// Threads: the engine's thread runs every op, and pool threads join in. Each op's work is in
// units that threads claim from a counter, in chunks, so a slow or preempted thread holds back
// no one. Pool threads wait for the next op by spinning on a generation counter, then sleep in
// memory.atomic.wait once nothing has come for a while. An op waits only for the threads that
// joined it: once the engine's thread has run out of units it closes the op, and a thread that
// comes later skips it. Without shared memory there is one thread and nothing waits.
use core::arch::wasm32::*;
use core::sync::atomic::{AtomicU32, Ordering::*};

static GEN: AtomicU32 = AtomicU32::new(0);
/// The next unit of the op in flight that no thread has claimed.
static NEXT: AtomicU32 = AtomicU32::new(0);
static DONE: AtomicU32 = AtomicU32::new(0);
/// The op in flight's generation (high 16 bits), whether it is closed (bit 15), and the pool
/// threads that joined it.
static ENTRY: AtomicU32 = AtomicU32::new(0);
const CLOSED: u32 = 0x8000;
static JOB: AtomicU32 = AtomicU32::new(0);
/// Threads taking part in the op in flight.
static JOB_THREADS: AtomicU32 = AtomicU32::new(1);
static THREADS: AtomicU32 = AtomicU32::new(1);
static SLEEPERS: AtomicU32 = AtomicU32::new(0);
static SCRATCH: AtomicU32 = AtomicU32::new(0);
static SCRATCH_PER: AtomicU32 = AtomicU32::new(0);
static TILE: AtomicU32 = AtomicU32::new(1 << 16 | 4 << 8 | 4);
/// Spins before a pool thread sleeps: a few milliseconds.
const SPINS: u32 = 1 << 20;

/// Bytes of scratch each thread has.
pub fn scratch_bytes() -> usize {
    SCRATCH_PER.load(Relaxed) as usize
}

pub fn scratch(i: usize) -> *mut u8 {
    (SCRATCH.load(Relaxed) as usize + i * scratch_bytes()) as *mut u8
}

/// A GEMM tile: dot products (rows x 4 or 2 x 8) or broadcasts (rows x 8 or 16), whichever
/// runs fastest here (cpu.ts times them).
pub fn tile_shape() -> (bool, usize, usize) {
    let t = TILE.load(Relaxed) as usize;
    (t >> 16 != 0, (t >> 8) & 0xff, t & 0xff)
}

#[no_mangle]
pub extern "C" fn set_tile(dot: u32, rows: u32, cols: u32) {
    TILE.store(dot << 16 | rows << 8 | cols, SeqCst);
}

static GEMV_MODE: AtomicU32 = AtomicU32::new(1);
/// The GEMV's arithmetic: 0 for f32, 1 for int16 inputs with integer dot products.
pub fn gemv_mode() -> u32 {
    GEMV_MODE.load(Relaxed)
}

#[no_mangle]
pub extern "C" fn set_gemv(mode: u32) {
    GEMV_MODE.store(mode, SeqCst);
}

pub fn threads() -> usize {
    THREADS.load(Relaxed) as usize
}

/// The threads and each one's scratch. Called by the engine's thread between runs.
#[no_mangle]
pub extern "C" fn setup(threads: u32, scratch: u32, per: u32) {
    THREADS.store(threads.max(1), SeqCst);
    SCRATCH.store(scratch, SeqCst);
    SCRATCH_PER.store(per, SeqCst);
}

/// Runs `op` on every thread, each with its index, and returns when all are done.
pub unsafe fn parallel(op: *const u32) {
    let nt = threads();
    NEXT.store(0, SeqCst);
    if nt == 1 {
        crate::ops::part(op, 0, 1);
        return;
    }
    JOB.store(op as u32, SeqCst);
    JOB_THREADS.store(nt as u32, SeqCst);
    DONE.store(0, SeqCst);
    let gen = GEN.load(SeqCst).wrapping_add(1);
    ENTRY.store(gen << 16, SeqCst);
    GEN.store(gen, SeqCst);
    if SLEEPERS.load(SeqCst) > 0 {
        memory_atomic_notify(GEN.as_ptr() as *mut i32, u32::MAX);
    }
    crate::ops::part(op, 0, nt);
    let joined = ENTRY.fetch_or(CLOSED, SeqCst) & 0x7fff;
    while DONE.load(SeqCst) < joined {}
}

/// Joins op `gen` unless it is closed or past.
fn join(gen: u32) -> bool {
    let mut cur = ENTRY.load(SeqCst);
    loop {
        if cur >> 16 != gen & 0xffff || cur & CLOSED != 0 {
            return false;
        }
        match ENTRY.compare_exchange(cur, cur + 1, SeqCst, SeqCst) {
            Ok(_) => return true,
            Err(now) => cur = now,
        }
    }
}

/// A pool thread's loop: thread `i` (from 1) takes its part of each op. It never returns.
#[no_mangle]
pub unsafe extern "C" fn worker(i: u32) {
    let mut seen = GEN.load(SeqCst);
    loop {
        let mut spins = 0u32;
        loop {
            let g = GEN.load(SeqCst);
            if g != seen {
                seen = g;
                break;
            }
            spins += 1;
            if spins > SPINS {
                SLEEPERS.fetch_add(1, SeqCst);
                if GEN.load(SeqCst) == seen {
                    memory_atomic_wait32(GEN.as_ptr() as *mut i32, seen as i32, -1);
                }
                SLEEPERS.fetch_sub(1, SeqCst);
                spins = 0;
            }
        }
        let nt = JOB_THREADS.load(SeqCst);
        if i < nt && join(seen) {
            crate::ops::part(JOB.load(SeqCst) as *const u32, i as usize, nt as usize);
            DONE.fetch_add(1, SeqCst);
        }
    }
}

/// The units of an op that this thread takes: chunks of about a quarter of a thread's share,
/// claimed from a counter, so a slow or preempted thread holds back no one. One loop per op.
pub fn claimed(total: usize, nt: usize) -> Claimed {
    let chunk = if nt == 1 { total.max(1) } else { (total / (nt * 4)).max(1) };
    Claimed { total, chunk, at: 0, end: 0 }
}

pub struct Claimed {
    total: usize,
    chunk: usize,
    at: usize,
    end: usize,
}

impl Iterator for Claimed {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.at == self.end {
            let a = NEXT.fetch_add(self.chunk as u32, SeqCst) as usize;
            if a >= self.total {
                return None;
            }
            self.at = a;
            self.end = (a + self.chunk).min(self.total);
        }
        self.at += 1;
        Some(self.at - 1)
    }
}

/// Part `i` of `n` of `total` items, in whole multiples of `align` (the last part takes the rest).
#[inline]
pub fn share(total: usize, i: usize, n: usize, align: usize) -> (usize, usize) {
    let units = total.div_ceil(align);
    let lo = units * i / n * align;
    let hi = (units * (i + 1) / n * align).min(total);
    (lo.min(total), hi)
}
