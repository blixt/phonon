// Threads: the engine's thread runs every op, and with N threads each op's work is cut in N
// parts. Pool threads wait for the next op by spinning on a generation counter, then sleep in
// memory.atomic.wait once nothing has come for a while. Without shared memory there is one
// thread and nothing waits.
use core::arch::wasm32::*;
use core::sync::atomic::{AtomicU32, Ordering::*};

static GEN: AtomicU32 = AtomicU32::new(0);
static DONE: AtomicU32 = AtomicU32::new(0);
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
    if nt == 1 {
        crate::ops::part(op, 0, 1);
        return;
    }
    JOB.store(op as u32, SeqCst);
    JOB_THREADS.store(nt as u32, SeqCst);
    DONE.store(0, SeqCst);
    GEN.fetch_add(1, SeqCst);
    if SLEEPERS.load(SeqCst) > 0 {
        memory_atomic_notify(GEN.as_ptr() as *mut i32, u32::MAX);
    }
    crate::ops::part(op, 0, nt);
    while DONE.load(SeqCst) < (nt - 1) as u32 {}
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
        if i < nt {
            crate::ops::part(JOB.load(SeqCst) as *const u32, i as usize, nt as usize);
            DONE.fetch_add(1, SeqCst);
        }
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
