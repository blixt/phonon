// Each op is a descriptor of u32 words in memory: its kind, then its fields in the order cpu.ts
// writes them (cpu.ts FIELDS). Pointers are byte offsets. `run` executes a list of ops in order,
// each on every thread. The small ops are here; the matrix products and attention have their
// own files.
use crate::math::*;
use crate::pool::{claimed, parallel, scratch};
use core::arch::wasm32::*;

pub const COPY: u32 = 0;
pub const CLEAR: u32 = 1;
pub const GEMV: u32 = 2;
pub const GEMM: u32 = 3;
pub const LAYERNORM: u32 = 4;
pub const ROPE: u32 = 5;
pub const ATTENTION: u32 = 6;
pub const ATTN_STEP: u32 = 7;
pub const ALIGN_ROW: u32 = 8;
pub const EMBED: u32 = 9;
pub const NOISE: u32 = 10;
pub const FINISH: u32 = 11;
pub const UPSAMPLE: u32 = 12;
pub const SHIFT: u32 = 13;
pub const ELU: u32 = 14;

// The state's u32 slots (graph.ts STATE).
pub const POS: usize = 0;
pub const FRAME: usize = 1;
pub const END: usize = 2;
pub const CHUNK: usize = 3;
pub const TEXT: usize = 4;
pub const AFTER: usize = 5;
pub const SEED: usize = 6;
pub const SCALE: usize = 7;

pub const GATE_STEP: u32 = 1;
pub const GATE_CHUNK: u32 = 2;

#[inline(always)]
pub unsafe fn w(op: *const u32, k: usize) -> u32 {
    *op.add(k)
}
#[inline(always)]
pub unsafe fn wu(op: *const u32, k: usize) -> usize {
    *op.add(k) as usize
}
#[inline(always)]
pub unsafe fn wf(op: *const u32, k: usize) -> f32 {
    f32::from_bits(*op.add(k))
}
#[inline(always)]
pub unsafe fn wp<T>(op: *const u32, k: usize) -> *mut T {
    *op.add(k) as usize as *mut T
}

/// Work past the end of speech returns at once.
#[inline(always)]
pub unsafe fn gated(gate: u32, st: *const u32) -> bool {
    let end = *st.add(END);
    match gate {
        GATE_STEP => end != 0 && *st.add(FRAME) >= end,
        GATE_CHUNK => end != 0 && *st.add(CHUNK) / 16 >= end,
        _ => false,
    }
}

/// Runs `n` ops listed at `list`, in order.
#[no_mangle]
pub unsafe extern "C" fn run(list: *const u32, n: u32) {
    for i in 0..n as usize {
        let op = *list.add(i) as usize as *const u32;
        match *op {
            COPY => core::ptr::copy(wp::<u8>(op, 1), wp::<u8>(op, 2), wu(op, 3)),
            CLEAR => core::ptr::write_bytes(wp::<u8>(op, 1), 0, wu(op, 2)),
            NOISE | FINISH | ALIGN_ROW | SHIFT => part(op, 0, 1),
            _ => parallel(op),
        }
    }
}

/// Thread `i` of `nt`'s part of `op`.
pub unsafe fn part(op: *const u32, i: usize, nt: usize) {
    match *op {
        GEMV => crate::gemv::gemv(op, i, nt),
        GEMM => crate::gemm::gemm(op, i, nt),
        LAYERNORM => layernorm(op, i, nt),
        ROPE => crate::attn::rope(op, i, nt),
        ATTENTION => crate::attn::attention(op, i, nt),
        ATTN_STEP => crate::attn::step(op, i, nt),
        ALIGN_ROW => crate::attn::align_row(op),
        EMBED => embed(op, i, nt),
        NOISE => noise(op),
        FINISH => finish(op),
        UPSAMPLE => upsample(op, i, nt),
        SHIFT => shift(op),
        ELU => elu(op, i, nt),
        _ => {}
    }
}

/// Mean and 1 / sqrt(variance + eps) of 512 values, in two passes.
pub unsafe fn moments(x: *const f32, eps: f32) -> (f32, f32) {
    let mut s = f32x4_splat(0.0);
    for c in (0..512).step_by(4) {
        s = f32x4_add(s, v128_load(x.add(c) as *const v128));
    }
    let mean = hsum(s) / 512.0;
    let m = f32x4_splat(mean);
    let mut q = f32x4_splat(0.0);
    for c in (0..512).step_by(4) {
        let d = f32x4_sub(v128_load(x.add(c) as *const v128), m);
        q = madd(d, d, q);
    }
    (mean, 1.0 / sqrt(hsum(q) / 512.0 + eps))
}

/// LayerNorm: x, y, rows, w, b, eps, gate, state
unsafe fn layernorm(op: *const u32, _i: usize, nt: usize) {
    if gated(w(op, 7), wp(op, 8)) {
        return;
    }
    let (x, y) = (wp::<f32>(op, 1), wp::<f32>(op, 2));
    let (g, b) = (wp::<u16>(op, 4), wp::<u16>(op, 5));
    for r in claimed(wu(op, 3), nt) {
        let (xr, yr) = (x.add(r * 512), y.add(r * 512));
        let (mean, inv) = moments(xr, wf(op, 6));
        let (m, s) = (f32x4_splat(mean), f32x4_splat(inv));
        for c in (0..512).step_by(4) {
            let v = f32x4_mul(f32x4_sub(v128_load(xr.add(c) as *const v128), m), s);
            v128_store(yr.add(c) as *mut v128, madd(v, h4(g.add(c)), h4(b.add(c))));
        }
    }
}

/// Embedding rows: ids, x, rows, bos, pad, embed, embed scales, bos_embed
unsafe fn embed(op: *const u32, _i: usize, nt: usize) {
    let (ids, x) = (wp::<u32>(op, 1), wp::<f32>(op, 2));
    let (bos, pad) = (w(op, 4), w(op, 5));
    let (ew, es, bv) = (wp::<i8>(op, 6), wp::<u16>(op, 7), wp::<u16>(op, 8));
    for r in claimed(wu(op, 3), nt) {
        let id = *ids.add(r);
        let xr = x.add(r * 512);
        for c in (0..512).step_by(16) {
            let (a, b, cc, d) = if id == bos {
                (h4(bv.add(c)), h4(bv.add(c + 4)), h4(bv.add(c + 8)), h4(bv.add(c + 12)))
            } else if id == pad {
                let z = f32x4_splat(0.0);
                (z, z, z, z)
            } else {
                let at = id as usize * 512 + c;
                let s = f32x4_splat(hf(es, at / 32));
                let (a, b, cc, d) = i8x16_f32(v128_load(ew.add(at) as *const v128));
                (f32x4_mul(a, s), f32x4_mul(b, s), f32x4_mul(cc, s), f32x4_mul(d, s))
            };
            v128_store(xr.add(c) as *mut v128, a);
            v128_store(xr.add(c + 4) as *mut v128, b);
            v128_store(xr.add(c + 8) as *mut v128, cc);
            v128_store(xr.add(c + 12) as *mut v128, d);
        }
    }
}

fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// The next latent's noise: z, state
unsafe fn noise(op: *const u32) {
    let st = wp::<u32>(op, 2);
    if gated(GATE_STEP, st) {
        return;
    }
    let z = wp::<f32>(op, 1);
    let f = *st.add(FRAME);
    let scale = f32::from_bits(*st.add(SCALE));
    for i in 0..32u32 {
        let a = pcg(*st.add(SEED) ^ pcg(f.wrapping_mul(64).wrapping_add(i * 2)));
        let b = pcg(a ^ 0x9e37_79b9);
        let u1 = ((a >> 8) as f32 + 1.0) / 16_777_217.0;
        let u2 = (b >> 8) as f32 / 16_777_216.0;
        let r = f64_sqrt(-2.0 * ln(u1 as f64)) as f32;
        let c = sincos((6.283_185_3f32 * u2) as f64).1 as f32;
        *z.add(i as usize) = r * c * scale;
    }
}

/// The end of a step: x, lat, lats, state, eos.w, eos.b, norm.w, norm.b, eps, threshold
unsafe fn finish(op: *const u32) {
    let st = wp::<u32>(op, 4);
    if gated(GATE_STEP, st) {
        return;
    }
    let (x, lat, lats) = (wp::<f32>(op, 1), wp::<f32>(op, 2), wp::<f32>(op, 3));
    let (ew, eb, nw, nb) = (wp::<u16>(op, 5), wp::<u16>(op, 6), wp::<u16>(op, 7), wp::<u16>(op, 8));
    let (mean, inv) = moments(x, wf(op, 9));
    let mut acc = f32x4_splat(0.0);
    for c in (0..512).step_by(4) {
        let v = f32x4_mul(f32x4_sub(v128_load(x.add(c) as *const v128), f32x4_splat(mean)), f32x4_splat(inv));
        let n = madd(v, h4(nw.add(c)), h4(nb.add(c)));
        acc = madd(n, h4(ew.add(c)), acc);
    }
    let logit = hsum(acc) + hf(eb, 0);
    let f = *st.add(FRAME);
    core::ptr::copy(lat, lats.add(f as usize * 32), 32);
    *st.add(POS) += 1;
    *st.add(FRAME) = f + 1;
    if *st.add(END) == 0 && logit > wf(op, 10) {
        *st.add(END) = f + *st.add(AFTER) + 1;
    }
}

/// The codec's input: lats, y, T, up, up scales, lat_mean, lat_std, state. Each 200 Hz position
/// t of the chunk is 2 taps of the denormalized latents of frames t / 16 and t / 16 - 1.
unsafe fn upsample(op: *const u32, i: usize, nt: usize) {
    let st = wp::<u32>(op, 8);
    if gated(GATE_CHUNK, st) {
        return;
    }
    let (lats, y) = (wp::<f32>(op, 1), wp::<f32>(op, 2));
    let (uw, us, mean, std) = (wp::<i8>(op, 4), wp::<u16>(op, 5), wp::<u16>(op, 6), wp::<u16>(op, 7));
    let chunk = *st.add(CHUNK) as usize;
    let lat = scratch(i) as *mut f32;
    for tl in claimed(wu(op, 3), nt) {
        let t = chunk + tl;
        let (ph, q) = (t % 16, t / 16);
        // The two frames' latents, denormalized (zero before the first frame).
        for j in 0..2 {
            for c in (0..32).step_by(4) {
                let v = if q >= j {
                    madd(v128_load(lats.add((q - j) * 32 + c) as *const v128), h4(std.add(c)), h4(mean.add(c)))
                } else {
                    f32x4_splat(0.0)
                };
                v128_store(lat.add(j * 32 + c) as *mut v128, v);
            }
        }
        let yr = y.add(tl * 512);
        for c in 0..512 {
            let mut acc = f32x4_splat(0.0);
            for j in 0..2 {
                let row = (c * 16 + ph) * 2 + j;
                let s = f32x4_splat(hf(us, row));
                let wr = uw.add(row * 32);
                let (a, b, cc, d) = i8x16_f32(v128_load(wr as *const v128));
                let (e, f, g, h) = i8x16_f32(v128_load(wr.add(16) as *const v128));
                let l = lat.add(j * 32);
                let mut p = f32x4_mul(a, v128_load(l as *const v128));
                p = madd(b, v128_load(l.add(4) as *const v128), p);
                p = madd(cc, v128_load(l.add(8) as *const v128), p);
                p = madd(d, v128_load(l.add(12) as *const v128), p);
                p = madd(e, v128_load(l.add(16) as *const v128), p);
                p = madd(f, v128_load(l.add(20) as *const v128), p);
                p = madd(g, v128_load(l.add(24) as *const v128), p);
                p = madd(h, v128_load(l.add(28) as *const v128), p);
                acc = madd(p, s, acc);
            }
            *yr.add(c) = hsum(acc);
        }
    }
}

/// Each stage's last rows to its front: T, n, then each stage's buf, ctx, rate, ch.
unsafe fn shift(op: *const u32) {
    let t = wu(op, 1);
    for s in 0..wu(op, 2) {
        let b = wp::<f32>(op, 3 + 4 * s);
        let (ctx, rate, ch) = (wu(op, 4 + 4 * s), wu(op, 5 + 4 * s), wu(op, 6 + 4 * s));
        core::ptr::copy(b.add(t * rate * ch), b, ctx * ch);
    }
}

/// ELU of a stage before a convolution reads it: src, dst, count, gate, state
unsafe fn elu(op: *const u32, _i: usize, nt: usize) {
    if gated(w(op, 4), wp(op, 5)) {
        return;
    }
    let (src, dst) = (wp::<f32>(op, 1), wp::<f32>(op, 2));
    for u in claimed(wu(op, 3) / 4, nt) {
        let k = 4 * u;
        v128_store(dst.add(k) as *mut v128, elu4(v128_load(src.add(k) as *const v128)));
    }
}
