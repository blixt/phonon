// An LM step's matrix-vector product, y = epilogue(W x), with int8 weights and one f16 scale per
// 32. Each thread first prepares the input in its scratch (LayerNorm, or the attention splits'
// merge), then takes a share of the output rows, 4 at a time.
//
// Fields: x, xo, w, ws, y, K, N, act, bias, res, r, g, go, v, ln, ln.w, ln.b, eps, mod, so, ho,
// splits, parts, state.
use crate::math::*;
use crate::ops::*;
use crate::pool::{claimed, scratch};
use core::arch::wasm32::*;

pub const RES_ADD: u32 = 1;
pub const RES_GATE: u32 = 2;
pub const RES_VEC: u32 = 3;
pub const LN_ON: u32 = 1;
pub const LN_AFFINE: u32 = 2;
pub const LN_MODULATE: u32 = 4;

pub unsafe fn gemv(op: *const u32, i: usize, nt: usize) {
    if gated(GATE_STEP, wp(op, 24)) {
        return;
    }
    let (k, n) = (wu(op, 6), wu(op, 7));
    let xs = scratch(i) as *mut f32;
    prepare(op, xs, k);
    let (w, ws) = (wp::<i8>(op, 3), wp::<u16>(op, 4));
    if crate::pool::gemv_mode() == 1 && k % 128 == 0 {
        // The input in int16, one scale per 32 values, then integer dot products.
        let xq = xs.add(k) as *mut i16;
        let sx = xq.add(k) as *mut f32;
        quantize(xs, k, xq, sx);
        for b in claimed(n / 4, nt) {
            epilogue(op, 4 * b, rows_q16(xq, sx, w, ws, k, 4 * b));
        }
        return;
    }
    for b in claimed(n / 4, nt) {
        epilogue(op, 4 * b, rows4(xs, w, ws, k, 4 * b));
    }
}

/// The input, ready for the rows: LayerNorm (with its affine weights and adaLN modulation), the
/// merge of attention splits, or a copy.
unsafe fn prepare(op: *const u32, xs: *mut f32, k: usize) {
    let ln = w(op, 15);
    let splits = wu(op, 22);
    if ln & LN_ON != 0 {
        let x = wp::<f32>(op, 1).add(wu(op, 2));
        // One pass, sum and sum of squares, as the WGSL kernel.
        let (mut s1, mut s2) = (f32x4_splat(0.0), f32x4_splat(0.0));
        for c in (0..512).step_by(4) {
            let v = v128_load(x.add(c) as *const v128);
            s1 = f32x4_add(s1, v);
            s2 = madd(v, v, s2);
        }
        let mean = hsum(s1) / 512.0;
        let var = (hsum(s2) / 512.0 - mean * mean).max(0.0);
        let inv = f32x4_splat(1.0 / sqrt(var + wf(op, 18)));
        let m = f32x4_splat(mean);
        let (lg, lb) = (wp::<u16>(op, 16), wp::<u16>(op, 17));
        let md = wp::<f32>(op, 19);
        let (so, ho) = (wu(op, 20), wu(op, 21));
        for c in (0..512).step_by(4) {
            let mut v = f32x4_mul(f32x4_sub(v128_load(x.add(c) as *const v128), m), inv);
            if ln & LN_AFFINE != 0 {
                v = madd(v, h4(lg.add(c)), h4(lb.add(c)));
            }
            if ln & LN_MODULATE != 0 {
                let scale = f32x4_add(f32x4_splat(1.0), v128_load(md.add(so + c) as *const v128));
                v = madd(v, scale, v128_load(md.add(ho + c) as *const v128));
            }
            v128_store(xs.add(c) as *mut v128, v);
        }
    } else if splits > 0 {
        // parts: [split][head] blocks of 66 floats: 64 dims, max, sum.
        let parts = wp::<f32>(op, 23);
        for h in 0..8 {
            let mut mx = -1e30f32;
            for s in 0..splits {
                mx = mx.max(*parts.add((s * 8 + h) * 66 + 64));
            }
            let mut num = [f32x4_splat(0.0); 16];
            let mut den = 0.0f32;
            for s in 0..splits {
                let b = parts.add((s * 8 + h) * 66);
                let f = exp1(*b.add(64) - mx);
                den += *b.add(65) * f;
                let fv = f32x4_splat(f);
                for (d, n) in num.iter_mut().enumerate() {
                    *n = madd(v128_load(b.add(d * 4) as *const v128), fv, *n);
                }
            }
            let inv = f32x4_splat(1.0 / den);
            for (d, n) in num.iter().enumerate() {
                v128_store(xs.add(h * 64 + d * 4) as *mut v128, f32x4_mul(*n, inv));
            }
        }
    } else {
        core::ptr::copy(wp::<f32>(op, 1).add(wu(op, 2)), xs, k);
    }
}

/// Each group of 32 values of x as int16 (rounded, saturating) and its scale, max |x| / 32767.
unsafe fn quantize(xs: *const f32, k: usize, xq: *mut i16, sx: *mut f32) {
    for g in 0..k / 32 {
        let x = xs.add(g * 32);
        let mut mx = f32x4_splat(0.0);
        for c in (0..32).step_by(4) {
            mx = f32x4_max(mx, f32x4_abs(v128_load(x.add(c) as *const v128)));
        }
        let m = crate::math::hmax(mx);
        let (scale, inv) = if m > 0.0 { (m / 32767.0, 32767.0 / m) } else { (1.0, 0.0) };
        *sx.add(g) = scale;
        let iv = f32x4_splat(inv);
        for c in (0..32).step_by(8) {
            let a = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(v128_load(x.add(c) as *const v128), iv)));
            let b = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_mul(v128_load(x.add(c + 4) as *const v128), iv)));
            v128_store(xq.add(g * 32 + c) as *mut v128, i16x8_narrow_i32x4(a, b));
        }
    }
}

/// One group's integer dot product of 32 int8 weights and 32 int16 inputs, in 4 lanes.
#[inline(always)]
unsafe fn dot_q16(xq: *const i16, wr: *const i8) -> v128 {
    let b0 = v128_load(wr as *const v128);
    let b1 = v128_load(wr.add(16) as *const v128);
    let d0 = i32x4_dot_i16x8(i16x8_extend_low_i8x16(b0), v128_load(xq as *const v128));
    let d1 = i32x4_dot_i16x8(i16x8_extend_high_i8x16(b0), v128_load(xq.add(8) as *const v128));
    let d2 = i32x4_dot_i16x8(i16x8_extend_low_i8x16(b1), v128_load(xq.add(16) as *const v128));
    let d3 = i32x4_dot_i16x8(i16x8_extend_high_i8x16(b1), v128_load(xq.add(24) as *const v128));
    i32x4_add(i32x4_add(d0, d1), i32x4_add(d2, d3))
}

/// Rows n..n + 4: four groups at a time, each group's sum scaled by its weight scale times its
/// input scale.
#[inline(always)]
unsafe fn rows_q16(xq: *const i16, sx: *const f32, w: *const i8, ws: *const u16, k: usize, n: usize) -> v128 {
    let groups = k / 32;
    let mut acc = [f32x4_splat(0.0); 4];
    for g in (0..groups).step_by(4) {
        let xs = v128_load(sx.add(g) as *const v128);
        let x = xq.add(g * 32);
        for (r, a) in acc.iter_mut().enumerate() {
            let row = n + r;
            let wr = w.add(row * k + g * 32);
            let c = f32x4_mul(h4(ws.add(row * groups + g)), xs);
            *a = madd(f32x4_convert_i32x4(dot_q16(x, wr)), i32x4_shuffle::<0, 0, 0, 0>(c, c), *a);
            *a = madd(f32x4_convert_i32x4(dot_q16(x.add(32), wr.add(32))), i32x4_shuffle::<1, 1, 1, 1>(c, c), *a);
            *a = madd(f32x4_convert_i32x4(dot_q16(x.add(64), wr.add(64))), i32x4_shuffle::<2, 2, 2, 2>(c, c), *a);
            *a = madd(f32x4_convert_i32x4(dot_q16(x.add(96), wr.add(96))), i32x4_shuffle::<3, 3, 3, 3>(c, c), *a);
        }
    }
    sums(acc[0], acc[1], acc[2], acc[3])
}

/// The lane sums of four vectors, as one vector.
#[inline(always)]
fn sums(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(a, b), i32x4_shuffle::<2, 6, 3, 7>(a, b));
    let cd = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(c, d), i32x4_shuffle::<2, 6, 3, 7>(c, d));
    f32x4_add(i64x2_shuffle::<0, 2>(ab, cd), i64x2_shuffle::<1, 3>(ab, cd))
}

#[inline(always)]
unsafe fn group(xs: *const f32, wr: *const i8) -> v128 {
    let (a, b, c, d) = i8x16_f32(v128_load(wr as *const v128));
    let (e, f, g, h) = i8x16_f32(v128_load(wr.add(16) as *const v128));
    let mut p = f32x4_mul(a, v128_load(xs as *const v128));
    p = madd(b, v128_load(xs.add(4) as *const v128), p);
    p = madd(c, v128_load(xs.add(8) as *const v128), p);
    p = madd(d, v128_load(xs.add(12) as *const v128), p);
    p = madd(e, v128_load(xs.add(16) as *const v128), p);
    p = madd(f, v128_load(xs.add(20) as *const v128), p);
    p = madd(g, v128_load(xs.add(24) as *const v128), p);
    madd(h, v128_load(xs.add(28) as *const v128), p)
}

#[inline(always)]
unsafe fn rows4(xs: *const f32, w: *const i8, ws: *const u16, k: usize, n: usize) -> v128 {
    let groups = k / 32;
    let (mut a0, mut a1, mut a2, mut a3) = (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
    let (w0, w1, w2, w3) = (w.add(n * k), w.add((n + 1) * k), w.add((n + 2) * k), w.add((n + 3) * k));
    let (s0, s1, s2, s3) = (ws.add(n * groups), ws.add((n + 1) * groups), ws.add((n + 2) * groups), ws.add((n + 3) * groups));
    for g in 0..groups {
        let x = xs.add(g * 32);
        a0 = madd(group(x, w0.add(g * 32)), f32x4_splat(hf(s0, g)), a0);
        a1 = madd(group(x, w1.add(g * 32)), f32x4_splat(hf(s1, g)), a1);
        a2 = madd(group(x, w2.add(g * 32)), f32x4_splat(hf(s2, g)), a2);
        a3 = madd(group(x, w3.add(g * 32)), f32x4_splat(hf(s3, g)), a3);
    }
    sums(a0, a1, a2, a3)
}

/// Bias, activation and residual of rows n..n + 4, then the store.
#[inline(always)]
unsafe fn epilogue(op: *const u32, n: usize, acc: v128) {
    let mut v = acc;
    let bias = wp::<u16>(op, 9);
    if !bias.is_null() {
        v = f32x4_add(v, h4(bias.add(n)));
    }
    v = act4(w(op, 8), v);
    let r = || v128_load(wp::<f32>(op, 11).add(n) as *const v128);
    match w(op, 10) {
        RES_ADD => v = f32x4_add(v, r()),
        RES_GATE => v = madd(v128_load(wp::<f32>(op, 12).add(wu(op, 13) + n) as *const v128), v, r()),
        RES_VEC => v = f32x4_add(v, h4(wp::<u16>(op, 14).add(n))),
        _ => {}
    }
    v128_store(wp::<f32>(op, 5).add(n) as *mut v128, v);
}
