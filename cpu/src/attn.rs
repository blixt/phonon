// Attention. Keys and values are f32, one row of 512 per position (8 heads of 64). The LM's
// cache is at fixed positions, the codec's a ring. Rotary angles are f32 (position times an
// f32 frequency) and their sine and cosine come from a table for the LM's positions, else from
// f64 math.
use crate::math::*;
use crate::ops::*;
use crate::pool::{claimed, scratch};
use core::arch::wasm32::*;

/// cos and sin of each of the 32 frequencies at `pos`, into `cs` ([32] cos then [32] sin).
unsafe fn angles(pos: usize, freqs: *const f32, table: *const f32, tn: usize, cs: *mut f32) {
    if !table.is_null() && pos < tn {
        core::ptr::copy(table.add(pos * 64), cs, 64);
        return;
    }
    for d in 0..32 {
        let a = pos as f32 * *freqs.add(d);
        let (s, c) = sincos(a as f64);
        *cs.add(d) = c as f32;
        *cs.add(32 + d) = s as f32;
    }
}

/// The table of `angles` for positions 0..n.
#[no_mangle]
pub unsafe extern "C" fn rope_table(freqs: *const f32, out: *mut f32, n: u32) {
    for pos in 0..n as usize {
        angles(pos, freqs, core::ptr::null(), 0, out.add(pos * 64));
    }
}

/// x rotated, 64 values of a head from `x`, scaled by `k`, into `out`.
#[inline(always)]
unsafe fn rotate(x: *const f32, cs: *const f32, k: f32, out: *mut f32) {
    let kv = f32x4_splat(k);
    for d in (0..32).step_by(4) {
        let lo = v128_load(x.add(d) as *const v128);
        let hi = v128_load(x.add(d + 32) as *const v128);
        let c = v128_load(cs.add(d) as *const v128);
        let s = v128_load(cs.add(32 + d) as *const v128);
        let a = f32x4_sub(f32x4_mul(lo, c), f32x4_mul(hi, s));
        let b = f32x4_add(f32x4_mul(hi, c), f32x4_mul(lo, s));
        v128_store(out.add(d) as *mut v128, f32x4_mul(a, kv));
        v128_store(out.add(d + 32) as *mut v128, f32x4_mul(b, kv));
    }
}

/// RoPE of rows: qkv, q, kc, vc, rows, pos0, freqs, codec, ring, state, table, table rows
pub unsafe fn rope(op: *const u32, i: usize, nt: usize) {
    let st = wp::<u32>(op, 10);
    let codec = w(op, 8) != 0;
    if codec && gated(GATE_CHUNK, st) {
        return;
    }
    let (qkv, q, kc, vc) = (wp::<f32>(op, 1), wp::<f32>(op, 2), wp::<f32>(op, 3), wp::<f32>(op, 4));
    let at = if codec { *st.add(CHUNK) as usize } else { 0 };
    let ring = wu(op, 9);
    let cs = scratch(i) as *mut f32;
    for m in claimed(wu(op, 5), nt) {
        let pos = wu(op, 6) + at + m;
        let slot = if codec { pos % ring } else { pos };
        angles(pos, wp(op, 7), wp(op, 11), wu(op, 12), cs);
        for h in 0..8 {
            let b = qkv.add(m * 1536 + h * 64);
            rotate(b, cs, 1.0, q.add(m * 512 + h * 64));
            rotate(b.add(512), cs, 1.0, kc.add(slot * 512 + h * 64));
            core::ptr::copy(b.add(1024), vc.add(slot * 512 + h * 64), 64);
        }
    }
}

#[inline(always)]
unsafe fn dot64(q: &[v128; 16], k: *const f32) -> f32 {
    let mut a = f32x4_mul(q[0], v128_load(k as *const v128));
    let mut b = f32x4_mul(q[1], v128_load(k.add(4) as *const v128));
    for e in (2..16).step_by(2) {
        a = madd(q[e], v128_load(k.add(4 * e) as *const v128), a);
        b = madd(q[e + 1], v128_load(k.add(4 * e + 4) as *const v128), b);
    }
    hsum(f32x4_add(a, b))
}

#[inline(always)]
unsafe fn load64(p: *const f32, scale: f32) -> [v128; 16] {
    let s = f32x4_splat(scale);
    let mut q = [f32x4_splat(0.0); 16];
    for (e, v) in q.iter_mut().enumerate() {
        *v = f32x4_mul(v128_load(p.add(4 * e) as *const v128), s);
    }
    q
}

/// Scores of keys [j0, j1) into `sc` (from index 0), their max, then e^(score - max) in place,
/// and the values summed with those weights into `acc`. Returns (max, sum).
#[inline(always)]
unsafe fn softmax_values(
    q: &[v128; 16],
    kc: *const f32,
    vc: *const f32,
    slot: impl Fn(usize) -> usize,
    j0: usize,
    j1: usize,
    sc: *mut f32,
    acc: &mut [v128; 16],
    mut keep: impl FnMut(usize, f32),
) -> (f32, f32) {
    let mut mx = -1e30f32;
    for j in j0..j1 {
        let d = dot64(q, kc.add(slot(j) * 512));
        keep(j, d);
        *sc.add(j - j0) = d;
        mx = mx.max(d);
    }
    let n = j1 - j0;
    let m = f32x4_splat(mx);
    let mut s = f32x4_splat(0.0);
    let mut j = 0;
    while j + 4 <= n {
        let e = exp4(f32x4_sub(v128_load(sc.add(j) as *const v128), m));
        v128_store(sc.add(j) as *mut v128, e);
        s = f32x4_add(s, e);
        j += 4;
    }
    let mut sum = hsum(s);
    while j < n {
        let e = exp1(*sc.add(j) - mx);
        *sc.add(j) = e;
        sum += e;
        j += 1;
    }
    for j in j0..j1 {
        let p = f32x4_splat(*sc.add(j - j0));
        let v = vc.add(slot(j) * 512);
        for (e, a) in acc.iter_mut().enumerate() {
            *a = madd(p, v128_load(v.add(4 * e) as *const v128), *a);
        }
    }
    (mx, sum)
}

/// Causal attention of many rows: q, kc, vc, out, rows, pos0, codec, window, ring, state
pub unsafe fn attention(op: *const u32, i: usize, nt: usize) {
    let st = wp::<u32>(op, 10);
    let codec = w(op, 7) != 0;
    if codec && gated(GATE_CHUNK, st) {
        return;
    }
    let (q, kc, vc, out) = (wp::<f32>(op, 1), wp::<f32>(op, 2), wp::<f32>(op, 3), wp::<f32>(op, 4));
    let (window, ring) = (wu(op, 8), wu(op, 9));
    let base = wu(op, 6) + if codec { *st.add(CHUNK) as usize } else { 0 };
    let sc = scratch(i) as *mut f32;
    for t in claimed(wu(op, 5) * 8, nt) {
        let (m, h) = (t / 8, t % 8);
        let pos = base + m;
        let lo = if window > 0 && pos + 1 > window { pos + 1 - window } else { 0 };
        let qv = load64(q.add(m * 512 + h * 64), 0.125);
        let mut acc = [f32x4_splat(0.0); 16];
        let slot = |j: usize| if codec { j % ring } else { j };
        let (_, sum) = softmax_values(&qv, kc.add(h * 64), vc.add(h * 64), slot, lo, pos + 1, sc, &mut acc, |_, _| {});
        let inv = f32x4_splat(1.0 / sum);
        let o = out.add(m * 512 + h * 64);
        for (e, a) in acc.iter().enumerate() {
            v128_store(o.add(4 * e) as *mut v128, f32x4_mul(*a, inv));
        }
    }
}

/// An LM step's attention, split over the keys: qkv, kc, vc, parts, splits, freqs, align, al, lo,
/// width, head, state, table, table rows. Each (split, head) leaves its unnormalized output, max
/// and sum in parts[split][head][66].
pub unsafe fn step(op: *const u32, i: usize, nt: usize) {
    let st = wp::<u32>(op, 12);
    if gated(GATE_STEP, st) {
        return;
    }
    let (qkv, kc, vc, parts) = (wp::<f32>(op, 1), wp::<f32>(op, 2), wp::<f32>(op, 3), wp::<f32>(op, 4));
    let splits = wu(op, 5);
    let pos = *st.add(POS) as usize;
    let n = pos + 1;
    let per = n.div_ceil(splits);
    let align = w(op, 7) != 0;
    let (al, lo, width, head) = (wp::<f32>(op, 8), wu(op, 9), wu(op, 10), wu(op, 11));
    let (text, frame) = (*st.add(TEXT) as usize, *st.add(FRAME) as usize);
    let cs = scratch(i) as *mut f32;
    let qf = cs.add(64);
    let sc = cs.add(128);
    angles(pos, wp(op, 6), wp(op, 13), wu(op, 14), cs);
    for t in claimed(splits * 8, nt) {
        let (sp, h) = (t / 8, t % 8);
        let hb = h * 64;
        let j0 = sp * per;
        let j1 = n.min(j0 + per);
        let j0 = j0.min(j1);
        rotate(qkv.add(hb), cs, 0.125, qf);
        if pos >= j0 && pos < j1 {
            rotate(qkv.add(512 + hb), cs, 1.0, kc.add(pos * 512 + hb));
            core::ptr::copy(qkv.add(1024 + hb), vc.add(pos * 512 + hb), 64);
        }
        let qv = load64(qf, 1.0);
        let mut acc = [f32x4_splat(0.0); 16];
        let keep = |j: usize, d: f32| {
            if align && h == head && j >= lo && j - lo < text {
                *al.add(frame * width + j - lo) = d;
            }
        };
        let (mx, sum) = softmax_values(&qv, kc.add(hb), vc.add(hb), |j| j, j0, j1, sc, &mut acc, keep);
        let pb = parts.add((sp * 8 + h) * 66);
        for (e, a) in acc.iter().enumerate() {
            v128_store(pb.add(4 * e) as *mut v128, *a);
        }
        *pb.add(64) = if j1 <= j0 { -1e30 } else { mx };
        *pb.add(65) = sum;
    }
}

/// Frame 0's alignment row: q, kc, al, lo, head, state
pub unsafe fn align_row(op: *const u32) {
    let st = wp::<u32>(op, 6);
    let (q, kc, al) = (wp::<f32>(op, 1), wp::<f32>(op, 2), wp::<f32>(op, 3));
    let (lo, head) = (wu(op, 4), wu(op, 5));
    let m = *st.add(POS) as usize - lo;
    let qv = load64(q.add(m * 512 + head * 64), 0.125);
    for i in 0..*st.add(TEXT) as usize {
        *al.add(i) = dot64(&qv, kc.add((lo + i) * 512 + head * 64));
    }
}
