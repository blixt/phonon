// Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. Every load
// reads row m of A as K contiguous values: a linear layer's row, a convolution's k input rows
// from row m (each input row has cin channels), or a transposed convolution's input rows m and
// m + 1, its two taps swapped as the weights are packed. A thread dequantizes 8 columns of W at
// a time into a panel of f32 in its scratch, [K][8], then runs 4 rows by 8 columns at a time over
// it, with the epilogue in SIMD.
//
// Fields: x, w, ws (0: f16 weights), y, M, K, N, act, bias, res, r, yo, ro, load, cin, stride,
// gate, state.
use crate::math::*;
use crate::ops::*;
use crate::pool::{scratch, scratch_bytes, share};
use core::arch::wasm32::*;

const LINEAR: u32 = 0;
const CONVT: u32 = 2;
const NR: usize = 8;

struct Shape {
    m: usize,
    k: usize,
    n: usize,
    phases: usize,
    /// Elements between A rows.
    astride: usize,
    /// Elements per weight row (all phases of an output channel).
    wrow: usize,
    cin: usize,
    convt: bool,
}

pub unsafe fn gemm(op: *const u32, i: usize, nt: usize) {
    if gated(w(op, 17), wp(op, 18)) {
        return;
    }
    let (m, k, n) = (wu(op, 5), wu(op, 6), wu(op, 7));
    let load = w(op, 14);
    let convt = load == CONVT;
    let phases = if convt { wu(op, 16) } else { 1 };
    let cin = wu(op, 15);
    let s = Shape {
        m,
        k,
        n,
        phases,
        astride: if load == LINEAR { k } else { cin },
        wrow: phases * k,
        cin,
        convt,
    };
    let blocks = n.div_ceil(NR);
    let units = phases * blocks;
    // Rows split too when there are few column blocks, at least 16 rows each.
    let mparts = (4 * nt).div_ceil(units).min(m.div_ceil(16)).max(1);
    let tasks = units * mparts;
    let (t0, t1) = share(tasks, i, nt, 1);
    let panel = scratch(i) as *mut f32;
    if s.k * NR * 4 > scratch_bytes() {
        return;
    }
    let mut packed = usize::MAX;
    for t in t0..t1 {
        let (unit, part) = (t / mparts, t % mparts);
        let (ph, nb) = (unit / blocks, unit % blocks);
        if unit != packed {
            pack(op, &s, panel, ph, nb * NR);
            packed = unit;
        }
        let (m0, m1) = share(m, part, mparts, 4);
        run(op, &s, panel, ph, nb * NR, m0, m1);
    }
}

/// Columns n0..n0 + 8 of phase `ph` of W as f32, [K][8], zero past N.
unsafe fn pack(op: *const u32, s: &Shape, panel: *mut f32, ph: usize, n0: usize) {
    let (w8, ws, w16) = (wp::<i8>(op, 2), wp::<u16>(op, 3), wp::<u16>(op, 2));
    let q8 = !ws.is_null();
    for j in 0..NR {
        let n = n0 + j;
        if n >= s.n {
            for kk in 0..s.k {
                *panel.add(kk * NR + j) = 0.0;
            }
            continue;
        }
        let base = n * s.wrow + ph * s.k;
        // A's index kk reads weight index src(kk): a transposed convolution's taps swap halves.
        let mut kk = 0;
        while kk < s.k {
            let src = if s.convt { if kk < s.cin { kk + s.cin } else { kk - s.cin } } else { kk };
            let at = base + src;
            let p = panel.add(kk * NR + j);
            if q8 {
                let sc = f32x4_splat(hf(ws, at / 32));
                let (a, b, c, d) = i8x16_f32(v128_load(w8.add(at) as *const v128));
                for (q, v) in [a, b, c, d].iter().enumerate() {
                    let v = f32x4_mul(*v, sc);
                    *p.add((4 * q) * NR) = f32x4_extract_lane::<0>(v);
                    *p.add((4 * q + 1) * NR) = f32x4_extract_lane::<1>(v);
                    *p.add((4 * q + 2) * NR) = f32x4_extract_lane::<2>(v);
                    *p.add((4 * q + 3) * NR) = f32x4_extract_lane::<3>(v);
                }
                kk += 16;
            } else {
                *p = hf(w16, at);
                kk += 1;
            }
        }
    }
}

/// Rows m0..m1 (m0 a multiple of 4) against one panel.
unsafe fn run(op: *const u32, s: &Shape, panel: *const f32, ph: usize, n0: usize, m0: usize, m1: usize) {
    let x = wp::<f32>(op, 1);
    let mut m = m0;
    while m + 4 <= m1 {
        let a = x.add(m * s.astride);
        let st = s.astride;
        let (mut c00, mut c01, mut c10, mut c11) = (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let (mut c20, mut c21, mut c30, mut c31) = (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        for kk in 0..s.k {
            let b0 = v128_load(panel.add(kk * NR) as *const v128);
            let b1 = v128_load(panel.add(kk * NR + 4) as *const v128);
            let a0 = v128_load32_splat(a.add(kk) as *const u32);
            let a1 = v128_load32_splat(a.add(st + kk) as *const u32);
            let a2 = v128_load32_splat(a.add(2 * st + kk) as *const u32);
            let a3 = v128_load32_splat(a.add(3 * st + kk) as *const u32);
            c00 = madd(a0, b0, c00);
            c01 = madd(a0, b1, c01);
            c10 = madd(a1, b0, c10);
            c11 = madd(a1, b1, c11);
            c20 = madd(a2, b0, c20);
            c21 = madd(a2, b1, c21);
            c30 = madd(a3, b0, c30);
            c31 = madd(a3, b1, c31);
        }
        store(op, s, ph, m, n0, c00, c01);
        store(op, s, ph, m + 1, n0, c10, c11);
        store(op, s, ph, m + 2, n0, c20, c21);
        store(op, s, ph, m + 3, n0, c30, c31);
        m += 4;
    }
    while m < m1 {
        let a = x.add(m * s.astride);
        let (mut c0, mut c1) = (f32x4_splat(0.0), f32x4_splat(0.0));
        for kk in 0..s.k {
            let av = v128_load32_splat(a.add(kk) as *const u32);
            c0 = madd(av, v128_load(panel.add(kk * NR) as *const v128), c0);
            c1 = madd(av, v128_load(panel.add(kk * NR + 4) as *const v128), c1);
        }
        store(op, s, ph, m, n0, c0, c1);
        m += 1;
    }
}

/// Bias, activation and residual for row m, columns n0..n0 + 8, then the store.
#[inline(always)]
unsafe fn store(op: *const u32, s: &Shape, ph: usize, m: usize, n0: usize, v0: v128, v1: v128) {
    let orow = m * s.phases + ph;
    let (y, r) = (wp::<f32>(op, 4), wp::<f32>(op, 11));
    let o = wu(op, 12) + orow * s.n + n0;
    let ro = wu(op, 13) + orow * s.n + n0;
    let bias = wp::<u16>(op, 9);
    let act = w(op, 8);
    let res = w(op, 10) != 0;
    if n0 + NR <= s.n {
        let (mut v0, mut v1) = (v0, v1);
        if !bias.is_null() {
            v0 = f32x4_add(v0, h4(bias.add(n0)));
            v1 = f32x4_add(v1, h4(bias.add(n0 + 4)));
        }
        if act != ACT_NONE {
            v0 = act4(act, v0);
            v1 = act4(act, v1);
        }
        if res {
            v0 = f32x4_add(v0, v128_load(r.add(ro) as *const v128));
            v1 = f32x4_add(v1, v128_load(r.add(ro + 4) as *const v128));
        }
        v128_store(y.add(o) as *mut v128, v0);
        v128_store(y.add(o + 4) as *mut v128, v1);
    } else {
        let mut lanes = [0f32; 8];
        v128_store(lanes.as_mut_ptr() as *mut v128, v0);
        v128_store(lanes.as_mut_ptr().add(4) as *mut v128, v1);
        for (j, l) in lanes.iter().enumerate().take(s.n - n0) {
            let mut v = *l;
            if !bias.is_null() {
                v += hf(bias, n0 + j);
            }
            v = act1(act, v);
            if res {
                v += *r.add(ro + j);
            }
            *y.add(o + j) = v;
        }
    }
}
