// Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. Every load
// reads row m of A as K contiguous values: a linear layer's row, a convolution's k input rows
// from row m (each input row has cin channels), or a transposed convolution's input rows m and
// m + 1, its two taps swapped as the weights are packed. A thread dequantizes NR columns of W at
// a time into a panel of f32 in its scratch, [K][NR], then runs tiles of MR rows by NR columns
// over it, with the epilogue in SIMD. The tile's shape is the fastest of a few on this device
// (cpu.ts times them once).
//
// Fields: x, w, ws (0: f16 weights), y, M, K, N, act, bias, res, r, yo, ro, load, cin, stride,
// gate, state.
use crate::math::*;
use crate::ops::*;
use crate::pool::{scratch, scratch_bytes, share, tile_shape};
use core::arch::wasm32::*;

const LINEAR: u32 = 0;
const CONVT: u32 = 2;

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
    /// Columns per panel: 8 or 16, or 4 for the dot-product tiles.
    nr: usize,
    /// Dot-product tiles: the panel is [4][K], each column's weights in a row.
    dot: bool,
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
        nr: 8,
        dot: false,
    };
    let (dot, mr, nr) = tile_shape();
    let dot = dot && k % 4 == 0;
    let wide = nr == 16 && n >= 64 && k * 16 * 4 <= scratch_bytes();
    let nr = if dot { nr.min(8) } else if wide { 16 } else { 8 };
    let s = Shape { nr, dot, ..s };
    let nr = s.nr;
    let blocks = n.div_ceil(nr);
    let units = phases * blocks;
    // Rows split too when there are few column blocks, at least 16 rows each.
    let mparts = (4 * nt).div_ceil(units).min(m.div_ceil(16)).max(1);
    let tasks = units * mparts;
    let (t0, t1) = share(tasks, i, nt, 1);
    let panel = scratch(i) as *mut f32;
    if s.k * nr * 4 > scratch_bytes() {
        return;
    }
    let mut packed = usize::MAX;
    for t in t0..t1 {
        let (unit, part) = (t / mparts, t % mparts);
        let (ph, nb) = (unit / blocks, unit % blocks);
        if unit != packed {
            pack(op, &s, panel, ph, nb * nr);
            packed = unit;
        }
        let (m0, m1) = share(m, part, mparts, 4);
        run(op, &s, mr, panel, ph, nb * nr, m0, m1);
    }
}

/// Columns n0..n0 + NR of phase `ph` of W as f32, [K][NR], zero past N.
unsafe fn pack(op: *const u32, s: &Shape, panel: *mut f32, ph: usize, n0: usize) {
    let (w8, ws, w16) = (wp::<i8>(op, 2), wp::<u16>(op, 3), wp::<u16>(op, 2));
    let nr = s.nr;
    // A's index kk reads weight index src(kk): a transposed convolution's taps swap halves.
    let src = |kk: usize| if s.convt { if kk < s.cin { kk + s.cin } else { kk - s.cin } } else { kk };
    if s.dot {
        for j in 0..nr {
            let n = n0 + j;
            let row = panel.add(j * s.k);
            if n >= s.n {
                core::ptr::write_bytes(row, 0, s.k);
            } else if ws.is_null() || s.k % 16 != 0 {
                for kk in 0..s.k {
                    let at = n * s.wrow + ph * s.k + src(kk);
                    *row.add(kk) = if ws.is_null() { hf(w16, at) } else { *w8.add(at) as f32 * hf(ws, at / 32) };
                }
            } else {
                for kk in (0..s.k).step_by(16) {
                    let at = n * s.wrow + ph * s.k + src(kk);
                    let sc = f32x4_splat(hf(ws, at / 32));
                    let (a, b, c, d) = i8x16_f32(v128_load(w8.add(at) as *const v128));
                    v128_store(row.add(kk) as *mut v128, f32x4_mul(a, sc));
                    v128_store(row.add(kk + 4) as *mut v128, f32x4_mul(b, sc));
                    v128_store(row.add(kk + 8) as *mut v128, f32x4_mul(c, sc));
                    v128_store(row.add(kk + 12) as *mut v128, f32x4_mul(d, sc));
                }
            }
        }
        return;
    }
    if ws.is_null() || s.k % 16 != 0 {
        for j in 0..nr {
            let n = n0 + j;
            for kk in 0..s.k {
                *panel.add(kk * nr + j) = if n >= s.n {
                    0.0
                } else {
                    let at = n * s.wrow + ph * s.k + src(kk);
                    if ws.is_null() {
                        hf(w16, at)
                    } else {
                        *w8.add(at) as f32 * hf(ws, at / 32)
                    }
                };
            }
        }
        return;
    }
    // Four columns and 16 values of K at a time, transposed in registers.
    for j in (0..nr).step_by(4) {
        let mut kk = 0;
        while kk < s.k {
            let mut c = [[f32x4_splat(0.0); 4]; 4];
            for (q, cq) in c.iter_mut().enumerate() {
                let n = n0 + j + q;
                if n < s.n {
                    let at = n * s.wrow + ph * s.k + src(kk);
                    let sc = f32x4_splat(hf(ws, at / 32));
                    let (a, b, cc, d) = i8x16_f32(v128_load(w8.add(at) as *const v128));
                    *cq = [f32x4_mul(a, sc), f32x4_mul(b, sc), f32x4_mul(cc, sc), f32x4_mul(d, sc)];
                }
            }
            for g in 0..4 {
                let (r0, r1, r2, r3) = transpose(c[0][g], c[1][g], c[2][g], c[3][g]);
                let p = panel.add((kk + 4 * g) * nr + j);
                v128_store(p as *mut v128, r0);
                v128_store(p.add(nr) as *mut v128, r1);
                v128_store(p.add(2 * nr) as *mut v128, r2);
                v128_store(p.add(3 * nr) as *mut v128, r3);
            }
            kk += 16;
        }
    }
}

#[inline(always)]
fn transpose(a: v128, b: v128, c: v128, d: v128) -> (v128, v128, v128, v128) {
    let ab0 = i32x4_shuffle::<0, 4, 1, 5>(a, b);
    let ab1 = i32x4_shuffle::<2, 6, 3, 7>(a, b);
    let cd0 = i32x4_shuffle::<0, 4, 1, 5>(c, d);
    let cd1 = i32x4_shuffle::<2, 6, 3, 7>(c, d);
    (
        i64x2_shuffle::<0, 2>(ab0, cd0),
        i64x2_shuffle::<1, 3>(ab0, cd0),
        i64x2_shuffle::<0, 2>(ab1, cd1),
        i64x2_shuffle::<1, 3>(ab1, cd1),
    )
}

/// Rows m0..m1 against one panel, in tiles of `mr` rows (then 4, then 1).
unsafe fn run(op: *const u32, s: &Shape, mr: usize, panel: *const f32, ph: usize, n0: usize, m0: usize, m1: usize) {
    let x = wp::<f32>(op, 1);
    let mut m = m0;
    if s.dot {
        if s.nr == 8 {
            m = dots::<2, 8>(op, s, x, panel, ph, n0, m, m1);
            dots::<1, 8>(op, s, x, panel, ph, n0, m, m1);
            return;
        }
        match mr {
            6 => m = dots::<6, 4>(op, s, x, panel, ph, n0, m, m1),
            4 => m = dots::<4, 4>(op, s, x, panel, ph, n0, m, m1),
            _ => {}
        }
        m = dots::<2, 4>(op, s, x, panel, ph, n0, m, m1);
        dots::<1, 4>(op, s, x, panel, ph, n0, m, m1);
        return;
    }
    if s.nr == 16 {
        match mr {
            6 => m = rows::<6, 4>(op, s, x, panel, ph, n0, m, m1),
            _ => {}
        }
        m = rows::<4, 4>(op, s, x, panel, ph, n0, m, m1);
        rows::<1, 4>(op, s, x, panel, ph, n0, m, m1);
        return;
    }
    match mr {
        8 => m = rows::<8, 2>(op, s, x, panel, ph, n0, m, m1),
        6 => m = rows::<6, 2>(op, s, x, panel, ph, n0, m, m1),
        _ => {}
    }
    m = rows::<4, 2>(op, s, x, panel, ph, n0, m, m1);
    rows::<1, 2>(op, s, x, panel, ph, n0, m, m1);
}

/// Whole tiles of MR rows by 4 NB columns from m, while they fit before m1; answers the first
/// row left.
#[inline(always)]
unsafe fn rows<const MR: usize, const NB: usize>(
    op: *const u32,
    s: &Shape,
    x: *const f32,
    panel: *const f32,
    ph: usize,
    n0: usize,
    mut m: usize,
    m1: usize,
) -> usize {
    while m + MR <= m1 {
        let c = tile::<MR, NB>(x.add(m * s.astride), s.astride, panel, s.k);
        for (r, cr) in c.iter().enumerate() {
            for (b, v) in cr.iter().enumerate() {
                store(op, s, ph, m + r, n0 + 4 * b, *v);
            }
        }
        m += MR;
    }
    m
}

/// Dot-product tiles of MR rows by NC columns from m, while they fit before m1: each column's
/// weights are a row of the panel, and the sums along K stay in lanes until the tile's end.
#[inline(always)]
unsafe fn dots<const MR: usize, const NC: usize>(
    op: *const u32,
    s: &Shape,
    x: *const f32,
    panel: *const f32,
    ph: usize,
    n0: usize,
    mut m: usize,
    m1: usize,
) -> usize {
    let k = s.k;
    while m + MR <= m1 {
        let a = x.add(m * s.astride);
        let mut c = [[f32x4_splat(0.0); NC]; MR];
        let mut kk = 0;
        while kk < k {
            let mut b = [f32x4_splat(0.0); NC];
            for (q, bq) in b.iter_mut().enumerate() {
                *bq = v128_load(panel.add(q * k + kk) as *const v128);
            }
            for (r, cr) in c.iter_mut().enumerate() {
                let av = v128_load(a.add(r * s.astride + kk) as *const v128);
                for (q, cq) in cr.iter_mut().enumerate() {
                    *cq = madd(av, b[q], *cq);
                }
            }
            kk += 4;
        }
        for (r, cr) in c.iter().enumerate() {
            for q in (0..NC).step_by(4) {
                store(op, s, ph, m + r, n0 + q, sums(cr[q], cr[q + 1], cr[q + 2], cr[q + 3]));
            }
        }
        m += MR;
    }
    m
}

/// The lane sums of four vectors, as one vector.
#[inline(always)]
fn sums(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let ab = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(a, b), i32x4_shuffle::<2, 6, 3, 7>(a, b));
    let cd = f32x4_add(i32x4_shuffle::<0, 4, 1, 5>(c, d), i32x4_shuffle::<2, 6, 3, 7>(c, d));
    f32x4_add(i64x2_shuffle::<0, 2>(ab, cd), i64x2_shuffle::<1, 3>(ab, cd))
}

/// MR rows of A (rows `st` apart) times the [K][4 NB] panel.
#[inline(always)]
unsafe fn tile<const MR: usize, const NB: usize>(a: *const f32, st: usize, panel: *const f32, k: usize) -> [[v128; NB]; MR] {
    let mut c = [[f32x4_splat(0.0); NB]; MR];
    let nr = 4 * NB;
    let step = |c: &mut [[v128; NB]; MR], kk: usize| {
        let mut b = [f32x4_splat(0.0); NB];
        for (q, bq) in b.iter_mut().enumerate() {
            *bq = v128_load(panel.add(kk * nr + 4 * q) as *const v128);
        }
        for (r, cr) in c.iter_mut().enumerate() {
            let av = v128_load32_splat(a.add(r * st + kk) as *const u32);
            for (q, cq) in cr.iter_mut().enumerate() {
                *cq = madd(av, b[q], *cq);
            }
        }
    };
    let mut kk = 0;
    while kk + 2 <= k {
        step(&mut c, kk);
        step(&mut c, kk + 1);
        kk += 2;
    }
    if kk < k {
        step(&mut c, kk);
    }
    c
}

/// Bias, activation and residual for row m, columns n0..n0 + 4, then the store.
#[inline(always)]
unsafe fn store(op: *const u32, s: &Shape, ph: usize, m: usize, n0: usize, v: v128) {
    if n0 >= s.n {
        return;
    }
    let orow = m * s.phases + ph;
    let (y, r) = (wp::<f32>(op, 4), wp::<f32>(op, 11));
    let o = wu(op, 12) + orow * s.n + n0;
    let ro = wu(op, 13) + orow * s.n + n0;
    let bias = wp::<u16>(op, 9);
    let act = w(op, 8);
    let res = w(op, 10) != 0;
    if n0 + 4 <= s.n {
        let mut v = v;
        if !bias.is_null() {
            v = f32x4_add(v, h4(bias.add(n0)));
        }
        if act != ACT_NONE {
            v = act4(act, v);
        }
        if res {
            v = f32x4_add(v, v128_load(r.add(ro) as *const v128));
        }
        v128_store(y.add(o) as *mut v128, v);
    } else {
        let mut lanes = [0f32; 4];
        v128_store(lanes.as_mut_ptr() as *mut v128, v);
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
