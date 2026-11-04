// SIMD helpers: f16 to f32, exp, the activations, and f64 sine, cosine and log for the
// rotary angles and the noise, so the module needs no imports.
use core::arch::wasm32::*;

/// a * b + c: one fused instruction with relaxed SIMD, else a multiply and an add.
#[inline(always)]
pub fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(f32x4_mul(a, b), c)
    }
}

#[inline(always)]
pub fn hsum(v: v128) -> f32 {
    let s = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let s = f32x4_add(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
    f32x4_extract_lane::<0>(s)
}

#[inline(always)]
pub fn hmax(v: v128) -> f32 {
    let s = f32x4_max(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let s = f32x4_max(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
    f32x4_extract_lane::<0>(s)
}

const TWO_112: u32 = 0x7780_0000;

/// An f16 value: its magnitude bits moved into an f32 and scaled by 2^112, which is exact for
/// normal and subnormal values alike.
#[inline(always)]
pub fn h2f(h: u16) -> f32 {
    let f = f32::from_bits(((h as u32) & 0x7fff) << 13) * f32::from_bits(TWO_112);
    if h & 0x8000 != 0 {
        -f
    } else {
        f
    }
}

#[inline(always)]
pub unsafe fn hf(p: *const u16, i: usize) -> f32 {
    h2f(*p.add(i))
}

/// Four f16 values from `p`.
#[inline(always)]
pub unsafe fn h4(p: *const u16) -> v128 {
    let v = u32x4_load_extend_u16x4(p);
    let mag = i32x4_shl(v128_and(v, u32x4_splat(0x7fff)), 13);
    let f = f32x4_mul(mag, f32x4_splat(f32::from_bits(TWO_112)));
    v128_or(f, i32x4_shl(v128_and(v, u32x4_splat(0x8000)), 16))
}

/// The 16 int8 values of `b` as four f32x4.
#[inline(always)]
pub fn i8x16_f32(b: v128) -> (v128, v128, v128, v128) {
    let lo = i16x8_extend_low_i8x16(b);
    let hi = i16x8_extend_high_i8x16(b);
    (
        f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo)),
        f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo)),
        f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi)),
        f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi)),
    )
}

/// e^x for each lane (Cephes' expf: about 1 ulp), for x from -87 up.
#[inline(always)]
pub fn exp4(x: v128) -> v128 {
    let x = f32x4_min(f32x4_max(x, f32x4_splat(-87.3)), f32x4_splat(88.37));
    let n = f32x4_nearest(f32x4_mul(x, f32x4_splat(core::f32::consts::LOG2_E)));
    let r = f32x4_sub(x, f32x4_mul(n, f32x4_splat(0.693_359_4)));
    let r = f32x4_add(r, f32x4_mul(n, f32x4_splat(2.121_944_4e-4)));
    let mut p = f32x4_splat(1.987_569_1e-4);
    p = madd(p, r, f32x4_splat(1.398_2e-3));
    p = madd(p, r, f32x4_splat(8.333_452e-3));
    p = madd(p, r, f32x4_splat(4.166_579_6e-2));
    p = madd(p, r, f32x4_splat(1.666_666_5e-1));
    p = madd(p, r, f32x4_splat(0.5));
    let r2 = f32x4_mul(r, r);
    let y = f32x4_add(madd(p, r2, r), f32x4_splat(1.0));
    let e = i32x4_shl(i32x4_add(i32x4_trunc_sat_f32x4(n), i32x4_splat(127)), 23);
    f32x4_mul(y, e)
}

#[inline(always)]
pub fn exp1(x: f32) -> f32 {
    f32x4_extract_lane::<0>(exp4(f32x4_splat(x)))
}

/// GELU with the erf of Numerical Recipes' erfc (relative error under 1.2e-7), as the WGSL
/// kernels compute it.
#[inline(always)]
pub fn gelu4(v: v128) -> v128 {
    let z = f32x4_mul(v, f32x4_splat(0.707_106_77));
    let a = f32x4_abs(z);
    let t = f32x4_div(f32x4_splat(1.0), madd(a, f32x4_splat(0.5), f32x4_splat(1.0)));
    let mut p = f32x4_splat(0.170_872_77);
    p = madd(p, t, f32x4_splat(-0.822_152_23));
    p = madd(p, t, f32x4_splat(1.488_515_9));
    p = madd(p, t, f32x4_splat(-1.135_203_98));
    p = madd(p, t, f32x4_splat(0.278_868_07));
    p = madd(p, t, f32x4_splat(-0.186_288_06));
    p = madd(p, t, f32x4_splat(0.096_784_18));
    p = madd(p, t, f32x4_splat(0.374_091_96));
    p = madd(p, t, f32x4_splat(1.000_023_7));
    p = madd(p, t, f32x4_splat(-1.265_512_2));
    let r = f32x4_mul(t, exp4(f32x4_sub(p, f32x4_mul(a, a))));
    let one = f32x4_splat(1.0);
    let erf = v128_bitselect(f32x4_sub(one, r), f32x4_sub(r, one), f32x4_ge(z, f32x4_splat(0.0)));
    f32x4_mul(f32x4_mul(f32x4_splat(0.5), v), f32x4_add(one, erf))
}

#[inline(always)]
pub fn silu4(v: v128) -> v128 {
    f32x4_div(v, f32x4_add(f32x4_splat(1.0), exp4(f32x4_neg(v))))
}

#[inline(always)]
pub fn elu4(v: v128) -> v128 {
    let e = f32x4_sub(exp4(v), f32x4_splat(1.0));
    v128_bitselect(v, e, f32x4_gt(v, f32x4_splat(0.0)))
}

pub const ACT_NONE: u32 = 0;
pub const ACT_GELU: u32 = 1;
pub const ACT_SILU: u32 = 2;

#[inline(always)]
pub fn act4(act: u32, v: v128) -> v128 {
    match act {
        ACT_GELU => gelu4(v),
        ACT_SILU => silu4(v),
        _ => v,
    }
}

#[inline(always)]
pub fn act1(act: u32, v: f32) -> f32 {
    f32x4_extract_lane::<0>(act4(act, f32x4_splat(v)))
}

#[inline(always)]
fn round(x: f64) -> f64 {
    f64_nearest(x)
}

/// sin and cos of x (|x| up to about 1e6), to about 1e-16: Cody-Waite reduction by pi/2, then
/// Taylor series on [-pi/4, pi/4].
pub fn sincos(x: f64) -> (f64, f64) {
    const PIO2_1: f64 = 1.570_796_326_734_125_6;
    const PIO2_1T: f64 = 6.077_100_506_506_192e-11;
    let q = round(x * core::f64::consts::FRAC_2_PI);
    let r = (x - q * PIO2_1) - q * PIO2_1T;
    let r2 = r * r;
    let mut s = SIN[7];
    for k in (0..7).rev() {
        s = s * r2 + SIN[k];
    }
    let mut c = COS[8];
    for k in (0..8).rev() {
        c = c * r2 + COS[k];
    }
    let s = r * s;
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Taylor coefficients of sin r / r and cos r in r^2.
const SIN: [f64; 8] = [1.0, -0.16666666666666666, 0.008333333333333333, -0.0001984126984126984, 2.7557319223985893e-06, -2.505210838544172e-08, 1.6059043836821613e-10, -7.647163731819816e-13];
const COS: [f64; 9] = [1.0, -0.5, 0.041666666666666664, -0.001388888888888889, 2.48015873015873e-05, -2.755731922398589e-07, 2.08767569878681e-09, -1.1470745597729725e-11, 4.779477332387385e-14];

/// ln x for x > 0, to about 1e-16: x = m 2^e with m in [sqrt(1/2), sqrt(2)), then the atanh
/// series of (m - 1) / (m + 1).
pub fn ln(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut t = 0.0;
    let mut k = 25.0;
    while k >= 1.0 {
        t = 1.0 / k + s2 * t;
        k -= 2.0;
    }
    e as f64 * core::f64::consts::LN_2 + 2.0 * s * t
}

#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    f32_sqrt(x)
}
