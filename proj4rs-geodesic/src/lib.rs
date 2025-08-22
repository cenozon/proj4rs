/*
 * Ported from the C implementation of GeographicLib's geodesic algorithms.
 *
 * Original C source notice:
 *
 * This is a C implementation of the geodesic algorithms described in
 *
 *   C. F. F. Karney,
 *   Algorithms for geodesics,
 *   J. Geodesy <b>87</b>, 43--55 (2013);
 *   https://doi.org/10.1007/s00190-012-0578-z
 *   Addenda: https://geographiclib.sourceforge.io/geod-addenda.html
 *
 * See the comments in geodesic.h for documentation.
 *
 * Copyright (c) Charles Karney (2012-2022) <charles@karney.com> and licensed
 * under the MIT/X11 License.  For more information, see
 * https://geographiclib.sourceforge.io/
 */

//! Rust geodesic calculations ported from the C implementation of GeographicLib.
//! This crate previously bound to C; it now provides a pure Rust port
//! implementing the same algorithms and numerical behavior for the
//! direct and inverse geodesic problems on an ellipsoid.

// Constants matching the original C implementation
const GEOGRAPHICLIB_GEODESIC_ORDER: usize = 6;
const N_A1: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_C1: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_C1P: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_A2: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_C2: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_A3: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_A3X: usize = N_A3;
const N_C3: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
const N_C3X: usize = (N_C3 * (N_C3 - 1)) / 2;
#[cfg(feature = "full-calc")]
const N_C4: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
#[cfg(feature = "full-calc")]
const N_C4X: usize = (N_C4 * (N_C4 + 1)) / 2;
const N_C: usize = GEOGRAPHICLIB_GEODESIC_ORDER + 1; // scratch size
                                                     // Helper consts for fixed-order series
const N_C3M1: usize = N_C3 - 1;

// degrees constants: qd = 90, hd = 180, td = 360
const QD: f64 = 90.0;
const HD: f64 = 180.0;
const TD: f64 = 360.0;

// Static numeric constants (compile-time computable)
const PI: f64 = std::f64::consts::PI;
const DEGREE: f64 = PI / HD;
const TOL0: f64 = f64::EPSILON;
const TOL1: f64 = 200.0 * TOL0;
const TOLB: f64 = TOL0;
const MAXIT1: u32 = 20;
const MAXIT2: u32 = MAXIT1 + f64::MANTISSA_DIGITS + 10;

// cached-wgs84: cross-target once cell/lock selection
#[cfg(all(feature = "cached-wgs84", target_arch = "wasm32"))]
use std::cell::OnceCell as WgsCell;
#[cfg(all(feature = "cached-wgs84", not(target_arch = "wasm32")))]
use std::sync::OnceLock as WgsCell;
#[cfg(feature = "cached-wgs84")]
static WGS84: WgsCell<Geodesic> = WgsCell::new();

// Module-level coefficient tables (hoisted from builders)
// These are read-only constants; keeping them here avoids rebuilding arrays.
#[rustfmt::skip]
const A1M1F_COEFF: [f64; 5] = [
    1.0, 4.0, 64.0, 0.0, 256.0,
];

#[rustfmt::skip]
const C1F_COEFF: [f64; 18] = [
    -1.0, 6.0, -16.0, 32.0,
    -9.0, 64.0, -128.0, 2048.0,
    9.0, -16.0, 768.0,
    3.0, -5.0, 512.0,
    -7.0, 1280.0,
    -7.0, 2048.0,
];

#[rustfmt::skip]
const C1PF_COEFF: [f64; 18] = [
    205.0, -432.0, 768.0, 1536.0,
    4005.0, -4736.0, 3840.0, 12288.0,
    -225.0, 116.0, 384.0,
    -7173.0, 2695.0, 7680.0,
    3467.0, 7680.0,
    38081.0, 61440.0,
];

#[rustfmt::skip]
const A2M1F_COEFF: [f64; 5] = [
    -11.0, -28.0, -192.0, 0.0, 256.0,
];

#[rustfmt::skip]
const C2F_COEFF: [f64; 18] = [
    1.0, 2.0, 16.0, 32.0,
    35.0, 64.0, 384.0, 2048.0,
    15.0, 80.0, 768.0,
    7.0, 35.0, 512.0,
    63.0, 1280.0,
    77.0, 2048.0,
];

#[rustfmt::skip]
const A3_COEFF: [f64; 18] = [
    -3.0, 128.0,
    -2.0, -3.0, 64.0,
    -1.0, -3.0, -1.0, 16.0,
    3.0, -1.0, -2.0, 8.0,
    1.0, -1.0, 2.0,
    1.0, 1.0,
];

#[rustfmt::skip]
const C3_COEFF: [f64; 45] = [
    3.0, 128.0, 2.0, 5.0, 128.0,
    -1.0, 3.0, 3.0, 64.0,
    -1.0, 0.0, 1.0, 8.0,
    -1.0, 1.0, 4.0, 5.0, 256.0,
    1.0, 3.0, 128.0, -3.0, -2.0, 3.0, 64.0,
    1.0, -3.0, 2.0, 32.0, 7.0, 512.0,
    -10.0, 9.0, 384.0, 5.0, -9.0, 5.0, 192.0,
    7.0, 512.0, -14.0, 7.0, 512.0, 21.0, 2560.0,
];

#[cfg(feature = "full-calc")]
#[rustfmt::skip]
const C4_COEFF: [f64; 77] = [
    97.0, 15015.0, 1088.0, 156.0, 45045.0,
    -224.0, -4784.0, 1573.0, 45045.0,
    -10656.0, 14144.0, -4576.0, -858.0, 45045.0,
    64.0, 624.0, -4576.0, 6864.0, -3003.0, 15015.0,
    100.0, 208.0, 572.0, 3432.0, -12012.0, 30030.0, 45045.0,
    1.0, 9009.0, -2944.0, 468.0, 135135.0,
    5792.0, 1040.0, -1287.0, 135135.0,
    5952.0, -11648.0, 9152.0, -2574.0, 135135.0,
    -64.0, -624.0, 4576.0, -6864.0, 3003.0, 135135.0,
    8.0, 10725.0, 1856.0, -936.0, 225225.0,
    -8448.0, 4992.0, -1144.0, 225225.0,
    -1440.0, 4160.0, -4576.0, 1716.0, 225225.0,
    -136.0, 63063.0, 1024.0, -208.0, 105105.0,
    3584.0, -3328.0, 1144.0, 315315.0,
    -128.0, 135135.0, -2560.0, 832.0, 405405.0,
    128.0, 99099.0,
];

#[derive(Clone)]
pub struct Geodesic {
    a: f64,
    f: f64,
    f1: f64,
    e2: f64,
    ep2: f64,
    n: f64,
    b: f64,
    c2: f64,
    etol2: f64,
    // Per-instance numeric params
    tiny: f64,
    tol2: f64,
    xthresh: f64,
    a3x: [f64; N_A3X],
    c3x: [f64; N_C3X],
    #[cfg(feature = "full-calc")]
    c4x: [f64; N_C4X],
}

impl std::fmt::Display for Geodesic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Geodesic {{ a: {}, f: {} }}", self.a, self.f)
    }
}
impl std::fmt::Debug for Geodesic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Geodesic {{ a: {}, f: {} }}", self.a, self.f)
    }
}

// No global mutable state; dynamic numeric params are stored per-instance.

#[inline(always)]
fn sq(x: f64) -> f64 {
    x * x
}

#[inline]
fn sumx(u: f64, v: f64) -> (f64, f64) {
    // error-free sum per the C implementation
    let s = u + v;
    let up = s - v;
    let vpp = s - up;
    let up = up - u;
    let vpp = vpp - v;
    let t = if s != 0.0 { -(up + vpp) } else { s };
    (s, t)
}

#[inline(always)]
fn polyvalx(n: isize, p: &[f64], x: f64) -> f64 {
    let mut y = if n < 0 { 0.0 } else { p[0] };
    let mut i = 1usize;
    let mut n = n;
    while { n -= 1; n } >= 0 {
        // Avoid fused multiply-add to mirror C rounding behavior
        y = y * x + p[i];
        i += 1;
    }
    y
}

#[inline(always)]
fn norm2(sinx: &mut f64, cosx: &mut f64) {
    let r = sinx.hypot(*cosx);
    *sinx /= r;
    *cosx /= r;
}

#[inline(always)]
fn ang_normalize(x: f64) -> f64 {
    let y = remainder_ieee(x, TD);
    if y.abs() == HD {
        y.copysign(x)
    } else {
        y
    }
}

#[inline(always)]
fn lat_fix(x: f64) -> f64 {
    if x.abs() > QD {
        f64::NAN
    } else {
        x
    }
}

#[inline(always)]
fn ang_diff(x: f64, y: f64) -> (f64, f64) {
    // difference y - x reduced, with error term
    let (d1, t1) = sumx(remainder_ieee(-x, TD), remainder_ieee(y, TD));
    let (mut d, t2) = sumx(remainder_ieee(d1, TD), t1);
    if d == 0.0 || d.abs() == HD {
        d = d.copysign(if t2 == 0.0 { y - x } else { -t2 });
    }
    (d, t2)
}

#[inline(always)]
fn round_ties_to_even(x: f64) -> f64 {
    // Round to nearest with ties-to-even (like nearbyint with FE_TONEAREST)
    let f = x.floor();
    let d = x - f; // in [0,1)
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else {
        // exactly halfway: choose even
        if (f as i64) & 1 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

#[inline(always)]
fn remainder_ieee(x: f64, y: f64) -> f64 {
    // IEEE-754 style remainder: r = x - y * n with n = round_ties_to_even(x/y)
    let n = round_ties_to_even(x / y);
    x - y * n
}

#[inline(always)]
fn ang_round(x: f64) -> f64 {
    let z = 1.0 / 16.0;
    let mut y = x.abs();
    let w = z - y;
    y = if w > 0.0 { z - w } else { y };
    y.copysign(x)
}

#[inline(always)]
fn remquo90(x: f64) -> (f64, i32) {
    // Emulate C remquo(x, 90deg): r in [-45,45] with ties-to-even quotient
    let q = round_ties_to_even(x / QD);
    let r = x - q * QD;
    (r, (q as i64 & 3) as i32)
}

#[inline(always)]
fn sincosdx(x: f64) -> (f64, f64) {
    let (r0, q) = remquo90(x);
    let r = r0 * DEGREE; // radians
    let s = r.sin();
    let c = r.cos();
    let (mut so, mut co) = match (q as u32) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    // Match C special cases: ensure correct signed zeros
    co += 0.0;
    if so == 0.0 {
        so = so.copysign(x);
    }
    (so, co)
}

#[inline(always)]
fn sincosde(x: f64, t: f64) -> (f64, f64) {
    let (r0, q) = remquo90(x);
    let r = ang_round(r0 + t) * DEGREE;
    let s = r.sin();
    let c = r.cos();
    let (mut so, mut co) = match (q as u32) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    // Match C special cases: ensure correct signed zeros
    co += 0.0;
    if so == 0.0 {
        so = so.copysign(x);
    }
    (so, co)
}

#[inline(always)]
fn atan2dx(y: f64, mut x: f64) -> f64 {
    let mut q = 0i32;
    let mut yy = y;
    if yy.abs() > x.abs() {
        std::mem::swap(&mut x, &mut yy);
        q = 2;
    }
    if x.is_sign_negative() {
        x = -x;
        q += 1;
    }
    let mut ang = yy.atan2(x) / DEGREE;
    match q {
        1 => {
            ang = (HD).copysign(yy) - ang;
        }
        2 => {
            ang = QD - ang;
        }
        3 => {
            ang += -QD;
        }
        _ => {}
    }
    ang
}

// Series helpers (ported)

#[inline(always)]
fn sin_cos_series_const<const N: usize>(sinp: bool, sinx: f64, cosx: f64, c: &[f64]) -> f64 {
    // Clenshaw summation matching the C reference exactly.
    // c is indexed such that c[1..N] (and c[0] for cos series) are used.
    let ar = 2.0 * (cosx - sinx) * (cosx + sinx); // 2*cos(2*x)
                                                  // Pointer to one past the last coefficient used, i.e., c + (N + sinp)
    let mut p = N + if sinp { 1 } else { 0 };
    // Accumulators for the sum
    let mut y0 = if (N & 1) != 0 {
        p -= 1;
        c[p]
    } else {
        0.0
    };
    let mut y1 = 0.0;
    // Now N is even; unroll loop by 2 like the C code
    let mut k = N / 2;
    while k > 0 {
        p -= 1; // --c
        y1 = ar * y0 - y1 + c[p];
        p -= 1; // --c
        y0 = ar * y1 - y0 + c[p];
        k -= 1;
    }
    if sinp {
        2.0 * sinx * cosx * y0
    } else {
        cosx * (y0 - y1)
    }
}

// Forward declarations of coefficient builders
// (forward decls not required in Rust)

#[allow(clippy::too_many_arguments)]
fn lengths(
    g: &Geodesic,
    eps: f64,
    sig12: f64,
    ssig1: f64,
    csig1: f64,
    dn1: f64,
    ssig2: f64,
    csig2: f64,
    dn2: f64,
    cbet1: f64,
    cbet2: f64,
    ps12b: Option<&mut f64>,
    pm12b: Option<&mut f64>,
    pm0: Option<&mut f64>,
    p_m12: Option<&mut f64>,
    p_m21: Option<&mut f64>,
) {
    let mut ca = [0.0f64; N_C];
    let mut cb = [0.0f64; N_C];
    let mut m0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let redlp = pm12b.is_some() || pm0.is_some() || p_m12.is_some() || p_m21.is_some();
    if ps12b.is_some() || redlp {
        a1 = a1m1f(eps);
        c1f(eps, &mut ca);
        if redlp {
            a2 = a2m1f(eps);
            c2f(eps, &mut cb);
            m0 = a1 - a2;
            a2 += 1.0;
        }
        a1 += 1.0;
    }
    let mut j12 = 0.0;
    if let Some(ps12b) = ps12b {
        let b1 = sin_cos_series_const::<N_C1>(true, ssig2, csig2, &ca)
            - sin_cos_series_const::<N_C1>(true, ssig1, csig1, &ca);
        *ps12b = a1 * (sig12 + b1);
        if redlp {
            let b2 = sin_cos_series_const::<N_C2>(true, ssig2, csig2, &cb)
                - sin_cos_series_const::<N_C2>(true, ssig1, csig1, &cb);
            j12 = m0 * sig12 + (a1 * b1 - a2 * b2);
        }
    } else if redlp {
        for l in 1..=N_C2 {
            cb[l] = a1 * ca[l] - a2 * cb[l];
        }
        j12 = m0 * sig12
            + (sin_cos_series_const::<N_C2>(true, ssig2, csig2, &cb)
                - sin_cos_series_const::<N_C2>(true, ssig1, csig1, &cb));
    }
    if let Some(pm0) = pm0 {
        *pm0 = m0;
    }
    if let Some(pm12b) = pm12b {
        *pm12b = dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) - csig1 * csig2 * j12;
    }
    if p_m12.is_some() || p_m21.is_some() {
        let csig12 = csig1 * csig2 + ssig1 * ssig2;
        let t = g.ep2 * (cbet1 - cbet2) * (cbet1 + cbet2) / (dn1 + dn2);
        if let Some(pm12) = p_m12 {
            *pm12 = csig12 + (t * ssig2 - csig2 * j12) * ssig1 / dn1;
        }
        if let Some(pm21) = p_m21 {
            *pm21 = csig12 - (t * ssig1 - csig1 * j12) * ssig2 / dn2;
        }
    }
}

#[cold]
fn astroid(x: f64, y: f64) -> f64 {
    let p = sq(x);
    let q = sq(y);
    let r = (p + q - 1.0) / 6.0;
    if !(q == 0.0 && r <= 0.0) {
        let s = p * q / 4.0; // S = r^3 * s
        let r2 = sq(r);
        let r3 = r * r2;
        let disc = s * (s + 2.0 * r3);
        let mut u = r;
        if disc >= 0.0 {
            let mut t3 = s + r3;
            let root = disc.sqrt();
            t3 += if t3 < 0.0 { -root } else { root };
            let t = t3.cbrt();
            u += t + if t != 0.0 { r2 / t } else { 0.0 };
        } else {
            let ang = (-disc).sqrt().atan2(-(s + r3));
            u += 2.0 * r * (ang / 3.0).cos();
        }
        let v = (sq(u) + q).sqrt();
        let uv = if u < 0.0 { q / (v - u) } else { u + v };
        let w = (uv - q) / (2.0 * v);
        return uv / ((uv + sq(w)).sqrt() + w);
    }
    0.0
}

#[allow(clippy::too_many_arguments)]
fn inverse_start(
    g: &Geodesic,
    sbet1: f64,
    cbet1: f64,
    dn1: f64,
    sbet2: f64,
    cbet2: f64,
    dn2: f64,
    lam12: f64,
    slam12: f64,
    clam12: f64,
    psalp1: &mut f64,
    pcalp1: &mut f64,
    psalp2: &mut f64,
    pcalp2: &mut f64,
    pdnm: &mut f64,
) -> f64 {
    let mut salp1: f64;
    let mut calp1: f64;
    let mut salp2 = 0.0;
    let mut calp2 = 0.0;
    let mut dnm = 0.0;
    let mut sig12 = -1.0;
    let sbet12 = sbet2 * cbet1 - cbet2 * sbet1;
    let cbet12 = cbet2 * cbet1 + sbet2 * sbet1;
    let sbet12a = sbet2 * cbet1 + cbet2 * sbet1;
    let shortline = cbet12 >= 0.0 && sbet12 < 0.5 && cbet2 * lam12 < 0.5;
    let (somg12, comg12) = if shortline {
        let mut sbetm2 = sq(sbet1 + sbet2);
        sbetm2 /= sbetm2 + sq(cbet1 + cbet2);
        dnm = (1.0 + g.ep2 * sbetm2).sqrt();
        let omg12 = lam12 / (g.f1 * dnm);
        (omg12.sin(), omg12.cos())
    } else {
        (slam12, clam12)
    };

    salp1 = cbet2 * somg12;
    calp1 = if comg12 >= 0.0 {
        sbet12 + cbet2 * sbet1 * sq(somg12) / (1.0 + comg12)
    } else {
        sbet12a - cbet2 * sbet1 * sq(somg12) / (1.0 - comg12)
    };

    let ssig12 = salp1.hypot(calp1);
    let csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12;

    if shortline && ssig12 < g.etol2 {
        salp2 = cbet1 * somg12;
        calp2 = sbet12
            - cbet1
                * sbet2
                * if comg12 >= 0.0 {
                    sq(somg12) / (1.0 + comg12)
                } else {
                    1.0 - comg12
                };
        norm2(&mut salp2, &mut calp2);
        sig12 = ssig12.atan2(csig12);
    } else if g.n.abs() > 0.1 || csig12 >= 0.0 || ssig12 >= 6.0 * g.n.abs() * PI * sq(cbet1) {
        // OK
    } else {
        let lam12x = (-slam12).atan2(-clam12);
        let (x, y, lamscale) = if g.f >= 0.0 {
            let k2 = sq(sbet1) * g.ep2;
            let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
            let lamscale = g.f * cbet1 * a3f(g, eps) * PI;
            let betscale = lamscale * cbet1;
            (lam12x / lamscale, sbet12a / betscale, lamscale)
        } else {
            let cbet12a = cbet2 * cbet1 - sbet2 * sbet1;
            let bet12a = sbet12a.atan2(cbet12a);
            let mut m12b = 0.0;
            let mut m0 = 0.0;
            lengths(
                g,
                g.n,
                PI + bet12a,
                sbet1,
                -cbet1,
                dn1,
                sbet2,
                cbet2,
                dn2,
                cbet1,
                cbet2,
                None,
                Some(&mut m12b),
                Some(&mut m0),
                None,
                None,
            );
            let x = -1.0 + m12b / (cbet1 * cbet2 * m0 * PI);
            let betscale = if x < -0.01 {
                sbet12a / x
            } else {
                -g.f * sq(cbet1) * PI
            };
            let lamscale = betscale / cbet1;
            let y = lam12x / lamscale;
            (x, y, lamscale)
        };
        if y > -TOL1 && x > -1.0 - g.xthresh {
            if g.f >= 0.0 {
                salp1 = (-x).min(1.0);
                calp1 = -(1.0 - sq(salp1)).sqrt();
            } else {
                let base = if x > -TOL1 { 0.0 } else { -1.0 };
                calp1 = if base > x { base } else { x };
                salp1 = (1.0 - sq(calp1)).sqrt();
            }
        } else {
            let k = astroid(x, y);
            let omg12 = lamscale
                * (if g.f >= 0.0 {
                    -x * k / (1.0 + k)
                } else {
                    -y * (1.0 + k) / k
                });
            let somg12 = omg12.sin();
            let comg12 = omg12.cos();
            salp1 = cbet2 * somg12;
            calp1 = sbet12
                - cbet2
                    * sbet1
                    * (if comg12 >= 0.0 {
                        sq(somg12) / (1.0 + comg12)
                    } else {
                        1.0 - comg12
                    });
        }
    }

    // Sanity check on starting guess to match C behavior
    if salp1.is_nan() || salp1 > 0.0 {
        norm2(&mut salp1, &mut calp1);
    } else {
        salp1 = 1.0;
        calp1 = 0.0;
    }
    *psalp1 = salp1;
    *pcalp1 = calp1;
    *psalp2 = salp2;
    *pcalp2 = calp2;
    *pdnm = dnm;
    sig12
}

// Lambda12 function is lengthy; port essential parts used by inverse iteration
#[allow(clippy::too_many_arguments)]
fn lambda12(
    g: &Geodesic,
    sbet1: f64,
    cbet1: f64,
    dn1: f64,
    sbet2: f64,
    cbet2: f64,
    dn2: f64,
    salp1: f64,
    mut calp1: f64,
    slam120: f64,
    clam120: f64,
    psalp2: &mut f64,
    pcalp2: &mut f64,
    psig12: &mut f64,
    pssig1: &mut f64,
    pcsig1: &mut f64,
    pssig2: &mut f64,
    pcsig2: &mut f64,
    peps: &mut f64,
    pdomg12: &mut f64,
    diffp: bool,
    pdlam12: &mut f64,
) -> f64 {
    let mut ssig1: f64;
    let mut csig1: f64;
    let mut ssig2: f64;
    let mut csig2: f64;
    let mut dlam12 = 0.0;
    if sbet1 == 0.0 && calp1 == 0.0 {
        calp1 = -g.tiny;
    }
    let salp0 = salp1 * cbet1;
    let calp0 = calp1.hypot(salp1 * sbet1);
    ssig1 = sbet1;
    let somg1 = salp0 * sbet1;
    csig1 = calp1 * cbet1;
    let comg1 = csig1;
    norm2(&mut ssig1, &mut csig1);
    let salp2 = if cbet2 != cbet1 { salp0 / cbet2 } else { salp1 };
    let calp2 = if cbet2 != cbet1 || sbet2.abs() != -sbet1 {
        ((sq(calp1 * cbet1)
            + if cbet1 < -sbet1 {
                (cbet2 - cbet1) * (cbet1 + cbet2)
            } else {
                (sbet1 - sbet2) * (sbet1 + sbet2)
            })
            / sq(cbet2))
        .sqrt()
    } else {
        calp1.abs()
    };
    ssig2 = sbet2;
    let somg2 = salp0 * sbet2;
    csig2 = calp2 * cbet2;
    let comg2 = csig2;
    norm2(&mut ssig2, &mut csig2);
    let sig12 = (0.0f64.max(csig1 * ssig2 - ssig1 * csig2)).atan2(csig1 * csig2 + ssig1 * ssig2);
    let somg12 = 0.0f64.max(comg1 * somg2 - somg1 * comg2);
    let comg12 = comg1 * comg2 + somg1 * somg2;
    let eta = (somg12 * clam120 - comg12 * slam120).atan2(comg12 * clam120 + somg12 * slam120);
    let k2 = sq(calp0) * g.ep2;
    let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
    let mut ca = [0.0f64; N_C];
    c3f(g, eps, &mut ca);
    let b312 = sin_cos_series_const::<N_C3M1>(true, ssig2, csig2, &ca)
        - sin_cos_series_const::<N_C3M1>(true, ssig1, csig1, &ca);
    let domg12 = -g.f * a3f(g, eps) * salp0 * (sig12 + b312);
    let lam12 = eta + domg12;
    if diffp {
        if calp2 == 0.0 {
            dlam12 = -2.0 * g.f1 * dn1 / sbet1;
        } else {
            let mut dlam12_tmp = 0.0;
            lengths(
                g,
                eps,
                sig12,
                ssig1,
                csig1,
                dn1,
                ssig2,
                csig2,
                dn2,
                cbet1,
                cbet2,
                None,
                Some(&mut dlam12_tmp),
                None,
                None,
                None,
            );
            dlam12 = dlam12_tmp * g.f1 / (calp2 * cbet2);
        }
    }
    *psalp2 = salp2;
    *pcalp2 = calp2;
    *psig12 = sig12;
    *pssig1 = ssig1;
    *pcsig1 = csig1;
    *pssig2 = ssig2;
    *pcsig2 = csig2;
    *peps = eps;
    *pdomg12 = domg12;
    if diffp {
        *pdlam12 = dlam12;
    }
    lam12
}

// Coefficient functions (translated closely)
fn a1m1f(eps: f64) -> f64 {
    // (1 - eps)*A1 - 1, polynomial in eps^2 of order 3
    let m = N_A1 / 2;
    let t = polyvalx(m as isize, &A1M1F_COEFF, sq(eps)) / A1M1F_COEFF[m + 1];
    (t + eps) / (1.0 - eps)
}

fn c1f(eps: f64, c: &mut [f64]) {
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, c_l) in c.iter_mut().enumerate().take(N_C1 + 1).skip(1) {
        let m = (N_C1 - l) / 2;
        *c_l = d * polyvalx(m as isize, &C1F_COEFF[o..], eps2) / C1F_COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

fn c1pf(eps: f64, c: &mut [f64]) {
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, c_l) in c.iter_mut().enumerate().take(N_C1P + 1).skip(1) {
        let m = (N_C1P - l) / 2;
        *c_l = d * polyvalx(m as isize, &C1PF_COEFF[o..], eps2) / C1PF_COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

fn a2m1f(eps: f64) -> f64 {
    let m = N_A2 / 2;
    let t = polyvalx(m as isize, &A2M1F_COEFF, sq(eps)) / A2M1F_COEFF[m + 1];
    (t - eps) / (1.0 + eps)
}

fn c2f(eps: f64, c: &mut [f64]) {
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, c_l) in c.iter_mut().enumerate().take(N_C2 + 1).skip(1) {
        let m = (N_C2 - l) / 2;
        *c_l = d * polyvalx(m as isize, &C2F_COEFF[o..], eps2) / C2F_COEFF[o + m + 1];
        o += m + 2;
        d *= eps;
    }
}

fn a3coeff(g: &mut Geodesic) {
    let mut o = 0usize;
    for (k, j) in (0..N_A3).rev().enumerate() {
        let m = std::cmp::min(N_A3 - j - 1, j);
        g.a3x[k] = polyvalx(m as isize, &A3_COEFF[o..], g.n) / A3_COEFF[o + m + 1];
        o += m + 2;
    }
}

fn c3coeff(g: &mut Geodesic) {
    let mut o = 0usize;
    let mut k = 0usize;
    for l in 1..N_C3 {
        // index C3[l]
        for j in (l..N_C3).rev() {
            let m = std::cmp::min(N_C3 - j - 1, j);
            g.c3x[k] = polyvalx(m as isize, &C3_COEFF[o..], g.n) / C3_COEFF[o + m + 1];
            k += 1;
            o += m + 2;
        }
    }
}

#[cfg(feature = "full-calc")]
fn c4coeff(g: &mut Geodesic) {
    let mut o = 0usize;
    let mut k = 0usize;
    for l in 0..N_C4 {
        for j in (l..N_C4).rev() {
            let m = N_C4 - j - 1;
            g.c4x[k] = polyvalx(m as isize, &C4_COEFF[o..], g.n) / C4_COEFF[o + m + 1];
            k += 1;
            o += m + 2;
        }
    }
}

fn a3f(g: &Geodesic, eps: f64) -> f64 {
    // Evaluate A3 using polynomial in eps
    polyvalx((N_A3X - 1) as isize, &g.a3x, eps)
}

fn c3f(g: &Geodesic, eps: f64, c: &mut [f64]) {
    // Evaluate C3 coeffs; c[1..nC3-1] set
    let mut mult = 1.0f64;
    let mut o = 0usize;
    c[0] = 0.0;
    for (l, c_l) in c.iter_mut().enumerate().take(N_C3).skip(1) {
        let m = N_C3 - l - 1;
        mult *= eps;
        *c_l = mult * polyvalx(m as isize, &g.c3x[o..], eps);
        o += m + 1;
    }
}

#[allow(dead_code)]
#[cfg(feature = "full-calc")]
fn c4f(g: &Geodesic, eps: f64, c: &mut [f64]) {
    // Evaluate C4 coeffs; c[0..nC4-1] set
    let mut mult = 1.0f64;
    let mut o = 0usize;
    for (l, c_l) in c.iter_mut().enumerate().take(N_C4) {
        let m = N_C4 - l - 1;
        *c_l = mult * polyvalx(m as isize, &g.c4x[o..], eps);
        o += m + 1;
        mult *= eps;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DirectResult {
    pub lat2: f64,
    pub lon2: f64,
    pub azi2: f64,
    #[cfg(feature = "full-calc")]
    pub m12: f64,
    #[cfg(feature = "full-calc")]
    pub m12_scale: f64,
    #[cfg(feature = "full-calc")]
    pub m21_scale: f64,
    #[cfg(feature = "full-calc")]
    pub s12_area: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct InverseResult {
    pub s12: f64,
    pub azi1: f64,
    pub azi2: f64,
    #[cfg(feature = "full-calc")]
    pub m12: f64,
    #[cfg(feature = "full-calc")]
    pub m12_scale: f64,
    #[cfg(feature = "full-calc")]
    pub m21_scale: f64,
    #[cfg(feature = "full-calc")]
    pub s12_area: f64,
}

impl Geodesic {
    pub fn new(a: f64, f: f64) -> Self {
        let mut g = Geodesic {
            a,
            f,
            f1: 0.0,
            e2: 0.0,
            ep2: 0.0,
            n: 0.0,
            b: 0.0,
            c2: 0.0,
            etol2: 0.0,
            tiny: 0.0,
            tol2: 0.0,
            xthresh: 0.0,
            a3x: [0.0; N_A3X],
            c3x: [0.0; N_C3X],
            #[cfg(feature = "full-calc")]
            c4x: [0.0; N_C4X],
        };
        g.f1 = 1.0 - g.f;
        g.e2 = g.f * (2.0 - g.f);
        g.ep2 = g.e2 / sq(g.f1);
        g.n = g.f / (2.0 - g.f);
        g.b = g.a * g.f1;
        g.c2 = (sq(g.a)
            + sq(g.b) * {
                if g.e2 == 0.0 {
                    1.0
                } else {
                    let t = g.e2.abs().sqrt();
                    (if g.e2 > 0.0 {
                        (g.e2).sqrt().atanh()
                    } else {
                        (-g.e2).sqrt().atan()
                    }) / t
                }
            })
            / 2.0;
        // Per-instance numeric params
        g.tiny = f64::MIN_POSITIVE.sqrt();
        g.tol2 = TOL0.sqrt();
        g.xthresh = 1000.0 * g.tol2;
        g.etol2 =
            0.1 * g.tol2 / ((g.f.abs().max(0.001) * (1.0f64).min(1.0 - g.f / 2.0) / 2.0).sqrt());
        a3coeff(&mut g);
        c3coeff(&mut g);
        #[cfg(feature = "full-calc")]
        c4coeff(&mut g);
        g
    }

    pub fn wgs84() -> Self {
        const A: f64 = 6_378_137.0;
        const F: f64 = 1.0 / 298.257_223_563;
        Self::new(A, F)
    }

    #[cfg(feature = "cached-wgs84")]
    pub fn wgs84_ref() -> &'static Geodesic {
        WGS84.get_or_init(|| Self::wgs84())
    }

    pub fn inverse(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> InverseResult {
        let (s12, azi1, azi2) = self.geninverse(lat1, lon1, lat2, lon2);
        #[cfg(feature = "full-calc")]
        {
            let line = self.line(lat1, lon1, azi1);
            let dr = line.position_distance(s12);
            InverseResult {
                s12,
                azi1,
                azi2,
                m12: dr.m12,
                m12_scale: dr.m12_scale,
                m21_scale: dr.m21_scale,
                s12_area: dr.s12_area,
            }
        }
        #[cfg(not(feature = "full-calc"))]
        {
            InverseResult { s12, azi1, azi2 }
        }
    }

    pub fn direct(&self, lat1: f64, lon1: f64, azi1: f64, s12: f64) -> DirectResult {
        let line = self.line(lat1, lon1, azi1);
        line.position_distance(s12)
    }

    // Internal helpers: a minimal port of geninverse and gendirect used by API
    fn geninverse(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64, f64) {
        let (mut lon12, mut lon12s) = ang_diff(lon1, lon2);
        let lonsign = if lon12.is_sign_negative() { -1.0 } else { 1.0 };
        lon12 *= lonsign;
        lon12s *= lonsign;
        let lam12 = lon12 * DEGREE;
        let (slam12, clam12) = sincosde(lon12, lon12s);
        lon12s = (HD - lon12) - lon12s;

        let mut lat1 = ang_round(lat_fix(lat1));
        let mut lat2 = ang_round(lat_fix(lat2));
        let swapp = if lat1.abs() < lat2.abs() || lat2.is_nan() {
            -1.0
        } else {
            1.0
        };
        let mut lonsign_i = lonsign;
        if swapp < 0.0 {
            lonsign_i *= -1.0;
            std::mem::swap(&mut lat1, &mut lat2);
        }
        let latsign = if lat1.is_sign_negative() { 1.0 } else { -1.0 };
        lat1 *= latsign;
        lat2 *= latsign;

        let (mut sbet1, mut cbet1) = sincosdx(lat1);
        sbet1 *= self.f1;
        norm2(&mut sbet1, &mut cbet1);
        cbet1 = cbet1.max(self.tiny);
        let (mut sbet2, mut cbet2) = sincosdx(lat2);
        sbet2 *= self.f1;
        norm2(&mut sbet2, &mut cbet2);
        cbet2 = cbet2.max(self.tiny);
        if cbet1 < -sbet1 {
            if cbet2 == cbet1 {
                sbet2 = sbet1.copysign(sbet2);
            }
        } else if sbet2.abs() == -sbet1 {
            cbet2 = cbet1;
        }
        let dn1 = (1.0 + self.ep2 * sq(sbet1)).sqrt();
        let dn2 = (1.0 + self.ep2 * sq(sbet2)).sqrt();

        let mut meridian = lat1 == -QD || slam12 == 0.0;
        let mut s12x = 0.0;
        let mut salp1 = 0.0;
        let mut calp1 = 0.0;
        let mut salp2 = 0.0;
        let mut calp2 = 0.0;
        let mut sig12 = 0.0;
        if meridian {
            let ssig1 = sbet1;
            let csig1 = cbet1 * clam12;
            let ssig2 = sbet2;
            let csig2 = 1.0 * cbet2;
            salp1 = slam12;
            calp1 = clam12;
            salp2 = 0.0;
            calp2 = 1.0;
            sig12 =
                (0.0f64.max(csig1 * ssig2 - ssig1 * csig2)).atan2(csig1 * csig2 + ssig1 * ssig2);
            let mut s12b = 0.0;
            let mut m12b = 0.0;
            lengths(
                self,
                self.n,
                sig12,
                ssig1,
                csig1,
                dn1,
                ssig2,
                csig2,
                dn2,
                cbet1,
                cbet2,
                Some(&mut s12b),
                Some(&mut m12b),
                None,
                None,
                None,
            );
            s12x = s12b;
            let m12x = m12b;
            if sig12 < 1.0 || m12x >= 0.0 {
                if sig12 < 3.0 * self.tiny || (sig12 < TOL0 && (s12x < 0.0 || m12x < 0.0)) {
                    sig12 = 0.0;
                    s12x = 0.0;
                }
                s12x *= self.b;
            } else {
                // m12 < 0, i.e., prolate and too close to antipodal; fall back
                meridian = false;
            }
        }
        if !meridian && sbet1 == 0.0 && (self.f <= 0.0 || lon12s >= self.f * HD) {
            salp1 = 1.0;
            calp1 = 0.0;
            salp2 = 1.0;
            calp2 = 0.0;
            s12x = self.a * lam12;
        } else if !meridian {
            let mut dnm = 0.0;
            let mut salp1t = 0.0;
            let mut calp1t = 0.0;
            let mut salp2t = 0.0;
            let mut calp2t = 0.0;
            let sig12_guess = inverse_start(
                self,
                sbet1,
                cbet1,
                dn1,
                sbet2,
                cbet2,
                dn2,
                lam12,
                slam12,
                clam12,
                &mut salp1t,
                &mut calp1t,
                &mut salp2t,
                &mut calp2t,
                &mut dnm,
            );
            if sig12_guess >= 0.0 {
                s12x = sig12_guess * self.b * dnm;
                salp1 = salp1t;
                calp1 = calp1t;
                salp2 = salp2t;
                calp2 = calp2t;
                // sig12 not needed for outputs in this branch
            } else {
                // Newton iteration port (simplified but robust)
                let mut ssig1 = 0.0;
                let mut csig1 = 0.0;
                let mut ssig2 = 0.0;
                let mut csig2 = 0.0;
                let mut eps = 0.0;
                let mut domg12 = 0.0;
                let mut numit = 0u32;
                let mut salp1a = self.tiny;
                let mut calp1a = 1.0;
                let mut salp1b = self.tiny;
                let mut calp1b = -1.0;
                let mut tripn = false;
                let mut tripb = false;
                salp1 = salp1t;
                calp1 = calp1t;
                loop {
                    let mut dv = 0.0;
                    let v = lambda12(
                        self,
                        sbet1,
                        cbet1,
                        dn1,
                        sbet2,
                        cbet2,
                        dn2,
                        salp1,
                        calp1,
                        slam12,
                        clam12,
                        &mut salp2,
                        &mut calp2,
                        &mut sig12,
                        &mut ssig1,
                        &mut csig1,
                        &mut ssig2,
                        &mut csig2,
                        &mut eps,
                        &mut domg12,
                        numit < MAXIT1,
                        &mut dv,
                    );
                    if tripb
                        || (v.abs().is_nan() || v.abs() < (if tripn { 8.0 } else { 1.0 }) * TOL0)
                        || numit == MAXIT2
                    {
                        break;
                    }
                    if v > 0.0 && (numit > MAXIT1 || calp1 / salp1 > calp1b / salp1b) {
                        salp1b = salp1;
                        calp1b = calp1;
                    } else if v < 0.0 && (numit > MAXIT1 || calp1 / salp1 < calp1a / salp1a) {
                        salp1a = salp1;
                        calp1a = calp1;
                    }
                    if numit < MAXIT1 && dv > 0.0 {
                        let dalp1 = -v / dv;
                        if dalp1.abs() < PI {
                            let (sd, cd) = dalp1.sin_cos();
                            let nsalp1 = salp1 * cd + calp1 * sd;
                            if nsalp1 > 0.0 {
                                calp1 = calp1 * cd - salp1 * sd;
                                salp1 = nsalp1;
                                norm2(&mut salp1, &mut calp1);
                                tripn = v.abs() <= 16.0 * TOL0;
                                numit += 1;
                                continue;
                            }
                        }
                    }
                    salp1 = 0.5 * (salp1a + salp1b);
                    calp1 = 0.5 * (calp1a + calp1b);
                    norm2(&mut salp1, &mut calp1);
                    tripn = false;
                    tripb = ((salp1a - salp1).abs() + (calp1a - calp1)) < TOLB
                        || ((salp1 - salp1b).abs() + (calp1 - calp1b)) < TOLB;
                    numit += 1;
                }
                let mut s12b = 0.0;
                let mut m12b = 0.0;
                lengths(
                    self,
                    eps,
                    sig12,
                    ssig1,
                    csig1,
                    dn1,
                    ssig2,
                    csig2,
                    dn2,
                    cbet1,
                    cbet2,
                    Some(&mut s12b),
                    Some(&mut m12b),
                    None,
                    None,
                    None,
                );
                s12x = s12b * self.b;
            }
        }

        // Apply swap and sign adjustments like C implementation
        if swapp < 0.0 {
            std::mem::swap(&mut salp1, &mut salp2);
            std::mem::swap(&mut calp1, &mut calp2);
        }
        salp1 *= swapp * lonsign_i;
        calp1 *= swapp * latsign;
        salp2 *= swapp * lonsign_i;
        calp2 *= swapp * latsign;
        // Output s12, azi1, azi2
        let s12 = 0.0 + s12x; // convert -0 to 0
        let az1 = atan2dx(salp1, calp1);
        let az2 = atan2dx(salp2, calp2);
        (s12, az1, az2)
    }
}

// Public line type for repeated direct evaluations with shared precomputation
pub struct GeodesicLine<'a> {
    g: &'a Geodesic,
    lon1: f64,
    salp0: f64,
    calp0: f64,
    ssig1: f64,
    csig1: f64,
    k2: f64,
    a1m1: f64,
    a3c: f64,
    c1a: [f64; N_C],
    c1pa: [f64; N_C],
    c3a: [f64; N_C],
    b11: f64,
    stau1: f64,
    ctau1: f64,
}

impl Geodesic {
    pub fn line<'a>(&'a self, lat1: f64, lon1: f64, azi1: f64) -> GeodesicLine<'a> {
        let azi1n = ang_normalize(azi1);
        let (salp1, calp1) = sincosdx(ang_round(azi1n));
        let lat1f = ang_round(lat_fix(lat1));
        let (mut sbet1, mut cbet1) = sincosdx(lat1f);
        sbet1 *= self.f1;
        norm2(&mut sbet1, &mut cbet1);
        cbet1 = cbet1.max(self.tiny);
        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);
        let mut ssig1 = sbet1;
        let mut csig1 = if sbet1 != 0.0 || calp1 != 0.0 {
            cbet1 * calp1
        } else {
            1.0
        };
        norm2(&mut ssig1, &mut csig1);
        let k2 = sq(calp0) * self.ep2;
        let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
        let a1m1 = a1m1f(eps);
        let mut c1a = [0.0f64; N_C];
        c1f(eps, &mut c1a);
        let b11 = sin_cos_series_const::<N_C1>(true, ssig1, csig1, &c1a);
        let (stau1, ctau1) = {
            let s = b11.sin();
            let c = b11.cos();
            (ssig1 * c + csig1 * s, csig1 * c - ssig1 * s)
        };
        let mut c1pa = [0.0f64; N_C];
        c1pf(eps, &mut c1pa);
        let mut c3a = [0.0f64; N_C];
        c3f(self, eps, &mut c3a);
        let a3c = -self.f * salp0 * a3f(self, eps);

        GeodesicLine {
            g: self,
            lon1,
            salp0,
            calp0,
            ssig1,
            csig1,
            k2,
            a1m1,
            a3c,
            c1a,
            c1pa,
            c3a,
            b11,
            stau1,
            ctau1,
        }
    }
}

impl<'a> GeodesicLine<'a> {
    pub fn position_distance(&self, s12: f64) -> DirectResult {
        let tau12 = s12 / (self.g.b * (1.0 + self.a1m1));
        let (s_t12, c_t12) = tau12.sin_cos();
        let b12 = -sin_cos_series_const::<N_C1P>(
            true,
            self.stau1 * c_t12 + self.ctau1 * s_t12,
            self.ctau1 * c_t12 - self.stau1 * s_t12,
            &self.c1pa,
        );
        let mut sig12 = tau12 - (b12 - self.b11);
        let mut ssig12 = sig12.sin();
        let mut csig12 = sig12.cos();
        if self.g.f.abs() > 0.01 {
            let ssig2 = self.ssig1 * csig12 + self.csig1 * ssig12;
            let csig2 = self.csig1 * csig12 - self.ssig1 * ssig12;
            let b12b = sin_cos_series_const::<N_C1>(true, ssig2, csig2, &self.c1a);
            let serr = (1.0 + self.a1m1) * (sig12 + (b12b - self.b11)) - s12 / self.g.b;
            sig12 -= serr / (1.0 + self.k2 * sq(ssig2)).sqrt();
            ssig12 = sig12.sin();
            csig12 = sig12.cos();
        }
        let ssig2 = self.ssig1 * csig12 + self.csig1 * ssig12;
        let mut csig2 = self.csig1 * csig12 - self.ssig1 * ssig12;
        let sbet2 = self.calp0 * ssig2;
        let mut cbet2 = self.salp0.hypot(self.calp0 * csig2);
        if cbet2 == 0.0 {
            cbet2 = self.g.tiny;
            csig2 = self.g.tiny;
        }
        let salp2 = self.salp0;
        let calp2 = self.calp0 * csig2;
        let lat2 = atan2dx(sbet2, self.g.f1 * cbet2);
        let somg1 = self.salp0 * self.ssig1;
        let comg1 = self.csig1;
        let somg2 = self.salp0 * ssig2;
        let comg2 = csig2;
        let omg12 = (somg2 * comg1 - comg2 * somg1).atan2(comg2 * comg1 + somg2 * somg1);
        let lam12 = omg12
            + self.a3c
                * (sig12
                    + (sin_cos_series_const::<N_C3M1>(true, ssig2, csig2, &self.c3a)
                        - sin_cos_series_const::<N_C3M1>(true, self.ssig1, self.csig1, &self.c3a)));
        let lon2 = ang_normalize(ang_normalize(self.lon1) + ang_normalize(lam12 / DEGREE));
        let azi2 = atan2dx(salp2, calp2);
        #[cfg(feature = "full-calc")]
        {
            // Reduced length, scales, and area
            let sbet1 = self.calp0 * self.ssig1;
            let dn1 = (1.0 + self.g.ep2 * sq(sbet1)).sqrt();
            let dn2 = (1.0 + self.g.ep2 * sq(sbet2)).sqrt();
            let mut m12b = 0.0f64;
            let mut m12_scale = 0.0f64;
            let mut m21_scale = 0.0f64;
            // cbet1 and eps derived from line state
            let cbet1 = self.salp0.hypot(self.calp0 * self.csig1);
            let eps = self.k2 / (2.0 * (1.0 + (1.0 + self.k2).sqrt()) + self.k2);
            lengths(
                self.g,
                eps,
                sig12,
                self.ssig1,
                self.csig1,
                dn1,
                ssig2,
                csig2,
                dn2,
                cbet1,
                cbet2,
                None,
                Some(&mut m12b),
                None,
                Some(&mut m12_scale),
                Some(&mut m21_scale),
            );
            let mut c4a = [0.0f64; N_C4];
            c4f(self.g, eps, &mut c4a);
            let b41 = sin_cos_series_const::<N_C4>(false, self.ssig1, self.csig1, &c4a);
            let b42 = sin_cos_series_const::<N_C4>(false, ssig2, csig2, &c4a);
            let a4 = sq(self.g.a) * self.g.e2 * self.calp0 * self.salp0;
            let csig12 = self.csig1 * csig2 + self.ssig1 * ssig2;
            let ssig12l = ssig12;
            let (salp12, calp12) = if self.calp0 == 0.0 || self.salp0 == 0.0 {
                // Recover (salp1, calp1) from line state
                let salp1 = self.salp0 / cbet1;
                let calp1 = self.csig1 / cbet1;
                (salp2 * calp1 - calp2 * salp1, calp2 * calp1 + salp2 * salp1)
            } else {
                let salp12 = self.calp0
                    * self.salp0
                    * (if csig12 <= 0.0 {
                        self.csig1 * (1.0 - csig12) + ssig12l * self.ssig1
                    } else {
                        ssig12l * (self.csig1 * ssig12l / (1.0 + csig12) + self.ssig1)
                    });
                let calp12 = sq(self.salp0) + sq(self.calp0) * self.csig1 * csig2;
                (salp12, calp12)
            };
            DirectResult {
                lat2,
                lon2,
                azi2,
                m12: m12b * self.g.b,
                m12_scale,
                m21_scale,
                s12_area: self.g.c2 * (salp12.atan2(calp12)) + a4 * (b42 - b41),
            }
        }
        #[cfg(not(feature = "full-calc"))]
        {
            DirectResult { lat2, lon2, azi2 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Geodesic;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn approx_ang(a: f64, b: f64) -> bool {
        approx(a, b, 1e-12)
    }

    fn approx_m(a: f64, b: f64) -> bool {
        // Functional parity: allow 1 mm absolute tolerance
        approx(a, b, 1e-3)
    }

    #[test]
    fn roundtrip_equator() {
        let g = {
            #[cfg(feature = "cached-wgs84")]
            {
                Geodesic::wgs84_ref().clone()
            }
            #[cfg(not(feature = "cached-wgs84"))]
            {
                Geodesic::wgs84()
            }
        };
        let lat1 = 0.0;
        let lon1 = 0.0;
        let s12 = 1_000_000.0; // 1000 km
        let azi1 = 90.0;
        let dr = g.direct(lat1, lon1, azi1, s12);
        let ir = g.inverse(lat1, lon1, dr.lat2, dr.lon2);
        assert!(approx_m(ir.s12, s12), "{} {}", ir.s12, s12);
        assert!(approx_ang(ir.azi1, azi1));
        assert!(approx_ang(dr.azi2, ir.azi2));
    }

    #[test]
    fn near_antipodal_strip_and_astroid() {
        let g = {
            #[cfg(feature = "cached-wgs84")]
            {
                Geodesic::wgs84_ref().clone()
            }
            #[cfg(not(feature = "cached-wgs84"))]
            {
                Geodesic::wgs84()
            }
        };
        // Slightly off antipodal to trigger astroid/strip logic
        let lat1 = 10.0;
        let lon1 = -20.0;
        let lat2 = -10.0;
        let lon2 = 160.0 + 1.0e-9;
        let ir = g.inverse(lat1, lon1, lat2, lon2);
        let dr = g.direct(lat1, lon1, ir.azi1, ir.s12);
        // roundtrip distance
        let ir_rt = g.inverse(lat1, lon1, dr.lat2, dr.lon2);
        assert!(
            (ir_rt.s12 - ir.s12).abs() <= 0.5,
            "|Δs|={} m",
            (ir_rt.s12 - ir.s12).abs()
        );
        // closeness to target within decimeter
        let ir_res = g.inverse(dr.lat2, dr.lon2, lat2, lon2);
        assert!(ir_res.s12 <= 0.1, "residual={} m", ir_res.s12);
    }

    #[test]
    fn very_short_lines() {
        let g = {
            #[cfg(feature = "cached-wgs84")]
            {
                Geodesic::wgs84_ref().clone()
            }
            #[cfg(not(feature = "cached-wgs84"))]
            {
                Geodesic::wgs84()
            }
        };
        let lat1 = 23.0;
        let lon1 = -45.0;
        let azi1 = 12.0;
        let s12 = 1e-3; // 1 mm
        let dr = g.direct(lat1, lon1, azi1, s12);
        let ir = g.inverse(lat1, lon1, dr.lat2, dr.lon2);
        assert!(approx_m(ir.s12, s12));
    }

    #[test]
    fn poles_and_meridians() {
        let g = {
            #[cfg(feature = "cached-wgs84")]
            {
                Geodesic::wgs84_ref().clone()
            }
            #[cfg(not(feature = "cached-wgs84"))]
            {
                Geodesic::wgs84()
            }
        };
        for &(lat1, lon1, azi1) in &[(90.0, 10.0, 180.0), (-90.0, -20.0, 0.0), (0.0, 0.0, 0.0)] {
            let s12 = 500_000.0; // 500 km
            let dr = g.direct(lat1, lon1, azi1, s12);
            let ir = g.inverse(lat1, lon1, dr.lat2, dr.lon2);
            assert!(approx_m(ir.s12, s12));
        }
    }

    #[test]
    fn prolate_ellipsoid_cases() {
        // Prolate: f < 0; use modest eccentricity to exercise alt branches
        let g = Geodesic::new(6_378_137.0, -1.0 / 150.0);
        let lat1 = 30.0;
        let lon1 = 10.0;
        let lat2 = -22.0;
        let lon2 = 170.0;
        let ir = g.inverse(lat1, lon1, lat2, lon2);
        let dr = g.direct(lat1, lon1, ir.azi1, ir.s12);
        let ir_rt = g.inverse(lat1, lon1, dr.lat2, dr.lon2);
        assert!(
            (ir_rt.s12 - ir.s12).abs() <= 5.0,
            "|Δs|={} m",
            (ir_rt.s12 - ir.s12).abs()
        );
        let ir_res = g.inverse(dr.lat2, dr.lon2, lat2, lon2);
        assert!(ir_res.s12 <= 5.0, "residual={} m", ir_res.s12);
    }
    #[test]
    fn dist_az_test() {
        struct TestCase {
            pub lat1: f64,
            pub lon1: f64,
            pub azi1: f64,
            pub lat2: f64,
            pub lon2: f64,
            pub azi2: f64,
            pub s12: f64,
            //pub a12: f64,
            //pub m12: f64,
            //pub mm12: f64, // M12
            //pub mm21: f64, // M21
            //pub ss12: f64, // S12
        }
        impl TestCase {
            fn vec(v: &[f64]) -> Self {
                Self {
                    lat1: v[0],
                    lon1: v[1],
                    azi1: v[2],
                    lat2: v[3],
                    lon2: v[4],
                    azi2: v[5],
                    s12: v[6],
                    //a12: v[7],
                    //m12: v[8],
                    //mm12: v[9],
                    //mm21: v[10],
                    //ss12: v[11],
                }
            }
        }

        let testcases = [
            TestCase::vec(&[
                35.60777,
                -139.44815,
                111.098748429560326,
                -11.17491,
                -69.95921,
                129.289270889708762,
                8935244.5604818305,
                80.50729714281974,
                6273170.2055303837,
                0.16606318447386067,
                0.16479116945612937,
                12841384694976.432,
            ]),
            TestCase::vec(&[
                55.52454,
                106.05087,
                22.020059880982801,
                77.03196,
                197.18234,
                109.112041110671519,
                4105086.1713924406,
                36.892740690445894,
                3828869.3344387607,
                0.80076349608092607,
                0.80101006984201008,
                61674961290615.615,
            ]),
            TestCase::vec(&[
                -21.97856,
                142.59065,
                -32.44456876433189,
                41.84138,
                98.56635,
                -41.84359951440466,
                8394328.894657671,
                75.62930491011522,
                6161154.5773110616,
                0.24816339233950381,
                0.24930251203627892,
                -6637997720646.717,
            ]),
            TestCase::vec(&[
                -66.99028,
                112.2363,
                173.73491240878403,
                -12.70631,
                285.90344,
                2.512956620913668,
                11150344.2312080241,
                100.278634181155759,
                6289939.5670446687,
                -0.17199490274700385,
                -0.17722569526345708,
                -121287239862139.744,
            ]),
            TestCase::vec(&[
                -17.42761,
                173.34268,
                -159.033557661192928,
                -15.84784,
                5.93557,
                -20.787484651536988,
                16076603.1631180673,
                144.640108810286253,
                3732902.1583877189,
                -0.81273638700070476,
                -0.81299800519154474,
                97825992354058.708,
            ]),
            TestCase::vec(&[
                32.84994,
                48.28919,
                150.492927788121982,
                -56.28556,
                202.29132,
                48.113449399816759,
                16727068.9438164461,
                150.565799985466607,
                3147838.1910180939,
                -0.87334918086923126,
                -0.86505036767110637,
                -72445258525585.010,
            ]),
            TestCase::vec(&[
                6.96833,
                52.74123,
                92.581585386317712,
                -7.39675,
                206.17291,
                90.721692165923907,
                17102477.2496958388,
                154.147366239113561,
                2772035.6169917581,
                -0.89991282520302447,
                -0.89986892177110739,
                -1311796973197.995,
            ]),
            TestCase::vec(&[
                -50.56724,
                -16.30485,
                -105.439679907590164,
                -33.56571,
                -94.97412,
                -47.348547835650331,
                6455670.5118668696,
                58.083719495371259,
                5409150.7979815838,
                0.53053508035997263,
                0.52988722644436602,
                41071447902810.047,
            ]),
            TestCase::vec(&[
                -58.93002,
                -8.90775,
                140.965397902500679,
                -8.91104,
                133.13503,
                19.255429433416599,
                11756066.0219864627,
                105.755691241406877,
                6151101.2270708536,
                -0.26548622269867183,
                -0.27068483874510741,
                -86143460552774.735,
            ]),
            TestCase::vec(&[
                -68.82867,
                -74.28391,
                93.774347763114881,
                -50.63005,
                -8.36685,
                34.65564085411343,
                3956936.926063544,
                35.572254987389284,
                3708890.9544062657,
                0.81443963736383502,
                0.81420859815358342,
                -41845309450093.787,
            ]),
            TestCase::vec(&[
                -10.62672,
                -32.0898,
                -86.426713286747751,
                5.883,
                -134.31681,
                -80.473780971034875,
                11470869.3864563009,
                103.387395634504061,
                6184411.6622659713,
                -0.23138683500430237,
                -0.23155097622286792,
                4198803992123.548,
            ]),
            TestCase::vec(&[
                -21.76221,
                166.90563,
                29.319421206936428,
                48.72884,
                213.97627,
                43.508671946410168,
                9098627.3986554915,
                81.963476716121964,
                6299240.9166992283,
                0.13965943368590333,
                0.14152969707656796,
                10024709850277.476,
            ]),
            TestCase::vec(&[
                -19.79938,
                -174.47484,
                71.167275780171533,
                -11.99349,
                -154.35109,
                65.589099775199228,
                2319004.8601169389,
                20.896611684802389,
                2267960.8703918325,
                0.93427001867125849,
                0.93424887135032789,
                -3935477535005.785,
            ]),
            TestCase::vec(&[
                -11.95887,
                -116.94513,
                92.712619830452549,
                4.57352,
                7.16501,
                78.64960934409585,
                13834722.5801401374,
                124.688684161089762,
                5228093.177931598,
                -0.56879356755666463,
                -0.56918731952397221,
                -9919582785894.853,
            ]),
            TestCase::vec(&[
                -87.85331,
                85.66836,
                -65.120313040242748,
                66.48646,
                16.09921,
                -4.888658719272296,
                17286615.3147144645,
                155.58592449699137,
                2635887.4729110181,
                -0.90697975771398578,
                -0.91095608883042767,
                42667211366919.534,
            ]),
            TestCase::vec(&[
                1.74708,
                128.32011,
                -101.584843631173858,
                -11.16617,
                11.87109,
                -86.325793296437476,
                12942901.1241347408,
                116.650512484301857,
                5682744.8413270572,
                -0.44857868222697644,
                -0.44824490340007729,
                10763055294345.653,
            ]),
            TestCase::vec(&[
                -25.72959,
                -144.90758,
                -153.647468693117198,
                -57.70581,
                -269.17879,
                -48.343983158876487,
                9413446.7452453107,
                84.664533838404295,
                6356176.6898881281,
                0.09492245755254703,
                0.09737058264766572,
                74515122850712.444,
            ]),
            TestCase::vec(&[
                -41.22777,
                122.32875,
                14.285113402275739,
                -7.57291,
                130.37946,
                10.805303085187369,
                3812686.035106021,
                34.34330804743883,
                3588703.8812128856,
                0.82605222593217889,
                0.82572158200920196,
                -2456961531057.857,
            ]),
            TestCase::vec(&[
                11.01307,
                138.25278,
                79.43682622782374,
                6.62726,
                247.05981,
                103.708090215522657,
                11911190.819018408,
                107.341669954114577,
                6070904.722786735,
                -0.29767608923657404,
                -0.29785143390252321,
                17121631423099.696,
            ]),
            TestCase::vec(&[
                -29.47124,
                95.14681,
                -163.779130441688382,
                -27.46601,
                -69.15955,
                -15.909335945554969,
                13487015.8381145492,
                121.294026715742277,
                5481428.9945736388,
                -0.51527225545373252,
                -0.51556587964721788,
                104679964020340.318,
            ]),
        ];
        let g = {
            #[cfg(feature = "cached-wgs84")]
            {
                crate::Geodesic::wgs84_ref().clone()
            }
            #[cfg(not(feature = "cached-wgs84"))]
            {
                crate::Geodesic::wgs84()
            }
        };
        for (i, t) in testcases.iter().enumerate() {
            let ir = g.inverse(t.lat1, t.lon1, t.lat2, t.lon2);
            if (ir.s12 - t.s12).abs() >= 1e-8
                || (ir.azi1 - t.azi1).abs() >= 1e-13
                || (ir.azi2 - t.azi2).abs() >= 1e-13
            {
                panic!(
                    "case {} mismatch: s={} exp={} Δs={}; azi1={} exp={} Δ1={}; azi2={} exp={} Δ2={}",
                    i, ir.s12, t.s12, ir.s12 - t.s12, ir.azi1, t.azi1, ir.azi1 - t.azi1, ir.azi2, t.azi2, ir.azi2 - t.azi2
                );
            }
        }
        let ir = g.inverse(0.0, 0.0, 0.0, 10.0);
        let s0 = 1113194.9079327357;
        assert!((ir.s12 - s0).abs() < 1e-5, "{} {}", ir.s12, s0);
        assert_eq!(ir.azi1, 90.0);
        assert_eq!(ir.azi2, 90.0);
    }

    #[test]
    fn test_debug() {
        use crate::Geodesic;
        let g = Geodesic::new(6_378_145.0, 1.0 / 298.25);
        assert_eq!(
            format!("{}", g),
            "Geodesic { a: 6378145, f: 0.003352891869237217 }"
        );
    }
}
