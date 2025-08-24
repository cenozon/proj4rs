// Series summations and coefficient builders used by geodesic calculations.
// The layout mirrors the C implementation to ensure identical numerics.
use crate::constants::*;
use crate::geodesic::Geodesic;
use crate::math::{polyvalx, sq};

// Clenshaw summation for sine/cosine series
#[inline(always)]
pub(crate) fn sin_cos_series_const<const N: usize>(
    sinp: bool,
    sinx: f64,
    cosx: f64,
    c: &[f64],
) -> f64 {
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

// Coefficient functions (translated closely)
pub(crate) fn a1m1f(eps: f64) -> f64 {
    // (1 - eps)*A1 - 1, polynomial in eps^2 of order 3
    let m = N_A1 / 2;
    let t = polyvalx(m as isize, &A1M1F_COEFF, sq(eps)) / A1M1F_COEFF[m + 1];
    (t + eps) / (1.0 - eps)
}

pub(crate) fn c1f(eps: f64, c: &mut [f64]) {
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

pub(crate) fn c1pf(eps: f64, c: &mut [f64]) {
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

pub(crate) fn a2m1f(eps: f64) -> f64 {
    let m = N_A2 / 2;
    let t = polyvalx(m as isize, &A2M1F_COEFF, sq(eps)) / A2M1F_COEFF[m + 1];
    (t - eps) / (1.0 + eps)
}

pub(crate) fn c2f(eps: f64, c: &mut [f64]) {
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

pub(crate) fn a3coeff(g: &mut Geodesic) {
    let mut o = 0usize;
    for (k, j) in (0..N_A3).rev().enumerate() {
        let m = std::cmp::min(N_A3 - j - 1, j);
        g.a3x[k] = polyvalx(m as isize, &A3_COEFF[o..], g.n) / A3_COEFF[o + m + 1];
        o += m + 2;
    }
}

pub(crate) fn c3coeff(g: &mut Geodesic) {
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
pub(crate) fn c4coeff(g: &mut Geodesic) {
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

pub(crate) fn a3f(g: &Geodesic, eps: f64) -> f64 {
    // Evaluate A3 using polynomial in eps
    polyvalx((N_A3X - 1) as isize, &g.a3x, eps)
}

pub(crate) fn c3f(g: &Geodesic, eps: f64, c: &mut [f64]) {
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

#[cfg(feature = "full-calc")]
pub(crate) fn c4f(g: &Geodesic, eps: f64, c: &mut [f64]) {
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
