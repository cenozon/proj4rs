// Small mathematical helpers and angular utilities.
// These mirror the behavior of the C reference implementation.
use crate::constants::{DEGREE, HD, QD, TD};

#[inline(always)]
pub(crate) fn sq(x: f64) -> f64 {
    x * x
}

#[inline]
pub(crate) fn sumx(u: f64, v: f64) -> (f64, f64) {
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
pub(crate) fn polyvalx(n: isize, p: &[f64], x: f64) -> f64 {
    let mut y = if n < 0 { 0.0 } else { p[0] };
    let mut i = 1usize;
    let mut n = n;
    while {
        n -= 1;
        n
    } >= 0
    {
        y = y.mul_add(x, p[i]);
        i += 1;
    }
    y
}

#[inline(always)]
pub(crate) fn norm2(sinx: &mut f64, cosx: &mut f64) {
    let r = sinx.hypot(*cosx);
    *sinx /= r;
    *cosx /= r;
}

#[inline(always)]
pub(crate) fn ang_normalize(x: f64) -> f64 {
    let y = remainder_ieee(x, TD);
    if y.abs() == HD {
        y.copysign(x)
    } else {
        y
    }
}

#[inline(always)]
pub(crate) fn lat_fix(x: f64) -> f64 {
    if x.abs() > QD {
        f64::NAN
    } else {
        x
    }
}

#[inline(always)]
pub(crate) fn ang_diff(x: f64, y: f64) -> (f64, f64) {
    // difference y - x reduced, with error term
    let (d1, t1) = sumx(remainder_ieee(-x, TD), remainder_ieee(y, TD));
    let (mut d, t2) = sumx(remainder_ieee(d1, TD), t1);
    if d == 0.0 || d.abs() == HD {
        d = d.copysign(if t2 == 0.0 { y - x } else { -t2 });
    }
    (d, t2)
}

#[inline(always)]
pub(crate) fn round_ties_to_even(x: f64) -> f64 {
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
pub(crate) fn remainder_ieee(x: f64, y: f64) -> f64 {
    // IEEE-754 style remainder: r = x - y * n with n = round_ties_to_even(x/y)
    let n = round_ties_to_even(x / y);
    x - y * n
}

#[inline(always)]
pub(crate) fn ang_round(x: f64) -> f64 {
    let z = 1.0 / 16.0;
    let mut y = x.abs();
    let w = z - y;
    y = if w > 0.0 { z - w } else { y };
    y.copysign(x)
}

#[inline(always)]
pub(crate) fn remquo90(x: f64) -> (f64, i32) {
    // Emulate C remquo(x, 90deg): r in [-45,45] with ties-to-even quotient
    let q = round_ties_to_even(x / QD);
    let r = x - q * QD;
    (r, (q as i64 & 3) as i32)
}

#[inline(always)]
pub(crate) fn sincosdx(x: f64) -> (f64, f64) {
    let (r0, q) = remquo90(x);
    let r = r0 * DEGREE; // radians
    let (s, c) = r.sin_cos();
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
pub(crate) fn sincosde(x: f64, t: f64) -> (f64, f64) {
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
pub(crate) fn atan2dx(y: f64, mut x: f64) -> f64 {
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
