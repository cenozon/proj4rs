// Core algorithmic helpers for the geodesic calculations.
// These functions are internal and used by Geodesic/GeodesicLine implementations.
use crate::constants::*;
use crate::geodesic::Geodesic;
use crate::math::{norm2, sq};
use crate::series::{a1m1f, a2m1f, a3f, c1f, c2f, c3f, sin_cos_series_const};

#[allow(clippy::too_many_arguments)]
pub(crate) fn lengths(
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
pub(crate) fn astroid(x: f64, y: f64) -> f64 {
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
pub(crate) fn inverse_start(
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
pub(crate) fn lambda12(
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
