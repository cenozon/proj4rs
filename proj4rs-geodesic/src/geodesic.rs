//! Geodesic ellipsoid with direct and inverse problem solvers.
//!
//! Construct with `Geodesic::new(a, f)` or `Geodesic::wgs84()`. Use:
//! - `inverse(lat1, lon1, lat2, lon2)` for distance and azimuths.
//! - `direct(lat1, lon1, azi1, s12)` for destination point given distance.
//! - `line(lat1, lon1, azi1)` to build a reusable `GeodesicLine` for repeated direct solves.
use std::fmt::{Debug, Display};

use crate::algorithms::{inverse_start, lambda12, lengths};
use crate::constants::*;
use crate::math::{ang_diff, ang_round, atan2dx, lat_fix, norm2, sincosde, sincosdx, sq};
#[cfg(feature = "full-calc")]
use crate::series::c4coeff;
use crate::series::{a3coeff, c3coeff};

/// Result of a direct geodesic computation.
///
/// Returned by `Geodesic::direct` and by `GeodesicLine::position_distance`.
/// Values are in degrees for angles and meters for distances.
#[derive(Copy, Clone, Debug)]
pub struct DirectResult {
    /// Latitude of the second point (degrees).
    pub lat2: f64,
    /// Longitude of the second point (degrees) normalized to [-180, 180).
    pub lon2: f64,
    /// Forward azimuth at the second point (degrees).
    pub azi2: f64,
    /// Reduced length (meters).
    #[cfg(feature = "full-calc")]
    pub m12: f64,
    /// Geodesic scale M12 (dimensionless).
    #[cfg(feature = "full-calc")]
    pub m12_scale: f64,
    /// Geodesic scale M21 (dimensionless).
    #[cfg(feature = "full-calc")]
    pub m21_scale: f64,
    /// Area between the geodesic and equator (m^2).
    #[cfg(feature = "full-calc")]
    pub s12_area: f64,
}

/// Result of an inverse geodesic computation.
///
/// Returned by `Geodesic::inverse`.
/// Values are in degrees for angles and meters for distances.
#[derive(Copy, Clone, Debug)]
pub struct InverseResult {
    /// Geodesic distance between the points (meters).
    pub s12: f64,
    /// Forward azimuth at the first point (degrees).
    pub azi1: f64,
    /// Forward azimuth at the second point (degrees).
    pub azi2: f64,
    /// Reduced length (meters). Present with `full-calc` feature.
    #[cfg(feature = "full-calc")]
    pub m12: f64,
    /// Geodesic scale M12 (dimensionless). Present with `full-calc` feature.
    #[cfg(feature = "full-calc")]
    pub m12_scale: f64,
    /// Geodesic scale M21 (dimensionless). Present with `full-calc` feature.
    #[cfg(feature = "full-calc")]
    pub m21_scale: f64,
    /// Area between the geodesic and equator (m^2). Present with `full-calc` feature.
    #[cfg(feature = "full-calc")]
    pub s12_area: f64,
}

/// Ellipsoidal model and precomputed coefficients for geodesic calculations.
///
/// - `a`: semi-major axis (meters).
/// - `f`: flattening; prolate ellipsoids are supported via negative `f`.
#[derive(Clone)]
pub struct Geodesic {
    pub(crate) a: f64,
    pub(crate) f: f64,
    pub(crate) f1: f64,
    pub(crate) e2: f64,
    pub(crate) ep2: f64,
    pub(crate) n: f64,
    pub(crate) b: f64,
    pub(crate) c2: f64,
    pub(crate) etol2: f64,
    // Per-instance numeric params
    pub(crate) tiny: f64,
    pub(crate) tol2: f64,
    pub(crate) xthresh: f64,
    pub(crate) a3x: [f64; N_A3X],
    pub(crate) c3x: [f64; N_C3X],
    #[cfg(feature = "full-calc")]
    pub(crate) c4x: [f64; N_C4X],
}

impl Display for Geodesic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Geodesic {{ a: {}, f: {} }}", self.a, self.f)
    }
}

impl Debug for Geodesic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Geodesic {{ a: {}, f: {} }}", self.a, self.f)
    }
}

impl Geodesic {
    /// Create a new ellipsoid with semi-major axis `a` (meters) and flattening `f`.
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

    /// WGS84 ellipsoid parameters (semi-major axis and flattening).
    pub fn wgs84() -> Self {
        const A: f64 = 6_378_137.0;
        const F: f64 = 1.0 / 298.257_223_563;
        Self::new(A, F)
    }

    /// Solve the inverse geodesic problem.
    ///
    /// Given two points `(lat1, lon1)` and `(lat2, lon2)` in degrees, returns
    /// the geodesic distance and forward azimuths at each point.
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

    /// Solve the direct geodesic problem.
    ///
    /// Given a start point `(lat1, lon1)` in degrees, a forward azimuth `azi1`
    /// in degrees, and a distance `s12` in meters, returns the destination point
    /// and forward azimuth at destination.
    pub fn direct(&self, lat1: f64, lon1: f64, azi1: f64, s12: f64) -> DirectResult {
        let line = self.line(lat1, lon1, azi1);
        line.position_distance(s12)
    }

    // Internal helpers: a minimal port of geninverse used by API
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
