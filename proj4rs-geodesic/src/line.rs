//! Geodesic line with precomputed state for repeated direct evaluations.
use crate::constants::*;
use crate::geodesic::{DirectResult, Geodesic};
use crate::math::{ang_normalize, ang_round, atan2dx, lat_fix, norm2, sincosdx, sq};
use crate::series::{a3f, c1f, c1pf, c3f, sin_cos_series_const};
#[cfg(feature = "full-calc")]
use crate::algorithms::lengths;
#[cfg(feature = "full-calc")]
use crate::series::c4f;


/// A line on the ellipsoid defined by a start point and azimuth.
///
/// Use `Geodesic::line` to construct and `position_distance` to evaluate points
/// by distance along the line.
pub struct GeodesicLine<'a> {
    pub(crate) g: &'a Geodesic,
    pub(crate) lon1: f64,
    pub(crate) salp0: f64,
    pub(crate) calp0: f64,
    pub(crate) ssig1: f64,
    pub(crate) csig1: f64,
    pub(crate) k2: f64,
    pub(crate) a1m1: f64,
    pub(crate) a3c: f64,
    pub(crate) c1a: [f64; N_C],
    pub(crate) c1pa: [f64; N_C],
    pub(crate) c3a: [f64; N_C],
    pub(crate) b11: f64,
    pub(crate) stau1: f64,
    pub(crate) ctau1: f64,
}

impl<'a> GeodesicLine<'a> {
    /// Compute the position obtained by moving `s12` meters along the line.
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
        let lon2 = crate::math::ang_normalize(
            crate::math::ang_normalize(self.lon1) + crate::math::ang_normalize(lam12 / DEGREE),
        );
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

impl Geodesic {
    /// Build a `GeodesicLine` from `lat1, lon1` and forward azimuth `azi1` in degrees.
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
        let a1m1 = crate::series::a1m1f(eps);
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
