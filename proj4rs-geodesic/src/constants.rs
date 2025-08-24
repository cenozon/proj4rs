// Internal constants and coefficient tables for geodesic series.
// These are kept private to the crate and consumed by algorithms/series.
// They are ported from GeographicLib's reference implementation.
//
// Constants matching the original C implementation
pub(crate) const GEOGRAPHICLIB_GEODESIC_ORDER: usize = 6;
pub(crate) const N_A1: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_C1: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_C1P: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_A2: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_C2: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_A3: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_A3X: usize = N_A3;
pub(crate) const N_C3: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
pub(crate) const N_C3X: usize = (N_C3 * (N_C3 - 1)) / 2;
#[cfg(feature = "full-calc")]
pub(crate) const N_C4: usize = GEOGRAPHICLIB_GEODESIC_ORDER;
#[cfg(feature = "full-calc")]
pub(crate) const N_C4X: usize = (N_C4 * (N_C4 + 1)) / 2;
pub(crate) const N_C: usize = GEOGRAPHICLIB_GEODESIC_ORDER + 1; // scratch size
                                                                // Helper consts for fixed-order series
pub(crate) const N_C3M1: usize = N_C3 - 1;
// degrees constants: qd = 90, hd = 180, td = 360
pub(crate) const QD: f64 = 90.0;
pub(crate) const HD: f64 = 180.0;
pub(crate) const TD: f64 = 360.0;
// Static numeric constants (compile-time computable)
pub(crate) const PI: f64 = std::f64::consts::PI;
pub(crate) const DEGREE: f64 = PI / HD;
pub(crate) const TOL0: f64 = f64::EPSILON;
pub(crate) const TOL1: f64 = 200.0 * TOL0;
pub(crate) const TOLB: f64 = TOL0;
pub(crate) const MAXIT1: u32 = 20;
pub(crate) const MAXIT2: u32 = MAXIT1 + f64::MANTISSA_DIGITS + 10;

// Coefficient tables (read-only)
pub(crate) const A1M1F_COEFF: [f64; 5] = [1.0, 4.0, 64.0, 0.0, 256.0];

pub(crate) const C1F_COEFF: [f64; 18] = [
    -1.0, 6.0, -16.0, 32.0, -9.0, 64.0, -128.0, 2048.0, 9.0, -16.0, 768.0, 3.0, -5.0, 512.0, -7.0,
    1280.0, -7.0, 2048.0,
];

pub(crate) const C1PF_COEFF: [f64; 18] = [
    205.0, -432.0, 768.0, 1536.0, 4005.0, -4736.0, 3840.0, 12288.0, -225.0, 116.0, 384.0, -7173.0,
    2695.0, 7680.0, 3467.0, 7680.0, 38081.0, 61440.0,
];

pub(crate) const A2M1F_COEFF: [f64; 5] = [-11.0, -28.0, -192.0, 0.0, 256.0];

pub(crate) const C2F_COEFF: [f64; 18] = [
    1.0, 2.0, 16.0, 32.0, 35.0, 64.0, 384.0, 2048.0, 15.0, 80.0, 768.0, 7.0, 35.0, 512.0, 63.0,
    1280.0, 77.0, 2048.0,
];

pub(crate) const A3_COEFF: [f64; 18] = [
    -3.0, 128.0, -2.0, -3.0, 64.0, -1.0, -3.0, -1.0, 16.0, 3.0, -1.0, -2.0, 8.0, 1.0, -1.0, 2.0,
    1.0, 1.0,
];

pub(crate) const C3_COEFF: [f64; 45] = [
    3.0, 128.0, 2.0, 5.0, 128.0, -1.0, 3.0, 3.0, 64.0, -1.0, 0.0, 1.0, 8.0, -1.0, 1.0, 4.0, 5.0,
    256.0, 1.0, 3.0, 128.0, -3.0, -2.0, 3.0, 64.0, 1.0, -3.0, 2.0, 32.0, 7.0, 512.0, -10.0, 9.0,
    384.0, 5.0, -9.0, 5.0, 192.0, 7.0, 512.0, -14.0, 7.0, 512.0, 21.0, 2560.0,
];

#[cfg(feature = "full-calc")]
pub(crate) const C4_COEFF: [f64; 77] = [
    97.0, 15015.0, 1088.0, 156.0, 45045.0, -224.0, -4784.0, 1573.0, 45045.0, -10656.0, 14144.0,
    -4576.0, -858.0, 45045.0, 64.0, 624.0, -4576.0, 6864.0, -3003.0, 15015.0, 100.0, 208.0, 572.0,
    3432.0, -12012.0, 30030.0, 45045.0, 1.0, 9009.0, -2944.0, 468.0, 135135.0, 5792.0, 1040.0,
    -1287.0, 135135.0, 5952.0, -11648.0, 9152.0, -2574.0, 135135.0, -64.0, -624.0, 4576.0, -6864.0,
    3003.0, 135135.0, 8.0, 10725.0, 1856.0, -936.0, 225225.0, -8448.0, 4992.0, -1144.0, 225225.0,
    -1440.0, 4160.0, -4576.0, 1716.0, 225225.0, -136.0, 63063.0, 1024.0, -208.0, 105105.0, 3584.0,
    -3328.0, 1144.0, 315315.0, -128.0, 135135.0, -2560.0, 832.0, 405405.0, 128.0, 99099.0,
];
