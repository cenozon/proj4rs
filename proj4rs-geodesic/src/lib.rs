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
//! Pure-Rust geodesic calculations ported from the C implementation of GeographicLib.
//!
//! The public API remains:
//! - `Geodesic`: create ellipsoids (e.g., `Geodesic::wgs84()`) and run direct/inverse problems.
//! - `GeodesicLine`: reuse precomputed state for repeated direct evaluations.
//! - `DirectResult` and `InverseResult`: result carriers from the above operations.
mod algorithms;
mod constants;
mod geodesic;
mod line;
mod math;
mod series;
#[cfg(test)]
mod tests;

// Public API
pub use geodesic::{DirectResult, Geodesic, InverseResult};
pub use line::GeodesicLine;
