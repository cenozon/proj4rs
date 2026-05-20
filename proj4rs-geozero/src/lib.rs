//! Streaming coordinate reprojection adaptor between [`proj4rs`] and
//! [`geozero`].
//!
//! This crate exposes [`StreamingReproject`], a [`GeomProcessor`] wrapper that
//! reprojects every emitted coordinate from a source CRS to a destination CRS
//! via [`proj4rs::transform`] and forwards the transformed coordinate to an
//! inner processor. No intermediate `geo_types::Geometry` (or any other
//! materialised geometry container) is allocated — coordinates are
//! reprojected as they flow through the geozero callback stream.
//!
//! Typical usage drives a WKB reader on one end and a tile/byte writer on the
//! other:
//!
//! ```ignore
//! use proj4rs::Proj;
//! use proj4rs_geozero::StreamingReproject;
//! use geozero::wkb::Wkb;
//! use geozero::GeozeroGeometry;
//!
//! let src = Proj::from_proj_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs").unwrap();
//! let dst = Proj::from_proj_string(
//!     "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 \
//!      +units=m +nadgrids=@null +wktext +no_defs",
//! )
//! .unwrap();
//!
//! let mut writer = /* any GeomProcessor receiving projected coords */;
//! let mut reproj = StreamingReproject::new(&src, &dst, &mut writer);
//! let wkb = Wkb(&blob);
//! wkb.process_geom(&mut reproj).unwrap();
//! ```
//!
//! Errors from the underlying [`proj4rs`] transform (e.g. a coordinate that
//! falls outside the destination CRS's valid domain) surface as
//! [`GeozeroError::Geometry`] with the proj4rs error rendered into the
//! message.

use geozero::error::{GeozeroError, Result};
use geozero::{CoordDimensions, GeomProcessor};
use proj4rs::adaptors::{transform_xy, transform_xyz};
use proj4rs::errors::Error as ProjError;
use proj4rs::Proj;

/// Wrap an inner [`GeomProcessor`] so that every coordinate it receives has
/// first been reprojected from `src` to `dst` via [`proj4rs`].
///
/// The wrapper is held by `&mut` reference to the inner processor, so the
/// caller retains ownership and can inspect the inner state after streaming
/// completes.
///
/// All non-coordinate callbacks (begin/end pairs for points, line strings,
/// polygons, collections, curves, surfaces, triangles, and the SRID hook) are
/// forwarded verbatim to the inner processor.
pub struct StreamingReproject<'a, P: GeomProcessor> {
    src: &'a Proj,
    dst: &'a Proj,
    inner: &'a mut P,
}

impl<'a, P: GeomProcessor> StreamingReproject<'a, P> {
    /// Build a new streaming reprojector.
    ///
    /// `src` is the CRS of the coordinates emitted by whatever drives this
    /// processor (e.g. the WKB reader). `dst` is the CRS the inner processor
    /// expects to receive.
    pub fn new(src: &'a Proj, dst: &'a Proj, inner: &'a mut P) -> Self {
        Self { src, dst, inner }
    }

    fn project_xy(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        transform_xy(self.src, self.dst, x, y).map_err(proj_err_to_geozero)
    }

    fn project_xyz(&self, x: f64, y: f64, z: f64) -> Result<(f64, f64, f64)> {
        transform_xyz(self.src, self.dst, x, y, z).map_err(proj_err_to_geozero)
    }
}

fn proj_err_to_geozero(err: ProjError) -> GeozeroError {
    GeozeroError::Geometry(err.to_string())
}

impl<P: GeomProcessor> GeomProcessor for StreamingReproject<'_, P> {
    fn dimensions(&self) -> CoordDimensions {
        self.inner.dimensions()
    }

    fn multi_dim(&self) -> bool {
        self.inner.multi_dim()
    }

    fn srid(&mut self, srid: Option<i32>) -> Result<()> {
        self.inner.srid(srid)
    }

    fn xy(&mut self, x: f64, y: f64, idx: usize) -> Result<()> {
        let (x, y) = self.project_xy(x, y)?;
        self.inner.xy(x, y, idx)
    }

    fn coordinate(
        &mut self,
        x: f64,
        y: f64,
        z: Option<f64>,
        m: Option<f64>,
        t: Option<f64>,
        tm: Option<u64>,
        idx: usize,
    ) -> Result<()> {
        let (x, y, z) = match z {
            Some(zv) => {
                let (x, y, zv) = self.project_xyz(x, y, zv)?;
                (x, y, Some(zv))
            }
            None => {
                let (x, y) = self.project_xy(x, y)?;
                (x, y, None)
            }
        };
        self.inner.coordinate(x, y, z, m, t, tm, idx)
    }

    fn empty_point(&mut self, idx: usize) -> Result<()> {
        self.inner.empty_point(idx)
    }

    fn point_begin(&mut self, idx: usize) -> Result<()> {
        self.inner.point_begin(idx)
    }

    fn point_end(&mut self, idx: usize) -> Result<()> {
        self.inner.point_end(idx)
    }

    fn multipoint_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.multipoint_begin(size, idx)
    }

    fn multipoint_end(&mut self, idx: usize) -> Result<()> {
        self.inner.multipoint_end(idx)
    }

    fn linestring_begin(&mut self, tagged: bool, size: usize, idx: usize) -> Result<()> {
        self.inner.linestring_begin(tagged, size, idx)
    }

    fn linestring_end(&mut self, tagged: bool, idx: usize) -> Result<()> {
        self.inner.linestring_end(tagged, idx)
    }

    fn multilinestring_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.multilinestring_begin(size, idx)
    }

    fn multilinestring_end(&mut self, idx: usize) -> Result<()> {
        self.inner.multilinestring_end(idx)
    }

    fn polygon_begin(&mut self, tagged: bool, size: usize, idx: usize) -> Result<()> {
        self.inner.polygon_begin(tagged, size, idx)
    }

    fn polygon_end(&mut self, tagged: bool, idx: usize) -> Result<()> {
        self.inner.polygon_end(tagged, idx)
    }

    fn multipolygon_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.multipolygon_begin(size, idx)
    }

    fn multipolygon_end(&mut self, idx: usize) -> Result<()> {
        self.inner.multipolygon_end(idx)
    }

    fn geometrycollection_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.geometrycollection_begin(size, idx)
    }

    fn geometrycollection_end(&mut self, idx: usize) -> Result<()> {
        self.inner.geometrycollection_end(idx)
    }

    fn circularstring_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.circularstring_begin(size, idx)
    }

    fn circularstring_end(&mut self, idx: usize) -> Result<()> {
        self.inner.circularstring_end(idx)
    }

    fn compoundcurve_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.compoundcurve_begin(size, idx)
    }

    fn compoundcurve_end(&mut self, idx: usize) -> Result<()> {
        self.inner.compoundcurve_end(idx)
    }

    fn curvepolygon_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.curvepolygon_begin(size, idx)
    }

    fn curvepolygon_end(&mut self, idx: usize) -> Result<()> {
        self.inner.curvepolygon_end(idx)
    }

    fn multicurve_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.multicurve_begin(size, idx)
    }

    fn multicurve_end(&mut self, idx: usize) -> Result<()> {
        self.inner.multicurve_end(idx)
    }

    fn multisurface_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.multisurface_begin(size, idx)
    }

    fn multisurface_end(&mut self, idx: usize) -> Result<()> {
        self.inner.multisurface_end(idx)
    }

    fn triangle_begin(&mut self, tagged: bool, size: usize, idx: usize) -> Result<()> {
        self.inner.triangle_begin(tagged, size, idx)
    }

    fn triangle_end(&mut self, tagged: bool, idx: usize) -> Result<()> {
        self.inner.triangle_end(tagged, idx)
    }

    fn polyhedralsurface_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.polyhedralsurface_begin(size, idx)
    }

    fn polyhedralsurface_end(&mut self, idx: usize) -> Result<()> {
        self.inner.polyhedralsurface_end(idx)
    }

    fn tin_begin(&mut self, size: usize, idx: usize) -> Result<()> {
        self.inner.tin_begin(size, idx)
    }

    fn tin_end(&mut self, idx: usize) -> Result<()> {
        self.inner.tin_end(idx)
    }
}
