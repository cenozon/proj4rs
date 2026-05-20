use geozero::error::{GeozeroError, Result};
use geozero::GeomProcessor;
use proj4rs::Proj;
use proj4rs_geozero::StreamingReproject;

const WGS84: &str = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs";
const WEB_MERCATOR: &str = concat!(
    "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 ",
    "+units=m +nadgrids=@null +wktext +no_defs",
);

#[derive(Default)]
struct CoordCollector {
    coords: Vec<(f64, f64)>,
}

impl GeomProcessor for CoordCollector {
    fn xy(&mut self, x: f64, y: f64, _idx: usize) -> Result<()> {
        self.coords.push((x, y));
        Ok(())
    }
}

#[test]
fn xy_origin_reprojects_to_mercator_origin() {
    // Arrange
    let src = Proj::from_proj_string(WGS84).unwrap();
    let dst = Proj::from_proj_string(WEB_MERCATOR).unwrap();
    let mut collector = CoordCollector::default();
    let mut reproj = StreamingReproject::new(&src, &dst, &mut collector);

    // Act
    reproj.xy(0.0_f64, 0.0_f64, 0).unwrap();

    // Assert
    assert_eq!(collector.coords.len(), 1);
    let (x, y) = collector.coords[0];
    assert!(x.abs() < 1.0e-6, "expected x near 0, got {x}");
    assert!(y.abs() < 1.0e-6, "expected y near 0, got {y}");
}

#[test]
fn linestring_callbacks_forward_and_coords_reproject() {
    // Arrange
    let src = Proj::from_proj_string(WGS84).unwrap();
    let dst = Proj::from_proj_string(WEB_MERCATOR).unwrap();
    let mut collector = CoordCollector::default();
    let mut reproj = StreamingReproject::new(&src, &dst, &mut collector);
    // proj4rs's longlat input is in radians.
    let pts_rad: Vec<(f64, f64)> = vec![
        (0.0_f64.to_radians(), 0.0_f64.to_radians()),
        (10.0_f64.to_radians(), 0.0_f64.to_radians()),
        (10.0_f64.to_radians(), 10.0_f64.to_radians()),
    ];

    // Act
    let result = drive_linestring(&mut reproj, &pts_rad);

    // Assert
    result.unwrap();
    assert_eq!(collector.coords.len(), 3);
    let (x0, y0) = collector.coords[0];
    assert!(x0.abs() < 1.0e-6, "expected x0 near 0, got {x0}");
    assert!(y0.abs() < 1.0e-6, "expected y0 near 0, got {y0}");
    let (x1, y1) = collector.coords[1];
    // 10 degrees lon in Web Mercator metres at equator.
    let expected_x1 = 10.0_f64.to_radians() * 6_378_137.0;
    assert!(
        (x1 - expected_x1).abs() < 1.0e-3,
        "expected x1 ~= {expected_x1}, got {x1}"
    );
    assert!(y1.abs() < 1.0e-6, "expected y1 near 0, got {y1}");
    let (x2, y2) = collector.coords[2];
    assert!(
        (x2 - expected_x1).abs() < 1.0e-3,
        "expected x2 ~= {expected_x1}, got {x2}"
    );
    assert!(y2 > 0.0, "expected y2 > 0 for positive latitude, got {y2}");
}

#[test]
fn out_of_range_latitude_returns_geozero_error() {
    // Arrange
    let src = Proj::from_proj_string(WGS84).unwrap();
    let dst = Proj::from_proj_string(WEB_MERCATOR).unwrap();
    let mut collector = CoordCollector::default();
    let mut reproj = StreamingReproject::new(&src, &dst, &mut collector);
    // 100 degrees latitude in radians — beyond pi/2, invalid for Mercator.
    let bad_lat_rad = 100.0_f64.to_radians();

    // Act
    let result = reproj.xy(0.0_f64, bad_lat_rad, 0);

    // Assert
    let err = result.expect_err("expected reprojection failure for lat > 90");
    assert!(
        matches!(err, GeozeroError::Geometry(_)),
        "expected GeozeroError::Geometry, got {err:?}"
    );
    assert!(
        collector.coords.is_empty(),
        "inner processor must not see invalid coords"
    );
}

fn drive_linestring<P: GeomProcessor>(p: &mut P, pts: &[(f64, f64)]) -> Result<()> {
    p.linestring_begin(true, pts.len(), 0)?;
    for (i, (x, y)) in pts.iter().enumerate() {
        p.xy(*x, *y, i)?;
    }
    p.linestring_end(true, 0)
}
