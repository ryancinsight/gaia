use super::*;

type Point = Point3<f64>;
type Line = Polyline<f64>;

fn point(x: f64, y: f64, z: f64) -> Point {
    Point::new(x, y, z)
}

#[test]
fn construction_rejects_short_and_non_finite_inputs() {
    assert_eq!(Line::new(Vec::new()), Err(PolylineError::TooFewPoints(0)));
    assert_eq!(
        Line::new(vec![point(0.0, 0.0, 0.0)]),
        Err(PolylineError::TooFewPoints(1))
    );
    assert_eq!(
        Line::new(vec![point(0.0, 0.0, 0.0), point(f64::NAN, 1.0, 2.0)]),
        Err(PolylineError::NonFinitePoint { index: 1 })
    );
    assert_eq!(
        Line::new(vec![point(f64::INFINITY, 0.0, 0.0), point(1.0, 0.0, 0.0),]),
        Err(PolylineError::NonFinitePoint { index: 0 })
    );
}

#[test]
fn ordered_vertices_define_segments_and_endpoints() -> Result<(), PolylineError> {
    let line = Line::new(vec![
        point(0.0, 0.0, 0.0),
        point(1.0, 0.0, 0.0),
        point(1.0, 2.0, 0.0),
    ])?;
    assert_eq!(line.len(), 3);
    assert!(!line.is_empty());
    assert_eq!(line.segment_count(), 2);
    assert_eq!(line.first(), point(0.0, 0.0, 0.0));
    assert_eq!(line.last(), point(1.0, 2.0, 0.0));
    assert_eq!(line.segments().count(), 2);
    Ok(())
}

#[test]
fn arc_length_sums_each_segment() -> Result<(), PolylineError> {
    let line = Line::new(vec![
        point(0.0, 0.0, 0.0),
        point(3.0, 4.0, 0.0),
        point(3.0, 4.0, 12.0),
    ])?;
    assert!((line.arc_length() - 17.0).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn aabb_encloses_every_vertex() -> Result<(), PolylineError> {
    let line = Line::new(vec![
        point(-1.0, 2.0, 3.0),
        point(4.0, -5.0, 0.0),
        point(0.0, 0.0, 7.0),
    ])?;
    let bounds = line.aabb();
    assert_eq!(bounds.min, point(-1.0, -5.0, 0.0));
    assert_eq!(bounds.max, point(4.0, 2.0, 7.0));
    Ok(())
}

#[test]
fn scalar_contract_covers_f32() -> Result<(), PolylineError> {
    let line = Polyline::<f32>::new(vec![Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 4.0, 0.0)])?;
    assert!((line.arc_length() - 5.0).abs() < f32::EPSILON);
    Ok(())
}
