use super::*;
use eunomia::NumericElement;

type Point = Point3<f64>;
type Line = Polyline<f64>;

fn point(x: f64, y: f64, z: f64) -> Point {
    Point::new(x, y, z)
}

fn assert_finite_validation<T: Scalar>() {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let non_finite = [
        <T as NumericElement>::NAN,
        <T as NumericElement>::INFINITY,
        -<T as NumericElement>::INFINITY,
    ];

    for value in non_finite {
        for coordinate in 0..3 {
            let invalid = match coordinate {
                0 => Point3::new(value, one, one),
                1 => Point3::new(one, value, one),
                2 => Point3::new(one, one, value),
                _ => unreachable!("invariant: coordinate comes from 0..3"),
            };
            assert_eq!(
                Polyline::<T>::new(vec![Point3::new(zero, zero, zero), invalid]),
                Err(PolylineError::NonFinitePoint { index: 1 })
            );
        }
    }

    assert_eq!(
        Polyline::<T>::new(vec![
            Point3::new(<T as NumericElement>::NAN, zero, zero),
            Point3::new(one, one, one),
        ]),
        Err(PolylineError::NonFinitePoint { index: 0 })
    );
}

fn assert_relative_eq<T: Scalar>(actual: T, expected: T) {
    let relative_error = (actual - expected).abs() / expected.abs();
    // The provider norm performs six rounded scalar operations for this
    // conditioned two-component reference, bounded by eight epsilons.
    let tolerance = T::EPSILON * <T as Scalar>::from_f64(8.0);
    assert!(relative_error <= tolerance);
}

fn assert_range_stable_arc_length<T: Scalar>(large: T, small: T) -> Result<(), PolylineError> {
    let zero = <T as NumericElement>::ZERO;
    for magnitude in [large, small] {
        let line = Polyline::<T>::new(vec![
            Point3::new(zero, zero, zero),
            Point3::new(magnitude, magnitude, zero),
        ])?;
        assert_relative_eq(line.arc_length(), magnitude * T::SQRT_2);
    }
    Ok(())
}

#[test]
fn construction_rejects_short_and_non_finite_inputs() {
    assert_eq!(Line::new(Vec::new()), Err(PolylineError::TooFewPoints(0)));
    assert_eq!(
        Line::new(vec![point(0.0, 0.0, 0.0)]),
        Err(PolylineError::TooFewPoints(1))
    );
    assert_finite_validation::<f32>();
    assert_finite_validation::<f64>();
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

#[test]
fn arc_length_is_range_stable_for_supported_fields() -> Result<(), PolylineError> {
    assert_range_stable_arc_length(1.0e20_f32, 1.0e-30_f32)?;
    assert_range_stable_arc_length(1.0e200_f64, 1.0e-200_f64)
}
