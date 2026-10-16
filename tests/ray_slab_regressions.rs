//! Analytical ray/AABB contract regressions in each native scalar precision.

use eunomia::NumericElement;
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::{Aabb, Ray};
use leto::geometry::{Point3, Vector3};

fn point<T: Scalar>(coordinates: [T; 3]) -> Point3<T> {
    Point3::new(coordinates[0], coordinates[1], coordinates[2])
}

fn assert_tiny_slabs<T: Scalar>(tiny: T, on_boundary: bool) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = <T as Scalar>::from_int(2);
    let four = <T as Scalar>::from_int(4);
    for positive in [true, false] {
        let component = if positive { tiny } else { -tiny };
        for rotation in 0..3 {
            let rotate = |mut coordinates: [T; 3]| {
                coordinates.rotate_left(rotation);
                coordinates
            };
            let origin = rotate([zero, if on_boundary { zero } else { -component }, zero]);
            let direction = rotate([one, component, zero]);
            let lower = rotate([zero, if positive { zero } else { -tiny }, -one]);
            let upper = rotate([four, if positive { tiny } else { zero }, one]);
            let direction = Vector3::new(direction[0], direction[1], direction[2]);
            let ray = Ray::try_new(point(origin), direction).expect("finite nonzero direction");
            // Squared tiny components underflow while 1+tiny^2 rounds to 1;
            // normalization therefore preserves these stored coordinates.
            assert_eq!(ray.direction(), direction);
            let expected = if on_boundary { (zero, one) } else { (one, two) };
            // y=-tiny+tiny*t crosses [0,tiny] at exactly t=1 and t=2;
            // y=tiny*t crosses it at t=0 and t=1. The dominant slab is [0,4].
            assert_eq!(
                ray.intersect_aabb(&Aabb::new(point(lower), point(upper))),
                Some(expected),
                "rotation={rotation}, positive={positive}, boundary={on_boundary}"
            );
        }
    }
}

fn assert_halfline_contract<T: Scalar>() {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = <T as Scalar>::from_int(2);
    let ray = Ray::try_new(point([zero; 3]), Vector3::new(one, zero, zero)).expect("unit ray");
    let behind = Aabb::new(point([-two, -one, -one]), point([-one, one, one]));
    assert_eq!(ray.intersect_aabb(&behind), None);
    let touching_origin = Aabb::new(point([-one; 3]), point([zero, one, one]));
    assert_eq!(ray.intersect_aabb(&touching_origin), Some((-one, zero)));
    let around_origin = Aabb::new(point([-one; 3]), point([one; 3]));
    assert_eq!(ray.intersect_aabb(&around_origin), Some((-one, one)));
}

fn assert_empty_contract<T: Scalar>() {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let ray = Ray::try_new(point([zero; 3]), Vector3::new(one, one, one)).expect("diagonal ray");
    assert_eq!(ray.intersect_aabb(&Aabb::empty()), None);
    assert_eq!(
        ray.intersect_aabb(&Aabb::new(point([one; 3]), point([-one; 3]))),
        None
    );
    assert_eq!(
        ray.intersect_aabb(&Aabb::new(point([zero; 3]), point([zero; 3]))),
        Some((zero, zero))
    );
}

#[test]
fn tiny_slabs_single() {
    assert_tiny_slabs::<f32>(f32::from_bits(1), false);
}
#[test]
fn tiny_slabs_double() {
    assert_tiny_slabs::<f64>(f64::from_bits(1), false);
}
#[test]
fn boundary_slabs_single() {
    assert_tiny_slabs::<f32>(f32::from_bits(1), true);
}
#[test]
fn boundary_slabs_double() {
    assert_tiny_slabs::<f64>(f64::from_bits(1), true);
}
#[test]
fn halfline_single() {
    assert_halfline_contract::<f32>();
}
#[test]
fn halfline_double() {
    assert_halfline_contract::<f64>();
}
#[test]
fn empty_single() {
    assert_empty_contract::<f32>();
}
#[test]
fn empty_double() {
    assert_empty_contract::<f64>();
}

fn assert_finite_product_overflow<T: Scalar>(input: T, delta: T, component: T, maximum: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    for rotation in 0..3 {
        let rotate = |mut coordinates: [T; 3]| {
            coordinates.rotate_left(rotation);
            coordinates
        };
        let direction = rotate([one, input, zero]);
        let ray = Ray::try_new(
            point([zero; 3]),
            Vector3::new(direction[0], direction[1], direction[2]),
        )
        .expect("finite direction");
        assert_eq!(
            ray.direction()[if rotation == 0 { 0 } else { 3 - rotation }],
            component
        );
        let bounds = Aabb::new(
            point(rotate([delta, -maximum, -one])),
            point(rotate([delta, maximum, one])),
        );
        // Independent exact rational arithmetic places delta/component above
        // the midpoint of MAX and its predecessor and at or below MAX.
        // The correctly rounded distance therefore is finite MAX, even though
        // multiplying delta by the rounded finite reciprocal overflows.
        assert_eq!(
            ray.intersect_aabb(&bounds),
            Some((maximum, maximum)),
            "rotation={rotation}"
        );
    }
}

#[test]
fn product_overflow_single() {
    assert_finite_product_overflow(
        f32::from_bits(0x3ff3_3333),
        f32::from_bits(0x7eee_7650),
        f32::from_bits(0x3eee_7651),
        f32::MAX,
    );
}

#[test]
fn product_overflow_double() {
    assert_finite_product_overflow(
        f64::from_bits(0x3ffc_cccc_cccc_cccd),
        f64::from_bits(0x7fdf_14c6_16bc_880a),
        f64::from_bits(0x3fdf_14c6_16bc_880b),
        f64::MAX,
    );
}
