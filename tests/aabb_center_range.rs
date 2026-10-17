//! A finite box has a representable midpoint even when its endpoint sum overflows.

use eunomia::NumericElement;
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::Aabb;
use leto::geometry::Point3;

fn point<T: Scalar>(p: [T; 3]) -> Point3<T> {
    Point3::new(p[0], p[1], p[2])
}

fn assert_large_center<T: Scalar>(lower: T, upper: T, expected: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let three = <T as Scalar>::from_int(3);
    for positive in [true, false] {
        let (lower, upper, expected) = if positive {
            (lower, upper, expected)
        } else {
            (-upper, -lower, -expected)
        };
        for rotation in 0..3 {
            let rotate = |mut p: [T; 3]| {
                p.rotate_left(rotation);
                point(p)
            };
            let bounds = Aabb::new(rotate([lower, -one, -three]), rotate([upper, one, one]));
            let center = bounds.center();
            assert_eq!(center, rotate([expected, zero, -one]));
            assert!(bounds.contains_point(&center));
        }
    }
}

fn assert_ordinary_centers<T: Scalar>(maximum: T, tiny: T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = <T as Scalar>::from_int(2);
    let six = <T as Scalar>::from_int(6);
    let bounds = Aabb::new(point([-six, zero, -two]), point([two, two, zero]));
    assert_eq!(bounds.center(), point([-two, one, -one]));
    let span = Aabb::new(point([-maximum; 3]), point([maximum; 3]));
    assert_eq!(span.center(), point([zero; 3]));
    // Always halving the endpoints first would lose these representable values.
    for value in [-tiny, tiny] {
        let bounds = Aabb::new(point([value; 3]), point([value; 3]));
        assert_eq!(bounds.center(), point([value; 3]));
    }
}

fn preserve_inverted_arithmetic<T: Scalar>(maximum: T, tiny: T) {
    let half = <T as Scalar>::from_f64(0.5);
    let infinity = <T as eunomia::RealField>::infinity();
    let bounds = Aabb::new(
        point([maximum, tiny, -tiny]),
        point([maximum * half, tiny, -tiny]),
    );
    // An inverted box has no selected checked contract here. Preserve the
    // original arithmetic result for the whole box, including other axes.
    assert_eq!(bounds.center(), point([infinity, tiny, -tiny]));
}

#[test]
fn singleton_extremes_binary32() {
    assert_large_center(f32::MAX, f32::MAX, f32::MAX);
}

#[test]
fn singleton_extremes_binary64() {
    assert_large_center(f64::MAX, f64::MAX, f64::MAX);
}

#[test]
fn positive_width_extremes_binary32() {
    assert_large_center(f32::MAX * 0.5, f32::MAX, f32::from_bits(0x7f3f_ffff));
    let previous = f32::from_bits(0x7f7f_fffe);
    // The exact average is halfway between the adjacent endpoints; even wins.
    assert_large_center(previous, f32::MAX, previous);
}

#[test]
fn positive_width_extremes_binary64() {
    assert_large_center(
        f64::MAX * 0.5,
        f64::MAX,
        f64::from_bits(0x7fe7_ffff_ffff_ffff),
    );
    let previous = f64::from_bits(0x7fef_ffff_ffff_fffe);
    assert_large_center(previous, f64::MAX, previous);
}

#[test]
fn ordinary_and_subnormal_centers_binary32() {
    assert_ordinary_centers(f32::MAX, f32::from_bits(1));
}

#[test]
fn ordinary_and_subnormal_centers_binary64() {
    assert_ordinary_centers(f64::MAX, f64::from_bits(1));
}

#[test]
fn inverted_arithmetic_is_preserved_binary32() {
    preserve_inverted_arithmetic(f32::MAX, f32::from_bits(1));
}

#[test]
fn inverted_arithmetic_is_preserved_binary64() {
    preserve_inverted_arithmetic(f64::MAX, f64::from_bits(1));
}
