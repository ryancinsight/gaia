//! Independent finite-segment range audit on unchanged published 0.6.8.
#![expect(
    clippy::unwrap_used,
    reason = "Audit fixtures require valid finite intersections"
)]
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::plane::Plane;
use leto::geometry::{Point3, Vector3};

fn symmetric<T: Scalar>(large: T) {
    let zero = T::from_int(0);
    let one = T::from_int(1);
    let half = one / T::from_int(2);
    let mut normal = Vector3::new(one, zero, zero);
    let mut endpoint = Point3::new(large, zero, zero);
    for _ in 0..3 {
        let plane = Plane::new(normal, zero);
        let opposite = Point3::new(-endpoint.x, -endpoint.y, -endpoint.z);
        assert_eq!(plane.signed_distance(&endpoint), large);
        assert_eq!(plane.signed_distance(&opposite), -large);
        // Both endpoints are finite. The exact midpoint is the origin, on the
        // unit-normal plane; reversing endpoints retains t=1/2.
        assert_eq!(plane.signed_distance(&Point3::origin()), zero);
        assert_eq!(plane.intersect_segment(&endpoint, &opposite).unwrap(), half);
        assert_eq!(plane.intersect_segment(&opposite, &endpoint).unwrap(), half);
        endpoint = Point3::new(endpoint.y, endpoint.z, endpoint.x);
        normal = Vector3::new(normal.y, normal.z, normal.x);
    }
}

fn ordinary<T: Scalar>() {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(0));
    let a = Point3::new(f(-3), f(0), f(0));
    let b = Point3::new(f(1), f(0), f(0));
    let t = plane.intersect_segment(&a, &b).unwrap();
    assert_eq!(t, f(3) / f(4));
    assert_eq!((f(1) - t) * a.x + t * b.x, f(0));
}
fn endpoint<T: Scalar>() {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(-1));
    let a = Point3::new(f(1), f(0), f(0));
    let b = Point3::new(f(3), f(0), f(0));
    assert_eq!(plane.intersect_segment(&a, &b).unwrap(), f(0));
    assert_eq!(plane.intersect_segment(&b, &a).unwrap(), f(1));
}
fn parallel<T: Scalar>() {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(0));
    assert_eq!(
        plane.intersect_segment(
            &Point3::new(f(1), f(0), f(0)),
            &Point3::new(f(1), f(2), f(0))
        ),
        None
    );
}
#[test]
fn symmetric_power_f32() {
    symmetric(2.0_f32.powi(127));
}
#[test]
fn symmetric_max_f32() {
    symmetric(f32::MAX);
}
#[test]
fn symmetric_power_f64() {
    symmetric(2.0_f64.powi(1023));
}
#[test]
fn symmetric_max_f64() {
    symmetric(f64::MAX);
}
#[test]
fn ordinary_f32() {
    ordinary::<f32>();
}
#[test]
fn ordinary_f64() {
    ordinary::<f64>();
}
#[test]
fn endpoint_f32() {
    endpoint::<f32>();
}
#[test]
fn endpoint_f64() {
    endpoint::<f64>();
}
#[test]
fn parallel_f32() {
    parallel::<f32>();
}
#[test]
fn parallel_f64() {
    parallel::<f64>();
}

fn asymmetric<T: Scalar>(large: T) {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(0));
    let a = Point3::new(large, f(0), f(0));
    let b = Point3::new(-large / f(2), f(0), f(0));
    let t = plane.intersect_segment(&a, &b).unwrap();
    let reverse = plane.intersect_segment(&b, &a).unwrap();
    assert!((t - f(2) / f(3)).abs() <= <T as eunomia::RealField>::EPSILON);
    assert!((reverse - f(1) / f(3)).abs() <= <T as eunomia::RealField>::EPSILON);
    assert!((t + reverse - f(1)).abs() <= <T as eunomia::RealField>::EPSILON);
}
fn small_and_cancelling<T: Scalar>(tiny: T) {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(0));
    let a = Point3::new(-tiny, f(0), f(0));
    let b = Point3::new(tiny, f(0), f(0));
    assert_eq!(
        plane.intersect_segment_with_eps(&a, &b, f(0)).unwrap(),
        f(1) / f(2)
    );
    assert_eq!(plane.intersect_segment(&a, &b), None);
    let epsilon = <T as eunomia::RealField>::EPSILON;
    let a = Point3::new(f(1) + f(4) * epsilon, f(0), f(0));
    let b = Point3::new(f(1) + f(2) * epsilon, f(0), f(0));
    // Exact close same-sign subtraction; the legacy extrapolation clamp stays.
    assert_eq!(
        plane.intersect_segment_with_eps(&a, &b, f(0)).unwrap(),
        f(1)
    );
    assert_eq!(
        plane.intersect_segment_with_eps(&b, &a, f(0)).unwrap(),
        f(0)
    );
}
fn nonfinite_contract<T: Scalar>(nan: T, infinity: T, large: T) {
    let f = T::from_int;
    let plane = Plane::new(Vector3::new(f(1), f(0), f(0)), f(0));
    for (a, b) in [
        (nan, f(1)),
        (infinity, -f(1)),
        (-infinity, f(1)),
        (infinity, -infinity),
    ] {
        let a = Point3::new(a, f(0), f(0));
        let b = Point3::new(b, f(0), f(0));
        // The existing native ratio/clamp yields zero for these invalid
        // distances; this patch does not introduce a checked-input policy.
        assert_eq!(
            plane.intersect_segment_with_eps(&a, &b, f(0)).unwrap(),
            f(0)
        );
    }
    let a = Point3::new(large, f(0), f(0));
    let b = Point3::new(-large, f(0), f(0));
    for eps in [nan, infinity, -f(1)] {
        assert_eq!(plane.intersect_segment_with_eps(&a, &b, eps).unwrap(), f(0));
    }
    let a = Point3::new(f(1), f(0), f(0));
    let b = Point3::new(f(1), f(1), f(0));
    assert_eq!(plane.intersect_segment_with_eps(&a, &b, infinity), None);
    // Legacy negative epsilon bypasses parallel rejection: 1/0 clamps to 1.
    assert_eq!(
        plane.intersect_segment_with_eps(&a, &b, -f(1)).unwrap(),
        f(1)
    );
}
#[test]
fn asymmetric_f32() {
    asymmetric(f32::MAX);
}
#[test]
fn asymmetric_f64() {
    asymmetric(f64::MAX);
}
#[test]
fn small_cancellation_f32() {
    small_and_cancelling(f32::from_bits(1));
}
#[test]
fn small_cancellation_f64() {
    small_and_cancelling(f64::from_bits(1));
}
#[test]
fn nonfinite_f32() {
    nonfinite_contract(f32::NAN, f32::INFINITY, f32::MAX);
}
#[test]
fn nonfinite_f64() {
    nonfinite_contract(f64::NAN, f64::INFINITY, f64::MAX);
}
