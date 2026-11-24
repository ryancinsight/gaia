//! Translation and cancellation regressions for native Newell recovery.
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::newell_normal;
use leto::geometry::{Point3, Vector3};

fn check<T: Scalar>(length: T, height: T) {
    let zero = T::from_int(0);
    let one = T::from_int(1);
    let mut shifted = [
        Point3::new(one, -length, height),
        Point3::new(-one, one, height),
        Point3::new(-one, zero, height),
    ];
    let mut unshifted = shifted.map(|p| Point3::new(p.x, p.y, zero));
    let mut expected = Vector3::new(zero, zero, one);
    for _ in 0..3 {
        // Independent reviewer fixture: first-origin L+1 rounds to L, but
        // the alternate-origin native cross and exact doubled area are 2.
        let stable = (shifted[2] - shifted[1]).cross(shifted[0] - shifted[1]);
        assert_eq!(stable, expected * (one + one));
        for _ in 0..3 {
            assert_eq!(newell_normal(&unshifted), Some(expected));
            assert_eq!(newell_normal(&shifted), Some(expected));
            let reversed = [shifted[0], shifted[2], shifted[1]];
            assert_eq!(newell_normal(&reversed), Some(-expected));
            shifted.rotate_left(1);
            unshifted.rotate_left(1);
        }
        shifted = shifted.map(|p| Point3::new(p.y, p.z, p.x));
        unshifted = unshifted.map(|p| Point3::new(p.y, p.z, p.x));
        expected = Vector3::new(expected.y, expected.z, expected.x);
    }
}

fn cancellation<T: Scalar>(length: T, height: T, retain_small_area: bool) {
    let zero = T::from_int(0);
    let one = T::from_int(1);
    let two = one + one;
    let origin = Point3::new(zero, zero, height);
    // Three closed subloops have exact doubled areas L², 2, -L². The
    // positive and negative large subloops cancel. Without the middle loop
    // the exact vector area is zero; with it the normal is positive unit Z.
    let large_x = Point3::new(length, zero, height);
    let large_y = Point3::new(zero, length, height);
    let small_x = Point3::new(one, zero, height);
    let small_y = Point3::new(zero, two, height);
    let mut vertices = vec![origin, large_x, large_y, origin];
    if retain_small_area {
        vertices.extend([small_x, small_y, origin]);
    }
    vertices.extend([large_y, large_x]);
    let expected = if retain_small_area {
        Some(Vector3::new(zero, zero, one))
    } else {
        None
    };
    for _ in 0..vertices.len() {
        assert_eq!(newell_normal(&vertices), expected);
        let reversed: Vec<_> = vertices.iter().rev().copied().collect();
        assert_eq!(newell_normal(&reversed), expected.map(|n| -n));
        vertices.rotate_left(1);
    }
}

#[test]
fn translated_native_single() {
    check(2.0_f32.powi(24), 2.0_f32.powi(127));
}
#[test]
fn translated_native_double() {
    check(2.0_f64.powi(53), 2.0_f64.powi(1023));
}
#[test]
fn retained_cancellation_single() {
    cancellation(2.0_f32.powi(24), 2.0_f32.powi(127), true);
}
#[test]
fn retained_cancellation_double() {
    cancellation(2.0_f64.powi(53), 2.0_f64.powi(1023), true);
}
#[test]
fn exact_cancellation_single() {
    cancellation(2.0_f32.powi(24), 2.0_f32.powi(127), false);
}
#[test]
fn exact_cancellation_double() {
    cancellation(2.0_f64.powi(53), 2.0_f64.powi(1023), false);
}
