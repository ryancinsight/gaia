//! Exact-zero and winding regressions for Newell determinant cancellation.
#![expect(clippy::unwrap_used, reason = "Fixed nine-element indices fit i32")]
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::newell_normal;
use leto::geometry::{Point3, Vector3};

fn fixture<T: Scalar>(length: T, height: T, cross_subtrahend: T) -> Vec<Point3<T>> {
    let z = T::from_int(0);
    let one = T::from_int(1);
    // Three closed subloops: +1, L²-k, -L². Every stored coordinate,
    // anchored difference and individual product is exactly representable.
    // The subtraction inside the middle cross alone rounds L²-k to L².
    vec![
        Point3::new(z, z, height),
        Point3::new(one, z, height),
        Point3::new(z, one, height),
        Point3::new(z, z, height),
        Point3::new(length, one, height),
        Point3::new(cross_subtrahend, length, height),
        Point3::new(z, z, height),
        Point3::new(z, length, height),
        Point3::new(length, z, height),
    ]
}
fn check_zero<T: Scalar>(length: T, height: T) {
    let vertices = fixture(length, height, T::from_int(1));
    assert_eq!(
        newell_normal(&vertices),
        None,
        "exact doubled area 1+(L²-1)-L²=0"
    );
}
fn check_negative<T: Scalar>(length: T, height: T) {
    let vertices = fixture(length, height, T::from_int(2));
    let z = T::from_int(0);
    let one = T::from_int(1);
    assert_eq!(
        newell_normal(&vertices),
        Some(Vector3::new(z, z, -one)),
        "exact doubled area 1+(L²-2)-L²=-1"
    );
}
#[test]
fn exact_zero_single() {
    check_zero(2.0_f32.powi(24), 2.0_f32.powi(127));
}
#[test]
fn exact_zero_double() {
    check_zero(2.0_f64.powi(53), 2.0_f64.powi(1023));
}
#[test]
fn negative_area_single() {
    check_negative(2.0_f32.powi(24), 2.0_f32.powi(127));
}
#[test]
fn negative_area_double() {
    check_negative(2.0_f64.powi(53), 2.0_f64.powi(1023));
}

fn grid<T: Scalar>(length: T, height: T) {
    let zero = T::from_int(0);
    let one = T::from_int(1);
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..=4 {
        let base = fixture(length, height, T::from_int(k));
        let mut groups = [[base[1], base[2]], [base[4], base[5]], [base[7], base[8]]];
        let mut origin = base[0];
        let mut positive = Vector3::new(zero, zero, one);
        for _ in 0..3 {
            let expected = match k {
                1 => None,
                0 => Some(positive),
                _ => Some(-positive),
            };
            for order in orders {
                let mut vertices = Vec::new();
                for index in order {
                    vertices.extend([origin, groups[index][0], groups[index][1]]);
                }
                for _ in 0..vertices.len() {
                    assert_eq!(newell_normal(&vertices), expected, "k={k}");
                    let reverse: Vec<_> = vertices.iter().rev().copied().collect();
                    assert_eq!(
                        newell_normal(&reverse),
                        expected.map(|n| -n),
                        "reversed k={k}"
                    );
                    vertices.rotate_left(1);
                }
            }
            for group in &mut groups {
                for p in group {
                    *p = Point3::new(p.y, p.z, p.x);
                }
            }
            origin = Point3::new(origin.y, origin.z, origin.x);
            positive = Vector3::new(positive.y, positive.z, positive.x);
        }
    }
}
fn multiscale<T: Scalar>(powers: &[T], height: T) {
    let z = T::from_int(0);
    let one = T::from_int(1);
    let origin = Point3::new(z, z, height);
    for residual in [-1, 0, 1] {
        let mut vertices = Vec::new();
        for &power in powers {
            vertices.extend([
                origin,
                Point3::new(power, z, height),
                Point3::new(z, one, height),
            ]);
        }
        if residual != 0 {
            vertices.extend([
                origin,
                Point3::new(one, z, height),
                Point3::new(z, T::from_int(residual), height),
            ]);
        }
        for &power in powers.iter().rev() {
            vertices.extend([
                origin,
                Point3::new(z, one, height),
                Point3::new(power, z, height),
            ]);
        }
        let expected = if residual == 0 {
            None
        } else {
            Some(Vector3::new(z, z, T::from_int(residual)))
        };
        assert_eq!(
            newell_normal(&vertices),
            expected,
            "multiscale residual {residual}"
        );
        vertices.reverse();
        assert_eq!(newell_normal(&vertices), expected.map(|n| -n));
    }
}
#[test]
fn determinant_grid_single() {
    grid(2.0_f32.powi(24), 2.0_f32.powi(127));
}
#[test]
fn determinant_grid_double() {
    grid(2.0_f64.powi(53), 2.0_f64.powi(1023));
}
#[test]
fn multiscale_single() {
    let p: [f32; 9] = core::array::from_fn(|i| 2.0_f32.powi(-100 + 25 * i32::try_from(i).unwrap()));
    multiscale(&p, 2.0_f32.powi(127));
}
#[test]
fn multiscale_double() {
    let p: [f64; 9] = core::array::from_fn(|i| 2.0_f64.powi(-216 + 54 * i32::try_from(i).unwrap()));
    multiscale(&p, 2.0_f64.powi(1023));
}

fn product_roundoff<T: Scalar>(length: T, height: T) {
    let z = T::from_int(0);
    let one = T::from_int(1);
    // L²-(L-1)(L+1)=1, but the second native product rounds to L².
    // The fused product residual is 1 even before determinant subtraction.
    let mut v = [
        Point3::new(z, z, height),
        Point3::new(length, length - one, height),
        Point3::new(length + one, length, height),
    ];
    let mut expected = Vector3::new(z, z, one);
    for _ in 0..3 {
        for _ in 0..3 {
            assert_eq!(newell_normal(&v), Some(expected));
            assert_eq!(newell_normal(&[v[0], v[2], v[1]]), Some(-expected));
            v.rotate_left(1);
        }
        v = v.map(|p| Point3::new(p.y, p.z, p.x));
        expected = Vector3::new(expected.y, expected.z, expected.x);
    }
}
#[test]
fn product_residual_single() {
    product_roundoff(2.0_f32.powi(23), 2.0_f32.powi(127));
}
#[test]
fn product_residual_double() {
    product_roundoff(2.0_f64.powi(52), 2.0_f64.powi(1023));
}
