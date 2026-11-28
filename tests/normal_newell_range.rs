//! `newell_normal` range contracts for every shipped scalar.

use core::cmp::Ordering;
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::newell_normal;
use leto::geometry::{Point3, Vector3};

/// The accumulation loop of `newell_normal`, kept here as the reference its
/// results are pinned to.
fn legacy_sum<T: Scalar>(vertices: &[Point3<T>]) -> Vector3<T> {
    let mut normal = Vector3::<T>::zeros();
    for (curr, next) in vertices.iter().zip(vertices.iter().cycle().skip(1)) {
        normal[0] += (curr.y - next.y) * (curr.z + next.z);
        normal[1] += (curr.z - next.z) * (curr.x + next.x);
        normal[2] += (curr.x - next.x) * (curr.y + next.y);
    }
    normal
}

fn assert_legacy<T: Scalar>(vertices: &[Point3<T>]) {
    let sum = legacy_sum(vertices);
    let len = sum.norm();
    // NaN passes the legacy `len < tolerance` degeneracy test.
    let expected = if len < T::tolerance() {
        None
    } else {
        Some(sum / len)
    };
    match (newell_normal(vertices), expected) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            for axis in 0..3 {
                let order = Scalar::total_cmp(&actual[axis], &expected[axis]);
                assert_eq!(order, Ordering::Equal);
            }
        }
        values => panic!("changed legacy result: {values:?}"),
    }
}

/// Finite legacy sums keep their bits, and so does every poisoned coordinate.
/// A seventh is not representable, so the scaled copies round in every
/// product and sum, and those roundings are part of the pinned bits. The
/// doubled area of the last two triangles is the tolerance, which the
/// `len < tolerance` test admits (leto's scaled norm of `(0, 0, a)` is exactly
/// `a`), and half of it, which that test rejects.
fn verify_legacy_bits<T: Scalar>(large: T) {
    let (z, one) = (T::from_int(0), T::from_int(1));
    let seventh = one / T::from_int(7);
    for shift in -16..=16 {
        let mut points = [(0, 0, 0), (3, 0, 1), (4, 2, -1), (0, 3, 2)]
            .map(|(a, b, c)| Point3::new(T::from_int(a), T::from_int(b), T::from_int(c)));
        for p in &mut points {
            *p += Vector3::new(T::from_int(shift), T::from_int(-shift), T::from_int(shift));
        }
        assert_legacy(&points);
        assert_legacy(&points.map(|p| Point3::new(p.x * seventh, p.y * seventh, p.z * seventh)));
    }
    let sliver = |area| [(z, z), (one, z), (z, area)].map(|(x, y)| Point3::new(x, y, z));
    let (admitted, rejected) = (T::tolerance(), T::tolerance() / T::from_int(2));
    let up = Some(Vector3::new(z, z, one));
    assert_eq!(newell_normal(&sliver(admitted)), up);
    assert_eq!(newell_normal(&sliver(rejected)), None);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for k in 0..9 {
            let mut points = [Point3::new(large, large, large); 3];
            let poison = <T as Scalar>::from_f64(invalid);
            match k % 3 {
                0 => points[k / 3].x = poison,
                1 => points[k / 3].y = poison,
                _ => points[k / 3].z = poison,
            }
            assert_legacy(&points);
        }
    }
}

#[test]
fn ordinary_or_nonfinite_input_keeps_legacy_bits() {
    verify_legacy_bits(2.0_f32.powi(127));
    verify_legacy_bits(2.0_f64.powi(1023));
}
