//! `newell_normal` range contracts for every shipped scalar: finite results
//! where the legacy accumulation overflows, legacy bits elsewhere.

use core::cmp::Ordering;
use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::newell_normal;
use leto::geometry::{Point3, Vector3};

/// The accumulation loop of `newell_normal` before the recovery path existed.
fn legacy_sum<T: Scalar>(vertices: &[Point3<T>]) -> Vector3<T> {
    let mut normal = Vector3::<T>::zeros();
    for (curr, next) in vertices.iter().zip(vertices.iter().cycle().skip(1)) {
        normal[0] += (curr.y - next.y) * (curr.z + next.z);
        normal[1] += (curr.z - next.z) * (curr.x + next.x);
        normal[2] += (curr.x - next.x) * (curr.y + next.y);
    }
    normal
}

fn is_finite<T: Scalar>(normal: &Vector3<T>) -> bool {
    (0..3).all(|axis| <T as eunomia::NumericElement>::is_finite(normal[axis]))
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

fn unit_z<T: Scalar>(sign: i32) -> Vector3<T> {
    Vector3::new(T::from_int(0), T::from_int(0), T::from_int(sign))
}

/// Checks `expected` for every cyclic start vertex, both windings (which
/// negate it) and all three axis rotations (which rotate it). Each start
/// vertex is a different fan anchor over the same exact sum.
fn assert_exact_variants<T: Scalar>(vertices: &[Point3<T>], expected: Option<Vector3<T>>) {
    let (mut vertices, mut expected) = (vertices.to_vec(), expected);
    for _ in 0..3 {
        // Premise: the legacy accumulation fails here, so the cold path runs.
        assert!(!is_finite(&legacy_sum(&vertices)));
        for _ in 0..vertices.len() {
            assert_eq!(newell_normal(&vertices), expected);
            let reversed: Vec<_> = vertices.iter().rev().copied().collect();
            assert_eq!(newell_normal(&reversed), expected.map(|n| -n));
            vertices.rotate_left(1);
        }
        for p in &mut vertices {
            *p = Point3::new(p.y, p.z, p.x);
        }
        expected = expected.map(|n| Vector3::new(n.y, n.z, n.x));
    }
}

/// Triangles with exact doubled areas 9, 2, 1 and `T::tolerance()` under a
/// large height. `1 + length` is not representable (the Two-Diff residual
/// keeps it) and the native product `(half - 1)(half + 1)` rounds to `half^2`
/// (the fused residual restores `half^2 - 1`), where `half = length / 2` is
/// the largest power of two with `half + 1` representable. The unchanged
/// `len < tolerance` test admits a doubled area equal to the tolerance (leto's
/// scaled norm of `(0, 0, a)` is exactly `a`) and rejects half of it.
fn verify_exact_triangles<T: Scalar>(length: T, height: T) {
    let (z, one, three) = (T::from_int(0), T::from_int(1), T::from_int(3));
    let (half, tolerance) = (length / T::from_int(2), T::tolerance());
    let at = |x, y| Point3::new(x, y, height);
    for triangle in [
        [at(z, z), at(three, z), at(z, three)],
        [at(one, -length), at(-one, one), at(-one, z)],
        [at(z, z), at(half, half - one), at(half + one, half)],
        [at(z, z), at(one, z), at(z, tolerance)],
    ] {
        assert_exact_variants(&triangle, Some(unit_z(1)));
    }
    let rejected = [at(z, z), at(one, z), at(z, tolerance / T::from_int(2))];
    assert_exact_variants(&rejected, None);
}

/// Closed subloops through one shared vertex, so their doubled areas add to
/// `1 - k`. With areas 1, `length^2 - k` and `-length^2`, differences from
/// the shared vertex and their products are exact and only the middle loop's
/// subtraction rounds. With areas `length^2 + 2 length - k`, `2 - length^2`
/// and `-2 length - 1`, the first loop's two differences from the shared
/// vertex round in crossed coordinates and their residuals multiply to 1.
fn verify_cancelling_subloops<T: Scalar>(length: T, height: T) {
    let (z, one, two) = (T::from_int(0), T::from_int(1), T::from_int(2));
    let at = |x, y| Point3::new(x, y, height);
    for k in 0..3 {
        let exact = [
            [at(z, z), at(one, z), at(z, one)],
            [at(z, z), at(length, one), at(T::from_int(k), length)],
            [at(z, z), at(z, length), at(length, z)],
        ];
        let rounded = [
            [at(one, one), at(-length, z), at(T::from_int(-k), -length)],
            [at(one, one), at(z, length), at(length + two, z)],
            [at(one, one), at(z, z), at(-two * length, one)],
        ];
        for loops in [exact, rounded] {
            assert_exact_variants(&loops.concat(), (k != 1).then(|| unit_z(1 - k)));
        }
    }
}

/// More expansion components than the inline capacity, over nine scales.
fn verify_many_scales<T: Scalar>(powers: &[T], height: T) {
    let (z, one) = (T::from_int(0), T::from_int(1));
    let (origin, up) = (Point3::new(z, z, height), Point3::new(z, one, height));
    let along = |x| Point3::new(x, z, height);
    for residual in [-1, 0, 1] {
        let mut vertices = Vec::new();
        vertices.extend(powers.iter().flat_map(|&p| [origin, along(p), up]));
        if residual != 0 {
            let tail = Point3::new(z, T::from_int(residual), height);
            vertices.extend([origin, along(one), tail]);
        }
        vertices.extend(powers.iter().rev().flat_map(|&p| [origin, up, along(p)]));
        assert_exact_variants(&vertices, (residual != 0).then(|| unit_z(residual)));
    }
}

/// Finite legacy sums keep their bits. So does every poisoned coordinate, a
/// difference that overflows (once against only zero factors) and a pair of
/// finite products `root^2 / 2` whose sum overflows.
fn verify_legacy_bits<T: Scalar>(large: T, maximum: T, root: T) {
    for shift in -16..=16 {
        let mut points = [(0, 0, 0), (3, 0, 1), (4, 2, -1), (0, 3, 2)]
            .map(|(a, b, c)| Point3::new(T::from_int(a), T::from_int(b), T::from_int(c)));
        for p in &mut points {
            *p += Vector3::new(T::from_int(shift), T::from_int(-shift), T::from_int(shift));
        }
        assert_legacy(&points);
    }
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
    let (z, three, half) = (T::from_int(0), T::from_int(3), root / T::from_int(2));
    let at = Point3::<T>::new;
    for points in [
        [at(maximum, z, z), at(-maximum, three, z), at(z, z, three)],
        [at(maximum, z, z), at(-maximum, z, z), at(z, z, z)],
        [at(z, z, z), at(root, -root, z), at(half, half, z)],
    ] {
        assert!(!is_finite(&legacy_sum(&points)));
        assert_legacy(&points);
    }
}

#[test]
fn translated_triangles_keep_exact_normals() {
    verify_exact_triangles(2.0_f32.powi(24), 2.0_f32.powi(127));
    verify_exact_triangles(2.0_f64.powi(53), 2.0_f64.powi(1023));
}

#[test]
fn cancelling_subloops_have_exact_sign_and_zero() {
    verify_cancelling_subloops(2.0_f32.powi(24), 2.0_f32.powi(127));
    verify_cancelling_subloops(2.0_f64.powi(53), 2.0_f64.powi(1023));
}

#[test]
fn expansion_beyond_inline_capacity_keeps_every_component() {
    let single: Vec<_> = (0..9).map(|i| 2.0_f32.powi(25 * i - 100)).collect();
    let double: Vec<_> = (0..9).map(|i| 2.0_f64.powi(54 * i - 216)).collect();
    verify_many_scales(&single, 2.0_f32.powi(127));
    verify_many_scales(&double, 2.0_f64.powi(1023));
}

#[test]
fn unrepresentable_or_nonfinite_input_keeps_legacy_bits() {
    verify_legacy_bits(2.0_f32.powi(127), f32::MAX, 2.0_f32.powi(64));
    verify_legacy_bits(2.0_f64.powi(1023), f64::MAX, 2.0_f64.powi(512));
}
