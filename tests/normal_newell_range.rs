//! `newell_normal` range contracts for every shipped scalar: recovered
//! normals where the legacy accumulation overflows, legacy bits elsewhere.

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
            assert_eq!(newell_normal(&vertices), expected, "{vertices:?}");
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
/// scaled norm of `(0, 0, a)` is exactly `a`) and rejects half of it. The
/// sliver's unscaled fan products `side^2` overflow; its doubled area
/// `side^2 EPSILON` does not.
fn verify_exact_triangles<T: Scalar>(length: T, height: T, side: T) {
    let (z, one, three) = (T::from_int(0), T::from_int(1), T::from_int(3));
    let (half, tolerance) = (length / T::from_int(2), T::tolerance());
    let at = |x, y| Point3::new(x, y, height);
    let tip = side + side * T::EPSILON;
    let sliver = |h| [(z, z), (side, side), (side, tip)].map(|(x, y)| Point3::new(x, y, h));
    for triangle in [
        [at(z, z), at(three, z), at(z, three)],
        [at(one, -length), at(-one, one), at(-one, z)],
        [at(z, z), at(half, half - one), at(half + one, half)],
        [at(z, z), at(one, z), at(z, tolerance)],
        sliver(height),
        sliver(z),
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

/// Nine separated scales, so an expansion holds nine components at once.
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

/// Finite legacy sums keep their bits, and so does every poisoned coordinate.
fn verify_legacy_bits<T: Scalar>(large: T) {
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
}

/// Polygons the cold path must scale, with `lobe = 3 root / 4` and
/// `big = lobe^2`. The lobes' fan terms `lobe^2` are finite and overflow when
/// two are added. `1 / big` vanishes under the scale that tames `big`, so
/// that scale must go to `big` alone. `fine = grain (1 + EPSILON)` loses its
/// last bit under the whole scale, so it must be shared with `grain`, or the
/// loops of area `fine grain` and its negative stop cancelling. A difference
/// `2 MAX` overflows before any scaling (once against only zero factors);
/// beside it `faint` gives a doubled area `faint MAX` at least the tolerance,
/// and for `f64` under four times it, so the halved vertices must count
/// twice in the degeneracy test. The last sum, `root^2`, is not
/// representable while its direction is. Each corner `(height, 0, 0),
/// (-back, a, a), (0, 0, 0)` has an exact sum, a power of two times
/// `(0, -1, 1)`, and exact scaled products, so its unit normal is
/// `(0, -h, h)` with `h = 1 / sqrt(2)`. Each `a` in `slants` leaves a
/// subnormal scaled sum, for `f32` down to the least subnormal; `steep` has
/// an ordinary one. From either far vertex `reach` lies strictly between
/// `MAX / 2` and `MAX`, where only halved differences are finite. The
/// expansion and the norm's quotients and squares are exact; the root and the
/// quotient round once each here and in `h`: four unit roundoffs, `2 EPSILON`.
fn verify_scaled_products<T: Scalar>(root: T, grain: T, height: T, faint: T, slants: &[T]) {
    let (z, one, max) = (T::from_int(0), T::from_int(1), T::max_value());
    let (half, lobe) = (
        root / T::from_int(2),
        root * T::from_int(3) / T::from_int(4),
    );
    let (big, fine, up) = (lobe * lobe, grain + grain * T::EPSILON, Some(unit_z(1)));
    let p = |x, y| Point3::new(x, y, z);
    let q = |x, y| Point3::new(x, y, height);
    let lobes = [p(z, z), p(lobe, z), p(z, lobe), p(-lobe, z), p(z, lobe)];
    assert_exact_variants(&lobes, up);
    assert_exact_variants(&[q(z, z), q(big, z), q(z, one / big)], up);
    let loops = [
        [q(z, z), q(fine, z), q(z, grain)],
        [q(z, z), q(z, one), q(fine * grain, z)],
        [q(z, z), q(big, z), q(z, big)],
        [q(z, z), q(z, big), q(big, z)],
    ];
    assert_exact_variants(&loops.concat(), None);
    assert_exact_variants(&[p(max, z), p(-max, faint), p(z, z)], up);
    assert_exact_variants(&[p(max, z), p(-max, z), p(z, z)], None);
    assert_exact_variants(&[p(z, z), p(root, -root), p(half, half)], up);
    let (two, h) = (one + one, one / (one + one).sqrt());
    let (bound, steep) = (two * T::EPSILON * h, (height + height / two, one));
    for (back, a) in slants.iter().map(|&a| (height, a)).chain([steep]) {
        let mut corner = [(height, z), (-back, a), (z, z)].map(|(x, y)| Point3::new(x, y, y));
        for _ in 0..3 {
            assert!(!is_finite(&legacy_sum(&corner)));
            let off = newell_normal(&corner).map_or([one; 3], |n| [n.x, n.y + h, n.z - h]);
            assert!(off.iter().all(|e| e.abs() <= bound), "{corner:?} {off:?}");
            corner.rotate_left(1);
        }
    }
}

/// Unit roundoffs between a recovered component and the unit vector of the
/// exact sum: the 11 that `newell_normal` documents and derives, and 1 for
/// the correctly rounded quotient `k / 7` this test compares with.
const RECOVERED_ROUNDOFFS: i32 = 12;

/// A triangle at `height` with edges `fine (3, -2, 0)` and `step (0, 2, -1)`,
/// so its sum is `fine step (2, 3, 6)`, of length `7 fine step`. With
/// `fine = 1 + 8 EPSILON` and `step = (1 - EPSILON) height / 2` the vertices
/// are representable, the products need the scale, and each rounds when its
/// expansion is summed.
fn verify_tilted_triangle<T: Scalar>(height: T) {
    let [z, one, two, three] = [0, 1, 2, 3].map(T::from_int);
    let fine = one + T::from_int(8) * T::EPSILON;
    let step = (one - T::EPSILON) * height / two;
    let anchor = Point3::new(z, z, height);
    let points = [
        anchor,
        anchor + Vector3::new(three * fine, -two * fine, z),
        anchor + Vector3::new(z, two * step, -step),
    ];
    assert!(!is_finite(&legacy_sum(&points)));
    let normal = newell_normal(&points).unwrap_or_else(Vector3::zeros);
    for (axis, k) in [2, 3, 6].into_iter().enumerate() {
        let expected = T::from_int(k) / T::from_int(7);
        let bound = T::from_int(RECOVERED_ROUNDOFFS) * (T::EPSILON / two) * expected;
        assert!((normal[axis] - expected).abs() <= bound, "{normal:?}");
    }
}

#[test]
fn translated_triangles_keep_exact_normals() {
    verify_exact_triangles(2.0_f32.powi(24), 2.0_f32.powi(127), 2.0_f32.powi(70));
    verify_exact_triangles(2.0_f64.powi(53), 2.0_f64.powi(1023), 2.0_f64.powi(520));
}

#[test]
fn cancelling_subloops_have_exact_sign_and_zero() {
    verify_cancelling_subloops(2.0_f32.powi(24), 2.0_f32.powi(127));
    verify_cancelling_subloops(2.0_f64.powi(53), 2.0_f64.powi(1023));
}

#[test]
fn nine_separated_scales_keep_every_component() {
    // No `powi` exponent is below the least normal one; see `faint` below.
    let low = 2.0_f32.powi(-64);
    let single: Vec<_> = (0..9).map(|i| low * 2.0_f32.powi(25 * i - 79)).collect();
    let double: Vec<_> = (0..9).map(|i| 2.0_f64.powi(54 * i - 216)).collect();
    verify_many_scales(&single, 2.0_f32.powi(127));
    verify_many_scales(&double, 2.0_f64.powi(1023));
}

#[test]
fn ordinary_or_nonfinite_input_keeps_legacy_bits() {
    verify_legacy_bits(2.0_f32.powi(127));
    verify_legacy_bits(2.0_f64.powi(1023));
}

#[test]
fn overflowing_products_and_differences_are_scaled() {
    // `powi` may return one over an overflowed power, which is zero, so each
    // subnormal is the least normal value divided by a power of two.
    let (far, tiny) = (2.0_f32.powi(127), f32::MIN_POSITIVE);
    let slants = [0, 10, 14].map(|k| tiny / 2.0_f32.powi(k));
    let (root, faint) = (2.0_f32.powi(64), tiny / 512.0);
    verify_scaled_products(root, 2.0_f32.powi(5), far, faint, &slants);
    let (far, tiny) = (2.0_f64.powi(1023), f64::MIN_POSITIVE);
    let slants = [0, 23].map(|k| tiny / 2.0_f64.powi(k));
    let (root, faint) = (2.0_f64.powi(512), tiny / 2.0_f64.powi(31));
    verify_scaled_products(root, 2.0_f64.powi(60), far, faint, &slants);
}

#[test]
fn tilted_scaled_triangle_meets_the_accuracy_bound() {
    verify_tilted_triangle(2.0_f32.powi(127));
    verify_tilted_triangle(2.0_f64.powi(1023));
}
