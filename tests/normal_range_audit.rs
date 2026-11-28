//! Triangle centroid range contracts: finite results where the raw sum
//! overflows, legacy bits elsewhere. Every check runs for each shipped scalar.

use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::triangle_centroid;
use leto::geometry::Point3;

fn translated_triangle<T: Scalar>(x: T) -> [Point3<T>; 3] {
    let zero = T::from_int(0);
    let three = T::from_int(3);
    [
        Point3::new(x, zero, zero),
        Point3::new(x, three, zero),
        Point3::new(x, zero, three),
    ]
}

fn verify_centroid<T: Scalar>(x: T) {
    let [a, b, c] = translated_triangle(x);
    // All three stored x coordinates are identical; their exact mean is x.
    // The remaining exact means are (0+3+0)/3 and (0+0+3)/3.
    let center = triangle_centroid(&a, &b, &c);
    assert_eq!(
        (center.x, center.y, center.z),
        (x, T::from_int(1), T::from_int(1))
    );
}

#[test]
fn ordinary_centroid_control() {
    verify_centroid(1.0_f32);
    verify_centroid(1.0_f64);
}

#[test]
fn equal_large_coordinates_keep_their_value() {
    verify_centroid(2.0_f32.powi(127));
    verify_centroid(2.0_f64.powi(1023));
}

fn legacy_centroid<T: Scalar>(values: [T; 3]) -> T {
    (values[0] + values[1] + values[2]) * <T as Scalar>::from_f64(1.0 / 3.0)
}

fn coordinate_mean<T: Scalar>(values: [T; 3]) -> T {
    let zero = T::from_int(0);
    triangle_centroid(
        &Point3::new(values[0], zero, zero),
        &Point3::new(values[1], zero, zero),
        &Point3::new(values[2], zero, zero),
    )
    .x
}

fn assert_same<T: Scalar>(actual: T, expected: T) {
    assert_eq!(
        Scalar::total_cmp(&actual, &expected),
        core::cmp::Ordering::Equal
    );
}

fn verify_centroid_cancellation<T: Scalar + Into<f64>>(large: T, maximum: T, ulp: T) {
    let third = <T as Scalar>::from_f64(1.0 / 3.0);
    for sign in [T::from_int(1), T::from_int(-1)] {
        // The original first sum overflows, but cancellation leaves large.
        assert_same(
            coordinate_mean([large * sign, large * sign, -large * sign]),
            large * sign * third,
        );
        // Cancel MAX with -MAX before the much smaller residual is scaled.
        let values = [maximum * sign, ulp * sign, -maximum * sign];
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            assert_same(
                coordinate_mean(permutation.map(|i| values[i])),
                ulp * sign * third,
            );
        }
        // Half an ulp is a tie near MAX; this order overflows, so it recovers.
        let tie = [maximum * sign, ulp / T::from_int(2) * sign, -maximum * sign];
        if !<T as eunomia::NumericElement>::is_finite(tie[0] + tie[1] + tie[2]) {
            assert_recovery_bound(tie, maximum);
        }
        // A constant-coordinate mean must retain the endpoint itself.
        assert_same(coordinate_mean([maximum * sign; 3]), maximum * sign);
    }
}

/// Sign, mantissa and exponent with `|x| = mantissa * 2^exponent`.
fn decompose(x: f64) -> (bool, u128, i32) {
    let bits = x.to_bits();
    let field = i32::try_from((bits >> 52) & 0x7ff).expect("invariant: 11-bit field");
    let fraction = u128::from(bits & ((1 << 52) - 1));
    match field {
        0 => (bits >> 63 == 1, fraction, -1074),
        _ => (bits >> 63 == 1, fraction | 1 << 52, field - 1075),
    }
}

/// Integer image of `x` in units of `2^unit_exp`, truncated toward zero.
fn fixed_point<T: Into<f64>>(x: T, unit_exp: i32) -> i128 {
    let (negative, mantissa, exponent) = decompose(x.into());
    let shift = exponent - unit_exp;
    let magnitude = if shift >= 0 {
        mantissa << shift
    } else {
        mantissa >> shift.unsigned_abs().min(127)
    };
    let sign: i128 = if negative { -1 } else { 1 };
    sign * i128::try_from(magnitude).expect("invariant: below 2^118 units")
}

/// Asserts the Rustdoc bounds (times 3, since `3 mean = s`) against the exact
/// sum `s` in integer units of `2^(e - 118)`, `2^e > MAX`: same sign `4u|s|`,
/// mixed signs `u(|lo + hi| + 3|s|)`, overflowing tail `3u|s|`, with
/// `u = 2^-p` applied as a right shift. The 3 units of slack are the
/// truncation of the three coordinates, below `2^-60` of the smallest
/// overflowing `|s|` (`ulp(MAX) / 2`), so it is bookkeeping, not a tolerance.
fn assert_recovery_bound<T: Scalar + Into<f64>>(triple: [T; 3], maximum: T) {
    let (_, max_mantissa, max_exponent) = decompose(maximum.into());
    let width = u128::BITS - max_mantissa.leading_zeros();
    let precision = width - max_mantissa.trailing_zeros();
    let unit_exp = max_exponent + i32::try_from(width).expect("invariant: width") - 118;
    let mut sorted = triple;
    sorted.sort_by(Scalar::total_cmp);
    let [lo, mid, hi] = sorted;
    let units = |x: T| fixed_point(x, unit_exp);
    let exact_sum = units(lo) + units(mid) + units(hi);
    let actual = coordinate_mean(triple);
    let error = (3 * units(actual) - exact_sum).abs();
    let ceil_shift = |n: i128| (n + (1_i128 << precision) - 1) >> precision;
    let bound = if lo >= T::from_int(0) || hi <= T::from_int(0) {
        ceil_shift(4 * exact_sum.abs())
    } else if <T as eunomia::NumericElement>::is_finite((lo + hi) + mid) {
        ceil_shift((units(lo) + units(hi)).abs() + 3 * exact_sum.abs())
    } else {
        ceil_shift(3 * exact_sum.abs())
    };
    assert!(
        error <= bound + 3,
        "{triple:?} -> {actual:?}: |3 result - sum| = {error} units, bound {bound}"
    );
    assert!(actual >= lo && actual <= hi, "{triple:?} -> {actual:?}");
}

fn verify_recovered_mean_bounds<T: Scalar + Into<f64>>(maximum: T, tiny: T) {
    let zero = T::from_int(0);
    let half = maximum / T::from_int(2);
    let values = [
        -maximum,
        -half,
        -T::from_int(1),
        -tiny,
        -zero,
        zero,
        tiny,
        T::from_int(1),
        half,
        maximum,
    ];
    for a in values {
        for b in values {
            for c in values {
                let triple = [a, b, c];
                let actual = coordinate_mean(triple);
                if <T as eunomia::NumericElement>::is_finite(a + b + c) {
                    assert_same(actual, legacy_centroid(triple));
                } else {
                    assert!(
                        <T as eunomia::NumericElement>::is_finite(actual),
                        "{triple:?} -> {actual:?}"
                    );
                    assert_recovery_bound(triple, maximum);
                }
            }
        }
    }
}

#[test]
fn recovered_mean_bounds_and_finite_legacy_bits() {
    verify_recovered_mean_bounds(f32::MAX, f32::from_bits(1));
    verify_recovered_mean_bounds(f64::MAX, f64::from_bits(1));
}

#[test]
fn overflow_cancellation_keeps_residual() {
    verify_centroid_cancellation(2.0_f32.powi(127), f32::MAX, 2.0_f32.powi(104));
    verify_centroid_cancellation(2.0_f64.powi(1023), f64::MAX, 2.0_f64.powi(971));
}

fn verify_ordinary_centroid_bits<T: Scalar>(tiny: T, large: T) {
    let zero = T::from_int(0);
    for values in [
        [tiny; 3],
        [-tiny; 3],
        [large, -large, tiny],
        [large, tiny, -large],
        [zero, -zero, -zero],
    ] {
        assert_same(coordinate_mean(values), legacy_centroid(values));
    }
    for a in -8..=8 {
        for b in -8..=8 {
            let values = [T::from_int(a), T::from_int(b), T::from_int(3 - a - b)];
            assert_same(coordinate_mean(values), legacy_centroid(values));
        }
    }
}

#[test]
fn centroid_ordinary_small_and_subnormal_bits() {
    verify_ordinary_centroid_bits(f32::from_bits(1), 2.0_f32.powi(64));
    verify_ordinary_centroid_bits(f64::from_bits(1), 2.0_f64.powi(512));
}

fn verify_nonfinite_coordinate<T: Scalar>() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = <T as Scalar>::from_f64(invalid);
        let values = [invalid, T::from_int(2), T::from_int(3)];
        assert_same(coordinate_mean(values), legacy_centroid(values));
    }
}

#[test]
fn nonfinite_coordinate_keeps_legacy_centroid() {
    verify_nonfinite_coordinate::<f32>();
    verify_nonfinite_coordinate::<f64>();
}

/// Each of the nine coordinates poisoned in turn keeps every axis legacy.
fn verify_nonfinite_vertex<T: Scalar>(large: T) {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (vertex, axis) in (0..9).map(|k| (k / 3, k % 3)) {
            let mut points = [Point3::new(large, large, large); 3];
            let poison = <T as Scalar>::from_f64(invalid);
            match axis {
                0 => points[vertex].x = poison,
                1 => points[vertex].y = poison,
                _ => points[vertex].z = poison,
            }
            let actual = triangle_centroid(&points[0], &points[1], &points[2]);
            for other in 0..3 {
                assert_same(
                    actual[other],
                    legacy_centroid([points[0][other], points[1][other], points[2][other]]),
                );
            }
        }
    }
}

#[test]
fn nonfinite_vertex_keeps_legacy_centroid_axes() {
    verify_nonfinite_vertex(2.0_f32.powi(127));
    verify_nonfinite_vertex(2.0_f64.powi(1023));
}
