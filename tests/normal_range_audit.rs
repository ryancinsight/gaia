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

fn verify_centroid_cancellation<T: Scalar>(large: T, maximum: T, ulp: T) {
    let third = <T as Scalar>::from_f64(1.0 / 3.0);
    for sign in [T::from_int(1), T::from_int(-1)] {
        // The original first sum overflows, but cancellation leaves large.
        assert_same(
            coordinate_mean([large * sign, large * sign, -large * sign]),
            large * sign * third,
        );
        // Cancel MAX with -MAX before the much smaller residual is scaled.
        let values = [maximum * sign, ulp * sign, -maximum * sign];
        // Half of ulp is a tie for sums near MAX. Where the original left to
        // right sum is finite its bits are kept (and may round the residual
        // away); where it overflows, recovery must preserve the residual.
        let tie = [maximum * sign, ulp / T::from_int(2) * sign, -maximum * sign];
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
            let tied = permutation.map(|i| tie[i]);
            if !<T as eunomia::NumericElement>::is_finite(tied[0] + tied[1] + tied[2]) {
                assert_same(coordinate_mean(tied), ulp / T::from_int(2) * sign * third);
            }
        }
        // A constant-coordinate mean must retain the endpoint itself.
        assert_same(coordinate_mean([maximum * sign; 3]), maximum * sign);
    }
}

fn verify_recovered_mean_bounds<T: Scalar>(maximum: T, tiny: T) {
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
                    let lo = triple
                        .into_iter()
                        .reduce(|x, y| if x < y { x } else { y })
                        .expect("three values");
                    let hi = triple
                        .into_iter()
                        .reduce(|x, y| if x > y { x } else { y })
                        .expect("three values");
                    assert!(actual >= lo && actual <= hi, "{triple:?} -> {actual:?}");
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

fn verify_nonfinite_vertex<T: Scalar>(large: T) {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut points = translated_triangle(large);
        points[0].y = <T as Scalar>::from_f64(invalid);
        let actual = triangle_centroid(&points[0], &points[1], &points[2]);
        for axis in 0..3 {
            assert_same(
                actual[axis],
                legacy_centroid([points[0][axis], points[1][axis], points[2][axis]]),
            );
        }
    }
}

#[test]
fn nonfinite_vertex_keeps_legacy_centroid_axes() {
    verify_nonfinite_vertex(2.0_f32.powi(127));
    verify_nonfinite_vertex(2.0_f64.powi(1023));
}
