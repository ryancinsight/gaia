//! Finite geometry contracts on unchanged Gaia main; audit-only regressions.

use gaia::domain::core::scalar::Scalar;
use gaia::domain::geometry::normal::{newell_normal, triangle_centroid, triangle_normal};
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

fn verify_newell<T: Scalar>(x: T) {
    // CCW triangle in the yz plane. Translating its x coordinate changes
    // neither its cross product (9,0,0) nor its unit normal (1,0,0).
    let vertices = translated_triangle(x);
    let normal = newell_normal(&vertices).expect("nondegenerate finite triangle");
    assert_eq!(
        (normal.x, normal.y, normal.z),
        (T::from_int(1), T::from_int(0), T::from_int(0))
    );
}

fn verify_triangle_normal<T: Scalar>(side: T) {
    let zero = T::from_int(0);
    let a = Point3::new(zero, zero, zero);
    let b = Point3::new(side, zero, zero);
    let c = Point3::new(zero, side, zero);
    // The cross product is (0,0,side^2), finite in every fixture below.
    let normal = triangle_normal(&a, &b, &c).expect("nondegenerate finite triangle");
    assert_eq!((normal.x, normal.y, normal.z), (zero, zero, T::from_int(1)));
}

#[test]
fn ordinary_centroid_control() {
    verify_centroid(1.0_f32);
    verify_centroid(1.0_f64);
}

#[test]
fn centroid_binary32_finite_mean() {
    verify_centroid(2.0_f32.powi(127));
}

#[test]
fn centroid_binary64_finite_mean() {
    verify_centroid(2.0_f64.powi(1023));
}

#[test]
fn ordinary_newell_control() {
    verify_newell(1.0_f32);
    verify_newell(1.0_f64);
}

#[test]
fn newell_binary32_translation_invariance() {
    verify_newell(2.0_f32.powi(127));
}

#[test]
fn newell_binary64_translation_invariance() {
    verify_newell(2.0_f64.powi(1023));
}

#[test]
fn ordinary_triangle_normal_control() {
    verify_triangle_normal(1.0_f32);
    verify_triangle_normal(1.0_f64);
}

fn verify_translated_triangle_normal<T: Scalar>(x: T) {
    let [a, b, c] = translated_triangle(x);
    let normal = triangle_normal(&a, &b, &c).expect("finite cross product (9,0,0)");
    assert_eq!(
        (normal.x, normal.y, normal.z),
        (T::from_int(1), T::from_int(0), T::from_int(0))
    );
}

#[test]
fn translated_triangle_normal_control() {
    verify_translated_triangle_normal(2.0_f32.powi(127));
    verify_translated_triangle_normal(2.0_f64.powi(1023));
}

#[test]
fn triangle_normal_binary32_finite_cross() {
    verify_triangle_normal(2.0_f32.powi(32));
}

#[test]
fn triangle_normal_binary64_finite_cross() {
    verify_triangle_normal(2.0_f64.powi(256));
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
fn overflow_cancellation_binary32() {
    verify_centroid_cancellation(2.0_f32.powi(127), f32::MAX, 2.0_f32.powi(104));
}

#[test]
fn overflow_cancellation_binary64() {
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

fn legacy_newell<T: Scalar>(vertices: &[Point3<T>]) -> Option<leto::geometry::Vector3<T>> {
    if vertices.len() < 3 {
        return None;
    }
    let mut normal = leto::geometry::Vector3::<T>::zeros();
    for i in 0..vertices.len() {
        let curr = &vertices[i];
        let next = &vertices[(i + 1) % vertices.len()];
        normal[0] += (curr.y - next.y) * (curr.z + next.z);
        normal[1] += (curr.z - next.z) * (curr.x + next.x);
        normal[2] += (curr.x - next.x) * (curr.y + next.y);
    }
    let len = normal.norm();
    if len < T::tolerance() {
        None
    } else {
        Some(normal / len)
    }
}

fn assert_normal_same<T: Scalar>(vertices: &[Point3<T>]) {
    match (newell_normal(vertices), legacy_newell(vertices)) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            for axis in 0..3 {
                assert_same(actual[axis], expected[axis]);
            }
        }
        values => panic!("changed legacy result: {values:?}"),
    }
}

fn verify_newell_small_and_cancellation<T: Scalar>(large: T, tiny: T) {
    let zero = T::from_int(0);
    let one = T::from_int(1);
    for x in [one, large] {
        // Tiny triangles retain the existing absolute degeneracy threshold.
        let points = [
            Point3::new(x, zero, zero),
            Point3::new(x, tiny, zero),
            Point3::new(x, zero, tiny),
        ];
        assert!(newell_normal(&points).is_none());
        // Two oppositely wound loops have exactly cancelling signed area.
        let [a, b, c] = translated_triangle(x);
        assert!(newell_normal(&[a, b, c, a, c, b]).is_none());
        // A small but admitted triangle has an exact positive x normal.
        let side = one / T::from_int(64);
        let points = [
            Point3::new(x, zero, zero),
            Point3::new(x, side, zero),
            Point3::new(x, zero, side),
        ];
        let normal = newell_normal(&points).expect("area exceeds both native tolerances");
        assert_eq!((normal.x, normal.y, normal.z), (one, zero, zero));
    }
}

#[test]
fn newell_small_and_cancellation_binary32() {
    verify_newell_small_and_cancellation(2.0_f32.powi(127), f32::from_bits(1));
}

#[test]
fn newell_small_and_cancellation_binary64() {
    verify_newell_small_and_cancellation(2.0_f64.powi(1023), f64::from_bits(1));
}

fn verify_newell_ordinary_bits<T: Scalar>() {
    for shift in -16..=16 {
        let x = T::from_int(shift);
        let mut points = [
            Point3::new(T::from_int(0), T::from_int(0), T::from_int(0)),
            Point3::new(T::from_int(3), T::from_int(0), T::from_int(1)),
            Point3::new(T::from_int(4), T::from_int(2), T::from_int(-1)),
            Point3::new(T::from_int(0), T::from_int(3), T::from_int(2)),
        ];
        for p in &mut points {
            p.x += x;
            p.y -= x;
            p.z += x;
        }
        assert_normal_same(&points);
    }
}

#[test]
fn newell_ordinary_accumulation_bits() {
    verify_newell_ordinary_bits::<f32>();
    verify_newell_ordinary_bits::<f64>();
}

fn verify_nonfinite_results<T: Scalar>() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = <T as Scalar>::from_f64(invalid);
        let values = [invalid, T::from_int(2), T::from_int(3)];
        assert_same(coordinate_mean(values), legacy_centroid(values));
        let points = translated_triangle(invalid);
        assert_normal_same(&points);
    }
}

#[test]
fn nonfinite_results_keep_legacy_behavior() {
    verify_nonfinite_results::<f32>();
    verify_nonfinite_results::<f64>();
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
        assert_normal_same(&points);
    }
}

#[test]
fn nonfinite_vertices_preserve_whole_legacy_results() {
    verify_nonfinite_vertex(2.0_f32.powi(127));
    verify_nonfinite_vertex(2.0_f64.powi(1023));
}

fn verify_newell_coordinate_planes<T: Scalar>(large: T) {
    let zero = T::from_int(0);
    let three = T::from_int(3);
    for axis in 0..3 {
        let mut a = [zero; 3];
        a[axis] = large;
        let mut b = a;
        let mut c = a;
        b[(axis + 1) % 3] = three;
        c[(axis + 2) % 3] = three;
        let points = [a, b, c].map(|p| Point3::new(p[0], p[1], p[2]));
        let normal = newell_normal(&points).expect("nondegenerate coordinate-plane triangle");
        for component in 0..3 {
            assert_same(
                normal[component],
                if component == axis {
                    T::from_int(1)
                } else {
                    zero
                },
            );
        }
    }
}

#[test]
fn newell_translation_all_coordinate_planes() {
    verify_newell_coordinate_planes(2.0_f32.powi(127));
    verify_newell_coordinate_planes(2.0_f64.powi(1023));
}
