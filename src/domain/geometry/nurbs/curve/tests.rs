use super::super::knot::KnotVector;
use super::*;
use crate::domain::core::scalar::{Real, Scalar};
use leto::geometry::Vector as SVector;

type V3 = SVector<Real, 3>;
type V2 = SVector<Real, 2>;

fn linear_rational_curve<T: Scalar>(
    points: [SVector<T, 2>; 2],
    weights: [T; 2],
) -> NurbsCurve<2, T> {
    NurbsCurve::new(
        points.to_vec(),
        weights.to_vec(),
        KnotVector::<T>::clamped_uniform(1, 1),
        1,
    )
    .unwrap()
}

fn v3(x: Real, y: Real, z: Real) -> V3 {
    V3::new(x, y, z)
}

// ── BSplineCurve ─────────────────────────────────────────────────────────

#[test]
fn bspline_linear_interpolates_endpoints() {
    // Linear (p=1), 2 control points: C(0)=P0, C(1)=P1
    let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 3.0)];
    let curve = BSplineCurve::clamped(pts, 1);
    let start = curve.point(0.0);
    let end = curve.point(1.0);
    assert!((start - v3(0.0, 0.0, 0.0)).norm() < 1e-12);
    assert!((end - v3(1.0, 2.0, 3.0)).norm() < 1e-12);
}

#[test]
fn bspline_quadratic_midpoint() {
    // Quadratic with 3 control points: P0=(0,0,0), P1=(1,2,0), P2=(2,0,0)
    // At t=0.5 the result should be between the control points
    let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 0.0), v3(2.0, 0.0, 0.0)];
    let curve = BSplineCurve::clamped(pts, 2);
    let mid = curve.point(0.5);
    // B-spline interpolates convex hull: y should be positive and x near 1
    assert!(mid[0] > 0.9 && mid[0] < 1.1);
    assert!(mid[1] > 0.0);
}

#[test]
fn bspline_tangent_linear() {
    // Linear curve from (0,0,0) to (1,1,1): tangent should be constant (1,1,1)
    let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)];
    let curve = BSplineCurve::clamped(pts, 1);
    let (_, tan) = curve.point_and_tangent(0.5);
    // Tangent direction is (1,1,1), magnitude = degree * (P1-P0) / knot diff
    assert!(
        tan.dot(v3(1.0, 1.0, 1.0)) > 0.0,
        "tangent must point in positive direction"
    );
}

#[test]
fn bspline_sample_uniform_count() {
    let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), v3(2.0, 0.0, 0.0)];
    let curve = BSplineCurve::clamped(pts, 2);
    let samples = curve.sample_uniform(11);
    assert_eq!(samples.len(), 11);
}

#[test]
fn bspline_creation_errors() {
    let kv = KnotVector::clamped_uniform(2, 2);
    // Wrong degree: knot vector has n+p+2=3+1+2... let's just test no control points
    let empty: Vec<V3> = vec![];
    assert!(BSplineCurve::new(empty, kv, 2).is_err());
}

// ── NurbsCurve ───────────────────────────────────────────────────────────

#[test]
fn nurbs_unit_weight_matches_bspline() {
    // NURBS with w=1 everywhere should equal B-spline
    let pts = vec![v3(0.0, 0.0, 0.0), v3(0.5, 1.0, 0.0), v3(1.0, 0.0, 0.0)];
    let bs = BSplineCurve::clamped(pts.clone(), 2);
    let weights = vec![1.0_f64; 3];
    let kv = KnotVector::clamped_uniform(2, 2);
    let nc = NurbsCurve::new(pts, weights, kv, 2).unwrap();
    for i in 0..=10 {
        let t = Real::from(i) / 10.0;
        let pb = bs.point(t);
        let pn = nc.point(t);
        assert!(
            (pb - pn).norm() < 1e-12,
            "unit-weight NURBS ≠ B-spline at t={}: |diff|={}",
            t,
            (pb - pn).norm()
        );
    }
}

#[test]
fn nurbs_quarter_circle() {
    // Exact unit quarter-circle in XY plane:
    // P0=(1,0), w0=1  P1=(1,1), w1=1/√2  P2=(0,1), w2=1
    let sq2_inv: Real = <Real as Scalar>::from_f64(std::f64::consts::FRAC_1_SQRT_2);
    let ctrl = vec![V2::new(1.0, 0.0), V2::new(1.0, 1.0), V2::new(0.0, 1.0)];
    let weights = vec![1.0, sq2_inv, 1.0];
    let knots = KnotVector::try_new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();

    // Sample 11 points and check that they lie on the unit circle
    for i in 0..=10 {
        let t = Real::from(i) / 10.0;
        let pt = curve.point(t);
        let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
        assert!(
            (r - 1.0).abs() < 1e-10,
            "quarter-circle point not on unit circle at t={t}: r={r}"
        );
    }
}

#[test]
fn nurbs_endpoints_interpolate() {
    // Any clamped NURBS must pass through first and last control points
    let ctrl = vec![v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0), v3(7.0, 8.0, 9.0)];
    let weights = vec![1.0, 0.5, 2.0];
    let knots = KnotVector::clamped_uniform(2, 2);
    let curve = NurbsCurve::new(ctrl.clone(), weights, knots, 2).unwrap();
    let start = curve.point(0.0);
    let end = curve.point(1.0);
    assert!((start - ctrl[0]).norm() < 1e-12, "start should equal P0");
    assert!((end - ctrl[2]).norm() < 1e-12, "end should equal P2");
}

#[test]
fn nurbs_non_positive_weight_errors() {
    let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
    let weights = vec![1.0, 0.0]; // zero weight is invalid
    let knots = KnotVector::clamped_uniform(1, 1);
    assert!(matches!(
        NurbsCurve::new(ctrl, weights, knots, 1),
        Err(CurveError::NonPositiveWeight { index: 1 })
    ));
}

#[test]
fn nurbs_aabb_contains_control_points() {
    let ctrl = vec![v3(-1.0, -2.0, -3.0), v3(0.0, 0.0, 0.0), v3(4.0, 5.0, 6.0)];
    let weights = vec![1.0, 1.5, 1.0];
    let knots = KnotVector::clamped_uniform(2, 2);
    let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
    let aabb = curve.aabb(50);
    // Curve lies in convex hull of control points; aabb should be non-degenerate
    assert!(aabb.min.x <= 0.0 && aabb.max.x >= 1.0);
}

#[test]
fn nurbs_tangent_non_zero_for_non_degenerate_curve() {
    let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 0.0), v3(2.0, 0.0, 0.0)];
    let weights = vec![1.0_f64; 3];
    let knots = KnotVector::clamped_uniform(2, 2);
    let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
    let (_, tan) = curve.point_and_tangent(0.5);
    assert!(tan.norm() > 0.0, "tangent at midpoint should be non-zero");
}

// ── Generic instantiation ───────────────────────────────────────────────

/// A dyadic linear curve evaluates identically at `f32` and `f64`:
/// all knot values, parameters, and control coordinates are exact at
/// both precisions, so native-`T` evaluation is the same computation.
#[test]
fn f32_instantiation_interpolates_dyadic_endpoints() {
    let curve32 = {
        let pts = vec![
            SVector::<f32, 3>::new(0.0, 0.0, 0.0),
            SVector::<f32, 3>::new(1.0, 0.5, 0.25),
        ];
        BSplineCurve::<3, f32>::clamped(pts, 1)
    };
    let curve64 = {
        let pts = vec![
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 0.5, 0.25),
        ];
        BSplineCurve::<3, f64>::clamped(pts, 1)
    };
    for i in 0..=8 {
        let t32 = <f32 as Scalar>::from_f64(f64::from(i) / 8.0);
        let t64 = f64::from(i) / 8.0;
        let p32 = curve32.point(t32);
        let p64 = curve64.point(t64);
        for k in 0..3 {
            assert_eq!(
                p32[k].to_bits(),
                <f32 as Scalar>::from_f64(p64[k]).to_bits()
            );
        }
    }
    let start = curve32.point(0.0);
    let end = curve32.point(1.0);
    assert_eq!(start[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(end[2].to_bits(), 0.25_f32.to_bits());
}

/// Unit-weight NURBS equals the B-spline at `f32` — the identity holds
/// per instantiation, not only at `f64`.
#[test]
fn f32_unit_weight_nurbs_matches_bspline() {
    let pts32 = vec![
        SVector::<f32, 3>::new(0.0, 0.0, 0.0),
        SVector::<f32, 3>::new(0.5, 1.0, 0.0),
        SVector::<f32, 3>::new(1.0, 0.0, 0.0),
    ];
    let bs = BSplineCurve::<3, f32>::clamped(pts32.clone(), 2);
    let nc = NurbsCurve::<3, f32>::from_bspline(bs.clone());
    for i in 0..=10 {
        let t = <f32 as Scalar>::from_f64(f64::from(i) / 10.0);
        let pb = bs.point(t);
        let pn = nc.point(t);
        assert!((pb - pn).norm() < 1e-6);
    }
}

/// A rational quarter-circle at `f32`: the non-dyadic weight 1/√2
/// rounds at `f32` precision, so the radius deviates from 1 by the
/// propagated weight error. Bound: `C = A/W` with
/// `∂C/∂w₁ = N₁·(P₁ − C)/W`, and `N₁ ≤ 1`, `|P₁ − C| ≤ √2`,
/// `W ≥ min wᵢ = √2/2`, so `|r − 1| ≤ 2·δw` with
/// `δw = ulp(√2/2) = 2⁻²⁴`; the assertion carries 8× headroom over
/// `2·2⁻²⁴ ≈ 1.2e-7`.
#[test]
fn f32_rational_quarter_circle_radius_is_weight_rounding_bounded() {
    let w = <f32 as Scalar>::from_f64(std::f64::consts::FRAC_1_SQRT_2);
    let ctrl = vec![
        SVector::<f32, 2>::new(1.0, 0.0),
        SVector::<f32, 2>::new(1.0, 1.0),
        SVector::<f32, 2>::new(0.0, 1.0),
    ];
    let weights = vec![1.0_f32, w, 1.0];
    let knots = KnotVector::<f32>::try_new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let curve = NurbsCurve::<2, f32>::new(ctrl, weights, knots, 2).unwrap();
    for i in 0..=16 {
        let t = <f32 as Scalar>::from_f64(f64::from(i) / 16.0);
        let pt = curve.point(t);
        let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
        assert!(
            (r - 1.0).abs() < 1e-6,
            "f32 quarter-circle radius drift exceeds the weight-rounding bound at t={t}: r={r}"
        );
    }
}

fn assert_extreme_weight_curve_tangent<T: Scalar>(tiny: T, large: T) {
    let zero = <T as Scalar>::from_f64(0.0);
    let small_point = SVector::<T, 2>::new(tiny, zero);
    let curve = linear_rational_curve([SVector::<T, 2>::zeros(), small_point], [tiny, large]);

    let (point, tangent) = curve.point_and_tangent(zero);
    assert_eq!(point, SVector::<T, 2>::zeros());
    assert_eq!(tangent, SVector::<T, 2>::new(large, zero));

    let half = <T as Scalar>::from_f64(0.5);
    let curve = linear_rational_curve(
        [
            SVector::<T, 2>::new(-(large * half), zero),
            SVector::<T, 2>::new(large, zero),
        ],
        [<T as Scalar>::from_f64(1.0), <T as Scalar>::from_f64(0.25)],
    );
    let (_, tangent) = curve.point_and_tangent(zero);
    assert_eq!(tangent[0], large * <T as Scalar>::from_f64(0.375));
}

/// The endpoint derivative survives weight-ratio and coordinate-subtraction overflow.
#[test]
fn rational_curve_derivative_scales_weight_and_coordinate_factors() {
    assert_extreme_weight_curve_tangent::<f32>(f32::from_bits(1), f32::MAX);
    assert_extreme_weight_curve_tangent::<f64>(f64::from_bits(1), f64::MAX);
}

/// A small weight multiplied by a large coordinate remains representable even
/// when normalizing the weight first would underflow to zero.
#[test]
fn rational_curve_preserves_scaled_point_contributions() {
    let large = 2.0_f64.powi(600);
    let tiny = 2.0_f64.powi(-600);
    let curve = linear_rational_curve(
        [
            SVector::<f64, 2>::new(large, 0.0),
            SVector::<f64, 2>::zeros(),
        ],
        [tiny, large],
    );

    assert_eq!(curve.point(0.5)[0], tiny);
    let (point, tangent) = curve.point_and_tangent(0.5);
    assert_eq!(point[0], tiny);
    assert_eq!(tangent[0], -2.0_f64.powi(-598));
}

/// The tangent of a dyadic linear curve evaluates exactly at `f32`:
/// every knot, control point, and parameter is dyadic, so the
/// degree-lowered recurrence involves no rounding, and the analytic
/// value `C' = (P₁ − P₀)/(ξᵢ₊₁ − ξᵢ) = P₁ − P₀` is reproduced bit-for-bit.
#[test]
fn f32_derivative_of_dyadic_linear_curve_is_exact() {
    let pts = vec![
        SVector::<f32, 3>::new(0.0, 0.0, 0.0),
        SVector::<f32, 3>::new(1.0, 0.5, 0.25),
    ];
    let curve = BSplineCurve::<3, f32>::clamped(pts, 1);
    let (_, tan) = curve.point_and_tangent(0.5);
    assert_eq!(tan[0].to_bits(), 1.0_f32.to_bits());
    assert_eq!(tan[1].to_bits(), 0.5_f32.to_bits());
    assert_eq!(tan[2].to_bits(), 0.25_f32.to_bits());
}
