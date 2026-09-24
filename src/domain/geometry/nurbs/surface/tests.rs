use super::super::knot::KnotVector;
use super::*;
use crate::domain::core::scalar::{Point3r, Real, Scalar, Vector3r};
use leto::geometry::Point3;

fn pt(x: Real, y: Real, z: Real) -> Point3r {
    Point3r::new(x, y, z)
}

fn grid_2x2(pts: &[Point3r]) -> ControlGrid {
    ControlGrid::new(pts.to_vec(), 2, 2)
}

// -- BSplineSurface --

#[test]
fn bspline_bilinear_corners() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let surf = BSplineSurface::clamped(grid_2x2(&pts), 1, 1).unwrap();
    assert!((surf.point(0.0, 0.0) - pt(0.0, 0.0, 0.0)).norm() < 1e-12);
    assert!((surf.point(1.0, 0.0) - pt(1.0, 0.0, 0.0)).norm() < 1e-12);
    assert!((surf.point(0.0, 1.0) - pt(0.0, 1.0, 0.0)).norm() < 1e-12);
    assert!((surf.point(1.0, 1.0) - pt(1.0, 1.0, 0.0)).norm() < 1e-12);
}

#[test]
fn bspline_bilinear_midpoint() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let surf = BSplineSurface::clamped(grid_2x2(&pts), 1, 1).unwrap();
    let mid = surf.point(0.5, 0.5);
    assert!((mid - pt(0.5, 0.5, 0.0)).norm() < 1e-12);
}

#[test]
fn bspline_normal_flat_patch() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let surf = BSplineSurface::clamped(grid_2x2(&pts), 1, 1).unwrap();
    let n = surf.normal(0.5, 0.5).expect("should have valid normal");
    assert!((n.into_inner().dot(Vector3r::new(0.0, 0.0, 1.0)) - 1.0).abs() < 1e-10);
}

#[test]
fn bspline_wrong_knot_count_errors() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let grid = grid_2x2(&pts);
    let ku = KnotVector::try_new(vec![0.0, 1.0]).unwrap(); // too short
    let kv = KnotVector::clamped_uniform(1, 1);
    assert!(BSplineSurface::new(grid, ku, kv, 1, 1).is_err());
}

// -- NurbsSurface --

#[test]
fn nurbs_unit_weight_matches_bspline() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let bs = BSplineSurface::clamped(ControlGrid::new(pts.clone(), 2, 2), 1, 1).unwrap();
    let ns = NurbsSurface::from_bspline(bs.clone());
    for i in 0..=5 {
        for j in 0..=5 {
            let u = Real::from(i) / 5.0;
            let v = Real::from(j) / 5.0;
            let pb = bs.point(u, v);
            let pn = ns.point(u, v);
            assert!(
                (pb - pn).norm() < 1e-12,
                "unit-weight NURBS != B-spline at ({u}, {v})"
            );
        }
    }
}

#[test]
fn nurbs_clamped_corners_interpolated() {
    let pts = vec![
        pt(0.0, 0.0, 1.0),
        pt(1.0, 0.0, 2.0),
        pt(0.0, 1.0, 3.0),
        pt(1.0, 1.0, 4.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts.clone(), 2, 2), 1, 1).unwrap();
    assert!((surf.point(0.0, 0.0) - pts[0]).norm() < 1e-12);
    assert!((surf.point(1.0, 0.0) - pts[1]).norm() < 1e-12);
    assert!((surf.point(0.0, 1.0) - pts[2]).norm() < 1e-12);
    assert!((surf.point(1.0, 1.0) - pts[3]).norm() < 1e-12);
}

#[test]
fn nurbs_aabb_non_degenerate() {
    let pts = vec![
        pt(-1.0, -1.0, 0.0),
        pt(1.0, -1.0, 0.0),
        pt(-1.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
    let aabb = surf.aabb(8);
    assert!(aabb.max.x > 0.0 && aabb.min.x < 0.0);
}

#[test]
fn nurbs_normal_not_zero() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
    assert!(surf.normal(0.5, 0.5).is_some());
}

#[test]
fn weight_grid_mismatch_errors() {
    let pts = vec![
        pt(0.0, 0.0, 0.0),
        pt(1.0, 0.0, 0.0),
        pt(0.0, 1.0, 0.0),
        pt(1.0, 1.0, 0.0),
    ];
    let grid = ControlGrid::new(pts, 2, 2);
    let wrong_weights = WeightGrid::uniform(3, 3); // 3x3 != 2x2
    let ku = KnotVector::clamped_uniform(1, 1);
    let kv = KnotVector::clamped_uniform(1, 1);
    assert!(NurbsSurface::new(grid, wrong_weights, ku, kv, 1, 1).is_err());
}

/// The scalar seam monomorphizes: a dyadic bilinear patch evaluates
/// identically at `f32` and `f64` — every control coordinate, knot, and
/// parameter is exact at both precisions.
#[test]
fn f32_instantiation_matches_f64_on_dyadic_patch() {
    let corners32 = [
        Point3::<f32>::new(0.0, 0.0, 0.0),
        Point3::<f32>::new(1.0, 0.0, 0.25),
        Point3::<f32>::new(0.0, 1.0, 0.5),
        Point3::<f32>::new(1.0, 1.0, 0.75),
    ];
    let corners64 = [
        Point3::<f64>::new(0.0, 0.0, 0.0),
        Point3::<f64>::new(1.0, 0.0, 0.25),
        Point3::<f64>::new(0.0, 1.0, 0.5),
        Point3::<f64>::new(1.0, 1.0, 0.75),
    ];
    let s32 =
        BSplineSurface::<f32>::clamped(ControlGrid::new(corners32.to_vec(), 2, 2), 1, 1).unwrap();
    let s64 =
        BSplineSurface::<f64>::clamped(ControlGrid::new(corners64.to_vec(), 2, 2), 1, 1).unwrap();
    for i in 0..=4 {
        for j in 0..=4 {
            let u = f64::from(i) / 4.0;
            let v = f64::from(j) / 4.0;
            let p32 = s32.point(<f32 as Scalar>::from_f64(u), <f32 as Scalar>::from_f64(v));
            let p64 = s64.point(u, v);
            assert_eq!(p32.x.to_bits(), <f32 as Scalar>::from_f64(p64.x).to_bits());
            assert_eq!(p32.y.to_bits(), <f32 as Scalar>::from_f64(p64.y).to_bits());
            assert_eq!(p32.z.to_bits(), <f32 as Scalar>::from_f64(p64.z).to_bits());
        }
    }
}
