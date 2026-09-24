use crate::domain::core::scalar::{Real, Scalar};

use super::super::curve::NurbsCurve;
use super::super::knot::KnotVector;
use super::super::parameter::uniform_parameter;
use super::super::surface::{ControlGrid, NurbsSurface};
use super::*;
use crate::domain::core::scalar::Point3r;
use leto::geometry::Vector as SVector;

type V3 = SVector<Real, 3>;
fn v3(x: Real, y: Real, z: Real) -> V3 {
    V3::new(x, y, z)
}
fn pt3(x: Real, y: Real, z: Real) -> Point3r {
    Point3r::new(x, y, z)
}

// -- Surface tessellation --

#[test]
fn tessellate_flat_surface_produces_triangles() {
    let pts = vec![
        pt3(0.0, 0.0, 0.0),
        pt3(1.0, 0.0, 0.0),
        pt3(0.0, 1.0, 0.0),
        pt3(1.0, 1.0, 0.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
    let opts = TessellationOptions {
        min_segments: 2,
        max_angle_deg: 5.0,
        max_depth: 3,
    };
    let mesh = tessellate_surface(&surf, &opts);
    // 2*2 coarse grid -> 4 quads -> 8 triangles (all flat, no subdivision)
    assert_eq!(
        mesh.face_count(),
        8,
        "flat 2x2 grid should give exactly 8 triangles"
    );
    assert!(mesh.vertex_count() >= 3);
}

#[test]
fn tessellate_flat_surface_vertices_on_plane() {
    let pts = vec![
        pt3(0.0, 0.0, 0.0),
        pt3(2.0, 0.0, 0.0),
        pt3(0.0, 2.0, 0.0),
        pt3(2.0, 2.0, 0.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
    let opts = TessellationOptions {
        min_segments: 4,
        max_angle_deg: 5.0,
        max_depth: 4,
    };
    let mesh = tessellate_surface(&surf, &opts);
    // Every vertex of a flat (z=0) surface should have z == 0
    for i in 0..mesh.vertex_count() {
        let pos = mesh
            .vertices
            .position(crate::domain::core::index::VertexId::new(i as u32));
        assert!(pos.z.abs() < 1e-10, "vertex z={} should be ~0", pos.z);
    }
}

#[test]
fn tessellate_surface_min_segments_respected() {
    let pts = vec![
        pt3(0.0, 0.0, 0.0),
        pt3(1.0, 0.0, 0.0),
        pt3(0.0, 1.0, 0.0),
        pt3(1.0, 1.0, 0.0),
    ];
    let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
    let opts = TessellationOptions {
        min_segments: 8,
        max_angle_deg: 100.0,
        max_depth: 0,
    };
    // max_angle=100 deg means no subdivision beyond min_segments; depth=0 means stop immediately
    // 8x8 = 64 quads -> 128 triangles
    let mesh = tessellate_surface(&surf, &opts);
    assert_eq!(mesh.face_count(), 128);
}

// -- Curve tessellation --

#[test]
fn tessellate_line_segment_returns_endpoints() {
    // Linear NURBS from (0,0,0) to (1,1,1)
    let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)];
    let weights = vec![1.0_f64; 2];
    let knots = KnotVector::clamped_uniform(1, 1);
    let curve = NurbsCurve::new(ctrl, weights, knots, 1).unwrap();
    let opts = TessellationOptions {
        min_segments: 4,
        max_angle_deg: 5.0,
        max_depth: 4,
    };
    let pts = tessellate_curve(&curve, &opts);
    assert!(pts.len() >= 2);
    // First point near (0,0,0), last point near (1,1,1)
    assert!((pts[0] - pt3(0.0, 0.0, 0.0)).norm() < 1e-12);
    assert!((pts[pts.len() - 1] - pt3(1.0, 1.0, 1.0)).norm() < 1e-12);
}

#[test]
fn tessellate_curve_monotone_along_line() {
    // Straight-line NURBS: tessellate should produce monotone x values
    let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
    let weights = vec![1.0_f64; 2];
    let knots = KnotVector::clamped_uniform(1, 1);
    let curve = NurbsCurve::new(ctrl, weights, knots, 1).unwrap();
    let opts = TessellationOptions::default();
    let pts = tessellate_curve(&curve, &opts);
    // x should increase monotonically (or be equal)
    for w in pts.windows(2) {
        assert!(w[1].x >= w[0].x - 1e-12, "x should be non-decreasing");
    }
}

#[test]
fn tessellate_curve_at_least_min_segments_plus_one() {
    let ctrl = vec![v3(0.0, 0.0, 0.0), v3(0.5, 1.0, 0.0), v3(1.0, 0.0, 0.0)];
    let weights = vec![1.0_f64; 3];
    let knots = KnotVector::clamped_uniform(2, 2);
    let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
    let opts = TessellationOptions {
        min_segments: 8,
        max_angle_deg: 100.0,
        max_depth: 0,
    };
    let pts = tessellate_curve(&curve, &opts);
    // With max_angle=100 and max_depth=0, only min_segments cuts are made
    assert_eq!(
        pts.len(),
        opts.min_segments + 1,
        "should have min_segments+1 points"
    );
}

/// The flat patch's welded tessellation vertices match direct evaluations
/// at the same three-segment grid parameters.
#[test]
fn tessellation_vertices_match_evaluated_grid() {
    let corners32 = [
        leto::geometry::Point3::<f32>::new(0.0, 0.0, 0.0),
        leto::geometry::Point3::<f32>::new(1.0, 0.0, 0.0),
        leto::geometry::Point3::<f32>::new(0.0, 1.0, 0.0),
        leto::geometry::Point3::<f32>::new(1.0, 1.0, 0.0),
    ];
    let surf =
        NurbsSurface::<f32>::clamped(ControlGrid::new(corners32.to_vec(), 2, 2), 1, 1).unwrap();
    let opts = TessellationOptions::<f32> {
        min_segments: 3,
        max_angle_deg: <f32 as Scalar>::from_f64(180.0),
        max_depth: 0,
    };
    let mesh = tessellate_surface(&surf, &opts);
    assert_eq!(
        mesh.vertex_count(),
        16,
        "the 4x4 parameter grid has 16 corners"
    );

    let mesh_bits: Vec<[u32; 3]> = mesh
        .vertices
        .positions()
        .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
        .collect();
    for i in 0..4 {
        for j in 0..4 {
            let u = uniform_parameter(0.0, 1.0, i, 3);
            let v = uniform_parameter(0.0, 1.0, j, 3);
            let expected = surf.point(u, v);
            let key = [
                expected.x.to_bits(),
                expected.y.to_bits(),
                expected.z.to_bits(),
            ];
            assert!(
                mesh_bits.contains(&key),
                "native-parameter position ({i}, {j}) must be welded into the mesh bit-for-bit"
            );
        }
    }
}

/// The scalar seam monomorphizes: a dyadic flat patch tessellates into an
/// `IndexedMesh<f32>` with the same topology as the `f64` run, and every
/// welded vertex coordinate is exactly representable at both precisions.
#[test]
fn f32_tessellation_matches_f64_on_dyadic_patch() {
    let corners32 = [
        leto::geometry::Point3::<f32>::new(0.0, 0.0, 0.0),
        leto::geometry::Point3::<f32>::new(1.0, 0.0, 0.0),
        leto::geometry::Point3::<f32>::new(0.0, 1.0, 0.0),
        leto::geometry::Point3::<f32>::new(1.0, 1.0, 0.0),
    ];
    let corners64 = [
        leto::geometry::Point3::<f64>::new(0.0, 0.0, 0.0),
        leto::geometry::Point3::<f64>::new(1.0, 0.0, 0.0),
        leto::geometry::Point3::<f64>::new(0.0, 1.0, 0.0),
        leto::geometry::Point3::<f64>::new(1.0, 1.0, 0.0),
    ];
    let surf32 =
        NurbsSurface::<f32>::clamped(ControlGrid::new(corners32.to_vec(), 2, 2), 1, 1).unwrap();
    let surf64 =
        NurbsSurface::<f64>::clamped(ControlGrid::new(corners64.to_vec(), 2, 2), 1, 1).unwrap();
    let opts32 = TessellationOptions::<f32> {
        min_segments: 2,
        max_angle_deg: <f32 as Scalar>::from_f64(5.0),
        max_depth: 3,
    };
    let opts64 = TessellationOptions::<f64> {
        min_segments: 2,
        max_angle_deg: 5.0,
        max_depth: 3,
    };
    let mesh32 = tessellate_surface(&surf32, &opts32);
    let mesh64 = tessellate_surface(&surf64, &opts64);
    assert_eq!(mesh32.face_count(), mesh64.face_count());
    assert_eq!(mesh32.vertex_count(), mesh64.vertex_count());
    for i in 0..mesh32.vertex_count() {
        let p32 = mesh32
            .vertices
            .position(crate::domain::core::index::VertexId::new(i as u32));
        let p64 = mesh64
            .vertices
            .position(crate::domain::core::index::VertexId::new(i as u32));
        assert_eq!(p32.x.to_bits(), <f32 as Scalar>::from_f64(p64.x).to_bits());
        assert_eq!(p32.y.to_bits(), <f32 as Scalar>::from_f64(p64.y).to_bits());
        assert_eq!(p32.z.to_bits(), <f32 as Scalar>::from_f64(p64.z).to_bits());
    }
}

/// A dyadic linear curve tessellates identically at `f32` and `f64`:
/// the quarter-point parameters and straight-line evaluations are exact
/// at both precisions.
#[test]
fn f32_curve_tessellation_matches_f64_on_dyadic_segment() {
    let curve32 = NurbsCurve::<3, f32>::new(
        vec![
            SVector::<f32, 3>::new(0.0, 0.0, 0.0),
            SVector::<f32, 3>::new(1.0, 0.5, 0.25),
        ],
        vec![1.0_f32; 2],
        KnotVector::<f32>::clamped_uniform(1, 1),
        1,
    )
    .unwrap();
    let curve64 = NurbsCurve::<3, f64>::new(
        vec![
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 0.5, 0.25),
        ],
        vec![1.0_f64; 2],
        KnotVector::<f64>::clamped_uniform(1, 1),
        1,
    )
    .unwrap();
    let pts32 = tessellate_curve(
        &curve32,
        &TessellationOptions::<f32> {
            min_segments: 4,
            max_angle_deg: <f32 as Scalar>::from_f64(5.0),
            max_depth: 2,
        },
    );
    let pts64 = tessellate_curve(
        &curve64,
        &TessellationOptions::<f64> {
            min_segments: 4,
            max_angle_deg: 5.0,
            max_depth: 2,
        },
    );
    assert_eq!(pts32.len(), pts64.len());
    for (p32, p64) in pts32.iter().zip(pts64.iter()) {
        assert_eq!(p32.x.to_bits(), <f32 as Scalar>::from_f64(p64.x).to_bits());
        assert_eq!(p32.y.to_bits(), <f32 as Scalar>::from_f64(p64.y).to_bits());
        assert_eq!(p32.z.to_bits(), <f32 as Scalar>::from_f64(p64.z).to_bits());
    }
}
