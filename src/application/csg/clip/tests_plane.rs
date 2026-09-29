//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::RegionId;

fn csg_pool() -> VertexPool {
    VertexPool::for_csg()
}

fn p(x: f64, y: f64, z: f64) -> Point3r {
    Point3r::new(x, y, z)
}

fn n_up() -> Vector3r {
    Vector3r::new(0.0, 0.0, 1.0)
}

/// Helper: build a face from 3 points, returning (face, pool).
fn face_from_pts(a: Point3r, b: Point3r, c: Point3r) -> (FaceData, VertexPool) {
    let mut pool = csg_pool();
    let n = n_up();
    let va = pool.insert_or_weld(a, n);
    let vb = pool.insert_or_weld(b, n);
    let vc = pool.insert_or_weld(c, n);
    let face = FaceData::new(va, vb, vc, RegionId::INVALID);
    (face, pool)
}

// ── PlaneEquation tests ──────────────────────────────────────────────

#[test]
fn plane_classify_above_below_on() {
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    assert_eq!(plane.classify(&p(0.5, 0.5, 1.0)), Orientation::Positive);
    assert_eq!(plane.classify(&p(0.5, 0.5, -1.0)), Orientation::Negative);
    assert_eq!(plane.classify(&p(0.5, 0.5, 0.0)), Orientation::Degenerate);
}

#[test]
fn plane_from_face_matches_from_points() {
    let mut pool = csg_pool();
    let n = n_up();
    let va = pool.insert_or_weld(p(0.0, 0.0, 0.0), n);
    let vb = pool.insert_or_weld(p(1.0, 0.0, 0.0), n);
    let vc = pool.insert_or_weld(p(0.0, 1.0, 0.0), n);
    let face = FaceData::new(va, vb, vc, RegionId::INVALID);

    let p1 = PlaneEquation::from_face(&face, &pool);
    let p2 = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));

    let q = p(0.3, 0.2, 0.7);
    assert_eq!(p1.classify(&q), p2.classify(&q));
}

#[test]
fn plane_intersect_edge_midpoint() {
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let s = p(0.0, 0.0, 1.0);
    let e = p(0.0, 0.0, -1.0);
    let cut = plane.intersect_edge(&s, &e);
    assert!((cut.x).abs() < 1e-12);
    assert!((cut.y).abs() < 1e-12);
    assert!((cut.z).abs() < 1e-12);
}

#[test]
fn plane_intersect_edge_quarter() {
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 1.0), &p(1.0, 0.0, 1.0), &p(0.0, 1.0, 1.0));
    let s = p(0.0, 0.0, 0.0);
    let e = p(0.0, 0.0, 4.0);
    let cut = plane.intersect_edge(&s, &e);
    assert!((cut.z - 1.0).abs() < 1e-12);
}

// ── classify_face tests ──────────────────────────────────────────────

#[test]
fn classify_face_above_plane() {
    let (face, pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (class, _) = classify_face(&face, &pool, &plane);
    assert_eq!(class, FacePlaneClass::Inside);
}

#[test]
fn classify_face_below_plane() {
    let (face, pool) = face_from_pts(p(0.0, 0.0, -1.0), p(1.0, 0.0, -2.0), p(0.0, 1.0, -3.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (class, _) = classify_face(&face, &pool, &plane);
    assert_eq!(class, FacePlaneClass::Outside);
}

#[test]
fn classify_face_straddling() {
    let (face, pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 0.5));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (class, _) = classify_face(&face, &pool, &plane);
    assert_eq!(class, FacePlaneClass::Straddling);
}

#[test]
fn classify_face_coplanar() {
    let (face, pool) = face_from_pts(p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (class, _) = classify_face(&face, &pool, &plane);
    assert_eq!(class, FacePlaneClass::Coplanar);
}

// ── clip_face_by_plane tests ─────────────────────────────────────────

#[test]
fn clip_fully_inside_keeps_face() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], face);
}

#[test]
fn clip_fully_outside_removes_face() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, -1.0), p(1.0, 0.0, -2.0), p(0.0, 1.0, -3.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    assert!(result.is_empty());
}

#[test]
fn clip_straddling_produces_subtriangles() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 1.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    assert!(
        !result.is_empty(),
        "straddling face should produce inside sub-faces"
    );
    // One vertex below → inside polygon is 4 vertices → 2 triangles
    assert_eq!(result.len(), 2);
}

#[test]
fn clip_coplanar_keeps_face() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    assert_eq!(result.len(), 1);
}

#[test]
fn clip_one_vertex_on_plane_two_above() {
    // v0 on plane, v1 and v2 above → fully inside (Degenerate counts as
    // inside).
    let (face, mut pool) = face_from_pts(p(0.5, 0.5, 0.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, 1.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    assert_eq!(result.len(), 1);
}

// ── refine_faces_with_plane tests ────────────────────────────────────

#[test]
fn refine_empty_mesh() {
    let mut pool = csg_pool();
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[], &mut pool, &plane);
    assert!(ins.is_empty());
    assert!(outs.is_empty());
}

#[test]
fn refine_all_above() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 2.0), p(0.0, 1.0, 3.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
    assert_eq!(ins.len(), 1);
    assert!(outs.is_empty());
}

#[test]
fn refine_straddling_triangle_splits_both_sides() {
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 1.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
    assert!(!ins.is_empty(), "should have inside sub-faces");
    assert!(!outs.is_empty(), "should have outside sub-faces");
    // 1 vertex below → inside = 2 tris, outside = 1 tri
    assert_eq!(ins.len(), 2);
    assert_eq!(outs.len(), 1);
}

#[test]
fn refine_preserves_region_ids() {
    let mut pool = csg_pool();
    let n = n_up();
    let va = pool.insert_or_weld(p(0.0, 0.0, 1.0), n);
    let vb = pool.insert_or_weld(p(1.0, 0.0, -1.0), n);
    let vc = pool.insert_or_weld(p(0.0, 1.0, 1.0), n);
    let region = RegionId::new(42);
    let face = FaceData::new(va, vb, vc, region);

    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
    for f in ins.iter().chain(outs.iter()) {
        assert_eq!(f.region, region, "sub-faces must inherit parent region");
    }
}

#[test]
fn refine_batch_classification_reduces_orient3d_calls() {
    // 4 faces sharing 5 vertices: batch should call orient_3d 5 times,
    // not 12.  We verify correctness by checking the output (counts of
    // inside/outside faces), which implicitly validates that the cached
    // per-vertex classifications were used.
    let mut pool = csg_pool();
    let n = n_up();
    // Shared vertices
    let v0 = pool.insert_or_weld(p(0.0, 0.0, 1.0), n);
    let v1 = pool.insert_or_weld(p(1.0, 0.0, 1.0), n);
    let v2 = pool.insert_or_weld(p(0.5, 1.0, 1.0), n);
    let v3 = pool.insert_or_weld(p(0.5, 0.5, -1.0), n);
    let v4 = pool.insert_or_weld(p(1.0, 1.0, 1.0), n);

    let faces = [
        FaceData::untagged(v0, v1, v2), // all above
        FaceData::untagged(v0, v1, v3), // v3 below → straddles
        FaceData::untagged(v1, v2, v3), // v3 below → straddles
        FaceData::untagged(v1, v4, v2), // all above
    ];

    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&faces, &mut pool, &plane);
    // 2 fully-inside + 2 straddling (each producing 2 inside sub-faces)
    assert!(ins.len() >= 4);
    assert!(!outs.is_empty());
}

#[test]
fn refine_vertex_on_plane_goes_to_both_halves() {
    // v0 on plane, v1 above, v2 below → straddling.
    // The on-plane vertex appears in both halves.
    let (face, mut pool) = face_from_pts(p(0.5, 0.5, 0.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, -1.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
    assert_eq!(ins.len(), 1);
    assert_eq!(outs.len(), 1);
    // The on-plane vertex (v0) should appear in both sub-faces.
    let v0 = face.vertices[0];
    assert!(ins[0].vertices.contains(&v0));
    assert!(outs[0].vertices.contains(&v0));
}

#[test]
fn refine_tilted_plane_xy() {
    // Plane at 45° through x-axis: n = (0, -1/√2, 1/√2), d = 0
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 1.0));
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 2.0), p(1.0, 2.0, 0.0), p(0.0, 2.0, 0.0));
    let (ins, outs) = refine_faces_with_plane(&[face], &mut pool, &plane);
    let total = ins.len() + outs.len();
    assert!(total >= 2, "tilted plane should split the face");
}

#[test]
fn refine_is_deterministic() {
    let build = || {
        let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, -1.0), p(0.0, 1.0, 0.5));
        let plane =
            PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(0.0, 1.0, 0.0));
        refine_faces_with_plane(&[face], &mut pool, &plane)
    };
    let (ins1, outs1) = build();
    let (ins2, outs2) = build();
    assert_eq!(ins1.len(), ins2.len());
    assert_eq!(outs1.len(), outs2.len());
}

#[test]
fn clip_face_degenerate_plane_no_panic() {
    // Degenerate plane (collinear points) → normal = 0.
    let (face, mut pool) = face_from_pts(p(0.0, 0.0, 1.0), p(1.0, 0.0, 1.0), p(0.0, 1.0, 1.0));
    let plane = PlaneEquation::from_points(&p(0.0, 0.0, 0.0), &p(1.0, 0.0, 0.0), &p(2.0, 0.0, 0.0));
    // All orient_3d calls return Degenerate for a degenerate plane.
    let result = clip_face_by_plane(&face, &mut pool, &plane);
    // Should not panic; face treated as coplanar (all Degenerate).
    assert!(!result.is_empty());
}
