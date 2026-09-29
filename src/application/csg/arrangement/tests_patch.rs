//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::{Point3r, Vector3r};

/// Helper: build a VertexPool and insert vertices at given positions.
fn pool_with_positions(pts: &[Point3r]) -> (VertexPool, Vec<VertexId>) {
    let mut pool = VertexPool::new(1e-6_f64);
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let ids: Vec<VertexId> = pts.iter().map(|&p| pool.insert_or_weld(p, n)).collect();
    (pool, ids)
}

fn p(x: Real, y: Real, z: Real) -> Point3r {
    Point3r::new(x, y, z)
}

// ── Empty / no-op cases ───────────────────────────────────────────────

/// Patching an empty face list must not panic and must leave faces empty.
#[test]
fn patch_empty_faces() {
    let pool = VertexPool::new(1e-6_f64);
    let mut faces: Vec<FaceData> = Vec::new();
    patch_small_boundary_holes(&mut faces, &pool);
    assert!(faces.is_empty());
}

/// A single isolated triangle has three boundary edges. The patcher should
/// not panic but also cannot close such a loop (it needs matching reverse
/// edges from adjacent faces). The face list must remain non-empty.
#[test]
fn patch_single_triangle_survives() {
    let (pool, v) = pool_with_positions(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0)]);
    let mut faces = vec![FaceData::untagged(v[0], v[1], v[2])];
    patch_small_boundary_holes(&mut faces, &pool);
    assert!(!faces.is_empty(), "single triangle must not be deleted");
}

// ── Already-closed mesh ──────────────────────────────────────────────

/// A watertight tetrahedron has no boundary edges. Patching should be a
/// no-op and preserve all 4 faces.
#[test]
fn patch_closed_tetrahedron_is_noop() {
    let (pool, v) = pool_with_positions(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.5, 1.0, 0.0),
        p(0.5, 0.5, 1.0),
    ]);
    // 4-face closed tetrahedron (consistent CCW winding from outside)
    let mut faces = vec![
        FaceData::untagged(v[0], v[2], v[1]), // bottom (viewed from -Z)
        FaceData::untagged(v[0], v[1], v[3]), // front
        FaceData::untagged(v[1], v[2], v[3]), // right
        FaceData::untagged(v[2], v[0], v[3]), // left
    ];
    let before = faces.len();
    patch_small_boundary_holes(&mut faces, &pool);
    assert_eq!(
        faces.len(),
        before,
        "closed tetrahedron must not gain or lose faces"
    );
    // Verify still closed
    let boundary = boundary_half_edges(&faces);
    assert!(boundary.is_empty(), "tetrahedron must remain watertight");
}

// ── Degenerate face removal ──────────────────────────────────────────

/// Step 1 of patch removes degenerate (zero-area) faces. A face with two
/// coincident vertices should be cleaned out.
#[test]
fn patch_removes_degenerate_faces() {
    let (pool, v) = pool_with_positions(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0)]);
    let mut faces = vec![
        FaceData::untagged(v[0], v[1], v[2]), // good triangle
        FaceData::untagged(v[0], v[0], v[1]), // degenerate: v0 == v0
    ];
    patch_small_boundary_holes(&mut faces, &pool);
    // Degenerate face should have been removed in Step 1
    for f in &faces {
        assert!(
            f.vertices[0] != f.vertices[1]
                && f.vertices[1] != f.vertices[2]
                && f.vertices[0] != f.vertices[2],
            "no degenerate faces should remain"
        );
    }
}

// ── Duplicate face removal ───────────────────────────────────────────

/// Step 2 of patch deduplicates faces with the same vertex set.
#[test]
fn patch_removes_duplicate_faces() {
    let (pool, v) = pool_with_positions(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(0.5, 0.5, 1.0),
    ]);
    // Closed tetrahedron with one face duplicated
    let mut faces = vec![
        FaceData::untagged(v[0], v[2], v[1]),
        FaceData::untagged(v[0], v[1], v[3]),
        FaceData::untagged(v[1], v[2], v[3]),
        FaceData::untagged(v[2], v[0], v[3]),
        FaceData::untagged(v[0], v[1], v[3]), // duplicate of face 1
    ];
    patch_small_boundary_holes(&mut faces, &pool);
    // Should have exactly 4 unique faces
    assert_eq!(faces.len(), 4, "duplicate face must be removed");
}

// ── Non-manifold edge repair ─────────────────────────────────────────

/// Step 3 of patch resolves non-manifold edges by keeping the larger-area
/// face.
#[test]
fn patch_resolves_non_manifold_by_keeping_larger() {
    let (pool, v) = pool_with_positions(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.5, 1.0, 0.0),  // large triangle apex
        p(0.5, 0.01, 0.0), // sliver triangle apex (near the base)
    ]);
    // Two faces share directed edge v0->v1: big triangle and tiny sliver
    let mut faces = vec![
        FaceData::untagged(v[0], v[1], v[2]), // big area
        FaceData::untagged(v[0], v[1], v[3]), // tiny area
    ];
    patch_small_boundary_holes(&mut faces, &pool);
    // The sliver should be removed (smaller area on same directed edge)
    assert_eq!(
        faces.len(),
        1,
        "non-manifold repair must keep only one face per half-edge"
    );
    // The surviving face should be the big one
    let surviving = &faces[0];
    assert!(
        surviving.vertices.contains(&v[2]),
        "bigger-area face must survive non-manifold resolution"
    );
}

// ── Boundary hole patching (triangle hole) ───────────────────────────

/// A box missing one quad face (= 2 triangles) creates a rectangular
/// boundary loop. The patcher should fill it.
#[test]
fn patch_fills_quad_hole_in_box() {
    // Build an axis-aligned unit cube with one face (2 triangles) missing.
    //
    //   v4──v5
    //   │    │   top (z=1)
    //   v7──v6
    //
    //   v0──v1
    //   │    │   bottom (z=0)
    //   v3──v2
    let pts = [
        p(0.0, 0.0, 0.0), // 0
        p(1.0, 0.0, 0.0), // 1
        p(1.0, 1.0, 0.0), // 2
        p(0.0, 1.0, 0.0), // 3
        p(0.0, 0.0, 1.0), // 4
        p(1.0, 0.0, 1.0), // 5
        p(1.0, 1.0, 1.0), // 6
        p(0.0, 1.0, 1.0), // 7
    ];
    let (pool, v) = pool_with_positions(&pts);

    // 5 faces × 2 triangles = 10 triangles; omit the +Y face (v2,v3,v7,v6)
    let mut faces = vec![
        // bottom (z=0, normal -Z)
        FaceData::untagged(v[0], v[2], v[1]),
        FaceData::untagged(v[0], v[3], v[2]),
        // top (z=1, normal +Z)
        FaceData::untagged(v[4], v[5], v[6]),
        FaceData::untagged(v[4], v[6], v[7]),
        // front (y=0, normal -Y)
        FaceData::untagged(v[0], v[1], v[5]),
        FaceData::untagged(v[0], v[5], v[4]),
        // back: OMITTED — this is the hole
        // left (x=0, normal -X)
        FaceData::untagged(v[0], v[4], v[7]),
        FaceData::untagged(v[0], v[7], v[3]),
        // right (x=1, normal +X)
        FaceData::untagged(v[1], v[2], v[6]),
        FaceData::untagged(v[1], v[6], v[5]),
    ];

    let boundary_before = boundary_half_edges(&faces);
    assert!(
        !boundary_before.is_empty(),
        "box with missing face must have boundary edges"
    );

    patch_small_boundary_holes(&mut faces, &pool);

    let boundary_after = boundary_half_edges(&faces);
    assert!(
        boundary_after.is_empty(),
        "patcher must close the quad hole — {} boundary edges remain",
        boundary_after.len()
    );
    // Should have original 10 + 2 patch triangles = 12
    assert!(
        faces.len() >= 12,
        "patched box must have at least 12 faces, got {}",
        faces.len()
    );
}

// ── Sliver face tolerance ────────────────────────────────────────────

/// Face with area ratio < 1e-8 * max_edge should be removed by Step 1.
#[test]
fn patch_removes_extreme_sliver() {
    let (pool, v) = pool_with_positions(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.5, 1e-10, 0.0), // extreme sliver
        p(0.5, 0.5, 0.0),   // normal vertex for a good triangle
    ]);
    let mut faces = vec![
        FaceData::untagged(v[0], v[1], v[2]), // sliver
        FaceData::untagged(v[0], v[1], v[3]), // good
    ];
    patch_small_boundary_holes(&mut faces, &pool);
    // Sliver should be gone
    assert_eq!(faces.len(), 1, "extreme sliver must be removed");
    assert!(
        faces[0].vertices.contains(&v[3]),
        "good triangle must survive"
    );
}

// ── Determinism ──────────────────────────────────────────────────────

/// Running patch twice on the same input must produce the same output.
#[test]
fn patch_is_idempotent() {
    let (pool, v) = pool_with_positions(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(1.0, 1.0, 0.0),
    ]);
    let make_faces = || {
        vec![
            FaceData::untagged(v[0], v[1], v[2]),
            FaceData::untagged(v[1], v[3], v[2]),
        ]
    };

    let mut faces1 = make_faces();
    patch_small_boundary_holes(&mut faces1, &pool);
    let snapshot1: Vec<[VertexId; 3]> = faces1.iter().map(|f| f.vertices).collect();

    let mut faces2 = make_faces();
    patch_small_boundary_holes(&mut faces2, &pool);
    let snapshot2: Vec<[VertexId; 3]> = faces2.iter().map(|f| f.vertices).collect();

    assert_eq!(snapshot1, snapshot2, "patch must be deterministic");
}
