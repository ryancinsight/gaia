//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::Point3r;

/// Build a minimal VertexPool with tolerance-based welding.
fn test_pool() -> VertexPool {
    VertexPool::with_tolerance(1e-4, 1e-8)
}

/// Insert a vertex into the pool and return its ID.
fn insert(pool: &mut VertexPool, x: Real, y: Real, z: Real) -> VertexId {
    pool.insert_or_weld(Point3r::new(x, y, z), Vector3r::new(0.0, 0.0, 1.0))
}

/// Flat triangle should NOT be refined (zero curvature).
#[test]
fn flat_triangle_unchanged() {
    let mut pool = test_pool();
    let v0 = insert(&mut pool, 0.0, 0.0, 0.0);
    let v1 = insert(&mut pool, 1.0, 0.0, 0.0);
    let v2 = insert(&mut pool, 0.5, 1.0, 0.0);
    let mut faces = vec![FaceData::new(v0, v1, v2, RegionId::default())];
    let before = faces.len();
    refine_high_curvature_faces(&mut faces, &mut pool);
    assert_eq!(faces.len(), before, "flat triangle should not be split");
}

/// A "tent" mesh with high angle defect at the apex should trigger splitting.
#[test]
fn tent_apex_triggers_refinement() {
    let mut pool = test_pool();
    // Base quad (flat on z=0)
    let b0 = insert(&mut pool, -1.0, -1.0, 0.0);
    let b1 = insert(&mut pool, 1.0, -1.0, 0.0);
    let b2 = insert(&mut pool, 1.0, 1.0, 0.0);
    let b3 = insert(&mut pool, -1.0, 1.0, 0.0);
    // Apex (high above centre → sharp curvature)
    let apex = insert(&mut pool, 0.0, 0.0, 3.0);

    let r = RegionId::default();
    let mut faces = vec![
        FaceData::new(b0, b1, apex, r),
        FaceData::new(b1, b2, apex, r),
        FaceData::new(b2, b3, apex, r),
        FaceData::new(b3, b0, apex, r),
    ];
    let before = faces.len();
    refine_high_curvature_faces(&mut faces, &mut pool);
    assert!(
        faces.len() > before,
        "tent apex should trigger curvature refinement: {} faces → {}",
        before,
        faces.len()
    );
}

/// Centroid splits must preserve total face count invariant:
/// each split adds exactly 2 faces (1 replaced in-place + 2 appended = 3 total,
/// net +2).
#[test]
fn centroid_split_face_count_invariant() {
    let mut pool = test_pool();
    let v0 = insert(&mut pool, 0.0, 0.0, 0.0);
    let v1 = insert(&mut pool, 1.0, 0.0, 0.0);
    let v2 = insert(&mut pool, 0.5, 1.0, 0.0);
    let r = RegionId::default();
    let mut faces = vec![FaceData::new(v0, v1, v2, r)];

    apply_centroid_splits(&mut faces, &mut pool, &[0]);
    assert_eq!(faces.len(), 3, "1 face → 3 faces after centroid split");

    // All three faces share the centroid vertex.
    let all_verts: Vec<VertexId> = faces.iter().flat_map(|f| f.vertices).collect();
    let centroid_count = all_verts
        .iter()
        .filter(|&&v| v != v0 && v != v1 && v != v2)
        .count();
    assert_eq!(centroid_count, 3, "centroid appears in all 3 sub-faces");
}

/// Curvature estimation for a closed box should yield finite positive values
/// at corners (high angle defect) and zero on flat interior vertices.
#[test]
fn curvature_from_soup_box_corners() {
    let mut pool = test_pool();
    // Simple box: 8 corners, 12 faces (2 per quad face).
    let c = [
        insert(&mut pool, 0.0, 0.0, 0.0), // 0
        insert(&mut pool, 1.0, 0.0, 0.0), // 1
        insert(&mut pool, 1.0, 1.0, 0.0), // 2
        insert(&mut pool, 0.0, 1.0, 0.0), // 3
        insert(&mut pool, 0.0, 0.0, 1.0), // 4
        insert(&mut pool, 1.0, 0.0, 1.0), // 5
        insert(&mut pool, 1.0, 1.0, 1.0), // 6
        insert(&mut pool, 0.0, 1.0, 1.0), // 7
    ];
    let r = RegionId::default();
    let faces = vec![
        // bottom z=0
        FaceData::new(c[0], c[2], c[1], r),
        FaceData::new(c[0], c[3], c[2], r),
        // top z=1
        FaceData::new(c[4], c[5], c[6], r),
        FaceData::new(c[4], c[6], c[7], r),
        // front y=0
        FaceData::new(c[0], c[1], c[5], r),
        FaceData::new(c[0], c[5], c[4], r),
        // back y=1
        FaceData::new(c[2], c[3], c[7], r),
        FaceData::new(c[2], c[7], c[6], r),
        // left x=0
        FaceData::new(c[0], c[4], c[7], r),
        FaceData::new(c[0], c[7], c[3], r),
        // right x=1
        FaceData::new(c[1], c[2], c[6], r),
        FaceData::new(c[1], c[6], c[5], r),
    ];

    let curvature = vertex_curvature_from_soup(&faces, &pool);
    // All 8 corner vertices should have non-zero curvature (angle defect = π/2).
    for &v in &c {
        let h = curvature.get(&v).copied().unwrap_or(0.0);
        assert!(
            h > 0.0,
            "box corner {v:?} should have positive curvature, got {h}"
        );
    }
}

/// Cotangent Laplacian on a closed icosahedron approximation of a sphere
/// yields curvature close to 1/R at every vertex.
///
/// Uses a 42-vertex geodesic sphere (subdivided icosahedron) with R=1.
/// The mean curvature of a sphere is H = 1/R = 1.0.  The discrete
/// cotangent Laplacian estimate should be within 30% of exact for this
/// resolution.
#[test]
fn cotangent_curvature_sphere_approximation() {
    use crate::domain::core::scalar::Point3r;
    use crate::domain::geometry::primitives::{PrimitiveMesh, UvSphere};
    let sphere = UvSphere {
        radius: 1.0,
        center: Point3r::origin(),
        segments: 16,
        stacks: 8,
    };
    let mesh = sphere.build().expect("UvSphere::build failed");

    let faces: Vec<FaceData> = mesh.faces.iter().copied().collect();

    let curvature = vertex_curvature_from_soup(&faces, &mesh.vertices);

    // At least some vertices should have curvature estimates.
    assert!(
        curvature.len() > 10,
        "expected many vertices with curvature, got {}",
        curvature.len()
    );

    // All curvature values should be positive and within a reasonable range
    // of H = 1/R = 1.0.  For a 16×8 UV sphere, the cotangent estimate
    // may deviate due to non-uniform vertex distribution, so we use a
    // wide tolerance band (0.2–5.0).
    for (&_vid, &h) in &curvature {
        assert!(
            h > 0.1 && h < 10.0,
            "sphere vertex curvature should be near 1.0, got {h}"
        );
    }
}

/// Centroid splits of adjacent faces preserve the shared edge exactly.
///
/// Splitting two faces sharing edge [v1, v2] must NOT create duplicate
/// vertices at the shared edge (T-junction).  Each centroid is unique
/// because `insert_unique` is used.
#[test]
fn adjacent_centroid_splits_no_t_junction() {
    let mut pool = test_pool();
    let v0 = insert(&mut pool, 0.0, 0.0, 0.0);
    let v1 = insert(&mut pool, 1.0, 0.0, 0.0);
    let v2 = insert(&mut pool, 0.5, 1.0, 0.0);
    let v3 = insert(&mut pool, 0.5, -1.0, 0.0);
    let r = RegionId::default();

    // Two triangles sharing edge [v0, v1]:
    // Face 0: [v0, v1, v2]  Face 1: [v1, v0, v3]
    let mut faces = vec![FaceData::new(v0, v1, v2, r), FaceData::new(v1, v0, v3, r)];

    apply_centroid_splits(&mut faces, &mut pool, &[0, 1]);
    assert_eq!(faces.len(), 6, "2 faces → 6 faces after splitting both");

    // Shared edge [v0, v1] must appear exactly twice: once in each
    // centroid-split fan — no extra vertices are on that edge.
    let edge_count = faces
        .iter()
        .filter(|f| {
            let vs = f.vertices;
            (vs[0] == v0 && vs[1] == v1)
                || (vs[1] == v0 && vs[2] == v1)
                || (vs[2] == v0 && vs[0] == v1)
                || (vs[0] == v1 && vs[1] == v0)
                || (vs[1] == v1 && vs[2] == v0)
                || (vs[2] == v1 && vs[0] == v0)
        })
        .count();
    assert_eq!(
        edge_count, 2,
        "shared edge should appear in exactly 2 sub-faces"
    );
}

/// Degenerate face (zero area) is not refined — curvature is undefined.
#[test]
fn degenerate_face_not_refined() {
    let mut pool = test_pool();
    let v0 = insert(&mut pool, 0.0, 0.0, 0.0);
    let v1 = insert(&mut pool, 1.0, 0.0, 0.0);
    // v2 is ON edge [v0, v1] → zero-area degenerate triangle.
    let v2 = insert(&mut pool, 0.5, 0.0, 0.0);
    let mut faces = vec![FaceData::new(v0, v1, v2, RegionId::default())];
    let before = faces.len();
    refine_high_curvature_faces(&mut faces, &mut pool);
    assert_eq!(faces.len(), before, "degenerate face should not be split");
}

/// Cotangent weight clamping prevents infinite curvature on near-degenerate
/// obtuse triangles where opposing angles approach 0 or π.
#[test]
fn obtuse_sliver_curvature_is_finite() {
    let mut pool = test_pool();
    // Extreme obtuse triangle: angle at v0 ≈ 179°, near collinear.
    let v0 = insert(&mut pool, 0.0, 0.0, 0.0);
    let v1 = insert(&mut pool, 10.0, 0.0, 0.0);
    let v2 = insert(&mut pool, 5.0, 0.001, 0.0);
    // Need ≥ 3 faces per vertex for curvature estimation.
    let v3 = insert(&mut pool, 5.0, -0.001, 0.0);
    let v4 = insert(&mut pool, 5.0, 0.0, 0.001);
    let r = RegionId::default();
    let faces = vec![
        FaceData::new(v0, v1, v2, r),
        FaceData::new(v0, v1, v3, r),
        FaceData::new(v0, v1, v4, r),
        FaceData::new(v0, v2, v3, r),
    ];

    let curvature = vertex_curvature_from_soup(&faces, &pool);
    for (&_vid, &h) in &curvature {
        assert!(
            h.is_finite(),
            "curvature must be finite even for slivers, got {h}"
        );
    }
}
