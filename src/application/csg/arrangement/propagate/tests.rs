//! Contract tests for both propagation passes and the adjacency map.

use super::adjacency::AdjacentFaces;
use super::{inject_cap_seam_into_barrels, propagate_seam_vertices};
use crate::application::csg::intersect::SnapSegment;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use hashbrown::HashSet;

fn contains_param_split(segs: &[SnapSegment], a: Point3r, b: Point3r, t: Real, tol: Real) -> bool {
    let x = a + (b - a) * t;
    segs.iter().any(|s| {
        ((s.start).distance_squared(a) < tol && (s.end).distance_squared(x) < tol)
            || ((s.start).distance_squared(x) < tol && (s.end).distance_squared(b) < tol)
    })
}

#[test]
fn propagate_seam_vertices_injects_crossing_split_into_adjacent_face() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(0.5, 1.0, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(0.5, -1.0, 0.0), n);
    let faces = vec![FaceData::untagged(a, b, c), FaceData::untagged(b, a, d)];

    let mut segs = vec![Vec::new(); 2];
    segs[0].push(SnapSegment {
        start: Point3r::new(0.25, 0.5, 0.0),
        end: Point3r::new(0.25, -0.5, 0.0),
    });

    propagate_seam_vertices(&faces, &mut segs, &pool);

    let injected = &segs[1];
    let pa = *pool.position(a);
    let pb = *pool.position(b);
    assert!(
        injected.len() >= 2,
        "crossing split should create at least two sub-segments on shared edge"
    );
    assert!(
        contains_param_split(injected, pa, pb, 0.25, 1e-10),
        "adjacent face should receive split at crossing parameter t=0.25"
    );
}

#[test]
fn adversarial_near_parallel_crossing_is_propagated() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(1.0, 1.0e-10, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(0.2, 1.0, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(0.2, -1.0, 0.0), n);
    let faces = vec![FaceData::untagged(a, b, c), FaceData::untagged(b, a, d)];

    let mut segs = vec![Vec::new(); 2];
    segs[0].push(SnapSegment {
        start: Point3r::new(0.5, -1.0e-10, 0.0),
        end: Point3r::new(0.5000000001, 1.0e-10, 0.0),
    });

    propagate_seam_vertices(&faces, &mut segs, &pool);
    let injected = &segs[1];
    assert!(
        !injected.is_empty(),
        "adjacent face should receive injected segments for near-parallel crossing"
    );
}

// ── inject_cap_seam_into_barrels tests ────────────────────────────────────

/// Build a single barrel rim face: v0=(0,0,0), v1=(1,0,0) on plane z=0,
/// v2=(0.5,0,−1) off-plane.  Rim edge is `[v0,v1]` along X.
fn single_rim_face(pool: &mut VertexPool) -> (Vec<FaceData>, Point3r, Vector3r) {
    let nz = Vector3r::new(0.0, 0.0, 1.0);
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), nz);
    let v1 = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), nz);
    let v2 = pool.insert_or_weld(Point3r::new(0.5, 0.0, -1.0), nz);
    let faces = vec![FaceData::untagged(v0, v1, v2)];
    let plane_pt = Point3r::new(0.0, 0.0, 0.0);
    let plane_n = Vector3r::new(0.0, 0.0, 1.0);
    (faces, plane_pt, plane_n)
}

/// Seam position at t=0.25 on a 1m rim edge is found by the spatial hash and
/// produces 2 sub-segment injections.
#[test]
fn inject_cap_seam_finds_seam_at_quarter_param() {
    let mut pool = VertexPool::default_millifluidic();
    let (faces, plane_pt, plane_n) = single_rim_face(&mut pool);
    let coplanar_used = HashSet::new();
    let seam_positions = vec![Point3r::new(0.25, 0.0, 0.0)];
    let mut segs_out = vec![Vec::new(); faces.len()];

    inject_cap_seam_into_barrels(
        &faces,
        &coplanar_used,
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    let injected = &segs_out[0];
    assert!(
        !injected.is_empty(),
        "seam at t=0.25 should generate sub-segments"
    );
    assert_eq!(injected.len(), 2, "one seam point creates 2 sub-segments");
}

/// A seam point that does not coincide with any of the five sample points
/// must still be discovered on a long rim edge by the adaptive hash cell.
#[test]
fn inject_cap_seam_off_sample_position_on_long_rim_edge_is_detected() {
    let mut pool = VertexPool::default_millifluidic();
    let nz = Vector3r::new(0.0, 0.0, 1.0);
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), nz);
    let v1 = pool.insert_or_weld(Point3r::new(8.0, 0.0, 0.0), nz);
    let v2 = pool.insert_or_weld(Point3r::new(4.0, 0.0, -1.0), nz);
    let faces = vec![FaceData::untagged(v0, v1, v2)];
    let plane_pt = Point3r::new(0.0, 0.0, 0.0);
    let plane_n = Vector3r::new(0.0, 0.0, 1.0);
    let seam_positions = vec![Point3r::new(1.0, 0.0, 0.0)];
    let mut segs_out = vec![Vec::new(); 1];

    inject_cap_seam_into_barrels(
        &faces,
        &HashSet::new(),
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    let injected = &segs_out[0];
    assert!(
        contains_param_split(
            injected,
            Point3r::new(0.0, 0.0, 0.0),
            Point3r::new(8.0, 0.0, 0.0),
            0.125,
            1e-10
        ),
        "adaptive cap-seam injection should detect off-sample seam position on long rim edge"
    );
}

/// Seam position not on the cap plane (z=0.5) must not inject into any face.
#[test]
fn inject_cap_seam_ignores_off_plane_position() {
    let mut pool = VertexPool::default_millifluidic();
    let (faces, plane_pt, plane_n) = single_rim_face(&mut pool);
    let coplanar_used = HashSet::new();
    // z=0.5 — off-plane, should be rejected by the ds.abs() guard.
    let seam_positions = vec![Point3r::new(0.5, 0.0, 0.5)];
    let mut segs_out = vec![Vec::new(); faces.len()];

    inject_cap_seam_into_barrels(
        &faces,
        &coplanar_used,
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    assert!(
        segs_out.iter().all(|s| s.is_empty()),
        "off-plane seam position must not inject into any face"
    );
}

/// Two seam positions (given out of order) produce 3 sorted sub-segments.
#[test]
fn inject_cap_seam_multiple_positions_generate_sorted_sub_intervals() {
    let mut pool = VertexPool::default_millifluidic();
    let (faces, plane_pt, plane_n) = single_rim_face(&mut pool);
    let coplanar_used = HashSet::new();
    // Intentionally insert t=0.75 before t=0.25 to verify sorting.
    let seam_positions = vec![Point3r::new(0.75, 0.0, 0.0), Point3r::new(0.25, 0.0, 0.0)];
    let mut segs_out = vec![Vec::new(); faces.len()];

    inject_cap_seam_into_barrels(
        &faces,
        &coplanar_used,
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    let injected = &segs_out[0];
    assert_eq!(injected.len(), 3, "two seam points create 3 sub-segments");
}

/// A face with only 1 on-plane vertex is not a rim face and must not receive
/// any injected segments.
#[test]
fn inject_cap_seam_non_rim_face_is_skipped() {
    let mut pool = VertexPool::default_millifluidic();
    let nz = Vector3r::new(0.0, 0.0, 1.0);
    // Only v0 is on the cap plane z=0 → on_count=1 → not a rim face.
    let v0 = pool.insert_or_weld(Point3r::new(0.5, 0.0, 0.0), nz);
    let v1 = pool.insert_or_weld(Point3r::new(0.0, 0.5, -1.0), nz);
    let v2 = pool.insert_or_weld(Point3r::new(1.0, 0.5, -1.0), nz);
    let faces = vec![FaceData::untagged(v0, v1, v2)];
    let coplanar_used = HashSet::new();
    let plane_pt = Point3r::new(0.0, 0.0, 0.0);
    let plane_n = Vector3r::new(0.0, 0.0, 1.0);
    let seam_positions = vec![Point3r::new(0.5, 0.0, 0.0)];
    let mut segs_out = vec![Vec::new(); faces.len()];

    inject_cap_seam_into_barrels(
        &faces,
        &coplanar_used,
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    assert!(
        segs_out[0].is_empty(),
        "non-rim face (1 on-plane vertex) must not receive injected segments"
    );
}

/// Adversarial: seam position not on the rim edge (off to the side) is
/// rejected even though it passes the plane check.
#[test]
fn inject_cap_seam_position_beside_rim_edge_is_rejected() {
    let mut pool = VertexPool::default_millifluidic();
    let (faces, plane_pt, plane_n) = single_rim_face(&mut pool);
    let coplanar_used = HashSet::new();
    // y=0.5 puts the point on the cap plane but off the rim edge [0,0,0]→[1,0,0].
    let seam_positions = vec![Point3r::new(0.5, 0.5, 0.0)];
    let mut segs_out = vec![Vec::new(); faces.len()];

    inject_cap_seam_into_barrels(
        &faces,
        &coplanar_used,
        &plane_pt,
        &plane_n,
        &seam_positions,
        &mut segs_out,
        &pool,
    );

    assert!(
        segs_out[0].is_empty(),
        "seam position beside rim edge must not inject (off-edge but on-plane)"
    );
}

// ── Axis-pair threshold ───────────────────────────────────────────────────

/// The near-parallel rejection threshold must be a length²: it is compared
/// against a 2-D determinant of two in-plane lengths.
///
/// A threshold of degree 1 would make the accepted angle grow with scale, so
/// the same geometry at 10 µm and at 1 m would be classified differently — the
/// failure class `scale_robustness_tests` exists to catch.
#[test]
fn min_axis_determinant_is_homogeneous_of_degree_two() {
    use super::seam::min_axis_determinant;

    let (edge_len_sq, seg_len_sq) = (1.0_f64, 0.25_f64);
    let base = min_axis_determinant(edge_len_sq, seg_len_sq);
    assert!(base > 0.0);
    for scale in [1e-5_f64, 1e-3, 1e3, 1e5] {
        let scaled = min_axis_determinant(edge_len_sq * scale * scale, seg_len_sq * scale * scale);
        let expected = base * scale * scale;
        assert!(
            (scaled - expected).abs() <= expected * 1e-12,
            "threshold must scale as length²: at scale {scale:e} got {scaled:e}, want {expected:e}"
        );
    }
}

/// The same near-parallel configuration must be accepted or rejected the same
/// way at every scale: scaling the mesh must not change the decision.
///
/// Under the previous length-form threshold this failed — the ratio of
/// determinant to threshold grew linearly with scale, so a genuinely
/// near-parallel pair was rejected on a micro-scale mesh and accepted on the
/// same mesh scaled up.
#[test]
fn near_parallel_rejection_is_scale_equivariant() {
    use super::seam::min_axis_determinant;

    // 2-D determinant of edge × segment for a pair `tilt` radians apart, at unit
    // scale: ≈ |e||s|·sin(tilt).
    let tilt = 1.0e-14_f64;
    let unit_decision = tilt < min_axis_determinant(1.0, 1.0);
    for scale in [1e-5_f64, 1e-3, 1.0, 1e3, 1e5] {
        let determinant = tilt * scale * scale;
        let threshold = min_axis_determinant(scale * scale, scale * scale);
        assert_eq!(
            determinant < threshold,
            unit_decision,
            "decision must not depend on scale (scale {scale:e})"
        );
    }
}

/// Two triangles sharing an edge along +X, plus a snap segment `tilt` radians
/// off that edge and offset off the edge line so it is not read as collinear.
fn near_parallel_scenario(
    scale: Real,
    tilt: Real,
) -> (VertexPool, Vec<FaceData>, Vec<Vec<SnapSegment>>) {
    let mut pool = VertexPool::new(scale * 1e-3);
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(scale, 0.0, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(0.5 * scale, scale, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(0.5 * scale, -scale, 0.0), n);
    let faces = vec![FaceData::untagged(a, b, c), FaceData::untagged(b, a, d)];
    let mut segs = vec![Vec::new(); 2];
    segs[0].push(SnapSegment {
        start: Point3r::new(0.25 * scale, 0.02 * scale, 0.0),
        end: Point3r::new(0.75 * scale, 0.02 * scale + tilt * 0.5 * scale, 0.0),
    });
    (pool, faces, segs)
}

/// The widened threshold must not change the decisions taken on unit-scale or
/// macro-scale geometry — it only stops over-rejecting below micro scale.
///
/// This runs the pass on a near-parallel pair at five scales: all five must
/// agree on the number of injected sub-segments, so the scale-equivariance the
/// threshold now has is also what the pass exhibits.
#[test]
fn near_parallel_propagation_agrees_across_scales() {
    let mut injected = Vec::new();
    for scale in [1e-5_f64, 1e-3, 1.0, 1e3, 1e5] {
        let (pool, faces, mut segs) = near_parallel_scenario(scale, 1e-12);
        propagate_seam_vertices(&faces, &mut segs, &pool);
        injected.push(segs[1].len());
    }
    assert!(
        injected.windows(2).all(|w| w[0] == w[1]),
        "the same near-parallel configuration must propagate identically at every scale: {injected:?}"
    );
}

#[test]
fn adjacent_faces_preserves_more_than_u8_incident_faces() {
    let mut adjacent = AdjacentFaces::default();
    for face_index in 0..=u8::MAX as usize {
        adjacent.push(face_index);
    }

    let faces: Vec<_> = adjacent.into_iter().copied().collect();
    assert_eq!(faces.len(), 256);
    assert_eq!(faces.first().copied(), Some(0));
    assert_eq!(faces.last().copied(), Some(255));
}
