//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::{Point3r, Vector3r};
use proptest::prelude::*;

fn build_greedy_nearest_merge_map_bruteforce_reference(
    bnd_verts: &[VertexId],
    max_dist_sq: Real,
    pool: &VertexPool,
) -> HashMap<VertexId, VertexId> {
    let mut merge_map: HashMap<VertexId, VertexId> = HashMap::new();
    for (i, &vi) in bnd_verts.iter().enumerate() {
        if merge_map.contains_key(&vi) {
            continue;
        }
        let pi = pool.position(vi);
        let mut best_d = max_dist_sq;
        let mut best_j: Option<VertexId> = None;
        for &vj in bnd_verts.iter().skip(i + 1) {
            if merge_map.contains_key(&vj) {
                continue;
            }
            let d = (pool.position(vj) - pi).norm_squared();
            if d < best_d {
                best_d = d;
                best_j = Some(vj);
            }
        }
        if let Some(vj) = best_j {
            merge_map.insert(vj, vi);
        }
    }
    merge_map
}

#[test]
fn mnn_pairs_two_disjoint_close_pairs() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.000, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(0.001, 0.0, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(1.000, 0.0, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(1.001, 0.0, 0.0), n);
    let mut verts = vec![a, b, c, d];
    verts.sort();
    let map = build_mutual_nearest_merge_map(&verts, 0.01, &pool);
    assert_eq!(map.len(), 2, "two disjoint pairs should be matched");
    assert!(map.get(&b).is_some_and(|&k| k == a) || map.get(&a).is_some_and(|&k| k == b));
    assert!(map.get(&d).is_some_and(|&k| k == c) || map.get(&c).is_some_and(|&k| k == d));
}

#[test]
fn adversarial_star_keeps_single_mutual_pair() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let center = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let l1 = pool.insert_or_weld(Point3r::new(0.001, 0.0, 0.0), n);
    let l2 = pool.insert_or_weld(Point3r::new(-0.001, 0.0, 0.0), n);
    let l3 = pool.insert_or_weld(Point3r::new(0.0, 0.001, 0.0), n);
    let mut verts = vec![center, l1, l2, l3];
    verts.sort();
    let map = build_mutual_nearest_merge_map(&verts, 0.01, &pool);
    assert_eq!(
        map.len(),
        1,
        "MNN should avoid collapsing a fan into the center in one pass"
    );
}

#[test]
fn greedy_grid_matches_bruteforce_reference_small_case() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let mut verts = vec![
        pool.insert_or_weld(Point3r::new(0.000, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(0.003, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(0.007, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(0.100, 0.0, 0.0), n),
        pool.insert_or_weld(Point3r::new(0.102, 0.0, 0.0), n),
    ];
    verts.sort();
    verts.dedup();

    let max_dist_sq = 0.01 * 0.01;
    let fast = build_greedy_nearest_merge_map(&verts, max_dist_sq, &pool);
    let brute = build_greedy_nearest_merge_map_bruteforce_reference(&verts, max_dist_sq, &pool);
    assert_eq!(fast, brute);
}

proptest! {
    #[test]
    fn greedy_grid_matches_bruteforce_reference_property(
        coords in prop::collection::vec((-50_i16..50_i16, -50_i16..50_i16, -10_i16..10_i16), 4..28),
        max_step in 1_i16..20_i16
    ) {
        let mut pool = VertexPool::default_millifluidic();
        let n = Vector3r::new(0.0, 0.0, 1.0);
        let mut verts = Vec::new();

        for (x, y, z) in coords {
            let p = Point3r::new(f64::from(x) * 0.01, f64::from(y) * 0.01, f64::from(z) * 0.01);
            verts.push(pool.insert_or_weld(p, n));
        }

        verts.sort();
        verts.dedup();

        let max_d = f64::from(max_step) * 0.01;
        let max_dist_sq = max_d * max_d;

        let fast = build_greedy_nearest_merge_map(&verts, max_dist_sq, &pool);
        let brute = build_greedy_nearest_merge_map_bruteforce_reference(&verts, max_dist_sq, &pool);
        prop_assert_eq!(fast, brute);
    }
}

// ── Seam stitch tests ────────────────────────────────────────────────

/// Stitching an empty face set must not panic.
#[test]
fn stitch_boundary_seams_empty_faces() {
    let pool = VertexPool::default_millifluidic();
    let mut faces: Vec<FaceData> = Vec::new();
    stitch_boundary_seams(&mut faces, &pool);
    assert!(faces.is_empty());
}

/// A closed tetrahedron should have no boundary seams; stitching is a no-op.
#[test]
fn stitch_boundary_seams_closed_mesh_noop() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let v1 = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let v2 = pool.insert_or_weld(Point3r::new(0.5, 1.0, 0.0), n);
    let v3 = pool.insert_or_weld(Point3r::new(0.5, 0.5, 1.0), n);
    let mut faces = vec![
        FaceData::untagged(v0, v2, v1),
        FaceData::untagged(v0, v1, v3),
        FaceData::untagged(v1, v2, v3),
        FaceData::untagged(v2, v0, v3),
    ];
    let before = faces.len();
    stitch_boundary_seams(&mut faces, &pool);
    assert_eq!(
        faces.len(),
        before,
        "closed mesh must not change under stitch"
    );
}

/// Conservative stitch on empty faces must not panic.
#[test]
fn stitch_boundary_seams_conservative_empty() {
    let pool = VertexPool::default_millifluidic();
    let mut faces: Vec<FaceData> = Vec::new();
    stitch_boundary_seams_conservative(&mut faces, &pool);
    assert!(faces.is_empty());
}

/// Determinism: stitching the same input twice must produce the same output.
#[test]
fn stitch_boundary_seams_deterministic() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let v0 = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let v1 = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let v2 = pool.insert_or_weld(Point3r::new(0.5, 1.0, 0.0), n);

    let make_faces = || vec![FaceData::untagged(v0, v1, v2)];

    let mut faces1 = make_faces();
    stitch_boundary_seams(&mut faces1, &pool);

    let mut faces2 = make_faces();
    stitch_boundary_seams(&mut faces2, &pool);

    let s1: Vec<_> = faces1.iter().map(|f| f.vertices).collect();
    let s2: Vec<_> = faces2.iter().map(|f| f.vertices).collect();
    assert_eq!(s1, s2, "stitch must be deterministic");
}
