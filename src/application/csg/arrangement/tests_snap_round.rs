//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::scalar::{Point3r, Vector3r};

#[test]
fn snap_round_splits_exact_tjunction_with_endpoint_constraint() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(2.0, 0.0, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(0.0, 1.0, 0.0), n);
    let m = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(2.0, 1.0, 0.0), n);

    let mut faces = vec![FaceData::untagged(a, b, c), FaceData::untagged(m, b, d)];
    snap_round_tjunctions(&mut faces, &pool);

    assert_eq!(
        faces.len(),
        3,
        "one constrained exact split should be applied"
    );
    let with_mc = faces
        .iter()
        .filter(|f| f.vertices.contains(&m) && f.vertices.contains(&c))
        .count();
    assert_eq!(
        with_mc, 2,
        "split should replace [a,b,c] with two triangles using split vertex m"
    );
}

#[test]
fn snap_round_splits_exact_collinear_vertex_without_endpoint_constraint() {
    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let a = pool.insert_or_weld(Point3r::new(0.0, 0.0, 0.0), n);
    let b = pool.insert_or_weld(Point3r::new(2.0, 0.0, 0.0), n);
    let c = pool.insert_or_weld(Point3r::new(0.0, 1.0, 0.0), n);
    let m = pool.insert_or_weld(Point3r::new(1.0, 0.0, 0.0), n);
    let d = pool.insert_or_weld(Point3r::new(1.0, 1.0, 0.0), n);
    let e = pool.insert_or_weld(Point3r::new(2.0, 1.0, 0.0), n);

    // m lies exactly on [a,b] but is not boundary-adjacent to a or b.
    let mut faces = vec![FaceData::untagged(a, b, c), FaceData::untagged(m, d, e)];
    snap_round_tjunctions(&mut faces, &pool);

    assert_eq!(
        faces.len(),
        3,
        "exact on-edge boundary vertices should trigger a split even without endpoint adjacency"
    );
    assert!(
        faces
            .iter()
            .filter(|face| face.vertices.contains(&m))
            .count()
            >= 2,
        "split faces should contain the exact T-junction vertex"
    );
}

#[test]
fn adversarial_endpoint_index_matches_full_scan_candidates() {
    use super::super::mesh_ops::boundary_half_edges;

    let mut pool = VertexPool::default_millifluidic();
    let n = Vector3r::new(0.0, 0.0, 1.0);
    let ids: Vec<VertexId> = (0..7)
        .map(|i| pool.insert_or_weld(Point3r::new(Real::from(i), 0.0, 0.0), n))
        .collect();
    let [v0, v1, v2, v3, v4, v5, v6] = <[VertexId; 7]>::try_from(ids).expect("7 ids");

    let faces = vec![
        FaceData::untagged(v0, v1, v2),
        FaceData::untagged(v0, v2, v3),
        FaceData::untagged(v0, v3, v4),
        FaceData::untagged(v0, v4, v1),
        FaceData::untagged(v2, v5, v4),
        FaceData::untagged(v4, v5, v6),
    ];

    let boundary = boundary_half_edges(&faces);
    let bnd_adj = build_boundary_adjacency(&boundary);
    let endpoint_index = build_endpoint_edge_index(&faces);

    let mut boundary_vertices: Vec<VertexId> = boundary.iter().flat_map(|&(a, b)| [a, b]).collect();
    boundary_vertices.sort();
    boundary_vertices.dedup();

    for v in boundary_vertices {
        let mut full_scan: HashSet<(usize, VertexId, VertexId)> = HashSet::new();
        for (fi, face) in faces.iter().enumerate() {
            for edge_idx in 0..3_usize {
                let a = face.vertices[edge_idx];
                let b = face.vertices[(edge_idx + 1) % 3];
                if !endpoint_constrained(v, a, b, &bnd_adj) {
                    continue;
                }
                let (mn, mx) = if a < b { (a, b) } else { (b, a) };
                full_scan.insert((fi, mn, mx));
            }
        }

        let indexed: HashSet<(usize, VertexId, VertexId)> =
            candidate_face_edges_for_vertex(v, &bnd_adj, &endpoint_index)
                .into_iter()
                .map(|(fi, a, b)| {
                    let (mn, mx) = if a < b { (a, b) } else { (b, a) };
                    (fi, mn, mx)
                })
                .collect();

        assert_eq!(
            indexed, full_scan,
            "endpoint-index candidate set must match full-scan set"
        );
    }
}
