//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::RegionId;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};

fn vid(n: u32) -> VertexId {
    VertexId::new(n)
}

fn tetra_store() -> FaceStore {
    let mut s = FaceStore::new();
    s.push(FaceData::new(vid(0), vid(2), vid(1), RegionId::INVALID));
    s.push(FaceData::new(vid(0), vid(1), vid(3), RegionId::INVALID));
    s.push(FaceData::new(vid(1), vid(2), vid(3), RegionId::INVALID));
    s.push(FaceData::new(vid(2), vid(0), vid(3), RegionId::INVALID));
    s
}

/// A tetrahedron has exactly 6 edges: $\binom{4}{2} = 6$.
///
/// # Theorem — Tetrahedron Edge Count
///
/// **Statement.** A tetrahedron on 4 vertices has exactly 6 edges.
///
/// **Proof.** Each pair of distinct vertices is connected by an edge.
/// There are $\binom{4}{2} = 6$ such pairs.  ∎
#[test]
fn tet_edge_count_is_six() {
    let fs = tetra_store();
    let es = EdgeStore::from_face_store(&fs);
    assert_eq!(es.len(), 6);
}

/// Every edge of a closed tetrahedron is manifold (shared by exactly 2 faces).
///
/// # Theorem — Tetrahedron Manifold Edges
///
/// **Statement.** In a closed tetrahedron, each of the 6 edges is shared
/// by exactly 2 of the 4 triangular faces.
///
/// **Proof.** Each edge `{a, b}` is the intersection of the two faces
/// containing both `a` and `b`.  Since $\binom{4-2}{1} + 1 = 2$ faces
/// contain any given edge (choose 1 of the remaining 2 vertices for
/// each face), every edge has valence 2.  ∎
#[test]
fn tet_all_edges_manifold() {
    let fs = tetra_store();
    let es = EdgeStore::from_face_store(&fs);
    for edge in es.iter() {
        assert!(
            edge.is_manifold(),
            "edge {:?} has valence {} (expected 2)",
            edge.vertices,
            edge.valence()
        );
    }
    assert_eq!(es.boundary_edge_count(), 0);
}

/// `find_edge` is canonical — order of arguments does not matter.
///
/// # Theorem — Canonical Edge Lookup
///
/// **Statement.** `find_edge(a, b)` and `find_edge(b, a)` return the
/// same `EdgeId`.
///
/// **Proof.** Both calls compute the canonical key
/// `(min(a, b), max(a, b))` and look it up in the same `edge_map`.  ∎
#[test]
fn find_edge_canonical_order() {
    let fs = tetra_store();
    let es = EdgeStore::from_face_store(&fs);
    for (index, edge) in es.iter().enumerate() {
        let (a, b) = edge.vertices;
        let id_ab = es.find_edge(a, b);
        let id_ba = es.find_edge(b, a);
        assert_eq!(id_ab, id_ba, "canonical order violated for {:?}", (a, b));
        // A stored edge must resolve to its own id, not merely to something.
        assert_eq!(
            id_ab,
            Some(EdgeId::from_usize(index)),
            "find_edge must return the stored edge's own id for {:?}",
            (a, b)
        );
    }
}

/// A single triangle has 3 boundary edges (valence 1).
#[test]
fn single_triangle_boundary_edges() {
    let mut fs = FaceStore::new();
    fs.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    let es = EdgeStore::from_face_store(&fs);
    assert_eq!(es.len(), 3);
    assert_eq!(es.boundary_edge_count(), 3);
    for edge in es.iter() {
        assert!(edge.is_boundary());
    }
}

/// Non-manifold edge detection: 3 faces sharing one edge.
///
/// # Theorem — Non-Manifold Detection
///
/// **Statement.** An edge shared by $k > 2$ faces is classified as
/// non-manifold (valence $k$).
///
/// **Proof.** `register_edge` pushes each face_id into the edge's
/// face list.  After processing all faces, `edge.faces.len() == k`,
/// and `is_non_manifold()` returns `k > 2`.  ∎
#[test]
fn non_manifold_edge_detected() {
    let mut fs = FaceStore::new();
    // 3 triangles sharing edge (v0, v1)
    fs.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(1), vid(3), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(1), vid(4), RegionId::INVALID));
    let es = EdgeStore::from_face_store(&fs);
    let nm = es.non_manifold_edges();
    assert_eq!(nm.len(), 1, "exactly one non-manifold edge expected");
    let e = es.get(nm[0]);
    assert_eq!(e.valence(), 3);
    assert!(e.is_non_manifold());
}

/// Empty face store produces empty edge store.
#[test]
fn empty_face_store_empty_edges() {
    let fs = FaceStore::new();
    let es = EdgeStore::from_face_store(&fs);
    assert!(es.is_empty());
    assert_eq!(es.len(), 0);
    assert_eq!(es.boundary_edge_count(), 0);
}

/// Cube (12 triangles, 8 vertices) has 18 edges — all manifold.
///
/// # Theorem — Cube Edge Count
///
/// **Statement.** A triangulated cube with 8 vertices, 12 triangles,
/// and 6 diagonal edges has E = 18.  By Euler: V - E + F = 2 →
/// 8 - E + 12 = 2 → E = 18.
///
/// **Proof.** Direct application of the Euler formula for a closed
/// genus-0 surface.  ∎
#[test]
fn cube_edge_count_and_manifold() {
    let mut fs = FaceStore::new();
    // Front z=0: v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0)
    // Back z=1: v4(0,0,1) v5(1,0,1) v6(1,1,1) v7(0,1,1)
    fs.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(2), vid(3), RegionId::INVALID));
    fs.push(FaceData::new(vid(4), vid(6), vid(5), RegionId::INVALID));
    fs.push(FaceData::new(vid(4), vid(7), vid(6), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(5), vid(1), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(4), vid(5), RegionId::INVALID));
    fs.push(FaceData::new(vid(3), vid(2), vid(6), RegionId::INVALID));
    fs.push(FaceData::new(vid(3), vid(6), vid(7), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(3), vid(7), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(7), vid(4), RegionId::INVALID));
    fs.push(FaceData::new(vid(1), vid(5), vid(6), RegionId::INVALID));
    fs.push(FaceData::new(vid(1), vid(6), vid(2), RegionId::INVALID));
    let es = EdgeStore::from_face_store(&fs);
    assert_eq!(es.len(), 18);
    assert_eq!(es.boundary_edge_count(), 0);
    for edge in es.iter() {
        assert!(
            edge.is_manifold(),
            "cube edge {:?} has valence {} (expected 2)",
            edge.vertices,
            edge.valence()
        );
    }
}

/// `clear()` resets both the edge vec and the edge map.
#[test]
fn clear_resets_store() {
    let fs = tetra_store();
    let mut es = EdgeStore::from_face_store(&fs);
    assert!(!es.is_empty());
    es.clear();
    assert!(es.is_empty());
    assert_eq!(es.len(), 0);
}
