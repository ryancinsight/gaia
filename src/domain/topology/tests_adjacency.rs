//! Tests for the parent module, extracted from the module body.

use super::*;
use crate::domain::core::index::RegionId;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};

fn vid(n: u32) -> VertexId {
    VertexId::new(n)
}

/// Build a tetrahedron face store (4 faces, 4 vertices, 6 edges).
fn tetra_store() -> FaceStore {
    let mut s = FaceStore::new();
    // 4 faces of tetrahedron v0..v3:
    s.push(FaceData::new(vid(0), vid(2), vid(1), RegionId::INVALID));
    s.push(FaceData::new(vid(0), vid(1), vid(3), RegionId::INVALID));
    s.push(FaceData::new(vid(1), vid(2), vid(3), RegionId::INVALID));
    s.push(FaceData::new(vid(2), vid(0), vid(3), RegionId::INVALID));
    s
}

/// Build a minimal cube face store (12 faces = 6 quads triangulated,
/// 8 vertices, 18 edges).
fn cube_store() -> FaceStore {
    let mut s = FaceStore::new();
    // Front (z=0): v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0)
    // Back  (z=1): v4(0,0,1) v5(1,0,1) v6(1,1,1) v7(0,1,1)
    // Front face
    s.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    s.push(FaceData::new(vid(0), vid(2), vid(3), RegionId::INVALID));
    // Back face
    s.push(FaceData::new(vid(4), vid(6), vid(5), RegionId::INVALID));
    s.push(FaceData::new(vid(4), vid(7), vid(6), RegionId::INVALID));
    // Bottom face (y=0)
    s.push(FaceData::new(vid(0), vid(5), vid(1), RegionId::INVALID));
    s.push(FaceData::new(vid(0), vid(4), vid(5), RegionId::INVALID));
    // Top face (y=1)
    s.push(FaceData::new(vid(3), vid(2), vid(6), RegionId::INVALID));
    s.push(FaceData::new(vid(3), vid(6), vid(7), RegionId::INVALID));
    // Left face (x=0)
    s.push(FaceData::new(vid(0), vid(3), vid(7), RegionId::INVALID));
    s.push(FaceData::new(vid(0), vid(7), vid(4), RegionId::INVALID));
    // Right face (x=1)
    s.push(FaceData::new(vid(1), vid(5), vid(6), RegionId::INVALID));
    s.push(FaceData::new(vid(1), vid(6), vid(2), RegionId::INVALID));
    s
}

fn adj_from(fs: &FaceStore) -> AdjacencyGraph {
    let es = EdgeStore::from_face_store(fs);
    AdjacencyGraph::build(fs, &es)
}

/// Every vertex of a tetrahedron is adjacent to the other 3 (K₄ graph).
///
/// # Theorem — Complete Vertex Adjacency of the Tetrahedron
///
/// **Statement.** In a tetrahedron with 4 vertices, each vertex is
/// connected to every other vertex by an edge, so `vertex_valence(v) = 3`
/// for all v.
///
/// **Proof.** A tetrahedron has $\binom{4}{2} = 6$ edges.  Each of the 4
/// vertices is incident to exactly $4 - 1 = 3$ edges, and no two edges
/// share both endpoints.  ∎
#[test]
fn tet_vertex_valence_is_three() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    for i in 0..4 {
        assert_eq!(
            adj.vertex_valence(vid(i)),
            3,
            "vertex {i} should have valence 3 in a tetrahedron"
        );
    }
}

/// Each vertex of a tetrahedron is incident to exactly 3 faces.
///
/// # Theorem — Vertex-Face Incidence of the Tetrahedron
///
/// **Statement.** Each vertex of a tetrahedron belongs to exactly 3 of the
/// 4 faces (every face except the one opposite the vertex).
///
/// **Proof.** Face $i$ is the triangle on the 3 vertices excluding vertex
/// $i$.  So vertex $v$ is absent from exactly one face and present in
/// the remaining $4 - 1 = 3$ faces.  ∎
#[test]
fn tet_vertex_faces_count_is_three() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    for i in 0..4 {
        assert_eq!(
            adj.vertex_faces(vid(i)).len(),
            3,
            "vertex {i} should be incident to 3 faces in a tetrahedron"
        );
    }
}

/// Each face of a tetrahedron is adjacent to exactly 3 other faces (K₄).
///
/// # Theorem — Face Adjacency is Complete for the Tetrahedron
///
/// **Statement.** In the face-adjacency graph of a tetrahedron, every face
/// is adjacent to all 3 other faces.
///
/// **Proof.** Each triangular face shares an edge with each of the other 3
/// faces (a tetrahedron has 6 edges and each face has 3 edges — each edge
/// is shared with exactly one other face).  The face-adjacency graph is
/// therefore K₄.  ∎
#[test]
fn tet_face_neighbors_count_is_three() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    for i in 0..4 {
        assert_eq!(
            adj.face_neighbors(FaceId::from_usize(i)).len(),
            3,
            "face {i} should have 3 face-neighbors in a tetrahedron"
        );
    }
}

/// Cube: each vertex should have valence 3.
///
/// # Theorem — Cube Vertex Valence
///
/// **Statement.** In a triangulated cube (12 triangles, 8 vertices, 18
/// edges), each vertex has degree 3 in the cube's edge graph (ignoring
/// the diagonal edges introduced by triangulation, the cube graph is
/// the 3-regular graph Q₃).
///
/// **Proof.** Each cubic vertex is the corner of exactly 3 mutually-
/// perpendicular faces, contributing 3 original edges.  Triangulation
/// introduces diagonals, increasing degree.  We verify empirically that
/// the actual triangulated valence satisfies `valence(v) ≥ 3`.  ∎
#[test]
fn cube_vertex_valence_at_least_three() {
    let fs = cube_store();
    let adj = adj_from(&fs);
    for i in 0..8 {
        let val = adj.vertex_valence(vid(i));
        assert!(val >= 3, "cube vertex {i} has valence {val} (expected ≥ 3)");
    }
}

/// Cube: 12 faces, each face has ≥ 1 neighbor.
#[test]
fn cube_all_faces_have_neighbors() {
    let fs = cube_store();
    let adj = adj_from(&fs);
    assert_eq!(adj.num_faces(), 12);
    for i in 0..12 {
        assert!(
            !adj.face_neighbors(FaceId::from_usize(i)).is_empty(),
            "cube face {i} should have at least one neighbor"
        );
    }
}

/// Empty mesh — build should succeed and return empty results.
///
/// # Theorem — Empty Graph Adjacency
///
/// **Statement.** An empty mesh (0 faces, 0 edges) produces an adjacency
/// graph with `num_vertices() == 0` and `num_faces() == 0`, and all
/// lookups return empty slices.
///
/// **Proof.** With no faces, the max-vertex-ID scan yields 0.  With no
/// edges, no adjacency entries are created.  ∎
#[test]
fn empty_mesh_no_panic() {
    let fs = FaceStore::new();
    let adj = adj_from(&fs);
    assert_eq!(adj.num_vertices(), 0);
    assert_eq!(adj.num_faces(), 0);
    assert!(adj.vertex_neighbors(vid(0)).is_empty());
    assert!(adj.vertex_faces(vid(0)).is_empty());
    assert!(adj.face_neighbors(FaceId::from_usize(0)).is_empty());
}

/// Single isolated triangle — 3 vertices each with valence 2.
///
/// # Theorem — Triangle Graph Valence
///
/// **Statement.** A single triangle has 3 edges and each vertex is
/// incident to exactly 2 edges, giving valence 2.
///
/// **Proof.** The triangle cycle graph C₃ is 2-regular.  ∎
#[test]
fn single_triangle_valence_two() {
    let mut fs = FaceStore::new();
    fs.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    let adj = adj_from(&fs);
    for i in 0..3 {
        assert_eq!(adj.vertex_valence(vid(i)), 2);
        assert_eq!(adj.vertex_faces(vid(i)).len(), 1);
    }
    // Single face has no face-neighbor (no edge shared with another face).
    assert!(adj.face_neighbors(FaceId::from_usize(0)).is_empty());
}

/// Non-manifold edge — 3 faces sharing one edge.
///
/// This is a known failure mode in mesh libraries: when an edge is
/// non-manifold (shared by >2 faces), the face-neighbor relationship
/// must include all pairwise adjacencies, not just the first 2.
///
/// # Theorem — Non-Manifold Edge Face-Pair Count
///
/// **Statement.** An edge shared by k faces generates $\binom{k}{2}$
/// face-neighbor pairs.  For k = 3, this is 3 pairs.
///
/// **Proof.** The inner loop over `(i, j)` with `i < j` enumerates all
/// $\binom{k}{2}$ unordered pairs.  Each pair generates two directed
/// entries in `face_neighbors` (one per face).  ∎
#[test]
fn non_manifold_edge_all_face_pairs() {
    // Three triangles sharing edge (v0, v1): apexes at v2, v3, v4.
    let mut fs = FaceStore::new();
    fs.push(FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(1), vid(3), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(1), vid(4), RegionId::INVALID));
    let adj = adj_from(&fs);

    // Each face should see the other 2 as neighbors (via the shared edge).
    for i in 0..3 {
        let nbrs = adj.face_neighbors(FaceId::from_usize(i));
        assert!(
            nbrs.len() >= 2,
            "face {} has {} neighbors; expected ≥ 2 for 3-face non-manifold edge",
            i,
            nbrs.len()
        );
    }
}

/// Duplicate triangles share three edges, but the public relation keeps
/// one neighboring face entry after packed-row deduplication.
#[test]
fn duplicate_faces_are_deduplicated() {
    let mut fs = FaceStore::new();
    let face = FaceData::new(vid(0), vid(1), vid(2), RegionId::INVALID);
    fs.push(face);
    fs.push(face);

    let adj = adj_from(&fs);
    assert_eq!(adj.face_neighbors(FaceId::from_usize(0)), &[FaceId(1)]);
    assert_eq!(adj.face_neighbors(FaceId::from_usize(1)), &[FaceId(0)]);
}

/// Bowtie vertex (pinch vertex) — two fans joined at one vertex.
///
/// This is a known failure mode in mesh libraries: a vertex shared by
/// two separate face fans should report ALL incident faces, not just
/// the first fan discovered by traversal.
///
/// # Theorem — Bowtie Vertex Incidence
///
/// **Statement.** A pinch vertex v shared by two triangular fans of
/// sizes $f_1$ and $f_2$ has $|vertex\_faces(v)| = f_1 + f_2$.
///
/// **Proof.** The face-pass in `build()` pushes each incident face
/// independently; it does not rely on edge-walking or fan-traversal,
/// so it cannot miss faces from either fan.  ∎
#[test]
fn bowtie_vertex_all_incident_faces() {
    // Fan 1: triangles (v0,v1,v4), (v0,v2,v1), (v0,v3,v2)
    // Fan 2: triangles (v0,v5,v6), (v0,v6,v7)
    // v0 is the bowtie vertex connecting both fans
    let mut fs = FaceStore::new();
    fs.push(FaceData::new(vid(0), vid(1), vid(4), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(2), vid(1), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(3), vid(2), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(5), vid(6), RegionId::INVALID));
    fs.push(FaceData::new(vid(0), vid(6), vid(7), RegionId::INVALID));

    let adj = adj_from(&fs);
    // v0 should see all 5 faces.
    assert_eq!(
        adj.vertex_faces(vid(0)).len(),
        5,
        "bowtie vertex v0 must see all 5 incident faces"
    );
    // v0's neighbors: v1, v2, v3, v4, v5, v6, v7 = 7 distinct vertices.
    assert_eq!(
        adj.vertex_valence(vid(0)),
        7,
        "bowtie vertex v0 must see all 7 neighbors"
    );
}

/// Two disjoint tetrahedra — vertex-out-of-range queries return empty.
///
/// # Theorem — Safe Out-of-Range Lookup
///
/// **Statement.** Querying `vertex_neighbors`, `vertex_faces`, or
/// `face_neighbors` with an ID beyond the dense array returns `&[]`.
///
/// **Proof.** Each accessor checks both row offsets with `.get()` and
/// returns the default empty slice when either offset is out of bounds.
/// ∎
#[test]
fn out_of_range_id_returns_empty() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    let far = vid(999);
    assert!(adj.vertex_neighbors(far).is_empty());
    assert!(adj.vertex_faces(far).is_empty());
    assert!(adj.face_neighbors(FaceId::from_usize(999)).is_empty());
}

/// Vertex-neighbor uniqueness: no duplicate neighbors in a tetrahedron.
///
/// Validates the vertex-neighbor uniqueness theorem: since EdgeStore
/// stores each undirected edge exactly once, the push-both-directions
/// step cannot create duplicates.
#[test]
fn vertex_neighbors_no_duplicates() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    for i in 0..4 {
        let nbrs = adj.vertex_neighbors(vid(i));
        let mut sorted = nbrs.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), before, "vertex {i} has duplicate neighbors");
    }
}

/// Face-neighbor symmetry: if face A lists face B as a neighbor,
/// face B lists face A.
///
/// # Theorem — Face-Neighbor Symmetry
///
/// **Statement.** The face-adjacency relation is symmetric: if face f₁
/// is in `face_neighbors(f₀)`, then f₀ is in `face_neighbors(f₁)`.
///
/// **Proof.** The edge-pass inserts both directions for every face-pair
/// `(fi, fj)` sharing an edge: `face_neighbors[fi].push(fj)` and
/// `face_neighbors[fj].push(fi)`.  ∎
#[test]
fn face_neighbor_symmetry() {
    let fs = tetra_store();
    let adj = adj_from(&fs);
    for i in 0..fs.len() {
        let fid = FaceId::from_usize(i);
        for &nbr in adj.face_neighbors(fid) {
            assert!(
                adj.face_neighbors(nbr).contains(&fid),
                "face {i} lists face {nbr:?} as neighbor, but not vice-versa"
            );
        }
    }
}
