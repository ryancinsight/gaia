//! Vertex and face adjacency graph.
//!
//! Provides dense O(1)-lookup adjacency maps for vertex-vertex (1-ring),
//! vertex-face (incidence), and face-face (edge-sharing) queries.
//!
//! # Algorithm — N-ary Dense Adjacency Construction
//!
//! All three adjacency maps are built by first counting each row's required
//! capacity, then filling contiguous value buffers addressed by row offsets:
//!
//! 1. **Count pass** (O(F + E)): Count vertex-face incidence, vertex valence,
//!    and pre-dedup face-neighbor entries.
//! 2. **Fill pass** (O(F + E)): For each face, record vertex → face incidence;
//!    for each edge, record vertex → vertex adjacency
//!    and, for each pair of faces sharing the edge, record face → face adjacency.
//!
//! Packed rows indexed by `VertexId::as_usize()` and `FaceId::as_usize()` use
//! one offset table and one contiguous value buffer per relation.  This keeps
//! lookups O(1) while avoiding one heap allocation and one `Vec` header per
//! entity.
//!
//! # Theorem — Vertex-Neighbor Uniqueness
//!
//! **Statement.** When the `EdgeStore` contains each undirected edge exactly
//! once (canonical `(min, max)` key), iterating edges and pushing both
//! directions into `vertex_neighbors` produces zero duplicates.
//!
//! **Proof.** Each undirected edge `{a, b}` has exactly one representative
//! in the `EdgeStore` with vertices `(min(a,b), max(a,b))`.  The edge-pass
//! pushes `b` into `vertex_neighbors[a]` and `a` into `vertex_neighbors[b]`.
//! Since no other entry in the store has the same canonical key, no other
//! iteration step pushes `b` into `vertex_neighbors[a]` (or vice-versa).
//! Therefore every `(vertex, neighbor)` pair appears exactly once.  ∎
//!
//! # Theorem — Face-Neighbor Correctness
//!
//! **Statement.** For a valid triangle mesh where no two distinct faces share
//! more than one edge, the face-neighbor lists produced by the edge-pass
//! contain no duplicates.
//!
//! **Proof.** Each edge contributes at most one `(fi, fj)` pair to
//! `face_neighbors`.  If `fi` and `fj` shared two distinct edges, they would
//! share at least 3 vertices — but two distinct triangles on the same 3
//! vertices are identical or reverse-wound, contradicting distinctness.
//! Hence each `(fi, fj)` pair appears at most once.  ∎
//!
//! A defensive sort+dedup is retained for `face_neighbors` to handle
//! degenerate input (e.g., duplicated faces with different IDs).  Deduplication
//! compacts the packed rows in place before the value buffer is frozen.
//!
//! # Complexity
//!
//! Construction: **O(V + E + F)** time, **O(V + F + Σdeg)** space.
//! Lookups: **O(1)** (direct array index, no hashing).

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::topology::PackedRows;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::FaceStore;

/// Pre-built adjacency graph for vertex-vertex and vertex-face queries.
///
/// Uses packed rows indexed by entity ID for O(1) lookups with zero hash
/// overhead and no per-entity heap allocations after construction.
pub struct AdjacencyGraph {
    /// vertex → list of adjacent vertices (1-ring neighborhood).
    /// Indexed by `VertexId::as_usize()`.  No duplicates (see module-level
    /// vertex-neighbor uniqueness theorem).
    vertex_neighbors: PackedRows<VertexId>,
    /// vertex → list of incident faces.
    /// Indexed by `VertexId::as_usize()`.
    vertex_faces: PackedRows<FaceId>,
    /// face → list of adjacent faces (sharing an edge).
    /// Indexed by `FaceId::as_usize()`.
    face_neighbors: PackedRows<FaceId>,
}

impl AdjacencyGraph {
    /// Build the adjacency graph from edge and face stores.
    ///
    /// Counts list capacities before filling dense adjacency maps, avoiding
    /// incremental reallocation while preserving O(V + E + F) construction.
    #[must_use]
    pub fn build(face_store: &FaceStore, edge_store: &EdgeStore) -> Self {
        let n_faces = face_store.len();

        // Compute the required vertex-array length from the maximum vertex ID
        // referenced by any face.  O(F) scan.
        let n_vertices = face_store
            .iter_enumerated()
            .flat_map(|(_, f)| f.vertices.iter())
            .map(|v| v.as_usize() + 1)
            .max()
            .unwrap_or(0);

        let mut vertex_neighbor_counts = vec![0usize; n_vertices];
        let mut vertex_face_counts = vec![0usize; n_vertices];
        let mut face_neighbor_counts = vec![0usize; n_faces];

        for (_, face) in face_store.iter_enumerated() {
            for &vid in &face.vertices {
                vertex_face_counts[vid.as_usize()] += 1;
            }
        }

        for edge in edge_store.iter() {
            let (a, b) = edge.vertices;
            vertex_neighbor_counts[a.as_usize()] += 1;
            vertex_neighbor_counts[b.as_usize()] += 1;

            let neighbor_count = edge.faces.len().saturating_sub(1);
            for &face in &edge.faces {
                face_neighbor_counts[face.as_usize()] += neighbor_count;
            }
        }

        let (mut vertex_neighbors, mut vertex_neighbor_cursors) =
            PackedRows::from_counts(vertex_neighbor_counts);
        let (mut vertex_faces, mut vertex_face_cursors) =
            PackedRows::from_counts(vertex_face_counts);
        let (mut face_neighbors, mut face_neighbor_cursors) =
            PackedRows::from_counts(face_neighbor_counts);

        // Pass 1 — vertex → face incidence from face store.
        for (fid, face) in face_store.iter_enumerated() {
            for &vid in &face.vertices {
                vertex_faces.write(&mut vertex_face_cursors, vid.as_usize(), fid);
            }
        }

        // Pass 2 — vertex-vertex and face-face from edge store.
        for edge in edge_store.iter() {
            let (a, b) = edge.vertices;
            vertex_neighbors.write(&mut vertex_neighbor_cursors, a.as_usize(), b);
            vertex_neighbors.write(&mut vertex_neighbor_cursors, b.as_usize(), a);

            // All face-pairs sharing this edge are neighbors.
            for i in 0..edge.faces.len() {
                for j in (i + 1)..edge.faces.len() {
                    let fi = edge.faces[i];
                    let fj = edge.faces[j];
                    face_neighbors.write(&mut face_neighbor_cursors, fi.as_usize(), fj);
                    face_neighbors.write(&mut face_neighbor_cursors, fj.as_usize(), fi);
                }
            }
        }

        // Vertex neighbors: duplicates are impossible (uniqueness theorem).
        // Face neighbors: defensive dedup for degenerate input.
        face_neighbors.sort_dedup();

        Self {
            vertex_neighbors,
            vertex_faces,
            face_neighbors,
        }
    }

    /// Get the 1-ring vertex neighborhood.
    #[must_use]
    pub fn vertex_neighbors(&self, v: VertexId) -> &[VertexId] {
        self.vertex_neighbors.get(v.as_usize())
    }

    /// Get faces incident to a vertex.
    #[must_use]
    pub fn vertex_faces(&self, v: VertexId) -> &[FaceId] {
        self.vertex_faces.get(v.as_usize())
    }

    /// Get faces neighboring a given face (sharing an edge).
    #[must_use]
    pub fn face_neighbors(&self, f: FaceId) -> &[FaceId] {
        self.face_neighbors.get(f.as_usize())
    }

    /// Vertex valence (number of adjacent vertices).
    #[must_use]
    pub fn vertex_valence(&self, v: VertexId) -> usize {
        self.vertex_neighbors(v).len()
    }

    /// Number of vertices tracked in the adjacency graph.
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.vertex_neighbors.len()
    }

    /// Number of faces tracked in the adjacency graph.
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.face_neighbors.len()
    }
}

#[cfg(test)]
#[path = "tests_adjacency.rs"]
mod tests;
