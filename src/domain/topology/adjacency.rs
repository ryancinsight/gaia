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
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::FaceStore;

/// Contiguous adjacency rows addressed by an offset table.
///
/// The row count is `offsets.len() - 1`; row `r` occupies
/// `values[offsets[r]..offsets[r + 1]]`.  The generic storage is independent
/// of mesh scalar precision and is instantiated only for the index types used
/// by the graph.
struct PackedRows<T> {
    offsets: Box<[usize]>,
    values: Box<[T]>,
}

impl<T: Copy + Default> PackedRows<T> {
    /// Allocate one value buffer from exact row capacities and return cursors
    /// positioned at each row's start.
    fn from_counts(counts: Vec<usize>) -> (Self, Vec<usize>) {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);

        let mut total = 0usize;
        for count in counts {
            total = total
                .checked_add(count)
                .expect("invariant: adjacency storage size fits usize");
            offsets.push(total);
        }

        let cursors = offsets[..offsets.len() - 1].to_vec();
        let values = vec![T::default(); total].into_boxed_slice();

        (
            Self {
                offsets: offsets.into_boxed_slice(),
                values,
            },
            cursors,
        )
    }

    /// Write the next value in a row during the fill pass.
    #[inline]
    fn write(&mut self, cursors: &mut [usize], row: usize, value: T) {
        let cursor = cursors
            .get_mut(row)
            .expect("invariant: adjacency row exists");
        let end = *self
            .offsets
            .get(row + 1)
            .expect("invariant: adjacency row end exists");
        let index = *cursor;
        debug_assert!(index < end, "invariant: adjacency row capacity is exact");
        *self
            .values
            .get_mut(index)
            .expect("invariant: adjacency row has remaining capacity") = value;
        *cursor = index + 1;
    }
}

impl<T> PackedRows<T> {
    /// Return a row, or an empty slice for an out-of-range ID.
    #[inline]
    fn get(&self, row: usize) -> &[T] {
        let Some(next) = row.checked_add(1) else {
            return &[];
        };

        match (self.offsets.get(row), self.offsets.get(next)) {
            (Some(&start), Some(&end)) => &self.values[start..end],
            _ => &[],
        }
    }

    /// Number of rows in the packed relation.
    #[inline]
    fn row_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

impl<T: Copy + Ord> PackedRows<T> {
    /// Sort and deduplicate every row, compacting the value buffer in place.
    fn sort_dedup(&mut self) {
        let row_count = self.row_count();
        let mut offsets = Vec::with_capacity(row_count + 1);
        offsets.push(0);
        let mut compacted = 0usize;

        for row in 0..row_count {
            let start = self.offsets[row];
            let end = self.offsets[row + 1];
            let unique_len = {
                let row_values = &mut self.values[start..end];
                row_values.sort_unstable();
                dedup_sorted(row_values)
            };

            if compacted != start {
                self.values
                    .copy_within(start..start + unique_len, compacted);
            }
            compacted += unique_len;
            offsets.push(compacted);
        }

        let values = std::mem::replace(&mut self.values, Vec::<T>::new().into_boxed_slice());
        let mut values = values.into_vec();
        values.truncate(compacted);
        self.values = values.into_boxed_slice();
        self.offsets = offsets.into_boxed_slice();
    }
}

/// Return the unique prefix length of a sorted slice while compacting it.
fn dedup_sorted<T: Copy + Eq>(values: &mut [T]) -> usize {
    let mut unique_len = 0usize;
    for read in 0..values.len() {
        if unique_len == 0 || values[read] != values[unique_len - 1] {
            if read != unique_len {
                values.swap(read, unique_len);
            }
            unique_len += 1;
        }
    }
    unique_len
}

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
        self.vertex_neighbors.row_count()
    }

    /// Number of faces tracked in the adjacency graph.
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.face_neighbors.row_count()
    }
}

#[cfg(test)]
mod tests {
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
}
