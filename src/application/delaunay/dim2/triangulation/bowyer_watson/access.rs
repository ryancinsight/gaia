//! Read-only accessors and compaction for [`super::DelaunayTriangulation`].

use super::{
    DelaunayTriangulation, GHOST_TRIANGLE, PslgVertex, PslgVertexId, Real, Triangle, TriangleId,
};

impl DelaunayTriangulation {
    // ── Public query API ──────────────────────────────────────────────────

    /// Number of real (non-super-triangle) vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.num_real_vertices
    }

    /// Number of alive triangles (including those incident to super-triangle vertices).
    #[must_use]
    pub fn triangle_count_raw(&self) -> usize {
        self.triangles.iter().filter(|t| t.alive).count()
    }

    /// Number of interior triangles (excluding those touching super-triangle vertices).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles
            .iter()
            .filter(|t| t.alive && !self.is_super_triangle(t))
            .count()
    }

    /// Check if a triangle is incident to a super-triangle vertex.
    #[inline]
    fn is_super_triangle(&self, tri: &Triangle) -> bool {
        tri.vertices.iter().any(|v| self.super_verts.contains(v))
    }

    /// Iterate over all interior (non-super) alive triangles.
    pub fn interior_triangles(&self) -> impl Iterator<Item = (TriangleId, &Triangle)> {
        self.triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.alive && !self.is_super_triangle(t))
            .map(|(i, t)| (TriangleId::from_usize(i), t))
    }

    /// Iterate over all alive triangles (including super-triangle ones).
    pub fn all_alive_triangles(&self) -> impl Iterator<Item = (TriangleId, &Triangle)> {
        self.triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.alive)
            .map(|(i, t)| (TriangleId::from_usize(i), t))
    }

    /// Access a vertex by ID.
    #[inline]
    #[must_use]
    pub fn vertex(&self, id: PslgVertexId) -> &PslgVertex {
        &self.vertices[id.idx()]
    }

    /// Access a triangle by ID.
    #[inline]
    #[must_use]
    pub fn triangle(&self, id: TriangleId) -> &Triangle {
        &self.triangles[id.idx()]
    }

    /// Access a mutable triangle by ID.
    #[inline]
    pub(crate) fn triangle_mut(&mut self, id: TriangleId) -> &mut Triangle {
        &mut self.triangles[id.idx()]
    }

    /// Access the full vertex slice.
    #[must_use]
    pub fn vertices(&self) -> &[PslgVertex] {
        &self.vertices
    }

    /// Access the full triangle slice.
    #[must_use]
    pub fn triangles_slice(&self) -> &[Triangle] {
        &self.triangles
    }

    /// Mutable access to the triangle slice.
    pub(crate) fn triangles_mut(&mut self) -> &mut Vec<Triangle> {
        &mut self.triangles
    }

    /// Insert a new vertex into the vertex pool and return its ID.
    ///
    /// Used by the refinement algorithm.
    pub(crate) fn add_vertex(&mut self, v: PslgVertex) -> PslgVertexId {
        let id = PslgVertexId::from_usize(self.vertices.len());
        self.vertices.push(v);
        self.vert_to_tri.push(GHOST_TRIANGLE);
        id
    }

    /// Insert a Steiner point into the triangulation.
    ///
    /// Returns the new vertex ID.
    pub(crate) fn insert_steiner(&mut self, x: Real, y: Real) -> PslgVertexId {
        let vid = self.add_vertex(PslgVertex::new(x, y));
        self.insert_vertex(vid);
        self.num_real_vertices += 1;
        vid
    }

    /// Read-only access to the vertex→triangle hint array.
    ///
    /// For each vertex `v`, `vert_to_tri_slice()[v.idx()]` holds one alive
    /// triangle incident to `v` (or [`GHOST_TRIANGLE`] if the vertex has not
    /// been inserted yet).
    ///
    /// # Invariant — Vertex-to-Triangle Validity
    ///
    /// **Statement**: After every insertion, for every inserted real vertex
    /// `$v_i$` with `$i < n$`, `$\text{vert\_to\_tri}[i]$` refers to an alive
    /// triangle whose vertex list contains $v_i$.
    ///
    /// **Proof**: Each of `insert_in_triangle`, `insert_on_edge`, and
    /// `insert_on_hull_edge` explicitly sets `vert_to_tri[v]` for every
    /// vertex of every newly created triangle.  The subsequent `flip_fix`
    /// calls update the hints for all four affected vertices after each
    /// flip.  Since flips only rearrange existing alive triangles (marking
    /// old ones dead and rewriting in-place), the hint is always updated
    /// to an alive triangle containing the vertex.  ∎
    #[must_use]
    pub fn vert_to_tri_slice(&self) -> &[TriangleId] {
        &self.vert_to_tri
    }

    /// Remove dead (tombstoned) triangles and remap all adjacency links.
    ///
    /// # Theorem — Compaction Invariant Preservation
    ///
    /// **Statement**: Let $T$ be a triangulation with $D$ dead and $A$ alive
    /// triangles.  `compact()` produces a triangulation $T'$ with exactly
    /// $A$ triangles such that:
    ///
    /// 1. Every alive triangle in $T$ maps bijectively to a triangle in $T'$.
    /// 2. Adjacency is preserved: $\text{adj}_{T'}(f(t), e) = f(\text{adj}_T(t, e))$
    ///    for all alive $t$ and edges $e$, where $f$ is the remapping.
    /// 3. Vertex→triangle hints remain valid.
    /// 4. Constrained-edge flags are preserved.
    ///
    /// **Proof sketch**: The remapping $f : [0 \dots A{+}D) \to [0 \dots A)$
    /// is a monotone injection on alive indices.  Because dead triangles are
    /// never referenced by alive triangles' adjacency lists (each insertion
    /// and flip only writes alive triangle IDs), every adjacency entry is
    /// either `GHOST_TRIANGLE` or an alive triangle ID.  Replacing each
    /// alive ID by its $f$-image preserves the bijection.  The `vert_to_tri`
    /// hints point to alive triangles by the vertex-to-triangle invariant
    /// above, so remapping them through $f$ keeps them valid.  ∎
    ///
    /// # Complexity
    ///
    /// $O(A + D)$ time and $O(A + D)$ auxiliary space for the remap table.
    pub fn compact(&mut self) {
        let n = self.triangles.len();
        let mut remap = vec![GHOST_TRIANGLE; n];
        let mut new_idx = 0u32;
        for i in 0..n {
            if self.triangles[i].alive {
                remap[i] = TriangleId::new(new_idx);
                new_idx += 1;
            }
        }

        // Compact the triangle array in-place.
        let mut write = 0;
        for read in 0..n {
            if self.triangles[read].alive {
                self.triangles[write] = self.triangles[read];
                for a in &mut self.triangles[write].adj {
                    if *a != GHOST_TRIANGLE {
                        *a = remap[a.idx()];
                    }
                }
                write += 1;
            }
        }
        self.triangles.truncate(write);

        // Remap vertex→triangle hints.
        for hint in &mut self.vert_to_tri {
            if *hint != GHOST_TRIANGLE && hint.idx() < n {
                *hint = remap[hint.idx()];
            }
        }

        // Remap last_triangle.
        if self.last_triangle != GHOST_TRIANGLE && self.last_triangle.idx() < n {
            self.last_triangle = remap[self.last_triangle.idx()];
        } else {
            self.last_triangle = if self.triangles.is_empty() {
                GHOST_TRIANGLE
            } else {
                TriangleId::new(0)
            };
        }
    }
}
