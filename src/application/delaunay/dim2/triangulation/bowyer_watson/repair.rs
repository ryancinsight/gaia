//! Edge-flip repair and Delaunay restoration for [`super::DelaunayTriangulation`].

use super::{GHOST_TRIANGLE, Orientation, TriangleId, incircle, orient_2d,};

impl DelaunayTriangulation {
    /// Returns the neighbor triangle ID that shared `edge` before the flip.
    #[inline]
    pub(crate) fn flip_shared_edge(&mut self, tid: TriangleId, edge: usize) -> TriangleId {
        let tri = self.triangles[tid.idx()];
        let nbr_tid = tri.adj[edge];
        let nbr_tri = self.triangles[nbr_tid.idx()];
        let nbr_edge = nbr_tri.shared_edge(tid).expect("adjacency broken");

        // Preserve existing constrained flags on non-flipped boundary edges.
        let tri_cons = tri.constrained;
        let nbr_cons = nbr_tri.constrained;

        let v_opp_t = tri.vertices[edge]; // vertex opposite the shared edge in tid
        let v_opp_n = nbr_tri.vertices[nbr_edge]; // vertex opposite in nbr
        let (va, vb) = tri.edge_vertices(edge); // shared edge vertices

        let adj_tid_va = tri.adj[(edge + 1) % 3];
        let adj_tid_vb = tri.adj[(edge + 2) % 3];
        let adj_nbr_va = nbr_tri.adj[(nbr_edge + 2) % 3];
        let adj_nbr_vb = nbr_tri.adj[(nbr_edge + 1) % 3];

        // Rewrite tid → (v_opp_t, v_opp_n, vb).
        self.triangles[tid.idx()].vertices = [v_opp_t, v_opp_n, vb];
        self.triangles[tid.idx()].adj = [adj_nbr_va, adj_tid_va, nbr_tid];
        self.triangles[tid.idx()].constrained = [
            nbr_cons[(nbr_edge + 2) % 3],
            tri_cons[(edge + 1) % 3],
            false, // New diagonal is never constrained.
        ];

        // Rewrite nbr → (v_opp_n, v_opp_t, va).
        self.triangles[nbr_tid.idx()].vertices = [v_opp_n, v_opp_t, va];
        self.triangles[nbr_tid.idx()].adj = [adj_tid_vb, adj_nbr_vb, tid];
        self.triangles[nbr_tid.idx()].constrained = [
            tri_cons[(edge + 2) % 3],
            nbr_cons[(nbr_edge + 1) % 3],
            false, // New diagonal is never constrained.
        ];

        // Fix external adjacency.
        self.fix_adjacency(adj_nbr_va, nbr_tid, tid);
        self.fix_adjacency(adj_tid_vb, tid, nbr_tid);

        // Update vert_to_tri: vertices may have moved between triangles.
        self.vert_to_tri[v_opp_t.idx()] = tid;
        self.vert_to_tri[v_opp_n.idx()] = nbr_tid;
        self.vert_to_tri[va.idx()] = nbr_tid;
        self.vert_to_tri[vb.idx()] = tid;

        nbr_tid
    }

    /// Iterative Delaunay edge-flip restoration.
    ///
    /// If the edge `edge` of triangle `tid` violates the Delaunay criterion
    /// (the opposite vertex is inside the circumcircle), flip the edge and
    /// push the two new external edges onto the work stack.
    ///
    /// # Theorem — Flip Termination
    ///
    /// Each flip strictly increases the minimum angle of the affected
    /// quadrilateral.  Since the angle space is bounded and the triangulation
    /// is finite, the flip sequence terminates.
    ///
    /// # Implementation Note
    ///
    /// Uses an explicit stack instead of recursion to avoid stack overflow on
    /// large meshes where deep flip cascades can occur.
    pub(super) fn flip_fix(&mut self, start_tid: TriangleId, start_edge: usize) {
        self.restore_delaunay_edges(vec![(start_tid, start_edge)]);
    }

    /// Restore local Delaunayhood for a stack of candidate edges.
    ///
    /// Each stack item is `(triangle_id, local_edge_index)` and is processed
    /// with iterative Lawson flips until all reachable non-constrained edges
    /// satisfy the in-circle criterion.
    pub(crate) fn restore_delaunay_edges(&mut self, mut stack: Vec<(TriangleId, usize)>) {
        while let Some((tid, edge)) = stack.pop() {
            if !self.triangles[tid.idx()].alive {
                continue;
            }

            let nbr = self.triangles[tid.idx()].adj[edge];
            if nbr == GHOST_TRIANGLE {
                continue;
            }

            // Check the constraint flag — never flip a constrained edge.
            if self.triangles[tid.idx()].constrained[edge] {
                continue;
            }

            let tri = &self.triangles[tid.idx()];
            let nbr_tri = &self.triangles[nbr.idx()];

            let v_opp_t = tri.vertices[edge]; // vertex opposite the shared edge in tid
            let nbr_edge = nbr_tri.shared_edge(tid).expect("adjacency broken");
            let v_opp_n = nbr_tri.vertices[nbr_edge]; // vertex opposite in nbr

            // Shared edge vertices.
            let (va, vb) = tri.edge_vertices(edge);

            // In-circle test: is v_opp_n inside circumcircle of (v_opp_t, va, vb)?
            let pa = self.vertices[va.idx()].to_point2();
            let pb = self.vertices[vb.idx()].to_point2();
            let pc = self.vertices[v_opp_t.idx()].to_point2();
            let pd = self.vertices[v_opp_n.idx()].to_point2();

            // orient_2d(a,b,c) must be positive for the incircle test to be correct.
            let ort = orient_2d(&pa, &pb, &pc);
            let inside = if ort == Orientation::Positive {
                incircle(&pa, &pb, &pc, &pd) == Orientation::Positive
            } else if ort == Orientation::Negative {
                incircle(&pb, &pa, &pc, &pd) == Orientation::Positive
            } else {
                // Degenerate triangle — skip.
                continue;
            };

            if !inside {
                continue;
            }

            // Perform the edge flip.
            let nbr = self.flip_shared_edge(tid, edge);

            // Push the two new external edges for further checking.
            stack.push((tid, 0)); // (v_opp_n, vb) from old nbr
            stack.push((nbr, 1)); // (va, v_opp_n) from old nbr
        }
    }
}
