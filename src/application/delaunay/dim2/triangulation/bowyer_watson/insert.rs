//! Incremental vertex insertion: cavity split, fan fill, and adjacency repair.

use super::DelaunayTriangulation;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::dim2::triangulation::triangle::{
    Triangle, TriangleId, GHOST_TRIANGLE,
};

impl DelaunayTriangulation {
    /// Insert vertex `vid` strictly inside triangle `tid`.
    ///
    /// Splits `tid` into 3 sub-triangles and restores Delaunay via edge flips.
    pub(super) fn insert_in_triangle(&mut self, vid: PslgVertexId, tid: TriangleId) {
        let old = self.triangles[tid.idx()];
        let [v0, v1, v2] = old.vertices;
        let [a0, a1, a2] = old.adj;

        // Kill the old triangle.
        self.triangles[tid.idx()].alive = false;

        // Create 3 new triangles.
        //   t0: (vid, v0, v1)  — replaces tid's slot is reused for t0
        //   t1: (vid, v1, v2)
        //   t2: (vid, v2, v0)
        let t0 = self.push_triangle(Triangle::new(vid, v0, v1));
        let t1 = self.push_triangle(Triangle::new(vid, v1, v2));
        let t2 = self.push_triangle(Triangle::new(vid, v2, v0));

        // Internal adjacency: each new triangle is adjacent to the other two.
        //  t0: edge opposite vid (edge 0) is between v0,v1 → adj[0] = old adj[2]
        //      edge opposite v0 (edge 1) is vid,v1 → adj[1] = t1
        //      edge opposite v1 (edge 2) is vid,v0 → adj[2] = t2
        self.triangles[t0.idx()].adj = [a2, t1, t2];
        self.triangles[t1.idx()].adj = [a0, t2, t0];
        self.triangles[t2.idx()].adj = [a1, t0, t1];

        // Fix external adjacency: the old neighbors pointed to `tid` — update them.
        self.fix_adjacency(a2, tid, t0);
        self.fix_adjacency(a0, tid, t1);
        self.fix_adjacency(a1, tid, t2);

        // Update vert_to_tri for all affected vertices.
        self.vert_to_tri[vid.idx()] = t0;
        self.vert_to_tri[v0.idx()] = t0;
        self.vert_to_tri[v1.idx()] = t1;
        self.vert_to_tri[v2.idx()] = t2;

        self.last_triangle = t0;

        // Restore Delaunay property via edge flips.
        self.flip_fix(t0, 0);
        self.flip_fix(t1, 0);
        self.flip_fix(t2, 0);
    }

    /// Insert vertex `vid` on edge `edge` of triangle `tid`.
    ///
    /// Splits the two triangles sharing the edge into 4 sub-triangles.
    #[expect(
        clippy::similar_names,
        reason = "standard adjacent-vertex and adjacent-triangle naming for local retriangulation"
    )]
    pub(super) fn insert_on_edge(&mut self, vid: PslgVertexId, tid: TriangleId, edge: usize) {
        let nbr_tid = self.triangles[tid.idx()].adj[edge];

        if nbr_tid == GHOST_TRIANGLE {
            // Edge is on the convex hull — only split the single triangle.
            self.insert_on_hull_edge(vid, tid, edge);
            return;
        }

        let old_t = self.triangles[tid.idx()];
        let old_n = self.triangles[nbr_tid.idx()];

        // Vertices of tid: v_opp is opposite the shared edge.
        let v_opp_t = old_t.vertices[edge];
        let (va, vb) = old_t.edge_vertices(edge); // shared edge vertices

        // Find the opposite vertex in the neighbor.
        let nbr_edge = old_n.shared_edge(tid).expect("adjacency broken");
        let v_opp_n = old_n.vertices[nbr_edge];

        // Kill old triangles.
        self.triangles[tid.idx()].alive = false;
        self.triangles[nbr_tid.idx()].alive = false;

        // Create 4 new triangles:
        //   t0: (vid, v_opp_t, va)
        //   t1: (vid, vb, v_opp_t)
        //   t2: (vid, va, v_opp_n)
        //   t3: (vid, v_opp_n, vb)
        let t0 = self.push_triangle(Triangle::new(vid, v_opp_t, va));
        let t1 = self.push_triangle(Triangle::new(vid, vb, v_opp_t));
        let t2 = self.push_triangle(Triangle::new(vid, va, v_opp_n));
        let t3 = self.push_triangle(Triangle::new(vid, v_opp_n, vb));

        // Adjacency of old_t edges (excluding the shared edge):
        let adj_t_va = old_t.adj[(edge + 2) % 3]; // edge opposite vb in old_t = v_opp_t → va
        let adj_t_vb = old_t.adj[(edge + 1) % 3]; // edge opposite va in old_t = vb → v_opp_t

        let adj_n_va = old_n.adj[(nbr_edge + 1) % 3]; // edge opposite vb in old_n
        let adj_n_vb = old_n.adj[(nbr_edge + 2) % 3]; // edge opposite va in old_n

        // t0: (vid, v_opp_t, va)
        //  edge 0 (opp vid) = v_opp_t→va → adj_t_va
        //  edge 1 (opp v_opp_t) = vid→va → t2
        //  edge 2 (opp va) = vid→v_opp_t → t1
        self.triangles[t0.idx()].adj = [adj_t_va, t2, t1];

        // t1: (vid, vb, v_opp_t)
        //  edge 0 (opp vid) = vb→v_opp_t → adj_t_vb
        //  edge 1 (opp vb) = vid→v_opp_t → t0
        //  edge 2 (opp v_opp_t) = vid→vb → t3
        self.triangles[t1.idx()].adj = [adj_t_vb, t0, t3];

        // t2: (vid, va, v_opp_n)
        //  edge 0 (opp vid) = va→v_opp_n → adj_n_va
        //  edge 1 (opp va) = vid→v_opp_n → t3
        //  edge 2 (opp v_opp_n) = vid→va → t0
        self.triangles[t2.idx()].adj = [adj_n_va, t3, t0];

        // t3: (vid, v_opp_n, vb)
        //  edge 0 (opp vid) = v_opp_n→vb → adj_n_vb
        //  edge 1 (opp v_opp_n) = vid→vb → t1
        //  edge 2 (opp vb) = vid→v_opp_n → t2
        self.triangles[t3.idx()].adj = [adj_n_vb, t1, t2];

        // Fix external neighbors.
        self.fix_adjacency(adj_t_va, tid, t0);
        self.fix_adjacency(adj_t_vb, tid, t1);
        self.fix_adjacency(adj_n_va, nbr_tid, t2);
        self.fix_adjacency(adj_n_vb, nbr_tid, t3);

        // Update vert_to_tri for all affected vertices.
        self.vert_to_tri[vid.idx()] = t0;
        self.vert_to_tri[v_opp_t.idx()] = t0;
        self.vert_to_tri[va.idx()] = t2;
        self.vert_to_tri[vb.idx()] = t1;
        self.vert_to_tri[v_opp_n.idx()] = t2;

        self.last_triangle = t0;

        // Restore Delaunay.
        self.flip_fix(t0, 0);
        self.flip_fix(t1, 0);
        self.flip_fix(t2, 0);
        self.flip_fix(t3, 0);
    }

    /// Handle the special case where the edge is on the hull boundary.
    #[expect(
        clippy::similar_names,
        reason = "standard adjacent-vertex and adjacent-triangle naming for local retriangulation"
    )]
    pub(super) fn insert_on_hull_edge(&mut self, vid: PslgVertexId, tid: TriangleId, edge: usize) {
        let old = self.triangles[tid.idx()];
        let v_opp = old.vertices[edge];
        let (va, vb) = old.edge_vertices(edge);

        self.triangles[tid.idx()].alive = false;

        // Split into 2 triangles.
        let t0 = self.push_triangle(Triangle::new(vid, v_opp, va));
        let t1 = self.push_triangle(Triangle::new(vid, vb, v_opp));

        let adj_va = old.adj[(edge + 2) % 3];
        let adj_vb = old.adj[(edge + 1) % 3];

        self.triangles[t0.idx()].adj = [adj_va, GHOST_TRIANGLE, t1];
        self.triangles[t1.idx()].adj = [adj_vb, t0, GHOST_TRIANGLE];

        self.fix_adjacency(adj_va, tid, t0);
        self.fix_adjacency(adj_vb, tid, t1);

        // Update vert_to_tri for all affected vertices.
        self.vert_to_tri[vid.idx()] = t0;
        self.vert_to_tri[v_opp.idx()] = t0;
        self.vert_to_tri[va.idx()] = t0;
        self.vert_to_tri[vb.idx()] = t1;

        self.last_triangle = t0;

        self.flip_fix(t0, 0);
        self.flip_fix(t1, 0);
    }

    /// Push a new triangle and return its ID.
    fn push_triangle(&mut self, tri: Triangle) -> TriangleId {
        let id = TriangleId::from_usize(self.triangles.len());
        self.triangles.push(tri);
        id
    }

    /// Fix external adjacency: replace `old_tid` with `new_tid` in the
    /// neighbor's adjacency list.
    fn fix_adjacency(&mut self, nbr: TriangleId, old_tid: TriangleId, new_tid: TriangleId) {
        if nbr == GHOST_TRIANGLE {
            return;
        }
        for a in &mut self.triangles[nbr.idx()].adj {
            if *a == old_tid {
                *a = new_tid;
                return;
            }
        }
    }

    /// Flip a shared internal edge and patch adjacency/constraint metadata.
    ///
    /// Rewrites the two incident triangles from `(va, vb)` diagonal to
    /// `(v_opp_t, v_opp_n)` and updates external neighbors and `vert_to_tri`.
    ///
    /// Returns the neighbor triangle ID that shared `edge` before the flip.
    #[inline]
    #[expect(
        clippy::similar_names,
        reason = "standard adjacent-vertex and adjacent-triangle naming for edge flips"
    )]
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
}
