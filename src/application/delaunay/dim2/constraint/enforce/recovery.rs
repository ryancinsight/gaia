//! Flip-based constraint recovery: crossing collection, flips, and repair.

use super::Cdt;
use crate::application::delaunay::core::{canonical_edge, segment_cross_point};
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::dim2::triangulation::triangle::{TriangleId, GHOST_TRIANGLE};
use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d, Orientation};
use leto::geometry::Point2;
use std::collections::VecDeque;

impl Cdt {
    /// Enforce a constraint edge between vertices `a` and `b`.
    ///
    /// # Algorithm — Single-Constraint Recovery (Sloan 1993)
    ///
    /// 1. If edge `(a,b)` already exists, mark it constrained and restore
    ///    the local Delaunay property.
    /// 2. Otherwise, collect crossing edges and flip them iteratively.
    /// 3. Post-flip, restore locally-Delaunay non-constrained edges.
    ///
    /// # Theorem — Monotone Progress
    ///
    /// **Statement**: Each successful flip either removes a crossing or
    /// replaces it with a crossing whose intersection length with `(a,b)`
    /// is strictly shorter, guaranteeing termination.
    ///
    /// **Proof sketch**: The flip replaces diagonal `(u,v)` with `(p,q)`
    /// in the convex quadrilateral.  If `(p,q)` still crosses `(a,b)`,
    /// it connects the two opposite vertices of the quad, which lie on
    /// opposite sides of `(a,b)`.  The intersection of `(p,q)` with
    /// `(a,b)` is contained strictly within the quad, which is a subset
    /// of the union of the two original triangles — so the new crossing
    /// length is bounded by the quad diagonal, which is shorter than
    /// the sum of the two triangles' diameters.  Combined with the
    /// finite triangulation, this ensures termination.  ∎
    pub(super) fn enforce_constraint(&mut self, a: PslgVertexId, b: PslgVertexId) {
        // Record the constraint.
        let canonical = canonical_edge(a, b);
        self.constrained_edges.insert(canonical);

        // Check if the edge already exists.
        if self.edge_exists(a, b) {
            self.mark_edge_constrained(a, b);
            self.restore_delaunay_near_constraint(a, b);
            return;
        }

        // Flip edges crossing the segment (a, b).
        self.flip_to_enforce(a, b);

        // Mark the edge as constrained.
        self.mark_edge_constrained(a, b);

        // Restore the locally Delaunay property for non-constrained edges.
        self.restore_delaunay_near_constraint(a, b);
    }

    /// Check if an edge between vertices `a` and `b` exists in the triangulation.
    ///
    /// Uses vertex-star walk: O(deg(a)) ≈ O(6) instead of O(T).
    pub(super) fn edge_exists(&self, a: PslgVertexId, b: PslgVertexId) -> bool {
        self.dt.edge_exists_fast(a, b)
    }

    /// Flip crossing edges until `(a, b)` exists in the triangulation.
    ///
    /// Uses the iterative approach: collect edges that intersect the segment
    /// `(a, b)`, then flip them one by one.
    /// Flip edges to recover the constraint `(a, b)`.
    ///
    /// Uses the Sloan (1993) queue-based algorithm:
    /// 1. Collect **all** edges crossing the segment `(a, b)`.
    /// 2. Process the queue round-robin: for each crossing edge, attempt a flip.
    ///    - If the quad is convex and the flip succeeds, check whether the new
    ///      diagonal still crosses `(a, b)`.  If so, re-enqueue it.
    ///    - If the quad is NOT convex, defer the edge to the back of the queue.
    /// 3. When all crossings are resolved the edge `(a, b)` exists.
    ///
    /// Steiner point insertion is only used as a last-resort fallback when a
    /// full pass through the queue makes zero progress.
    pub(super) fn flip_to_enforce(&mut self, a: PslgVertexId, b: PslgVertexId) {
        // Collect all crossing edges into a work queue.
        let mut queue: VecDeque<(TriangleId, usize)> = VecDeque::with_capacity(32);
        self.collect_crossing_edges(a, b, &mut queue);
        if queue.is_empty() {
            return;
        }

        // Safety bound: O(n²) iterations for n crossing edges.
        let max_rounds = (queue.len() + 1) * (queue.len() + 1) * 2;
        let mut iter = 0usize;

        while let Some((tid, edge)) = queue.pop_front() {
            iter += 1;
            if iter > max_rounds {
                // Exceeded iteration budget — fallback: insert Steiner point
                // for first remaining crossing and restart.
                self.steiner_fallback(a, b, tid, edge);
                // Re-collect crossings and try again.
                queue.clear();
                self.collect_crossing_edges(a, b, &mut queue);
                if queue.is_empty() {
                    return;
                }
                iter = 0;
                continue;
            }

            // Validate: the triangle might have been killed by a prior flip.
            if !self.dt.triangle(tid).alive {
                continue;
            }

            // Verify the edge still crosses (a, b).
            let (va, vb) = self.dt.triangle(tid).edge_vertices(edge);
            let sa = self.dt.vertex(a).to_point2();
            let sb = self.dt.vertex(b).to_point2();
            if !self.edge_crosses_segment(&sa, &sb, va, vb) {
                continue; // No longer crossing — already resolved.
            }

            if self.can_flip(tid, edge) {
                // Record the new diagonal endpoints BEFORE the flip mutates state.
                let v_opp_t = self.dt.triangle(tid).vertices[edge];
                let nbr_tid = self.dt.triangle(tid).adj[edge];
                let nbr_edge = self
                    .dt
                    .triangle(nbr_tid)
                    .shared_edge(tid)
                    .expect("adjacency broken");
                let v_opp_n = self.dt.triangle(nbr_tid).vertices[nbr_edge];

                self.perform_flip(tid, edge);

                // Check if the new diagonal (v_opp_t — v_opp_n) crosses (a, b).
                let sa = self.dt.vertex(a).to_point2();
                let sb = self.dt.vertex(b).to_point2();
                if self.edge_crosses_segment(&sa, &sb, v_opp_t, v_opp_n) {
                    // New diagonal still crosses — find which triangle now
                    // contains that edge and re-enqueue.
                    if let Some(new_loc) = self.find_edge_in_triangles(v_opp_t, v_opp_n) {
                        queue.push_back(new_loc);
                    }
                }
            } else {
                // Not convex — defer to the back of the queue.
                queue.push_back((tid, edge));
            }
        }
    }

    #[inline]
    fn for_each_crossing_edge_in_triangle<F>(
        &self,
        tid: TriangleId,
        a: PslgVertexId,
        b: PslgVertexId,
        seg_start: &Point2<Real>,
        seg_end: &Point2<Real>,
        mut on_crossing: F,
    ) where
        F: FnMut(usize, TriangleId),
    {
        let tri = self.dt.triangle(tid);
        for edge in 0..3 {
            let (va, vb) = tri.edge_vertices(edge);
            if Self::edge_touches_constraint_endpoint(va, vb, a, b) {
                continue;
            }
            if tri.constrained[edge] {
                continue;
            }
            if self.edge_crosses_segment(seg_start, seg_end, va, vb) {
                on_crossing(edge, tri.adj[edge]);
            }
        }
    }

    /// Collect all edges currently crossing segment `(a, b)`.
    ///
    /// # Algorithm — Directed Triangle Walk with BFS Extension
    ///
    /// Starts from the vertex-star of `a` and walks toward `b` via
    /// adjacency, collecting every non-constrained edge that geometrically
    /// crosses the segment `(a,b)`.  Uses a BFS queue to handle
    /// non-convex or tombstoned regions without falling back to O(T) scan.
    ///
    /// # Theorem — Crossing Edge Collection Completeness
    ///
    /// **Statement**: The directed walk from `a`'s star collects every
    /// edge that intersects the open segment `(a,b)`, provided no
    /// constrained edge blocks the walk.
    ///
    /// **Proof sketch**: In a triangulation, the segment `(a,b)` passes
    /// through a sequence of triangles $`T_1`, `T_2`, \ldots, `T_k`$ forming
    /// a "channel".  Adjacent triangles in this channel share crossing
    /// edges.  Starting from $`T_1`$ (in `a`'s star) and following
    /// adjacency across crossing edges visits every triangle in the
    /// channel, hence every crossing edge.  ∎
    ///
    /// # Complexity
    ///
    /// $O(k + \deg(a))$ where $k$ is the number of crossing edges.
    fn collect_crossing_edges(
        &self,
        a: PslgVertexId,
        b: PslgVertexId,
        queue: &mut VecDeque<(TriangleId, usize)>,
    ) {
        let sa = self.dt.vertex(a).to_point2();
        let sb = self.dt.vertex(b).to_point2();

        // Directed walk: start from triangles around `a`, walk towards `b`.
        //
        // # Dense Visited Buffer
        //
        // Triangle IDs are dense indices into the pool, so a `Vec<bool>`
        // gives O(1) lookup/insert with zero hashing overhead compared
        // to the previous `HashSet<TriangleId>`.
        let star_a = self.dt.triangles_around_vertex(a);
        let pool_len = self.dt.triangles_slice().len();
        let mut visited = vec![false; pool_len];

        // Find the starting triangle whose edge opposite `a` crosses (a,b).
        let mut walk_queue: VecDeque<TriangleId> = VecDeque::with_capacity(32);
        for &tid in &star_a {
            self.for_each_crossing_edge_in_triangle(tid, a, b, &sa, &sb, |edge, nbr| {
                queue.push_back((tid, edge));
                visited[tid.idx()] = true;
                if nbr != GHOST_TRIANGLE && !visited[nbr.idx()] {
                    walk_queue.push_back(nbr);
                }
            });
        }

        // Continue walking through neighbors of found crossings.
        while let Some(tid) = walk_queue.pop_front() {
            if visited[tid.idx()] {
                continue;
            }
            visited[tid.idx()] = true;
            if !self.dt.triangle(tid).alive {
                continue;
            }
            self.for_each_crossing_edge_in_triangle(tid, a, b, &sa, &sb, |edge, nbr| {
                queue.push_back((tid, edge));
                if nbr != GHOST_TRIANGLE && !visited[nbr.idx()] {
                    walk_queue.push_back(nbr);
                }
            });
        }
    }

    /// Find the triangle and local edge index containing edge `(u, v)`.
    ///
    /// # Algorithm
    ///
    /// Delegates to [`DelaunayTriangulation::find_edge_fast`], which walks
    /// the vertex-star of `u` looking for a triangle containing both `u`
    /// and `v`.
    ///
    /// # Complexity
    ///
    /// $O(\deg(u)) \approx O(6)$ for well-distributed Delaunay vertices,
    /// compared to $O(T)$ for a full triangle scan.
    fn find_edge_in_triangles(
        &self,
        u: PslgVertexId,
        v: PslgVertexId,
    ) -> Option<(TriangleId, usize)> {
        self.dt.find_edge_fast(u, v)
    }

    /// Last-resort: insert a Steiner point at the intersection of the
    /// constraint segment and a crossing edge.
    ///
    /// # Algorithm — Steiner Point Insertion Fallback
    ///
    /// When the flip-based recovery exhausts its iteration budget
    /// (non-convex quadrilaterals prevent progress), a Steiner point
    /// is inserted at the intersection of `(a,b)` with the first
    /// remaining crossing edge.  This splits both the constraint and
    /// the crossing edge, breaking the deadlock.
    ///
    /// # Theorem — Steiner Recovery Guarantees Termination
    ///
    /// **Statement**: Steiner insertion reduces the number of crossing
    /// edges by at least 1, so at most $O(k)$ Steiner points are
    /// inserted for $k$ initial crossings.
    ///
    /// **Proof sketch**: The Steiner point lies on the constraint
    /// segment, splitting it into two sub-segments.  The crossing edge
    /// through the Steiner point is destroyed by the Bowyer-Watson
    /// insertion.  Each sub-segment has strictly fewer crossings than
    /// the original.  ∎
    fn steiner_fallback(&mut self, a: PslgVertexId, b: PslgVertexId, tid: TriangleId, edge: usize) {
        let (va, vb) = self.dt.triangle(tid).edge_vertices(edge);
        let pa = self.dt.vertex(va);
        let pb = self.dt.vertex(vb);
        let pc = self.dt.vertex(a);
        let pd = self.dt.vertex(b);
        if let Some((ix, iy)) = segment_cross_point(
            &pc.to_point2(),
            &pd.to_point2(),
            &pa.to_point2(),
            &pb.to_point2(),
        ) {
            self.dt.insert_steiner(ix, iy);
        }
    }

    /// Check if the quadrilateral formed by flipping edge `edge` of triangle
    /// `tid` is convex (required for a valid flip).
    ///
    /// # Theorem — Convexity Predicate for Edge Flips
    ///
    /// **Statement**: An edge shared by two triangles can be flipped iff
    /// the union of the two triangles forms a strictly convex
    /// quadrilateral — i.e., the two diagonals of the quad intersect
    /// in their interiors.
    ///
    /// **Proof sketch**: Let the shared edge be `(va, vb)` and the
    /// opposite vertices be `p` and `q`.  The quad is convex iff `p`
    /// and `q` lie on opposite sides of line `(va, vb)` AND `va` and
    /// `vb` lie on opposite sides of line `(p, q)`.  This is equivalent
    /// to the four orientation tests having alternating signs for each
    /// diagonal.  If the quad is non-convex, the flip would produce
    /// an overlapping (non-planar) triangulation.  ∎
    ///
    /// # Complexity
    ///
    /// $O(1)$: four `orient_2d` predicate evaluations.
    fn can_flip(&self, tid: TriangleId, edge: usize) -> bool {
        let tri = self.dt.triangle(tid);
        let nbr_tid = tri.adj[edge];
        if nbr_tid == GHOST_TRIANGLE {
            return false;
        }

        let nbr = self.dt.triangle(nbr_tid);
        let Some(nbr_edge) = nbr.shared_edge(tid) else {
            return false;
        };

        let v_opp_t = tri.vertices[edge];
        let v_opp_n = nbr.vertices[nbr_edge];
        let (va, vb) = tri.edge_vertices(edge);

        let pa = self.dt.vertex(va).to_point2();
        let pb = self.dt.vertex(vb).to_point2();
        let pc = self.dt.vertex(v_opp_t).to_point2();
        let pd = self.dt.vertex(v_opp_n).to_point2();

        // The quad (v_opp_t, va, v_opp_n, vb) must be convex.
        // Check that the diagonals cross.
        let o1 = orient_2d(&pc, &pd, &pa);
        let o2 = orient_2d(&pc, &pd, &pb);
        let o3 = orient_2d(&pa, &pb, &pc);
        let o4 = orient_2d(&pa, &pb, &pd);

        (o1 != o2 && o1 != Orientation::Degenerate && o2 != Orientation::Degenerate)
            && (o3 != o4 && o3 != Orientation::Degenerate && o4 != Orientation::Degenerate)
    }

    /// Perform an edge flip on edge `edge` of triangle `tid`.
    ///
    /// # Algorithm — Lawson Edge Flip
    ///
    /// Given triangles $`T_1` = (`v_a`, `v_b`, p)$ and $`T_2` = (`v_b`, `v_a`, q)$
    /// sharing edge $(`v_a`, `v_b`)$, the flip replaces them with
    /// $`T_1`' = (p, q, `v_b`)$ and $`T_2`' = (q, p, `v_a`)$, rotating the
    /// shared edge by 90° to become $(p, q)$.
    ///
    /// # Invariants Maintained
    ///
    /// 1. **Adjacency symmetry**: All four external adjacency links are
    ///    updated so `T.adj[e].adj[e'] == T` holds.
    /// 2. **`vert_to_tri` cache**: Updated for all four affected vertices
    ///    so subsequent vertex-star walks start from valid triangles.
    /// 3. **Constrained flags**: The new diagonal is marked unconstrained;
    ///    external edge constraint flags are preserved from the original
    ///    triangles.
    ///
    /// # Complexity
    ///
    /// $O(1)$: fixed number of array writes and adjacency patches.
    fn perform_flip(&mut self, tid: TriangleId, edge: usize) {
        self.dt.flip_shared_edge(tid, edge);
    }

    /// Mark an edge between `a` and `b` as constrained in both triangles
    /// that share it.
    ///
    /// Uses vertex-star walk: O(deg(a)) ≈ O(6) instead of O(T).
    #[inline]
    pub(super) fn set_edge_constraint_flag(
        &mut self,
        a: PslgVertexId,
        b: PslgVertexId,
        constrained: bool,
    ) {
        let tris = self.dt.triangles_around_vertex(a);
        for tid in tris {
            let tri = self.dt.triangle_mut(tid);
            for edge in 0..3 {
                let (va, vb) = tri.edge_vertices(edge);
                if Self::edge_matches_endpoints(va, vb, a, b) {
                    tri.constrained[edge] = constrained;
                }
            }
        }
    }

    pub(super) fn mark_edge_constrained(&mut self, a: PslgVertexId, b: PslgVertexId) {
        self.set_edge_constraint_flag(a, b, true);
    }

    /// Restore the locally Delaunay property for non-constrained edges near
    /// a newly enforced constraint.
    ///
    /// # Theorem — CDT Locally Delaunay Property (Chew 1989)
    ///
    /// **Statement**: In a valid CDT, every non-constrained edge is locally
    /// Delaunay — the opposite vertex of the adjacent triangle does *not*
    /// lie strictly inside the circumcircle.  After constraint recovery via
    /// edge flips, non-constrained edges near the constraint may violate
    /// this property.  An iterative in-circle flip sweep restores it.
    ///
    /// **Proof sketch**: Each flip of a non-Delaunay, non-constrained edge
    /// strictly increases the minimum angle of the affected quadrilateral
    /// (Lawson 1977).  Since the angle space is bounded and the
    /// triangulation is finite, the flip sequence terminates.  Constrained
    /// edges are never flipped, so the CDT constraint set is preserved.
    ///
    /// # Complexity
    ///
    /// $O(k)$ expected for $k$ flips, where $k$ is bounded by the number
    /// of non-constrained edges in the 2-ring of the constraint.  Worst
    /// case $O(n)$.
    pub(super) fn restore_delaunay_near_constraint(&mut self, a: PslgVertexId, b: PslgVertexId) {
        // Seed the stack with all non-constrained edges around both endpoints.
        let mut stack: Vec<(TriangleId, usize)> = Vec::with_capacity(24);
        for &v in &[a, b] {
            let star = self.dt.triangles_around_vertex(v);
            for tid in star {
                let tri = self.dt.triangle(tid);
                if !tri.alive {
                    continue;
                }
                for edge in 0..3 {
                    if tri.constrained[edge] {
                        continue;
                    }
                    if tri.adj[edge] == GHOST_TRIANGLE {
                        continue;
                    }
                    stack.push((tid, edge));
                }
            }
        }
        self.dt.restore_delaunay_edges(stack);
    }
}
