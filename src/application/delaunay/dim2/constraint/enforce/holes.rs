//! Hole-region triangle removal by flood-fill from seed points.

use super::Cdt;
use crate::application::delaunay::dim2::triangulation::adjacency::Adjacency;
use crate::application::delaunay::dim2::triangulation::triangle::{TriangleId, GHOST_TRIANGLE};
use crate::domain::core::scalar::Real;
use std::collections::VecDeque;

impl Cdt {
    /// Remove triangles that are inside a hole region via flood-fill.
    pub(super) fn remove_hole_triangles(&mut self, hx: Real, hy: Real) {
        // Find the triangle containing the hole seed.
        let start = {
            let first_alive = self
                .dt
                .triangles_slice()
                .iter()
                .position(|t| t.alive)
                .map(TriangleId::from_usize);

            match first_alive {
                Some(s) => s,
                None => return,
            }
        };

        let loc = self.dt.locate_point(start, hx, hy);
        let Some(
            crate::application::delaunay::dim2::triangulation::locate::Location::Inside(seed_tid)
            | crate::application::delaunay::dim2::triangulation::locate::Location::OnEdge(
                seed_tid,
                _,
            )
            | crate::application::delaunay::dim2::triangulation::locate::Location::OnVertex(
                seed_tid,
                _,
            ),
        ) = loc
        else {
            return;
        };

        // Flood-fill from the seed, stopping at constrained edges.
        //
        // # Dense Visited Buffer
        //
        // Triangle IDs are dense pool indices.  A `Vec<bool>` eliminates
        // hash overhead of `HashSet<TriangleId>` and gives cache-friendly
        // sequential access during the adjacency-patching pass.
        let pool_len = self.dt.triangles_slice().len();
        let mut queue = VecDeque::with_capacity(32);
        let mut visited = vec![false; pool_len];
        queue.push_back(seed_tid);
        visited[seed_tid.idx()] = true;

        while let Some(tid) = queue.pop_front() {
            let tri = *self.dt.triangle(tid);
            self.dt.triangle_mut(tid).alive = false;

            for edge in 0..3 {
                if tri.constrained[edge] {
                    continue; // Don't cross constraint edges.
                }
                let nbr = tri.adj[edge];
                if nbr != GHOST_TRIANGLE && !visited[nbr.idx()] && self.dt.triangle(nbr).alive {
                    visited[nbr.idx()] = true;
                    queue.push_back(nbr);
                }
            }
        }

        // Patch adjacency: alive triangles on the hole boundary still have
        // adjacency links pointing to the now-dead hole triangles.  Set
        // those links to GHOST_TRIANGLE to maintain the symmetric-adjacency
        // invariant.
        //
        // Theorem — Adjacency Consistency After Hole Removal
        //
        // Statement: After flood-fill hole removal, setting adj[e] =
        // GHOST_TRIANGLE for every alive triangle whose neighbor across
        // edge e was killed restores the symmetric-adjacency invariant.
        //
        // Proof: Let T_alive be an alive boundary triangle and T_dead its
        // killed neighbor across edge e.  Before removal, T_alive.adj[e] =
        // T_dead and T_dead.adj[e'] = T_alive for some e'.  After killing
        // T_dead, T_dead is no longer alive so T_alive.adj[e] is a dangling
        // reference.  Setting T_alive.adj[e] = GHOST_TRIANGLE makes the
        // boundary edge a hull edge, consistent with the invariant that
        // GHOST_TRIANGLE neighbours are not checked for symmetry.  ∎
        for (i, &is_dead) in visited.iter().enumerate() {
            if !is_dead {
                continue;
            }
            let tid = TriangleId::from_usize(i);
            let tri = *self.dt.triangle(tid);
            for edge in 0..3 {
                let nbr = tri.adj[edge];
                if nbr == GHOST_TRIANGLE || visited[nbr.idx()] {
                    continue;
                }
                // nbr is alive — patch its adjacency back to GHOST.
                if let Some(back_edge) = Adjacency::find_edge(self.dt.triangles_slice(), nbr, tid) {
                    Adjacency::link_one(self.dt.triangles_mut(), nbr, back_edge, GHOST_TRIANGLE);
                }
            }
        }
    }
}
