//! `Cdt` read/write accessors and incremental constraint bookkeeping.

use super::Cdt;
use crate::application::delaunay::core::canonical_edge;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::dim2::triangulation::bowyer_watson::DelaunayTriangulation;
use hashbrown::HashSet;

impl Cdt {
    // ── Public API ──────────────────────────────────────────────────

    /// Access the underlying triangulation.
    #[must_use]
    pub fn triangulation(&self) -> &DelaunayTriangulation {
        &self.dt
    }

    /// Consume the CDT, returning the underlying triangulation.
    #[must_use]
    pub fn into_triangulation(self) -> DelaunayTriangulation {
        self.dt
    }

    /// Mutable access to the underlying triangulation.
    pub(crate) fn triangulation_mut(&mut self) -> &mut DelaunayTriangulation {
        &mut self.dt
    }

    /// Check if an edge is constrained.
    #[must_use]
    pub fn is_constrained(&self, a: PslgVertexId, b: PslgVertexId) -> bool {
        let canonical = canonical_edge(a, b);
        self.constrained_edges.contains(&canonical)
    }

    /// The set of all constrained edges.
    #[must_use]
    pub fn constrained_edges(&self) -> &HashSet<(PslgVertexId, PslgVertexId)> {
        &self.constrained_edges
    }

    /// Insert a new constraint edge (used by refinement after segment splits).
    pub(crate) fn add_constraint(&mut self, a: PslgVertexId, b: PslgVertexId) {
        self.enforce_constraint(a, b);
    }

    /// Remove a constraint edge (used when splitting a segment).
    ///
    /// Uses vertex-star walk: O(deg(a)) instead of O(T).
    pub(crate) fn remove_constraint(&mut self, a: PslgVertexId, b: PslgVertexId) {
        let canonical = canonical_edge(a, b);
        self.constrained_edges.remove(&canonical);
        // Unmark the edge in the triangulation (it may no longer exist).
        self.set_edge_constraint_flag(a, b, false);
    }
}
