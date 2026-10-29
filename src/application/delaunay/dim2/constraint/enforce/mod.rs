//! Constraint edge enforcement — turning a Delaunay triangulation into a CDT.
//!
//! # Algorithm — Edge-Flip Constraint Recovery
//!
//! Given a Delaunay triangulation and a constraint segment `(a, b)`:
//!
//! 1. If the edge `(a, b)` already exists in the triangulation, mark it as
//!    constrained and return.
//! 2. Otherwise, collect all edges that cross the segment `(a, b)`.
//! 3. Iteratively flip crossing edges.  After each flip, the new diagonal
//!    either no longer crosses `(a, b)` (progress) or it crosses but with a
//!    shorter intersection (bounded by the geometry).
//! 4. When the flipping sequence terminates, the edge `(a, b)` exists in the
//!    triangulation.  Mark it as constrained.
//!
//! # Theorem — Constraint Recovery Termination
//!
//! **Statement**: For a set of non-crossing constraint segments, the iterative
//! flip-based constraint recovery terminates in $O(n^2)$ flips in the worst
//! case, and $O(n)$ flips on average for well-distributed inputs.
//!
//! **Proof sketch**: Each flip replaces a crossing edge with a non-crossing
//! one relative to the constraint segment.  Since the number of possible
//! triangulations is finite and we make monotone progress (the set of
//! crossing edges shrinks), the process terminates.
//!
//! # Hole Removal
//!
//! After all constraints are enforced, triangles inside hole regions (identified
//! by flood-fill from hole seed points) are removed.

use hashbrown::HashSet;

use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::application::delaunay::dim2::triangulation::bowyer_watson::DelaunayTriangulation;

/// Constrained Delaunay Triangulation.
///
/// Wraps a [`DelaunayTriangulation`] with enforced constraint edges from a PSLG.
///
/// # Theorem — CDT Uniqueness
///
/// **Statement**: Given a valid PSLG (no crossing segments), the CDT is unique
/// up to co-circular degeneracies.  That is, if no four input points are
/// co-circular, the CDT is the unique triangulation that:
/// 1. Contains all constraint segments as edges.
/// 2. Maximises the minimum angle among all such triangulations.
///
/// **Proof sketch**: Among all triangulations containing the constraints, the
/// CDT is obtained by making every non-constrained edge locally Delaunay.
/// The Delaunay criterion (maximize minimum angle) has a unique optimum when
/// no four points are co-circular.
///
/// # Example
///
/// ```rust,no_run
/// use gaia::application::delaunay::{Pslg, Cdt};
///
/// let mut pslg = Pslg::new();
/// let a = pslg.add_vertex(0.0, 0.0);
/// let b = pslg.add_vertex(1.0, 0.0);
/// let c = pslg.add_vertex(0.5, 1.0);
/// pslg.add_segment(a, b);
/// pslg.add_segment(b, c);
/// pslg.add_segment(c, a);
///
/// let cdt = Cdt::from_pslg(&pslg);
/// assert!(cdt.triangulation().is_delaunay());
/// ```
pub struct Cdt {
    /// The underlying triangulation.
    dt: DelaunayTriangulation,
    /// The set of constrained edges (canonical vertex pairs).
    constrained_edges: HashSet<(PslgVertexId, PslgVertexId)>,
}

mod access;
mod construct;
mod edge_tests;
mod holes;
mod recovery;
