//! Incremental Bowyer-Watson Delaunay triangulation.
//!
//! # Theorem — Bowyer-Watson Correctness
//!
//! **Statement** (Bowyer 1981, Watson 1981): Given $DT(P)$ and a new point
//! $p$, the algorithm identifies and removes the *bad cavity* (triangles
//! whose circumcircle contains $p$), then fan-connects $p$ to the cavity
//! boundary.  The result is $DT(P \cup \{p\})$.
//!
//! **Proof sketch**: By the empty-circumcircle characterisation, exactly
//! those triangles whose circumcircle contains $p$ are invalidated.
//! Fan-connecting $p$ to the cavity boundary creates triangles whose
//! circumcircles contain only $p$ and the two edge endpoints — hence
//! Delaunay.
//!
//! # Complexity
//!
//! Expected $O(n \log n)$ (uniformly distributed); worst-case $O(n^2)$.

use super::triangle::{Triangle, TriangleId};
use crate::application::delaunay::dim2::pslg::vertex::{PslgVertex, PslgVertexId};

/// The Delaunay triangulation data structure.
///
/// Stores vertices and triangles as flat arrays with explicit adjacency.
/// Supports incremental point insertion via the Bowyer-Watson algorithm.
///
/// # Vertex-to-Triangle Adjacency
///
/// A `vert_to_tri` map is maintained: for each vertex `v`, `vert_to_tri[v]`
/// contains one incident alive triangle.  This enables O(deg(v)) edge
/// lookups instead of O(T) full scans — a **critical** performance
/// improvement for constraint enforcement where `edge_exists()`,
/// `collect_crossing_edges()`, and `mark_edge_constrained()` are called
/// per constraint segment.
///
/// # Example
///
/// ```rust,no_run
/// // DelaunayTriangulation is an internal type used by the CDT pipeline.
/// // Access it through the public Cdt and Pslg APIs in gaia::application::delaunay.
/// let _ = ();
/// ```
pub struct DelaunayTriangulation {
    /// Vertex pool (includes 3 super-triangle vertices at the end).
    pub(crate) vertices: Vec<PslgVertex>,
    /// Triangle pool (tombstoned entries have `alive == false`).
    pub(crate) triangles: Vec<Triangle>,
    /// Index of the last triangle used for point-location (walk hint).
    last_triangle: TriangleId,
    /// Number of real (non-super-triangle) vertices.
    pub(crate) num_real_vertices: usize,
    /// IDs of the three super-triangle vertices.
    pub(crate) super_verts: [PslgVertexId; 3],
    /// Vertex → one incident alive triangle.  Enables O(deg(v)) local
    /// traversal for edge queries instead of O(T) global scans.
    pub(crate) vert_to_tri: Vec<TriangleId>,
    /// Epoch stamps for O(1) amortised cycle detection in the Lawson walk.
    /// `locate_epoch[tid]` holds the generation when triangle `tid` was last
    /// visited.
    locate_epoch: Vec<u32>,
    /// Generation counter, incremented once per `locate_point` call.
    locate_gen: u32,
}

// Child slices add `impl DelaunayTriangulation` blocks: construction,
// insertion, edge-flipping, and read-only queries plus compaction.

mod construct;
mod flips;
mod insert;
mod query;
