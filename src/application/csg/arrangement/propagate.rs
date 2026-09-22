//! Seam vertex propagation for arrangement CSG.
//!
//! After intersection detection produces snap-segments (the Steiner points
//! along mesh–mesh intersection curves), adjacent faces that share edges
//! with intersected faces must also receive those vertices.  Without this
//! step, the mesh has **T-junctions**: a vertex sits on an edge of an
//! adjacent triangle but is not topologically connected to it, breaking
//! the manifold property.
//!
//! This module provides two propagation strategies:
//!
//! 1. [`propagate_seam_vertices`] — For general co-refinement: examines
//!    each intersected face's snap-segments and injects Steiner points into
//!    neighbouring faces that share edges with the intersected face.
//!
//! 2. [`inject_cap_seam_into_barrels`] — For coplanar dispatch: injects
//!    boundary vertices of resolved coplanar groups into adjacent
//!    non-coplanar ("barrel") faces, preventing T-junctions at the
//!    coplanar/non-coplanar boundary.
//!
//! ## Algorithm — Edge-Adjacent Propagation
//!
//! For each snap-segment endpoint `v`, identify the edge `e = (a, b)` of
//! the target face that contains `v` (within collinearity tolerance).
//! Then inject a zero-length sub-interval `[v, v]` into `e`'s segment list,
//! which forces the downstream CDT to include `v` as a constrained vertex.
//!
//! ## Theorem — Propagation Completeness
//!
//! If every snap-segment endpoint `v` that lies on a shared edge `e` is
//! propagated to all faces incident to `e`, then after CDT co-refinement,
//! no T-junctions remain at shared edges.
//!
//! *Proof.*  A T-junction at edge `e` requires a vertex `v` on `e` that is
//! in the refined triangulation of one face but not the other.  Since
//! `v` was added as a constrained point to the CDT of every face containing
//! `e`, the CDT includes `v` as a vertex in all triangulations.  Therefore
//! no T-junction can exist.  ∎
//!
//! ## References
//!
//! - Shewchuk, J. R. (1996). "Triangle: Engineering a 2D quality mesh
//!   generator."  Provides CDT guarantees used by propagation.
//!
//! ## Layout
//!
//! This file is the facade: the implementation is split by concern into
//! `tolerances` (the named thresholds and the question each answers),
//! `adjacency` (the inline-two/rest-heap edge→face map), `seam` (the
//! seam-vertex pass and its fixed-point wrapper), and `barrels` (the
//! coplanar cap-seam pass). The file stays a `propagate.rs` rather than a
//! `propagate/mod.rs` so the manifest-passthrough guard sees only
//! re-exports here.

mod adjacency;
mod barrels;
mod seam;
mod tolerances;

pub use barrels::inject_cap_seam_into_barrels;
pub use seam::propagate_seam_vertices;
pub(crate) use seam::propagate_seam_vertices_until_stable;

#[cfg(test)]
mod tests;
