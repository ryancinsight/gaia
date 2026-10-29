//! Accelerated plane-based clip and refine operations.
//!
//! # CGAL 6.1 Insight — Plane Clippers (2025)
//!
//! CGAL 6.1 reimplemented `clip()`, `split()`, and introduced
//! `refine_with_plane()` achieving 10× speedup by:
//!
//! 1. **Cached plane equations** — compute normal + offset once, reuse for all
//!    orientation tests against that plane.
//! 2. **Batch vertex classification** — classify every unique vertex against the
//!    cut plane in a single O(V) pass before processing faces, reducing
//!    redundant `orient_3d` calls from O(3F) to O(V).
//! 3. **Direct plane clipping** — split straddling faces at the plane directly,
//!    producing 1–3 sub-triangles without the overhead of full CDT
//!    co-refinement.
//!
//! These techniques prevent cascading degenerates (thin walls from repeated
//! subdivision) and complement corefinement for hybrid workflows (e.g.,
//! pre-split at symmetry planes before full Boolean).
//!
//! # Theorem — Batch Vertex Classification Equivalence
//!
//! Let V be the set of unique vertex IDs referenced by F faces, and let P be a
//! plane defined by three CCW points.  Per-face classification calls
//! `orient_3d` 3|F| times, but since vertices are shared, |V| ≤ 3|F| with
//! equality only for isolated triangles.  Batch classification calls
//! `orient_3d` exactly |V| times.  Since `orient_3d` is a pure function of its
//! arguments, the per-vertex results are identical in both approaches.  ∎
//!
//! # References
//!
//! - Sébastien Loriot, Mael Rouxel-Labbé, Jane Tournois, Ilker O. Yaz,
//!   "Polygon Mesh Processing", CGAL 6.1, 2025.

mod clip;
mod equation;
mod refine;

#[cfg(test)]
mod tests;

pub use clip::{classify_face, clip_face_by_plane};
pub use equation::{FacePlaneClass, PlaneEquation};
pub use refine::refine_faces_with_plane;
