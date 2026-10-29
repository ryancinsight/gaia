//! 3D Delaunay tetrahedralization using Bowyer-Watson.
//!
//! # Theorem 2: Empty Circumsphere Criterion (Delaunay Condition)
//!
//! **Statement**: A tetrahedral mesh is strictly Delaunay if and only if no
//! vertex in the mesh lies strictly within the circumscribing sphere of any
//! other tetrahedron in the mesh.
//!
//! **Proof sketch for Zero-Allocation Cavity Buffer**: By evaluating the empty circumsphere criterion
//! for each existing tetrahedron $`T_i`$, we construct the set $C$ of tetrahedra violating the invariant
//! with regarding to the new vertex $P$. The boundaries of $C$ form a simply-connected star-shaped
//! polyhedron. The construction of new tetrahedra connecting $P$ to the cavity boundary is independent
//! of the global mesh history, therefore the data structures tracking the cavity faces and the set of
//! surviving elements ($T \\setminus C$) can be transient and strictly localized. Because they represent
//! purely temporary geometric state, pre-allocating a persistent `HashMap` and `Vec`, and executing `.clear()`
//! between generic point insertions is mathematically identical and topologically isomorphic to
//! fresh allocations, avoiding per-insertion cavity-buffer churn. Capacity
//! growth remains possible when the caller exceeds the initial heuristic.
//!
//! The storage, vector arithmetic, and predicate decisions are generic over
//! `T: Scalar`: the exact Shewchuk predicates evaluate the caller's
//! stored-precision coordinates directly (`f32` promotes losslessly into
//! the `f64` expansion arithmetic, `f64` is the identity), so the kernel
//! honours native precision for both supported scalars (ADR 0005).

use hashbrown::{HashMap, HashSet};
use leto::geometry::Point3;

use crate::domain::core::scalar::Scalar;

mod insert;
mod kernel;
mod types;

#[cfg(test)]
mod tests;

pub use types::{Face, Tetrahedron};

/// A rigorously verifiable 3D Delaunay geometry engine.
pub struct BowyerWatson3D<T: Scalar> {
    /// All inserted vertices plus the 4 super-tetrahedron anchor vertices at the tail.
    pub vertices: Vec<Point3<T>>,
    /// Stable-index storage for tetrahedra. Empty slots are `None`.
    pub tetrahedra: Vec<Option<Tetrahedron<T>>>,
    /// Free-list of reused indices.
    free_list: Vec<usize>,

    /// Adjacency mapping: A face maps to up to 2 tetrahedron pseudo-indices.
    /// Default empty slot is `usize::MAX`.
    face_tets: HashMap<Face, [usize; 2]>,

    /// Seed index for adjacency BFS.
    last_inserted_tet: usize,

    // -- Zero-Allocation Cavity State Buffers --
    cavity_cache: HashMap<Face, usize>,
    cavity_faces: Vec<(Face, usize)>,

    // -- BFS Queues --
    bad_tets: Vec<usize>,
    visited_tets: Vec<usize>,
    visited_flags: Vec<bool>,

    // -- BFS Seed Search Cache --
    search_q: Vec<usize>,
    search_visited: HashSet<usize>,

    super_idx: usize,
}
