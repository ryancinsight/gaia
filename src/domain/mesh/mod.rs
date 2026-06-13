//! # Mesh Types
//!
//! This module provides the consolidated mesh type:
//!
//! - **[`IndexedMesh`]** — the single source of truth (SSOT) surface and volume mesh,
//!   combining `VertexPool` (spatial-hash dedup), `FaceStore`, `EdgeStore`,
//!   `AttributeStore`, `Vec<Cell>`, and `GhostCell` topology.

#[allow(missing_docs)]
pub mod indexed;
#[allow(missing_docs)]
pub mod halfedge;

pub use indexed::{IndexedMesh, MeshBuilder};
pub use halfedge::{HalfEdgeMesh, with_mesh};

