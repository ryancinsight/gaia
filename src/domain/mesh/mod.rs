//! # Mesh Types
//!
//! This module provides the consolidated mesh type:
//!
//! - **[`IndexedMesh`]** — the single source of truth (SSOT) surface and volume mesh,
//!   combining `VertexPool` (spatial-hash dedup), `FaceStore`, `EdgeStore`,
//!   `AttributeStore`, `Vec<Cell>`, and `GhostCell` topology.

pub mod halfedge;
pub mod indexed;
pub mod tetrahedral;

pub use halfedge::{with_mesh, HalfEdgeMesh};
pub use indexed::{IndexedMesh, MeshBuilder};
pub use tetrahedral::TetrahedralMeshBuilder;
