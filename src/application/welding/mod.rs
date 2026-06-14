//! Vertex welding and snapping.
//!
//! Provides spatial hash-based vertex deduplication (also inside `VertexPool`)
//! and additional snap-to-grid / snap-to-vertex utilities for post-processing.
//!
//! ## Primary API
//!
//! - **[`SnappingGrid`]** — unified 26-neighbor grid for coordinate snapping
//!   AND vertex deduplication in one structure.  The grid is mesh-agnostic and
//!   returns stable welded indices for callers to map into their mesh storage.
//!
//! ## Additional API
//!
//! - [`SpatialHashGrid`] — point-set dedup only (no coordinate snapping).
//! - [`MeshWelder`] — batch vertex welding for existing `IndexedMesh`es.

pub mod snap;
pub mod spatial_hash;
pub mod welder;

pub use snap::{GridCell, SnappingGrid};
pub use spatial_hash::SpatialHashGrid;
pub use welder::MeshWelder;
