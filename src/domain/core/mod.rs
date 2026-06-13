//! Core types: scalar abstraction, indices, errors, constants.
//!
//! Single source of truth for all fundamental types used across the crate.

pub mod constants;
pub mod error;
pub mod index;
pub mod scalar;

pub use error::{require, Error, ErrorContext, MeshError, MeshResult, NurbsKind, Result};
pub use index::{
    EdgeId, FaceId, FaceKey, HalfEdgeKey, PatchKey, RegionId, RegionKey, VertexId, VertexKey,
};
pub use scalar::{Real, Scalar};
