//! Error taxonomy for mesh and geometry operations.
//!
//! # Dependency contract
//!
//! This module lives in `domain::core` and **must not** import from the
//! `application` layer. Application-layer error types (e.g.
//! `PslgValidationError`) are owned by the application layer and are
//! never pulled into this enum. Application code converts its own errors
//! into `gaia::Error::InvalidInput` or returns its own `Result` type.


use thiserror::Error as ThisError;

use crate::domain::core::index::{EdgeId, FaceId, VertexId};
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::nurbs::curve::CurveError;
use crate::domain::geometry::nurbs::knot::KnotError;
use crate::domain::geometry::nurbs::surface::SurfaceError;
use crate::domain::geometry::primitives::PrimitiveError;

/// Alias for `std::result::Result<T, MeshError>`.
pub type MeshResult<T> = std::result::Result<T, MeshError>;

/// Comprehensive error type covering all mesh operations.
#[derive(ThisError, Debug)]
pub enum MeshError {
    // ── Topology ──────────────────────────────────────────────
    /// A face references a vertex that does not exist.
    #[error("face {face} references invalid vertex {vertex}")]
    InvalidVertexRef {
        /// The face containing the bad reference.
        face: FaceId,
        /// The invalid vertex index.
        vertex: VertexId,
    },

    /// An edge is non-manifold (shared by != 2 faces).
    #[error("edge {edge} is non-manifold: shared by {count} faces (expected 2)")]
    NonManifoldEdge {
        /// The problematic edge.
        edge: EdgeId,
        /// Actual face count.
        count: usize,
    },

    /// A vertex is non-manifold (fan not a single disk).
    #[error("vertex {vertex} is non-manifold")]
    NonManifoldVertex {
        /// The problematic vertex.
        vertex: VertexId,
    },

    /// Mesh has inconsistent winding order.
    #[error("inconsistent winding at face {face}")]
    InconsistentWinding {
        /// The face with wrong orientation.
        face: FaceId,
    },

    /// Mesh is not connected (multiple components).
    #[error("mesh has {count} disconnected components (expected 1)")]
    Disconnected {
        /// Number of components.
        count: usize,
    },

    // ── Geometry ──────────────────────────────────────────────
    /// Degenerate face (zero area).
    #[error("face {face} is degenerate (area ≈ 0)")]
    DegenerateFace {
        /// The zero-area face.
        face: FaceId,
    },

    /// Self-intersection detected.
    #[error("self-intersection between faces {a} and {b}")]
    SelfIntersection {
        /// First intersecting face.
        a: FaceId,
        /// Second intersecting face.
        b: FaceId,
    },

    /// A coordinate is NaN or Inf.
    #[error("invalid coordinate at vertex {vertex}: {point:?}")]
    InvalidCoordinate {
        /// Vertex with bad coordinates.
        vertex: VertexId,
        /// The bad point.
        point: Point3r,
    },

    // ── Watertight ────────────────────────────────────────────
    /// Mesh is not watertight (boundary edges exist).
    #[error("mesh has {count} boundary edges — not watertight")]
    NotWatertight {
        /// Number of boundary edges.
        count: usize,
    },

    /// Gap detected between regions.
    #[error("gap of width {width:.6} mm between regions")]
    GapDetected {
        /// Gap width in model units.
        width: f64,
    },

    // ── CSG ───────────────────────────────────────────────────
    /// BSP tree construction failed.
    #[error("BSP construction failed: {reason}")]
    BspError {
        /// Reason for failure.
        reason: String,
    },

    /// Boolean operation produced empty result.
    #[error("boolean {op} produced empty mesh")]
    EmptyBooleanResult {
        /// Operation name.
        op: String,
    },

    // ── Quality ───────────────────────────────────────────────
    /// Mesh fails quality threshold.
    #[error("quality score {score:.3} below threshold {threshold:.3}")]
    QualityBelowThreshold {
        /// Achieved score.
        score: f64,
        /// Required minimum.
        threshold: f64,
    },

    // ── Channel ───────────────────────────────────────────────
    /// Invalid channel geometry.
    #[error("channel error: {message}")]
    ChannelError {
        /// Description.
        message: String,
    },

    // ── IO ────────────────────────────────────────────────────
    /// IO failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic message.
    #[error("{0}")]
    Other(String),
}

// ── NURBS error kind ─────────────────────────────────────────────────────

/// NURBS-related error kind covering curves, knot vectors, and surfaces.
#[derive(ThisError, Debug)]
pub enum NurbsKind {
    /// B-spline or NURBS curve construction error.
    #[error("NURBS curve error: {0}")]
    Curve(#[from] CurveError),
    /// Knot vector validation error.
    #[error("knot vector error: {0}")]
    Knot(#[from] KnotError),
    /// NURBS surface construction error.
    #[error("NURBS surface error: {0}")]
    Surface(#[from] SurfaceError),
}

// ── Unified error type ───────────────────────────────────────────────────

/// Unified error type for all gaia operations.
///
/// Mirrors the cfd-core pattern: a single top-level enum with
/// domain-specific [`Kind`](crate) sub-enums for structured matching.
///
/// Each variant wraps a domain error kind. Use [`ErrorContext`] to add
/// contextual messages, and [`require`] to convert `Option` to `Result`.
#[derive(ThisError, Debug)]
pub enum Error {
    /// Mesh topology, geometry, watertight, CSG, quality, or channel error.
    #[error(transparent)]
    Mesh(#[from] MeshError),

    /// NURBS curve, knot vector, or surface error.
    #[error(transparent)]
    Nurbs(#[from] NurbsKind),

    /// Primitive mesh builder error.
    #[error(transparent)]
    Primitive(#[from] PrimitiveError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation / deserialisation error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic invalid-input error.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Generic error with additional context.
    #[error("{context}: {source}")]
    WithContext {
        /// Context description.
        context: String,
        /// Underlying error.
        #[source]
        source: Box<Error>,
    },
}

// ── Convenience From impls ────────────────────────────────────────────────

impl From<CurveError> for Error {
    fn from(e: CurveError) -> Self {
        Error::Nurbs(NurbsKind::Curve(e))
    }
}

impl From<KnotError> for Error {
    fn from(e: KnotError) -> Self {
        Error::Nurbs(NurbsKind::Knot(e))
    }
}

impl From<SurfaceError> for Error {
    fn from(e: SurfaceError) -> Self {
        Error::Nurbs(NurbsKind::Surface(e))
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::InvalidInput(msg)
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::InvalidInput(msg.to_string())
    }
}

// ── Result type alias ─────────────────────────────────────────────────────

/// Result type alias for gaia operations.
pub type Result<T> = std::result::Result<T, Error>;

// ── Error context extension ───────────────────────────────────────────────

/// Extension trait for adding context to errors.
pub trait ErrorContext<T> {
    /// Wrap the error with a contextual message.
    ///
    /// # Errors
    /// Returns the original error wrapped with added context.
    fn context(self, msg: impl Into<String>) -> Result<T>;

    /// Wrap the error with a lazily-evaluated contextual message.
    ///
    /// # Errors
    /// Returns the original error wrapped with context from the closure.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T> ErrorContext<T> for Result<T> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            context: msg.into(),
            source: Box::new(e),
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| Error::WithContext {
            context: f(),
            source: Box::new(e),
        })
    }
}

impl<T> ErrorContext<T> for MeshResult<T> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            context: msg.into(),
            source: Box::new(Error::Mesh(e)),
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| Error::WithContext {
            context: f(),
            source: Box::new(Error::Mesh(e)),
        })
    }
}

/// Convert an `Option` into a `Result`, using the message on `None`.
///
/// # Errors
/// Returns `Error::InvalidInput` if the option is `None`.
pub fn require<T>(opt: Option<T>, msg: impl Into<String>) -> Result<T> {
    opt.ok_or_else(|| Error::InvalidInput(msg.into()))
}
