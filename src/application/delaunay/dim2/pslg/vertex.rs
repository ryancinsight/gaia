//! Vertex type for the PSLG / Delaunay triangulation.
//!
//! Each vertex carries a 2-D position suitable for exact Shewchuk predicates.
//!
//! # Design
//!
//! `PslgVertexId` is a strongly-typed `u32` newtype following the same pattern
//! as [`crate::domain::core::index::VertexId`].  It indexes into the flat vertex array
//! stored in [`super::graph::Pslg`].

use std::fmt;

use crate::domain::core::scalar::{Real, Scalar};

/// A 2-D vertex position for the Delaunay triangulation.
///
/// Stored contiguously in the PSLG vertex pool. Coordinates use `T`; `f64` is
/// the default precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PslgVertex<T = Real> {
    /// X-coordinate.
    pub x: T,
    /// Y-coordinate.
    pub y: T,
}

impl PslgVertex<Real> {
    /// Create a new vertex at `(x, y)`.
    #[inline]
    #[must_use]
    pub fn new(x: Real, y: Real) -> Self {
        Self { x, y }
    }
}

impl<T: Scalar> PslgVertex<T> {
    /// Squared Euclidean distance to another vertex.
    #[inline]
    #[must_use]
    pub fn dist_sq(&self, other: &Self) -> T {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another vertex.
    #[inline]
    #[must_use]
    pub fn dist(&self, other: &Self) -> T {
        self.dist_sq(other).sqrt()
    }

    /// Midpoint between `self` and `other`.
    #[inline]
    #[must_use]
    pub fn midpoint(&self, other: &Self) -> Self {
        Self {
            x: <T as Scalar>::from_f64(0.5) * (self.x + other.x),
            y: <T as Scalar>::from_f64(0.5) * (self.y + other.y),
        }
    }

    /// Convert to a `leto::geometry::Point2<T>` for predicate calls.
    #[inline]
    #[must_use]
    pub fn to_point2(&self) -> leto::geometry::Point2<T> {
        leto::geometry::Point2::new(self.x, self.y)
    }
}

impl<T: Scalar> From<[T; 2]> for PslgVertex<T> {
    #[inline]
    fn from(arr: [T; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }
}

impl<T: Scalar> From<(T, T)> for PslgVertex<T> {
    #[inline]
    fn from((x, y): (T, T)) -> Self {
        Self { x, y }
    }
}

/// Strongly-typed vertex index into the PSLG vertex array.
///
/// Follows the same newtype-index pattern as [`crate::domain::core::index::VertexId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PslgVertexId(pub u32);

impl PslgVertexId {
    /// Create from a raw `u32`.
    #[inline]
    #[must_use]
    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Create from `usize`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self(n as u32)
    }

    /// Raw `u32` index.
    #[inline]
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// As `usize`.
    #[inline]
    #[must_use]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for PslgVertexId {
    #[inline]
    fn from(n: usize) -> Self {
        Self(n as u32)
    }
}

impl fmt::Display for PslgVertexId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Sentinel value representing "no vertex" / the super-triangle ghost vertex.
pub const GHOST_VERTEX: PslgVertexId = PslgVertexId(u32::MAX);
