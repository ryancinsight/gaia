//! Vertex record and inline spatial-cell index types for [`super::VertexPool`].

use crate::domain::core::scalar::Scalar;
use leto::geometry::{Point3, Vector3};

// ── VertexData<T> ────────────────────────────────────────────────────────────

/// Data stored per vertex — position + surface normal.
#[derive(Clone, Debug)]
pub struct VertexData<T: Scalar = f64> {
    /// Position in 3-D space.
    pub position: Point3<T>,
    /// Surface normal (may be zero for interior vertices).
    pub normal: Vector3<T>,
}

impl<T: Scalar> VertexData<T> {
    /// Create a vertex with explicit position and normal.
    pub fn new(position: Point3<T>, normal: Vector3<T>) -> Self {
        Self { position, normal }
    }

    /// Create a vertex with position only (zero normal).
    pub fn from_position(position: Point3<T>) -> Self {
        Self {
            position,
            normal: Vector3::zeros(),
        }
    }

    /// Linear interpolation between two vertices.
    ///
    /// Position is linearly interpolated; normal is renormalised.
    pub fn lerp(&self, other: &Self, t: T) -> Self {
        let one_minus_t = <T as eunomia::NumericElement>::ONE - t;
        let position = Point3::from(self.position.coords * one_minus_t + other.position.coords * t);
        let n = self.normal * one_minus_t + other.normal * t;
        let len = n.norm();
        let normal = if len > <T as eunomia::NumericElement>::ZERO {
            n / len
        } else {
            Vector3::zeros()
        };
        Self { position, normal }
    }
}

// ── CellIndices ──────────────────────────────────────────────────────────────

/// Stack-allocated, inline representation for spatial hash cell indices.
///
/// Avoids heap allocations for cells containing a single index (the most common case),
/// falling back to a boxed slice for cells containing multiple indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CellIndices {
    /// Exactly 1 index (inline stack-allocated).
    One(u32),
    /// Multiple indices (heap-allocated fallback).
    Many(Box<[u32]>),
}

impl CellIndices {
    /// Get the first index in the cell.
    #[inline]
    #[must_use]
    pub fn first(&self) -> u32 {
        match self {
            Self::One(idx) => *idx,
            Self::Many(slice) => slice[0],
        }
    }

    /// Push a new index to the cell.
    pub fn push(&mut self, idx: u32) {
        let current = std::mem::replace(self, Self::One(idx));
        *self = match current {
            Self::One(i0) => {
                let v = vec![i0, idx];
                Self::Many(v.into_boxed_slice())
            }
            Self::Many(slice) => {
                let mut v = slice.into_vec();
                v.push(idx);
                Self::Many(v.into_boxed_slice())
            }
        };
    }

    /// Access the indices as a slice of `u32`.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::One(idx) => std::slice::from_ref(idx),
            Self::Many(slice) => slice,
        }
    }

    /// Iterate over indices in this cell.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.as_slice().iter()
    }
}

impl std::ops::Deref for CellIndices {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a CellIndices {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
