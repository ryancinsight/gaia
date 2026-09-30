//! Constraint segment in the PSLG.
//!
//! A segment is a directed edge between two PSLG vertices that must appear
//! as an edge in the resulting CDT.

use std::fmt;

use super::vertex::PslgVertexId;

/// A constraint segment connecting two PSLG vertices.
///
/// The CDT guarantees that an edge between `start` and `end` will exist in
/// the final triangulation.  Segments may share endpoints but must not cross
/// in their interiors (the PSLG validity requirement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PslgSegment {
    /// Start vertex.
    pub start: PslgVertexId,
    /// End vertex.
    pub end: PslgVertexId,
}

impl PslgSegment {
    /// Create a new segment from `start` to `end`.
    #[inline]
    #[must_use]
    pub fn new(start: PslgVertexId, end: PslgVertexId) -> Self {
        Self { start, end }
    }

    /// Return the canonical (sorted) edge key for deduplication.
    ///
    /// `(min(start, end), max(start, end))`.
    #[inline]
    #[must_use]
    pub fn canonical(&self) -> (PslgVertexId, PslgVertexId) {
        if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    /// Check if this segment is degenerate (start == end).
    #[inline]
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.start == self.end
    }
}

/// Strongly-typed segment index into the PSLG segment array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PslgSegmentId(pub u32);

impl PslgSegmentId {
    /// Create from raw `u32`.
    #[inline]
    #[must_use]
    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Create from `usize`.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `u32::MAX`.
    #[inline]
    #[must_use]
    pub fn from_usize(n: usize) -> Self {
        Self(u32::try_from(n).expect("segment index fits in u32"))
    }

    /// Raw index.
    #[inline]
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// As `usize`.
    ///
    /// # Panics
    ///
    /// Panics only on targets where `usize` cannot represent every `u32`.
    #[inline]
    #[must_use]
    pub fn idx(self) -> usize {
        usize::try_from(self.0).expect("segment index fits in usize")
    }
}

impl From<usize> for PslgSegmentId {
    #[inline]
    fn from(n: usize) -> Self {
        Self::from_usize(n)
    }
}

impl fmt::Display for PslgSegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}
