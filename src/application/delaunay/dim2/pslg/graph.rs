//! Planar Straight-Line Graph (PSLG) — the input to CDT.
//!
//! A PSLG consists of:
//! - A set of **vertices** with 2-D coordinates.
//! - A set of **constraint segments** connecting pairs of vertices.
//! - Optional **hole seeds** — points inside regions that should be removed
//!   from the triangulation.
//!
//! # Invariant
//!
//! No two constraint segments may cross in their interiors.  Segments may
//! share endpoints.  The caller is responsible for ensuring this; the CDT
//! will produce undefined results if segments cross.
//!
//! # Theorem — PSLG Validity
//!
//! **Statement**: A set of segments $S$ forms a valid PSLG if and only if
//! no two segments in $S$ share an interior point.  (Shared endpoints are
//! permitted.)
//!
//! **Proof sketch**: The definition of a planar subdivision requires that
//! edges intersect only at shared vertices.  If two segments cross, the
//! crossing point is not a vertex, violating the subdivision property.

use crate::domain::core::scalar::Real;

use super::segment::{PslgSegment, PslgSegmentId};
use super::vertex::{PslgVertex, PslgVertexId};

/// Validation errors for a [`Pslg`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PslgValidationError {
    /// A vertex coordinate is NaN or infinite.
    ///
    /// # Rationale
    ///
    /// Non-finite coordinates produce undefined results in orientation and
    /// in-circle predicates.  Rejecting them at the PSLG level prevents
    /// silent corruption downstream.
    NonFiniteVertex {
        /// The offending vertex index.
        vertex: PslgVertexId,
    },
    /// Segment endpoint index is out of range of the vertex list.
    SegmentVertexOutOfRange {
        /// Segment id with invalid endpoint reference.
        segment: PslgSegmentId,
        /// Start vertex id.
        start: PslgVertexId,
        /// End vertex id.
        end: PslgVertexId,
        /// Total number of vertices present in the PSLG.
        vertex_count: usize,
    },
    /// Segment start and end are identical.
    DegenerateSegment {
        /// Degenerate segment id.
        segment: PslgSegmentId,
        /// Collapsed endpoint id.
        vertex: PslgVertexId,
    },
    /// Two segments are duplicates (same canonical endpoints).
    DuplicateSegment {
        /// First segment id.
        first: PslgSegmentId,
        /// Second segment id.
        second: PslgSegmentId,
        /// Canonical first endpoint.
        a: PslgVertexId,
        /// Canonical second endpoint.
        b: PslgVertexId,
    },
    /// Two segments intersect in their interiors or overlap collinearly.
    IntersectingSegments {
        /// First intersecting segment id.
        first: PslgSegmentId,
        /// Second intersecting segment id.
        second: PslgSegmentId,
    },
    /// Two distinct vertex indices map to the same geometric position (or
    /// a position indistinguishable within floating-point tolerance).
    ///
    /// # Rationale
    ///
    /// Coincident vertices cause degenerate zero-length edges in the
    /// Delaunay triangulation, breaking orient and incircle predicates.
    CoincidentVertices {
        /// First vertex index.
        first: PslgVertexId,
        /// Second vertex index.
        second: PslgVertexId,
    },
}

impl core::fmt::Display for PslgValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteVertex { vertex } => {
                write!(
                    f,
                    "vertex {vertex:?} has non-finite (NaN or ±∞) coordinates"
                )
            }
            Self::SegmentVertexOutOfRange {
                segment,
                start,
                end,
                vertex_count,
            } => write!(
                f,
                "segment {segment} references out-of-range vertex ids ({start}, {end}) with vertex_count={vertex_count}",
            ),
            Self::DegenerateSegment { segment, vertex } => {
                write!(f, "segment {segment} is degenerate at vertex {vertex}")
            }
            Self::DuplicateSegment {
                first,
                second,
                a,
                b,
            } => write!(
                f,
                "segments {first} and {second} are duplicates of edge ({a}, {b})",
            ),
            Self::IntersectingSegments { first, second } => {
                write!(
                    f,
                    "segments {first} and {second} intersect in their interiors"
                )
            }
            Self::CoincidentVertices { first, second } => {
                write!(f, "vertices {first:?} and {second:?} are coincident")
            }
        }
    }
}

impl std::error::Error for PslgValidationError {}

/// A Planar Straight-Line Graph — the canonical input to CDT.
///
/// # Example
///
/// ```rust,ignore
/// use gaia::application::delaunay::Pslg;
///
/// let mut pslg = Pslg::new();
/// let a = pslg.add_vertex(0.0, 0.0);
/// let b = pslg.add_vertex(1.0, 0.0);
/// let c = pslg.add_vertex(0.5, 0.866);
/// pslg.add_segment(a, b);
/// pslg.add_segment(b, c);
/// pslg.add_segment(c, a);
/// ```
#[derive(Clone, Debug)]
pub struct Pslg {
    /// Vertex positions.
    pub(super) vertices: Vec<PslgVertex>,
    /// Constraint segments.
    pub(super) segments: Vec<PslgSegment>,
    /// Hole seed points — each point inside a region to be removed.
    pub(super) holes: Vec<PslgVertex>,
}

impl Pslg {
    /// Create an empty PSLG.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            segments: Vec::new(),
            holes: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(num_vertices: usize, num_segments: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(num_vertices),
            segments: Vec::with_capacity(num_segments),
            holes: Vec::new(),
        }
    }

    // ── Vertex operations ─────────────────────────────────────────────────

    /// Add a vertex at `(x, y)` and return its ID.
    pub fn add_vertex(&mut self, x: Real, y: Real) -> PslgVertexId {
        let id = PslgVertexId::from_usize(self.vertices.len());
        self.vertices.push(PslgVertex::new(x, y));
        id
    }

    /// Add a vertex from a `PslgVertex` value.
    pub fn add_vertex_value(&mut self, v: PslgVertex) -> PslgVertexId {
        let id = PslgVertexId::from_usize(self.vertices.len());
        self.vertices.push(v);
        id
    }

    /// Number of vertices.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get vertex by ID.
    #[inline]
    #[must_use]
    pub fn vertex(&self, id: PslgVertexId) -> &PslgVertex {
        &self.vertices[id.idx()]
    }

    /// Slice of all vertex positions.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[PslgVertex] {
        &self.vertices
    }

    // ── Segment operations ────────────────────────────────────────────────

    /// Add a constraint segment between two existing vertices.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if `start == end` or if either ID is out of range.
    pub fn add_segment(&mut self, start: PslgVertexId, end: PslgVertexId) -> PslgSegmentId {
        debug_assert_ne!(start, end, "degenerate segment");
        debug_assert!(
            start.idx() < self.vertices.len() && end.idx() < self.vertices.len(),
            "segment vertex out of range"
        );
        let id = PslgSegmentId::from_usize(self.segments.len());
        self.segments.push(PslgSegment::new(start, end));
        id
    }

    /// Number of constraint segments.
    #[inline]
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get segment by ID.
    #[inline]
    #[must_use]
    pub fn segment(&self, id: PslgSegmentId) -> &PslgSegment {
        &self.segments[id.idx()]
    }

    /// Slice of all segments.
    #[inline]
    #[must_use]
    pub fn segments(&self) -> &[PslgSegment] {
        &self.segments
    }

    #[cfg(test)]
    pub(crate) fn segments_mut_for_test_only(&mut self) -> &mut Vec<PslgSegment> {
        &mut self.segments
    }

    // ── Hole operations ───────────────────────────────────────────────────

    /// Mark a point as a hole seed.
    ///
    /// All triangles whose centroid is reachable from this point without
    /// crossing a constraint segment will be removed.
    pub fn add_hole(&mut self, x: Real, y: Real) {
        self.holes.push(PslgVertex::new(x, y));
    }

    /// Slice of all hole seeds.
    #[inline]
    #[must_use]
    pub fn holes(&self) -> &[PslgVertex] {
        &self.holes
    }

    // ── Bounding box ──────────────────────────────────────────────────────

    /// Compute the axis-aligned bounding box `(min, max)`.
    ///
    /// Returns `None` if the PSLG has fewer than 1 vertex.
    #[must_use]
    pub fn bounding_box(&self) -> Option<(PslgVertex, PslgVertex)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_x = min_x;
        let mut max_y = min_y;
        for v in &self.vertices[1..] {
            if v.x < min_x {
                min_x = v.x;
            }
            if v.y < min_y {
                min_y = v.y;
            }
            if v.x > max_x {
                max_x = v.x;
            }
            if v.y > max_y {
                max_y = v.y;
            }
        }
        Some((PslgVertex::new(min_x, min_y), PslgVertex::new(max_x, max_y)))
    }
}

impl Default for Pslg {
    fn default() -> Self {
        Self::new()
    }
}
