use super::graph::{Pslg, PslgValidationError};
use super::intersection::{collinear_overlap_interior, segments_intersect_closed};
use super::segment::{PslgSegment, PslgSegmentId};
use super::vertex::{PslgVertex, PslgVertexId};
use crate::domain::core::scalar::Scalar;

const COINCIDENT_TOLERANCE_ULPS: f64 = 128.0;

/// Check that every vertex has finite coordinates and that no two distinct
/// vertices are closer than the ULP-scaled coincident tolerance.
fn check_vertex_positions<T: Scalar>(
    vertices: &[PslgVertex<T>],
) -> Result<(), PslgValidationError> {
    // Phase 1: non-finite coordinate check.
    for (i, v) in vertices.iter().enumerate() {
        if !v.x.is_finite() || !v.y.is_finite() {
            return Err(PslgValidationError::NonFiniteVertex {
                vertex: PslgVertexId::from_usize(i),
            });
        }
    }

    // Phase 2: coincident vertex check (O(n²) pairwise).
    let Some((first, remaining)) = vertices.split_first() else {
        return Ok(());
    };
    if remaining.is_empty() {
        return Ok(());
    }
    let mut min_x = first.x;
    let mut max_x = first.x;
    let mut min_y = first.y;
    let mut max_y = first.y;
    let mut max_abs = if first.x.abs() > first.y.abs() {
        first.x.abs()
    } else {
        first.y.abs()
    };
    for v in remaining {
        if v.x < min_x {
            min_x = v.x;
        }
        if v.x > max_x {
            max_x = v.x;
        }
        if v.y < min_y {
            min_y = v.y;
        }
        if v.y > max_y {
            max_y = v.y;
        }
        if v.x.abs() > max_abs {
            max_abs = v.x.abs();
        }
        if v.y.abs() > max_abs {
            max_abs = v.y.abs();
        }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let diag_sq = dx * dx + dy * dy;
    let abs_sq = max_abs * max_abs;
    let scale_sq = if diag_sq > abs_sq { diag_sq } else { abs_sq };
    let tolerance =
        <T as Scalar>::from_f64(COINCIDENT_TOLERANCE_ULPS) * <T as eunomia::RealField>::EPSILON;
    let coin_tol = scale_sq * tolerance * tolerance;
    for (i, first) in vertices.iter().enumerate() {
        for (j, second) in vertices.iter().enumerate().skip(i + 1) {
            let ddx = first.x - second.x;
            let ddy = first.y - second.y;
            if ddx * ddx + ddy * ddy < coin_tol {
                return Err(PslgValidationError::CoincidentVertices {
                    first: PslgVertexId::from_usize(i),
                    second: PslgVertexId::from_usize(j),
                });
            }
        }
    }
    Ok(())
}

/// Check segment endpoint ranges, duplicate segments, and intersections.
fn check_segment_topology<T: Scalar>(
    vertices: &[PslgVertex<T>],
    segments: &[PslgSegment],
) -> Result<(), PslgValidationError> {
    use hashbrown::HashMap;

    let n_vertices = vertices.len();

    // Phase 3: endpoint range and degeneracy.
    for (idx, seg) in segments.iter().copied().enumerate() {
        let sid = PslgSegmentId::from_usize(idx);
        if seg.start.idx() >= n_vertices || seg.end.idx() >= n_vertices {
            return Err(PslgValidationError::SegmentVertexOutOfRange {
                segment: sid,
                start: seg.start,
                end: seg.end,
                vertex_count: n_vertices,
            });
        }
        if seg.is_degenerate() {
            return Err(PslgValidationError::DegenerateSegment {
                segment: sid,
                vertex: seg.start,
            });
        }
    }

    // Phase 4: duplicate segments.
    let mut seen: HashMap<(PslgVertexId, PslgVertexId), PslgSegmentId> =
        HashMap::with_capacity(segments.len());
    for (idx, seg) in segments.iter().copied().enumerate() {
        let sid = PslgSegmentId::from_usize(idx);
        let key = seg.canonical();
        if let Some(first) = seen.insert(key, sid) {
            return Err(PslgValidationError::DuplicateSegment {
                first,
                second: sid,
                a: key.0,
                b: key.1,
            });
        }
    }

    // Phase 5: intersecting segment pairs (O(s²)).
    for (i, s1) in segments.iter().enumerate() {
        for (j, s2) in segments.iter().enumerate().skip(i + 1) {
            let share_endpoint = s1.start == s2.start
                || s1.start == s2.end
                || s1.end == s2.start
                || s1.end == s2.end;
            let a1 = vertices
                .get(s1.start.idx())
                .expect("invariant: segment endpoints were range-checked")
                .to_point2();
            let a2 = vertices
                .get(s1.end.idx())
                .expect("invariant: segment endpoints were range-checked")
                .to_point2();
            let b1 = vertices
                .get(s2.start.idx())
                .expect("invariant: segment endpoints were range-checked")
                .to_point2();
            let b2 = vertices
                .get(s2.end.idx())
                .expect("invariant: segment endpoints were range-checked")
                .to_point2();
            if !segments_intersect_closed(&a1, &a2, &b1, &b2) {
                continue;
            }
            if share_endpoint && !collinear_overlap_interior(&a1, &a2, &b1, &b2) {
                continue;
            }
            return Err(PslgValidationError::IntersectingSegments {
                first: PslgSegmentId::from_usize(i),
                second: PslgSegmentId::from_usize(j),
            });
        }
    }

    Ok(())
}

impl<T: Scalar> Pslg<T> {
    /// Validate PSLG topological constraints.
    ///
    /// Checks:
    /// - Segment endpoint indices are in range.
    /// - No degenerate segments.
    /// - No duplicate segments.
    /// - No segment-segment interior intersections (shared endpoints allowed).
    ///
    /// # Errors
    ///
    /// Returns [`PslgValidationError::NonFiniteVertex`] for NaN or infinite
    /// coordinates, [`PslgValidationError::CoincidentVertices`] for distinct
    /// vertices that collapse within tolerance, and the corresponding segment
    /// validation variants for out-of-range endpoints, degenerate segments,
    /// duplicate segments, or intersecting constraints.
    ///
    /// # Panics
    ///
    /// Panics if a segment endpoint index becomes invalid after the explicit
    /// range-check phase, violating the internal assumption behind the
    /// `expect(...)` lookups used during pairwise segment validation.
    pub fn validate(&self) -> Result<(), PslgValidationError> {
        check_vertex_positions(&self.vertices)?;
        check_segment_topology(&self.vertices, &self.segments)
    }
}
