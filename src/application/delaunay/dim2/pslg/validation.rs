use super::graph::{Pslg, PslgValidationError};
use super::intersection::{collinear_overlap_interior, segments_intersect_closed};
use super::segment::PslgSegmentId;
use super::vertex::PslgVertexId;
use crate::domain::core::scalar::Scalar;

const COINCIDENT_TOLERANCE_ULPS: f64 = 128.0;

impl<T: Scalar> Pslg<T> {
    /// Validate PSLG topological constraints.
    ///
    /// Checks:
    /// - Segment endpoint indices are in range.
    /// - No degenerate segments.
    /// - No duplicate segments.
    /// - No segment-segment interior intersections (shared endpoints allowed).
    ///
    /// # Panics
    ///
    /// Panics if a segment endpoint index becomes invalid after the explicit
    /// range-check phase, violating the internal assumption behind the
    /// `expect(...)` lookups used during pairwise segment validation.
    pub fn validate(&self) -> Result<(), PslgValidationError> {
        use hashbrown::HashMap;

        let n_vertices = self.vertices.len();

        // Check for non-finite coordinates before anything else.
        for (i, v) in self.vertices.iter().enumerate() {
            if !v.x.is_finite() || !v.y.is_finite() {
                return Err(PslgValidationError::NonFiniteVertex {
                    vertex: PslgVertexId::from_usize(i),
                });
            }
        }

        // Check for coincident vertices (O(n²) pairwise).
        // Two vertices are coincident if their separation is indistinguishable
        // from zero relative to the characteristic scale of the PSLG.
        //
        // The characteristic scale is max(bbox_diagonal, max_abs_coord) so
        // that coincident-vertex detection works regardless of whether the
        // point cloud has spread (large diagonal) or is clustered near a
        // single location (tiny diagonal, large absolute coordinates).
        if let Some((first, remaining)) = self.vertices.split_first()
            && !remaining.is_empty()
        {
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
            let diag_sq = {
                let dx = max_x - min_x;
                let dy = max_y - min_y;
                dx * dx + dy * dy
            };
            // Use the larger of (diagonal², max_abs²) as scale².
            let abs_sq = max_abs * max_abs;
            let scale_sq = if diag_sq > abs_sq { diag_sq } else { abs_sq };
            // The former f64 threshold was (128·ε)²; retain that relative
            // guard using the active scalar's machine epsilon.
            let tolerance = <T as Scalar>::from_f64(COINCIDENT_TOLERANCE_ULPS)
                * <T as eunomia::RealField>::EPSILON;
            let coin_tol = scale_sq * tolerance * tolerance;
            for (i, first) in self.vertices.iter().enumerate() {
                for (j, second) in self.vertices.iter().enumerate().skip(i + 1) {
                    let dx = first.x - second.x;
                    let dy = first.y - second.y;
                    if dx * dx + dy * dy < coin_tol {
                        return Err(PslgValidationError::CoincidentVertices {
                            first: PslgVertexId::from_usize(i),
                            second: PslgVertexId::from_usize(j),
                        });
                    }
                }
            }
        }

        for (idx, seg) in self.segments.iter().copied().enumerate() {
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

        let mut seen: HashMap<(PslgVertexId, PslgVertexId), PslgSegmentId> =
            HashMap::with_capacity(self.segments.len());
        for (idx, seg) in self.segments.iter().copied().enumerate() {
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

        for (i, s1) in self.segments.iter().enumerate() {
            for (j, s2) in self.segments.iter().enumerate().skip(i + 1) {
                let share_endpoint = s1.start == s2.start
                    || s1.start == s2.end
                    || s1.end == s2.start
                    || s1.end == s2.end;

                let a1 = self
                    .vertices
                    .get(s1.start.idx())
                    .expect("invariant: segment endpoints were range-checked")
                    .to_point2();
                let a2 = self
                    .vertices
                    .get(s1.end.idx())
                    .expect("invariant: segment endpoints were range-checked")
                    .to_point2();
                let b1 = self
                    .vertices
                    .get(s2.start.idx())
                    .expect("invariant: segment endpoints were range-checked")
                    .to_point2();
                let b2 = self
                    .vertices
                    .get(s2.end.idx())
                    .expect("invariant: segment endpoints were range-checked")
                    .to_point2();

                if !segments_intersect_closed(&a1, &a2, &b1, &b2) {
                    continue;
                }

                // Shared endpoints are allowed only when they do not overlap
                // beyond that endpoint (collinear overlap is invalid).
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
}
