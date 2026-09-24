use crate::domain::core::scalar::Real;

use super::graph::{Pslg, PslgValidationError};
use super::intersection::{collinear_overlap_interior, segments_intersect_closed};
use super::segment::PslgSegmentId;
use super::vertex::PslgVertexId;

impl Pslg {
    /// Validate PSLG topological constraints.
    ///
    /// Checks:
    /// - Segment endpoint indices are in range.
    /// - No degenerate segments.
    /// - No duplicate segments.
    /// - No segment-segment interior intersections (shared endpoints allowed).
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
        if n_vertices >= 2 {
            let mut max_abs: Real = 0.0;
            let (min_x, max_x, min_y, max_y) = self.vertices.iter().fold(
                (Real::MAX, Real::MIN, Real::MAX, Real::MIN),
                |(lo_x, hi_x, lo_y, hi_y), v| {
                    (lo_x.min(v.x), hi_x.max(v.x), lo_y.min(v.y), hi_y.max(v.y))
                },
            );
            for v in &self.vertices {
                max_abs = max_abs.max(v.x.abs()).max(v.y.abs());
            }
            let diag_sq = {
                let dx = max_x - min_x;
                let dy = max_y - min_y;
                dx * dx + dy * dy
            };
            // Use the larger of (diagonal², max_abs²) as scale².
            let scale_sq = diag_sq.max(max_abs * max_abs);
            // Tolerance ≈ (128ε)² × scale² with ε = 2.22e-16 → 8.1e-28.
            let coin_tol = scale_sq * 8.1e-28;
            for i in 0..n_vertices {
                for j in (i + 1)..n_vertices {
                    let dx = self.vertices[i].x - self.vertices[j].x;
                    let dy = self.vertices[i].y - self.vertices[j].y;
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

        for i in 0..self.segments.len() {
            for j in (i + 1)..self.segments.len() {
                let s1 = self.segments[i];
                let s2 = self.segments[j];

                let share_endpoint = s1.start == s2.start
                    || s1.start == s2.end
                    || s1.end == s2.start
                    || s1.end == s2.end;

                let a1 = self.vertices[s1.start.idx()].to_point2();
                let a2 = self.vertices[s1.end.idx()].to_point2();
                let b1 = self.vertices[s2.start.idx()].to_point2();
                let b2 = self.vertices[s2.end.idx()].to_point2();

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
