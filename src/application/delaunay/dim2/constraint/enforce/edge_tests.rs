//! Cheap edge/segment predicates used during constraint recovery.

use super::Cdt;
use crate::application::delaunay::core::segments_cross_proper;
use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::domain::core::scalar::Real;
use leto::geometry::Point2;

impl Cdt {
    #[inline]
    pub(super) fn edge_matches_endpoints(
        edge_start: PslgVertexId,
        edge_end: PslgVertexId,
        a: PslgVertexId,
        b: PslgVertexId,
    ) -> bool {
        (edge_start == a && edge_end == b) || (edge_start == b && edge_end == a)
    }

    #[inline]
    pub(super) fn edge_touches_constraint_endpoint(
        edge_start: PslgVertexId,
        edge_end: PslgVertexId,
        constraint_start: PslgVertexId,
        constraint_end: PslgVertexId,
    ) -> bool {
        edge_start == constraint_start
            || edge_start == constraint_end
            || edge_end == constraint_start
            || edge_end == constraint_end
    }

    #[inline]
    pub(super) fn edge_crosses_segment(
        &self,
        seg_start: &Point2<Real>,
        seg_end: &Point2<Real>,
        edge_start: PslgVertexId,
        edge_end: PslgVertexId,
    ) -> bool {
        let edge_a = self.dt.vertex(edge_start).to_point2();
        let edge_b = self.dt.vertex(edge_end).to_point2();
        segments_cross_proper(seg_start, seg_end, &edge_a, &edge_b)
    }
}
