//! Core shared Delaunay components.

use crate::application::delaunay::dim2::pslg::vertex::PslgVertexId;
use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d, Orientation};
use leto::geometry::Point2;

/// Canonicalize an undirected edge key by sorting endpoint IDs.
#[inline]
#[must_use]
pub(crate) fn canonical_edge(a: PslgVertexId, b: PslgVertexId) -> (PslgVertexId, PslgVertexId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Test if two line segments properly cross (share an interior point).
#[inline]
#[must_use]
pub(crate) fn segments_cross_proper(
    a1: &Point2<Real>,
    a2: &Point2<Real>,
    b1: &Point2<Real>,
    b2: &Point2<Real>,
) -> bool {
    let o1 = orient_2d(a1, a2, b1);
    let o2 = orient_2d(a1, a2, b2);
    let o3 = orient_2d(b1, b2, a1);
    let o4 = orient_2d(b1, b2, a2);

    o1 != o2
        && o3 != o4
        && o1 != Orientation::Degenerate
        && o2 != Orientation::Degenerate
        && o3 != Orientation::Degenerate
        && o4 != Orientation::Degenerate
}

/// Compute the f64 parametric crossing point of two non-parallel line segments.
///
/// Uses a scale-relative parallelism guard so the threshold adapts to the
/// coordinate magnitude: $|\text{denom}| < |e_a| \cdot |e_b| \cdot 10^{-14}$.
#[inline]
#[must_use]
pub(crate) fn segment_cross_point(
    a1: &Point2<Real>,
    a2: &Point2<Real>,
    b1: &Point2<Real>,
    b2: &Point2<Real>,
) -> Option<(Real, Real)> {
    let dx_a = a2.x - a1.x;
    let dy_a = a2.y - a1.y;
    let dx_b = b2.x - b1.x;
    let dy_b = b2.y - b1.y;
    let denom = dx_a * dy_b - dy_a * dx_b;
    let len_a_sq = dx_a * dx_a + dy_a * dy_a;
    let len_b_sq = dx_b * dx_b + dy_b * dy_b;
    let scale = (len_a_sq * len_b_sq).sqrt().max(1e-30);
    if denom.abs() < scale * 1e-14 {
        return None;
    }
    let t = ((b1.x - a1.x) * dy_b - (b1.y - a1.y) * dx_b) / denom;
    Some((a1.x + t * dx_a, a1.y + t * dy_a))
}
