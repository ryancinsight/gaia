use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d, Orientation};
use leto::geometry::Point2;

pub(super) fn segments_intersect_closed(
    a1: &Point2<Real>,
    a2: &Point2<Real>,
    b1: &Point2<Real>,
    b2: &Point2<Real>,
) -> bool {
    let o1 = orient_2d(a1, a2, b1);
    let o2 = orient_2d(a1, a2, b2);
    let o3 = orient_2d(b1, b2, a1);
    let o4 = orient_2d(b1, b2, a2);

    if o1 != Orientation::Degenerate
        && o2 != Orientation::Degenerate
        && o3 != Orientation::Degenerate
        && o4 != Orientation::Degenerate
    {
        return o1 != o2 && o3 != o4;
    }

    if o1 == Orientation::Degenerate && on_segment(a1, a2, b1) {
        return true;
    }
    if o2 == Orientation::Degenerate && on_segment(a1, a2, b2) {
        return true;
    }
    if o3 == Orientation::Degenerate && on_segment(b1, b2, a1) {
        return true;
    }
    if o4 == Orientation::Degenerate && on_segment(b1, b2, a2) {
        return true;
    }

    false
}

pub(super) fn on_segment(a: &Point2<Real>, b: &Point2<Real>, p: &Point2<Real>) -> bool {
    p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x) && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
}

pub(super) fn collinear_overlap_interior(
    a1: &Point2<Real>,
    a2: &Point2<Real>,
    b1: &Point2<Real>,
    b2: &Point2<Real>,
) -> bool {
    // Non-collinear cannot overlap interiorly.
    if orient_2d(a1, a2, b1) != Orientation::Degenerate
        || orient_2d(a1, a2, b2) != Orientation::Degenerate
    {
        return false;
    }

    let use_x = (a2.x - a1.x).abs() >= (a2.y - a1.y).abs();
    let (a_lo, a_hi, b_lo, b_hi) = if use_x {
        (
            a1.x.min(a2.x),
            a1.x.max(a2.x),
            b1.x.min(b2.x),
            b1.x.max(b2.x),
        )
    } else {
        (
            a1.y.min(a2.y),
            a1.y.max(a2.y),
            b1.y.min(b2.y),
            b1.y.max(b2.y),
        )
    };

    let overlap = a_hi.min(b_hi) - a_lo.max(b_lo);
    overlap > 0.0
}

/// Compute the f64 parametric crossing point of two non-parallel line segments.
///
/// Uses a scale-relative parallelism guard so the threshold adapts to the
/// coordinate magnitude: $|\text{denom}| < |e_a| \cdot |e_b| \cdot 10^{-14}$.
pub(super) fn segment_cross_point(
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
