use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::predicates::{orient_2d, Orientation};
use leto::geometry::Point2;

const RELATIVE_INTERSECTION_TOLERANCE_ULPS: f64 = 64.0;

pub(super) fn relative_intersection_tolerance<T: Scalar>() -> T {
    <T as Scalar>::from_f64(RELATIVE_INTERSECTION_TOLERANCE_ULPS)
        * <T as eunomia::RealField>::EPSILON
}

pub(super) fn segments_intersect_closed<T: Scalar>(
    a1: &Point2<T>,
    a2: &Point2<T>,
    b1: &Point2<T>,
    b2: &Point2<T>,
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

pub(super) fn on_segment<T: Scalar>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> bool {
    p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x) && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
}

pub(super) fn collinear_overlap_interior<T: Scalar>(
    a1: &Point2<T>,
    a2: &Point2<T>,
    b1: &Point2<T>,
    b2: &Point2<T>,
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
    overlap > T::ZERO
}

/// Compute the parametric crossing point of two non-parallel line segments.
///
/// Returns `None` when the displacements are zero or non-finite, or when the
/// normalized determinant is zero in the active precision.
///
/// The caller has already established a proper crossing from exact
/// orientation signs, so a scale-relative angle threshold would discard valid
/// shallow crossings.
#[expect(
    clippy::similar_names,
    reason = "standard segment-endpoint and displacement naming for intersection math"
)]
pub(super) fn segment_cross_point<T: Scalar>(
    a1: &Point2<T>,
    a2: &Point2<T>,
    b1: &Point2<T>,
    b2: &Point2<T>,
) -> Option<(T, T)> {
    let dx_a = a2.x - a1.x;
    let dy_a = a2.y - a1.y;
    let dx_b = b2.x - b1.x;
    let dy_b = b2.y - b1.y;
    let dx_q = b1.x - a1.x;
    let dy_q = b1.y - a1.y;
    let zero = T::ZERO;
    let scale = [dx_a, dy_a, dx_b, dy_b, dx_q, dy_q]
        .into_iter()
        .map(T::abs)
        .fold(zero, |largest, value| {
            if value.total_cmp(&largest).is_gt() {
                value
            } else {
                largest
            }
        });
    if scale == zero || !scale.is_finite() {
        return None;
    }

    // Dividing every displacement by the same largest component leaves the
    // cross-product ratio unchanged while keeping products in range when the
    // segments are uniformly tiny or large.
    let dx_a = dx_a / scale;
    let dy_a = dy_a / scale;
    let dx_b = dx_b / scale;
    let dy_b = dy_b / scale;
    let dx_q = dx_q / scale;
    let dy_q = dy_q / scale;
    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom == T::ZERO {
        return None;
    }
    let t = (dx_q * dy_b - dy_q * dx_b) / denom;
    let one = T::ONE;
    Some(((one - t) * a1.x + t * a2.x, (one - t) * a1.y + t * a2.y))
}

#[cfg(test)]
mod tests {
    use super::segment_cross_point;
    use crate::domain::core::scalar::Scalar;
    use crate::domain::geometry::predicates::{orient_2d, Orientation};
    use leto::geometry::Point2;

    fn shallow_crossing<T: Scalar>() {
        let zero = T::ZERO;
        let one = T::ONE;
        let offset = T::from_int(16) * <T as eunomia::RealField>::EPSILON;
        let a1 = Point2::new(zero, zero);
        let a2 = Point2::new(one, zero);
        let b1 = Point2::new(zero, offset);
        let b2 = Point2::new(one, -offset);

        assert_eq!(orient_2d(&a1, &a2, &b1), Orientation::Positive);
        assert_eq!(orient_2d(&a1, &a2, &b2), Orientation::Negative);
        assert_eq!(
            segment_cross_point(&a1, &a2, &b1, &b2),
            Some((<T as Scalar>::from_f64(0.5), zero))
        );
    }

    fn scale_extremes<T: Scalar>(small_scale: f64, large_scale: f64) {
        let zero = T::ZERO;
        let scale_values = [
            <T as Scalar>::from_f64(small_scale),
            <T as Scalar>::from_f64(large_scale),
        ];
        for scale in scale_values {
            let a1 = Point2::new(-scale, -scale);
            let a2 = Point2::new(scale, scale);
            let b1 = Point2::new(-scale, scale);
            let b2 = Point2::new(scale, -scale);
            assert_eq!(segment_cross_point(&a1, &a2, &b1, &b2), Some((zero, zero)));
        }
    }

    #[test]
    fn shallow_crossings_use_orientation_evidence_without_angle_cutoff() {
        shallow_crossing::<f32>();
        shallow_crossing::<f64>();
    }

    #[test]
    fn construction_reports_a_determinant_lost_to_cancellation() {
        let a1 = Point2::new(0.0_f32, 0.0);
        let a2 = Point2::new(8192.0, 8191.0);
        let b1 = Point2::new(0.5, 0.5);
        let b2 = Point2::new(8191.5, 8190.5);

        assert_eq!(orient_2d(&a1, &a2, &b1), Orientation::Positive);
        assert_eq!(orient_2d(&a1, &a2, &b2), Orientation::Negative);
        assert_eq!(orient_2d(&b1, &b2, &a1), Orientation::Negative);
        assert_eq!(orient_2d(&b1, &b2, &a2), Orientation::Positive);
        assert_eq!(segment_cross_point(&a1, &a2, &b1, &b2), None);
    }

    #[test]
    fn crossing_construction_scales_displacements_before_products() {
        scale_extremes::<f32>(2.0_f64.powi(-80), 2.0_f64.powi(80));
        scale_extremes::<f64>(2.0_f64.powi(-600), 2.0_f64.powi(600));
    }
}
