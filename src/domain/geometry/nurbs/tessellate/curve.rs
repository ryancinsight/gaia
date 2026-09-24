use super::super::curve::NurbsCurve;
use super::super::parameter::uniform_parameter;
use super::{angle_deg, TessellationOptions};
use crate::domain::core::scalar::Scalar;
use leto::geometry::UnitVector3;

/// Adaptively tessellate a 3-D NURBS curve into an ordered polyline.
///
/// The returned points are in order from `t_min` to `t_max`, including both
/// endpoints.  Consecutive segment angle above `opts.max_angle_deg` triggers
/// recursive subdivision of that segment.
///
/// # Example
/// ```rust,ignore
/// use gaia::domain::geometry::nurbs::tessellate::{TessellationOptions, tessellate_curve};
///
/// let pts = tessellate_curve(&my_curve, &TessellationOptions::default());
/// assert!(pts.len() >= 2);
/// ```
#[must_use]
pub fn tessellate_curve<T: Scalar>(
    curve: &NurbsCurve<3, T>,
    opts: &TessellationOptions<T>,
) -> Vec<leto::geometry::Point3<T>> {
    let (t0, t1) = curve.domain();
    let segs = opts.min_segments.max(1);

    // Start with the first point, then adaptively fill in each segment
    let mut result: Vec<leto::geometry::Point3<T>> = Vec::with_capacity(segs * 2 + 1);
    result.push(leto::geometry::Point3::from(curve.point(t0)));

    for i in 0..segs {
        let ta = uniform_parameter(t0, t1, i, segs);
        let tb = uniform_parameter(t0, t1, i + 1, segs);
        subdivide_curve_segment(curve, ta, tb, 0, opts, &mut result);
    }
    result
}

// ---------------------------------------------------------------------------
// Curve subdivision helpers
// ---------------------------------------------------------------------------

/// Recursively subdivide a curve segment `[ta, tb]`, appending points up to
/// (but not including) `ta`'s value, and including `tb`'s endpoint.
fn subdivide_curve_segment<T: Scalar>(
    curve: &NurbsCurve<3, T>,
    ta: T,
    tb: T,
    depth: usize,
    opts: &TessellationOptions<T>,
    result: &mut Vec<leto::geometry::Point3<T>>,
) {
    if depth >= opts.max_depth {
        result.push(leto::geometry::Point3::from(curve.point(tb)));
        return;
    }

    // Compute tangent angle between endpoints
    let (_, tan_a) = curve.point_and_tangent(ta);
    let (_, tan_b) = curve.point_and_tangent(tb);

    let need_split = if let (Some(ua), Some(ub)) = (
        UnitVector3::try_new(tan_a, <T as Scalar>::from_f64(1e-15)),
        UnitVector3::try_new(tan_b, <T as Scalar>::from_f64(1e-15)),
    ) {
        angle_deg(ua, ub) > opts.max_angle_deg
    } else {
        // Degenerate tangent — insert midpoint to be safe
        true
    };

    if need_split {
        let tm = (ta + tb) * <T as Scalar>::from_f64(0.5);
        subdivide_curve_segment(curve, ta, tm, depth + 1, opts, result);
        subdivide_curve_segment(curve, tm, tb, depth + 1, opts, result);
    } else {
        result.push(leto::geometry::Point3::from(curve.point(tb)));
    }
}

// ---------------------------------------------------------------------------
