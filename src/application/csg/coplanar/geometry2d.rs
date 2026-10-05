//! 2-D point and AABB helpers.

#[cfg(test)]
use eunomia::NumericElement;

use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::predicates::{orient_2d_arr, Orientation};

/// Test whether 2-D point `(px,py)` lies inside or on the boundary of the
/// CCW-wound triangle `(ax,ay)→(bx,by)→(cx,cy)` using exact arithmetic.
///
/// # Theorem — Degenerate Triangle Rejection
///
/// **Statement**: A zero-area (degenerate) triangle contains no points by
/// definition.  When `orient_2d(a,b,c) == Degenerate`, the three vertices
/// are collinear and the triangle degenerates to a line segment or point,
/// enclosing zero area.
///
/// **Proof**: A triangle in ℝ² encloses area iff its signed area
/// `½|det([b−a, c−a])| > 0`, which is equivalent to `orient_2d(a,b,c) ≠ 0`.
/// When the determinant is zero the "triangle" is a 1-D simplex with empty
/// interior.  Returning `false` prevents false-positive containment results
/// that would cause incorrect coplanar Boolean fragment classification.
///
/// **Consequence**: Callers receive `false` for degenerate triangles rather
/// than the previous behavior where all collinear points were classified as
/// "inside" (since no edge had both positive and negative orientations).  ∎
#[expect(
    clippy::too_many_arguments,
    reason = "2-D triangle-containment predicate: 3 vertices + 1 query = 8 scalar coordinates; struct grouping would obscure the mathematical structure"
)]
#[inline]
pub(crate) fn point_in_tri_2d_exact<T: Scalar>(
    px: T,
    py: T,
    ax: T,
    ay: T,
    bx: T,
    by: T,
    cx: T,
    cy: T,
) -> bool {
    let p = [px, py];
    let a = [ax, ay];
    let b = [bx, by];
    let c = [cx, cy];

    // Reject degenerate (zero-area) triangles: collinear vertices enclose
    // no area and therefore contain no points.
    let tri_ori = orient_2d_arr(a, b, c);
    if tri_ori == Orientation::Degenerate {
        return false;
    }

    let d0 = orient_2d_arr(a, b, p);
    let d1 = orient_2d_arr(b, c, p);
    let d2 = orient_2d_arr(c, a, p);

    let neg =
        d0 == Orientation::Negative || d1 == Orientation::Negative || d2 == Orientation::Negative;
    let pos =
        d0 == Orientation::Positive || d1 == Orientation::Positive || d2 == Orientation::Positive;

    // Inside or on boundary iff all edge orientations agree with the triangle
    // winding (or are degenerate = on-edge).
    !(neg && pos)
}

/// Test whether 2-D point is inside the union of selected triangles.
///
/// `indices` refers into `tris`; duplicates are tolerated and behave as a set.
#[inline]
pub(crate) fn point_in_union_2d_exact_indexed<T: Scalar>(
    px: T,
    py: T,
    tris: &[[T; 6]],
    indices: &[usize],
) -> bool {
    indices.iter().any(|&i| {
        let t = tris[i];
        point_in_tri_2d_exact(px, py, t[0], t[1], t[2], t[3], t[4], t[5])
    })
}

/// 2-D AABB of a triangle: `[min_u, min_v, max_u, max_v]`.
#[inline]
pub(crate) fn aabb2<T: Scalar>(ax: T, ay: T, bx: T, by: T, cx: T, cy: T) -> [T; 4] {
    [
        ax.min_scalar(bx).min_scalar(cx),
        ay.min_scalar(by).min_scalar(cy),
        ax.max_scalar(bx).max_scalar(cx),
        ay.max_scalar(by).max_scalar(cy),
    ]
}

/// True if two 2-D AABBs intersect (inclusive boundary).
#[inline]
pub(crate) fn aabb_overlaps<T: Scalar>(a: &[T; 4], b: &[T; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Unsigned area of a 2-D polygon via the shoelace formula.
#[cfg(test)]
#[inline]
pub(crate) fn polygon_area_2d<T: Scalar + std::ops::Neg<Output = T>>(poly: &[[T; 2]]) -> T {
    let n = poly.len();
    if n < 3 {
        return <T as NumericElement>::ZERO;
    }
    let mut sum = <T as NumericElement>::ZERO;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    <T as NumericElement>::abs(sum) * <T as Scalar>::from_f64(0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed CCW triangle correctly contains its centroid.
    #[test]
    fn point_in_tri_2d_exact_interior_point() {
        assert!(point_in_tri_2d_exact(
            0.25, 0.25, // centroid-ish
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ));
    }

    /// A point outside the triangle is rejected.
    #[test]
    fn point_in_tri_2d_exact_exterior_point() {
        assert!(!point_in_tri_2d_exact(
            2.0, 2.0, // far outside
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ));
    }

    /// A point on the edge of the triangle is accepted (boundary-inclusive).
    #[test]
    fn point_in_tri_2d_exact_edge_point() {
        assert!(point_in_tri_2d_exact(
            0.5, 0.0, // midpoint of edge a→b
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ));
    }

    /// Degenerate triangle (collinear vertices) rejects all points.
    ///
    /// Validates the Phase 5a guard: zero-area triangles contain nothing.
    #[test]
    fn point_in_tri_2d_exact_degenerate_rejects_collinear_point() {
        // Triangle degenerates to the segment (0,0)→(2,0).
        // Point (1,0) is on that segment but the "triangle" has zero area.
        assert!(!point_in_tri_2d_exact(
            1.0, 0.0, // on the degenerate segment
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
        ));
    }

    /// Degenerate triangle rejects points off the line too.
    #[test]
    fn point_in_tri_2d_exact_degenerate_rejects_off_line_point() {
        assert!(!point_in_tri_2d_exact(
            1.0, 1.0, // off the degenerate line
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
        ));
    }

    /// Degenerate triangle where all three vertices coincide.
    #[test]
    fn point_in_tri_2d_exact_degenerate_single_point() {
        assert!(!point_in_tri_2d_exact(
            0.0, 0.0, // same as all vertices
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ));
    }
}
