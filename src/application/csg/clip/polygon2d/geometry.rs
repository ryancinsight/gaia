//! Geometry helpers for 2-D polygon clipping.
//!
//! Provides fundamental geometric predicates and utilities shared across
//! all clipping algorithms: signed area, convexity test, winding-number
//! point-in-polygon, and segment-segment intersection.

use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d_arr, Orientation};

/// Signed area of a 2-D polygon (positive = CCW, negative = CW).
pub(crate) fn signed_area(poly: &[[Real; 2]]) -> Real {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    sum * 0.5
}

/// Unsigned area of a 2-D polygon.
#[inline]
#[must_use]
pub fn polygon_area(poly: &[[Real; 2]]) -> Real {
    signed_area(poly).abs()
}

/// Test if a simple polygon is convex.
#[cfg(test)]
pub(crate) fn is_convex(poly: &[[Real; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return true;
    }
    let mut sign = Orientation::Degenerate;
    for i in 0..n {
        let j = (i + 1) % n;
        let k = (i + 2) % n;
        let ori = orient_2d_arr(poly[i], poly[j], poly[k]);
        if ori == Orientation::Degenerate {
            continue;
        }
        if sign == Orientation::Degenerate {
            sign = ori;
        } else if sign != ori {
            return false;
        }
    }
    true
}

/// Ensure a polygon is in CCW winding order.
pub(crate) fn ensure_ccw(poly: &mut [[Real; 2]]) {
    if winding_ccw(poly) == Some(false) {
        poly.reverse();
    }
}

/// Test if the 2-D point `p` lies inside or on the boundary of triangle
/// `(a, b, c)`.
///
/// Exact: the three edge orientations must not mix signs.  A point lying on an
/// edge is `Degenerate` for that edge and therefore counts as inside
/// (boundary-inclusive) — which is what the boundary-loop ear test needs.
#[inline]
pub(crate) fn point_in_triangle(
    p: &[Real; 2],
    a: &[Real; 2],
    b: &[Real; 2],
    c: &[Real; 2],
) -> bool {
    let d1 = orient_2d_arr(*a, *b, *p);
    let d2 = orient_2d_arr(*b, *c, *p);
    let d3 = orient_2d_arr(*c, *a, *p);
    let has_neg =
        d1 == Orientation::Negative || d2 == Orientation::Negative || d3 == Orientation::Negative;
    let has_pos =
        d1 == Orientation::Positive || d2 == Orientation::Positive || d3 == Orientation::Positive;
    !(has_neg && has_pos)
}

/// Determine the winding direction of a 2-D polygon.
///
/// Returns `Some(true)` for counter-clockwise and `Some(false)` for clockwise
/// winding, or `None` when the polygon has no resolvable orientation (fewer
/// than three vertices, or exactly zero area).
///
/// Orientation is resolved with the exact `orient_2d_arr` predicate: starting at
/// the leftmost-then-lowest vertex (always convex for a simple polygon), the
/// walk returns the first non-degenerate turn.  The shoelace [`signed_area`] is
/// kept **only** as a documented fallback for the fully-degenerate case where
/// every visited turn is exactly collinear.
pub(crate) fn winding_ccw(pts: &[[Real; 2]]) -> Option<bool> {
    let n = pts.len();
    if n < 3 {
        return None;
    }

    let mut min_i = 0usize;
    for i in 1..n {
        if pts[i][0] < pts[min_i][0] || (pts[i][0] == pts[min_i][0] && pts[i][1] < pts[min_i][1]) {
            min_i = i;
        }
    }

    for k in 0..n {
        let i = (min_i + k) % n;
        let prev = (i + n - 1) % n;
        let next = (i + 1) % n;
        match orient_2d_arr(pts[prev], pts[i], pts[next]) {
            Orientation::Positive => return Some(true),
            Orientation::Negative => return Some(false),
            Orientation::Degenerate => {}
        }
    }

    // Documented fallback: no turn resolved the winding exactly, so use the
    // shoelace sign.  Only reached for fully-degenerate (collinear) inputs.
    let area = signed_area(pts);
    if area > 0.0 {
        Some(true)
    } else if area < 0.0 {
        Some(false)
    } else {
        None
    }
}

/// Segment-segment intersection parameter.
/// Returns `(t, s)` where `t` is the parameter along `(p1→p2)` and `s` along `(p3→p4)`.
///
/// ## Algorithm
///
/// 1. Compute direction vectors `d1 = p2-p1`, `d2 = p4-p3`.
/// 2. Reject only if `d1` and `d2` are exactly parallel via exact
///    orientation (`orient_2d_arr([0,0], d1, d2) == Degenerate`).
/// 3. Solve the 2x2 linear system for `(t,s)` by Cramer's rule.
///
/// ## Theorem — Parallelism Equivalence
///
/// Two 2-D direction vectors are parallel iff the orientation determinant of
/// `(0, d1, d2)` is exactly zero.
///
/// **Proof sketch.**
/// The orientation determinant is the 2x2 determinant `d1.x*d2.y-d1.y*d2.x`,
/// which is the signed area of the parallelogram spanned by `d1,d2`.
/// Zero area is equivalent to linear dependence (parallel vectors). ∎
pub(crate) fn seg_intersect(
    p1: [Real; 2],
    p2: [Real; 2],
    p3: [Real; 2],
    p4: [Real; 2],
) -> Option<(Real, Real)> {
    let d1x = p2[0] - p1[0];
    let d1y = p2[1] - p1[1];
    let d2x = p4[0] - p3[0];
    let d2y = p4[1] - p3[1];

    if orient_2d_arr([0.0, 0.0], [d1x, d1y], [d2x, d2y]) == Orientation::Degenerate {
        return None;
    }

    let denom = d1x * d2y - d1y * d2x;
    let dx = p3[0] - p1[0];
    let dy = p3[1] - p1[1];
    let t = (dx * d2y - dy * d2x) / denom;
    let s = (dx * d1y - dy * d1x) / denom;
    Some((t, s))
}

/// Point-in-polygon test using winding number (robust for concave polygons).
pub(crate) fn point_in_polygon(px: Real, py: Real, poly: &[[Real; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let p = [px, py];
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = poly[i][1];
        let yj = poly[j][1];
        if yi <= py {
            if yj > py && orient_2d_arr(poly[i], poly[j], p) == Orientation::Positive {
                winding += 1;
            }
        } else if yj <= py && orient_2d_arr(poly[i], poly[j], p) == Orientation::Negative {
            winding -= 1;
        }
    }
    winding != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Real, b: Real, tol: Real) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_signed_area_ccw_triangle() {
        let tri = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let area = signed_area(&tri);
        assert!(area > 0.0, "CCW triangle should have positive signed area");
        assert!(approx_eq(area, 0.5, 1e-12));
    }

    #[test]
    fn test_signed_area_cw_triangle() {
        let tri = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let area = signed_area(&tri);
        assert!(area < 0.0, "CW triangle should have negative signed area");
    }

    #[test]
    fn test_is_convex_square() {
        let sq = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(is_convex(&sq));
    }

    #[test]
    fn test_is_convex_l_shape() {
        let l = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        assert!(!is_convex(&l));
    }

    #[test]
    fn test_point_in_polygon_inside() {
        let sq = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert!(point_in_polygon(1.0, 1.0, &sq));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let sq = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert!(!point_in_polygon(3.0, 1.0, &sq));
    }

    #[test]
    fn test_seg_intersect_crossing() {
        // The diagonals of the unit square cross at their shared midpoint.
        let (t, u) = seg_intersect([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0])
            .expect("crossing segments must produce parameters");
        assert!((t - 0.5).abs() < 1e-12, "expected t = 0.5, got {t}");
        assert!((u - 0.5).abs() < 1e-12, "expected u = 0.5, got {u}");
    }

    #[test]
    fn test_seg_intersect_nearly_parallel_not_dropped() {
        // determinant = 5e-21 (below legacy threshold), but non-zero exactly.
        let (p1, p2) = ([0.0, 0.0], [1.0e-10, 1.0e-10]);
        let (p3, p4) = ([0.0, 1.0e-10], [2.0e-10, 3.5e-10]);
        let (t, u) = seg_intersect(p1, p2, p3, p4)
            .expect("non-parallel directions must not be rejected by epsilon threshold");

        // The parameters must place both segments at the same point.
        for axis in 0..2 {
            let on_first = p1[axis] + t * (p2[axis] - p1[axis]);
            let on_second = p3[axis] + u * (p4[axis] - p3[axis]);
            assert!(
                (on_first - on_second).abs() <= 1.0e-22,
                "axis {axis}: t={t} gives {on_first}, u={u} gives {on_second}"
            );
        }
    }

    #[test]
    fn test_winding_ccw_ccw_and_cw_triangles() {
        let ccw = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        assert_eq!(winding_ccw(&ccw), Some(true));
        assert_eq!(winding_ccw(&cw), Some(false));
    }

    #[test]
    fn test_winding_ccw_degenerate_is_none() {
        let collinear = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        assert_eq!(winding_ccw(&collinear), None);
        assert_eq!(winding_ccw(&[[0.0, 0.0], [1.0, 1.0]]), None);
    }

    #[test]
    fn test_ensure_ccw_normalises_cw_polygon() {
        let mut cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        ensure_ccw(&mut cw);
        assert_eq!(winding_ccw(&cw), Some(true));
    }

    #[test]
    fn test_point_in_triangle_inside_on_edge_and_outside() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert!(point_in_triangle(&[0.25, 0.25], &a, &b, &c), "interior");
        assert!(point_in_triangle(&[0.5, 0.0], &a, &b, &c), "on edge a→b");
        assert!(!point_in_triangle(&[1.0, 1.0], &a, &b, &c), "exterior");
    }
}
