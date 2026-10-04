//! Sutherland-Hodgman convex polygon clipping.
//!
//! Optimal for convex clip regions: O(n) per clipping edge, O(n·m) total.
//! Only supports convex clip polygons — for concave clips, use the hybrid
//! dispatcher which selects an appropriate general algorithm.
//!
//! # References
//!
//! Sutherland & Hodgman (1974), "Reentrant polygon clipping"

use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;

/// Evaluates the 2-D cross product (unscaled signed distance).
/// MUST be evaluated in standard floats to construct precise `t` interpolations.
#[inline]
fn edge_distance<T: Scalar>(ax: T, ay: T, bx: T, by: T, px: T, py: T) -> T {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

fn sh_clip_halfplane_into<T: Scalar>(
    poly: &[[T; 2]],
    ax: T,
    ay: T,
    bx: T,
    by: T,
    out: &mut Vec<[T; 2]>,
) {
    out.clear();
    if poly.len() < 2 {
        return;
    }
    let intersection_eps: Real = 1e-30;
    let intersection_eps = <T as Scalar>::from_f64(intersection_eps);
    out.reserve(poly.len() + 1);
    let n = poly.len();
    for i in 0..n {
        let s = poly[i];
        let e = poly[(i + 1) % n];
        let sc = edge_distance(ax, ay, bx, by, s[0], s[1]);
        let ec = edge_distance(ax, ay, bx, by, e[0], e[1]);

        // Use the same float values for inside checking to perfectly synchronize
        // with the numeric interpolation branching.
        let s_in = sc >= <T as NumericElement>::ZERO;
        let e_in = ec >= <T as NumericElement>::ZERO;
        match (s_in, e_in) {
            (true, true) => out.push(e),
            (true, false) => {
                let denom = sc - ec;
                if denom.abs() > intersection_eps {
                    let t = sc / denom;
                    out.push([s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t]);
                }
            }
            (false, true) => {
                let denom = sc - ec;
                if denom.abs() > intersection_eps {
                    let t = sc / denom;
                    out.push([s[0] + (e[0] - s[0]) * t, s[1] + (e[1] - s[1]) * t]);
                }
                out.push(e);
            }
            (false, false) => {}
        }
    }
}

/// Clip a polygon against the left half-plane of directed edge (ax,ay)→(bx,by).
///
/// Retained from the original implementation because it is still the canonical polygon clipper.
/// Optimal for convex clip regions (one pass per edge, O(n) total).
pub fn sh_clip_halfplane<T: Scalar>(poly: &[[T; 2]], ax: T, ay: T, bx: T, by: T) -> Vec<[T; 2]> {
    let mut out = Vec::with_capacity(poly.len().saturating_add(1));
    sh_clip_halfplane_into(poly, ax, ay, bx, by, &mut out);
    out
}

/// Clip subject polygon to the inside of a CCW convex clip polygon
/// using iterated Sutherland-Hodgman half-plane clips.
pub fn sh_clip_convex<T: Scalar>(subject: &[[T; 2]], clip: &[[T; 2]]) -> Vec<[T; 2]> {
    let n = clip.len();
    if n < 3 || subject.len() < 3 {
        return Vec::new();
    }
    let mut result = subject.to_vec();
    let mut scratch = Vec::with_capacity(subject.len().saturating_add(1));
    for i in 0..n {
        let j = (i + 1) % n;
        sh_clip_halfplane_into(
            &result,
            clip[i][0],
            clip[i][1],
            clip[j][0],
            clip[j][1],
            &mut scratch,
        );
        if scratch.len() < 3 {
            return Vec::new();
        }
        std::mem::swap(&mut result, &mut scratch);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::super::geometry::polygon_area;
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn sh_clip_two_squares_intersection() {
        let subject = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let clip = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let result = sh_clip_convex(&subject, &clip);
        assert!(result.len() >= 3);
        let area = polygon_area(&result);
        assert!(
            approx_eq(area, 1.0, 0.01),
            "intersection area should be 1.0, got {area}"
        );
    }

    #[test]
    fn sh_clip_triangle_inside_square() {
        let tri = vec![[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]];
        let sq = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let result = sh_clip_convex(&tri, &sq);
        assert!(result.len() >= 3);
        let area = polygon_area(&result);
        let tri_area = polygon_area(&tri);
        assert!(
            approx_eq(area, tri_area, 1e-10),
            "triangle fully inside square"
        );
    }
}
