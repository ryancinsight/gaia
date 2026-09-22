//! Shared exact 3-D geometric predicates for CSG pipelines.
//!
//! ## Algorithm
//!
//! This module centralizes exact helper predicates used by arrangement,
//! coplanar, and seam-propagation stages:
//!
//! 1. `triangle_is_degenerate_exact`:
//!    exact 3-D triangle degeneracy via three projected `orient_2d_arr` tests.
//! 2. `collinear_3d_exact`:
//!    exact collinearity of point `p` with segment endpoints `(a,b)`.
//! 3. `point_on_segment_exact`:
//!    strict interior point-on-segment (`0 < t < 1`) after exact collinearity.
//!
//! ## Theorem — Projection-Minor Equivalence in 3-D
//!
//! Let vectors `u = b-a` and `v = p-a`.
//! `u` and `v` are linearly dependent in `R^3` iff all three 2x2 minors
//! (`xy`, `xz`, `yz`) vanish. These minors are exactly the determinants tested
//! by `orient_2d_arr` on projected coordinates.
//!
//! **Proof sketch.**
//! Linear dependence in `R^3` is equivalent to rank 1 of matrix `[u v]`.
//! Rank 1 iff every 2x2 minor is zero. Each projected orientation determinant
//! computes one of these minors exactly (Shewchuk predicates). ∎

use crate::domain::core::scalar::{Point3r, Real};
use crate::domain::geometry::predicates::{orient_2d_arr, Orientation};

/// Exact 3-D triangle degeneracy check.
#[inline]
pub(crate) fn triangle_is_degenerate_exact(a: &Point3r, b: &Point3r, c: &Point3r) -> bool {
    orient_2d_arr([a.x, a.y], [b.x, b.y], [c.x, c.y]) == Orientation::Degenerate
        && orient_2d_arr([a.x, a.z], [b.x, b.z], [c.x, c.z]) == Orientation::Degenerate
        && orient_2d_arr([a.y, a.z], [b.y, b.z], [c.y, c.z]) == Orientation::Degenerate
}

/// Exact 3-D collinearity check for point `p` with segment `(a,b)`.
#[inline]
pub(crate) fn collinear_3d_exact(a: &Point3r, b: &Point3r, p: &Point3r) -> bool {
    orient_2d_arr([a.x, a.y], [b.x, b.y], [p.x, p.y]) == Orientation::Degenerate
        && orient_2d_arr([a.x, a.z], [b.x, b.z], [p.x, p.z]) == Orientation::Degenerate
        && orient_2d_arr([a.y, a.z], [b.y, b.z], [p.y, p.z]) == Orientation::Degenerate
}

/// Exact strict-interior point-on-segment parameter.
///
/// Returns `Some(t)` only when `p = a + t*(b-a)` with `0 < t < 1`.
#[inline]
pub(crate) fn point_on_segment_exact(a: &Point3r, b: &Point3r, p: &Point3r) -> Option<Real> {
    if !collinear_3d_exact(a, b, p) {
        return None;
    }
    let edge = *b - *a;
    let edge_len_sq = edge.dot(edge);
    if edge_len_sq <= 0.0 {
        return None;
    }
    let t = (*p - *a).dot(edge) / edge_len_sq;
    if t > 0.0 && t < 1.0 {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: Real, y: Real, z: Real) -> Point3r {
        Point3r::new(x, y, z)
    }

    #[test]
    fn triangle_is_degenerate_exact_detects_collinear_triangle() {
        assert!(triangle_is_degenerate_exact(
            &p(0.0, 0.0, 0.0),
            &p(1.0, 0.0, 0.0),
            &p(2.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn point_on_segment_exact_accepts_true_collinear_midpoint() {
        let t = point_on_segment_exact(&p(0.0, 0.0, 0.0), &p(1.0, 2.0, 3.0), &p(0.5, 1.0, 1.5))
            .expect("a collinear midpoint lies on the segment");
        // The midpoint's parameter is what makes it the midpoint.
        assert!((t - 0.5).abs() < 1e-12, "expected t = 0.5, got {t}");
    }

    #[test]
    fn point_on_segment_exact_rejects_projection_false_positive() {
        let t = point_on_segment_exact(&p(0.0, 0.0, 0.0), &p(1.0, 1.0, 1.0), &p(0.5, 0.5, 0.6));
        assert!(t.is_none());
    }
}
