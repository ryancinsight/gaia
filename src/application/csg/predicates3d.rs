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
//! The helpers are generic over the `Scalar` seam: the projected orientation
//! tests are exact for the stored precision (ADR 0005), and the segment
//! parameter arithmetic executes in `T`. The `f64` CSG pipelines instantiate
//! `T = Real` without annotation.
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

use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::predicates::{orient_2d_arr, Orientation};
use leto::geometry::Point3;

/// Exact 3-D triangle degeneracy check.
#[inline]
pub(crate) fn triangle_is_degenerate_exact<T: Scalar>(
    a: &Point3<T>,
    b: &Point3<T>,
    c: &Point3<T>,
) -> bool {
    orient_2d_arr([a.x, a.y], [b.x, b.y], [c.x, c.y]) == Orientation::Degenerate
        && orient_2d_arr([a.x, a.z], [b.x, b.z], [c.x, c.z]) == Orientation::Degenerate
        && orient_2d_arr([a.y, a.z], [b.y, b.z], [c.y, c.z]) == Orientation::Degenerate
}

/// Exact 3-D collinearity check for point `p` with segment `(a,b)`.
#[inline]
pub(crate) fn collinear_3d_exact<T: Scalar>(a: &Point3<T>, b: &Point3<T>, p: &Point3<T>) -> bool {
    orient_2d_arr([a.x, a.y], [b.x, b.y], [p.x, p.y]) == Orientation::Degenerate
        && orient_2d_arr([a.x, a.z], [b.x, b.z], [p.x, p.z]) == Orientation::Degenerate
        && orient_2d_arr([a.y, a.z], [b.y, b.z], [p.y, p.z]) == Orientation::Degenerate
}

/// Exact strict-interior point-on-segment parameter.
///
/// Returns `Some(t)` only when `p = a + t*(b-a)` with `0 < t < 1`, evaluated in
/// native `T` arithmetic after the exact collinearity gate.
#[inline]
pub(crate) fn point_on_segment_exact<T: Scalar>(
    a: &Point3<T>,
    b: &Point3<T>,
    p: &Point3<T>,
) -> Option<T> {
    if !collinear_3d_exact(a, b, p) {
        return None;
    }
    let edge = *b - *a;
    let edge_len_sq = edge.dot(edge);
    if edge_len_sq <= <T as eunomia::NumericElement>::ZERO {
        return None;
    }
    let t = (*p - *a).dot(edge) / edge_len_sq;
    if t > <T as eunomia::NumericElement>::ZERO && t < <T as eunomia::NumericElement>::ONE {
        Some(t)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Point3<f64> {
        Point3::new(x, y, z)
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

    /// The helpers monomorphize over the `Scalar` seam; the `f32`
    /// instantiation must produce the same segment parameter the `f64`
    /// instantiation computes on dyadic (exactly representable) inputs.
    #[test]
    fn f32_instantiation_matches_f64_on_dyadic_inputs() {
        let a32 = Point3::new(0.0_f32, 0.0, 0.0);
        let b32 = Point3::new(0.5_f32, 0.25, 0.125);
        let p32 = Point3::new(0.25_f32, 0.125, 0.0625);
        let t32 = point_on_segment_exact(&a32, &b32, &p32);
        let t64 = point_on_segment_exact(
            &Point3::new(f64::from(a32.x), f64::from(a32.y), f64::from(a32.z)),
            &Point3::new(f64::from(b32.x), f64::from(b32.y), f64::from(b32.z)),
            &Point3::new(f64::from(p32.x), f64::from(p32.y), f64::from(p32.z)),
        );
        let (Some(t32), Some(t64)) = (t32, t64) else {
            panic!("invariant: the dyadic midpoint lies strictly inside the segment at both precisions");
        };
        assert_eq!(
            t32.to_bits(),
            <f32 as Scalar>::from_f64(t64).to_bits(),
            "the exact parameter must be precision-independent"
        );
        assert!(
            triangle_is_degenerate_exact(&a32, &b32, &p32),
            "invariant: a point on the segment makes the triple exactly collinear"
        );
    }
}
