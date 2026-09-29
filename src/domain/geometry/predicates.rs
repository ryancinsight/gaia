//! # Exact Geometric Predicates
//!
//! Thin wrappers around [Shewchuk's adaptive-precision arithmetic][shewchuk]
//! as provided by the `geometry-predicates` crate.  All functions in this
//! module run in **exact arithmetic** — they never return a wrong sign due to
//! floating-point rounding, even for nearly-degenerate configurations.
//!
//! Every wrapper is generic over
//! [`Scalar`]: coordinates of the
//! caller's precision promote losslessly into the `f64` expansion arithmetic
//! (`f32` is a strict subset of `f64`; `f64` promotes as the identity), so
//! the returned sign is exact for the stored `T`-precision configuration —
//! no decision is made about geometry that is not there (ADR 0005).
//!
//! ## Theorem — Shewchuk Adaptive Arithmetic
//!
//! Let `fl(e)` denote the floating-point evaluation of an expression `e`.
//! Shewchuk's method computes `sign(e)` exactly by maintaining an *expansion*
//! — a non-overlapping sequence of floating-point numbers whose sum equals `e`
//! exactly.  The adaptive stages refine the expansion only until the sign is
//! determined, achieving near-floating-point speed for well-separated inputs
//! while falling back to full multi-precision for near-degenerate cases.
//!
//! ## Available Predicates
//!
//! | Function | Returns | Meaning |
//! |---|---|---|
//! | [`orient_2d`] | [`Orientation`] | Sign of the 2-D cross product |
//! | [`orient_3d`] | [`Orientation`] | Sign of the 3×3 tet volume determinant |
//! | [`incircle`] | [`Orientation`] | Point inside/on/outside circumcircle |
//! | [`insphere`] | [`Orientation`] | Point inside/on/outside circumsphere |
//!
//! ## Diagram
//!
//! ```text
//! orient_2d(a, b, c):
//!
//!      c
//!     /
//!    /  CCW (+) → Positive
//!   a ──── b
//!
//!   det = | ax  ay  1 |
//!         | bx  by  1 |  > 0 ↔ CCW
//!         | cx  cy  1 |
//! ```
//!
//! [shewchuk]: https://www.cs.cmu.edu/~quake/robust.html

use geometry_predicates as gp;
use leto::geometry::Point2;

use crate::domain::core::scalar::Scalar;

/// The sign of an orientation determinant.
///
/// Returned by all predicate functions.  The `Degenerate` variant corresponds
/// to an exact zero determinant (collinear / coplanar / co-spherical inputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Determinant is strictly positive (CCW / above / inside).
    Positive,
    /// Determinant is exactly zero (degenerate configuration).
    Degenerate,
    /// Determinant is strictly negative (CW / below / outside).
    Negative,
}

impl Orientation {
    /// Convert a raw determinant value to an `Orientation`.
    #[inline]
    #[must_use]
    pub fn from_det(d: f64) -> Self {
        if d > 0.0 {
            Orientation::Positive
        } else if d < 0.0 {
            Orientation::Negative
        } else {
            Orientation::Degenerate
        }
    }

    /// Returns `true` if the orientation is [`Positive`](Orientation::Positive).
    #[inline]
    #[must_use]
    pub fn is_positive(self) -> bool {
        self == Orientation::Positive
    }

    /// Returns `true` if the orientation is [`Negative`](Orientation::Negative).
    #[inline]
    #[must_use]
    pub fn is_negative(self) -> bool {
        self == Orientation::Negative
    }

    /// Returns `true` for a degenerate (zero) determinant.
    #[inline]
    #[must_use]
    pub fn is_degenerate(self) -> bool {
        self == Orientation::Degenerate
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Promote a scalar of precision `T` into the predicate's `f64` arithmetic.
///
/// The promotion is exact over the sealed `Scalar` set: every `f32` value is
/// exactly representable in `f64` and `f64` promotes as the identity, so the
/// predicate evaluates the exact sign of the stored `T`-precision
/// configuration — no coordinate is rounded on the way in (ADR 0005).
#[inline]
fn promote<T: Scalar>(v: T) -> f64 {
    <T as eunomia::NumericElement>::to_f64(v)
}

// ── 2-D predicates ────────────────────────────────────────────────────────────

/// **Exact 2-D orientation test.**
///
/// Returns the sign of the 2×2 determinant:
///
/// ```text
/// | ax - cx   ay - cy |
/// | bx - cx   by - cy |
/// ```
///
/// - [`Positive`][Orientation::Positive] — `a`, `b`, `c` are in CCW order.
/// - [`Negative`][Orientation::Negative] — `a`, `b`, `c` are in CW order.
/// - [`Degenerate`][Orientation::Degenerate] — `a`, `b`, `c` are collinear.
///
/// # Example
/// ```rust
/// use gaia::domain::geometry::predicates::{orient_2d, Orientation};
/// use leto::geometry::Point2;
///
/// let a = Point2::new(0.0_f64, 0.0);
/// let b = Point2::new(1.0, 0.0);
/// let c = Point2::new(0.0, 1.0);
/// assert_eq!(orient_2d(&a, &b, &c), Orientation::Positive); // CCW
/// ```
#[inline]
#[must_use]
pub fn orient_2d<T: Scalar>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> Orientation {
    let det = gp::orient2d(
        [promote(a.x), promote(a.y)],
        [promote(b.x), promote(b.y)],
        [promote(c.x), promote(c.y)],
    );
    Orientation::from_det(det)
}

/// **Exact 2-D orientation test from raw arrays.**
///
/// Convenience overload accepting `[T; 2]` arrays.
#[inline]
#[must_use]
pub fn orient_2d_arr<T: Scalar>(a: [T; 2], b: [T; 2], c: [T; 2]) -> Orientation {
    let det = gp::orient2d(
        [promote(a[0]), promote(a[1])],
        [promote(b[0]), promote(b[1])],
        [promote(c[0]), promote(c[1])],
    );
    Orientation::from_det(det)
}

// ── 3-D predicates ────────────────────────────────────────────────────────────

/// **Exact 3-D orientation test.**
///
/// Returns the sign of the **signed volume** of the tetrahedron `(a, b, c, d)`:
///
/// - [`Positive`][Orientation::Positive] — `d` lies *above* the plane `abc`
///   when `a→b→c` is counter-clockwise (right-hand rule); positive signed volume.
/// - [`Negative`][Orientation::Negative] — `d` lies *below* the plane.
/// - [`Degenerate`][Orientation::Degenerate] — all four points are coplanar.
///
/// Used in the BVH Boolean narrow phase to determine on which side of a
/// supporting plane a vertex lies.
///
/// # Example
/// ```rust
/// use gaia::domain::geometry::predicates::{orient_3d, Orientation};
///
/// // Tetrahedron with positive volume
/// let o  = [0.0_f64, 0.0, 0.0];
/// let ex = [1.0, 0.0, 0.0];
/// let ey = [0.0, 1.0, 0.0];
/// let ez = [0.0, 0.0, 1.0];
/// assert_eq!(orient_3d(o, ex, ey, ez), Orientation::Positive);
/// ```
#[inline]
#[must_use]
pub fn orient_3d<T: Scalar>(a: [T; 3], b: [T; 3], c: [T; 3], d: [T; 3]) -> Orientation {
    let to64 = |v: [T; 3]| [promote(v[0]), promote(v[1]), promote(v[2])];
    // gp::orient3d returns positive when d is *below* the abc plane (Shewchuk
    // convention).  We negate so that our API convention is "d above = Positive",
    // which matches standard signed-volume / right-hand-rule intuition.
    let det = -gp::orient3d(to64(a), to64(b), to64(c), to64(d));
    Orientation::from_det(det)
}

/// Convenience wrapper: accept `leto::geometry::Point3<T>`.
#[inline]
#[must_use]
pub fn orient_3d_pts<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
    d: &leto::geometry::Point3<T>,
) -> Orientation {
    orient_3d(
        [a.x, a.y, a.z],
        [b.x, b.y, b.z],
        [c.x, c.y, c.z],
        [d.x, d.y, d.z],
    )
}

// ── In-circle / In-sphere predicates ─────────────────────────────────────────

/// **Exact in-circle test.**
///
/// Returns the sign of the 3×3 determinant that determines whether point `d`
/// lies inside, on, or outside the circumcircle of triangle `abc` (with `abc`
/// in CCW order):
///
/// ```text
/// | ax - dx   ay - dy   (ax²+ay²) - (dx²+dy²) |
/// | bx - dx   by - dy   (bx²+by²) - (dx²+dy²) |
/// | cx - dx   cy - dy   (cx²+cy²) - (dx²+dy²) |
/// ```
///
/// - [`Positive`][Orientation::Positive] — `d` is strictly inside the circumcircle.
/// - [`Negative`][Orientation::Negative] — `d` is strictly outside.
/// - [`Degenerate`][Orientation::Degenerate] — `d` lies exactly on the circle.
///
/// Used in Delaunay mesh refinement to enforce the empty-circumcircle property.
#[inline]
#[must_use]
pub fn incircle<T: Scalar>(
    a: &Point2<T>,
    b: &Point2<T>,
    c: &Point2<T>,
    d: &Point2<T>,
) -> Orientation {
    let det = gp::incircle(
        [promote(a.x), promote(a.y)],
        [promote(b.x), promote(b.y)],
        [promote(c.x), promote(c.y)],
        [promote(d.x), promote(d.y)],
    );
    Orientation::from_det(det)
}

/// **Exact in-sphere test.**
///
/// Returns the sign of the 4×4 determinant that determines whether point `e`
/// lies inside, on, or outside the circumsphere of tetrahedron `abcd` (with
/// `abcd` in positive orientation):
///
/// - [`Positive`][Orientation::Positive] — `e` is strictly inside the sphere.
/// - [`Negative`][Orientation::Negative] — `e` is strictly outside.
/// - [`Degenerate`][Orientation::Degenerate] — `e` lies on the sphere.
///
/// Used in 3-D Delaunay mesh generation to enforce the Delaunay property.
#[must_use]
pub fn insphere<T: Scalar>(a: [T; 3], b: [T; 3], c: [T; 3], d: [T; 3], e: [T; 3]) -> Orientation {
    let to64 = |v: [T; 3]| [promote(v[0]), promote(v[1]), promote(v[2])];
    let det = gp::insphere(to64(a), to64(b), to64(c), to64(d), to64(e));
    Orientation::from_det(det)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_predicates.rs"]
mod tests;
