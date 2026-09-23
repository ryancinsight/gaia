//! Exact geometric predicates for topological validation.
//!
//! Exposes robust exact orientation tests to prevent floating-point
//! heuristics from causing degenerate topological failures like non-manifold
//! edge creation. These functions wrap adaptive multi-precision arithmetic.

use crate::domain::core::scalar::Point3r;

/// Exact algebraic sign representing geometric orientation.
///
/// # Theorem — Geometric Robustness
///
/// Shewchuk's adaptive-precision arithmetic guarantees that the sign of the
/// orientation determinant is computed exactly, even for nearly-collinear or
/// nearly-coplanar point configurations. No epsilon-based fallbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sign {
    /// Points are in clockwise order (negative determinant).
    Negative = -1,
    /// Points are collinear/coplanar (zero determinant).
    Zero = 0,
    /// Points are in counter-clockwise order (positive determinant).
    Positive = 1,
}

impl Sign {
    /// Convert the exact expansion floating-point result into a strict sign.
    #[inline]
    #[must_use]
    pub fn from_exact_f64(v: f64) -> Self {
        if v > 0.0 {
            Sign::Positive
        } else if v < 0.0 {
            Sign::Negative
        } else {
            Sign::Zero
        }
    }

    /// Returns `true` if the sign is positive (counter-clockwise).
    #[inline]
    #[must_use]
    pub fn is_positive(self) -> bool {
        self == Sign::Positive
    }

    /// Returns `true` if the sign is negative (clockwise).
    #[inline]
    #[must_use]
    pub fn is_negative(self) -> bool {
        self == Sign::Negative
    }

    /// Returns `true` if the sign is zero (collinear/coplanar).
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self == Sign::Zero
    }
}

/// Exact 3D orientation predicate.
///
/// Returns whether the point `d` is strictly above, strictly below, or perfectly
/// coplanar with the oriented plane defined by `a`, `b`, and `c`.
///
/// This evaluation is mathematically exact and immune to floating-point epsilon noise.
#[inline]
#[must_use]
pub fn orient3d(a: &Point3r, b: &Point3r, c: &Point3r, d: &Point3r) -> Sign {
    let pa = [a.x, a.y, a.z];
    let pb = [b.x, b.y, b.z];
    let pc = [c.x, c.y, c.z];
    let pd = [d.x, d.y, d.z];

    let det = geometry_predicates::orient3d(pa, pb, pc, pd);
    Sign::from_exact_f64(det)
}

/// Exact 2D orientation predicate (Sutherland-Hodgman / coplanar clipping).
///
/// Returns whether the point `c` lies strictly left, strictly right, or perfectly
/// collinear with the directed line from `a` to `b` in the 2D plane (X-Y).
#[inline]
#[must_use]
pub fn orient2d(a: &Point3r, b: &Point3r, c: &Point3r) -> Sign {
    let pa = [a.x, a.y];
    let pb = [b.x, b.y];
    let pc = [c.x, c.y];

    let det = geometry_predicates::orient2d(pa, pb, pc);
    Sign::from_exact_f64(det)
}

/// Exact incircle predicate in 2D.
#[inline]
#[must_use]
pub fn incircle2d(a: &Point3r, b: &Point3r, c: &Point3r, d: &Point3r) -> Sign {
    let pa = [a.x, a.y];
    let pb = [b.x, b.y];
    let pc = [c.x, c.y];
    let pd = [d.x, d.y];

    let det = geometry_predicates::incircle(pa, pb, pc, pd);
    Sign::from_exact_f64(det)
}

/// Exact insphere predicate in 3D.
#[inline]
#[must_use]
pub fn insphere3d(a: &Point3r, b: &Point3r, c: &Point3r, d: &Point3r, e: &Point3r) -> Sign {
    let pa = [a.x, a.y, a.z];
    let pb = [b.x, b.y, b.z];
    let pc = [c.x, c.y, c.z];
    let pd = [d.x, d.y, d.z];
    let pe = [e.x, e.y, e.z];

    let det = geometry_predicates::insphere(pa, pb, pc, pd, pe);
    Sign::from_exact_f64(det)
}

#[cfg(test)]
mod gaia_004_regression {
    use super::*;
    use crate::domain::geometry::predicates::{
        orient_2d as geo_orient_2d, orient_3d as geo_orient_3d, Orientation,
    };
    use leto::geometry::Point2;

    fn p(x: f64, y: f64, z: f64) -> Point3r {
        Point3r::new(x, y, z)
    }

    /// Naive f64 evaluation of the 2-D orientation determinant — the
    /// non-robust oracle the raw wrappers must beat (GAIA-004).
    fn naive_orient_2d(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> f64 {
        (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    }

    /// The 2⁻¹⁰⁴ cancellation construction: exact determinant positive,
    /// naive f64 cancels to exactly zero. The raw `orient2d` wrapper must
    /// resolve the true sign, proving the check is live at gaia's own
    /// boundary rather than assumed from the dependency.
    #[test]
    fn raw_orient2d_resolves_cancellation_naive_reports_collinear() {
        let t = 1.0 + f64::EPSILON;
        let u = 1.0 + 2.0 * f64::EPSILON;
        let a = p(0.0, 0.0, 0.0);
        let b = p(t, 1.0, 0.0);
        let c = p(u, t, 0.0);

        assert_eq!(naive_orient_2d(0.0, 0.0, t, 1.0, u, t), 0.0);
        assert_eq!(orient2d(&a, &b, &c), Sign::Positive);
    }

    /// The same construction lifted to 3-D: the raw `orient3d` wrapper
    /// follows Shewchuk's convention (positive when `d` is below `abc`),
    /// so the positive signed volume maps to `Sign::Negative` here — the
    /// exact mirror of the geometry module's right-hand-rule wrapper, which
    /// the cross-module pin below asserts pointwise.
    #[test]
    fn raw_orient3d_resolves_cancellation_with_shewchuk_sign() {
        let t = 1.0 + f64::EPSILON;
        let u = 1.0 + 2.0 * f64::EPSILON;
        let a = p(0.0, 0.0, 0.0);
        let b = p(t, 1.0, 0.0);
        let c = p(u, t, 0.0);
        let d = p(0.0, 0.0, 1.0);

        let naive = (b.x * c.y - b.y * c.x) * d.z;
        assert_eq!(naive, 0.0);
        assert_eq!(orient3d(&a, &b, &c, &d), Sign::Negative);
    }

    /// The two predicate modules carry opposite `orient3d` sign conventions
    /// by design: the geometry wrapper negates Shewchuk's result for the
    /// right-hand rule while this module passes it through. The pin holds
    /// pointwise on a generic tet and collapses to agreement on a coplanar
    /// one, so a silent negation drift in either module fails here.
    #[test]
    fn geometry_and_topology_orient3d_conventions_are_pinned_opposite() {
        let cases = [
            (
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            (
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 1.0],
            ),
            (
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ),
        ];
        for (a, b, c, d) in cases {
            let (pa, pb, pc, pd) = (
                p(a[0], a[1], a[2]),
                p(b[0], b[1], b[2]),
                p(c[0], c[1], c[2]),
                p(d[0], d[1], d[2]),
            );
            let raw = orient3d(&pa, &pb, &pc, &pd);
            let geo = geo_orient_3d(a, b, c, d);
            let expected_geo = match raw {
                Sign::Positive => Orientation::Negative,
                Sign::Negative => Orientation::Positive,
                Sign::Zero => Orientation::Degenerate,
            };
            assert_eq!(
                geo, expected_geo,
                "convention pin broke on {a:?} {b:?} {c:?} {d:?}"
            );
        }
    }

    /// `Sign::from_exact_f64` is the raw module's `Orientation::from_det`
    /// analogue: boundary values map to the strict three-way sign.
    #[test]
    fn from_exact_f64_maps_boundary_values() {
        assert_eq!(Sign::from_exact_f64(0.0), Sign::Zero);
        assert_eq!(Sign::from_exact_f64(-0.0), Sign::Zero);
        assert_eq!(Sign::from_exact_f64(f64::MIN_POSITIVE), Sign::Positive);
        assert_eq!(Sign::from_exact_f64(-f64::MIN_POSITIVE), Sign::Negative);
    }

    /// The 2-D permutation law on the raw wrapper: swapping the first two
    /// arguments flips the sign; a collinear triple stays `Zero`.
    #[test]
    fn raw_orient2d_is_alternating_under_transposition() {
        let a = p(0.0, 0.0, 0.0);
        let b = p(1.0, 0.0, 0.0);
        let c = p(0.0, 1.0, 0.0);
        assert_eq!(orient2d(&a, &b, &c), Sign::Positive);
        assert_eq!(orient2d(&b, &a, &c), Sign::Negative);
        // The 3-cycle (a,b,c) -> (c,a,b) is even: orientation preserved.
        assert_eq!(orient2d(&c, &a, &b), Sign::Positive);
        // A transposition flips it.
        assert_eq!(orient2d(&a, &c, &b), Sign::Negative);
        let collinear_c = p(2.0, 0.0, 0.0);
        assert_eq!(orient2d(&a, &b, &collinear_c), Sign::Zero);
        let _ = geo_orient_2d(
            &Point2::new(0.0, 0.0),
            &Point2::new(1.0, 0.0),
            &Point2::new(0.0, 1.0),
        );
    }
}
