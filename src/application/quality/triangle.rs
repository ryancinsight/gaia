//! Per-triangle quality measurements.
//!
//! ## DRY / SSOT note
//!
//! All angle-based metrics (`min_angle`, `max_angle`, `equiangle_skewness`)
//! delegate to the private `triangle_angles` helper which computes the three
//! interior angles **exactly once** per triangle.  Previously each function
//! recomputed the same six normalized edge vectors independently.

use crate::domain::core::scalar::{Point3r, Real, Scalar};
use eunomia::{NumericElement, RealField};

// ── Private helper ─────────────────────────────────────────────────────────────

/// Compute the three interior angles of the triangle `(a, b, c)` in radians.
///
/// Returns `[angle_at_a, angle_at_b, angle_at_c]`.
///
/// Uses clamped `acos` to guard against floating-point values outside `[-1, 1]`
/// produced by roundoff. Degenerate triangles return `[NaN; 3]` because at
/// least one angle is undefined.
///
/// # Theorem — Interior Angle Sum
///
/// For a non-degenerate Euclidean triangle, the three interior angles sum to
/// exactly π radians.  The implementation derives all three angles from the
/// same three edge vectors, so the computed values share one geometric source
/// of truth; tests assert the π-sum and equilateral angle contracts within a
/// bounded floating-point tolerance.
#[inline]
pub(crate) fn triangle_angles<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> [T; 3] {
    let ab = b - a;
    let ac = c - a;
    let bc = c - b;

    let lab = ab.norm();
    let lac = ac.norm();
    let lbc = bc.norm();

    if lab <= <T as NumericElement>::ZERO
        || lac <= <T as NumericElement>::ZERO
        || lbc <= <T as NumericElement>::ZERO
    {
        return [<T as NumericElement>::NAN; 3];
    }

    #[inline]
    fn angle<T: Scalar>(dot: T, lhs_len: T, rhs_len: T) -> T {
        (dot / (lhs_len * rhs_len))
            .clamp(-<T as NumericElement>::ONE, <T as NumericElement>::ONE)
            .acos()
    }

    [
        angle(ab.dot(ac), lab, lac),     // angle at A
        angle((-ab).dot(bc), lab, lbc),  // angle at B
        angle((-ac).dot(-bc), lac, lbc), // angle at C
    ]
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Triangle aspect ratio: `longest_edge / shortest_altitude`.
///
/// A perfect equilateral triangle has aspect ratio ~1.155.
/// CFD meshes typically want aspect ratio < 5.
#[inline]
#[must_use]
pub(crate) fn aspect_ratio_native<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> T {
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let edges = [ab.norm(), bc.norm(), ca.norm()];
    let longest = edges
        .iter()
        .copied()
        .fold(<T as NumericElement>::ZERO, |lhs, rhs| lhs.max_scalar(rhs));
    // ab already computed — reuse for cross product to avoid a second subtraction.
    let area_twice = ab.cross(-ca).norm();

    // Compare the area against a scale-relative bound.  An absolute epsilon
    // incorrectly classifies valid micro-scale or macro-scale triangles.
    let scale = longest * longest;
    if longest <= <T as NumericElement>::ZERO || area_twice <= <T as RealField>::EPSILON * scale {
        return <T as NumericElement>::INFINITY; // Degenerate
    }

    let shortest_alt = area_twice / longest;
    longest / shortest_alt
}

/// Minimum interior angle of a triangle (in radians).
#[inline]
#[must_use]
pub(crate) fn min_angle_native<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> T {
    let [a0, a1, a2] = triangle_angles(a, b, c);
    a0.min(a1).min(a2)
}

/// Maximum interior angle of a triangle (in radians).
#[inline]
#[must_use]
pub(crate) fn max_angle_native<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> T {
    let [a0, a1, a2] = triangle_angles(a, b, c);
    a0.max(a1).max(a2)
}

/// Equiangle skewness: measures deviation from equilateral.
///
/// Range: 0 (equilateral) → 1 (degenerate).
///
/// Computed from a single `triangle_angles` call — min and max are
/// both extracted from the same `[Real; 3]` array, avoiding the prior 2×
/// angle computation overhead.
#[inline]
#[must_use]
pub(crate) fn equiangle_skewness_native<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> T {
    let ideal = <T as RealField>::PI / <T as Scalar>::from_f64(3.0); // 60° for equilateral triangle
    let angles = triangle_angles(a, b, c);
    let max_a = angles
        .iter()
        .copied()
        .fold(-<T as NumericElement>::INFINITY, |lhs, rhs| {
            lhs.max_scalar(rhs)
        });
    let min_a = angles
        .iter()
        .copied()
        .fold(<T as NumericElement>::INFINITY, |lhs, rhs| {
            lhs.min_scalar(rhs)
        });

    let skew_max = (max_a - ideal) / (<T as RealField>::PI - ideal);
    let skew_min = (ideal - min_a) / ideal;

    if skew_max.is_nan() || skew_min.is_nan() {
        <T as NumericElement>::NAN
    } else {
        skew_max.max_scalar(skew_min)
    }
}

/// Edge length ratio: `shortest_edge / longest_edge`.
///
/// 1.0 = all edges equal, approaching 0 = bad quality.
#[inline]
#[must_use]
pub(crate) fn edge_length_ratio_native<T: Scalar>(
    a: &leto::geometry::Point3<T>,
    b: &leto::geometry::Point3<T>,
    c: &leto::geometry::Point3<T>,
) -> T {
    let edges = [(b - a).norm(), (c - b).norm(), (a - c).norm()];
    let shortest = edges
        .iter()
        .copied()
        .fold(<T as NumericElement>::INFINITY, |lhs, rhs| {
            lhs.min_scalar(rhs)
        });
    let longest = edges
        .iter()
        .copied()
        .fold(<T as NumericElement>::ZERO, |lhs, rhs| lhs.max_scalar(rhs));

    if longest <= <T as NumericElement>::ZERO {
        return <T as NumericElement>::ZERO;
    }

    shortest / longest
}

/// Triangle aspect ratio: `longest_edge / shortest_altitude`.
#[inline]
#[must_use]
pub fn aspect_ratio(a: &Point3r, b: &Point3r, c: &Point3r) -> Real {
    aspect_ratio_native(a, b, c)
}

/// Minimum interior angle of a triangle (in radians).
#[inline]
#[must_use]
pub fn min_angle(a: &Point3r, b: &Point3r, c: &Point3r) -> Real {
    min_angle_native(a, b, c)
}

/// Maximum interior angle of a triangle (in radians).
#[inline]
#[must_use]
pub fn max_angle(a: &Point3r, b: &Point3r, c: &Point3r) -> Real {
    max_angle_native(a, b, c)
}

/// Equiangle skewness: deviation from an equilateral triangle.
#[inline]
#[must_use]
pub fn equiangle_skewness(a: &Point3r, b: &Point3r, c: &Point3r) -> Real {
    equiangle_skewness_native(a, b, c)
}

/// Edge length ratio: `shortest_edge / longest_edge`.
#[inline]
#[must_use]
pub fn edge_length_ratio(a: &Point3r, b: &Point3r, c: &Point3r) -> Real {
    edge_length_ratio_native(a, b, c)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::{Point3r, Real};

    fn equilateral() -> (Point3r, Point3r, Point3r) {
        use std::f64::consts::SQRT_2;
        let s = SQRT_2;
        (
            Point3r::new(0.0, 0.0, 0.0),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(
                0.5,
                s * 0.5 * std::f64::consts::FRAC_1_SQRT_2 * 2.0_f64.sqrt(),
                0.0,
            ),
        )
    }

    /// triangle_angles sums to π for an equilateral triangle.
    #[test]
    fn triangle_angles_sum_to_pi_for_equilateral() {
        let (a, b, c) = equilateral();
        let angles = triangle_angles(&a, &b, &c);
        let sum: Real = angles.iter().sum();
        assert!(
            (sum - std::f64::consts::PI).abs() < 1e-10,
            "angle sum {sum} should be π"
        );
    }

    /// Equilateral triangle has all angles equal to π/3.
    #[test]
    fn triangle_angles_equilateral_all_sixty_degrees() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
        let [a0, a1, a2] = triangle_angles(&a, &b, &c);
        let sixty = std::f64::consts::PI / 3.0;
        assert!((a0 - sixty).abs() < 1e-10, "angle at A = {a0}");
        assert!((a1 - sixty).abs() < 1e-10, "angle at B = {a1}");
        assert!((a2 - sixty).abs() < 1e-10, "angle at C = {a2}");
    }

    /// min_angle and max_angle agree for equilateral triangle.
    #[test]
    fn min_max_angle_consistent_equilateral() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
        let sixty = std::f64::consts::PI / 3.0;
        assert!((min_angle(&a, &b, &c) - sixty).abs() < 1e-10);
        assert!((max_angle(&a, &b, &c) - sixty).abs() < 1e-10);
    }

    /// Equiangle skewness of equilateral triangle is ~0.
    #[test]
    fn skewness_zero_for_equilateral() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
        assert!(
            equiangle_skewness(&a, &b, &c).abs() < 1e-10,
            "skewness of equilateral should be ~0"
        );
    }

    /// Aspect ratio of equilateral triangle ≈ 1.155.
    #[test]
    fn aspect_ratio_equilateral_near_one() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
        let ar = aspect_ratio(&a, &b, &c);
        assert!(
            (ar - 1.1547).abs() < 0.001,
            "equilateral aspect ratio {ar} should be ~1.155"
        );
    }

    /// Edge length ratio of equilateral triangle is 1.0.
    #[test]
    fn edge_ratio_equilateral_is_one() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);
        assert!((edge_length_ratio(&a, &b, &c) - 1.0).abs() < 1e-10);
    }

    /// Degenerate (zero-area) triangle returns INFINITY for aspect ratio.
    #[test]
    fn degenerate_aspect_ratio_is_infinity() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(1.0, 0.0, 0.0);
        let c = Point3r::new(2.0, 0.0, 0.0); // collinear
        assert!(aspect_ratio(&a, &b, &c).is_infinite());
    }

    /// Degenerate angle geometry returns NaN because collinear duplicate points
    /// do not define all three interior angles.
    #[test]
    fn degenerate_triangle_angles_are_nan() {
        let a = Point3r::new(0.0, 0.0, 0.0);
        let b = Point3r::new(0.0, 0.0, 0.0);
        let c = Point3r::new(1.0, 0.0, 0.0);
        let angles = triangle_angles(&a, &b, &c);
        assert!(angles.iter().all(|angle| angle.is_nan()));
    }
}
