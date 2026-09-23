//! Plane representation and polygon splitting.
//!
//! Adapted from csgrs's BSP plane, but operating on indexed vertices rather
//! than owned `Polygon<S>` structs. The plane is defined by the Hessian
//! normal form: `n · x + d = 0`.

use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::{Point3, Vector3};

/// Classification of a point relative to a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointClassification {
    /// On the positive (front) side of the plane.
    Front,
    /// On the negative (back) side of the plane.
    Back,
    /// Within tolerance of the plane.
    Coplanar,
}

/// An oriented plane in Hessian normal form: `normal · x + w = 0`.
///
/// Generic over the scalar seam with an `f64` default; the classification
/// tolerance is [`Scalar::tolerance`], so each precision uses its own
/// documented geometry tolerance (`f64`: 1 nm, `f32`: 10 µm).
#[derive(Clone, Copy, Debug)]
pub struct Plane<T = Real> {
    /// Unit normal vector.
    pub normal: Vector3<T>,
    /// Signed distance from origin (w = -normal · `point_on_plane`).
    pub w: T,
}

impl<T: Scalar> Plane<T> {
    /// Create a plane from a normal and signed distance.
    #[inline]
    #[must_use]
    pub fn new(normal: Vector3<T>, w: T) -> Self {
        Self { normal, w }
    }

    /// Create a plane from a normal and a point on the plane.
    #[inline]
    #[must_use]
    pub fn from_normal_and_point(normal: Vector3<T>, point: &Point3<T>) -> Self {
        let n = normal.normalize();
        let w = -n.dot(point.coords);
        Self { normal: n, w }
    }

    /// Create a plane from three non-collinear points (CCW winding → outward normal).
    #[inline]
    #[must_use]
    pub fn from_three_points(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> Option<Self> {
        let ab = b - a;
        let ac = c - a;
        let cross = ab.cross(ac);
        let len = cross.norm();
        if len < T::tolerance() {
            return None; // Degenerate (collinear points)
        }
        let normal = cross / len;
        let w = -normal.dot(a.coords);
        Some(Self { normal, w })
    }

    /// Signed distance from a point to this plane.
    ///
    /// Positive = front side, negative = back side, ~0 = coplanar.
    #[inline]
    #[must_use]
    pub fn signed_distance(&self, point: &Point3<T>) -> T {
        self.normal.dot(point.coords) + self.w
    }

    /// Classify a point relative to this plane (using the scalar's tolerance).
    #[inline]
    #[must_use]
    pub fn classify_point(&self, point: &Point3<T>) -> PointClassification {
        self.classify_point_with_eps(point, T::tolerance())
    }

    /// Classify a point with a custom epsilon (used by BSP operations).
    #[inline]
    #[must_use]
    pub fn classify_point_with_eps(&self, point: &Point3<T>, eps: T) -> PointClassification {
        let dist = self.signed_distance(point);
        if dist > eps {
            PointClassification::Front
        } else if dist < -eps {
            PointClassification::Back
        } else {
            PointClassification::Coplanar
        }
    }

    /// Flip the plane (reverse normal and w).
    #[inline]
    #[must_use]
    pub fn flip(&self) -> Self {
        Self {
            normal: -self.normal,
            w: -self.w,
        }
    }

    /// Compute the intersection parameter `t` along the line segment `a → b`.
    ///
    /// Returns `None` if the segment is parallel to the plane.
    #[inline]
    #[must_use]
    pub fn intersect_segment(&self, a: &Point3<T>, b: &Point3<T>) -> Option<T> {
        self.intersect_segment_with_eps(a, b, T::tolerance())
    }

    /// Compute the intersection parameter with a custom epsilon.
    #[inline]
    #[must_use]
    pub fn intersect_segment_with_eps(&self, a: &Point3<T>, b: &Point3<T>, eps: T) -> Option<T> {
        let da = self.signed_distance(a);
        let db = self.signed_distance(b);
        let denom = da - db;
        if denom.abs() < eps {
            return None;
        }
        let t = da / denom;
        // Clamp to `[0,1]` to avoid extrapolation from accumulated FP error
        Some(
            t.max(<T as NumericElement>::ZERO)
                .min(<T as NumericElement>::ONE),
        )
    }
}

impl<T: Scalar> PartialEq for Plane<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let tol = T::tolerance();
        (self.normal - other.normal).norm() < tol && (self.w - other.w).abs() < tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::{Point3r, Vector3r};

    fn basis_plane<T: Scalar>() -> Plane<T> {
        let f = <T as Scalar>::from_f64;
        Plane::from_three_points(
            &Point3::new(f(0.0), f(0.0), f(0.0)),
            &Point3::new(f(1.0), f(0.0), f(0.0)),
            &Point3::new(f(0.0), f(1.0), f(0.0)),
        )
        .expect("invariant: the basis triangle is non-degenerate")
    }

    #[test]
    fn from_three_points_classifies_corners() {
        let plane = Plane::from_three_points(
            &Point3r::new(0.0, 0.0, 0.0),
            &Point3r::new(1.0, 0.0, 0.0),
            &Point3r::new(0.0, 1.0, 0.0),
        )
        .expect("invariant: the basis triangle is non-degenerate");
        assert_eq!(
            plane.classify_point(&Point3r::new(0.0, 0.0, 1.0)),
            PointClassification::Front
        );
        assert_eq!(
            plane.classify_point(&Point3r::new(0.0, 0.0, -1.0)),
            PointClassification::Back
        );
        assert_eq!(
            plane.classify_point(&Point3r::new(0.25, 0.75, 0.0)),
            PointClassification::Coplanar
        );
    }

    #[test]
    fn degenerate_triple_is_rejected() {
        assert!(Plane::from_three_points(
            &Point3r::new(0.0, 0.0, 0.0),
            &Point3r::new(1.0, 1.0, 1.0),
            &Point3r::new(2.0, 2.0, 2.0)
        )
        .is_none());
    }

    #[test]
    fn intersect_segment_midpoint_parameter() {
        let plane = Plane::new(Vector3r::new(0.0, 0.0, 1.0), 0.0);
        let t = plane
            .intersect_segment(&Point3r::new(0.0, 0.0, -1.0), &Point3r::new(0.0, 0.0, 1.0))
            .expect("invariant: the segment crosses the plane");
        assert!((t - 0.5).abs() < 1e-12);
    }

    /// The scalar seam monomorphizes: dyadic coordinates classify identically
    /// at `f32` and `f64` — every input value is exact at both precisions and
    /// the comparison distances dominate both tolerances.
    #[test]
    fn f32_instantiation_matches_f64_on_dyadic_inputs() {
        let p32 = basis_plane::<f32>();
        let p64 = basis_plane::<f64>();
        let q32 = Point3::<f32>::new(0.25, 0.25, 0.5);
        let q64 = Point3::<f64>::new(0.25, 0.25, 0.5);
        assert_eq!(p32.classify_point(&q32), p64.classify_point(&q64));
        assert_eq!(p32.classify_point(&q32), PointClassification::Front);
        let below32 = Point3::<f32>::new(0.0, 0.0, -0.5);
        let below64 = Point3::<f64>::new(0.0, 0.0, -0.5);
        assert_eq!(p32.classify_point(&below32), p64.classify_point(&below64));
    }
}
