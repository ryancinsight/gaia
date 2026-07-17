//! Directed half-lines in affine three-dimensional space.

use crate::domain::core::scalar::{Real, Scalar};
use crate::domain::geometry::Aabb;
use core::fmt;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::geometry::{Point3, UnitVector3, Vector3};

/// Violation of the finite, non-degenerate ray construction contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RayError {
    /// The origin contains a non-finite coordinate.
    NonFiniteOrigin,
    /// The direction contains a non-finite coordinate.
    NonFiniteDirection,
    /// The direction has zero Euclidean length.
    ZeroDirection,
}

impl fmt::Display for RayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteOrigin => formatter.write_str("ray origin must be finite"),
            Self::NonFiniteDirection => formatter.write_str("ray direction must be finite"),
            Self::ZeroDirection => formatter.write_str("ray direction must be non-zero"),
        }
    }
}

impl std::error::Error for RayError {}

/// A directed half-line with a finite origin and unit direction.
///
/// Construction through [`Ray::try_new`] rejects zero-length and
/// non-finite direction vectors, so [`Ray::point_at`] preserves the physical
/// distance parameterization `origin + direction × distance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray<T: Scalar = Real> {
    origin: Point3<T>,
    direction: UnitVector3<T>,
}

impl<T: Scalar> Ray<T> {
    /// Construct a ray from finite origin and non-zero direction coordinates.
    ///
    /// The direction is scaled before normalization, preserving a finite
    /// non-zero vector even when its squared norm would overflow or underflow
    /// in the scalar's native precision.
    ///
    /// # Errors
    ///
    /// Returns [`RayError::NonFiniteOrigin`] or [`RayError::NonFiniteDirection`]
    /// for non-finite coordinates, and [`RayError::ZeroDirection`] for a zero
    /// direction vector.
    pub fn try_new(origin: Point3<T>, direction: Vector3<T>) -> Result<Self, RayError> {
        if !<T as NumericElement>::is_finite(origin.x)
            || !<T as NumericElement>::is_finite(origin.y)
            || !<T as NumericElement>::is_finite(origin.z)
        {
            return Err(RayError::NonFiniteOrigin);
        }
        if !<T as NumericElement>::is_finite(direction.x)
            || !<T as NumericElement>::is_finite(direction.y)
            || !<T as NumericElement>::is_finite(direction.z)
        {
            return Err(RayError::NonFiniteDirection);
        }

        let scale = <T as NumericElement>::abs(direction.x)
            .max_scalar(<T as NumericElement>::abs(direction.y))
            .max_scalar(<T as NumericElement>::abs(direction.z));
        if scale <= <T as NumericElement>::ZERO {
            return Err(RayError::ZeroDirection);
        }

        Ok(Self {
            origin,
            direction: UnitVector3::new_normalize(direction / scale),
        })
    }

    /// Return the point at the supplied signed distance along this ray.
    #[must_use]
    pub fn point_at(self, distance: T) -> Point3<T> {
        self.origin + self.direction.into_inner() * distance
    }

    /// Return the ray origin.
    #[must_use]
    pub const fn origin(self) -> Point3<T> {
        self.origin
    }

    /// Return the unit direction.
    #[must_use]
    pub fn direction(self) -> Vector3<T> {
        self.direction.into_inner()
    }

    /// Intersect this ray with an axis-aligned bounding box (slab method).
    ///
    /// Returns the signed entry and exit distances `(t_enter, t_exit)` measured
    /// along the unit direction with `t_enter <= t_exit`, or `None` when the ray
    /// misses the box. Because the direction is unit-normalized, the distances
    /// are in the ray's world-space length units; an origin inside the box
    /// yields a negative `t_enter`.
    ///
    /// A component parallel to a slab (`direction[axis] == 0`) contributes no
    /// bound but rejects the ray when the origin lies outside that slab. The
    /// unit-direction invariant guarantees at least one non-parallel axis, so a
    /// hit always yields finite distances.
    #[must_use]
    pub fn intersect_aabb(self, aabb: &Aabb<T>) -> Option<(T, T)> {
        let origin = self.origin;
        let dir = self.direction.into_inner();
        let mut t_enter = <T as RealField>::neg_infinity();
        let mut t_exit = <T as RealField>::infinity();
        for axis in 0..3 {
            let o = origin[axis];
            let d = dir[axis];
            let lo = aabb.min[axis];
            let hi = aabb.max[axis];
            if <T as NumericElement>::abs(d) <= <T as NumericElement>::ZERO {
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv = <T as FloatElement>::recip(d);
                let mut t0 = (lo - o) * inv;
                let mut t1 = (hi - o) * inv;
                if t0 > t1 {
                    core::mem::swap(&mut t0, &mut t1);
                }
                t_enter = <T as NumericElement>::max_scalar(t_enter, t0);
                t_exit = <T as NumericElement>::min_scalar(t_exit, t1);
                if t_enter > t_exit {
                    return None;
                }
            }
        }
        Some((t_enter, t_exit))
    }
}

#[cfg(test)]
mod tests {
    use super::{Ray, RayError};
    use crate::domain::core::scalar::Scalar;
    use crate::domain::geometry::Aabb;
    use eunomia::{NumericElement, RealField};
    use leto::geometry::{Point3, Vector3};

    fn scalar<T: Scalar>(value: f64) -> T {
        <T as Scalar>::from_f64(value)
    }

    fn assert_normalization_contract<T: Scalar>() {
        let origin = Point3::new(scalar(1.0), scalar(2.0), scalar(3.0));
        let ray = Ray::<T>::try_new(origin, Vector3::new(scalar(3.0), scalar(4.0), scalar(0.0)))
            .expect("invariant: finite non-zero direction must construct a ray");

        // Dot product, square root, reciprocal, and two products each round once;
        // 16ε bounds that fixed-depth native-precision evaluation chain.
        let norm_error: T = (ray.direction().norm() - <T as NumericElement>::ONE).abs();
        let norm_error_bound: T = scalar::<T>(16.0) * <T as RealField>::EPSILON;
        assert!(norm_error <= norm_error_bound);
        assert_eq!(ray.origin(), origin);
    }

    fn assert_invalid_input_contract<T: Scalar>() {
        assert_eq!(
            Ray::<T>::try_new(Point3::origin(), Vector3::zeros()),
            Err(RayError::ZeroDirection)
        );
        assert_eq!(
            Ray::<T>::try_new(
                Point3::new(<T as RealField>::nan(), scalar(0.0), scalar(0.0)),
                Vector3::new(scalar(1.0), scalar(0.0), scalar(0.0)),
            ),
            Err(RayError::NonFiniteOrigin)
        );
        assert_eq!(
            Ray::<T>::try_new(
                Point3::origin(),
                Vector3::new(<T as RealField>::infinity(), scalar(0.0), scalar(0.0)),
            ),
            Err(RayError::NonFiniteDirection)
        );
    }

    fn assert_point_at_contract<T: Scalar>() {
        let ray = Ray::<T>::try_new(
            Point3::new(scalar(1.0), scalar(2.0), scalar(3.0)),
            Vector3::new(scalar(0.0), scalar(0.0), scalar(2.0)),
        )
        .expect("invariant: finite non-zero direction must construct a ray");

        assert_eq!(
            ray.point_at(scalar(4.0)),
            Point3::new(scalar(1.0), scalar(2.0), scalar(7.0))
        );
    }

    fn unit_box<T: Scalar>() -> Aabb<T> {
        Aabb::new(
            Point3::new(scalar(0.0), scalar(0.0), scalar(0.0)),
            Point3::new(scalar(4.0), scalar(4.0), scalar(4.0)),
        )
    }

    fn ray<T: Scalar>(o: [f64; 3], d: [f64; 3]) -> Ray<T> {
        Ray::try_new(
            Point3::new(scalar(o[0]), scalar(o[1]), scalar(o[2])),
            Vector3::new(scalar(d[0]), scalar(d[1]), scalar(d[2])),
        )
        .expect("invariant: finite non-zero direction must construct a ray")
    }

    fn assert_aabb_hit_contract<T: Scalar>() {
        // Axis-aligned unit direction gives exact slab distances (no rounding):
        // origin x=-5, box spans x in [0, 4] → enter at t=5, exit at t=9.
        let (enter, exit) = ray::<T>([-5.0, 2.0, 2.0], [1.0, 0.0, 0.0])
            .intersect_aabb(&unit_box())
            .expect("ray crosses the box");
        assert_eq!(enter, scalar(5.0));
        assert_eq!(exit, scalar(9.0));
    }

    fn assert_aabb_parallel_miss_contract<T: Scalar>() {
        // Direction parallel to the x-slab with y = 10 outside [0, 4] → miss.
        assert_eq!(
            ray::<T>([-5.0, 10.0, 2.0], [1.0, 0.0, 0.0]).intersect_aabb(&unit_box::<T>()),
            None
        );
    }

    fn assert_aabb_interior_origin_contract<T: Scalar>() {
        // Origin inside the box → negative entry distance, positive exit.
        let (enter, exit) = ray::<T>([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
            .intersect_aabb(&unit_box())
            .expect("ray originates inside the box");
        assert_eq!(enter, scalar(-2.0));
        assert_eq!(exit, scalar(2.0));
    }

    #[test]
    fn ray_contract_holds_for_each_supported_scalar() {
        assert_normalization_contract::<f32>();
        assert_normalization_contract::<f64>();
        assert_invalid_input_contract::<f32>();
        assert_invalid_input_contract::<f64>();
        assert_point_at_contract::<f32>();
        assert_point_at_contract::<f64>();
        assert_aabb_hit_contract::<f32>();
        assert_aabb_hit_contract::<f64>();
        assert_aabb_parallel_miss_contract::<f32>();
        assert_aabb_parallel_miss_contract::<f64>();
        assert_aabb_interior_origin_contract::<f32>();
        assert_aabb_interior_origin_contract::<f64>();
    }
}
