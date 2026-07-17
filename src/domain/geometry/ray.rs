//! Directed half-lines in affine three-dimensional space.

use crate::domain::core::scalar::{Real, Scalar};
use core::fmt;
use eunomia::NumericElement;
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
}

#[cfg(test)]
mod tests {
    use super::{Ray, RayError};
    use crate::domain::core::scalar::Scalar;
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

    #[test]
    fn ray_contract_holds_for_each_supported_scalar() {
        assert_normalization_contract::<f32>();
        assert_normalization_contract::<f64>();
        assert_invalid_input_contract::<f32>();
        assert_invalid_input_contract::<f64>();
        assert_point_at_contract::<f32>();
        assert_point_at_contract::<f64>();
    }
}
