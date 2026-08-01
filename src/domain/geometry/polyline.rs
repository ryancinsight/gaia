//! Validated open polyline geometry.
//!
//! A [`Polyline`] is an ordered sequence of finite 3-D points connected by
//! straight segments. It is open: the final point is not connected back to the
//! first unless callers include that closing point explicitly.

use eunomia::NumericElement;
use leto::geometry::Point3;
use thiserror::Error as ThisError;

use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::Aabb;

/// Error returned when constructing a [`Polyline`].
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum PolylineError {
    /// Fewer than two points were provided.
    #[error("polyline requires at least 2 points, got {0}")]
    TooFewPoints(usize),
    /// A point contains NaN or infinity.
    #[error("polyline point at index {index} is not finite")]
    NonFinitePoint {
        /// Input point index.
        index: usize,
    },
}

/// An immutable open polyline in three-dimensional space.
///
/// Construction validates at least two finite points. Storage is frozen as a
/// boxed slice so retained capacity cannot exceed the vertex count.
///
/// # Examples
///
/// ```
/// use gaia::Polyline;
/// use leto::geometry::Point3;
///
/// let line = Polyline::<f64>::new(vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(3.0, 4.0, 0.0),
/// ])?;
/// assert_eq!(line.segment_count(), 1);
/// assert!((line.arc_length() - 5.0).abs() < f64::EPSILON);
/// # Ok::<(), gaia::PolylineError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Polyline<T: Scalar = f64> {
    points: Box<[Point3<T>]>,
}

impl<T: Scalar> Polyline<T> {
    /// Create a polyline from ordered vertex positions.
    ///
    /// # Errors
    ///
    /// Returns [`PolylineError::TooFewPoints`] for fewer than two points or
    /// [`PolylineError::NonFinitePoint`] for the first NaN/infinite point.
    pub fn new(points: Vec<Point3<T>>) -> Result<Self, PolylineError> {
        if points.len() < 2 {
            return Err(PolylineError::TooFewPoints(points.len()));
        }
        if let Some((index, _)) = points.iter().enumerate().find(|(_, point)| {
            !<T as NumericElement>::is_finite(point.x)
                || !<T as NumericElement>::is_finite(point.y)
                || !<T as NumericElement>::is_finite(point.z)
        }) {
            return Err(PolylineError::NonFinitePoint { index });
        }
        Ok(Self {
            points: points.into_boxed_slice(),
        })
    }

    /// Number of vertices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether this polyline contains no vertices.
    ///
    /// A successfully constructed polyline is never empty; this method keeps
    /// the collection-style `len`/`is_empty` contract complete.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of straight segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.points.len() - 1
    }

    /// Borrow the ordered vertex positions.
    #[must_use]
    pub fn points(&self) -> &[Point3<T>] {
        &self.points
    }

    /// First vertex.
    #[must_use]
    pub fn first(&self) -> Point3<T> {
        self.points
            .first()
            .copied()
            .expect("invariant: a polyline contains at least two points")
    }

    /// Last vertex.
    #[must_use]
    pub fn last(&self) -> Point3<T> {
        self.points
            .last()
            .copied()
            .expect("invariant: a polyline contains at least two points")
    }

    /// Sum of Euclidean segment lengths.
    #[must_use]
    pub fn arc_length(&self) -> T {
        self.points
            .windows(2)
            .map(|segment| (segment[1] - segment[0]).norm())
            .fold(<T as NumericElement>::ZERO, |total, length| total + length)
    }

    /// Axis-aligned bounding box enclosing every vertex.
    #[must_use]
    pub fn aabb(&self) -> Aabb<T> {
        Aabb::from_points(self.points.iter())
    }

    /// Iterate over consecutive segment endpoints.
    pub fn segments(&self) -> impl Iterator<Item = (&Point3<T>, &Point3<T>)> {
        self.points
            .windows(2)
            .map(|segment| (&segment[0], &segment[1]))
    }
}

#[cfg(test)]
mod tests;
