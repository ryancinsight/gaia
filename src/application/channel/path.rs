//! Channel centerline paths.

use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use thiserror::Error as ThisError;

/// Error returned when constructing a channel centerline path.
#[derive(Clone, Debug, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum ChannelPathError {
    /// Fewer than two waypoints were provided.
    #[error("channel path requires at least 2 points, got {0}")]
    TooFewPoints(usize),
    /// A waypoint contains a non-finite coordinate.
    #[error("channel path point at index {index} is not finite")]
    NonFinitePoint {
        /// Index of the invalid waypoint.
        index: usize,
    },
    /// Adjacent waypoints are identical, so the segment has no direction.
    #[error("channel path segment {index} has zero length")]
    DegenerateSegment {
        /// Index of the first waypoint in the zero-length segment.
        index: usize,
    },
}

/// A channel centerline path defined by ordered waypoints.
#[derive(Clone, Debug)]
pub struct ChannelPath {
    /// Ordered waypoints (3D positions of the channel centerline).
    points: Box<[Point3r]>,
}

impl ChannelPath {
    /// Create a path from a set of waypoints.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelPathError::TooFewPoints`] when fewer than two
    /// waypoints are supplied, [`ChannelPathError::NonFinitePoint`] for a
    /// non-finite coordinate, or [`ChannelPathError::DegenerateSegment`] for
    /// adjacent duplicate waypoints.
    pub fn new(points: Vec<Point3r>) -> Result<Self, ChannelPathError> {
        if points.len() < 2 {
            return Err(ChannelPathError::TooFewPoints(points.len()));
        }
        if let Some((index, _)) = points
            .iter()
            .enumerate()
            .find(|(_, point)| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
        {
            return Err(ChannelPathError::NonFinitePoint { index });
        }
        if let Some(index) = points
            .windows(2)
            .position(|window| (window[1] - window[0]).norm_squared() == 0.0)
        {
            return Err(ChannelPathError::DegenerateSegment { index });
        }
        Ok(Self {
            points: points.into_boxed_slice(),
        })
    }

    /// Create a straight-line path between two points.
    pub fn straight(start: Point3r, end: Point3r) -> Result<Self, ChannelPathError> {
        Self::new(vec![start, end])
    }

    /// Get the waypoints.
    #[must_use]
    pub fn points(&self) -> &[Point3r] {
        &self.points
    }

    /// Number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.points.len() - 1
    }

    /// Total path length.
    #[must_use]
    pub fn length(&self) -> Real {
        self.points.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
    }

    /// Direction at a given segment index (normalized).
    #[must_use]
    pub fn segment_direction(&self, segment: usize) -> Option<Vector3r> {
        let end_index = segment.checked_add(1)?;
        let [start, end] = self.points.get(segment..=end_index)? else {
            return None;
        };
        Some((*end - *start).normalize())
    }

    /// Compute a stable local frame at each waypoint.
    ///
    /// Uses averaged tangents for interior points and a fixed up-vector fallback
    /// to keep planar channel sweeps from twisting between segments.
    #[must_use]
    pub fn compute_frames(&self) -> Vec<FrenetFrame> {
        let n = self.points.len();
        let mut frames = Vec::with_capacity(n);

        for i in 0..n {
            let tangent = if i == 0 {
                (self.points[1] - self.points[0]).normalize()
            } else if i == n - 1 {
                (self.points[n - 1] - self.points[n - 2]).normalize()
            } else {
                ((self.points[i + 1] - self.points[i]).normalize()
                    + (self.points[i] - self.points[i - 1]).normalize())
                .normalize()
            };

            // Use a consistent Up vector (Z-axis) to prevent twisting and guarantee perfectly
            // aligned seams across distinct channel segments in planar schematics.
            let mut up = Vector3r::new(0.0, 0.0, 1.0);

            // If tangent is parallel to Z-axis, fallback to X-axis as up
            if tangent.z.abs() > 0.999 {
                up = Vector3r::new(1.0, 0.0, 0.0);
            }

            // local X (normal) is perpendicular to tangent and up
            let normal = up.cross(tangent).normalize();
            // local Y (binormal) is perpendicular to normal and tangent, pointing "up"
            let binormal = tangent.cross(normal).normalize();

            frames.push(FrenetFrame {
                position: self.points[i],
                tangent,
                normal,
                binormal,
            });
        }

        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_waypoint_sequences() {
        let too_few = ChannelPath::new(Vec::new()).expect_err("empty path must fail");
        assert_eq!(too_few, ChannelPathError::TooFewPoints(0));

        let non_finite =
            ChannelPath::new(vec![Point3r::new(f64::NAN, 0.0, 0.0), Point3r::origin()])
                .expect_err("non-finite path must fail");
        assert_eq!(non_finite, ChannelPathError::NonFinitePoint { index: 0 });

        let duplicate = ChannelPath::new(vec![Point3r::origin(), Point3r::origin()])
            .expect_err("zero-length segment must fail");
        assert_eq!(duplicate, ChannelPathError::DegenerateSegment { index: 0 });
    }

    #[test]
    fn stores_valid_paths_without_excess_capacity() {
        let path = ChannelPath::new(vec![
            Point3r::origin(),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(1.0, 1.0, 0.0),
        ])
        .expect("valid path");
        assert_eq!(path.points().len(), 3);
        assert_eq!(path.segment_count(), 2);
        assert_eq!(path.segment_direction(2), None);
        assert_eq!(
            path.segment_direction(0),
            Some(Vector3r::new(1.0, 0.0, 0.0))
        );
    }
}

/// Local coordinate frame at a point along the path.
#[derive(Clone, Debug)]
pub struct FrenetFrame {
    /// Position on the path.
    pub position: Point3r,
    /// Tangent direction (forward along path).
    pub tangent: Vector3r,
    /// Normal direction (perpendicular to tangent).
    pub normal: Vector3r,
    /// Binormal direction (tangent × normal).
    pub binormal: Vector3r,
}
