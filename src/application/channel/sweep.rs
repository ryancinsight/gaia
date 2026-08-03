//! Profile sweep along a path to generate channel geometry.
//!
//! This is the core extrusion engine — the equivalent of blue2mesh's
//! `ExtrusionEngine` but producing indexed mesh faces directly.

use crate::application::channel::path::ChannelPath;
use crate::application::channel::profile::ChannelProfile;
use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::Real;
use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use thiserror::Error as ThisError;

/// Error returned when sweep inputs violate their structural contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ThisError)]
#[non_exhaustive]
pub enum SweepError {
    /// Width scales must provide one value for each path station.
    #[error("sweep width scale count {actual} does not match path station count {expected}")]
    WidthScaleCountMismatch {
        /// Number of path stations required by the sweep.
        expected: usize,
        /// Number of supplied width scales.
        actual: usize,
    },
}

/// Sweep mesher: sweeps a 2D profile along a 3D path.
pub struct SweepMesher {
    /// Whether to cap the start of the sweep.
    pub cap_start: bool,
    /// Whether to cap the end of the sweep.
    pub cap_end: bool,
}

impl SweepMesher {
    /// Create with default settings (both ends capped).
    #[must_use]
    pub fn new() -> Self {
        Self {
            cap_start: true,
            cap_end: true,
        }
    }

    /// Sweep a profile along a path, producing indexed faces.
    ///
    /// Returns the list of generated faces. New vertices are inserted into
    /// `vertex_pool` via welding.
    pub fn sweep(
        &self,
        profile: &ChannelProfile,
        path: &ChannelPath,
        vertex_pool: &mut VertexPool,
        region: RegionId,
    ) -> Vec<FaceData> {
        self.sweep_inner(profile, path, |_| 1.0, vertex_pool, region)
    }

    /// Sweep a profile with variable width scaling along a path.
    ///
    /// `width_scales` must have the same length as the path stations.
    ///
    /// # Errors
    ///
    /// Returns [`SweepError::WidthScaleCountMismatch`] when the number of
    /// scales differs from the number of path stations. The vertex pool is
    /// unchanged on this error.
    pub fn sweep_variable(
        &self,
        profile: &ChannelProfile,
        path: &ChannelPath,
        width_scales: &[Real],
        vertex_pool: &mut VertexPool,
        region: RegionId,
    ) -> Result<Vec<FaceData>, SweepError> {
        let n_stations = path.points().len();
        if width_scales.len() != n_stations {
            return Err(SweepError::WidthScaleCountMismatch {
                expected: n_stations,
                actual: width_scales.len(),
            });
        }
        Ok(self.sweep_inner(profile, path, |i| width_scales[i], vertex_pool, region))
    }

    /// Canonical sweep kernel shared by `sweep` and `sweep_variable`.
    ///
    /// `scale_x_fn(station_index)` returns the X-scale at that station.
    fn sweep_inner(
        &self,
        profile: &ChannelProfile,
        path: &ChannelPath,
        scale_x_fn: impl Fn(usize) -> Real,
        vertex_pool: &mut VertexPool,
        region: RegionId,
    ) -> Vec<FaceData> {
        let profile_pts = profile.generate_points();
        let frames = path.compute_frames();
        let n_profile = profile_pts.len();
        let n_stations = frames.len();

        let mut rings: Vec<Vec<VertexId>> = Vec::with_capacity(n_stations);
        for (i, frame) in frames.iter().enumerate() {
            let scale_x = scale_x_fn(i);
            let mut ring = Vec::with_capacity(n_profile);
            for pt2d in &profile_pts {
                let pos =
                    frame.position + frame.normal * (pt2d[0] * scale_x) + frame.binormal * pt2d[1];
                let outward = (pos - frame.position).normalize();
                let vid = vertex_pool.insert_or_weld(pos, outward);
                ring.push(vid);
            }
            rings.push(ring);
        }

        let mut faces = Vec::new();

        for s in 0..(n_stations - 1) {
            let ring_a = &rings[s];
            let ring_b = &rings[s + 1];
            for i in 0..n_profile {
                let j = (i + 1) % n_profile;
                faces.push(FaceData::new(ring_a[i], ring_b[j], ring_b[i], region));
                faces.push(FaceData::new(ring_a[i], ring_a[j], ring_b[j], region));
            }
        }

        if self.cap_start && n_profile >= 3 {
            let center_pos = frames[0].position;
            let center_normal = -frames[0].tangent;
            let center = vertex_pool.insert_or_weld(center_pos, center_normal);
            let ring = &rings[0];
            for i in 0..n_profile {
                let j = (i + 1) % n_profile;
                faces.push(FaceData::new(center, ring[j], ring[i], region));
            }
        }

        if self.cap_end && n_profile >= 3 {
            let last = n_stations - 1;
            let center_pos = frames[last].position;
            let center_normal = frames[last].tangent;
            let center = vertex_pool.insert_or_weld(center_pos, center_normal);
            let ring = &rings[last];
            for i in 0..n_profile {
                let j = (i + 1) % n_profile;
                faces.push(FaceData::new(center, ring[i], ring[j], region));
            }
        }

        faces
    }
}

impl Default for SweepMesher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Point3r;

    #[test]
    fn reports_width_scale_count_mismatch_without_mutating_the_pool() {
        let path = ChannelPath::new(vec![
            Point3r::origin(),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(2.0, 0.0, 0.0),
        ])
        .expect("valid path");
        let profile = ChannelProfile::Rectangular {
            width: 1.0,
            height: 1.0,
        };
        let mut vertex_pool = VertexPool::default_millifluidic();
        let before = vertex_pool.len();
        let error = SweepMesher::new()
            .sweep_variable(
                &profile,
                &path,
                &[1.0, 0.9],
                &mut vertex_pool,
                RegionId::from_usize(0),
            )
            .expect_err("mismatched scales must fail");

        assert_eq!(
            error,
            SweepError::WidthScaleCountMismatch {
                expected: 3,
                actual: 2,
            }
        );
        assert_eq!(vertex_pool.len(), before);
    }

    #[test]
    fn matching_width_scales_reach_the_canonical_sweep_kernel() {
        let path = ChannelPath::new(vec![
            Point3r::origin(),
            Point3r::new(1.0, 0.0, 0.0),
            Point3r::new(2.0, 0.0, 0.0),
        ])
        .expect("valid path");
        let profile = ChannelProfile::Rectangular {
            width: 1.0,
            height: 1.0,
        };
        let mut vertex_pool = VertexPool::default_millifluidic();
        let faces = SweepMesher::new()
            .sweep_variable(
                &profile,
                &path,
                &[1.0, 0.9, 1.1],
                &mut vertex_pool,
                RegionId::from_usize(0),
            )
            .expect("matching scales must build");

        assert_eq!(faces.len(), 24);
        assert_eq!(vertex_pool.len(), 14);
    }
}
