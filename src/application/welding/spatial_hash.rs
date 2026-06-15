//! Standalone spatial hash grid for point queries.
//!
//! This is the same algorithm used inside `VertexPool`, but exposed as a
//! generic utility for any point set.
//!
//! `GridCell` is defined in `snap` (the canonical SSOT) and re-exported here
//! to eliminate the duplicate struct that previously existed in both modules.

use hashbrown::HashMap;

use crate::domain::core::scalar::{Point3r, Real};

/// Canonical 3-D grid cell key — re-exported from [`snap::GridCell`].
///
/// [`snap::GridCell`]: crate::application::welding::snap::GridCell
pub use super::snap::GridCell;

/// A spatial hash grid mapping 3D points to bucket indices.
pub struct SpatialHashGrid {
    /// Buckets: grid cell → list of indices.
    buckets: HashMap<GridCell, Vec<u32>>,
    /// Reciprocal of cell size.
    inv_cell_size: Real,
}

impl SpatialHashGrid {
    /// Create a new spatial hash grid.
    ///
    /// `cell_size` should be roughly the search radius for typical queries.
    #[must_use]
    pub fn new(cell_size: Real) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            buckets: HashMap::new(),
            inv_cell_size: 1.0 / cell_size,
        }
    }

    /// Insert a point with a given index.
    pub fn insert(&mut self, point: &Point3r, index: u32) {
        let cell = GridCell::from_point(point, self.inv_cell_size);
        self.buckets
            .entry(cell)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(index);
    }

    /// Query: find all indices within `radius` of `point`.
    ///
    /// Searches the 3×3×3 neighborhood and filters by actual distance.
    #[must_use]
    pub fn query_radius(&self, point: &Point3r, radius: Real, positions: &[Point3r]) -> Vec<u32> {
        let cell = GridCell::from_point(point, self.inv_cell_size);
        let radius_sq = radius * radius;
        let mut results = Vec::new();

        for neighbor in cell.neighborhood_27() {
            if let Some(indices) = self.buckets.get(&neighbor) {
                for &idx in indices {
                    let dist_sq = (positions[idx as usize] - point).norm_squared();
                    if dist_sq <= radius_sq {
                        results.push(idx);
                    }
                }
            }
        }

        results
    }

    /// Find the nearest point within `radius`, if any.
    #[must_use]
    pub fn query_nearest(
        &self,
        point: &Point3r,
        radius: Real,
        positions: &[Point3r],
    ) -> Option<u32> {
        let cell = GridCell::from_point(point, self.inv_cell_size);
        let radius_sq = radius * radius;
        let mut best: Option<(u32, Real)> = None;

        for neighbor in cell.neighborhood_27() {
            if let Some(indices) = self.buckets.get(&neighbor) {
                for &idx in indices {
                    let dist_sq = (positions[idx as usize] - point).norm_squared();
                    if dist_sq <= radius_sq && best.is_none_or(|(_, d)| dist_sq < d) {
                        best = Some((idx, dist_sq));
                    }
                }
            }
        }

        best.map(|(idx, _)| idx)
    }

    /// Clear the grid.
    pub fn clear(&mut self) {
        self.buckets.clear();
    }
}
