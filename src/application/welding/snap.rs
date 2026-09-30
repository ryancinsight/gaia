//! # Unified Vertex Snapping and Welding
//!
//! This module unifies coordinate snapping and deduplication in a single
//! [`SnappingGrid`] that owns the canonical vertex set.
//!
//! ## Algorithm — 26-Neighbor Search
//!
//! Each vertex position is *quantized* to a grid cell via **round-half-up**:
//!
//! ```text
//! cell(x, y, z) = (floor(x/ε + 0.5), floor(y/ε + 0.5), floor(z/ε + 0.5))
//! ```
//!
//! # Theorem — Deterministic Quantization
//!
//! `floor(v + 0.5)` (round-half-up) is a single-valued function for all real
//! inputs including negative values.  Rust's `.round()` uses round-half-away-
//! from-zero, which maps `-0.5 → -1` while floor-based maps `-0.5 → 0`.  When
//! two distinct floating-point computation paths to the same geometric point
//! straddle a half-integer boundary with opposite-sign rounding errors, `.round()`
//! can assign different grid cells; `floor(v + 0.5)` always assigns the same
//! cell for any sign of the tie-breaking error. ∎
//!
//! When inserting a new point, the grid searches all **26 face-, edge-, and
//! corner-adjacent neighbors** plus the home cell itself (27 cells total).
//! This prevents "ghost duplicates" at cell boundaries: a point within `ε` of
//! a neighbor-cell wall will still be found regardless of which side of the
//! wall it falls on.
//!
//! ## Complexity
//!
//! | Operation | Expected | Worst case |
//! |---|---|---|
//! | `insert_or_weld` | O(1) | O(k) where k = vertices per cell |
//! | `query_nearest`  | O(1) | O(k) |
//! | Memory           | O(n) | O(n) |
//!
//! ## Diagram
//!
//! ```text
//! 26-neighbor cells (3-D cross-section, center = ★):
//!
//!   z-1 layer       z=0 layer        z+1 layer
//!  ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐
//!  │ · │ · │ · │  │ · │ · │ · │  │ · │ · │ · │
//!  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
//!  │ · │ · │ · │  │ · │ ★ │ · │  │ · │ · │ · │
//!  ├───┼───┼───┤  ├───┼───┼───┤  ├───┼───┼───┤
//!  │ · │ · │ · │  │ · │ · │ · │  │ · │ · │ · │
//!  └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘
//!   9 neighbors     8 neighbors     9 neighbors
//!                   (+ center)
//! ```
//!
//! ## Integration with `HalfEdgeMesh<'id>`
//!
//! [`SnappingGrid`] is **mesh-agnostic**: it stores positions and returns
//! opaque `u32` indices. Callers map those indices into their own mesh storage.

use hashbrown::HashMap;

use crate::domain::core::scalar::{Point3r, Real, Scalar};
use crate::infrastructure::storage::CellIndices;

// ── GridCell ─────────────────────────────────────────────────────────────────

/// Canonical 3-D grid cell coordinate — SSOT for all spatial hash and welding
/// consumers in the crate.
///
/// Uses `i64` so that negative coordinates and very large models are handled
/// correctly without overflow for any mesh that fits in ±9 × 10¹² ε-units.
///
/// Re-exported by [`spatial_hash`] to eliminate the duplicate struct.
///
/// [`spatial_hash`]: crate::application::welding::spatial_hash
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GridCell {
    /// Quantized X index.
    pub x: i64,
    /// Quantized Y index.
    pub y: i64,
    /// Quantized Z index.
    pub z: i64,
}

impl GridCell {
    /// Quantize a point using **floor** quantization (suitable for range queries).
    ///
    /// Floor maps each point to the cell at or below it on every axis.
    /// Used by `SpatialHashGrid` for O(1) bucket lookup.
    #[inline]
    #[must_use]
    pub fn from_point(p: &Point3r, inv_cell_size: Real) -> Self {
        Self {
            x: (p.x * inv_cell_size).floor() as i64,
            y: (p.y * inv_cell_size).floor() as i64,
            z: (p.z * inv_cell_size).floor() as i64,
        }
    }

    /// Quantize a point using **round-half-up** (suitable for vertex welding).
    ///
    /// `floor(v + 0.5)` is single-valued for all real inputs including negative
    /// values. Rust `.round()` uses round-half-away-from-zero which maps
    /// `-0.5 \u2192 -1` while floor-based maps `-0.5 \u2192 0`. When two floating-point
    /// paths to the same geometric point straddle a half-integer boundary,
    /// `.round()` can assign different cells; `floor(v + 0.5)` always assigns
    /// the same cell. \u220e
    #[inline]
    #[must_use]
    pub fn from_point_round(p: &Point3r, inv_eps: Real) -> Self {
        Self {
            x: (p.x * inv_eps + 0.5).floor() as i64,
            y: (p.y * inv_eps + 0.5).floor() as i64,
            z: (p.z * inv_eps + 0.5).floor() as i64,
        }
    }

    /// Reconstruct the canonical snapped position for this cell.
    #[inline]
    #[must_use]
    pub fn to_point(self, eps: Real) -> Point3r {
        Point3r::new(
            Real::from_index(self.x) * eps,
            Real::from_index(self.y) * eps,
            Real::from_index(self.z) * eps,
        )
    }

    /// Iterator over the 26 neighboring cells **plus self** (27 total).
    ///
    /// Covers all face-, edge-, and corner-adjacent cells so that a welding
    /// query cannot miss a vertex that lies just across a cell boundary.
    #[inline]
    pub fn neighborhood_27(self) -> impl Iterator<Item = GridCell> {
        (-1i64..=1).flat_map(move |dz| {
            (-1i64..=1).flat_map(move |dy| {
                (-1i64..=1).map(move |dx| GridCell {
                    x: self.x + dx,
                    y: self.y + dy,
                    z: self.z + dz,
                })
            })
        })
    }
}

// ── SnappingGrid ──────────────────────────────────────────────────────────────

/// Unified vertex snapping and welding structure.
///
/// Owns a flat list of deduplicated positions.  On each
/// [`insert_or_weld`][SnappingGrid::insert_or_weld] call it either returns the
/// index of an existing vertex within ε, or inserts the new (snapped) position
/// and returns its fresh index.
///
/// # Precision
///
/// Two points `p` and `q` are considered the *same* vertex if
/// `‖p − q‖² ≤ ε²`.  The weld distance is always `ε`; the grid cell size is
/// also `ε`, so the 26-neighbor search guarantees no missed welds.
///
/// # Thread safety
///
/// `SnappingGrid` is **not** `Send`/`Sync` — protect it with a `Mutex` if
/// parallel insertion is required.
pub struct SnappingGrid {
    /// Grid cell → list of `positions` indices stored in that cell.
    buckets: HashMap<GridCell, CellIndices>,
    /// Flat array of all accepted (snapped) positions.
    positions: Vec<Point3r>,
    /// Snap tolerance ε.
    eps: Real,
    /// 1 / ε for quantization.
    inv_eps: Real,
}

impl SnappingGrid {
    /// Create a new snapping grid with tolerance `eps`.
    ///
    /// `eps` is the maximum distance at which two points are considered
    /// identical and will be welded together.  For millifluidic meshes the
    /// recommended value is `1e-6` (1 μm).
    ///
    /// # Panics
    /// Panics if `eps` is not finite and positive.
    #[must_use]
    pub fn new(eps: Real) -> Self {
        assert!(
            eps.is_finite() && eps > 0.0,
            "eps must be finite and positive"
        );
        Self {
            buckets: HashMap::new(),
            positions: Vec::new(),
            eps,
            inv_eps: 1.0 / eps,
        }
    }

    /// Create a new snapping grid with tolerance `eps` and pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if `eps` is not finite or is less than or equal to zero.
    #[must_use]
    pub fn with_capacity(capacity: usize, eps: Real) -> Self {
        assert!(
            eps.is_finite() && eps > 0.0,
            "eps must be finite and positive"
        );
        Self {
            buckets: HashMap::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            eps,
            inv_eps: 1.0 / eps,
        }
    }

    /// Create a snapping grid suitable for millifluidic devices (ε = 1 μm).
    #[must_use]
    pub fn millifluidic() -> Self {
        Self::new(1e-6)
    }

    /// Tolerance ε.
    #[inline]
    #[must_use]
    pub fn eps(&self) -> Real {
        self.eps
    }

    /// Number of unique vertices stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if no vertices have been inserted yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Read-only slice of all stored positions.
    #[inline]
    #[must_use]
    pub fn positions(&self) -> &[Point3r] {
        &self.positions
    }

    /// Look up the position for a given index.
    ///
    /// Returns `None` for out-of-range indices.
    #[inline]
    #[must_use]
    pub fn position(&self, idx: u32) -> Option<Point3r> {
        self.positions.get(idx as usize).copied()
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Search all 27 cells around `home` for the nearest vertex within `eps_sq`.
    ///
    /// Returns `Some((index, dist_sq))` for the closest vertex within `ε²`, or
    /// `None` if no vertex qualifies.
    ///
    /// Extracted from `insert_or_weld` and `query_nearest` to satisfy SSOT:
    /// both previously contained an identical loop body.
    #[inline]
    fn find_nearest_in_27(
        buckets: &HashMap<GridCell, CellIndices>,
        positions: &[Point3r],
        home: GridCell,
        query: &Point3r,
        eps_sq: Real,
    ) -> Option<(u32, Real)> {
        let mut best: Option<(u32, Real)> = None;
        for cell in home.neighborhood_27() {
            if let Some(indices) = buckets.get(&cell) {
                for &idx in indices {
                    let dist_sq = (positions[idx as usize] - query).norm_squared();
                    if dist_sq <= eps_sq {
                        match best {
                            None => best = Some((idx, dist_sq)),
                            Some((_, d)) if dist_sq < d => best = Some((idx, dist_sq)),
                            _ => {}
                        }
                    }
                }
            }
        }
        best
    }

    // ── Core operation ────────────────────────────────────────────────────

    /// Insert `point` into the grid, or weld it to an existing vertex.
    ///
    /// If any stored vertex is within ε of `point`, returns the index of the
    /// nearest such vertex.  Otherwise snaps `point` to the canonical grid
    /// position and inserts it, returning the new index.
    ///
    /// The search covers all 26 neighbors plus the home cell, so no duplicate
    /// can hide across a cell boundary.
    ///
    /// # Returns
    /// A `(index, is_new)` pair.  `is_new` is `true` when a fresh vertex was
    /// added, `false` when an existing vertex was reused.
    ///
    /// # Panics
    ///
    /// Panics if the snapped vertex count exceeds `u32::MAX`.
    #[inline]
    pub fn insert_or_weld(&mut self, point: Point3r) -> (u32, bool) {
        let home = GridCell::from_point_round(&point, self.inv_eps);
        let eps_sq = self.eps * self.eps;

        if let Some((idx, _)) =
            Self::find_nearest_in_27(&self.buckets, &self.positions, home, &point, eps_sq)
        {
            return (idx, false);
        }

        // New vertex: snap to grid center and insert
        let snapped = home.to_point(self.eps);
        let new_idx = u32::try_from(self.positions.len()).expect("vertex count fits in u32");
        self.positions.push(snapped);
        self.buckets
            .entry(home)
            .and_modify(|indices| indices.push(new_idx))
            .or_insert(CellIndices::One(new_idx));
        (new_idx, true)
    }

    /// Query the nearest vertex within ε of `point` without inserting.
    ///
    /// Returns `None` if no vertex is within ε.
    #[inline]
    #[must_use]
    pub fn query_nearest(&self, point: &Point3r) -> Option<u32> {
        let home = GridCell::from_point_round(point, self.inv_eps);
        let eps_sq = self.eps * self.eps;
        Self::find_nearest_in_27(&self.buckets, &self.positions, home, point, eps_sq)
            .map(|(idx, _)| idx)
    }

    /// Query all vertices within ε of `point` without inserting.
    #[must_use]
    pub fn query_within_eps(&self, point: &Point3r) -> Vec<u32> {
        let home = GridCell::from_point_round(point, self.inv_eps);
        let eps_sq = self.eps * self.eps;
        let mut results = Vec::new();

        for cell in home.neighborhood_27() {
            if let Some(indices) = self.buckets.get(&cell) {
                for &idx in indices {
                    let dist_sq = (self.positions[idx as usize] - point).norm_squared();
                    if dist_sq <= eps_sq {
                        results.push(idx);
                    }
                }
            }
        }

        results
    }

    /// Clear all vertices, resetting the grid to empty.
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.positions.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "snap_tests.rs"]
mod tests;
