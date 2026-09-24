use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::Point3;

// ControlGrid
// ---------------------------------------------------------------------------

/// A rectangular grid of 3-D control points stored row-major.
///
/// Rows follow the `v` direction and columns follow the `u` direction.
/// `grid.get(i, j)` returns row `i`, column `j`.
#[derive(Clone, Debug)]
pub struct ControlGrid<T = Real> {
    data: Vec<Point3<T>>,
    n_rows: usize,
    n_cols: usize,
}

impl<T: Scalar> ControlGrid<T> {
    /// Create from a flat row-major vector.
    ///
    /// # Panics
    /// Panics if `data.len() != n_rows * n_cols`.
    #[must_use]
    pub fn new(data: Vec<Point3<T>>, n_rows: usize, n_cols: usize) -> Self {
        assert_eq!(
            data.len(),
            n_rows * n_cols,
            "data.len() must equal n_rows * n_cols"
        );
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    /// Number of rows (v direction).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (u direction).
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Access control point at `(row, col)`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> Point3<T> {
        self.data[i * self.n_cols + j]
    }
}

// ---------------------------------------------------------------------------
// WeightGrid
// ---------------------------------------------------------------------------

/// A rectangular grid of positive NURBS weights, stored row-major.
#[derive(Clone, Debug)]
pub struct WeightGrid<T = Real> {
    data: Vec<T>,
    n_rows: usize,
    n_cols: usize,
}

impl<T: Scalar> WeightGrid<T> {
    /// All-ones weight grid (equivalent to B-spline).
    #[must_use]
    pub fn uniform(n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: vec![<T as NumericElement>::ONE; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    /// Create from a flat row-major vector.
    ///
    /// # Panics
    /// Panics if any weight is <= 0 or the data length does not match the
    /// supplied dimensions.
    #[must_use]
    pub fn new(data: Vec<T>, n_rows: usize, n_cols: usize) -> Self {
        assert_eq!(data.len(), n_rows * n_cols);
        for (i, &w) in data.iter().enumerate() {
            assert!(
                w > <T as NumericElement>::ZERO,
                "weight[{i}] = {w} is not positive"
            );
        }
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    /// Access weight at `(row, col)`.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.n_cols + j]
    }

    /// Dimensions.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    /// Dimensions.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
}
