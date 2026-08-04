//! Quality metric definitions.

use crate::domain::core::scalar::Real;
use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;

/// A quality metric measurement.
#[derive(Clone, Debug)]
pub struct QualityMetric {
    /// Minimum value across all elements.
    pub min: Real,
    /// Maximum value across all elements.
    pub max: Real,
    /// Mean value across all elements.
    pub mean: Real,
    /// Number of elements measured.
    pub count: usize,
}

impl QualityMetric {
    /// Create from a slice of values.
    pub fn from_values(values: &[Real]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        if values.iter().any(|value| value.is_nan()) {
            return Some(Self {
                min: Real::NAN,
                max: Real::NAN,
                mean: Real::NAN,
                count: values.len(),
            });
        }
        let min = values.iter().copied().fold(Real::INFINITY, Real::min);
        let max = values.iter().copied().fold(Real::NEG_INFINITY, Real::max);
        let sum: Real = values.iter().sum();
        Some(Self {
            min,
            max,
            mean: sum / values.len() as Real,
            count: values.len(),
        })
    }

    /// Create a report metric from native-precision values.
    ///
    /// The extrema and sum are reduced in `T` and converted to `f64` only at
    /// the reporting boundary. This keeps validation arithmetic in the mesh's
    /// scalar precision instead of silently widening `f32` geometry.
    pub fn from_scalar_values<T: Scalar>(values: &[T]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        if values.iter().copied().any(<T as NumericElement>::is_nan) {
            return Some(Self {
                min: Real::NAN,
                max: Real::NAN,
                mean: Real::NAN,
                count: values.len(),
            });
        }
        let min = values
            .iter()
            .copied()
            .fold(<T as NumericElement>::INFINITY, |lhs, rhs| {
                lhs.min_scalar(rhs)
            });
        let max = values
            .iter()
            .copied()
            .fold(-<T as NumericElement>::INFINITY, |lhs, rhs| {
                lhs.max_scalar(rhs)
            });
        let sum = values
            .iter()
            .copied()
            .fold(<T as NumericElement>::ZERO, |sum, value| sum + value);
        let count = <T as Scalar>::from_f64(values.len() as f64);
        Some(Self {
            min: min.to_f64(),
            max: max.to_f64(),
            mean: (sum / count).to_f64(),
            count: values.len(),
        })
    }

    /// Number of elements below a threshold.
    #[must_use]
    pub fn count_below(values: &[Real], threshold: Real) -> usize {
        values.iter().filter(|&&v| v < threshold).count()
    }
}
