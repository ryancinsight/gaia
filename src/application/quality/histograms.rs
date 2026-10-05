//! Fixed-width histogram for mesh quality distributions.
//!
//! ## Complexity
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | `compute` | O(n)       | n = value count |
//! | `exact_percentile` | O(n log n) | strictly sort-based exact percentile |
//!
//! ## References
//!
//! Sturges, H.A. (1926). "The Choice of a Class Interval". *JASA* 21(153): 65-66.

use crate::domain::core::scalar::{Real, Scalar};
use eunomia::{FloatElement, NumericElement};

/// Compute the exact p-th percentile (0.0 = min, 1.0 = max) from a slice of values.
///
/// Non-finite values (NaN, +/-Inf) are ignored.
#[must_use]
pub fn exact_percentile(values: &[Real], p: f64) -> Option<Real> {
    exact_percentile_scalar::<Real>(values, p)
}

/// Generic exact percentile over any `T: Scalar`.
///
/// Non-finite values are ignored. Complexity O(n log n).
#[must_use]
pub(crate) fn exact_percentile_scalar<T: Scalar>(values: &[T], p: f64) -> Option<T> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded percentile position is clamped into the finite slice bounds before the checked integer conversion"
    )]
    fn rounded_index(value: f64) -> usize {
        usize::try_from(value.round() as i64).expect("percentile index fits in usize")
    }

    let mut finite: Vec<T> = values
        .iter()
        .copied()
        .filter(|v| <T as NumericElement>::is_finite(*v))
        .collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_position = p.clamp(0.0, 1.0) * (finite.len() - 1) as f64;
    let target_idx = rounded_index(target_position);
    Some(finite[target_idx])
}

// ── Public histogram (Real = f64) ──────────────────────────────────────────────

/// Fixed-width histogram over a set of scalar values.
///
/// `edges` has `n_bins + 1` elements: `edges[i]` is the left boundary of bin `i`,
/// and `edges[n_bins]` is the right boundary of the last bin.
/// `bins[i]` is the count of values in `[edges[i], edges[i+1])`.
/// Values exactly equal to `max` fall into the last bin.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Per-bin counts.
    pub bins: Vec<usize>,
    /// Bin boundary values; length = `bins.len() + 1`.
    pub edges: Vec<Real>,
    /// Minimum value observed.
    pub min: Real,
    /// Maximum value observed.
    pub max: Real,
}

impl Histogram {
    /// Build a fixed-width histogram from `values` with `n_bins` bins.
    ///
    /// Returns `None` when `values` is empty or `n_bins` is zero.
    ///
    /// All finite values are bucketed. Non-finite values (NaN, +/-Inf) are silently
    /// skipped so that degenerate triangle metrics do not corrupt the histogram.
    #[must_use]
    pub fn compute(values: &[Real], n_bins: usize) -> Option<Self> {
        HistogramT::compute(values, n_bins).map(HistogramT::into_real_histogram)
    }

    /// Number of bins.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.bins.len()
    }

    /// Bin width (uniform).
    #[must_use]
    pub fn bin_width(&self) -> Real {
        if self.edges.len() < 2 {
            return 0.0;
        }
        self.edges[1] - self.edges[0]
    }

    /// Bin midpoint for bin index `i`.
    #[must_use]
    pub fn midpoint(&self, i: usize) -> Real {
        0.5 * (self.edges[i] + self.edges[i + 1])
    }
}

// ── Generic histogram (T: Scalar) ─────────────────────────────────────────────

/// Fixed-width histogram generic over `T: Scalar`.
///
/// Keeps histogram arithmetic in the input precision; edges and extrema are
/// stored as `T`, not widened to `f64`.
#[derive(Clone, Debug)]
pub(crate) struct HistogramT<T> {
    /// Per-bin counts.
    pub bins: Vec<usize>,
    /// Bin boundary values; length = `bins.len() + 1`.
    pub edges: Vec<T>,
    /// Minimum value observed.
    pub min: T,
    /// Maximum value observed.
    pub max: T,
}

impl<T: Scalar> HistogramT<T> {
    /// Build a fixed-width histogram from `values` with `n_bins` bins.
    ///
    /// Returns `None` when `values` is empty or `n_bins` is zero.
    #[must_use]
    pub fn compute(values: &[T], n_bins: usize) -> Option<Self> {
        if values.is_empty() || n_bins == 0 {
            return None;
        }
        let finite: Vec<T> = values
            .iter()
            .copied()
            .filter(|v| <T as NumericElement>::is_finite(*v))
            .collect();
        if finite.is_empty() {
            return None;
        }
        let min = finite
            .iter()
            .copied()
            .fold(<T as NumericElement>::INFINITY, T::min_scalar);
        let max = finite
            .iter()
            .copied()
            .fold(-<T as NumericElement>::INFINITY, T::max_scalar);

        let mut edges = Vec::with_capacity(n_bins + 1);
        let range = max - min;
        let epsilon = <T as Scalar>::from_f64(f64::EPSILON);
        let bin_w = if range < epsilon {
            <T as NumericElement>::ONE
        } else {
            range / <T as FloatElement>::from_count(n_bins)
        };
        for i in 0..=n_bins {
            edges.push(min + <T as FloatElement>::from_count(i) * bin_w);
        }
        // Invariant: the 0..=n_bins loop pushes n_bins+1 elements.
        *edges
            .last_mut()
            .expect("invariant: edges has n_bins+1 elements, always non-empty") = max + epsilon;

        let mut bins = vec![0usize; n_bins];
        let inv_w = <T as NumericElement>::ONE / bin_w;
        for &v in &finite {
            let bucket_f = (v - min) * inv_w;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the floored histogram bucket coordinate is non-negative and converted through a checked integer boundary"
            )]
            let idx =
                usize::try_from(bucket_f.to_f64().floor() as i64).expect("bin index fits in usize");
            let idx = idx.min(n_bins - 1);
            bins[idx] += 1;
        }

        Some(Self {
            bins,
            edges,
            min,
            max,
        })
    }

    /// Number of bins.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.bins.len()
    }

    /// Bin width (uniform).
    #[must_use]
    pub fn bin_width(&self) -> T {
        if self.edges.len() < 2 {
            return <T as NumericElement>::ZERO;
        }
        self.edges[1] - self.edges[0]
    }

    /// Bin midpoint for bin index `i`.
    #[must_use]
    pub fn midpoint(&self, i: usize) -> T {
        let half = <T as Scalar>::from_f64(0.5);
        half * (self.edges[i] + self.edges[i + 1])
    }
}

impl HistogramT<Real> {
    /// Convert into the public `Histogram` type (zero-cost for `Real = f64`).
    fn into_real_histogram(self) -> Histogram {
        Histogram {
            bins: self.bins,
            edges: self.edges,
            min: self.min,
            max: self.max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_uniform_values_fills_all_bins() {
        let values: Vec<Real> = (0..100).map(Real::from).collect();
        let h = Histogram::compute(&values, 10).unwrap();
        assert_eq!(h.bins.len(), 10);
        for &count in &h.bins {
            assert!(count > 0, "all bins should be non-empty for uniform data");
        }
        assert_eq!(h.bins.iter().sum::<usize>(), 100);
    }

    #[test]
    fn histogram_single_value_all_in_one_bin() {
        let values = vec![42.0_f64; 50];
        let h = Histogram::compute(&values, 5).unwrap();
        assert_eq!(h.bins.iter().sum::<usize>(), 50);
    }

    #[test]
    fn histogram_nan_is_skipped() {
        let values = vec![1.0, f64::NAN, 2.0, 3.0];
        let h = Histogram::compute(&values, 3).unwrap();
        assert_eq!(h.bins.iter().sum::<usize>(), 3);
    }

    #[test]
    fn histogram_empty_returns_none() {
        assert!(Histogram::compute(&[], 5).is_none());
    }

    #[test]
    fn histogram_zero_bins_returns_none() {
        assert!(Histogram::compute(&[1.0, 2.0], 0).is_none());
    }

    #[test]
    fn exact_percentile_median_is_exact() {
        let values: Vec<Real> = (0..=100).map(Real::from).collect();
        let p50 = exact_percentile(&values, 0.5).unwrap();
        assert!(
            (p50 - 50.0).abs() < 1e-12,
            "exact median should be exactly 50, got {p50}"
        );
    }

    #[test]
    fn histogram_generic_f32_and_f64_agree() {
        let values_f64: Vec<f64> = (0_u8..50).map(f64::from).collect();
        let values_f32: Vec<f32> = (0_u8..50).map(f32::from).collect();
        let h64 = HistogramT::compute(&values_f64, 5).unwrap();
        let h32 = HistogramT::compute(&values_f32, 5).unwrap();
        assert_eq!(
            h64.bins, h32.bins,
            "f32 and f64 histograms should agree on bin counts"
        );
        assert_eq!(h64.n_bins(), 5);
        assert_eq!(h32.n_bins(), 5);
    }

    #[test]
    fn exact_percentile_scalar_f32_matches_f64() {
        let values_f64: Vec<f64> = (0_u8..=100).map(f64::from).collect();
        let values_f32: Vec<f32> = (0_u8..=100).map(f32::from).collect();
        let p50_f64 = exact_percentile_scalar::<f64>(&values_f64, 0.5).unwrap();
        let p50_f32 = exact_percentile_scalar::<f32>(&values_f32, 0.5).unwrap();
        assert!((p50_f64 - 50.0).abs() < 1e-10);
        assert!((f64::from(p50_f32) - 50.0).abs() < 1e-4);
    }
}
