use super::super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::super::knot::KnotVector;
use super::super::parameter::uniform_parameter;
use super::CurveError;
use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::Vector as SVector;

// ── BSplineCurve ──────────────────────────────────────────────────────────────

/// A non-rational B-spline curve of degree `p` in `D`-dimensional space.
///
/// All weights are implicitly 1.  For rational curves (conics, circles, etc.)
/// use [`super::NurbsCurve`].
///
/// # Invariants
///
/// - `knots.len() == n + p + 2` where `n+1 = control_points.len()`.
/// - `p ≥ 1`.
/// - The knot vector is non-decreasing.
///
/// # Diagram
///
/// ```text
/// Control polygon:   P0 ── P1 ── P2 ── P3
///                           │
///                      B-spline curve
///                           │
/// C(t): ════════════════════════════════  (t ∈ [ξ₀, ξₘ])
/// ```
#[derive(Clone, Debug)]
pub struct BSplineCurve<const D: usize, T = Real> {
    /// Control points in D-dimensional space.
    pub(super) control_points: Vec<SVector<T, D>>,
    /// Validated knot vector.
    pub(super) knots: KnotVector<T>,
    /// Polynomial degree.
    pub(super) degree: usize,
}

impl<const D: usize, T: Scalar> BSplineCurve<D, T> {
    /// Create a new B-spline curve.
    ///
    /// # Errors
    /// Returns [`CurveError`] if the knot count, degree, or control-point
    /// count is inconsistent.
    pub fn new(
        control_points: Vec<SVector<T, D>>,
        knots: KnotVector<T>,
        degree: usize,
    ) -> Result<Self, CurveError> {
        if control_points.is_empty() {
            return Err(CurveError::NoControlPoints);
        }
        if degree == 0 {
            return Err(CurveError::ZeroDegree);
        }
        let n = control_points.len() - 1;
        let expected = n + degree + 2;
        if knots.len() != expected {
            return Err(CurveError::KnotCountMismatch {
                got: knots.len(),
                expected,
            });
        }
        Ok(Self {
            control_points,
            knots,
            degree,
        })
    }

    /// Create a clamped uniform B-spline with the given control points and degree.
    ///
    /// The knot vector is constructed automatically.
    ///
    /// # Panics
    /// Panics if `degree == 0` or `control_points.len() < degree + 1`.
    #[must_use]
    pub fn clamped(control_points: Vec<SVector<T, D>>, degree: usize) -> Self {
        assert!(degree >= 1, "degree must be ≥ 1");
        let n = control_points.len() - 1;
        assert!(n >= degree, "need at least degree+1 control points");
        let knots = KnotVector::clamped_uniform(n, degree);
        Self {
            control_points,
            knots,
            degree,
        }
    }

    /// The parameter domain `[t_min, t_max]`.
    #[must_use]
    pub fn domain(&self) -> (T, T) {
        self.knots.domain()
    }

    /// Number of control points.
    #[must_use]
    pub fn num_control_points(&self) -> usize {
        self.control_points.len()
    }

    /// Polynomial degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Evaluate the curve at parameter `t`.
    ///
    /// Uses de Boor's algorithm via [`super::super::basis::eval_basis_to_slice`] —
    /// O(p²) per evaluation.
    ///
    /// # Panics
    /// Panics if `t` is outside the knot domain.
    #[must_use]
    pub fn point(&self, t: T) -> SVector<T, D> {
        let n = self.control_points.len() - 1;
        let span = self.knots.find_span(t, n);
        let zero = <T as NumericElement>::ZERO;
        let mut basis_buf = [zero; 9];
        let mut basis_vec;
        let basis = if self.degree <= 8 {
            &mut basis_buf[..=self.degree]
        } else {
            basis_vec = vec![zero; self.degree + 1];
            &mut basis_vec[..]
        };
        eval_basis_to_slice(span, t, self.degree, &self.knots, basis);
        let p = self.degree;
        let mut result = SVector::<T, D>::zeros();
        for (j, &b) in basis.iter().enumerate() {
            result += self.control_points[span - p + j] * b;
        }
        result
    }

    /// Evaluate the curve and its first derivative at parameter `t`.
    ///
    /// Returns `(C(t), C'(t))`.
    ///
    /// # Panics
    /// Panics if `t` is outside the knot domain.
    #[must_use]
    pub fn point_and_tangent(&self, t: T) -> (SVector<T, D>, SVector<T, D>) {
        let n = self.control_points.len() - 1;
        let span = self.knots.find_span(t, n);
        let zero = <T as NumericElement>::ZERO;
        let mut basis_buf = [zero; 9];
        let mut dbasis_buf = [zero; 9];
        let mut basis_vec;
        let mut dbasis_vec;
        let (basis, dbasis) = if self.degree <= 8 {
            (
                &mut basis_buf[..=self.degree],
                &mut dbasis_buf[..=self.degree],
            )
        } else {
            basis_vec = vec![zero; self.degree + 1];
            dbasis_vec = vec![zero; self.degree + 1];
            (&mut basis_vec[..], &mut dbasis_vec[..])
        };
        eval_basis_and_deriv_to_slice(span, t, self.degree, &self.knots, basis, dbasis);
        let p = self.degree;
        let mut pt = SVector::<T, D>::zeros();
        let mut tan = SVector::<T, D>::zeros();
        for j in 0..=p {
            let cp = self.control_points[span - p + j];
            pt += cp * basis[j];
            tan += cp * dbasis[j];
        }
        (pt, tan)
    }

    /// Sample `count` uniformly spaced points on the curve.
    ///
    /// Includes both endpoints.
    #[must_use]
    pub fn sample_uniform(&self, count: usize) -> Vec<SVector<T, D>> {
        assert!(count >= 2, "need at least 2 samples");
        let (lo, hi) = self.domain();
        (0..count)
            .map(|i| {
                let t = uniform_parameter(lo, hi, i, count - 1);
                self.point(t)
            })
            .collect()
    }
}
