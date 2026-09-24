use super::super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::super::knot::KnotVector;
use super::super::parameter::uniform_parameter;
use super::{BSplineCurve, CurveError};
use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::Vector as SVector;
// ── NurbsCurve ────────────────────────────────────────────────────────────────

/// A rational B-spline (NURBS) curve of degree `p` in `D`-dimensional space.
///
/// Each control point `Pᵢ` has an associated positive weight `wᵢ`.  When all
/// weights are equal to 1.0, the curve is identical to a B-spline.  Non-uniform
/// weights allow exact representation of conics (circles, ellipses, parabolas).
///
/// # Invariants
///
/// - `control_points.len() == weights.len()`.
/// - All weights `wᵢ > 0`.
/// - `knots.len() == n + p + 2` where `n+1 == control_points.len()`.
///
/// # Theorem — Rational Partition of Unity
///
/// The NURBS basis functions `R_{i,p}(t) = N_{i,p}(t) · wᵢ / W(t)` where
/// `W(t) = Σ N_{i,p}(t) · wᵢ` satisfy `Σ R_{i,p}(t) = 1`, inheriting
/// the convex hull and affine invariance properties from B-splines.
#[derive(Clone, Debug)]
pub struct NurbsCurve<const D: usize, T = Real> {
    /// Control points in D-dimensional Euclidean space.
    control_points: Vec<SVector<T, D>>,
    /// Positive weights, one per control point.
    weights: Vec<T>,
    /// Validated knot vector.
    knots: KnotVector<T>,
    /// Polynomial degree.
    degree: usize,
}

impl<const D: usize, T: Scalar> NurbsCurve<D, T> {
    /// Create a new NURBS curve.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError`] if:
    /// - `control_points` or `weights` are empty.
    /// - `weights.len() != control_points.len()`.
    /// - Any weight is ≤ 0.
    /// - The knot count doesn't satisfy `n + p + 2`.
    pub fn new(
        control_points: Vec<SVector<T, D>>,
        weights: Vec<T>,
        knots: KnotVector<T>,
        degree: usize,
    ) -> Result<Self, CurveError> {
        if control_points.is_empty() {
            return Err(CurveError::NoControlPoints);
        }
        if degree == 0 {
            return Err(CurveError::ZeroDegree);
        }
        if weights.len() != control_points.len() {
            return Err(CurveError::WeightsMismatch {
                weights: weights.len(),
                control_points: control_points.len(),
            });
        }
        for (i, &w) in weights.iter().enumerate() {
            if w <= <T as NumericElement>::ZERO {
                return Err(CurveError::NonPositiveWeight { index: i });
            }
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
            weights,
            knots,
            degree,
        })
    }

    /// Create a NURBS curve from a B-spline (all weights = 1).
    #[must_use]
    pub fn from_bspline(curve: BSplineCurve<D, T>) -> Self {
        let n = curve.control_points.len();
        let weights = vec![<T as NumericElement>::ONE; n];
        Self {
            control_points: curve.control_points,
            weights,
            knots: curve.knots,
            degree: curve.degree,
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

    /// Weights slice.
    #[must_use]
    pub fn weights(&self) -> &[T] {
        &self.weights
    }

    /// Evaluate the NURBS curve at parameter `t`.
    ///
    /// ```text
    ///        Σᵢ N_{i,p}(t) · wᵢ · Pᵢ
    /// C(t) = ─────────────────────────
    ///           Σᵢ N_{i,p}(t) · wᵢ
    /// ```
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

        let mut num = SVector::<T, D>::zeros();
        let mut den: T = zero;
        for (j, &b) in basis.iter().enumerate() {
            let w = self.weights[span - p + j];
            let bw = b * w;
            num += self.control_points[span - p + j] * bw;
            den += bw;
        }
        // Guard against degenerate knot spans where all basis weights are 0
        if den.abs() < <T as Scalar>::from_f64(1e-15) {
            return self.control_points[span - p];
        }
        num / den
    }

    /// Evaluate the NURBS curve and its first derivative at parameter `t`.
    ///
    /// Uses the quotient rule:
    /// ```text
    /// C'(t) = (A'(t) · W(t) − A(t) · W'(t)) / W(t)²
    /// ```
    /// where `A(t) = Σ N_{i,p}(t)·wᵢ·Pᵢ` and `W(t) = Σ N_{i,p}(t)·wᵢ`.
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

        let mut a = SVector::<T, D>::zeros(); // Σ N·w·P
        let mut da = SVector::<T, D>::zeros(); // Σ N'·w·P
        let mut w: T = zero; // Σ N·w
        let mut dw: T = zero; // Σ N'·w

        for j in 0..=p {
            let cp = self.control_points[span - p + j];
            let wj = self.weights[span - p + j];
            a += cp * (basis[j] * wj);
            da += cp * (dbasis[j] * wj);
            w += basis[j] * wj;
            dw += dbasis[j] * wj;
        }

        let guard = <T as Scalar>::from_f64(1e-15);
        let pt = if w.abs() < guard {
            self.control_points[span - p]
        } else {
            a / w
        };

        let tan = if w.abs() < guard {
            SVector::<T, D>::zeros()
        } else {
            (da - pt * dw) / w
        };

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

impl<T: Scalar> NurbsCurve<3, T> {
    /// Compute an axis-aligned bounding box over `resolution` samples.
    ///
    /// By the convex hull property of NURBS, the bounding box of the control
    /// points is a conservative bound; this provides a tighter empirical bound.
    #[must_use]
    pub fn aabb(&self, resolution: usize) -> crate::domain::geometry::Aabb<T> {
        use crate::domain::geometry::Aabb;
        use leto::geometry::Point3;
        let mut aabb = Aabb::<T>::empty();
        let (lo, hi) = self.domain();
        let res = resolution.max(8);
        for i in 0..=res {
            let t = uniform_parameter(lo, hi, i, res);
            let pt = self.point(t);
            aabb.expand(&Point3::new(pt[0], pt[1], pt[2]));
        }
        aabb
    }
}
