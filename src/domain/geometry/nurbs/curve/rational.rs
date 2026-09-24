use super::super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::super::knot::KnotVector;
use super::super::parameter::uniform_parameter;
use super::super::ratio::{rational_value, scaled_rational_term};
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
        self.rational_value(span - self.degree, basis).0
    }

    /// Evaluate the NURBS curve and its first derivative at parameter `t`.
    ///
    /// The quotient rule is evaluated in difference form:
    /// ```text
    /// C'(t) = Σᵢ N'ᵢ,ₚ(t) · wᵢ · (Pᵢ − C(t)) / W(t)
    /// ```
    /// where `W(t) = Σ N_{i,p}(t)·wᵢ`.
    /// Finite factors and coordinate differences are scaled before
    /// multiplication or subtraction to preserve representable components.
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
        let (pt, w, exponent) = self.rational_value(span - self.degree, basis);
        if w == zero {
            return (pt, SVector::<T, D>::zeros());
        }
        let p = self.degree;
        let mut tan = SVector::<T, D>::zeros();
        for j in 0..=p {
            if dbasis[j].abs() <= zero {
                continue;
            }
            let cp = self.control_points[span - p + j];
            if cp == pt {
                continue;
            }
            let raw_weight = self.weights[span - p + j];
            for dimension in 0..D {
                if cp[dimension] != pt[dimension] {
                    tan[dimension] += scaled_rational_term(
                        [dbasis[j], <T as NumericElement>::ONE, raw_weight],
                        [<T as NumericElement>::ONE, w],
                        cp[dimension],
                        pt[dimension],
                        -exponent,
                    );
                }
            }
        }
        (pt, tan)
    }

    fn rational_value(&self, start: usize, basis: &[T]) -> (SVector<T, D>, T, i32) {
        let zero = <T as NumericElement>::ZERO;
        let terms = basis
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(j, coefficient)| {
                (coefficient.abs() > zero).then_some((
                    [
                        coefficient,
                        <T as NumericElement>::ONE,
                        self.weights[start + j],
                    ],
                    self.control_points[start + j].data,
                ))
            });
        let (point, denominator, exponent) = rational_value(terms, self.control_points[start].data);
        (SVector::from(point), denominator, exponent)
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
