//! # NURBS and B-Spline Curves
//!
//! Provides `BSplineCurve` (non-rational) and `NurbsCurve` (rational) for any
//! embedding dimension.  Dimension `D` is a const-generic type parameter
//! so all allocations are stack-based for D ≤ 4, and `T` is the
//! [`Scalar`] precision seam with an
//! `f64` default, so existing callers compile without annotation.
//!
//! ## Mathematical Foundation
//!
//! ### B-Spline Curve (non-rational)
//!
//! ```text
//! C(t) = Σᵢ N_{i,p}(t) · Pᵢ
//! ```
//!
//! where `N_{i,p}` are the B-spline basis functions computed by Cox–de Boor
//! recursion (see [`super::basis`]).
//!
//! ### NURBS Curve (rational)
//!
//! ```text
//!        Σᵢ N_{i,p}(t) · wᵢ · Pᵢ
//! C(t) = ─────────────────────────
//!           Σᵢ N_{i,p}(t) · wᵢ
//! ```
//!
//! ### Theorem — Partition of Unity
//!
//! For any parameter value `t` in the domain:
//! ```text
//! Σᵢ N_{i,p}(t) = 1
//! ```
//! This ensures that a B-spline curve is an affine combination of its control
//! points and that translations and rotations of the control polygon map
//! exactly to the curve.
//!
//! ### Theorem — Convex Hull Property
//!
//! Each point `C(t)` lies in the convex hull of the control points local to the
//! active knot span.  For p+1 overlapping spans this is a "local convex hull".
//! Critical for conservative AABB computation in BVH construction.
//!
//! ## Example
//!
//! ```rust,no_run
//! use gaia::domain::geometry::nurbs::knot::KnotVector;
//! use gaia::domain::geometry::nurbs::curve::NurbsCurve;
//! use leto::geometry::Vector as SVector;
//!
//! // Quadratic NURBS arc (quarter circle in XY plane)
//! let ctrl = vec![
//!     SVector::<f64, 3>::new(1.0, 0.0, 0.0),
//!     SVector::<f64, 3>::new(1.0, 1.0, 0.0),
//!     SVector::<f64, 3>::new(0.0, 1.0, 0.0),
//! ];
//! let weights = vec![1.0_f64, std::f64::consts::FRAC_1_SQRT_2, 1.0];
//! let knots = KnotVector::try_new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
//! let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
//!
//! let mid = curve.point(0.5);
//! let scale = mid[0].hypot(mid[1]);
//! assert!((scale - 1.0).abs() < 1e-10, "point should be on unit circle");
//! ```

use thiserror::Error as ThisError;

use super::basis::{eval_basis_and_deriv_to_slice, eval_basis_to_slice};
use super::knot::{KnotError, KnotVector};
use crate::domain::core::scalar::{Real, Scalar};
use eunomia::NumericElement;
use leto::geometry::Vector as SVector;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Error returned when constructing a B-spline or NURBS curve fails.
#[derive(Clone, Debug, PartialEq, ThisError)]
pub enum CurveError {
    /// Knot vector is invalid.
    #[error("knot error: {0}")]
    Knot(#[from] KnotError),
    /// Vector of control points is empty.
    #[error("no control points")]
    NoControlPoints,
    /// Number of knots is inconsistent with degree and control-point count.
    #[error("expected {expected} knots (n+p+2), got {got}")]
    KnotCountMismatch {
        /// Number of knots provided.
        got: usize,
        /// Number of knots expected (`n + p + 2`).
        expected: usize,
    },
    /// Degree is zero — undefined for B-splines.
    #[error("degree must be ≥ 1")]
    ZeroDegree,
    /// Weights vector length does not match control-point count.
    #[error("weights length {weights} != control points {control_points}")]
    WeightsMismatch {
        /// Number of weights provided.
        weights: usize,
        /// Number of control points.
        control_points: usize,
    },
    /// A weight is ≤ 0, which makes the NURBS curve ill-defined.
    #[error("weight[{index}] ≤ 0")]
    NonPositiveWeight {
        /// Index of the offending weight.
        index: usize,
    },
}

// ── BSplineCurve ──────────────────────────────────────────────────────────────

/// A non-rational B-spline curve of degree `p` in `D`-dimensional space.
///
/// All weights are implicitly 1.  For rational curves (conics, circles, etc.)
/// use [`NurbsCurve`].
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
    control_points: Vec<SVector<T, D>>,
    /// Validated knot vector.
    knots: KnotVector<T>,
    /// Polynomial degree.
    degree: usize,
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
    /// Uses de Boor's algorithm via [`super::basis::eval_basis_to_slice`] —
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
                let t = lo
                    + (hi - lo)
                        * (<T as Scalar>::from_f64(i as f64)
                            / <T as Scalar>::from_f64((count - 1) as f64));
                self.point(t)
            })
            .collect()
    }
}

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
                let t = lo
                    + (hi - lo)
                        * (<T as Scalar>::from_f64(i as f64)
                            / <T as Scalar>::from_f64((count - 1) as f64));
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
            let t = lo
                + (hi - lo)
                    * (<T as Scalar>::from_f64(i as f64) / <T as Scalar>::from_f64(res as f64));
            let pt = self.point(t);
            aabb.expand(&Point3::new(pt[0], pt[1], pt[2]));
        }
        aabb
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use leto::geometry::Vector as SVector;

    type V3 = SVector<Real, 3>;
    type V2 = SVector<Real, 2>;

    fn v3(x: Real, y: Real, z: Real) -> V3 {
        V3::new(x, y, z)
    }

    // ── BSplineCurve ─────────────────────────────────────────────────────────

    #[test]
    fn bspline_linear_interpolates_endpoints() {
        // Linear (p=1), 2 control points: C(0)=P0, C(1)=P1
        let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 3.0)];
        let curve = BSplineCurve::clamped(pts, 1);
        let start = curve.point(0.0);
        let end = curve.point(1.0);
        assert!((start - v3(0.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((end - v3(1.0, 2.0, 3.0)).norm() < 1e-12);
    }

    #[test]
    fn bspline_quadratic_midpoint() {
        // Quadratic with 3 control points: P0=(0,0,0), P1=(1,2,0), P2=(2,0,0)
        // At t=0.5 the result should be between the control points
        let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 2.0, 0.0), v3(2.0, 0.0, 0.0)];
        let curve = BSplineCurve::clamped(pts, 2);
        let mid = curve.point(0.5);
        // B-spline interpolates convex hull: y should be positive and x near 1
        assert!(mid[0] > 0.9 && mid[0] < 1.1);
        assert!(mid[1] > 0.0);
    }

    #[test]
    fn bspline_tangent_linear() {
        // Linear curve from (0,0,0) to (1,1,1): tangent should be constant (1,1,1)
        let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)];
        let curve = BSplineCurve::clamped(pts, 1);
        let (_, tan) = curve.point_and_tangent(0.5);
        // Tangent direction is (1,1,1), magnitude = degree * (P1-P0) / knot diff
        assert!(
            tan.dot(v3(1.0, 1.0, 1.0)) > 0.0,
            "tangent must point in positive direction"
        );
    }

    #[test]
    fn bspline_sample_uniform_count() {
        let pts = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), v3(2.0, 0.0, 0.0)];
        let curve = BSplineCurve::clamped(pts, 2);
        let samples = curve.sample_uniform(11);
        assert_eq!(samples.len(), 11);
    }

    #[test]
    fn bspline_creation_errors() {
        let kv = KnotVector::clamped_uniform(2, 2);
        // Wrong degree: knot vector has n+p+2=3+1+2... let's just test no control points
        let empty: Vec<V3> = vec![];
        assert!(BSplineCurve::new(empty, kv, 2).is_err());
    }

    // ── NurbsCurve ───────────────────────────────────────────────────────────

    #[test]
    fn nurbs_unit_weight_matches_bspline() {
        // NURBS with w=1 everywhere should equal B-spline
        let pts = vec![v3(0.0, 0.0, 0.0), v3(0.5, 1.0, 0.0), v3(1.0, 0.0, 0.0)];
        let bs = BSplineCurve::clamped(pts.clone(), 2);
        let weights = vec![1.0_f64; 3];
        let kv = KnotVector::clamped_uniform(2, 2);
        let nc = NurbsCurve::new(pts, weights, kv, 2).unwrap();
        for i in 0..=10 {
            let t = Real::from(i) / 10.0;
            let pb = bs.point(t);
            let pn = nc.point(t);
            assert!(
                (pb - pn).norm() < 1e-12,
                "unit-weight NURBS ≠ B-spline at t={}: |diff|={}",
                t,
                (pb - pn).norm()
            );
        }
    }

    #[test]
    fn nurbs_quarter_circle() {
        // Exact unit quarter-circle in XY plane:
        // P0=(1,0), w0=1  P1=(1,1), w1=1/√2  P2=(0,1), w2=1
        let sq2_inv: Real = <Real as Scalar>::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        let ctrl = vec![V2::new(1.0, 0.0), V2::new(1.0, 1.0), V2::new(0.0, 1.0)];
        let weights = vec![1.0, sq2_inv, 1.0];
        let knots = KnotVector::try_new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();

        // Sample 11 points and check that they lie on the unit circle
        for i in 0..=10 {
            let t = Real::from(i) / 10.0;
            let pt = curve.point(t);
            let r = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-10,
                "quarter-circle point not on unit circle at t={t}: r={r}"
            );
        }
    }

    #[test]
    fn nurbs_endpoints_interpolate() {
        // Any clamped NURBS must pass through first and last control points
        let ctrl = vec![v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0), v3(7.0, 8.0, 9.0)];
        let weights = vec![1.0, 0.5, 2.0];
        let knots = KnotVector::clamped_uniform(2, 2);
        let curve = NurbsCurve::new(ctrl.clone(), weights, knots, 2).unwrap();
        let start = curve.point(0.0);
        let end = curve.point(1.0);
        assert!((start - ctrl[0]).norm() < 1e-12, "start should equal P0");
        assert!((end - ctrl[2]).norm() < 1e-12, "end should equal P2");
    }

    #[test]
    fn nurbs_non_positive_weight_errors() {
        let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
        let weights = vec![1.0, 0.0]; // zero weight is invalid
        let knots = KnotVector::clamped_uniform(1, 1);
        assert!(matches!(
            NurbsCurve::new(ctrl, weights, knots, 1),
            Err(CurveError::NonPositiveWeight { index: 1 })
        ));
    }

    #[test]
    fn nurbs_aabb_contains_control_points() {
        let ctrl = vec![v3(-1.0, -2.0, -3.0), v3(0.0, 0.0, 0.0), v3(4.0, 5.0, 6.0)];
        let weights = vec![1.0, 1.5, 1.0];
        let knots = KnotVector::clamped_uniform(2, 2);
        let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
        let aabb = curve.aabb(50);
        // Curve lies in convex hull of control points; aabb should be non-degenerate
        assert!(aabb.min.x <= 0.0 && aabb.max.x >= 1.0);
    }

    #[test]
    fn nurbs_tangent_non_zero_for_non_degenerate_curve() {
        let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 0.0), v3(2.0, 0.0, 0.0)];
        let weights = vec![1.0_f64; 3];
        let knots = KnotVector::clamped_uniform(2, 2);
        let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
        let (_, tan) = curve.point_and_tangent(0.5);
        assert!(tan.norm() > 0.0, "tangent at midpoint should be non-zero");
    }

    // ── Generic instantiation ───────────────────────────────────────────────

    /// A dyadic linear curve evaluates identically at `f32` and `f64`:
    /// all knot values, parameters, and control coordinates are exact at
    /// both precisions, so native-`T` evaluation is the same computation.
    #[test]
    fn f32_instantiation_interpolates_dyadic_endpoints() {
        let curve32 = {
            let pts = vec![
                SVector::<f32, 3>::new(0.0, 0.0, 0.0),
                SVector::<f32, 3>::new(1.0, 0.5, 0.25),
            ];
            BSplineCurve::<3, f32>::clamped(pts, 1)
        };
        let curve64 = {
            let pts = vec![
                SVector::<f64, 3>::new(0.0, 0.0, 0.0),
                SVector::<f64, 3>::new(1.0, 0.5, 0.25),
            ];
            BSplineCurve::<3, f64>::clamped(pts, 1)
        };
        for i in 0..=8 {
            let t32 = <f32 as Scalar>::from_f64(f64::from(i) / 8.0);
            let t64 = f64::from(i) / 8.0;
            let p32 = curve32.point(t32);
            let p64 = curve64.point(t64);
            for k in 0..3 {
                assert_eq!(
                    p32[k].to_bits(),
                    <f32 as Scalar>::from_f64(p64[k]).to_bits()
                );
            }
        }
        let start = curve32.point(0.0);
        let end = curve32.point(1.0);
        assert_eq!(start[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(end[2].to_bits(), 0.25_f32.to_bits());
    }

    /// Unit-weight NURBS equals the B-spline at `f32` — the identity holds
    /// per instantiation, not only at `f64`.
    #[test]
    fn f32_unit_weight_nurbs_matches_bspline() {
        let pts32 = vec![
            SVector::<f32, 3>::new(0.0, 0.0, 0.0),
            SVector::<f32, 3>::new(0.5, 1.0, 0.0),
            SVector::<f32, 3>::new(1.0, 0.0, 0.0),
        ];
        let bs = BSplineCurve::<3, f32>::clamped(pts32.clone(), 2);
        let nc = NurbsCurve::<3, f32>::from_bspline(bs.clone());
        for i in 0..=10 {
            let t = <f32 as Scalar>::from_f64(f64::from(i) / 10.0);
            let pb = bs.point(t);
            let pn = nc.point(t);
            assert!((pb - pn).norm() < 1e-6);
        }
    }
}
