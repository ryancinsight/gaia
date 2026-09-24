use super::super::knot::KnotError;
use thiserror::Error as ThisError;

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
