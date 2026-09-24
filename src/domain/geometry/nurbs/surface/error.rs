use thiserror::Error as ThisError;

/// Error returned when constructing a B-spline or NURBS surface.
#[derive(Clone, Debug, PartialEq, ThisError)]
pub enum SurfaceError {
    /// Control grid is empty.
    #[error("control grid is empty")]
    EmptyControlGrid,
    /// Degree in the named direction is zero.
    #[error("degree in {direction} direction must be >= 1")]
    ZeroDegree {
        /// Direction character: `'u'` or `'v'`.
        direction: char,
    },
    /// Knot count does not satisfy `n + p + 2`.
    #[error("knot-{direction}: expected {expected} knots, got {got}")]
    KnotCountMismatch {
        /// Direction character: `'u'` or `'v'`.
        direction: char,
        /// Actual number of knots supplied.
        got: usize,
        /// Required number of knots (`n + degree + 2`).
        expected: usize,
    },
    /// Weight grid dimensions differ from the control grid.
    #[error("weight grid dimensions differ from control grid")]
    WeightGridMismatch,
}

// ---------------------------------------------------------------------------
