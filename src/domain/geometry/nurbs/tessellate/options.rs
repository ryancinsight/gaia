use crate::domain::core::scalar::{Real, Scalar};

// TessellationOptions
// ---------------------------------------------------------------------------

/// Options controlling curvature-adaptive tessellation.
#[derive(Clone, Debug)]
pub struct TessellationOptions<T = Real> {
    /// Maximum angle (degrees) between adjacent surface/curve normals
    /// before a cell or segment is subdivided. Default: 5.0.
    pub max_angle_deg: T,
    /// Minimum number of segments per parametric direction (>= 1).
    /// The tessellation always produces at least this many divisions
    /// even on flat faces. Default: 4.
    pub min_segments: usize,
    /// Maximum additional recursion depth beyond `min_segments`. Default: 6.
    pub max_depth: usize,
}

impl<T: Scalar> Default for TessellationOptions<T> {
    fn default() -> Self {
        Self {
            max_angle_deg: <T as Scalar>::from_f64(5.0),
            min_segments: 4,
            max_depth: 6,
        }
    }
}

impl<T: Scalar> TessellationOptions<T> {
    /// Create with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum deviation angle in degrees (builder pattern).
    #[must_use]
    pub fn with_max_angle(mut self, deg: T) -> Self {
        self.max_angle_deg = deg;
        self
    }

    /// Set the minimum number of parameter segments (builder pattern).
    #[must_use]
    pub fn with_min_segments(mut self, n: usize) -> Self {
        self.min_segments = n.max(1);
        self
    }
}

// ---------------------------------------------------------------------------
