//! Mesh-wide quality validation.

use crate::application::quality::metrics::QualityMetric;
use crate::application::quality::triangle;
use crate::application::quality::triangle::triangle_angles;
use crate::domain::core::constants;
use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::scalar::{Point3r, Real, Scalar};
use crate::infrastructure::storage::face_store::FaceStore;
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Quality thresholds for mesh validation.
#[derive(Clone, Debug)]
pub struct QualityThresholds {
    /// Maximum allowed aspect ratio (default: 5.0).
    pub max_aspect_ratio: Real,
    /// Minimum allowed angle in radians (default: 15°).
    pub min_angle: Real,
    /// Maximum allowed equiangle skewness (default: 0.8).
    pub max_skewness: Real,
    /// Minimum allowed edge length ratio (default: 0.1).
    pub min_edge_ratio: Real,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            max_aspect_ratio: constants::DEFAULT_MAX_ASPECT_RATIO,
            min_angle: constants::DEFAULT_MIN_ANGLE,
            max_skewness: constants::DEFAULT_MAX_SKEWNESS,
            min_edge_ratio: constants::DEFAULT_MIN_EDGE_RATIO,
        }
    }
}

/// Quality validation report.
#[derive(Clone, Debug)]
pub struct QualityReport {
    /// Aspect ratio statistics.
    pub aspect_ratio: Option<QualityMetric>,
    /// Minimum angle statistics (radians).
    pub min_angle: Option<QualityMetric>,
    /// Skewness statistics.
    pub skewness: Option<QualityMetric>,
    /// Edge ratio statistics.
    pub edge_ratio: Option<QualityMetric>,
    /// Number of faces failing quality thresholds.
    pub failing_faces: usize,
    /// Total faces evaluated.
    pub total_faces: usize,
    /// Pass/fail.
    pub passed: bool,
}

/// Mesh quality validator.
pub struct MeshValidator {
    thresholds: QualityThresholds,
}

/// Convert a generic `Point3<T>` position to the `f64` working precision
/// required by the quality triangle functions.
///
/// Hoisted from the per-face closure in `MeshValidator::validate` to eliminate
/// closure re-creation overhead on every loop iteration.
#[inline]
fn point_to_p3r<T: Scalar>(p: &nalgebra::Point3<T>) -> Point3r {
    use num_traits::ToPrimitive as TP;
    Point3r::new(
        TP::to_f64(&p.x).unwrap_or(0.0),
        TP::to_f64(&p.y).unwrap_or(0.0),
        TP::to_f64(&p.z).unwrap_or(0.0),
    )
}

impl MeshValidator {
    /// Create with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: QualityThresholds::default(),
        }
    }

    /// Create with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: QualityThresholds) -> Self {
        Self { thresholds }
    }

    /// Validate mesh quality.
    pub fn validate<T: Scalar>(
        &self,
        face_store: &FaceStore,
        vertex_pool: &VertexPool<T>,
    ) -> QualityReport {
        let n = face_store.len();
        let mut aspect_ratios = Vec::with_capacity(n);
        let mut min_angles = Vec::with_capacity(n);
        let mut skewnesses = Vec::with_capacity(n);
        let mut edge_ratios = Vec::with_capacity(n);
        let mut failing = 0usize;

        for (_, face) in face_store.iter_enumerated() {
            let a = point_to_p3r(vertex_pool.position(face.vertices[0]));
            let b = point_to_p3r(vertex_pool.position(face.vertices[1]));
            let c = point_to_p3r(vertex_pool.position(face.vertices[2]));

            let ar = triangle::aspect_ratio(&a, &b, &c);
            let er = triangle::edge_length_ratio(&a, &b, &c);
            // Compute all three angles once, derive min, max, and skewness from
            // that single result — avoids 2x redundant normalize passes that
            // would occur by calling min_angle + equiangle_skewness separately.
            let angles = triangle_angles(&a, &b, &c);
            let ideal = constants::PI / 3.0;
            let max_a = angles.iter().copied().fold(Real::NEG_INFINITY, Real::max);
            let ma = angles.iter().copied().fold(Real::INFINITY, Real::min);
            let sk = ((max_a - ideal) / (constants::PI - ideal))
                .max((ideal - ma) / ideal);

            aspect_ratios.push(ar);
            min_angles.push(ma);
            skewnesses.push(sk);
            edge_ratios.push(er);

            if ar > self.thresholds.max_aspect_ratio
                || ma < self.thresholds.min_angle
                || sk > self.thresholds.max_skewness
                || er < self.thresholds.min_edge_ratio
            {
                failing += 1;
            }
        }

        QualityReport {
            aspect_ratio: QualityMetric::from_values(&aspect_ratios),
            min_angle: QualityMetric::from_values(&min_angles),
            skewness: QualityMetric::from_values(&skewnesses),
            edge_ratio: QualityMetric::from_values(&edge_ratios),
            failing_faces: failing,
            total_faces: n,
            passed: failing == 0,
        }
    }

    /// Validate and return error if quality is below threshold.
    pub fn assert_quality<T: Scalar>(
        &self,
        face_store: &FaceStore,
        vertex_pool: &VertexPool<T>,
    ) -> MeshResult<QualityReport> {
        let report = self.validate(face_store, vertex_pool);
        if !report.passed {
            return Err(MeshError::QualityBelowThreshold {
                score: 1.0 - (report.failing_faces as f64 / report.total_faces as f64),
                threshold: 1.0,
            });
        }
        Ok(report)
    }
}

impl Default for MeshValidator {
    fn default() -> Self {
        Self::new()
    }
}
