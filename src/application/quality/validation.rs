//! Mesh-wide quality validation.

use crate::application::quality::metrics::QualityMetric;
use crate::application::quality::triangle;
use crate::application::quality::triangle::triangle_angles;
use crate::domain::core::constants;
use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::scalar::{Real, Scalar};
use crate::infrastructure::storage::face_store::FaceStore;
use crate::infrastructure::storage::vertex_pool::VertexPool;
use eunomia::FloatElement;
use eunomia::NumericElement;

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

impl QualityThresholds {
    /// Create quality thresholds from typed [`aequitas`] quantities.
    ///
    /// Using typed quantities prevents unit confusion (e.g. passing degrees
    /// where radians are required, or a skewness ratio as an angle).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gaia::application::quality::validation::QualityThresholds;
    /// use aequitas::systems::si::quantities::{Angle, Dimensionless};
    ///
    /// let thresholds = QualityThresholds::from_typed(
    ///     Dimensionless::from_base(10.0),  // max aspect ratio
    ///     Angle::from_base(15_f64.to_radians()),  // min angle
    ///     Dimensionless::from_base(0.85),  // max skewness
    ///     Dimensionless::from_base(0.05),  // min edge ratio
    /// );
    /// ```
    #[must_use]
    pub fn from_typed(
        max_aspect_ratio: aequitas::systems::si::quantities::Dimensionless<Real>,
        min_angle: aequitas::systems::si::quantities::Angle<Real>,
        max_skewness: aequitas::systems::si::quantities::Dimensionless<Real>,
        min_edge_ratio: aequitas::systems::si::quantities::Dimensionless<Real>,
    ) -> Self {
        Self {
            max_aspect_ratio: max_aspect_ratio.into_base(),
            min_angle: min_angle.into_base(),
            max_skewness: max_skewness.into_base(),
            min_edge_ratio: min_edge_ratio.into_base(),
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
        let max_aspect_ratio = <T as Scalar>::from_f64(self.thresholds.max_aspect_ratio);
        let min_angle = <T as Scalar>::from_f64(self.thresholds.min_angle);
        let max_skewness = <T as Scalar>::from_f64(self.thresholds.max_skewness);
        let min_edge_ratio = <T as Scalar>::from_f64(self.thresholds.min_edge_ratio);
        let ideal = <T as eunomia::RealField>::PI / T::from_int(3);

        for (_, face) in face_store.iter_enumerated() {
            let a = vertex_pool.position(face.vertices[0]);
            let b = vertex_pool.position(face.vertices[1]);
            let c = vertex_pool.position(face.vertices[2]);

            let ar = triangle::aspect_ratio_native(a, b, c);
            let er = triangle::edge_length_ratio_native(a, b, c);
            // Compute all three angles once, derive min, max, and skewness from
            // that single result — avoids 2x redundant normalize passes that
            // would occur by calling min_angle + equiangle_skewness separately.
            let angles = triangle_angles(a, b, c);
            let max_a = angles
                .iter()
                .copied()
                .fold(-<T as NumericElement>::INFINITY, |lhs, rhs| {
                    lhs.max_scalar(rhs)
                });
            let ma = angles
                .iter()
                .copied()
                .fold(<T as NumericElement>::INFINITY, |lhs, rhs| {
                    lhs.min_scalar(rhs)
                });
            let skew_max = (max_a - ideal) / (<T as eunomia::RealField>::PI - ideal);
            let skew_min = (ideal - ma) / ideal;
            let sk = if skew_max.is_nan() || skew_min.is_nan() {
                <T as NumericElement>::NAN
            } else {
                skew_max.max_scalar(skew_min)
            };

            aspect_ratios.push(ar);
            min_angles.push(ma);
            skewnesses.push(sk);
            edge_ratios.push(er);

            if !ar.is_finite()
                || !ma.is_finite()
                || !sk.is_finite()
                || !er.is_finite()
                || ar > max_aspect_ratio
                || ma < min_angle
                || sk > max_skewness
                || er < min_edge_ratio
            {
                failing += 1;
            }
        }

        QualityReport {
            aspect_ratio: QualityMetric::from_scalar_values(&aspect_ratios),
            min_angle: QualityMetric::from_scalar_values(&min_angles),
            skewness: QualityMetric::from_scalar_values(&skewnesses),
            edge_ratio: QualityMetric::from_scalar_values(&edge_ratios),
            failing_faces: failing,
            total_faces: n,
            passed: failing == 0,
        }
    }

    /// Validate and return error if quality is below threshold.
    ///
    /// # Errors
    ///
    /// Returns [`MeshError::QualityBelowThreshold`] if any face has a
    /// non-finite metric or violates the configured aspect-ratio, minimum-angle,
    /// skewness, or edge-ratio thresholds.
    pub fn assert_quality<T: Scalar>(
        &self,
        face_store: &FaceStore,
        vertex_pool: &VertexPool<T>,
    ) -> MeshResult<QualityReport> {
        let report = self.validate(face_store, vertex_pool);
        if !report.passed {
            return Err(MeshError::QualityBelowThreshold {
                score: 1.0
                    - (Real::from_count(report.failing_faces)
                        / Real::from_count(report.total_faces)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::IndexedMesh;
    use leto::geometry::Point3;

    #[test]
    fn validates_single_precision_geometry_in_native_precision() {
        let mut mesh = IndexedMesh::<f32>::new();
        let a = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
        let b = mesh.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
        let c = mesh.add_vertex_pos(Point3::new(0.0, 1.0, 0.0));
        mesh.add_face(a, b, c);

        let report = MeshValidator::default().validate(&mesh.faces, &mesh.vertices);

        assert!(report.passed);
        assert_eq!(report.total_faces, 1);
        assert_eq!(report.min_angle.expect("one face").count, 1);
    }

    #[test]
    fn rejects_degenerate_faces_instead_of_accepting_nan_metrics() {
        let mut mesh = IndexedMesh::<f64>::new();
        let a = mesh.add_vertex_pos(Point3::new(0.0, 0.0, 0.0));
        let b = mesh.add_vertex_pos(Point3::new(1.0, 0.0, 0.0));
        mesh.add_face(a, b, a);

        let report = MeshValidator::default().validate(&mesh.faces, &mesh.vertices);

        assert_eq!(report.failing_faces, 1);
        assert!(!report.passed);
    }

    #[test]
    fn typed_thresholds_match_scalar_thresholds() {
        use aequitas::systems::si::quantities::{Angle, Dimensionless};

        let typed = QualityThresholds::from_typed(
            Dimensionless::from_base(10.0),
            Angle::from_base(0.261_799), // ≈ 15°
            Dimensionless::from_base(0.8),
            Dimensionless::from_base(0.1),
        );
        assert!((typed.max_aspect_ratio - 10.0).abs() < 1e-12);
        assert!((typed.min_angle - 0.261_799).abs() < 1e-6);
        assert!((typed.max_skewness - 0.8).abs() < 1e-12);
        assert!((typed.min_edge_ratio - 0.1).abs() < 1e-12);
    }
}
