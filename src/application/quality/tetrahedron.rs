//! Native-precision quality metrics for linear tetrahedral cells.
//!
//! The metrics are dimensionless except for volume.  They are evaluated in
//! the mesh scalar `T` and converted to `f64` only when a report is assembled,
//! so `IndexedMesh<f32>` does not silently execute its quality kernel in
//! double precision.

use leto::geometry::{Point3, Vector3};

use super::boundary::{
    assess_boundary_cells, BoundaryFacetQualityCriteria, BoundaryTetrahedralQualityAcceptance,
};
use crate::application::quality::metrics::QualityMetric;
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::{Cell, ElementType};

/// Per-tetrahedron quality values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedronQuality<T> {
    /// Positive geometric volume.
    pub volume: T,
    /// Circumradius divided by the shortest edge length.
    pub radius_edge_ratio: T,
    /// Smallest interior dihedral angle in radians.
    pub min_dihedral_angle: T,
    /// Volume normalized so an equilateral tetrahedron has value one.
    pub normalized_volume: T,
}

/// Summary statistics for the tetrahedral cells in an indexed mesh.
#[derive(Clone, Debug)]
pub struct TetrahedralQualityReport {
    /// Volume statistics for valid tetrahedral cells.
    pub volume: Option<QualityMetric>,
    /// Radius-edge ratio statistics for valid tetrahedral cells.
    pub radius_edge_ratio: Option<QualityMetric>,
    /// Minimum-dihedral-angle statistics in radians for valid cells.
    pub min_dihedral_angle: Option<QualityMetric>,
    /// Normalized-volume statistics for valid cells.
    pub normalized_volume: Option<QualityMetric>,
    /// Number of tetrahedral cells with valid geometry.
    pub valid_cell_count: usize,
    /// Number of tetrahedral cells that could not be measured.
    pub invalid_cell_count: usize,
}

/// Error returned when a tetrahedral acceptance criterion is not meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TetrahedralQualityCriteriaError {
    /// The radius-edge upper bound is not finite and strictly positive.
    InvalidMaxRadiusEdgeRatio,
    /// The minimum dihedral-angle lower bound is outside `[0, pi]` or is not finite.
    InvalidMinDihedralAngle,
    /// The normalized-volume lower bound is outside `[0, 1]` or is not finite.
    InvalidMinNormalizedVolume,
    /// The optional maximum-volume upper bound is not finite and strictly positive.
    InvalidMaxVolume,
}

/// Explicit acceptance criteria for tetrahedral volume cells.
///
/// The criteria intentionally have no `Default` implementation. Quality and
/// sizing thresholds are consumer policy, not universal mesh invariants, so a
/// caller must provide them explicitly. The constructor validates the domain
/// once; subsequent classification is a branch-only native-precision kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedralQualityCriteria<T> {
    max_radius_edge_ratio: T,
    min_dihedral_angle: T,
    min_normalized_volume: T,
    max_volume: Option<T>,
}

impl<T: Scalar> TetrahedralQualityCriteria<T> {
    /// Create validated shape and optional cell-size criteria.
    ///
    /// `max_volume` is an upper bound in the mesh's coordinate units. A value
    /// of `None` disables the size criterion without changing shape checks.
    ///
    /// # Errors
    ///
    /// Returns a typed error when any bound is non-finite or outside its
    /// mathematical domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use gaia::application::quality::TetrahedralQualityCriteria;
    ///
    /// let criteria = TetrahedralQualityCriteria::<f64>::try_new(
    ///     2.0,
    ///     0.25,
    ///     0.1,
    ///     Some(1.0),
    /// )
    /// .expect("criteria are finite and valid");
    /// assert_eq!(criteria.max_volume(), Some(1.0));
    /// ```
    #[must_use = "handle invalid tetrahedral criteria"]
    pub fn try_new(
        max_radius_edge_ratio: T,
        min_dihedral_angle: T,
        min_normalized_volume: T,
        max_volume: Option<T>,
    ) -> Result<Self, TetrahedralQualityCriteriaError> {
        let zero = <T as eunomia::NumericElement>::ZERO;
        let one = <T as eunomia::NumericElement>::ONE;
        let pi = <T as eunomia::RealField>::PI;
        if !<T as eunomia::NumericElement>::is_finite(max_radius_edge_ratio)
            || max_radius_edge_ratio <= zero
        {
            return Err(TetrahedralQualityCriteriaError::InvalidMaxRadiusEdgeRatio);
        }
        if !<T as eunomia::NumericElement>::is_finite(min_dihedral_angle)
            || min_dihedral_angle < zero
            || min_dihedral_angle > pi
        {
            return Err(TetrahedralQualityCriteriaError::InvalidMinDihedralAngle);
        }
        if !<T as eunomia::NumericElement>::is_finite(min_normalized_volume)
            || min_normalized_volume < zero
            || min_normalized_volume > one
        {
            return Err(TetrahedralQualityCriteriaError::InvalidMinNormalizedVolume);
        }
        if max_volume.is_some_and(|volume| {
            !<T as eunomia::NumericElement>::is_finite(volume) || volume <= zero
        }) {
            return Err(TetrahedralQualityCriteriaError::InvalidMaxVolume);
        }

        Ok(Self {
            max_radius_edge_ratio,
            min_dihedral_angle,
            min_normalized_volume,
            max_volume,
        })
    }

    /// Return the maximum accepted radius-edge ratio.
    #[must_use]
    pub fn max_radius_edge_ratio(&self) -> T {
        self.max_radius_edge_ratio
    }

    /// Return the minimum accepted interior dihedral angle in radians.
    #[must_use]
    pub fn min_dihedral_angle(&self) -> T {
        self.min_dihedral_angle
    }

    /// Return the minimum accepted equilateral-normalized volume.
    #[must_use]
    pub fn min_normalized_volume(&self) -> T {
        self.min_normalized_volume
    }

    /// Return the optional maximum accepted cell volume.
    #[must_use]
    pub fn max_volume(&self) -> Option<T> {
        self.max_volume
    }

    /// Classify one measured tetrahedron against these criteria.
    #[must_use]
    pub fn classify(&self, quality: TetrahedronQuality<T>) -> TetrahedronQualityClass {
        let finite = <T as eunomia::NumericElement>::is_finite(quality.volume)
            && <T as eunomia::NumericElement>::is_finite(quality.radius_edge_ratio)
            && <T as eunomia::NumericElement>::is_finite(quality.min_dihedral_angle)
            && <T as eunomia::NumericElement>::is_finite(quality.normalized_volume);
        let zero = <T as eunomia::NumericElement>::ZERO;
        let one = <T as eunomia::NumericElement>::ONE;
        let pi = <T as eunomia::RealField>::PI;
        if !finite
            || quality.volume <= zero
            || quality.radius_edge_ratio <= zero
            || quality.min_dihedral_angle < zero
            || quality.min_dihedral_angle > pi
            || quality.normalized_volume < zero
            || quality.normalized_volume > one
        {
            return TetrahedronQualityClass::Invalid;
        }

        let poor_radius_edge = quality.radius_edge_ratio > self.max_radius_edge_ratio;
        let poor_dihedral = quality.min_dihedral_angle < self.min_dihedral_angle;
        let poor_normalized_volume = quality.normalized_volume < self.min_normalized_volume;

        // A sliver candidate has the characteristic combination that the
        // radius-edge test alone cannot reject: low angle and low normalized
        // volume while the radius-edge bound still passes.
        if !poor_radius_edge && poor_dihedral && poor_normalized_volume {
            return TetrahedronQualityClass::Sliver;
        }
        if poor_radius_edge || poor_dihedral || poor_normalized_volume {
            return TetrahedronQualityClass::PoorShape;
        }
        if self
            .max_volume
            .is_some_and(|max_volume| quality.volume > max_volume)
        {
            return TetrahedronQualityClass::Oversized;
        }
        TetrahedronQualityClass::Accepted
    }

    /// Assess every tetrahedral cell in an indexed mesh.
    ///
    /// Non-tetrahedral cells are ignored. Invalid tetrahedra are counted and
    /// never treated as accepted. `None` means the mesh has no tetrahedral
    /// cells.
    #[must_use]
    pub fn assess(&self, mesh: &IndexedMesh<T>) -> Option<TetrahedralQualityAcceptance> {
        if !mesh
            .cells
            .iter()
            .any(|cell| cell.element_type == ElementType::Tetrahedron)
        {
            return None;
        }

        let mut acceptance = TetrahedralQualityAcceptance::default();
        for measurement in tetrahedral_measurements(mesh) {
            let Some(quality) = measurement else {
                acceptance.invalid_cell_count += 1;
                continue;
            };
            match self.classify(quality) {
                TetrahedronQualityClass::Accepted => acceptance.accepted_cell_count += 1,
                TetrahedronQualityClass::Sliver => acceptance.sliver_count += 1,
                TetrahedronQualityClass::PoorShape => acceptance.poor_shape_count += 1,
                TetrahedronQualityClass::Oversized => acceptance.oversized_cell_count += 1,
                TetrahedronQualityClass::Invalid => acceptance.invalid_cell_count += 1,
            }
        }
        Some(acceptance)
    }

    /// Assess tetrahedral cells that touch the geometric boundary.
    ///
    /// A boundary cell passes only when its cell criteria and every exposed
    /// triangular facet satisfy their respective policies. Boundary facets
    /// are identified by face incidence: exactly one tetrahedral cell must
    /// reference the face. Malformed tetrahedral topology is counted as an
    /// invalid boundary cell instead of being treated as an interior cell.
    ///
    /// The facet policy must be constructed through
    /// [`BoundaryFacetQualityCriteria::try_new`], which validates its bounds.
    #[must_use]
    pub fn assess_boundary(
        &self,
        mesh: &IndexedMesh<T>,
        facet_criteria: &BoundaryFacetQualityCriteria<T>,
    ) -> Option<BoundaryTetrahedralQualityAcceptance> {
        assess_boundary_cells(self, facet_criteria, mesh)
    }
}

/// Quality category assigned to one tetrahedral cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TetrahedronQualityClass {
    /// The cell satisfies every configured shape and size bound.
    Accepted,
    /// The cell is a radius-edge-safe sliver candidate with low angle and volume shape scores.
    Sliver,
    /// The cell violates one or more shape bounds without matching the sliver signature.
    PoorShape,
    /// The cell satisfies shape bounds but exceeds the configured volume bound.
    Oversized,
    /// The measurement contains a non-finite or non-positive value.
    Invalid,
}

/// Counts produced by [`TetrahedralQualityCriteria::assess`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TetrahedralQualityAcceptance {
    /// Number of cells satisfying all criteria.
    pub accepted_cell_count: usize,
    /// Number of cells matching the sliver signature.
    pub sliver_count: usize,
    /// Number of cells failing a shape criterion without the sliver signature.
    pub poor_shape_count: usize,
    /// Number of shape-valid cells exceeding the maximum volume.
    pub oversized_cell_count: usize,
    /// Number of malformed or degenerate tetrahedral cells.
    pub invalid_cell_count: usize,
}

impl TetrahedralQualityAcceptance {
    /// Return the number of cells rejected by the criteria.
    #[must_use]
    pub fn rejected_cell_count(self) -> usize {
        self.sliver_count
            + self.poor_shape_count
            + self.oversized_cell_count
            + self.invalid_cell_count
    }

    /// Return whether every measured tetrahedral cell was accepted.
    #[must_use]
    pub fn passed(self) -> bool {
        self.rejected_cell_count() == 0
    }
}

const EDGES: [(usize, usize, usize, usize); 6] = [
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (1, 2, 0, 3),
    (1, 3, 0, 2),
    (2, 3, 0, 1),
];

fn tetrahedral_measurements<T: Scalar>(
    mesh: &IndexedMesh<T>,
) -> impl Iterator<Item = Option<TetrahedronQuality<T>>> + '_ {
    mesh.cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .map(|cell| cell_tetrahedron_quality(mesh, cell))
}

pub(crate) fn cell_tetrahedron_quality<T: Scalar>(
    mesh: &IndexedMesh<T>,
    cell: &Cell,
) -> Option<TetrahedronQuality<T>> {
    if cell.element_type != ElementType::Tetrahedron {
        return None;
    }
    let vertex_ids: [usize; 4] = cell.vertex_ids.as_slice().try_into().ok()?;
    if vertex_ids.iter().any(|&id| id >= mesh.vertices.len()) {
        return None;
    }
    let points = vertex_ids.map(|id| {
        *mesh
            .vertices
            .position(crate::domain::core::index::VertexId::from_usize(id))
    });
    tetrahedron_quality(points)
}

#[inline]
fn finite_point<T: Scalar>(point: &Point3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(point.x)
        && <T as eunomia::NumericElement>::is_finite(point.y)
        && <T as eunomia::NumericElement>::is_finite(point.z)
}

#[inline]
fn finite_vector<T: Scalar>(vector: &Vector3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(vector.x)
        && <T as eunomia::NumericElement>::is_finite(vector.y)
        && <T as eunomia::NumericElement>::is_finite(vector.z)
}

/// Measure one tetrahedron in native scalar precision.
///
/// `None` means that a coordinate is non-finite, the tetrahedron is
/// degenerate, or a derived metric is non-finite.  The returned angle is the
/// interior dihedral angle at the worst edge; lower values indicate a sliver.
#[must_use]
pub fn tetrahedron_quality<T: Scalar>(points: [Point3<T>; 4]) -> Option<TetrahedronQuality<T>> {
    if points.iter().any(|point| !finite_point(point)) {
        return None;
    }

    let [a, b, c, d] = points;
    let u = b - a;
    let v = c - a;
    let w = d - a;
    let signed_six_volume = u.dot(v.cross(w));
    let zero = <T as eunomia::NumericElement>::ZERO;
    if !<T as eunomia::NumericElement>::is_finite(signed_six_volume) || signed_six_volume == zero {
        return None;
    }

    let six = <T as Scalar>::from_f64(6.0);
    let volume = signed_six_volume.abs() / six;
    let two = <T as eunomia::NumericElement>::ONE + <T as eunomia::NumericElement>::ONE;
    let center_numerator = v.cross(w) * u.norm_squared()
        + w.cross(u) * v.norm_squared()
        + u.cross(v) * w.norm_squared();
    let circumcenter = center_numerator / (signed_six_volume * two);
    let radius_squared = circumcenter.norm_squared();
    if !finite_vector(&center_numerator)
        || !finite_vector(&circumcenter)
        || !<T as eunomia::NumericElement>::is_finite(radius_squared)
        || radius_squared <= zero
    {
        return None;
    }

    let mut edge_squared = [zero; 6];
    let mut shortest_edge_squared = <T as eunomia::NumericElement>::INFINITY;
    for (index, &(i, j, _, _)) in EDGES.iter().enumerate() {
        let edge = points[j] - points[i];
        let squared = edge.norm_squared();
        if !<T as eunomia::NumericElement>::is_finite(squared) || squared <= zero {
            return None;
        }
        edge_squared[index] = squared;
        shortest_edge_squared = shortest_edge_squared.min_scalar(squared);
    }

    let radius_edge_ratio = radius_squared.sqrt() / shortest_edge_squared.sqrt();
    let mut min_dihedral_angle = <T as eunomia::NumericElement>::INFINITY;
    for &(i, j, k, l) in &EDGES {
        let edge = points[j] - points[i];
        let edge_squared = edge.norm_squared();
        let face_k = points[k] - points[i];
        let face_l = points[l] - points[i];
        let projected_k = face_k - edge * (face_k.dot(edge) / edge_squared);
        let projected_l = face_l - edge * (face_l.dot(edge) / edge_squared);
        let product = projected_k.norm_squared() * projected_l.norm_squared();
        if product <= zero || !<T as eunomia::NumericElement>::is_finite(product) {
            return None;
        }
        let cosine = (projected_k.dot(projected_l) / product.sqrt())
            .max_scalar(-<T as eunomia::NumericElement>::ONE)
            .min_scalar(<T as eunomia::NumericElement>::ONE);
        min_dihedral_angle = min_dihedral_angle.min_scalar(cosine.acos());
    }

    let edge_sum = edge_squared
        .iter()
        .copied()
        .fold(zero, |sum, squared| sum + squared);
    let edge_mean_squared = edge_sum / six;
    let edge_rms = edge_mean_squared.sqrt();
    let normalized_volume = ((six + six) * volume)
        / (<T as Scalar>::from_f64(2.0).sqrt() * edge_rms * edge_mean_squared);
    if !<T as eunomia::NumericElement>::is_finite(radius_edge_ratio)
        || !<T as eunomia::NumericElement>::is_finite(min_dihedral_angle)
        || !<T as eunomia::NumericElement>::is_finite(normalized_volume)
    {
        return None;
    }

    Some(TetrahedronQuality {
        volume,
        radius_edge_ratio,
        min_dihedral_angle,
        normalized_volume,
    })
}

/// Summarize native-precision quality metrics for tetrahedral cells.
///
/// Non-tetrahedral cells are ignored.  A malformed or degenerate tetrahedron
/// increments `invalid_cell_count`; it is not converted into a default metric.
/// `None` is returned when the mesh contains no tetrahedral cells.
#[must_use]
pub fn tetrahedral_quality_report<T: Scalar>(
    mesh: &IndexedMesh<T>,
) -> Option<TetrahedralQualityReport> {
    let tetrahedral_cell_count = mesh
        .cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .count();
    if tetrahedral_cell_count == 0 {
        return None;
    }

    let mut volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut radius_edge_ratios = Vec::with_capacity(tetrahedral_cell_count);
    let mut min_dihedral_angles = Vec::with_capacity(tetrahedral_cell_count);
    let mut normalized_volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut invalid_cell_count = 0;

    for measurement in tetrahedral_measurements(mesh) {
        let Some(quality) = measurement else {
            invalid_cell_count += 1;
            continue;
        };
        volumes.push(quality.volume);
        radius_edge_ratios.push(quality.radius_edge_ratio);
        min_dihedral_angles.push(quality.min_dihedral_angle);
        normalized_volumes.push(quality.normalized_volume);
    }

    Some(TetrahedralQualityReport {
        volume: QualityMetric::from_scalar_values(&volumes),
        radius_edge_ratio: QualityMetric::from_scalar_values(&radius_edge_ratios),
        min_dihedral_angle: QualityMetric::from_scalar_values(&min_dihedral_angles),
        normalized_volume: QualityMetric::from_scalar_values(&normalized_volumes),
        valid_cell_count: volumes.len(),
        invalid_cell_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::TetrahedralMeshBuilder;

    fn analytical_quality<T: Scalar>() -> TetrahedronQuality<T> {
        let scalar = |value| <T as Scalar>::from_f64(value);
        tetrahedron_quality([
            Point3::new(scalar(0.0), scalar(0.0), scalar(0.0)),
            Point3::new(scalar(1.0), scalar(0.0), scalar(0.0)),
            Point3::new(scalar(0.0), scalar(1.0), scalar(0.0)),
            Point3::new(scalar(0.0), scalar(0.0), scalar(1.0)),
        ])
        .expect("analytical tetrahedron is non-degenerate")
    }

    fn assert_analytical_quality<T: Scalar>() {
        let quality = analytical_quality::<T>();
        assert!((quality.volume.to_f64() - 1.0 / 6.0).abs() < 1e-6);
        assert!((quality.radius_edge_ratio.to_f64() - 3.0_f64.sqrt() / 2.0).abs() < 1e-6);
        assert!((quality.normalized_volume.to_f64() - 0.769800358919501).abs() < 1e-6);
        assert!(quality.min_dihedral_angle.to_f64().to_degrees() > 54.0);
    }

    #[test]
    fn analytical_tetrahedron_has_expected_native_metrics() {
        assert_analytical_quality::<f32>();
        assert_analytical_quality::<f64>();
    }

    #[test]
    fn sliver_has_worse_dihedral_and_normalized_volume() {
        let regular = analytical_quality::<f64>();
        let sliver = tetrahedron_quality([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.45, 0.45, 1e-4),
        ])
        .expect("sliver remains non-degenerate");
        assert!(sliver.min_dihedral_angle < regular.min_dihedral_angle);
        assert!(sliver.normalized_volume < regular.normalized_volume);
        assert!(sliver.radius_edge_ratio > regular.radius_edge_ratio);
    }

    #[test]
    fn shape_metrics_are_translation_invariant_and_volume_scales_cubically() {
        let points: [Point3<f64>; 4] = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let translated_and_scaled = points.map(|point| {
            Point3::new(
                11.0 + 7.0 * point.x,
                -3.0 + 7.0 * point.y,
                5.0 + 7.0 * point.z,
            )
        });
        let baseline = tetrahedron_quality(points).expect("baseline is valid");
        let transformed =
            tetrahedron_quality(translated_and_scaled).expect("transformed cell is valid");

        assert!((transformed.volume - baseline.volume * 343.0).abs() < 1e-12);
        assert!((transformed.radius_edge_ratio - baseline.radius_edge_ratio).abs() < 1e-12);
        assert!((transformed.min_dihedral_angle - baseline.min_dihedral_angle).abs() < 1e-12);
        assert!((transformed.normalized_volume - baseline.normalized_volume).abs() < 1e-12);
    }

    #[test]
    fn criteria_distinguish_slivers_shape_failures_and_oversized_cells() {
        let criteria = TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 0.5, Some(1.0))
            .expect("criteria are valid");
        let sliver = TetrahedronQuality {
            volume: 0.1,
            radius_edge_ratio: 1.5,
            min_dihedral_angle: 0.25,
            normalized_volume: 0.25,
        };
        let poor_shape = TetrahedronQuality {
            volume: 0.1,
            radius_edge_ratio: 2.5,
            min_dihedral_angle: 0.75,
            normalized_volume: 0.75,
        };
        let oversized = TetrahedronQuality {
            volume: 2.0,
            radius_edge_ratio: 1.5,
            min_dihedral_angle: 0.75,
            normalized_volume: 0.75,
        };
        let invalid = TetrahedronQuality {
            volume: 0.1,
            radius_edge_ratio: 1.5,
            min_dihedral_angle: 0.75,
            normalized_volume: 1.1,
        };

        assert_eq!(criteria.classify(sliver), TetrahedronQualityClass::Sliver);
        assert_eq!(
            criteria.classify(poor_shape),
            TetrahedronQualityClass::PoorShape
        );
        assert_eq!(
            criteria.classify(oversized),
            TetrahedronQualityClass::Oversized
        );
        assert_eq!(criteria.classify(invalid), TetrahedronQualityClass::Invalid);
    }

    #[test]
    fn criteria_assessment_is_native_and_counts_invalid_cells() {
        fn exercise<T: Scalar>() {
            let scalar = |value| <T as Scalar>::from_f64(value);
            let mut builder = TetrahedralMeshBuilder::<T>::new();
            let valid = [
                builder.vertex_array([scalar(0.0), scalar(0.0), scalar(0.0)]),
                builder.vertex_array([scalar(1.0), scalar(0.0), scalar(0.0)]),
                builder.vertex_array([scalar(0.0), scalar(1.0), scalar(0.0)]),
                builder.vertex_array([scalar(0.0), scalar(0.0), scalar(1.0)]),
            ];
            builder
                .tetrahedron(valid)
                .expect("valid tetrahedron is inserted");
            let mut mesh = builder.build();
            mesh.cells
                .push(crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0));

            let criteria = TetrahedralQualityCriteria::try_new(
                scalar(2.0),
                scalar(0.5),
                scalar(0.5),
                Some(scalar(1.0)),
            )
            .expect("criteria are valid");
            let acceptance = criteria.assess(&mesh).expect("tetrahedral cells exist");
            assert_eq!(acceptance.accepted_cell_count, 1);
            assert_eq!(acceptance.invalid_cell_count, 1);
            assert_eq!(acceptance.rejected_cell_count(), 1);
            assert!(!acceptance.passed());
        }

        exercise::<f32>();
        exercise::<f64>();
    }

    #[test]
    fn criteria_reject_non_finite_and_out_of_domain_bounds() {
        assert_eq!(
            TetrahedralQualityCriteria::<f64>::try_new(0.0, 0.5, 0.5, None),
            Err(TetrahedralQualityCriteriaError::InvalidMaxRadiusEdgeRatio)
        );
        assert_eq!(
            TetrahedralQualityCriteria::<f64>::try_new(2.0, -0.1, 0.5, None),
            Err(TetrahedralQualityCriteriaError::InvalidMinDihedralAngle)
        );
        assert_eq!(
            TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 1.1, None),
            Err(TetrahedralQualityCriteriaError::InvalidMinNormalizedVolume)
        );
        assert_eq!(
            TetrahedralQualityCriteria::<f64>::try_new(2.0, 0.5, 0.5, Some(0.0)),
            Err(TetrahedralQualityCriteriaError::InvalidMaxVolume)
        );
    }

    #[test]
    fn report_counts_invalid_tetrahedra_without_defaulting_metrics() {
        let mut builder = TetrahedralMeshBuilder::<f64>::new();
        let a = builder.vertex_xyz(0.0, 0.0, 0.0);
        let b = builder.vertex_xyz(1.0, 0.0, 0.0);
        let c = builder.vertex_xyz(0.0, 1.0, 0.0);
        let d = builder.vertex_xyz(0.0, 0.0, 1.0);
        builder
            .tetrahedron([a, b, c, d])
            .expect("valid tetrahedron");
        let mut mesh = builder.build();
        mesh.cells
            .push(crate::domain::topology::Cell::tetrahedron(0, 0, 0, 0));
        let report = tetrahedral_quality_report(&mesh).expect("tetrahedral cells exist");
        assert_eq!(report.valid_cell_count, 1);
        assert_eq!(report.invalid_cell_count, 1);
        assert_eq!(report.volume.expect("valid metric").count, 1);
    }
}
