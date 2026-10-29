//! Tetrahedral acceptance criteria: validation, classification, assessment.

use aequitas::systems::si::quantities::{Angle, Dimensionless};

use crate::application::quality::boundary::{
    assess_boundary_cells, BoundaryFacetQualityCriteria, BoundaryTetrahedralQualityAcceptance,
};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::ElementType;

use super::measure::tetrahedral_measurements;
use super::types::{TetrahedralQualityCriteriaError, TetrahedronQuality};

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

    /// Create validated criteria with aequitas-typed angle/ratio bounds.
    ///
    /// This constructor is a dimension-safe wrapper over [`Self::try_new`]:
    /// the radius-edge ratio and normalized-volume bounds are dimensionless,
    /// while the dihedral-angle bound is explicitly typed as radians.
    ///
    /// # Errors
    ///
    /// Returns the same validation error as [`Self::try_new`] when any bound
    /// is non-finite or outside its mathematical domain after conversion to
    /// base units.
    #[must_use = "handle invalid tetrahedral criteria"]
    pub fn try_new_typed(
        max_radius_edge_ratio: Dimensionless<T>,
        min_dihedral_angle: Angle<T>,
        min_normalized_volume: Dimensionless<T>,
        max_volume: Option<T>,
    ) -> Result<Self, TetrahedralQualityCriteriaError> {
        Self::try_new(
            max_radius_edge_ratio.into_base(),
            min_dihedral_angle.into_base(),
            min_normalized_volume.into_base(),
            max_volume,
        )
    }

    /// Return the maximum accepted radius-edge ratio.
    #[must_use]
    pub fn max_radius_edge_ratio(&self) -> T {
        self.max_radius_edge_ratio
    }

    /// Return the maximum accepted radius-edge ratio as a typed quantity.
    #[must_use]
    pub fn max_radius_edge_ratio_quantity(&self) -> Dimensionless<T> {
        Dimensionless::from_base(self.max_radius_edge_ratio)
    }

    /// Return the minimum accepted interior dihedral angle in radians.
    #[must_use]
    pub fn min_dihedral_angle(&self) -> T {
        self.min_dihedral_angle
    }

    /// Return the minimum accepted interior dihedral angle as a typed quantity.
    #[must_use]
    pub fn min_dihedral_angle_quantity(&self) -> Angle<T> {
        Angle::from_base(self.min_dihedral_angle)
    }

    /// Return the minimum accepted equilateral-normalized volume.
    #[must_use]
    pub fn min_normalized_volume(&self) -> T {
        self.min_normalized_volume
    }

    /// Return the minimum accepted normalized volume as a typed quantity.
    #[must_use]
    pub fn min_normalized_volume_quantity(&self) -> Dimensionless<T> {
        Dimensionless::from_base(self.min_normalized_volume)
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
