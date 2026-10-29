//! Core tetrahedral data types: per-cell metrics, reports, criteria errors.

use crate::application::quality::metrics::QualityMetric;

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
