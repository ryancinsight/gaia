//! Native-precision quality metrics for linear tetrahedral cells.
//!
//! The metrics are dimensionless except for volume.  They are evaluated in
//! the mesh scalar `T` and converted to `f64` only when a report is assembled,
//! so `IndexedMesh<f32>` does not silently execute its quality kernel in
//! double precision.

mod criteria;
mod measure;
mod types;

#[cfg(test)]
mod tests;

pub use criteria::{
    TetrahedralQualityAcceptance, TetrahedralQualityCriteria, TetrahedronQualityClass,
};
pub(crate) use measure::cell_tetrahedron_quality;
pub use measure::{tetrahedral_quality_report, tetrahedron_quality};
pub use types::{TetrahedralQualityCriteriaError, TetrahedralQualityReport, TetrahedronQuality};
