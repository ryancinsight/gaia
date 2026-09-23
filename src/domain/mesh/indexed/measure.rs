//! Measurement and validation queries: extent, area, volume, watertightness,
//! and the quality report.

use super::IndexedMesh;
use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::aabb::Aabb;

impl<T: Scalar> IndexedMesh<T> {
    // ── Geometric queries ─────────────────────────────────────────────────

    /// Axis-aligned bounding box.
    pub fn bounding_box(&self) -> Aabb<T> {
        Aabb::from_points(self.vertices.positions())
    }

    /// Total surface area of all triangles.
    pub fn surface_area(&self) -> T {
        use crate::domain::geometry::measure;
        measure::total_surface_area(self.faces.iter_enumerated().map(|(_, f)| {
            (
                self.vertices.position(f.vertices[0]),
                self.vertices.position(f.vertices[1]),
                self.vertices.position(f.vertices[2]),
            )
        }))
    }

    /// Signed volume (positive for outward-oriented closed mesh).
    pub fn signed_volume(&self) -> T {
        use crate::domain::geometry::measure;
        measure::total_signed_volume(self.faces.iter_enumerated().map(|(_, f)| {
            (
                self.vertices.position(f.vertices[0]),
                self.vertices.position(f.vertices[1]),
                self.vertices.position(f.vertices[2]),
            )
        }))
    }

    // ── Validation ────────────────────────────────────────────────────────

    /// Check watertightness (rebuilds edges if needed).
    pub fn is_watertight(&mut self) -> bool {
        self.rebuild_edges();
        let edges = self
            .edges
            .as_ref()
            .expect("invariant: rebuild_edges() sets edges to Some");
        let report = crate::application::watertight::check::check_watertight(
            &self.vertices,
            &self.faces,
            edges,
        );
        report.is_watertight
    }

    /// Run quality validation against default thresholds.
    pub fn quality_report(&self) -> crate::application::quality::validation::QualityReport {
        let validator = crate::application::quality::validation::MeshValidator::default();
        validator.validate(&self.faces, &self.vertices)
    }
}
