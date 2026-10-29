//! `Cdt` constructors: Delaunay build plus full constraint enforcement.

use super::Cdt;
use crate::application::delaunay::dim2::pslg::graph::Pslg;
use crate::application::delaunay::dim2::pslg::graph::PslgValidationError;
use crate::application::delaunay::dim2::triangulation::bowyer_watson::DelaunayTriangulation;
use crate::domain::core::scalar::Real;
use hashbrown::HashSet;

impl Cdt {
    /// Build a CDT from a PSLG.
    ///
    /// 1. Inserts all PSLG vertices into a Delaunay triangulation.
    /// 2. Enforces all constraint segments.
    /// 3. Removes triangles in hole regions.
    ///
    /// # Panics
    ///
    /// Panics if the PSLG contains degenerate/duplicate segments,
    /// out-of-range endpoints, or segment intersections. Use
    /// [`try_from_pslg`](Self::try_from_pslg) for fallible construction.
    #[must_use]
    #[track_caller]
    pub fn from_pslg(pslg: &Pslg) -> Self {
        Self::try_from_pslg(pslg)
            .unwrap_or_else(|err| panic!("Invalid PSLG for CDT construction: {err}"))
    }

    /// Build a CDT from a PSLG with validation.
    ///
    /// Returns an error if the PSLG contains degenerate/duplicate segments,
    /// out-of-range segment endpoints, or segment intersections.
    ///
    /// # Errors
    ///
    /// Returns [`PslgValidationError::NonFiniteVertex`] if any vertex
    /// coordinate is NaN or infinite, [`PslgValidationError::CoincidentVertices`]
    /// if two vertices collapse within tolerance, and the corresponding
    /// segment validation variants for out-of-range endpoints, degenerate
    /// segments, duplicate segments, or intersecting constraints.
    pub fn try_from_pslg(pslg: &Pslg) -> Result<Self, PslgValidationError> {
        pslg.validate()?;

        // Build the Delaunay triangulation from the PSLG vertices.
        let points: Vec<(Real, Real)> = pslg.vertices().iter().map(|v| (v.x, v.y)).collect();

        let dt = DelaunayTriangulation::from_points(&points);

        let mut cdt = Self {
            dt,
            constrained_edges: HashSet::with_capacity(pslg.segments().len()),
        };

        // Enforce each constraint segment.
        for seg in pslg.segments() {
            cdt.enforce_constraint(seg.start, seg.end);
        }

        // Remove hole triangles.
        for hole in pslg.holes() {
            cdt.remove_hole_triangles(hole.x, hole.y);
        }

        Ok(cdt)
    }
}
