//! Branching (bifurcation / trifurcation) mesh builder.
//!
//! Builds a structured mesh for a Y-shaped or T-shaped branching passage.
//! Use [`BranchingMeshBuilder::build_surface`] for the modern [`IndexedMesh`]
//! boundary-surface output.
//!
//! ## Design Note
//!
//! All geometry and arithmetic is performed in `f64` (`Real`).  A generic
//! `<T: Scalar>` parameter would be a fake generic (core_invariants rule 2)
//! because the algorithm uses `sin`/`cos`, square-root normalisation, and CSG
//! Boolean union â€” all operating natively in `f64`.  Parametrising `T` would
//! silently zero-out geometry via `unwrap_or(0.0)` on conversion failure.

use crate::application::channel::venturi::BuildError;
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::Real;
use crate::domain::mesh::IndexedMesh;

mod tube;

/// Builds a branching (bifurcation) flow passage mesh.
///
/// All length/geometry parameters are in metres (`f64`).
#[derive(Clone, Debug)]
pub struct BranchingMeshBuilder {
    /// Parent tube diameter (m).
    pub d_parent: Real,
    /// Parent tube length (m).
    pub l_parent: Real,
    /// Daughter tube diameter (m).
    pub d_daughter: Real,
    /// Daughter tube length (m).
    pub l_daughter: Real,
    /// Half-angle of branching (radians), in `(0, π/2)`.
    pub branching_angle: Real,
    /// Axial mesh resolution per tube segment; must be at least four rings.
    pub resolution: usize,
    /// Number of daughter branches (2 = bifurcation, 3 = trifurcation).
    pub n_daughters: usize,
}

impl BranchingMeshBuilder {
    /// Create a symmetric bifurcation (1 parent, 2 daughters).
    #[must_use]
    pub fn bifurcation(
        d_parent: Real,
        l_parent: Real,
        d_daughter: Real,
        l_daughter: Real,
        branching_angle: Real,
        resolution: usize,
    ) -> Self {
        Self {
            d_parent,
            l_parent,
            d_daughter,
            l_daughter,
            branching_angle,
            resolution,
            n_daughters: 2,
        }
    }

    /// Create a symmetric trifurcation (1 parent, 3 daughters).
    #[must_use]
    pub fn trifurcation(
        d_parent: Real,
        l_parent: Real,
        d_daughter: Real,
        l_daughter: Real,
        branching_angle: Real,
        resolution: usize,
    ) -> Self {
        Self {
            d_parent,
            l_parent,
            d_daughter,
            l_daughter,
            branching_angle,
            resolution,
            n_daughters: 3,
        }
    }

    /// Build a watertight surface mesh (parent + daughter walls, inlet, and outlet caps).
    ///
    /// Region IDs:
    /// - `RegionId(0)` â€” wall (all tube surfaces)
    /// - `RegionId(1)` â€” inlet cap (parent inlet)
    /// - `RegionId(2+d)` â€” outlet cap for daughter `d`
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when a dimension, angle, daughter count, or
    /// resolution violates the builder contract, when capacity arithmetic
    /// overflows, or when the CSG union cannot produce a watertight result.
    pub fn build_surface(&self) -> Result<IndexedMesh, BuildError> {
        build_branching_surface(self)
    }
}

fn build_branching_surface(b: &BranchingMeshBuilder) -> Result<IndexedMesh, BuildError> {
    let mesh = build_branching_union(b)?;
    validate_branching_result(&mesh, b)?;
    Ok(mesh)
}

fn build_branching_union(b: &BranchingMeshBuilder) -> Result<IndexedMesh, BuildError> {
    validate_parameters(b)?;

    let d_parent = b.d_parent;
    let l_parent = b.l_parent;
    let d_daughter = b.d_daughter;
    let l_daughter = b.l_daughter;
    let branching_angle = b.branching_angle;

    let r_parent = d_parent / 2.0_f64;
    let r_daughter = d_daughter / 2.0_f64;
    let n_ax = b.resolution;
    // Angular resolution derived from the tube leaf â€” consistent with
    // venturi/serpentine.
    let n_ang = tube::ANGULAR_SEGMENTS;
    let vertex_capacity = n_ax
        .checked_mul(n_ang)
        .and_then(|capacity| capacity.checked_add(2))
        .ok_or_else(|| BuildError("branching resolution overflows vertex capacity".into()))?;
    let face_capacity = n_ax
        .checked_sub(1)
        .and_then(|steps| steps.checked_mul(n_ang))
        .and_then(|faces| faces.checked_mul(2))
        .and_then(|faces| faces.checked_add(n_ang.checked_mul(2)?))
        .ok_or_else(|| BuildError("branching resolution overflows face capacity".into()))?;

    let mut meshes = Vec::with_capacity(1 + b.n_daughters);

    // 1. Parent tube â€” extend slightly past l_parent to ensure solid overlap for CSG union.
    let parent_overlap = r_parent * 1.5;
    let mesh_parent = tube::build_closed_tube(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, l_parent + parent_overlap),
        r_parent,
        n_ax,
        true,
        0,
        vertex_capacity,
        face_capacity,
    );
    meshes.push(mesh_parent);

    // 2. Daughter tubes
    for d in 0..b.n_daughters {
        let angle_step = if b.n_daughters == 1 {
            0.0_f64
        } else {
            branching_angle * (d as f64 - (b.n_daughters - 1) as f64 / 2.0_f64)
        };
        let sin_a = angle_step.sin();
        let cos_a = angle_step.cos();

        // Start daughter tube deep inside the parent to guarantee volume overlap.
        let overlap_dist = r_parent * 1.5;
        let start_x = -overlap_dist * sin_a;
        let start_y = 0.0;
        let start_z = l_parent - overlap_dist * cos_a;

        // End position
        let run_dist = l_daughter + overlap_dist;
        let dx = run_dist * sin_a;
        let dy = 0.0;
        let dz = run_dist * cos_a;

        let mesh_d = tube::build_closed_tube(
            (start_x, start_y, start_z),
            (dx, dy, dz),
            r_daughter,
            n_ax,
            false,
            d,
            vertex_capacity,
            face_capacity,
        );
        meshes.push(mesh_d);
    }

    // 3. Boolean Union across all branch bounds.
    use crate::application::csg::boolean::{csg_boolean_nary, BooleanOp};
    csg_boolean_nary(BooleanOp::Union, &meshes)
        .map_err(|e| BuildError(format!("CSG Boolean failed on branch connection: {e:?}")))
}

fn validate_parameters(b: &BranchingMeshBuilder) -> Result<(), BuildError> {
    let dimensions = [
        ("d_parent", b.d_parent),
        ("l_parent", b.l_parent),
        ("d_daughter", b.d_daughter),
        ("l_daughter", b.l_daughter),
    ];
    if let Some((name, value)) = dimensions
        .into_iter()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(BuildError(format!(
            "branching parameter {name} must be finite and > 0, got {value}"
        )));
    }
    if !b.branching_angle.is_finite()
        || b.branching_angle <= 0.0
        || b.branching_angle >= std::f64::consts::FRAC_PI_2
    {
        return Err(BuildError(format!(
            "branching_angle must be finite and lie in (0, π/2), got {}",
            b.branching_angle
        )));
    }
    if b.resolution < 4 {
        return Err(BuildError(format!(
            "resolution must be at least 4, got {}",
            b.resolution
        )));
    }
    if !matches!(b.n_daughters, 2 | 3) {
        return Err(BuildError(format!(
            "n_daughters must be 2 or 3, got {}",
            b.n_daughters
        )));
    }
    Ok(())
}

fn validate_branching_result(
    mesh: &IndexedMesh,
    b: &BranchingMeshBuilder,
) -> Result<(), BuildError> {
    let r_daughter = b.d_daughter / 2.0;
    let half_daughters = (b.n_daughters - 1) as Real / 2.0;
    for daughter in 0..b.n_daughters {
        let angle = b.branching_angle * (daughter as Real - half_daughters);
        let expected_x = b.l_daughter * angle.sin();
        let expected_z = b.l_parent + b.l_daughter * angle.cos();
        let has_outlet_neighborhood = mesh.vertices.positions().any(|point| {
            let dx = point.x - expected_x;
            let dy = point.y;
            let dz = point.z - expected_z;
            dx * dx + dy * dy + dz * dz <= (r_daughter * 1.5).powi(2)
        });
        let outlet_region = RegionId::from_usize(2 + daughter);
        let has_outlet_region = mesh.faces.iter().any(|face| face.region == outlet_region);
        if !has_outlet_neighborhood || !has_outlet_region {
            return Err(BuildError(format!(
                "CSG Boolean omitted daughter outlet geometry for daughter {daughter}"
            )));
        }
    }
    Ok(())
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bifurcation_struct_construction() {
        // Validates parameter binding without running the expensive CSG pipeline.
        let b = BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 4);
        assert_eq!(b.n_daughters, 2);
        assert!((b.d_parent - 0.004).abs() < 1e-14);
        assert_eq!(b.resolution, 4);
    }

    #[test]
    fn trifurcation_struct_construction() {
        let b = BranchingMeshBuilder::trifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 6);
        assert_eq!(b.n_daughters, 3);
    }

    #[test]
    fn invalid_branching_parameters_are_rejected() {
        let mut invalid_daughters =
            BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 4);
        invalid_daughters.n_daughters = 4;
        let cases = [
            (
                BranchingMeshBuilder::bifurcation(0.0, 0.020, 0.002, 0.015, 0.5, 4),
                "branching parameter d_parent must be finite and > 0, got 0",
            ),
            (
                BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.0, 4),
                "branching_angle must be finite and lie in (0, π/2), got 0",
            ),
            (
                BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 3),
                "resolution must be at least 4, got 3",
            ),
            (invalid_daughters, "n_daughters must be 2 or 3, got 4"),
            (
                BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, usize::MAX),
                "branching resolution overflows vertex capacity",
            ),
        ];
        for (builder, expected) in cases {
            let error = match builder.build_surface() {
                Ok(_) => panic!("invalid branching parameters unexpectedly built a mesh"),
                Err(error) => error,
            };
            assert_eq!(error.0, expected);
        }
    }

    #[test]
    fn small_daughter_trifurcation_builds_without_weld_collapse() {
        let mut mesh = BranchingMeshBuilder::trifurcation(0.004, 0.020, 0.001, 0.010, 0.5, 4)
            .build_surface()
            .expect("small daughter trifurcation");
        mesh.rebuild_edges();
        let report = crate::application::watertight::check::check_watertight(
            &mesh.vertices,
            &mesh.faces,
            mesh.edges_ref().expect("edges rebuilt"),
        );
        assert!(
            report.is_watertight,
            "small daughter mesh must be watertight"
        );
        assert!(
            report.signed_volume > 0.0,
            "small daughter volume must be positive"
        );
    }

    #[test]
    fn representative_branching_union_preserves_watertight_outlets() {
        let builder = BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 4);
        let mut mesh = build_branching_union(&builder).expect("branch union");
        mesh.rebuild_edges();
        let report = crate::application::watertight::check::check_watertight(
            &mesh.vertices,
            &mesh.faces,
            mesh.edges_ref().expect("edges rebuilt"),
        );
        assert!(
            report.is_watertight,
            "branch union must be watertight: boundary={} non_manifold={}",
            report.boundary_edge_count, report.non_manifold_edge_count,
        );
        for daughter in 0..builder.n_daughters {
            let outlet_region = RegionId::from_usize(2 + daughter);
            assert!(
                mesh.faces.iter().any(|face| face.region == outlet_region),
                "branch union omitted outlet region {daughter}",
            );
        }
    }

    #[test]
    fn representative_branching_cases_build_watertight_meshes() {
        let builders = [
            BranchingMeshBuilder::bifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 4),
            BranchingMeshBuilder::trifurcation(0.004, 0.020, 0.002, 0.015, 0.5, 6),
        ];
        for builder in builders {
            let mut mesh = builder
                .build_surface()
                .expect("representative branching mesh");
            mesh.rebuild_edges();
            let report = crate::application::watertight::check::check_watertight(
                &mesh.vertices,
                &mesh.faces,
                mesh.edges_ref().expect("edges rebuilt"),
            );
            assert!(
                report.is_watertight,
                "representative branch must be watertight"
            );
            assert!(
                report.signed_volume > 0.0,
                "representative branch volume must be positive"
            );
            for daughter in 0..builder.n_daughters {
                let outlet_region = RegionId::from_usize(2 + daughter);
                assert!(
                    mesh.faces.iter().any(|face| face.region == outlet_region),
                    "representative branch omitted outlet region {daughter}",
                );
            }
        }
    }
}
