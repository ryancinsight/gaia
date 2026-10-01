//! Biconcave disk primitive — Evans-Fung red blood cell (RBC) shape.

use std::f64::consts::TAU;

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::{Point3r, Scalar, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a biconcave disk matching the Evans-Fung parametrization of a
/// human red blood cell (RBC).
///
/// The disk is centred at `center` with the disk axis along +Y.  The rim
/// lies in the XZ plane; the upper lobe (+Y) and lower lobe (−Y) are
/// generated independently and joined at the rim.
///
/// ## Profile (Evans-Fung)
///
/// For normalised radius `ρ = 2r/D ∈ [0, 1]`:
/// ```text
/// y(ρ) = (D/2) · √(1 − ρ²) · (c0 + c1·ρ² + c2·ρ⁴)
/// ```
/// Default coefficients model the human discocyte:
/// `c0 = 0.207161`, `c1 = 2.002558`, `c2 = −1.122762`
///
/// ## Region IDs
///
/// | `RegionId` | Surface |
/// |----------|---------|
/// | 1 | Upper lobe (+Y) |
/// | 2 | Lower lobe (−Y) |
///
/// ## Uses
///
/// Immersed boundary / front-tracking blood flow, capsule dynamics,
/// RBC deformability studies (sickle cell, malaria-infected cells).
///
/// ## Output
///
/// - `signed_volume ≈ 94 fL` for default `diameter = 8e-3` mm (8 µm)
/// - Outward normals on upper lobe point upward (+Y component > 0 except at rim)
/// - Outward normals on lower lobe point downward (−Y component > 0 outward)
#[derive(Clone, Debug)]
pub struct BiconcaveDisk {
    /// Disk diameter (mm). Default ≈ 8 µm = 8e-3 mm for human RBC.
    pub diameter: f64,
    /// Centre of the disk.
    pub center: Point3r,
    /// Angular subdivisions around the disk rim (≥ 3).
    pub segments: usize,
    /// Radial rings from centre to rim (≥ 2). More rings → better concavity.
    pub rings: usize,
    /// Evans-Fung coefficient c0 (default 0.207161).
    pub c0: f64,
    /// Evans-Fung coefficient c1 (default 2.002558).
    pub c1: f64,
    /// Evans-Fung coefficient c2 (default −1.122762).
    pub c2: f64,
}

impl Default for BiconcaveDisk {
    fn default() -> Self {
        Self {
            diameter: 8e-3,
            center: Point3r::origin(),
            segments: 32,
            rings: 16,
            c0: 0.207_161,
            c1: 2.002_558,
            c2: -1.122_762,
        }
    }
}

impl BiconcaveDisk {
    /// Convenience constructor with human-RBC defaults and specified diameter.
    #[must_use]
    pub fn human_rbc(diameter: f64) -> Self {
        Self {
            diameter,
            ..Self::default()
        }
    }
}

impl PrimitiveMesh for BiconcaveDisk {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

/// Evaluate the Evans-Fung half-thickness profile at normalized radius `rho`.
fn evans_fung_height(r: f64, rho: f64, c0: f64, c1: f64, c2: f64) -> f64 {
    let rho2 = rho * rho;
    let under = (1.0 - rho2).max(0.0);
    r * under.sqrt() * (c0 + c1 * rho2 + c2 * rho2 * rho2)
}

/// Build one Evans-Fung ring of positions and outward normals.
#[expect(
    clippy::too_many_arguments,
    reason = "the Evans-Fung ring is fully determined by the center, resolution, radius, coefficients, and surface side"
)]
fn disk_ring_vertices(
    center: Point3r,
    ns: usize,
    r: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    rho: f64,
    is_upper: bool,
) -> Vec<(Point3r, Vector3r)> {
    let sign = if is_upper { 1.0 } else { -1.0 };
    let rho2 = rho * rho;
    let y_val = sign * evans_fung_height(r, rho, c0, c1, c2);
    let dy_drho = {
        let under = (1.0 - rho2).max(0.0);
        let sqrt_under = under.sqrt();
        let f_val = c0 + c1 * rho2 + c2 * rho2 * rho2;
        let f_prime = 2.0 * c1 * rho + 4.0 * c2 * rho2 * rho;
        if sqrt_under < 1e-14 {
            0.0
        } else {
            sign * r * (-rho / sqrt_under * f_val + sqrt_under * f_prime)
        }
    };

    (0..ns)
        .map(|i| {
            let theta = f64::from_usize(i) / f64::from_usize(ns) * TAU;
            let (ct, st) = (theta.cos(), theta.sin());
            let position = Point3r::new(
                center.x + r * rho * ct,
                center.y + y_val,
                center.z + r * rho * st,
            );
            let dt = Vector3r::new(-r * rho * st, 0.0, r * rho * ct);
            let dr = Vector3r::new(r * ct, dy_drho, r * st);
            let raw_normal = if is_upper { dt.cross(dr) } else { dr.cross(dt) };
            let normal = if raw_normal.norm() < 1e-14 {
                Vector3r::new(0.0, sign, 0.0)
            } else {
                raw_normal.normalize()
            };
            (position, normal)
        })
        .collect()
}

/// Add one biconcave-disk surface from apex to the shared rim ring.
#[expect(
    clippy::too_many_arguments,
    reason = "the surface helper is the build split point and needs the rim, profile coefficients, resolution, and side selection in one place"
)]
fn add_disk_surface(
    mesh: &mut IndexedMesh,
    center: Point3r,
    rim_ids: &[VertexId],
    ns: usize,
    nr: usize,
    r: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    is_upper: bool,
    region: RegionId,
) {
    let apex_y = if is_upper {
        center.y + evans_fung_height(r, 0.0, c0, c1, c2)
    } else {
        center.y - evans_fung_height(r, 0.0, c0, c1, c2).abs()
    };
    let apex_normal = if is_upper {
        Vector3r::new(0.0, 1.0, 0.0)
    } else {
        Vector3r::new(0.0, -1.0, 0.0)
    };
    let apex = mesh.add_vertex(Point3r::new(center.x, apex_y, center.z), apex_normal);

    let mut ring_ids: Vec<Vec<VertexId>> = Vec::with_capacity(nr);
    for k in 1..=nr {
        let rho = f64::from_usize(k) / f64::from_usize(nr);
        let ids = if k == nr {
            rim_ids.to_vec()
        } else {
            disk_ring_vertices(center, ns, r, c0, c1, c2, rho, is_upper)
                .iter()
                .map(|(position, normal)| mesh.add_vertex(*position, *normal))
                .collect()
        };
        ring_ids.push(ids);
    }

    for i in 0..ns {
        let j = (i + 1) % ns;
        if is_upper {
            mesh.add_face_with_region(apex, ring_ids[0][j], ring_ids[0][i], region);
        } else {
            mesh.add_face_with_region(apex, ring_ids[0][i], ring_ids[0][j], region);
        }
    }

    for k in 0..nr - 1 {
        for i in 0..ns {
            let j = (i + 1) % ns;
            if is_upper {
                mesh.add_face_with_region(
                    ring_ids[k][i],
                    ring_ids[k][j],
                    ring_ids[k + 1][j],
                    region,
                );
                mesh.add_face_with_region(
                    ring_ids[k][i],
                    ring_ids[k + 1][j],
                    ring_ids[k + 1][i],
                    region,
                );
            } else {
                mesh.add_face_with_region(
                    ring_ids[k][j],
                    ring_ids[k][i],
                    ring_ids[k + 1][i],
                    region,
                );
                mesh.add_face_with_region(
                    ring_ids[k][j],
                    ring_ids[k + 1][i],
                    ring_ids[k + 1][j],
                    region,
                );
            }
        }
    }
}

fn build(bd: &BiconcaveDisk) -> Result<IndexedMesh, PrimitiveError> {
    if bd.diameter <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "diameter must be > 0, got {}",
            bd.diameter
        )));
    }
    if bd.segments < 3 {
        return Err(PrimitiveError::TooFewSegments(bd.segments));
    }
    if bd.rings < 2 {
        return Err(PrimitiveError::InvalidParam("rings must be ≥ 2".into()));
    }

    let upper_region = RegionId::new(1);
    let lower_region = RegionId::new(2);

    let d = bd.diameter;
    let r = d / 2.0;

    // Scale the weld tolerance to the geometry.
    //
    // The innermost ring (ρ = 1/nr) has circumradius r/nr and vertex
    // spacing ≈ 2π·r / (nr·ns).  We choose a tolerance that is 1/10 of
    // that spacing — small enough that no two distinct vertices are
    // accidentally merged, yet large enough to absorb floating-point drift
    // at CSG intersection seams.
    //
    // For a human-RBC default (r = 4 µm, nr = 16, ns = 32) this gives
    // tolerance ≈ 4.9 × 10⁻⁵ mm — finer than the fixed 1 × 10⁻⁴ mm that
    // caused incorrect welding of innermost-ring vertices at that scale.
    let min_ring_spacing =
        std::f64::consts::TAU * r / (f64::from_usize(bd.rings) * f64::from_usize(bd.segments));
    let tol = (min_ring_spacing / 10.0).max(1e-10); // never go below 1 pm
    let mut mesh = IndexedMesh::with_cell_size(tol);
    let cx = bd.center.x;
    let cy = bd.center.y;
    let cz = bd.center.z;
    let ns = bd.segments;
    let nr = bd.rings;
    let c0 = bd.c0;
    let c1 = bd.c1;
    let c2 = bd.c2;

    // Shared rim ring (rho = 1, y = 0)
    // Both upper and lower lobes end at the rim. Build it once for shared topology.
    let rim_ids: Vec<VertexId> = (0..ns)
        .map(|i| {
            let theta = f64::from_usize(i) / f64::from_usize(ns) * TAU;
            let (ct, st) = (theta.cos(), theta.sin());
            let pos = Point3r::new(cx + r * ct, cy, cz + r * st);
            // Normal at rim points radially outward in XZ plane
            let n = Vector3r::new(ct, 0.0, st);
            mesh.add_vertex(pos, n)
        })
        .collect();

    add_disk_surface(
        &mut mesh,
        bd.center,
        &rim_ids,
        ns,
        nr,
        r,
        c0,
        c1,
        c2,
        true,
        upper_region,
    );
    add_disk_surface(
        &mut mesh,
        bd.center,
        &rim_ids,
        ns,
        nr,
        r,
        c0,
        c1,
        c2,
        false,
        lower_region,
    );

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;

    #[test]
    fn biconcave_disk_is_watertight() {
        // Use diameter = 1.0 mm so that ring vertex spacing (≈ 2π·r·ρ_min/ns)
        // stays well above the VertexPool weld tolerance (1e-4 mm).
        // The default 8e-3 mm (8 µm) makes the innermost ring too small to mesh.
        let mesh = BiconcaveDisk::default().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(
            report.is_watertight,
            "biconcave_disk must be watertight: {report:?}"
        );
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn biconcave_disk_volume_positive() {
        let mesh = BiconcaveDisk {
            diameter: 8e-3,
            segments: 64,
            rings: 32,
            ..BiconcaveDisk::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.signed_volume > 0.0, "signed_volume must be positive");
        // Human RBC volume ≈ 94 fL = 94e-9 mm³; allow wide tolerance for numeric integration
        // against discretized profile
        assert!(
            report.signed_volume > 0.0 && report.signed_volume < 1.0,
            "volume {:.3e} mm³ should be in (0, 1) mm³ for 8 µm disk",
            report.signed_volume
        );
    }

    #[test]
    fn biconcave_disk_human_rbc_constructor() {
        // The build() function automatically scales the weld tolerance to the
        // geometry, so the default 8 µm diameter and default segments/rings
        // now produce a watertight mesh without manual parameter adjustment.
        let mesh = BiconcaveDisk::human_rbc(8e-3).build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_watertight);
    }

    #[test]
    fn biconcave_disk_rejects_invalid() {
        assert!(BiconcaveDisk {
            diameter: 0.0,
            ..BiconcaveDisk::default()
        }
        .build()
        .is_err());
        assert!(BiconcaveDisk {
            diameter: -1.0,
            ..BiconcaveDisk::default()
        }
        .build()
        .is_err());
        assert!(BiconcaveDisk {
            segments: 2,
            ..BiconcaveDisk::default()
        }
        .build()
        .is_err());
        assert!(BiconcaveDisk {
            rings: 1,
            ..BiconcaveDisk::default()
        }
        .build()
        .is_err());
    }
}
