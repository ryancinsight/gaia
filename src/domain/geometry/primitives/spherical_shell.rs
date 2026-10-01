//! Hollow sphere (spherical shell) primitive.

use std::f64::consts::PI;
use std::f64::consts::TAU;

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::{Point3r, Scalar, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a hollow sphere — two concentric sphere surfaces connected by
/// annular polar caps.
///
/// The shell is centred at `center`.  The outer surface has outward normals
/// pointing *away* from the centre; the inner surface has outward normals
/// pointing *toward* the centre (i.e., outward from the enclosed solid
/// material).
///
/// ## Construction
///
/// Both sphere surfaces are UV-parametrised over φ ∈ [φ₁, π − φ₁] where
/// φ₁ = π / stacks (one step from each pole).  This leaves a small polar
/// opening at each end that is closed by an annular quad strip — exactly
/// the same topology as [`Pipe`].
///
/// ## Topology
///
/// The through-bore (the polar hole connecting outer and inner surfaces)
/// creates a genus-1 handle, so `V − E + F = 0`  (χ = 0), identical to
/// `Torus` and [`Pipe`].
///
/// ## Output
///
/// - `signed_volume = (4/3)·π·(r_outer³ − r_inner³)`
/// - All faces tagged `RegionId(1)`
///
/// [`Pipe`]: super::pipe::Pipe
#[derive(Clone, Debug)]
pub struct SphericalShell {
    /// Centre of the shell.
    pub center: Point3r,
    /// Outer sphere radius (mm).
    pub outer_radius: f64,
    /// Inner sphere (cavity) radius (mm). Must be < `outer_radius`.
    pub inner_radius: f64,
    /// Angular subdivisions around the equator (≥ 3).
    pub segments: usize,
    /// Latitude subdivisions (≥ 3). Determines shell resolution.
    pub stacks: usize,
}

impl Default for SphericalShell {
    fn default() -> Self {
        Self {
            center: Point3r::origin(),
            outer_radius: 1.0,
            inner_radius: 0.9,
            segments: 32,
            stacks: 16,
        }
    }
}

impl PrimitiveMesh for SphericalShell {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

/// Build the latitude rings for one shell surface.
fn build_shell_rings(
    mesh: &mut IndexedMesh,
    r: f64,
    center: Point3r,
    ns: usize,
    nk: usize,
    is_outer: bool,
) -> Vec<Vec<VertexId>> {
    let segments = f64::from_usize(ns);
    let stacks = f64::from_usize(nk);

    (0..nk - 1)
        .map(|k| {
            let phi = f64::from_usize(k + 1) / stacks * PI;
            let sp = phi.sin();
            let cp = phi.cos();
            let y = center.y + r * cp;

            (0..ns)
                .map(|j| {
                    let theta = f64::from_usize(j) / segments * TAU;
                    let ct = theta.cos();
                    let st = theta.sin();
                    let position = Point3r::new(center.x + r * sp * ct, y, center.z + r * sp * st);
                    let normal = if is_outer {
                        Vector3r::new(sp * ct, cp, sp * st)
                    } else {
                        Vector3r::new(-sp * ct, -cp, -sp * st)
                    };
                    mesh.add_vertex(position, normal)
                })
                .collect()
        })
        .collect()
}

/// Add the outer and inner ring bands between adjacent shell latitudes.
fn add_shell_bands(
    mesh: &mut IndexedMesh,
    outer_rings: &[Vec<VertexId>],
    inner_rings: &[Vec<VertexId>],
    ns: usize,
    region: RegionId,
) {
    for k in 0..outer_rings.len() - 1 {
        for j in 0..ns {
            let j1 = (j + 1) % ns;
            let vu0 = outer_rings[k][j];
            let vu1 = outer_rings[k][j1];
            let vl0 = outer_rings[k + 1][j];
            let vl1 = outer_rings[k + 1][j1];
            mesh.add_face_with_region(vu0, vl0, vl1, region);
            mesh.add_face_with_region(vu0, vl1, vu1, region);
        }
    }

    for k in 0..inner_rings.len() - 1 {
        for j in 0..ns {
            let j1 = (j + 1) % ns;
            let vu0 = inner_rings[k][j];
            let vu1 = inner_rings[k][j1];
            let vl0 = inner_rings[k + 1][j];
            let vl1 = inner_rings[k + 1][j1];
            mesh.add_face_with_region(vu0, vu1, vl1, region);
            mesh.add_face_with_region(vu0, vl1, vl0, region);
        }
    }
}

/// Add the annular polar caps that connect the shell surfaces.
fn add_shell_caps(
    mesh: &mut IndexedMesh,
    outer_rings: &[Vec<VertexId>],
    inner_rings: &[Vec<VertexId>],
    ns: usize,
    region: RegionId,
) {
    for j in 0..ns {
        let j1 = (j + 1) % ns;
        let oi = outer_rings[0][j];
        let oi1 = outer_rings[0][j1];
        let ii = inner_rings[0][j];
        let ii1 = inner_rings[0][j1];
        mesh.add_face_with_region(oi, ii1, ii, region);
        mesh.add_face_with_region(oi, oi1, ii1, region);
    }

    let outer_last = outer_rings.len() - 1;
    let inner_last = inner_rings.len() - 1;
    for j in 0..ns {
        let j1 = (j + 1) % ns;
        let oi = outer_rings[outer_last][j];
        let oi1 = outer_rings[outer_last][j1];
        let ii = inner_rings[inner_last][j];
        let ii1 = inner_rings[inner_last][j1];
        mesh.add_face_with_region(ii, ii1, oi1, region);
        mesh.add_face_with_region(ii, oi1, oi, region);
    }
}

fn build(s: &SphericalShell) -> Result<IndexedMesh, PrimitiveError> {
    if s.inner_radius <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "inner_radius must be > 0, got {}",
            s.inner_radius
        )));
    }
    if s.outer_radius <= s.inner_radius {
        return Err(PrimitiveError::InvalidParam(format!(
            "outer_radius ({}) must be > inner_radius ({})",
            s.outer_radius, s.inner_radius
        )));
    }
    if s.segments < 3 {
        return Err(PrimitiveError::TooFewSegments(s.segments));
    }
    if s.stacks < 3 {
        return Err(PrimitiveError::InvalidParam("stacks must be ≥ 3".into()));
    }

    let region = RegionId::new(1);
    let mut mesh = IndexedMesh::new();

    let ro = s.outer_radius;
    let ri = s.inner_radius;
    let cx = s.center.x;
    let cy = s.center.y;
    let cz = s.center.z;
    let ns = s.segments;
    let nk = s.stacks;
    let center = Point3r::new(cx, cy, cz);
    let outer_rings = build_shell_rings(&mut mesh, ro, center, ns, nk, true);
    let inner_rings = build_shell_rings(&mut mesh, ri, center, ns, nk, false);
    add_shell_bands(&mut mesh, &outer_rings, &inner_rings, ns, region);
    add_shell_caps(&mut mesh, &outer_rings, &inner_rings, ns, region);

    // All sections (outer lateral, inner lateral, north/south polar caps) are built
    // with consistent inward winding. Flip all faces to obtain outward-pointing normals
    // and positive signed volume.
    mesh.flip_faces();

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;
    use std::f64::consts::PI;

    #[test]
    fn spherical_shell_is_closed_and_oriented() {
        let mesh = SphericalShell::default().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_closed, "shell must be closed");
        assert!(
            report.orientation_consistent,
            "shell must be consistently oriented"
        );
        // Genus 1 (through-bore at poles) → χ = 0
        assert_eq!(
            report.euler_characteristic,
            Some(0),
            "spherical shell Euler characteristic must be 0 (genus 1)"
        );
        assert!(report.is_watertight, "shell passes watertight check");
    }

    #[test]
    fn spherical_shell_volume_positive_and_approximately_correct() {
        let (ri, ro) = (0.9_f64, 1.0_f64);
        let mesh = SphericalShell {
            outer_radius: ro,
            inner_radius: ri,
            segments: 64,
            stacks: 32,
            ..SphericalShell::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.signed_volume > 0.0);
        let expected = 4.0 / 3.0 * PI * (ro * ro * ro - ri * ri * ri);
        let error = (report.signed_volume - expected).abs() / expected;
        assert!(
            error < 0.01,
            "volume error {:.4}% should be < 1%",
            error * 100.0
        );
    }

    #[test]
    fn spherical_shell_rejects_invalid_params() {
        assert!(SphericalShell {
            inner_radius: 0.0,
            ..SphericalShell::default()
        }
        .build()
        .is_err());
        assert!(SphericalShell {
            outer_radius: 0.8,
            inner_radius: 0.9,
            ..SphericalShell::default()
        }
        .build()
        .is_err());
        assert!(SphericalShell {
            segments: 2,
            ..SphericalShell::default()
        }
        .build()
        .is_err());
        assert!(SphericalShell {
            stacks: 2,
            ..SphericalShell::default()
        }
        .build()
        .is_err());
    }
}
