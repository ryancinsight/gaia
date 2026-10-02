//! Capsule primitive (cylinder + hemispherical end caps).

use eunomia::FloatElement;
use std::f64::consts::{PI, TAU};

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a capsule: a closed right cylinder capped with two hemispheres.
///
/// The capsule is aligned with the +Y axis and centred at `center`.
/// The total height is `cylinder_height + 2 × radius`.
///
/// ## Mesh structure
///
/// | Section | Faces |
/// |---------|-------|
/// | Top hemisphere (`hemisphere_stacks` stacks from +Y apex to equator) | `segments × (hemisphere_stacks − 1) × 2 + segments` |
/// | Lateral cylinder | `2 × segments` |
/// | Bottom hemisphere (`hemisphere_stacks` stacks from equator to −Y apex) | `segments × (hemisphere_stacks − 1) × 2 + segments` |
///
/// Shared equatorial rings at `y = center.y ± cylinder_height/2` are
/// automatically welded by `VertexPool` spatial-hash deduplication.
///
/// ## Uses
///
/// Bacteria (E. coli, B. subtilis), pharmaceutical drug-delivery capsules,
/// elongated droplets, flexible particles in Lagrangian–Eulerian blood flow.
///
/// ## Output
///
/// - `signed_volume ≈ π r² (cylinder_height + 4r/3)`
/// - All faces tagged `RegionId(1)`
#[derive(Clone, Debug)]
pub struct Capsule {
    /// Hemisphere and cylinder radius (mm).
    pub radius: f64,
    /// Length of the cylindrical midsection (mm). May be 0 (sphere).
    pub cylinder_height: f64,
    /// Centre of the capsule.
    pub center: Point3r,
    /// Angular subdivisions around the axis (≥ 3).
    pub segments: usize,
    /// Latitude subdivisions per hemisphere (≥ 1).
    pub hemisphere_stacks: usize,
}

impl Default for Capsule {
    fn default() -> Self {
        Self {
            radius: 0.5,
            cylinder_height: 1.0,
            center: Point3r::origin(),
            segments: 32,
            hemisphere_stacks: 8,
        }
    }
}

impl PrimitiveMesh for Capsule {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

/// Add one hemispherical cap to a capsule mesh.
#[expect(
    clippy::too_many_arguments,
    reason = "the helper mirrors the capsule build parameters needed to place one hemisphere without introducing a one-off configuration type"
)]
fn add_hemisphere(
    mesh: &mut IndexedMesh,
    r: f64,
    center_y: f64,
    cx: f64,
    cz: f64,
    flip: bool,
    ns: usize,
    hs: usize,
    region: RegionId,
) {
    let segments = f64::from_count(ns);
    let hemisphere_stacks = f64::from_count(hs);

    for i in 0..ns {
        let t0 = f64::from_count(i) / segments * TAU;
        let t1 = f64::from_count(i + 1) / segments * TAU;
        for j in 0..hs {
            let (phi0, phi1) = if flip {
                (
                    PI / 2.0 + f64::from_count(j) / hemisphere_stacks * PI / 2.0,
                    PI / 2.0 + f64::from_count(j + 1) / hemisphere_stacks * PI / 2.0,
                )
            } else {
                (
                    f64::from_count(j) / hemisphere_stacks * PI / 2.0,
                    f64::from_count(j + 1) / hemisphere_stacks * PI / 2.0,
                )
            };

            let vertex_at = |theta: f64, phi: f64| -> (Point3r, Vector3r) {
                let sp = phi.sin();
                let cp = phi.cos();
                let ct = theta.cos();
                let st = theta.sin();
                let normal = Vector3r::new(sp * ct, cp, sp * st);
                let point = Point3r::new(cx + r * sp * ct, center_y + r * cp, cz + r * sp * st);
                (point, normal)
            };

            let (p00, n00) = vertex_at(t0, phi0);
            let (p10, n10) = vertex_at(t1, phi0);
            let (p11, n11) = vertex_at(t1, phi1);
            let (p01, n01) = vertex_at(t0, phi1);

            let v00 = mesh.add_vertex(p00, n00);
            let v10 = mesh.add_vertex(p10, n10);
            let v11 = mesh.add_vertex(p11, n11);
            let v01 = mesh.add_vertex(p01, n01);

            if !flip && j == 0 {
                mesh.add_face_with_region(v10, v11, v01, region);
            } else if flip && j == hs - 1 {
                mesh.add_face_with_region(v00, v10, v01, region);
            } else {
                mesh.add_face_with_region(v00, v10, v11, region);
                mesh.add_face_with_region(v00, v11, v01, region);
            }
        }
    }
}

/// Add the cylindrical mid-band shared by the two capsule hemispheres.
#[expect(
    clippy::too_many_arguments,
    reason = "the helper needs radius, centers, tessellation, and region inputs to reuse the capsule cylinder band without obscuring the geometry"
)]
fn add_cylinder_band(
    mesh: &mut IndexedMesh,
    r: f64,
    cx: f64,
    cy_bot: f64,
    cy_top: f64,
    cz: f64,
    ns: usize,
    region: RegionId,
) {
    let segments = f64::from_count(ns);
    for i in 0..ns {
        let t0 = f64::from_count(i) / segments * TAU;
        let t1 = f64::from_count(i + 1) / segments * TAU;
        let (c0, s0) = (t0.cos(), t0.sin());
        let (c1, s1) = (t1.cos(), t1.sin());
        let n0 = Vector3r::new(c0, 0.0, s0);
        let n1 = Vector3r::new(c1, 0.0, s1);

        let vb0 = mesh.add_vertex(Point3r::new(cx + r * c0, cy_bot, cz + r * s0), n0);
        let vb1 = mesh.add_vertex(Point3r::new(cx + r * c1, cy_bot, cz + r * s1), n1);
        let vt0 = mesh.add_vertex(Point3r::new(cx + r * c0, cy_top, cz + r * s0), n0);
        let vt1 = mesh.add_vertex(Point3r::new(cx + r * c1, cy_top, cz + r * s1), n1);

        mesh.add_face_with_region(vb0, vt0, vt1, region);
        mesh.add_face_with_region(vb0, vt1, vb1, region);
    }
}

fn build(cap: &Capsule) -> Result<IndexedMesh, PrimitiveError> {
    if cap.radius <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "radius must be > 0, got {}",
            cap.radius
        )));
    }
    if cap.cylinder_height < 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "cylinder_height must be ≥ 0, got {}",
            cap.cylinder_height
        )));
    }
    if cap.segments < 3 {
        return Err(PrimitiveError::TooFewSegments(cap.segments));
    }
    if cap.hemisphere_stacks < 1 {
        return Err(PrimitiveError::InvalidParam(
            "hemisphere_stacks must be ≥ 1".into(),
        ));
    }

    let region = RegionId::new(1);
    let mut mesh = IndexedMesh::new();
    let r = cap.radius;
    let hl = cap.cylinder_height;
    let cx = cap.center.x;
    let cy = cap.center.y;
    let cz = cap.center.z;
    let ns = cap.segments;
    let hs = cap.hemisphere_stacks;

    // Y-offsets for the two hemisphere centres (= cylinder cap positions).
    let top_cy = cy + hl / 2.0;
    let bot_cy = cy - hl / 2.0;

    add_hemisphere(&mut mesh, r, top_cy, cx, cz, false, ns, hs, region);

    // ── Cylinder lateral (only when cylinder_height > 0) ────────────────────
    if hl > 0.0 {
        add_cylinder_band(&mut mesh, r, cx, bot_cy, top_cy, cz, ns, region);
    }

    add_hemisphere(&mut mesh, r, bot_cy, cx, cz, true, ns, hs, region);

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;
    use std::f64::consts::PI;

    #[test]
    fn capsule_is_watertight() {
        let mesh = Capsule::default().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_watertight, "capsule must be watertight");
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn capsule_volume_positive_and_approximately_correct() {
        let r = 0.5_f64;
        let h = 2.0_f64;
        let mesh = Capsule {
            radius: r,
            cylinder_height: h,
            segments: 64,
            hemisphere_stacks: 16,
            ..Capsule::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.signed_volume > 0.0);
        // V = π r² (h + 4r/3)
        let expected = PI * r * r * (h + 4.0 * r / 3.0);
        let error = (report.signed_volume - expected).abs() / expected;
        assert!(
            error < 0.005,
            "volume error {:.4}% should be < 0.5%",
            error * 100.0
        );
    }

    #[test]
    fn capsule_zero_height_is_sphere() {
        // cylinder_height = 0 → capsule becomes a sphere
        let r = 1.0_f64;
        let mesh = Capsule {
            radius: r,
            cylinder_height: 0.0,
            segments: 64,
            hemisphere_stacks: 16,
            ..Capsule::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_watertight);
        let expected = 4.0 / 3.0 * PI * r * r * r;
        let error = (report.signed_volume - expected).abs() / expected;
        assert!(
            error < 0.005,
            "sphere-degenerate error {:.4}%",
            error * 100.0
        );
    }

    #[test]
    fn capsule_rejects_invalid_params() {
        assert!(Capsule {
            radius: 0.0,
            ..Capsule::default()
        }
        .build()
        .is_err());
        assert!(Capsule {
            radius: -1.0,
            ..Capsule::default()
        }
        .build()
        .is_err());
        assert!(Capsule {
            cylinder_height: -0.1,
            ..Capsule::default()
        }
        .build()
        .is_err());
        assert!(Capsule {
            segments: 2,
            ..Capsule::default()
        }
        .build()
        .is_err());
        assert!(Capsule {
            hemisphere_stacks: 0,
            ..Capsule::default()
        }
        .build()
        .is_err());
    }
}
