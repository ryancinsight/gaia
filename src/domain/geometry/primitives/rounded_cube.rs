//! Rounded cube (filleted box) primitive.

use eunomia::FloatElement;
use std::f64::consts::{PI, TAU};

use super::{PrimitiveError, PrimitiveMesh};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a box with cylindrical edge fillets and spherical corner octants.
///
/// The box occupies the region
/// `[origin.x, origin.x + width] × [origin.y, origin.y + height] × [origin.z, origin.z + depth]`.
/// Every edge is replaced by a quarter-cylinder of radius `corner_radius`
/// and every corner by a sphere octant of the same radius.
///
/// ## Geometry decomposition
///
/// | Region type | Count | Faces each |
/// |-------------|-------|-----------|
/// | Flat face panels (rectangular) | 6 | `2 × (u × v)` |
/// | Quarter-cylinder edge strips | 12 | `2 × cs × len_segments` |
/// | Sphere octant corners | 8 | `cs × cs × 2 + cs` apex triangles |
///
/// where `cs = corner_segments` and all shared boundary rings are welded by
/// `VertexPool` spatial-hash deduplication.
///
/// ## Validation
///
/// - `corner_radius ≤ min(width, height, depth) / 2`
/// - `corner_segments ≥ 1`
/// - All dimensions > 0
///
/// ## Output
///
/// - All faces tagged `RegionId(1)`
/// - `signed_volume ≈ w·h·d − (4−π)·r²·(w+h+d) + V_sphere_correction`
///   (the exact value approaches `w·h·d` as `r → 0`)
#[derive(Clone, Debug)]
pub struct RoundedCube {
    /// Corner of the bounding box (minimum x, y, z).
    pub origin: Point3r,
    /// Extent along +X (mm).
    pub width: f64,
    /// Extent along +Y (mm).
    pub height: f64,
    /// Extent along +Z (mm).
    pub depth: f64,
    /// Fillet radius (mm). Must be ≤ `min(w, h, d) / 2`.
    pub corner_radius: f64,
    /// Angular segments per quarter-turn of the fillets (≥ 1).
    pub corner_segments: usize,
}

impl Default for RoundedCube {
    fn default() -> Self {
        Self {
            origin: Point3r::origin(),
            width: 2.0,
            height: 2.0,
            depth: 2.0,
            corner_radius: 0.2,
            corner_segments: 4,
        }
    }
}

impl PrimitiveMesh for RoundedCube {
    fn build(&self) -> Result<IndexedMesh, PrimitiveError> {
        build(self)
    }
}

/// Inner-box bounds shared by the flat panels, edge fillets, and corner octants.
type InnerBoxBounds = (f64, f64, f64, f64, f64, f64);

/// Add one rectangular face panel with a uniform normal and region tag.
fn add_region_quad(
    mesh: &mut IndexedMesh,
    p00: Point3r,
    p10: Point3r,
    p11: Point3r,
    p01: Point3r,
    n: Vector3r,
    region: RegionId,
) {
    let v00 = mesh.add_vertex(p00, n);
    let v10 = mesh.add_vertex(p10, n);
    let v11 = mesh.add_vertex(p11, n);
    let v01 = mesh.add_vertex(p01, n);
    mesh.add_face_with_region(v00, v10, v11, region);
    mesh.add_face_with_region(v00, v11, v01, region);
}

/// Add the six rectangular face panels spanning the flat portions of the box.
fn add_flat_face_panels(mesh: &mut IndexedMesh, bounds: InnerBoxBounds, r: f64, region: RegionId) {
    let (x0, x1, y0, y1, z0, z1) = bounds;

    add_region_quad(
        mesh,
        Point3r::new(x0 - r, y0, z0),
        Point3r::new(x0 - r, y1, z0),
        Point3r::new(x0 - r, y1, z1),
        Point3r::new(x0 - r, y0, z1),
        -Vector3r::x(),
        region,
    );
    add_region_quad(
        mesh,
        Point3r::new(x1 + r, y0, z1),
        Point3r::new(x1 + r, y1, z1),
        Point3r::new(x1 + r, y1, z0),
        Point3r::new(x1 + r, y0, z0),
        Vector3r::x(),
        region,
    );
    add_region_quad(
        mesh,
        Point3r::new(x0, y0 - r, z1),
        Point3r::new(x1, y0 - r, z1),
        Point3r::new(x1, y0 - r, z0),
        Point3r::new(x0, y0 - r, z0),
        -Vector3r::y(),
        region,
    );
    add_region_quad(
        mesh,
        Point3r::new(x0, y1 + r, z0),
        Point3r::new(x1, y1 + r, z0),
        Point3r::new(x1, y1 + r, z1),
        Point3r::new(x0, y1 + r, z1),
        Vector3r::y(),
        region,
    );
    add_region_quad(
        mesh,
        Point3r::new(x0, y0, z0 - r),
        Point3r::new(x1, y0, z0 - r),
        Point3r::new(x1, y1, z0 - r),
        Point3r::new(x0, y1, z0 - r),
        -Vector3r::z(),
        region,
    );
    add_region_quad(
        mesh,
        Point3r::new(x1, y0, z1 + r),
        Point3r::new(x0, y0, z1 + r),
        Point3r::new(x0, y1, z1 + r),
        Point3r::new(x1, y1, z1 + r),
        Vector3r::z(),
        region,
    );
}

/// Add one quarter-cylinder strip swept between two endpoints along a box edge.
fn add_edge_strip(
    mesh: &mut IndexedMesh,
    cs: usize,
    a_start: f64,
    a_end: f64,
    mut sample: impl FnMut(f64) -> (Point3r, Point3r, Vector3r),
    reverse_winding: bool,
    region: RegionId,
) {
    for k in 0..cs {
        let a0 = a_start + f64::from_count(k) / f64::from_count(cs) * (a_end - a_start);
        let a1 = a_start + f64::from_count(k + 1) / f64::from_count(cs) * (a_end - a_start);
        let (pb0, pt0, n0) = sample(a0);
        let (pb1, pt1, n1) = sample(a1);
        let vb0 = mesh.add_vertex(pb0, n0);
        let vt0 = mesh.add_vertex(pt0, n0);
        let vb1 = mesh.add_vertex(pb1, n1);
        let vt1 = mesh.add_vertex(pt1, n1);

        if reverse_winding {
            mesh.add_face_with_region(vb0, vt1, vt0, region);
            mesh.add_face_with_region(vb0, vb1, vt1, region);
        } else {
            mesh.add_face_with_region(vb0, vt0, vt1, region);
            mesh.add_face_with_region(vb0, vt1, vb1, region);
        }
    }
}

/// Add the twelve quarter-cylinder fillet strips along the box edges.
fn add_edge_cylinder_strips(
    mesh: &mut IndexedMesh,
    bounds: InnerBoxBounds,
    r: f64,
    cs: usize,
    region: RegionId,
) {
    let (x0, x1, y0, y1, z0, z1) = bounds;

    let z_edges = [
        (x0, y0, PI, 3.0 * PI / 2.0),
        (x1, y0, 3.0 * PI / 2.0, TAU),
        (x1, y1, 0.0, PI / 2.0),
        (x0, y1, PI / 2.0, PI),
    ];
    for (cx, cy, a_start, a_end) in z_edges {
        add_edge_strip(
            mesh,
            cs,
            a_start,
            a_end,
            |angle| {
                let normal = Vector3r::new(angle.cos(), angle.sin(), 0.0);
                (
                    Point3r::new(cx + r * angle.cos(), cy + r * angle.sin(), z0),
                    Point3r::new(cx + r * angle.cos(), cy + r * angle.sin(), z1),
                    normal,
                )
            },
            false,
            region,
        );
    }

    let x_edges = [
        (y0, z0, PI, 3.0 * PI / 2.0),
        (y0, z1, 3.0 * PI / 2.0, TAU),
        (y1, z1, 0.0, PI / 2.0),
        (y1, z0, PI / 2.0, PI),
    ];
    for (cy, cz, a_start, a_end) in x_edges {
        add_edge_strip(
            mesh,
            cs,
            a_start,
            a_end,
            |angle| {
                let normal = Vector3r::new(0.0, angle.sin(), angle.cos());
                (
                    Point3r::new(x0, cy + r * angle.sin(), cz + r * angle.cos()),
                    Point3r::new(x1, cy + r * angle.sin(), cz + r * angle.cos()),
                    normal,
                )
            },
            true,
            region,
        );
    }

    let y_edges = [
        (x0, z0, PI, 3.0 * PI / 2.0),
        (x1, z0, 3.0 * PI / 2.0, TAU),
        (x1, z1, 0.0, PI / 2.0),
        (x0, z1, PI / 2.0, PI),
    ];
    for (cx, cz, a_start, a_end) in y_edges {
        add_edge_strip(
            mesh,
            cs,
            a_start,
            a_end,
            |angle| {
                let normal = Vector3r::new(angle.cos(), 0.0, angle.sin());
                (
                    Point3r::new(cx + r * angle.cos(), y0, cz + r * angle.sin()),
                    Point3r::new(cx + r * angle.cos(), y1, cz + r * angle.sin()),
                    normal,
                )
            },
            true,
            region,
        );
    }
}

/// Evaluate one position-normal sample on a spherical corner octant.
fn corner_octant_sample(
    center: Point3r,
    signs: (f64, f64, f64),
    r: f64,
    u: f64,
    v: f64,
) -> (Point3r, Vector3r) {
    let (sx, sy, sz) = signs;
    let nx = sx * u.cos();
    let ny = sy * u.sin() * v.cos();
    let nz = sz * u.sin() * v.sin();
    let normal = Vector3r::new(nx, ny, nz);
    let position = Point3r::new(center.x + r * nx, center.y + r * ny, center.z + r * nz);
    (position, normal)
}

/// Add one sphere-octant fillet patch at a single rounded cube corner.
fn add_corner_octant(
    mesh: &mut IndexedMesh,
    center: Point3r,
    signs: (f64, f64, f64),
    r: f64,
    cs: usize,
    region: RegionId,
) {
    let (sx, sy, sz) = signs;
    let parity_positive = (sx * sy * sz) > 0.0;

    for iu in 0..cs {
        for iv in 0..cs {
            let u0 = f64::from_count(iu) / f64::from_count(cs) * PI / 2.0;
            let u1 = f64::from_count(iu + 1) / f64::from_count(cs) * PI / 2.0;
            let v0 = f64::from_count(iv) / f64::from_count(cs) * PI / 2.0;
            let v1 = f64::from_count(iv + 1) / f64::from_count(cs) * PI / 2.0;

            let (p00, n00) = corner_octant_sample(center, signs, r, u0, v0);
            let (p10, n10) = corner_octant_sample(center, signs, r, u1, v0);
            let (p11, n11) = corner_octant_sample(center, signs, r, u1, v1);
            let (p01, n01) = corner_octant_sample(center, signs, r, u0, v1);

            let v00 = mesh.add_vertex(p00, n00);
            let v10 = mesh.add_vertex(p10, n10);
            let v11 = mesh.add_vertex(p11, n11);
            let v01 = mesh.add_vertex(p01, n01);

            if iu == 0 {
                if parity_positive {
                    mesh.add_face_with_region(v10, v01, v11, region);
                } else {
                    mesh.add_face_with_region(v10, v11, v01, region);
                }
            } else if parity_positive {
                mesh.add_face_with_region(v00, v01, v11, region);
                mesh.add_face_with_region(v00, v11, v10, region);
            } else {
                mesh.add_face_with_region(v00, v10, v11, region);
                mesh.add_face_with_region(v00, v11, v01, region);
            }
        }
    }
}

/// Add the eight spherical corner-octant fillets that connect the edge strips.
fn add_sphere_corner_octants(
    mesh: &mut IndexedMesh,
    bounds: InnerBoxBounds,
    r: f64,
    cs: usize,
    region: RegionId,
) {
    let (x0, x1, y0, y1, z0, z1) = bounds;

    for sx in [-1.0_f64, 1.0] {
        for sy in [-1.0_f64, 1.0] {
            for sz in [-1.0_f64, 1.0] {
                let center = Point3r::new(
                    if sx > 0.0 { x1 } else { x0 },
                    if sy > 0.0 { y1 } else { y0 },
                    if sz > 0.0 { z1 } else { z0 },
                );
                add_corner_octant(mesh, center, (sx, sy, sz), r, cs, region);
            }
        }
    }
}

fn build(rc: &RoundedCube) -> Result<IndexedMesh, PrimitiveError> {
    let (w, h, d) = (rc.width, rc.height, rc.depth);
    let r = rc.corner_radius;
    let cs = rc.corner_segments;

    if w <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "width must be > 0, got {w}"
        )));
    }
    if h <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "height must be > 0, got {h}"
        )));
    }
    if d <= 0.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "depth must be > 0, got {d}"
        )));
    }
    if r <= 0.0 || r > w / 2.0 || r > h / 2.0 || r > d / 2.0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "corner_radius must be in (0, min(w,h,d)/2] = (0, {}], got {r}",
            w.min(h).min(d) / 2.0
        )));
    }
    if cs < 1 {
        return Err(PrimitiveError::InvalidParam(
            "corner_segments must be ≥ 1".into(),
        ));
    }

    let region = RegionId::new(1);
    let mut mesh = IndexedMesh::new();

    // Inner box corners (after subtracting r from all sides).
    let x0 = rc.origin.x + r;
    let x1 = rc.origin.x + w - r;
    let y0 = rc.origin.y + r;
    let y1 = rc.origin.y + h - r;
    let z0 = rc.origin.z + r;
    let z1 = rc.origin.z + d - r;
    let bounds = (x0, x1, y0, y1, z0, z1);
    add_flat_face_panels(&mut mesh, bounds, r, region);
    add_edge_cylinder_strips(&mut mesh, bounds, r, cs, region);
    add_sphere_corner_octants(&mut mesh, bounds, r, cs, region);

    // All sections (flat panels, Z-edge strips, X/Y-edge strips reversed, octant corners)
    // are constructed with inward winding. Flip all faces to obtain outward normals.
    mesh.flip_faces();

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::watertight::check::check_watertight;
    use crate::infrastructure::storage::edge_store::EdgeStore;

    #[test]
    fn rounded_cube_is_watertight() {
        let mesh = RoundedCube::default().build().unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(
            report.is_watertight,
            "rounded_cube must be watertight: {report:?}"
        );
        assert_eq!(report.euler_characteristic, Some(2));
    }

    #[test]
    fn rounded_cube_volume_positive() {
        let mesh = RoundedCube {
            width: 4.0,
            height: 3.0,
            depth: 2.0,
            corner_radius: 0.3,
            corner_segments: 6,
            ..RoundedCube::default()
        }
        .build()
        .unwrap();
        let edges = EdgeStore::from_face_store(&mesh.faces);
        let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
        assert!(report.is_watertight);
        assert!(report.signed_volume > 0.0);
        // Volume must be less than bounding box
        assert!(report.signed_volume < 4.0 * 3.0 * 2.0);
    }

    #[test]
    fn rounded_cube_rejects_invalid_params() {
        assert!(RoundedCube {
            width: 0.0,
            ..RoundedCube::default()
        }
        .build()
        .is_err());
        assert!(RoundedCube {
            corner_radius: 0.0,
            ..RoundedCube::default()
        }
        .build()
        .is_err());
        // corner_radius > min(w,h,d)/2
        assert!(RoundedCube {
            corner_radius: 1.5,
            width: 2.0,
            height: 2.0,
            depth: 2.0,
            ..RoundedCube::default()
        }
        .build()
        .is_err());
        assert!(RoundedCube {
            corner_segments: 0,
            ..RoundedCube::default()
        }
        .build()
        .is_err());
    }
}
