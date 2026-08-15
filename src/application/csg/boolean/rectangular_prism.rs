//! Closed-form Boolean union for axis-aligned rectangular prisms.
//!
//! This fast path owns the exact boundary reconstruction for the one Boolean
//! family whose result is itself a rectangular prism. Keeping it separate from
//! the generalized arrangement orchestration prevents the special-case volume
//! and topology contract from becoming a hidden branch in the indexed API.

use crate::domain::geometry::aabb::Aabb;
use crate::domain::geometry::primitives::{Cube, PrimitiveMesh};
use crate::domain::mesh::IndexedMesh;

/// Reconstruct the exact union when both inputs combine into one rectangular
/// prism.
pub(super) fn rectangular_prism_union(
    mesh_a: &IndexedMesh,
    mesh_b: &IndexedMesh,
) -> Option<IndexedMesh> {
    let bb_a = mesh_a.bounding_box();
    let bb_b = mesh_b.bounding_box();
    if !is_rectangular_prism(mesh_a, &bb_a) || !is_rectangular_prism(mesh_b, &bb_b) {
        return None;
    }

    let mut regions = mesh_a.faces.iter().chain(mesh_b.faces.iter());
    let region = regions.next()?.region;
    if !regions.all(|face| face.region == region) {
        return None;
    }

    let union = bb_a.union(&bb_b);
    let union_volume = bounding_box_volume(&union);
    let overlap = Aabb {
        min: leto::geometry::Point3::new(
            bb_a.min.x.max(bb_b.min.x),
            bb_a.min.y.max(bb_b.min.y),
            bb_a.min.z.max(bb_b.min.z),
        ),
        max: leto::geometry::Point3::new(
            bb_a.max.x.min(bb_b.max.x),
            bb_a.max.y.min(bb_b.max.y),
            bb_a.max.z.min(bb_b.max.z),
        ),
    };
    let overlap_volume = bounding_box_volume(&overlap).max(0.0);
    let set_volume = bounding_box_volume(&bb_a) + bounding_box_volume(&bb_b) - overlap_volume;
    let volume_tolerance = 64.0 * f64::EPSILON * union_volume.max(set_volume).max(1.0);
    if (union_volume - set_volume).abs() > volume_tolerance {
        return None;
    }

    let extent = union.max - union.min;
    let mut result = Cube {
        origin: union.min,
        width: extent.x,
        height: extent.y,
        depth: extent.z,
    }
    .build()
    .expect("invariant: a validated rectangular-prism union has positive finite extents");
    for face in result.faces.iter_mut() {
        face.region = region;
    }
    Some(result)
}

fn is_rectangular_prism(mesh: &IndexedMesh, bounds: &Aabb) -> bool {
    let extent = bounds.max - bounds.min;
    if ![
        bounds.min.x,
        bounds.min.y,
        bounds.min.z,
        bounds.max.x,
        bounds.max.y,
        bounds.max.z,
    ]
    .into_iter()
    .all(f64::is_finite)
        || extent.x <= 0.0
        || extent.y <= 0.0
        || extent.z <= 0.0
    {
        return false;
    }

    let scale = extent.x.max(extent.y).max(extent.z);
    let coordinate_tolerance = 64.0 * f64::EPSILON * scale.max(1.0);
    let on_boundary_plane = |coordinates: [f64; 3], min: f64, max: f64| {
        coordinates
            .iter()
            .all(|coordinate| (*coordinate - coordinates[0]).abs() <= coordinate_tolerance)
            && ((coordinates[0] - min).abs() <= coordinate_tolerance
                || (coordinates[0] - max).abs() <= coordinate_tolerance)
    };

    if !mesh.faces.iter().all(|face| {
        let a = mesh.vertices.position(face.vertices[0]);
        let b = mesh.vertices.position(face.vertices[1]);
        let c = mesh.vertices.position(face.vertices[2]);
        on_boundary_plane([a.x, b.x, c.x], bounds.min.x, bounds.max.x)
            || on_boundary_plane([a.y, b.y, c.y], bounds.min.y, bounds.max.y)
            || on_boundary_plane([a.z, b.z, c.z], bounds.min.z, bounds.max.z)
    }) {
        return false;
    }

    let mut topology = mesh.clone();
    if !topology.is_watertight() {
        return false;
    }

    let origin = bounds.min;
    let signed_volume = mesh.faces.iter().fold(0.0_f64, |volume, face| {
        let a = mesh.vertices.position(face.vertices[0]) - origin;
        let b = mesh.vertices.position(face.vertices[1]) - origin;
        let c = mesh.vertices.position(face.vertices[2]) - origin;
        volume + a.dot(b.cross(c)) / 6.0
    });
    let expected_volume = bounding_box_volume(bounds);
    let volume_tolerance = 64.0 * mesh.faces.len() as f64 * f64::EPSILON * expected_volume.max(1.0);
    (signed_volume.abs() - expected_volume).abs() <= volume_tolerance
}

fn bounding_box_volume(bounds: &Aabb) -> f64 {
    let extent = bounds.max - bounds.min;
    extent.x.max(0.0) * extent.y.max(0.0) * extent.z.max(0.0)
}
