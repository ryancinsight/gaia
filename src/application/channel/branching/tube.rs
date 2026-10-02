//! Closed-tube construction for branching channel operands.
//!
//! This leaf owns the ring frame, cap regions, and snap-cell policy shared by
//! the parent and daughter tube operands. The branching module owns only the
//! topology and Boolean composition of those operands.

use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::vertex_pool::DEFAULT_MESH_CELL_SIZE;
use eunomia::FloatElement;

pub(super) const ANGULAR_SEGMENTS: usize = 32;

/// Connect two successive tube rings with outward-oriented wall quads.
fn add_tube_wall_band(
    mesh: &mut IndexedMesh,
    previous_ring: &[crate::domain::core::index::VertexId],
    next_ring: &[crate::domain::core::index::VertexId],
    region: RegionId,
) {
    for ia in 0..ANGULAR_SEGMENTS {
        let ia1 = (ia + 1) % ANGULAR_SEGMENTS;
        let v00 = previous_ring[ia];
        let v01 = previous_ring[ia1];
        let v10 = next_ring[ia];
        let v11 = next_ring[ia1];
        mesh.add_face_with_region(v00, v01, v10, region);
        mesh.add_face_with_region(v01, v11, v10, region);
    }
}

/// Build one closed, region-labelled tube operand for a branch composition.
#[expect(
    clippy::too_many_arguments,
    reason = "tube geometry requires origin, direction, radius, step count, and allocation hints — no natural grouping without obscuring the geometric meaning"
)]
pub(super) fn build_closed_tube(
    origin: (Real, Real, Real),
    dir: (Real, Real, Real),
    radius: Real,
    n_steps: usize,
    is_parent: bool,
    daughter_index: usize,
    vertex_capacity: usize,
    face_capacity: usize,
) -> IndexedMesh {
    // Keep the default snap cell for ordinary channels. When a
    // circumferential edge is smaller than that cell, shrink only this
    // operand's cell so adjacent ring vertices remain distinct.
    let angular_edge =
        2.0 * radius * (std::f64::consts::PI / Real::from_count(ANGULAR_SEGMENTS)).sin();
    let cell_size = if angular_edge < DEFAULT_MESH_CELL_SIZE {
        angular_edge * 0.25
    } else {
        DEFAULT_MESH_CELL_SIZE
    };
    let mut mesh =
        IndexedMesh::with_capacity_and_cell_size(vertex_capacity, face_capacity, 0, cell_size);
    let (ox, oy, oz) = origin;
    let (dx, dy, dz) = dir;
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let (udx, udy, udz) = (dx / len, dy / len, dz / len);

    // Compute a stable radial basis via Gram-Schmidt against a reference axis.
    let (ex, ey, ez) = if udz.abs() < 0.9 {
        let (lx, ly, lz) = (0.0, 0.0, 1.0);
        let dot = udx * lx + udy * ly + udz * lz;
        let (sx, sy, sz) = (lx - dot * udx, ly - dot * udy, lz - dot * udz);
        let slen = (sx * sx + sy * sy + sz * sz).sqrt();
        (sx / slen, sy / slen, sz / slen)
    } else {
        let (lx, ly, lz) = (1.0, 0.0, 0.0);
        let dot = udx * lx + udy * ly + udz * lz;
        let (sx, sy, sz) = (lx - dot * udx, ly - dot * udy, lz - dot * udz);
        let slen = (sx * sx + sy * sy + sz * sz).sqrt();
        (sx / slen, sy / slen, sz / slen)
    };
    let (fx, fy, fz) = (
        udy * ez - udz * ey,
        udz * ex - udx * ez,
        udx * ey - udy * ex,
    );

    let mut first_ring = Vec::with_capacity(ANGULAR_SEGMENTS);
    let mut previous_ring = Vec::with_capacity(ANGULAR_SEGMENTS);
    let mut ring = Vec::with_capacity(ANGULAR_SEGMENTS);
    let wall_region = RegionId::from_usize(0);
    for i in 0..n_steps {
        let t = Real::from_count(i) / Real::from_count(n_steps - 1);
        let cx = ox + dx * t;
        let cy = oy + dy * t;
        let cz = oz + dz * t;
        ring.clear();
        for ia in 0..ANGULAR_SEGMENTS {
            let theta =
                std::f64::consts::TAU * Real::from_count(ia) / Real::from_count(ANGULAR_SEGMENTS);
            let (sin_t, cos_t) = theta.sin_cos();
            let normal_x = cos_t * ex + sin_t * fx;
            let normal_y = cos_t * ey + sin_t * fy;
            let normal_z = cos_t * ez + sin_t * fz;
            let vid = mesh.add_vertex(
                Point3r::new(
                    cx + radius * normal_x,
                    cy + radius * normal_y,
                    cz + radius * normal_z,
                ),
                Vector3r::new(normal_x, normal_y, normal_z),
            );
            ring.push(vid);
        }
        if i == 0 {
            first_ring.clone_from(&ring);
        } else {
            add_tube_wall_band(&mut mesh, &previous_ring, &ring, wall_region);
        }
        std::mem::swap(&mut previous_ring, &mut ring);
    }

    // Inlet cap (starts at t=0, normal = -dir).
    let inlet_center = mesh.add_vertex(Point3r::new(ox, oy, oz), Vector3r::new(-udx, -udy, -udz));
    let inlet_region = RegionId::from_usize(1);
    for ia in 0..ANGULAR_SEGMENTS {
        let ia1 = (ia + 1) % ANGULAR_SEGMENTS;
        let face =
            mesh.add_face_with_region(inlet_center, first_ring[ia1], first_ring[ia], inlet_region);
        if is_parent {
            mesh.mark_boundary(face, "inlet");
        }
    }

    // Outlet cap (ends at t=1, normal = dir).
    let outlet_center = mesh.add_vertex(
        Point3r::new(ox + dx, oy + dy, oz + dz),
        Vector3r::new(udx, udy, udz),
    );
    let outlet_region = RegionId::from_usize(2 + daughter_index);
    for ia in 0..ANGULAR_SEGMENTS {
        let ia1 = (ia + 1) % ANGULAR_SEGMENTS;
        let face = mesh.add_face_with_region(
            outlet_center,
            previous_ring[ia],
            previous_ring[ia1],
            outlet_region,
        );
        if !is_parent {
            mesh.mark_boundary(face, format!("outlet_{daughter_index}"));
        }
    }

    mesh
}
