//! Branching (bifurcation / trifurcation) mesh builder.
//!
//! Builds a structured mesh for a Y-shaped or T-shaped branching passage.
//! Use [`BranchingMeshBuilder::build_surface`] for the modern [`IndexedMesh`]
//! boundary-surface output.

use nalgebra::RealField;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::application::channel::venturi::BuildError;
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a branching (bifurcation) flow passage mesh.
///
/// All length/geometry parameters are in metres.
#[derive(Clone, Debug)]
pub struct BranchingMeshBuilder<T: Copy + RealField> {
    d_parent: T,
    l_parent: T,
    d_daughter: T,
    l_daughter: T,
    branching_angle: T,
    resolution: usize,
    n_daughters: usize,
}

impl<T: Copy + RealField + Float + FromPrimitive> BranchingMeshBuilder<T> {
    /// Create a symmetric bifurcation (1 parent, 2 daughters).
    pub fn bifurcation(
        d_parent: T,
        l_parent: T,
        d_daughter: T,
        l_daughter: T,
        branching_angle: T,
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
    pub fn trifurcation(
        d_parent: T,
        l_parent: T,
        d_daughter: T,
        l_daughter: T,
        branching_angle: T,
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
    /// - `RegionId(0)` — wall (all tube surfaces)
    /// - `RegionId(1)` — inlet cap (parent inlet)
    /// - `RegionId(2+d)` — outlet cap for daughter `d`
    pub fn build_surface(&self) -> Result<IndexedMesh, BuildError> {
        build_branching_surface(self)
    }
}

fn build_branching_surface<T: Copy + RealField + Float + FromPrimitive + ToPrimitive>(
    b: &BranchingMeshBuilder<T>,
) -> Result<IndexedMesh, BuildError> {
    let d_parent = b.d_parent.to_f64().unwrap_or(0.0);
    let l_parent = b.l_parent.to_f64().unwrap_or(0.0);
    let d_daughter = b.d_daughter.to_f64().unwrap_or(0.0);
    let l_daughter = b.l_daughter.to_f64().unwrap_or(0.0);
    let branching_angle = b.branching_angle.to_f64().unwrap_or(0.0);

    let r_parent = d_parent / 2.0_f64;
    let r_daughter = d_daughter / 2.0_f64;
    let n_ax = b.resolution.max(4);
    let n_ang: usize = 32;

    let wall_region = RegionId::from_usize(0);

    // Helper: build a watertight closed tube.
    let build_closed_tube = |origin: (Real, Real, Real),
                             dir: (Real, Real, Real),
                             r: Real,
                             n_steps: usize,
                             is_parent: bool,
                             d_idx: usize|
     -> IndexedMesh {
        let mut mesh = IndexedMesh::new();
        let (ox, oy, oz) = origin;
        let (dx, dy, dz) = dir;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let (udx, udy, udz) = (dx / len, dy / len, dz / len);

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
        let (fx, fy, fz) = (udy * ez - udz * ey, udz * ex - udx * ez, udx * ey - udy * ex);

        let mut rings = Vec::with_capacity(n_steps);
        for i in 0..n_steps {
            let t = i as Real / (n_steps - 1) as Real;
            let cx = ox + dx * t;
            let cy = oy + dy * t;
            let cz = oz + dz * t;
            let mut ring = Vec::with_capacity(n_ang);
            for ia in 0..n_ang {
                let theta = std::f64::consts::TAU * ia as Real / n_ang as Real;
                let (sin_t, cos_t) = theta.sin_cos();
                let nx_v = cos_t * ex + sin_t * fx;
                let ny_v = cos_t * ey + sin_t * fy;
                let nz_v = cos_t * ez + sin_t * fz;
                let vid = mesh.add_vertex(
                    Point3r::new(cx + r * nx_v, cy + r * ny_v, cz + r * nz_v),
                    Vector3r::new(nx_v, ny_v, nz_v),
                );
                ring.push(vid);
            }
            rings.push(ring);
        }

        // Walls
        for iz in 0..(n_steps - 1) {
            for ia in 0..n_ang {
                let ia1 = (ia + 1) % n_ang;
                let v00 = rings[iz][ia];
                let v01 = rings[iz][ia1];
                let v10 = rings[iz + 1][ia];
                let v11 = rings[iz + 1][ia1];
                mesh.add_face_with_region(v00, v10, v01, wall_region);
                mesh.add_face_with_region(v01, v10, v11, wall_region);
            }
        }

        // Inlet cap (starts at t=0, normal = -dir)
        let ic = mesh.add_vertex(Point3r::new(ox, oy, oz), Vector3r::new(-udx, -udy, -udz));
        let inlet_region = RegionId::from_usize(1);
        for ia in 0..n_ang {
            let ia1 = (ia + 1) % n_ang;
            let fid = mesh.add_face_with_region(ic, rings[0][ia1], rings[0][ia], inlet_region);
            if is_parent {
                mesh.mark_boundary(fid, "inlet");
            }
        }

        // Outlet cap (ends at t=1, normal = dir)
        let oc = mesh.add_vertex(Point3r::new(ox + dx, oy + dy, oz + dz), Vector3r::new(udx, udy, udz));
        let outlet_region = RegionId::from_usize(2 + d_idx);
        let last = n_steps - 1;
        for ia in 0..n_ang {
            let ia1 = (ia + 1) % n_ang;
            let fid = mesh.add_face_with_region(oc, rings[last][ia], rings[last][ia1], outlet_region);
            if !is_parent {
                mesh.mark_boundary(fid, format!("outlet_{d_idx}"));
            }
        }

        mesh
    };

    let mut meshes = Vec::new();

    // 1. Parent tube
    // Extend slightly past l_parent to ensure solid overlap for CSG union
    let parent_overlap = r_parent * 1.5;
    let mesh_parent = build_closed_tube(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, l_parent + parent_overlap),
        r_parent,
        n_ax,
        true,
        0,
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

        // Start daughter tube deep inside the parent to guarantee volume overlap
        let overlap_dist = r_parent * 1.5;
        let start_x = -overlap_dist * sin_a;
        let start_y = 0.0;
        let start_z = l_parent - overlap_dist * cos_a;

        // End position
        let run_dist = l_daughter + overlap_dist;
        let dx = run_dist * sin_a;
        let dy = 0.0;
        let dz = run_dist * cos_a;

        let mesh_d = build_closed_tube(
            (start_x, start_y, start_z),
            (dx, dy, dz),
            r_daughter,
            n_ax,
            false,
            d,
        );
        meshes.push(mesh_d);
    }

    // 3. Exact evaluate Boolean Union across all branch bounds
    use crate::application::csg::boolean::{BooleanOp, csg_boolean_nary};
    let union_mesh = csg_boolean_nary(BooleanOp::Union, &meshes)
        .map_err(|e| BuildError(format!("CSG Boolean failed on branch connection: {e:?}")))?;

    Ok(union_mesh)
}
