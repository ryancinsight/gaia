//! Shared Marching Cubes extraction engine for implicit TPMS surfaces.
//!
//! ## Algorithm
//!
//! Lorensen & Cline (1987) marching cubes on a uniform axis-aligned voxel
//! grid.  For each of the `n³` voxel cubes the sign pattern of the corners is
//! used to look up which of the 12 edges contain a zero crossing; linear
//! interpolation along each edge gives the vertex position; the triangle table
//! maps each of the 256 sign configurations to up to 5 triangles.
//!
//! ## Shared data
//!
//! `EDGE_TABLE` and `TRI_TABLE` are the canonical Lorensen & Cline lookup
//! tables — there is exactly **one copy** in the entire codebase, located here.
//! All TPMS primitive builders delegate to [`extract`].
//!
//! ## Theorem — Correctness
//!
//! A triangle edge intersects the zero level-set iff the two endpoint field
//! values have opposite signs.  Linear interpolation along the edge gives a
//! zero crossing that is first-order accurate; as grid spacing `h → 0` the
//! extracted surface converges to the true level-set at rate `O(h)` in
//! Hausdorff distance (Lorensen & Cline 1987).

use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

mod tables;
pub use tables::{CORNERS, EDGE_TABLE, EDGES, TRI_TABLE};


/// Contiguous cache for the three axis-aligned edge families of a voxel grid.
///
/// Every grid edge has one canonical axis and integer lattice coordinate, so
/// a hash key is unnecessary. The arrays contain one slot per possible grid
/// edge; [`Self::UNMAPPED`] means that the edge has not produced a mesh vertex
/// yet. Valid mesh IDs are strictly below `u32::MAX`, so the packed slots do
/// not need a separate occupancy bitmap.
struct EdgeVertexCache {
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    n: usize,
    grid_size: usize,
}

impl EdgeVertexCache {
    const UNMAPPED: u32 = u32::MAX;

    fn new(n: usize) -> Self {
        let grid_size = n + 1;
        let x_edges = n * grid_size * grid_size;
        let y_edges = grid_size * n * grid_size;
        let z_edges = grid_size * grid_size * n;
        Self {
            x: vec![Self::UNMAPPED; x_edges],
            y: vec![Self::UNMAPPED; y_edges],
            z: vec![Self::UNMAPPED; z_edges],
            n,
            grid_size,
        }
    }

    #[inline]
    fn x_index(n: usize, grid_size: usize, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * grid_size + iy) * n + ix
    }

    #[inline]
    fn y_index(n: usize, grid_size: usize, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * n + iy) * grid_size + ix
    }

    #[inline]
    fn z_index(grid_size: usize, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * grid_size + iy) * grid_size + ix
    }

    /// Return the canonical cache slot for one of the twelve local cube edges.
    ///
    /// The mapping follows [`CORNERS`] and [`EDGES`]. The local edge index is
    /// produced by the fixed `EDGES` table in [`extract`], so the final arm is
    /// a programmer-error guard rather than an input fallback.
    #[inline]
    fn slot(&mut self, ix: usize, iy: usize, iz: usize, edge: usize) -> &mut u32 {
        let n = self.n;
        let grid_size = self.grid_size;
        match edge {
            0 => &mut self.x[Self::x_index(n, grid_size, ix, iy, iz)],
            1 => &mut self.y[Self::y_index(n, grid_size, ix + 1, iy, iz)],
            2 => &mut self.x[Self::x_index(n, grid_size, ix, iy + 1, iz)],
            3 => &mut self.y[Self::y_index(n, grid_size, ix, iy, iz)],
            4 => &mut self.x[Self::x_index(n, grid_size, ix, iy, iz + 1)],
            5 => &mut self.y[Self::y_index(n, grid_size, ix + 1, iy, iz + 1)],
            6 => &mut self.x[Self::x_index(n, grid_size, ix, iy + 1, iz + 1)],
            7 => &mut self.y[Self::y_index(n, grid_size, ix, iy, iz + 1)],
            8 => &mut self.z[Self::z_index(grid_size, ix, iy, iz)],
            9 => &mut self.z[Self::z_index(grid_size, ix + 1, iy, iz)],
            10 => &mut self.z[Self::z_index(grid_size, ix + 1, iy + 1, iz)],
            11 => &mut self.z[Self::z_index(grid_size, ix, iy + 1, iz)],
            _ => panic!("invariant: marching-cubes edge index {edge} outside table"),
        }
    }
}

// ── Extraction engine ─────────────────────────────────────────────────────────

trait SurfaceEvaluator {
    fn field(&self, x: f64, y: f64, z: f64, k: f64) -> f64;

    fn gradient(&self, x: f64, y: f64, z: f64, k: f64) -> Vector3r;
}

struct ClosureEvaluator<F, G> {
    field: F,
    gradient: G,
}

impl<F, G> SurfaceEvaluator for ClosureEvaluator<F, G>
where
    F: Fn(f64, f64, f64, f64) -> f64,
    G: Fn(f64, f64, f64, f64) -> Vector3r,
{
    #[inline]
    fn field(&self, x: f64, y: f64, z: f64, k: f64) -> f64 {
        (self.field)(x, y, z, k)
    }

    #[inline]
    fn gradient(&self, x: f64, y: f64, z: f64, k: f64) -> Vector3r {
        (self.gradient)(x, y, z, k)
    }
}

impl<S: super::Tpms + ?Sized> SurfaceEvaluator for S {
    #[inline]
    fn field(&self, x: f64, y: f64, z: f64, k: f64) -> f64 {
        super::Tpms::field(self, x, y, z, k)
    }

    #[inline]
    fn gradient(&self, x: f64, y: f64, z: f64, k: f64) -> Vector3r {
        super::Tpms::gradient(self, x, y, z, k)
    }
}

/// Marching cubes extraction parameters.
pub struct McParams {
    /// Clip-sphere radius (mm).  Triangles whose centroid exceeds this are discarded.
    pub radius: f64,
    /// Marching grid voxels per axis (the domain is `[-radius, radius]³`).
    pub resolution: usize,
    /// Wave number `k = 2π / period` for the TPMS field.
    pub k: f64,
    /// Level-set threshold subtracted from the field before sign testing.
    pub iso_value: f64,
}

/// Extract a TPMS iso-surface from `field_fn` into `mesh`.
///
/// `field_fn(x, y, z, k) → f64` — the implicit field to march.
/// `gradient_fn(x, y, z, k) → Vector3r` — normalised surface normal at `(x,y,z)`.
pub fn extract(
    mesh: &mut IndexedMesh,
    params: &McParams,
    field_fn: impl Fn(f64, f64, f64, f64) -> f64,
    gradient_fn: impl Fn(f64, f64, f64, f64) -> Vector3r,
) {
    let evaluator = ClosureEvaluator {
        field: field_fn,
        gradient: gradient_fn,
    };
    extract_impl(mesh, params, &evaluator);
}

pub(crate) fn extract_surface<S: super::Tpms>(
    mesh: &mut IndexedMesh,
    params: &McParams,
    surface: &S,
) {
    extract_impl(mesh, params, surface);
}

fn extract_impl<E: SurfaceEvaluator + ?Sized>(
    mesh: &mut IndexedMesh,
    params: &McParams,
    evaluator: &E,
) {
    let n = params.resolution;
    let r = params.radius;
    let k = params.k;
    let iso = params.iso_value;
    let r_sq = r * r;
    let step = 2.0 * r / n as f64;
    let gs = n + 1;

    // Pre-sample field on (n+1)³ grid.
    let mut field = vec![0.0_f64; gs * gs * gs];
    let idx = |ix: usize, iy: usize, iz: usize| iz * gs * gs + iy * gs + ix;
    for iz in 0..=n {
        for iy in 0..=n {
            for ix in 0..=n {
                let wx = -r + ix as f64 * step;
                let wy = -r + iy as f64 * step;
                let wz = -r + iz as f64 * step;
                field[idx(ix, iy, iz)] = evaluator.field(wx, wy, wz, k) - iso;
            }
        }
    }

    let mut cache = EdgeVertexCache::new(n);

    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                // Gather corner field values and compute sign-config integer.
                let mut cube_vals = [0.0_f64; 8];
                let mut cube_cfg: usize = 0;
                for (ci, &(dx, dy, dz)) in CORNERS.iter().enumerate() {
                    let v = field[idx(ix + dx as usize, iy + dy as usize, iz + dz as usize)];
                    cube_vals[ci] = v;
                    if v < 0.0 {
                        cube_cfg |= 1 << ci;
                    }
                }

                let emask = EDGE_TABLE[cube_cfg];
                if emask == 0 {
                    continue;
                }

                // Resolve or create a vertex for each intersected edge.
                let mut edge_vids: [Option<VertexId>; 12] = [None; 12];
                for (ei, &[ca, cb]) in EDGES.iter().enumerate() {
                    if emask & (1 << ei) == 0 {
                        continue;
                    }
                    let slot = cache.slot(ix, iy, iz, ei);
                    let vid = match *slot {
                        EdgeVertexCache::UNMAPPED => {
                            let (ax, ay, az) = (
                                ix + CORNERS[ca].0 as usize,
                                iy + CORNERS[ca].1 as usize,
                                iz + CORNERS[ca].2 as usize,
                            );
                            let (bx, by, bz) = (
                                ix + CORNERS[cb].0 as usize,
                                iy + CORNERS[cb].1 as usize,
                                iz + CORNERS[cb].2 as usize,
                            );
                            let va = cube_vals[ca];
                            let vb = cube_vals[cb];
                            let t = if (vb - va).abs() > 1e-15 {
                                (-va / (vb - va)).clamp(0.0, 1.0)
                            } else {
                                0.5
                            };
                            let wx = -r + (ax as f64 * (1.0 - t) + bx as f64 * t) * step;
                            let wy = -r + (ay as f64 * (1.0 - t) + by as f64 * t) * step;
                            let wz = -r + (az as f64 * (1.0 - t) + bz as f64 * t) * step;
                            let normal = evaluator.gradient(wx, wy, wz, k);
                            let vid = mesh.add_vertex(Point3r::new(wx, wy, wz), normal);
                            debug_assert_ne!(vid.raw(), EdgeVertexCache::UNMAPPED);
                            *slot = vid.raw();
                            vid
                        }
                        raw => VertexId::new(raw),
                    };
                    edge_vids[ei] = Some(vid);
                }

                // Emit triangles, culling those outside the clip sphere.
                let tri_row = &TRI_TABLE[cube_cfg];
                let mut ti = 0;
                while ti + 2 < 16 && tri_row[ti] >= 0 {
                    let e0 = tri_row[ti] as usize;
                    let e1 = tri_row[ti + 1] as usize;
                    let e2 = tri_row[ti + 2] as usize;
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (edge_vids[e0], edge_vids[e1], edge_vids[e2])
                    {
                        let p0 = mesh.vertices.position(v0);
                        let p1 = mesh.vertices.position(v1);
                        let p2 = mesh.vertices.position(v2);
                        let cx = (p0.x + p1.x + p2.x) / 3.0;
                        let cy = (p0.y + p1.y + p2.y) / 3.0;
                        let cz = (p0.z + p1.z + p2.z) / 3.0;
                        if cx * cx + cy * cy + cz * cz <= r_sq {
                            mesh.add_face(v0, v1, v2);
                        }
                    }
                    ti += 3;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EdgeVertexCache;

    #[test]
    fn edge_cache_shares_each_lattice_edge_across_cells() {
        let mut cache = EdgeVertexCache::new(4);

        *cache.slot(1, 1, 1, 0) = 10;
        assert_eq!(*cache.slot(1, 0, 1, 2), 10);

        *cache.slot(1, 1, 1, 1) = 11;
        assert_eq!(*cache.slot(2, 1, 1, 3), 11);

        *cache.slot(1, 1, 1, 8) = 12;
        assert_eq!(*cache.slot(0, 1, 1, 9), 12);
        assert_eq!(*cache.slot(1, 0, 1, 11), 12);
    }
}
