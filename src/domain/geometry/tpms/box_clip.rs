//! AABB-clipped TPMS extraction — fills a rectangular volume with a TPMS mesh.
//!
//! This module mirrors [`build_tpms_sphere`](super::build_tpms_sphere) but clips
//! the marching-cubes extraction to an axis-aligned bounding box instead of a
//! sphere.  This is the geometry kernel used when filling a shell cuboid cavity
//! with a TPMS lattice network.
//!
//! ## Theorem — Grid Convergence
//!
//! The same `O(h)` Hausdorff convergence rate (Lorensen & Cline 1987) applies:
//! as `resolution → ∞`, the extracted surface converges to the true `{F = c}`
//! level-set inside the box.

use crate::domain::geometry::primitives::PrimitiveError;
use crate::domain::geometry::tpms::marching_cubes;
use crate::domain::geometry::tpms::Tpms;
use crate::domain::geometry::tpms::Vector3r;
use crate::domain::mesh::IndexedMesh;
use eunomia::FloatElement;

use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Point3r;
use hashbrown::HashMap;
use moirai::ParallelSlice;

#[inline]
fn triangle_edge_index(edge: i8) -> usize {
    usize::from(
        u8::try_from(edge)
            .expect("TRI_TABLE emits non-negative edge indices while the sentinel guard is active"),
    )
}

/// Evaluate the signed-distance field of the clipping box at one world-space sample point.
fn box_sdf(wx: f64, wy: f64, wz: f64, bounds: [f64; 6]) -> f64 {
    let [x0, y0, z0, x1, y1, z1] = bounds;
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    let cz = (z0 + z1) * 0.5;
    let hx = (x1 - x0) * 0.5;
    let hy = (y1 - y0) * 0.5;
    let hz = (z1 - z0) * 0.5;
    let qx = (wx - cx).abs() - hx;
    let qy = (wy - cy).abs() - hy;
    let qz = (wz - cz).abs() - hz;
    qx.max(0.0).hypot(qy.max(0.0)).hypot(qz.max(0.0)) + qx.max(qy).max(qz).min(0.0)
}

/// Pre-sample the clipped TPMS field on the padded marching-cubes lattice.
///
/// # Parallelism
///
/// Each z-slice of the field is independent and computed in parallel via moirai.
/// The outer `iz` loop is the moirai work unit; inner `iy × ix` loops run
/// sequentially within each worker because `gs × gs` elements per slice fit in
/// L1 cache on modern hardware.
fn sample_box_field<S: Tpms + Send + Sync>(
    surface: &S,
    params: &TpmsBoxParams,
    k: f64,
    gs: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    let [x0, y0, z0, ..] = params.bounds;
    let iso = params.iso_value;
    let bounds = params.bounds;

    // Each iz-slice is independent: write to field[iz*gs*gs .. (iz+1)*gs*gs].
    let iz_indices: Vec<usize> = (0..gs).collect();
    let slices: Vec<Box<[f64]>> = iz_indices.par().map_collect(|&iz| {
        let wz = z0 + (f64::from_count(iz) - 1.0) * dz;
        let mut slice = vec![0.0_f64; gs * gs];
        for iy in 0..gs {
            let wy = y0 + (f64::from_count(iy) - 1.0) * dy;
            for ix in 0..gs {
                let wx = x0 + (f64::from_count(ix) - 1.0) * dx;
                let tpms_val = surface.field(wx, wy, wz, k) - iso;
                slice[iy * gs + ix] = tpms_val.max(box_sdf(wx, wy, wz, bounds));
            }
        }
        slice.into_boxed_slice()
    });

    // Concatenate slices in iz order into the full field buffer.
    let mut field = Vec::with_capacity(gs * gs * gs);
    for slice in slices {
        field.extend_from_slice(&slice);
    }
    field
}

/// Choose an inward-pointing box-wall normal when the interpolated vertex lies on the box boundary.
fn box_boundary_normal(
    wx: f64,
    wy: f64,
    wz: f64,
    bounds: [f64; 6],
    fallback_normal: Vector3r,
) -> Vector3r {
    let [x0, y0, z0, x1, y1, z1] = bounds;
    if box_sdf(wx, wy, wz, bounds).abs() >= 1e-5 {
        return fallback_normal;
    }

    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    if (wx - x0).abs() < 1e-5 {
        nx = -1.0;
    } else if (wx - x1).abs() < 1e-5 {
        nx = 1.0;
    }
    if (wy - y0).abs() < 1e-5 {
        ny = -1.0;
    } else if (wy - y1).abs() < 1e-5 {
        ny = 1.0;
    }
    if (wz - z0).abs() < 1e-5 {
        nz = -1.0;
    } else if (wz - z1).abs() < 1e-5 {
        nz = 1.0;
    }
    let boundary_normal = Vector3r::new(nx, ny, nz);
    if boundary_normal.norm_squared() > 1e-6 {
        boundary_normal.normalize()
    } else {
        fallback_normal
    }
}

/// Interpolate one marching-cubes edge crossing and add the resulting vertex to the mesh.
fn interpolate_box_vertex(
    mesh: &mut IndexedMesh,
    a: (usize, usize, usize),
    b: (usize, usize, usize),
    edge_values: (f64, f64),
    spacing: (f64, f64, f64),
    bounds: [f64; 6],
    gradient_at: impl Fn(f64, f64, f64) -> Vector3r,
) -> VertexId {
    let [x0, y0, z0, ..] = bounds;
    let (va, vb) = edge_values;
    let (dx, dy, dz) = spacing;
    let t = if (vb - va).abs() > 1e-15 {
        (-va / (vb - va)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let wx = x0 + (f64::from_count(a.0) * (1.0 - t) + f64::from_count(b.0) * t - 1.0) * dx;
    let wy = y0 + (f64::from_count(a.1) * (1.0 - t) + f64::from_count(b.1) * t - 1.0) * dy;
    let wz = z0 + (f64::from_count(a.2) * (1.0 - t) + f64::from_count(b.2) * t - 1.0) * dz;
    let normal = box_boundary_normal(wx, wy, wz, bounds, gradient_at(wx, wy, wz));
    mesh.add_vertex(Point3r::new(wx, wy, wz), normal)
}

// ── Parameters ────────────────────────────────────────────────────────────────

/// Parameters for AABB-clipped TPMS extraction.
///
/// The TPMS surface is extracted inside the box
/// `[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`.
#[derive(Clone, Debug)]
pub struct TpmsBoxParams {
    /// AABB bounds: `[x_min, y_min, z_min, x_max, y_max, z_max]` in mm.
    pub bounds: [f64; 6],
    /// TPMS unit-cell period (mm).  `k = 2π / period`.
    pub period: f64,
    /// Voxels per axis.  Higher → denser, more accurate.
    pub resolution: usize,
    /// Level-set iso-value.  `0.0` = exact minimal surface mid-sheet.
    pub iso_value: f64,
}

impl TpmsBoxParams {
    /// Validate all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitiveError::InvalidParam`] for degenerate or non-finite
    /// bounds, non-positive period, or resolution < 4.
    pub fn validate(&self) -> Result<(), PrimitiveError> {
        let [x0, y0, z0, x1, y1, z1] = self.bounds;
        if !x0.is_finite()
            || !y0.is_finite()
            || !z0.is_finite()
            || !x1.is_finite()
            || !y1.is_finite()
            || !z1.is_finite()
        {
            return Err(PrimitiveError::InvalidParam(
                "all AABB bounds must be finite".to_string(),
            ));
        }
        if x1 <= x0 || y1 <= y0 || z1 <= z0 {
            return Err(PrimitiveError::InvalidParam(format!(
                "AABB must have positive extents: ({x0},{y0},{z0})→({x1},{y1},{z1})"
            )));
        }
        if self.period <= 0.0 {
            return Err(PrimitiveError::InvalidParam(format!(
                "period must be > 0, got {}",
                self.period
            )));
        }
        if self.resolution < 4 {
            return Err(PrimitiveError::InvalidParam(format!(
                "resolution must be >= 4, got {}",
                self.resolution
            )));
        }
        Ok(())
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Extract a TPMS mesh clipped to an axis-aligned bounding box.
///
/// This is the rectangular counterpart of [`build_tpms_sphere`](super::build_tpms_sphere).
/// The marching-cubes grid spans `[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`
/// with `resolution` voxels per axis.  All extracted triangles lie inside the box
/// (no centroid distance culling).
///
/// # Errors
///
/// Returns [`PrimitiveError::InvalidParam`] on parameter validation failure.
pub fn build_tpms_box<S: Tpms>(
    surface: &S,
    params: &TpmsBoxParams,
) -> Result<IndexedMesh, PrimitiveError> {
    params.validate()?;

    let [x0, y0, z0, ..] = params.bounds;
    let k = std::f64::consts::TAU / params.period;
    let n = params.resolution;

    let resolution = f64::from_count(n);
    let dx = (params.bounds[3] - x0) / resolution;
    let dy = (params.bounds[4] - y0) / resolution;
    let dz = (params.bounds[5] - z0) / resolution;
    // Pad by 1 voxel on each side so the marching cubes bounds enclose the box
    let gs = n + 3;

    // Pre-sample field on padded grid.
    let field = sample_box_field(surface, params, k, gs, dx, dy, dz);
    let idx = |ix: usize, iy: usize, iz: usize| iz * gs * gs + iy * gs + ix;

    let mut mesh = IndexedMesh::new();
    let mut cache: HashMap<(usize, usize, usize, usize), VertexId> =
        HashMap::with_capacity(gs * gs * 3);

    for iz in 0..(gs - 1) {
        for iy in 0..(gs - 1) {
            for ix in 0..(gs - 1) {
                // Corner field values and sign configuration.
                let mut cube_vals = [0.0_f64; 8];
                let mut cube_cfg: usize = 0;
                for (ci, &(cdx, cdy, cdz)) in marching_cubes::CORNERS.iter().enumerate() {
                    let v = field[idx(ix + cdx, iy + cdy, iz + cdz)];
                    cube_vals[ci] = v;
                    if v < 0.0 {
                        cube_cfg |= 1 << ci;
                    }
                }

                let emask = marching_cubes::EDGE_TABLE[cube_cfg];
                if emask == 0 {
                    continue;
                }

                // Resolve or create vertex for each intersected edge.
                let mut edge_vids: [Option<VertexId>; 12] = [None; 12];
                for (ei, &[ca, cb]) in marching_cubes::EDGES.iter().enumerate() {
                    if emask & (1 << ei) == 0 {
                        continue;
                    }
                    let vid = *cache.entry((ix, iy, iz, ei)).or_insert_with(|| {
                        let a = (
                            ix + marching_cubes::CORNERS[ca].0,
                            iy + marching_cubes::CORNERS[ca].1,
                            iz + marching_cubes::CORNERS[ca].2,
                        );
                        let b = (
                            ix + marching_cubes::CORNERS[cb].0,
                            iy + marching_cubes::CORNERS[cb].1,
                            iz + marching_cubes::CORNERS[cb].2,
                        );
                        interpolate_box_vertex(
                            &mut mesh,
                            a,
                            b,
                            (cube_vals[ca], cube_vals[cb]),
                            (dx, dy, dz),
                            params.bounds,
                            |wx, wy, wz| surface.gradient(wx, wy, wz, k),
                        )
                    });
                    edge_vids[ei] = Some(vid);
                }

                // Emit triangles
                let tri_row = &marching_cubes::TRI_TABLE[cube_cfg];
                let mut ti = 0;
                while ti + 2 < 16 && tri_row[ti] >= 0 {
                    let e0 = triangle_edge_index(tri_row[ti]);
                    let e1 = triangle_edge_index(tri_row[ti + 1]);
                    let e2 = triangle_edge_index(tri_row[ti + 2]);
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (edge_vids[e0], edge_vids[e1], edge_vids[e2])
                    {
                        mesh.add_face(v0, v1, v2);
                    }
                    ti += 3;
                }
            }
        }
    }

    Ok(mesh)
}

// ── Graded builder ────────────────────────────────────────────────────────────

/// Validate AABB bounds and resolution (shared by `build_tpms_box` and the
/// graded variant).
fn validate_box_bounds(bounds: &[f64; 6], resolution: usize) -> Result<(), PrimitiveError> {
    let [x0, y0, z0, x1, y1, z1] = *bounds;
    if !x0.is_finite()
        || !y0.is_finite()
        || !z0.is_finite()
        || !x1.is_finite()
        || !y1.is_finite()
        || !z1.is_finite()
    {
        return Err(PrimitiveError::InvalidParam(
            "all AABB bounds must be finite".to_string(),
        ));
    }
    if x1 <= x0 || y1 <= y0 || z1 <= z0 {
        return Err(PrimitiveError::InvalidParam(format!(
            "AABB must have positive extents: ({x0},{y0},{z0})→({x1},{y1},{z1})"
        )));
    }
    if resolution < 4 {
        return Err(PrimitiveError::InvalidParam(format!(
            "resolution must be >= 4, got {resolution}"
        )));
    }
    Ok(())
}

/// Extract a TPMS mesh with **spatially-varying period** inside an AABB.
///
/// This is the adaptive counterpart of [`build_tpms_box`].  Instead of a
/// single global period, the caller provides a closure `period_fn(x, y, z)`
/// that returns the local period (mm) at each world coordinate.
///
/// This enables graded pore structures for size-based cell separation:
/// fine period at the periphery (small pores → block RBCs) and coarse
/// period at the center (large pores → pass WBCs/CTCs).
///
/// # Arguments
///
/// * `surface` — the TPMS implicit surface to extract.
/// * `bounds` — AABB `[x_min, y_min, z_min, x_max, y_max, z_max]`.
/// * `resolution` — voxels per axis (≥ 4).
/// * `iso_value` — level-set threshold to subtract from the field.
/// * `period_fn` — closure `(x, y, z) → period_mm` for spatially-varying
///   period.  Must return a positive, finite value for all inputs.
///
/// # Errors
///
/// Returns [`PrimitiveError::InvalidParam`] on degenerate bounds or low
/// resolution.
pub fn build_tpms_box_graded<S: Tpms>(
    surface: &S,
    bounds: [f64; 6],
    resolution: usize,
    iso_value: f64,
    period_fn: impl Fn(f64, f64, f64) -> f64,
) -> Result<IndexedMesh, PrimitiveError> {
    validate_box_bounds(&bounds, resolution)?;

    let [x0, y0, z0, ..] = bounds;
    let n = resolution;
    let iso = iso_value;

    let resolution = f64::from_count(n);
    let dx = (bounds[3] - x0) / resolution;
    let dy = (bounds[4] - y0) / resolution;
    let dz = (bounds[5] - z0) / resolution;
    let gs = n + 3;

    // Pre-sample field on (n+1)³ grid with spatially-varying k.
    let mut field = vec![0.0_f64; gs * gs * gs];
    let idx = |ix: usize, iy: usize, iz: usize| iz * gs * gs + iy * gs + ix;
    for iz in 0..gs {
        for iy in 0..gs {
            for ix in 0..gs {
                let wx = x0 + (f64::from_count(ix) - 1.0) * dx;
                let wy = y0 + (f64::from_count(iy) - 1.0) * dy;
                let wz = z0 + (f64::from_count(iz) - 1.0) * dz;
                let local_period = period_fn(wx, wy, wz).max(1e-12);
                let local_k = std::f64::consts::TAU / local_period;
                let tpms_val = surface.field(wx, wy, wz, local_k) - iso;
                field[idx(ix, iy, iz)] = tpms_val.max(box_sdf(wx, wy, wz, bounds));
            }
        }
    }

    let mut mesh = IndexedMesh::new();
    let mut cache: HashMap<(usize, usize, usize, usize), VertexId> =
        HashMap::with_capacity(gs * gs * 3);

    for iz in 0..(gs - 1) {
        for iy in 0..(gs - 1) {
            for ix in 0..(gs - 1) {
                let mut cube_vals = [0.0_f64; 8];
                let mut cube_cfg: usize = 0;
                for (ci, &(cdx, cdy, cdz)) in marching_cubes::CORNERS.iter().enumerate() {
                    let v = field[idx(ix + cdx, iy + cdy, iz + cdz)];
                    cube_vals[ci] = v;
                    if v < 0.0 {
                        cube_cfg |= 1 << ci;
                    }
                }

                let emask = marching_cubes::EDGE_TABLE[cube_cfg];
                if emask == 0 {
                    continue;
                }

                let mut edge_vids: [Option<VertexId>; 12] = [None; 12];
                for (ei, &[ca, cb]) in marching_cubes::EDGES.iter().enumerate() {
                    if emask & (1 << ei) == 0 {
                        continue;
                    }
                    let vid = *cache.entry((ix, iy, iz, ei)).or_insert_with(|| {
                        let a = (
                            ix + marching_cubes::CORNERS[ca].0,
                            iy + marching_cubes::CORNERS[ca].1,
                            iz + marching_cubes::CORNERS[ca].2,
                        );
                        let b = (
                            ix + marching_cubes::CORNERS[cb].0,
                            iy + marching_cubes::CORNERS[cb].1,
                            iz + marching_cubes::CORNERS[cb].2,
                        );
                        interpolate_box_vertex(
                            &mut mesh,
                            a,
                            b,
                            (cube_vals[ca], cube_vals[cb]),
                            (dx, dy, dz),
                            bounds,
                            |wx, wy, wz| {
                                let local_period = period_fn(wx, wy, wz).max(1e-12);
                                let local_k = std::f64::consts::TAU / local_period;
                                surface.gradient(wx, wy, wz, local_k)
                            },
                        )
                    });
                    edge_vids[ei] = Some(vid);
                }

                let tri_row = &marching_cubes::TRI_TABLE[cube_cfg];
                let mut ti = 0;
                while ti + 2 < 16 && tri_row[ti] >= 0 {
                    let e0 = triangle_edge_index(tri_row[ti]);
                    let e1 = triangle_edge_index(tri_row[ti + 1]);
                    let e2 = triangle_edge_index(tri_row[ti + 2]);
                    if let (Some(v0), Some(v1), Some(v2)) =
                        (edge_vids[e0], edge_vids[e1], edge_vids[e2])
                    {
                        mesh.add_face(v0, v1, v2);
                    }
                    ti += 3;
                }
            }
        }
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::geometry::tpms::Gyroid;

    #[test]
    fn box_clip_produces_nonempty_mesh() {
        let params = TpmsBoxParams {
            bounds: [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
            period: 2.5,
            resolution: 16,
            iso_value: 0.0,
        };
        let mesh = build_tpms_box(&Gyroid, &params).expect("should succeed");
        assert!(
            mesh.face_count() > 0,
            "gyroid-in-box must produce at least one face"
        );
        assert!(
            mesh.vertex_count() > 0,
            "gyroid-in-box must produce at least one vertex"
        );
    }

    #[test]
    fn box_clip_validates_degenerate_bounds() {
        let params = TpmsBoxParams {
            bounds: [5.0, 0.0, 0.0, 5.0, 10.0, 10.0], // x_max == x_min
            period: 2.5,
            resolution: 16,
            iso_value: 0.0,
        };
        assert!(build_tpms_box(&Gyroid, &params).is_err());
    }

    #[test]
    fn box_clip_validates_low_resolution() {
        let params = TpmsBoxParams {
            bounds: [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            period: 2.5,
            resolution: 2,
            iso_value: 0.0,
        };
        assert!(build_tpms_box(&Gyroid, &params).is_err());
    }

    #[test]
    fn box_clip_all_vertices_within_bounds() {
        let params = TpmsBoxParams {
            bounds: [-3.0, -2.0, -1.0, 4.0, 5.0, 6.0],
            period: 3.0,
            resolution: 16,
            iso_value: 0.0,
        };
        let mesh = build_tpms_box(&Gyroid, &params).expect("should succeed");
        let eps = params.period / f64::from_count(params.resolution); // one voxel tolerance
        for vid in 0..mesh.vertex_count() {
            let p = mesh.vertices.position(VertexId(vid as u32));
            assert!(
                p.x >= -3.0 - eps && p.x <= 4.0 + eps,
                "vertex x={} out of bounds",
                p.x,
            );
            assert!(
                p.y >= -2.0 - eps && p.y <= 5.0 + eps,
                "vertex y={} out of bounds",
                p.y,
            );
            assert!(
                p.z >= -1.0 - eps && p.z <= 6.0 + eps,
                "vertex z={} out of bounds",
                p.z,
            );
        }
    }

    // ── Graded builder tests ──────────────────────────────────────────────

    #[test]
    fn graded_uniform_matches_box_clip() {
        // A graded builder with constant period should produce the same mesh
        // topology as build_tpms_box (same face count ± small tolerance from
        // floating point differences).
        let bounds = [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0];
        let period = 2.5;
        let params = TpmsBoxParams {
            bounds,
            period,
            resolution: 16,
            iso_value: 0.0,
        };
        let uniform = build_tpms_box(&Gyroid, &params).unwrap();
        let graded = build_tpms_box_graded(&Gyroid, bounds, 16, 0.0, |_x, _y, _z| period).unwrap();
        assert_eq!(
            uniform.face_count(),
            graded.face_count(),
            "constant-period graded must produce same face count as uniform"
        );
    }

    #[test]
    fn graded_mesh_nonempty() {
        // A graded mesh with period varying from 1.5 (walls) to 5.0 (center)
        // should produce a non-empty mesh.
        let bounds = [0.0, 0.0, 0.0, 10.0, 10.0, 5.0];
        let mesh = build_tpms_box_graded(&Gyroid, bounds, 20, 0.0, |_x, y, _z| {
            // Y ranges [0, 10]: center at 5.0
            let y_frac = y / 10.0;
            let wall_dist = (2.0 * (y_frac - 0.5)).abs();
            5.0 * (1.0 - wall_dist) + 1.5 * wall_dist
        })
        .unwrap();
        assert!(mesh.face_count() > 0, "graded gyroid must produce faces");
    }

    #[test]
    fn graded_rejects_degenerate_bounds() {
        assert!(build_tpms_box_graded(
            &Gyroid,
            [0.0, 0.0, 0.0, 0.0, 10.0, 10.0],
            16,
            0.0,
            |_, _, _| 3.0,
        )
        .is_err());
    }
}
