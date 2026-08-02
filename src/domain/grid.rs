//! Structured grid builder.
//!
//! Generates a regular Cartesian grid over the unit cube `[0,1]`³,
//! subdivided into nx×ny×nz hexahedra (each decomposed to 5 tetrahedra).
//!
//! This module is a **volume/FEM tool** — it intentionally uses `Mesh<T>` for
//! hexahedral cell topology and is exempt from the surface-mesh deprecation.

use crate::domain::core::index::VertexId;
use crate::domain::mesh::{IndexedMesh, TetrahedralMeshBuilder};

/// Error type for grid building.
#[derive(Debug)]
pub struct GridError(pub String);

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "grid error: {}", self.0)
    }
}

impl std::error::Error for GridError {}

/// Builds a structured hexahedral grid over the unit cube.
///
/// `nx`, `ny`, `nz` are the number of *cells* (not nodes) along each axis.
pub struct StructuredGridBuilder {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl StructuredGridBuilder {
    /// Create a builder with `nx × ny × nz` cells.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz }
    }

    /// Build the mesh.
    pub fn build(self) -> Result<IndexedMesh<f64>, GridError> {
        build_structured_grid(self.nx, self.ny, self.nz)
    }
}

fn build_structured_grid(nx: usize, ny: usize, nz: usize) -> Result<IndexedMesh<f64>, GridError> {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let nz = nz.max(1);

    let vnx = nx + 1;
    let vny = ny + 1;
    let vnz = nz + 1;

    let mut builder =
        TetrahedralMeshBuilder::<f64>::with_capacity(vnx * vny * vnz, nx * ny * nz * 5);
    let mut v_ids = Vec::with_capacity(vnx * vny * vnz);

    // Create corner vertices on a regular grid.
    for iz in 0..vnz {
        for iy in 0..vny {
            for ix in 0..vnx {
                let x = ix as f64 / nx as f64;
                let y = iy as f64 / ny as f64;
                let z = iz as f64 / nz as f64;
                v_ids.push(builder.vertex_array([x, y, z]));
            }
        }
    }

    let v_idx = |ix: usize, iy: usize, iz: usize| v_ids[iz * vny * vnx + iy * vnx + ix];

    // Create cells: each hex cell is split into 5 tetrahedra.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // 8 corner indices of the hex cell.
                let v: [VertexId; 8] = [
                    v_idx(ix, iy, iz),
                    v_idx(ix + 1, iy, iz),
                    v_idx(ix + 1, iy + 1, iz),
                    v_idx(ix, iy + 1, iz),
                    v_idx(ix, iy, iz + 1),
                    v_idx(ix + 1, iy, iz + 1),
                    v_idx(ix + 1, iy + 1, iz + 1),
                    v_idx(ix, iy + 1, iz + 1),
                ];

                // Alternating 5-tet decomposition to ensure conforming faces.
                if (ix + iy + iz) % 2 == 0 {
                    let tets_a: [[VertexId; 4]; 5] = [
                        [v[0], v[1], v[3], v[4]],
                        [v[1], v[2], v[3], v[6]],
                        [v[4], v[5], v[1], v[6]], // Swapped v6 and v1
                        [v[4], v[7], v[6], v[3]],
                        [v[1], v[3], v[4], v[6]],
                    ];
                    for tet in tets_a {
                        builder
                            .tetrahedron(tet)
                            .map_err(|error| GridError(error.to_string()))?;
                    }
                } else {
                    let tets_b: [[VertexId; 4]; 5] = [
                        [v[1], v[0], v[5], v[2]], // Swapped v2 and v5
                        [v[3], v[0], v[2], v[7]],
                        [v[4], v[0], v[7], v[5]], // Swapped v5 and v7
                        [v[6], v[2], v[5], v[7]],
                        [v[0], v[2], v[7], v[5]], // Swapped v5 and v7
                    ];
                    for tet in tets_b {
                        builder
                            .tetrahedron(tet)
                            .map_err(|error| GridError(error.to_string()))?;
                    }
                }
            }
        }
    }

    let mut mesh = builder.build();

    // Label boundary faces.
    let mut boundary_updates = Vec::with_capacity(mesh.faces.len());
    for f_idx in mesh.boundary_faces() {
        let face = mesh.faces.get(f_idx);
        let [v0, v1, v2] = face.vertices;
        let p0 = mesh.vertices.position(v0);
        let p1 = mesh.vertices.position(v1);
        let p2 = mesh.vertices.position(v2);

        let all_bottom = p0.z < 1e-9 && p1.z < 1e-9 && p2.z < 1e-9;
        let all_top = p0.z > 1.0 - 1e-9 && p1.z > 1.0 - 1e-9 && p2.z > 1.0 - 1e-9;
        let all_front = p0.y < 1e-9 && p1.y < 1e-9 && p2.y < 1e-9;
        let all_back = p0.y > 1.0 - 1e-9 && p1.y > 1.0 - 1e-9 && p2.y > 1.0 - 1e-9;
        let all_left = p0.x < 1e-9 && p1.x < 1e-9 && p2.x < 1e-9;
        let all_right = p0.x > 1.0 - 1e-9 && p1.x > 1.0 - 1e-9 && p2.x > 1.0 - 1e-9;
        if all_bottom {
            boundary_updates.push((f_idx, "inlet"));
        } else if all_top {
            boundary_updates.push((f_idx, "outlet"));
        } else if all_front || all_back || all_left || all_right {
            boundary_updates.push((f_idx, "wall"));
        }
    }
    for (f_idx, label) in boundary_updates {
        mesh.mark_boundary(f_idx, label);
    }

    Ok(mesh)
}
