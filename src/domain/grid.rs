//! Structured grid builder.
//!
//! Provides regular Cartesian grids over the unit cube `[0,1]`³ as either
//! validated tetrahedral cells or triangulated-face hexahedral cells.
//!
//! This module is a **volume/FEM tool** — it intentionally uses `Mesh<T>` for
//! hexahedral cell topology and is exempt from the surface-mesh deprecation.

use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::mesh::{IndexedMesh, TetrahedralMeshBuilder};
use crate::domain::topology::{Cell, ElementType};
use eunomia::FloatElement;
use leto::geometry::Point3;

/// Error type for grid building.
#[derive(Debug)]
pub struct GridError(pub String);

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "grid error: {}", self.0)
    }
}

impl std::error::Error for GridError {}

/// Per-cell tetrahedral decomposition for [`StructuredGridBuilder`].
///
/// Both splits are conforming on translation-invariant lattices (shared faces
/// triangulate identically), so the assembled volume is watertight either way.
/// The choice is a consumer contract, not a quality default: kwavers' FEM
/// convergence proof is calibrated on the six-tet body-diagonal split
/// (shape-regularity κ ≤ √3), while the five-tet alternating split uses
/// fewer cells per hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TetDecomposition {
    /// Alternating five-tetrahedra split (conforming faces by parity).
    #[default]
    FiveTetAlternating,
    /// Six tetrahedra sharing the cell body diagonal `v000 → v111`.
    SixTetBodyDiagonal,
}

/// Builds a structured hexahedral grid as validated tetrahedral cells.
///
/// `nx`, `ny`, `nz` are the number of *cells* (not nodes) along each axis.
/// By default the grid covers the unit cube `[0,1]`³ with the alternating
/// five-tet split — the historical behavior. [`Self::with_origin`],
/// [`Self::with_spacing`], and [`Self::with_decomposition`] parameterize the
/// placement and the per-cell split without changing the default output.
pub struct StructuredGridBuilder {
    nx: usize,
    ny: usize,
    nz: usize,
    origin: [f64; 3],
    spacing: Option<[f64; 3]>,
    decomposition: TetDecomposition,
}

impl StructuredGridBuilder {
    /// Create a builder with `nx × ny × nz` cells.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            origin: [0.0; 3],
            spacing: None,
            decomposition: TetDecomposition::default(),
        }
    }

    /// Place the grid minimum corner at `origin` (default `[0, 0, 0]`).
    #[must_use]
    pub fn with_origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = origin;
        self
    }

    /// Use explicit per-axis vertex spacing (default: unit-cube `1/n`).
    ///
    /// [`Self::build`] rejects non-finite or non-positive spacing with a
    /// [`GridError`]; the unit-cube default path is untouched.
    #[must_use]
    pub fn with_spacing(mut self, spacing: [f64; 3]) -> Self {
        self.spacing = Some(spacing);
        self
    }

    /// Select the per-cell tetrahedral split (default alternating five-tet).
    #[must_use]
    pub fn with_decomposition(mut self, decomposition: TetDecomposition) -> Self {
        self.decomposition = decomposition;
        self
    }

    /// Build the mesh.
    ///
    /// Boundary faces carry geometric labels (`inlet` at the minimum-z plane,
    /// `outlet` at maximum-z, `wall` elsewhere) against the actual grid box,
    /// and every cell is orientation-normalized by [`TetrahedralMeshBuilder`].
    ///
    /// # Errors
    ///
    /// Returns [`GridError`] for non-finite origin, non-finite/non-positive
    /// spacing, or if the tetrahedral decomposition produces an invalid cell.
    pub fn build(self) -> Result<IndexedMesh<f64>, GridError> {
        if self.origin.iter().any(|c| !c.is_finite()) {
            return Err(GridError(format!(
                "structured grid origin must be finite, got {:?}",
                self.origin
            )));
        }
        if let Some(spacing) = self.spacing
            && spacing.iter().any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(GridError(format!(
                "structured grid spacing must be finite and positive, got {spacing:?}"
            )));
        }
        build_structured_grid(
            self.nx,
            self.ny,
            self.nz,
            self.origin,
            self.spacing,
            self.decomposition,
        )
    }
}

/// Builds a structured hexahedral grid whose cells are represented by
/// triangulated boundary faces for [`crate::application::hierarchy::hex_to_tet::HexToTetConverter`].
pub struct StructuredHexGridBuilder {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl StructuredHexGridBuilder {
    /// Create a builder with `nx × ny × nz` hexahedral cells.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz }
    }

    /// Build a triangulated-face hexahedral mesh.
    #[must_use]
    pub fn build(self) -> IndexedMesh<f64> {
        build_structured_hex_grid(self.nx, self.ny, self.nz)
    }
}

fn build_structured_hex_grid(nx: usize, ny: usize, nz: usize) -> IndexedMesh<f64> {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let nz = nz.max(1);
    let vertex_capacity = (nx + 1) * (ny + 1) * (nz + 1);
    let cell_capacity = nx * ny * nz;
    let mut mesh = IndexedMesh::with_capacity(vertex_capacity, cell_capacity * 12, cell_capacity);
    let mut vertex_ids = Vec::with_capacity(vertex_capacity);

    for iz in 0..=nz {
        for iy in 0..=ny {
            for ix in 0..=nx {
                vertex_ids.push(mesh.add_vertex_pos(Point3::new(
                    f64::from_count(ix) / f64::from_count(nx),
                    f64::from_count(iy) / f64::from_count(ny),
                    f64::from_count(iz) / f64::from_count(nz),
                )));
            }
        }
    }

    let vertex =
        |ix: usize, iy: usize, iz: usize| vertex_ids[iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let hex = [
                    vertex(ix, iy, iz),
                    vertex(ix + 1, iy, iz),
                    vertex(ix + 1, iy + 1, iz),
                    vertex(ix, iy + 1, iz),
                    vertex(ix, iy, iz + 1),
                    vertex(ix + 1, iy, iz + 1),
                    vertex(ix + 1, iy + 1, iz + 1),
                    vertex(ix, iy + 1, iz + 1),
                ];
                let triangles = [
                    [hex[0], hex[1], hex[2]],
                    [hex[0], hex[2], hex[3]],
                    [hex[4], hex[6], hex[5]],
                    [hex[4], hex[7], hex[6]],
                    [hex[0], hex[4], hex[5]],
                    [hex[0], hex[5], hex[1]],
                    [hex[3], hex[2], hex[6]],
                    [hex[3], hex[6], hex[7]],
                    [hex[0], hex[3], hex[7]],
                    [hex[0], hex[7], hex[4]],
                    [hex[1], hex[5], hex[6]],
                    [hex[1], hex[6], hex[2]],
                ];
                let faces = triangles.map(|[v0, v1, v2]| mesh.add_face(v0, v1, v2));
                mesh.add_cell(Cell {
                    faces: faces.map(FaceId::as_usize).to_vec(),
                    element_type: ElementType::Hexahedron,
                    vertex_ids: Vec::new(),
                });
            }
        }
    }

    mesh
}

/// Shared construction for [`StructuredGridBuilder`]: regular vertex lattice
/// over `[origin, origin + cells * spacing]`, per-cell tetrahedral split per
/// `decomposition`, geometric boundary labels against the actual grid box.
///
/// `spacing = None` reproduces the historical unit-cube lattice bit-for-bit
/// (division by the cell count); `Some` places vertices at
/// `origin + index * spacing`.
fn build_structured_grid(
    nx: usize,
    ny: usize,
    nz: usize,
    origin: [f64; 3],
    spacing: Option<[f64; 3]>,
    decomposition: TetDecomposition,
) -> Result<IndexedMesh<f64>, GridError> {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let nz = nz.max(1);

    let vnx = nx + 1;
    let vny = ny + 1;
    let vnz = nz + 1;

    let tets_per_cell = match decomposition {
        TetDecomposition::FiveTetAlternating => 5,
        TetDecomposition::SixTetBodyDiagonal => 6,
    };
    let mut builder =
        TetrahedralMeshBuilder::<f64>::with_capacity(vnx * vny * vnz, nx * ny * nz * tets_per_cell);
    let mut v_ids = Vec::with_capacity(vnx * vny * vnz);

    // Axis coordinate: historical division path when spacing is defaulted so
    // existing outputs are unchanged; affine placement otherwise.
    let axis_coord = |origin: f64, index: usize, cells: usize, spacing: Option<f64>| match spacing {
        None => f64::from_count(index) / f64::from_count(cells),
        Some(step) => f64::from_count(index).mul_add(step, origin),
    };

    // Create corner vertices on a regular grid.
    for iz in 0..vnz {
        for iy in 0..vny {
            for ix in 0..vnx {
                let x = axis_coord(origin[0], ix, nx, spacing.map(|s| s[0]));
                let y = axis_coord(origin[1], iy, ny, spacing.map(|s| s[1]));
                let z = axis_coord(origin[2], iz, nz, spacing.map(|s| s[2]));
                v_ids.push(builder.vertex_array([x, y, z]));
            }
        }
    }

    let vertex_id_at = |ix: usize, iy: usize, iz: usize| v_ids[iz * vny * vnx + iy * vnx + ix];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // 8 corner indices of the hex cell.
                let v: [VertexId; 8] = [
                    vertex_id_at(ix, iy, iz),
                    vertex_id_at(ix + 1, iy, iz),
                    vertex_id_at(ix + 1, iy + 1, iz),
                    vertex_id_at(ix, iy + 1, iz),
                    vertex_id_at(ix, iy, iz + 1),
                    vertex_id_at(ix + 1, iy, iz + 1),
                    vertex_id_at(ix + 1, iy + 1, iz + 1),
                    vertex_id_at(ix, iy + 1, iz + 1),
                ];

                match decomposition {
                    TetDecomposition::FiveTetAlternating => {
                        // Alternating 5-tet decomposition to ensure conforming faces.
                        let tets: [[VertexId; 4]; 5] = if (ix + iy + iz) % 2 == 0 {
                            [
                                [v[0], v[1], v[3], v[4]],
                                [v[1], v[2], v[3], v[6]],
                                [v[4], v[5], v[1], v[6]], // Swapped v6 and v1
                                [v[4], v[7], v[6], v[3]],
                                [v[1], v[3], v[4], v[6]],
                            ]
                        } else {
                            [
                                [v[1], v[0], v[5], v[2]], // Swapped v2 and v5
                                [v[3], v[0], v[2], v[7]],
                                [v[4], v[0], v[7], v[5]], // Swapped v5 and v7
                                [v[6], v[2], v[5], v[7]],
                                [v[0], v[2], v[7], v[5]], // Swapped v5 and v7
                            ]
                        };
                        for tet in tets {
                            builder
                                .tetrahedron(tet)
                                .map_err(|error| GridError(error.to_string()))?;
                        }
                    }
                    TetDecomposition::SixTetBodyDiagonal => {
                        // Six tetrahedra sharing the body diagonal v000 → v111.
                        // The pattern is translation-invariant, so shared faces
                        // triangulate identically and the volume stays conforming.
                        // Corners: c000=v[0], c100=v[1], c110=v[2], c010=v[3],
                        // c001=v[4], c101=v[5], c111=v[6], c011=v[7].
                        let tets: [[VertexId; 4]; 6] = [
                            [v[0], v[1], v[2], v[6]],
                            [v[0], v[1], v[5], v[6]],
                            [v[0], v[3], v[2], v[6]],
                            [v[0], v[3], v[7], v[6]],
                            [v[0], v[4], v[5], v[6]],
                            [v[0], v[4], v[7], v[6]],
                        ];
                        for tet in tets {
                            builder
                                .tetrahedron(tet)
                                .map_err(|error| GridError(error.to_string()))?;
                        }
                    }
                }
            }
        }
    }

    let mut mesh = builder.build();
    label_grid_boundaries(&mut mesh, origin, [nx, ny, nz], spacing);

    Ok(mesh)
}

/// Label boundary faces against the actual grid box.
///
/// `inlet` at the minimum-z plane, `outlet` at maximum-z, `wall` elsewhere.
/// Tolerance scales with the largest box extent (unit cube: `1e-9`, matching
/// the historical literal comparisons).
fn label_grid_boundaries(
    mesh: &mut IndexedMesh<f64>,
    origin: [f64; 3],
    cells: [usize; 3],
    spacing: Option<[f64; 3]>,
) {
    let extent = |axis: usize| match spacing {
        None => 1.0,
        Some(step) => f64::from_count(cells[axis]) * step[axis],
    };
    let max_extent = extent(0).max(extent(1)).max(extent(2));
    let eps = 1e-9 * max_extent;
    let lo = origin;
    let hi = [
        origin[0] + extent(0),
        origin[1] + extent(1),
        origin[2] + extent(2),
    ];
    let on_lo = |p: f64, axis: usize| (p - lo[axis]).abs() <= eps;
    let on_hi = |p: f64, axis: usize| (p - hi[axis]).abs() <= eps;

    let mut boundary_updates = Vec::with_capacity(mesh.faces.len());
    for f_idx in mesh.boundary_faces() {
        let face = mesh.faces.get(f_idx);
        let [v0, v1, v2] = face.vertices;
        let p0 = mesh.vertices.position(v0);
        let p1 = mesh.vertices.position(v1);
        let p2 = mesh.vertices.position(v2);

        let all_bottom = on_lo(p0.z, 2) && on_lo(p1.z, 2) && on_lo(p2.z, 2);
        let all_top = on_hi(p0.z, 2) && on_hi(p1.z, 2) && on_hi(p2.z, 2);
        let all_front = on_lo(p0.y, 1) && on_lo(p1.y, 1) && on_lo(p2.y, 1);
        let all_back = on_hi(p0.y, 1) && on_hi(p1.y, 1) && on_hi(p2.y, 1);
        let all_left = on_lo(p0.x, 0) && on_lo(p1.x, 0) && on_lo(p2.x, 0);
        let all_right = on_hi(p0.x, 0) && on_hi(p1.x, 0) && on_hi(p2.x, 0);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cell-sum volume: `signed_volume()` sums boundary faces, which is not
    /// the volume of a tetrahedral partition — sum the canonical per-cell
    /// measurements instead (all cells must measure finite and positive).
    fn total_cell_volume(mesh: &IndexedMesh<f64>) -> f64 {
        use crate::application::quality::tetrahedron::cell_tetrahedron_quality;
        let mut total = 0.0;
        for cell in mesh.cells() {
            let quality =
                cell_tetrahedron_quality(mesh, cell).expect("structured-grid cell measures finite");
            assert!(quality.volume > 0.0);
            total += quality.volume;
        }
        total
    }

    #[test]
    fn default_builder_preserves_unit_cube_five_tet_contract() {
        let mesh = StructuredGridBuilder::new(2, 2, 2).build().unwrap();
        assert_eq!(mesh.cell_count(), 2 * 2 * 2 * 5);
        assert_eq!(mesh.vertex_count(), 27);
        assert!((total_cell_volume(&mesh) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn six_tet_body_diagonal_covers_arbitrary_box() {
        let mesh = StructuredGridBuilder::new(2, 1, 1)
            .with_origin([1.0, 2.0, 3.0])
            .with_spacing([0.5, 2.0, 3.0])
            .with_decomposition(TetDecomposition::SixTetBodyDiagonal)
            .build()
            .unwrap();
        // 2 cells × 6 tets; 3 × 2 × 2 vertices; volume 2·(0.5·2·3) = 6.
        assert_eq!(mesh.cell_count(), 12);
        assert_eq!(mesh.vertex_count(), 12);
        assert!((total_cell_volume(&mesh) - 6.0).abs() < 1e-9);
        // Every cell is a valid tetrahedron over four distinct vertices.
        for cell in mesh.cells() {
            assert_eq!(cell.element_type, ElementType::Tetrahedron);
            assert_eq!(cell.vertex_ids.len(), 4);
        }
        // Geometric boundary labels against the actual box.
        let mut labels: Vec<&str> = mesh
            .boundary_labels
            .values()
            .map(std::borrow::Cow::as_ref)
            .collect();
        labels.sort_unstable();
        labels.dedup();
        assert!(labels.contains(&"inlet"));
        assert!(labels.contains(&"outlet"));
        assert!(labels.contains(&"wall"));
    }

    #[test]
    fn rejects_non_positive_spacing_and_non_finite_origin() {
        // `unwrap_err` would need `IndexedMesh: Debug`; assert on `err()`.
        let bad_spacing = StructuredGridBuilder::new(1, 1, 1)
            .with_spacing([1.0, 0.0, 1.0])
            .build()
            .err()
            .expect("zero spacing is rejected");
        assert!(bad_spacing.to_string().contains("spacing"));
        let bad_origin = StructuredGridBuilder::new(1, 1, 1)
            .with_origin([f64::NAN, 0.0, 0.0])
            .build()
            .err()
            .expect("NaN origin is rejected");
        assert!(bad_origin.to_string().contains("origin"));
    }
}
