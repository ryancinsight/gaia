//! Venturi tube mesh builder.
//!
//! Builds a structured mesh for a Venturi flow passage.
//! Use [`VenturiMeshBuilder::build_surface`] for the [`IndexedMesh`]
//! boundary-surface output.
//!
//! ## Design Note
//!
//! All arithmetic is `f64` (`Real`). A generic `<T: Scalar>` parameter is a
//! fake generic (`core_invariants` Â§2): the body unconditionally converts Tâ†’f64
//! before computation; parametrising T adds no numerical benefit.

use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Real, Vector3r};
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::PackedRows;
use eunomia::FloatElement;

/// Error type for Venturi mesh building.
#[derive(Debug)]
pub struct BuildError(pub String);

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mesh build error: {}", self.0)
    }
}

impl std::error::Error for BuildError {}

/// Builds a Venturi tube mesh.
///
/// All length parameters are in metres (`f64`).
///
/// ```rust
/// use gaia::VenturiMeshBuilder;
/// let mesh = VenturiMeshBuilder::new(0.010, 0.004, 0.020, 0.040, 0.010, 0.060, 0.020)
///     .build_surface()
///     .unwrap();
/// assert!(!mesh.faces.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct VenturiMeshBuilder {
    /// Inlet diameter (m).
    pub d_inlet: Real,
    /// Throat diameter (m).
    pub d_throat: Real,
    /// Inlet section length (m).
    pub l_inlet: Real,
    /// Convergent section length (m).
    pub l_convergent: Real,
    /// Throat section length (m).
    pub l_throat: Real,
    /// Divergent section length (m).
    pub l_divergent: Real,
    /// Outlet section length (m).
    pub l_outlet: Real,
    resolution_x: usize,
    resolution_y: usize,
    circular: bool,
}

impl VenturiMeshBuilder {
    /// Create a Venturi mesh builder with the given geometry (all in metres).
    #[must_use]
    pub fn new(
        d_inlet: Real,
        d_throat: Real,
        l_inlet: Real,
        l_convergent: Real,
        l_throat: Real,
        l_divergent: Real,
        l_outlet: Real,
    ) -> Self {
        Self {
            d_inlet,
            d_throat,
            l_inlet,
            l_convergent,
            l_throat,
            l_divergent,
            l_outlet,
            resolution_x: 8,
            resolution_y: 4,
            circular: true,
        }
    }

    /// Create a Venturi mesh builder from typed
    /// [`aequitas`](aequitas::systems::si::quantities) `Length` quantities.
    ///
    /// All values are stored in SI metres internally.  Using typed quantities
    /// prevents accidental unit mix-ups between diameters and lengths.
    ///
    /// # Example
    ///
    /// ```rust
    /// use gaia::VenturiMeshBuilder;
    /// use gaia::domain::core::constants::{length_mm, default_channel_diameter};
    ///
    /// let builder = VenturiMeshBuilder::from_quantities(
    ///     length_mm(10.0),  // d_inlet
    ///     length_mm(4.0),   // d_throat
    ///     length_mm(20.0),  // l_inlet
    ///     length_mm(40.0),  // l_convergent
    ///     length_mm(10.0),  // l_throat
    ///     length_mm(60.0),  // l_divergent
    ///     length_mm(20.0),  // l_outlet
    /// );
    /// let mesh = builder.build_surface().unwrap();
    /// assert!(!mesh.faces.is_empty());
    /// ```
    #[must_use]
    pub fn from_quantities(
        d_inlet: aequitas::systems::si::quantities::Length<Real>,
        d_throat: aequitas::systems::si::quantities::Length<Real>,
        l_inlet: aequitas::systems::si::quantities::Length<Real>,
        l_convergent: aequitas::systems::si::quantities::Length<Real>,
        l_throat: aequitas::systems::si::quantities::Length<Real>,
        l_divergent: aequitas::systems::si::quantities::Length<Real>,
        l_outlet: aequitas::systems::si::quantities::Length<Real>,
    ) -> Self {
        Self::new(
            d_inlet.into_base(),
            d_throat.into_base(),
            l_inlet.into_base(),
            l_convergent.into_base(),
            l_throat.into_base(),
            l_divergent.into_base(),
            l_outlet.into_base(),
        )
    }

    /// Set the mesh resolution (axial Ã— radial).
    #[must_use]
    pub fn with_resolution(mut self, x: usize, y: usize) -> Self {
        self.resolution_x = x;
        self.resolution_y = y;
        self
    }

    /// Use a circular cross-section (default) vs. square.
    #[must_use]
    pub fn with_circular(mut self, circular: bool) -> Self {
        self.circular = circular;
        self
    }

    /// Build a watertight surface mesh.
    ///
    /// # Errors
    ///
    /// This constructor is currently infallible and always returns `Ok`.
    pub fn build_surface(&self) -> Result<IndexedMesh, BuildError> {
        Ok(build_venturi_surface(self))
    }
}

fn build_venturi_surface(b: &VenturiMeshBuilder) -> IndexedMesh {
    let (d_in, d_th) = (b.d_inlet, b.d_throat);
    let (inlet_length, convergent_length, throat_length, divergent_length, outlet_length) = (
        b.l_inlet,
        b.l_convergent,
        b.l_throat,
        b.l_divergent,
        b.l_outlet,
    );
    let nx = b.resolution_x.max(2);
    let n_ang: usize = if b.circular {
        b.resolution_y.max(2) * 4
    } else {
        4
    };
    let total_l =
        inlet_length + convergent_length + throat_length + divergent_length + outlet_length;

    let wall_region = RegionId::from_usize(0);
    let inlet_region = RegionId::from_usize(1);
    let outlet_region = RegionId::from_usize(2);

    let radius_at = |z: Real| -> Real {
        let (r_in, r_th) = (d_in / 2.0, d_th / 2.0);
        let (z1, z2, z3, z4) = (
            inlet_length,
            inlet_length + convergent_length,
            inlet_length + convergent_length + throat_length,
            inlet_length + convergent_length + throat_length + divergent_length,
        );
        if z <= z1 {
            r_in
        } else if z <= z2 {
            r_in + (r_th - r_in) * (z - z1) / convergent_length
        } else if z <= z3 {
            r_th
        } else if z <= z4 {
            r_th + (r_in - r_th) * (z - z3) / divergent_length
        } else {
            r_in
        }
    };

    let mut mesh = IndexedMesh::new();
    let mut ring_values: Vec<crate::domain::core::index::VertexId> = Vec::with_capacity(nx * n_ang);
    let mut ring_offsets: Vec<usize> = Vec::with_capacity(nx + 1);
    ring_offsets.push(0);
    for i in 0..nx {
        let z = total_l * Real::from_count(i) / Real::from_count(nx - 1);
        let r = radius_at(z);
        for ia in 0..n_ang {
            let theta = std::f64::consts::TAU * Real::from_count(ia) / Real::from_count(n_ang);
            let (sin_t, cos_t) = theta.sin_cos();
            ring_values.push(mesh.add_vertex(
                Point3r::new(r * cos_t, r * sin_t, z),
                Vector3r::new(cos_t, sin_t, 0.0),
            ));
        }
        ring_offsets.push(ring_values.len());
    }
    let rings = PackedRows::from_parts(ring_offsets, ring_values);

    for iz in 0..(nx - 1) {
        for ia in 0..n_ang {
            let ia1 = (ia + 1) % n_ang;
            let (v00, v01) = (rings[iz][ia], rings[iz][ia1]);
            let (v10, v11) = (rings[iz + 1][ia], rings[iz + 1][ia1]);
            mesh.add_face_with_region(v00, v01, v11, wall_region);
            mesh.add_face_with_region(v00, v11, v10, wall_region);
        }
    }

    let ic = mesh.add_vertex(Point3r::new(0.0, 0.0, 0.0), Vector3r::new(0.0, 0.0, -1.0));
    for ia in 0..n_ang {
        let ia1 = (ia + 1) % n_ang;
        mesh.add_face_with_region(ic, rings[0][ia1], rings[0][ia], inlet_region);
    }

    let oc = mesh.add_vertex(
        Point3r::new(0.0, 0.0, total_l),
        Vector3r::new(0.0, 0.0, 1.0),
    );
    let last = nx - 1;
    for ia in 0..n_ang {
        let ia1 = (ia + 1) % n_ang;
        mesh.add_face_with_region(oc, rings[last][ia], rings[last][ia1], outlet_region);
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::index::{FaceId, VertexId};

    /// Pins the pre-refactor ring layout the same way as the serpentine
    /// builder's characterization test: ring 0 sits at `z = 0`, inside the
    /// inlet section (`0 <= l_inlet`), so its radius is the closed-form
    /// `d_inlet / 2` with no piecewise taper applied. The `PackedRows` storage
    /// swap must not reorder or drop rows, so this must hold unchanged
    /// before and after it.
    #[test]
    fn venturi_ring_topology_is_pinned() {
        let mesh = VenturiMeshBuilder::new(0.010, 0.004, 0.020, 0.040, 0.010, 0.060, 0.020)
            .with_resolution(4, 2)
            .build_surface()
            .expect("should succeed");

        // nx = resolution_x.max(2) = 4.
        // n_ang = if circular { resolution_y.max(2) * 4 } = 8.
        let nx = 4usize;
        let n_ang = 8usize;
        assert_eq!(mesh.vertices.len(), nx * n_ang + 2);
        assert_eq!(
            mesh.faces.as_slice().len(),
            2 * n_ang * (nx - 1) + 2 * n_ang
        );

        let r = 0.010 / 2.0;
        let eps = 1e-12; // several ULPs above f64 sin_cos error at this magnitude
        for ia in 0..n_ang {
            let theta = std::f64::consts::TAU * ia as f64 / n_ang as f64;
            let (sin_t, cos_t) = theta.sin_cos();
            let actual = mesh.vertices.position(VertexId::from_usize(ia));
            assert!(
                (actual.x - r * cos_t).abs() < eps
                    && (actual.y - r * sin_t).abs() < eps
                    && actual.z.abs() < eps,
                "ring-0 vertex {ia}: expected ({}, {}, 0.0), got ({}, {}, {})",
                r * cos_t,
                r * sin_t,
                actual.x,
                actual.y,
                actual.z
            );
        }

        let f0 = mesh.faces.get(FaceId::from_usize(0));
        assert_eq!(
            f0.vertices,
            [
                VertexId::from_usize(0),
                VertexId::from_usize(1),
                VertexId::from_usize(n_ang + 1),
            ]
        );
        let f1 = mesh.faces.get(FaceId::from_usize(1));
        assert_eq!(
            f1.vertices,
            [
                VertexId::from_usize(0),
                VertexId::from_usize(n_ang + 1),
                VertexId::from_usize(n_ang),
            ]
        );
    }

    #[test]
    fn venturi_produces_non_empty_mesh() {
        let mesh = VenturiMeshBuilder::new(0.010, 0.004, 0.020, 0.040, 0.010, 0.060, 0.020)
            .build_surface()
            .expect("should succeed");
        assert!(!mesh.faces.is_empty());
        assert!(!mesh.vertices.is_empty());
    }

    #[test]
    fn venturi_resolution_affects_face_count() {
        let lo = VenturiMeshBuilder::new(0.010, 0.004, 0.020, 0.040, 0.010, 0.060, 0.020)
            .with_resolution(4, 2)
            .build_surface()
            .unwrap();
        let hi = VenturiMeshBuilder::new(0.010, 0.004, 0.020, 0.040, 0.010, 0.060, 0.020)
            .with_resolution(8, 4)
            .build_surface()
            .unwrap();
        assert!(hi.faces.len() > lo.faces.len());
    }
}
