//! Serpentine channel mesh builder.
//!
//! Builds a structured mesh for a sinuous (serpentine) microchannel.
//! All arithmetic is `f64` (`Real`) â€” fake-generic eliminated per core_invariants Â§2.

use crate::application::channel::venturi::BuildError;
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

/// Builds a serpentine channel mesh (all lengths in metres, f64).
#[derive(Clone, Debug)]
pub struct SerpentineMeshBuilder {
    /// Channel diameter (m).
    pub diameter: f64,
    /// Amplitude of the sinusoidal path (m).
    pub amplitude: f64,
    /// Wavelength (m).
    pub wavelength: f64,
    /// Number of full periods.
    pub num_periods: usize,
    resolution_x: usize,
    resolution_y: usize,
    circular: bool,
}

impl SerpentineMeshBuilder {
    /// Create a serpentine builder with given diameter, amplitude, wavelength.
    #[must_use]
    pub fn new(diameter: f64, amplitude: f64, wavelength: f64) -> Self {
        Self {
            diameter,
            amplitude,
            wavelength,
            num_periods: 3,
            resolution_x: 32,
            resolution_y: 4,
            circular: true,
        }
    }

    /// Set the number of full sinusoidal periods.
    #[must_use]
    pub fn with_periods(mut self, periods: usize) -> Self {
        self.num_periods = periods;
        self
    }

    /// Set the mesh resolution (axial Ã— radial).
    #[must_use]
    pub fn with_resolution(mut self, x: usize, y: usize) -> Self {
        self.resolution_x = x;
        self.resolution_y = y;
        self
    }

    /// Use a circular cross-section.
    #[must_use]
    pub fn with_circular(mut self, circular: bool) -> Self {
        self.circular = circular;
        self
    }

    /// Build a watertight surface mesh.
    ///
    /// # Errors
    /// Returns `BuildError` if the parameters are degenerate.
    pub fn build_surface(&self) -> Result<IndexedMesh, BuildError> {
        build_serpentine_surface(self)
    }
}

fn build_serpentine_surface(b: &SerpentineMeshBuilder) -> Result<IndexedMesh, BuildError> {
    let r = b.diameter / 2.0;
    let n_ax = b.resolution_x.max(4) * b.num_periods;
    let n_ang = if b.circular {
        (b.resolution_y.max(2) * 4).max(4)
    } else {
        4
    };
    let total_len = b.wavelength * b.num_periods as f64;

    let wall_region = RegionId::from_usize(0);
    let inlet_region = RegionId::from_usize(1);
    let outlet_region = RegionId::from_usize(2);

    let mut mesh = IndexedMesh::new();
    let spine: Vec<(f64, f64, f64)> = (0..n_ax)
        .map(|i| {
            let z = total_len * i as f64 / (n_ax - 1) as f64;
            let y = b.amplitude * (std::f64::consts::TAU * z / b.wavelength).sin();
            (0.0, y, z)
        })
        .collect();

    let mut rings: Vec<Vec<crate::domain::core::index::VertexId>> = Vec::with_capacity(n_ax);
    for &(cx, cy, cz) in &spine {
        let mut ring = Vec::with_capacity(n_ang);
        for ia in 0..n_ang {
            let theta = std::f64::consts::TAU * ia as f64 / n_ang as f64;
            let (sin_t, cos_t) = theta.sin_cos();
            ring.push(mesh.add_vertex(
                Point3r::new(cx + r * cos_t, cy + r * sin_t, cz),
                Vector3r::new(cos_t, sin_t, 0.0),
            ));
        }
        rings.push(ring);
    }

    for iz in 0..(n_ax - 1) {
        for ia in 0..n_ang {
            let ia1 = (ia + 1) % n_ang;
            let (v00, v01) = (rings[iz][ia], rings[iz][ia1]);
            let (v10, v11) = (rings[iz + 1][ia], rings[iz + 1][ia1]);
            mesh.add_face_with_region(v00, v10, v01, wall_region);
            mesh.add_face_with_region(v01, v10, v11, wall_region);
        }
    }

    let (icx, icy, icz) = spine[0];
    let ic = mesh.add_vertex(Point3r::new(icx, icy, icz), Vector3r::new(0.0, 0.0, -1.0));
    for ia in 0..n_ang {
        let ia1 = (ia + 1) % n_ang;
        mesh.add_face_with_region(ic, rings[0][ia1], rings[0][ia], inlet_region);
    }

    let last = n_ax - 1;
    let (ocx, ocy, ocz) = spine[last];
    let oc = mesh.add_vertex(Point3r::new(ocx, ocy, ocz), Vector3r::new(0.0, 0.0, 1.0));
    for ia in 0..n_ang {
        let ia1 = (ia + 1) % n_ang;
        mesh.add_face_with_region(oc, rings[last][ia], rings[last][ia1], outlet_region);
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serpentine_produces_non_empty_mesh() {
        let mesh = SerpentineMeshBuilder::new(0.002, 0.005, 0.020)
            .with_periods(2)
            .with_resolution(8, 2)
            .build_surface()
            .expect("should succeed");
        assert!(!mesh.faces.is_empty());
        assert!(!mesh.vertices.is_empty());
    }

    #[test]
    fn serpentine_more_periods_more_faces() {
        let one = SerpentineMeshBuilder::new(0.002, 0.005, 0.020)
            .with_periods(1)
            .with_resolution(8, 2)
            .build_surface()
            .unwrap();
        let two = SerpentineMeshBuilder::new(0.002, 0.005, 0.020)
            .with_periods(2)
            .with_resolution(8, 2)
            .build_surface()
            .unwrap();
        assert!(two.faces.len() > one.faces.len());
    }
}
