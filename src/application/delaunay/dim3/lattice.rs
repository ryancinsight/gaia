//! Body-Centered Cubic (BCC) Lattice Seeding and SDF Volumetric Meshing.
//!
//! Generates an unstructured `IndexedMesh<T>` conforming to an implicit `Sdf3D` surface
//! using gradient descent and the robust-predicate `BowyerWatson3D`
//! tetrahedralizer.

use crate::application::delaunay::dim3::sdf::Sdf3D;
use crate::application::delaunay::dim3::tetrahedralize::BowyerWatson3D;
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::indexed::IndexedMesh;
use hashbrown::{HashMap, HashSet};
use leto::geometry::{Point3, Vector3};

/// An implicit-to-explicit tetrahedral mesh generator.
pub struct SdfMesher<T: Scalar> {
    /// Nominal edge length for the internal BCC lattice.
    pub cell_size: T,
    /// Number of gradient descent steps for boundary projection.
    pub snap_iterations: usize,
    /// Distance threshold normalized to cell_size for points that undergo snapping.
    pub snap_radius: T,
}

impl<T: Scalar> SdfMesher<T> {
    /// Create a new volumetric mesher with a target characteristic element edge length.
    pub fn new(cell_size: T) -> Self {
        Self {
            cell_size,
            snap_iterations: 15,
            snap_radius: <T as Scalar>::from_f64(1.5),
        }
    }
}

mod build;
