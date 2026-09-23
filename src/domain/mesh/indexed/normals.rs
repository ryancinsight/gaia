//! Vertex-normal recomputation from the current face winding.

use super::IndexedMesh;
use crate::domain::core::index::VertexId;
use crate::domain::core::scalar::Scalar;
use leto::geometry::Vector3;

impl<T: Scalar> IndexedMesh<T> {
    // ── Normal Recomputation ──────────────────────────────────

    /// Recompute all vertex normals from face geometry.
    ///
    /// After CSG operations, face winding may have changed, but the vertex
    /// normals stored in the pool are the original normals. This method
    /// recalculates normals based on the current face winding, averaging
    /// contributions from all faces that share each vertex.
    ///
    /// For each face, the normal is computed from the cross product:
    /// `n = normalize((v1 - v0) × (v2 - v0))`
    ///
    /// Each vertex's normal is the average of all face normals that use it.
    pub fn recompute_normals(&mut self) {
        use crate::domain::geometry::normal::triangle_normal;

        let mut normal_sums: Vec<Vector3<T>> = vec![Vector3::<T>::zeros(); self.vertices.len()];
        let mut counts: Vec<usize> = vec![0; self.vertices.len()];

        for (_, face) in self.faces.iter_enumerated() {
            let a = self.vertices.position(face.vertices[0]);
            let b = self.vertices.position(face.vertices[1]);
            let c = self.vertices.position(face.vertices[2]);

            let face_normal = triangle_normal(a, b, c).unwrap_or_else(|| Vector3::<T>::z());

            for &vi in &face.vertices {
                normal_sums[vi.as_usize()] += face_normal;
                counts[vi.as_usize()] += 1;
            }
        }

        for (i, (sum, count)) in normal_sums.iter().zip(counts.iter()).enumerate() {
            if *count > 0 {
                let avg = *sum / <T as Scalar>::from_f64(*count as f64);
                let len = avg.norm();
                if len > <T as Scalar>::from_f64(1e-12) {
                    self.vertices.set_normal(VertexId::new(i as u32), avg / len);
                }
            }
        }
    }
}
