//! Repair phases for reconstructed Boolean meshes.

mod collapse;
mod edges;
mod merge;
mod pipeline;
mod vertices;

pub(super) use pipeline::postprocess_boolean_mesh;

#[cfg(test)]
pub(super) use edges::split_non_manifold_edges;

use crate::infrastructure::storage::face_store::FaceData;
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Path-compressing union-find: return the root of `x` with halving compression.
///
/// # Invariant
/// `parent[root] == root`. Path halving rewrites each traversed node to its
/// grandparent, shortening future finds without allocating an auxiliary stack.
#[inline(always)]
fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Compute the unit normal of a face, or `None` if degenerate.
#[inline]
fn face_normal_of(face: &FaceData, vertices: &VertexPool) -> Option<leto::geometry::Vector3<f64>> {
    crate::domain::geometry::normal::triangle_normal(
        vertices.position(face.vertices[0]),
        vertices.position(face.vertices[1]),
        vertices.position(face.vertices[2]),
    )
}
