//! Boundary sealing: close holes in an otherwise-manifold mesh.

use crate::domain::core::index::{RegionId, VertexId};
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::topology::boundary_loops;
use crate::infrastructure::storage::edge_store::EdgeStore;
use crate::infrastructure::storage::face_store::{FaceData, FaceStore};
use crate::infrastructure::storage::vertex_pool::VertexPool;

/// Bound on the vertices visited while walking a *single* boundary loop.
///
/// A walk that has not closed by then is abandoned, so malformed adjacency
/// cannot spin. This bounds the walk only: a loop that does close is returned
/// whatever its size, because the fan below triangulates any length.
const MAX_BOUNDARY_PATH_LEN: usize = 4096;

/// Seal boundary loops by fan triangulation from the centroid.
///
/// For each connected boundary loop, insert a centroid vertex and create
/// triangles from each boundary edge to the centroid.
///
/// Returns the number of faces added.
#[must_use]
pub fn seal_boundary_loops(
    vertex_pool: &mut VertexPool,
    face_store: &mut FaceStore,
    edge_store: &EdgeStore,
    region: RegionId,
) -> usize {
    let boundary = edge_store.boundary_edges();
    if boundary.is_empty() {
        return 0;
    }

    // Collect boundary edges as directed pairs
    let mut boundary_pairs: Vec<(VertexId, VertexId)> = Vec::with_capacity(boundary.len());
    for &eid in &boundary {
        let edge = edge_store.get(eid);
        // For boundary edges (valence 1), the single adjacent face determines the winding.
        let face_id = edge.faces[0];
        let face = face_store.get(face_id);

        let (v0, v1) = edge.vertices;

        // Check if the face uses the edge as (v0, v1) or (v1, v0).
        // The boundary loop should run in the opposite direction to seal it.
        // Face edges are: (f.v[0], f.v[1]), (f.v[1], f.v[2]), (f.v[2], f.v[0])
        let mut is_forward = false;
        let [a, b, c] = face.vertices;
        if (a == v0 && b == v1) || (b == v0 && c == v1) || (c == v0 && a == v1) {
            is_forward = true;
        }

        if is_forward {
            // Face has (v0 -> v1). Boundary loop must be (v1 -> v0).
            boundary_pairs.push((v1, v0));
        } else {
            // Face has (v1 -> v0). Boundary loop must be (v0 -> v1).
            boundary_pairs.push((v0, v1));
        }
    }

    // Find connected loops. The walk is bounded; the returned loops are not,
    // since a sealed hole of any size is still fan-triangulated from its
    // centroid.
    let loops = boundary_loops::trace_loops(&boundary_pairs, MAX_BOUNDARY_PATH_LEN, usize::MAX);

    let mut faces_added = 0;
    for boundary_loop in &loops {
        // `trace_loops` only returns loops of at least three vertices, so the
        // fan below never produces a degenerate face.
        debug_assert!(boundary_loop.len() >= 3);

        // Compute centroid
        let mut centroid = Point3r::origin();
        for &vid in boundary_loop {
            centroid.coords += vertex_pool.position(vid).coords;
        }
        centroid.coords /= boundary_loop.len() as crate::domain::core::scalar::Real;

        let centroid_id = vertex_pool.insert_or_weld(centroid, Vector3r::zeros());

        // Fan triangulate
        for i in 0..boundary_loop.len() {
            let j = (i + 1) % boundary_loop.len();
            face_store.push(FaceData {
                vertices: [boundary_loop[i], boundary_loop[j], centroid_id],
                region,
            });
            faces_added += 1;
        }
    }

    faces_added
}
