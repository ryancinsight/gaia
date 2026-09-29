//! Tests for the mesh-arrangement module surface.

use super::*;
use crate::application::csg::arrangement::boolean_csg::csg_boolean as arrangement_csg_boolean;
use crate::application::csg::boolean::{csg_boolean, BooleanOp};
use crate::application::watertight::check::{check_watertight, WatertightReport};
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::geometry::primitives::UvSphere;
use crate::domain::mesh::IndexedMesh;

/// Convenience wrapper: rebuild edges then return a full watertight report.
fn watertight_report(mesh: &mut IndexedMesh) -> WatertightReport {
    mesh.rebuild_edges();
    check_watertight(&mesh.vertices, &mesh.faces, mesh.edges_ref().unwrap())
}

/// Low-level boolean without the watertight post-check.
///
/// Use this for edge-case regression tests where the mesh is known to be
/// geometrically correct but may have topological defects (e.g. coplanar
/// caps that currently produce boundary seams).
fn boolean_raw(op: BooleanOp, mesh_a: &IndexedMesh, mesh_b: &IndexedMesh) -> IndexedMesh {
    use crate::domain::core::index::VertexId;
    use crate::infrastructure::storage::face_store::FaceData;
    use crate::infrastructure::storage::vertex_pool::VertexPool;
    use hashbrown::HashMap;
    let mut combined = VertexPool::default_millifluidic();
    let mut remap_a: HashMap<VertexId, VertexId> = HashMap::new();
    for (old_id, _) in mesh_a.vertices.iter() {
        let pos = *mesh_a.vertices.position(old_id);
        let nrm = *mesh_a.vertices.normal(old_id);
        remap_a.insert(old_id, combined.insert_or_weld(pos, nrm));
    }
    let mut remap_b: HashMap<VertexId, VertexId> = HashMap::new();
    for (old_id, _) in mesh_b.vertices.iter() {
        let pos = *mesh_b.vertices.position(old_id);
        let nrm = *mesh_b.vertices.normal(old_id);
        remap_b.insert(old_id, combined.insert_or_weld(pos, nrm));
    }
    let faces_a: Vec<FaceData> = mesh_a
        .faces
        .iter()
        .map(|f| FaceData {
            vertices: f.vertices.map(|v| remap_a[&v]),
            region: f.region,
        })
        .collect();
    let faces_b: Vec<FaceData> = mesh_b
        .faces
        .iter()
        .map(|f| FaceData {
            vertices: f.vertices.map(|v| remap_b[&v]),
            region: f.region,
        })
        .collect();
    let input_slice: &[Vec<FaceData>] = &[faces_a, faces_b];
    let result_faces = arrangement_csg_boolean(op, input_slice, &mut combined)
        .expect("csg_boolean should not error");
    super::super::reconstruct::reconstruct_mesh(&result_faces, &combined)
}

/// Build a UV sphere centred at `(cx, cy, cz)` with radius `r` and the
/// given latitude/longitude resolution.
fn make_sphere(cx: f64, cy: f64, cz: f64, r: f64, stacks: usize, segments: usize) -> IndexedMesh {
    use crate::domain::geometry::primitives::PrimitiveMesh;
    let sphere = UvSphere {
        radius: r,
        center: Point3r::new(cx, cy, cz),
        segments,
        stacks,
    };
    sphere.build().expect("UvSphere::build failed")
}

/// Compute the signed volume of an `IndexedMesh` using the divergence theorem.
///
/// `vol = (1/6) * ÃŽÂ£_face  (v0 Ã‚Â· (v1 Ãƒâ€” v2))`
fn signed_volume(mesh: &IndexedMesh) -> f64 {
    let mut vol = 0.0_f64;
    for face in mesh.faces.iter() {
        let a = mesh.vertices.position(face.vertices[0]);
        let b = mesh.vertices.position(face.vertices[1]);
        let c = mesh.vertices.position(face.vertices[2]);
        vol += a.x * (b.y * c.z - b.z * c.y)
            + a.y * (b.z * c.x - b.x * c.z)
            + a.z * (b.x * c.y - b.y * c.x);
    }
    (vol / 6.0).abs()
}

mod part1;
mod part2;
mod part3;
