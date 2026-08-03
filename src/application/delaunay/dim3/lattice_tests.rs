use super::sdf::SphereSdf;
use super::SdfMesher;
use crate::domain::mesh::indexed::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;
use leto::geometry::Point3;

type MeshSignature = (
    Vec<[f64; 3]>,
    Vec<([u32; 3], u32)>,
    Vec<(Vec<usize>, crate::domain::topology::ElementType, Vec<usize>)>,
);

fn signature(mesh: &IndexedMesh<f64>) -> MeshSignature {
    (
        mesh.vertices
            .positions()
            .map(|point| [point.x, point.y, point.z])
            .collect(),
        mesh.faces
            .iter()
            .map(|face: &FaceData| (face.vertices.map(|vertex| vertex.raw()), face.region.0))
            .collect(),
        mesh.cells
            .iter()
            .map(|cell| {
                (
                    cell.faces.clone(),
                    cell.element_type,
                    cell.vertex_ids.clone(),
                )
            })
            .collect(),
    )
}

#[test]
fn repeated_sdf_meshing_is_reproducible() {
    let sdf = SphereSdf {
        center: Point3::origin(),
        radius: 1.0,
    };
    let mesher = SdfMesher::new(0.8);

    let first = mesher.build_volume(&sdf);
    let second = mesher.build_volume(&sdf);

    assert_eq!(signature(&first), signature(&second));
}
