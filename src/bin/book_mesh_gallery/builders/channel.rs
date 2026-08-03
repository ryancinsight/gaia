use gaia::application::channel::{
    BranchingMeshBuilder, ChannelPath, ChannelProfile, SerpentineMeshBuilder, SubstrateBuilder,
    SweepMesher, VenturiMeshBuilder,
};
use gaia::domain::core::index::RegionId;
use gaia::domain::core::scalar::Point3r;
use gaia::domain::mesh::IndexedMesh;
use gaia::infrastructure::storage::face_store::FaceData;
use gaia::infrastructure::storage::vertex_pool::VertexPool;

use super::super::model::{BuildBlocker, GalleryResult, MeshCase};

fn face_soup_mesh(faces: &[FaceData], pool: &VertexPool) -> IndexedMesh {
    let mut mesh = IndexedMesh::with_capacity(pool.len(), faces.len(), 0);
    for (_, vertex) in pool.iter() {
        mesh.add_vertex_unique(vertex.position, vertex.normal);
    }
    for face in faces {
        mesh.add_face_with_region(
            face.vertices[0],
            face.vertices[1],
            face.vertices[2],
            face.region,
        );
    }
    mesh.rebuild_edges();
    mesh
}

pub(crate) fn cases() -> GalleryResult<(Vec<MeshCase>, Vec<BuildBlocker>)> {
    let mut cases = Vec::with_capacity(6);
    let mut blockers = Vec::new();
    match BranchingMeshBuilder::bifurcation(0.004, 0.02, 0.002, 0.015, 0.5, 4).build_surface() {
        Ok(mesh) => cases.push(MeshCase {
            slug: "branching-bifurcation",
            title: "Branching bifurcation",
            source: "src/application/channel/branching.rs",
            parameters: "bifurcation, diameter=0.004, lengths=0.02, angle=0.5, resolution=4",
            mesh,
        }),
        Err(error) => blockers.push(BuildBlocker {
            category: "Channel",
            family: "Branching bifurcation",
            source: "src/application/channel/branching.rs",
            error: error.to_string(),
        }),
    }
    match BranchingMeshBuilder::trifurcation(0.004, 0.02, 0.002, 0.015, 0.5, 6).build_surface() {
        Ok(mesh) => cases.push(MeshCase {
            slug: "branching-trifurcation",
            title: "Branching trifurcation",
            source: "src/application/channel/branching.rs",
            parameters: "trifurcation, diameter=0.004, lengths=0.02, angle=0.5, resolution=6",
            mesh,
        }),
        Err(error) => blockers.push(BuildBlocker {
            category: "Channel",
            family: "Branching trifurcation",
            source: "src/application/channel/branching.rs",
            error: error.to_string(),
        }),
    }
    cases.push(MeshCase {
        slug: "serpentine-channel",
        title: "Serpentine channel",
        source: "src/application/channel/serpentine.rs",
        parameters:
            "diameter=0.002, amplitude=0.004, wavelength=0.01, periods=2, resolution=(12,4)",
        mesh: SerpentineMeshBuilder::new(0.002, 0.004, 0.01)
            .with_periods(2)
            .with_resolution(12, 4)
            .build_surface()?,
    });
    cases.push(MeshCase {
        slug: "venturi-channel",
        title: "Venturi channel",
        source: "src/application/channel/venturi.rs",
        parameters: "diameters=(0.01,0.004), lengths=(0.02,0.04,0.01,0.06,0.02), resolution=(8,4)",
        mesh: VenturiMeshBuilder::new(0.01, 0.004, 0.02, 0.04, 0.01, 0.06, 0.02)
            .with_resolution(8, 4)
            .build_surface()?,
    });
    cases.push(MeshCase {
        slug: "substrate",
        title: "Millifluidic substrate",
        source: "src/application/channel/substrate.rs",
        parameters: "width=0.04, depth=0.03, height=0.004",
        mesh: SubstrateBuilder::new(0.04, 0.03, 0.004).build_indexed()?,
    });

    let path = ChannelPath::new(vec![
        Point3r::new(0.0, 0.0, 0.0),
        Point3r::new(0.01, 0.0, 0.0),
        Point3r::new(0.02, 0.006, 0.002),
        Point3r::new(0.03, 0.006, 0.002),
    ]);
    let profile = ChannelProfile::RoundedRectangular {
        width: 0.004,
        height: 0.002,
        corner_radius: 0.0005,
        corner_segments: 3,
    };
    let mut pool = VertexPool::default_millifluidic();
    let faces = SweepMesher::new().sweep(&profile, &path, &mut pool, RegionId::from_usize(0));
    cases.push(MeshCase {
        slug: "profile-sweep",
        title: "Profile sweep",
        source: "src/application/channel/sweep.rs",
        parameters: "rounded rectangular profile, four centerline waypoints",
        mesh: face_soup_mesh(&faces, &pool),
    });

    Ok((cases, blockers))
}
