#![cfg(test)]

use super::super::rectangular_prism::rectangular_prism_union;
use super::repair::split_non_manifold_edges;
use super::{csg_boolean, csg_boolean_nary};
use crate::application::csg::boolean::normalization::{
    normalization_transform, normalize_operand, NormalizedOperand,
};
use crate::application::csg::boolean::BooleanOp;
use crate::application::watertight::check::check_watertight;
use crate::domain::core::scalar::Point3r;
use crate::domain::geometry::primitives::{Cube, Cylinder, Disk, PrimitiveMesh, UvSphere};
use crate::domain::mesh::IndexedMesh;
use crate::infrastructure::storage::face_store::FaceData;

fn sphere() -> IndexedMesh {
    UvSphere {
        radius: 1.0,
        center: Point3r::origin(),
        segments: 16,
        stacks: 8,
    }
    .build()
    .expect("sphere build")
}

fn cylinder() -> IndexedMesh {
    Cylinder {
        base_center: Point3r::new(0.0, -1.5, 0.0),
        radius: 0.4,
        height: 3.0,
        segments: 16,
    }
    .build()
    .expect("cylinder build")
}

fn cube_a() -> IndexedMesh {
    Cube {
        origin: Point3r::new(-1.0, -1.0, -1.0),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_a build")
}

fn cube_b() -> IndexedMesh {
    Cube {
        origin: Point3r::new(-0.5, -0.5, -0.5),
        width: 2.0,
        height: 2.0,
        depth: 2.0,
    }
    .build()
    .expect("cube_b build")
}

fn disk_a() -> IndexedMesh {
    Disk {
        center: Point3r::new(0.0, 0.0, 0.0),
        radius: 1.0,
        segments: 16,
    }
    .build()
    .expect("disk_a build")
}

fn disk_b() -> IndexedMesh {
    Disk {
        center: Point3r::new(0.5, 0.0, 0.0),
        radius: 1.0,
        segments: 16,
    }
    .build()
    .expect("disk_b build")
}

/// Assert a 3-D CSG result is watertight with a positive signed volume.
fn assert_3d_watertight(mut mesh: IndexedMesh) {
    mesh.rebuild_edges();
    let report = check_watertight(&mesh.vertices, &mesh.faces, mesh.edges_ref().unwrap());
    assert!(
        report.is_watertight,
        "CSG result must be watertight: {} boundary edge(s), {} non-manifold edge(s)",
        report.boundary_edge_count, report.non_manifold_edge_count,
    );
    assert!(
        mesh.signed_volume() > 0.0,
        "CSG result must have positive signed volume (outward-oriented normals)",
    );
}

fn component_count(mesh: &mut IndexedMesh) -> usize {
    use crate::domain::topology::connectivity::connected_components;
    use crate::domain::topology::AdjacencyGraph;

    mesh.rebuild_edges();
    let adjacency = AdjacencyGraph::build(&mesh.faces, mesh.edges_ref().unwrap());
    connected_components(&mesh.faces, &adjacency).len()
}

fn symmetric_parallel_cylinders(segments: usize) -> (IndexedMesh, IndexedMesh) {
    let radius = 0.6;
    let height = 3.0;
    let separation = radius;
    let cyl_a = Cylinder {
        base_center: Point3r::new(-separation / 2.0, -height / 2.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("symmetric cyl_a build");
    let cyl_b = Cylinder {
        base_center: Point3r::new(separation / 2.0, -height / 2.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("symmetric cyl_b build");
    (cyl_a, cyl_b)
}

fn planar_branch(angle_from_x: f64, radius: f64, height: f64, segments: usize) -> IndexedMesh {
    use crate::application::csg::CsgNode;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let raw = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius,
        height,
        segments,
    }
    .build()
    .expect("branch build");
    let rotation = UnitQuaternion::<f64>::from_axis_angle(
        Vector3::z_axis(),
        angle_from_x - std::f64::consts::FRAC_PI_2,
    );
    CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(raw))),
        iso: Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
    }
    .evaluate()
    .expect("branch transform")
}

fn planar_trunk(radius: f64, height: f64, extension: f64, segments: usize) -> IndexedMesh {
    use crate::application::csg::CsgNode;
    use leto::geometry::{Isometry3, Translation3, UnitQuaternion, Vector3};

    let raw = Cylinder {
        base_center: Point3r::new(0.0, 0.0, 0.0),
        radius,
        height: height + extension,
        segments,
    }
    .build()
    .expect("trunk build");
    let rotation =
        UnitQuaternion::<f64>::from_axis_angle(Vector3::z_axis(), -std::f64::consts::FRAC_PI_2);
    CsgNode::Transform {
        node: Box::new(CsgNode::Leaf(Box::new(raw))),
        iso: Isometry3::from_parts(Translation3::new(-height, 0.0, 0.0), rotation),
    }
    .evaluate()
    .expect("trunk transform")
}

fn quadfurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 20.0, -20.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

fn trifurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [45.0_f64, 90.0, -45.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

fn pentafurcation_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 30.0, 0.0, -30.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

// ── cube × cylinder coplanar (caps flush with cube walls) ──────────────────

fn cylinder_coplanar() -> IndexedMesh {
    Cylinder {
        base_center: Point3r::new(0.0, -1.0, 0.0),
        radius: 0.4,
        height: 2.0,
        segments: 16,
    }
    .build()
    .expect("cylinder_coplanar build")
}

// ── Trifurcation 60° pinch-vertex regression ─────────────────────────

fn trifurcation_60deg_meshes() -> Vec<IndexedMesh> {
    let radius = 0.5;
    let height = 3.0;
    let extension = radius * 0.10;
    let segments = 32;
    let mut meshes = vec![planar_trunk(radius, height, extension, segments)];
    for angle_deg in [60.0_f64, 90.0, -60.0] {
        meshes.push(planar_branch(
            angle_deg.to_radians(),
            radius,
            height,
            segments,
        ));
    }
    meshes
}

#[path = "indexed_tests/part1.rs"]
mod part1;
#[path = "indexed_tests/part2.rs"]
mod part2;
