use gaia::application::csg::boolean::{csg_boolean_nary, BooleanOp};
use gaia::application::delaunay::dim3::{sdf::SphereSdf, SdfMesher};
use gaia::application::hierarchy::{
    hex_to_tet::HexToTetConverter, hierarchical_mesh::P2MeshConverter,
};
use gaia::domain::core::scalar::Point3r;
use gaia::domain::geometry::primitives::PrimitiveMesh;
use gaia::domain::grid::{StructuredGridBuilder, StructuredHexGridBuilder};

use super::super::model::{GalleryResult, MeshCase};

pub(crate) fn cases() -> GalleryResult<Vec<MeshCase>> {
    let tetrahedral_grid = StructuredGridBuilder::new(2, 2, 2).build()?;
    let hexahedral_grid = StructuredHexGridBuilder::new(2, 2, 2).build();
    let hex_to_tet = HexToTetConverter::convert(&hexahedral_grid);
    let p2 = P2MeshConverter::convert_to_p2(&tetrahedral_grid);

    let cube_a = gaia::Cube {
        origin: Point3r::origin(),
        width: 1.0,
        height: 1.0,
        depth: 1.0,
    }
    .build()?;
    let sdf = SphereSdf {
        center: Point3r::origin(),
        radius: 1.0,
    };
    let sdf_volume = SdfMesher::new(0.8).build_volume(&sdf);

    Ok(vec![
        MeshCase {
            slug: "structured-tetrahedral-grid",
            title: "Structured tetrahedral grid",
            source: "src/domain/grid.rs",
            parameters: "cells=(2,2,2), five-tetrahedron decomposition",
            mesh: tetrahedral_grid,
        },
        MeshCase {
            slug: "structured-hexahedral-grid",
            title: "Structured hexahedral grid",
            source: "src/domain/grid.rs",
            parameters: "cells=(2,2,2), triangulated hexahedral boundary",
            mesh: hexahedral_grid,
        },
        MeshCase {
            slug: "hex-to-tet",
            title: "Hexahedron to tetrahedron",
            source: "src/application/hierarchy/hex_to_tet.rs",
            parameters: "2×2×2 structured hexahedral input",
            mesh: hex_to_tet,
        },
        MeshCase {
            slug: "p2-refinement",
            title: "P2 surface refinement",
            source: "src/application/hierarchy/hierarchical_mesh.rs",
            parameters: "1:4 refinement of the structured tetrahedral boundary",
            mesh: p2,
        },
        MeshCase {
            slug: "csg-nary-identity",
            title: "CSG n-ary identity",
            source: "src/application/csg/boolean/indexed.rs",
            parameters: "BooleanOp::Union with one unit-cube operand",
            mesh: csg_boolean_nary(BooleanOp::Union, &[cube_a])?,
        },
        MeshCase {
            slug: "sdf-tetrahedral-volume",
            title: "SDF tetrahedral volume",
            source: "src/application/delaunay/dim3/lattice.rs",
            parameters: "SphereSdf radius=1, cell_size=0.8",
            mesh: sdf_volume,
        },
    ])
}
