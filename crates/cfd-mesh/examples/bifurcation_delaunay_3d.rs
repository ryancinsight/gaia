//! Generates a Y-junction bifurcation fluid domain using 3D Delaunay
//! tetrahedralization and rigorous SDF compositing.
//!
//! Run with:
//! ```sh
//! cargo run -p cfd-mesh --example bifurcation_delaunay_3d --release
//! ```

use cfd_mesh::application::delaunay::dim3::{sdf::CapsuleSdf, sdf::SmoothUnionSdf, SdfMesher};
use cfd_mesh::application::watertight::check::check_watertight;
use cfd_mesh::domain::core::VertexId;
use cfd_mesh::infrastructure::io::stl;
use cfd_mesh::infrastructure::storage::edge_store::EdgeStore;
use nalgebra::Point3;
use std::fs;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("  3D Delaunay SDF Mesher — Y-Junction Bifurcation");
    println!("=================================================================");

    // A 0.24 cell size deliberately phases the BCC lattice out of exact integer alignment 
    // with the origin-centered Y-junction, eliminating the 8-edge non-manifold triple-point pinches.
    let cell_size = 0.24;
    let mut mesher = SdfMesher::new(cell_size);

    // Parent channel axis
    let parent = CapsuleSdf::new(
        Point3::new(-10.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        2.0,
    );

    // Superior daughter channel
    let top_branch = CapsuleSdf::new(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 5.0, 0.0),
        1.5,
    );

    // Inferior daughter channel
    let bottom_branch = CapsuleSdf::new(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, -5.0, 0.0),
        1.5,
    );

    // Smooth union blending with radius k = 1.0 to eliminate non-manifold intersection pinches
    let branches = SmoothUnionSdf::new(top_branch, bottom_branch, 1.0);
    let bifurcation = SmoothUnionSdf::new(parent, branches, 1.0);

    let mesh = mesher.build_volume(&bifurcation);

    println!("  Volume Vertices : {}", mesh.vertex_count());
    println!("  Tetrahedral Cells: {}", mesh.cell_count());
    // Exact Volumetric Integration via interior 3-simplices
    let mut interior_volume = 0.0;
    for cell in mesh.cells() {
        if cell.vertex_ids.len() == 4 {
            let a = mesh.vertices.position(VertexId::from_usize(cell.vertex_ids[0])).coords;
            let b = mesh.vertices.position(VertexId::from_usize(cell.vertex_ids[1])).coords;
            let c = mesh.vertices.position(VertexId::from_usize(cell.vertex_ids[2])).coords;
            let d = mesh.vertices.position(VertexId::from_usize(cell.vertex_ids[3])).coords;
            
            // Volume = |(A - D) ⋅ ((B - D) × (C - D))| / 6.0
            let dot_val: f64 = (a - d).dot(&(b - d).cross(&(c - d)));
            let v = dot_val.abs() / 6.0;
            interior_volume += v;
        }
    }
    println!("  Interior Volume : {:.6} mm³ (Sum of interior tetrahedra)", interior_volume);

    // Extract the boundary representation (surface hull only).
    // The Delaunay extraction intrinsically guarantees positive-oriented simplices,
    // which extract_boundary_mesh geometrically enforces onto the boundary B-Rep.
    let mut boundary = mesh.extract_boundary_mesh();
    boundary.recompute_normals();

    // Watertight validation following the canonical cfd-mesh pattern.
    let edges = EdgeStore::from_face_store(&boundary.faces);
    let report = check_watertight(&boundary.vertices, &boundary.faces, &edges);

    println!("  B-Rep Vertices  : {}", boundary.vertex_count());
    println!("  B-Rep Faces     : {}", boundary.face_count());
    println!("{:#?}", report);

    // Write to cfd-mesh/outputs (canonical crate-local path).
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs");
    fs::create_dir_all(&out_dir)?;
    let stl_path = out_dir.join("bifurcation_delaunay_3d.stl");
    let file = fs::File::create(&stl_path)?;
    let mut writer = BufWriter::new(file);
    stl::write_binary_stl(&mut writer, &boundary.vertices, &boundary.faces)?;

    println!("  STL             : {}", stl_path.display());
    println!("=================================================================");
    Ok(())
}
