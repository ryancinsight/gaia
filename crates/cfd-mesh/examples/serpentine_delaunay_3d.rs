//! Generates a serpentine channel fluid domain using 3D Delaunay
//! tetrahedralization and rigorous SDF compositing.
//!
//! Run with:
//! ```sh
//! cargo run -p cfd-mesh --example serpentine_delaunay_3d --release
//! ```

use cfd_mesh::application::delaunay::dim3::lattice::SdfMesher;
use cfd_mesh::application::delaunay::dim3::sdf::{CapsuleSdf, SmoothUnionSdf};
use cfd_mesh::application::watertight::check::check_watertight;
use cfd_mesh::infrastructure::io::stl;
use cfd_mesh::infrastructure::storage::edge_store::EdgeStore;
use nalgebra::Point3;
use std::fs;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("  3D Delaunay SDF Mesher — Serpentine Channel");
    println!("=================================================================");

    // To prevent severe discrete chordal aliasing on the extreme inner U-turn bend (R=0.5),
    // the max cell scale must drop below the target features.
    let cell_size = 0.25;
    let mesher = SdfMesher::new(cell_size);

    // Segment 1 (Forward flow)
    let seg1 = CapsuleSdf::new(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        1.0,
    );

    // Segment 2 (Transverse U-turn)
    let seg2 = CapsuleSdf::new(
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 5.0, 0.0),
        1.0,
    );

    // Segment 3 (Return flow)
    let seg3 = CapsuleSdf::new(
        Point3::new(10.0, 5.0, 0.0),
        Point3::new(0.0, 5.0, 0.0),
        1.0,
    );

    // Smooth union blending with radius k = 1.0 to guarantee C1 continuity and eliminate checkerboard pinches
    let u1 = SmoothUnionSdf::new(seg1, seg2, 1.0);
    let serpentine = SmoothUnionSdf::new(u1, seg3, 1.0);

    let mesh = mesher.build_volume(&serpentine);

    println!("  Volume Vertices : {}", mesh.vertex_count());
    println!("  Tetrahedral Cells: {}", mesh.cell_count());
    println!("  Volume Faces    : {}", mesh.face_count());

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
    let stl_path = out_dir.join("serpentine_delaunay_3d.stl");
    let file = fs::File::create(&stl_path)?;
    let mut writer = BufWriter::new(file);
    stl::write_binary_stl(&mut writer, &boundary.vertices, &boundary.faces)?;

    println!("  STL             : {}", stl_path.display());
    println!("=================================================================");
    Ok(())
}
