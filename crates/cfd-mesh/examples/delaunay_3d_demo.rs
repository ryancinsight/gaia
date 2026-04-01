//! Example: 3D Delaunay tetrahedralization of an SDF sphere.
//!
//! Generates a watertight tetrahedral volume mesh from an implicit sphere,
//! validates manifold invariants, and exports a binary STL.
//!
//! Run with:
//! ```sh
//! cargo run -p cfd-mesh --example delaunay_3d_demo --release
//! ```

use cfd_mesh::application::delaunay::dim3::lattice::SdfMesher;
use cfd_mesh::application::delaunay::dim3::sdf::SphereSdf;
use cfd_mesh::application::watertight::check::check_watertight;
use cfd_mesh::infrastructure::io::stl;
use cfd_mesh::infrastructure::storage::edge_store::EdgeStore;
use nalgebra::Point3;
use std::fs;
use std::io::BufWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================================");
    println!("  3D Delaunay SDF Mesher — Sphere r=5.0, cell_size=0.25");
    println!("=================================================================");

    let cell_size = 0.25;
    let mesher: SdfMesher<f64> = SdfMesher::new(cell_size);

    let center: Point3<f64> = Point3::origin();
    let sphere = SphereSdf { center, radius: 5.0 };

    let mesh = mesher.build_volume(&sphere);

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
    let expected_vol = 4.0 * std::f64::consts::PI * 5.0_f64.powi(3) / 3.0;

    println!("  B-Rep Vertices  : {}", boundary.vertex_count());
    println!("  B-Rep Faces     : {}", boundary.face_count());
    println!("  Watertight      : {}", report.is_watertight);
    println!(
        "  Volume          : {:.4} mm³  (expected {:.4})",
        report.signed_volume, expected_vol
    );
    println!(
        "  Euler χ         : {:?}  (expected 2)",
        report.euler_characteristic
    );

    // Write to cfd-mesh/outputs (canonical crate-local path).
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs");
    fs::create_dir_all(&out_dir)?;
    let stl_path = out_dir.join("delaunay_3d_demo.stl");
    let file = fs::File::create(&stl_path)?;
    let mut writer = BufWriter::new(file);
    stl::write_binary_stl(&mut writer, &boundary.vertices, &boundary.faces)?;

    println!("  STL             : {}", stl_path.display());
    println!("=================================================================");
    Ok(())
}
