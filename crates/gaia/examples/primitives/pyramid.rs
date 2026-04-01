//! Export: Pyramid → outputs/primitives/pyramid.stl
use gaia::application::watertight::check::check_watertight;
use gaia::domain::geometry::primitives::PrimitiveMesh;
use gaia::infrastructure::io::stl;
use gaia::infrastructure::storage::edge_store::EdgeStore;
use gaia::Pyramid;
use std::fs;
use std::io::BufWriter;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = Pyramid::default().build()?;
    let edges = EdgeStore::from_face_store(&mesh.faces);
    let report = check_watertight(&mesh.vertices, &mesh.faces, &edges);
    println!(
        "Pyramid V={} F={} watertight={} vol={:.4} chi={:?}",
        mesh.vertices.len(),
        mesh.faces.len(),
        report.is_watertight,
        report.signed_volume,
        report.euler_characteristic
    );
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs/primitives");
    fs::create_dir_all(&out)?;
    stl::write_binary_stl(
        &mut BufWriter::new(fs::File::create(out.join("pyramid.stl"))?),
        &mesh.vertices,
        &mesh.faces,
    )?;
    Ok(())
}
