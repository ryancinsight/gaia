//! Generate the reviewed mesh-family sheets used by the Gaia mdBook.

#[path = "book_mesh_gallery/builders/mod.rs"]
mod builders;
#[path = "book_mesh_gallery/manifest.rs"]
mod manifest;
#[path = "book_mesh_gallery/model.rs"]
mod model;
#[path = "book_mesh_gallery/render.rs"]
mod render;
#[path = "book_mesh_gallery/watertightness.rs"]
mod watertightness;

use std::fs;
use std::path::PathBuf;

use model::GalleryResult;

fn output_dir() -> GalleryResult<PathBuf> {
    let mut args = std::env::args_os();
    let _program = args.next();
    let path = args.next().map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("docs/book")
    });
    if args.next().is_some() {
        return Err("expected at most one output directory argument".into());
    }
    Ok(path)
}

fn main() -> GalleryResult<()> {
    let root = output_dir()?;
    let figures = root.join("figures");
    fs::create_dir_all(&figures)?;

    let primitive_cases = builders::primitive::cases()?;
    let (channel_cases, blockers) = builders::channel::cases()?;
    let topology_cases = builders::topology::cases()?;
    let (watertight_cases, watertight_rejections) = watertightness::cases()?;

    render::sheet(
        &primitive_cases,
        "Gaia analytic primitive mesh families",
        &figures.join("primitive-mesh-families.svg"),
    )?;
    render::diagnostic_sheet(
        &watertight_cases,
        &watertight_rejections,
        &figures.join("watertightness-diagnostics.svg"),
    )?;
    render::sheet(
        &channel_cases,
        "Gaia channel and sweep mesh families",
        &figures.join("channel-mesh-families.svg"),
    )?;
    render::sheet(
        &topology_cases,
        "Gaia topology and volume mesh families",
        &figures.join("topology-mesh-families.svg"),
    )?;
    manifest::write(
        &primitive_cases,
        &channel_cases,
        &topology_cases,
        &blockers,
        &root.join("figure_manifest.md"),
    )?;
    watertightness::write_manifest(
        &watertight_cases,
        &watertight_rejections,
        &root.join("watertightness_manifest.md"),
    )?;

    println!(
        "generated {} primitive, {} channel, {} topology, and {} watertightness cases in {}",
        primitive_cases.len(),
        channel_cases.len(),
        topology_cases.len(),
        watertight_cases.len(),
        root.display()
    );
    Ok(())
}
