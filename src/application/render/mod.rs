//! Software mesh rendering: a camera, a z-buffered rasteriser, and flat shading.
//!
//! This module turns an [`IndexedMesh`](crate::domain::mesh::IndexedMesh) into
//! pixels. It is deliberately split from whatever owns a window:
//!
//! - [`OrbitCamera`] is the interaction model — orbit, pan, dolly, fit —
//!   expressed as semantic operations, with no notion of pixels or events.
//! - [`Renderer`] owns a depth buffer and writes colour into a slice the caller
//!   supplies, so a host can render straight into its own framebuffer.
//! - [`transform`] holds the two matrix constructions the pipeline needs.
//!
//! ## Host split
//!
//! Nothing here knows what a window, a surface, or an input event is. A host
//! supplies three things and gets a frame back:
//!
//! 1. a colour buffer of `width * height` `u32` pixels,
//! 2. mouse and keyboard events, mapped onto the camera's semantic operations,
//! 3. a present call.
//!
//! [`Rgba8::packed`] and [`Renderer::render`] agree with the host framebuffer
//! used by `metis-platform` (`0xAARRGGBB`, row-major, top-down), so the handoff
//! is a copy with no swizzle. Keeping that agreement here rather than in the
//! host means the renderer can be exercised headlessly in this crate's tests.
//!
//! ## Example
//!
//! ```rust,no_run
//! use gaia::application::render::{OrbitCamera, RenderSettings, Renderer};
//! use gaia::domain::core::scalar::Point3r;
//! use gaia::domain::mesh::IndexedMesh;
//!
//! let mesh: IndexedMesh = IndexedMesh::new();
//! let (width, height) = (800_u32, 600_u32);
//!
//! let mut camera = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 10.0);
//! camera.fit(&mesh.bounding_box(), f64::from(width) / f64::from(height));
//!
//! let mut renderer = Renderer::new(width, height)?;
//! let mut pixels = vec![0_u32; (width * height) as usize];
//! let stats = renderer.render(&mesh, &camera, &mut pixels, &RenderSettings::default())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod camera;
pub mod raster;
pub mod transform;

pub use camera::{OrbitCamera, DEFAULT_FOV_Y};
pub use raster::{
    CullMode, Material, RenderError, RenderSettings, RenderStats, Renderer, Rgba8, MAX_PIXELS,
};
pub use transform::Mat4;
