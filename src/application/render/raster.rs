//! Z-buffered software rasterisation of indexed triangle meshes.
//!
//! The renderer owns only what a frame cannot be drawn without and the host
//! cannot supply — the depth buffer — and writes colour into a slice the caller
//! owns. That keeps the handoff to a host framebuffer a plain copy rather than
//! a second full-resolution allocation per frame, and it means this module
//! never needs to know what a window is.
//!
//! ## Pipeline
//!
//! 1. **Transform.** Each face's three positions go to clip space through the
//!    camera's view-projection matrix.
//! 2. **Cull.** Back-facing triangles are dropped in world space, before any
//!    per-vertex work, using the face normal against the view direction.
//! 3. **Clip.** Triangles straddling the near plane are clipped against it, so
//!    the perspective divide never runs on a vanishing `w`. The far plane is
//!    left to the depth test.
//! 4. **Rasterise.** Each surviving triangle is scanned over its screen-space
//!    bounding box with half-space edge functions and tested against the depth
//!    buffer.
//!
//! ## Depth
//!
//! The depth key is `1/w`, which is *affine* in screen space, so it interpolates
//! with plain barycentric weights and needs no perspective correction. Larger
//! `1/w` is nearer, so the test keeps the maximum. `w` equals the eye-space
//! distance in front of the camera under [`super::transform::perspective`]; the
//! test `perspective_w_is_eye_space_depth` pins that, because depth ordering
//! silently inverts if it ever stops holding.
//!
//! ## Shading
//!
//! Flat, one colour per triangle, lit by a headlight on the view axis with an
//! ambient floor — the same default a VTK surface render uses. Lambert uses
//! `|n·l|` so a mesh with inconsistent winding is still readable rather than
//! half black; the host can correct the winding with
//! [`crate::domain::mesh::IndexedMesh::orient_outward`].
//!
//! ## Module layout
//!
//! One concern per module, so a stage can be read and changed without the
//! others:
//!
//! - [`color`] — the colour type and its host packing.
//! - [`settings`] — cull mode, material, and the settings bundle.
//! - [`stats`] — the per-frame counters.
//! - [`error`] — why a frame could not be rendered.
//! - [`renderer`] — the depth buffer, the viewport, and the frame loop.
//! - `clip` and `rasterize` — the two internal stages, kept private because
//!   their signatures are an implementation detail of the frame loop.

/// Largest colour buffer the renderer will allocate a depth buffer for.
///
/// 16 M pixels, matching the host framebuffer limit in `metis-platform`, so a
/// renderer and a surface agree on what "too large" means.
pub const MAX_PIXELS: usize = 16 * 1024 * 1024;

pub mod color;
pub mod error;
pub mod renderer;
pub mod settings;
pub mod stats;

mod clip;
mod rasterize;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_support;

pub use color::Rgba8;
pub use error::RenderError;
pub use renderer::Renderer;
pub use settings::{CullMode, Material, RenderSettings};
pub use stats::RenderStats;
