//! Why a frame could not be rendered.

use thiserror::Error as ThisError;

use super::MAX_PIXELS;

/// Why a frame could not be rendered.
#[derive(Debug, Clone, PartialEq, Eq, ThisError)]
pub enum RenderError {
    /// The requested dimensions exceed [`MAX_PIXELS`].
    #[error("viewport {width}x{height} exceeds the {MAX_PIXELS}-pixel limit")]
    DimensionsTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },

    /// The colour slice handed to
    /// [`Renderer::render`](super::renderer::Renderer::render) is too short.
    #[error("colour buffer holds {provided} pixels but {needed} are required")]
    ColorBufferTooSmall {
        /// `width * height` for the renderer's current size.
        needed: usize,
        /// Length of the slice the caller passed.
        provided: usize,
    },

    /// The camera and aspect do not define a projection.
    ///
    /// Reachable when the eye rounds to the target — a tiny distance from a
    /// target far enough from the origin to absorb it — so the camera has no
    /// view direction.
    #[error("camera and aspect do not define a view-projection matrix")]
    DegenerateCamera,
}
