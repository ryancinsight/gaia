//! The renderer: a depth buffer, a viewport, and the per-frame loop.

use crate::domain::core::scalar::Real;
use crate::domain::mesh::indexed::IndexedMesh;

use super::super::camera::OrbitCamera;
use super::super::transform::{to_homogeneous, transform};
use super::clip::clip_against_near_plane;
use super::error::RenderError;
use super::rasterize::{rasterize, screen_triangle};
use super::settings::{CullMode, RenderSettings};
use super::stats::RenderStats;
use super::MAX_PIXELS;

/// Rasterises meshes into a caller-owned colour buffer.
#[derive(Debug, Clone)]
pub struct Renderer {
    width: u32,
    height: u32,
    depth: Vec<Real>,
}

impl Renderer {
    /// Create a renderer for a `width` x `height` viewport.
    ///
    /// # Errors
    ///
    /// [`RenderError::DimensionsTooLarge`] when `width * height` exceeds
    /// [`MAX_PIXELS`]. A zero dimension is allowed and renders nothing, because
    /// a window can legitimately be collapsed to nothing mid-drag.
    pub fn new(width: u32, height: u32) -> Result<Self, RenderError> {
        let mut renderer = Self {
            width: 0,
            height: 0,
            depth: Vec::new(),
        };
        renderer.resize(width, height)?;
        Ok(renderer)
    }

    /// Change the viewport size, reallocating the depth buffer.
    ///
    /// # Errors
    ///
    /// [`RenderError::DimensionsTooLarge`] as for [`Self::new`].
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        let pixels = pixel_count(width, height);
        if pixels > MAX_PIXELS {
            return Err(RenderError::DimensionsTooLarge { width, height });
        }
        self.width = width;
        self.height = height;
        self.depth.clear();
        self.depth.resize(pixels, 0.0);
        Ok(())
    }

    /// The current viewport width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The current viewport height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The depth buffer, row-major, holding `1/w` per pixel.
    ///
    /// Exposed for tests and for a host that wants to pick a world position out
    /// of a pixel; a value of zero means nothing was drawn there.
    #[must_use]
    pub fn depth(&self) -> &[Real] {
        &self.depth
    }

    /// Render `mesh` into `color`.
    ///
    /// `color` is row-major `0xAARRGGBB`, `width * height` entries, and is fully
    /// overwritten — background first, then triangles. Only its first
    /// `width * height` entries are touched, so a host may pass a larger buffer.
    ///
    /// # Errors
    ///
    /// [`RenderError::ColorBufferTooSmall`] when `color` cannot hold the frame,
    /// or [`RenderError::DegenerateCamera`] when the camera has no usable
    /// projection for the current aspect.
    pub fn render(
        &mut self,
        mesh: &IndexedMesh<Real>,
        camera: &OrbitCamera,
        color: &mut [u32],
        settings: &RenderSettings,
    ) -> Result<RenderStats, RenderError> {
        let mut stats = RenderStats::default();
        let pixels = pixel_count(self.width, self.height);
        if pixels == 0 {
            return Ok(stats);
        }
        if color.len() < pixels {
            return Err(RenderError::ColorBufferTooSmall {
                needed: pixels,
                provided: color.len(),
            });
        }

        let aspect = Real::from(self.width) / Real::from(self.height);
        let view_projection = camera
            .view_projection(aspect)
            .ok_or(RenderError::DegenerateCamera)?;
        let near = camera.near();

        let background = settings.background.packed();
        color[..pixels].fill(background);
        self.depth[..pixels].fill(0.0);

        // The headlight sits on the view axis, pointing from the target back
        // towards the eye.
        let forward = camera.forward();
        let light = -forward;

        for face in mesh.faces.as_slice() {
            stats.faces_considered += 1;
            let [ia, ib, ic] = face.vertices;
            let pa = *mesh.vertices.position(ia);
            let pb = *mesh.vertices.position(ib);
            let pc = *mesh.vertices.position(ic);

            let raw_normal = (pb - pa).cross(pc - pa);
            let normal_len = raw_normal.norm();
            if !normal_len.is_finite() || normal_len <= 0.0 {
                stats.degenerate_faces += 1;
                continue;
            }
            let normal = raw_normal / normal_len;

            if settings.cull == CullMode::Back && normal.dot(forward) >= 0.0 {
                stats.backface_culled += 1;
                continue;
            }

            let color_word = settings.material.shade(&normal, &light).packed();

            let clip = [
                transform(&view_projection, &to_homogeneous(&pa)),
                transform(&view_projection, &to_homogeneous(&pb)),
                transform(&view_projection, &to_homogeneous(&pc)),
            ];
            let (polygon, count) = clip_against_near_plane(&clip, near);
            if count < 3 {
                stats.near_plane_clipped += 1;
                continue;
            }

            // Fan-triangulate the clipped polygon. The near plane can turn a
            // triangle into a quad, and a fan covers both cases.
            for i in 1..count - 1 {
                let Some(tri) = screen_triangle(
                    [polygon[0], polygon[i], polygon[i + 1]],
                    self.width,
                    self.height,
                ) else {
                    continue;
                };
                stats.triangles_rasterized += 1;
                stats.fragments_passed += rasterize(
                    tri,
                    color_word,
                    self.width,
                    self.height,
                    &mut color[..pixels],
                    &mut self.depth[..pixels],
                );
            }
        }

        Ok(stats)
    }
}

/// `width * height` as a pixel count, saturating rather than wrapping.
fn pixel_count(width: u32, height: u32) -> usize {
    (width as usize).saturating_mul(height as usize)
}
