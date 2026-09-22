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

use thiserror::Error as ThisError;

use crate::domain::core::scalar::{Real, Vector3r};
use crate::domain::mesh::indexed::IndexedMesh;

use super::camera::OrbitCamera;
use super::transform::{lerp4, to_homogeneous, transform, Vec4};

/// Largest colour buffer the renderer will allocate a depth buffer for.
///
/// 16 M pixels, matching the host framebuffer limit in `metis-platform`, so a
/// renderer and a surface agree on what "too large" means.
pub const MAX_PIXELS: usize = 16 * 1024 * 1024;

/// An 8-bit-per-channel colour.
///
/// [`Self::packed`] produces `0xAARRGGBB`, which is the format the host
/// framebuffer expects, so a rendered frame can be copied across without a
/// swizzle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rgba8 {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel.
    pub a: u8,
}

impl Rgba8 {
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Mid grey.
    pub const GRAY: Self = Self::rgb(128, 128, 128);
    /// Dark grey, the default viewport background.
    pub const DARK_GRAY: Self = Self::rgb(32, 34, 38);
    /// Light grey.
    pub const LIGHT_GRAY: Self = Self::rgb(200, 200, 200);

    /// Create a colour from its four channels.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque colour from three channels.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Pack into the host framebuffer's `0xAARRGGBB` word.
    #[must_use]
    pub const fn packed(self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }

    /// Scale the colour channels by `factor`, clamping into range.
    ///
    /// Alpha is left alone: scaling coverage is not what a lighting model
    /// means, and a partially transparent surface is the host's decision. A
    /// `NaN` factor yields zero rather than propagating a `NaN` into a channel
    /// cast, which would be an unspecified value.
    #[must_use]
    pub fn scaled(self, factor: Real) -> Self {
        let scale = |channel: u8| -> u8 {
            let value = Real::from(channel) * factor;
            if value.is_nan() || value <= 0.0 {
                0
            } else if value >= 255.0 {
                255
            } else {
                value as u8
            }
        };
        Self {
            r: scale(self.r),
            g: scale(self.g),
            b: scale(self.b),
            a: self.a,
        }
    }
}

/// Which triangles to discard before rasterising them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// Drop triangles whose outward normal faces away from the camera.
    ///
    /// Correct and roughly twice as fast for a closed, consistently wound
    /// surface — which is what this crate produces. Use [`Self::None`] for an
    /// open surface, or for one whose winding has not been corrected.
    #[default]
    Back,
    /// Keep every triangle; the depth test resolves visibility on its own.
    None,
}

/// A flat-shaded surface material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// The unlit surface colour.
    pub base_color: Rgba8,
    /// Fraction of the base colour present with no light contribution.
    pub ambient: Real,
    /// Fraction of the base colour that the diffuse term may add.
    pub diffuse: Real,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Rgba8::rgb(180, 190, 200),
            ambient: 0.25,
            diffuse: 0.75,
        }
    }
}

impl Material {
    /// The default neutral surface.
    #[must_use]
    pub fn surface() -> Self {
        Self::default()
    }

    /// Builder form of `base_color`.
    #[must_use]
    pub const fn with_base_color(mut self, base_color: Rgba8) -> Self {
        self.base_color = base_color;
        self
    }

    /// The flat colour for a triangle with unit outward `normal`, lit by a
    /// headlight along `light_dir`.
    ///
    /// The Lambert term is `|n·l|`: an absolute value, so a triangle facing away
    /// from the light is shaded as if it faced towards it. That is deliberate.
    /// A viewer is often handed a mesh whose winding is not yet consistent, and
    /// `max(0, n·l)` would render half of such a mesh black and make the winding
    /// look like a lighting bug.
    #[must_use]
    pub fn shade(&self, normal: &Vector3r, light_dir: &Vector3r) -> Rgba8 {
        let lambert = normal.dot(*light_dir).abs().clamp(0.0, 1.0);
        let intensity = (self.ambient + self.diffuse * lambert).clamp(0.0, 1.0);
        self.base_color.scaled(intensity)
    }
}

/// What to draw and how.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Colour written where no triangle covers a pixel.
    pub background: Rgba8,
    /// The surface material.
    pub material: Material,
    /// Which triangles to discard.
    pub cull: CullMode,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            background: Rgba8::DARK_GRAY,
            material: Material::surface(),
            cull: CullMode::Back,
        }
    }
}

/// What one call to [`Renderer::render`] did.
///
/// Reported rather than logged, so a host can put it on a status line and a
/// test can assert on it. Every counter is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderStats {
    /// Faces examined in the mesh.
    pub faces_considered: usize,
    /// Faces skipped for having no usable world normal.
    pub degenerate_faces: usize,
    /// Faces dropped by [`CullMode::Back`].
    pub backface_culled: usize,
    /// Faces that produced no triangle at all after near-plane clipping.
    pub near_plane_clipped: usize,
    /// Triangles handed to the rasteriser, after fan-triangulating clipped
    /// polygons.
    pub triangles_rasterized: usize,
    /// Pixels that passed the depth test.
    ///
    /// A pixel whose centre falls on an edge shared by two triangles can be
    /// counted once per triangle, so this is an upper bound on the number of
    /// distinct pixels covered, and equal to it in every case where no pixel
    /// centre lands on a shared edge. [`Renderer::render`] uses a strict
    /// comparison against the stored depth, so a triangle cannot overwrite a
    /// pixel already written by a coplanar neighbour at exactly equal depth.
    pub fragments_passed: usize,
}

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

    /// The colour slice handed to [`Renderer::render`] is too short.
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

/// A screen-space vertex: pixel position plus the depth key.
#[derive(Debug, Clone, Copy)]
struct ScreenVertex {
    x: Real,
    y: Real,
    inv_w: Real,
}

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

/// Clip a triangle against the near plane `w >= near` in clip space.
///
/// Sutherland-Hodgman over a single plane. A triangle clipped by one plane has
/// at most four sides, so the output is a fixed array plus a count and this
/// allocates nothing.
fn clip_against_near_plane(triangle: &[Vec4; 3], near: Real) -> ([Vec4; 4], usize) {
    let mut out = [Vec4::zeros(); 4];
    let mut count = 0;
    for i in 0..3 {
        let current = triangle[i];
        let next = triangle[(i + 1) % 3];
        let current_inside = current[3] >= near;
        let next_inside = next[3] >= near;
        if current_inside {
            out[count] = current;
            count += 1;
        }
        if current_inside != next_inside {
            let denom = next[3] - current[3];
            // The two `w` values straddle `near`, so the denominator is
            // non-zero for finite input. Guard anyway: a non-finite `w` would
            // otherwise produce a NaN vertex that propagates through the whole
            // fan without ever failing a comparison.
            if denom != 0.0 && denom.is_finite() {
                let t = (near - current[3]) / denom;
                out[count] = lerp4(&current, &next, t);
                count += 1;
            }
        }
    }
    (out, count)
}

/// Perspective-divide and map three clip-space vertices to the viewport.
///
/// Returns `None` if any vertex has a non-positive or non-finite `w`, which
/// cannot happen after near-plane clipping but would otherwise divide by zero.
fn screen_triangle(clip: [Vec4; 3], width: u32, height: u32) -> Option<[ScreenVertex; 3]> {
    let mut out = [ScreenVertex {
        x: 0.0,
        y: 0.0,
        inv_w: 0.0,
    }; 3];
    for (slot, v) in out.iter_mut().zip(clip.iter()) {
        let w = v[3];
        if !w.is_finite() || w <= 0.0 {
            return None;
        }
        let inv_w = 1.0 / w;
        let ndc_x = v[0] * inv_w;
        let ndc_y = v[1] * inv_w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() {
            return None;
        }
        slot.x = (ndc_x + 1.0) * 0.5 * Real::from(width);
        // Screen y grows downward, so the normalised y axis is flipped here and
        // nowhere else.
        slot.y = (1.0 - ndc_y) * 0.5 * Real::from(height);
        slot.inv_w = inv_w;
    }
    Some(out)
}

/// Twice the signed area of the triangle `(a, b, p)`, in screen coordinates.
#[inline]
fn edge(a: &ScreenVertex, b: &ScreenVertex, px: Real, py: Real) -> Real {
    (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
}

/// Fill a triangle, testing and updating the depth buffer.
///
/// Returns the number of pixels that passed the depth test. A half-space test
/// with a consistent winding gives every pixel exactly one owning triangle per
/// surface, so adjacent triangles neither seam nor double-shade; a pixel whose
/// centre lies exactly on a shared edge can still be counted twice, which is
/// what [`RenderStats::fragments_passed`] documents.
fn rasterize(
    mut v: [ScreenVertex; 3],
    color: u32,
    width: u32,
    height: u32,
    pixels: &mut [u32],
    depth: &mut [Real],
) -> usize {
    let mut area = edge(&v[0], &v[1], v[2].x, v[2].y);
    if area == 0.0 || !area.is_finite() {
        return 0;
    }
    if area < 0.0 {
        v.swap(1, 2);
        area = -area;
    }

    let max_x = Real::from(width) - 1.0;
    let max_y = Real::from(height) - 1.0;
    let min_px = v
        .iter()
        .map(|s| s.x)
        .fold(Real::INFINITY, Real::min)
        .max(0.0);
    let max_px = v
        .iter()
        .map(|s| s.x)
        .fold(Real::NEG_INFINITY, Real::max)
        .min(max_x);
    let min_py = v
        .iter()
        .map(|s| s.y)
        .fold(Real::INFINITY, Real::min)
        .max(0.0);
    let max_py = v
        .iter()
        .map(|s| s.y)
        .fold(Real::NEG_INFINITY, Real::max)
        .min(max_y);
    // Screen coordinates are finite by the time they reach here, because
    // `screen_triangle` rejects a non-finite normalised coordinate.
    if min_px > max_px || min_py > max_py {
        return 0;
    }

    let x0 = min_px.floor() as i64;
    let x1 = max_px.floor() as i64;
    let y0 = min_py.floor() as i64;
    let y1 = max_py.floor() as i64;
    let stride = width as usize;

    let mut written = 0;
    for y in y0..=y1 {
        let py = y as Real + 0.5;
        for x in x0..=x1 {
            let px = x as Real + 0.5;
            let w0 = edge(&v[1], &v[2], px, py);
            let w1 = edge(&v[2], &v[0], px, py);
            let w2 = edge(&v[0], &v[1], px, py);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            // `1/w` is affine in screen space, so plain barycentric weights are
            // already perspective-correct for it.
            let inv_w = (w0 * v[0].inv_w + w1 * v[1].inv_w + w2 * v[2].inv_w) / area;
            if inv_w.is_nan() || inv_w <= 0.0 {
                continue;
            }
            let index = y as usize * stride + x as usize;
            if inv_w <= depth[index] {
                continue;
            }
            depth[index] = inv_w;
            pixels[index] = color;
            written += 1;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Point3r;

    /// An axis-aligned cube of half-extent 1, wound outward.
    fn cube() -> IndexedMesh<Real> {
        let mut mesh = IndexedMesh::new();
        let ids: Vec<_> = [
            Point3r::new(-1.0, -1.0, -1.0),
            Point3r::new(1.0, -1.0, -1.0),
            Point3r::new(1.0, 1.0, -1.0),
            Point3r::new(-1.0, 1.0, -1.0),
            Point3r::new(-1.0, -1.0, 1.0),
            Point3r::new(1.0, -1.0, 1.0),
            Point3r::new(1.0, 1.0, 1.0),
            Point3r::new(-1.0, 1.0, 1.0),
        ]
        .into_iter()
        .map(|p| mesh.add_vertex_pos(p))
        .collect();
        for q in [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 4, 7, 3],
        ] {
            mesh.add_face(ids[q[0]], ids[q[1]], ids[q[2]]);
            mesh.add_face(ids[q[0]], ids[q[2]], ids[q[3]]);
        }
        mesh
    }

    /// A quad spanning y and z in `[-1, 1]` at a fixed `x`, so it faces the
    /// camera's default `+x` eye.
    fn quad_facing_camera(mesh: &mut IndexedMesh<Real>, x: Real) {
        let a = mesh.add_vertex_pos(Point3r::new(x, -1.0, -1.0));
        let b = mesh.add_vertex_pos(Point3r::new(x, 1.0, -1.0));
        let c = mesh.add_vertex_pos(Point3r::new(x, 1.0, 1.0));
        let d = mesh.add_vertex_pos(Point3r::new(x, -1.0, 1.0));
        mesh.add_face(a, b, c);
        mesh.add_face(a, c, d);
    }

    /// The default camera looks along `-x` from `+x`, so this views the cube
    /// from a corner.
    fn camera_facing_cube() -> OrbitCamera {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 8.0);
        cam.orbit(0.6, 0.4);
        cam.set_clip_planes(0.1, 100.0);
        cam
    }

    /// Colour and depth coverage must agree, and the reported fragment count
    /// must match the distinct pixels covered up to shared-edge double counts.
    fn assert_coverage_agrees(
        renderer: &Renderer,
        color: &[u32],
        background: Rgba8,
        stats: RenderStats,
    ) {
        let background = background.packed();
        let covered = color.iter().filter(|&&p| p != background).count();
        let with_depth = renderer.depth().iter().filter(|&&d| d > 0.0).count();
        assert_eq!(
            covered, with_depth,
            "colour coverage ({covered}) and depth coverage ({with_depth}) disagree"
        );
        assert!(
            stats.fragments_passed >= covered,
            "a covered pixel must have been written: {} writes for {covered} pixels",
            stats.fragments_passed
        );
        assert!(
            stats.fragments_passed <= covered + covered / 50 + 1,
            "only shared-edge pixels may be counted twice: {} writes for {covered} pixels",
            stats.fragments_passed
        );
    }

    #[test]
    fn packed_matches_the_host_framebuffer_word_order() {
        // 0xAARRGGBB, the same value `u32::from_be_bytes([a, r, g, b])` gives.
        assert_eq!(Rgba8::new(0x11, 0x22, 0x33, 0x44).packed(), 0x4411_2233);
        assert_eq!(Rgba8::WHITE.packed(), 0xFFFF_FFFF);
        assert_eq!(Rgba8::BLACK.packed(), 0xFF00_0000);
    }

    #[test]
    fn scaled_clamps_and_leaves_alpha_alone() {
        let c = Rgba8::new(100, 200, 250, 128);
        assert_eq!(c.scaled(0.0), Rgba8::new(0, 0, 0, 128));
        assert_eq!(c.scaled(1.0), c);
        assert_eq!(c.scaled(10.0), Rgba8::new(255, 255, 255, 128));
        assert_eq!(c.scaled(-1.0), Rgba8::new(0, 0, 0, 128));
        assert_eq!(c.scaled(f64::NAN), Rgba8::new(0, 0, 0, 128));
    }

    #[test]
    fn shade_uses_an_ambient_floor_and_an_absolute_lambert_term() {
        let m = Material {
            base_color: Rgba8::rgb(200, 200, 200),
            ambient: 0.25,
            diffuse: 0.75,
        };
        let light = Vector3r::new(1.0, 0.0, 0.0);
        let facing = m.shade(&Vector3r::new(1.0, 0.0, 0.0), &light);
        let away = m.shade(&Vector3r::new(-1.0, 0.0, 0.0), &light);
        let edge_on = m.shade(&Vector3r::new(0.0, 0.0, 1.0), &light);
        assert_eq!(facing.r, 200);
        assert_eq!(away, facing, "the Lambert term is absolute");
        assert_eq!(edge_on.r, 50, "ambient only at 90 degrees");
    }

    #[test]
    fn dimensions_over_the_pixel_limit_are_refused() {
        assert!(matches!(
            Renderer::new(MAX_PIXELS as u32, 2),
            Err(RenderError::DimensionsTooLarge { .. })
        ));
        assert!(Renderer::new(64, 64).is_ok());
        // A collapsed window is legal and simply draws nothing.
        assert!(Renderer::new(0, 0).is_ok());
    }

    #[test]
    fn render_rejects_a_short_colour_buffer() {
        let mut renderer = Renderer::new(16, 16).expect("small viewport");
        let mut color = vec![0_u32; 16 * 16 - 1];
        let mesh = cube();
        let cam = camera_facing_cube();
        let settings = RenderSettings::default();
        assert!(matches!(
            renderer.render(&mesh, &cam, &mut color, &settings),
            Err(RenderError::ColorBufferTooSmall { .. })
        ));
    }

    #[test]
    fn a_zero_sized_viewport_renders_nothing_without_error() {
        let mut renderer = Renderer::new(0, 0).expect("collapsed viewport");
        let mut color: Vec<u32> = Vec::new();
        let stats = renderer
            .render(
                &cube(),
                &camera_facing_cube(),
                &mut color,
                &RenderSettings::default(),
            )
            .expect("a collapsed viewport is not an error");
        assert_eq!(stats, RenderStats::default());
    }

    /// The cube must actually be drawn: pixels differ from the background and
    /// the depth buffer is populated exactly where they do.
    #[test]
    fn a_framed_cube_covers_pixels_and_fills_depth() {
        let mut renderer = Renderer::new(120, 90).expect("viewport");
        let mut color = vec![0_u32; 120 * 90];
        let settings = RenderSettings::default();
        let stats = renderer
            .render(&cube(), &camera_facing_cube(), &mut color, &settings)
            .expect("render");

        assert_eq!(stats.faces_considered, 12);
        assert_eq!(stats.degenerate_faces, 0);
        assert!(
            stats.fragments_passed > 1000,
            "the cube should cover a large area, got {}",
            stats.fragments_passed
        );
        assert!(stats.triangles_rasterized >= 3);
        assert_coverage_agrees(&renderer, &color, settings.background, stats);
    }

    /// Culling removes triangles, not pixels: on a closed convex surface every
    /// back-facing triangle is behind a front-facing one, so the image must be
    /// identical. Only the *write count* may differ, because an unculled
    /// back-facing fragment is written and then overwritten by the nearer
    /// front-facing one at the same pixel.
    #[test]
    fn culling_backfaces_removes_triangles_but_not_pixels() {
        let mut renderer = Renderer::new(120, 90).expect("viewport");
        let cam = camera_facing_cube();
        let mut culled_color = vec![0_u32; 120 * 90];
        let mut unculled_color = vec![0_u32; 120 * 90];

        let culled = renderer
            .render(&cube(), &cam, &mut culled_color, &RenderSettings::default())
            .expect("render");
        let unculled = renderer
            .render(
                &cube(),
                &cam,
                &mut unculled_color,
                &RenderSettings {
                    cull: CullMode::None,
                    ..RenderSettings::default()
                },
            )
            .expect("render");

        assert_eq!(culled.backface_culled + culled.triangles_rasterized, 12);
        assert_eq!(unculled.backface_culled, 0);
        assert_eq!(unculled.triangles_rasterized, 12);
        assert!(
            unculled.triangles_rasterized > culled.triangles_rasterized,
            "culling should submit fewer triangles"
        );
        assert_eq!(
            culled_color, unculled_color,
            "culling must not change a single pixel"
        );
        // Every pixel the culled pass fills is also filled by the unculled pass,
        // so dropping back faces can never lower the write count. It is not an
        // equality: a back-facing fragment is only *written* when it is drawn
        // before the nearer front-facing one, so the excess depends on face
        // order.
        assert!(
            unculled.fragments_passed >= culled.fragments_passed,
            "unculled writes {} should be at least the culled {}",
            unculled.fragments_passed,
            culled.fragments_passed
        );
    }

    /// A face with three identical positions has no normal and must be skipped
    /// rather than producing a NaN transform.
    #[test]
    fn a_degenerate_face_is_counted_and_skipped() {
        let mut mesh = IndexedMesh::new();
        let v = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        mesh.add_face(v, v, v);
        let mut renderer = Renderer::new(32, 32).expect("viewport");
        let mut color = vec![0_u32; 32 * 32];
        let stats = renderer
            .render(
                &mesh,
                &camera_facing_cube(),
                &mut color,
                &RenderSettings::default(),
            )
            .expect("render");
        assert_eq!(stats.degenerate_faces, 1);
        assert_eq!(stats.triangles_rasterized, 0);
        assert_eq!(stats.fragments_passed, 0);
    }

    /// A camera inside a closed mesh draws nothing while back-face culling is
    /// on: every front-facing face is behind the eye, and every face in front of
    /// the eye is back-facing. Pinned because it is the blank frame a host has
    /// to be able to explain rather than report as a renderer failure, and
    /// because the same scene drawn without culling is not blank.
    #[test]
    fn a_camera_inside_a_closed_mesh_draws_nothing_while_culling() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 0.2);
        cam.orbit(0.6, 0.4);
        cam.set_clip_planes(0.1, 100.0);
        let mut renderer = Renderer::new(120, 90).expect("viewport");
        let mut color = vec![0_u32; 120 * 90];

        let culled = renderer
            .render(&cube(), &cam, &mut color, &RenderSettings::default())
            .expect("render");
        assert_eq!(
            culled.fragments_passed, 0,
            "inside the cube every face in front of the eye is back-facing"
        );

        // Without culling the far interior walls are visible. The walls between
        // the eye and the target are behind the eye, so this is also a
        // near-plane clipping case.
        let unculled = renderer
            .render(
                &cube(),
                &cam,
                &mut color,
                &RenderSettings {
                    cull: CullMode::None,
                    ..RenderSettings::default()
                },
            )
            .expect("render");
        assert!(
            unculled.near_plane_clipped > 0,
            "the walls behind the eye must be clipped"
        );
        assert!(
            unculled.fragments_passed > 0,
            "the far interior walls are in front of the eye"
        );
        assert_coverage_agrees(
            &renderer,
            &color,
            RenderSettings::default().background,
            unculled,
        );
    }

    /// A triangle that straddles the near plane must be clipped, not projected
    /// through the eye. The retained part lands inside the footprint the
    /// unclipped triangle would have covered; a projection through the eye would
    /// instead fling it far outside that footprint.
    #[test]
    fn geometry_crossing_the_near_plane_is_clipped_not_smeared() {
        let mut mesh = IndexedMesh::new();
        // The default camera looks down -x from +x, so eye-space depth is
        // `distance - x`. These depths are 0.5, 0.5 and 1.5, and the face normal
        // (1, 0, 1) still faces the camera, so a near plane at 0.8 cuts the
        // triangle in half rather than removing it.
        let a = mesh.add_vertex_pos(Point3r::new(1.5, -0.5, -0.5));
        let b = mesh.add_vertex_pos(Point3r::new(1.5, 0.5, -0.5));
        let c = mesh.add_vertex_pos(Point3r::new(0.5, -0.5, 0.5));
        mesh.add_face(a, b, c);

        let camera_with_near = |near: Real| {
            let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 2.0);
            cam.set_clip_planes(near, 100.0);
            cam
        };
        let settings = RenderSettings::default();
        let background = settings.background.packed();
        let mut renderer = Renderer::new(120, 90).expect("viewport");

        let mut clipped_color = vec![0_u32; 120 * 90];
        let clipped = renderer
            .render(&mesh, &camera_with_near(0.8), &mut clipped_color, &settings)
            .expect("render");
        assert_eq!(
            clipped.triangles_rasterized, 1,
            "the retained part is one triangle"
        );
        assert_eq!(
            clipped.near_plane_clipped, 0,
            "the triangle straddles the plane, so it is not dropped whole"
        );
        assert!(
            clipped.fragments_passed > 0,
            "the retained part is in front of the near plane"
        );
        assert_coverage_agrees(&renderer, &clipped_color, settings.background, clipped);
        for (index, &pixel) in clipped_color.iter().enumerate() {
            if pixel == background {
                continue;
            }
            assert!(
                renderer.depth()[index].is_finite(),
                "pixel {index} has a non-finite depth"
            );
        }

        // The same scene with a near plane that keeps the whole triangle. The
        // near plane does not enter the x and y projection, so the two renders
        // share a footprint and only the covered area differs.
        let mut whole_color = vec![0_u32; 120 * 90];
        let whole = renderer
            .render(&mesh, &camera_with_near(0.01), &mut whole_color, &settings)
            .expect("render");
        assert_eq!(whole.near_plane_clipped, 0);
        assert_eq!(whole.triangles_rasterized, 1);

        // The retained remnant is a sub-region of the unclipped triangle, so its
        // footprint lies inside the unclipped footprint. Along the shared
        // boundary the two are not pixel-identical: the remnant's edge is a
        // sub-segment of the original edge, so the half-space value for a pixel
        // centre *on* that line rounds differently and the pixel is included by
        // one render and not the other. Requiring every differing pixel to touch
        // the unclipped footprint keeps the assertion meaningful without pinning
        // a rounding decision, and a projection through the eye would instead
        // place pixels far outside it.
        const WIDTH: usize = 120;
        const HEIGHT: usize = 90;
        assert_eq!(clipped_color.len(), WIDTH * HEIGHT);
        let mut clipped_pixels = 0_usize;
        let mut stranded = Vec::new();
        for index in 0..clipped_color.len() {
            if clipped_color[index] == background {
                continue;
            }
            clipped_pixels += 1;
            if whole_color[index] != background {
                continue;
            }
            let (x, y) = (index % WIDTH, index / WIDTH);
            let touches = (y.saturating_sub(1)..=(y + 1).min(HEIGHT - 1)).any(|row| {
                (x.saturating_sub(1)..=(x + 1).min(WIDTH - 1))
                    .any(|column| whole_color[row * WIDTH + column] != background)
            });
            if !touches {
                stranded.push(index);
            }
        }
        assert!(
            stranded.is_empty(),
            "{} clipped pixels are not even adjacent to the unclipped footprint: \
             {stranded:?}",
            stranded.len()
        );
        let whole_pixels = whole_color.iter().filter(|&&p| p != background).count();
        assert!(
            clipped_pixels < whole_pixels,
            "clipping must discard the part behind the near plane: \
             {clipped_pixels} covered against {whole_pixels} unclipped"
        );
    }

    /// Geometry entirely behind the eye has `w < near` at every vertex, so it
    /// must be clipped away rather than projected through the origin.
    #[test]
    fn geometry_behind_the_camera_is_clipped_away() {
        let mut mesh = IndexedMesh::new();
        // The default camera sits at x = +6 looking towards -x, so a quad at
        // x = +10 is behind the eye.
        quad_facing_camera(&mut mesh, 10.0);
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 6.0);
        cam.set_clip_planes(0.1, 100.0);

        let mut renderer = Renderer::new(64, 64).expect("viewport");
        let mut color = vec![0_u32; 64 * 64];
        let stats = renderer
            .render(
                &mesh,
                &cam,
                &mut color,
                &RenderSettings {
                    cull: CullMode::None,
                    ..RenderSettings::default()
                },
            )
            .expect("render");

        assert_eq!(stats.near_plane_clipped, 2, "both faces are behind the eye");
        assert_eq!(stats.triangles_rasterized, 0);
        assert_eq!(stats.fragments_passed, 0);
        assert!(color
            .iter()
            .all(|&p| p == RenderSettings::default().background.packed()));
    }

    /// Geometry outside the frustum but in front of the camera must also draw
    /// nothing, without being mistaken for a clipping failure.
    #[test]
    fn geometry_outside_the_frustum_draws_nothing() {
        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 8.0);
        cam.set_clip_planes(0.1, 100.0);
        // Pan the target far along the camera's up vector so the cube leaves
        // the viewport entirely.
        cam.pan_world(0.0, 10_000.0);
        let mut renderer = Renderer::new(60, 60).expect("viewport");
        let mut color = vec![0_u32; 60 * 60];
        let stats = renderer
            .render(&cube(), &cam, &mut color, &RenderSettings::default())
            .expect("render");
        assert_eq!(stats.fragments_passed, 0);
        assert_eq!(
            stats.near_plane_clipped, 0,
            "the cube is in front of the camera, just off screen"
        );
    }

    /// A camera whose eye rounds to its target has no view direction, and that
    /// must be reported rather than rendered as garbage.
    #[test]
    fn a_camera_with_no_view_direction_is_reported() {
        // At this magnitude a distance of 1e-9 is below the representable
        // offset from the target, so `eye` equals `target` exactly.
        let cam = OrbitCamera::new(Point3r::new(1.0e300, 0.0, 0.0), 1.0e-9);
        assert!(cam.view().is_none(), "the eye rounds onto the target");

        let mut renderer = Renderer::new(8, 8).expect("viewport");
        let mut color = vec![0_u32; 64];
        assert!(matches!(
            renderer.render(&cube(), &cam, &mut color, &RenderSettings::default()),
            Err(RenderError::DegenerateCamera)
        ));
    }

    /// Two coplanar quads at different depths must resolve by depth, not by
    /// submission order: the nearer one wins.
    #[test]
    fn depth_resolves_occlusion_independently_of_submission_order() {
        let mut mesh = IndexedMesh::new();
        quad_facing_camera(&mut mesh, 0.0); // 6 units from the eye
        quad_facing_camera(&mut mesh, -1.0); // 7 units from the eye

        let mut cam = OrbitCamera::new(Point3r::new(0.0, 0.0, 0.0), 6.0);
        cam.set_clip_planes(0.1, 100.0);
        let mut renderer = Renderer::new(40, 40).expect("viewport");
        let mut color = vec![0_u32; 40 * 40];
        renderer
            .render(
                &mesh,
                &cam,
                &mut color,
                &RenderSettings {
                    cull: CullMode::None,
                    ..RenderSettings::default()
                },
            )
            .expect("render");

        // The centre pixel is covered by both quads; the nearer one must win,
        // so its depth key is the larger of 1/6 and 1/7.
        let centre = 20 * 40 + 20;
        let depth = renderer.depth()[centre];
        assert!(
            depth > 1.0 / 6.5 && depth < 1.0 / 5.5,
            "the centre should hold the near quad at 1/6, got {depth}"
        );
    }

    /// A mesh drawn twice must produce identical output: nothing in the
    /// rasteriser may depend on previous frame state beyond the depth buffer,
    /// which is cleared.
    #[test]
    fn rendering_is_deterministic_across_frames() {
        let mut renderer = Renderer::new(64, 48).expect("viewport");
        let cam = camera_facing_cube();
        let settings = RenderSettings::default();
        let mut first = vec![0_u32; 64 * 48];
        let mut second = vec![0_u32; 64 * 48];
        let a = renderer
            .render(&cube(), &cam, &mut first, &settings)
            .expect("render");
        let b = renderer
            .render(&cube(), &cam, &mut second, &settings)
            .expect("render");
        assert_eq!(a, b);
        assert_eq!(first, second);
    }

    /// Resizing must keep the frame consistent: the depth buffer is resized
    /// with the colour buffer, so a stale pixel cannot survive a resize.
    #[test]
    fn resizing_reallocates_depth_and_clears_it() {
        let mut renderer = Renderer::new(32, 32).expect("viewport");
        let mut color = vec![0_u32; 32 * 32];
        renderer
            .render(
                &cube(),
                &camera_facing_cube(),
                &mut color,
                &RenderSettings::default(),
            )
            .expect("render");
        assert!(renderer.depth().iter().any(|&d| d > 0.0));

        renderer.resize(64, 48).expect("resize");
        assert_eq!(renderer.depth().len(), 64 * 48);
        assert!(
            renderer.depth().iter().all(|&d| d == 0.0),
            "a resize must not leave depth from the previous size"
        );

        let mut color = vec![0_u32; 64 * 48];
        let stats = renderer
            .render(
                &cube(),
                &camera_facing_cube(),
                &mut color,
                &RenderSettings::default(),
            )
            .expect("render");
        assert!(stats.fragments_passed > 0);
    }
}
