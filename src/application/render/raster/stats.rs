//! Exact per-frame counters, reported rather than logged.

/// What one call to [`Renderer::render`](super::renderer::Renderer::render) did.
///
/// Reported rather than logged, so a host can put it on a status line and a
/// test can assert on it. Every counter is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RenderStats {
    /// Faces examined in the mesh.
    pub faces_considered: usize,
    /// Faces skipped for having no usable world normal.
    pub degenerate_faces: usize,
    /// Faces dropped by [`CullMode::Back`](super::settings::CullMode::Back).
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
    /// centre lands on a shared edge.
    /// [`Renderer::render`](super::renderer::Renderer::render) uses a strict
    /// comparison against the stored depth, so a triangle cannot overwrite a
    /// pixel already written by a coplanar neighbour at exactly equal depth.
    pub fragments_passed: usize,
}
