//! Scanline fill of a screen-space triangle, testing a depth buffer.

use crate::domain::core::scalar::Real;

use super::super::transform::Vec4;

/// A screen-space vertex: pixel position plus the depth key.
#[derive(Debug, Clone, Copy)]
pub(super) struct ScreenVertex {
    pub(super) x: Real,
    pub(super) y: Real,
    pub(super) inv_w: Real,
}

/// Perspective-divide and map three clip-space vertices to the viewport.
///
/// Returns `None` if any vertex has a non-positive or non-finite `w`, which
/// cannot happen after near-plane clipping but would otherwise divide by zero.
pub(super) fn screen_triangle(
    clip: [Vec4; 3],
    width: u32,
    height: u32,
) -> Option<[ScreenVertex; 3]> {
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
/// what [`RenderStats::fragments_passed`](super::stats::RenderStats::fragments_passed)
/// documents.
pub(super) fn rasterize(
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
