//! Near-plane clipping, the stage that keeps the perspective divide safe.

use crate::domain::core::scalar::Real;

use super::super::transform::{lerp4, Vec4};

/// Clip a triangle against the near plane `w >= near` in clip space.
///
/// Sutherland-Hodgman over a single plane. A triangle clipped by one plane has
/// at most four sides, so the output is a fixed array plus a count and this
/// allocates nothing.
pub(super) fn clip_against_near_plane(triangle: &[Vec4; 3], near: Real) -> ([Vec4; 4], usize) {
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

#[cfg(test)]
mod tests {
    use crate::domain::core::scalar::{Point3r, Real};
    use crate::domain::mesh::indexed::IndexedMesh;

    use super::super::super::camera::OrbitCamera;
    use super::super::renderer::Renderer;
    use super::super::settings::RenderSettings;
    use super::super::tests_support::assert_coverage_agrees;

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
}
