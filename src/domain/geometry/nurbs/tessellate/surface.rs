use super::super::parameter::uniform_parameter;
use super::super::surface::NurbsSurface;
use super::{angle_deg, TessellationOptions};
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use eunomia::NumericElement;
use leto::geometry::UnitVector3;

/// Maximum normal deviation (degrees) at the four corners of a parameter quad.
/// Returns 0.0 if fewer than 2 normals could be computed (degenerate surface).
fn quad_max_angle_deg<T: Scalar>(surf: &NurbsSurface<T>, u0: T, v0: T, u1: T, v1: T) -> T {
    let corners = [(u0, v0), (u1, v0), (u0, v1), (u1, v1)];
    let mut normals = [UnitVector3::new_normalize(leto::geometry::Vector3::z()); 4];
    let mut count = 0;
    for &(u, v) in &corners {
        if let Some(n) = surf.normal(u, v) {
            normals[count] = n;
            count += 1;
        }
    }

    let mut max_a: T = <T as NumericElement>::ZERO;
    for i in 0..count {
        for j in (i + 1)..count {
            let a = angle_deg(normals[i], normals[j]);
            if a > max_a {
                max_a = a;
            }
        }
    }
    max_a
}

/// Recursively subdivide a parameter quad until the normal deviation falls
/// below the threshold or the maximum depth is reached.
/// Appends leaf quads `(u0, v0, u1, v1)` to `leaves`.
fn subdivide_quad<T: Scalar>(
    surf: &NurbsSurface<T>,
    u0: T,
    v0: T,
    u1: T,
    v1: T,
    depth: usize,
    opts: &TessellationOptions<T>,
    leaves: &mut Vec<(T, T, T, T)>,
) {
    if depth >= opts.max_depth {
        leaves.push((u0, v0, u1, v1));
        return;
    }
    if quad_max_angle_deg(surf, u0, v0, u1, v1) <= opts.max_angle_deg {
        leaves.push((u0, v0, u1, v1));
        return;
    }
    // Bisect the longer parametric edge
    let du = u1 - u0;
    let dv = v1 - v0;
    let half = <T as Scalar>::from_f64(0.5);
    if du >= dv {
        let um = (u0 + u1) * half;
        subdivide_quad(surf, u0, v0, um, v1, depth + 1, opts, leaves);
        subdivide_quad(surf, um, v0, u1, v1, depth + 1, opts, leaves);
    } else {
        let vm = (v0 + v1) * half;
        subdivide_quad(surf, u0, v0, u1, vm, depth + 1, opts, leaves);
        subdivide_quad(surf, u0, vm, u1, v1, depth + 1, opts, leaves);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Adaptively tessellate a NURBS surface into an `IndexedMesh`.
///
/// Vertices shared between adjacent triangles are automatically welded by
/// the `VertexPool` inside `IndexedMesh`.
///
/// # Example
/// ```rust,ignore
/// use gaia::domain::geometry::nurbs::surface::NurbsSurface;
/// use gaia::domain::geometry::nurbs::tessellate::{TessellationOptions, tessellate_surface};
///
/// let opts = TessellationOptions::new().with_max_angle(2.0).with_min_segments(8);
/// let mesh = tessellate_surface(&my_surf, &opts);
/// assert!(mesh.face_count() > 0);
/// ```
#[must_use]
pub fn tessellate_surface<T: Scalar>(
    surf: &NurbsSurface<T>,
    opts: &TessellationOptions<T>,
) -> IndexedMesh<T> {
    let ((u0, u1), (v0, v1)) = surf.domain();
    let segs = opts.min_segments.max(1);

    // Collect all leaf quads via adaptive subdivision
    let mut leaves: Vec<(T, T, T, T)> = Vec::with_capacity(segs * segs);
    for i in 0..segs {
        for j in 0..segs {
            let ua = uniform_parameter(u0, u1, i, segs);
            let ub = uniform_parameter(u0, u1, i + 1, segs);
            let va = uniform_parameter(v0, v1, j, segs);
            let vb = uniform_parameter(v0, v1, j + 1, segs);
            subdivide_quad(surf, ua, va, ub, vb, 0, opts, &mut leaves);
        }
    }

    // Triangulate leaf quads; VertexPool welds shared corners automatically
    let mut mesh = IndexedMesh::<T>::new();
    for (qu0, qv0, qu1, qv1) in &leaves {
        let p00 = surf.point(*qu0, *qv0);
        let p10 = surf.point(*qu1, *qv0);
        let p01 = surf.point(*qu0, *qv1);
        let p11 = surf.point(*qu1, *qv1);

        let v00 = mesh.add_vertex_pos(p00);
        let v10 = mesh.add_vertex_pos(p10);
        let v01 = mesh.add_vertex_pos(p01);
        let v11 = mesh.add_vertex_pos(p11);

        // Two counter-clockwise triangles per quad
        mesh.add_face(v00, v10, v01);
        mesh.add_face(v10, v11, v01);
    }

    mesh
}

// ---------------------------------------------------------------------------
