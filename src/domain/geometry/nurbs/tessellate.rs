//! Curvature-adaptive tessellation of NURBS curves and surfaces.
//!
//! # Surface tessellation algorithm
//!
//! 1. Start with a `min_segments` x `min_segments` coarse parameter grid.
//! 2. For each quad cell evaluate the surface normal at all 4 corners.
//! 3. If the maximum angle between any two corner normals exceeds
//!    `max_angle_deg`, bisect the longer parameter edge and recurse.
//! 4. Collect leaf quads; triangulate each into 2 triangles.
//! 5. Pass every vertex position through `IndexedMesh::add_vertex_pos`, which
//!    uses the built-in `VertexPool` spatial-hash to weld coincident vertices.
//!
//! # Curve tessellation algorithm
//!
//! 1. Start with `min_segments` uniform parameter samples.
//! 2. For each segment evaluate the tangent at both endpoints.
//! 3. If the angle between tangents exceeds `max_angle_deg`, insert the
//!    midpoint and recurse on both halves.
//! 4. Return the ordered list of 3-D sample positions.

use super::curve::NurbsCurve;
use super::surface::NurbsSurface;
use crate::domain::core::scalar::{Real, Scalar};
use crate::domain::mesh::IndexedMesh;
use eunomia::NumericElement;
use leto::geometry::UnitVector3;

// ---------------------------------------------------------------------------
// TessellationOptions
// ---------------------------------------------------------------------------

/// Options controlling curvature-adaptive tessellation.
#[derive(Clone, Debug)]
pub struct TessellationOptions<T = Real> {
    /// Maximum angle (degrees) between adjacent surface/curve normals
    /// before a cell or segment is subdivided. Default: 5.0.
    pub max_angle_deg: T,
    /// Minimum number of segments per parametric direction (>= 1).
    /// The tessellation always produces at least this many divisions
    /// even on flat faces. Default: 4.
    pub min_segments: usize,
    /// Maximum additional recursion depth beyond `min_segments`. Default: 6.
    pub max_depth: usize,
}

impl<T: Scalar> Default for TessellationOptions<T> {
    fn default() -> Self {
        Self {
            max_angle_deg: <T as Scalar>::from_f64(5.0),
            min_segments: 4,
            max_depth: 6,
        }
    }
}

impl<T: Scalar> TessellationOptions<T> {
    /// Create with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum deviation angle in degrees (builder pattern).
    #[must_use]
    pub fn with_max_angle(mut self, deg: T) -> Self {
        self.max_angle_deg = deg;
        self
    }

    /// Set the minimum number of parameter segments (builder pattern).
    #[must_use]
    pub fn with_min_segments(mut self, n: usize) -> Self {
        self.min_segments = n.max(1);
        self
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Angle in degrees between two unit vectors (clamped to [0, 180]).
#[inline]
fn angle_deg<T: Scalar>(a: UnitVector3<T>, b: UnitVector3<T>) -> T {
    let cos_t = a
        .into_inner()
        .dot(b.into_inner())
        .clamp(<T as Scalar>::from_f64(-1.0), <T as Scalar>::from_f64(1.0));
    cos_t.acos().to_degrees()
}

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
            let ua = u0 + (u1 - u0) * <T as Scalar>::from_f64(i as f64 / segs as f64);
            let ub = u0 + (u1 - u0) * <T as Scalar>::from_f64((i + 1) as f64 / segs as f64);
            let va = v0 + (v1 - v0) * <T as Scalar>::from_f64(j as f64 / segs as f64);
            let vb = v0 + (v1 - v0) * <T as Scalar>::from_f64((j + 1) as f64 / segs as f64);
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

/// Adaptively tessellate a 3-D NURBS curve into an ordered polyline.
///
/// The returned points are in order from `t_min` to `t_max`, including both
/// endpoints.  Consecutive segment angle above `opts.max_angle_deg` triggers
/// recursive subdivision of that segment.
///
/// # Example
/// ```rust,ignore
/// use gaia::domain::geometry::nurbs::tessellate::{TessellationOptions, tessellate_curve};
///
/// let pts = tessellate_curve(&my_curve, &TessellationOptions::default());
/// assert!(pts.len() >= 2);
/// ```
#[must_use]
pub fn tessellate_curve<T: Scalar>(
    curve: &NurbsCurve<3, T>,
    opts: &TessellationOptions<T>,
) -> Vec<leto::geometry::Point3<T>> {
    let (t0, t1) = curve.domain();
    let segs = opts.min_segments.max(1);

    // Start with the first point, then adaptively fill in each segment
    let mut result: Vec<leto::geometry::Point3<T>> = Vec::with_capacity(segs * 2 + 1);
    result.push(leto::geometry::Point3::from(curve.point(t0)));

    for i in 0..segs {
        let ta = t0 + (t1 - t0) * <T as Scalar>::from_f64(i as f64 / segs as f64);
        let tb = t0 + (t1 - t0) * <T as Scalar>::from_f64((i + 1) as f64 / segs as f64);
        subdivide_curve_segment(curve, ta, tb, 0, opts, &mut result);
    }
    result
}

// ---------------------------------------------------------------------------
// Curve subdivision helpers
// ---------------------------------------------------------------------------

/// Recursively subdivide a curve segment `[ta, tb]`, appending points up to
/// (but not including) `ta`'s value, and including `tb`'s endpoint.
fn subdivide_curve_segment<T: Scalar>(
    curve: &NurbsCurve<3, T>,
    ta: T,
    tb: T,
    depth: usize,
    opts: &TessellationOptions<T>,
    result: &mut Vec<leto::geometry::Point3<T>>,
) {
    if depth >= opts.max_depth {
        result.push(leto::geometry::Point3::from(curve.point(tb)));
        return;
    }

    // Compute tangent angle between endpoints
    let (_, tan_a) = curve.point_and_tangent(ta);
    let (_, tan_b) = curve.point_and_tangent(tb);

    let need_split = if let (Some(ua), Some(ub)) = (
        UnitVector3::try_new(tan_a, <T as Scalar>::from_f64(1e-15)),
        UnitVector3::try_new(tan_b, <T as Scalar>::from_f64(1e-15)),
    ) {
        angle_deg(ua, ub) > opts.max_angle_deg
    } else {
        // Degenerate tangent — insert midpoint to be safe
        true
    };

    if need_split {
        let tm = (ta + tb) * <T as Scalar>::from_f64(0.5);
        subdivide_curve_segment(curve, ta, tm, depth + 1, opts, result);
        subdivide_curve_segment(curve, tm, tb, depth + 1, opts, result);
    } else {
        result.push(leto::geometry::Point3::from(curve.point(tb)));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::curve::NurbsCurve;
    use super::super::knot::KnotVector;
    use super::super::surface::{ControlGrid, NurbsSurface};
    use super::*;
    use crate::domain::core::scalar::Point3r;
    use leto::geometry::Vector as SVector;

    type V3 = SVector<Real, 3>;
    fn v3(x: Real, y: Real, z: Real) -> V3 {
        V3::new(x, y, z)
    }
    fn pt3(x: Real, y: Real, z: Real) -> Point3r {
        Point3r::new(x, y, z)
    }

    // -- Surface tessellation --

    #[test]
    fn tessellate_flat_surface_produces_triangles() {
        let pts = vec![
            pt3(0.0, 0.0, 0.0),
            pt3(1.0, 0.0, 0.0),
            pt3(0.0, 1.0, 0.0),
            pt3(1.0, 1.0, 0.0),
        ];
        let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
        let opts = TessellationOptions {
            min_segments: 2,
            max_angle_deg: 5.0,
            max_depth: 3,
        };
        let mesh = tessellate_surface(&surf, &opts);
        // 2*2 coarse grid -> 4 quads -> 8 triangles (all flat, no subdivision)
        assert_eq!(
            mesh.face_count(),
            8,
            "flat 2x2 grid should give exactly 8 triangles"
        );
        assert!(mesh.vertex_count() >= 3);
    }

    #[test]
    fn tessellate_flat_surface_vertices_on_plane() {
        let pts = vec![
            pt3(0.0, 0.0, 0.0),
            pt3(2.0, 0.0, 0.0),
            pt3(0.0, 2.0, 0.0),
            pt3(2.0, 2.0, 0.0),
        ];
        let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
        let opts = TessellationOptions {
            min_segments: 4,
            max_angle_deg: 5.0,
            max_depth: 4,
        };
        let mesh = tessellate_surface(&surf, &opts);
        // Every vertex of a flat (z=0) surface should have z == 0
        for i in 0..mesh.vertex_count() {
            let pos = mesh
                .vertices
                .position(crate::domain::core::index::VertexId::new(i as u32));
            assert!(pos.z.abs() < 1e-10, "vertex z={} should be ~0", pos.z);
        }
    }

    #[test]
    fn tessellate_surface_min_segments_respected() {
        let pts = vec![
            pt3(0.0, 0.0, 0.0),
            pt3(1.0, 0.0, 0.0),
            pt3(0.0, 1.0, 0.0),
            pt3(1.0, 1.0, 0.0),
        ];
        let surf = NurbsSurface::clamped(ControlGrid::new(pts, 2, 2), 1, 1).unwrap();
        let opts = TessellationOptions {
            min_segments: 8,
            max_angle_deg: 100.0,
            max_depth: 0,
        };
        // max_angle=100 deg means no subdivision beyond min_segments; depth=0 means stop immediately
        // 8x8 = 64 quads -> 128 triangles
        let mesh = tessellate_surface(&surf, &opts);
        assert_eq!(mesh.face_count(), 128);
    }

    // -- Curve tessellation --

    #[test]
    fn tessellate_line_segment_returns_endpoints() {
        // Linear NURBS from (0,0,0) to (1,1,1)
        let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)];
        let weights = vec![1.0_f64; 2];
        let knots = KnotVector::clamped_uniform(1, 1);
        let curve = NurbsCurve::new(ctrl, weights, knots, 1).unwrap();
        let opts = TessellationOptions {
            min_segments: 4,
            max_angle_deg: 5.0,
            max_depth: 4,
        };
        let pts = tessellate_curve(&curve, &opts);
        assert!(pts.len() >= 2);
        // First point near (0,0,0), last point near (1,1,1)
        assert!((pts[0] - pt3(0.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((pts[pts.len() - 1] - pt3(1.0, 1.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn tessellate_curve_monotone_along_line() {
        // Straight-line NURBS: tessellate should produce monotone x values
        let ctrl = vec![v3(0.0, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
        let weights = vec![1.0_f64; 2];
        let knots = KnotVector::clamped_uniform(1, 1);
        let curve = NurbsCurve::new(ctrl, weights, knots, 1).unwrap();
        let opts = TessellationOptions::default();
        let pts = tessellate_curve(&curve, &opts);
        // x should increase monotonically (or be equal)
        for w in pts.windows(2) {
            assert!(w[1].x >= w[0].x - 1e-12, "x should be non-decreasing");
        }
    }

    #[test]
    fn tessellate_curve_at_least_min_segments_plus_one() {
        let ctrl = vec![v3(0.0, 0.0, 0.0), v3(0.5, 1.0, 0.0), v3(1.0, 0.0, 0.0)];
        let weights = vec![1.0_f64; 3];
        let knots = KnotVector::clamped_uniform(2, 2);
        let curve = NurbsCurve::new(ctrl, weights, knots, 2).unwrap();
        let opts = TessellationOptions {
            min_segments: 8,
            max_angle_deg: 100.0,
            max_depth: 0,
        };
        let pts = tessellate_curve(&curve, &opts);
        // With max_angle=100 and max_depth=0, only min_segments cuts are made
        assert_eq!(
            pts.len(),
            opts.min_segments + 1,
            "should have min_segments+1 points"
        );
    }

    /// The scalar seam monomorphizes: a dyadic flat patch tessellates into an
    /// `IndexedMesh<f32>` with the same topology as the `f64` run, and every
    /// welded vertex coordinate is exactly representable at both precisions.
    #[test]
    fn f32_tessellation_matches_f64_on_dyadic_patch() {
        let corners32 = [
            leto::geometry::Point3::<f32>::new(0.0, 0.0, 0.0),
            leto::geometry::Point3::<f32>::new(1.0, 0.0, 0.0),
            leto::geometry::Point3::<f32>::new(0.0, 1.0, 0.0),
            leto::geometry::Point3::<f32>::new(1.0, 1.0, 0.0),
        ];
        let corners64 = [
            leto::geometry::Point3::<f64>::new(0.0, 0.0, 0.0),
            leto::geometry::Point3::<f64>::new(1.0, 0.0, 0.0),
            leto::geometry::Point3::<f64>::new(0.0, 1.0, 0.0),
            leto::geometry::Point3::<f64>::new(1.0, 1.0, 0.0),
        ];
        let surf32 =
            NurbsSurface::<f32>::clamped(ControlGrid::new(corners32.to_vec(), 2, 2), 1, 1).unwrap();
        let surf64 =
            NurbsSurface::<f64>::clamped(ControlGrid::new(corners64.to_vec(), 2, 2), 1, 1).unwrap();
        let opts32 = TessellationOptions::<f32> {
            min_segments: 2,
            max_angle_deg: <f32 as Scalar>::from_f64(5.0),
            max_depth: 3,
        };
        let opts64 = TessellationOptions::<f64> {
            min_segments: 2,
            max_angle_deg: 5.0,
            max_depth: 3,
        };
        let mesh32 = tessellate_surface(&surf32, &opts32);
        let mesh64 = tessellate_surface(&surf64, &opts64);
        assert_eq!(mesh32.face_count(), mesh64.face_count());
        assert_eq!(mesh32.vertex_count(), mesh64.vertex_count());
        for i in 0..mesh32.vertex_count() {
            let p32 = mesh32
                .vertices
                .position(crate::domain::core::index::VertexId::new(i as u32));
            let p64 = mesh64
                .vertices
                .position(crate::domain::core::index::VertexId::new(i as u32));
            assert_eq!(p32.x.to_bits(), <f32 as Scalar>::from_f64(p64.x).to_bits());
            assert_eq!(p32.y.to_bits(), <f32 as Scalar>::from_f64(p64.y).to_bits());
            assert_eq!(p32.z.to_bits(), <f32 as Scalar>::from_f64(p64.z).to_bits());
        }
    }
}
