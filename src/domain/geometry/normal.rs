//! Normal computation for triangles and polygon fans — generic over `T: Scalar`.

use crate::domain::core::scalar::Scalar;
use leto::geometry::{Point3, Vector3};

/// Compute the face normal of a triangle (CCW winding → outward).
///
/// Returns `None` if the triangle is degenerate.
#[inline]
#[must_use]
pub fn triangle_normal<T: Scalar>(
    a: &Point3<T>,
    b: &Point3<T>,
    c: &Point3<T>,
) -> Option<Vector3<T>> {
    let ab = b - a;
    let ac = c - a;
    let cross = ab.cross(ac);
    let len = cross.norm();
    if len < T::tolerance() {
        return None;
    }
    Some(cross / len)
}

/// Area-weighted normal of a triangle (magnitude = 2 × area).
///
/// Cheaper than `triangle_normal` when the caller only needs the direction sign,
/// not the unit normal.
#[inline]
#[must_use]
pub fn triangle_area_normal<T: Scalar>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> Vector3<T> {
    let ab = b - a;
    let ac = c - a;
    ab.cross(ac)
}

/// Centroid (arithmetic mean) of a triangle.
///
/// Canonical SSOT implementation — used by CSG classify, GWN, and channel builders.
/// Avoids the O(1) `to_f64()` cast chain found in duplicate implementations.
#[inline]
#[must_use]
pub fn triangle_centroid<T: Scalar>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> Point3<T> {
    let third = <T as Scalar>::from_f64(1.0 / 3.0);
    Point3::new(
        centroid_component::<T, 0>(a, b, c, third),
        centroid_component::<T, 1>(a, b, c, third),
        centroid_component::<T, 2>(a, b, c, third),
    )
}

#[inline]
fn centroid_component<T: Scalar, const AXIS: usize>(
    a: &Point3<T>,
    b: &Point3<T>,
    c: &Point3<T>,
    third: T,
) -> T {
    let sum = a[AXIS] + b[AXIS] + c[AXIS];
    let center = sum * third;
    if <T as eunomia::NumericElement>::is_finite(sum) {
        center
    } else {
        recover_centroid_component::<T, AXIS>(a, b, c, center, third)
    }
}

#[cold]
#[inline]
fn recover_centroid_component<T: Scalar, const AXIS: usize>(
    a: &Point3<T>,
    b: &Point3<T>,
    c: &Point3<T>,
    center: T,
    third: T,
) -> T {
    if !(finite_point(a) & finite_point(b) & finite_point(c)) {
        return center;
    }
    let (mut a, mut b, mut c) = (a[AXIS], b[AXIS], c[AXIS]);
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }
    if b > c {
        core::mem::swap(&mut b, &mut c);
    }
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }
    let three = T::from_int(3);
    let zero = T::from_int(0);
    if a >= zero || c <= zero {
        // Same-sign endpoints have finite median-relative differences with
        // opposite signs. Recentring also retains an equal-coordinate mean
        // exactly, instead of rounding three individually divided endpoints.
        return b + ((a - b) + (c - b)) / three;
    }
    // Opposite extremes cannot overflow when added.
    let extremes = a + c;
    // Cancel opposite extremes before scaling, preserving small residuals.
    let ordered_sum = extremes + b;
    if <T as eunomia::NumericElement>::is_finite(ordered_sum) {
        ordered_sum * third
    } else {
        // Only two same-sign finite terms remain. Their divided sum is
        // bounded by 2*MAX/3 and neither can be a tiny cancellation term.
        extremes / three + b / three
    }
}

#[inline]
fn finite_point<T: Scalar>(point: &Point3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(point.x)
        & <T as eunomia::NumericElement>::is_finite(point.y)
        & <T as eunomia::NumericElement>::is_finite(point.z)
}

/// Newell's method for computing the normal of a polygon with ≥3 vertices.
///
/// Robust for non-planar polygons and concave cases.
/// Returns `None` if the polygon is degenerate.
#[inline]
pub fn newell_normal<T: Scalar>(vertices: &[Point3<T>]) -> Option<Vector3<T>> {
    if vertices.len() < 3 {
        return None;
    }
    let mut normal = Vector3::<T>::zeros();
    let n = vertices.len();
    for i in 0..n {
        let curr = &vertices[i];
        let next = &vertices[(i + 1) % n];
        normal[0] += (curr.y - next.y) * (curr.z + next.z);
        normal[1] += (curr.z - next.z) * (curr.x + next.x);
        normal[2] += (curr.x - next.x) * (curr.y + next.y);
    }
    if ![normal.x, normal.y, normal.z]
        .into_iter()
        .all(<T as eunomia::NumericElement>::is_finite)
    {
        normal = recover_newell_normal(vertices, normal);
    }
    let len = normal.norm();
    if len < T::tolerance() {
        return None;
    }
    Some(normal / len)
}

#[cold]
#[inline(never)]
fn recover_newell_normal<T: Scalar>(vertices: &[Point3<T>], original: Vector3<T>) -> Vector3<T> {
    if !vertices.iter().all(finite_point) {
        return original;
    }
    // Newell's vector is the closed sum of cross products. Subtracting one
    // origin leaves that sum invariant, equivalently summing an anchored fan.
    // Translation is removed before products; the original loop and its bits
    // remain the fast path. No temporary vertex collection is required.
    let origin = vertices[0];
    let mut previous = vertices[1] - origin;
    let mut normal = Vector3::<T>::zeros();
    for vertex in &vertices[2..] {
        let current = *vertex - origin;
        normal += previous.cross(current);
        previous = current;
    }
    if [normal.x, normal.y, normal.z]
        .into_iter()
        .all(<T as eunomia::NumericElement>::is_finite)
    {
        normal
    } else {
        // Unrepresentable anchored arithmetic has no new rejection policy.
        original
    }
}

/// Compute vertex normal by averaging adjacent face normals.
///
/// Returns normalised average, or `None` if all normals cancel.
#[inline]
pub fn average_normal<'a, T: Scalar + 'a>(
    face_normals: impl Iterator<Item = &'a Vector3<T>>,
) -> Option<Vector3<T>> {
    let mut sum = Vector3::<T>::zeros();
    let mut count = 0usize;
    for n in face_normals {
        sum += *n;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let len = sum.norm();
    if len < T::tolerance() {
        return None;
    }
    Some(sum / len)
}

/// Angle-weighted vertex normal: weight each face normal by the interior angle.
///
/// `faces`: iterator of `(face_normal, angle_at_vertex)` pairs.
#[inline]
pub fn angle_weighted_normal<T: Scalar>(
    faces: impl Iterator<Item = (Vector3<T>, T)>,
) -> Option<Vector3<T>> {
    let mut sum = Vector3::<T>::zeros();
    for (normal, angle) in faces {
        sum += normal * angle;
    }
    let len = sum.norm();
    if len < T::tolerance() {
        return None;
    }
    Some(sum / len)
}
