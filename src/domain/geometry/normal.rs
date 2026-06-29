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
    let cross = ab.cross(&ac);
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
    ab.cross(&ac)
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
        (a.x + b.x + c.x) * third,
        (a.y + b.y + c.y) * third,
        (a.z + b.z + c.z) * third,
    )
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
    let len = normal.norm();
    if len < T::tolerance() {
        return None;
    }
    Some(normal / len)
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
        sum += n;
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
