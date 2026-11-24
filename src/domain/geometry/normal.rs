//! Normal computation for triangles and polygon fans — generic over `T: Scalar`.

use crate::domain::core::scalar::Scalar;
use leto::geometry::{Point3, Vector3};

mod expansion;
use expansion::ExactSum;

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
///
/// # Contract
///
/// Let `u` be the unit roundoff of `T` (`2^-24` for `f32`, `2^-53` for `f64`;
/// IEEE 754 round-to-nearest) and let `mean` be the real-number mean of the
/// three stored coordinates. Per coordinate:
///
/// - When the raw sum `(a + b) + c` is finite, the result is that sum times
///   the rounded `1/3`, bit for bit as before. No accuracy claim is added.
/// - When the raw sum overflows and every coordinate of all three vertices is
///   finite, the result is finite instead of infinity and obeys the bound
///   below. The tests also check it stays within the range of the inputs.
/// - When any input vertex is nonfinite, every coordinate keeps the legacy
///   result, including NaN and infinity propagation.
///
/// On the overflow path, three equal coordinates give that coordinate exactly;
/// on the finite-sum path the legacy `(3x) * (1/3)` rounding applies and may
/// differ from `x` by one ulp. No result is claimed to be the correctly rounded
/// mean: the bounds below apply, and some results are one ulp from it.
///
/// # Recovery error bound
///
/// Sort the coordinate as `lo <= m <= hi`. Each operation `x op y` is modelled
/// as `(x op y)(1 + d)` with `|d| <= u` for correctly rounded IEEE 754
/// operations that neither overflow nor underflow; terms of order `u^2` are
/// dropped. No error-free transformation is used here.
///
/// Size of the mean. Let `p`, `q`, `r` be the coordinates in addition order and
/// `ulp` the spacing just below `MAX`. Round-to-nearest overflows only when
/// the exact result has magnitude at least `MAX + ulp / 2`. If `p + q`
/// overflows, `|p + q| >= MAX + ulp / 2` and `|r| <= MAX`, so
/// `|p + q + r| >= ulp / 2`. Otherwise `t = fl(p + q)` is finite, the sum
/// `t + r` overflows, and `|t - (p + q)| <= ulp / 2`, so
/// `|p + q + r| >= MAX`. Hence `|mean| >= ulp / 6` whenever the sum overflows
/// (above `2^101` for `f32`, `2^968` for `f64`), so half a smallest subnormal
/// of underflow error is negligible against the bounds below.
///
/// - Same sign (`lo >= 0` or `hi <= 0`): the result is
///   `m + ((lo - m) + (hi - m)) / 3`. The two differences have opposite signs,
///   so neither they nor their sum overflow. Their rounding errors, the sum
///   and the division by 3 each contribute at most `u (hi - lo) / 3`
///   (`|lo - m| + |hi - m| = hi - lo`), and the final addition adds
///   `u |mean|`. For same-sign coordinates `hi - lo <= max(|lo|, |hi|) <=
///   3 |mean|`, giving `|result - mean| <= 4u |mean|`. Equal coordinates give
///   zero differences and the exact result `m`.
/// - Mixed signs (`lo < 0 < hi`): `lo + hi` cannot overflow, so it is formed
///   first and cancels large opposite coordinates before scaling. If adding
///   `m` stays finite the result is that sum times the rounded `1/3`, with
///   `|result - mean| <= u (|lo + hi| / 3 + 3 |mean|)`. Otherwise `lo + hi` and
///   `m` share a sign and `(lo + hi) / 3 + m / 3` gives
///   `|result - mean| <= 3u |mean|`.
///
/// The cold path is entered only after the plain sum is nonfinite, so ordinary
/// inputs pay one finiteness test per coordinate.
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
///
/// # Contract
///
/// - Finite accumulated components give the legacy result bit for bit,
///   rounding error included: a distant polygon whose sums stay finite is
///   not corrected.
/// - If a component is nonfinite and every vertex is finite, a cold path
///   recomputes all three. Subtracting one point from every vertex leaves
///   the closed sum of cross products unchanged, so the path sums the fan
///   anchored at the first vertex. Each difference from that vertex is an
///   unevaluated pair (Two-Diff, section 2.3), each product a value plus its
///   fused-multiply residual, and each sum a floating expansion
///   (Grow-Expansion, section 2.4 Theorem 10, over Knuth's Two-Sum, Theorem
///   7). While no difference, product or partial sum overflows and no
///   product residual underflows, the expansion equals the real-number fan
///   sum of the stored coordinates, and an expansion with no components is
///   exactly zero. Only the collapse to one value rounds; no accuracy bound
///   is claimed for it.
/// - If a vertex, or any value computed on the cold path, is nonfinite, the
///   legacy components are kept, so NaN and infinity propagate as before.
///
/// The degeneracy test and normalisation are unchanged and apply to
/// whichever vector results.
///
/// Sections and theorems are those of J. R. Shewchuk, "Adaptive Precision
/// Floating-Point Arithmetic and Fast Robust Geometric Predicates", Discrete
/// & Computational Geometry 18(3):305-363, 1997 (report CMU-CS-96-140R).
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
    // The fan anchored at the first vertex; see the `newell_normal` contract.
    let origin = &vertices[0];
    let mut previous = compensated_difference(&vertices[1], origin);
    let mut sums: [ExactSum<T, 8>; 3] = core::array::from_fn(|_| ExactSum::new());
    for vertex in &vertices[2..] {
        let current = compensated_difference(vertex, origin);
        // Keep both signed products of each determinant separate. Reducing
        // cross() first would discard its subtraction residual before summing.
        for (u, v) in [
            (previous.0, current.0),
            (previous.0, current.1),
            (previous.1, current.0),
            (previous.1, current.1),
        ] {
            for (axis, sum) in sums.iter_mut().enumerate() {
                let first = (axis + 1) % 3;
                let second = (axis + 2) % 3;
                if !sum.add_product(u[first], v[second]) || !sum.add_product(-u[second], v[first]) {
                    return original;
                }
            }
        }
        previous = current;
    }
    let normal = Vector3::new(sums[0].estimate(), sums[1].estimate(), sums[2].estimate());
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

// Two-Diff: `low` is the exact rounding error of a finite `high`. A nonfinite
// `high` makes a nonfinite product, which `ExactSum::add_product` refuses.
#[inline]
fn compensated_difference<T: Scalar>(
    point: &Point3<T>,
    origin: &Point3<T>,
) -> (Vector3<T>, Vector3<T>) {
    let high = *point - *origin;
    let residual = |a: T, b: T, difference: T| {
        let virtual_b = a - difference;
        (a - (difference + virtual_b)) + (virtual_b - b)
    };
    let low = Vector3::new(
        residual(point.x, origin.x, high.x),
        residual(point.y, origin.y, high.y),
        residual(point.z, origin.z, high.z),
    );
    (high, low)
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
