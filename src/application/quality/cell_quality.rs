//! CFD volume-cell quality metrics.
//!
//! Two metrics that `OpenFOAM` reports for every internal face:
//!
//! | Metric | Formula | Good | Acceptable | Invalid |
//! |--------|---------|------|------------|---------|
//! | Non-orthogonality | ∠(face normal, d) | < 70° | < 85° | ≥ 90° |
//! | Skewness | \|fc - Pi\| / \|fc - `C_owner`\| | < 0.5 | < 0.85 | ≥ 1.0 |
//!
//! where **d** is the owner→neighbour centroid vector, **fc** is the face
//! centre, and **Pi** is the point where **d** intersects the face plane.

use eunomia::{FloatElement, NumericElement};
use leto::geometry::Point3;

use crate::application::quality::metrics::QualityMetric;
use crate::domain::core::index::{FaceId, VertexId};
use crate::domain::core::scalar::{Real, Scalar};
use crate::domain::mesh::IndexedMesh;

// ── Per-face computations ─────────────────────────────────────────────────────

/// Non-orthogonality in degrees: angle between the face normal and the
/// owner→neighbour centroid vector **d**.
///
/// Returns `None` when either cell identifier is not present.
#[expect(
    clippy::many_single_char_names,
    reason = "standard face-vertex, normal, and direction-vector naming"
)]
#[must_use]
pub fn face_non_orthogonality<T: Scalar>(
    face: FaceId,
    owner: usize,
    neighbour: usize,
    mesh: &IndexedMesh<T>,
) -> Option<T> {
    let face_data = mesh.faces.get(face);
    let [va, vb, vc] = face_data.vertices;
    let a = mesh.vertices.position(va);
    let b = mesh.vertices.position(vb);
    let c = mesh.vertices.position(vc);
    let eps = <T as Scalar>::from_f64(1e-30);

    let n = (b - a).cross(c - a);
    if n.norm_squared() < eps {
        return Some(<T as NumericElement>::ZERO);
    }
    let n = n.normalize();

    let c_owner = cell_centroid(owner, mesh)?;
    let c_neigh = cell_centroid(neighbour, mesh)?;
    let d = c_neigh - c_owner;
    if d.norm_squared() < eps {
        return Some(<T as NumericElement>::ZERO);
    }
    let d = d.normalize();

    let cos_theta = n.dot(d).abs().min_scalar(<T as NumericElement>::ONE);
    Some(cos_theta.acos() * <T as Scalar>::from_f64(180.0 / std::f64::consts::PI))
}

/// Skewness: ratio of (face-centre deviation from the owner→neighbour
/// intersection point) to (distance from face centre to owner centroid).
///
/// Returns `None` when either cell identifier is not present.
#[expect(
    clippy::many_single_char_names,
    reason = "standard face-vertex, normal, and line-parameter naming"
)]
#[must_use]
pub fn face_skewness<T: Scalar>(
    face: FaceId,
    owner: usize,
    neighbour: usize,
    mesh: &IndexedMesh<T>,
) -> Option<T> {
    let face_data = mesh.faces.get(face);
    let [va, vb, vc] = face_data.vertices;
    let a = mesh.vertices.position(va);
    let b = mesh.vertices.position(vb);
    let c = mesh.vertices.position(vc);
    let eps = <T as Scalar>::from_f64(1e-30);

    // Face centre = centroid of the triangle.
    let fc = Point3::from((a.coords + b.coords + c.coords) / <T as FloatElement>::from_count(3));

    // Face normal (unnormalised; used for plane intersection).
    let n = (b - a).cross(c - a);
    if n.norm_squared() < eps {
        return Some(<T as NumericElement>::ZERO);
    }

    let c_owner = cell_centroid(owner, mesh)?;
    let c_neigh = cell_centroid(neighbour, mesh)?;

    // Parametric intersection of the d-line with the face plane:
    //   P(t) = c_owner + t * d,  and  n · (P(t) - fc) = 0
    let d = c_neigh - c_owner;
    let denom = n.dot(d);
    if denom.abs() < eps {
        return Some(<T as NumericElement>::ZERO);
    }
    let t = n.dot(fc - c_owner) / denom;
    let p_i = c_owner + d * t; // intersection point on face plane

    let deviation = (fc - p_i).norm();
    let ref_dist = (fc - c_owner).norm();
    if ref_dist < eps {
        return Some(<T as NumericElement>::ZERO);
    }
    Some(deviation / ref_dist)
}

// ── Report ────────────────────────────────────────────────────────────────────

/// Summary of volume-cell quality for an entire [`IndexedMesh`].
#[derive(Clone, Debug)]
pub struct CellQualityReport {
    /// Non-orthogonality statistics (degrees) over all internal faces.
    pub non_orthogonality: QualityMetric,
    /// Skewness statistics over all internal faces.
    pub skewness: QualityMetric,
    /// Number of internal faces with non-orthogonality > 70°.
    pub high_non_orthogonality_count: usize,
    /// Number of internal faces with skewness > 0.85.
    pub high_skewness_count: usize,
    /// Total internal faces evaluated.
    pub internal_face_count: usize,
}

/// Compute cell quality metrics for all internal faces.
///
/// Returns `None` when the mesh has no volumetric cells or no internal faces.
#[must_use]
pub fn cell_quality_report<T: Scalar>(mesh: &IndexedMesh<T>) -> Option<CellQualityReport> {
    if mesh.cell_count() == 0 {
        return None;
    }

    // Build face → (owner, optional neighbour) map.
    // hashbrown::HashMap is used for lower per-lookup overhead vs std HashMap.
    let mut face_owner: hashbrown::HashMap<FaceId, usize> =
        hashbrown::HashMap::with_capacity(mesh.face_count());
    let mut face_neighbour: hashbrown::HashMap<FaceId, usize> =
        hashbrown::HashMap::with_capacity(mesh.face_count() / 2);

    for (cell_id, cell) in mesh.cells().iter().enumerate() {
        for &fi in &cell.faces {
            let fi = FaceId::from_usize(fi);
            if let hashbrown::hash_map::Entry::Vacant(e) = face_owner.entry(fi) {
                e.insert(cell_id);
            } else {
                face_neighbour.insert(fi, cell_id);
            }
        }
    }

    let mut non_orth_vals: Vec<Real> = Vec::with_capacity(face_owner.len());
    let mut skew_vals: Vec<Real> = Vec::with_capacity(face_owner.len());

    for (&fi, &owner) in &face_owner {
        let Some(&neighbour) = face_neighbour.get(&fi) else {
            continue;
        };
        let (Some(non_orthogonality), Some(skewness)) = (
            face_non_orthogonality(fi, owner, neighbour, mesh),
            face_skewness(fi, owner, neighbour, mesh),
        ) else {
            continue;
        };
        non_orth_vals.push(<T as NumericElement>::to_f64(non_orthogonality));
        skew_vals.push(<T as NumericElement>::to_f64(skewness));
    }

    let non_orthogonality = QualityMetric::from_values(&non_orth_vals)?;
    let skewness = QualityMetric::from_values(&skew_vals)?;
    let high_no = non_orth_vals.iter().filter(|&&v| v > 70.0).count();
    let high_sk = skew_vals.iter().filter(|&&v| v > 0.85).count();

    Some(CellQualityReport {
        non_orthogonality,
        skewness,
        high_non_orthogonality_count: high_no,
        high_skewness_count: high_sk,
        internal_face_count: non_orth_vals.len(),
    })
}

// ── Helper ────────────────────────────────────────────────────────────────────

/// Centroid of a cell: arithmetic mean of its vertices.
#[must_use]
pub fn cell_centroid<T: Scalar>(cell_id: usize, mesh: &IndexedMesh<T>) -> Option<Point3<T>> {
    let cell = mesh.cells().get(cell_id)?;

    // Prefer vertex_ids if populated; fall back to face-vertex union.
    if !cell.vertex_ids.is_empty() {
        let sum: leto::geometry::Vector3<T> = cell
            .vertex_ids
            .iter()
            .map(|&vi| mesh.vertices.position(VertexId::from_usize(vi)).coords)
            .fold(leto::geometry::Vector3::zeros(), |sum, position| {
                sum + position
            });
        return Some(Point3::from(
            sum / <T as FloatElement>::from_count(cell.vertex_ids.len()),
        ));
    }

    let mut sum = leto::geometry::Vector3::<T>::zeros();
    let mut count = 0usize;
    // hashbrown::HashSet gives O(1) amortised membership test vs O(n) Vec::contains.
    let mut seen: hashbrown::HashSet<_> = hashbrown::HashSet::with_capacity(cell.faces.len() * 3);
    for &fi in &cell.faces {
        for &vi in &mesh.faces.get(FaceId::from_usize(fi)).vertices {
            if seen.insert(vi) {
                sum += mesh.vertices.position(vi).coords;
                count += 1;
            }
        }
    }
    if count == 0 {
        None
    } else {
        Some(Point3::from(sum / <T as FloatElement>::from_count(count)))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::StructuredGridBuilder;

    #[test]
    fn cell_quality_report_none_for_surface_mesh() {
        let mesh = IndexedMesh::<f64>::new();
        assert!(cell_quality_report(&mesh).is_none());
    }

    #[test]
    fn regular_grid_non_orthogonality_is_finite() {
        // The 5-tet-per-hex decomposition creates diagonal faces; non-orthogonality
        // can be up to 90° on such faces.  This test verifies the metric is
        // computed correctly (finite, in range [0°, 90°]).
        let mesh = StructuredGridBuilder::new(2, 2, 2).build().unwrap();
        let report = cell_quality_report(&mesh).expect("volume mesh should have report");
        assert!(report.non_orthogonality.max.is_finite());
        assert!(report.non_orthogonality.min >= 0.0);
        assert!(report.non_orthogonality.max <= 90.0 + 1e-9);
        assert!(report.skewness.min >= 0.0);
    }

    #[test]
    fn cell_quality_report_has_internal_faces() {
        let mesh = StructuredGridBuilder::new(2, 2, 2).build().unwrap();
        let report = cell_quality_report(&mesh).unwrap();
        assert!(
            report.internal_face_count > 0,
            "2×2×2 grid should have internal faces"
        );
    }
}
