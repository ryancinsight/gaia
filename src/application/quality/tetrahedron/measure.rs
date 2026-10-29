//! Native-precision tetrahedral measurement kernels and mesh reports.

use leto::geometry::{Point3, Vector3};

use crate::application::quality::metrics::QualityMetric;
use crate::domain::core::scalar::Scalar;
use crate::domain::mesh::IndexedMesh;
use crate::domain::topology::{Cell, ElementType};

use super::types::{TetrahedralQualityReport, TetrahedronQuality};

const EDGES: [(usize, usize, usize, usize); 6] = [
    (0, 1, 2, 3),
    (0, 2, 1, 3),
    (0, 3, 1, 2),
    (1, 2, 0, 3),
    (1, 3, 0, 2),
    (2, 3, 0, 1),
];

pub(super) fn tetrahedral_measurements<T: Scalar>(
    mesh: &IndexedMesh<T>,
) -> impl Iterator<Item = Option<TetrahedronQuality<T>>> + '_ {
    mesh.cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .map(|cell| cell_tetrahedron_quality(mesh, cell))
}

pub(crate) fn cell_tetrahedron_quality<T: Scalar>(
    mesh: &IndexedMesh<T>,
    cell: &Cell,
) -> Option<TetrahedronQuality<T>> {
    if cell.element_type != ElementType::Tetrahedron {
        return None;
    }
    let vertex_ids: [usize; 4] = cell.vertex_ids.as_slice().try_into().ok()?;
    if vertex_ids.iter().any(|&id| id >= mesh.vertices.len()) {
        return None;
    }
    let points = vertex_ids.map(|id| {
        *mesh
            .vertices
            .position(crate::domain::core::index::VertexId::from_usize(id))
    });
    tetrahedron_quality(points)
}

#[inline]
fn finite_point<T: Scalar>(point: &Point3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(point.x)
        && <T as eunomia::NumericElement>::is_finite(point.y)
        && <T as eunomia::NumericElement>::is_finite(point.z)
}

#[inline]
fn finite_vector<T: Scalar>(vector: &Vector3<T>) -> bool {
    <T as eunomia::NumericElement>::is_finite(vector.x)
        && <T as eunomia::NumericElement>::is_finite(vector.y)
        && <T as eunomia::NumericElement>::is_finite(vector.z)
}

/// Measure one tetrahedron in native scalar precision.
///
/// `None` means that a coordinate is non-finite, the tetrahedron is
/// degenerate, or a derived metric is non-finite.  The returned angle is the
/// interior dihedral angle at the worst edge; lower values indicate a sliver.
#[expect(
    clippy::many_single_char_names,
    reason = "standard tetrahedron vertex and basis-vector naming"
)]
#[must_use]
pub fn tetrahedron_quality<T: Scalar>(points: [Point3<T>; 4]) -> Option<TetrahedronQuality<T>> {
    if points.iter().any(|point| !finite_point(point)) {
        return None;
    }

    let [a, b, c, d] = points;
    let u = b - a;
    let v = c - a;
    let w = d - a;
    let signed_six_volume = u.dot(v.cross(w));
    let zero = <T as eunomia::NumericElement>::ZERO;
    if !<T as eunomia::NumericElement>::is_finite(signed_six_volume) || signed_six_volume == zero {
        return None;
    }

    let six = T::from_int(6);
    let volume = signed_six_volume.abs() / six;
    let two = <T as eunomia::NumericElement>::ONE + <T as eunomia::NumericElement>::ONE;
    let center_numerator = v.cross(w) * u.norm_squared()
        + w.cross(u) * v.norm_squared()
        + u.cross(v) * w.norm_squared();
    let circumcenter = center_numerator / (signed_six_volume * two);
    let radius_squared = circumcenter.norm_squared();
    if !finite_vector(&center_numerator)
        || !finite_vector(&circumcenter)
        || !<T as eunomia::NumericElement>::is_finite(radius_squared)
        || radius_squared <= zero
    {
        return None;
    }

    let mut edge_squared = [zero; 6];
    let mut shortest_edge_squared = <T as eunomia::NumericElement>::INFINITY;
    for (index, &(i, j, _, _)) in EDGES.iter().enumerate() {
        let edge = points[j] - points[i];
        let squared = edge.norm_squared();
        if !<T as eunomia::NumericElement>::is_finite(squared) || squared <= zero {
            return None;
        }
        edge_squared[index] = squared;
        shortest_edge_squared = shortest_edge_squared.min_scalar(squared);
    }

    let radius_edge_ratio = radius_squared.sqrt() / shortest_edge_squared.sqrt();
    let mut min_dihedral_angle = <T as eunomia::NumericElement>::INFINITY;
    for &(i, j, k, l) in &EDGES {
        let edge = points[j] - points[i];
        let edge_squared = edge.norm_squared();
        let face_k = points[k] - points[i];
        let face_l = points[l] - points[i];
        let projected_k = face_k - edge * (face_k.dot(edge) / edge_squared);
        let projected_l = face_l - edge * (face_l.dot(edge) / edge_squared);
        let product = projected_k.norm_squared() * projected_l.norm_squared();
        if product <= zero || !<T as eunomia::NumericElement>::is_finite(product) {
            return None;
        }
        let cosine = (projected_k.dot(projected_l) / product.sqrt())
            .max_scalar(-<T as eunomia::NumericElement>::ONE)
            .min_scalar(<T as eunomia::NumericElement>::ONE);
        min_dihedral_angle = min_dihedral_angle.min_scalar(cosine.acos());
    }

    let edge_sum = edge_squared
        .iter()
        .copied()
        .fold(zero, |sum, squared| sum + squared);
    let edge_mean_squared = edge_sum / six;
    let edge_rms = edge_mean_squared.sqrt();
    let normalized_volume =
        ((six + six) * volume) / (T::from_int(2).sqrt() * edge_rms * edge_mean_squared);
    if !<T as eunomia::NumericElement>::is_finite(radius_edge_ratio)
        || !<T as eunomia::NumericElement>::is_finite(min_dihedral_angle)
        || !<T as eunomia::NumericElement>::is_finite(normalized_volume)
    {
        return None;
    }

    Some(TetrahedronQuality {
        volume,
        radius_edge_ratio,
        min_dihedral_angle,
        normalized_volume,
    })
}

/// Summarize native-precision quality metrics for tetrahedral cells.
///
/// Non-tetrahedral cells are ignored.  A malformed or degenerate tetrahedron
/// increments `invalid_cell_count`; it is not converted into a default metric.
/// `None` is returned when the mesh contains no tetrahedral cells.
#[must_use]
pub fn tetrahedral_quality_report<T: Scalar>(
    mesh: &IndexedMesh<T>,
) -> Option<TetrahedralQualityReport> {
    let tetrahedral_cell_count = mesh
        .cells
        .iter()
        .filter(|cell| cell.element_type == ElementType::Tetrahedron)
        .count();
    if tetrahedral_cell_count == 0 {
        return None;
    }

    let mut volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut radius_edge_ratios = Vec::with_capacity(tetrahedral_cell_count);
    let mut min_dihedral_angles = Vec::with_capacity(tetrahedral_cell_count);
    let mut normalized_volumes = Vec::with_capacity(tetrahedral_cell_count);
    let mut invalid_cell_count = 0;

    for measurement in tetrahedral_measurements(mesh) {
        let Some(quality) = measurement else {
            invalid_cell_count += 1;
            continue;
        };
        volumes.push(quality.volume);
        radius_edge_ratios.push(quality.radius_edge_ratio);
        min_dihedral_angles.push(quality.min_dihedral_angle);
        normalized_volumes.push(quality.normalized_volume);
    }

    Some(TetrahedralQualityReport {
        volume: QualityMetric::from_scalar_values(&volumes),
        radius_edge_ratio: QualityMetric::from_scalar_values(&radius_edge_ratios),
        min_dihedral_angle: QualityMetric::from_scalar_values(&min_dihedral_angles),
        normalized_volume: QualityMetric::from_scalar_values(&normalized_volumes),
        valid_cell_count: volumes.len(),
        invalid_cell_count,
    })
}
