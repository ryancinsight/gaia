//! Winding Number Normal Consistency scoring for prepared triangle meshes.

use super::gwn::{vertex_offsets, PreparedFace};
use crate::domain::core::scalar::{Point3r, Vector3r};

/// Exact analytical spatial gradient of the generalized winding number.
///
/// For a triangle, let `N = va · (vb × vc)` and
/// `D = |va||vb||vc| + (va·vb)|vc| + (vb·vc)|va| + (vc·va)|vb|`. Then
/// `∇Ω = 2 (D ∇N − N ∇D) / (N² + D²)`, with `∇N = −normal` and
/// `∇D = −(Kₐ va + Kᵦ vb + K𝒸 vc)`, where
/// `Kₐ = (|vb||vc| + vb·vc)/|va| + |vb| + |vc|`, with cyclic definitions for
/// `Kᵦ` and `K𝒸`. The implementation sums this expression over faces and
/// divides by `4π`.
fn gwn_gradient_prepared(query: &Point3r, faces: &[PreparedFace]) -> Vector3r {
    let mut grad_sum = Vector3r::zeros();
    for face in faces {
        let Some((va, vb, vc)) = vertex_offsets(query, face) else {
            continue;
        };
        let la = va.norm();
        let lb = vb.norm();
        let lc = vc.norm();

        let num = va.dot(vb.cross(vc));
        let den = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;

        let den_sq = den * den + num * num;
        if den_sq < 1e-60 {
            continue;
        }

        // `atan2` has a branch cut when the face contains the query in its
        // plane; use the principal-value gradient there instead of a singular
        // derivative across that discontinuity.
        if num.abs() < 1e-12 && den < 0.0 {
            continue;
        }

        let k_a = (lb * lc + vb.dot(vc)) / la + lb + lc;
        let k_b = (la * lc + va.dot(vc)) / lb + la + lc;
        let k_c = (la * lb + va.dot(vb)) / lc + la + lb;

        let grad_n = -face.normal;
        let grad_d = -(va * k_a + vb * k_b + vc * k_c);

        let grad_omega = (grad_n * den - grad_d * num) * 2.0 / den_sq;
        grad_sum += grad_omega;
    }
    grad_sum / (4.0 * std::f64::consts::PI)
}

/// Winding Number Normal Consistency score at a surface point.
///
/// The score is the cosine between `−∇GWN` and `normal`, so positive values
/// indicate codirectional normals and negative values indicate an inverted
/// orientation. The score is defined as zero when either vector has negligible
/// squared length.
///
/// The WNNC property uses the codirection between negative winding-number
/// gradients and consistently oriented surface normals ([Lin et al., 2024](https://arxiv.org/abs/2405.16634)).
/// This function applies that score to a prepared triangle mesh using the
/// analytical solid-angle gradient.
#[must_use]
pub fn wnnc_score(point: &Point3r, normal: &Vector3r, faces: &[PreparedFace]) -> f64 {
    let grad = gwn_gradient_prepared(point, faces);

    let grad_norm_sq = grad.norm_squared();
    let normal_norm_sq = normal.norm_squared();
    if grad_norm_sq < 1e-60 || normal_norm_sq < 1e-60 {
        return 0.0;
    }
    let neg_grad_dot_n = -grad.dot(*normal);
    neg_grad_dot_n / (grad_norm_sq.sqrt() * normal_norm_sq.sqrt())
}
