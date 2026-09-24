//! # B-Spline Basis Function Evaluation
//!
//! Implements the **Cox–de Boor recursion** for computing B-spline basis
//! functions `N_{i,p}(ξ)` and their first derivatives `N'_{i,p}(ξ)`.
//!
//! ## Algorithm — triangular basis evaluation
//!
//! Rather than recursively recomputing shared sub-expressions, this module
//! uses the triangular recurrence in Piegl and Tiller, Algorithm A2.2
//! (Chapter 2, §2.5, p. 70):
//!
//! ```text
//! N[0] = 1
//! for j = 1..=p:
//!   left[j] = t - U[i+1-j]
//!   right[j] = U[i+j] - t
//!   saved = 0
//!   for r = 0..j:
//!     current = N[r]
//!     denominator = right[r+1] + left[j-r]
//!     N[r] = saved + current * (right[r+1] / denominator)
//!     saved = current * (left[j-r] / denominator)
//!   N[j] = saved
//! ```
//!
//! This yields the `p+1` non-zero basis functions `N_{i-p,p}(ξ)` through
//! `N_{i,p}(ξ)` in O(p²) time. A zero knot-difference denominator contributes
//! zero, as in the Cox–de Boor convention (Piegl and Tiller, §2.2, Eq. 2.10).
//! Ratios are formed before multiplying by the current basis value so a
//! finite ratio does not overflow through an intermediate reciprocal.
//!
//! Reference: [Piegl and Tiller, *The NURBS Book*, 2nd ed.](https://link.springer.com/book/10.1007/978-3-642-59223-2),
//! Chapter 2, §§2.2 and 2.5, pp. 47–79.
//!
//! ## Theorem — Partition of Unity
//!
//! For any valid knot span and any `ξ` in `[ξᵢ, ξᵢ₊₁)`:
//!
//! ```text
//! Σⱼ₌₀ᵖ N_{i-p+j, p}(ξ) = 1
//! ```
//!
//! The recurrence preserves this identity in exact arithmetic; floating-point
//! evaluations incur rounding error.

use super::knot::KnotVector;
use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;

const STACK_DEGREE_LIMIT: usize = 8;
const STACK_WORK_LEN: usize = STACK_DEGREE_LIMIT + 1;

#[inline]
fn assert_output_len(name: &str, len: usize, required: usize) {
    assert!(
        len >= required,
        "invariant: {name} length must be at least degree + 1; got {len}, required {required}"
    );
}

/// Divide, mapping an exactly zero denominator (repeated knots) to zero.
///
/// No absolute small-value threshold is applied. For repeated knots, an exact
/// zero denominator contributes zero. See Piegl and Tiller, *The NURBS Book*,
/// 2nd ed., Chapter 2, §2.2, Eq. 2.10.
#[inline]
fn safe_div<T: Scalar>(num: T, denom: T) -> T {
    if denom == <T as NumericElement>::ZERO {
        <T as NumericElement>::ZERO
    } else {
        num / denom
    }
}

fn fill_basis<T: Scalar>(
    span: usize,
    t: T,
    p: usize,
    knots: &KnotVector<T>,
    out: &mut [T],
    left: &mut [T],
    right: &mut [T],
) {
    debug_assert!(out.len() > p);
    debug_assert!(left.len() > p);
    debug_assert!(right.len() > p);

    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    out[0] = one;
    for j in 1..=p {
        left[j] = t - knots.get(span + 1 - j);
        right[j] = knots.get(span + j) - t;
        let mut saved = zero;
        for r in 0..j {
            let current = out[r];
            let denominator = right[r + 1] + left[j - r];
            let right_ratio = safe_div(right[r + 1], denominator);
            let left_ratio = safe_div(left[j - r], denominator);
            out[r] = saved + current * right_ratio;
            saved = current * left_ratio;
        }
        out[j] = saved;
    }
}

/// Evaluate the `p+1` non-zero B-spline basis functions into `out`.
///
/// The caller owns the output storage, so degree-8-or-smaller callers can use
/// stack buffers and execute without heap allocation. Higher degrees allocate
/// only the internal Cox-de Boor work buffers.
///
/// # Panics
///
/// Panics if `out.len() < p + 1`, or if `span`, `p`, and `knots` do not
/// describe a valid active knot span.
pub fn eval_basis_to_slice<T: Scalar>(
    span: usize,
    t: T,
    p: usize,
    knots: &KnotVector<T>,
    out: &mut [T],
) {
    let required = p + 1;
    assert_output_len("basis output", out.len(), required);

    if p <= STACK_DEGREE_LIMIT {
        let mut left = [<T as NumericElement>::ZERO; STACK_WORK_LEN];
        let mut right = [<T as NumericElement>::ZERO; STACK_WORK_LEN];
        fill_basis(
            span,
            t,
            p,
            knots,
            out,
            &mut left[..required],
            &mut right[..required],
        );
    } else {
        let mut left = vec![<T as NumericElement>::ZERO; required];
        let mut right = vec![<T as NumericElement>::ZERO; required];
        fill_basis(span, t, p, knots, out, &mut left, &mut right);
    }
}

/// Evaluate the `p+1` non-zero B-spline basis functions at parameter `t`.
///
/// Returns a heap-allocated vector. Use [`eval_basis_to_slice`] to avoid allocation.
#[must_use]
pub fn eval_basis<T: Scalar>(span: usize, t: T, p: usize, knots: &KnotVector<T>) -> Vec<T> {
    let mut n = vec![<T as NumericElement>::ZERO; p + 1];
    eval_basis_to_slice(span, t, p, knots, &mut n);
    n
}

/// Evaluate all non-zero basis functions and their first derivatives into slices.
///
/// Derivatives use the standard degree-lowered recurrence:
///
/// ```text
/// dN_{i,p}/dt = p * (N_{i,p-1}/(ξ_{i+p} - ξ_i)
///                 - N_{i+1,p-1}/(ξ_{i+p+1} - ξ_{i+1}))
/// ```
///
/// The caller owns both output slices, so degree-8-or-smaller callers can keep
/// all outputs and lower-degree work storage on the stack.
///
/// # Panics
///
/// Panics if either output slice has length `< p + 1`, or if `span`, `p`, and
/// `knots` do not describe a valid active knot span.
pub fn eval_basis_and_deriv_to_slice<T: Scalar>(
    span: usize,
    t: T,
    p: usize,
    knots: &KnotVector<T>,
    out_n: &mut [T],
    out_dn: &mut [T],
) {
    let required = p + 1;
    assert_output_len("basis output", out_n.len(), required);
    assert_output_len("basis derivative output", out_dn.len(), required);

    eval_basis_to_slice(span, t, p, knots, out_n);

    if p == 0 {
        out_dn[0] = <T as NumericElement>::ZERO;
        return;
    }

    let mut lower_stack = [<T as NumericElement>::ZERO; STACK_WORK_LEN];
    let mut lower_heap;
    let lower = if p <= STACK_DEGREE_LIMIT {
        &mut lower_stack[..p]
    } else {
        lower_heap = vec![<T as NumericElement>::ZERO; p];
        &mut lower_heap
    };
    eval_basis_to_slice(span, t, p - 1, knots, lower);

    let pp = <T as Scalar>::from_f64(p as f64);
    for j in 0..=p {
        let i = span - p + j;
        let left = if j == 0 {
            <T as NumericElement>::ZERO
        } else {
            safe_div(lower[j - 1], knots.get(i + p) - knots.get(i))
        };
        let right = if j == p {
            <T as NumericElement>::ZERO
        } else {
            safe_div(lower[j], knots.get(i + p + 1) - knots.get(i + 1))
        };
        out_dn[j] = pp * (left - right);
    }
}

/// Evaluate all non-zero basis functions **and their first derivatives**.
///
/// Returns a pair of heap-allocated vectors. Use
/// [`eval_basis_and_deriv_to_slice`] to avoid allocation.
#[must_use]
pub fn eval_basis_and_deriv<T: Scalar>(
    span: usize,
    t: T,
    p: usize,
    knots: &KnotVector<T>,
) -> (Vec<T>, Vec<T>) {
    let mut n = vec![<T as NumericElement>::ZERO; p + 1];
    let mut dn = vec![<T as NumericElement>::ZERO; p + 1];
    eval_basis_and_deriv_to_slice(span, t, p, knots, &mut n, &mut dn);
    (n, dn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::geometry::nurbs::knot::KnotVector;

    #[test]
    fn partition_of_unity() {
        // Cubic with 5 control points, clamped uniform
        let kv = KnotVector::<f64>::clamped_uniform(4, 3);
        for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let span = kv.find_span(t, 4);
            let n = eval_basis(span, t, 3, &kv);
            let sum: f64 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity violated at t={t}: sum={sum}"
            );
        }
    }

    #[test]
    fn non_negativity() {
        let kv = KnotVector::<f64>::clamped_uniform(5, 3);
        for i in 0..20 {
            let t = f64::from(i) / 20.0;
            let span = kv.find_span(t, 5);
            let n = eval_basis(span, t, 3, &kv);
            for (j, &v) in n.iter().enumerate() {
                assert!(v >= -1e-14, "basis N[{j}] = {v} < 0 at t={t}");
            }
        }
    }

    #[test]
    fn linear_basis_midpoint() {
        // Linear (p=1), 2 control points: knots = [0,0,1,1]
        let kv = KnotVector::<f64>::try_new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let span = kv.find_span(0.5, 1);
        let n = eval_basis(span, 0.5, 1, &kv);
        // Both basis functions should be 0.5 at t=0.5
        assert!((n[0] - 0.5).abs() < 1e-14);
        assert!((n[1] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn slice_basis_matches_allocating_basis_for_stack_degree() {
        let kv = KnotVector::<f64>::clamped_uniform(5, 3);
        let t = 0.375;
        let span = kv.find_span(t, 5);
        let expected = eval_basis(span, t, 3, &kv);
        let mut actual = [0.0; 4];

        eval_basis_to_slice(span, t, 3, &kv, &mut actual);

        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn slice_basis_matches_allocating_basis_for_heap_degree() {
        let kv = KnotVector::<f64>::clamped_uniform(9, 9);
        let t = 0.5;
        let span = kv.find_span(t, 9);
        let expected = eval_basis(span, t, 9, &kv);
        let mut actual = vec![0.0; 10];

        eval_basis_to_slice(span, t, 9, &kv, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn derivative_slice_matches_allocating_wrapper() {
        let kv = KnotVector::<f64>::clamped_uniform(6, 3);
        let t = 0.42;
        let span = kv.find_span(t, 6);
        let (expected_n, expected_dn) = eval_basis_and_deriv(span, t, 3, &kv);
        let mut actual_n = [0.0; 4];
        let mut actual_dn = [0.0; 4];

        eval_basis_and_deriv_to_slice(span, t, 3, &kv, &mut actual_n, &mut actual_dn);

        assert_eq!(actual_n.as_slice(), expected_n.as_slice());
        assert_eq!(actual_dn.as_slice(), expected_dn.as_slice());
    }

    #[test]
    fn derivatives_match_central_difference_inside_span() {
        let kv = KnotVector::<f64>::clamped_uniform(6, 3);
        let t = 0.42;
        let h = 1.0e-6;
        let span = kv.find_span(t, 6);
        assert_eq!(span, kv.find_span(t - h, 6));
        assert_eq!(span, kv.find_span(t + h, 6));

        let (_, dn) = eval_basis_and_deriv(span, t, 3, &kv);
        let lo = eval_basis(span, t - h, 3, &kv);
        let hi = eval_basis(span, t + h, 3, &kv);

        for j in 0..=3 {
            let finite_difference = (hi[j] - lo[j]) / (2.0 * h);
            assert!(
                (dn[j] - finite_difference).abs() < 1.0e-9,
                "basis derivative mismatch at j={j}: analytic={}, finite_difference={finite_difference}",
                dn[j]
            );
        }
    }

    fn assert_linear_basis_on_span<T: Scalar>(width: T, inverse_width: T) {
        let zero = <T as NumericElement>::ZERO;
        let half = <T as Scalar>::from_f64(0.5);
        let t = width * half;
        let kv = KnotVector::try_new(vec![zero, zero, width, width]).unwrap();
        let span = kv.find_span(t, 1);

        let (basis, derivatives) = eval_basis_and_deriv(span, t, 1, &kv);

        assert_eq!(basis.as_slice(), &[half, half]);
        assert_eq!(
            derivatives.as_slice(),
            &[zero - inverse_width, inverse_width]
        );
    }

    /// Dyadic knot widths and their reciprocals are exactly representable in
    /// both scalar types; these exact assertions therefore need no tolerance.
    #[test]
    fn narrow_positive_spans_preserve_basis_and_derivatives() {
        assert_linear_basis_on_span(
            <f32 as Scalar>::from_f64(2.0_f64.powi(-60)),
            <f32 as Scalar>::from_f64(2.0_f64.powi(60)),
        );
        assert_linear_basis_on_span(
            <f64 as Scalar>::from_f64(2.0_f64.powi(-60)),
            <f64 as Scalar>::from_f64(2.0_f64.powi(60)),
        );
    }

    /// The ratio remains finite although the reciprocal of this subnormal
    /// interval exceeds the finite `f32` range.
    #[test]
    fn subnormal_positive_span_keeps_f32_basis_finite() {
        let width = <f32 as Scalar>::from_f64(2.0_f64.powi(-130));
        let half = <f32 as Scalar>::from_f64(0.5);
        let zero = <f32 as NumericElement>::ZERO;
        let t = width * half;
        let kv = KnotVector::try_new(vec![zero, zero, width, width]).unwrap();
        let span = kv.find_span(t, 1);

        let basis = eval_basis(span, t, 1, &kv);

        assert_eq!(basis.as_slice(), &[half, half]);
        assert!(basis.iter().all(|value| value.is_finite()));
    }

    /// The scalar seam monomorphizes: the `f32` instantiation holds the
    /// partition-of-unity invariant at its own precision. Tolerance derives
    /// from the Cox–de Boor error growth at degree 3 — O(p²·ε_f32) ≈
    /// 9 · 2⁻⁴⁴ ≈ 5.4e-7 — with 4× headroom; knot values are exact dyadics.
    #[test]
    fn f32_partition_of_unity() {
        let kv = KnotVector::<f32>::clamped_uniform(4, 3);
        for i in 0..=8 {
            let t = <f32 as Scalar>::from_f64(f64::from(i)) / 8.0;
            let span = kv.find_span(t, 4);
            let n = eval_basis(span, t, 3, &kv);
            let sum: f32 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 2e-6,
                "f32 partition of unity violated at t={t}: sum={sum}"
            );
        }
    }

    /// Partition of unity at `f32` on the heap-degree path: degree 9 exceeds
    /// the stack limit of 8, so the evaluation exercises the allocating
    /// branch. Error growth is O(p²·ε_f32) ≈ 81·2⁻²⁴ ≈ 4.8e-6; the
    /// assertion carries 4× headroom.
    #[test]
    fn f32_partition_of_unity_at_heap_degree() {
        let kv = KnotVector::<f32>::clamped_uniform(9, 9);
        for i in 0..=8 {
            let t = <f32 as Scalar>::from_f64(f64::from(i)) / 8.0;
            let span = kv.find_span(t, 9);
            let n = eval_basis(span, t, 9, &kv);
            let sum: f32 = n.iter().sum();
            assert!(
                (sum - 1.0).abs() < 2e-5,
                "f32 heap-degree partition of unity violated at t={t}: sum={sum}"
            );
        }
    }
}
