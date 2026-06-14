//! # B-Spline Basis Function Evaluation
//!
//! Implements the **Cox–de Boor recursion** for computing B-spline basis
//! functions `N_{i,p}(ξ)` and their first derivatives `N'_{i,p}(ξ)`.
//!
//! ## Algorithm — de Boor's Algorithm
//!
//! Rather than computing all `N_{i,p}` recursively (which recomputes shared
//! sub-expressions), this module uses the *triangular table* approach:
//!
//! ```text
//! N[0] = [N_{i-p,0}, N_{i-p+1,0}, …, N_{i,0}]  (only the active span is 1)
//!
//! For d = 1 to p:
//!   For j = 0 to p-d:
//!     left  = ξ − ξ_{i-p+d+j}
//!     right = ξ_{i+1+j} − ξ
//!     saved = N[j] * left / (left + right)
//!     N[j]  = saved + N[j+1] * right / (left + right)
//! ```
//!
//! This yields the `p+1` non-zero basis functions `N_{i-p,p}(ξ)` through
//! `N_{i,p}(ξ)` in O(p²) time.
//!
//! ## Theorem — Partition of Unity
//!
//! For any valid knot span and any `ξ` in `[ξᵢ, ξᵢ₊₁)`:
//!
//! ```text
//! Σⱼ₌₀ᵖ N_{i-p+j, p}(ξ) = 1
//! ```
//!
//! All evaluations in this module maintain this invariant.

use super::knot::KnotVector;
use crate::domain::core::scalar::Real;

const STACK_DEGREE_LIMIT: usize = 8;
const STACK_WORK_LEN: usize = STACK_DEGREE_LIMIT + 1;

#[inline]
fn assert_output_len(name: &str, len: usize, required: usize) {
    assert!(
        len >= required,
        "invariant: {name} length must be at least degree + 1; got {len}, required {required}"
    );
}

#[inline]
fn safe_div(num: Real, denom: Real) -> Real {
    if denom.abs() < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

fn fill_basis(
    span: usize,
    t: Real,
    p: usize,
    knots: &KnotVector,
    out: &mut [Real],
    left: &mut [Real],
    right: &mut [Real],
) {
    debug_assert!(out.len() > p);
    debug_assert!(left.len() > p);
    debug_assert!(right.len() > p);

    out[0] = 1.0;
    for j in 1..=p {
        left[j] = t - knots.get(span + 1 - j);
        right[j] = knots.get(span + j) - t;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = safe_div(out[r], right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
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
pub fn eval_basis_to_slice(span: usize, t: Real, p: usize, knots: &KnotVector, out: &mut [Real]) {
    let required = p + 1;
    assert_output_len("basis output", out.len(), required);

    if p <= STACK_DEGREE_LIMIT {
        let mut left = [0.0; STACK_WORK_LEN];
        let mut right = [0.0; STACK_WORK_LEN];
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
        let mut left = vec![0.0; required];
        let mut right = vec![0.0; required];
        fill_basis(span, t, p, knots, out, &mut left, &mut right);
    }
}

/// Evaluate the `p+1` non-zero B-spline basis functions at parameter `t`.
///
/// Returns a heap-allocated vector. Use [`eval_basis_to_slice`] to avoid allocation.
#[must_use]
pub fn eval_basis(span: usize, t: Real, p: usize, knots: &KnotVector) -> Vec<Real> {
    let mut n = vec![0.0 as Real; p + 1];
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
pub fn eval_basis_and_deriv_to_slice(
    span: usize,
    t: Real,
    p: usize,
    knots: &KnotVector,
    out_n: &mut [Real],
    out_dn: &mut [Real],
) {
    let required = p + 1;
    assert_output_len("basis output", out_n.len(), required);
    assert_output_len("basis derivative output", out_dn.len(), required);

    eval_basis_to_slice(span, t, p, knots, out_n);

    if p == 0 {
        out_dn[0] = 0.0;
        return;
    }

    let mut lower_stack = [0.0; STACK_WORK_LEN];
    let mut lower_heap;
    let lower = if p <= STACK_DEGREE_LIMIT {
        &mut lower_stack[..p]
    } else {
        lower_heap = vec![0.0; p];
        &mut lower_heap
    };
    eval_basis_to_slice(span, t, p - 1, knots, lower);

    let pp = p as Real;
    for j in 0..=p {
        let i = span - p + j;
        let left = if j == 0 {
            0.0
        } else {
            safe_div(lower[j - 1], knots.get(i + p) - knots.get(i))
        };
        let right = if j == p {
            0.0
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
pub fn eval_basis_and_deriv(
    span: usize,
    t: Real,
    p: usize,
    knots: &KnotVector,
) -> (Vec<Real>, Vec<Real>) {
    let mut n = vec![0.0 as Real; p + 1];
    let mut dn = vec![0.0 as Real; p + 1];
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
        let kv = KnotVector::clamped_uniform(4, 3);
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
        let kv = KnotVector::clamped_uniform(5, 3);
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
        let kv = KnotVector::try_new(vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let span = kv.find_span(0.5, 1);
        let n = eval_basis(span, 0.5, 1, &kv);
        // Both basis functions should be 0.5 at t=0.5
        assert!((n[0] - 0.5).abs() < 1e-14);
        assert!((n[1] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn slice_basis_matches_allocating_basis_for_stack_degree() {
        let kv = KnotVector::clamped_uniform(5, 3);
        let t = 0.375;
        let span = kv.find_span(t, 5);
        let expected = eval_basis(span, t, 3, &kv);
        let mut actual = [0.0; 4];

        eval_basis_to_slice(span, t, 3, &kv, &mut actual);

        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn slice_basis_matches_allocating_basis_for_heap_degree() {
        let kv = KnotVector::clamped_uniform(9, 9);
        let t = 0.5;
        let span = kv.find_span(t, 9);
        let expected = eval_basis(span, t, 9, &kv);
        let mut actual = vec![0.0; 10];

        eval_basis_to_slice(span, t, 9, &kv, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn derivative_slice_matches_allocating_wrapper() {
        let kv = KnotVector::clamped_uniform(6, 3);
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
        let kv = KnotVector::clamped_uniform(6, 3);
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
}
