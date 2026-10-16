//! Shared provider boundary; return only a determinant sign to public wrappers.
//!
//! Keep legacy nonfinite-input behavior separate from the finite exactness
//! repair (GAIA-027). Such inputs still take the same expansion calls and
//! converters; this change adds no error variant, precondition or panic.

use super::exact;
use geometry_predicates as gp;

#[inline]
pub(crate) fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let points = [a, b, c];
    if exact::expansion_range(&points) || !exact::finite(&points) {
        gp::orient2d(a, b, c)
    } else {
        exact::orient2d(points)
    }
}

#[inline]
pub(crate) fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let points = [a, b, c, d];
    if exact::expansion_range(&points) || !exact::finite(&points) {
        gp::orient3d(a, b, c, d)
    } else {
        exact::orient3d(points)
    }
}

#[inline]
pub(crate) fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let points = [a, b, c, d];
    if exact::expansion_range(&points) || !exact::finite(&points) {
        gp::incircle(a, b, c, d)
    } else {
        exact::incircle(points)
    }
}

#[inline]
pub(crate) fn insphere(
    first: [f64; 3],
    second: [f64; 3],
    third: [f64; 3],
    fourth: [f64; 3],
    query: [f64; 3],
) -> f64 {
    let points = [first, second, third, fourth, query];
    if exact::expansion_range(&points) || !exact::finite(&points) {
        gp::insphere(first, second, third, fourth, query)
    } else {
        exact::insphere(points)
    }
}
