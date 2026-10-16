//! Exact finite-box classification after positive-infinity slab collapse.
//!
//! Stored-coordinate endpoint ratios are (bound-origin)/direction. Comparing
//! two ratios needs no floating subtraction, multiplication, or division:
//! the determinant with rows [a,b,1], [oa,ob,1], [da,db,0] is
//! -((a-oa)*db - (b-ob)*da). Its sign, adjusted for the denominator signs,
//! orders the exact ratios. The shared full-exponent predicate supplies that
//! sign (ADRs 0005/0009), including original-point subtraction overflow.

use crate::domain::core::scalar::Scalar;
use crate::domain::geometry::{predicates::evaluate, Aabb};
use eunomia::NumericElement;
use leto::geometry::{Point3, Vector3};

#[derive(Clone, Copy)]
struct Endpoint {
    bound: f64,
    origin: f64,
    direction: f64,
}

impl Endpoint {
    fn before(self, other: Self) -> bool {
        let determinant = evaluate::orient3d(
            [self.bound, other.bound, 1.0],
            [self.origin, other.origin, 1.0],
            [self.direction, other.direction, 0.0],
            [0.0; 3],
        );
        if (self.direction > 0.0) == (other.direction > 0.0) {
            determinant >= 0.0
        } else {
            determinant <= 0.0
        }
    }

    fn nonnegative(self) -> bool {
        if self.direction > 0.0 {
            self.bound >= self.origin
        } else {
            self.bound <= self.origin
        }
    }
}

/// Keep legacy nonfinite-box behavior; remove only a proven finite-box miss.
/// This does not select an invalid-input or returned-distance representation.
#[cold]
pub(super) fn retains_hit<T: Scalar>(
    origin: Point3<T>,
    direction: Vector3<T>,
    bounds: &Aabb<T>,
) -> bool {
    if !(0..3).all(|axis| {
        <T as NumericElement>::is_finite(bounds.min[axis])
            && <T as NumericElement>::is_finite(bounds.max[axis])
    }) {
        return true;
    }
    // Only an exact decision crosses this seam. There is no widened native
    // construction or quotient: f32 values promote losslessly, f64 identically.
    intersect(
        std::array::from_fn(|axis| origin[axis].to_f64()),
        std::array::from_fn(|axis| direction[axis].to_f64()),
        std::array::from_fn(|axis| bounds.min[axis].to_f64()),
        std::array::from_fn(|axis| bounds.max[axis].to_f64()),
    )
}

/// At most three active axes require four endpoint selections and one final
/// comparison. Each calls the existing bounded exact orient3d implementation;
/// no extra numeric provider, unbounded search, or dynamic collection occurs.
fn intersect(origin: [f64; 3], direction: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
    let mut entry: Option<Endpoint> = None;
    let mut exit: Option<Endpoint> = None;
    for axis in 0..3 {
        if lo[axis] > hi[axis] {
            return false;
        }
        if direction[axis] == 0.0 {
            if origin[axis] < lo[axis] || origin[axis] > hi[axis] {
                return false;
            }
            continue;
        }
        let endpoint = |bound| Endpoint {
            bound,
            origin: origin[axis],
            direction: direction[axis],
        };
        let (lower, upper) = if direction[axis] > 0.0 {
            (endpoint(lo[axis]), endpoint(hi[axis]))
        } else {
            (endpoint(hi[axis]), endpoint(lo[axis]))
        };
        if entry.is_none_or(|current| current.before(lower)) {
            entry = Some(lower);
        }
        if exit.is_none_or(|current| upper.before(current)) {
            exit = Some(upper);
        }
    }
    match (entry, exit) {
        (Some(entry), Some(exit)) => exit.nonnegative() && entry.before(exit),
        _ => true,
    }
}
