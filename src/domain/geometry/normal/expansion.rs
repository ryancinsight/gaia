//! Floating-point expansions for the Newell recovery path; theorems are
//! those of the paper `newell_normal` cites.

use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;

/// An exact sum: nonadjacent components in order of increasing magnitude.
pub(super) struct ExactSum<T>(Vec<T>);

impl<T: Scalar> ExactSum<T> {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    /// Adds `shrink^2 * a * b` for a power of two `shrink <= 1`, to within
    /// two least subnormals and exactly when no scaled bit falls below one.
    /// A factor of magnitude at least 1 keeps every bit under one `shrink`:
    /// the loop in `scaled_fan_sum` leaves `shrink > 1 / (16 sqrt(n MAX))`,
    /// far above `eta / EPSILON`. A smaller factor stays unscaled: the other
    /// then rounds by at most one least subnormal, and the product by less.
    pub(super) fn add_product(&mut self, a: T, b: T, shrink: T) {
        let (small, large) = if a.abs() < b.abs() { (a, b) } else { (b, a) };
        let (x, y) = if small.abs() < <T as NumericElement>::ONE {
            (small, large * shrink * shrink)
        } else {
            (small * shrink, large * shrink)
        };
        let product = x * y;
        // A product's rounding error is representable (Theorem 18), so the
        // fused `x * y - product` returns it exactly unless a bit of it lies
        // below the least subnormal.
        self.add(x.scalar_fmadd(y, -product));
        self.add(product);
    }

    // Grow-Expansion with zero elimination: every Two-Sum error survives and
    // the final carry is stored last.
    fn add(&mut self, value: T) {
        let (mut carry, mut written) = (value, 0);
        for index in 0..self.0.len() {
            let word = self.0[index];
            let sum = carry + word;
            let virtual_word = sum - carry;
            let error = (carry - (sum - virtual_word)) + (word - virtual_word);
            if error != <T as NumericElement>::ZERO {
                self.0[written] = error;
                written += 1;
            }
            carry = sum;
        }
        self.0.truncate(written);
        if carry != <T as NumericElement>::ZERO {
            self.0.push(carry);
        }
    }

    /// The components summed from the smallest; each addition may round.
    pub(super) fn estimate(&self) -> T {
        let zero = <T as NumericElement>::ZERO;
        self.0.iter().copied().fold(zero, |a, b| a + b)
    }
}
