//! Native floating expansions for the nonfinite Newell recovery path.

use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;

// Inline capacity is a storage choice, not an arithmetic precision limit.
// A spill retains every component rather than truncating residuals.
pub(super) struct NativeSum<T: Scalar, const INLINE: usize> {
    inline: [T; INLINE],
    spill: Option<Vec<T>>,
    len: usize,
}

impl<T: Scalar, const INLINE: usize> NativeSum<T, INLINE> {
    pub(super) fn new() -> Self {
        Self {
            inline: [<T as NumericElement>::ZERO; INLINE],
            spill: None,
            len: 0,
        }
    }

    pub(super) fn add_product(&mut self, a: T, b: T) -> bool {
        let product = a * b;
        if !<T as NumericElement>::is_finite(product) {
            return false;
        }
        // After the finite-product guard, a zero operand makes both the
        // product and its fused residual exactly zero (including signed zero).
        if a == <T as NumericElement>::ZERO || b == <T as NumericElement>::ZERO {
            return true;
        }
        // The native fused operation recovers product roundoff whenever that
        // residual is representable. No T->wider->T arithmetic occurs.
        let error = a.scalar_fmadd(b, -product);
        self.add(error) && self.add(product)
    }

    fn add(&mut self, value: T) -> bool {
        if !<T as NumericElement>::is_finite(value) {
            return false;
        }
        if value == <T as NumericElement>::ZERO {
            return true;
        }
        if self.len == INLINE && self.spill.is_none() {
            let mut spill = Vec::with_capacity(INLINE.saturating_mul(2).max(1));
            spill.extend_from_slice(&self.inline[..self.len]);
            self.spill = Some(spill);
        }
        let words = match self.spill.as_deref_mut() {
            Some(words) => words,
            None => &mut self.inline[..self.len],
        };
        let mut carry = value;
        let mut written = 0;
        // Grow-expansion addition: each exact TwoSum residual survives, with
        // the final carry stored last. Components stay ordered/nonoverlapping.
        for index in 0..self.len {
            let word = words[index];
            let sum = carry + word;
            let virtual_word = sum - carry;
            let error = (carry - (sum - virtual_word)) + (word - virtual_word);
            if !<T as NumericElement>::is_finite(sum) || !<T as NumericElement>::is_finite(error) {
                return false;
            }
            if error != <T as NumericElement>::ZERO {
                words[written] = error;
                written += 1;
            }
            carry = sum;
        }
        if let Some(spill) = &mut self.spill {
            spill.truncate(written);
            if carry != <T as NumericElement>::ZERO {
                spill.push(carry);
                written += 1;
            }
        } else if carry != <T as NumericElement>::ZERO {
            self.inline[written] = carry;
            written += 1;
        }
        self.len = written;
        true
    }

    pub(super) fn estimate(&self) -> T {
        let words = match self.spill.as_deref() {
            Some(words) => words,
            None => &self.inline[..self.len],
        };
        words
            .iter()
            .copied()
            .fold(<T as NumericElement>::ZERO, |a, b| a + b)
    }
}
