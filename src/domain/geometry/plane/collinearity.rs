//! Exact stored-point collinearity, with a native-precision nonzero filter.
//!
//! The caller already rejects nonfinite/vanishing native cross norms. A zero
//! rounded edge difference is an exact equality of two stored finite scalars:
//! IEEE subtraction cannot lose a nonzero difference below the least subnormal.
//! Thus a nonzero minor with one zero factor is certainly nonzero exactly.
//!
//! Otherwise let `u = EPSILON/2` and `S = |fl(ab_i ac_j)| + |fl(ab_j ac_i)|`.
//! Each product contains two subtractions and one multiplication. For normal
//! results its relative error is bounded by `gamma_3 = 3u/(1-3u)`; subnormal
//! subtraction is exact and each underflowed product adds at most eta/2,
//! where eta is the least subnormal. If the exact minor is zero, its computed
//! magnitude is therefore at most 6u S + 4 eta (including subtraction and
//! expressing the bound in computed products). Our evaluated 16u(S+tol²)
//! exceeds this even after bound-rounding for both sealed precisions. The
//! tol² term dominates underflow: u tol² >> eta in binary32 and binary64.
//! Overflow of S makes the filter inconclusive, never a nonzero certificate.
//!
//! The fallback adds the six signed products of original coordinates as exact
//! dyadic integers. Binary64 has at most 53 significand bits and a least-bit
//! exponent from -1074 to 971. A product therefore has at most 106 bits and a
//! shift from 0 to 4090 relative to 2^-2148; six products need at most 4199
//! bits. Two 66-limb unsigned sums suffice without overflow, allocation, or
//! floating-point products. Equality of the sums is exactly a zero minor.
//! Lossless f32-to-f64 promotion is a decision-only boundary (ADR 0005).

use crate::domain::core::scalar::Scalar;
use eunomia::NumericElement;
use leto::geometry::{Point3, Vector3};

/// Decide collinearity after the caller validates a finite, nonzero cross norm.
#[inline]
pub(super) fn is_collinear<T: Scalar>(
    a: &Point3<T>,
    b: &Point3<T>,
    c: &Point3<T>,
    ab: Vector3<T>,
    ac: Vector3<T>,
    cross: Vector3<T>,
) -> bool {
    let zero = <T as NumericElement>::ZERO;
    let (row, column, minor) = if cross.z != zero {
        (0, 1, cross.z)
    } else if cross.x != zero {
        (1, 2, cross.x)
    } else {
        (2, 0, cross.y)
    };
    if ab[row] == zero || ac[column] == zero || ab[column] == zero || ac[row] == zero {
        return false;
    }
    let permanent = (ab[row] * ac[column]).abs() + (ab[column] * ac[row]).abs();
    let bound = T::from_int(8) * T::EPSILON * (permanent + T::tolerance_sq());
    if minor.abs() > bound {
        return false;
    }
    exact_collinear(a, b, c)
}

/// Keep dyadic decoding and its stack storage outside the common inline path.
#[cold]
fn exact_collinear<T: Scalar>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> bool {
    [[0, 1], [1, 2], [2, 0]].into_iter().all(|[i, j]| {
        zero_minor(
            [a[i], a[j]].map(<T as NumericElement>::to_f64),
            [b[i], b[j]].map(<T as NumericElement>::to_f64),
            [c[i], c[j]].map(<T as NumericElement>::to_f64),
        )
    })
}

// The significand/exponent bounds are properties of the exact binary64
// representation, not geometric tolerances or empirically selected limits.
const SIGNIFICAND_BITS: usize = 53;
const MAX_LEAST_BIT_SHIFT: usize = 2045;
const SUM_BITS: usize = 2 * MAX_LEAST_BIT_SHIFT + 2 * SIGNIFICAND_BITS + 3;
const WORD_BITS: usize = 64;
const LIMBS: usize = SUM_BITS.div_ceil(WORD_BITS);

/// Sign, integer significand, and nonnegative least-bit shift from 2^-1074.
#[inline]
fn dyadic(value: f64) -> (bool, u64, usize) {
    let bits = value.to_bits();
    let exponent = usize::try_from((bits >> 52) & 0x7ff).expect("eleven bits fit usize");
    let significand = (bits & ((1_u64 << 52) - 1)) | (u64::from(exponent != 0) << 52);
    (bits >> 63 != 0, significand, exponent.saturating_sub(1))
}

/// Add one word and propagate carry; six 106-bit products fit the proven bound.
#[inline]
fn add_word(sum: &mut [u64; LIMBS], mut index: usize, mut word: u64) {
    while word != 0 {
        let (value, carry) = sum[index].overflowing_add(word);
        sum[index] = value;
        word = u64::from(carry);
        index += 1;
    }
}

/// Exact zero of the original-point orientation determinant; inputs are finite.
#[cold]
fn zero_minor(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let [ax, ay] = a.map(dyadic);
    let [bx, by] = b.map(dyadic);
    let [cx, cy] = c.map(dyadic);
    let mut sums = [[0_u64; LIMBS]; 2];
    // ax by + bx cy + cx ay - ax cy - bx ay - cx by.
    for (left, right, subtract) in [
        (ax, by, false),
        (bx, cy, false),
        (cx, ay, false),
        (ax, cy, true),
        (bx, ay, true),
        (cx, by, true),
    ] {
        let product = u128::from(left.1) * u128::from(right.1);
        let sum = &mut sums[usize::from(left.0 ^ right.0 ^ subtract)];
        let shift = left.2 + right.2;
        let index = shift / WORD_BITS;
        let offset = shift % WORD_BITS;
        let bytes = product.to_le_bytes();
        let (low, high) = bytes.split_at(size_of::<u64>());
        let low = u64::from_le_bytes(low.try_into().expect("low word has eight bytes"));
        let high = u64::from_le_bytes(high.try_into().expect("high word has eight bytes"));
        if offset == 0 {
            add_word(sum, index, low);
            add_word(sum, index + 1, high);
        } else {
            add_word(sum, index, low << offset);
            add_word(
                sum,
                index + 1,
                (high << offset) | (low >> (WORD_BITS - offset)),
            );
            add_word(sum, index + 2, high >> (WORD_BITS - offset));
        }
    }
    sums[0] == sums[1]
}

#[cfg(test)]
mod tests {
    use super::zero_minor;

    #[test]
    fn exact_zero_covers_binary64_exponent_range() {
        // Determinants 2^e * 2^-1074 and 2^e * 2^1023 are nonzero as
        // rationals, including products outside floating-point range.
        for exponent in -1074..=1023 {
            let x = if exponent < -1022 {
                f64::from_bits(1_u64 << (exponent + 1074))
            } else {
                2.0_f64.powi(exponent)
            };
            for y in [f64::from_bits(1), 1.0, 2.0_f64.powi(1023)] {
                assert!(!zero_minor([0.0, 0.0], [x, 0.0], [0.0, y]));
                assert!(zero_minor([-x, -y], [0.0, 0.0], [x, y]));
                // One adjacent representable coordinate breaks y/x equality.
                assert!(!zero_minor([-x, -y], [0.0, 0.0], [x, y.next_up()]));
            }
        }
        assert!(zero_minor([-f64::MAX; 2], [0.0; 2], [f64::MAX; 2]));
        assert!(!zero_minor(
            [-f64::MAX; 2],
            [0.0; 2],
            [f64::MAX, f64::MAX.next_down()]
        ));
    }
}
