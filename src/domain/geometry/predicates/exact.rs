//! Full-exponent finite predicate signs. See ADR 0009 for bounds and selection.
//!
//! Finite binary64 coordinates are integral multiples of 2^-1074. Decoding
//! their integer coefficients preserves all stored bits, including subnormals.
//! Each determinant is homogeneous, so the common positive dyadic factor
//! changes its magnitude but never its sign. No floating product or rounded
//! coordinate transformation occurs on this path.

use num_bigint::{BigInt, Sign};

/// The existing expansion provider proves its range for binary exponents
/// [-142,201]. Use complete bins [-142,200], plus signed zero; the upper bin
/// boundary stays strictly below 2^201. NaN/infinity fail this classifier.
#[inline]
pub(super) fn expansion_range<const N: usize, const D: usize>(points: &[[f64; D]; N]) -> bool {
    points.iter().flatten().fold(true, |safe, coordinate| {
        let bits = coordinate.to_bits();
        let exponent = (bits >> 52) & 0x7ff;
        safe & (bits << 1 == 0 || exponent.wrapping_sub(881) <= 342)
    })
}

#[inline]
pub(super) fn finite<const N: usize, const D: usize>(points: &[[f64; D]; N]) -> bool {
    points
        .iter()
        .flatten()
        .all(|coordinate| coordinate.is_finite())
}

/// For normal inputs, biased exponent e gives integer coefficient m << (e-1).
/// For subnormals, the stored fraction already is that coefficient. Sign is
/// applied after decoding; both signed zeros map to the same integer zero.
fn parts(coordinate: f64) -> (bool, u64, usize) {
    debug_assert!(coordinate.is_finite());
    let bits = coordinate.to_bits();
    let exponent = u16::try_from((bits >> 52) & 0x7ff).expect("11-bit exponent fits u16");
    let fraction = bits & ((1_u64 << 52) - 1);
    let significand = fraction | if exponent == 0 { 0 } else { 1_u64 << 52 };
    (
        bits >> 63 != 0,
        significand,
        usize::from(exponent.saturating_sub(1)),
    )
}

fn integer_in_basis(coordinate: f64, common_shift: usize) -> BigInt {
    let (negative, significand, shift) = parts(coordinate);
    if significand == 0 {
        return BigInt::ZERO;
    }
    let trailing = usize::try_from(significand.trailing_zeros()).expect("at most 52 trailing bits");
    let value = BigInt::from(significand >> trailing) << (shift + trailing - common_shift);
    if negative {
        -value
    } else {
        value
    }
}

fn differences<const N: usize, const D: usize>(
    points: [[f64; D]; N],
    origin: [f64; D],
) -> [[BigInt; D]; N] {
    // Remove only a proven common integer power of two before arithmetic.
    // This changes the exact dyadic basis, never a stored floating value.
    // All nonzero coefficients are divisible by this factor; no bit is lost.
    // Homogeneity preserves the determinant sign.
    let common_shift = points
        .iter()
        .chain(std::iter::once(&origin))
        .flatten()
        .filter_map(|coordinate| {
            let (_, significand, shift) = parts(*coordinate);
            (significand != 0).then(|| {
                shift
                    + usize::try_from(significand.trailing_zeros())
                        .expect("at most 52 trailing bits")
            })
        })
        .min()
        .unwrap_or(0);
    let origin = origin.map(|coordinate| integer_in_basis(coordinate, common_shift));
    points.map(|point| {
        std::array::from_fn(|axis| integer_in_basis(point[axis], common_shift) - &origin[axis])
    })
}

/// Leibniz sum over at most four columns. Only nonzero partial products are
/// visited. `used` records selected columns; prior columns greater than the
/// next column contribute exactly the new permutation inversions.
fn determinant<const N: usize>(matrix: &[[BigInt; N]; N]) -> BigInt {
    fn visit<const N: usize>(
        matrix: &[[BigInt; N]; N],
        row: usize,
        used: u8,
        negative: bool,
        product: &BigInt,
        sum: &mut BigInt,
    ) {
        if row == N {
            if negative {
                *sum -= product;
            } else {
                *sum += product;
            }
            return;
        }
        for column in 0..N {
            let bit = 1_u8 << column;
            if used & bit != 0 || matrix[row][column] == BigInt::ZERO {
                continue;
            }
            let odd = (used >> (column + 1)).count_ones() & 1 != 0;
            visit(
                matrix,
                row + 1,
                used | bit,
                negative ^ odd,
                &(product * &matrix[row][column]),
                sum,
            );
        }
    }
    const { assert!(N <= 4) };
    let mut sum = BigInt::ZERO;
    visit(matrix, 0, 0, false, &BigInt::from(1_u8), &mut sum);
    sum
}

fn sign(value: &BigInt) -> f64 {
    match value.sign() {
        Sign::Minus => -1.0,
        Sign::NoSign => 0.0,
        Sign::Plus => 1.0,
    }
}

#[cold]
pub(super) fn orient2d([a, b, c]: [[f64; 2]; 3]) -> f64 {
    sign(&determinant(&differences([a, b], c)))
}

/// Preserve the expansion provider's "d below abc is positive" convention.
#[cold]
pub(super) fn orient3d([a, b, c, d]: [[f64; 3]; 4]) -> f64 {
    sign(&determinant(&differences([a, b, c], d)))
}

#[cold]
pub(super) fn incircle([a, b, c, d]: [[f64; 2]; 4]) -> f64 {
    let matrix = differences([a, b, c], d).map(|[x, y]| {
        let lift = &x * &x + &y * &y;
        [x, y, lift]
    });
    sign(&determinant(&matrix))
}

#[cold]
pub(super) fn insphere([first, second, third, fourth, query]: [[f64; 3]; 5]) -> f64 {
    let matrix = differences([first, second, third, fourth], query).map(|[x, y, z]| {
        let lift = &x * &x + &y * &y + &z * &z;
        [x, y, z, lift]
    });
    sign(&determinant(&matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(coordinate: f64) -> BigInt {
        integer_in_basis(coordinate, 0)
    }

    #[test]
    fn integer_decoding_preserves_ieee_endpoints() {
        assert_eq!(integer(0.0), BigInt::ZERO);
        assert_eq!(integer(-0.0), BigInt::ZERO);
        assert_eq!(integer(f64::from_bits(1)), BigInt::from(1_u8));
        assert_eq!(integer(-f64::from_bits(1)), BigInt::from(-1_i8));
        assert_eq!(integer(f64::MIN_POSITIVE), BigInt::from(1_u8) << 52);
        let maximum = BigInt::from((1_u64 << 53) - 1) << 2045;
        assert_eq!(integer(f64::MAX), maximum);
        assert_eq!(integer(-f64::MAX), -maximum);
        assert_eq!(integer(f64::MAX).bits(), 2098);
        assert_eq!(
            differences([[2.0_f64.powi(600), 0.0]], [0.0, 0.0]),
            [[BigInt::from(1_u8), BigInt::ZERO]]
        );
    }

    #[test]
    fn provider_classifier_preserves_proven_bins_and_signed_zero() {
        let lower = f64::from_bits(881_u64 << 52);
        let upper = f64::from_bits(1224_u64 << 52);
        assert!(expansion_range(&[[
            0.0,
            -0.0,
            lower,
            -lower,
            1.0,
            f64::from_bits(upper.to_bits() - 1)
        ]]));
        assert!(!expansion_range(&[[f64::from_bits(lower.to_bits() - 1)]]));
        assert!(!expansion_range(&[[upper]]));
        assert!(!expansion_range(&[[f64::from_bits(1)]]));
        assert!(!expansion_range(&[[f64::MAX]]));
        assert!(!finite(&[[f64::INFINITY]]));
        assert!(!finite(&[[f64::NAN]]));
    }
}
