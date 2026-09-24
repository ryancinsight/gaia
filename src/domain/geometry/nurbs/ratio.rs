use crate::domain::core::scalar::Scalar;
use eunomia::{FloatElement, NumericElement};

pub(super) fn rational_value<T: Scalar, const D: usize>(
    terms: impl Iterator<Item = ([T; 3], [T; D])> + Clone,
    fallback: [T; D],
) -> ([T; D], T, i32) {
    let zero = <T as NumericElement>::ZERO;
    let exponent = terms
        .clone()
        .filter(|(factors, _)| {
            factors
                .iter()
                .all(|&factor| factor != zero && <T as NumericElement>::is_finite(factor))
        })
        .map(|(factors, _)| {
            factors
                .into_iter()
                .map(|factor| {
                    <T as FloatElement>::binary_exponent(factor)
                        .expect("invariant: active finite factors have binary exponents")
                })
                .sum()
        })
        .max()
        .unwrap_or(0);
    let mut denominator = zero;
    for (factors, _) in terms.clone() {
        denominator += scaled_rational_term(
            factors,
            [<T as NumericElement>::ONE; 2],
            <T as NumericElement>::ONE,
            zero,
            -exponent,
        );
    }
    if denominator == zero {
        return (fallback, denominator, exponent);
    }
    let mut point = [zero; D];
    for (factors, coordinates) in terms {
        for (component, coordinate) in point.iter_mut().zip(coordinates) {
            *component += scaled_rational_term(
                factors,
                [<T as NumericElement>::ONE, denominator],
                coordinate,
                zero,
                -exponent,
            );
        }
    }
    (point, denominator, exponent)
}

/// Evaluate a rational product term without losing a representable result to
/// intermediate overflow or underflow.
///
/// Finite factors are binary-scaled before multiplication. If the coordinate
/// difference itself overflows, its operands are scaled before subtraction.
/// Zero and non-finite factors retain direct IEEE arithmetic.
pub(super) fn scaled_rational_term<T: Scalar>(
    numerator: [T; 3],
    denominator: [T; 2],
    positive: T,
    negative: T,
    exponent_offset: i32,
) -> T {
    if positive == negative && <T as NumericElement>::is_finite(positive) {
        return <T as NumericElement>::ZERO;
    }
    let difference = positive - negative;
    let factors = [numerator[0], numerator[1], numerator[2], difference];
    if <T as NumericElement>::is_finite(difference) {
        return scaled_term(factors, denominator, exponent_offset).unwrap_or_else(|| {
            <T as FloatElement>::scale_binary(direct_term(factors, denominator), exponent_offset)
        });
    }
    if !<T as NumericElement>::is_finite(positive) || !<T as NumericElement>::is_finite(negative) {
        return direct_term(factors, denominator);
    }

    let coordinate_exponent = [positive, negative]
        .into_iter()
        .filter_map(<T as FloatElement>::binary_exponent)
        .max()
        .expect("invariant: finite coordinate subtraction overflow has a nonzero operand");
    let difference = <T as FloatElement>::scale_binary(positive, -coordinate_exponent)
        - <T as FloatElement>::scale_binary(negative, -coordinate_exponent);
    let scaled_factors = [numerator[0], numerator[1], numerator[2], difference];
    scaled_term(
        scaled_factors,
        denominator,
        coordinate_exponent + exponent_offset,
    )
    .unwrap_or_else(|| {
        <T as FloatElement>::scale_binary(direct_term(factors, denominator), exponent_offset)
    })
}

fn scaled_term<T: Scalar>(numerator: [T; 4], denominator: [T; 2], offset: i32) -> Option<T> {
    let mut significand = <T as NumericElement>::ONE;
    let mut exponent = i64::from(offset);

    for factor in numerator {
        let factor_exponent = <T as FloatElement>::binary_exponent(factor)?;
        significand *= <T as FloatElement>::scale_binary(factor, -factor_exponent);
        exponent += i64::from(factor_exponent);
    }

    for factor in denominator {
        let factor_exponent = <T as FloatElement>::binary_exponent(factor)?;
        significand = significand / <T as FloatElement>::scale_binary(factor, -factor_exponent);
        exponent -= i64::from(factor_exponent);
    }

    let exponent =
        i32::try_from(exponent).expect("invariant: seven finite scalar exponents fit within i32");
    Some(<T as FloatElement>::scale_binary(significand, exponent))
}

fn direct_term<T: Scalar>(numerator: [T; 4], denominator: [T; 2]) -> T {
    let [derivative_basis, other_basis, raw_weight, difference] = numerator;
    let [weight_scale, weight_sum] = denominator;
    ((derivative_basis * other_basis) * (raw_weight / weight_scale) / weight_sum) * difference
}
