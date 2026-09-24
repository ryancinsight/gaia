use crate::domain::core::scalar::Scalar;
use eunomia::{FloatElement, NumericElement};

pub(super) fn rational_value<T: Scalar, const D: usize>(
    terms: impl Iterator<Item = ([T; 3], [T; D])> + Clone,
    fallback: [T; D],
) -> ([T; D], T, i32) {
    let zero = <T as NumericElement>::ZERO;
    let exponent = terms
        .clone()
        .filter_map(|(factors, _)| product_exponent(factors))
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
    for (dimension, component) in point.iter_mut().enumerate() {
        let finite_terms = <T as NumericElement>::is_finite(denominator)
            && terms.clone().all(|(factors, coordinates)| {
                factors.into_iter().all(<T as NumericElement>::is_finite)
                    && <T as NumericElement>::is_finite(coordinates[dimension])
            });
        *component = if finite_terms {
            scaled_quotient_sum(
                terms.clone().map(|(factors, coordinates)| {
                    scaled_term_parts(
                        [factors[0], factors[1], factors[2], coordinates[dimension]],
                        [<T as NumericElement>::ONE; 2],
                        -exponent,
                    )
                }),
                denominator,
            )
        } else {
            let one = <T as NumericElement>::ONE;
            terms
                .clone()
                .map(|(factors, coordinates)| {
                    scaled_rational_term(
                        factors,
                        [one, denominator],
                        coordinates[dimension],
                        zero,
                        -exponent,
                    )
                })
                .fold(zero, |sum, term| sum + term)
        };
    }
    (point, denominator, exponent)
}

fn product_exponent<T: Scalar>(factors: impl IntoIterator<Item = T>) -> Option<i32> {
    factors.into_iter().try_fold(0, |exponent, factor| {
        Some(exponent + <T as FloatElement>::binary_exponent(factor)?)
    })
}

#[derive(Clone, Copy)]
struct ScaledValue<T> {
    significand: T,
    exponent: i32,
}

// Degree-eight basis buffers carry nine terms per axis in curve and surface evaluators.
const INLINE_RATIONAL_TERMS: usize = 9 * 9;

/// Sum binary-scaled terms before dividing by the rational denominator.
fn scaled_quotient_sum<T: Scalar>(
    terms: impl Iterator<Item = Option<ScaledValue<T>>>,
    divisor: T,
) -> T {
    let zero = <T as NumericElement>::ZERO;
    let divisor_exponent = <T as FloatElement>::binary_exponent(divisor)
        .expect("invariant: finite rational denominator is nonzero");
    let divisor_significand = <T as FloatElement>::scale_binary(divisor, -divisor_exponent);
    let empty = ScaledValue {
        significand: zero,
        exponent: 0,
    };
    let mut inline = [empty; INLINE_RATIONAL_TERMS];
    let mut inline_len = 0;
    let mut overflow = Vec::new();
    for term in terms.flatten() {
        if inline_len < INLINE_RATIONAL_TERMS {
            inline[inline_len] = term;
            inline_len += 1;
        } else {
            overflow.push(term);
        }
    }
    if overflow.is_empty() {
        let mut expansion = [empty; INLINE_RATIONAL_TERMS];
        let mut scratch = [empty; INLINE_RATIONAL_TERMS];
        let expansion_len = sum_expansion(&mut inline[..inline_len], &mut expansion, &mut scratch);
        for component in &mut expansion[..expansion_len] {
            *component = divide_scaled(*component, divisor_significand, divisor_exponent);
        }
        let quotient_len =
            sum_expansion(&mut expansion[..expansion_len], &mut scratch, &mut inline);
        return rounded_expansion(&scratch[..quotient_len], zero);
    }

    overflow.extend_from_slice(&inline[..inline_len]);
    let mut expansion = vec![empty; overflow.len()];
    let mut scratch = vec![empty; overflow.len()];
    let expansion_len = sum_expansion(&mut overflow, &mut expansion, &mut scratch);
    for component in &mut expansion[..expansion_len] {
        *component = divide_scaled(*component, divisor_significand, divisor_exponent);
    }
    let quotient_len = sum_expansion(&mut expansion[..expansion_len], &mut scratch, &mut overflow);
    rounded_expansion(&overflow[..quotient_len], zero)
}

fn sum_expansion<T: Scalar>(
    terms: &mut [ScaledValue<T>],
    expansion: &mut [ScaledValue<T>],
    scratch: &mut [ScaledValue<T>],
) -> usize {
    terms.sort_unstable_by_key(|term| term.exponent);
    let mut expansion_len = 0;

    for &term in terms.iter() {
        let mut accumulator = Some(term);
        let mut scratch_len = 0;
        for &component in &expansion[..expansion_len] {
            let (sum, error) = match accumulator {
                Some(accumulator) => add_scaled(accumulator, component),
                None => (Some(component), None),
            };
            if let Some(error) = error {
                scratch[scratch_len] = error;
                scratch_len += 1;
            }
            accumulator = sum;
        }
        if let Some(accumulator) = accumulator {
            scratch[scratch_len] = accumulator;
            scratch_len += 1;
        }
        expansion[..scratch_len].copy_from_slice(&scratch[..scratch_len]);
        expansion_len = scratch_len;
    }

    expansion_len
}

fn divide_scaled<T: Scalar>(
    value: ScaledValue<T>,
    divisor: T,
    divisor_exponent: i32,
) -> ScaledValue<T> {
    scaled_value(
        value.significand / divisor,
        value
            .exponent
            .checked_sub(divisor_exponent)
            .expect("invariant: scaled rational quotient exponents fit within i32"),
    )
    .expect("invariant: finite nonzero scaled terms remain nonzero after division")
}

fn rounded_expansion<T: Scalar>(expansion: &[ScaledValue<T>], zero: T) -> T {
    let Some(maximum_exponent) = expansion.iter().map(|term| term.exponent).max() else {
        return zero;
    };
    let normalized = expansion
        .iter()
        .map(|term| {
            <T as FloatElement>::scale_binary(term.significand, term.exponent - maximum_exponent)
        })
        .fold(zero, |sum, term| sum + term);
    <T as FloatElement>::scale_binary(normalized, maximum_exponent)
}

fn add_scaled<T: Scalar>(
    left: ScaledValue<T>,
    right: ScaledValue<T>,
) -> (Option<ScaledValue<T>>, Option<ScaledValue<T>>) {
    let (larger, smaller) = if left.exponent >= right.exponent {
        (left, right)
    } else {
        (right, left)
    };
    let exponent_gap = i64::from(larger.exponent) - i64::from(smaller.exponent);
    let Ok(alignment) = i32::try_from(exponent_gap) else {
        return (Some(larger), Some(smaller));
    };
    let aligned_smaller = <T as FloatElement>::scale_binary(smaller.significand, -alignment);
    if aligned_smaller == <T as NumericElement>::ZERO {
        return (Some(larger), Some(smaller));
    }
    if <T as FloatElement>::scale_binary(aligned_smaller, alignment) != smaller.significand {
        return (Some(larger), Some(smaller));
    }

    let sum = larger.significand + aligned_smaller;
    let virtual_right = sum - larger.significand;
    let error = (larger.significand - (sum - virtual_right)) + (aligned_smaller - virtual_right);
    (
        scaled_value(sum, larger.exponent),
        scaled_value(error, larger.exponent),
    )
}

fn scaled_value<T: Scalar>(value: T, exponent: i32) -> Option<ScaledValue<T>> {
    let value_exponent = <T as FloatElement>::binary_exponent(value)?;
    Some(ScaledValue {
        significand: <T as FloatElement>::scale_binary(value, -value_exponent),
        exponent: exponent
            .checked_add(value_exponent)
            .expect("invariant: scaled rational exponents fit within i32"),
    })
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
    let term = scaled_term_parts(numerator, denominator, offset)?;
    Some(<T as FloatElement>::scale_binary(
        term.significand,
        term.exponent,
    ))
}

fn scaled_term_parts<T: Scalar>(
    numerator: [T; 4],
    denominator: [T; 2],
    offset: i32,
) -> Option<ScaledValue<T>> {
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
    scaled_value(significand, exponent)
}

fn direct_term<T: Scalar>(numerator: [T; 4], denominator: [T; 2]) -> T {
    let [derivative_basis, other_basis, raw_weight, difference] = numerator;
    let [weight_scale, weight_sum] = denominator;
    ((derivative_basis * other_basis) * (raw_weight / weight_scale) / weight_sum) * difference
}
