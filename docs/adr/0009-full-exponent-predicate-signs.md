# ADR 0009: Preserve finite predicate signs across the binary64 exponent range

- Status: Accepted
- Change class: [patch]
- Item: GAIA-026

## Context

Both public predicate families promise exact signs for stored finite coordinates.
The current geometry-predicates 0.3.0 provider documents exponent limitations.
Saved regressions demonstrate overflow-to-NaN, silent underflow-to-zero and
cyclic inconsistency in all four determinant families. Lossless promotion in
ADR 0005 preserves coordinates, but does not extend expansion exponent range.

Current selected Eunomia, Aequitas, Leto and Melinoe expose no reusable signed
integer or full-exponent determinant provider. Eunomia's private five-word
reduced-format kernel is neither public nor large enough. Switching to robust
1.2 merely changes Shewchuk transcription, without a demonstrated range repair.
Use current maintained num-bigint 0.5.1 privately, with defaults disabled,
rather than introducing a custom large-integer carry/multiplication kernel.

## Decision

Route both public wrapper families through one private evaluation boundary.
Retain the existing adaptive provider when every coordinate is signed zero or
has binary exponent in [-142,200]. Complete bins stay inside the provider's
stated [-142,201] safe input exponent envelope. Outside that envelope, finite
inputs use a cold exact signed determinant. Do not infer correctness from a
nonzero result, catch only NaN, round inputs or heuristically rescale floats.

Every finite binary64 value is n * 2^-1074 for an integer n. Decode n directly
from its IEEE bits, including signed zero and subnormals. Remove the minimum
shared power of two from those integer coefficients before arithmetic: the
factor divides every coefficient exactly, so this is a compact exact dyadic
basis, with no floating rescaling, rounding or discarded coordinate bits.
Differences and
squared lifts use integer arithmetic. The translated 2x2/3x3 orientation,
3x3 incircle and 4x4 insphere determinants have homogeneous degrees 2/3/4/5.
The removed common dyadic factor is positive, so integer and original signs
agree exactly. A bounded Leibniz enumeration sums at most 24 signed terms;
zero factors prune only exact-zero terms. Both established 3-D orientation
conventions and circle/sphere winding conventions remain unchanged.

Input coefficients have magnitude below 2^2098. Differences are below 2^2099;
three-dimensional squared lifts are below 2^4200. Thus no insphere term reaches
2^10497, and the sum of 24 terms stays below 2^10502. Other families need at
most 4199/6300/8400 bits. Fixed matrix sizes and at most 65 recursive visits
bound operand sizes and arithmetic counts independently of conditioning.
The maintained integer provider uses heap storage; measured allocation/peak
live-byte costs must accompany the draft. No allocation reaches the adaptive
path. No native construction coefficient or physical tolerance changes.

GAIA-027 remains separate: nonfinite input takes the same legacy expansion
calls and converters. This patch introduces no invalid enum case, new panic,
error return or public precondition. A future explicit invalid-input contract
needs its own versioning and caller audit.

## Verification

Keep the original red analytical tests and independent rational determinants.
Add full-exponent stored-bit oracle cases, cancellation, exact degeneracy,
winding/permutation properties and both public families. Test debug/release,
actual f32 promotion, the classifier boundaries and IEEE endpoints. Measure
ordinary/adversarial constructor time and allocation/peak live bytes against
unchanged base; do not present local microbenchmarks as mesh throughput.

Require native tests, diagnostics/docs, supply-chain/hygiene, fresh/normalized
consumers, published-baseline API, package verification, normal push, hosted
gates and independent exact-head review. New predicate work remains draft and
unpublished until those conditions are assessed. Preserve all unrelated holds.
