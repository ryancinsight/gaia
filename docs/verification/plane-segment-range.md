# Native plane/segment ratio range

GAIA-037, provisional patch candidate 0.6.10, bases on published main 0.6.8 at
5eca6a6779b062dacd752dfc756e63a384193a71. No held #184 normal source is included.
The candidate version avoids collision with #184's declared 0.6.9; it does not
reserve or approve that release. Merge/publication stay held for independent
exact-head review, all gates and release-order/version reconciliation. Rebase
onto any accepted newer main and renew review/gates/package after base/version
changes. Preserve #184's centroid/resource and all other owner/service holds.

## Contract and repair

For unit-normal plane x=0, symmetric finite endpoints ±2^127 (f32), ±2^1023
(f64), or ±MAX have exact intersection parameter 1/2. The original native
difference da-db overflows and the clamped quotient collapses to zero.

Keep the existing parallel rejection abs(da-db)<eps, finite-denominator division
and clamp exactly. A nonfinite denominator enters cold recovery. Only finite
da/db and finite nonnegative eps are eligible. All other inputs retain the
original da/denom arithmetic; no checked-input policy, error variant or panic
is added. Negative-zero epsilon remains zero; strictly negative/NaN/infinite
epsilon policies remain unchanged, including legacy degenerate results.

Finite da/db overflow subtraction only with opposite signs. An overflow needs
the smaller magnitude at least the rounding boundary near MAX, well above
subnormal range for the sealed f32/f64 types. Thus both exact native halves
are representable and nonzero. Their difference is finite, nonzero, bounded by
MAX and has the numerator's sign. The reordered ratio lies in [0,1]. Halving
is confined to this path: minimum-subnormal crossings and close same-sign
cancellation retain the old arithmetic. No widening or tolerance change occurs.
The two rounded operations (scaled subtraction and division) admit the
conservative relative error bound gamma3=3u/(1-3u), native u=2^-24 or 2^-53.
This is an external rounding-analysis bound, not a production tolerance or
universal correctly-rounded result claim.

This repair starts at finite computed signed distances. It does not recover
an earlier dot-product/offset overflow, change public Plane construction or
classification, or make later endpoint interpolation range-safe.

## Reproductions and oracle

Original unchanged-main audit: four failures and six controls, debug/release.
Expanded baseline: six range failures and ten controls; repaired sixteen tests
cover both precisions, symmetric/asymmetric overflow, reverse direction/axes,
endpoint/parallel results, close cancellation, minimum-subnormal crossings and
existing nonfinite-distance/invalid-epsilon behavior. A preliminary new invalid-
epsilon control expected zero for 1/0; baseline and arithmetic show the existing
clamp returns one. That expectation is corrected and the unchanged baseline
replayed; its raw output is retained. None of the six valid-input failures is
removed or relaxed.

An external caller links actual published-main original source and candidate
libraries on one coherent provider graph. The stored-bit corpus contains 8192
cases per precision (16384 per O2/O3 profile). In each profile 15107 Option/
output records match the old bits exactly, including invalid policies, while
1277 recovered ratios are finite, within [0,1], and satisfy the exact Fraction
reference bound above. O2/O3 output files are byte-identical. Finite samples do
not establish universal rounding correctness; exact analytical regressions and
unchanged-source contracts remain separate evidence.

## Caller and resource measurements

One ARM64 Mac, Rust 1.97, standard O2/O3 profiles and default LTO/codegen settings.
Seven alternating baseline/candidate samples per native precision/context use
500000 direct/indirect calls, or 1024 streaming batches of 1024 segments. Runtime
ordinary axis/non-axis planes and finite overflowing cases preserve complete
black-boxed Option values. Input construction and computed-value validation are
outside timing. No concurrent local build/test workload runs during measurement.

| Profile | Precision | Ordinary caller | Baseline ns/segment | Candidate | Change |
|---|---|---|---:|---:|---:|
| O2 | f32 | direct | 1.200 | 1.346 | +12.24% |
| O2 | f32 | indirect | 1.275 | 1.385 | +8.61% |
| O2 | f32 | stream | 1.223 | 1.339 | +9.44% |
| O2 | f64 | direct | 1.201 | 1.341 | +11.68% |
| O2 | f64 | indirect | 1.266 | 1.398 | +10.42% |
| O2 | f64 | stream | 1.218 | 1.353 | +11.05% |
| O3 | f32 | direct | 1.220 | 1.330 | +9.02% |
| O3 | f32 | indirect | 1.276 | 1.386 | +8.55% |
| O3 | f32 | stream | 1.231 | 1.340 | +8.84% |
| O3 | f64 | direct | 1.201 | 1.343 | +11.77% |
| O3 | f64 | indirect | 1.274 | 1.389 | +9.00% |
| O3 | f64 | stream | 1.220 | 1.350 | +10.66% |

Correct repaired cold medians across direct/indirect/stream contexts:

| Profile | Precision | Candidate ns/segment range |
|---|---|---:|
| O2 | f32 | 2.040–2.787 |
| O2 | f64 | 2.046–2.816 |
| O3 | f32 | 2.046–2.786 |
| O3 | f64 | 2.239–2.543 |

The baseline's cold output is wrong and is not a throughput target. Every
sampled caller requests zero allocation calls/bytes; setup is excluded. Ordinary
costs are roughly 0.11–0.15 ns/segment (+8.5–12.3%) here and remain explicit for
independent acceptance. No unavoidable, globally optimal, zero-overhead, RSS,
cache, universal stack or whole-mesh guarantee follows. Raw samples and standard
disassembly are retained; no forced-inline, target-CPU, optimizer threshold,
lint, test budget or suppression change is used.

## Separate parallel-policy follow-up

GAIA-038 records existing custom eps=0 behavior: equal off-plane signed
values +1/+1 or -1/-1 in f32/f64 yield Some(1) or Some(0), since abs(0)<0
is false. Actual old/new caller probes reproduce all four. The default entry
point documents parallel rejection; the custom-zero policy needs explicit
resolution in a separate repair. GAIA-037 preserves this finite-denominator
behavior and does not claim to close that gap.

## Final acceptance

Source/normalized/fresh-consumer, strict diagnostics, current supply-chain,
published-0.6.8 patch API, complete-history conformance and hosted gates must
pass on the final head. Independent review and release-order reconciliation
remain mandatory before merge/publication. Hosted Miri selects only two
GhostCell tests; plane/projection/normal coverage is UNVERIFIED. Keep
GAIA-NEXTEST-PIPE, Horae first publication and all owner/service holds intact.
