# Finite renderer projection recovery

Scope: GAIA-033/034, patch candidate 0.6.8, based on current main a9f3dd3.
No public signature, invalid-input policy, provider, test budget or compiler
setting changes. The original audit checkout and failing debug/release logs
remain external evidence; expanded unchanged-source tests fail four cases and
pass five controls before the repair.

## Numerical decision

For finite 0 < n < f, set d=f-n. The exact depth coefficients are
-(f+n)/d and -2fn/d. At n=2^512,f=2^513 they are -3 and -2^514;
the previous 2*f*n overflows before multiplication by 1/(n-f).
Tiny frusta can instead zero that product or overflow the reciprocal.

Keep each original finite, nonzero depth coefficient. Only a collapsed or
nonfinite result enters the cold recovery: r=f/(f-n), scale=1-2r,
offset=-2*(n*r). The positive difference is representable; for distinct native
positive floats, spacing bounds r (at most 2^53 for f64). The distance product
is scaled first, avoiding an overflow unrelated to the final coefficient.
A coefficient whose actual magnitude exceeds native range may remain infinite.
This is not a universal correctly-rounded matrix/renderer guarantee.

For clip=(w,2w,0,w), w=2^-1074, the exact NDC is (1,2,0).
Reciprocal-first evaluation overflows. Normal denominators cannot overflow
1/w, so the public fast path routes only zero/subnormal w to a cold helper.
The helper retains signed-zero rejection and original multiplication whenever
1/w is finite or a vector component is nonfinite; otherwise it divides finite
components by w directly. No epsilon, widening, clamp or nonfinite rejection
policy is introduced.

## Value coverage

Nine focused tests cover original failures, signed subnormal quotients,
signed zero, ordinary coefficient/result bits, nonfinite behavior and existing
frustum rejection/positive-infinite-aspect behavior. Every binary near exponent
from 2^-1074 through 2^1020 (2095 cases), with far=2*near, checks exact
coefficients -3/-4near and both near/far NDC endpoints -1/+1.
Finite inputs outside these fixtures and matrix-vector overflow elsewhere
remain subject to native arithmetic; the repair does not claim universal
full-range rendering.

## Original author cost and memory

The external measurement links actual unchanged 0.6.7 and candidate libraries
in one consumer, with opaque function-pointer wrappers and complete black-boxed
outputs. Seven alternating paired samples per standard opt2/3 profile use
500000 calls each on one ARM64 Mac. Median perspective costs are
6.391 -> 6.439 ns (opt2) and 6.525 -> 6.623 ns (opt3), +0.76%/+1.49%.
Median NDC costs are 3.168 -> 3.321 ns and 3.101 -> 3.252 ns,
+4.83%/+4.88%. Retain this measured overhead; it is not a no-regression claim.
Cold recovered paths are measured separately. Sampled allocation calls and
allocated bytes are zero; this is not a universal resource bound or a
whole-renderer throughput measurement. No custom optimizer thresholds, forced
inlining, lint waivers or test-budget changes are used.

## Acceptance

Full native computation passes 1233 tests with one existing skip; doctests pass
57 with one existing ignore. Strict diagnostics and a real patch comparison
against published 0.6.7 pass 223 applicable API checks (30 inapplicable).
Normalized packaging, fresh-consumer/3MF, normal hosted gates and independent
exact-head review remain acceptance checks before merge/publication. The
GAIA-NEXTEST-PIPE follow-up remains open; a completed run does not close it.

## Caller-cost renewal after independent review

The original author caller above reports about 4.8–4.9% ordinary NDC overhead.
The independent fixed caller reports 0.73–0.80 ns, about 33–38%, at the original
reviewed head. Both contexts remain retained evidence; neither bounds other
callers or whole-renderer throughput.

Generated ARM64 code explains a concrete part of the difference: baseline NDC
is inlined into the independent caller, while the reviewed head calls it out
of line. An inline hint alone removes that call, but the cold helper returning
`Option<Vector3>` still forces a 32-byte option temporary and a second vector
copy in that caller. Returning a vector from the cold helper, with zero-w
rejection in the public function, removes the temporary. The caller's frame
returns from 112 to 80 bytes. A normal `#[inline]` exposes the small fast path;
no forced inlining, LTO/CPU tuning, optimization threshold or gate changes occur.

The denominator guard uses the IEEE binary64 exponent mask derived from the
infinity bit pattern: exactly zero exponent bits mean signed zero or subnormal.
Normal, infinity and NaN inputs take the same arithmetic path as before.
No numerical expression or nonfinite rejection contract changes. The original
4114 independent cases plus 12288 exponent/sign/significand probes produce
byte-identical reviewed/new outputs at standard O2 and O3.

New seven-sample alternating measurements link the exact same independent
caller source and the exact original author source against baseline, reviewed
and optimized libraries. Setup is excluded; complete outputs remain black-boxed.

| Context | Profile | Baseline NDC ns | Reviewed NDC ns | Optimized NDC ns |
|---|---|---:|---:|---:|
| Independent fixed caller | O2 | 2.2402 | 2.9853 | 2.1892 |
| Original author caller | O2 | 3.1641 | 3.3213 | 3.1642 |
| Independent fixed caller | O3 | 2.1416 | 2.9487 | 2.1897 |
| Original author caller | O3 | 3.0930 | 3.2564 | 3.0968 |

The optimized independent caller differs from its co-measured baseline by
-0.051/+0.048 ns at O2/O3; the author caller differs by +0.0001/+0.0038 ns.
These are sampled medians, not an assertion of zero or unavoidable overhead.
Cold correct-path costs remain context-dependent: author cold NDC medians rise
3.107 -> 3.579 ns (O2) and 3.023 -> 3.449 ns (O3), while the independent caller
measures 3.600 -> 2.635 ns and 2.955 -> 2.592 ns. Preserve these observations;
incorrect baseline cold results are not throughput targets. Sampled allocation
calls and bytes remain zero; RSS, universal stack and whole-product bounds are
not established. This changed head requires renewed independent exact-head
review and all current final gates before any merge/publication.

## Miri scope

The configured hosted Miri job executes only two GhostCell Send/Sync tests
(1162 library tests filtered in the original review); it executes zero
projection tests. Projection Miri coverage remains UNVERIFIED. Native numerical
checks and hosted Miri success do not establish that missing coverage.
