# Native centroid and Newell range recovery

Scope: GAIA-035/036, patch 0.6.9, based on the accepted projection PR #183.
The 0.6.8 parent integrates at 5eca6a6779b062dacd752dfc756e63a384193a71,
with exactly the independently reviewed tree. Its registry archive, source,
crate-gaia-mesh-v0.6.8 tag, main CI and automatic docs verify. This draft is
rebased onto that integration. Renew its numerical, published-baseline API,
package, hosted and publication-scope gates; independent exact-head review and
geometry performance acceptance remain required before merge/publication.

## Centroid decision

The original centroid is ((a+b)+c)*third independently per coordinate. Keep
that expression and all output bits whenever the raw sum is finite. Only a
nonfinite sum invokes native recovery. If any input vertex is nonfinite,
return the entire original point's coordinate results, preserving the existing
nonfinite behavior rather than inventing an invalid-input policy.

Sort a recovered coordinate's three finite inputs. Same-sign inputs use
m+((lo-m)+(hi-m))/3. The differences are finite, with opposite signs, and the
adjustment stays in the endpoints' interval. Equal coordinates return m exactly.
The first prototype divided two grouped same-sign terms separately; the new
boundary grid catches a one-ulp error for three equal large coordinates. That
failed evidence remains retained. No clamp or constant-value special case is used.

For mixed signs, cancel lo+hi first. That sum cannot overflow. If adding m
remains finite, use the original third multiplication; otherwise only two
same-sign terms remain and (lo+hi)/3+m/3 is bounded by 2*MAX/3. This order
preserves small residuals after cancellation of large opposing coordinates.
No widening, arbitrary epsilon, dependency or public interface change occurs.
This is native arithmetic, not a universal correctly-rounded mean guarantee.

The scalar recovery is const-parameterized by coordinate. Fixed three-component
finite checks avoid retaining unused-axis cold calls in projected consumers;
standard generated code computes only the selected axis on the ordinary path.
Do not replace this with a shared whole-point fast guard that couples axes.

## Newell decision

Newell's raw vector equals the closed sum of consecutive vertex cross products.
Translating all vertices by one origin leaves that sum invariant; equivalently,
sum the anchored fan (v_i-o) cross (v_(i+1)-o). Retain the original loop and its
bits whenever its accumulated vector is finite. Only nonfinite accumulations
with entirely finite input vertices use recovery. Existing norm/tolerance
handling then applies, including cancelling loops and tiny/subnormal triangles.

Independent review of the first implementation finds that a native first-origin
subtraction can erase an edge: (1,-L,H),(-1,1,H),(-1,0,H), for native L=2^24/2^53
and H=2^127/2^1023, has exact vector area (0,0,2), yet L+1 rounds to L. Merely
finding a finite fan result does not justify returning a degenerate normal.

Recovery retains each difference's high and low parts with native TwoDiff.
The next independent review finds that computing cross() before compensation
still rounds L²-k to L². Subloops with areas 1,L²-1,-L² invent a positive unit
normal for exact zero; using k=2 gives the wrong winding. Every individual
product in that reviewer fixture is finite and exact.

Recovery now feeds both signed products of every determinant separately into
a native grow-expansion accumulator. The native fused-multiply-add seam also
retains representable product roundoff before accumulation. After the finite
product guard, an exactly zero operand skips the exactly zero fused residual;
nonzero operands with an underflowed product retain the fused path. All four high/low
edge pairings remain. Finite TwoSum transforms retain each addition residual;
there is no pre-rounded determinant or fixed two-word truncation. Eight inline
components are a storage choice: an adaptive spill retains further components.
No float widens and neither the ordinary loop nor degeneracy threshold changes.

Ten additional regressions cover exact-zero/negative-area reviewer fixtures,
k=0..4, all six subloop orders, cyclic origins, all coordinate planes and reverse
winding. Nine separated scales exercise spill storage and retained/zero winding;
L²-(L-1)(L+1)=1 checks fused product residuals. The prior translated-edge and
L²+2-L²/exact-zero controls remain.

Arithmetic still has native range limits. Nonfinite products, residuals or
partial sums return the original result; residuals below the minimum subnormal
cannot be represented. This is not a universal exact-product, arbitrary-range
or correctly-rounded-normal guarantee. The represented reviewer fixtures and
expanded cancellation cases have analytically determined normals above the
unchanged threshold, or exact zero.

The earlier allocation tables describe earlier heads. This repaired recovery
uses inline expansion storage and may allocate on a spill. Renew separate
ordinary, standard cold and multiscale spill allocation/timing measurements;
do not carry forward an unconditional zero-allocation claim.

Nonfinite input vertices preserve the original result. If native anchored
arithmetic itself is nonfinite, retain the original result; this repair does
not claim universal full-range polygon normals, scaled exact products or a new
range/error policy. The proven translated fixtures have finite anchored edges,
products and sums. Original triangle_normal and unrelated normal APIs are unchanged.

## Values and independent oracle

Original audit: four f32/f64 baseline failures with six controls; expanded
pre-fix suite: eight failures/nine controls. Twenty candidate tests cover the
original findings, cancellation permutations, MAX/equal-coordinate bounds,
signed zero, subnormal/small values, all three coordinate planes and complete
legacy behavior for nonfinite vertices. A 1000-triple grid per native precision
checks finite recovered means within endpoints and exact finite legacy bits.

An independent 8000-case stored-bit Fraction oracle checks 5753 finite legacy
results bit-for-bit and 2247 recovered cases. Recovered results are finite,
within their endpoint bounds, and differ from exact stored-coordinate means
by at most gamma8*abs(mean)+2 minimum-subnormal units, gamma8=8u/(1-8u), with
native u=2^-24 or 2^-53. This is an external rounding-analysis check, not a
production tolerance or widening step. Finite samples do not prove all rounding.

## Preserved pre-integration caller and allocation measurements

The external consumer links actual baseline/candidate libraries and computes
in native f32/f64. Seven alternating samples per standard O2/O3 profile use
500000 calls. Whole-point and projected centroid wrappers, triangle Newell and
runtime-slice 32-vertex polygons are separate contexts. Complete outputs remain
black-boxed. Setup is excluded; all sampled allocation calls/bytes are zero.
Earlier prototypes and a concurrent-load run remain retained; this table uses
a run without concurrent local builds/tests.

| Profile | Precision | Caller | Baseline ns | Candidate ns | Median change |
|---|---|---|---:|---:|---:|
| O2 | f32 | centroid | 2.809 | 3.816 | +35.83% |
| O2 | f32 | projected_centroid | 2.421 | 2.473 | +2.13% |
| O2 | f32 | newell | 6.717 | 7.404 | +10.23% |
| O2 | f32 | polygon32 | 23.601 | 23.892 | +1.23% |
| O2 | f64 | centroid | 2.985 | 3.933 | +31.76% |
| O2 | f64 | projected_centroid | 2.441 | 2.594 | +6.26% |
| O2 | f64 | newell | 7.092 | 7.730 | +8.99% |
| O2 | f64 | polygon32 | 27.815 | 27.841 | +0.09% |
| O3 | f32 | centroid | 2.967 | 3.882 | +30.86% |
| O3 | f32 | projected_centroid | 2.459 | 2.603 | +5.88% |
| O3 | f32 | newell | 6.664 | 7.306 | +9.65% |
| O3 | f32 | polygon32 | 22.439 | 22.801 | +1.61% |
| O3 | f64 | centroid | 3.009 | 3.974 | +32.06% |
| O3 | f64 | projected_centroid | 2.424 | 2.631 | +8.56% |
| O3 | f64 | newell | 40.810 | 7.631 | -81.30% |
| O3 | f64 | polygon32 | 27.777 | 27.779 | +0.01% |

The whole-point centroid cost remains about 31–36% in these tiny callers;
projected costs are lower after unused-axis recovery removal. Preserve that
remaining cost for independent performance acceptance; do not call it
unavoidable or a no-regression result. The O3 f64 baseline triangle caller has
a much higher cost in this specific layout; do not infer a general Newell
speedup from that row. 32-vertex slices provide a separate loop context.
No universal timing, RSS/stack, allocation or whole-mesh throughput bound follows.
No forced-inline, LTO/CPU tuning, optimizer-threshold, lint or budget changes occur.
The pre-integration measurements remain preserved. Renew final-head measurements
against the actual 0.6.8 source after integration; acceptance remains independent.

### Post-integration performance blocker

The unchanged harness rebuilt against the actual 0.6.8 source exposes a larger
O3 f64 whole-centroid regression: 3.018 -> 20.161 ns (+568%). A second run of
that same immutable executable repeats 2.996 -> 20.019 ns. Preserve both raw
runs; this is a performance acceptance blocker, not an accepted tradeoff.
The O2 whole-point rows remain +32–36%, projected rows +5–8%, triangle Newell
rows +8–10%, and 32-vertex rows +0.2–1.9% in the initial renewed samples.
The earlier O3 f64 baseline Newell anomaly does not recur: 7.102 -> 7.675 ns.

Those prior-head sampled allocation calls/bytes are zero; repaired expansion
spill measurements renew separately. Ordinary whole-point
code retains three finite guards, with cold calls guarded off for finite sums;
projected code retains only the selected-axis guard/helper. The O3 f64 wrapper
has 98 instructions and a 160-byte stack frame (O2: 95/160). These observations
do not establish the latency's root cause. Keep the exact binary, disassembly,
repeat and earlier contexts for independent review. Do not merge/publish 0.6.9
before this performance blocker and independent exact-head review are resolved.

## Verification and holds

The original range head passes 1253 native tests with one existing skip; doctests pass
57 with one existing ignore. Its twenty regressions pass debug/release. Strict
Clippy and patch API comparison against the coherent 0.6.8 source pass
(223 applicable, 30 inapplicable). Repeat against the actual published archive
after integration. Final normalized/fresh consumer, current hosted gates and
independent exact-head review remain acceptance requirements.
The configured Miri job runs two GhostCell tests, not these geometry or projection
regressions. Their Miri coverage is UNVERIFIED. Keep GAIA-NEXTEST-PIPE and all
publication, service and owner holds unchanged.

### Bounded centroid investigation

The frozen executable itself changes from about 3.9 to 20.1 ns when only its
path length changes. Eight matched-path comparisons retain slow cases in both
original and deferred-multiply binaries (about 21 ns). An equivalent finite
predicate compiles to the same ordinary instructions; hoisting sums increases
the frame and does not repair the slow case. No centroid prototype is accepted.
The independent 3.005 -> 3.940 ns repeat and the retained 3 -> 20 ns samples
therefore coexist as measured placement contexts. The hardware cause is not
established; the 160-byte frame and conditional cold calls alone do not prove it.
All sampled allocations remain zero. Keep the performance acceptance hold;
this investigation neither dismisses the slowdown nor claims unavoidable cost.
