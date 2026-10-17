# Native centroid and Newell range recovery

Scope: GAIA-035/036, patch 0.6.9, stacked on projection PR #183 (0.6.8).
The stack parent may change under its renewed independent performance review.
Do not merge this child before that dependency integrates; rebase onto accepted
main and renew exact-head numerical, API, package, hosted and publication-scope
gates after integration. No merge/publication/credential/service action is part
of this repair.

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
bits whenever its accumulated vector is finite. For nonfinite accumulations
with entirely finite input vertices, recompute that anchored sum without any
temporary vertex allocation. Existing norm/tolerance handling then applies,
including cancelling loops and tiny/subnormal triangles returning None.

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

## Caller and allocation measurements

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
Final-head measurements and acceptance must renew after the stack integrates.

## Verification and holds

Full native computation passes 1253 tests with one existing skip; doctests pass
57 with one existing ignore. Twenty regressions pass debug/release. Strict
Clippy and patch API comparison against coherent 0.6.8 pass (223 applicable,
30 inapplicable). Final normalized/fresh consumer, current hosted gates and
independent exact-head review remain acceptance requirements.
The configured Miri job runs two GhostCell tests, not these geometry or projection
regressions. Their Miri coverage is UNVERIFIED. Keep GAIA-NEXTEST-PIPE and all
publication, service and owner holds unchanged.
