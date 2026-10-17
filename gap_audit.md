# Gaia provider compatibility audit

## GAIA-035 / GAIA-036 — Native normal-range repair [patch]

A separate 0.6.9 draft now bases on #183's accepted 0.6.8 integration, 5eca6a6.
Pre-integration source/evidence remain preserved; final gates and independent
exact-head geometry/performance acceptance renew after rebasing. Finite
centroid sum overflow is recovered with
native sorted cancellation or same-sign median anchoring; Newell retries only
nonfinite accumulations using compensated origin-relative edges and fan terms
for finite vertices. Independent P2 review exposes erased small edges in the
first native fan; TwoDiff residuals and TwoSum accumulation retain those edges
and L²+2-L² cancellation without changing thresholds. Six new regressions cover
both precisions, winding/origin/axis changes and exact-zero cancellation.
Ordinary bits, complete nonfinite results, tolerance and signed-zero/subnormal
behavior remain covered. The 20 regressions, 2000 bounds-grid triples and
8000 exact-rational records pass; source-parent patch API and native/docs gates
pass. Post-integration O3 f64 whole-centroid cost is 3.018 -> 20.161 ns,
repeated 2.996 -> 20.019 ns on the same binary; acceptance remains blocked.
No root cause or unavoidable-cost claim follows from the preserved codegen. See
`docs/verification/normal-range.md` for proof limits and all caller contexts.
The configured two-test GhostCell Miri job does not cover these functions;
geometry/projection Miri and GAIA-NEXTEST-PIPE remain unresolved scope.

## GAIA-034 caller-cost renewal [patch]

Independent review at the original projection head has no actionable code
finding but measures a larger NDC caller cost than the original author context.
Keep that acceptance issue explicit. A normal inline hint, exact exponent-bit
classification and vector-only cold return remove the observed option temporary
and reduce the sampled ordinary gap in both preserved callers. No arithmetic,
policy, gate or compiler threshold changes. Independent exact-head renewal at
17068c5 accepts both frozen caller contexts and the disclosed cold tradeoff.
Normal merge 5eca6a6 has the identical reviewed tree; sole 0.6.8 registry source,
archive/tag, main CI and automatic docs verify. Hosted Miri covers two GhostCell
tests and zero projection tests; no broader performance or Miri claim follows.

## GAIA-033 / GAIA-034 — Renderer finite range [patch]

The original bounded audit reproduces two valid finite-input failures on
unchanged main. This separate 0.6.8 repair retains finite nonzero perspective
coefficient bits and ordinary reciprocal/multiplication results, recovering
only collapsed/overflowed depth coefficients and finite-vector quotients after
reciprocal overflow. Zero and nonfinite input behavior remains unchanged.
The mathematical fixture, native bounds, measured costs and remaining scope
are recorded in `docs/verification/projection-range.md`. Full native values
pass 1233/one skip; docs pass 57/one ignore. These completions do not resolve
GAIA-NEXTEST-PIPE or establish universal numerical/resource behavior.

## GAIA-032 finite AABB center range [patch]

The documented arithmetic centre of a finite ordered box is bounded by its
endpoints. A singleton at MAX therefore has center MAX, yet the unchanged
native sum/half returns infinity. Positive-width extreme intervals fail too.
Four analytical tests fail on unchanged 0.6.5 and accepted #181 in debug and
standard opt2/3; four controls pass. The actual main integration 012fa6e1 has the
identical #181 tree and AABB source. No active PR owns this subsystem.

Keep native sum/half bits on ordinary coordinates; recover only overflowing
coordinates of fully finite ordered boxes using exact halves of large same-sign
normal endpoints. Preserve the complete old result for nonfinite/inverted boxes.
Always halving would lose least-subnormal singletons. No invalid-input policy,
widening, epsilon, dependency or API change applies. Ordinary means are not
promised universally correctly rounded; whole-mesh callers require separate
measurement.

The independent review of the original draft finds P2 shared point-guard coupling and unused
axis work. Check raw sums before scaling for the full-point fast return, then
return independent scalar recovery results with normal inline hints. Strict
infinity comparisons express finite/ordered eligibility. ARM64 f32/f64 x/y/z
ordinary projections at opt2/3 each load just two endpoints and execute one add
and multiply without a call. Forced inlining fails the existing Clippy gate;
no lint waiver or custom compiler threshold is used in the selected repair.

The original eight tests preserve four baseline failures/four controls; all
repaired debug/opt2/opt3 clients pass. Every mode replays 60108 boxes (30762 author
and 29346 independent review cases) with whole-point and projected outputs
bit-identical to the prior fixed head. Exact author overflow oracles, signed
zero, subnormal, nonfinite and inverted coverage remain intact. Local computed
results: 1224 native passes/one skip; 57 doctests/one ignore; strict diagnostics,
supply-chain/hygiene and patch API checks against published 0.6.6 and old #182
pass (223/30 instrument skips each). The independent native run reports one
output-handle LEAK despite exit zero; retain its named-test/log evidence under
GAIA-NEXTEST-PIPE. A subsequent no-leak-line run does not establish a cause or
close that followup; do not describe the entire native run as clean.

Seven alternating baseline/old/repair samples use standard Rust 1.97 opt2/3,
direct/opaque/indirect callers and axis/full-point streams. Ordinary full-point
repair range is 0.82-1.40 ns; maximum paired baseline overhead is 26.66%/0.184 ns,
versus the original roughly 0.65 ns overhead. Direct f64 opt3 is 0.82 ns versus
old 1.38/base 0.77; f64 stream-axis is 0.41 versus old 0.83/base 0.40. The f32
stream-axis tradeoff remains 0.37 versus old 0.78/base 0.20 ns (92% relative).
Cold repair range is 2.94-3.48 ns. All sampled paths allocate zero requested heap;
these are warmed ARM64 primitive samples, not universal latency, RSS/stack/cache,
whole-mesh or formal resource bounds. Final package/normalized consumer/dry-run,
normal push/hosted gates and independent exact-head re-review remain required
before a fresh sole-0.6.7 lifecycle assessment.

## GAIA-031 finite rounded ray/slab ties [patch]

The ray/AABB method promises an intersection, with `None` for a miss. No
conservative-candidate allowance is documented. The 456 saved finite ties have
strictly disjoint exact stored-coordinate intervals; 228 boxes have positive
width on all axes. Near 0.5, a stored x=y ray cannot reach unequal point slabs
or disjoint positive-width x/y intervals. Parent #180 returns an equal finite
interval anyway. Four analytical regressions fail in both build modes while
real contacts pass. This is a classification defect under the existing contract,
separate from the quality of the approximate distance representation.

The isolated 0.6.6 candidate extends the already-established exact finite-box
postfilter to native endpoint ties. It certifies contained origins and cardinal
rays without determinant work. Endpoint sign normalization is exact; common
origins/denominators cancel, allowing direct bound order. Other ambiguous ratios
use the existing exact predicate seam, with at most five calls. True-hit output
bits, nonfinite-bound behavior and unrepresentable-distance policy are preserved.
Unequal native intervals are outside this bounded change; it does not establish
universal exact classification or a conservative/correctly-rounded enclosure.

PR #180 publishes 0.6.5 at 0e2f98a22dca3911843fae27cd409807872b33c2; the
reviewed tree, actual archive/VCS/420 Rust files/manifest/tag, main CI and docs
verify. This isolated followup requires its own independent review and gates.

### GAIA-031 independent performance finding and bounded repair

The first independent review demonstrates an ordinary untied f64 axis call
returning the same `Some((2,4))` but taking about 8.6 ns versus 3.2 ns in a
saved executable. Freshly linked identical assembly can measure about 3.5 ns;
this is an artifact/caller-sensitive result, not a universal slowdown claim.
An inline hint alone leaves another slow context. Keep the complete tie-only
decision cold and non-inlined, with the public slab kernel inlineable. Standard
level-2/3 direct, opaque-wrapper and function-pointer codegen restores the
observed double method frame from 192 to 144 bytes, retaining three reciprocal
divisions. The frame observation alone does not establish the full timing cause.

Numerical expressions and every exact certificate are unchanged. All three
corpora (13502/16592/10224 cases) retain original #181 outputs in debug and both
optimized client modes. Add two true-contact value tests with origin y=eta,
x=0.5, y in [0.5,next(0.5)] and eta the least subnormal. Such finite contacts
can require allocating full-exponent predicate work: the independent review
samples about 1.1-1.3 us, 35 allocation calls and 384/2088 requested live bytes,
zero retained. The cheap roughly 12 ns contact fixtures are not a general
latency/allocation bound. Repaired exact-head gates and independent re-review
remain required; no flag, provider, budget, NaN or range-policy changes apply.

## Eunomia 0.10 / Leto 0.46 public boundary [major]

Gaia175's provider update at `273eb814` keeps Gaia 0.5.3 and Leto ^0.45.
Standard `cargo publish --dry-run --locked -p gaia-mesh` fails with 149 type
errors: the normalized graph combines Gaia's Eunomia 0.10 with published
Leto 0.45.2's Eunomia 0.9.2. The Git-locked build alone hides that mismatch.

Gaia's public Scalar trait extends Eunomia::RealField, and its public geometry
aliases/mesh fields expose Leto types. An unchanged external generic consumer
builds and runs on published Gaia 0.5.3/Eunomia 0.9.2. It fails E0277 against
the coherent upgraded package because Scalar no longer supplies the old
RealField bound. The traits are nominally different even when their source
spelling matches. The API scanner's 196 passing checks do not detect this
external identity loss; the real consumer establishes the breaking boundary.

Version Gaia at 0.6 and require Leto ^0.46, retaining the existing Eunomia 0.10,
Aequitas 0.3 and memory 0.10 admissions. Moirai runtime/dev floors admit 0.9.3.
The canonical lock retains all 166 registry records; production Rust,
features/defaults, geometry tolerances and test budgets are unchanged.

Reproduce after a clean commit:

```sh
cargo +1.97.0 publish --dry-run --locked -p gaia-mesh
python3 scripts/check_provider_boundary.py --evidence-dir target/provider-boundary-proof
```

The probe requires the normalized 0.6 package and runs identical source in
real registry-backed baseline/current consumers. A successful probe means
the old consumer passes and the upgraded public-bound case fails with E0277.
It adds no production source replacement, old-provider fallback or alias.
See [ADR0008](docs/adr/0008-eunomia-010-provider-boundary.md) and
[GAIA-EUNOMIA-010](backlog.md#gaia-eunomia-010). Final-head gates/review remain
required before integration/publication; the existing checks remain intact.

## Moirai 0.9 adoption [patch]

Gaia 0.5.2 excludes the additive Moirai 0.9 provider through its `<0.9.0` bound.
Metis's Rust page implementation cannot resolve its canonical Git graph while
that bound remains. Gaia 0.5.3 admits 0.7/0.8/0.9 through the existing scheduling
surface; source geometry algorithms and all unrelated provider pins retain
exactly their preceding identities.

Actual verification: 1,159 mesh/topology/property tests pass with one existing
ignored test in each of two candidate runs. Nextest reports one output-pipe
leak per run, in different tests: nonempty-serpentine mesh and matching-width
sweep. Focused serpentine and the preceding committed baseline's full suite
exit cleanly. The cause remains unverified; no timeout, workload or assertion
changes suppress the observation. Track it in
[GAIA-NEXTEST-PIPE](backlog.md#gaia-nextest-pipe).

Moirai [PR 619](https://github.com/ryancinsight/Moirai/pull/619) passes its
required Windows/browser/API gates and merges at
`7828f5df7cf76b7fdec68695ee8b2a49da8e5ba7`. Gaia's standalone lock records that
default-branch identity; unrelated pins retain their preceding values. Gaia
publication and Metis consumer acceptance remain pending. Active integration
work lives in [backlog.md](backlog.md#gaia-moirai-09-coherence).

The merged-provider rerun passes all 1,159 computed tests in 12.989 seconds
without a reported pipe leak; strict all-target Clippy passes. This observation
does not establish the earlier leaks' cause, so their investigation stays open.
