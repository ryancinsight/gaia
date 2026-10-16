# Gaia provider compatibility audit

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
