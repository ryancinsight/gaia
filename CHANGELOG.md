# Changelog

## [0.6.6] - Unreleased

### Fixed

- [patch] Reject finite-box misses whose distinct slab intervals collapse to
  equal native finite distances. Check exact stored-coordinate interval order
  at ties, preserving inclusive genuine contact and the existing returned
  distance bits. The existing infinity-collapse repair remains in effect.
- [patch] Certify origin-containing and axis-aligned candidates directly;
  compare same-origin/same-denominator endpoint ratios by their bounds after
  exact sign normalization. Other ambiguous ties use the shared exact predicate.
  NaN/infinite-bound policy, range representation and all providers are unchanged.

- [patch] Keep the entire tie-only classification decision cold and non-inlined,
  with the public slab kernel inlineable, to repair an ordinary untied caller
  regression under standard consumer flags. Preserve all numerical expressions,
  exact certificates and native tuples. Add mixed-exponent finite true-contact
  controls and document that some finite ties allocate and take over a microsecond
  in sampled exact-predicate cases; cheap contacts do not bound the general cost.

### Sequencing

PR #180 publishes 0.6.5 at 0e2f98a22dca3911843fae27cd409807872b33c2.
This isolated followup bases on that verified integration and requires its own
independent exact-head review, applicable gates and current publication-scope
assessment before any 0.6.6 merge or publication.

## [0.6.5] - 2026-10-16

### Fixed

- [patch] Reject proven finite-box misses when native positive entry and exit
  distances collapse to infinity. Use at most five exact stored-coordinate
  ratio comparisons through the existing predicate provider; retain true hits
  with native infinite distances. Keep NaN/infinite-bound behavior, public API,
  physical tolerances and provider versions unchanged.
- [patch] Record 456 pre-existing finite rounded-endpoint false hits as GAIA-031;
  this bounded overflow repair does not fix or certify those cases.

### Sequencing

PR #180 is independently accepted at bf1379266f0d9374bbde386de4b25f4918571cc6
and merges at 0e2f98a22dca3911843fae27cd409807872b33c2 with the same reviewed
tree. Sole 0.6.5 actual archive/VCS/all 420 Rust files/original manifest/tag,
main CI and authorized automatic docs verify. Independent cost samples reach
19% ordinary overhead, 5.35 us exceptional latency and 3104 requested live heap
bytes with none retained; ordinary samples allocate nothing. These are warmed
local ARM64 fixture measurements, not formal bounds or whole-mesh throughput.

## [0.6.4] - 2026-10-16

### Fixed

- [patch] Recover representable ray/AABB slab distances when a reciprocal or
  reciprocal product overflows, using native division for nonfinite products.
  Preserve zero boundary distances for subnormal direction components. Reject
  empty/inverted boxes and intervals strictly behind the ray origin, while
  preserving signed entry distances for origins inside the box and inclusive
  contact at the origin. No direction floor, scalar widening or tolerance
  change is introduced. Distances outside the scalar range and NaN bounds
  remain a separate checked-contract design item (GAIA-029).
- [patch] Keep the exceptional quotient in a cold, non-inlined generic helper,
  preventing speculative fallback divisions on ordinary finite-product slabs.
  Preserve every operand, selection decision and returned interval.

### Integration evidence — 2026-10-16

GAIA-028 closes through PR #179 at
`bcd3d50e78eb821f2cc189752e4b0c23f2778af9`, with the independently accepted
bf55af4 tree. Main CI, automatic API release gate/publication and authorized
Pages deployment pass. Only Gaia 0.6.4 publishes; the actual registry archive
checksum is `8912137124c5ae691724e4ffc42c2e1bdfacf294d1253fd52a57102d37bd275c`.
All 417 packaged Rust files, original manifest and VCS match the merge; its
release tag targets the same commit. GAIA-029/030/031 and the output-pipe
followup stay separate. No settings, credentials or other service change.

## [0.6.3] - 2026-10-16

### Fixed
- [patch] Preserve exact signs for every finite stored coordinate across all
  orientation, incircle and insphere wrappers, including overflow, underflow
  and cyclic-order inconsistencies. Keep both existing 3-D sign conventions.
- [patch] Keep Delaunay tetrahedron winding/circumsphere and CSG projected
  degeneracy decisions correct on finite extreme-exponent fixtures.

### Changed
- [patch] Share the existing adaptive provider in its proven exponent range and
  a bounded exact-integer sign fallback elsewhere (ADR 0009). Use current private
  num-bigint 0.5.1 with defaults disabled; public provider identities are unchanged.
  Exact shared dyadic-factor removal reduces integer work without rescaling or
  rounding any float. Physical tolerances and native constructions stay unchanged.
- [patch] Preserve legacy nonfinite-input behavior pending GAIA-027's separate
  versioned contract. The finite-input repair adds no enum case, panic or error API.

### Integration evidence — 2026-10-16

GAIA-026 closes through PR #178 at `9e494758dd7d906f753abcea4378005e3501dd36`.
The integrated tree matches the accepted exact-head independent review
(`418a2e4`); main native CI and the automatic release/API gate pass. The actual
0.6.3 registry archive checksum is
`92679b8095428f14b7e9d32783cfea4bc17132cd8d97c6046c69e608aeb63f16`;
all 416 packaged Rust files and original manifest match the merge. The release
tag targets that merge and authorized Pages deployment succeeds. Only Gaia
0.6.3 publishes. Invalid predicate inputs remain GAIA-027; the output-pipe
followup remains open. Independent validation adds 11,346 rational oracle
cases in both build modes; this is not exhaustive IEEE enumeration or a
whole-provider audit.

## [0.6.2] - 2026-10-16

### Fixed

- [patch] Compare the doubled triangle area in `Plane::from_three_points`
  with the squared scalar tolerance, preserving resolved small triangles in
  both precisions. Reject non-finite coordinates and unrepresentable plane
  arithmetic instead of returning a plane with NaN or zero normals.
  Reject exact collinearity through the canonical original-point orientation
  predicates, so cross-product cancellation cannot invent an arbitrary plane.
  Reject exactly collinear stored points despite native cancellation noise;
  certify ordinary nonzero minors with native arithmetic and evaluate ambiguous
  cases using full-exponent exact dyadic integer sums. Preserve valid finite
  high-aspect triangles in both windings. Plane coefficients remain in their
  native precision; the accepted ZIP 9.0.4/provider graph remains unchanged.
  Try every cheap projected-minor certificate before exact fallback, avoiding
  expensive integer work when one projection is noisy but another resolves
  the triangle; cover the chosen path without timing-dependent assertions.

- [patch] Restore fresh source consumers after Moirai PR #625 advances the
  provider workspace to 0.10.0. Advance both internal runtime declarations and
  their nine lock identities to the published current cohort; retain disabled
  defaults and the parallel feature. No Gaia Rust source or public provider
  identity changes. The normal published API comparison and stricter patch
  diagnostic pass; identical external generic code retains the exposed
  Eunomia 0.10 / Leto 0.46 contract in published and source consumers.

## [0.6.1] - Unreleased

### Fixed

- [patch] Restore fresh-consumer resolution by advancing the private 3MF ZIP
  dependency from the entirely yanked 2.x series to current 9.0.4. Keep
  defaults disabled and Deflate enabled; Gaia's public API and first-party
  provider identities remain unchanged. CI now resolves a lock-free external
  consumer and checks its real 3MF geometry with independent ZIP/XML readers.

### Integration evidence — 2026-10-16

GAIA-025 closes through PR #177 and gaia-mesh 0.6.1. The normal merge's tree
matches the exact-head independent review and all required checks. The actual
registry archive matches its index checksum and merge-commit VCS/source receipt;
the release tag and authorized Pages deployment are verified. Only Gaia ships.
The authentic unchanged locked API comparison passes independently (196 checks;
stricter patch diagnostic 223). Raw fresh 0.6.0 baseline resolution remains
inconclusive because ZIP 2.x is yanked; workflow tolerance is not an API verdict.

## [0.6.0] - Unreleased

### Breaking

- [major] Move the public `Scalar: RealField` contract to Eunomia 0.10 and
  public geometry types to Leto 0.46. Generic consumers using the old Eunomia
  0.9 bound must select the new provider identity and advance Gaia to 0.6.
  ADR0008 records the boundary; the external-consumer regression reproduces it.
- Require the coherent Eunomia 0.10, Leto 0.46, Aequitas 0.3, memory 0.10 and
  Moirai 0.9.3 cohort through existing Git sources/features/defaults. This also
  fixes normalized package verification against the published providers.
  Geometry kernels, precision tolerances and all 166 registry records remain
  unchanged.

## [0.5.3] - Unreleased

### Fixed

- [patch] Admit Moirai 0.9 through the existing scheduling API in runtime and
  development dependencies (`>=0.7.0, <0.10.0`). The standalone lock records
  the actual published Rust page provider revision; unrelated provider pins
  retain their existing identities. Geometry and public APIs remain unchanged.


## [0.5.2] - Unreleased

### Fixed

- Admit the API-compatible Moirai 0.8 provider alongside 0.7 (`>=0.7.0, <0.9.0`)
  in both runtime and development dependencies. Metis's shared Git source can
  now resolve one provider identity rather than fail on Gaia's former 0.7-only
  range. No geometry, scheduling API or scalar implementation changes.
- The standalone lock binds the integration to Moirai's pushed Rust browser and
  macOS scope-wait correction. Unrelated provider revisions remain pinned.
- Preserve the existing explicit narrowing contract through Eunomia's primitive
  `CastFrom` seam and compare precision-specific tolerances after exact widening
  in the test. IEEE 754 midpoint/subnormal/signed-zero/Infinity bit oracles replace
  two bare casts without changing geometry tolerances or precision semantics.

### Integration evidence — 2026-10-12

Gaia PR [169](https://github.com/ryancinsight/gaia/pull/169) merges at
`2279ae294ff76bc925a18c65a152bfe64f6e13f0`. Metis PR
[493](https://github.com/ryancinsight/metis/pull/493) consumes Gaia 0.5.2
and one Moirai 0.8.2 Git identity in its frozen workspace/fuzz graphs; unrelated
Metis provider pins remain unchanged. The consumer passes its protected
Windows, Apple, API, lock and parser gates. The provider-graph backlog item is
complete; broader scalar conversion and geometry work keeps its existing owners.
