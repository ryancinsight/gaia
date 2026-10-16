# Changelog

## [0.6.1] - Unreleased

### Fixed

- [patch] Restore fresh-consumer resolution by advancing the private 3MF ZIP
  dependency from the entirely yanked 2.x series to current 9.0.4. Keep
  defaults disabled and Deflate enabled; Gaia's public API and first-party
  provider identities remain unchanged. CI now resolves a lock-free external
  consumer and checks its real 3MF geometry with independent ZIP/XML readers.

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
