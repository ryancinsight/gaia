# Changelog

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
