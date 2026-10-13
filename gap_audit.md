# Gaia provider compatibility audit

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
