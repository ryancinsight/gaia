# Gaia provider compatibility audit

## Moirai 0.9 adoption [patch]

Gaia 0.5.2 excludes the additive Moirai 0.9 provider through its `<0.9.0` bound.
Metis's Rust page implementation cannot resolve its canonical Git graph while
that bound remains. Gaia 0.5.3 admits 0.7/0.8/0.9 through the existing scheduling
surface; source geometry algorithms and all unrelated provider pins retain
exactly their preceding identities.

Actual verification: 1,159 mesh/topology/property tests pass with one existing
ignored test. Nextest reports an output-pipe leak for the nonempty-serpentine mesh test
in the full run; the same computed test exits cleanly on focused reproduction.
The unchanged full suite runs again to check the observed exit condition. Protected provider publication and Metis consumer verification
remain pending. Active work lives in [backlog.md](backlog.md#gaia-moirai-09-coherence).
