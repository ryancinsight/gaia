# Gaia provider compatibility audit

## Moirai0.9 adoption [patch]

Gaia0.5.2 excludes the additive Moirai0.9 provider through its `<0.9.0` bound.
Metis's Rust page implementation cannot resolve its canonical Git graph while
that bound remains. Gaia0.5.3 admits0.7/0.8/0.9 through the existing scheduling
surface; source geometry algorithms and all unrelated provider pins retain
exactly their preceding identities.

Actual verification:1,159 mesh/topology/property tests pass with one existing
ignored test. Nextest reports one leaky test; its concrete owner still requires
inspection. Protected provider publication and Metis consumer verification
remain pending. Active work lives in [backlog.md](backlog.md#gaia-moirai-09-coherence).
