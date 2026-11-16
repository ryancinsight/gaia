# ADR 0010: Gaia release scope for 1.0

- Status: Accepted
- Change class: [arch]
- Item: GAIA-RELEASE-LADDER

## Context

Gaia (`gaia-mesh`) is the stack's geometry, meshing, NURBS, and topology
provider. The board carries some thirty open items across exact-predicate
robustness, the `T: Scalar` seam migration, verification gates, docs, and
file-size debt, with statuses ranging from blocked correctness repairs to
incrementally delivered hardening. No milestone ladder orders them toward a
stable 1.0, and no record states what 1.0 freezes or leaves out.

## Decision

Adopt a seven-rung ladder to 1.0. The ladder opens `backlog.md`; every open
item carries exactly one `milestone:` assignment.

- **M-robustness**: exact-predicate and range correctness defects are
  repaired (GAIA-037, GAIA-038, GAIA-033, GAIA-034, GAIA-032, GAIA-029,
  GAIA-027). Exit: each item's acceptance oracle passes at head with
  renewed independent review; no open correctness item in the predicate
  surface.
- **M-seam**: the `T: Scalar` precision contract completes (GAIA-003) and
  rational NURBS weights are validated at the construction boundary
  (GAIA-021). Exit: zero internal `Real` sites outside the intentional f64
  boundary, cast ratchets at zero, and the NURBS error variants typed and
  migration-documented.
- **M-coherence**: provider coherence and test hygiene hold
  (GAIA-MOIRAI-09-COHERENCE, GAIA-NEXTEST-PIPE, GAIA-040). Exit: fresh
  consumers resolve one coherent provider set, the suite exits without
  leaked output pipes, and no test shares a fixed temp path.
- **M-gates**: verification gates are mechanized (GAIA-005, GAIA-007,
  GAIA-008, GAIA-012, GAIA-016, GAIA-017, GAIA-018, GAIA-019, GAIA-020,
  GAIA-039). Exit: doctests runnable, Miri covers the `unsafe` items,
  supply-chain/semver/float-comparison/bench budgets run in CI, and
  unformatted code cannot reach main.
- **M-docs**: the architecture is recorded and taught (GAIA-010, GAIA-011,
  GAIA-015, GAIA-022). Exit: decided architecture has ADRs, the book
  derives the theory, refinement/remeshing is delivered or declared a
  non-goal in the README, and the legacy checklist is consolidated.
- **M-debt**: file-size and lint debt burns down (GAIA-013,
  GAIA-RESCUE-QUEUE) with Edition 2024 already landed (GAIA-014, exit
  evidence). Exit: every source file at or under the 500-line target or
  recorded as a cohesion exception, with ratchet counts monotone
  decreasing.
- **M-1.0** (1.0.0): stable geometry release (**authority-gated**). Exit:
  `cargo semver-checks`, publish dry-run, `cargo doc`, the full
  nextest/doctest/Miri suite green, and zero open correctness items.
  **Veto point:** release authority sets the API-freeze contents; this
  milestone never forces `publish = true` without it.
- **M-ecosystem**: consumer-owned migrations against the stable surface
  (ares contact geometry, CFDrs meshes, kwavers acoustic meshes). Exit:
  consumer receipts on their own boards; does not gate 1.0.

### Veto points

1. The 1.0 API-freeze contents (no source determines them; release
   authority sets them).
2. The GAIA-015 capability decision: constrained 3-D refinement and
   remeshing land behind a consumer-driven contract or are declared
   non-goals in the README; this ADR does not pick it.

## Rejected alternatives

- Freezing 1.0 on the untyped predicate inputs: GAIA-027/GAIA-029 record
  that NaN maps silently today; a stability promise cannot cover a
  contract that cannot distinguish invalid input.
- Carrying refinement/remeshing as silent stretch goals: the 2026-08-04
  audit records them as absent; they enter only through GAIA-015's
  explicit decision.

## Overturning evidence

A consumer failure traceable to a frozen API, or a new audit finding in
the predicate surface, reopens the affected rung. Any saved regression
failing at head overturns the corresponding exit.
