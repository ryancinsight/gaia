# Backlog

<a id="gaia-stable-qualification"></a>
## GAIA-STABLE-QUALIFICATION — gaia stable-release qualification [patch] — todo
- priority: verification; milestone: M-robustness; basis: `ce7513c375ff`.
- Outcome: Reproduce the plane–segment midpoint error on the current geometry release. Follow the [release scope](docs/adr/0010-release-scope-1-0.md) and `M-robustness`. Planning targets for `gaia-mesh` are 0.8.1, then 0.9.0 proposed for the scalar seam, then 1.0.0. Verification alone can advance the current version.
- Acceptance and sequence: [stable-release checklist](CHECKLIST.md#stable-release-checklist).

<a id="M-robustness"></a>
## M-robustness: exact-predicate and range correctness defects are repaired — needs: none
- exit: each item's acceptance oracle passes at head with renewed independent review; no open correctness item in the predicate surface.
<a id="M-seam"></a>
## M-seam: the `T: Scalar` precision contract completes — needs: none
- exit: zero internal `Real` sites outside the intentional f64 boundary, cast ratchets at zero, NURBS error variants typed and migration-documented.
<a id="M-coherence"></a>
## M-coherence: provider coherence and test hygiene hold — needs: none
- exit: fresh consumers resolve one coherent provider set, the suite exits without leaked pipes, no test shares a fixed temp path.

- [patch] The source lock selects Mnemosyne Arena 0.7.2 and Memory 0.10.2
  for the published Gaia 0.8 provider cohort.
<a id="M-gates"></a>
## M-gates: verification gates are mechanized — needs: none
- exit: doctests runnable, Miri covers the `unsafe` items, supply-chain/semver/float-comparison/bench budgets run in CI, unformatted code cannot reach main.
<a id="M-docs"></a>
## M-docs: the architecture is recorded and taught — needs: none
- exit: decided architecture has ADRs, the book derives the theory, refinement/remeshing delivered or declared a non-goal, checklist consolidated.
<a id="M-debt"></a>
## M-debt: file-size and lint debt burns down — needs: none
- exit: every source file at or under the 500-line target or a recorded cohesion exception; ratchet counts monotone decreasing.
<a id="M-1.0"></a>
## Milestone M-1.0 (1.0.0): stable geometry release (authority-gated) — needs: M-robustness, M-seam, M-coherence, M-gates, M-docs, M-debt
- exit: `cargo semver-checks`, publish dry-run, `cargo doc`, full nextest/doctest/Miri green, zero open correctness items. Veto: release authority sets the API-freeze contents; never forces `publish = true` without it.
<a id="M-ecosystem"></a>
## M-ecosystem: consumer-owned migrations (ares, CFDrs, kwavers) — needs: M-1.0
- exit: consumer receipts on their own boards; does not gate 1.0.
- ladder: [ADR 0010](docs/adr/0010-release-scope-1-0.md); every open item below carries exactly one `milestone:`.

<a id="GAIA-037"></a>
## GAIA-037 — Review the merged finite-range repairs (GAIA-032, GAIA-033, GAIA-034, GAIA-037) [patch]
- status: todo; priority: verification; milestone: M-robustness; needs: none
- outcome: the AABB-center (GAIA-032), perspective-depth (GAIA-033), perspective-quotient (GAIA-034) and plane-segment-parameter (GAIA-037) repairs merged without independent exact-head review, which the M-robustness exit requires. Each repair adds no widening, tolerance, dependency or public API change.
- acceptance: an independent judge posts a verdict per head with the cases each item names; findings are fixed forward. Heads: #182 `240a245` (GAIA-032), #183 `17068c5` (GAIA-033, GAIA-034), #185 `ba78874` (GAIA-037).
- cases GAIA-032: `tests/aabb_center_range.rs` singleton at MAX and positive-width extremes (f32, f64), ordinary, subnormal and inverted controls; the four regressions fail on 0.6.5 and the 0.6.6 source (oracle `scripts/regressions/check_aabb_center_oracle.py`); sampled full-point cost at most 26.66%/0.184 ns, residual f32 stream-axis 0.37 ns against a 0.20 ns baseline.
- cases GAIA-033: depth tests in `tests/projection_range_audit.rs`; near=2^512, far=2^513 gives -3/-2^514 and all 2095 dyadic near exponents map both endpoints and coefficients; ordinary coefficient bits and the existing rejection policy are unchanged.
- cases GAIA-034: `small_w_division_preserves_exact_values_and_signed_zero`, `nonzero_subnormal_perspective_divide_retains_representable_values` and `ordinary_ndc_and_nonfinite_components_keep_original_results` in the same file (w=2^-1074, clip=(w,2w,0,w) gives (1,2,0)); ordinary-caller cost (+0.76% to +4.88% by the author, 33-38% in one independent caller; `docs/verification/projection-range.md`); projection Miri coverage UNVERIFIED, the hosted job runs two GhostCell tests.
- cases GAIA-037: `tests/plane_segment_range_audit.rs` symmetric finite endpoints at native MAX give t=1/2, plus asymmetric, cancellation, subnormal, parallel and nonfinite controls; parallel test, finite-denominator division and clamp unchanged; cost in `docs/verification/plane-segment-range.md`.
- evidence (`9daa782`): #182 `a9f3dd3`, #183 `5eca6a6` and #185 `ea7411d` are merged ancestors of main; the three test files are in the tree; `CI` on main succeeds at `9daa782`; `gh pr view` shows zero reviews and `gh api .../statuses` zero statuses on each head.
- scope: `src/domain/geometry/aabb.rs`, `src/domain/geometry/plane.rs`, `src/application/render/transform.rs`, `tests/aabb_center_range.rs`, `tests/projection_range_audit.rs`, `tests/plane_segment_range_audit.rs`.
- next step: dispatch the judge at current main with the cases above. `CHECKLIST.md:7` owns the GAIA-037 re-run on the 0.8.0 baseline; GAIA-038 owns the custom-zero-epsilon policy; GAIA-NEXTEST-PIPE stays open.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`

<a id="GAIA-038"></a>
## GAIA-038 — Zero-epsilon parallel rejection policy [patch]
- milestone: M-robustness.
- status: blocked; owner: unassigned; target: unassigned; priority: correctness.
- evidence: unit-normal x=0 plane, equal off-plane signed distances +1/+1
  or -1/-1, custom eps=0, f32/f64. Both unchanged 0.6.8 and GAIA-037 return
  Some(1) or Some(0), rather than parallel rejection: abs(0)<0 is false.
- contract: the default entry point documents None for parallel segments;
  establish the custom-zero-epsilon policy explicitly before a separate repair.
  Keep exact zero distinct from minimum-subnormal nonparallel crossings.
- scope: pre-existing finite-denominator behavior, intentionally preserved by
  GAIA-037. No fix or acceptance is claimed by its overflow recovery draft.

<a id="GAIA-035"></a>
## GAIA-035 — Finite triangle centroids and Newell normals (GAIA-036) [patch]
- status: todo; priority: correctness; milestone: M-robustness; needs: none
- outcome: `triangle_centroid` and `newell_normal` in `src/domain/geometry/normal.rs` return the representable result for finite inputs: finite equal large x coordinates give their exact mean, not infinity; translated finite coordinate-plane triangles give their exact unit normal, not NaN (f32, f64). Ordinary bits, nonfinite behavior and thresholds are unchanged.
- acceptance: `tests/normal_range_audit.rs` (20 tests), `tests/normal_determinant_regression.rs` (10) and `tests/normal_translation_regression.rs` (6) from `refs/pull/184/head` pass on a rebase onto current main, the ten determinant tests fail on the head before the repair (PR #184 body), and an independent exact-head review records no finding; the O3 f64 centroid regression 3.018 to 20.161 ns (`docs/verification/normal-range.md` in that ref) is removed or accepted by that review with its measurements recorded.
- evidence (`9daa782`): PR #184 is CLOSED, released by the stale-claim sweep on 2026-11-10; `git ls-remote origin refs/pull/184/head` resolves its head, and its tests, `src/domain/geometry/normal/expansion.rs` and the verification record are absent from main. Review of that head found no new correctness issue in verified scope; merge stayed held for centroid performance and resource acceptance. Holds: `gap_audit.md:32-35`, `CHECKLIST.md:15-19,35`.
- scope: `src/domain/geometry/normal.rs`, `gap_audit.md`, `CHECKLIST.md`.
- next step: centroid slice gaia#245; Newell recovery slice gaia#247 (`tests/normal_newell_range.rs`) covers only sums that overflow: a distant polygon whose sums stay finite keeps its legacy rounding error, which the last slice corrects or this outcome is reworded to accept; the accuracy bound for recovered normals and its test are the slice stacked on it. Remaining from `refs/pull/246/head`: the determinant grid's loop-order permutations and totals -2 and -3, the shared-edge and double-wound cancellation polygons, the minimum-subnormal triangle, and the zero-height, ordinary, all-vertex-nonfinite and `triangle_normal` controls of its `tests/normal_determinant_regression.rs`, `tests/normal_translation_regression.rs` and `tests/normal_range_audit.rs`; verification record rewrite, `gap_audit.md`/`CHECKLIST.md` hold text and this block's deletion, with the centroid and Newell performance clause measured.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`

<a id="GAIA-029"></a>
## GAIA-029 — Specify checked ray/slab range and invalid bounds [minor]
- milestone: M-robustness.
- status: blocked; owner: unclaimed; priority: correctness.
- outcome: A checked contract distinguishes an actual miss from an
  unrepresentable distance or invalid NaN box bound.
- evidence: A ray from MAX toward -MAX reaches a finite point at distance
  2*MAX, outside either native scalar range; legacy results contain infinity.
  Public Aabb fields accept NaN without a construction error. The current
  Option interval cannot distinguish range failure from an actual miss.
- boundary: GAIA-028 repairs representable arithmetic and existing empty/half-line
  semantics only. It does not add an error variant, panic, widening policy or
  claim exact correctly rounded results for every slab calculation.
- next: Select/version a checked API and audit callers before changing the
  legacy overflow/invalid-input result contract. Do not classify overflow as
  an ordinary miss merely to make returned values finite.

<a id="GAIA-027"></a>
## GAIA-027 — Specify invalid predicate inputs [minor]
- milestone: M-robustness.
- status: blocked; owner: unclaimed; priority: correctness.
- outcome: Invalid/nonfinite input is distinguished from exact degeneracy.
- scope: Orientation::from_det, Sign::from_exact_f64 and predicate callers.
- evidence: Both public converters map NaN to the zero variant; their three-way
  result types cannot represent invalid input. No explicit checked contract exists.
- boundary: GAIA-026 repairs finite inputs only. Nonfinite coordinates still
  execute the same expansion calls and converters; no error variant, new panic,
  return type, tolerance or precondition is silently introduced.
- next: Select the public invalid-input contract, classify/version the change,
  and audit caller handling with explicit NaN/infinity computed-value tests.

Shared state and ownership board for `gaia-mesh` (import path `gaia`).
This board is the ordered queue of open work. GAIA-022 folds the legacy
`CHECKLIST.md` execution state into this queue and removes the duplicate.

Schema per item: **outcome**, **priority**, **needs**, **scope**,
**acceptance oracle**, and **next step**; audited items also record **basis**.

Seeded 2026-08-20 by the Atlas gap audit (`atlas-gap-audit`) at
`4980732`. Every item cites the evidence that opened it.

Triage order: correctness → architecture → verification → tightening → feature.


<a id="GAIA-MOIRAI-09-COHERENCE"></a>
## GAIA-MOIRAI-09-COHERENCE — Verify current provider compatibility [patch]
- status: todo; priority: correctness; milestone: M-coherence; needs: none
- outcome: fresh Gaia and Metis consumers resolve one coherent Moirai provider identity through the existing internal scheduling API; Moirai types stay out of Gaia's public fields, signatures, bounds, aliases and re-exports.
- acceptance: at current main (a) a fresh consumer crate that depends on the packaged `gaia-mesh` runs `cargo generate-lockfile`, and its `Cargo.lock` holds one `moirai-runtime` entry and every `moirai-*` entry shares one version and one source (M-coherence exit), (b) `cargo package --locked` and (c) `cargo publish --dry-run --locked` pass in this repository; a second version or source in (a), or a failed command, is the defect.
- evidence (`9daa782`): from tracked files only; no cargo command has run for this item. `Cargo.lock` lists twelve `moirai-*` packages (async, async-macros, core, executor, pal, parallel, quic, runtime, scheduler, sync, transport, utils), all 0.12.0 from `Moirai.git#9e78bb77`; `Cargo.toml:82` and `:88` both require `moirai-runtime` 0.12.0 (PRs #191, #205). The 0.10.0 candidate and its receipts (Gaia 0.6.2, PR #176) are superseded: `git show e8ee1d2:backlog.md`.
- scope: the two `moirai-runtime` lines in `Cargo.toml` and the twelve-package `moirai-*` cohort in `Cargo.lock`.
- next step: run (a) to (c); delete this block citing the run if they pass, otherwise repair. Downstream Metis verification stays on its owner's board.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`

<a id="GAIA-NEXTEST-PIPE"></a>
## GAIA-NEXTEST-PIPE — Explain retained test output pipes [patch]
- milestone: M-coherence.
- status: todo
- priority: verification
- outcome: Every completed test releases its actual output handles.
- scope: Reproduce the unchanged suite's process and output-handle lifetimes.
- acceptance: Identify the owning process/handle and verify the original full suite exits without leaked output pipes; retain all computed assertions and deadlines.
- basis: Two provider-candidate runs pass 1,159 values but report one leak in different tests; focused reproduction and the preceding baseline exit cleanly.
- additional evidence: the GAIA-024 repair passes 1,166 values but once flags
  `cylinder_through_torus_union`; subsequent focused/full repeats exit cleanly.
- bounded evidence: 20 focused repetitions and eight full main-branch suites
  (1159 passes/one skip each) report no leak under unchanged capture/budgets.
  Read-only process/pipe snapshots are preserved; no owning writer/cause or
  production fix is established. These clean runs do not close this item.
- review recurrence: In the original #182 review, the independent native run reports
  1224 passes/one skip and one LEAK for
  `application::csg::coplanar::operations::tests::coplanar_buffers_preserve_packed_triangle_order`
  with exit zero. Preserve the raw report; ownership/cause remains unresolved.
  The component repair's local rerun has the same computed passes and no leak
  line; that observation does not close this item or certify the suite clean.
- next: Capture process/handle evidence around an actual reported leak, then correct the identified owner.


---

<a id="GAIA-003"></a>
## GAIA-003 — Collapse the `Real` alias onto the `Scalar` seam
- status: todo
- priority: architecture
- milestone: M-seam
- needs: none; the predicate seam landed in PR #73 / ADR 0005.
- outcome: `T: Scalar` is the internal precision contract; `Real` remains only as the public default type parameter.
- acceptance: `git grep -w Real -- src` reports no internal site outside the intentional `f64` boundary, which ADR 0007 names as "the CDT and CSG arrangement backends" at the exact-predicate boundary, and no workspace cast `allow` remains; each slice migrates one module family, updates its callers and passes its gate.
- evidence (`9daa782`): 722 `Real` hits in 106 `src/` files (`domain/core/constants.rs` 61, `render/camera.rs` 35, `channel/branching.rs` 31, `channel/venturi.rs` 26, `csg/coplanar/operations.rs` 24), from 848 initially. The `Cargo.toml` comment states 0 production cast sites (not re-measured) and keeps `cast_precision_loss = "allow"` for non-library targets. Delivery narrative: `git show e8ee1d2:backlog.md`.
- scope: `src/domain/core/scalar.rs` and its consumer modules; preserve public defaults. Index-to-float conversion uses eunomia `FloatElement::from_count` and `from_integer`; `Scalar` carries no conversion methods (GAIA-COUNT-CONV-001).
- next step: migrate `render/camera.rs`, then `channel/`, one family per slice; the CDT and CSG arrangement backends stay at `f64` per ADR 0007 and are excluded by path. ADR 0007 does not list those paths: `.to_f64()` occurs in `csg` and `delaunay` only in `arrangement/planar.rs` (8), `clip/polygon2d/geometry.rs` (2) and `dim3/lattice.rs` (13), and ADR 0005 makes the predicate wrappers generic over `T`, so the first slice records the path list in an ADR 0007 revision note.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`

---

<a id="GAIA-005"></a>
## GAIA-005 — Retire the 39 ignored doctests
- milestone: M-gates.

- **Outcome**: the public API's documented examples compile and run, so
  `cargo test --doc` is a real contract gate instead of covering 8 of 47
  examples.
- **Scope**: the 36 ```` ```rust,ignore ```` and 3 ```` ```ignore ```` fences
  across `src/`. Non-goals: ```` ```text ```` diagram blocks (97 of them),
  which are prose, not examples.
- **Acceptance oracle**: `rg -c '```rust,ignore|```ignore' src` reaches 0, or
  each surviving `ignore` is converted to `no_run` / `compile_fail` with a
  stated reason; `cargo test --doc --all-features` reports a runnable count
  matching the public item count in the touched modules.
- **Dependencies**: none. Burns down per module.
- **Risk / change class**: [verification] [docs] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: 36 `rust,ignore` + 3 `ignore` fences vs 4 `rust` + 1 `no_run`;
`CHECKLIST.md` Phase 52 records "All eight runnable doctests pass; 39
additional doctests remain intentionally ignored". Examples that never compile
rot silently — `src/lib.rs:12-19` and `src/domain/core/scalar.rs:46-52` are
both `rust,ignore`.

---

<a id="GAIA-007"></a>
## GAIA-007 — Miri gate for the GhostCell `Send`/`Sync` impls
- milestone: M-gates.

- **Outcome**: the crate's only two `unsafe` items are covered by the
  verification the deeper-gate rule requires for reachable unsafe.
- **Scope**: `src/infrastructure/permission/cell.rs:34` and `:40`;
  a `miri` job in `.github/workflows/ci.yml` under a nightly verification
  toolchain, with the pinned build toolchain unchanged.
- **Acceptance oracle**: `cargo +nightly miri nextest run` (or `miri test`
  over the permission and half-edge modules) is green in CI and covers a test
  that exercises branded aliasing across threads, not only construction.
- **Dependencies**: none.
- **Risk / change class**: [verification] [patch] — S.
- **Status**: todo. Delivered Phase 14 (2026-09-29): two cross-thread tests added to `cell.rs` (Send/Sync theorems documented inline); `miri` CI job added under nightly toolchain with `-Zmiri-strict-provenance`. **Owner**: root.

Evidence: `rg 'unsafe ' src` returns exactly `cell.rs:34` and `cell.rs:40`
(`unsafe impl Send`/`Sync` for `GhostCell`); `.github/workflows/ci.yml` now
includes a `miri` job targeting `infrastructure::permission::cell::tests`.

---

<a id="GAIA-008"></a>
## GAIA-008 — Supply-chain and semver gates
- milestone: M-gates.

- **Outcome**: CI enforces the checks the stack's engineering gates require for
  a published crate: advisory/licence/ban scanning, unused-dependency
  detection, and public-surface semver classification.
- **Scope**: `.github/workflows/ci.yml`, plus a `deny.toml`. Non-goals:
  changing the dependency set; the gates report first, remediation is its own
  item.
- **Acceptance oracle**: `cargo deny check`, `cargo machete`, and
  `cargo semver-checks check-release` run in CI; the semver job is required on
  any PR touching `pub` surface and gates `rust-release.yml`.
- **Dependencies**: none.
- **Risk / change class**: [verification] [security] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `.github/workflows/ci.yml` job `gate` has exactly five steps (fmt,
clippy, nextest, doctests, doc). `CHECKLIST.md` records semver comparisons run
by hand ("196 checks passed, 57 skipped") — a manual sequence performed more
than twice is a mechanization defect.

Progress (Phase 29, 2026-10-02):
- Added root `deny.toml` with RustSec advisories, license policy, crate-ban, and source-policy defaults for Gaia's dependency surface.
- Wired `cargo deny check` and `cargo machete` into `.github/workflows/ci.yml` so the main verification job now enforces supply-chain and unused-dependency gates alongside fmt/clippy/tests/docs.

---

<a id="GAIA-010"></a>
## GAIA-010 — Retroactive ADRs for the decided architecture
- milestone: M-docs.

- **Outcome**: the decisions the README already presents as settled have
  records, so a cold-start agent finds them by index instead of by prose.
- **Scope**: `docs/adr/`, `docs/adr/README.md` (generated by
  `scripts/adr-index.py`). Candidate subjects, each an as-built Accepted ADR
  grounded strictly in current code: the GhostCell brand + slotmap topology
  seam; the `Scalar` sealed-trait precision seam and the `Real` default; the
  exact-predicate boundary and where exactness ends (GWN); the CSG operand
  normalization band `0.5..=10.0`; the GAIA-LINT-1 ratchet as the conformance
  policy. Non-goals: inventing rationale — backfill is trigger-driven, one ADR
  per item that touches its scope, never a speculative sweep.
- **Acceptance oracle**: each new ADR cites its board item and the code it
  describes; `python scripts/adr-index.py check` passes; no ADR restates a
  README paragraph without naming the rejected alternative.
- **Dependencies**: GAIA-003 carries the remaining precision candidate; the
  predicate boundary and GWN band landed as ADRs 0005 and 0004.
- **Risk / change class**: [docs] [patch] — M.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `docs/adr/README.md` indexes five ADRs (0001 channel-path validation,
0002 boundary quality criteria, 0003 indexed CSG repair modules, 0004 GWN
thresholds, 0005 predicate boundary) for a 73 214-line kernel whose README § Core
Architecture presents six distinct architectural decisions as settled.

---

<a id="GAIA-011"></a>
## GAIA-011 — Domain book: teach the geometry, not only the contract
- milestone: M-docs.

- **Outcome**: the book teaches robust computational geometry from the ground
  up — exact predicate arithmetic, the Delaunay empty-circumsphere property
  and Ruppert termination, winding-number membership, manifold/Euler
  invariants — and then maps gaia's abstractions onto that theory. The current
  contract and gallery chapters stay; they become the applied layer.
- **Scope**: `docs/book/SUMMARY.md` and new chapters. Non-goals: migrating
  `CHECKLIST.md` status prose into the book; figures stay generated by
  `src/bin/book_mesh_gallery`, never hand-assembled.
- **Acceptance oracle**: `mdbook test` passes with runnable samples in each new
  chapter; each theory chapter cites a resolved primary reference (Shewchuk
  1997 for the predicates, Ruppert 1995 for refinement, Jacobson et al. for the
  generalized winding number) with a locator, and states its domain of
  validity.
- **Dependencies**: none; the GWN band decision is recorded in ADR 0004.
- **Risk / change class**: [docs] [patch] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `docs/book/SUMMARY.md` lists six chapters totalling 289 lines —
mesh-generation contract, Atlas ownership, gallery, two figure manifests, and
watertightness diagnostics. All describe what gaia guarantees; none derives
why the algorithms hold.

---

<a id="GAIA-012"></a>
## GAIA-012 — Benchmark and example runtime budgets
- milestone: M-gates.

- **Outcome**: the three criterion benches and the CI-safe examples carry
  enforced finite budgets, so a performance regression fails a gate instead of
  being noticed by hand.
- **Scope**: `.github/workflows/ci.yml`, `benches/{csg_performance,
  hex_to_tet_performance, tpms_performance}.rs`. Non-goals: adding benches;
  changing bench workloads to fit a budget (that is instrument tuning).
- **Acceptance oracle**: CI smoke-runs the bench binaries in single-iteration
  mode (`cargo test --benches` / criterion `--test`) inside the committed
  30 s nextest budget, and the CI-safe examples run within it; a committed
  per-binary wall-clock bound exists for full timing runs.
- **Dependencies**: GAIA-006 (example targets must exist before they can be
  budgeted).
- **Risk / change class**: [perf] [verification] [patch] — S.
- **Status**: todo. **Owner**: unclaimed.

Evidence: `.config/nextest.toml` budgets tests only; `.github/workflows/ci.yml`
never invokes a bench or example target beyond `clippy --all-targets`
type-checking, so no bench body has ever been executed by a gate.

---

<a id="GAIA-013"></a>
## GAIA-013 — GAIA-LINT-1 ratchet burn-down and file-size debt
- status: todo
- priority: tightening
- milestone: M-debt
- needs: none
- outcome: the allow entries under `[workspace.lints.clippy]` in `Cargo.toml` are deleted as their counts reach zero, and every `src/` file is at most 500 lines or carries a recorded cohesion exception (ADR 0003 owns CSG repair module ownership).
- acceptance: each increment lowers at least one measured count (allow entries, `cargo clippy --all-targets --all-features -- --force-warn clippy::too_many_lines` emissions, `src/` files over 500 lines) and raises none; an increment that deletes an allow line shows its zero emission count from that command.
- evidence (`9daa782`): 19 `src/` files exceed 500 lines (largest: `csg/arrangement/adversarial_tests.rs` 2312, `csg/arrangement/tests.rs` 1064, `csg/boolean/indexed_tests.rs` 1045, `domain/geometry/predicates.rs` 788). `Cargo.toml` keeps `cast_precision_loss = "allow"` and `too_many_lines = "allow"` with comments that state 0; the clippy emission counts were not re-measured, and the earlier "58 emissions" and "two `unwrap_used` sites" are superseded (`unwrap_used = "deny"`). Delivery narrative: `git show e8ee1d2:backlog.md`.
- scope: `Cargo.toml` `[workspace.lints.clippy]` and the over-500-line files. Non-goal: slicing that breaks domain cohesion.
- next step: run the force-warn command and delete each `allow` whose count is 0, then split `predicates.rs`, `csg/coplanar/operations.rs` and `csg/arrangement/patch.rs` by operation family. GAIA-003 retires the `cast_precision_loss` allow.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`

---

<a id="GAIA-014"></a>
## GAIA-014 — Edition 2024
- milestone: M-debt.
- **Outcome**: the crate builds on edition 2024, gaining
  `unsafe_op_in_unsafe_fn`, let-chains, and the current resolver behaviour the
  stack's other members already assume.
- **Scope**: `Cargo.toml` `edition`, plus whatever `cargo fix --edition`
  surfaces. Non-goals: raising the pinned toolchain (1.97.0 already supports
  it).
- **Acceptance oracle**: `cargo clippy --all-targets --all-features -- -D
  warnings` and the full nextest suite pass on edition 2024 with no new
  ratchet entries; `manual_let_else` (51 allowed) drops as let-chains land.
- **Dependencies**: none.
- **Risk / change class**: [patch] — M.
- **Status**: todo. Delivered with GAIA-013 / Phase 13. **Owner**: root.

Evidence: delivered with [GAIA-013](#GAIA-013): the manifest now uses edition
2024 and resolver 3 with the pinned Rust 1.97.0 toolchain. Strict all-target
Clippy and the full nextest suite pass.

---

<a id="GAIA-015"></a>
## GAIA-015 — Constrained 3-D refinement and remeshing (carried forward)
- milestone: M-docs.

- **Outcome**: the two P1 capability gaps the repo's own 2026-08-04 audit
  recorded are either delivered behind a consumer-driven acceptance contract or
  explicitly declared out of scope in the README so no reader infers them.
- **Scope**: `docs/mesh_library_gap_audit.md` rows "3-D constrained
  refinement" and "Remeshing/repair"; `src/application/delaunay/dim3/`.
  Non-goals: adding a public refinement API before the predicate contract,
  sizing-field contract, feature-protection criteria, and termination gates
  are specified — that gate is already recorded in `CHECKLIST.md` Phase 52.
- **Acceptance oracle**: either a sizing-field + radius-edge refinement loop
  with a proven termination bound and a boundary-feature protection test, or a
  README scope statement naming both as non-goals with the consumer driver
  that would reopen them.
- **Dependencies**: none — the predicate contract landed with GAIA-002
  (PR #73).
- **Risk / change class**: [arch] [minor] — L.
- **Status**: todo. **Owner**: unclaimed.

Evidence: re-verified 2026-08-20 against the current tree —
`rg -li 'remesh|decimat|advancing_front|SizingField' src` returns zero files;
`sliver` appears only in CSG fragment classification and quality *measurement*,
never in an optimization pass. The audit's finding still holds.

---

<a id="GAIA-016"></a>
## GAIA-016 — Pin exact floating-point test values
- milestone: M-gates.
- Status: todo. Delivered Phase 13 (test assertions) + Phase 17 (full discharge). `float_cmp = "allow"` removed from ratchet; 5 production sites carry `#[expect]` with documented reasons (tie-breaking, identity check, edge guard, convex hull pivot); test-code sites covered by `#![cfg_attr(test, expect(clippy::float_cmp))]` in lib.rs. Count: 44 → 25 (P13) → 0 (P17). **Owner**: root.
- priority: verification

<a id="GAIA-017"></a>
## GAIA-017 — Parse float-comparison diagnostics
- milestone: M-gates.
- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: Cargo JSON diagnostics become validated records with workspace-local
  source and macro-expansion chains.
- priority: verification
- needs: GAIA-016
- scope: `scripts/float_cmp_diagnostics.py` and focused tests.
- acceptance: malformed messages fail closed; external packages are excluded;
  local absolute, relative, generated, and expanded sources have exact tests.
- basis: `34f0229`
- next: implement the parser and its value-semantic tests under 400 changed lines.

<a id="GAIA-018"></a>
## GAIA-018 — Reject hidden float-comparison diagnostics
- milestone: M-gates.
- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: Rust lint attributes cannot conceal the measured diagnostic class.
- priority: verification
- needs: GAIA-016
- scope: `scripts/float_cmp_suppressions.py` and focused tests.
- acceptance: comments and strings are ignored; broad allows and non-local
  `float_cmp` suppression are rejected; narrow item expectations are accepted.
- basis: `34f0229`
- next: implement the scanner and adversarial lexical cases under 400 lines.

<a id="GAIA-019"></a>
## GAIA-019 — Enforce the float-comparison ceiling
- milestone: M-gates.
- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: the manifest ceiling is machine-checked against diagnostics and its
  accepted base value.
- priority: verification
- needs: GAIA-017, GAIA-018
- scope: `scripts/float_cmp_budget.py` manifest, base, and count rules plus tests.
- acceptance: missing, duplicate, malformed, or raised ceilings fail; 44
  emissions pass and 45 fail; shallow-history base resolution is tested.
- basis: `34f0229`
- next: implement the pure budget rules and CLI-free tests under 400 lines.

<a id="GAIA-020"></a>
## GAIA-020 — Gate float-comparison growth in CI
- milestone: M-gates.
- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: the repository gate runs strict Clippy once and rejects diagnostic or
  suppression growth on pull requests and main.
- priority: verification
- needs: GAIA-019
- scope: `.github/workflows/ci.yml`, Cargo execution in
  `scripts/float_cmp_budget.py`, and shared helpers in `scripts/lockfile.py`.
- acceptance: the script reports the 44-emission inventory, checks the base
  ceiling and suppressions, and the hosted gate passes on its exact revision.
- basis: `34f0229`
- next: integrate the runner and workflow under 400 changed lines.

<a id="GAIA-021"></a>
## GAIA-021 — Validate finite rational NURBS weights
- milestone: M-seam.
- Status: todo; priority: P1; integrator: unclaimed; last-update: 2026-09-24.

- outcome: rational curves and surfaces reject non-finite weights at their
  construction boundary while preserving the positive-weight contract.
- priority: correctness
- needs: GAIA-003
- scope: `src/domain/geometry/nurbs/{curve,surface}.rs`, public construction
  error enums, tests, Rustdoc, and the breaking-change migration guide.
- acceptance: NaN and positive infinity return typed exhaustive error variants;
  finite positive weights remain accepted; cargo-semver-checks classifies the
  enum change and a major-version migration documents exhaustive-match updates.
- basis: `34f0229`
- next: specify the public error migration before implementation.

<a id="GAIA-022"></a>
## GAIA-022 — Consolidate the legacy execution checklist
- milestone: M-docs.
- status: todo
- outcome: every open deliverable has one complete record in this queue.
- priority: verification
- needs: none
- scope: `CHECKLIST.md`, `backlog.md`, and docs linking to checklist anchors.
- acceptance: revalidate every unchecked entry against the tree and hosting;
  map it to an existing item or file a complete item; remove stale links;
  delete `CHECKLIST.md`; leave no references to it.
- basis: `e43c3e6` (current main).
- next: compare each unchecked entry with the current tree, backlog, and PRs.


<a id="GAIA-039"></a>
## GAIA-039 — Unformatted code can still reach main
- milestone: M-gates.
- status: todo
- outcome: unformatted code cannot reach `main`.
- priority: correctness
- needs: none
- scope: `.githooks/pre-push` (line 1123: `SKIP_LOCAL_GATE=1` exits the push phase before its fmt check; hook install is opt-in via the atlas stack script `scripts/atlas-lock-form.py install-hooks`, an atlas repository path); `main` branch protection.
- acceptance: a recorded decision on the enforcement mechanism, and a check showing an unformatted push or merge to `main` is rejected.
- evidence (2026-11-13, measured at `a0ecea2`): `gh api repos/ryancinsight/gaia/branches/main/protection` reports `enforce_admins` false, no required pull-request reviews, `strict` false; `gh api repos/ryancinsight/gaia/rulesets` is empty. Of the 42 first-parent commits in `532577e^..a0ecea2` (2026-10-05 onward), 5 have an empty `gh api repos/ryancinsight/gaia/commits/<sha>/pulls` (the other 37 return a PR): `532577e`, `f85f185`, `447adeb`, `dc56f42` (a merge), `541ba01`. The required contexts include "fmt + clippy + supply-chain + nextest + doctests + doc"; while `enforce_admins` is false, an admin push bypasses them (read 2026-11-13; when that context became required is not established).
- basis: `a0ecea2` (current main).
- next: decide between enforcing admins, requiring a PR, and removing the skip in the atlas-owned hook (`scripts/git-hooks/pre-push` in the atlas repository); no atlas item carries that upstream change at atlas `2b000c51b`, so it is filed at the next atlas filing sweep.

<a id="GAIA-RESCUE-QUEUE"></a>
## GAIA-RESCUE-QUEUE — Port the rescue branch's file splits onto main
- milestone: M-debt.
- status: todo
- outcome: the file splits and test-module extractions of rescue branch #134 are ported onto current main, each file at or under 500 lines, or recorded as a GAIA-013 cohesion exception.
- priority: correctness
- needs: none
- scope: the `.rs` paths #134 changes (kept at `refs/pull/134/head`); the board of this repository.
- acceptance: every `.rs` path #134 changes that exists on main is at or under 500 lines, or is recorded as a cohesion exception; the `cargo nextest list` names of ported modules are unchanged.
- basis: `2a3aa98` (current main).
- queue: `ryancinsight/gaia#134` (`rescue/audit-gaia-20260928`, 98 files, base `9d382c4`) is the only surviving source; its work is kept at `refs/pull/134/head`. #121 and #123 resolve against main (witnesses in the bodies of #209 and #210). #120 is subsumed by #134 plus a rescue-hook hunk that main's classifier dispatch supersedes and mid-edit defects read from its diff, not compiled (an orphaned doc comment in `bowyer_watson.rs`, a dropped `DelaunayTriangulation` import in `bowyer_watson/{access,repair}.rs`, stale imports in `lattice.rs`, `mod data;` dropped but still re-exported in `vertex_pool.rs`); its work is kept at `refs/pull/120/head`.
- port plan: main has moved 308 commits since #134's base and `git merge-tree` reports 31 conflicted paths (5 modify/delete), so it ports file by file, leaf-first. Extractions use `<mod>/tests.rs`, the layout main uses; the `partN.rs` and `tests_<x>.rs` layout does not port. Drop #134's two case-only renames (`CHECKLIST.md` to `checklist.md`, `agents.md` to `AGENTS.md`): main keeps the original names. 19 of the `.rs` paths #134 changes that exist on main still exceed 500 lines; `quality/boundary.rs`, `openfoam.rs`, `edge_store.rs`, `adjacency.rs`, `arrangement/stitch.rs`, `arrangement/snap_round.rs`, `arrangement/planar.rs`, `quality/normals.rs`, `primitives/revolution_sweep.rs` and `tpms/box_clip.rs` are already ported.
- decision for the lead: whether `partN` slicing of the three 1000+ line test files (`adversarial_tests.rs`, `tests.rs`, `indexed_tests.rs`) is acceptable under GAIA-013's cohesion non-goal, or each is re-split by operation family.
- next: port the remaining #134 files, each file at or under 500 lines.

<a id="GAIA-040"></a>
## GAIA-040 — OpenFOAM writer tests share fixed temp paths
- milestone: M-coherence.
- status: todo
- outcome: the `openfoam` tests write only under per-test temporary directories, so no test reads or overwrites another run's output.
- priority: verification
- needs: none
- scope: `src/infrastructure/io/openfoam/tests.rs`; `Cargo.toml` `[dev-dependencies]` only if a tempdir crate is chosen (none is declared today).
- acceptance: no test derives a path from a fixed name under `std::env::temp_dir()`; the six `infrastructure::io::openfoam::tests` pass under parallel `cargo nextest run -p gaia-mesh`, and a failing assertion leaves no directory behind.
- evidence (read at `c582974`, `tests.rs:23-123`): the tests join `gaia_of_test_five`, `_vcount`, `_fcount`, `_boundary`, `_named` and `_empty` onto `std::env::temp_dir()`; the five that write call `remove_dir_all` only after their last assertion, so a failed assertion leaves residue, and any two runs sharing the OS temp directory (peer trees) write the same paths.
- basis: `c582974` (current main).
- next: give each test a unique directory removed by a drop guard; reuse std unless adding a tempdir dev-dependency is accepted.

<a id="GAIA-ADR0008-PROVIDER-BOUND"></a>
## GAIA-ADR0008-PROVIDER-BOUND — Revise the stale Leto bound in ADR 0008 and gap_audit.md [patch]
- status: todo
- priority: verification
- milestone: M-docs
- needs: none
- outcome: ADR 0008 and `gap_audit.md` state the Leto bound the manifest admits, and every cross-reference in them resolves.
- acceptance: `git grep -n 'Leto \^0.46'` matches only a dated revision note in ADR 0008 naming `>=0.49.0, <0.51.0` and the commits that moved it; `gap_audit.md` has no link to `backlog.md#gaia-eunomia-010`; its section "Eunomia 0.10 / Leto 0.46 public boundary" is deleted in favor of the ADR or rewritten to match it; `docs/adr/README.md` agrees.
- evidence (origin/main `bbd8f08`): `docs/adr/0008-eunomia-010-provider-boundary.md:22` says "require Leto ^0.46"; `Cargo.toml:56-57` require leto `>=0.49.0, <0.51.0` and eunomia `0.12.0`, moved since the ADR (eunomia 0.10 to 0.11 at `92c8d11` and to 0.12 at `a84fe68`; leto 0.46 to 0.47.1 at `a84fe68`, 0.48 at `64a1098`, 0.49 at `f62d219`, then the current range); `gap_audit.md:169` links `backlog.md#gaia-eunomia-010`, an anchor `backlog.md` does not define; `gap_audit.md:137-170` repeats the same decision. Sources: gaia#225 verdict 1 F1, verdict 2 N2, verdict 3 F12.
- scope: `docs/adr/0008-eunomia-010-provider-boundary.md`, `docs/adr/README.md`, `gap_audit.md`.
- next step: rewrite the ADR 0008 Decision in place with the revision note (what, why, evidence), then delete the duplicate `gap_audit.md` section.
- basis: `bbd8f08730f16c4fca4e73039fd90b39280d87d1`

<a id="GAIA-RELEASE-CLASS-0610"></a>
## GAIA-RELEASE-CLASS-0610 — Release the Leto 0.51 geometry boundary in Gaia 0.7.0 [major]
- status: todo
- priority: correctness
- milestone: M-coherence
- needs: none
- outcome: Gaia receives the same Leto 0.51 geometry values as its consumers under the published Aequitas and Eunomia requirements.
- acceptance: release gate against 0.6.10, exact-head CI, normalized package build, and a fresh registry consumer exercising typed geometry and 3MF export.
- evidence: the published 0.6.10 API rejects a caller's Leto 0.51 points and vectors. The merged Gaia and Leto source pair returns exact plane offset −3. [ADR 0011](docs/adr/0011-published-provider-boundary.md) records the release decision.
- next step: verify the 0.7.0 candidate and record the registry receipt.

<a id="GAIA-FUZZ-TARGETS"></a>
## GAIA-FUZZ-TARGETS — Delete the inert fuzz target and run `fuzz/` in bounded CI [patch]
- status: todo
- priority: verification
- milestone: M-gates
- needs: none
- outcome: `fuzz/` holds only registered targets, and a bounded job builds and runs each parser target.
- acceptance: `fuzz/fuzz_targets/fuzz_read_stl_he.rs` is deleted and the `agents.md` I/O row names only registered targets; a scheduled or dedicated-nightly job builds and runs every `[[bin]]` in `fuzz/Cargo.toml` for a finite derived budget, so `git grep -n -i fuzz -- .github` is nonempty.
- evidence (`bbd8f08`): `fuzz/Cargo.toml` registers `fuzz_read_stl`, `fuzz_read_obj`, `fuzz_read_ply`; the tracked `fuzz_read_stl_he.rs` begins "REMOVED" and instructs `git rm`; `agents.md:706` still lists `fuzz_read_stl_he`; `git grep -n -i fuzz origin/main -- .github` is empty. Source: gaia#225 verdict F8.
- evidence (`9daa782`): after the rand 0.9 update `fuzz/Cargo.lock` was merged textually and no hosted job resolves it: `git grep -n -i fuzz -- .github` is empty. At `d4986c9`, `fuzz/Cargo.lock` resolves quinn-udp against windows-sys 0.61.2 while the root lock keeps 0.52.0. The job here must run `cargo metadata --locked --manifest-path fuzz/Cargo.toml` or build under `--locked`. Source: gaia#242 verdict F2, F8.
- scope: `fuzz/`, `agents.md`, `.github/workflows/`.
- next step: `git rm` the file and fix the `agents.md` row, then add the dedicated-nightly job.
- basis: `bbd8f08730f16c4fca4e73039fd90b39280d87d1`

<a id="GAIA-INDEX-EXPECT-INVARIANT"></a>
## GAIA-INDEX-EXPECT-INVARIANT — Make `raw_from_usize` state its invariant or return a typed error [patch]
- status: todo
- priority: verification
- milestone: M-debt
- needs: none
- outcome: `src/domain/core/index.rs` has no `expect` whose message lacks the `invariant:` prefix, and an index above `u32::MAX` is not an unexplained panic.
- acceptance: `raw_from_usize` either returns a typed error on overflow with its callers updated, or its `expect` message begins `invariant:` and names why overflow is unreachable at every caller; a test asserts the overflow behavior by value.
- evidence (`bbd8f08`): `index.rs:115` is `u32::try_from(n).expect("Index exceeds u32::MAX")`, a message that narrates the failure; the symbol appears in `index.rs` and `storage/vertex_pool.rs` (9 line references). Source: gaia#223 judge F3 (`639e20e`).
- scope: `src/domain/core/index.rs`, `src/infrastructure/storage/vertex_pool.rs`.
- next step: read the callers, then choose the typed error or the proven invariant.
- basis: `bbd8f08730f16c4fca4e73039fd90b39280d87d1`

<a id="GAIA-SDF-STREAM-PIN"></a>
## GAIA-SDF-STREAM-PIN — Settle and pin the seeded SDF mesher stream after rand 0.9 [patch]
- status: todo
- priority: verification
- milestone: M-coherence
- needs: none
- outcome: The tracked SDF tetrahedral volume figure and its manifest row are known to match `SdfMesher` output under rand 0.9; CI fails when a regenerated book figure differs from the tracked copy; a test pins the seeded stream so the next rand change shows; the CHANGELOG states that seeded output changed.
- acceptance: `cargo run --locked --quiet --bin book_mesh_gallery -- docs/book` followed by `git status --porcelain docs/book` prints nothing, or the regenerated figure and `docs/book/figure_manifest.md` row (158 vertices, 1058 faces, 451 cells at the basis) are committed (not run at the basis: unsettled); `git grep -n 'diff --exit-code' -- .github` prints a step after the gallery run (prints nothing at the basis); a test asserts exact output for `SdfMesher::new(0.8)` on the unit `SphereSdf` as a regression pin, since `lattice_tests.rs:38-48` compares two runs only; `git grep -n -w rand -- CHANGELOG.md` prints a [patch] entry (nothing at the basis).
- evidence (`9daa782`): `lattice.rs:277` seeds `StdRng`, then `shuffle` (:293, :297) and three `random_range` per point (:303-305); the judge of gaia#242 read rand 0.8.8 `seq/mod.rs:586-591` (downward, one draw per element) against 0.9.5 `seq/slice.rs:373-410` (upward, packed draws), so the permutation and jitter differ for a fixed seed. `book-pages.yml:75` regenerates the figures and nothing diffs them.
- scope: `src/application/delaunay/dim3/lattice_tests.rs`, `docs/book/figures/models/topology/sdf-tetrahedral-volume.svg`, `docs/book/figure_manifest.md`, `.github/workflows/book-pages.yml`, `CHANGELOG.md`.
- next step: run the gallery at the basis and diff; then add the diff step and the pin.
- basis: `d4986c9c2a7d0b7dbef240ec1e23fbfa334438ba`

<a id="GAIA-BOARD-ITEM-FORM"></a>
## GAIA-BOARD-ITEM-FORM — Bring open items to the 15-line schema with anchors [patch]
- status: todo
- priority: verification
- milestone: M-docs
- needs: none
- outcome: every open item is about 15 lines or fewer and has an `<a id>` anchor immediately before its heading.
- acceptance: heading-to-heading line count (anchor excluded) is at most 15 for every `## GAIA-` item and an anchor precedes each; long delivery narrative moves to commits, PRs or ADRs.
- evidence (`9daa782` plus this change): 12 of 33 `## GAIA-` items still exceed 15 lines by that count (GAIA-010 28, -008 27, -011 27, -015 27, -027 26, -NEXTEST-PIPE 25, -005 25, -012 24, -007 21, -014 21, -029 16, -021 16); every item has an anchor. Separately, GAIA-017 to -021 carry `priority: P1` outside the closed set (lint violations). `atlas-artifact-budget.py` reports item length only for per-item files and the lint checks neither length nor anchors, so no check sees the length class.
- scope: `backlog.md` only.
- next step: compact the twelve items above, largest first, keeping outcome, acceptance oracle and next step; GAIA-029 and -021 each exceed by one line.
- basis: `9daa782f4ac78aa9e900e6cb6a5912d85713885d`
