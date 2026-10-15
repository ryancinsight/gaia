# ADR 0008: Admit the Eunomia 0.10 numeric contract

- Status: Accepted
- Change class: [major], pre-1.0 minor increment
- Item: GAIA-EUNOMIA-010

## Context

Gaia publicly exposes `Scalar: eunomia::RealField` and Leto geometry types.
The published 0.5.3 package selects Eunomia 0.9 and Leto 0.45. Their replacement
with Eunomia 0.10/Leto 0.46 changes nominal provider identities even where
trait definitions or geometry values remain textually equal. Sealing Scalar
to f32/f64 does not preserve generic callers' inherited old-provider bounds.

PR175's first provider update retains Gaia 0.5.3 and Leto ^0.45. Its normalized
registry package fails with 149 errors because direct Eunomia 0.10 conflicts
with Leto 0.45.2's Eunomia 0.9 traits. Git-locked compilation is insufficient
to establish package compatibility.

## Decision

Version Gaia at 0.6 and require Leto ^0.46. Retain the existing Eunomia 0.10,
Aequitas 0.3 and memory 0.10 admissions; require Moirai runtime 0.9.3 in the
existing runtime/development declarations. Keep source URLs, features and
defaults unchanged. Preserve geometry kernels, precision tolerances, existing
tests and all 166 registry records. Incoming Metis/RITK requirements admit
Gaia 0.6 through their existing owner-controlled integration.

Use the new Eunomia identity for public numeric bounds. Do not introduce
compatibility aliases, duplicate provider traits, source substitutions or
old-provider fallbacks. Versions, lock and release records land atomically.

## Alternatives

- Keeping 0.5.3 understates the demonstrated public bound change.
- Keeping Leto ^0.45 fails standard package verification against the registry.
- Adding the old trait alongside the new trait masks the intended migration
  and duplicates ownership instead of completing the provider boundary.

## Verification

`scripts/check_provider_boundary.py` compiles/runs identical generic source
against published Gaia 0.5.3/Eunomia 0.9.2, then expects E0277 against the
normalized 0.6 package: `T: Gaia::Scalar` no longer supplies the old RealField
bound. It uses real registry dependencies and no mocks. Run a standard clean
package dry-run first. Its optional evidence directory retains manifests,
locks and compiler logs. The API scanner does not detect this external
nominal-identity loss; this executed consumer fills that coverage gap.

Full native/property tests, strict all-target Clippy, docs/doctests,
supply-chain/hygiene, lock/ADR/API gates and standard package verification
remain required on the final revision. Preserve the output-pipe investigation
and all timeouts/workloads. Miri remains deferred locally; no existing CI job
is disabled. Final integration/publication requires applicable gates/review.
