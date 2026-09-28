# ADR 0006: PSLG scalar-relative tolerances

- Status: Accepted
- Date: 2026-09-24
- Board item: GAIA-003

## Context

`Pslg` now stores vertices in `T: Scalar`, defaulting to `Real`. Its
validation and crossing operations must therefore express geometric
tolerances relative to the active scalar precision. The previous implementation
used fixed `f64` values: `1e-14` for overlap and crossing-angle decisions and
`8.1e-28` for coincident vertices.

## Decision

- Keep orientation classification on `orient_2d`, whose stored-`T` inputs are
  promoted losslessly to the exact-predicate implementation as recorded in
  [ADR 0005](0005-native-precision-predicate-boundary.md).
- Set overlap and welding tolerance to `64 * epsilon(T)`. The former `1e-14`
  cutoff is about 45 `f64` epsilons; rounding upward to 64 provides one
  precision-relative threshold for both supported scalar types.
- Set the coincident-vertex tolerance to `(128 * epsilon(T))² * scale²`, where
  `scale²` is the larger of the bounding-box diagonal squared and the largest
  absolute coordinate squared. This preserves the former scale-relative
  construction while removing its `f64`-specific constant.
- Construct proper crossings in `T`. Normalize every segment displacement by
  the largest displacement component before cross products; the common scale
  cancels from the intersection parameter and keeps products in range for
  uniformly small or large inputs. Do not apply an angle cutoff after exact
  orientation has established a proper crossing.
- A normalized determinant that rounds to zero in `T` still cannot produce an
  intersection point. `resolve_crossings` documents this limit, and callers
  validate the result; the CSG path checks through `Cdt::try_from_pslg`.

## Alternatives

- Fixed `f64` constants do not scale to `f32` and produce a precision-dependent
  topology.
- A relative angle cutoff discards proper shallow crossings already classified
  by exact orientation.
- Widening the crossing construction changes the native-precision contract
  recorded in ADR 0005.

## Consequences

PSLG behavior is generic over `f32` and `f64`; existing `Pslg` call sites keep
the `f64` default. Tests cover shallow crossings below the former angle cutoff,
uniformly small and large crossing scales, relative-tolerance boundaries, and
the detectable failure when a `T`-precision determinant cancels to zero.
Callers that require a planar result must check `validate()` after resolution.

Overturning evidence: a first-party exact construction that preserves the
native-precision API and resolves crossings whose normalized determinant
cancels to zero.
