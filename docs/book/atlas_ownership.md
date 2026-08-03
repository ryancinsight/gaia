# Atlas ownership and provider boundaries

Gaia is the Atlas mesh-generation SSOT. Its consumers may select a mesh
through a stable `IndexedMesh` contract, but mesh topology, welding, primitive
construction, channel sweeps, CSG, and volume conversion remain owned by Gaia.
This keeps one implementation of each meshing algorithm and prevents
CFDrs, Kwavers, Helios, RITK, or downstream private consumers from growing
divergent local meshers.

Gaia already uses Atlas providers at their domain seams:

| Role | Provider | Gaia boundary |
| --- | --- | --- |
| Geometry and points | `leto` | `Point3`, `Vector3`, and geometry operations |
| Numeric element contract | `eunomia` | `Scalar` bounds and native arithmetic |
| Permission/state infrastructure | `melinoe` | infrastructure-level permission integration |
| Optional execution/memory integration | `moirai`, `mnemosyne` | optional workspace capabilities, not mesh ownership |

Provider use is direct where the role exists; Gaia does not add a consumer-owned
adapter layer to conceal a missing provider operation. When another Atlas repo
needs a mesh capability, the capability belongs in Gaia if it is a meshing
operation, and the consumer adds only the contract test and boundary conversion
needed for its domain.

The ownership rule is intentionally narrower than “every consumer must depend
on every provider.” Consumers depend on Gaia's published mesh contract and on
their own domain providers. They do not inherit Gaia's internal storage or
infrastructure dependencies merely to construct a mesh.
