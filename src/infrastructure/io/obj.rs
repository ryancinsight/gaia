//! Wavefront OBJ import and export.
//!
//! Supports triangle meshes with vertex positions and normals.

use hashbrown::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::core::index::RegionId;
use crate::domain::core::scalar::{Point3r, Vector3r};
use crate::domain::mesh::IndexedMesh;

use super::parse;

// =============================================================================
//  Export
// =============================================================================

/// Write an [`IndexedMesh`] as Wavefront OBJ.
///
/// Emits `v` (position), `vn` (normal), and `f` (face) records.
/// OBJ uses 1-based indexing.
pub fn write_obj<W: Write>(writer: &mut W, mesh: &IndexedMesh) -> MeshResult<()> {
    writeln!(writer, "# OBJ exported by gaia").map_err(MeshError::Io)?;

    // Build a contiguous index map: VertexId -> 0-based index.
    let mut id_to_idx: HashMap<crate::domain::core::index::VertexId, usize> =
        HashMap::with_capacity(mesh.vertex_count());

    // Emit vertices and normals in insertion order.
    for (idx, (vid, vdata)) in mesh.vertices.iter().enumerate() {
        id_to_idx.insert(vid, idx);
        let p = &vdata.position;
        writeln!(writer, "v {} {} {}", p.x, p.y, p.z).map_err(MeshError::Io)?;
    }

    for (_vid, vdata) in mesh.vertices.iter() {
        let n = &vdata.normal;
        writeln!(writer, "vn {} {} {}", n.x, n.y, n.z).map_err(MeshError::Io)?;
    }

    // Emit faces (1-indexed).
    for (_fid, face) in mesh.faces.iter_enumerated() {
        let i0 = id_to_idx[&face.vertices[0]] + 1;
        let i1 = id_to_idx[&face.vertices[1]] + 1;
        let i2 = id_to_idx[&face.vertices[2]] + 1;
        writeln!(writer, "f {i0}//{i0} {i1}//{i1} {i2}//{i2}").map_err(MeshError::Io)?;
    }

    Ok(())
}

// =============================================================================
//  Import
// =============================================================================

/// Read a Wavefront OBJ file into a new [`IndexedMesh`].
///
/// Supports `v`, `vn`, and `f` records. Face specifications may be:
/// - `f v1 v2 v3` (position only)
/// - `f v1//vn1 v2//vn2 v3//vn3` (position + normal)
/// - `f v1/vt1/vn1 ...` (position + texcoord + normal, texcoord ignored)
///
/// Polygonal faces with more than 3 vertices are fan-triangulated.
///
/// # Strictness
///
/// A record whose keyword this function understands must be well-formed;
/// only records it does not model are skipped. A face with fewer than three
/// vertices, a face naming a position the file never declared, and a `v`/`vn`
/// record with too few fields are all errors rather than silent omissions.
/// Dropping a `v` record silently would be worse than failing: it shifts every
/// later position index, so faces would resolve to the *wrong* vertices rather
/// than to none.
///
/// A *missing* normal is tolerated — an out-of-range normal index yields a zero
/// normal, since the face still has geometry without one — but a missing
/// position cannot be, because it leaves nothing.
///
/// # Errors
/// Returns [`MeshError::Io`] if reading fails, [`MeshError::Other`] if a record
/// is malformed, and [`MeshError::InvalidCoordinate`] if a position is NaN or
/// infinite.
pub fn read_obj<R: Read>(reader: R) -> MeshResult<IndexedMesh> {
    let buf = BufReader::new(reader);
    let mut positions: Vec<Point3r> = Vec::new();
    let mut normals: Vec<Vector3r> = Vec::new();
    let mut mesh = IndexedMesh::new();
    let region = RegionId::from_usize(0);

    for line in buf.lines() {
        let line = line.map_err(MeshError::Io)?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "v" if parts.len() >= 4 => {
                positions.push(parse::parse_point(
                    [parts[1], parts[2], parts[3]],
                    positions.len(),
                )?);
            }
            "vn" if parts.len() >= 4 => {
                normals.push(Vector3r::new(
                    parse::parse_real(parts[1])?,
                    parse::parse_real(parts[2])?,
                    parse::parse_real(parts[3])?,
                ));
            }
            "f" if parts.len() >= 4 => {
                // Parse face vertex indices.
                let verts: Vec<(usize, Option<usize>)> = parts
                    .iter()
                    .skip(1)
                    .map(|s| parse_face_vertex(s))
                    .collect::<MeshResult<_>>()?;

                // Resolve every index *before* touching the mesh, so a face
                // that fails validation cannot leave half its vertices behind.
                let resolved: Vec<(Point3r, Vector3r)> = verts
                    .iter()
                    .map(|&(pi, ni)| {
                        let position = parse::resolve(&positions, pi, "OBJ face vertex position")?;
                        let normal = ni
                            .and_then(|i| normals.get(i).copied())
                            .unwrap_or_else(Vector3r::zeros);
                        Ok((position, normal))
                    })
                    .collect::<MeshResult<_>>()?;

                let ids: Vec<_> = resolved
                    .into_iter()
                    .map(|(position, normal)| mesh.add_vertex(position, normal))
                    .collect();

                // Fan-triangulate. `windows(3)` states the "at least three
                // vertices" invariant structurally, so no `len() - 1` is
                // available to underflow.
                for triangle in ids.windows(3) {
                    mesh.add_face_with_region(triangle[0], triangle[1], triangle[2], region);
                }
            }
            "v" | "vn" => {
                return Err(MeshError::Other(format!(
                    "OBJ `{}` record has {} fields; 4 are required",
                    parts[0],
                    parts.len()
                )));
            }
            "f" => {
                return Err(MeshError::Other(format!(
                    "OBJ face has {} vertices; at least 3 are required",
                    parts.len() - 1
                )));
            }
            // Records this importer does not model carry no geometry it
            // claims to read: vt, mtllib, usemtl, o, g, s, l, p, vp, ...
            _ => {}
        }
    }

    Ok(mesh)
}

// =============================================================================
//  Fuzz entry point
// =============================================================================

/// Fuzz entry point for OBJ parsing.
///
/// Accepts arbitrary bytes and attempts to parse them as OBJ. This function
/// must **never panic** — every failure is returned as `Err`. Suitable as the
/// inner body of a `cargo-fuzz` target.
///
/// # Example (in a fuzz target)
/// ```rust,ignore
/// #![no_main]
/// libfuzzer_sys::fuzz_target!(|data: &[u8]| {
///     let _ = gaia::infrastructure::io::obj::fuzz_read_obj(data);
/// });
/// ```
pub fn fuzz_read_obj(data: &[u8]) -> MeshResult<IndexedMesh> {
    read_obj(std::io::Cursor::new(data))
}

/// Parse a face vertex specification like `v`, `v//vn`, or `v/vt/vn`.
/// Returns `(position_index, optional_normal_index)`, converted to 0-based.
///
/// # Errors
/// Returns [`MeshError::Other`] if the specification has more than the three
/// slash-separated fields OBJ defines, or if an index is not a positive
/// integer. OBJ indices are 1-based, so `0` is rejected rather than read as
/// the last element.
fn parse_face_vertex(s: &str) -> MeshResult<(usize, Option<usize>)> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() > 3 {
        return Err(MeshError::Other(format!(
            "invalid face vertex: {s} (at most 3 slash-separated fields)"
        )));
    }

    let pos_idx: usize = parts[0]
        .parse::<usize>()
        .map_err(|_| MeshError::Other(format!("invalid face vertex: {s}")))?
        .checked_sub(1)
        .ok_or_else(|| MeshError::Other(format!("face vertex index 0 is invalid: {s}")))?;

    let normal_idx = if parts.len() == 3 && !parts[2].is_empty() {
        Some(
            parts[2]
                .parse::<usize>()
                .map_err(|_| MeshError::Other(format!("invalid normal index: {s}")))?
                .checked_sub(1)
                .ok_or_else(|| MeshError::Other(format!("normal index 0 is invalid: {s}")))?,
        )
    } else {
        None
    };

    Ok((pos_idx, normal_idx))
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_rejects;

    #[test]
    fn obj_round_trip() {
        let mut mesh = IndexedMesh::new();
        let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        mesh.add_face(v0, v1, v2);

        let mut buf = Vec::new();
        write_obj(&mut buf, &mesh).unwrap();

        let mesh2 = read_obj(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(mesh2.vertex_count(), 3);
        assert_eq!(mesh2.face_count(), 1);
    }

    #[test]
    fn obj_quad_fan_triangulated() {
        let obj = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let mesh = read_obj(std::io::Cursor::new(&obj[..])).unwrap();
        // A quad should produce 2 triangles via fan triangulation.
        assert_eq!(mesh.face_count(), 2);
    }

    // ── Malformed input is an error, never a panic ────────────────────────

    /// Regression: the position index was read with `positions[pi]`, so a face
    /// naming a position the file never declared panicked on file-controlled
    /// input.
    #[test]
    fn obj_out_of_range_position_index_is_an_error() {
        for face in ["f 9999 1 2", "f 1 2 18446744073709551615", "f 4 1 2"] {
            let obj = format!("v 0 0 0\nv 1 0 0\nv 0 1 0\n{face}\n");
            assert_rejects(
                &read_obj(std::io::Cursor::new(obj.as_bytes())),
                "out of range",
            );
        }
    }

    /// Regression: this used to fall through to the "unsupported record" arm,
    /// so the face disappeared from the mesh without a word.
    #[test]
    fn obj_face_with_too_few_vertices_is_an_error() {
        let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n";
        assert_rejects(&read_obj(std::io::Cursor::new(&obj[..])), "at least 3");
    }

    /// A dropped `v` record shifts every later position index, so faces would
    /// resolve to the *wrong* vertices rather than to none.
    #[test]
    fn obj_short_vertex_record_is_an_error() {
        let obj = b"v 0 0 0\nv 1 0\nv 0 1 0\nf 1 2 3\n";
        assert_rejects(&read_obj(std::io::Cursor::new(&obj[..])), "4 are required");
    }

    /// `Real`'s parser accepts these spellings, so the syntax check alone let a
    /// non-finite coordinate into the mesh.
    #[test]
    fn obj_non_finite_position_is_an_error() {
        for spelling in ["nan", "inf", "-inf"] {
            let obj = format!("v {spelling} 0 0\n");
            assert_rejects(
                &read_obj(std::io::Cursor::new(obj.as_bytes())),
                "invalid coordinate at vertex 0",
            );
        }
    }

    /// The asymmetry with the position index is deliberate: a face still has
    /// geometry without a normal.
    #[test]
    fn obj_out_of_range_normal_index_yields_zero_normal() {
        let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//99 2//99 3//99\n";
        let mesh = read_obj(std::io::Cursor::new(&obj[..])).unwrap();
        assert_eq!(mesh.face_count(), 1);
        for (_vid, vertex) in mesh.vertices.iter() {
            assert_eq!(vertex.normal.x, 0.0);
            assert_eq!(vertex.normal.y, 0.0);
            assert_eq!(vertex.normal.z, 0.0);
        }
    }

    #[test]
    fn obj_over_specified_face_vertex_is_an_error() {
        let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/2/3/4 2 3\n";
        assert_rejects(&read_obj(std::io::Cursor::new(&obj[..])), "slash-separated");
    }

    /// Only records the importer does not model are skipped.
    #[test]
    fn obj_unmodelled_records_are_still_skipped() {
        let obj = b"mtllib x.mtl\no thing\ng grp\ns 1\nusemtl m\nv 0 0 0\n\
                    v 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let mesh = read_obj(std::io::Cursor::new(&obj[..])).unwrap();
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
    }

    /// The fuzz entry point's contract: arbitrary bytes, no panic.
    #[test]
    fn fuzz_read_obj_never_panics_on_adversarial_input() {
        let inputs: &[&[u8]] = &[
            b"",
            b"\n\n\n",
            b"v",
            b"f",
            b"v 0 0 0\nf 1 1 1\n",
            b"v nan nan nan\nf 1 1 1\n",
            b"v 0 0 0\nf 1/2/3/4 1 1\n",
            b"v 0 0 0\nf -1 -2 -3\n",
            b"v 0 0 0\nf 0 0 0\n",
            b"v 1e400 0 0\nf 1 1 1\n",
        ];
        for input in inputs {
            let _ = fuzz_read_obj(input);
        }
    }
}
