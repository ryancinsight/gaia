//! Stanford PLY import and export.
//!
//! Supports ASCII PLY with vertex positions, normals, and triangular faces.

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

/// Write an [`IndexedMesh`] as ASCII PLY.
pub fn write_ply<W: Write>(writer: &mut W, mesh: &IndexedMesh) -> MeshResult<()> {
    let vertex_count = mesh.vertex_count();
    let face_count = mesh.face_count();

    // Header
    writeln!(writer, "ply").map_err(MeshError::Io)?;
    writeln!(writer, "format ascii 1.0").map_err(MeshError::Io)?;
    writeln!(writer, "comment exported by gaia").map_err(MeshError::Io)?;
    writeln!(writer, "element vertex {vertex_count}").map_err(MeshError::Io)?;
    writeln!(writer, "property float x").map_err(MeshError::Io)?;
    writeln!(writer, "property float y").map_err(MeshError::Io)?;
    writeln!(writer, "property float z").map_err(MeshError::Io)?;
    writeln!(writer, "property float nx").map_err(MeshError::Io)?;
    writeln!(writer, "property float ny").map_err(MeshError::Io)?;
    writeln!(writer, "property float nz").map_err(MeshError::Io)?;
    writeln!(writer, "element face {face_count}").map_err(MeshError::Io)?;
    writeln!(writer, "property list uchar int vertex_indices").map_err(MeshError::Io)?;
    writeln!(writer, "end_header").map_err(MeshError::Io)?;

    // Build contiguous index map.
    let mut id_to_idx: HashMap<crate::domain::core::index::VertexId, usize> =
        HashMap::with_capacity(vertex_count);

    // Vertex data
    for (idx, (vid, vdata)) in mesh.vertices.iter().enumerate() {
        id_to_idx.insert(vid, idx);
        let p = &vdata.position;
        let n = &vdata.normal;
        writeln!(writer, "{} {} {} {} {} {}", p.x, p.y, p.z, n.x, n.y, n.z)
            .map_err(MeshError::Io)?;
    }

    // Face data
    for (_fid, face) in mesh.faces.iter_enumerated() {
        let i0 = id_to_idx[&face.vertices[0]];
        let i1 = id_to_idx[&face.vertices[1]];
        let i2 = id_to_idx[&face.vertices[2]];
        writeln!(writer, "3 {i0} {i1} {i2}").map_err(MeshError::Io)?;
    }

    Ok(())
}

// =============================================================================
//  Import
// =============================================================================

/// Cap on the speculative pre-allocation driven by the header's element counts.
///
/// Those counts are file-supplied *claims*, not facts. `element vertex
/// 999999999999` would otherwise ask the allocator for tens of terabytes before
/// a single body line is read, and a failed allocation aborts the process
/// instead of returning an error. Pre-allocation only buys an honest file a few
/// avoided reallocations, so it is capped; past the cap the vectors grow
/// normally.
const SPECULATIVE_RESERVE_CAP: usize = 1 << 16;

/// Read an ASCII PLY file into a new [`IndexedMesh`].
///
/// # Strictness
///
/// The header's `element` counts bound the loops that read the body, and a
/// body line that cannot supply the record it belongs to is an error rather
/// than a silent omission: a face declaring fewer than three vertices, a face
/// naming a vertex the file never declared, and a blank line where a face was
/// expected all fail the read.
///
/// # Errors
/// Returns [`MeshError::Other`] for a malformed header or record, and
/// [`MeshError::InvalidCoordinate`] if a position is NaN or infinite.
pub fn read_ply<R: Read>(reader: R) -> MeshResult<IndexedMesh> {
    let buf = BufReader::new(reader);
    let mut lines = buf.lines();

    // Parse header.
    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut in_vertex_props = false;
    // `usize`, not `u8`: this is incremented once per matching header line, so a
    // header long enough to overflow a byte would panic in a debug build. Only
    // `>= 3` is ever asked of the value.
    let mut normal_prop_count = 0usize;

    // Read the "ply" magic.
    let magic = next_line(&mut lines)?;
    if magic.trim() != "ply" {
        return Err(MeshError::Other("not a PLY file".to_owned()));
    }

    loop {
        let line = next_line(&mut lines)?;
        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        }
        if trimmed.starts_with("element vertex") {
            vertex_count = trimmed
                .split_whitespace()
                .nth(2)
                .ok_or_else(|| MeshError::Other("bad vertex count".to_owned()))?
                .parse()
                .map_err(|_| MeshError::Other("bad vertex count".to_owned()))?;
            in_vertex_props = true;
        } else if trimmed.starts_with("element face") {
            face_count = trimmed
                .split_whitespace()
                .nth(2)
                .ok_or_else(|| MeshError::Other("bad face count".to_owned()))?
                .parse()
                .map_err(|_| MeshError::Other("bad face count".to_owned()))?;
            in_vertex_props = false;
        } else if trimmed.starts_with("element") {
            in_vertex_props = false;
        }

        // Only a `property` line declares a property. A `comment` that happens
        // to contain " nx" does not, and counting it would claim normals that
        // the body never carries.
        if in_vertex_props
            && trimmed.starts_with("property")
            && (trimmed.contains(" nx") || trimmed.contains(" ny") || trimmed.contains(" nz"))
        {
            normal_prop_count += 1;
        }
    }

    let has_normals = normal_prop_count >= 3;

    // Read vertex data.
    let mut positions: Vec<Point3r> = Vec::with_capacity(vertex_count.min(SPECULATIVE_RESERVE_CAP));
    let mut normals_vec = Vec::with_capacity(if has_normals {
        vertex_count.min(SPECULATIVE_RESERVE_CAP)
    } else {
        0
    });

    for ordinal in 0..vertex_count {
        let line = next_line(&mut lines)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MeshError::Other("vertex line too short".to_owned()));
        }
        positions.push(parse::parse_point([parts[0], parts[1], parts[2]], ordinal)?);

        if has_normals && parts.len() >= 6 {
            normals_vec.push(Vector3r::new(
                parse::parse_real(parts[3])?,
                parse::parse_real(parts[4])?,
                parse::parse_real(parts[5])?,
            ));
        }
    }

    // Build indexed mesh from vertices.
    let mut mesh = IndexedMesh::new();
    let region = RegionId::from_usize(0);

    let vertex_ids: Vec<_> = (0..positions.len())
        .map(|i| {
            let n = normals_vec.get(i).copied().unwrap_or_else(Vector3r::zeros);
            mesh.add_vertex(positions[i], n)
        })
        .collect();

    // Read face data.
    for _ in 0..face_count {
        let line = next_line(&mut lines)?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            // A blank line where a face is expected consumes one of the
            // declared faces, so the last face in the file would be dropped
            // without a word.
            return Err(MeshError::Other(
                "blank line where a face was expected".to_owned(),
            ));
        }
        let n_verts: usize = parts[0]
            .parse()
            .map_err(|_| MeshError::Other("bad face vertex count".to_owned()))?;

        // A face needs three vertices to have any area, so fewer is a
        // malformed record. (`windows(3)` below would simply produce nothing,
        // which is why this is a strictness choice rather than a safety one.)
        if n_verts < 3 {
            return Err(MeshError::Other(format!(
                "PLY face declares {n_verts} vertices; at least 3 are required"
            )));
        }
        // Compare against `parts.len() - 1` rather than `n_verts + 1`: the
        // count is file-controlled, and `n_verts + 1` overflows for
        // `usize::MAX`.
        if n_verts > parts.len() - 1 {
            return Err(MeshError::Other("face line too short".to_owned()));
        }

        let face_verts: Vec<usize> = parts
            .iter()
            .skip(1)
            .take(n_verts)
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| MeshError::Other(format!("bad face index: {s}")))
            })
            .collect::<MeshResult<_>>()?;

        // Resolve every index before touching the mesh, so a face that fails
        // validation cannot leave half its triangles behind.
        let ids: Vec<_> = face_verts
            .iter()
            .map(|&vi| parse::resolve(&vertex_ids, vi, "PLY face vertex"))
            .collect::<MeshResult<_>>()?;

        // Fan-triangulate. `windows(3)` states the "at least three vertices"
        // invariant structurally, so no `len() - 1` is available to underflow.
        for triangle in ids.windows(3) {
            mesh.add_face_with_region(triangle[0], triangle[1], triangle[2], region);
        }
    }

    Ok(mesh)
}

// =============================================================================
//  Fuzz entry point
// =============================================================================

/// Fuzz entry point for PLY parsing.
///
/// Accepts arbitrary bytes and attempts to parse them as PLY. This function
/// must **never panic** — every failure is returned as `Err`. Suitable as the
/// inner body of a `cargo-fuzz` target.
///
/// # Example (in a fuzz target)
/// ```rust,ignore
/// #![no_main]
/// libfuzzer_sys::fuzz_target!(|data: &[u8]| {
///     let _ = gaia::infrastructure::io::ply::fuzz_read_ply(data);
/// });
/// ```
pub fn fuzz_read_ply(data: &[u8]) -> MeshResult<IndexedMesh> {
    read_ply(std::io::Cursor::new(data))
}

fn next_line(lines: &mut std::io::Lines<BufReader<impl Read>>) -> MeshResult<String> {
    lines
        .next()
        .ok_or_else(|| MeshError::Other("unexpected end of PLY file".to_owned()))?
        .map_err(MeshError::Io)
}

// =============================================================================
//  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_rejects;

    /// Assemble a minimal ASCII PLY from raw body text, so a test can put a
    /// deliberately malformed record exactly where the reader will look for it.
    fn ply_ascii(vertex_count: usize, vertices: &str, face_count: usize, faces: &str) -> String {
        format!(
            "ply\nformat ascii 1.0\n\
             element vertex {vertex_count}\n\
             property float x\nproperty float y\nproperty float z\n\
             element face {face_count}\n\
             property list uchar int vertex_indices\n\
             end_header\n\
             {vertices}{faces}"
        )
    }

    #[test]
    fn ply_round_trip() {
        let mut mesh = IndexedMesh::new();
        let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        mesh.add_face(v0, v1, v2);

        let mut buf = Vec::new();
        write_ply(&mut buf, &mesh).unwrap();

        let mesh2 = read_ply(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(mesh2.vertex_count(), 3);
        assert_eq!(mesh2.face_count(), 1);
    }

    #[test]
    fn ply_quad_fan_triangulated() {
        let ply = ply_ascii(4, "0 0 0\n1 0 0\n1 1 0\n0 1 0\n", 1, "4 0 1 2 3\n");
        let mesh = read_ply(std::io::Cursor::new(ply.as_bytes())).unwrap();
        assert_eq!(mesh.face_count(), 2);
    }

    // ── Malformed input is an error, never a panic ────────────────────────

    /// Regression: the vertex index was read with `vertex_ids[face_verts[0]]`,
    /// so a face naming a vertex the file never declared panicked.
    #[test]
    fn ply_out_of_range_vertex_index_is_an_error() {
        for face in ["3 0 1 99\n", "3 18446744073709551615 1 2\n"] {
            let ply = ply_ascii(3, "0 0 0\n1 0 0\n0 1 0\n", 1, face);
            assert_rejects(
                &read_ply(std::io::Cursor::new(ply.as_bytes())),
                "out of range",
            );
        }
    }

    /// Regression: `parts[1..=0]` is a reversed range, and slicing it panics.
    #[test]
    fn ply_zero_vertex_face_is_an_error() {
        let ply = ply_ascii(3, "0 0 0\n1 0 0\n0 1 0\n", 1, "0\n");
        assert_rejects(
            &read_ply(std::io::Cursor::new(ply.as_bytes())),
            "at least 3",
        );
    }

    /// Regression: `parts.len() < n_verts + 1` overflowed before it compared.
    #[test]
    fn ply_face_vertex_count_of_usize_max_does_not_overflow() {
        let ply = ply_ascii(
            3,
            "0 0 0\n1 0 0\n0 1 0\n",
            1,
            "18446744073709551615 0 1 2\n",
        );
        assert_rejects(
            &read_ply(std::io::Cursor::new(ply.as_bytes())),
            "face line too short",
        );
    }

    /// Regression: a blank line used to consume one of the declared faces, so
    /// the last face in the file was dropped without a word.
    #[test]
    fn ply_blank_line_where_a_face_was_expected_is_an_error() {
        let ply = ply_ascii(3, "0 0 0\n1 0 0\n0 1 0\n", 1, "\n3 0 1 2\n");
        assert_rejects(
            &read_ply(std::io::Cursor::new(ply.as_bytes())),
            "blank line",
        );
    }

    #[test]
    fn ply_non_finite_vertex_is_an_error() {
        for spelling in ["nan", "inf", "-inf"] {
            let ply = ply_ascii(1, &format!("{spelling} 0 0\n"), 0, "");
            assert_rejects(
                &read_ply(std::io::Cursor::new(ply.as_bytes())),
                "invalid coordinate at vertex 0",
            );
        }
    }

    /// The header count is a claim. Without the cap this asks the allocator for
    /// tens of terabytes and aborts the process instead of reporting an error.
    #[test]
    fn ply_absurd_vertex_count_does_not_pre_allocate() {
        let ply = "ply\nformat ascii 1.0\n\
                   element vertex 999999999999\n\
                   property float x\nproperty float y\nproperty float z\n\
                   end_header\n";
        assert_rejects(
            &read_ply(std::io::Cursor::new(ply.as_bytes())),
            "unexpected end of PLY file",
        );
    }

    /// Only `property` lines declare properties. These comments used to be
    /// counted, claiming normals the file never declared — and the six-field
    /// vertex lines then had their trailing columns read as those normals.
    #[test]
    fn ply_comment_mentioning_a_normal_property_declares_nothing() {
        let ply = ply_ascii(3, "0 0 0 9 9 9\n1 0 0 9 9 9\n0 1 0 9 9 9\n", 1, "3 0 1 2\n");
        let ply = ply.replace(
            "property float z\n",
            "property float z\ncomment nx\ncomment ny\ncomment nz\n",
        );
        let mesh = read_ply(std::io::Cursor::new(ply.as_bytes())).unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        for (_vid, vertex) in mesh.vertices.iter() {
            assert_eq!(vertex.normal.x, 0.0);
            assert_eq!(vertex.normal.y, 0.0);
            assert_eq!(vertex.normal.z, 0.0);
        }
    }

    /// The fuzz entry point's contract: arbitrary bytes, no panic.
    #[test]
    fn fuzz_read_ply_never_panics_on_adversarial_input() {
        let inputs: &[&[u8]] = &[
            b"",
            b"ply",
            b"ply\nformat ascii 1.0\n",
            b"ply\nformat ascii 1.0\nend_header\n",
            b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n",
            b"ply\nformat ascii 1.0\nelement face 1\nend_header\n0 0 0\n",
            b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\nnan nan nan\n",
            b"ply\nformat ascii 1.0\nelement vertex 0\nelement face 1\n\
              end_header\n18446744073709551615 0 0 0\n",
            b"ply\nformat ascii 1.0\nelement vertex 3\n\
              property float x\nproperty float y\nproperty float z\n\
              end_header\n0 0 0\n1 0 0\n0 1 0\n",
        ];
        for input in inputs {
            let _ = fuzz_read_ply(input);
        }
    }
}
