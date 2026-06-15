//! 3MF (3D Manufacturing Format) export.
//!
//! 3MF is the modern replacement for STL in additive manufacturing workflows.
//! It is a ZIP archive containing an XML model file at `3D/3dmodel.model`.

use hashbrown::HashMap;
use std::io::{Seek, Write};

use crate::domain::core::error::{MeshError, MeshResult};
use crate::domain::mesh::IndexedMesh;

/// Write an [`IndexedMesh`] as a 3MF archive.
///
/// The output is a ZIP archive containing the required content types and
/// relationships files, plus the `3D/3dmodel.model` XML payload.
pub fn write_3mf<W: Write + Seek>(writer: W, mesh: &IndexedMesh) -> MeshResult<()> {
    let mut zip = zip::ZipWriter::new(writer);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // [Content_Types].xml
    zip.start_file("[Content_Types].xml", options)
        .map_err(io_err)?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
        .map_err(MeshError::Io)?;

    // _rels/.rels
    zip.start_file("_rels/.rels", options).map_err(io_err)?;
    zip.write_all(RELS_XML.as_bytes()).map_err(MeshError::Io)?;

    // 3D/3dmodel.model
    zip.start_file("3D/3dmodel.model", options)
        .map_err(io_err)?;

    let model_xml = build_model_xml(mesh);
    zip.write_all(model_xml.as_bytes()).map_err(MeshError::Io)?;

    zip.finish().map_err(io_err)?;
    Ok(())
}

fn io_err(e: zip::result::ZipError) -> MeshError {
    MeshError::Other(format!("ZIP error: {e}"))
}

fn build_model_xml(mesh: &IndexedMesh) -> String {
    let vertex_count = mesh.vertex_count();
    let face_count = mesh.face_count();
    let mut xml = String::with_capacity(estimate_model_xml_capacity(vertex_count, face_count));
    xml.push_str(MODEL_HEADER);

    // Vertices
    xml.push_str("      <vertices>\n");

    let mut id_to_idx: HashMap<crate::domain::core::index::VertexId, usize> =
        HashMap::with_capacity(vertex_count);
    for (idx, (vid, vdata)) in mesh.vertices.iter().enumerate() {
        id_to_idx.insert(vid, idx);
        let p = &vdata.position;
        xml.push_str(&format!(
            "        <vertex x=\"{}\" y=\"{}\" z=\"{}\" />\n",
            p.x, p.y, p.z
        ));
    }
    xml.push_str("      </vertices>\n");

    // Triangles
    xml.push_str("      <triangles>\n");
    for (_fid, face) in mesh.faces.iter_enumerated() {
        let v1 = id_to_idx[&face.vertices[0]];
        let v2 = id_to_idx[&face.vertices[1]];
        let v3 = id_to_idx[&face.vertices[2]];
        xml.push_str(&format!(
            "        <triangle v1=\"{v1}\" v2=\"{v2}\" v3=\"{v3}\" />\n"
        ));
    }
    xml.push_str("      </triangles>\n");

    xml.push_str(MODEL_FOOTER);
    xml
}

const ESTIMATED_VERTEX_XML_BYTES: usize = 80;
const ESTIMATED_TRIANGLE_XML_BYTES: usize = 64;

fn estimate_model_xml_capacity(vertex_count: usize, face_count: usize) -> usize {
    MODEL_HEADER.len()
        + MODEL_FOOTER.len()
        + "      <vertices>\n".len()
        + "      </vertices>\n".len()
        + "      <triangles>\n".len()
        + "      </triangles>\n".len()
        + vertex_count.saturating_mul(ESTIMATED_VERTEX_XML_BYTES)
        + face_count.saturating_mul(ESTIMATED_TRIANGLE_XML_BYTES)
}

const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>
"#;

const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>
"#;

const MODEL_HEADER: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US"
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
"#;

const MODEL_FOOTER: &str = r#"      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" />
  </build>
</model>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::core::scalar::Point3r;

    #[test]
    fn model_xml_capacity_covers_static_and_per_element_payloads() {
        let empty_capacity = estimate_model_xml_capacity(0, 0);
        assert!(empty_capacity >= MODEL_HEADER.len() + MODEL_FOOTER.len());
        assert_eq!(
            estimate_model_xml_capacity(3, 1) - empty_capacity,
            3 * ESTIMATED_VERTEX_XML_BYTES + ESTIMATED_TRIANGLE_XML_BYTES
        );
    }

    #[test]
    fn write_3mf_produces_zip_archive() {
        let mut mesh = IndexedMesh::new();
        let v0 = mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        mesh.add_face(v0, v1, v2);

        let mut cursor = std::io::Cursor::new(Vec::new());
        write_3mf(&mut cursor, &mesh).expect("3MF export should succeed");
        let bytes = cursor.into_inner();

        assert_eq!(&bytes[..4], b"PK\x03\x04");
        assert!(bytes
            .windows("3D/3dmodel.model".len())
            .any(|window| { window == b"3D/3dmodel.model" }));
    }
}
