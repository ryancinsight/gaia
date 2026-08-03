use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use super::model::{GalleryResult, MeshCase};

const MAX_DRAW_FACES: usize = 900;
const PANEL_WIDTH: f64 = 280.0;
const PANEL_HEIGHT: f64 = 240.0;
const SHEET_COLUMNS: usize = 4;

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn project(point: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = point;
    [
        (x - y) * 0.866_025_403_8,
        (x + y) * 0.5 - z * 0.9,
        x + y + z,
    ]
}

fn render_panel(svg: &mut String, case: &MeshCase, left: f64, top: f64) {
    let positions: Vec<[f64; 3]> = case
        .mesh
        .vertices
        .positions()
        .map(|point| project([point.x, point.y, point.z]))
        .collect();

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for [x, y, _] in &positions {
        min_x = min_x.min(*x);
        max_x = max_x.max(*x);
        min_y = min_y.min(*y);
        max_y = max_y.max(*y);
    }
    let span_x = (max_x - min_x).max(1e-12);
    let span_y = (max_y - min_y).max(1e-12);
    let scale = 188.0 / span_x.max(span_y);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let map = |[x, y, depth]: [f64; 3]| {
        [
            left + 140.0 + (x - center_x) * scale,
            top + 116.0 - (y - center_y) * scale,
            depth,
        ]
    };

    let face_stride =
        case.mesh.faces.len().saturating_add(MAX_DRAW_FACES - 1) / MAX_DRAW_FACES.max(1);
    let face_stride = face_stride.max(1);
    let mut sampled_edges = BTreeSet::new();
    let mut sampled_faces = Vec::new();
    for (index, face) in case.mesh.faces.iter().enumerate() {
        if index % face_stride != 0 {
            continue;
        }
        let [a, b, c] = face.vertices;
        let Some(&pa) = positions.get(a.as_usize()) else {
            continue;
        };
        let Some(&pb) = positions.get(b.as_usize()) else {
            continue;
        };
        let Some(&pc) = positions.get(c.as_usize()) else {
            continue;
        };
        sampled_edges.extend(face.edges_canonical());
        let pa = map(pa);
        let pb = map(pb);
        let pc = map(pc);
        sampled_faces.push(([pa, pb, pc], (pa[2] + pb[2] + pc[2]) / 3.0));
    }
    sampled_faces.sort_by(|left, right| left.1.total_cmp(&right.1));

    let title = xml_escape(case.title);
    let _ = write!(
        svg,
        "<g><rect x=\"{left:.1}\" y=\"{top:.1}\" width=\"{PANEL_WIDTH:.1}\" height=\"{PANEL_HEIGHT:.1}\" rx=\"8\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>"
    );
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"600\" fill=\"#0f172a\">{title}</text>",
        left + 140.0,
        top + 21.0
    );
    let _ = write!(
        svg,
        "<g stroke=\"#2563eb\" stroke-width=\"0.7\" stroke-linejoin=\"round\">"
    );
    for (points, _) in sampled_faces {
        let _ = write!(
            svg,
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"#60a5fa\" fill-opacity=\"0.28\"/>",
            points[0][0],
            points[0][1],
            points[1][0],
            points[1][1],
            points[2][0],
            points[2][1]
        );
    }
    svg.push_str("</g><g stroke=\"#0f172a\" stroke-width=\"0.45\" stroke-opacity=\"0.68\">");
    for (a, b) in sampled_edges {
        let Some(&pa) = positions.get(a.as_usize()) else {
            continue;
        };
        let Some(&pb) = positions.get(b.as_usize()) else {
            continue;
        };
        let pa = map(pa);
        let pb = map(pb);
        let _ = write!(
            svg,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>",
            pa[0], pa[1], pb[0], pb[1]
        );
    }
    let _ = write!(
        svg,
        "</g><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"9\" fill=\"#475569\">V={} F={} C={}</text></g>",
        left + 140.0,
        top + 226.0,
        case.mesh.vertex_count(),
        case.mesh.faces.len(),
        case.mesh.cell_count()
    );
}

pub(crate) fn sheet(cases: &[MeshCase], title: &str, output: &Path) -> GalleryResult<()> {
    let rows = cases.len().saturating_add(SHEET_COLUMNS - 1) / SHEET_COLUMNS.max(1);
    let width = PANEL_WIDTH * SHEET_COLUMNS as f64;
    let height = PANEL_HEIGHT * rows as f64;
    let title = xml_escape(title);
    let mut svg = String::with_capacity(cases.len() * 40_000);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\" role=\"img\" aria-labelledby=\"sheet-title\"><title id=\"sheet-title\">{title}</title><desc>Generated from Gaia IndexedMesh values. Face display is bounded for readability; exact counts are in the manifest.</desc><rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
    );
    for (index, case) in cases.iter().enumerate() {
        let column = index % SHEET_COLUMNS;
        let row = index / SHEET_COLUMNS;
        render_panel(
            &mut svg,
            case,
            column as f64 * PANEL_WIDTH,
            row as f64 * PANEL_HEIGHT,
        );
    }
    svg.push_str("</svg>\n");
    fs::write(output, svg)?;
    Ok(())
}
