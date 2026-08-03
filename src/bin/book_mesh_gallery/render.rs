use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use super::model::{GalleryResult, MeshCase, WatertightCase, WatertightRejection};

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

const DIAGNOSTIC_PANEL_WIDTH: f64 = 280.0;
const DIAGNOSTIC_PANEL_HEIGHT: f64 = 260.0;
const DIAGNOSTIC_COLUMNS: usize = 3;

fn diagnostic_mesh_panel(
    svg: &mut String,
    case: &WatertightCase,
    left: f64,
    top: f64,
) -> GalleryResult<()> {
    let positions: Vec<[f64; 3]> = case
        .mesh
        .vertices
        .positions()
        .map(|point| project([point.x, point.y, point.z]))
        .collect();
    let (min_x, max_x, min_y, max_y) = positions.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), [x, y, _]| {
            (min_x.min(*x), max_x.max(*x), min_y.min(*y), max_y.max(*y))
        },
    );
    let span_x = (max_x - min_x).max(1e-12);
    let span_y = (max_y - min_y).max(1e-12);
    let scale = 154.0 / span_x.max(span_y);
    let center_x = (min_x + max_x) * 0.5;
    let center_y = (min_y + max_y) * 0.5;
    let map = |[x, y, depth]: [f64; 3]| {
        [
            left + 140.0 + (x - center_x) * scale,
            top + 103.0 - (y - center_y) * scale,
            depth,
        ]
    };

    let face_stride =
        case.mesh.faces.len().saturating_add(MAX_DRAW_FACES - 1) / MAX_DRAW_FACES.max(1);
    let mut sampled_faces = Vec::new();
    for (index, face) in case.mesh.faces.iter().enumerate() {
        if index % face_stride.max(1) != 0 {
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
        let pa = map(pa);
        let pb = map(pb);
        let pc = map(pc);
        sampled_faces.push(([pa, pb, pc], (pa[2] + pb[2] + pc[2]) / 3.0));
    }
    sampled_faces.sort_by(|left, right| left.1.total_cmp(&right.1));

    let title = xml_escape(case.title);
    let _ = write!(
        svg,
        "<g><rect x=\"{left:.1}\" y=\"{top:.1}\" width=\"{DIAGNOSTIC_PANEL_WIDTH:.1}\" height=\"{DIAGNOSTIC_PANEL_HEIGHT:.1}\" rx=\"8\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"600\" fill=\"#0f172a\">{title}</text>",
        left + 140.0,
        top + 21.0
    );
    svg.push_str("<g stroke=\"#2563eb\" stroke-width=\"0.7\" stroke-linejoin=\"round\">");
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
    svg.push_str("</g>");

    let Some(edges) = case.mesh.edges_ref() else {
        return Err(std::io::Error::other("diagnostic edge store was not built").into());
    };
    for edge in edges.iter() {
        let Some(&pa) = positions.get(edge.vertices.0.as_usize()) else {
            continue;
        };
        let Some(&pb) = positions.get(edge.vertices.1.as_usize()) else {
            continue;
        };
        let pa = map(pa);
        let pb = map(pb);
        let (color, width) = if edge.is_non_manifold() {
            ("#ea580c", 2.2)
        } else if edge.is_boundary() {
            ("#dc2626", 2.0)
        } else {
            ("#0f172a", 0.45)
        };
        let _ = write!(
            svg,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{color}\" stroke-width=\"{width:.1}\" stroke-opacity=\"0.8\"/>",
            pa[0], pa[1], pb[0], pb[1]
        );
    }
    let status = if case.report.is_watertight {
        ("WATERTIGHT", "#15803d")
    } else {
        ("FAILS CHECK", "#b91c1c")
    };
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" font-weight=\"600\" fill=\"{}\">{}</text><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"8\" fill=\"#475569\">closed={} boundary={} nonmanifold={}</text><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"8\" fill=\"#475569\">orientation={} V={} F={} χ={}</text>",
        left + 140.0,
        top + 185.0,
        status.1,
        status.0,
        left + 140.0,
        top + 202.0,
        case.report.is_closed,
        case.report.boundary_edge_count,
        case.report.non_manifold_edge_count,
        left + 140.0,
        top + 217.0,
        case.report.orientation_consistent,
        case.mesh.vertex_count(),
        case.mesh.faces.len(),
        case
            .report
            .euler_characteristic
            .map_or_else(|| "n/a".to_owned(), |value| value.to_string()),
    );
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"8\" fill=\"#64748b\">red=boundary, orange=non-manifold</text></g>",
        left + 140.0,
        top + 235.0
    );
    Ok(())
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let next_len = current.len() + usize::from(!current.is_empty()) + word.len();
        if next_len > width && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn rejection_panel(svg: &mut String, rejection: &WatertightRejection, left: f64, top: f64) {
    let _ = write!(
        svg,
        "<g><rect x=\"{left:.1}\" y=\"{top:.1}\" width=\"{DIAGNOSTIC_PANEL_WIDTH:.1}\" height=\"{DIAGNOSTIC_PANEL_HEIGHT:.1}\" rx=\"8\" fill=\"#fff7ed\" stroke=\"#fdba74\"/><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"600\" fill=\"#0f172a\">{}</text><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"17\" font-weight=\"600\" fill=\"#b91c1c\">REJECTED</text><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"8\" fill=\"#475569\">public builder returned no mesh</text>",
        left + 140.0,
        top + 21.0,
        xml_escape(rejection.title),
        left + 140.0,
        top + 79.0,
        left + 140.0,
        top + 105.0,
    );
    for (index, line) in wrap_text(rejection.parameters, 42).iter().enumerate() {
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"8\" fill=\"#7c2d12\">{}</text>",
            left + 140.0,
            top + 128.0 + index as f64 * 11.0,
            xml_escape(line),
        );
    }
    for (index, line) in wrap_text(&rejection.error, 42).iter().enumerate() {
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"7\" fill=\"#7c2d12\">{}</text>",
            left + 140.0,
            top + 164.0 + index as f64 * 10.0,
            xml_escape(line),
        );
    }
    let _ = write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"8\" fill=\"#64748b\">geometry is intentionally omitted</text></g>",
        left + 140.0,
        top + 226.0,
    );
}

pub(crate) fn diagnostic_sheet(
    cases: &[WatertightCase],
    rejections: &[WatertightRejection],
    output: &Path,
) -> GalleryResult<()> {
    let panel_count = cases.len() + rejections.len();
    let rows = panel_count.saturating_add(DIAGNOSTIC_COLUMNS - 1) / DIAGNOSTIC_COLUMNS.max(1);
    let width = DIAGNOSTIC_PANEL_WIDTH * DIAGNOSTIC_COLUMNS as f64;
    let height = DIAGNOSTIC_PANEL_HEIGHT * rows as f64;
    let mut svg = String::with_capacity(panel_count * 25_000);
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-labelledby=\"diagnostic-title\">");
    svg.push_str("<title id=\"diagnostic-title\">Gaia watertightness diagnostics</title><desc>Generated from Gaia mesh values. Red edges are boundary edges, orange edges are non-manifold edges, and rejected branch panels contain no fabricated geometry.</desc>");
    let _ = write!(
        svg,
        "<rect width=\"{width:.0}\" height=\"{height:.0}\" fill=\"white\"/>"
    );
    for (index, case) in cases.iter().enumerate() {
        let column = index % DIAGNOSTIC_COLUMNS;
        let row = index / DIAGNOSTIC_COLUMNS;
        diagnostic_mesh_panel(
            &mut svg,
            case,
            column as f64 * DIAGNOSTIC_PANEL_WIDTH,
            row as f64 * DIAGNOSTIC_PANEL_HEIGHT,
        )?;
    }
    for (index, rejection) in rejections.iter().enumerate() {
        let index = cases.len() + index;
        let column = index % DIAGNOSTIC_COLUMNS;
        let row = index / DIAGNOSTIC_COLUMNS;
        rejection_panel(
            &mut svg,
            rejection,
            column as f64 * DIAGNOSTIC_PANEL_WIDTH,
            row as f64 * DIAGNOSTIC_PANEL_HEIGHT,
        );
    }
    svg.push_str("</svg>\n");
    fs::write(output, svg)?;
    Ok(())
}
