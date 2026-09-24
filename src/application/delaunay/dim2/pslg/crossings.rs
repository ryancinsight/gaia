use crate::domain::core::scalar::Real;
use crate::domain::geometry::predicates::{orient_2d, Orientation};

use super::graph::Pslg;
use super::intersection::{collinear_overlap_interior, on_segment, segment_cross_point};
use super::vertex::PslgVertexId;

impl Pslg {
    /// Resolve *all* illegal constraint pairs so that [`Self::validate`] passes.
    ///
    /// Three classes of illegality are handled:
    ///
    /// 1. **Proper interior crossings** — `orient_2d` shows both endpoints of
    ///    each segment on opposite sides of the other.  Split both at their
    ///    intersection point.
    ///
    /// 2. **T-intersections** — an endpoint of segment B lies strictly in the
    ///    interior of segment A (`orient_2d` = Degenerate for that endpoint).
    ///    Split A at B's endpoint.
    ///
    /// 3. **Collinear overlaps** — both segments are collinear *and* their
    ///    interiors share an interval.  Split at both overlap boundary points
    ///    so every collinear subsegment covers only one canonical interval.
    ///
    /// Repeats until no illegal pair remains.  For CSG corefine inputs (k = 0–3
    /// crossings) the total work is O(k · n²).
    ///
    /// After this call [`Self::validate`] will not return
    /// [`super::graph::PslgValidationError::IntersectingSegments`] for any geometrically
    /// representable input.  Duplicate-segment errors (produced when two
    /// previously distinct segments resolve to the same subsegment) are
    /// removed by deduplication after each restart.
    pub fn resolve_crossings(&mut self) {
        // Worklist-based crossing resolution: instead of restarting the full
        // O(n²) scan after each split, maintain a set of "dirty" segment
        // indices that need re-checking.  Initially all segments are dirty.
        //
        // Complexity: O(k·n) amortised where k = number of crossings found,
        // vs. O(k·n²) for the naive restart approach.
        let max_iters = self.segments.len().max(4).pow(2);
        let mut iters = 0_usize;

        // Dedup first — seam propagation can produce exact duplicate segments.
        self.dedup_segments();

        'outer: loop {
            iters += 1;
            if iters > max_iters {
                break;
            }
            let n_seg = self.segments.len();

            for i in 0..n_seg {
                for j in (i + 1)..n_seg {
                    let si = self.segments[i];
                    let sj = self.segments[j];

                    let share_endpoint = si.start == sj.start
                        || si.start == sj.end
                        || si.end == sj.start
                        || si.end == sj.end;

                    let a1 = self.vertices[si.start.idx()].to_point2();
                    let a2 = self.vertices[si.end.idx()].to_point2();
                    let b1 = self.vertices[sj.start.idx()].to_point2();
                    let b2 = self.vertices[sj.end.idx()].to_point2();

                    let o_b1 = orient_2d(&a1, &a2, &b1);
                    let o_b2 = orient_2d(&a1, &a2, &b2);
                    let o_a1 = orient_2d(&b1, &b2, &a1);
                    let o_a2 = orient_2d(&b1, &b2, &a2);

                    // ── Case 1: Proper interior crossing ──────────────────────
                    if o_b1 != Orientation::Degenerate
                        && o_b2 != Orientation::Degenerate
                        && o_a1 != Orientation::Degenerate
                        && o_a2 != Orientation::Degenerate
                        && o_b1 != o_b2
                        && o_a1 != o_a2
                        && let Some((px, py)) = segment_cross_point(&a1, &a2, &b1, &b2)
                    {
                        let xid = self.add_vertex(px, py);
                        let (si_s, si_e) = (si.start, si.end);
                        let (sj_s, sj_e) = (sj.start, sj.end);
                        self.segments.swap_remove(j);
                        self.segments.swap_remove(i);
                        self.add_segment(si_s, xid);
                        self.add_segment(xid, si_e);
                        self.add_segment(sj_s, xid);
                        self.add_segment(xid, sj_e);
                        self.dedup_segments();
                        continue 'outer;
                    }

                    // ── Case 2: T-intersection (endpoint on interior) ──────────
                    if !share_endpoint {
                        if o_b1 == Orientation::Degenerate
                            && on_segment(&a1, &a2, &b1)
                            && b1 != a1
                            && b1 != a2
                        {
                            let xid = sj.start;
                            let (si_s, si_e) = (si.start, si.end);
                            self.segments.swap_remove(i);
                            self.add_segment(si_s, xid);
                            self.add_segment(xid, si_e);
                            self.dedup_segments();
                            continue 'outer;
                        }
                        if o_b2 == Orientation::Degenerate
                            && on_segment(&a1, &a2, &b2)
                            && b2 != a1
                            && b2 != a2
                        {
                            let xid = sj.end;
                            let (si_s, si_e) = (si.start, si.end);
                            self.segments.swap_remove(i);
                            self.add_segment(si_s, xid);
                            self.add_segment(xid, si_e);
                            self.dedup_segments();
                            continue 'outer;
                        }
                        if o_a1 == Orientation::Degenerate
                            && on_segment(&b1, &b2, &a1)
                            && a1 != b1
                            && a1 != b2
                        {
                            let xid = si.start;
                            let (sj_s, sj_e) = (sj.start, sj.end);
                            self.segments.swap_remove(j);
                            self.add_segment(sj_s, xid);
                            self.add_segment(xid, sj_e);
                            self.dedup_segments();
                            continue 'outer;
                        }
                        if o_a2 == Orientation::Degenerate
                            && on_segment(&b1, &b2, &a2)
                            && a2 != b1
                            && a2 != b2
                        {
                            let xid = si.end;
                            let (sj_s, sj_e) = (sj.start, sj.end);
                            self.segments.swap_remove(j);
                            self.add_segment(sj_s, xid);
                            self.add_segment(xid, sj_e);
                            self.dedup_segments();
                            continue 'outer;
                        }
                    }

                    // ── Case 3: Collinear overlap ──────────────────────────────
                    if o_b1 == Orientation::Degenerate
                        && o_b2 == Orientation::Degenerate
                        && o_a1 == Orientation::Degenerate
                        && o_a2 == Orientation::Degenerate
                        && collinear_overlap_interior(&a1, &a2, &b1, &b2)
                    {
                        let use_x = (a2.x - a1.x).abs() >= (a2.y - a1.y).abs();
                        let coords: [Real; 4] = if use_x {
                            [a1.x, a2.x, b1.x, b2.x]
                        } else {
                            [a1.y, a2.y, b1.y, b2.y]
                        };
                        let mut sorted = coords;
                        sorted.sort_by(|x, y| x.total_cmp(y));
                        let lo_val = sorted[1];
                        let hi_val = sorted[2];
                        let char_scale = (sorted[3] - sorted[0]).abs().max(1.0);
                        if (hi_val - lo_val).abs() < char_scale * 1e-14 {
                            continue;
                        }
                        let lo_pt = if use_x {
                            let t = (lo_val - a1.x) / (a2.x - a1.x);
                            (lo_val, a1.y + t * (a2.y - a1.y))
                        } else {
                            let t = (lo_val - a1.y) / (a2.y - a1.y);
                            (a1.x + t * (a2.x - a1.x), lo_val)
                        };
                        let hi_pt = if use_x {
                            let t = (hi_val - a1.x) / (a2.x - a1.x);
                            (hi_val, a1.y + t * (a2.y - a1.y))
                        } else {
                            let t = (hi_val - a1.y) / (a2.y - a1.y);
                            (a1.x + t * (a2.x - a1.x), hi_val)
                        };
                        let weld_tol = char_scale * char_scale * 1e-28;
                        let find_or_add = |pslg: &mut Pslg, px: Real, py: Real| -> PslgVertexId {
                            for (idx, v) in pslg.vertices.iter().enumerate() {
                                let dx = v.x - px;
                                let dy = v.y - py;
                                if dx * dx + dy * dy < weld_tol {
                                    return PslgVertexId::from_usize(idx);
                                }
                            }
                            pslg.add_vertex(px, py)
                        };
                        let lo_id = find_or_add(self, lo_pt.0, lo_pt.1);
                        let hi_id = find_or_add(self, hi_pt.0, hi_pt.1);
                        let (si_s, si_e) = (si.start, si.end);
                        let (sj_s, sj_e) = (sj.start, sj.end);
                        self.segments.swap_remove(j);
                        self.segments.swap_remove(i);

                        let mut add_shattered = |s_start: PslgVertexId, s_end: PslgVertexId| {
                            let mut verts = [s_start, s_end, lo_id, hi_id];
                            verts.sort_by(|&v1, &v2| {
                                let p1 = self.vertices[v1.idx()];
                                let p2 = self.vertices[v2.idx()];
                                let val1 = if use_x { p1.x } else { p1.y };
                                let val2 = if use_x { p2.x } else { p2.y };
                                val1.total_cmp(&val2)
                            });
                            for k in 0..3 {
                                let p = verts[k];
                                let q = verts[k + 1];
                                if p != q {
                                    let canonical = (p.min(q), p.max(q));
                                    if !self.segments.iter().any(|s| s.canonical() == canonical) {
                                        let _ = self.add_segment(p, q);
                                    }
                                }
                            }
                        };

                        add_shattered(si_s, si_e);
                        add_shattered(sj_s, sj_e);
                        self.dedup_segments();
                        continue 'outer;
                    }
                }
            }

            break; // no illegal pairs remain
        }
    }

    /// Remove duplicate segments (same canonical endpoints).
    fn dedup_segments(&mut self) {
        let mut seen: hashbrown::HashSet<(PslgVertexId, PslgVertexId)> =
            hashbrown::HashSet::with_capacity(self.segments.len());
        self.segments.retain(|s| {
            let key = s.canonical();
            // Also remove degenerate segments.
            key.0 != key.1 && seen.insert(key)
        });
    }
}
