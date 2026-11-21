"""Measure the lab's own two-column electrode designs (tWSe2-1 KLayout session) and write a report.

    python tools/learn_lab_designs.py [--session PATH.lys] [--out docs/wse2_design_report.md]
                                      [--images docs/training] [--json PATH]

Reads the session's views (Stack 1/3/4.gds + aligned microscope images), runs
:func:`ustack.lab_designs.analyse_design` on each layout, writes a Markdown report (per-stack
table, consensus across stacks, comparison with UStack's rules / presets and Hall-bar model) and
overlay images (layer 5/6/7 outlines on the aligned 100x images, cropped around the device).

The numbers are measurements of the lab's drawings, not approved values (AGENTS.md rule 1).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ustack import lab_designs as ld  # noqa: E402
from ustack.config import PRESET_DEFAULTS  # noqa: E402

SESSION = ROOT / "WSe2 device training" / "tWSe2-1_KLayout_session" / "tWSe2-1.lys"
OVERLAY_DIRS = ("3 - tWSe2 stack/AFM clean/", "TS/")
COLOURS = {"fingers": (60, 60, 255), "contact_gate": (60, 220, 60), "electrodes": (255, 60, 255), "markers": (0, 230, 255)}
LEGEND = {"fingers": "5 fingers", "contact_gate": "6 contact gate", "electrodes": "7 electrodes", "markers": "3 markers"}
OUT_WIDTH = 1000


# ---------------------------------------------------------------------------- overlays
def render_overlay(view: ld.SessionView, img: ld.SessionImage, analysis: ld.DesignAnalysis,
                   roles: dict, out_path: Path, margin_um: float = 22.0) -> dict | None:
    """Draw the design outlines on one aligned image, cropped around the device."""
    im = cv2.imread(str(img.path))
    if im is None:
        return None
    H, W = im.shape[:2]
    cx, cy = analysis.frame["device_centre_chip_um"]
    fin = ld.layer_polygons(view.gds, roles["fingers"], (cx, cy), 80.0)
    xs = [p[0] for poly in fin for p in poly]
    ys = [p[1] for poly in fin for p in poly]
    x0, x1, y0, y1 = min(xs) - margin_um, max(xs) + margin_um, min(ys) - margin_um, max(ys) + margin_um
    cols, rows = img.chip_to_px([x0, x1, x1, x0], [y0, y0, y1, y1], W, H)
    c0, c1 = int(max(0, np.floor(cols.min()))), int(min(W, np.ceil(cols.max())))
    r0, r1 = int(max(0, np.floor(rows.min()))), int(min(H, np.ceil(rows.max())))
    if c1 - c0 < 20 or r1 - r0 < 20:
        return None
    crop = im[r0:r1, c0:c1]
    scale = OUT_WIDTH / crop.shape[1]
    crop = cv2.resize(crop, (OUT_WIDTH, int(round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    for role in ("markers", "fingers", "contact_gate", "electrodes"):
        if role not in roles:
            continue
        for poly in ld.layer_polygons(view.gds, roles[role], (cx, cy), 150.0):
            pc, pr = img.chip_to_px([p[0] for p in poly], [p[1] for p in poly], W, H)
            pts = np.stack([(pc - c0) * scale, (pr - r0) * scale], axis=1)
            cv2.polylines(crop, [np.round(pts * 4).astype(np.int32)], True, COLOURS[role], 2, cv2.LINE_AA, shift=2)
    # legend and a 10 µm scale bar
    y = 24
    for role in ("fingers", "contact_gate", "electrodes", "markers"):
        cv2.putText(crop, LEGEND[role], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop, LEGEND[role], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS[role], 2, cv2.LINE_AA)
        y += 24
    bar_px = 10.0 / img.um_per_px * scale
    hh = crop.shape[0]
    cv2.line(crop, (OUT_WIDTH - 30 - int(bar_px), hh - 24), (OUT_WIDTH - 30, hh - 24), (255, 255, 255), 4)
    cv2.putText(crop, "10 um", (OUT_WIDTH - 30 - int(bar_px), hh - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return {"file": out_path, "source": img.file, "um_per_px": img.um_per_px}


# ---------------------------------------------------------------------------- report helpers
def fmt(v, nd=2) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return "–".join(fmt(x, nd) for x in v) if len(v) == 2 and all(isinstance(x, (int, float)) for x in v) else ", ".join(fmt(x, nd) for x in v)
    if isinstance(v, float):
        txt = f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:g}"
        return "0" if txt in ("-0", "") else txt
    return str(v)


def rows_for(a: ld.DesignAnalysis) -> dict[str, object]:
    """The quantities of the per-stack table (label → value), in display order."""
    t, b = a.columns.get("top", {}), a.columns.get("bottom", {})
    cg, es, ar = a.contact_gate, a.electrode_summary, a.array
    both = lambda k: f"{fmt(t.get(k))} / {fmt(b.get(k))}"  # noqa: E731
    med2 = lambda k: _med([t.get(k), b.get(k)])  # noqa: E731
    neither = es.get("neither_inner_ends_local_um") or []
    land = (a.landing or {}).get("smallest") or {}
    return {
        "Finger axis on the chip (deg)": a.frame.get("finger_axis_deg"),
        "Fingers per column (top / bottom)": f"{t.get('n', 0)} / {b.get('n', 0)}",
        "Finger width at the tip (µm)": med2("tip_width_um"),
        "Finger pitch at the tips (µm)": med2("pitch_at_tips_um"),
        "Finger spacing (edge-to-edge) at the tips (µm)": med2("spacing_at_tips_um"),
        "Column gap, tip to tip (µm)": ar.get("column_gap_um"),
        "Tips aligned: spread of tip line / top–bottom offset (µm)": f"{fmt(max(t.get('tip_alignment_spread_um', 0), b.get('tip_alignment_spread_um', 0)), 3)} / {fmt(ar.get('tips_facing', {}).get('max_abs_offset_um'), 3)}",
        "Array span along the gap at the tips (µm)": ar.get("span_at_tips_um"),
        "Straight section from the tip = bend point (µm, top / bottom)": both("straight_length_um"),
        "Finger centre-line length (µm, top / bottom median)": both("centreline_length_um"),
        "Finger extent along its axis (µm, top / bottom)": both("extent_along_axis_um"),
        "Full-length mean width (µm, top / bottom)": both("mean_width_um"),
        "Width beyond the bend (µm, top / bottom)": both("outer_width_um"),
        "Fan-out angle beyond the bend (deg, min–max, top)": t.get("outer_angle_deg_range"),
        "Fan-out angle beyond the bend (deg, min–max, bottom)": b.get("outer_angle_deg_range"),
        "Pitch at the far ends (µm, top / bottom)": both("far_end_pitch_um"),
        "Whole finger array (µm, along gap × across)": ar.get("total_extent_um"),
        "Contact-gate shapes": cg.get("n_shapes"),
        "Fingers under the CG (top / bottom)": f"{cg.get('top_fingers_covered', 0)} / {cg.get('bottom_fingers_covered', 0)}",
        "CG width on a finger (µm)": _med([cg.get("top_cg_width_on_finger_um"), cg.get("bottom_cg_width_on_finger_um")]),
        "CG overlap along the finger from the tip (µm, top / bottom)": f"{fmt(cg.get('top_overlap_length_um'))} / {fmt(cg.get('bottom_overlap_length_um'))}",
        "CG start, measured from the tip (µm)": _med([cg.get("top_overlap_from_tip_um"), cg.get("bottom_overlap_from_tip_um")]),
        "CG clearance to the gap over the array (µm; 0 = flush with tips)": fmt(_med([cg.get("top_clearance_to_gap_um"), cg.get("bottom_clearance_to_gap_um")]), 3),
        "CG crosses the gap outside the array (distance from array, µm)": (f"yes ({fmt(cg.get('gap_crossing_distance_from_array_um'))})" if cg.get("crosses_gap_outside_array") else ("no" if cg.get("n_shapes") else "–")),
        "Layer-7 electrodes near the device": es.get("n_near_device"),
        "… on fingers / on CG / neither": f"{es.get('by_kind', {}).get('finger', 0)} / {es.get('by_kind', {}).get('contact_gate', 0)} / {es.get('by_kind', {}).get('neither', 0)}",
        "Finger electrodes per column (top / bottom)": f"{es.get('finger_electrodes_per_column', {}).get('top', 0)} / {es.get('finger_electrodes_per_column', {}).get('bottom', 0)}",
        "Electrodes on more than one finger": es.get("electrodes_on_more_than_one_finger"),
        "Floating fingers (no electrode; top / bottom)": f"{es.get('n_floating', {}).get('top', 0)} / {es.get('n_floating', {}).get('bottom', 0)}",
        "Finger-electrode width near the device (µm, median)": es.get("finger_electrode_width_um"),
        "Electrode width near the device, all (µm, min–max)": es.get("width_near_device_range_um"),
        "Min electrode–electrode spacing near the device (µm)": es.get("min_spacing_near_device_um"),
        "Median nearest-neighbour electrode spacing (µm)": es.get("nearest_neighbour_spacing_median_um"),
        "Electrode lands on the finger at, from the tip (µm, median)": es.get("overlap_from_tip_um"),
        "… landing point range (µm)": es.get("overlap_from_tip_range_um"),
        "Electrode overlap length along the finger (µm, median)": es.get("overlap_length_along_finger_um"),
        "… overlap length range (µm)": es.get("overlap_length_range_um"),
        "Min clearance electrode → other fingers (µm)": es.get("clearance_to_other_fingers_min_um"),
        "Min clearance finger electrode → CG (µm)": es.get("clearance_to_contact_gate_min_um"),
        "All electrodes reach layer 4": es.get("all_reach_pads"),
        "'Neither' electrodes: inner ends (u, v) µm": "; ".join(f"({fmt(p[0], 1)}, {fmt(p[1], 1)})" for p in neither) or "–",
        "Landing square (layer 1): size / device offset from its centre (µm)": (f"{fmt(land.get('size_um'))} / ({fmt(-land['centre_offset_um'][0], 1)}, {fmt(-land['centre_offset_um'][1], 1)})" if land else "–"),
        "Pad-free square around the device (half-size, µm)": (a.markers or {}).get("pad_free_half_size_um"),
    }


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 4) if xs else None


def consensus(analyses: list[ld.DesignAnalysis]) -> list[tuple[str, str, str, str]]:
    """(quantity, median, spread, stacks used) over the stacks that have the quantity."""
    def col(a, c, k):
        return a.columns.get(c, {}).get(k)

    items = [
        ("Finger width at the tip (µm)", lambda a: [col(a, c, "tip_width_um") for c in ("top", "bottom")]),
        ("Finger pitch at the tips (µm)", lambda a: [col(a, c, "pitch_at_tips_um") for c in ("top", "bottom")]),
        ("Finger spacing at the tips (µm)", lambda a: [col(a, c, "spacing_at_tips_um") for c in ("top", "bottom")]),
        ("Column gap, tip to tip (µm)", lambda a: [a.array.get("column_gap_um")]),
        ("Fingers per column", lambda a: [col(a, c, "n") for c in ("top", "bottom")]),
        ("Array span at the tips (µm)", lambda a: [a.array.get("span_at_tips_um")]),
        ("Straight section from the tip (µm)", lambda a: [col(a, c, "straight_length_um") for c in ("top", "bottom")]),
        ("Finger centre-line length (µm)", lambda a: [col(a, c, "centreline_length_um") for c in ("top", "bottom")]),
        ("Finger width beyond the bend (µm)", lambda a: [col(a, c, "outer_width_um") for c in ("top", "bottom")]),
        ("Pitch at the far ends (µm)", lambda a: [col(a, c, "far_end_pitch_um") for c in ("top", "bottom")]),
        ("CG width on a finger (µm)", lambda a: [a.contact_gate.get(f"{c}_cg_width_on_finger_um") for c in ("top", "bottom")]),
        ("CG overlap length from the tip (µm)", lambda a: [a.contact_gate.get(f"{c}_overlap_length_um") for c in ("top", "bottom")]),
        ("CG clearance to the gap (µm)", lambda a: [a.contact_gate.get(f"{c}_clearance_to_gap_um") for c in ("top", "bottom")]),
        ("Finger-electrode width near the device (µm)", lambda a: [a.electrode_summary.get("finger_electrode_width_um")]),
        ("Min electrode spacing near the device (µm)", lambda a: [a.electrode_summary.get("min_spacing_near_device_um")]),
        ("Median nearest-neighbour electrode spacing (µm)", lambda a: [a.electrode_summary.get("nearest_neighbour_spacing_median_um")]),
        ("Electrode lands on the finger at, from the tip (µm)", lambda a: [a.electrode_summary.get("overlap_from_tip_um")]),
        ("Electrode overlap length along the finger (µm)", lambda a: [a.electrode_summary.get("overlap_length_along_finger_um")]),
        ("Min clearance electrode → other fingers (µm)", lambda a: [a.electrode_summary.get("clearance_to_other_fingers_min_um")]),
        ("'Neither' (S/D by the layer rule) electrodes", lambda a: [a.electrode_summary.get("by_kind", {}).get("neither")] if a.electrode_summary.get("n_near_device") else []),
    ]
    out = []
    for label, get in items:
        per = []
        used = []
        for a in analyses:
            vals = [v for v in get(a) if isinstance(v, (int, float))]
            if vals:
                per.append(statistics.median(vals))
                used.append(a.name)
        if not per:
            continue
        allv = [v for a in analyses for v in get(a) if isinstance(v, (int, float))]
        out.append((label, fmt(statistics.median(per), 3), f"{fmt(min(allv), 3)} – {fmt(max(allv), 3)}", ", ".join(used)))
    return out


def ustack_values() -> tuple[dict, dict]:
    rules = json.loads((ROOT / "config" / "rules.json").read_text(encoding="utf-8")).get("rules", {})
    presets = {k: dict(v) for k, v in PRESET_DEFAULTS.items()}
    pfile = ROOT / "config" / "presets.json"
    if pfile.is_file():
        for k, v in json.loads(pfile.read_text(encoding="utf-8")).get("presets", {}).items():
            presets[k] = dict(presets.get(k, {}), **v)
    return rules, presets


def rv(rules, rid):
    r = rules.get(rid) or {}
    return r.get("value"), r.get("status", "missing")


def pv(presets, key):
    p = presets.get(key) or {}
    return p.get("value"), p.get("status", "program")


# ---------------------------------------------------------------------------- report
def write_report(views, analyses, overlays, out: Path) -> None:
    rules, presets = ustack_values()
    by = {a.name: a for a in analyses}
    names = [a.name for a in analyses]
    L = []
    w = L.append
    w("# tWSe2 batch 1: what the lab's own designs measure")
    w("")
    w("> **These are measurements of the lab's own GDS drawings (Stack 1, 3, 4 of the tWSe2-1 KLayout "
      "session), made by `tools/learn_lab_designs.py` with `ustack/lab_designs.py`. They are not "
      "approved values.** None of them is written to `config/`. The lab has to confirm a number before "
      "it becomes a rule, preset or default (AGENTS.md rule 1).")
    w("")
    w("## Data and method")
    w("")
    w(f"- Session: `{SESSION.relative_to(ROOT)}`. Views in order: " + ", ".join(f"{v.index + 1} = {v.name}" for v in views) + ".")
    w("- Layer meanings: the user's message of 2026-11-16 (`WSE2_BATCH1_LAYERS`): 1 = device landing square, "
      "3 = alignment markers, 4 = bond pads and fan-out, 5 = pre-patterned fingers (\"hall bars\") in two "
      "columns, 6 = contact gate, 7 = electrodes. The session names 5/0 *InnerContacts*, 6/0 *CG* and "
      "7/0 *OuterContacts*.")
    w("- Local frame: the chip rotated so the fingers run along +v (top column at v > 0), origin at the "
      "middle of the column gap and of the finger array. *Tip* = the finger end at the gap; distances "
      "\"from the tip\" are along the finger axis.")
    w("- Widths come from slices across the finger axis (every 0.1 µm); the *straight section* ends where the "
      "centre or width changes by more than 0.01 µm. Mean widths and lengths of bent pieces are the "
      "equivalent strip (area A, half perimeter P/2: w·l = A, w + l = P/2). Electrode width near the "
      "device = the strip width of the electrode within ±10 µm of its innermost vertex.")
    w("- Electrodes count as near the device within 60 µm of the finger array. Layer-7 shapes there "
      "that touch none of layers 4/5/6 (text labels such as `S1`, `tWSe2 - 1`) are left out.")
    w("- Image placement: KLayout image matrix, u = col − W/2, v = H/2 − row, [x, y, w] = M·[u, v, 1]. "
      "100x images come out at " + ", ".join(sorted({fmt(i.um_per_px, 4) for v in views for i in v.images
                                                      if i.exists and i.magnification == '100x'})) + " µm/px. "
      "Missing image files referenced by the session: " + (", ".join(sorted({f'`{i.file}`' for v in views for i in v.images if not i.exists})) or "none") + ".")
    w("")
    w("## Per-stack measurements")
    w("")
    w("| Quantity | " + " | ".join(names) + " |")
    w("|---|" + "---|" * len(names))
    tables = {a.name: rows_for(a) for a in analyses}
    labels = list(tables[names[0]].keys())
    for lab in labels:
        w(f"| {lab} | " + " | ".join(fmt(tables[n].get(lab)) for n in names) + " |")
    w("")
    w("Stack 4 never reached the electrode step (no layer 7), so its electrode rows are empty and all its "
      "fingers count as floating.")
    w("")
    w("## Consensus across the stacks")
    w("")
    w("Median of the per-stack medians; the spread is the smallest and largest value seen on any column of any stack.")
    w("")
    w("| Quantity | Median | Spread | Stacks |")
    w("|---|---|---|---|")
    for row in consensus(analyses):
        w("| " + " | ".join(row) + " |")
    w("")
    w("## What the drawings show")
    w("")
    w(_observations(by))
    w("")
    w("## Compared with UStack's rules and presets")
    w("")
    w(_comparison(by, rules, presets))
    w("")
    w("## Where the lab's design differs from UStack's Hall-bar model")
    w("")
    w(_mismatches(by, presets))
    w("")
    w("## Overlays")
    w("")
    w("Red = layer 5 fingers, green = layer 6 contact gate, magenta = layer 7 electrodes, yellow = layer 3 "
      "markers, drawn with the session's image matrix on the aligned 100x bright-field images and "
      "cropped around the device.")
    w("")
    for o in overlays:
        rel = o["file"].relative_to(out.parent)
        w(f"**{o['stack']}**, `{o['source']}` ({fmt(o['um_per_px'], 4)} µm/px)")
        w("")
        w(f"![{o['stack']} overlay]({rel.as_posix()})")
        w("")
    w("## Reproduce")
    w("")
    w("```")
    w("~/Library/Caches/UStack-build/venv-arm64/bin/python tools/learn_lab_designs.py")
    w("```")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def _observations(by: dict) -> str:
    s1, s3, s4 = by.get("Stack 1"), by.get("Stack 3"), by.get("Stack 4")
    out = []
    if s1 and s3 and s4:
        out.append(f"- **Two facing columns of identical straight fingers.** Every finger is "
                   f"{fmt(s1.columns['top']['tip_width_um'])} µm wide at the tip; the tips of each column "
                   f"lie on one line and the top and bottom tips sit exactly opposite each other (offset ≤ "
                   f"{fmt(max(a.array['tips_facing']['max_abs_offset_um'] for a in (s1, s3, s4)), 3)} µm). Pitch "
                   f"{fmt(s1.columns['top']['pitch_at_tips_um'])} µm (Stack 1) and "
                   f"{fmt(s3.columns['top']['pitch_at_tips_um'])} µm (Stacks 3, 4), i.e. spacing "
                   f"{fmt(s1.columns['top']['spacing_at_tips_um'])} / {fmt(s3.columns['top']['spacing_at_tips_um'])} µm. "
                   f"The column gap changes with the flake: {fmt(s1.array['column_gap_um'])} µm (Stack 1), "
                   f"{fmt(s3.array['column_gap_um'])} µm (Stack 3), {fmt(s4.array['column_gap_um'])} µm (Stack 4).")
        out.append(f"- **Number of fingers follows the flake**: {s1.columns['top']['n']}/{s3.columns['top']['n']}/"
                   f"{s4.columns['top']['n']} per column; array span at the tips "
                   f"{fmt(s1.array['span_at_tips_um'], 1)}, {fmt(s3.array['span_at_tips_um'], 1)}, "
                   f"{fmt(s4.array['span_at_tips_um'], 1)} µm.")
        out.append(f"- **Straight, then fan out.** The fingers run straight from the tip for "
                   f"{fmt(s1.columns['top']['straight_length_um'], 1)} µm (Stack 1), "
                   f"{fmt(s3.columns['top']['straight_length_um'], 1)} / {fmt(s3.columns['bottom']['straight_length_um'], 1)} µm "
                   f"(Stack 3, top / bottom) and {fmt(s4.columns['top']['straight_length_um'], 1)} µm (Stack 4), then bend "
                   f"outward (up to ±{fmt(max(abs(x) for a in (s1, s3, s4) for c in ('top', 'bottom') for x in (a.columns[c]['outer_angle_deg_range'] or [0])), 0)}° "
                   f"from the finger axis), mostly widen to ≈1.0–1.2 µm, and spread to a far-end pitch of ≈3–4.6 µm. "
                   f"Stack 3's top fingers have a second bend (dog-leg); its outer width stays 0.78 µm.")
        out.append(f"- **Stack 4's array is rotated** {fmt(s4.frame['finger_axis_deg'] - 90, 1)}° on the chip "
                   f"(fingers at {fmt(s4.frame['finger_axis_deg'], 1)}°); Stacks 1 and 3 are axis-aligned.")
        out.append("- **Contact gate = a comb on top of the fingers.** Each CG tooth is a copy of the finger tip it gates "
                   "(same 0.8 µm width, starting flush with the tip, no side extension, no reach into the gap), "
                   f"{fmt(s3.contact_gate['top_overlap_length_um'], 1)}–{fmt(s1.contact_gate['top_overlap_length_um'], 1)} µm long, "
                   "joined by a bus bar behind the teeth and led out sideways. Only the fingers that get an electrode are "
                   f"gated (Stack 1: 7 + 7 of 9 + 9; Stack 3: 8 + 6, the same fingers as the electrodes). Stack 1 and 4 "
                   f"draw one CG per column; Stack 3 draws one CG for both columns that crosses the gap "
                   f"{fmt(s3.contact_gate.get('gap_crossing_distance_from_array_um'), 1)} µm beside the array.")
        e1, e3 = s1.electrode_summary, s3.electrode_summary
        out.append(f"- **One electrode per finger, landing on the fanned part.** Every layer-7 finger electrode "
                   f"touches exactly one finger, starting {fmt(e1['overlap_from_tip_range_um'][0], 1)}–"
                   f"{fmt(max(e1['overlap_from_tip_range_um'][1], e3['overlap_from_tip_range_um'][1]), 1)} µm from the tip (beyond the CG "
                   f"teeth; min clearance to the CG {fmt(min(e1['clearance_to_contact_gate_min_um'], e3['clearance_to_contact_gate_min_um']), 2)} µm) "
                   f"and running {fmt(e3['overlap_length_range_um'][0], 1)}–{fmt(e1['overlap_length_range_um'][1], 1)} µm along it. "
                   f"Electrodes are ≈{fmt(e3['finger_electrode_width_um'], 1)}–{fmt(e1['finger_electrode_width_um'], 1)} µm wide "
                   f"there; neighbouring electrodes come as close as {fmt(e1['min_spacing_near_device_um'])} µm (Stack 1) / "
                   f"{fmt(e3['min_spacing_near_device_um'])} µm (Stack 3). All reach layer 4 (the fan-out) about 200 µm out.")
        fl1, fl3 = e1["floating_fingers"], e3["floating_fingers"]
        out.append(f"- **Floating fingers.** Stack 3: {len(fl3['top'])} top + {len(fl3['bottom'])} bottom finger(s) "
                   f"without an electrode (top #{fl3['top']}, bottom #{fl3['bottom']}, counted left to right), as the user said. "
                   f"**Stack 1 also leaves {len(fl1['top'])} + {len(fl1['bottom'])} fingers floating** (top #{fl1['top']}, "
                   f"bottom #{fl1['bottom']}) and has no CG over them, so the same practice (connect only the fingers "
                   "the flake reaches) seems to apply there; worth confirming with the lab.")
        n1 = e1["neither_inner_ends_local_um"]
        n3 = e3["neither_inner_ends_local_um"]
        out.append(f"- **The 'neither' electrodes (S/D by the layer rule) are not at the device.** Each stack has two, and "
                   f"they end on the gap axis (|v| ≤ {fmt(max(abs(p[1]) for p in n1 + n3), 1)} µm) far beyond the finger array: "
                   f"u = {', '.join(fmt(p[0], 1) for p in n1)} µm (Stack 1) and {', '.join(fmt(p[0], 1) for p in n3)} µm (Stack 3), "
                   "with the array ending at ±9–10 µm. They end in a ≈3.5 µm-wide bar across the long dark strip that "
                   "runs through the gap in the overlays (the back-gate flake), so they look like **back-gate contacts at "
                   "both ends of the BG**, not source/drain. The lab's drawings have no dedicated S/D electrodes: any "
                   "finger can serve as source or drain.")
        out.append(f"- **Landing square and markers.** Layer 1 holds a 400 × 400 µm square (plus the 6 mm write field and "
                   f"the 22 × 15 mm chip); the device sits within {fmt(max(abs(c) for a in (s1, s3, s4) for c in a.landing['smallest']['centre_offset_um']), 0)} µm of its centre. "
                   "Near the device there are 2.5 µm checkerboard marker pairs (two squares touching at a corner) at "
                   "(±40, ±40) µm from the square's centre; at the top-left corner one 2.5 µm square sits next to a 5 µm "
                   "square (an orientation mark). 10 µm squares sit at (±195, ±195) µm. "
                   "Pads/fan-out (layer 4) stay ≥ ≈175 µm from the device.")
    return "\n".join(out)


def _comparison(by: dict, rules: dict, presets: dict) -> str:
    s1, s3, s4 = by.get("Stack 1"), by.get("Stack 3"), by.get("Stack 4")
    trio = [a for a in (s1, s3, s4) if a]
    ew, ews = rv(rules, "ebl.electrode_width_target_um")
    es_, ess = rv(rules, "ebl.electrode_spacing_target_um")
    cw, cws = rv(rules, "ebl.contact_width_min_um")
    cs, css = rv(rules, "ebl.contact_spacing_min_um")
    co, cos = rv(rules, "ebl.contact_overlap_um")
    sh, shs = rv(rules, "stack.alignment_shift_um")
    npairs, _ = pv(presets, "hall_bar.n_probe_pairs")
    wpa, _ = pv(presets, "hall_bar.width_per_arm")
    ppw, _ = pv(presets, "hall_bar.pitch_per_width")
    tmd, _ = pv(presets, "prepattern.tmd_overlap_spacings")
    cgr, _ = pv(presets, "cg.reach_spacings")
    gm, _ = pv(presets, "gates.margin_spacings")
    gaps = [a.array["column_gap_um"] for a in trio]
    pitches = [a.columns["top"]["pitch_at_tips_um"] for a in trio]
    spacings = [a.columns["top"]["spacing_at_tips_um"] for a in trio]
    straight = [a.columns[c]["straight_length_um"] for a in trio for c in ("top", "bottom")]
    el = [a for a in trio if a.electrode_summary.get("n_near_device")]
    ewid = [a.electrode_summary["finger_electrode_width_um"] for a in el]
    emin = [a.electrode_summary["min_spacing_near_device_um"] for a in el]
    ovl = [a.electrode_summary["overlap_length_along_finger_um"] for a in el]
    nfing = [a.columns["top"]["n"] for a in trio]
    sp = es_ or 0
    rows = [
        ("`ebl.electrode_width_target_um`", f"{fmt(ew)} ({ews})", f"finger tip width 0.8 µm in every stack; layer-7 electrodes {fmt(min(ewid), 2)}–{fmt(max(ewid), 2)} µm near the device",
         "Fingers sit exactly on the target; the EBL electrodes are drawn about twice as wide."),
        ("`ebl.contact_width_min_um`", f"{fmt(cw)} ({cws})", "narrowest drawn finger 0.78 µm (Stack 3 outer part)", "Above the hard minimum everywhere."),
        ("`ebl.electrode_spacing_target_um`", f"{fmt(es_)} ({ess})", f"finger spacing at the tips {fmt(min(spacings))}–{fmt(max(spacings))} µm; electrode–electrode min {fmt(min(emin))}–{fmt(max(emin))} µm",
         "Fingers are well above the target; the layer-7 fan-out goes **below** the 0.8 µm target (a DRC warning in UStack)."),
        ("`ebl.contact_spacing_min_um`", f"{fmt(cs)} ({css})", f"electrode–electrode min {fmt(min(emin))} µm", "Above the hard minimum."),
        ("`hall_bar.n_probe_pairs` (preset)", fmt(npairs), f"{', '.join(str(n) for n in nfing)} fingers per column (Stacks 1, 3, 4)",
         "The lab draws as many fingers as fit along the flake, not 2 pairs."),
        ("`hall_bar.width_per_arm` (preset): W = this × arm width", f"{fmt(wpa)} → W = {fmt((wpa or 0) * (ew or 0))} µm",
         f"Mak layout: W = column gap = {', '.join(fmt(g) for g in gaps)} µm, i.e. {fmt(min(gaps) / 0.8, 2)}–{fmt(max(gaps) / 0.8, 2)} × the finger width",
         "The gap is set by the BG / flake, not by a ratio to the arm width."),
        ("`hall_bar.pitch_per_width` (preset): pitch = this × W", f"{fmt(ppw)} → {fmt((ppw or 0) * min(gaps))}–{fmt((ppw or 0) * max(gaps))} µm for these gaps",
         f"pitch {fmt(min(pitches))}–{fmt(max(pitches))} µm = finger width + {fmt(min(spacings))}–{fmt(max(spacings))} µm",
         f"The lab's pitch is set by width + spacing (≈{fmt(min(pitches) / 0.8, 2)}× the finger width), independent of W; pitch/W = {fmt(min(p / g for p, g in zip(pitches, gaps)), 2)}–{fmt(max(p / g for p, g in zip(pitches, gaps)), 2)}."),
        ("`prepattern.tmd_overlap_spacings` (preset): device reaches this × spacing onto each electrode", f"{fmt(tmd)} → {fmt((tmd or 0) * sp)} µm",
         f"not in the GDS (needs the flake outline); the straight part of each finger is {fmt(min(straight), 1)}–{fmt(max(straight), 1)} µm",
         "Measure on the traced flake; the straight finger length bounds how far the flake may reach before the fingers fan out."),
        ("Mak electrode length = overlap + 2·gate margin (Configure)", f"g = max(shift {fmt(sh)}, {fmt(gm)} × spacing) → {fmt((tmd or 0) * sp + 2 * max(sh or 0, (gm or 0) * sp))} µm",
         f"straight finger section {fmt(min(straight), 1)}–{fmt(max(straight), 1)} µm, whole finger {fmt(min(a.columns[c]['centreline_length_um'] for a in trio for c in ('top', 'bottom')), 1)}–{fmt(max(a.columns[c]['centreline_length_um'] for a in trio for c in ('top', 'bottom')), 1)} µm",
         "The lab's fingers are much longer and fan out to meet the EBL electrodes."),
        ("`cg.reach_spacings` (preset): CG inner / outer reach and side extension = this × spacing", f"{fmt(cgr)} → {fmt((cgr or 0) * sp)} µm each",
         f"CG starts flush with the tip (inner reach 0), side extension 0 (same 0.8 µm width as the finger), length {fmt(min(a.contact_gate.get('bottom_overlap_length_um') or 99 for a in trio), 1)}–{fmt(max(a.contact_gate.get('top_overlap_length_um') or 0 for a in trio), 1)} µm",
         "Different shape: a comb of finger-width teeth, not a band."),
        ("`ebl.contact_overlap_um`", f"{fmt(co)} ({cos})", f"layer-7 electrode overlap on a finger {fmt(min(ovl), 1)}–{fmt(max(ovl), 1)} µm (median per stack)",
         "Different meaning (1D-contact overlap in UStack); the lab's metal-on-metal overlap is long."),
    ]
    out = ["| UStack value | UStack now (status) | Measured in the lab's drawings | Comment |", "|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _mismatches(by: dict, presets: dict) -> str:
    s1, s3, s4 = by.get("Stack 1"), by.get("Stack 3"), by.get("Stack 4")
    trio = [a for a in (s1, s3, s4) if a]
    nf = [a.columns["top"]["n"] for a in trio]
    cs = [a.columns[c] for a in trio for c in ("top", "bottom")]
    st = [c["straight_length_um"] for c in cs]
    fp = [c["far_end_pitch_um"] for c in cs]
    ex = [c["extent_along_axis_um"] for c in cs]
    ang = max(abs(x) for c in cs for x in (c["outer_angle_deg_range"] or [0]))
    return "\n".join([
        f"1. **Terminal count.** `hallbar.HallBar` has n probe pairs (preset {pv(presets, 'hall_bar.n_probe_pairs')[0]}) + S + D, one finger "
        f"per terminal. The lab draws {min(nf)}–{max(nf)} identical fingers per column ({2 * min(nf)}–{2 * max(nf)} terminals), one "
        "electrode each, and leaves the fingers the flake does not reach floating.",
        "2. **No source/drain legs.** UStack's bent S/D (legs of width `leg_w`, flush with the channel ends, running "
        "`arm_l` to ±y) has no counterpart: every finger is the same, so S and D are chosen among the fingers at "
        "measurement time. The two layer-7 electrodes that touch neither 5 nor 6 end 27–50 µm away along the gap axis, "
        "on the back-gate strip, i.e. they are BG contacts (one at each end), not S/D.",
        "3. **No channel polygon.** In UStack the pattern is a Hall-bar body (W × L) with arms; here the \"channel\" is "
        "the gap between the tip lines (W = 3–5 µm) and L is the finger span (18–25 µm). Probe pitch is width + spacing "
        "(2.2–2.3 µm), not 2 × W.",
        f"4. **Finger shape.** UStack arms are straight rectangles of `arm_l`. The lab's fingers are straight for "
        f"{fmt(min(st), 1)}–{fmt(max(st), 1)} µm from the tip, then bend (one or two bends, up to ≈±{fmt(ang, 0)}°), mostly widen "
        f"to ≈1.0–1.2 µm and fan out to a {fmt(min(fp), 1)}–{fmt(max(fp), 1)} µm pitch, ending {fmt(min(ex), 0)}–{fmt(max(ex), 0)} µm "
        "from the tip (along the finger axis); the EBL electrodes land on the fanned part, beyond the CG.",
        "5. **Contact gate.** UStack's contact gate is a band region with inner / outer reach and side extension "
        "(`cg.reach_spacings`). The lab's CG is a comb: one tooth per *contacted* finger, exactly the finger width, from "
        "the tip outward 5–11 µm, joined by a bus bar behind the teeth; one CG per column (Stacks 1, 4) or one for both "
        "columns crossing the gap beside the array (Stack 3).",
        "6. **Back gate.** UStack's Mak preset puts the BG in the gap (BG = channel + 2g, gap W = bottom hBN + 2·shift). "
        "The overlays show the dark BG strip running through the gap far beyond the array, contacted at both ends by "
        "layer-7 bars; UStack's BG geometry has no such contacts drawn by the pattern step.",
        "7. **Two EBL steps per device.** Layer 5 (fingers) is written before stacking (pre-pattern), layer 6 (CG) and "
        "layer 7 (electrodes, BG contacts) after; Stack 4 shows the state after the CG design but before layer 7. "
        "UStack's step list (pre-pattern contacts, contact gates, …) matches this order, but its Hall-bar geometry does not "
        "generate these finger / comb shapes.",
    ])


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", type=Path, default=SESSION)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "wse2_design_report.md")
    ap.add_argument("--images", type=Path, default=ROOT / "docs" / "training")
    ap.add_argument("--json", type=Path, default=None, help="also write the full measurements as JSON")
    args = ap.parse_args(argv)
    if not args.session.is_file():
        print(f"session not found: {args.session}", file=sys.stderr)
        return 1
    views = ld.load_session(args.session)
    roles = ld.WSE2_BATCH1_LAYERS
    analyses, overlays = [], []
    for v in views:
        if not v.gds or not v.gds.is_file():
            print(f"skip view {v.index}: layout {v.name} not found")
            continue
        a = ld.analyse_design(v.gds, roles, name=v.stem)
        analyses.append(a)
        for img in v.images:
            if not img.exists or img.magnification != "100x" or not any(img.file.startswith(d) for d in OVERLAY_DIRS):
                continue
            tag = "AFMclean" if img.file.startswith(OVERLAY_DIRS[0]) else "TS"
            o = render_overlay(v, img, a, roles, args.images / f"{v.stem.replace(' ', '')}_{tag}_100x_overlay.jpg")
            if o:
                o["stack"] = v.stem
                overlays.append(o)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(views, analyses, overlays, args.out)
    if args.json:
        args.json.write_text(json.dumps([a.to_json() for a in analyses], indent=1), encoding="utf-8")
    print(f"wrote {args.out} and {len(overlays)} overlay(s) in {args.images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
