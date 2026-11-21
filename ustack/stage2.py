"""Stage 2: the EBL pattern for one lithography step on a fabricated device.

Inputs are the aligned images (image → chip µm transform), the traced flake outlines and defects
in chip µm, the measured thicknesses / layer counts, the bond-pad GDS (markers and the leads to
route to) and the purposes of this step. :func:`design_step` places the Hall bar (keeping every
specified dimension; only its position and angle are chosen) and draws each purpose;
:func:`check_step` re-checks the result with the measured values; :func:`export_step` writes the
GDS with KLayout and an updated specification next to it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import klayout.db as kdb

from . import geometry as geo
from .alignment import Similarity
from .config import LabConfig, MissingItem, MissingValues
from .gds_io import write_gds
from .hallbar import HallBar, Terminal, escape_point, lead_path
from .model import PATTERN_FIELDS, DeviceSpec, Q, now_iso
from .validation import Issue, Report, is_insulator, thickness_nm, validate_stage1

PURPOSES: dict[str, str] = {
    "board_etch": "Board etch",
    "hallbar_etch": "Graphene Hall-bar etch",
    "contacts": "Contacts",
    "top_gate": "Gate (metal top gate)",
    "prepattern_contacts": "Pre-pattern contacts",
    "contact_gates": "Contact gates",
    "split_gates": "Split gates",
}

#: Which purposes make sense for a contact scheme (the UI offers these).
PURPOSES_BY_SCHEME = {
    "edge_1d": ["board_etch", "top_gate", "hallbar_etch", "contacts"],
    "prepatterned_bottom": ["prepattern_contacts", "contact_gates", "split_gates", "contacts"],
    "top_2d": ["board_etch", "top_gate", "hallbar_etch", "contacts"],
}

DEFECT_KINDS = ("bubble", "wrinkle", "step", "residue", "crack", "other")


@dataclass
class StepInputs:
    purposes: list[str]
    outlines: dict[str, list[geo.Poly]] = field(default_factory=dict)   # layer uid → chip-µm polygons
    defects: list[dict] = field(default_factory=list)                   # {"kind", "polygon"}
    measured: dict[str, dict] = field(default_factory=dict)             # uid → {"thickness_nm", "layer_count"}
    images: dict[str, str] = field(default_factory=dict)                # "5x", "50x", "100x", "100x_lut"
    transforms: dict[str, dict] = field(default_factory=dict)           # image key → Similarity json
    bondpad_gds: str | None = None
    marker_layer: tuple[int, int] | None = None
    lead_layer: tuple[int, int] | None = None
    site: tuple[float, float] | None = None
    placement: dict | None = None                                       # {"center", "angle_deg", "source"}
    etch_windows: list[geo.Poly] = field(default_factory=list)
    marker_polys: list[geo.Poly] = field(default_factory=list)
    lead_tips: list[tuple[float, float]] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {k: copy.deepcopy(getattr(self, k)) for k in ("purposes", "outlines", "defects", "measured", "images", "transforms",
                                                          "bondpad_gds", "marker_layer", "lead_layer", "site", "placement",
                                                          "etch_windows", "lead_tips")}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "StepInputs":
        obj = cls(list(d.get("purposes", [])))
        for k in ("outlines", "defects", "measured", "images", "transforms", "bondpad_gds", "site", "placement", "etch_windows", "lead_tips"):
            if k in d:
                setattr(obj, k, copy.deepcopy(d[k]))
        for k in ("marker_layer", "lead_layer"):
            if d.get(k):
                setattr(obj, k, tuple(d[k]))
        return obj


@dataclass
class Design:
    hall_bar: HallBar | None
    shapes: dict[str, list[geo.Poly]] = field(default_factory=dict)
    routes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    placement_score: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- measured spec
def measured_spec(spec: DeviceSpec, inputs: StepInputs) -> DeviceSpec:
    """The spec with the measured thicknesses / layer counts and traced outlines applied."""
    eff = spec.copy()
    for uid, m in inputs.measured.items():
        layer = eff.layer(uid)
        if layer is None:
            continue
        for key in ("thickness_nm", "layer_count"):
            if m.get(key) is not None:
                getattr(layer, key).set(float(m[key]), "measured")
    for uid, polys in inputs.outlines.items():
        layer = eff.layer(uid)
        if layer is not None and polys:
            layer.outline = [list(p) for p in max(polys, key=geo.poly_area)]
    return eff


def _require_pattern(spec: DeviceSpec, *keys: str) -> list[float]:
    missing, out = [], []
    for k in keys:
        q = spec.pattern_params.get(k)
        if q is None or q.value is None:
            label, unit = PATTERN_FIELDS[k]
            missing.append(MissingItem(f"spec:pattern.{k}", label, "µm" if unit == "um" else unit))
        else:
            out.append(q.value)
    if missing:
        raise MissingValues(missing)
    return out


def _outline_region(inputs: StepInputs, uid: str | None) -> kdb.Region:
    if uid is None:
        return kdb.Region()
    return geo.regions(inputs.outlines.get(uid, []))


def _uids(spec: DeviceSpec, *functions: str, material: str | None = None, role: str | None = None) -> list[str]:
    return [l.uid for l in spec.layers if (not functions or l.function in functions)
            and (material is None or l.material == material) and (role is None or l.role == role)]


def device_region(spec: DeviceSpec, inputs: StepInputs) -> kdb.Region:
    """Where every device layer is present (the twisted region of a tear-and-stack bilayer)."""
    reg = None
    for uid in _uids(spec, "device_layer"):
        r = _outline_region(inputs, uid)
        if r.is_empty():
            continue
        reg = r if reg is None else (reg & r)
    return reg if reg is not None else kdb.Region()


def defect_region(inputs: StepInputs, clearance: float) -> kdb.Region:
    reg = geo.regions(d["polygon"] for d in inputs.defects if len(d.get("polygon", [])) >= 3)
    return geo.sized(reg, clearance) if clearance else reg


def _graphite_gates(spec: DeviceSpec, inputs: StepInputs) -> kdb.Region:
    return geo.regions(p for uid in _uids(spec, "bottom_gate", "top_gate", material="graphite") for p in inputs.outlines.get(uid, []))


def _gate_region(spec: DeviceSpec, inputs: StepInputs, function: str) -> kdb.Region:
    """Traced outlines of every layer with ``function`` (graphite or a pre-patterned metal gate)."""
    return geo.regions(p for uid in _uids(spec, function) for p in inputs.outlines.get(uid, []))


def _range(value: Any) -> tuple[float, float] | None:
    """A rule value that may be a [lo, hi] list or a single number, as (lo, hi)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value if v is not None]
        return (min(nums), max(nums)) if nums else None
    return float(value), float(value)


#: Metal purposes (written in some step; must never touch each other across steps).
METAL_PURPOSES = ("contacts", "top_gate", "prepattern_contacts", "contact_gates", "split_gates")


# ---------------------------------------------------------------------- placement
def place_hall_bar(hb: HallBar, spec: DeviceSpec, inputs: StepInputs, config: LabConfig, steps_deg: float = 3.0) -> dict:
    """Choose the Hall bar's centre and angle. Hard constraints: the body (channel + arms) lies in
    the usable region (device layer, encapsulating hBN, minus defects grown by the clearance);
    for 1D contacts every terminal end is ≥ the 'beyond BG' distance outside the graphite BG.
    Preference: channel over the BG (and TG), then the largest margin to the usable edge."""
    if inputs.placement and inputs.placement.get("source") in ("user", "previous step"):
        hb.center = tuple(inputs.placement["center"])
        hb.angle_deg = float(inputs.placement["angle_deg"])
        return {"source": inputs.placement["source"]}
    edge = spec.contact_scheme in ("edge_1d", "top_2d")
    dev = device_region(spec, inputs)
    stage_a_tmd = dev.is_empty() and spec.contact_scheme == "prepatterned_bottom"
    if stage_a_tmd:
        # pre-pattern before the device layer exists: fingers on the bottom hBN (Xu) or on the bare chip (Mak)
        bbn = _outline_region(inputs, next(iter(_uids(spec, role="bottom_hbn")), None))
        if spec.family == "xu_tmd" and not bbn.is_empty():
            usable = bbn
        else:
            centre = inputs.site or (0.0, 0.0)
            hb.center, hb.angle_deg = tuple(centre), 0.0
            return {"source": "site centre", "note": "No flakes traced: the Hall bar is placed at the device site's centre; drag it if needed."}
    else:
        if dev.is_empty():
            raise MissingValues([MissingItem("trace:device_layer", "Trace the device layer outline on the 100x image first", kind="action")])
        usable = dev
        if edge:
            for role in ("bottom_hbn", "tg_hbn"):
                r = _outline_region(inputs, next(iter(_uids(spec, role=role)), None))
                if not r.is_empty():
                    usable = usable & r
    clearance = 0.0
    if inputs.defects:
        clearance = config.require("ebl.defect_clearance_um")[0]
    usable = usable - defect_region(inputs, clearance)
    if usable.is_empty():
        raise ValueError("No usable area left after removing the defects")
    bg = _gate_region(spec, inputs, "bottom_gate")
    tg = _gate_region(spec, inputs, "top_gate")
    beyond = overlap = None
    if edge and not bg.is_empty() and config.applicable("ebl.device_etch_beyond_bg_um", spec.family, {"graphene"}):
        beyond, overlap = config.require("ebl.device_etch_beyond_bg_um", "ebl.contact_overlap_um")
    levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    eroded = {d: (usable.sized(-geo.to_dbu(d)) if d else usable) for d in levels}
    bg_grown = geo.sized(bg, beyond) if beyond else None
    x0, y0, x1, y1 = geo.bbox(usable)
    best = None

    def evaluate(cx: float, cy: float, ang: float):
        hb.center, hb.angle_deg = (cx, cy), ang
        body = hb.body_region()
        if not (body - eroded[0.0]).is_empty():
            return None
        if bg_grown is not None:
            for t in hb.terminals():
                if not (geo.region(hb.terminal_end_box(t, overlap)) & bg_grown).is_empty():
                    return None
        margin = 0.0
        for d in levels[1:]:
            if (body - eroded[d]).is_empty():
                margin = d
            else:
                break
        span = geo.region(hb.probe_span())
        a = geo.area(span) or 1.0
        gated = geo.area(span & bg) / a if not bg.is_empty() else 1.0
        if not tg.is_empty():
            gated = min(gated, geo.area(span & tg) / a)
        return (round(gated, 2), margin)

    def search(cxs, cys, angles):
        nonlocal best
        for ang in angles:
            for cx in cxs:
                for cy in cys:
                    if not _inside(eroded[0.0], cx, cy):
                        continue
                    score = evaluate(cx, cy, ang)
                    if score is not None and (best is None or score > best[0]):
                        best = (score, (cx, cy), ang)

    n = 15
    xs = [x0 + (i + 0.5) * (x1 - x0) / n for i in range(n)]
    ys = [y0 + (i + 0.5) * (y1 - y0) / n for i in range(n)]
    angles = [i * steps_deg for i in range(int(round(180 / steps_deg)))]
    ux, uy = geo.centroid(max(geo.to_polys(usable), key=geo.poly_area))
    search([ux], [uy], angles)
    search(xs, ys, angles)
    if best is None:
        fl, fw = hb.footprint()
        raise ValueError(f"The Hall bar ({fl:.2f} × {fw:.2f} µm with the given parameters) does not fit in the usable area of the "
                         "device in any position or angle (device layer ∩ hBN, minus defects"
                         + (f", arms ≥ {beyond:g} µm beyond the BG" if beyond else "") + "). UStack does not shrink specified dimensions.")
    (score, (bx, by), ba) = best
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    fx = [bx + (i - 3) * dx / 3 for i in range(7)]
    fy = [by + (i - 3) * dy / 3 for i in range(7)]
    search(fx, fy, [ba + k for k in (-2, -1, 0, 1, 2)])
    (score, (bx, by), ba) = best
    hb.center, hb.angle_deg = (bx, by), ba % 180.0
    return {"source": "auto", "gated_fraction": score[0], "margin_um": score[1]}


def _inside(reg: kdb.Region, x: float, y: float) -> bool:
    p = kdb.Point(geo.to_dbu(x), geo.to_dbu(y))
    return any(poly.inside(p) for poly in reg.each_merged())


# ---------------------------------------------------------------------- leads
def _escape(t: Terminal, keep_out: kdb.Region, start_offset: float, min_straight: float = 0.0, step: float = 0.25,
            limit: float = 400.0) -> tuple[float, float]:
    """Where a lead leaving terminal ``t`` straight out first clears ``keep_out`` (at least
    ``min_straight`` from its start, so it never turns right at the electrode it leaves)."""
    d = start_offset + min_straight
    while d < limit:
        p = escape_point(t, d)
        if not _inside(keep_out, *p):
            return p
        d += step
    return escape_point(t, limit)


def assign_tips(starts: list[tuple[float, float]], tips: list[tuple[float, float]], center: tuple[float, float],
                tip_cost: list[float] | None = None) -> list[int | None]:
    """Order-preserving assignment of lead starts to pad tips around the device (no crossings in a
    radial fan-out), minimising the total length (+ ``tip_cost`` per tip used, e.g. a pad under a stray flake).
    Returns the tip index per start (None if fewer tips than starts)."""
    n, m = len(starts), len(tips)
    if n == 0 or m < n:
        return [None] * n
    ang = lambda p: math.atan2(p[1] - center[1], p[0] - center[0])
    s_order = sorted(range(n), key=lambda i: ang(starts[i]))
    t_order = sorted(range(m), key=lambda j: ang(tips[j]))
    best_cost, best_map = math.inf, None
    for shift in range(m):
        tt = t_order[shift:] + t_order[:shift]
        # DP: assign s_order[i] to tt[j] with j increasing
        INF = math.inf
        cost = [[INF] * (m + 1) for _ in range(n + 1)]
        choice = [[None] * (m + 1) for _ in range(n + 1)]
        for j in range(m + 1):
            cost[0][j] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                skip = cost[i][j - 1]
                take = cost[i - 1][j - 1] + math.dist(starts[s_order[i - 1]], tips[tt[j - 1]]) + (tip_cost[tt[j - 1]] if tip_cost else 0.0)
                if take <= skip:
                    cost[i][j], choice[i][j] = take, "take"
                else:
                    cost[i][j], choice[i][j] = skip, "skip"
        if cost[n][m] < best_cost:
            best_cost = cost[n][m]
            mapping = [None] * n
            i, j = n, m
            while i > 0 and j > 0:
                if choice[i][j] == "take":
                    mapping[s_order[i - 1]] = tt[j - 1]
                    i, j = i - 1, j - 1
                else:
                    j -= 1
            best_map = mapping
    return best_map or [None] * n


def route(terminals: list[tuple[str, Terminal, float]], keep_out: kdb.Region, tips: list[tuple[float, float]],
          center: tuple[float, float], width: float) -> tuple[list[geo.Poly], list[dict]]:
    """Leads from each terminal (name, terminal, start offset along its direction) out of
    ``keep_out`` and on to its pad tip: start → escape point → tip."""
    escapes = [(_escape(t, keep_out, off, 2 * width)) for _, t, off in terminals]
    mapping = assign_tips(escapes, tips, center)
    polys, routes = [], []
    for (name, t, off), esc, k in zip(terminals, escapes, mapping):
        start = escape_point(t, off)
        pts = [start, esc] if math.dist(start, esc) > 1e-6 else [start]
        if k is not None:
            pts.append(tips[k])
        polys.append(lead_path(pts, width) if len(pts) >= 2 else None)
        routes.append({"terminal": name, "points_um": [[round(x, 4), round(y, 4)] for x, y in pts], "routed_to_pad": k is not None,
                       "pad_tip_um": list(tips[k]) if k is not None else None})
    return polys, routes


# ---------------------------------------------------------------------- what a step needs
def required_values(spec: DeviceSpec, inputs: StepInputs, config: LabConfig) -> list[MissingItem]:
    """Every value the step's purposes need that nobody has supplied, in one list (so the user is
    asked once, not one value per attempt)."""
    eff = measured_spec(spec, inputs)
    items: list[MissingItem] = []
    try:
        HallBar.from_spec(eff)
    except MissingValues as exc:
        items += exc.items
    graphene = any(l.material == "graphene" for l in eff.device_layers())
    tags = {"graphene"} if graphene else {"tmd"}
    rules: list[str] = []
    pattern: list[str] = []
    edge = eff.contact_scheme in ("edge_1d", "top_2d")
    bg_traced = any(inputs.outlines.get(uid) for uid in _uids(eff, "bottom_gate"))
    if inputs.defects:
        rules.append("ebl.defect_clearance_um")
    if edge and bg_traced and config.applicable("ebl.device_etch_beyond_bg_um", eff.family, tags):
        rules += ["ebl.device_etch_beyond_bg_um", "ebl.contact_overlap_um"]
    p = set(inputs.purposes)
    if "hallbar_etch" in p:
        rules.append("ebl.etch_mask_overhang_um")
    if "board_etch" in p:
        rules += ["ebl.board_etch_keep_margin_um", "ebl.etch_mask_overhang_um"]
    if "contacts" in p and edge:
        rules.append("ebl.contact_overlap_um")
    if "top_gate" in p:
        pattern += ["tg_length_um", "tg_overhang_um"]
    if "contact_gates" in p:
        pattern += ["cg_inner_um", "cg_outer_um", "cg_side_um"]
    if "split_gates" in p:
        pattern.append("sg_gap_um")
    seen = set()
    for rid in rules:
        if rid not in seen and config.rules[rid].is_missing:
            items.append(config.missing_item(rid))
        seen.add(rid)
    for key in dict.fromkeys(pattern):
        q = eff.pattern_params.get(key)
        if q is None or q.value is None:
            label, unit = PATTERN_FIELDS[key]
            items.append(MissingItem(f"spec:pattern.{key}", label, "µm" if unit == "um" else unit))
    if p & {"hallbar_etch", "board_etch"} and graphene and not inputs.outlines.get(next(iter(_uids(eff, "device_layer", material="graphene")), "")):
        items.append(MissingItem("trace:graphene", "Trace the graphene outline on the 100x image", kind="action"))
    if "split_gates" in p and not all(inputs.outlines.get(next(iter(_uids(eff, f)), "")) for f in ("bottom_gate", "top_gate")):
        items.append(MissingItem("trace:gates", "Trace the bottom-gate and top-gate outlines on the 100x image", kind="action"))
    return items


# ---------------------------------------------------------------------- purposes
def design_step(spec: DeviceSpec, inputs: StepInputs, config: LabConfig) -> Design:
    """Draw every purpose of the step. Raises MissingValues listing everything still needed."""
    needed = required_values(spec, inputs, config)
    if needed:
        raise MissingValues(needed)
    eff = measured_spec(spec, inputs)
    hb = HallBar.from_spec(eff)
    family_tags = {"graphene"} if any(l.material == "graphene" for l in eff.device_layers()) else {"tmd"}
    if "hallbar_etch" in inputs.purposes and config.applicable("ebl.corner_radius_min_um", eff.family, family_tags):
        hb.corner_radius = float(config.value("ebl.corner_radius_min_um") or 0.0)
    style = eff.hall_bar.sd_style
    prev = inputs.placement or {}
    if prev.get("source") in ("user", "previous step") and prev.get("sd_style") in ("straight", "bent"):
        hb.sd_style = prev["sd_style"]          # the as-built shape of the earlier step
        style = prev["sd_style"]
    try:
        score, err = place_hall_bar(hb, eff, inputs, config), None
    except ValueError as exc:
        score, err = None, exc
    bg_reg = _gate_region(eff, inputs, "bottom_gate")
    if style == "auto" and not bg_reg.is_empty() and (score is None or score.get("gated_fraction", 1.0) < 0.99):
        # a back gate under the source/drain ends (e.g. a strip along the whole device): turn the S/D to the sides
        bent = dataclasses.replace(hb, sd_style="bent")
        try:
            s2 = place_hall_bar(bent, eff, inputs, config)
        except ValueError:
            s2 = None
        if s2 is not None and (score is None or s2.get("gated_fraction", 0.0) > score.get("gated_fraction", 0.0)):
            hb, score = bent, s2
            score["sd_style"] = "bent (automatic)"
    if score is None:
        raise err
    design = Design(hb)
    design.placement_score = score
    if hb.bent and score.get("sd_style"):
        design.notes.append("Source and drain bent to the sides (S to +y, D to −y): the back gate runs under the ends of the "
                            "channel, so straight S/D contacts would land on it.")
    if not bg_reg.is_empty() and design.placement_score.get("gated_fraction", 1.0) < 0.99:
        design.notes.append(f"Only {design.placement_score['gated_fraction']:.0%} of the probe span could be placed over the back gate "
                            "while every contacted end stays clear of it (e.g. a gate strip that runs under the source/drain ends). "
                            "Place the Hall bar by hand, or the device needs a different pattern.")
    lead_w = config.value("ebl.electrode_width_target_um")
    if lead_w is None:
        lead_w = config.require("ebl.contact_width_min_um")[0]
    design.notes.append(f"Lead width {lead_w:g} µm (rule ebl.electrode_width_target_um).")
    graphene = any(l.material == "graphene" for l in eff.device_layers())
    gr_uid = next(iter(_uids(eff, "device_layer", material="graphene")), None) if graphene else None
    gr_reg = _outline_region(inputs, gr_uid)
    dev = device_region(eff, inputs)
    graphite = _graphite_gates(eff, inputs)
    tips = [tuple(t) for t in inputs.lead_tips]
    centre = hb.center
    flakes_all = geo.regions(p for polys in inputs.outlines.values() for p in polys)
    missing: list[MissingItem] = []

    def need(*rids):
        try:
            return config.require(*rids)
        except MissingValues as exc:
            missing.extend(exc.items)
            return None

    def need_p(*keys):
        try:
            return _require_pattern(eff, *keys)
        except MissingValues as exc:
            missing.extend(exc.items)
            return None

    shapes: dict[str, list[geo.Poly]] = {}
    gate_keepout = geo.sized(graphite, config.value("ebl.contact_to_graphite_gate_min_um") or 0.0) if graphene else kdb.Region()

    # -- Hall-bar etch (edge contacts): positive = graphene grown by the overhang, minus the Hall bar
    if "hallbar_etch" in inputs.purposes:
        if gr_reg.is_empty():
            missing.append(MissingItem("trace:graphene", "Trace the graphene outline on the 100x image", kind="action"))
        vals = need("ebl.etch_mask_overhang_um")
        if vals and not gr_reg.is_empty():
            pos = geo.sized(gr_reg, vals[0]) - hb.body_region()
            shapes["hallbar_etch"] = geo.to_polys(geo.rounded(pos, hb.corner_radius) if hb.corner_radius else pos)
            shapes["ref_hallbar"] = geo.to_polys(hb.body_region())

    # -- Board etch: etch everything around except the kept region (graphene + margin, gates), with check windows
    if "board_etch" in inputs.purposes:
        vals = need("ebl.board_etch_keep_margin_um", "ebl.etch_mask_overhang_um")
        if gr_reg.is_empty() and graphene:
            missing.append(MissingItem("trace:graphene", "Trace the graphene outline on the 100x image", kind="action"))
        if vals and not flakes_all.is_empty():
            keep_margin, overhang = vals
            keep = geo.sized(gr_reg if graphene else dev, keep_margin) | graphite
            tgm = [p for uid in _uids(eff, "top_gate") for p in inputs.outlines.get(uid, [])]
            keep |= geo.regions(tgm)
            outer = geo.sized(flakes_all, overhang)
            etch = outer - keep
            windows = geo.regions(inputs.etch_windows)
            etch |= windows
            shapes["board_etch"] = geo.to_polys(etch)
            shapes["ref_board_keep"] = geo.to_polys(keep)
            if inputs.etch_windows:
                shapes["etch_windows"] = [list(w) for w in inputs.etch_windows]
            elif config.applicable("gr.board_etch_windows", eff.family, {"graphene"}):
                design.notes.append("No etch-check windows placed: add top-BN-only windows on the design canvas (Fab Manual: 'Protect two windows').")

    # Gates are drawn whenever their parameters are known (so later steps route around them and
    # every step assigns the same pad tips); they are written only in the step that has the purpose.
    def known(*keys):
        return all(eff.pattern_params.get(k) is not None and eff.pattern_params[k].value is not None for k in keys)

    gate_polys: dict[str, list[geo.Poly]] = {}
    tg_body_placed: geo.Poly | None = None
    terms: list[tuple[str, str, Terminal, float]] = []   # (layer key, name, terminal, start offset)

    # -- metal top gate: a body over the channel (probe span) and a lead that leaves through the widest gap
    #    between two probe arms and ends past the graphene edge. Only the channel is covered: graphene under the
    #    metal is not removed by the later Hall-bar etch, so a gate over the arms would join them, and a gate over
    #    the arm ends would sit where the 1D contacts are etched and written.
    if "top_gate" in inputs.purposes or known("tg_length_um", "tg_overhang_um"):
        vals = need_p("tg_length_um", "tg_overhang_um")
        if vals:
            tg_len, tg_over = vals
            spacing = config.value("ebl.electrode_spacing_target_um") or config.value("ebl.contact_spacing_min_um") or 0.0
            x_exit = _tg_exit(hb, tg_len, lead_w, spacing, config.value("ebl.contact_overlap_um") or 0.0)
            if x_exit is None:
                raise ValueError(f"The metal top gate ({tg_len:g} µm long) has no gap between the probe arms that is wide enough for "
                                 f"its {lead_w:g} µm lead plus {spacing:g} µm on each side. UStack does not change specified "
                                 "dimensions: lengthen the gate or space the probes further apart.")
            reach = _reach(hb, gr_reg if not gr_reg.is_empty() else dev, +1, lead_w / 2, x_exit)
            top = max(reach if reach is not None else hb.W / 2 + hb.arm_l, hb.W / 2 + hb.arm_l) + tg_over
            body = geo.rect(-tg_len / 2, -hb.W / 2, tg_len / 2, hb.W / 2)
            lead = lead_path([(x_exit, 0.0), (x_exit, top)], lead_w)
            gate_polys["top_gate"] = [geo.transform(p, hb.angle_deg, *hb.center) for p in (body, lead)]
            tg_body_placed = geo.transform(body, hb.angle_deg, *hb.center)
            (ex, ey), = geo.transform([(x_exit, top)], hb.angle_deg, *hb.center)
            (ux, uy), = geo.transform([(0.0, 1.0)], hb.angle_deg)
            terms.append(("top_gate", "TG", Terminal("TG", "gate", (ex, ey), (ux, uy), lead_w), 0.0))

    # -- pre-pattern contacts (metal fingers under the device layer)
    if "prepattern_contacts" in inputs.purposes:
        polys = [p for _, p in hb.arms()] + [p for _, p in hb.legs()]
        for t in hb.terminals():
            if t.kind in ("source", "drain") and not hb.bent:
                polys.append(lead_path([t.end, escape_point(t, hb.arm_l)], t.width))
        shapes["prepattern_contacts"] = geo.to_polys(geo.regions(polys))

    # -- Hall-bar terminals: edge-contact pads, or the outer ends of the pre-patterned fingers
    pads: list[geo.Poly] = []
    if spec.contact_scheme in ("edge_1d", "top_2d"):
        overlap = config.value("ebl.contact_overlap_um")
        if "contacts" in inputs.purposes:
            overlap = need("ebl.contact_overlap_um")
            overlap = overlap[0] if overlap else None
        for t in hb.terminals():
            if overlap is not None:
                pads.append(lead_path([escape_point(t, -overlap), escape_point(t, lead_w)], max(t.width, lead_w)))
            terms.append(("contacts", t.name, t, lead_w))
    else:
        lead_key = "prepattern_contacts" if ("prepattern_contacts" in inputs.purposes and "contacts" not in inputs.purposes) else "contacts"
        for t in hb.terminals():
            terms.append((lead_key, t.name, t, hb.arm_l if (t.kind in ("source", "drain") and not hb.bent) else 0.0))

    # -- contact gates: a band over the contact tips on each side and at each end
    if "contact_gates" in inputs.purposes or known("cg_inner_um", "cg_outer_um", "cg_side_um"):
        vals = need_p("cg_inner_um", "cg_outer_um", "cg_side_um")
        if vals:
            cin, cout, cside = vals
            xs = hb.probe_x()
            lt = hb.total_length
            if hb.bent:
                # the bent source / drain sit in the top / bottom electrode column: each band also covers its leg
                (xls, xld), half_leg = hb.leg_x(), hb.leg_width / 2
                xa_t, xb_t = min(xs + [xls - half_leg + hb.arm_w / 2]) - hb.arm_w / 2 - cside, max(xs) + hb.arm_w / 2 + cside
                xa_b, xb_b = min(xs) - hb.arm_w / 2 - cside, max(xs + [xld + half_leg - hb.arm_w / 2]) + hb.arm_w / 2 + cside
                bands = {
                    "CGT": (geo.rect(xa_t, hb.W / 2 - cin, xb_t, hb.W / 2 + cout), (xb_t, hb.W / 2 + cout), (1, 1)),
                    "CGB": (geo.rect(xa_b, -hb.W / 2 - cout, xb_b, -hb.W / 2 + cin), (xa_b, -hb.W / 2 - cout), (-1, -1)),
                }
            else:
                xa, xb = min(xs) - hb.arm_w / 2 - cside, max(xs) + hb.arm_w / 2 + cside
                bands = {
                    "CGT": (geo.rect(xa, hb.W / 2 - cin, xb, hb.W / 2 + cout), (xb, hb.W / 2 + cout), (1, 1)),
                    "CGB": (geo.rect(xa, -hb.W / 2 - cout, xb, -hb.W / 2 + cin), (xa, -hb.W / 2 - cout), (-1, -1)),
                    "CGS": (geo.rect(-lt / 2 - cout, -hb.W / 2 - cside, -lt / 2 + cin, hb.W / 2 + cside), (-lt / 2 - cout, hb.W / 2 + cside), (-1, 1)),
                    "CGD": (geo.rect(lt / 2 - cin, -hb.W / 2 - cside, lt / 2 + cout, hb.W / 2 + cside), (lt / 2 + cout, -hb.W / 2 - cside), (1, -1)),
                }
            polys = []
            for name, (local, corner, d) in bands.items():
                polys.append(geo.transform(local, hb.angle_deg, *hb.center))
                (ex, ey), = geo.transform([corner], hb.angle_deg, *hb.center)
                n = math.hypot(*d)
                (ux, uy), = geo.transform([(d[0] / n, d[1] / n)], hb.angle_deg)
                terms.append(("contact_gates", name, Terminal(name, "gate", (ex, ey), (ux, uy), lead_w), 0.0))
            gate_polys["contact_gates"] = polys

    # -- split gates: the BG-only regions of the device layer, clear of the TG and the contact gates
    if "split_gates" in inputs.purposes or known("sg_gap_um"):
        vals = need_p("sg_gap_um")
        bg = _outline_region(inputs, next(iter(_uids(eff, "bottom_gate")), None))
        tg = _outline_region(inputs, next(iter(_uids(eff, "top_gate")), None))
        if vals and not bg.is_empty() and not tg.is_empty():
            spacing = config.value("ebl.contact_spacing_min_um") or 0.0
            sg = (bg & (dev if not dev.is_empty() else bg)) - geo.sized(tg, vals[0])
            if "contact_gates" in gate_polys:
                sg -= geo.sized(geo.regions(gate_polys["contact_gates"]), spacing)
            polys = geo.to_polys(sg)
            gate_polys["split_gates"] = polys
            for i, p in enumerate(polys, 1):
                cx, cy = geo.centroid(p)
                d = math.hypot(cx - centre[0], cy - centre[1]) or 1.0
                terms.append(("split_gates", f"SG{i}", Terminal(f"SG{i}", "gate", (cx, cy), ((cx - centre[0]) / d, (cy - centre[1]) / d), lead_w), 0.0))

    # -- leads: all terminals routed together (one consistent tip assignment), each written with its layer
    keep_out = geo.sized(dev if not dev.is_empty() else hb.body_region(), lead_w) | gate_keepout
    leads, routes = route([(name, t, off) for _, name, t, off in terms], keep_out, tips, centre, lead_w)
    if not tips:
        design.notes.append("No pad tips from the bond-pad GDS: leads stop just outside the device. Choose the lead layer to route them.")
    by_layer: dict[str, list[geo.Poly]] = {}
    for (key, _, _, _), poly, r in zip(terms, leads, routes):
        r["layer"] = key
        if poly is not None:
            by_layer.setdefault(key, []).append(poly)
    design.routes = [r for r in routes if r["layer"] in inputs.purposes]
    if "contacts" in inputs.purposes:
        shapes["contacts"] = geo.to_polys(geo.regions(pads + by_layer.get("contacts", [])))
    elif pads:
        # where the contacts of a later step will land: other metal must keep clear of them
        shapes["ref_contacts"] = geo.to_polys(geo.regions(pads))
    for key in ("top_gate", "contact_gates", "split_gates"):
        if key in inputs.purposes and key in gate_polys:
            shapes[key] = geo.to_polys(geo.regions(gate_polys[key] + by_layer.get(key, [])))
    if "top_gate" in shapes:
        shapes["ref_tg_body"] = [tg_body_placed] if tg_body_placed else gate_polys["top_gate"]
    if "hallbar_etch" in shapes and "top_gate" in gate_polys:
        # graphene under the metal gate is protected anyway; the etch mask follows the gate (as in the lab's layouts)
        shapes["hallbar_etch"] = geo.to_polys(geo.regions(shapes["hallbar_etch"]) - geo.regions(gate_polys["top_gate"]))
    if "prepattern_contacts" in inputs.purposes and by_layer.get("prepattern_contacts"):
        shapes["prepattern_contacts"] = geo.to_polys(geo.regions(shapes["prepattern_contacts"]) | geo.regions(by_layer["prepattern_contacts"]))
    for key, polys in gate_polys.items():
        if key not in inputs.purposes:
            shapes[f"ref_{key}"] = polys
    if missing:
        uniq = {m.key: m for m in missing}
        raise MissingValues(list(uniq.values()))
    if inputs.marker_polys:
        shapes["align_marks"] = [list(p) for p in inputs.marker_polys]
    defect_polys = [d["polygon"] for d in inputs.defects if len(d.get("polygon", [])) >= 3]
    if defect_polys:
        shapes["ref_defects"] = defect_polys
    shapes["ref_flakes"] = [p for polys in inputs.outlines.values() for p in polys]
    design.shapes = shapes
    return design


def _tg_exit(hb: HallBar, tg_len: float, lead_w: float, spacing: float, overlap: float = 0.0) -> float | None:
    """Where (Hall-bar frame x) the top-gate lead leaves the channel on the +y side: the centre of
    the widest free stretch between probe arms, keeping ``spacing`` to every arm and to the
    source/drain contact, and attached to the gate body (|x| + lead_w/2 ≤ tg_len/2). None if no
    stretch is wide enough."""
    half = min(tg_len / 2, hb.total_length / 2 - overlap - spacing) - lead_w / 2
    if half < 0:
        return None
    blocked = [(x - hb.arm_w / 2 - spacing - lead_w / 2, x + hb.arm_w / 2 + spacing + lead_w / 2) for x in hb.probe_x()]
    if hb.bent:   # the source leg leaves on the +y side, like the gate lead
        xs = hb.leg_x()[0]
        blocked.append((xs - hb.leg_width / 2 - spacing - lead_w / 2, xs + hb.leg_width / 2 + spacing + lead_w / 2))
    blocked.sort()
    free, start = [], -half
    for a, b in blocked:
        if a > start:
            free.append((start, min(a, half)))
        start = max(start, b)
    if start < half:
        free.append((start, half))
    free = [(a, b) for a, b in free if b >= a]
    if not free:
        return None
    a, b = max(free, key=lambda ab: (round(ab[1] - ab[0], 6), -abs((ab[0] + ab[1]) / 2)))
    return (a + b) / 2


def _reach(hb: HallBar, reg: kdb.Region, side: int, half_width: float | None = None, x_center: float = 0.0) -> float | None:
    """How far ``reg`` extends from the channel axis toward +y (side = +1) or -y (side = -1),
    within ±``half_width`` of ``x_center`` along the channel (Hall bar frame, µm)."""
    if reg.is_empty():
        return None
    hw = half_width if half_width is not None else hb.L / 2 + hb.arm_w
    x0, y0, x1, y1 = geo.bbox(reg)
    big = 2 * (abs(x1 - x0) + abs(y1 - y0)) + 10
    strip = geo.rect(x_center - hw, 0, x_center + hw, big) if side > 0 else geo.rect(x_center - hw, -big, x_center + hw, 0)
    cut = reg & geo.region(geo.transform(strip, hb.angle_deg, *hb.center))
    if cut.is_empty():
        return None
    pts = [p for poly in geo.to_polys(cut) for p in poly]
    local = geo.transform(geo.transform(pts, 0, -hb.center[0], -hb.center[1]), -hb.angle_deg)
    return max(y * side for _, y in local)


# ---------------------------------------------------------------------- re-check
def check_step(spec: DeviceSpec, inputs: StepInputs, design: Design, config: LabConfig) -> Report:
    """Stage 2 DRC on the drawn pattern, using the measured values and traced outlines."""
    eff = measured_spec(spec, inputs)
    report = validate_stage1(eff, config)
    # keep only stage-1 findings that matter now (values and stack rules), re-tag them
    for i in report.issues:
        i.code = "stage1:" + i.code
    hb = design.hall_bar
    tags = {"graphene"} if any(l.material == "graphene" for l in eff.device_layers()) else {"tmd"}

    def add(sev, code, msg, rule=None, where=None):
        status = None
        if rule:
            r = config.rules.get(rule)
            status = r.status if r else None
            if r and r.status in ("pending", "literature"):
                msg += f" [rule {rule} is {r.status} approval]"
            if r and sev == "error" and r.severity in ("warning", "info"):
                sev = r.severity
        report.add(Issue(sev, code, msg, rule=rule, rule_status=status, where=where or []))

    def rule_on(rid):
        return config.applicable(rid, eff.family, tags) and config.value(rid) is not None

    shapes = {k: geo.regions(v) for k, v in design.shapes.items() if not k.startswith("ref_")}
    metal_keys = [k for k in ("contacts", "top_gate", "prepattern_contacts", "contact_gates", "split_gates") if k in shapes]
    # widths / spacings of every metal layer
    for key in metal_keys:
        reg = shapes[key]
        hard_w, tgt_w = config.value("ebl.contact_width_min_um"), config.value("ebl.electrode_width_target_um")
        hard_s, tgt_s = config.value("ebl.contact_spacing_min_um"), config.value("ebl.electrode_spacing_target_um")
        if hard_w:
            pts = geo.width_violations(reg, hard_w)
            if pts:
                add("error", "width_min", f"{PURPOSES.get(key, key)}: narrower than {hard_w:g} µm.", "ebl.contact_width_min_um", pts)
        if tgt_w:
            pts = geo.width_violations(reg, tgt_w)
            if pts:
                add("warning", "width_target", f"{PURPOSES.get(key, key)}: narrower than the {tgt_w:g} µm target.", "ebl.electrode_width_target_um", pts)
        if hard_s:
            pts = geo.space_violations(reg, hard_s)
            if pts:
                add("error", "space_min", f"{PURPOSES.get(key, key)}: gap below {hard_s:g} µm.", "ebl.contact_spacing_min_um", pts)
        if tgt_s:
            pts = geo.space_violations(reg, tgt_s)
            if pts:
                add("warning", "space_target", f"{PURPOSES.get(key, key)}: gap below the {tgt_s:g} µm target.", "ebl.electrode_spacing_target_um", pts)
    # metal layers written in different steps must not touch each other: this step's metal against the
    # metal of this step's other purposes, of earlier exported steps (as built) and of later steps (as planned)
    others: dict[str, kdb.Region] = {}
    for step in spec.stage2.get("steps", []):
        for key, polys in (step.get("patterns_um") or {}).items():
            if key in METAL_PURPOSES and key not in metal_keys and polys:
                others[key] = others.get(key, kdb.Region()) | geo.regions(polys)
    for key in METAL_PURPOSES:
        ref = design.shapes.get(f"ref_{key}")
        if ref and key not in metal_keys:
            others[key] = others.get(key, kdb.Region()) | geo.regions(ref)
    hard_s = config.value("ebl.contact_spacing_min_um")
    for i, a in enumerate(metal_keys):
        for b, reg_b, planned in [(k, shapes[k], False) for k in metal_keys[i + 1:]] + [(k, r, True) for k, r in others.items()]:
            if {a, b} == {"contacts", "prepattern_contacts"}:
                continue  # leads connect to the fingers on purpose
            ov = shapes[a] & reg_b
            when = " (other step)" if planned else ""
            if not ov.is_empty():
                add("error", "metal_overlap", f"{PURPOSES[a]} and {PURPOSES[b]}{when} overlap (short).", None,
                    [geo.centroid(p) for p in geo.to_polys(ov)][:10])
            elif hard_s:
                pts = geo.separation_violations(shapes[a], reg_b, hard_s)
                if pts:
                    add("error", "metal_gap", f"{PURPOSES[a]} and {PURPOSES[b]}{when} closer than {hard_s:g} µm.",
                        "ebl.contact_spacing_min_um", pts)
    # Hall bar etch
    if "hallbar_etch" in shapes and hb is not None:
        body = hb.body_region()
        if rule_on("ebl.arm_spacing_min_um"):
            pts = geo.space_violations(body, config.value("ebl.arm_spacing_min_um"))
            if pts:
                add("error", "arm_spacing", "Hall-bar arms closer than the minimum.", "ebl.arm_spacing_min_um", pts)
        tgt = config.value("ebl.electrode_spacing_target_um")
        if tgt:
            pts = geo.space_violations(body, tgt)
            if pts:
                add("warning", "arm_spacing_target", f"Hall-bar arms closer than the {tgt:g} µm target.", "ebl.electrode_spacing_target_um", pts)
    # graphene edge-contact rules
    gr_uid = next(iter(_uids(eff, "device_layer", material="graphene")), None)
    gr = _outline_region(inputs, gr_uid)
    graphite = _graphite_gates(eff, inputs)
    bg = _gate_region(eff, inputs, "bottom_gate")
    if "contacts" in shapes and eff.contact_scheme in ("edge_1d", "top_2d") and hb is not None:
        if rule_on("ebl.contact_to_graphite_gate_min_um") and not graphite.is_empty():
            d = config.value("ebl.contact_to_graphite_gate_min_um")
            pts = geo.separation_violations(shapes["contacts"], graphite, d)
            if pts:
                add("error", "contact_graphite", f"Contacts closer than {d:g} µm to a graphite gate.", "ebl.contact_to_graphite_gate_min_um", pts)
        if rule_on("ebl.device_etch_beyond_bg_um") and not bg.is_empty():
            d = config.value("ebl.device_etch_beyond_bg_um")
            ov = config.value("ebl.contact_overlap_um") or 0.0
            for t in hb.terminals():
                box = geo.region(hb.terminal_end_box(t, ov or 0.01))
                if not (box & geo.sized(bg, d)).is_empty():
                    add("error", "beyond_bg", f"Terminal {t.name}: the contacted end is less than {d:g} µm beyond the BG.", "ebl.device_etch_beyond_bg_um", [t.end])
        if not gr.is_empty():
            for t in hb.terminals():
                ov = config.value("ebl.contact_overlap_um") or 0.01
                box = geo.region(hb.terminal_end_box(t, ov))
                if (box & gr).is_empty() or (box & shapes["contacts"]).is_empty():
                    add("error", "contact_touch", f"Terminal {t.name}: the contact does not land on graphene.", None, [t.end])
    # the channel should be gated
    if hb is not None and not bg.is_empty():
        span = geo.region(hb.probe_span())
        frac = geo.area(span & bg) / (geo.area(span) or 1.0)
        if frac < 0.99:
            add("warning", "channel_not_gated", f"Only {frac:.0%} of the probe span lies over the back gate.", None, [hb.center])
    # defects
    if inputs.defects and hb is not None:
        clr = config.value("ebl.defect_clearance_um")
        if clr is None:
            report.add(Issue("missing", "rule_value", "Lab value needed: defect clearance", rule="ebl.defect_clearance_um",
                             ask=config.missing_item("ebl.defect_clearance_um")))
        else:
            dreg = defect_region(inputs, 0.0)
            pts = geo.separation_violations(hb.body_region(), dreg, clr)
            if pts:
                add("error", "defect", f"The Hall bar comes within {clr:g} µm of a marked defect.", "ebl.defect_clearance_um", pts)
    # board etch keeps the graphene
    if "board_etch" in shapes and not gr.is_empty() and rule_on("stack.board_etch_keep_contains_device"):
        cut = gr & shapes["board_etch"]
        if not cut.is_empty():
            add("error", "board_etch_cuts_device", "The board etch removes part of the graphene device layer.", "stack.board_etch_keep_contains_device",
                [geo.centroid(p) for p in geo.to_polys(cut)])
        keep = geo.regions(design.shapes.get("ref_board_keep", []))
        if not keep.is_empty() and geo.area(keep) <= geo.area(gr):
            add("error", "board_etch_area", "The area kept by the board etch is not larger than the graphene.", "stack.board_etch_keep_contains_device")
        if rule_on("gr.board_etch_windows") and not design.shapes.get("etch_windows"):
            add("warning", "etch_windows", "No top-BN-only etch-check windows in the board etch.", "gr.board_etch_windows")
        for w in design.shapes.get("etch_windows", []):
            wr = geo.region(w)
            conductors = gr | graphite
            if not (wr & conductors).is_empty():
                add("error", "etch_window_place", "An etch-check window lies on graphene or graphite (it must be top-hBN only).", "gr.board_etch_windows", [geo.centroid(w)])
    # top gate
    if "top_gate" in shapes:
        tg = shapes["top_gate"]
        if config.value("ebl.metal_tg_width_min_um"):
            pts = geo.width_violations(tg, config.value("ebl.metal_tg_width_min_um"))
            if pts:
                add("error", "tg_width", "Metal top gate narrower than the minimum.", "ebl.metal_tg_width_min_um", pts)
        if not gr.is_empty() and rule_on("gr.tg_extends_beyond_graphene"):
            # the gate must reach outside the graphene, so whatever connects to it lands off the graphene
            if (tg - gr).is_empty():
                add("warning", "tg_inside_graphene", "The metal top gate does not extend outside the graphene.", "gr.tg_extends_beyond_graphene")
            for r in design.routes:
                if r.get("layer") == "top_gate" and r["points_um"] and _inside(gr, *r["points_um"][0]):
                    add("warning", "tg_end_on_graphene", f"The {r['terminal']} lead starts on graphene: extend the gate past the graphene edge.",
                        "gr.tg_extends_beyond_graphene", [tuple(r["points_um"][0])])
    # Xu: contact gates on the middle hBN, middle hBN ≥ margin beyond the TMD
    if eff.family == "xu_tmd":
        mid = _outline_region(inputs, next(iter(_uids(eff, role="middle_hbn")), None))
        tmd = geo.regions(p for uid in _uids(eff, "device_layer") for p in inputs.outlines.get(uid, []))
        if not mid.is_empty() and not tmd.is_empty() and rule_on("xu.middle_hbn_edge_margin_um"):
            m = config.value("xu.middle_hbn_edge_margin_um")
            pts = geo.enclosure_violations(mid, tmd, m)
            if pts:
                add("error", "xu_margin", f"The middle hBN does not reach {m:g} µm beyond the TMD everywhere.", "xu.middle_hbn_edge_margin_um", pts)
        if "contact_gates" in shapes and not mid.is_empty() and not tmd.is_empty():
            # the contact gate may leave the middle hBN only where there is no TMD (its lead), never touching the TMD
            bare = (shapes["contact_gates"] & tmd) - mid
            if not bare.is_empty():
                add("error", "cg_on_tmd", "A contact gate touches the TMD outside the middle hBN.", "xu.middle_hbn_edge_margin_um",
                    [geo.centroid(p) for p in geo.to_polys(bare)])
    if "prepattern_contacts" in shapes and not bg.is_empty():
        bbn = _outline_region(inputs, next(iter(_uids(eff, role="bottom_hbn")), None))
        pre = next(iter(_uids(eff, role="prepattern_contacts")), None)
        bg_first = [u for u in _uids(eff, "bottom_gate") if pre is not None and eff.index(u) < eff.index(pre)]
        if bg_first:
            # the electrodes are written on the bottom hBN over the BG (Xu): only bare BG is a short
            short = (shapes["prepattern_contacts"] & bg) - bbn
            if not short.is_empty():
                add("error", "finger_bg", "Pre-patterned contacts touch the graphite BG where no bottom hBN covers it.", None,
                    [geo.centroid(p) for p in geo.to_polys(short)])
        else:
            # the BG lands beside the electrodes (Mak: in the gap between the two columns): never touching, with spacing
            ov = shapes["prepattern_contacts"] & bg
            if not ov.is_empty():
                add("error", "finger_bg", "Pre-patterned electrodes overlap the back gate (it lands in the gap between them): short.", None,
                    [geo.centroid(p) for p in geo.to_polys(ov)])
            elif hard_s:
                pts = geo.separation_violations(shapes["prepattern_contacts"], bg, hard_s)
                if pts:
                    add("error", "finger_bg_gap", f"Pre-patterned electrodes closer than {hard_s:g} µm to the back gate.",
                        "ebl.contact_spacing_min_um", pts)
    if eff.family == "mak_tmd":
        tg = _outline_region(inputs, next(iter(_uids(eff, "top_gate", material="graphite")), None))
        if not tg.is_empty() and not bg.is_empty() and rule_on("mak.tg_smaller_than_bg") and not (tg - bg).is_empty():
            add("warning", "mak_tg_inside_bg", "The top gate reaches outside the bottom gate.", "mak.tg_smaller_than_bg")
    # alignment marks are never written on
    marks = geo.regions(inputs.marker_polys)
    if not marks.is_empty():
        for key, reg in shapes.items():
            if key == "align_marks":
                continue
            ov = reg & marks
            if not ov.is_empty():
                add("error", "on_marker", f"{PURPOSES.get(key, key)} writes on an alignment marker.", None, [geo.centroid(p) for p in geo.to_polys(ov)])
    # contact metal vs the measured step (edge contacts)
    for r in design.routes:
        if not r["routed_to_pad"]:
            add("warning", "unrouted", f"Lead {r['terminal']} is not connected to a bond-pad lead.")
    order = {"error": 0, "missing": 1, "warning": 2, "info": 3}
    report.issues.sort(key=lambda i: order.get(i.severity, 9))
    return report


# ---------------------------------------------------------------------- etch plan (informational)
def etch_plan(spec: DeviceSpec, inputs: StepInputs, config: LabConfig) -> list[str]:
    """Etch times from the measured hBN thicknesses and the Fab Manual rates (pending rules)."""
    eff = measured_spec(spec, inputs)
    rate_bn = config.value("etch.bn_rate_nm_per_s")
    over = _range(config.value("etch.board_overetch_nm"))
    out = []
    tb, bb = eff.by_role("tg_hbn"), eff.by_role("bottom_hbn")
    t_top, t_bot = (thickness_nm(tb) if tb else None), (thickness_nm(bb) if bb else None)
    if "board_etch" in inputs.purposes and rate_bn and over and None not in (t_top, t_bot):
        lo, hi = t_top + t_bot + over[0], t_top + t_bot + over[1]
        out.append(f"Board etch (BN-Test ≈ {rate_bn:g} nm/s): {lo:.0f}–{hi:.0f} nm ≈ {lo / rate_bn:.0f}–{hi / rate_bn:.0f} s "
                   "(the rate rises with time; watch the check windows).")
    cb = _range(config.value("etch.contact_bottom_bn_nm"))
    if "contacts" in inputs.purposes and eff.contact_scheme == "edge_1d" and rate_bn and t_top is not None and cb:
        out.append(f"1D contacts: top hBN {t_top:g} nm ≈ {t_top / rate_bn:.0f} s BN-Test, graphene 7–10 s O2, then "
                   f"{cb[0]:g}–{cb[1]:g} nm of bottom hBN ≈ {cb[0] / rate_bn:.0f}–{cb[1] / rate_bn:.0f} s.")
    return out


# ---------------------------------------------------------------------- export
def export_step(spec: DeviceSpec, inputs: StepInputs, design: Design, report: Report, config: LabConfig, out_dir: Path,
                step_name: str) -> dict[str, Path]:
    """Write the GDS (chip frame) and the updated specification + report next to it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{step_name}"
    shapes = {k: v for k, v in design.shapes.items() if not k.startswith("ref_") or k in ("ref_flakes", "ref_defects")}
    texts = {}
    if design.hall_bar is not None and "contacts" in shapes:
        texts["contacts"] = [(t.name, *t.end) for t in design.hall_bar.terminals()]
    gds = write_gds(out_dir / f"{base}.gds", f"{spec.name[:24].replace(' ', '_')}_{step_name}"[:60], shapes, config, texts,
                    skip_reference=_reference_layers_missing(config))
    updated = spec.copy()
    eff = measured_spec(spec, inputs)
    for layer in updated.layers:
        src = eff.layer(layer.uid)
        if src is not None:
            layer.thickness_nm, layer.layer_count = src.thickness_nm, src.layer_count
    steps = updated.stage2.setdefault("steps", [])
    record = {
        "step": step_name, "time": now_iso(), "purposes": inputs.purposes,
        "inputs": inputs.to_json(),
        "hall_bar_as_built": design.hall_bar.to_json() if design.hall_bar else None,
        "placement": design.placement_score, "routes": design.routes, "notes": design.notes,
        "etch_plan": etch_plan(spec, inputs, config),
        "patterns_um": {k: v for k, v in design.shapes.items() if k not in ("ref_flakes", "align_marks")},
        "gds": gds.name, "gds_layers": {k: list(config.gds_layer(k)) for k in shapes if not config.layers.get(k) or not config.layers[k].is_missing},
        "drc": report.to_json(),
    }
    steps.append(record)
    if design.hall_bar is not None:
        updated.stage2["hall_bar_placement"] = {"center": list(design.hall_bar.center), "angle_deg": design.hall_bar.angle_deg,
                                                "source": "previous step", "step": step_name, "sd_style": design.hall_bar.sd_style}
    updated.history.append({"time": now_iso(), "note": f"Stage 2 step '{step_name}' exported ({', '.join(inputs.purposes)})"})
    spec_path = updated.save(out_dir / f"{base}_spec.json")
    md = out_dir / f"{base}_report.md"
    md.write_text(_report_md(updated, record, report), encoding="utf-8")
    return {"gds": gds, "spec": spec_path, "report": md, "updated_spec": updated}


def _reference_layers_missing(config: LabConfig) -> bool:
    return any(config.layers.get(k) is None or config.layers[k].is_missing for k in ("ref_flakes", "ref_defects"))


def _report_md(spec: DeviceSpec, record: dict, report: Report) -> str:
    lines = [f"# {spec.name} — EBL step '{record['step']}'", "", f"Generated by UStack on {record['time']}.", "",
             f"Purposes: {', '.join(PURPOSES.get(p, p) for p in record['purposes'])}", "", f"GDS: `{record['gds']}` "
             f"(layers: {', '.join(f'{k} {v[0]}/{v[1]}' for k, v in record['gds_layers'].items())})", ""]
    hb = record.get("hall_bar_as_built")
    if hb:
        lines += ["## Hall bar as built", "", f"- Centre ({hb['center_um'][0]:.3f}, {hb['center_um'][1]:.3f}) µm, angle {hb['angle_deg']:.2f}°",
                  f"- W {hb['W_um']:g} µm, L {hb['L_um']:g} µm, {hb['n_probe_pairs']} probe pairs, arms {hb['arm_width_um']:g} × {hb['arm_length_um']:g} µm",
                  f"- Placement: {record['placement']}", ""]
    if record.get("etch_plan"):
        lines += ["## Etch plan (from measured thicknesses; rates are pending rules)", ""] + [f"- {e}" for e in record["etch_plan"]] + [""]
    lines += ["## Design-rule check", "", f"**{report.summary()}**", ""]
    for i in report.issues:
        lines.append(f"- **{i.severity}** `{i.code}` {i.message}")
    if record.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in record["notes"]]
    return "\n".join(lines) + "\n"
