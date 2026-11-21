"""Learn the lab's own EBL design practice from its GDS files and KLayout sessions (Qt-free).

Two entry points:

* :func:`load_session` reads a KLayout session (``.lys``): its views, the layout each view shows
  and the images placed on it (KLayout's 3×3 image matrix, so pixels map to chip µm).
* :func:`analyse_design` measures a two-column pre-patterned electrode design (the Mak-style
  layout, AGENTS.md §2): fingers, contact gate, electrodes, landing area and markers near the
  device, all in µm, with KLayout regions (1 DBU = 1 nm).

Everything returned is a *measurement of the lab's drawing*, never an approved value: a caller
that wants to use a number as a default must have the lab confirm it first (AGENTS.md rule 1).

Image convention (KLayout ``img::Object``): an image of W×H pixels, pixel column ``col`` and row
``row`` (row 0 at the top) has image coordinates u = col − W/2, v = H/2 − row; the session matrix M
gives [x, y, w] = M·[u, v, 1] and chip µm = (x/w, y/w). Verified on the tWSe2-1 session by
overlaying the alignment markers of layer 3 on the 100x images.

Local frame used by :func:`analyse_design`: the chip frame rotated so the fingers run along +v
(the "top" column at larger v), origin at the device centre (middle of the column gap, middle of
the finger array). Distances "from the tip" are measured along the finger axis (v).
"""

from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import klayout.db as kdb
import numpy as np

DBU = 0.001  # µm per database unit
_BIG = 10**9  # DBU, "infinite" box half-size

#: Layer meanings of the lab's tWSe2 batch-1 designs (Stack 1/3/4.gds), as stated by the user in
#: the request of 2026-11-16: 1 = square area the device lands in, 3 = small square alignment
#: markers, 4 = bond pads and fan-out leads, 5 = pre-patterned "hall bar" fingers in two columns
#: with a long thin gap between them, 6 = contact gate on top of the fingers, 7 = electrodes that
#: connect fingers / contact gates / gates to the fan-out. A user statement about these files, not a
#: lab-wide layer convention (config/layers.json stays the authority for GDS export).
WSE2_BATCH1_LAYERS: dict[str, tuple[int, int]] = {
    "landing": (1, 0),
    "markers": (3, 0),
    "pads": (4, 0),
    "fingers": (5, 0),
    "contact_gate": (6, 0),
    "electrodes": (7, 0),
}

#: Analysis knobs (program choices for *measuring*, not design values).
NEAR_DEVICE_UM = 60.0      # electrodes / labels within this distance of the finger array count as "near"
SLICE_STEP_UM = 0.1        # sampling step along a finger
STRAIGHT_TOL_UM = 0.01      # a finger is "straight" while its centre and width stay within this
TIP_ZONE_UM = 1.0          # finger width "at the tip" = median width over the first TIP_ZONE_UM
ELECTRODE_END_BOX_UM = 10.0  # electrode width near the device: piece within ±this of its inner end


# ====================================================================== session (.lys)
@dataclass
class SessionImage:
    """One image placed in a KLayout view."""

    file: str                  # path as written in the session (relative to the .lys folder)
    path: Path                 # resolved path
    exists: bool
    matrix: list[list[float]]  # 3×3, [x, y, w] = M·[u, v, 1]
    visible: bool = True
    z_position: int = 0
    layer_binding: str = ""

    def _m(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    def px_to_chip(self, cols, rows, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel column/row (arrays or scalars) of a W×H image → chip µm (x, y)."""
        u = np.asarray(cols, dtype=float) - W / 2.0
        v = H / 2.0 - np.asarray(rows, dtype=float)
        m = self._m()
        x = m[0, 0] * u + m[0, 1] * v + m[0, 2]
        y = m[1, 0] * u + m[1, 1] * v + m[1, 2]
        w = m[2, 0] * u + m[2, 1] * v + m[2, 2]
        return x / w, y / w

    def chip_to_px(self, xs, ys, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
        """Chip µm → pixel column/row of a W×H image (inverse of :meth:`px_to_chip`)."""
        mi = np.linalg.inv(self._m())
        x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        u = mi[0, 0] * x + mi[0, 1] * y + mi[0, 2]
        v = mi[1, 0] * x + mi[1, 1] * y + mi[1, 2]
        w = mi[2, 0] * x + mi[2, 1] * y + mi[2, 2]
        return u / w + W / 2.0, H / 2.0 - v / w

    @property
    def um_per_px(self) -> float:
        """Scale at the image centre (√|det J| of the projective map at u = v = 0)."""
        a, b, c = self.matrix[0]
        d, e, f = self.matrix[1]
        g, h, i = self.matrix[2]
        j = np.array([[a * i - c * g, b * i - c * h], [d * i - f * g, e * i - f * h]]) / (i * i)
        return float(math.sqrt(abs(np.linalg.det(j))))

    @property
    def magnification(self) -> str:
        m = re.search(r"_(\d+x)_", Path(self.file).name)
        return m.group(1) if m else ""


@dataclass
class SessionView:
    """One KLayout view: the layout it shows and the images placed on it."""

    index: int
    name: str                  # layout name, e.g. "Stack 1.gds"
    gds: Path | None
    images: list[SessionImage] = field(default_factory=list)
    layer_names: dict[str, str] = field(default_factory=dict)  # "5/0" → "InnerContacts"

    @property
    def stem(self) -> str:
        return Path(self.name).stem


_MATRIX_RE = re.compile(r"matrix=\(([^)]*)\)\s*\(([^)]*)\)\s*\(([^)]*)\)")
_FILE_RE = re.compile(r"file='([^']*)'")


def _parse_image(value: str, folder: Path) -> SessionImage | None:
    mm, fm = _MATRIX_RE.search(value), _FILE_RE.search(value)
    if not mm or not fm:
        return None
    matrix = [[float(t) for t in group.split(",")] for group in mm.groups()]
    rel = fm.group(1)
    path = (folder / rel).resolve()
    vis = re.search(r"is_visible=(\w+)", value)
    z = re.search(r"z_position=(-?\d+)", value)
    lb = re.search(r"layer_binding=('[^']*' \([^)]*\)|[^;]*)", value)
    return SessionImage(file=rel, path=path, exists=path.is_file(), matrix=matrix,
                        visible=(vis.group(1) == "true") if vis else True,
                        z_position=int(z.group(1)) if z else 0,
                        layer_binding=lb.group(1) if lb else "")


def load_session(lys_path: str | Path) -> list[SessionView]:
    """Read a KLayout session: one :class:`SessionView` per ``<view>``, in session order. Image
    files that do not exist are kept with ``exists = False`` (callers skip them)."""
    lys = Path(lys_path)
    folder = lys.parent
    root = ET.parse(lys).getroot()
    layouts = {}
    for lay in root.findall("layout"):
        name = (lay.findtext("name") or "").strip()
        fp = (lay.findtext("file-path") or "").strip()
        layouts[name] = (folder / fp).resolve() if fp else None
    views = []
    for k, view in enumerate(root.findall("view")):
        ref = (view.findtext("cellviews/cellview/layout-ref") or "").strip()
        sv = SessionView(index=k, name=ref, gds=layouts.get(ref))
        for src in view.iter("source"):
            m = re.match(r"\s*'(\d+/\d+)\s*-\s*([^']*)'", src.text or "")
            if m:
                sv.layer_names[m.group(1)] = m.group(2).strip()
        for ann in view.iter("annotation"):
            if (ann.findtext("class") or "").startswith("img::"):
                img = _parse_image(ann.findtext("value") or "", folder)
                if img:
                    sv.images.append(img)
        views.append(sv)
    return views


# ====================================================================== region helpers
def _um(v: float) -> float:
    return round(v * DBU, 4)


def _dbu(v_um: float) -> int:
    return int(round(v_um / DBU))


def _read_layers(gds_path: Path, roles: dict[str, tuple[int, int]]) -> dict[str, kdb.Region]:
    layout = kdb.Layout()
    layout.read(str(gds_path))
    top = layout.top_cell()
    out = {}
    for role, (ln, dt) in roles.items():
        li = layout.find_layer(ln, dt)
        out[role] = _copy_region(top, li).merged()
    return out


def _strip(reg: kdb.Region) -> tuple[float, float]:
    """Equivalent strip (width, length) in µm of a region: area A and half-perimeter P/2 of a
    w × l strip satisfy A = w·l and P/2 = w + l."""
    a = reg.area() * DBU * DBU
    p2 = reg.perimeter() * DBU / 2.0
    disc = p2 * p2 - 4 * a
    if a <= 0:
        return 0.0, 0.0
    w = (p2 - math.sqrt(max(disc, 0.0))) / 2.0
    return w, (a / w if w else 0.0)


def _band(v0_um: float, v1_um: float) -> kdb.Region:
    lo, hi = sorted((v0_um, v1_um))
    return kdb.Region(kdb.Box(-_BIG, _dbu(lo), _BIG, _dbu(hi)))


def _box(cx: float, cy: float, hx: float, hy: float | None = None) -> kdb.Region:
    hy = hx if hy is None else hy
    return kdb.Region(kdb.Box(_dbu(cx - hx), _dbu(cy - hy), _dbu(cx + hx), _dbu(cy + hy)))


def _slice(reg: kdb.Region, v_um: float, h_um: float = 0.02) -> tuple[float, float] | None:
    """(width, centre u) of a region along the line v = v_um (a thin band of height h_um)."""
    piece = reg & _band(v_um - h_um / 2, v_um + h_um / 2)
    if piece.is_empty():
        return None
    b = piece.bbox()
    return piece.area() * DBU * DBU / h_um, (b.left + b.right) / 2 * DBU


def _centroid(reg: kdb.Region) -> tuple[float, float]:
    sx = sy = sa = 0.0
    for p in reg.each():
        a = p.area()
        c = p.bbox().center()
        sx, sy, sa = sx + c.x * a, sy + c.y * a, sa + a
    return (sx / sa * DBU, sy / sa * DBU) if sa else (0.0, 0.0)


def _min_distance(a: kdb.Region, b: kdb.Region, limit_um: float = 50.0) -> float | None:
    """Smallest distance (µm) between two regions, or None if farther than ``limit_um``."""
    if a.is_empty() or b.is_empty():
        return None
    if a.interacting(b).count():
        return 0.0
    pairs = a.separation_check(b, _dbu(limit_um), False, kdb.Metrics.Euclidian)
    ds = [_edge_distance(ep.first, ep.second) * DBU for ep in pairs.each()]
    return round(min(ds), 4) if ds else None


def _edge_distance(e1: kdb.Edge, e2: kdb.Edge) -> float:
    """Distance between two non-crossing segments (DBU); robust to zero-length edges, for which
    ``EdgePair.distance`` returns 0."""

    def pt_seg(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - ax - t * dx, py - ay - t * dy)

    a1, a2, b1, b2 = e1.p1, e1.p2, e2.p1, e2.p2
    return min(pt_seg(a1.x, a1.y, b1.x, b1.y, b2.x, b2.y), pt_seg(a2.x, a2.y, b1.x, b1.y, b2.x, b2.y),
               pt_seg(b1.x, b1.y, a1.x, a1.y, a2.x, a2.y), pt_seg(b2.x, b2.y, a1.x, a1.y, a2.x, a2.y))


def finger_axis_angle(fingers: kdb.Region) -> float:
    """Direction of the fingers (degrees in [0, 180)): the length-weighted mode of the edge
    directions (the long straight edges of the fingers dominate)."""
    edges = []
    hist = np.zeros(360)  # 0.5° bins
    for e in fingers.edges().each():
        dx, dy = e.dx(), e.dy()
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        edges.append((ang, math.hypot(dx, dy)))
        hist[int(ang * 2) % 360] += math.hypot(dx, dy)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    mode = int(np.argmax(smooth)) / 2.0 + 0.25
    # refine: length-weighted mean of the edges within ±0.75° of the mode (angles unwrapped)
    num = den = 0.0
    for ang, ln in edges:
        d = (ang - mode + 90.0) % 180.0 - 90.0
        if abs(d) <= 0.75:
            num, den = num + d * ln, den + ln
    return float(mode + (num / den if den else 0.0)) % 180.0


# ====================================================================== analysis results
@dataclass
class DesignAnalysis:
    name: str
    gds: str
    layer_roles: dict[str, list[int]]
    layer_roles_source: str
    frame: dict[str, Any]                 # angle, device centre (chip µm), local-frame convention
    fingers: list[dict] = field(default_factory=list)
    columns: dict[str, dict] = field(default_factory=dict)
    array: dict[str, Any] = field(default_factory=dict)
    contact_gate: dict[str, Any] = field(default_factory=dict)
    electrodes: list[dict] = field(default_factory=list)
    electrode_summary: dict[str, Any] = field(default_factory=dict)
    landing: dict[str, Any] = field(default_factory=dict)
    markers: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    status: str = "measured from the lab's GDS; not approved"

    def to_json(self) -> dict:
        return asdict(self)


# ====================================================================== fingers
def _finger_profile(f: kdb.Region, tip_v: float, sign: int, extent: float) -> dict:
    """Width and centre along a finger, from its tip (s = 0) outward."""
    ss, ws, cs = [], [], []
    s = SLICE_STEP_UM / 2
    while s < extent - SLICE_STEP_UM / 4:
        sl = _slice(f, tip_v + sign * s)
        if sl:
            ss.append(s)
            ws.append(sl[0])
            cs.append(sl[1])
        s += SLICE_STEP_UM
    return {"s": ss, "w": ws, "c": cs}


def _measure_finger(f: kdb.Region, column: str) -> dict:
    b = f.bbox()
    sign = 1 if column == "top" else -1
    tip_v = (b.bottom if sign > 0 else b.top) * DBU
    extent = b.height() * DBU
    prof = _finger_profile(f, tip_v, sign, extent)
    ss, ws, cs = prof["s"], prof["w"], prof["c"]
    tip_n = max(1, sum(1 for s in ss if s <= TIP_ZONE_UM))
    w0 = statistics.median(ws[:tip_n])
    c0 = statistics.median(cs[:tip_n])
    def off(s_):
        sl = _slice(f, tip_v + sign * s_)
        return sl is None or abs(sl[0] - w0) > STRAIGHT_TOL_UM or abs(sl[1] - c0) > STRAIGHT_TOL_UM

    straight = extent
    for k, s in enumerate(ss):
        if abs(ws[k] - w0) > STRAIGHT_TOL_UM or abs(cs[k] - c0) > STRAIGHT_TOL_UM:
            good, bad = (ss[k - 1] if k else 0.0), s
            for _ in range(10):  # bisect to ~0.1 nm·2^-10 of the step
                mid = (good + bad) / 2
                good, bad = (good, mid) if off(mid) else (mid, bad)
            straight = round(good, 3)
            break
    mean_w, centre_len = _strip(f)
    # the far end: centroid of the last 1 µm (along v) of the finger
    far_v0 = tip_v + sign * max(extent - 1.0, 0.0)
    far = f & _band(far_v0, tip_v + sign * extent)
    fu, fv = _centroid(far)
    out = {
        "column": column,
        "tip_u": round(c0, 4), "tip_v": round(tip_v, 4),
        "tip_width_um": round(w0, 4),
        "tip_edges_u": [round(c0 - w0 / 2, 4), round(c0 + w0 / 2, 4)],
        "straight_length_um": round(straight, 3),
        "extent_along_axis_um": round(extent, 3),
        "centreline_length_um": round(centre_len, 3),
        "mean_width_um": round(mean_w, 4),
        "far_end_u": round(fu, 3), "far_end_v": round(fv, 3),
        "lateral_offset_at_far_end_um": round(fu - c0, 3),
    }
    if straight < extent - 0.5:
        outer = f - _band(tip_v, tip_v + sign * (straight + 0.2))
        ow, ol = _strip(outer)
        bend_u = cs[min(range(len(ss)), key=lambda k: abs(ss[k] - straight))] if ss else c0
        dv = abs(fv - (tip_v + sign * straight))
        out.update({
            "bend_from_tip_um": round(straight, 3),
            "outer_width_um": round(ow, 4),
            "outer_length_um": round(ol, 3),
            "outer_angle_deg": round(math.degrees(math.atan2(fu - bend_u, dv)) if dv else 0.0, 2),
        })
    return out


def _split_columns(pieces: list[kdb.Region]) -> tuple[list[int], list[int], float]:
    """Indices of the top / bottom column (by the v of each piece's centroid, 2-means)."""
    vs = [_centroid(p)[1] for p in pieces]
    lo, hi = min(vs), max(vs)
    for _ in range(20):
        mid = (lo + hi) / 2
        a = [v for v in vs if v >= mid] or [hi]
        b = [v for v in vs if v < mid] or [lo]
        hi, lo = sum(a) / len(a), sum(b) / len(b)
    mid = (lo + hi) / 2
    top = [k for k, v in enumerate(vs) if v >= mid]
    bot = [k for k, v in enumerate(vs) if v < mid]
    return top, bot, mid


def _column_summary(fs: list[dict]) -> dict:
    if not fs:
        return {"n": 0}
    fs = sorted(fs, key=lambda d: d["tip_u"])
    tips_u = [d["tip_u"] for d in fs]
    pitch = [round(b - a, 4) for a, b in zip(tips_u, tips_u[1:])]
    spacing = [round(b["tip_edges_u"][0] - a["tip_edges_u"][1], 4) for a, b in zip(fs, fs[1:])]
    far = [(d["far_end_u"], d["far_end_v"]) for d in fs]
    far_pitch = [round(math.hypot(b[0] - a[0], b[1] - a[1]), 3) for a, b in zip(far, far[1:])]
    tv = [d["tip_v"] for d in fs]

    def med(xs):
        return round(statistics.median(xs), 4) if xs else None

    return {
        "n": len(fs),
        "tip_v_median": med(tv),
        "tip_alignment_spread_um": round(max(tv) - min(tv), 4),
        "tip_width_um": med([d["tip_width_um"] for d in fs]),
        "tip_width_range_um": [min(d["tip_width_um"] for d in fs), max(d["tip_width_um"] for d in fs)],
        "pitch_at_tips_um": med(pitch), "pitch_at_tips_all_um": pitch,
        "spacing_at_tips_um": med(spacing), "spacing_at_tips_all_um": spacing,
        "straight_length_um": med([d["straight_length_um"] for d in fs]),
        "straight_length_range_um": [min(d["straight_length_um"] for d in fs), max(d["straight_length_um"] for d in fs)],
        "extent_along_axis_um": med([d["extent_along_axis_um"] for d in fs]),
        "centreline_length_um": med([d["centreline_length_um"] for d in fs]),
        "centreline_length_range_um": [min(d["centreline_length_um"] for d in fs), max(d["centreline_length_um"] for d in fs)],
        "mean_width_um": med([d["mean_width_um"] for d in fs]),
        "outer_width_um": med([d["outer_width_um"] for d in fs if "outer_width_um" in d]),
        "outer_angle_deg_range": ([min(d["outer_angle_deg"] for d in fs if "outer_angle_deg" in d),
                                   max(d["outer_angle_deg"] for d in fs if "outer_angle_deg" in d)]
                                  if any("outer_angle_deg" in d for d in fs) else None),
        "far_end_pitch_um": med(far_pitch), "far_end_pitch_all_um": far_pitch,
        "span_at_tips_um": round(fs[-1]["tip_edges_u"][1] - fs[0]["tip_edges_u"][0], 4),
        "tips_u_range": [fs[0]["tip_edges_u"][0], fs[-1]["tip_edges_u"][1]],
    }


# ====================================================================== contact gate
def _contact_gate(cg: kdb.Region, fingers: list[dict], fregs: list[kdb.Region], cols: dict, gap_mid: float) -> dict:
    if cg.is_empty():
        return {"n_shapes": 0}
    shapes = []
    for p in cg.each():
        b = p.bbox()
        shapes.append({"bbox_local_um": [_um(b.left), _um(b.bottom), _um(b.right), _um(b.top)],
                       "size_um": [_um(b.width()), _um(b.height())], "area_um2": round(p.area() * DBU * DBU, 3)})
    per_finger = []
    for d, f in zip(fingers, fregs):
        ov = cg & f
        if ov.is_empty():
            continue
        b = ov.bbox()
        sign = 1 if d["column"] == "top" else -1
        s_vals = sorted(abs(v * DBU - d["tip_v"]) for v in (b.bottom, b.top))
        mid_s = (s_vals[0] + s_vals[1]) / 2
        win = _box(d["tip_u"], d["tip_v"] + sign * mid_s, d["tip_width_um"] / 2 + 0.6, 0.05)
        sl = _slice(cg & win, d["tip_v"] + sign * mid_s)
        per_finger.append({"column": d["column"], "index": d["index"],
                           "overlap_from_tip_um": round(s_vals[0], 3), "overlap_to_um": round(s_vals[1], 3),
                           "overlap_length_um": round(s_vals[1] - s_vals[0], 3),
                           "cg_width_on_finger_um": round(sl[0], 4) if sl else None,
                           "finger_width_um": d["tip_width_um"]})
    out = {"n_shapes": cg.count(), "shapes": shapes, "per_finger": per_finger}
    for col in ("top", "bottom"):
        rows = [r for r in per_finger if r["column"] == col]
        out[f"{col}_fingers_covered"] = len(rows)
        if rows:
            out[f"{col}_overlap_length_um"] = statistics.median(r["overlap_length_um"] for r in rows)
            out[f"{col}_overlap_from_tip_um"] = statistics.median(r["overlap_from_tip_um"] for r in rows)
            widths = [r["cg_width_on_finger_um"] for r in rows if r["cg_width_on_finger_um"]]
            out[f"{col}_cg_width_on_finger_um"] = statistics.median(widths) if widths else None
    # how close the CG comes to the column gap over the finger array (positive = stays behind
    # the tip line, 0 = flush with the tips); parts outside the array span are reported apart
    spans = [cols[c]["tips_u_range"] for c in ("top", "bottom") if cols.get(c, {}).get("n")]
    u0, u1 = (min(s[0] for s in spans), max(s[1] for s in spans)) if spans else (0.0, 0.0)
    over_array = kdb.Region(kdb.Box(_dbu(u0), -_BIG, _dbu(u1), _BIG))
    for col, sign in (("top", 1), ("bottom", -1)):
        c = cols.get(col, {})
        if not c.get("n"):
            continue
        half = cg & over_array & (_band(gap_mid, _BIG * DBU) if sign > 0 else _band(-_BIG * DBU, gap_mid))
        if half.is_empty():
            continue
        b = half.bbox()
        edge = (b.bottom if sign > 0 else b.top) * DBU
        out[f"{col}_clearance_to_gap_um"] = round(sign * (edge - c["tip_v_median"]), 4)
    top_v = cols.get("top", {}).get("tip_v_median")
    bot_v = cols.get("bottom", {}).get("tip_v_median")
    if top_v is not None and bot_v is not None:
        in_gap = cg & _band(bot_v + 0.01, top_v - 0.01)
        inside = in_gap & over_array
        outside = in_gap - over_array
        out["area_in_gap_over_array_um2"] = round(inside.area() * DBU * DBU, 3)
        out["crosses_gap_outside_array"] = not outside.is_empty()
        if not outside.is_empty():
            b = outside.bbox()
            out["gap_crossing_u_um"] = [_um(b.left), _um(b.right)]
            out["gap_crossing_distance_from_array_um"] = round(max(b.left * DBU - u1, u0 - b.right * DBU, 0.0), 3)
    return out


# ====================================================================== electrodes
def _electrode(e: kdb.Region, fingers: list[dict], fregs: list[kdb.Region], cg: kdb.Region, pads: kdb.Region) -> dict:
    """Measure one (merged) electrode polygon in the local frame."""
    # inner end: the vertex closest to the device centre (origin of the local frame)
    pts = [(q.x * DBU, q.y * DBU) for p in e.each() for q in p.each_point_hull()]
    iu, iv = min(pts, key=lambda t: math.hypot(*t))
    far = max(pts, key=lambda t: math.hypot(*t))
    end_piece = e & _box(iu, iv, ELECTRODE_END_BOX_UM)
    w_end, _ = _strip(end_piece)
    contacts = []
    for d, f in zip(fingers, fregs):
        ov = e & f
        if ov.is_empty():
            continue
        b = ov.bbox()
        s0 = min(abs(v * DBU - d["tip_v"]) for v in (b.bottom, b.top))
        s1 = max(abs(v * DBU - d["tip_v"]) for v in (b.bottom, b.top))
        fw = d.get("outer_width_um") or d["tip_width_um"]
        contacts.append({"column": d["column"], "index": d["index"],
                         "overlap_area_um2": round(ov.area() * DBU * DBU, 3),
                         "overlap_from_tip_um": round(s0, 3), "overlap_to_um": round(s1, 3),
                         "overlap_length_along_finger_um": round(ov.area() * DBU * DBU / fw, 3) if fw else None})
    cg_ov = e & cg
    kind = ("finger" if contacts and cg_ov.is_empty() else "contact_gate" if (not contacts and not cg_ov.is_empty())
            else "finger+contact_gate" if contacts else "neither")
    out = {
        "kind": kind,
        "inner_end_local_um": [round(iu, 3), round(iv, 3)],
        "inner_end_distance_um": round(math.hypot(iu, iv), 3),
        "far_point_local_um": [round(far[0], 3), round(far[1], 3)],
        "width_near_device_um": round(w_end, 4),
        "reaches_pads": bool(e.interacting(pads).count()) if not pads.is_empty() else False,
        "fingers": contacts,
    }
    if not cg_ov.is_empty():
        b = cg_ov.bbox()
        out["cg_overlap_area_um2"] = round(cg_ov.area() * DBU * DBU, 3)
        out["cg_overlap_bbox_um"] = [_um(b.left), _um(b.bottom), _um(b.right), _um(b.top)]
    return out


def _electrode_spacing(pieces: list[kdb.Region]) -> list[float]:
    """For each electrode piece, the distance to the nearest other piece (µm)."""
    out = []
    for k, p in enumerate(pieces):
        others = kdb.Region()
        for j, q in enumerate(pieces):
            if j != k:
                others += q
        d = _min_distance(p, others, 30.0)
        if d is not None:
            out.append(d)
    return out


# ====================================================================== landing area / markers
def _landing(land: kdb.Region, centre_chip: tuple[float, float]) -> dict:
    cx, cy = centre_chip
    pt = kdb.Region(kdb.Box(_dbu(cx) - 1, _dbu(cy) - 1, _dbu(cx) + 1, _dbu(cy) + 1))
    shapes = []
    # raw (unmerged) polygons: merging would hide a small landing square inside the chip outline
    containing = [p for p in land.each() if kdb.Region(p).interacting(pt).count()]
    for p in sorted(containing, key=lambda p: p.area()):
        b = p.bbox()
        shapes.append({"size_um": [_um(b.width()), _um(b.height())],
                       "centre_offset_um": [round(b.center().x * DBU - cx, 3), round(b.center().y * DBU - cy, 3)]})
    return {"containing_shapes": shapes, "smallest": shapes[0] if shapes else None}


def _markers(marks: kdb.Region, centre_chip: tuple[float, float], radius: float) -> dict:
    cx, cy = centre_chip
    near = marks.interacting(_box(cx, cy, radius))  # whole shapes, not clipped
    items = []
    for p in near.each():
        b = p.bbox()
        items.append({"dx_um": round(b.center().x * DBU - cx, 3), "dy_um": round(b.center().y * DBU - cy, 3),
                      "size_um": [_um(b.width()), _um(b.height())]})
    by_size: dict[str, list] = {}
    for it in items:
        by_size.setdefault(f"{it['size_um'][0]:g}x{it['size_um'][1]:g}", []).append([it["dx_um"], it["dy_um"]])
    return {"radius_um": radius, "n": len(items), "by_size": by_size, "items": items}


def _raw_region(gds_path: Path, layer: tuple[int, int]) -> kdb.Region:
    layout = kdb.Layout()
    layout.read(str(gds_path))
    return _copy_region(layout.top_cell(), layout.find_layer(*layer))


def _copy_region(top: kdb.Cell, li: int | None) -> kdb.Region:
    """Flattened copy of one layer (a Region built straight from a shape iterator keeps a
    reference to the layout, which may be gone when the caller uses it)."""
    reg = kdb.Region()
    if li is not None:
        reg.insert(top.begin_shapes_rec(li))
    return reg


# ====================================================================== main entry
def analyse_design(gds_path: str | Path, layer_roles: dict[str, tuple[int, int]] | None = None,
                   name: str | None = None) -> DesignAnalysis:
    """Measure a two-column pre-patterned electrode design. ``layer_roles`` maps the roles
    landing, markers, pads, fingers, contact_gate, electrodes to (layer, datatype); a role may be
    missing. ``None`` uses :data:`WSE2_BATCH1_LAYERS` (recorded in the result as the source)."""
    gds_path = Path(gds_path)
    source = "given by the caller"
    if layer_roles is None:
        layer_roles, source = WSE2_BATCH1_LAYERS, "WSE2_BATCH1_LAYERS (user's message, 2026-11-16)"
    regs = _read_layers(gds_path, layer_roles)
    for role in ("landing", "markers", "pads", "fingers", "contact_gate", "electrodes"):
        regs.setdefault(role, kdb.Region())
    result = DesignAnalysis(name=name or gds_path.stem, gds=str(gds_path),
                            layer_roles={k: list(v) for k, v in layer_roles.items()},
                            layer_roles_source=source, frame={})
    if regs["fingers"].is_empty():
        result.notes.append("no finger shapes on the fingers layer")
        return result

    # ---- local frame: fingers along +v, origin at the device centre
    theta = finger_axis_angle(regs["fingers"])
    rot = kdb.ICplxTrans(1.0, 90.0 - theta, False, 0, 0)
    fin = regs["fingers"].transformed(rot)
    pieces = [kdb.Region(p) for p in fin.each()]
    top_i, bot_i, _ = _split_columns(pieces)
    tips_top = [pieces[k].bbox().bottom * DBU for k in top_i]
    tips_bot = [pieces[k].bbox().top * DBU for k in bot_i]
    v_mid = (statistics.median(tips_top) + statistics.median(tips_bot)) / 2 if tips_top and tips_bot else 0.0
    all_tip = kdb.Region()
    for k in top_i:
        all_tip += pieces[k] & _band(tips_top[top_i.index(k)], tips_top[top_i.index(k)] + 0.5)
    for k in bot_i:
        all_tip += pieces[k] & _band(tips_bot[bot_i.index(k)] - 0.5, tips_bot[bot_i.index(k)])
    tb = all_tip.bbox()
    u_mid = (tb.left + tb.right) / 2 * DBU
    shift = kdb.ICplxTrans(1.0, 0.0, False, -_dbu(u_mid), -_dbu(v_mid))
    to_local = shift * rot
    to_chip = to_local.inverted()
    c_chip = to_chip * kdb.Point(0, 0)
    centre_chip = (c_chip.x * DBU, c_chip.y * DBU)
    result.frame = {"finger_axis_deg": round(theta, 3), "rotation_to_local_deg": round(90.0 - theta, 3),
                    "device_centre_chip_um": [round(centre_chip[0], 3), round(centre_chip[1], 3)],
                    "convention": "local (u, v): fingers along +v, top column at v > 0, origin at the middle "
                                  "of the column gap and of the finger array"}
    L = {role: r.transformed(to_local) for role, r in regs.items() if role in ("fingers", "contact_gate", "electrodes", "pads")}

    # ---- fingers
    fregs, fingers = [], []
    for col, idx in (("top", top_i), ("bottom", bot_i)):
        col_regs = [pieces[k].transformed(shift) for k in idx]
        meas = [_measure_finger(r, col) for r in col_regs]
        order = sorted(range(len(meas)), key=lambda k: meas[k]["tip_u"])
        for n, k in enumerate(order):
            meas[k]["index"] = n
            fingers.append(meas[k])
            fregs.append(col_regs[k])
    result.fingers = fingers
    cols = {c: _column_summary([d for d in fingers if d["column"] == c]) for c in ("top", "bottom")}
    result.columns = cols
    if cols["top"]["n"] and cols["bottom"]["n"]:
        t, b = cols["top"], cols["bottom"]
        result.array = {
            "column_gap_um": round(t["tip_v_median"] - b["tip_v_median"], 4),
            "column_gap_min_um": round(min(d["tip_v"] for d in fingers if d["column"] == "top")
                                       - max(d["tip_v"] for d in fingers if d["column"] == "bottom"), 4),
            "column_offset_u_um": round((t["tips_u_range"][0] + t["tips_u_range"][1]) / 2
                                        - (b["tips_u_range"][0] + b["tips_u_range"][1]) / 2, 4),
            "tips_facing": _tips_facing(fingers),
            "span_at_tips_um": round(max(t["tips_u_range"][1], b["tips_u_range"][1])
                                     - min(t["tips_u_range"][0], b["tips_u_range"][0]), 4),
            "total_extent_um": [_um(L["fingers"].bbox().width()), _um(L["fingers"].bbox().height())],
        }
    else:
        result.notes.append("fingers found in only one column")

    # ---- contact gate
    result.contact_gate = _contact_gate(L["contact_gate"], fingers, fregs, cols, 0.0)

    # ---- electrodes near the device
    fb = L["fingers"].bbox()
    near_box = kdb.Region(fb.enlarged(_dbu(NEAR_DEVICE_UM), _dbu(NEAR_DEVICE_UM)))
    connect = L["fingers"] + L["contact_gate"] + L["pads"]
    els, unconnected = [], 0
    for p in L["electrodes"].each():
        e = kdb.Region(p)
        if not e.interacting(near_box).count():
            continue
        if not e.interacting(connect).count():
            unconnected += 1
            continue
        els.append(e)
    meas = [_electrode(e, fingers, fregs, L["contact_gate"], L["pads"]) for e in els]
    order = sorted(range(len(meas)), key=lambda k: math.atan2(meas[k]["inner_end_local_um"][1], meas[k]["inner_end_local_um"][0]))
    meas = [meas[k] for k in order]
    els = [els[k] for k in order]
    clip = kdb.Region(fb.enlarged(_dbu(30.0), _dbu(30.0)))
    near_pieces = [e & clip for e in els]
    spacing = _electrode_spacing([p for p in near_pieces if not p.is_empty()])
    for m, e in zip(meas, els):
        others = kdb.Region()
        for d, f in zip(fingers, fregs):
            if not any(c["column"] == d["column"] and c["index"] == d["index"] for c in m["fingers"]):
                others += f
        m["clearance_to_other_fingers_um"] = _min_distance(e & clip, others, 30.0)
        if m["kind"] != "contact_gate":
            m["clearance_to_contact_gate_um"] = _min_distance(e & clip, L["contact_gate"], 30.0)
    for k, m in enumerate(meas):
        m["id"] = k
    result.electrodes = meas
    contacted = {(c["column"], c["index"]) for m in meas for c in m["fingers"]}
    floating = {col: [d["index"] for d in fingers if d["column"] == col and (col, d["index"]) not in contacted]
                for col in ("top", "bottom")}
    kinds = [m["kind"] for m in meas]
    fe = [m for m in meas if m["kind"] == "finger"]
    result.electrode_summary = {
        "n_near_device": len(meas),
        "n_unconnected_near_device": unconnected,
        "by_kind": {k: kinds.count(k) for k in sorted(set(kinds))},
        "finger_electrodes_per_column": {col: sum(1 for m in fe if m["fingers"] and m["fingers"][0]["column"] == col)
                                         for col in ("top", "bottom")},
        "electrodes_on_more_than_one_finger": sum(1 for m in meas if len(m["fingers"]) > 1),
        "floating_fingers": floating,
        "n_floating": {c: len(v) for c, v in floating.items()},
        "width_near_device_um": _med([m["width_near_device_um"] for m in meas]),
        "width_near_device_range_um": _rng([m["width_near_device_um"] for m in meas]),
        "finger_electrode_width_um": _med([m["width_near_device_um"] for m in fe]),
        "min_spacing_near_device_um": round(min(spacing), 4) if spacing else None,
        "nearest_neighbour_spacing_median_um": _med(spacing),
        "overlap_from_tip_um": _med([c["overlap_from_tip_um"] for m in fe for c in m["fingers"]]),
        "overlap_from_tip_range_um": _rng([c["overlap_from_tip_um"] for m in fe for c in m["fingers"]]),
        "overlap_length_along_finger_um": _med([c["overlap_length_along_finger_um"] for m in fe for c in m["fingers"]
                                                if c["overlap_length_along_finger_um"]]),
        "overlap_length_range_um": _rng([c["overlap_length_along_finger_um"] for m in fe for c in m["fingers"]
                                         if c["overlap_length_along_finger_um"]]),
        "clearance_to_other_fingers_min_um": _min([m["clearance_to_other_fingers_um"] for m in fe]),
        "clearance_to_contact_gate_min_um": _min([m.get("clearance_to_contact_gate_um") for m in fe]),
        "all_reach_pads": all(m["reaches_pads"] for m in meas) if meas else None,
        "neither_inner_ends_local_um": [m["inner_end_local_um"] for m in meas if m["kind"] == "neither"],
    }

    # ---- landing area and markers (chip frame, relative to the device centre)
    if "landing" in layer_roles:
        result.landing = _landing(_raw_region(gds_path, layer_roles["landing"]), centre_chip)
    if "markers" in layer_roles:
        sm = result.landing.get("smallest") if result.landing else None
        radius = (max(sm["size_um"]) / 2 if sm else 250.0)
        result.markers = _markers(_raw_region(gds_path, layer_roles["markers"]), centre_chip, radius)
        if not regs["pads"].is_empty():
            result.markers["pad_free_half_size_um"] = _clear_half_size(regs["pads"], centre_chip)
    return result


def _tips_facing(fingers: list[dict]) -> dict:
    """Do the top tips sit opposite the bottom tips? Offset of each top tip from the nearest
    bottom tip (u)."""
    tops = [d["tip_u"] for d in fingers if d["column"] == "top"]
    bots = [d["tip_u"] for d in fingers if d["column"] == "bottom"]
    if not tops or not bots:
        return {}
    offs = [min((t - b for b in bots), key=abs) for t in tops]
    return {"median_offset_um": round(statistics.median(offs), 4), "max_abs_offset_um": round(max(abs(o) for o in offs), 4)}


def _clear_half_size(pads: kdb.Region, centre: tuple[float, float]) -> float:
    """Half-size of the largest square around the device centre free of pad / fan-out shapes."""
    lo, hi = 0.0, 5000.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if pads.interacting(_box(centre[0], centre[1], mid)).count():
            hi = mid
        else:
            lo = mid
    return round(lo, 2)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _rng(xs):
    xs = [x for x in xs if x is not None]
    return [round(min(xs), 4), round(max(xs), 4)] if xs else None


def _min(xs):
    xs = [x for x in xs if x is not None]
    return round(min(xs), 4) if xs else None


def local_to_chip(analysis: DesignAnalysis, u: float, v: float) -> tuple[float, float]:
    """Local-frame µm → chip µm for a finished analysis."""
    a = math.radians(-analysis.frame["rotation_to_local_deg"])
    cx, cy = analysis.frame["device_centre_chip_um"]
    return cx + math.cos(a) * u - math.sin(a) * v, cy + math.sin(a) * u + math.cos(a) * v


def layer_polygons(gds_path: str | Path, layer: tuple[int, int], near: tuple[float, float] | None = None,
                   half_size: float = 100.0) -> list[list[tuple[float, float]]]:
    """Merged polygons (chip µm) of one layer, optionally only those within a box around ``near``."""
    reg = _raw_region(Path(gds_path), layer).merged()
    if near is not None:
        reg = reg.interacting(_box(near[0], near[1], half_size))
    out = []
    for p in reg.each():
        q = p.resolved_holes() if p.holes() else p
        out.append([(pt.x * DBU, pt.y * DBU) for pt in q.each_point_hull()])
    return out
