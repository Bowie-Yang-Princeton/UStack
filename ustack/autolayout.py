"""Flake layout › Auto design (user, 2026-11-21): place the traced flakes (turn + shift, never a change of shape or size,
never mirrored) so that the stack obeys the lab's rules and leaves room for the metal patterns of Stage 2 — the Hall-bar
finger array, the contact gate, the gate contacts and the electrodes' landing places (the electrodes themselves and their
routing are not designed here).

The stack is placed in stages, in the order the user proposed (each stage's choice fixes what the next one must contain):

1. **Device layers** — the first device layer stays where it is (the reference); every other one is shifted (and turned
   only when "Keep the twist angle" is off) so that the overlap (the twisted region) holds as many finger pairs as the
   bond pads can take. Search: a raster estimate of the pair count (``_pairs``: the region turned so the array axis is x,
   eroded by one pair strip, the longest run along x → pairs at the finger pitch) on a 1 µm grid of shifts pruned by the
   overlap area (one correlation), refined at 0.25 µm / 2°; the best few are fitted exactly (``fingers.fit_finger_array``).
2. **Bottom hBN** — must hold the twisted region and the fingers' straight parts (``fingers.bbn_margin_um`` inside),
   wants the fingers at their preset length and the device layers; then the **Hall bars are generated** (the real Stage 2
   generator, ``ebl_flow.generate``, on a scratch flow) — each finger as long as the bottom hBN allows.
3. **Bottom gate** — must lie under the channel between the finger tips, wants the twisted region, and must leave a free
   stretch outside the bottom hBN for its contact (an ``electrodes.bg_contact_um`` square ``electrodes.gate_contact_
   clearance_um`` from the hBN); that contact place is kept clear by every later flake.
4. **Middle hBN** (the contact-gate dielectric) — must hold the contact-gate comb (``cg.hbn_margin_um`` inside) and the
   twisted region (and the TMD by ``xu.middle_hbn_edge_margin_um`` when that rule applies), keeps off the place where each
   finger's electrode lands (its last ``electrodes.finger_overlap_um``, grown by ``electrodes.flake_clearance_um``); then
   the **contact gate is generated** (real generator).
5. **Top hBN, top gate, cap** — the top hBN holds the twisted region and the comb's teeth, keeping off the finger and
   contact-gate electrode landings; the top gate covers the twisted region, never touches a conductor its hBN does not
   cover, and leaves a contact place; the cap covers the twisted region and keeps off every contact place.

Every stage's translation search is one FFT correlation per turn of the flake (targets fixed, the turned flake the
kernel): "inside" = the must-hold pixels covered by the flake shrunk by its margin, "wanted" / "avoided" area, "never"
area and the exposed area for a contact — 10° turns on a 0.5 µm grid, then ±5° in 1° steps on a 0.25 µm grid around the
best few, then an exact check with KLayout booleans. A small beam (several candidates kept per stage) keeps the greedy
order from locking in an early choice that fails later. Every stack candidate is finally scored by the real generators,
the electrode landings (``ebl_flow.finger_landings``) and ``validate_stage1``.

``stack.alignment_shift_um`` (the lab's stacking misalignment) is added to every "must hold" margin and every clearance:
a flake transferred later can land that far from its planned place. Every lab value used is read from config/ and asked
for (``required_values``) when missing; the search constants below are program choices (raster size, angle steps)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import klayout.db as kdb
import numpy as np

from . import ebl_flow as ef
from . import fingers as fg
from . import geometry as geo
from .cancel import poll, set_check
from .config import LabConfig, MissingItem, MissingValues
from .model import DeviceSpec

Poly = list[tuple[float, float]]

COARSE_RES = 0.5            # µm per raster pixel, first search (program choice)
FINE_RES = 0.25             # µm per raster pixel, refinement and the device layers
COARSE_STEP_DEG = 10.0      # turns tried in the first search
FINE_HALF_DEG = 5.0         # refinement: ± this around a coarse turn …
FINE_STEP_DEG = 1.0         # … in these steps
PAIR_STEP_DEG = 6.0         # array axis angles tried by the pair estimate (refined at 2°)
SHIFT_STEP_UM = 1.0         # device-layer shifts tried
AVOID_WEIGHT = 20.0         # 1 µm² over a kept-clear place costs as much as 20 µm² of wanted cover
NMS_UM = 2.0                # candidates of one turn at least this far apart
MBN_UNDER_TG_COST = 0.05    # middle hBN under the top gate (× AVOID_WEIGHT per µm²): keeps the leads' ends off the gate
COARSE_SLACK = 0.08         # coarse pass: this share of the must-hold area may stick out (the fine pass decides)
DISTINCT_UM = 5.0           # placements kept for a later stage differ by this shift …
DISTINCT_DEG = 15.0         # … or this turn (a later stage that fails gets real alternatives)


class Cancelled(Exception):
    pass


# ====================================================================== roles and values
@dataclass
class Roles:
    devices: list            # device layers (bottom → top)
    fingers: object          # the Hall-bar contacts layer
    bbn: object              # hBN the fingers are written on
    bg: object | None        # bottom gate flake (conducting, not metal)
    mbn: object | None       # hBN under the contact gate
    cg: object | None        # contact-gate layer
    tbn: object | None       # top-gate dielectric
    tg: object | None        # top gate flake
    caps: list               # encapsulation flakes above the top gate
    others: list             # any other flake (placed over the twisted region)
    flakes: list             # every flake layer (non-metal, not the substrate), bottom → top
    config: LabConfig = None


def _is_flake(config: LabConfig, layer) -> bool:
    mat = config.material(layer.material) if layer.material else None
    return mat is not None and mat.category not in ("metal", "substrate")


def _conducting(config: LabConfig, layer) -> bool:
    mat = config.material(layer.material) if layer.material else None
    return mat is not None and mat.conducting == "yes"


def roles(spec: DeviceSpec, config: LabConfig) -> Roles:
    """The flakes' parts in the finger-array workflow; ValueError for stacks it does not handle."""
    idx = spec.layers.index
    fl = ef.step_layer(spec, "hall_bars")
    if fl is None:
        raise ValueError("Auto design places the flakes for the finger-array pattern (Stage 2): the stack has no Hall-bar "
                         "contacts layer.")
    devices = [spec.layer(u) for u in ef.device_layer_uids(spec)]
    if not devices:
        raise ValueError("The stack has no device layer.")
    bbn = ef.hbn_under(spec, "hall_bars")
    if bbn is None or any(idx(d) < idx(fl) for d in devices):
        raise ValueError("Auto design needs the fingers written on an hBN under the device layers (as in the Xu structure); "
                         "fingers pre-patterned on the chip under the bottom gate are not handled yet.")
    flakes = [l for l in spec.layers if _is_flake(config, l)]
    bg = next((l for l in reversed(spec.layers[:idx(bbn)]) if l.function == "bottom_gate" and _is_flake(config, l)), None)
    cg = ef.step_layer(spec, "contact_gate")
    mbn = ef.hbn_under(spec, "contact_gate") if cg is not None else None
    if mbn is not None and (mbn is bbn or idx(mbn) < idx(devices[-1])):
        mbn = None
    tbn = next((l for l in spec.layers if l.function == "top_gate_dielectric" and _is_flake(config, l)), None)
    tg = next((l for l in spec.layers if l.function == "top_gate" and _is_flake(config, l)), None)
    caps = [l for l in flakes if l.function == "encapsulation" and tg is not None and idx(l) > idx(tg)]
    known = {id(x) for x in devices + [bbn, bg, mbn, tbn, tg] + caps if x is not None}
    others = [l for l in flakes if id(l) not in known]
    return Roles(devices, fl, bbn, bg, mbn, cg, tbn, tg, caps, others, flakes, config)


@dataclass
class Values:
    p: fg.FingerParams
    shift: float             # stacking misalignment (added to every margin)
    clearance: float         # electrode metal outside the flakes above the fingers
    overlap: float           # electrode overlap on a finger (at most)
    gclr: float              # gate contact clearance
    contact: float           # gate contact size
    xu_margin: float | None  # middle hBN beyond the TMD (when the rule applies)
    graphite: float          # electrode / conductor to a graphite gate
    cap_pairs: int | None    # finger pairs the bond pads can take
    cg_overlap: float        # contact-gate electrode overlap on its bus lead
    config: LabConfig = None
    roles: Roles = None
    target: int | None = None   # finger pairs wanted (this run)


def required_values(spec: DeviceSpec, config: LabConfig) -> list[MissingItem]:
    """Every lab value the auto design needs and config/ does not have (asked all at once)."""
    r = roles(spec, config)
    items = []

    def preset(key, label, unit, hint):
        if config.preset(key) is None:
            items.append(MissingItem(f"preset:{key}", label, unit, hint=hint))
    rule = config.rules.get("stack.alignment_shift_um")
    if rule is not None and rule.is_missing:
        items.append(config.missing_item("stack.alignment_shift_um"))
    preset("fingers.bbn_margin_um", "Finger ends inside the bottom hBN by", "µm",
           "Each finger runs as far as the bottom hBN allows, ending this far inside it (saved as a preset).")
    if r.cg is not None:
        preset("cg.hbn_margin_um", "Contact gate inside its hBN by", "µm",
               "The whole contact gate stays this far inside the hBN it is written on (saved as a preset).")
        preset("cg.bus_tmd_gap_um", "Contact-gate bus from the TMD by", "µm",
               "The bus joining the teeth runs this far from the TMD (saved as a preset).")
    preset("electrodes.flake_clearance_um", "Electrode clearance from the flakes above the fingers", "µm",
           "Each finger's electrode lands where its metal is this far outside the flakes above the fingers (saved as a preset).")
    preset("electrodes.finger_overlap_um", "Electrode overlap on a finger (at most)", "µm",
           "A Hall-bar electrode runs at most this far along its finger (saved as a preset).")
    preset("electrodes.gate_contact_clearance_um", "Gate contact clearance", "µm",
           "A gate contact stays this far from the hBN over the gate and the other conductors (saved as a preset).")
    preset("electrodes.bg_contact_um", "Gate contact size", "µm", "Side of each gate's contact square (saved as a preset).")
    return items


def _values(spec: DeviceSpec, config: LabConfig, r: Roles, flow: ef.Flow | None) -> Values:
    missing = required_values(spec, config)
    if missing:
        raise MissingValues(missing)
    from .validation import Report, _Ctx
    ctx = _Ctx(spec, config, Report())
    xu = config.value("xu.middle_hbn_edge_margin_um") if ctx.applies("xu.middle_hbn_edge_margin_um") else None
    cap = None
    bp = ef.bondpads_for(flow) if flow is not None else None
    if bp is not None:
        others = (2 if r.cg is not None else 0) + sum(1 for g in (r.bg, r.tg) if g is not None and _conducting(config, g))
        cap = max(1, (len(bp.pad_ends()) - others) // 2)
    hb = flow.steps.get("hall_bars") if flow is not None else None
    if hb is not None and hb.max_pairs:
        cap = min(cap, hb.max_pairs) if cap else hb.max_pairs
    p = fg.FingerParams.from_config(config, hb.overrides if hb is not None else None)
    return Values(p, float(config.value("stack.alignment_shift_um") or 0.0), float(config.preset("electrodes.flake_clearance_um")),
                  float(config.preset("electrodes.finger_overlap_um")), float(config.preset("electrodes.gate_contact_clearance_um")),
                  float(config.preset("electrodes.bg_contact_um")), float(xu) if xu is not None else None,
                  float(config.value("ebl.contact_to_graphite_gate_min_um") or 0.0), cap,
                  float(config.preset("electrodes.cg_overlap_um") or 0.0), config, r)


# ====================================================================== geometry helpers
def _turn(poly, deg: float, c) -> Poly:
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(c[0] + ca * (x - c[0]) - sa * (y - c[1]), c[1] + sa * (x - c[0]) + ca * (y - c[1])) for x, y in poly]


def _moved(poly, dx: float, dy: float) -> Poly:
    return [(x + dx, y + dy) for x, y in poly]


def _place(poly, pose) -> Poly:
    deg, dx, dy = pose
    return _moved(_turn(poly, deg, geo.centroid(poly)), dx, dy)


def _bbox(polys) -> tuple[float, float, float, float] | None:
    pts = [q for poly in polys for q in poly]
    if not pts:
        return None
    a = np.asarray(pts, float)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _union_box(*boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _polys(reg: kdb.Region) -> list[Poly]:
    return [q for q in geo.to_polys(reg) if len(q) >= 3]


def _grown(polys, d: float) -> list[Poly]:
    if not polys:
        return []
    reg = geo.regions(polys).merged()
    return _polys(geo.sized(reg, d) if abs(d) > 1e-9 else reg)


def _mask(polys, x0: float, y0: float, res: float, shape) -> np.ndarray:
    """1 where a pixel's centre is inside any polygon (pixel (i, j) centred at (x0 + (j + ½)·res, y0 + (i + ½)·res))."""
    m = np.zeros(shape, np.uint8)
    for q in polys or []:
        if len(q) >= 3:
            pts = np.round(((np.asarray(q, float) - (x0, y0)) / res - 0.5) * 4.0).astype(np.int32)
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1, cv2.LINE_8, 2)    # one polygon at a time (overlaps stay filled)
    return m


# ====================================================================== finger pairs on a region (raster estimate)
def _runs(E: np.ndarray):
    """Per row of ``E``: the longest run of ones along x → [(length, row, start)]."""
    out = []
    if not E.any():
        return out
    Z = np.zeros((E.shape[0], 1), np.int8)
    D = np.diff(np.hstack([Z, E.astype(np.int8), Z]), axis=1)
    for r in np.nonzero(E.any(1))[0]:
        s, e = np.nonzero(D[r] == 1)[0], np.nonzero(D[r] == -1)[0]
        k = int(np.argmax(e - s))
        out.append((int(e[k] - s[k]), int(r), int(s[k])))
    return out


def _pairs(mask: np.ndarray, res: float, p: fg.FingerParams, angles, target: int | None = None,
           union: np.ndarray | None = None) -> dict:
    """The best straight finger array on ``mask`` (the twisted region): the region turned so the array axis is x,
    eroded by one pair strip (a finger width along x, gap + both tip overlaps along y), the longest run of valid centres
    along x → pairs at the finger pitch (slightly conservative: the kernel is one pixel larger). With ``target`` pairs,
    among the arrays holding that many the one where ``union`` (the device layers) reaches least far out along the
    fingers from the tips is best — the contact-gate teeth must run that far (shorter teeth, a smaller comb).
    Returns {"n", "angle", "slack", "reach" (µm, both columns)}."""
    best = {"n": 0, "angle": 0.0, "slack": 0, "reach": math.inf}
    if not mask.any():
        return best
    kh = int(math.ceil((p.gap + 2 * p.min_overlap) / res)) + 1
    kw = int(math.ceil(p.width / res)) + 1
    H, W = mask.shape
    D = int(math.ceil(math.hypot(H, W))) + 4
    oy, ox = (D - H) // 2, (D - W) // 2

    def padded(m):
        out = np.zeros((D, D), np.uint8)
        out[oy:oy + H, ox:ox + W] = m
        return out
    pad = padded(mask)
    upad = padded(union) if union is not None else None
    kernel = np.ones((kh, kw), np.uint8)
    pitch_px = p.pitch / res
    half_gap = p.gap / 2 / res

    def key(d):
        if target:
            return (d["n"] >= target, -d["reach"] if d["n"] >= target else d["n"], d["slack"])
        return (d["n"], d["slack"], 0)
    for a in angles:
        poll()
        M = cv2.getRotationMatrix2D((D / 2.0, D / 2.0), float(a), 1.0)    # rows = +y: this turns the region by −a
        R = cv2.warpAffine(pad, M, (D, D), flags=cv2.INTER_NEAREST)
        runs = _runs(cv2.erode(R, kernel))
        if not runs:
            continue
        colmax = colmin = None
        if upad is not None and target:
            U = cv2.warpAffine(upad, M, (D, D), flags=cv2.INTER_NEAREST)
            occ = U.any(0)
            rows = np.arange(D)[:, None]
            colmax = np.where(occ, np.where(U > 0, rows, -1).max(0), -1)
            colmin = np.where(occ, np.where(U > 0, rows, D).min(0), D)
        for length, row, start in runs:
            n = int(math.floor((length - 1) / pitch_px)) + 1
            if target and n > target:
                n = target
            need = int(math.ceil((n - 1) * pitch_px)) + 1
            slack = length - need
            reach = 0.0
            if colmax is not None:
                s0 = start + slack // 2                # the array centred in the run
                cols = slice(s0, s0 + need)
                top = max(0.0, float(colmax[cols].max()) - (row + half_gap))
                bottom = max(0.0, (row - half_gap) - float(colmin[cols].min()))
                reach = (top + bottom) * res
            d = {"n": n, "angle": float(a) % 180.0, "slack": slack, "reach": reach}
            if key(d) > key(best):
                best = d
    return best


# ====================================================================== one flake's placement search
@dataclass
class Targets:
    must: list = field(default_factory=list)          # inside the flake shrunk by ``margin``
    margin: float = 0.0
    want: list = field(default_factory=list)          # [(polys, weight)]: area covered (flake shrunk by margin) is good
    avoid: list = field(default_factory=list)         # [(polys, weight)]: area covered (whole flake) costs AVOID_WEIGHT·w
    hard: list = field(default_factory=list)          # never covered by the flake
    blocked: list | None = None                       # exposure: the flake's area outside these …
    expose_need: float = 0.0                          # … must be at least this (µm²)
    expose_cost: float = 0.02                         # each µm² exposed beyond 2× the need costs this (bare graphite
                                                      # blocks the router and the later flakes)
    accept: object = None                             # exact check of a placed outline → bool (e.g. a contact fits)


def _fft_size(n: int) -> int:
    return cv2.getOptimalDFTSize(int(n))


def _scan(flake: Poly, c, T: Targets, angles, res: float, frame, per_angle: int = 2, tolerant: bool = False):
    """Correlation search of ``flake`` turned by each of ``angles`` (about ``c``) over ``frame``: [(score, angle, dx, dy)].
    ``tolerant`` (the coarse pass): a little of the must-hold area may stick out (up to ``COARSE_SLACK`` of it, ranked
    below the placements that hold it all) and the hard area may be touched a little — a narrow window of turns that
    fits is found between the coarse steps by the fine pass, then checked exactly."""
    X0, Y0, X1, Y1 = frame
    W, H = int(math.ceil((X1 - X0) / res)) + 1, int(math.ceil((Y1 - Y0) / res)) + 1
    SH, SW = _fft_size(H), _fft_size(W)
    shape = (H, W)

    def spec_of(arr):
        return np.fft.rfft2(arr.astype(np.float32), s=(SH, SW))
    # thin targets (0.8 µm teeth) can fall between pixel centres: "must hold" and "never" targets are drawn ¾ pixel wider
    Km = _mask(_grown(T.must, 0.75 * res), X0, Y0, res, shape)
    n_must = float(Km.sum())
    must_f = spec_of(Km) if n_must else None
    Kw = np.zeros(shape, np.float32)
    for polys, w in T.want:
        Kw += w * _mask(polys, X0, Y0, res, shape)
    want_f = spec_of(Kw) if Kw.any() else None
    Ka = np.zeros(shape, np.float32)
    for polys, w in T.avoid:
        Ka += w * _mask(polys, X0, Y0, res, shape)
    avoid_f = spec_of(Ka) if Ka.any() else None
    Kh = _mask(_grown(T.hard, 0.75 * res), X0, Y0, res, shape)
    hard_f = spec_of(Kh) if Kh.any() else None
    free_f = spec_of(1 - _mask(T.blocked, X0, Y0, res, shape)) if T.blocked is not None and T.expose_need > 0 else None
    out = []
    er = np.ones((3, 3), np.uint8)
    for a in angles:
        poll()
        F = _turn(flake, float(a), c)
        fb = _bbox([F])
        fx0, fy0 = fb[0] - res, fb[1] - res
        h, w = int(math.ceil((fb[3] - fb[1]) / res)) + 3, int(math.ceil((fb[2] - fb[0]) / res)) + 3
        if h >= H or w >= W:
            continue
        Ff = _mask([F], fx0, fy0, res, (h, w))
        Fe_polys = _grown([F], -T.margin) if T.margin > 0 else [F]
        Fe = _mask(Fe_polys, fx0, fy0, res, (h, w))
        FFf = np.conj(spec_of(np.pad(Ff, ((0, SH - h), (0, SW - w)))))
        score = np.zeros((H - h + 1, W - w + 1), np.float32)
        ok = np.ones_like(score, bool)

        def corr(kf, FF):
            return np.fft.irfft2(kf * FF, s=(SH, SW))[:H - h + 1, :W - w + 1]
        if must_f is not None:
            if tolerant:
                FFe0 = np.conj(spec_of(np.pad(Fe, ((0, SH - h), (0, SW - w)))))
                out_px = n_must - corr(must_f, FFe0)
                ok &= out_px < COARSE_SLACK * n_must + 0.5
                score -= 1e3 * np.maximum(out_px, 0.0) * res * res
            else:
                Fc = cv2.erode(Fe, er)                 # conservative: the exact check follows
                FFc = np.conj(spec_of(np.pad(Fc, ((0, SH - h), (0, SW - w)))))
                ok &= corr(must_f, FFc) > n_must - 0.5
            if not ok.any():
                continue
        if hard_f is not None:
            hit = corr(hard_f, FFf)
            if tolerant:
                ok &= hit < COARSE_SLACK * float(Ff.sum()) + 0.5
                score -= 1e3 * np.maximum(hit, 0.0) * res * res
            else:
                ok &= hit < 0.5
        if want_f is not None:
            FFe = np.conj(spec_of(np.pad(Fe, ((0, SH - h), (0, SW - w)))))
            score += corr(want_f, FFe) * res * res
        if avoid_f is not None:
            score -= AVOID_WEIGHT * corr(avoid_f, FFf) * res * res
        if free_f is not None:
            expo = corr(free_f, FFf) * res * res
            ok &= expo >= T.expose_need
            score += np.minimum(expo, 2 * T.expose_need) - T.expose_cost * np.maximum(expo - 2 * T.expose_need, 0.0)
        if not ok.any():
            continue
        score = np.where(ok, score, -np.inf)
        r = max(1, int(round(NMS_UM / res)))
        for _ in range(per_angle):
            k = int(np.argmax(score))
            i, j = divmod(k, score.shape[1])
            s = float(score[i, j])
            if not np.isfinite(s):
                break
            out.append((s, float(a), X0 + j * res - fx0, Y0 + i * res - fy0))
            score[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = -np.inf
    out.sort(key=lambda t: -t[0])
    return out


def _exact_ok(placed: Poly, T: Targets) -> bool:
    reg = geo.region(placed)
    if T.must:
        inner = geo.sized(reg, -T.margin) if T.margin > 0 else reg
        if not (geo.regions(T.must).merged() - inner).is_empty():
            return False
    if T.hard and not (reg & geo.regions(T.hard).merged()).is_empty():
        return False
    return T.accept is None or bool(T.accept(placed))


def search_flake(flake: Poly, T: Targets, keep: int = 1, angles=None, check=None) -> list[tuple[float, tuple, Poly]]:
    """The best ``keep`` placements of ``flake`` for ``T``: [(score, (angle, dx, dy) about its centroid, placed outline)]."""
    c = geo.centroid(flake)
    radius = max(math.dist(c, q) for q in flake)
    box = _union_box(_bbox(T.must), _bbox([q for polys, _w in T.want for q in polys]),
                     _bbox([q for polys, _w in T.avoid for q in polys]), _bbox(T.hard))
    if box is None:
        return [(0.0, (0.0, 0.0, 0.0), [tuple(q) for q in flake])]
    pad = 2 * radius + 2.0
    frame = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    coarse = list(np.arange(0.0, 360.0, COARSE_STEP_DEG)) if angles is None else list(angles)
    first = _scan(flake, c, T, coarse, COARSE_RES, frame, tolerant=True)
    if check:
        check()
    got: list[tuple[float, tuple, Poly]] = []
    pool, seen = [], []
    for cand in first:                                 # coarse candidates spread over turns and places
        if all(abs(math.remainder(cand[1] - q[1], 360)) > DISTINCT_DEG or math.hypot(cand[2] - q[2], cand[3] - q[3]) > DISTINCT_UM
               for q in seen):
            seen.append(cand)
            pool.append(cand)
        if len(pool) >= max(6, 3 * keep):
            break
    for s0, a0, dx0, dy0 in pool:
        poll()
        placed0 = _moved(_turn(flake, a0, c), dx0, dy0)
        b = _bbox([placed0])
        m = 3.0 + radius * math.sin(math.radians(FINE_HALF_DEG)) + 2 * FINE_RES
        fine_frame = (b[0] - m, b[1] - m, b[2] + m, b[3] + m)
        fine_angles = [a0] if angles is not None and len(coarse) == 1 else \
            list(a0 + np.arange(-FINE_HALF_DEG, FINE_HALF_DEG + 1e-9, FINE_STEP_DEG))
        cands = _scan(flake, c, T, fine_angles, FINE_RES, fine_frame, per_angle=1)
        cands.sort(key=lambda t: -t[0])
        for s, a, dx, dy in cands[:4]:
            placed = _moved(_turn(flake, a, c), dx, dy)
            if all(math.dist(geo.centroid(placed), geo.centroid(g[2])) > DISTINCT_UM
                   or abs(math.remainder(a - g[1][0], 360)) > DISTINCT_DEG for g in got) and _exact_ok(placed, T):
                got.append((s, (math.remainder(a, 360.0), dx, dy), placed))
                break
        if check:
            check()
        if len(got) >= keep:
            break
    got.sort(key=lambda g: -g[0])
    return got[:keep]


# ====================================================================== stack state
@dataclass
class State:
    outlines: dict                                    # uid → placed outline (design frame)
    poses: dict = field(default_factory=dict)         # uid → (angle, dx, dy) about the original outline's centroid
    arr: fg.FingerArray | None = None
    comb: list = field(default_factory=list)          # contact-gate shapes (generated)
    reach: list = field(default_factory=list)         # contact-gate electrode landing points
    zones: dict = field(default_factory=dict)         # kept-clear places: "bg" / "tg" contacts
    score: float = 0.0
    notes: list = field(default_factory=list)
    failed: str = ""
    angle: float | None = None                        # array axis from the device stage's estimate
    side_over: float = 0.0                            # µm² of the top gate beyond the device on the bottom gate's side

    def copy(self) -> "State":
        return State(dict(self.outlines), dict(self.poses), self.arr, list(self.comb), list(self.reach), dict(self.zones),
                     self.score, list(self.notes), self.failed, self.angle, self.side_over)


def _scratch(spec: DeviceSpec, outlines: dict, copy: bool = True) -> tuple[DeviceSpec, ef.Flow]:
    """A flow whose steps all see ``outlines`` (and, with ``copy``, a spec copy carrying them as traced borders)."""
    s = spec.copy() if copy else spec
    if copy:
        for layer in s.layers:
            if layer.uid in outlines:
                layer.outline = [list(map(float, q)) for q in outlines[layer.uid]]
    flow = ef.Flow()
    for k in ("hall_bars", "contact_gate", "electrodes"):
        flow.step(k).outlines = {u: [[tuple(q) for q in poly]] for u, poly in outlines.items()}
    return s, flow


def _landing_zones(st: State, v: Values) -> list[Poly]:
    """Where each finger's electrode lands (its last ``overlap`` µm), grown by the clearance + misalignment."""
    arr = st.arr
    if arr is None:
        return []
    from .router import wire
    p = arr.params
    out = []
    for f in arr.fingers():
        L = p.length_at(f.column, f.index)
        d = max(p.land_from_tip, L - v.overlap)
        d = min(d, max(L - p.electrode_width, 0.0))
        out += _polys(geo.sized(wire(fg.finger_path(arr, f, d), p.electrode_width), v.clearance + v.shift))
    return out


def _cg_zones(st: State, v: Values) -> list[Poly]:
    """Where the contact-gate electrodes land: each bus lead's end, back along it by the overlap, + clearance."""
    rad = v.cg_overlap + v.clearance + v.shift
    return [[(x + rad * math.cos(2 * math.pi * k / 16), y + rad * math.sin(2 * math.pi * k / 16)) for k in range(16)]
            for x, y in st.reach]


def _contact_spot(gate: Poly, covers: list[Poly], keep: list[Poly], v: Values) -> Poly | None:
    """A gate contact square (with its clearance) on the part of ``gate`` no hBN covers: None if there is none."""
    exposed = geo.region(gate)
    if covers:
        exposed = exposed - geo.sized(geo.regions(covers).merged(), v.gclr + v.shift)
    if keep:
        exposed = exposed - geo.sized(geo.regions(keep).merged(), v.gclr + v.shift)
    core = geo.sized(exposed.merged(), -(v.contact / 2 + v.shift))
    pieces = _polys(core.merged())
    if not pieces:
        return None
    big = max(pieces, key=lambda q: abs(geo.poly_area(q)))
    cx, cy = geo.centroid(big)
    if not geo.point_in_poly(cx, cy, big):
        cx, cy = big[0]
    h = v.contact / 2 + v.shift
    return geo.rect(cx - h, cy - h, cx + h, cy + h)


# ====================================================================== the stages
def _teeth_reach(spec, arr: fg.FingerArray, outlines: dict, v: Values) -> float:
    """How long the contact-gate teeth must be (both columns, µm) so the bus clears the device layers."""
    tmd = [outlines[d.uid] for d in v.roles.devices if d.uid in outlines]
    try:
        lay, _n = fg.fit_contact_gate(arr, None, None, not ef.has_source_drain(spec), keepout=fg.tmd_keepout(tmd, v.p.cg_tmd_gap))
    except ValueError:
        return math.inf
    return float(sum((lay.get("tooth") or {}).values()))


def _device_stage(spec, r: Roles, outlines: dict, movable: set, v: Values, keep_twist: bool, keep: int, check,
                  target: int) -> list[State]:
    """The device layers placed so their overlap holds ``target`` finger pairs, as compactly as possible (the device
    layers reaching the least far out along the fingers: short contact-gate teeth, a comb the middle hBN can hold)."""
    p = v.p
    devs = r.devices
    ref = next((d for d in devs if d.uid not in movable), devs[0])
    base = State({ref.uid: [tuple(q) for q in outlines[ref.uid]]})
    rest = [d for d in devs if d is not ref]
    fixed_rest = [d for d in rest if d.uid not in movable]
    for d in fixed_rest:
        base.outlines[d.uid] = [tuple(q) for q in outlines[d.uid]]
    moving = [d for d in rest if d.uid in movable]
    res = FINE_RES
    states = [base]
    for n_d, layer in enumerate(moving):
        last = n_d == len(moving) - 1
        new_states = []
        for st in states:
            region = None
            for d in devs:
                if d.uid in st.outlines:
                    rg = geo.region(st.outlines[d.uid])
                    region = rg if region is None else region & rg
            A = _polys(region.merged())
            if not A:
                continue
            placed_union = [st.outlines[d.uid] for d in devs if d.uid in st.outlines]
            src = [tuple(q) for q in outlines[layer.uid]]
            c = geo.centroid(src)
            found = []
            for t in ([0.0] if keep_twist else list(np.arange(0.0, 360.0, COARSE_STEP_DEG))):
                B = _turn(src, t, c)
                bb = _bbox([B])
                h, w = int(math.ceil((bb[3] - bb[1]) / res)) + 3, int(math.ceil((bb[2] - bb[0]) / res)) + 3
                fx0, fy0 = bb[0] - res, bb[1] - res
                ab = _bbox(A)
                X0, Y0 = ab[0] - w * res, ab[1] - h * res
                Wf, Hf = int(math.ceil((ab[2] - X0) / res)) + w + 1, int(math.ceil((ab[3] - Y0) / res)) + h + 1
                Am = _mask(A, X0, Y0, res, (Hf, Wf))
                Um = _mask(placed_union, X0, Y0, res, (Hf, Wf))
                Bm = _mask([B], fx0, fy0, res, (h, w))
                ov = cv2.filter2D(Am.astype(np.float32), cv2.CV_32F, Bm.astype(np.float32), anchor=(0, 0),
                                  borderType=cv2.BORDER_CONSTANT)[:Hf - h + 1, :Wf - w + 1]
                need_px = (target - 1) * p.pitch / res * (p.gap + 2 * p.min_overlap) / res if target else 0
                floor_ = max(p.width * (p.gap + 2 * p.min_overlap) / (res * res), need_px)
                step = max(1, int(round(SHIFT_STEP_UM / res)))

                def at(i, j, angles):
                    O = Am[i:i + h, j:j + w] & Bm
                    U = Um[i:i + h, j:j + w] | Bm
                    return _pairs(O, res, p, angles, target, U)
                ii, jj = np.nonzero(ov[::step, ::step] >= floor_)
                for i, j in zip(ii * step, jj * step):
                    d = at(i, j, np.arange(0.0, 180.0, PAIR_STEP_DEG))
                    if d["n"]:
                        found.append((d, t, int(i), int(j), (X0, Y0, fx0, fy0, h, w, Hf, Wf)))
                if check:
                    check()

            def key(item):
                d = item[0]
                return (d["n"] >= target, -d["reach"], d["slack"]) if target else (d["n"], d["slack"], 0)
            found.sort(key=key, reverse=True)
            picked = []
            for f in found:                             # distinct shifts
                if all(abs(f[2] - q[2]) + abs(f[3] - q[3]) > 3.0 / res or f[1] != q[1] for q in picked):
                    picked.append(f)
                if len(picked) >= 6:
                    break
            for d0, t, i0, j0, (X0, Y0, fx0, fy0, h, w, Hf, Wf) in picked:      # ±0.75 µm, 2° axis steps
                poll()
                B = _turn(src, t, c)
                Am = _mask(A, X0, Y0, res, (Hf, Wf))
                Um = _mask(placed_union, X0, Y0, res, (Hf, Wf))
                Bm = _mask([B], fx0, fy0, res, (h, w))
                top = (d0, i0, j0)
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        i, j = i0 + di, j0 + dj
                        if not (0 <= i <= Hf - h and 0 <= j <= Wf - w):
                            continue
                        O = Am[i:i + h, j:j + w] & Bm
                        U = Um[i:i + h, j:j + w] | Bm
                        d = _pairs(O, res, p, np.arange(d0["angle"] - 6.0, d0["angle"] + 6.01, 2.0), target, U)
                        if key((d,)) > key((top[0],)):
                            top = (d, i, j)
                d, i, j = top
                dx, dy = X0 + j * res - fx0, Y0 + i * res - fy0
                s2 = st.copy()
                s2.outlines[layer.uid] = _moved(B, dx, dy)
                s2.poses[layer.uid] = (math.remainder(t, 360.0), dx, dy)
                s2.score = -d["reach"]
                s2.notes = list(st.notes)
                s2.angle = d["angle"]
                new_states.append(s2)
                if check:
                    check()
        new_states.sort(key=lambda s_: -s_.score)
        states = new_states[: (2 * keep + 2) if last else 1]
    out = []
    for st in states:                                  # exact: the array on the twisted region, the teeth it needs
        poll()
        tw = _twisted(spec, st)
        if not tw:
            continue
        ang = st.angle
        fit = None
        for a in ([ang - 1.0, ang, ang + 1.0] if ang is not None else [None]):
            try:
                f = fg.fit_finger_array(tw, p, a, target)
            except ValueError:
                continue
            if fit is None or (f.n, f.margin_um) > (fit.n, fit.margin_um):
                fit = f
        if fit is None:
            continue
        st.arr = fit.array
        reach = _teeth_reach(spec, fit.array, st.outlines, v)
        st.score = 1000.0 * min(fit.n, target or fit.n) - 10.0 * reach + fit.margin_um
        st.notes.append(f"Device layers: the twisted region holds {fit.n} finger pairs (axis {fit.array.angle_deg:.1f}°); "
                        f"contact-gate teeth {reach:.1f} µm in all.")
        out.append(st)
        if check:
            check()
    out.sort(key=lambda s_: -s_.score)
    uniq = []
    for s_ in out:
        if all(any(math.dist(geo.centroid(s_.outlines[d.uid]), geo.centroid(u.outlines[d.uid])) > 1.0 for d in moving) or not moving
               for u in uniq):
            uniq.append(s_)
    return uniq[:keep]


def _twisted(spec, st: State) -> list[Poly]:
    return ef.twisted_region(spec, {u: [q] for u, q in st.outlines.items()})


def _fingers_at(arr: fg.FingerArray, length: float | None = None) -> list[Poly]:
    return [arr.finger(c, i, length).polygon for c in ("top", "bottom") for i in range(arr.n)]


def _generate(spec, st: State, config: LabConfig, key: str) -> tuple[ef.FlowStep, ef.Flow]:
    """The real Stage 2 generator on a scratch flow: the Hall bars on the twisted region (the array axis and pair count of
    the device stage), or — with the Hall bars of ``st`` as they are — the contact gate."""
    s, flow = _scratch(spec, st.outlines, copy=False)   # (the generators read the geometry from the flow)
    hb = flow.step("hall_bars")
    if key == "contact_gate" and st.arr is not None and st.arr.params.lengths:
        hb.array = st.arr.to_json()
        return ef.generate(s, flow, "contact_gate", config, None, run_check=False), flow
    hb.fixed_angle = st.arr.angle_deg if st.arr is not None else None
    hb.max_pairs = st.arr.n if st.arr is not None else None
    out = ef.generate(s, flow, "hall_bars", config, None, run_check=False)
    if key == "contact_gate":
        out = ef.generate(s, flow, "contact_gate", config, None, run_check=False)
    return out, flow


def _place_stage(st: State, layer, T: Targets, outlines: dict, movable: set, keep: int, check) -> list[State]:
    src = [tuple(q) for q in outlines[layer.uid]]
    if layer.uid not in movable:                      # fixed: kept where it is (reported when it breaks a rule)
        s2 = st.copy()
        s2.outlines[layer.uid] = src
        if not _exact_ok(src, T):
            s2.notes.append(f"{layer.label} is fixed and does not meet its placement rules here.")
            s2.score -= 1e5
        return [s2]
    out = []
    for score, pose, placed in search_flake(src, T, keep, check=check):
        s2 = st.copy()
        s2.outlines[layer.uid] = placed
        s2.poses[layer.uid] = pose
        s2.score += score * 1e-3
        out.append(s2)
    return out


def _stage_bbn(spec, r, st, v, outlines, movable, keep, check):
    arr, p = st.arr, v.p
    tw = _twisted(spec, st)
    T = Targets(must=tw + _fingers_at(arr, min(p.straight, p.length)), margin=p.bbn_margin + v.shift,
                want=[(_fingers_at(arr, p.length), 1.0)] + [([st.outlines[d.uid]], 2.0) for d in r.devices])
    out = []
    for s2 in _place_stage(st, r.bbn, T, outlines, movable, keep, check):
        try:
            hb, _flow = _generate(spec, s2, v.config, "hall_bars")
        except ValueError as exc:
            if r.bbn.uid in movable:
                s2.failed = f"Hall bars: {exc}"
                continue
            s2.notes.append(f"Hall bars do not fit on the fixed {r.bbn.label}: {exc}")   # fixed: reported, not dropped
            s2.score -= 1e5
            out.append(s2)
            continue
        s2.arr = fg.FingerArray.from_json(hb.array)
        s2.notes.append(f"{r.bbn.label}: fingers {min(s2.arr.params.lengths.values() or [0]):.1f}–"
                        f"{max(s2.arr.params.lengths.values() or [0]):.1f} µm long.")
        out.append(s2)
    return out


def _stage_bg(spec, r, st, v, outlines, movable, keep, check):
    arr = st.arr
    covers = [st.outlines[r.bbn.uid]]
    need = (v.contact + 2 * v.shift) ** 2

    def accept(poly):
        return _contact_spot(poly, covers, [], v) is not None
    T = Targets(must=[arr.channel()], margin=v.shift, want=[(_twisted(spec, st), 1.0)],
                blocked=_grown(covers, v.gclr + v.shift), expose_need=need, expose_cost=1.0, accept=accept)
    placed = _place_stage(st, r.bg, T, outlines, movable, keep, check)
    if r.tg is not None and r.bg.uid in movable:
        # also placements sticking out on one side only (along / across the fingers, each way): the top gate can then
        # stick out on the other side (user, 2026-11-21); the top-gate stage and the final ranking choose
        for k in range(4):
            a = math.radians(arr.angle_deg + 90.0 * k)
            hp = _half_planes(spec, r, st, v, (math.cos(a), math.sin(a)))
            if hp is None:
                break
            T1 = Targets(must=T.must, margin=T.margin, want=T.want, avoid=[([hp[1]], GATE_SIDE_WEIGHT)], blocked=T.blocked,
                         expose_need=T.expose_need, expose_cost=T.expose_cost, accept=accept)
            for s2 in _place_stage(st, r.bg, T1, outlines, movable, 1, check):
                p2 = s2.outlines[r.bg.uid]
                if all(geo.area(geo.region(p2) ^ geo.region(q.outlines[r.bg.uid])) > 4.0 for q in placed):
                    placed.append(s2)
    out = []
    for s2 in placed:
        spot = _contact_spot(s2.outlines[r.bg.uid], covers, [], v)
        if spot is None:
            s2.notes.append(f"{r.bg.label}: no free place for its contact outside {r.bbn.label}.")
            s2.score -= 1e4
        else:
            s2.zones["bg"] = spot
        out.append(s2)
    return out


def _comb_estimate(spec, st: State, v: Values, n_skip: int = 1, extension: float | None = None) -> list[Poly]:
    """The contact-gate comb the middle hBN must hold: as the generator would draw it without an hBN (teeth as short as
    the TMD keep-out allows), with ``n_skip`` ungated fingers per column and the bus ``extension`` µm past the outer
    tooth (the generator itself shortens the bus leads down to one electrode width and leaves more fingers ungated when
    the hBN is tight — the minimal comb is what must fit)."""
    arr, p = st.arr, v.p
    tmd = [st.outlines[d.uid] for d in v.roles.devices]
    keepout = fg.tmd_keepout(tmd, p.cg_tmd_gap)
    leave_sd = not ef.has_source_drain(spec)
    lay, _n = fg.fit_contact_gate(arr, None, None, leave_sd, keepout=keepout)
    ext = p.cg_bus_extension if extension is None else extension
    comb = fg.contact_gate_comb(arr, skip=lay["skip"], dirs=lay["dirs"], extension={c: ext for c in lay["extension"]},
                                n_skip=n_skip if leave_sd else 0, keepout=keepout)
    return [c["polygon"] for c in comb.values()]


def _teeth_estimate(spec, st: State, v: Values, n_skip: int) -> list[Poly]:
    """The contact gate's teeth: each gated finger's first ``tooth`` µm (as short as the TMD keep-out allows), with
    ``n_skip`` outer fingers per column left ungated. The bus behind them is the generator's to fit (it shortens and
    trims it to the hBN)."""
    arr, p = st.arr, v.p
    tmd = [st.outlines[d.uid] for d in v.roles.devices]
    leave_sd = not ef.has_source_drain(spec)
    lay, _n = fg.fit_contact_gate(arr, None, None, leave_sd, keepout=fg.tmd_keepout(tmd, p.cg_tmd_gap))
    out = []
    for c in ("top", "bottom"):
        skip = fg._skipped(arr, lay["skip"].get(c), n_skip if leave_sd else 0)
        t = float((lay.get("tooth") or {}).get(c) or p.cg_tooth)
        out += [arr.finger(c, i, t).polygon for i in range(arr.n) if i not in skip]
    return out


def _stage_mbn(spec, r, st, v, outlines, movable, keep, check):
    p = v.p
    tw = _twisted(spec, st)
    tmd = [st.outlines[d.uid] for d in r.devices]
    extra = (v.xu_margin - p.cg_hbn_margin) if v.xu_margin is not None else 0.0
    base = tw + ((_grown(tmd, extra) if extra > 0 else tmd) if v.xu_margin is not None else [])
    avoid = [(_landing_zones(st, v), 1.0)] + ([([st.zones["bg"]], 1.0)] if "bg" in st.zones else [])
    if r.tg is not None and r.tg.uid in st.outlines:
        # the contact-gate leads run to this hBN's far corners: a middle hBN reaching far along the top gate puts their
        # ends (the electrode landings) under it — mildly discouraged here, rejected after generating
        avoid.append(([st.outlines[r.tg.uid]], MBN_UNDER_TG_COST))
    full = _comb_estimate(spec, st, v)
    out = []
    for n_skip in range(1, min(fg.CG_MAX_SKIP, st.arr.n - 1) + 1):     # fewest ungated fingers first
        minimal = _teeth_estimate(spec, st, v, n_skip)
        T = Targets(must=base + minimal, margin=p.cg_hbn_margin + v.shift, avoid=avoid, want=[(full, 1.0), (_grown(full, 6.0), 0.2)])
        for s2 in _place_stage(st, r.mbn, T, outlines, movable, keep + 4, check):
            try:
                cg, _flow = _generate(spec, s2, v.config, "contact_gate")
            except ValueError as exc:
                if r.mbn.uid in movable:
                    s2.failed = f"Contact gate: {exc}"
                    continue
                s2.notes.append(f"The contact gate does not fit on the fixed {r.mbn.label}: {exc}")
                s2.score -= 1e5
                out.append(s2)
                return out
            s2.comb = [list(map(tuple, q)) for q in cg.shapes.get("contact_gates", [])]
            s2.reach = [tuple(q) for q in (cg.comb or {}).get("reach", {}).values()]
            if r.tg is not None and r.tg.uid in s2.outlines:
                # the top gate is placed already: the contact-gate electrodes must land off it (the hBN that insulates the
                # gate would cover the landing), and the gate's contact must stay clear of the comb
                tg_reg = geo.region(s2.outlines[r.tg.uid])
                if any(not (tg_reg & geo.region(z)).is_empty() for z in _cg_zones(s2, v)):
                    s2.failed = "Contact gate: its electrodes would land under the top gate"
                    continue
                spot = _contact_spot(s2.outlines[r.tg.uid], [], _conductors(spec, r, s2) + _twisted(spec, s2), v)
                if spot is None:
                    s2.failed = "Contact gate: it leaves the top gate no room for its contact"
                    continue
                s2.zones["tg"] = spot
            s2.score -= 50.0 * ((cg.comb or {}).get("n_skip", 1) - 1)
            s2.notes.append(f"{r.mbn.label}: contact gate fits ({(cg.comb or {}).get('n_skip', 1)} ungated finger(s) per column).")
            out.append(s2)
            if len(out) >= keep:
                return out
        if out or r.mbn.uid not in movable:
            break
    return out


def _exposed_flakes(spec, r: Roles, st: State) -> list[Poly]:
    """Every conducting flake placed so far (but the top gate) where no insulating flake placed above it covers it."""
    out = []
    cfg = r.config
    for i, layer in enumerate(spec.layers):
        if layer.uid not in st.outlines or cfg is None or not _conducting(cfg, layer) or layer is r.tg:
            continue
        exposed = geo.region(st.outlines[layer.uid])
        for above in spec.layers[i + 1:]:
            if above.uid in st.outlines and not _conducting(cfg, above):
                exposed = exposed - geo.region(st.outlines[above.uid])
        out += _polys(exposed.merged())
    return out


def _conductors(spec, r: Roles, st: State) -> list[Poly]:
    """The conductors a later flake meets: the fingers, the contact gate, the bottom gate's contact place and every
    conducting flake placed so far where no insulating flake above it (placed so far) covers it (exposed graphite)."""
    out = []
    if st.arr is not None:
        out += [f.polygon for f in st.arr.fingers()]
    out += st.comb
    if "bg" in st.zones:
        out.append(st.zones["bg"])
    return out + _exposed_flakes(spec, r, st)


def _tg_crossings(spec, r: Roles, st: State, v: Values) -> list[Poly]:
    """Where the top gate lies over a conductor (fingers, contact gate, the bottom gate's contact place), grown by the
    contact-to-graphite clearance + misalignment: what its hBN must cover."""
    if r.tg is None or r.tg.uid not in st.outlines:
        return []
    cond = _conductors(spec, r, st)
    if not cond:
        return []
    clr = v.graphite + v.shift
    near = geo.sized(geo.regions(cond).merged(), clr) if clr > 0 else geo.regions(cond).merged()
    return _polys((near & geo.region(st.outlines[r.tg.uid])).merged())


def _stage_tbn(spec, r, st, v, outlines, movable, keep, check):
    """The top-gate hBN: holds the twisted region and every place the top gate crosses a conductor (it insulates the gate
    from them), keeps off the electrode landings and the contact places; covering the comb's teeth is welcome."""
    tw = _twisted(spec, st)
    cross = _tg_crossings(spec, r, st, v)
    avoid = [(_landing_zones(st, v), 1.0), (_cg_zones(st, v), 1.0)] + [([z], 1.0) for z in st.zones.values()]
    T = Targets(must=tw + cross, margin=v.shift, avoid=avoid, want=[(_grown(tw, 4.0), 0.5)] + ([(st.comb, 0.5)] if st.comb else []))
    return _place_stage(st, r.tbn, T, outlines, movable, keep, check)


GATE_SIDE_WEIGHT = 0.25     # top gate over the bottom gate's side (× AVOID_WEIGHT per µm²): the gates stick out opposite
GATE_OTHER_SIDE_WANT = 0.05 # … and a little reward per µm² on the other side
GATE_SIDE_NOTE_UM2 = 25.0   # a top gate further than this into the bottom gate's side is noted (and ranked lower)


def _half_planes(spec, r: Roles, st: State, v: Values, d, extent_of=None):
    """(beyond the device toward ``d``, beyond it the other way): two boxes, each starting past the twisted region (+ the
    contact-to-graphite clearance + shift + 2 µm) and as large as the gates can reach (a larger area would only widen
    the search raster)."""
    tw = _twisted(spec, st)
    if not tw:
        return None
    c = np.asarray(geo.centroid(max(tw, key=lambda q: abs(geo.poly_area(q)))), float)
    d = np.asarray(d, float) / float(np.hypot(*d))
    n = np.array([-d[1], d[0]])
    pts = np.asarray([q for poly in tw for q in poly], float)
    reach = float(np.max(np.abs((pts - c) @ d))) + v.graphite + v.shift + 2.0
    L = 1.0
    for g in (r.tg, r.bg):
        if g is not None and (g.uid in st.outlines or g.outline):
            q = np.asarray(st.outlines.get(g.uid) or g.outline, float)
            L = max(L, float(np.ptp(q, axis=0).max()) + 1.0)

    def half(sign):
        a = c + sign * d * reach
        b = c + sign * d * (reach + L)
        return [tuple(a + n * L), tuple(b + n * L), tuple(b - n * L), tuple(a - n * L)]
    return half(1.0), half(-1.0)


def _bg_side(spec, r: Roles, st: State, v: Values):
    """Where the bottom gate sticks out past the device (user, 2026-11-21: the top and bottom gate graphite should stick out
    on opposite sides, so they cannot touch when stacking): (the half-plane beyond the device on the bottom gate's side,
    the half-plane on the other side), or None. The side = from the twisted region's middle toward the bottom gate's
    contact place (else its part outside the twisted region)."""
    tw = _twisted(spec, st)
    if r.bg is None or r.bg.uid not in st.outlines or not tw:
        return None
    c = np.asarray(geo.centroid(max(tw, key=lambda q: abs(geo.poly_area(q)))), float)
    if "bg" in st.zones:
        target = np.asarray(geo.centroid(st.zones["bg"]), float)
    else:
        rest = _polys((geo.region(st.outlines[r.bg.uid]) - geo.regions(tw)).merged())
        if not rest:
            return None
        target = np.asarray(geo.centroid(max(rest, key=lambda q: abs(geo.poly_area(q)))), float)
    d = target - c
    if float(np.hypot(*d)) < 1e-6:
        return None
    return _half_planes(spec, r, st, v, d)


def _stage_tg(spec, r, st, v, outlines, movable, keep, check):
    """The top gate (placed before its hBN, which then covers wherever it crosses a conductor): covers the twisted region,
    crosses as few conductors as it can, never lies over an electrode landing or the bottom gate's contact place, and
    leaves a place for its own contact (``gate_contact_clearance`` from the conductors and the device)."""
    tw = _twisted(spec, st)
    cond = _conductors(spec, r, st)
    clr = v.graphite + v.shift
    # never over an electrode landing, the bottom gate's contact or another gate's exposed graphite (its hBN could not
    # cover a long stretch of it: the two gates would short); crossing fingers / the comb is allowed (its hBN covers them
    # there) but kept short
    exposed = _exposed_flakes(spec, r, st)
    hard = _landing_zones(st, v) + _cg_zones(st, v) + ([st.zones["bg"]] if "bg" in st.zones else []) + \
        (_grown(exposed, clr) if clr > 0 else exposed)
    keepaway = cond + tw
    need = (v.contact + 2 * v.shift) ** 2

    def accept(poly):
        return _contact_spot(poly, [], keepaway, v) is not None
    avoid = [(_grown(cond, clr) if clr > 0 else cond, 0.5)]
    want = []
    side = _bg_side(spec, r, st, v)
    if side is not None:                             # stick out on the side away from the bottom gate's (if it can)
        avoid.append(([side[0]], GATE_SIDE_WEIGHT))
        want.append(([side[1]], GATE_OTHER_SIDE_WANT))
    T = Targets(must=tw, margin=v.shift, hard=hard, avoid=avoid, want=want,
                blocked=_grown(keepaway, v.gclr + v.shift), expose_need=need, accept=accept)
    out = []
    for s2 in _place_stage(st, r.tg, T, outlines, movable, keep, check):
        spot = _contact_spot(s2.outlines[r.tg.uid], [], keepaway, v)
        if side is not None:
            over = s2.side_over = geo.area(geo.region(s2.outlines[r.tg.uid]) & geo.region(side[0]))
            if over > GATE_SIDE_NOTE_UM2:
                s2.notes.append(f"{r.tg.label} also sticks out on {r.bg.label}'s side ({over:.0f} µm² there): no placement "
                                "keeping the gates on opposite sides met the rules.")
        if spot is not None:
            s2.zones["tg"] = spot
        else:
            s2.notes.append(f"{r.tg.label}: no free place for its contact.")
            s2.score -= 1e4
        out.append(s2)
    return out


def _stage_cover(spec, r, st, v, layer, outlines, movable, keep, check):
    tw = _twisted(spec, st)
    avoid = [(_landing_zones(st, v), 1.0), (_cg_zones(st, v), 1.0)]
    hard = list(st.zones.values())
    T = Targets(must=tw, margin=v.shift, avoid=avoid, hard=hard, want=[(_grown(tw, 4.0), 0.5)])
    return _place_stage(st, layer, T, outlines, movable, keep, check)


# ====================================================================== final evaluation
@dataclass
class Candidate:
    outlines: dict
    poses: dict
    pairs: int
    errors: list
    blocked: list
    notes: list
    score: tuple
    fingers: list = field(default_factory=list)
    comb: list = field(default_factory=list)
    zones: dict = field(default_factory=dict)

    def summary(self) -> str:
        bits = [f"{self.pairs} finger pairs"]
        bits.append("contact gate fits" if self.comb else "no contact gate")
        bits.append(f"{len(self.errors)} rule error(s)" if self.errors else "rules met")
        if self.blocked:
            bits.append(f"{len(self.blocked)} finger electrode(s) without a clear landing")
        return " · ".join(bits)


def _evaluate(spec, r: Roles, st: State, v: Values, config: LabConfig) -> Candidate:
    from .validation import validate_stage1
    s, flow = _scratch(spec, st.outlines)
    errors, blocked = [], []
    rep = validate_stage1(s, config)
    errors = [i.message for i in rep.errors]
    if st.arr is not None:
        el = flow.step("electrodes")
        try:
            land, _notes = ef.finger_landings(s, flow, el, st.arr, st.arr.params, config)
            p = st.arr.params
            blocked = [f.name for f in st.arr.fingers() if land.get(f.name, 0) >= max(p.length_at(f.column, f.index) - p.electrode_width, 0) - 1e-6]
        except (ValueError, MissingValues):
            pass
    pairs = st.arr.n if st.arr is not None else 0
    contacts = sum(1 for k in ("bg", "tg") if k in st.zones)
    side = -int(st.side_over // GATE_SIDE_NOTE_UM2)            # gates sticking out on opposite sides first (if possible)
    score = (not st.failed, -len(errors), min(pairs, v.target or pairs), -len(blocked), contacts, side, st.score)
    return Candidate(dict(st.outlines), dict(st.poses), pairs, errors, blocked, st.notes, score,
                     [f.polygon for f in st.arr.fingers()] if st.arr is not None else [], list(st.comb), dict(st.zones))


# ====================================================================== entry point
def auto_design(spec: DeviceSpec, config: LabConfig, flow: ef.Flow | None = None, movable: set | None = None,
                keep_twist: bool = True, thorough: bool = False, progress=None, cancelled=None,
                pairs: int | None = None, debug=None) -> list[Candidate]:
    """Placements of the movable flakes (turn + shift of each traced border), best first (at most 3). ``movable`` =
    layer uids the design may move (default: every flake with a traced border); ``progress(text, fraction)``;
    ``cancelled()`` → True stops (raises :class:`Cancelled`)."""
    t0 = time.time()
    r = roles(spec, config)
    v = _values(spec, config, r, flow)
    missing = [l.label for l in r.flakes if not (l.outline and len(l.outline) >= 3)]
    if missing:
        raise ValueError("Trace every flake first (Flake images): " + ", ".join(missing) + " has no traced border.")
    outlines = {l.uid: [tuple(map(float, q)) for q in l.outline] for l in r.flakes}
    if movable is None:
        movable = set(outlines)
    if not movable & set(outlines):
        raise ValueError("No flake may move: every flake is fixed (on the chip in Stage 2, or unticked). Tick the flakes the "
                         "design may place.")
    beams = dict(device=3, bbn=2, bg=1, mbn=2, rest=1) if thorough else dict(device=2, bbn=1, bg=1, mbn=1, rest=1)
    # the top gate before the middle hBN: each middle-hBN placement is then kept only if its contact-gate electrodes land
    # off the top gate (where the contact-gate leads end depends on the middle hBN's corners; the other order failed
    # far more often on the lab's flakes)
    plan = ["device", "bbn"] + (["bg"] if r.bg is not None else []) + (["tg"] if r.tg is not None else []) + \
        (["mbn"] if r.mbn is not None else []) + (["tbn"] if r.tbn is not None else []) + \
        [f"cap:{l.uid}" for l in r.caps] + [f"other:{l.uid}" for l in r.others]
    total = len(plan)

    def check():
        if cancelled is not None and cancelled():
            raise Cancelled()
    set_check(check)                                 # the generators' inner loops stop too (Abort)
    try:
        return _auto_design(spec, config, r, v, outlines, movable, keep_twist, beams, plan, total, check, progress,
                            pairs, debug, t0)
    finally:
        set_check(None)


def _auto_design(spec, config, r, v, outlines, movable, keep_twist, beams, plan, total, check, progress, pairs, debug, t0):

    def say(i, text):
        if progress is not None:
            progress(text, i / max(total, 1))
        check()
    say(0, "Device layers: the overlap with the most finger pairs …")
    target = v.target = pairs or v.cap_pairs
    states = _device_stage(spec, r, outlines, movable, v, keep_twist, beams["device"], check, target)
    if not states:
        raise ValueError(f"No overlap of the device layers holds {target} finger pairs." if target else
                         "No overlap of the device layers holds even one finger pair.")
    fails = []
    keys = {"bbn": "bbn", "bg": "bg", "mbn": "mbn"}

    def run(i, st, keep):
        stage = plan[i]
        name = stage.split(":")[0]
        if name == "bbn":
            say(i, f"{r.bbn.label}: around the fingers, then the Hall bars generated …")
            return _stage_bbn(spec, r, st, v, outlines, movable, keep, check)
        if name == "bg":
            say(i, f"{r.bg.label}: under the channel, with room for its contact …")
            return _stage_bg(spec, r, st, v, outlines, movable, keep, check)
        if name == "mbn":
            say(i, f"{r.mbn.label}: around the contact gate, then the contact gate generated …")
            return _stage_mbn(spec, r, st, v, outlines, movable, keep, check)
        if name == "tbn":
            say(i, f"{r.tbn.label}: over the device and wherever the top gate crosses metal, off the electrode landings …")
            return _stage_tbn(spec, r, st, v, outlines, movable, keep, check)
        if name == "tg":
            say(i, f"{r.tg.label}: over the device, off the electrode landings, with room for its contact …")
            return _stage_tg(spec, r, st, v, outlines, movable, keep, check)
        layer = spec.layer(stage.split(":")[1])
        say(i, f"{layer.label}: over the device …")
        return _stage_cover(spec, r, st, v, layer, outlines, movable, keep, check)

    # stage by stage; a stage that finds nothing sends the search back to the previous one, which then keeps three times
    # as many alternatives (each stage widened once) — e.g. a middle hBN whose contact-gate leads end where the top gate
    # must pass is replaced by another of its placements
    inputs = {1: states}
    widened = {}
    i = 1
    while i < len(plan):
        nxt = []
        keep = keys.get(plan[i].split(":")[0], "rest")
        stage_fails = []
        for st in inputs[i]:
            got = run(i, st, beams[keep] * widened.get(i, 1))
            if not got:
                stage_fails.append(st.failed or "no placement meets the rules")
            nxt += [g for g in got if not g.failed] or []
            stage_fails += [g.failed for g in got if g.failed]
        if debug is not None:
            debug(plan[i], len(inputs[i]), len(nxt), stage_fails)
        if not nxt:
            fails = stage_fails
            back = next((j for j in range(i - 1, 0, -1) if j not in widened), None)
            if back is None:
                what = plan[i].split(":")[0]
                label = {"bbn": r.bbn, "bg": r.bg, "mbn": r.mbn, "tbn": r.tbn, "tg": r.tg}.get(what) or spec.layer(plan[i].split(":")[-1])
                raise ValueError(f"No layout meets the rules at {label.label if label is not None else what}: "
                                 + "; ".join(sorted(set(fails))[:3]))
            widened[back] = 4
            i = back
            continue
        inputs[i + 1] = sorted(nxt, key=lambda s_: -s_.score)[: max(beams.values()) * 2 * max(widened.get(i, 1), 1)]
        i += 1
    states = inputs[len(plan)]
    say(total, "Checking the candidates (rules, electrode landings) …")
    cands = sorted((_evaluate(spec, r, st, v, config) for st in states), key=lambda c: c.score, reverse=True)
    for c in cands:
        c.notes.append(f"Designed in {time.time() - t0:.0f} s.")
    return cands[:3]


def apply_candidate(spec: DeviceSpec, cand: Candidate) -> list[str]:
    """Move each flake's traced border (and its image) to the candidate's place; returns the uids moved."""
    moved = []
    for uid, (deg, dx, dy) in cand.poses.items():
        layer = spec.layer(uid)
        if layer is None or not layer.outline:
            continue
        a = math.radians(deg)
        m = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        c = np.asarray(geo.centroid(layer.outline), float)
        t = c - m @ c + np.array([dx, dy])
        layer.outline = [[float(x), float(y)] for x, y in np.asarray(layer.outline, float) @ m.T + t]
        if (layer.image or {}).get("um_per_px"):
            layer.image = ef._image_after(layer.image, m, t)
        elif layer.image is not None:
            layer.image["placed"] = True
        moved.append(uid)
    return moved
