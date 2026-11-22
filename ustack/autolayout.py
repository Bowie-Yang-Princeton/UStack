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
from .cancel import current_check, poll, set_check
from .config import LabConfig, MissingItem, MissingValues
from .model import DeviceSpec

Poly = list[tuple[float, float]]

COARSE_RES = 0.5            # µm per raster pixel, first search (program choice)
FINE_RES = 0.25             # µm per raster pixel, refinement and the device layers
COARSE_STEP_DEG = 10.0      # turns tried in the first search
FINE_HALF_DEG = 5.0         # refinement: ± this around a coarse turn …
FINE_STEP_DEG = 1.0         # … in these steps
PAIR_STEP_DEG = 10.0        # array axis angles tried by the pair estimate (refined ±6° at 2°)
SHIFT_STEP_UM = 1.0         # device-layer shifts tried
AVOID_WEIGHT = 20.0         # 1 µm² over a kept-clear place costs as much as 20 µm² of wanted cover
NMS_UM = 2.0                # candidates of one turn at least this far apart
MBN_UNDER_TG_COST = 0.05    # middle hBN under the top gate (× AVOID_WEIGHT per µm²): keeps the leads' ends off the gate
MBN_NEAR_TG_COST = 0.3      # … near it beyond the device (the leads' landings): kept clear more firmly (2026-11-22)
MBN_OVER_BG_COST = 0.15     # middle hBN over the bottom gate beyond the finger array: the contact-gate leads (ending at its
                            # corners) would run along the bottom gate (user, 2026-11-22: a gate leak)
LEAD_ON_BG_COST = 2.0
TBN_ON_LEADS_COST = 0.1     # top hBN over the contact-gate bars beyond the array (× AVOID_WEIGHT per µm²): left exposed       # score per µm² of contact-gate lead over the bottom gate (the generated comb, beyond the array)
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
    locked: set = field(default_factory=set)   # layers whose turn is set by their twist (shift only)
    fixed_arr: fg.FingerArray | None = None    # fingers already made (or kept): not generated, the flakes placed round them
    fixed_comb: dict | None = None             # contact gate already made (or kept): {"polys", "reach"}


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
    """Per row of ``E``: the longest run of ones along x → [(length, row, start)] (vectorised)."""
    a = _runs_arrays(E)
    return list(zip(a[0].tolist(), a[1].tolist(), a[2].tolist())) if a is not None else []


def _runs_arrays(E: np.ndarray):
    """(lengths, rows, starts) of the longest run of ones in each row that has one, or None."""
    if not E.any():
        return None
    Z = np.zeros((E.shape[0], 1), np.int8)
    D = np.diff(np.hstack([Z, (E > 0).astype(np.int8), Z]), axis=1)
    rs, cs = np.nonzero(D == 1)                       # run starts and ends come in the same (row-major) order
    _re, ce = np.nonzero(D == -1)
    lengths = ce - cs
    order = np.lexsort((-lengths, rs))                # by row, longest first
    first = np.ones(order.size, bool)
    first[1:] = rs[order][1:] != rs[order][:-1]
    sel = order[first]
    return lengths[sel], rs[sel], cs[sel]


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
        ra = _runs_arrays(cv2.erode(R, kernel))
        if ra is None:
            continue
        lengths, rws, starts = ra
        ns = np.floor((lengths - 1) / pitch_px).astype(int) + 1
        if target:
            ns = np.minimum(ns, target)
        needs = np.ceil((ns - 1) * pitch_px).astype(int) + 1
        slacks = lengths - needs
        hit = np.nonzero(ns >= target)[0] if target else np.array([], int)
        if not target or hit.size == 0 or upad is None:
            # no run holds the target (or no target): the most pairs, then the most slack
            k = int(np.lexsort((-slacks, -ns))[0])
            d = {"n": int(ns[k]), "angle": float(a) % 180.0, "slack": int(slacks[k]), "reach": 0.0}
            if key(d) > key(best):
                best = d
            continue
        U = cv2.warpAffine(upad, M, (D, D), flags=cv2.INTER_NEAREST)
        occ = U.any(0)
        rows = np.arange(D)[:, None]
        colmax = np.where(occ, np.where(U > 0, rows, -1).max(0), -1)
        colmin = np.where(occ, np.where(U > 0, rows, D).min(0), D)
        for k in hit:                                  # only the runs that hold the target: how far the device reaches out
            row, start, need, slack = int(rws[k]), int(starts[k]), int(needs[k]), int(slacks[k])
            s0 = start + slack // 2                    # the array centred in the run
            cols = slice(s0, s0 + need)
            top = max(0.0, float(colmax[cols].max()) - (row + half_gap))
            bottom = max(0.0, (row - half_gap) - float(colmin[cols].min()))
            d = {"n": int(ns[k]), "angle": float(a) % 180.0, "slack": slack, "reach": (top + bottom) * res}
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
    locked: tuple = ()                                # layers kept at their turn (twist set): shift only
    accept_key: object = None                         # what ``accept`` depends on (hashable): lets identical searches be reused

    def key(self):
        """A hashable summary of the targets (None when ``accept`` has no key): identical searches are reused."""
        if self.accept is not None and self.accept_key is None:
            return None
        return (_pkey(self.must), round(self.margin, 6), tuple((_pkey(p), w) for p, w in self.want),
                tuple((_pkey(p), w) for p, w in self.avoid), _pkey(self.hard),
                _pkey(self.blocked) if self.blocked is not None else None, round(self.expose_need, 6),
                round(self.expose_cost, 6), self.accept_key)


def _pkey(polys) -> tuple:
    return tuple(tuple((round(float(x), 4), round(float(y), 4)) for x, y in poly) for poly in (polys or []))


#: Searches already done in this run (``search_flake`` / ``_aligned``), keyed by flake, targets and options: backtracking
#: re-runs stages on the same states (user, 2026-11-22: the auto design made faster). Cleared at the start of each run.
_MEMO: dict = {}
_MEMO_MAX = 4000
#: Helper threads for the correlation search: the turns of one scan run in parallel (numpy's FFT releases the GIL).
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2))), thread_name_prefix="autolayout")
    return _POOL


def _memo(key, compute):
    if key is None:
        return compute()
    if key in _MEMO:
        return _MEMO[key]
    out = compute()
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.clear()
    _MEMO[key] = out
    return out


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
    er = np.ones((3, 3), np.uint8)
    score_f = None                                    # want and avoid combine into one transform (both linear)
    check_fn = current_check()

    def one(a):
        if check_fn is not None:
            check_fn()
        F = _turn(flake, float(a), c)
        fb = _bbox([F])
        fx0, fy0 = fb[0] - res, fb[1] - res
        h, w = int(math.ceil((fb[3] - fb[1]) / res)) + 3, int(math.ceil((fb[2] - fb[0]) / res)) + 3
        if h >= H or w >= W:
            return []
        Ff = _mask([F], fx0, fy0, res, (h, w))
        Fe_polys = _grown([F], -T.margin) if T.margin > 0 else [F]
        Fe = _mask(Fe_polys, fx0, fy0, res, (h, w))
        FFf = np.conj(spec_of(np.pad(Ff, ((0, SH - h), (0, SW - w)))))
        FFe = None
        score = np.zeros((H - h + 1, W - w + 1), np.float32)
        ok = np.ones_like(score, bool)

        def corr(kf, FF):
            return np.fft.irfft2(kf * FF, s=(SH, SW))[:H - h + 1, :W - w + 1]
        if must_f is not None:
            if tolerant:
                FFe = np.conj(spec_of(np.pad(Fe, ((0, SH - h), (0, SW - w)))))
                out_px = n_must - corr(must_f, FFe)
                ok &= out_px < COARSE_SLACK * n_must + 0.5
                score -= 1e3 * np.maximum(out_px, 0.0) * res * res
            else:
                Fc = cv2.erode(Fe, er)                 # conservative: the exact check follows
                FFc = np.conj(spec_of(np.pad(Fc, ((0, SH - h), (0, SW - w)))))
                ok &= corr(must_f, FFc) > n_must - 0.5
            if not ok.any():
                return []
        if hard_f is not None:
            hit = corr(hard_f, FFf)
            if tolerant:
                ok &= hit < COARSE_SLACK * float(Ff.sum()) + 0.5
                score -= 1e3 * np.maximum(hit, 0.0) * res * res
            else:
                ok &= hit < 0.5
            if not ok.any():
                return []
        if want_f is not None or avoid_f is not None:
            spec = 0
            if want_f is not None:
                if FFe is None:
                    FFe = np.conj(spec_of(np.pad(Fe, ((0, SH - h), (0, SW - w)))))
                spec = want_f * FFe
            if avoid_f is not None:
                spec = spec - AVOID_WEIGHT * avoid_f * FFf
            score += corr(spec, 1.0) * res * res
        if free_f is not None:
            expo = corr(free_f, FFf) * res * res
            ok &= expo >= T.expose_need
            score += np.minimum(expo, 2 * T.expose_need) - T.expose_cost * np.maximum(expo - 2 * T.expose_need, 0.0)
        if not ok.any():
            return []
        score = np.where(ok, score, -np.inf)
        r = max(1, int(round(NMS_UM / res)))
        got = []
        for _ in range(per_angle):
            k = int(np.argmax(score))
            i, j = divmod(k, score.shape[1])
            s_ = float(score[i, j])
            if not np.isfinite(s_):
                break
            got.append((s_, float(a), X0 + j * res - fx0, Y0 + i * res - fy0))
            score[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = -np.inf
        return got
    angles = list(angles)
    poll()
    if len(angles) > 1:
        out = [g for part in _pool().map(one, angles) for g in part]
    else:
        out = [g for a in angles for g in one(a)]
    out.sort(key=lambda t: -t[0])
    return out


def _targets_cache(T: Targets) -> dict:
    c = T.__dict__.get("_cache")
    if c is None:
        c = {}
        if T.must:
            c["must"] = geo.regions(T.must).merged()
            c["must_box"] = _bbox(T.must)
        if T.hard:
            c["hard"] = geo.regions(T.hard).merged()
            c["hard_box"] = _bbox(T.hard)
        T.__dict__["_cache"] = c
    return c


def _exact_ok(placed: Poly, T: Targets) -> bool:
    c = _targets_cache(T)
    pts = np.asarray(placed, float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if T.must:                                        # cheap first: what it must hold lies inside its box (less the margin)
        mb = c["must_box"]
        m = T.margin - 1e-6
        if mb[0] < x0 + m or mb[1] < y0 + m or mb[2] > x1 - m or mb[3] > y1 - m:
            return False
    reg = geo.region(placed)
    if T.must:
        inner = geo.sized(reg, -T.margin) if T.margin > 0 else reg
        if not (c["must"] - inner).is_empty():
            return False
    if T.hard:
        hb = c["hard_box"]
        if not (hb[2] < x0 or hb[0] > x1 or hb[3] < y0 or hb[1] > y1) and not (reg & c["hard"]).is_empty():
            return False
    return T.accept is None or bool(T.accept(placed))


def search_flake(flake: Poly, T: Targets, keep: int = 1, angles=None, check=None) -> list[tuple[float, tuple, Poly]]:
    """The best ``keep`` placements of ``flake`` for ``T``: [(score, (angle, dx, dy) about its centroid, placed outline)].
    Identical searches in one run are done once (``_MEMO``)."""
    tk = T.key()
    key = None if tk is None else ("search", _pkey([flake]), tk, keep, tuple(angles) if angles is not None else None)
    return list(_memo(key, lambda: _search_flake(flake, T, keep, angles, check)))


def _search_flake(flake: Poly, T: Targets, keep: int = 1, angles=None, check=None) -> list[tuple[float, tuple, Poly]]:
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
    gate_angle: float | None = None                   # angle between where the two gates stick out (°)
    blockers: list = field(default_factory=list)      # flakes that cannot hold this device overlap at any turn

    def copy(self) -> "State":
        return State(dict(self.outlines), dict(self.poses), self.arr, list(self.comb), list(self.reach), dict(self.zones),
                     self.score, list(self.notes), self.failed, self.angle, self.gate_angle, list(self.blockers))


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


#: Each µm² of a flake above the bottom hBN lying beyond its edge costs this (× AVOID_WEIGHT): metal deposited over a flake
#: that steps down off the bottom hBN can break at the step edge (user, 2026-11-22: keep every layer on the bottom hBN if
#: possible). Program choice: about the weight of a µm² of wanted cover.
STEP_EDGE_COST = 0.25
#: Bottom hBN: each µm² of the area the middle / top hBN can reach round the device that it covers is worth this (room for
#: them on it; user, 2026-11-22). Program choice: a quarter of a µm² of finger.
ROOM_WANT = 0.25


def _off_bbn(r: "Roles", st: "State", reach: float) -> list[Poly]:
    """The area round the bottom hBN (out to ``reach`` beyond its box) that is not on it — where a flake above it would
    step down off its edge; [] before the bottom hBN is placed."""
    if r.bbn is None or r.bbn.uid not in st.outlines:
        return []
    b = geo.region(st.outlines[r.bbn.uid])
    x0, y0, x1, y1 = _bbox([st.outlines[r.bbn.uid]])
    box = geo.region(geo.rect(x0 - reach, y0 - reach, x1 + reach, y1 + reach))
    return _polys((box - b).merged())


def _half_size(poly) -> float:
    """Half the long side of a flake's smallest enclosing rectangle (how far it reaches from its middle, roughly)."""
    (_cx, _cy), (w, h), _a = cv2.minAreaRect(np.asarray(poly, np.float32))
    return float(max(w, h)) / 2.0


def _inside_bbn(r: "Roles", st: "State", reach: float, margin: float) -> list[Poly]:
    """Where a flake above the bottom hBN must not lie to stay ``margin`` inside it (the outside of the bottom hBN shrunk)."""
    if r.bbn is None or r.bbn.uid not in st.outlines:
        return []
    inner = geo.sized(geo.region(st.outlines[r.bbn.uid]), -margin) if margin > 0 else geo.region(st.outlines[r.bbn.uid])
    x0, y0, x1, y1 = _bbox([st.outlines[r.bbn.uid]])
    box = geo.region(geo.rect(x0 - reach, y0 - reach, x1 + reach, y1 + reach))
    return _polys((box - inner).merged())


def _reach_of(poly) -> float:
    c = geo.centroid(poly)
    return 2.0 * max(math.dist(c, q) for q in poly) + 2.0


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


#: A finger electrode needs at least this many electrode widths of the finger's far end free (the generator slides it out
#: past the flakes above, ``ebl_flow.finger_landings``); program choice. The full overlap is preferred, not required.
MIN_LANDING_WIDTHS = 2.0


def _landing_min(st: State, v: Values) -> list[Poly]:
    """The least each finger's electrode needs: the finger's last ``MIN_LANDING_WIDTHS`` electrode widths, grown by the
    clearance + misalignment — what no flake above the fingers may cover (the full overlap, ``_landing_zones``, is wanted)."""
    arr = st.arr
    if arr is None:
        return []
    from .router import wire
    p = arr.params
    out = []
    for f in arr.fingers():
        L = p.length_at(f.column, f.index)
        d = max(p.land_from_tip, L - MIN_LANDING_WIDTHS * p.electrode_width)
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
def _leave_sd(spec) -> bool:
    """Whether outer fingers are left ungated as source / drain (no longer: ``ebl_flow.GATE_SOURCE_DRAIN``)."""
    return not ef.GATE_SOURCE_DRAIN and not ef.has_source_drain(spec)


def _teeth_reach(spec, arr: fg.FingerArray, outlines: dict, v: Values) -> float:
    """How long the contact-gate teeth must be (both columns, µm) so the bus clears the device layers."""
    tmd = [outlines[d.uid] for d in v.roles.devices if d.uid in outlines]
    try:
        lay, _n = fg.fit_contact_gate(arr, None, None, _leave_sd(spec), keepout=fg.tmd_keepout(tmd, v.p.cg_tmd_gap))
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
                cands_ij = list(zip((ii * step).tolist(), (jj * step).tolist()))
                angles0 = np.arange(0.0, 180.0, PAIR_STEP_DEG)
                # the candidate shifts in parallel (OpenCV and numpy release the GIL)
                for (i, j), d in zip(cands_ij, _pool().map(lambda ij: at(ij[0], ij[1], angles0), cands_ij)):
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
    return uniq                                      # ranked by the caller against the flakes that must hold them


def _rank_device_states(spec, r, states: list, v, outlines: dict, movable: set, keep: int) -> list:
    """The device overlaps the top gate can cover and the middle hBN can hold (by their margins + the misalignment in
    use) first — those they cannot are tried last, and said (user, 2026-11-22: the overlap / the TMD union with the comb's
    teeth may come out larger than those flakes). Copies: the states are reused at another misalignment."""
    ok, rest = [], []
    for s0 in states:
        poll()
        s_ = s0.copy()
        s_.blockers = _cover_problems(spec, r, s_, v, outlines, movable)
        (rest if s_.blockers else ok).append(s_)
        if len(ok) >= keep:
            break
    for s_ in rest:
        s_.notes.extend(s_.blockers)
        s_.score -= 1e5
    return (ok + rest)[:keep]


#: Placements of each device layer combined on fixed fingers, the turns tried, and how many combinations are checked
#: against the top gate / middle hBN (program choices).
ON_ARRAY_KEEP = 4
ON_ARRAY_STEP_DEG = 5.0
ON_ARRAY_TESTS = 12


def _beyond_array(arr: fg.FingerArray | None):
    """Everything beyond the finger array's span along its axis (past the outer fingers): where the contact-gate leads
    run. None without an array."""
    if arr is None:
        return None
    hs = arr.span / 2 + arr.params.width
    far = 1000.0
    return (geo.region(arr.to_chip([(hs, -far), (far, -far), (far, far), (hs, far)]))
            + geo.region(arr.to_chip([(-far, -far), (-hs, -far), (-hs, far), (-far, far)]))).merged()


def _pair_strips(arr: fg.FingerArray) -> list[Poly]:
    """Each finger pair's tips and the gap between them, with the overlap the device layers need on each tip: what the
    twisted region must hold for the array to work (as ``fit_finger_array`` requires)."""
    p = arr.params
    h = p.gap / 2 + p.min_overlap
    return [arr.to_chip([(arr.u_of(i) - p.width / 2, -h), (arr.u_of(i) + p.width / 2, -h), (arr.u_of(i) + p.width / 2, h),
                         (arr.u_of(i) - p.width / 2, h)]) for i in range(arr.n)]


def _device_on_array(spec, r: Roles, outlines: dict, movable: set, v: Values, keep_twist: bool, keep: int, check,
                     arr: fg.FingerArray) -> list[State]:
    """The device layers placed on fingers already made (user, 2026-11-22: re-running the auto design on a partly made
    sample): each device layer must hold every pair's tip strip (``_pair_strips``) — then so does their overlap — reaching as little as it can along the fingers beyond (short contact-gate teeth). With the twist kept, the
    movable device layers turn together (their twist kept), each then only shifted."""
    strips = _pair_strips(arr)
    beyond = _polys((geo.regions(_fingers_at(arr)).merged() - geo.regions(strips).merged()).merged())
    T = Targets(must=strips, margin=0.0, avoid=[(beyond, 0.05)])      # as ``fit_finger_array`` asks of the twisted region
    devs = list(r.devices)
    fixed = [d for d in devs if d.uid not in movable]
    moving = [d for d in devs if d.uid in movable]
    base = State({d.uid: [tuple(q) for q in outlines[d.uid]] for d in fixed}, arr=arr, angle=arr.angle_deg)
    for d in fixed:
        if not _exact_ok(base.outlines[d.uid], T):
            base.notes.append(f"{d.label} is fixed and does not hold the fingers' tips.")
            base.score -= 1e5
    results = []

    import itertools

    def combos(turn):
        """Every combination of a few placements of each moving layer at this turn."""
        options = []
        for d in moving:
            poll()
            src = [tuple(q) for q in outlines[d.uid]]
            c = geo.centroid(src)
            got = search_flake(_turn(src, turn, c), T, ON_ARRAY_KEEP, angles=[0.0]) if keep_twist else \
                search_flake(src, T, ON_ARRAY_KEEP)
            if not got:
                return []
            options.append([(d, g) for g in got])
        out = []
        for combo in itertools.product(*options):
            st = base.copy()
            for d, (sc, (a, dx, dy), placed) in combo:
                st.outlines[d.uid] = placed
                st.poses[d.uid] = (math.remainder((turn if keep_twist else 0.0) + a, 360.0), dx, dy)
            tw = geo.regions(_twisted(spec, st)).merged()
            union = geo.regions([st.outlines[d.uid] for d in devs]).merged()
            st.score += 1e-3 * sum(g[0] for _d, g in combo)
            out.append((geo.area(tw), geo.area(union), st))
        return out
    pool = []
    if not moving:
        pool = [(0.0, 0.0, base)]
    else:
        for t in (np.arange(0.0, 360.0, ON_ARRAY_STEP_DEG) if keep_twist else [0.0]):
            if check:
                check()
            pool += combos(float(t))
    # the flakes that must hold the device decide: tried in two orders (smallest overlap — the top gate holds it; smallest
    # union — the middle hBN encloses it under the Xu rule), alternately, until enough pass
    by_tw = sorted(pool, key=lambda q: q[0])
    by_union = sorted(pool, key=lambda q: q[1])
    tried, results, failed = set(), [], []
    for k in range(min(len(pool), ON_ARRAY_TESTS)):
        for lst in (by_tw, by_union):
            if k >= len(lst) or id(lst[k][2]) in tried:
                continue
            st = lst[k][2]
            tried.add(id(st))
            poll()
            st.blockers = _cover_problems(spec, r, st, v, outlines, movable)
            (failed if st.blockers else results).append(st)
        if len(results) >= keep:
            break
    if not results:
        results = failed[:1]
    for st in results:
        st.notes.append(f"Device layers: placed on the fingers as made ({arr.n} pairs; each holds every pair's tips).")
        st.notes.extend(st.blockers)
        st.score -= 1e5 * bool(st.blockers)
    results.sort(key=lambda s_: -s_.score)
    return results[:keep]


def _cover_problems(spec, r: Roles, st: State, v: Values, outlines: dict, movable: set) -> list[str]:
    """Why no layout can follow this device overlap: the top gate cannot cover the twisted region by the misalignment, or
    the middle hBN cannot hold the twisted region (+ the TMD by ``xu.middle_hbn_edge_margin_um`` when that rule applies)
    by ``cg.hbn_margin_um`` + the misalignment, at any turn."""
    tw = _twisted(spec, st)
    out = []
    if not tw:
        return out
    if r.tg is not None and r.tg.uid in outlines:
        clr = v.graphite + v.shift
        keep_off = (_grown(_landing_min(st, v), clr) if clr > 0 else _landing_min(st, v)) if st.arr is not None else []
        if not _can_hold(outlines[r.tg.uid], tw, v.shift, r.tg.uid in movable, keep_off):
            out.append(f"{r.tg.label} cannot cover the twisted region by {v.shift:g} µm (misalignment) at any turn"
                       + (" while keeping off the finger electrodes' landings." if keep_off else "."))
    if r.mbn is not None and r.mbn.uid in outlines:
        p = v.p
        tmd = [st.outlines[d.uid] for d in r.devices]
        extra = (v.xu_margin - p.cg_hbn_margin) if v.xu_margin is not None else 0.0
        base = tw + ((_grown(tmd, extra) if extra > 0 else tmd) if v.xu_margin is not None else [])
        margin = p.cg_hbn_margin + v.shift
        if st.arr is not None and v.fixed_comb is None:     # and the contact gate's teeth (every finger has one)
            try:
                base = base + _teeth_estimate(spec, st, v, 0 if not _leave_sd(spec) else 1)
            except ValueError:
                pass
        if not _can_hold(outlines[r.mbn.uid], base, margin, r.mbn.uid in movable):
            box = _bbox(base)
            what = ("the device layers and the contact gate's teeth" if v.xu_margin is not None else
                    "the twisted region and the contact gate's teeth")
            out.append(f"{r.mbn.label} cannot hold {what} (about {box[2] - box[0]:.0f} × {box[3] - box[1]:.0f} µm) by "
                       f"{(v.xu_margin if v.xu_margin is not None else p.cg_hbn_margin) + v.shift:g} µm at any turn"
                       + (f" (xu.middle_hbn_edge_margin_um {v.xu_margin:g} + misalignment {v.shift:g})" if v.xu_margin is not None
                          else f" (cg.hbn_margin_um {p.cg_hbn_margin:g} + misalignment {v.shift:g})") + ".")
    return out


def _can_hold(flake: Poly, must: list[Poly], margin: float, movable: bool, hard=()) -> bool:
    T = Targets(must=must, margin=margin, hard=list(hard))
    if not movable:
        return _exact_ok([tuple(q) for q in flake], T)
    return bool(search_flake([tuple(q) for q in flake], T, 1))


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
    angles = [0.0] if layer.uid in getattr(T, "locked", ()) else None
    for score, pose, placed in search_flake(src, T, keep, angles=angles, check=check):
        s2 = st.copy()
        s2.outlines[layer.uid] = placed
        s2.poses[layer.uid] = pose
        s2.score += score * 1e-3
        out.append(s2)
    return out


def _stage_bbn(spec, r, st, v, outlines, movable, keep, check):
    arr, p = st.arr, v.p
    tw = _twisted(spec, st)
    if v.fixed_arr is not None:                       # the fingers are made (kept): the bottom hBN holds them, nothing generated
        T = Targets(must=tw + _fingers_at(arr), margin=p.bbn_margin + v.shift,
                    want=[([st.outlines[d.uid] for d in r.devices], 2.0)], locked=tuple(v.locked))
        out = _place_stage(st, r.bbn, T, outlines, movable, keep, check)
        for s2 in out:
            s2.notes.append(f"{r.bbn.label}: holds the fingers as made ({arr.n} pairs, kept).")
        return out
    # room round the device for the middle and top hBN (user, 2026-11-22: they ended at the bottom hBN's very edge; the
    # user's layout moves the bottom hBN so both lie well inside it): the area they can reach round the twisted region
    reach = max((_half_size(outlines[l.uid]) for l in (r.mbn, r.tbn) if l is not None and l.uid in outlines), default=0.0)
    room = [([q for q in _grown(tw, reach)], ROOM_WANT)] if reach > 0 and tw else []
    T = Targets(must=tw + _fingers_at(arr, min(p.straight, p.length)), margin=p.bbn_margin + v.shift,
                want=[(_fingers_at(arr, p.length), 1.0)] + [([st.outlines[d.uid]], 2.0) for d in r.devices] + room)
    # the device layers wholly on the bottom hBN if it can hold them (no step edge under them: user, 2026-11-22)
    T_all = Targets(must=T.must + [st.outlines[d.uid] for d in r.devices], margin=T.margin, want=T.want)
    placed = _place_stage(st, r.bbn, T_all, outlines, movable, keep, check) if r.bbn.uid in movable else []
    if not placed:
        placed = _place_stage(st, r.bbn, T, outlines, movable, keep, check)
    out = []
    for s2 in placed:
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
    """The bottom gate: under the channel (with the misalignment margin), a free contact place outside the bottom hBN.
    First (user, 2026-11-22) along the channel's long side — turned so its long axis lies along the finger array's axis and
    slid along it to one side (``_aligned``), so it sticks out one way and the top gate can stick out the other way;
    the turn-and-shift search only when neither side works."""
    arr = st.arr
    covers = [st.outlines[r.bbn.uid]]
    need = (v.contact + 2 * v.shift) ** 2

    def accept(poly):
        return _contact_spot(poly, covers, [], v) is not None
    T = Targets(must=[arr.channel()], margin=v.shift, want=[(_twisted(spec, st), 1.0)],
                blocked=_grown(covers, v.gclr + v.shift), expose_need=need, expose_cost=1.0, accept=accept,
                accept_key=("bg", _pkey(covers)),
                locked=tuple(v.locked))
    placed = []
    if r.bg.uid in movable and r.bg.uid not in v.locked:
        u = (math.cos(math.radians(arr.angle_deg)), math.sin(math.radians(arr.angle_deg)))
        centre = geo.centroid(arr.channel())
        for sign in (1.0, -1.0):
            for score, pose, poly in _aligned(outlines[r.bg.uid], (sign * u[0], sign * u[1]), centre, T,
                                              prefer=[_twisted(spec, st)], check=check)[:1]:
                s2 = st.copy()
                s2.outlines[r.bg.uid], s2.poses[r.bg.uid] = poly, pose
                s2.score += score * 1e-3
                placed.append(s2)
        placed = placed[:max(keep, 2)]
    if not placed:
        placed = _place_stage(st, r.bg, T, outlines, movable, keep, check)
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
    leave_sd = _leave_sd(spec)
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
    leave_sd = _leave_sd(spec)
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
        # … and its surroundings beyond the device by the landing radius + graphite clearance: the leads end at this hBN's
        # far corners, and a landing there would be under the gate or its hBN (user, 2026-11-22)
        rad = v.cg_overlap + v.clearance + 2 * v.shift + v.graphite
        near_tg = geo.sized(geo.region(st.outlines[r.tg.uid]), rad) - geo.sized(geo.regions(tw).merged(), 2.0)
        avoid.append((_polys(near_tg.merged()), MBN_NEAR_TG_COST))
    if v.fixed_comb is not None:                      # the contact gate is made (kept): the middle hBN holds it, nothing generated
        T = Targets(must=base + v.fixed_comb["polys"], margin=p.cg_hbn_margin + v.shift,
                    avoid=avoid + [(_off_bbn(r, st, _reach_of(outlines[r.mbn.uid])), STEP_EDGE_COST)], locked=tuple(v.locked))
        out, bad = [], []
        for s2 in _place_stage(st, r.mbn, T, outlines, movable, keep, check):
            s2.comb = [list(map(tuple, q)) for q in v.fixed_comb["polys"]]
            s2.reach = [tuple(q) for q in v.fixed_comb.get("reach", [])]
            if r.tg is not None and r.tg.uid in s2.outlines:
                spot = _contact_spot(s2.outlines[r.tg.uid], [], _conductors(spec, r, s2) + _twisted(spec, s2), v)
                if spot is None:
                    s2.failed = "Contact gate: it leaves the top gate no room for its contact"
                    bad.append(s2)
                    continue
                s2.zones["tg"] = spot
            s2.notes.append(f"{r.mbn.label}: holds the contact gate as made (kept).")
            out.append(s2)
        return out or bad[:3]
    lead_zone = _beyond_array(st.arr)
    bg_far = []
    if r.bg is not None and r.bg.uid in st.outlines and lead_zone is not None:
        bg_far = _polys((geo.region(st.outlines[r.bg.uid]) & lead_zone).merged())
        if bg_far:
            avoid = avoid + [(bg_far, MBN_OVER_BG_COST)]
    full = _comb_estimate(spec, st, v)
    out, bad = [], []
    skips = range(1, min(fg.CG_MAX_SKIP, st.arr.n - 1) + 1) if _leave_sd(spec) else (0,)   # every finger gated (2026-11-22)
    for n_skip in skips:                                             # fewest ungated fingers first
        minimal = _teeth_estimate(spec, st, v, n_skip)
        T = Targets(must=base + minimal, margin=p.cg_hbn_margin + v.shift,
                    avoid=avoid + [(_off_bbn(r, st, _reach_of(outlines[r.mbn.uid])), STEP_EDGE_COST)],
                    want=[(full, 1.0), (_grown(full, 6.0), 0.2)])
        # inside the bottom hBN (by the misalignment) when it can be: tried first (user, 2026-11-22)
        T_in = Targets(must=T.must, margin=T.margin, avoid=T.avoid, want=T.want,
                       hard=_inside_bbn(r, st, _reach_of(outlines[r.mbn.uid]), v.shift))
        placed = _place_stage(st, r.mbn, T_in, outlines, movable, keep + 4, check) if T_in.hard and r.mbn.uid in movable else []
        for s2 in placed or _place_stage(st, r.mbn, T, outlines, movable, keep + 4, check):
            try:
                cg, _flow = _generate(spec, s2, v.config, "contact_gate")
            except ValueError as exc:
                if r.mbn.uid in movable:
                    s2.failed = f"Contact gate: {exc}"
                    bad.append(s2)
                    continue
                s2.notes.append(f"The contact gate does not fit on the fixed {r.mbn.label}: {exc}")
                s2.score -= 1e5
                out.append(s2)
                return out
            s2.comb = [list(map(tuple, q)) for q in cg.shapes.get("contact_gates", [])]
            s2.reach = [tuple(q) for q in (cg.comb or {}).get("reach", {}).values()]
            n_cols = len(cg.shapes.get("contact_gates", []))
            if len(s2.reach) < n_cols:                    # every bus needs its lead and a landing (user, 2026-11-22)
                s2.failed = "Contact gate: a bus has no room for its lead on this middle hBN"
                bad.append(s2)
                continue
            if r.tg is not None and r.tg.uid in s2.outlines:
                # the top gate is placed already: the contact-gate electrodes must land off it (the hBN that insulates the
                # gate would cover the landing), and the gate's contact must stay clear of the comb
                tg_reg = geo.region(s2.outlines[r.tg.uid])
                clr = v.graphite + v.shift                   # its hBN covers that much round any crossing: off the landing too
                if any(not (tg_reg & geo.sized(geo.region(z), clr)).is_empty() for z in _cg_zones(s2, v)):
                    s2.failed = "Contact gate: its electrodes would land under the top gate"
                    bad.append(s2)
                    continue
                spot = _contact_spot(s2.outlines[r.tg.uid], [], _conductors(spec, r, s2) + _twisted(spec, s2), v)
                if spot is None:
                    s2.failed = "Contact gate: it leaves the top gate no room for its contact"
                    bad.append(s2)
                    continue
                s2.zones["tg"] = spot
            k = (cg.comb or {}).get("n_skip", 0)
            s2.score -= 50.0 * max(k - 1, 0)
            if bg_far:                                    # the leads along the bottom gate: kept short (a gate leak)
                s2.score -= LEAD_ON_BG_COST * geo.area(geo.regions(s2.comb).merged() & geo.regions(bg_far).merged())
            s2.notes.append(f"{r.mbn.label}: contact gate fits (" + (f"{k} ungated finger(s) per column)." if k else
                                                                      "every finger gated, source / drain too)."))
            out.append(s2)
            if len(out) >= keep:
                return out
        if out or r.mbn.uid not in movable:
            break
    return out or bad[:3]


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
    from them), keeps off the electrode landings and the contact places, and leaves the contact-gate bars beyond the
    array exposed where it can; inside the bottom hBN when it can be."""
    tw = _twisted(spec, st)
    cross = _tg_crossings(spec, r, st, v)
    # never over an electrode landing — the fingers' or the contact-gate leads' (user, 2026-11-22: a top hBN covering a bus
    # lead's end leaves no place for its electrode) — nor a gate's contact place
    hard = _landing_min(st, v) + _cg_zones(st, v) + list(st.zones.values())
    # the contact-gate bars beyond the finger array left exposed where it can (user, 2026-11-22: the top hBN covered a
    # bar; the user's layout turns it so the bars run out from under it) — no reward for covering the comb any more
    zone = _beyond_array(st.arr)
    leads = _polys((geo.regions(st.comb).merged() & zone).merged()) if st.comb and zone is not None else []
    avoid = [(_landing_zones(st, v), 1.0), (_off_bbn(r, st, _reach_of(outlines[r.tbn.uid])), STEP_EDGE_COST)] + \
        ([(_grown(leads, v.clearance + v.shift), TBN_ON_LEADS_COST)] if leads else [])
    T = Targets(must=tw + cross, margin=v.shift, hard=hard, avoid=avoid, want=[(_grown(tw, 4.0), 0.5)], locked=tuple(v.locked))
    T_in = Targets(must=T.must, margin=T.margin, avoid=T.avoid, want=T.want, locked=T.locked,
                   hard=hard + _inside_bbn(r, st, _reach_of(outlines[r.tbn.uid]), v.shift))
    if r.tbn.uid in movable:
        got = _place_stage(st, r.tbn, T_in, outlines, movable, keep, check)
        if got:
            return got
    return _place_stage(st, r.tbn, T, outlines, movable, keep, check)


GATE_MIN_APART_DEG = 120.0  # the two gates stick out at least this far apart (user, 2026-11-22; ideally 180°)
GATE_TURNS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0)   # top gate: opposite the bottom gate first, then turned toward 120°
ALIGN_STEP_UM = 0.5         # slide step of an aligned gate (fine) …
ALIGN_COARSE_UM = 2.0       # … and the first, coarse pass
ALIGN_DISTINCT_UM = 2.0     # alternatives of an aligned gate differ by at least this much (or by their turn)
ALIGN_OFFSETS_UM = (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0)   # … and its offsets across


def _long_axis_deg(poly) -> float:
    """Direction of a flake's long side (°): the long side of its smallest enclosing rectangle."""
    (_cx, _cy), (w, h), ang = cv2.minAreaRect(np.asarray(poly, np.float32))
    return float(ang if w >= h else ang + 90.0)


def _aligned(src: Poly, d, centre, T: Targets, prefer=(), check=None) -> list[tuple[float, tuple, Poly]]:
    """A gate turned so its long side lies along ``d`` and slid from ``centre`` toward ``d`` as far as ``T`` allows (it then
    sticks out that way only), with small offsets across: [(score, pose, placed)], the farthest first, one per offset /
    end. ``prefer`` (must-hold areas tried first, e.g. the twisted region) — without them only ``T.must``."""
    tk = T.key()
    key = None if tk is None else ("aligned", _pkey([src]), (round(float(d[0]), 6), round(float(d[1]), 6)),
                                   (round(float(centre[0]), 4), round(float(centre[1]), 4)), tk, tuple(_pkey(g) for g in prefer))
    return list(_memo(key, lambda: _aligned_run(src, d, centre, T, prefer, check)))


def _aligned_run(src: Poly, d, centre, T: Targets, prefer=(), check=None) -> list[tuple[float, tuple, Poly]]:
    d = np.asarray(d, float) / float(np.hypot(*d))
    n = np.array([-d[1], d[0]])
    c0 = np.asarray(geo.centroid(src), float)
    ctr = np.asarray(centre, float)
    axis = math.degrees(math.atan2(d[1], d[0]))
    la = _long_axis_deg(src)
    must_pts = np.asarray([q for poly in (T.must + [q for g in prefer for q in g]) for q in poly], float) \
        if (T.must or prefer) else ctr[None, :]
    lo = float(np.min((must_pts - ctr) @ d))
    out = []
    for extra in ([list(prefer)] if prefer else []) + [[]]:
        T2 = Targets(must=T.must + [q for g in extra for q in g], margin=T.margin, hard=T.hard, accept=T.accept,
                     accept_key=T.accept_key)
        for flip in (0.0, 180.0):
            turn = math.remainder(axis - la + flip, 360.0)
            B = np.asarray(_turn(src, turn, tuple(c0)), float)
            proj = (B - c0) @ d
            a_hi = -float(np.min(proj)) + lo - T.margin       # the gate's near end at the far edge of what it must hold
            for b_off in ALIGN_OFFSETS_UM:
                if check is not None:
                    check()
                poll()
                def at(a_):
                    t_ = ctr + a_ * d + b_off * n - c0
                    pl = [(float(x), float(y)) for x, y in B + t_]
                    return (t_, pl) if _exact_ok(pl, T2) else None
                a, hit = a_hi, None
                while a >= 0.0:                                  # a < 0 would stick out the other way; coarse steps …
                    hit = at(a)
                    if hit is not None:
                        break
                    a -= ALIGN_COARSE_UM
                if hit is not None:                              # … then back out in fine steps as far as it still fits
                    for a2 in np.arange(min(a + ALIGN_COARSE_UM - ALIGN_STEP_UM, a_hi), a + 1e-9, -ALIGN_STEP_UM):
                        h2 = at(float(a2))
                        if h2 is not None:
                            a, hit = float(a2), h2
                            break
                    t, placed = hit
                    out.append((a - abs(b_off) + (5.0 if extra else 0.0), (turn, float(t[0]), float(t[1])), placed))
            if out and extra:
                break
        if out:
            break
    out.sort(key=lambda g: -g[0])
    return out


def _stick_dir(gate: Poly, tw: list[Poly], clr: float):
    """Where a gate sticks out past the device: the unit vector from the twisted region's middle to the middle of the
    gate's part beyond it (+ ``clr``), or None when it does not stick out one way (centred, or not at all)."""
    if not tw:
        return None
    big = max(tw, key=lambda q: abs(geo.poly_area(q)))
    c = np.asarray(geo.centroid(big), float)
    rest = _polys((geo.region(gate) - geo.sized(geo.regions(tw).merged(), clr)).merged())
    if not rest:
        return None
    w = np.array([abs(geo.poly_area(q)) for q in rest])
    m = np.sum([wi * np.asarray(geo.centroid(q), float) for wi, q in zip(w, rest)], axis=0) / max(w.sum(), 1e-12)
    v = m - c
    L = float(np.hypot(*v))
    radius = max(math.dist(tuple(c), q) for q in big)
    return v / L if L > 0.25 * radius + 1.0 else None


def _apart_deg(a, b) -> float | None:
    if a is None or b is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def _stage_tg(spec, r, st, v, outlines, movable, keep, check):
    """The top gate (placed before its hBN, which then covers wherever it crosses a conductor): covers the twisted region,
    never lies over an electrode landing or the bottom gate's contact place, and leaves a place for its own contact
    (``gate_contact_clearance`` from the conductors and the device). It sticks out opposite the bottom gate (user,
    2026-11-22: at least ``GATE_MIN_APART_DEG`` apart, ideally 180°, long side along that way): turned along the way
    opposite the bottom gate (then ±30°, ±60°) and slid out (``_aligned``); the turn-and-shift search only when none fits
    (noted)."""
    tw = _twisted(spec, st)
    cond = _conductors(spec, r, st)
    clr = v.graphite + v.shift
    exposed = _exposed_flakes(spec, r, st)
    # the electrode landings and contact places kept clear by the graphite clearance + misalignment as well: wherever the
    # gate crosses metal its hBN must cover that much round the crossing, and that hBN must not reach a landing either
    # (user, 2026-11-22: a top hBN over a contact-gate lead's end)
    spots = _landing_min(st, v) + _cg_zones(st, v) + ([st.zones["bg"]] if "bg" in st.zones else [])
    hard = (_grown(spots, clr) if clr > 0 else spots) + (_grown(exposed, clr) if clr > 0 else exposed)
    keepaway = cond + tw
    need = (v.contact + 2 * v.shift) ** 2

    def accept(poly):
        return _contact_spot(poly, [], keepaway, v) is not None
    T = Targets(must=tw, margin=v.shift, hard=hard, avoid=[(_grown(cond, clr) if clr > 0 else cond, 0.5),
                                                          (_grown(_landing_zones(st, v), clr) if clr > 0 else _landing_zones(st, v), 1.0)],
                blocked=_grown(keepaway, v.gclr + v.shift), expose_need=need, accept=accept, locked=tuple(v.locked),
                accept_key=("tg", _pkey(keepaway)))
    bg_dir = _stick_dir(st.outlines[r.bg.uid], tw, clr) if r.bg is not None and r.bg.uid in st.outlines else None
    placed = []
    if r.tg.uid in movable and r.tg.uid not in v.locked and tw:
        centre = geo.centroid(max(tw, key=lambda q: abs(geo.poly_area(q))))
        if bg_dir is not None:
            base = -bg_dir
            ways = []
            for k in GATE_TURNS_DEG:
                a = math.radians(k)
                ways.append((base[0] * math.cos(a) - base[1] * math.sin(a), base[0] * math.sin(a) + base[1] * math.cos(a)))
        else:
            a = math.radians(st.arr.angle_deg) if st.arr is not None else 0.0
            ways = [(math.cos(a), math.sin(a)), (-math.cos(a), -math.sin(a))]
        # the best of each way first (opposite the bottom gate first), then — when more are asked for (a later stage
        # failed) — the other offsets / ends of those ways: real alternatives, not the same placements again
        per_way = []
        for d in ways:
            got = _aligned(outlines[r.tg.uid], d, centre, T, check=check)
            if got:
                per_way.append(got)
                if len(per_way) >= keep:
                    break
        picked = []
        for rank in range(max((len(g) for g in per_way), default=0)):
            for got in per_way:
                if rank < len(got) and all(math.dist(geo.centroid(got[rank][2]), geo.centroid(q[2])) > ALIGN_DISTINCT_UM
                                           or abs(math.remainder(got[rank][1][0] - q[1][0], 360.0)) > 1.0 for q in picked):
                    picked.append(got[rank])
                if len(picked) >= keep:
                    break
            if len(picked) >= keep:
                break
        for score, pose, poly in picked:
            s2 = st.copy()
            s2.outlines[r.tg.uid], s2.poses[r.tg.uid] = poly, pose
            s2.score += score * 1e-3
            placed.append(s2)
    if not placed:
        placed = _place_stage(st, r.tg, T, outlines, movable, keep, check)
    out = []
    for s2 in placed:
        spot = _contact_spot(s2.outlines[r.tg.uid], [], keepaway, v)
        if r.bg is not None and r.bg.uid in s2.outlines:
            apart = _apart_deg(bg_dir, _stick_dir(s2.outlines[r.tg.uid], tw, clr))
            s2.gate_angle = apart
            if apart is None or apart < GATE_MIN_APART_DEG - 1e-6:
                s2.notes.append(f"{r.tg.label} and {r.bg.label} do not stick out on opposite sides"
                                + (f" (only {apart:.0f}° apart)" if apart is not None else "")
                                + f": no placement {GATE_MIN_APART_DEG:.0f}° or more apart met the rules.")
            else:
                s2.notes.append(f"{r.tg.label} sticks out {apart:.0f}° away from {r.bg.label}.")
        if spot is not None:
            s2.zones["tg"] = spot
        else:
            s2.notes.append(f"{r.tg.label}: no free place for its contact.")
            s2.score -= 1e4
        out.append(s2)
    return out


def _stage_cover(spec, r, st, v, layer, outlines, movable, keep, check):
    tw = _twisted(spec, st)
    hard = _landing_min(st, v) + _cg_zones(st, v) + list(st.zones.values())        # electrode landings and contacts stay free
    T = Targets(must=tw, margin=v.shift, hard=hard, want=[(_grown(tw, 4.0), 0.5)], locked=tuple(v.locked),
                avoid=[(_landing_zones(st, v), 1.0), (_off_bbn(r, st, _reach_of(outlines[layer.uid])), STEP_EDGE_COST)])
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
    gate_angle: float | None = None                   # how far apart the two gates stick out (°)
    axis: float | None = None                         # the finger array's axis (°)
    array: dict | None = None                         # the finger array (FingerArray.to_json, Flake-layout frame)
    reach: list = field(default_factory=list)         # contact-gate lead ends

    def to_json(self) -> dict:
        P = lambda poly: [[float(x), float(y)] for x, y in poly]     # noqa: E731
        return {"outlines": {u: P(q) for u, q in self.outlines.items()}, "poses": {u: list(map(float, p)) for u, p in self.poses.items()},
                "pairs": int(self.pairs), "errors": list(self.errors), "blocked": list(self.blocked), "notes": list(self.notes),
                "score": [float(x) if isinstance(x, (int, float)) else x for x in self.score],
                "fingers": [P(q) for q in self.fingers], "comb": [P(q) for q in self.comb],
                "zones": {k: P(q) for k, q in self.zones.items()}, "gate_angle": self.gate_angle, "axis": self.axis,
                "array": self.array, "reach": [list(map(float, q)) for q in self.reach]}

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        T = lambda poly: [tuple(q) for q in poly]                      # noqa: E731
        return cls({u: T(q) for u, q in d["outlines"].items()}, {u: tuple(p) for u, p in d.get("poses", {}).items()},
                   int(d.get("pairs", 0)), list(d.get("errors", [])), list(d.get("blocked", [])), list(d.get("notes", [])),
                   tuple(d.get("score", ())), [T(q) for q in d.get("fingers", [])], [T(q) for q in d.get("comb", [])],
                   {k: T(q) for k, q in d.get("zones", {}).items()}, d.get("gate_angle"), d.get("axis"), d.get("array"),
                   [tuple(q) for q in d.get("reach", [])])

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
    if st.comb and len(st.reach) < len(st.comb):
        blocked.append("a contact-gate bus without its lead")
    if st.reach:                                     # contact-gate electrodes: their landings under no flake above the comb
        above = [st.outlines[l.uid] for l in ([r.tbn] if r.tbn is not None else []) + list(r.caps) if l.uid in st.outlines]
        if above:
            cover = geo.regions(above).merged()
            for k, z in enumerate(_cg_zones(st, v)):
                core = geo.sized(geo.region(z), -(v.clearance + v.shift))
                if not (core & cover).is_empty():
                    blocked.append(f"contact-gate lead {k + 1}")
    if r.bbn is not None and r.bbn.uid in st.outlines:          # flakes above the bottom hBN stepping off its edge
        bbn_reg = geo.region(st.outlines[r.bbn.uid])
        above = spec.layers[spec.layers.index(r.bbn) + 1:]
        off = [(l.label, geo.area(geo.region(st.outlines[l.uid]) - bbn_reg)) for l in above if l.uid in st.outlines]
        off = [(n, a) for n, a in off if a > 1.0]
        gates = {l.label for l in (r.tg, r.bg) if l is not None}
        st.off_bbn = sum(a for n, a in off if n not in gates)     # gates must stick out; the others ranked by it
        if off:
            st.notes.append("Off the bottom hBN (a step edge under any metal there): "
                            + ", ".join(f"{n} {a:.0f} µm²" for n, a in off) + ".")
    pairs = st.arr.n if st.arr is not None else 0
    contacts = sum(1 for k in ("bg", "tg") if k in st.zones)
    side = 0 if (st.gate_angle is None and (r.bg is None or r.tg is None)) or \
        (st.gate_angle is not None and st.gate_angle >= GATE_MIN_APART_DEG - 1e-6) else -1   # gates opposite first
    score = (not st.failed, -len(errors), min(pairs, v.target or pairs), -len(blocked), contacts, side,
             -int(getattr(st, "off_bbn", 0.0) // 25.0), st.score)
    return Candidate(dict(st.outlines), dict(st.poses), pairs, errors, blocked, st.notes, score,
                     [f.polygon for f in st.arr.fingers()] if st.arr is not None else [], list(st.comb), dict(st.zones),
                     st.gate_angle, st.arr.angle_deg if st.arr is not None else None,
                     st.arr.to_json() if st.arr is not None else None, [list(q) for q in st.reach])


# ====================================================================== entry point
def auto_design(spec: DeviceSpec, config: LabConfig, flow: ef.Flow | None = None, movable: set | None = None,
                keep_twist: bool = True, thorough: bool = False, progress=None, cancelled=None,
                pairs: int | None = None, debug=None, fixed: dict | None = None) -> list[Candidate]:
    """Placements of the movable flakes (turn + shift of each traced border), best first (at most 3). ``movable`` =
    layer uids the design may move (default: every flake with a traced border); ``progress(text, fraction)``;
    ``cancelled()`` → True stops (raises :class:`Cancelled`)."""
    t0 = time.time()
    r = roles(spec, config)
    v = _values(spec, config, r, flow)
    v.locked = {u for u in ef.twist_layer_uids(spec) if u not in ef.device_layer_uids(spec)}
    if movable is None:
        movable = {l.uid for l in r.flakes}
    # metal already made (or kept from an earlier design), in the Flake-layout frame (``made_patterns``)
    fixed = fixed or {}
    if fixed.get("hall_bars") is not None:
        a = fixed["hall_bars"]
        v.fixed_arr = a if isinstance(a, fg.FingerArray) else fg.FingerArray.from_json(a)
    if fixed.get("contact_gate") is not None:
        if v.fixed_arr is None:
            raise ValueError("The contact gate can be kept only with the fingers it lies on: keep the Hall-bar fingers too.")
        v.fixed_comb = {"polys": [[tuple(map(float, q)) for q in poly] for poly in fixed["contact_gate"]["polys"]],
                        "reach": [tuple(map(float, q)) for q in fixed["contact_gate"].get("reach", [])]}
        # the comb is written on the middle hBN, which lies over the device layers: those are in place with it
        movable = set(movable) - set(comb_holders(r))
    missing = [l.label for l in r.flakes if not (l.outline and len(l.outline) >= 3)]
    if missing:
        raise ValueError("Trace every flake first (Flake images): " + ", ".join(missing) + " has no traced border.")
    # the device layers start as they lie in their pictures, each turned by its twist (user, 2026-11-22): the lowest one
    # at its picture's angle when it may move (else where it is); the others follow it by the twists
    # (lattice directions from each image's reference angle; flakes with a twist to a device layer follow it, turn kept)
    s0 = spec.copy()
    devs = [d.uid for d in r.devices if d.outline]
    tied = set(ef.twist_layer_uids(s0))
    pre = ef.apply_device_twist(s0, ref_image_angle=0.0 if devs and devs[0] in movable else None,
                                skip={u for u in tied if u not in movable})
    outlines = {l.uid: [tuple(map(float, q)) for q in s0.layer(l.uid).outline] for l in r.flakes}
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
    _MEMO.clear()
    try:
        cands = _auto_design(spec, config, r, v, outlines, movable, keep_twist, beams, plan, total, check, progress,
                             pairs, debug, t0)
    finally:
        set_check(None)
    for c in cands:                                  # poses from the traced borders as they are (the twist turn first)
        for u, d in pre.items():
            deg, dx, dy = c.poses.get(u, (0.0, 0.0, 0.0))
            c.poses[u] = (deg + d, dx, dy)
    return cands


def _auto_design(spec, config, r, v, outlines, movable, keep_twist, beams, plan, total, check, progress, pairs, debug, t0):

    def say(i, text):
        if progress is not None:
            progress(text, i / max(total, 1))
        check()
    if v.fixed_arr is not None:
        say(0, "Device layers: on the fingers as made …")
        v.target = v.fixed_arr.n
        pool = _device_on_array(spec, r, outlines, movable, v, keep_twist, beams["device"] * 3, check, v.fixed_arr)
        if not pool:
            raise ValueError(f"The device layers cannot hold the tips of the {v.fixed_arr.n} finger pairs already made "
                             "at any place (each must cover every pair's tips and the gap between them).")
    else:
        say(0, "Device layers: the overlap with the most finger pairs …")
        if pairs:
            v.target = pairs
            pool = _device_stage(spec, r, outlines, movable, v, keep_twist, beams["device"], check, pairs) or \
                _device_stage(spec, r, outlines, movable, v, keep_twist, 1, check, None)
            got = max((s_.arr.n for s_ in pool if s_.arr is not None), default=0)
            if got < pairs:
                raise ValueError(f"No overlap of the device layers holds {pairs} finger pairs"
                                 + (f" (at most {got} fit with these flakes)." if got else "."))
        else:
            # as many pairs as the flakes allow, the bond pads' count an upper limit; then the most compact overlap for it
            cap = v.cap_pairs
            pool = _device_stage(spec, r, outlines, movable, v, keep_twist, beams["device"], check, cap)
            if not pool and cap:                      # nothing near the cap: as many as fit
                pool = _device_stage(spec, r, outlines, movable, v, keep_twist, beams["device"], check, None)
            most = max((s_.arr.n for s_ in pool if s_.arr is not None), default=0)
            v.target = min(most, cap) if cap else most
            if most and (cap is None or most < cap):
                say(0, f"Device layers: {most} finger pairs at most — the most compact overlap for them …")
                pool = _device_stage(spec, r, outlines, movable, v, keep_twist, beams["device"], check, most) or pool
                if cap:
                    for s_ in pool:
                        s_.notes.append(f"The bond pads take {cap} finger pairs; these device layers hold at most {most}.")
        if not pool:
            raise ValueError("No overlap of the device layers holds even one finger pair.")
    # the stacking misalignment is added to every margin; when no layout keeps all of it, the run is repeated with half,
    # then none (user, 2026-11-22: the lab's own design on these flakes keeps the lab's margins, not the full misalignment
    # on top of each) — each candidate says which was used
    full = v.shift
    levels = [full] + ([round(full / 2, 3), 0.0] if full > 0 else [])
    reasons = []
    for n_lvl, lvl in enumerate(levels):
        v.shift = lvl
        # the backtracking an attempt may spend before the next, smaller margin is tried (the last attempt: as much as
        # it needs) — a margin that cannot work is given up quickly (user, 2026-11-22: keep the run time)
        budget = None if n_lvl == len(levels) - 1 else n_lvl + 1
        states = _rank_device_states(spec, r, pool, v, outlines, movable, beams["device"])
        if all(s_.blockers for s_ in states):
            reasons.append(f"misalignment {lvl:g} µm: " + " ".join(states[0].blockers))
            continue
        try:
            cands = _later_stages(spec, config, r, v, outlines, movable, beams, plan, total, check, say, debug, t0, states,
                                  budget)
        except ValueError as exc:
            reasons.append(f"misalignment {lvl:g} µm: {exc}")
            continue
        if lvl < full:
            for c in cands:
                c.notes.insert(0, f"Misalignment margin {lvl:g} µm of the lab's {full:g} µm (stack.alignment_shift_um): with "
                               "the full margin on every flake no layout exists.")
                c.score = c.score[:1] + (-levels.index(lvl),) + c.score[1:]
        return cands
    twist = [f"{d.label} {d.twist_deg.value:g}°" for d in r.devices if d.twist_deg.value]
    raise ValueError("No layout: " + " · ".join(reasons[:3]) + (f" (twist: {', '.join(twist)})" if twist else "")
                     + " Use larger flakes, another twist, or lower the margins in Rules / Preset dimensions.")


def _later_stages(spec, config, r, v, outlines, movable, beams, plan, total, check, say, debug, t0, states, budget=None):
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
            if budget is not None and len(widened) >= budget:
                back = None                            # this attempt's backtracking is spent
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


def comb_holders(r: Roles) -> list[str]:
    """The flakes a contact gate kept as it is needs where they are: the middle hBN it is written on and the device layers
    under it."""
    return ([r.mbn.uid] if r.mbn is not None else []) + [d.uid for d in r.devices]


def made_patterns(spec: DeviceSpec, flow: ef.Flow | None) -> dict:
    """Metal the auto design can keep as it is (user, 2026-11-22: re-running it on a partly made sample, or fixing parts
    by hand), in the Flake-layout frame: {"hall_bars": {"array", "source", "made", "text"}, "contact_gate": {"polys",
    "reach", "source", "made", "text"}}. Source "stage2" = generated in Stage 2 (as found written when located; mapped
    through the frame the flakes on the chip set), else "plan" = the auto design applied last. "made" = a later Stage 2
    step already has images or found flakes (the pattern is on the chip): the window ticks it by default."""
    out: dict = {}
    if flow is not None:
        view = ef.layout_view(spec, flow)
        frame = ef._pattern_frame(spec, flow, view)
        order = flow.order
        last = order[-1] if order else None

        def later_done(key):
            if key not in order:
                return False
            for k in order[order.index(key) + 1:]:
                st_ = flow.steps.get(k)
                if st_ is not None and (st_.images or any(ef.located_here(st_, u) for u in st_.outlines)
                                        or key in (st_.found_patterns or {})):
                    return True
            return False
        if frame is not None and np.linalg.det(frame[0]) > 0:
            m, t = frame
            rot = math.degrees(math.atan2(m[1, 0], m[0, 0]))
            arr = ef._array_of(flow, last)
            if arr is not None:
                (cx, cy), = ef.to_layout(frame, [arr.center])
                out["hall_bars"] = {"array": fg.FingerArray((cx, cy), arr.angle_deg - rot, arr.n, arr.params), "source": "stage2",
                                    "made": later_done("hall_bars"),
                                    "text": f"Hall-bar fingers from Stage 2 ({arr.n} pairs)"}
            cg = flow.steps.get("contact_gate")
            polys = ef.pattern_shapes(flow, last, "contact_gate").get("contact_gates") if cg is not None else None
            if polys and "hall_bars" in out:
                reach = [tuple(q) for q in ((cg.comb or {}).get("reach") or {}).values()]
                off = ef.pattern_offset(flow, last, "contact_gate")
                if off and reach:
                    reach = ef.move_points(reach, off)
                out["contact_gate"] = {"polys": [ef.to_layout(frame, q) for q in polys],
                                       "reach": ef.to_layout(frame, reach) if reach else [], "source": "stage2",
                                       "made": later_done("contact_gate"), "text": "Contact gate from Stage 2"}
    plan = ef.auto_plan(spec)
    if plan and plan.get("array") and "hall_bars" not in out:
        arr = fg.FingerArray.from_json(plan["array"])
        out["hall_bars"] = {"array": arr, "source": "plan", "made": False,
                            "text": f"Hall-bar fingers as the auto design planned them ({arr.n} pairs)"}
        if plan.get("comb") and "contact_gate" not in out:
            out["contact_gate"] = {"polys": [[tuple(q) for q in poly] for poly in plan["comb"]],
                                   "reach": [tuple(q) for q in plan.get("reach") or []], "source": "plan", "made": False,
                                   "text": "Contact gate as the auto design planned it"}
    return out


def apply_candidate(spec: DeviceSpec, cand: Candidate, cands: list | None = None) -> list[str]:
    """Put each flake's traced border (and its image) where the candidate has it and keep its planned metal (fingers,
    contact gate, gate-contact places; Flake-layout frame) in ``spec.stage2["auto_design"]`` — shown in Flake layout, the
    Structure tab and the auto design window until Stage 2 generates those patterns (``ebl_flow.layout_patterns``).
    ``cands`` (the run's layouts) are kept with it, so the window offers them again (user, 2026-11-22). The borders go to
    the candidate's outlines as they are (the rigid move from where each border is now), so any layout of the list can be
    applied after another. Returns the uids moved."""
    keep = list(cands) if cands else [cand]
    spec.stage2[ef.AUTO_PLAN_KEY] = {
        "fingers": [[list(map(float, q)) for q in poly] for poly in cand.fingers],
        "comb": [[list(map(float, q)) for q in poly] for poly in cand.comb],
        "zones": {k: [list(map(float, q)) for q in poly] for k, poly in cand.zones.items()},
        "outlines": {u: [list(map(float, q)) for q in poly] for u, poly in cand.outlines.items()},
        "pairs": int(cand.pairs), "summary": cand.summary(), "notes": list(cand.notes),
        "array": cand.array, "reach": [list(map(float, q)) for q in cand.reach],
        "candidates": [c.to_json() for c in keep], "applied": next((k for k, c in enumerate(keep) if c is cand), 0)}
    moved = []
    for uid, target in cand.outlines.items():
        layer = spec.layer(uid)
        if layer is None or not layer.outline:
            continue
        cur = [tuple(map(float, q)) for q in layer.outline]
        tgt = [tuple(map(float, q)) for q in target]
        if len(cur) == len(tgt) and np.allclose(np.asarray(cur), np.asarray(tgt), atol=1e-9):
            continue
        if len(cur) == len(tgt):
            m, t = ef._rigid(cur, tgt)
        else:                                          # the border was reshaped since: moved by its middle only
            c0, c1 = np.asarray(geo.centroid(cur), float), np.asarray(geo.centroid(tgt), float)
            m, t = np.eye(2), c1 - c0
            tgt = [tuple(map(float, q)) for q in np.asarray(cur) + t]
        layer.outline = [[float(x), float(y)] for x, y in tgt]
        if (layer.image or {}).get("um_per_px"):
            layer.image = ef._image_after(layer.image, m, t)
        elif layer.image is not None:
            layer.image["placed"] = True
        moved.append(uid)
    return moved


def saved_candidates(spec: DeviceSpec) -> tuple[list, int]:
    """The layouts of the run applied last (``apply_candidate``), and which of them was applied."""
    plan = ef.auto_plan(spec) or {}
    out = []
    for d in plan.get("candidates") or []:
        try:
            out.append(Candidate.from_json(d))
        except (KeyError, TypeError, ValueError):
            continue
    return out, int(plan.get("applied", 0) or 0)
