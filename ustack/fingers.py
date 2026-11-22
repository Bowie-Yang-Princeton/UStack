"""Finger-array pattern: the lab's two facing columns of identical fingers, sized to the twisted
region, plus the contact-gate comb, gate contacts and the electrodes to the bond pads.

Local frame of an array: ``u`` runs along the gap between the two columns (the array axis), ``v``
across it; the top column's tips sit at ``v = +gap/2`` and point to −v, the bottom column's tips
at ``v = −gap/2``. Chip µm = R(angle)·(u, v) + centre.

Finger shape (all numbers are presets in ``config/presets.json``, group *Finger array*):
straight for ``fingers.straight_um`` from the tip with ``fingers.width_um``, then a straight
fan-out piece ``fingers.far_width_um`` wide, ending in a stub as long as it is wide that runs along
v (square ends; the far ends of a column lie on one line). The fan angle is the one that puts the centres at ``u · far_pitch / pitch``
at ``fingers.length_um``; a longer finger (each column is fitted to the bottom hBN,
:func:`fit_finger_lengths`) keeps that angle, so its end lies further out.

**How many fingers** (:func:`fit_finger_array`): a finger *pair* at position ``u`` counts when the
strip ``[u ± width/2] × [±(gap/2 + min_overlap)]`` — the channel between the two tips and
``fingers.min_overlap_um`` of each finger — lies inside the twisted region. The region is
rasterised in the local frame (step = pitch / an integer ≈ 0.05 µm), eroded by that strip, and for
every angle (2° steps, refined to 0.25°), every phase of the pitch grid and every gap position the
longest run of consecutive valid positions is the finger count; ties go to the largest distance
from the region edge. The result is verified with KLayout booleans (a pair whose strip is not
inside the region is dropped). The user's design is never changed to fit: the dimensions are the
presets; only the position, angle and count are chosen.

Contact gate (:func:`contact_gate_comb`, :func:`fit_contact_gate`): one tooth per gated finger, the
finger's width, from the tip outward ``cg.tooth_length_um``, joined behind the teeth by a bus
``cg.bus_width_um`` wide (as in the lab's drawings) that leads out sideways up to
``cg.bus_extension_um`` beyond the outer tooth, where an electrode lands. Without source / drain
layers in the stack, the outer finger(s) of each column at opposite ends are left ungated (source and
drain) and each bus leads away from them (``CG_LAYOUTS``); the lead is shortened, the reversed layout
used, or one more finger per column left ungated, so the comb lies ``cg.hbn_margin_um`` inside the
hBN under it (the user's own fix on Device 2026-11-16). :func:`gated_fingers` /
:func:`cg_terminals` read a contact gate as it is (also when drawn by hand).

Gate contacts (:func:`gate_contacts`): a square of ``electrodes.bg_contact_um`` on the gate flake,
at the points of the flake farthest along ±u from the array (the lab's back-gate bars at both ends
of the gap strip), at least ``electrodes.gate_contact_clearance_um`` from the twisted region,
the fingers and the contact gate.

Electrodes (:func:`electrodes`): every terminal (finger far ends, contact-gate bus ends, gate
contacts) gets a lead ``electrodes.width_um`` wide: on a finger it starts
``electrodes.land_from_tip_um`` from the tip and follows the finger to its far end, leaves straight
for two lead widths and then goes straight to a bond-pad lead tip; tips are assigned in angular
order around the device (no crossings in a radial fan-out, ``stage2.assign_tips``). Crossings and
spacing between leads are reported by :func:`check_pattern`, not avoided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import klayout.db as kdb
import numpy as np

from . import geometry as geo
from .cancel import poll
from .config import LabConfig
from .hallbar import lead_path

Poly = list[tuple[float, float]]
RES_TARGET_UM = 0.05


@dataclass
class FingerParams:
    width: float = 0.8
    pitch: float = 2.2
    gap: float = 4.0
    min_overlap: float = 0.8
    straight: float = 9.5
    length: float = 21.8
    far_pitch: float = 3.8
    far_width: float = 1.1
    cg_tooth: float = 6.1
    cg_bus_width: float = 1.6
    cg_bus_extension: float = 5.0
    electrode_width: float = 1.6
    land_from_tip: float = 14.2
    contact_size: float = 3.5
    contact_clearance: float = 3.0
    bbn_margin: float | None = None
    cg_hbn_margin: float | None = None
    cg_tmd_gap: float | None = None          # the contact-gate bus this far from the TMD (device layers): teeth fitted per column
    length_top: float | None = None          # set by fit_finger_lengths: the top / bottom column's own length
    length_bottom: float | None = None
    lengths: dict | None = None              # each finger's own length ("top:3" → µm; user, 2026-11-20: they may differ)

    KEYS = {"width": "fingers.width_um", "pitch": "fingers.pitch_um", "gap": "fingers.column_gap_um",
            "min_overlap": "fingers.min_overlap_um", "straight": "fingers.straight_um", "length": "fingers.length_um",
            "far_pitch": "fingers.far_pitch_um", "far_width": "fingers.far_width_um", "cg_tooth": "cg.tooth_length_um",
            "cg_bus_width": "cg.bus_width_um", "cg_bus_extension": "cg.bus_extension_um", "electrode_width": "electrodes.width_um",
            "land_from_tip": "electrodes.land_from_tip_um", "contact_size": "electrodes.bg_contact_um",
            "contact_clearance": "electrodes.gate_contact_clearance_um", "bbn_margin": "fingers.bbn_margin_um",
            "cg_hbn_margin": "cg.hbn_margin_um", "cg_tmd_gap": "cg.bus_tmd_gap_um"}
    FITTED = ("length_top", "length_bottom", "lengths")

    def length_of(self, column: str) -> float:
        """Tip-to-far-end length of a column's fingers (fitted to the bottom hBN, else the preset)."""
        v = self.length_top if column == "top" else self.length_bottom
        return float(v) if v is not None else self.length

    def straight_of(self, column: str) -> float:
        return min(self.straight, self.length_of(column))

    def length_at(self, column: str, i: int) -> float:
        """Tip-to-far-end length of one finger (its own fitted length, else its column's)."""
        v = (self.lengths or {}).get(f"{column}:{i}")
        return float(v) if v is not None else self.length_of(column)

    def straight_at(self, column: str, i: int) -> float:
        return min(self.straight, self.length_at(column, i))

    @classmethod
    def from_config(cls, config: LabConfig, overrides: dict | None = None) -> "FingerParams":
        vals = {}
        for attr, key in cls.KEYS.items():
            v = (overrides or {}).get(attr)
            if v is None:
                v = config.preset(key)
            if v is not None:
                vals[attr] = float(v)
        return cls(**vals)

    def to_json(self) -> dict:
        out = {k: getattr(self, k) for k in (*self.KEYS, *self.FITTED)}
        if out.get("lengths") is not None:
            out["lengths"] = dict(out["lengths"])
        return out


@dataclass
class Finger:
    column: str                  # "top" | "bottom"
    index: int                   # 0 … n−1 along +u
    polygon: Poly                # chip µm
    centreline: Poly             # tip → bend → far end, chip µm
    direction: tuple[float, float]  # unit vector at the far end, pointing away from the array

    @property
    def tip(self) -> tuple[float, float]:
        return self.centreline[0]

    @property
    def far_end(self) -> tuple[float, float]:
        return self.centreline[-1]

    @property
    def name(self) -> str:
        return f"{'T' if self.column == 'top' else 'B'}{self.index + 1}"


@dataclass
class FingerArray:
    center: tuple[float, float]
    angle_deg: float
    n: int
    params: FingerParams

    # -------------------------------------------------------------- frames
    def to_chip(self, pts) -> Poly:
        a = math.radians(self.angle_deg)
        c, s = math.cos(a), math.sin(a)
        cx, cy = self.center
        return [(cx + c * u - s * v, cy + s * u + c * v) for u, v in pts]

    def to_local(self, pts) -> Poly:
        a = math.radians(self.angle_deg)
        c, s = math.cos(a), math.sin(a)
        cx, cy = self.center
        return [(c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)) for x, y in pts]

    def region_to_chip(self, local: kdb.Region) -> kdb.Region:
        """A region drawn in the local frame, merged, then rotated/moved as a whole (no slivers
        between pieces from snapping each one to the grid separately)."""
        t = kdb.ICplxTrans(1.0, self.angle_deg, False, geo.to_dbu(self.center[0]), geo.to_dbu(self.center[1]))
        return local.merged().transformed(t).merged()

    def dir_chip(self, du: float, dv: float) -> tuple[float, float]:
        a = math.radians(self.angle_deg)
        c, s = math.cos(a), math.sin(a)
        return (c * du - s * dv, s * du + c * dv)

    def u_of(self, i: int) -> float:
        return (i - (self.n - 1) / 2.0) * self.params.pitch

    @property
    def span(self) -> float:
        """Tip-line span of the array along u (outer finger edges)."""
        return (self.n - 1) * self.params.pitch + self.params.width

    # -------------------------------------------------------------- shapes
    def far_u(self, u: float, length: float, straight: float) -> float:
        """Where a finger at ``u`` ends along u. The fan keeps the presets' angle all the way out: at
        the preset length the pitch is ``far_pitch``; a longer finger (fitted to the bottom hBN) keeps
        spreading at the same angle, so the far ends fan out wider."""
        p = self.params
        if length <= straight:
            return u
        ref = p.length - p.straight
        k = (p.far_pitch / p.pitch - 1.0) / ref if ref > 1e-9 else 0.0
        return u * (1.0 + k * (length - straight))

    def fingers(self) -> list[Finger]:
        return [self.finger(column, i) for column in ("top", "bottom") for i in range(self.n)]

    def finger(self, column: str, i: int, length: float | None = None) -> Finger:
        """One finger (``length`` instead of its own, to try one)."""
        p = self.params
        sgn = 1.0 if column == "top" else -1.0
        length = p.length_at(column, i) if length is None else float(length)
        straight = min(p.straight, length)
        u = self.u_of(i)
        uf = self.far_u(u, length, straight)
        v0 = sgn * p.gap / 2
        vb = sgn * (p.gap / 2 + straight)
        vf = sgn * (p.gap / 2 + length)
        w2, f2 = p.width / 2, p.far_width / 2
        # the fan aims at the base of a stub (the last ``far_width`` µm, along v): every end is square (a slanted cut
        # would leave an acute tip under the width minimum); fingers of one length end on one line
        h = min(p.far_width, abs(vf - vb) / 2) if length > straight else 0.0
        v_cut = vf - sgn * h
        d = (uf - u, v_cut - vb) if length > straight else (0.0, sgn)
        n_ = math.hypot(*d) or 1.0
        du, dv = d[0] / n_, d[1] / n_
        nu, nv = -dv, du                                    # normal of the fan piece
        local = geo.region([(u - w2, v0), (u + w2, v0), (u + w2, vb), (u - w2, vb)])
        if length > straight:
            local.insert(geo.polygon([(u + nu * w2, vb + nv * w2), (u - nu * w2, vb - nv * w2),
                                      (uf - nu * f2, v_cut - nv * f2), (uf + nu * f2, v_cut + nv * f2)]))
            local.insert(geo.polygon([(uf - f2, v_cut), (uf + f2, v_cut), (uf + f2, vf), (uf - f2, vf)]))
            # the second joint: convex hull of the fan's square end and the stub's base (no notches)
            hull2 = cv2.convexHull(np.array([(uf - nu * f2, v_cut - nv * f2), (uf + nu * f2, v_cut + nv * f2),
                                             (uf - f2, v_cut), (uf + f2, v_cut)], np.float64).astype(np.float32))
            local.insert(geo.polygon([tuple(map(float, q[0])) for q in hull2]))
            # the joint: convex hull of the straight part's end and the fan piece's start (no notches)
            hull = cv2.convexHull(np.array([(u - w2, vb), (u + w2, vb), (u + nu * w2, vb + nv * w2),
                                            (u - nu * w2, vb - nv * w2)], np.float64).astype(np.float32))
            local.insert(geo.polygon([tuple(map(float, q[0])) for q in hull]))
        reg = self.region_to_chip(local)
        return Finger(column, i, geo.to_polys(reg)[0], self.to_chip([(u, v0), (u, vb), (uf, vf)]), self.dir_chip(du, dv))

    def strip(self, i: int, overlap: float | None = None) -> Poly:
        """The pair strip of finger pair ``i`` (channel + overlap of each tip), chip µm."""
        p = self.params
        ov = p.min_overlap if overlap is None else overlap
        u = self.u_of(i)
        h = p.gap / 2 + ov
        return self.to_chip([(u - p.width / 2, -h), (u + p.width / 2, -h), (u + p.width / 2, h), (u - p.width / 2, h)])

    def channel(self) -> Poly:
        """The gap between the tip lines over the array span (chip µm)."""
        p = self.params
        hs = self.span / 2
        return self.to_chip([(-hs, -p.gap / 2), (hs, -p.gap / 2), (hs, p.gap / 2), (-hs, p.gap / 2)])

    def point_at(self, finger: Finger, dist_from_tip: float) -> tuple[float, float]:
        """The point on a finger's centre line ``dist_from_tip`` from the tip (measured along v)."""
        p = self.params
        length, straight = p.length_at(finger.column, finger.index), p.straight_at(finger.column, finger.index)
        t, b, f = finger.centreline
        if dist_from_tip <= straight or length <= straight:
            r = min(dist_from_tip / max(straight, 1e-9), 1.0)
            return (t[0] + (b[0] - t[0]) * r, t[1] + (b[1] - t[1]) * r)
        r = min((dist_from_tip - straight) / (length - straight), 1.0)
        return (b[0] + (f[0] - b[0]) * r, b[1] + (f[1] - b[1]) * r)

    def to_json(self) -> dict:
        return {"center": [round(self.center[0], 4), round(self.center[1], 4)], "angle_deg": round(self.angle_deg, 4), "n": self.n,
                "params": self.params.to_json()}

    @classmethod
    def from_json(cls, d: dict) -> "FingerArray":
        return cls(tuple(d["center"]), float(d["angle_deg"]), int(d["n"]), FingerParams(**d["params"]))


# ====================================================================== fitting
@dataclass
class FitResult:
    array: FingerArray | None
    n: int
    margin_um: float
    notes: list[str] = field(default_factory=list)


def _longest_runs(valid: np.ndarray, dt: np.ndarray):
    """Per row of ``valid`` (H × K): longest run of True; returns (length, end index, min dt over the run)."""
    H, K = valid.shape
    run = np.zeros(H, np.int32)
    best = np.zeros(H, np.int32)
    best_end = np.zeros(H, np.int32)
    for k in range(K):
        run = (run + 1) * valid[:, k]
        better = run > best
        best = np.where(better, run, best)
        best_end = np.where(better, k, best_end)
    # min distance over the best run of each row (rows with best ≥ 1)
    margin = np.full(H, -1.0)
    for r in np.nonzero(best)[0]:
        e = best_end[r]
        seg = dt[r, e - best[r] + 1:e + 1]
        margin[r] = float(seg.min()) + 1e-3 * float(seg.mean())     # ties → the more central placement
    return best, best_end, margin


def _fit_at_angle(region_polys: list[Poly], centre: tuple[float, float], angle: float, p: FingerParams):
    """Best (n, margin, centre_chip) for one angle."""
    poll()
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    loc = [[(c * (x - centre[0]) + s * (y - centre[1]), -s * (x - centre[0]) + c * (y - centre[1])) for x, y in poly]
           for poly in region_polys]
    allp = np.concatenate([np.asarray(q) for q in loc])
    q = max(1, round(p.pitch / RES_TARGET_UM))
    res = p.pitch / q
    u0, v0 = allp.min(0) - 2 * res
    u1, v1 = allp.max(0) + 2 * res
    W, H = int(math.ceil((u1 - u0) / res)) + 1, int(math.ceil((v1 - v0) / res)) + 1
    if W * H > 25_000_000:
        raise ValueError("The twisted region is too large to fit fingers at this resolution")
    mask = np.zeros((H, W), np.uint8)
    for poly in loc:
        pts = np.round((np.asarray(poly) - (u0, v0)) / res).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    kw = max(1, int(round(p.width / res)))
    kh = max(1, int(round((p.gap + 2 * p.min_overlap) / res)))
    if kh >= H or kw >= W:
        return 0, -1.0, None
    ker = np.ones((kh | 1, kw | 1), np.uint8)       # odd sizes: anchored at the centre
    ero = cv2.erode(mask, ker, borderValue=0)
    if not ero.any():
        return 0, -1.0, None
    dt = cv2.distanceTransform(ero, cv2.DIST_L2, 3) * res
    best = (0, -1.0, None)
    for phase in range(q):
        sub, dsub = ero[:, phase::q].astype(bool), dt[:, phase::q]
        if sub.shape[1] == 0:
            continue
        length, end, margin = _longest_runs(sub, dsub)
        r = int(np.lexsort((margin, length))[-1])
        n, m = int(length[r]), float(margin[r])
        if (n, m) > best[:2]:
            k_end = int(end[r])
            cols = [phase + q * (k_end - j) for j in range(n)]
            uc = u0 + res * (sum(cols) / n)
            vc = v0 + res * r
            best = (n, m, (centre[0] + c * uc - s * vc, centre[1] + s * uc + c * vc))
    return best


def fit_finger_array(region_polys: list[Poly], params: FingerParams, angle_deg: float | None = None,
                     n_max: int | None = None, angle_step: float = 2.0) -> FitResult:
    """Place the largest finger array whose pairs all sit on the twisted region (see the module
    docstring). ``angle_deg`` fixes the array axis; ``n_max`` caps the count (centred)."""
    region_polys = [list(map(tuple, q)) for q in region_polys if len(q) >= 3]
    if not region_polys:
        raise ValueError("No twisted region: locate or draw it first")
    centre = geo.centroid(max(region_polys, key=geo.poly_area))
    angles = [angle_deg] if angle_deg is not None else list(np.arange(0.0, 180.0, angle_step))
    scored = [(_fit_at_angle(region_polys, centre, float(a), params), float(a)) for a in angles]
    (n, m, cc), ang = max(scored, key=lambda t: t[0][:2])
    if angle_deg is None and n > 0:
        for a in np.arange(ang - angle_step, ang + angle_step + 1e-9, 0.25):
            r = _fit_at_angle(region_polys, centre, float(a), params)
            if r[:2] > (n, m):
                (n, m, cc), ang = r, float(a)
    notes = []
    if n == 0 or cc is None:
        raise ValueError(f"Not even one finger pair fits: the twisted region must hold a {params.width:g} µm wide strip "
                         f"{params.gap + 2 * params.min_overlap:g} µm long (gap {params.gap:g} µm + {params.min_overlap:g} µm "
                         "of each finger). UStack does not change the finger dimensions; edit them in the presets if intended.")
    if n_max is not None and n > n_max:
        notes.append(f"{n} pairs fit; {n_max} drawn (your limit), centred in the run.")
        n = n_max
    arr = FingerArray(cc, ang % 360.0, n, params)
    # verify with exact booleans, dropping pairs at the ends whose strip is not inside
    reg = geo.regions(region_polys).merged()
    ok = [(geo.region(arr.strip(i)) - reg).is_empty() for i in range(n)]
    if not all(ok):
        lo = next(i for i, v in enumerate(ok) if v) if any(ok) else n
        hi = n - next(i for i, v in enumerate(reversed(ok)) if v) if any(ok) else 0
        run = [i for i in range(lo, hi) if ok[i]]
        if not run:
            raise ValueError("No finger pair fits inside the twisted region")
        u_mid = (arr.u_of(run[0]) + arr.u_of(run[-1])) / 2
        (cx, cy), = arr.to_chip([(u_mid, 0.0)])
        notes.append(f"{n - len(run)} pair(s) at the ends dropped after the exact check.")
        arr = FingerArray((cx, cy), arr.angle_deg, len(run), params)
    return FitResult(arr, arr.n, m, notes)


def fit_finger_lengths(arr: FingerArray, bbn_polys: list[Poly], margin: float, step: float = 0.05,
                       max_length: float = 400.0, avoid: kdb.Region | None = None) -> tuple[dict, list[str]]:
    """Each finger as long as possible on its own (user, 2026-11-20: the fingers need not be the same length): the largest
    tip-to-far-end length (to ``step``) at which it lies at least ``margin`` inside the bottom hBN (the fingers are written
    on it) and off ``avoid`` (bubbles / cracks grown by their clearance). Width, pitch, straight part and fan angle are the
    presets; only the lengths change. Returns ({"top": longest of the column, "bottom": …, "fingers": {"top:3": L, …}}, notes);
    a finger whose first part does not fit keeps its column's preset length (and is said so)."""
    allowed = geo.sized(geo.regions(bbn_polys).merged(), -margin)
    if avoid is not None and not avoid.is_empty():
        allowed = (allowed - avoid).merged()
    p0 = arr.params
    per: dict[str, float] = {}
    notes, short = [], {"top": [], "bottom": []}

    def fits(column: str, i: int, length: float) -> bool:
        return (geo.region(arr.finger(column, i, length).polygon) - allowed).is_empty()

    for column in ("top", "bottom"):
        for i in range(arr.n):
            lo = min(p0.min_overlap, p0.straight)
            if not fits(column, i, lo):
                short[column].append(i)
                continue
            good, bad = lo, None                       # coarse walk out (1 µm) to the first length that does not fit, then bisect
            L = lo + 1.0
            while L <= max_length:
                if fits(column, i, L):
                    good, L = L, L + 1.0
                else:
                    bad = L
                    break
            if bad is not None:
                while bad - good > step:
                    mid = (good + bad) / 2
                    good, bad = (mid, bad) if fits(column, i, mid) else (good, mid)
            per[f"{column}:{i}"] = math.floor(good / step) * step
        if short[column]:
            notes.append(f"{column.capitalize()} column, finger(s) {', '.join(str(i + 1) for i in short[column])}: even "
                         f"{min(p0.min_overlap, p0.straight):g} µm is not {margin:g} µm inside the bottom hBN (and off bubbles / "
                         f"cracks); the preset length {p0.length:g} µm is used for them.")
    out = {c: max((v for k, v in per.items() if k.startswith(c + ":")), default=None) for c in ("top", "bottom")}
    out["fingers"] = per
    return out, notes


# ====================================================================== other shapes
#: Contact-gate layouts tried in order (``fit_contact_gate``): which finger of each column is left
#: ungated as source / drain (the outer one at opposite ends, as in the lab's Stack 1 drawing) and the
#: side each bus leads out to (away from that finger).
CG_LAYOUTS = ({"skip": {"top": "first", "bottom": "last"}, "dirs": {"top": 1.0, "bottom": -1.0}},
              {"skip": {"top": "last", "bottom": "first"}, "dirs": {"top": -1.0, "bottom": 1.0}})


def _skipped(arr: "FingerArray", which, count: int = 1) -> set[int]:
    """The ``count`` outer fingers left ungated at the ``which`` end ("first" = lowest u, "last")."""
    if which in (None, "none") or count <= 0:
        return set()
    return set(range(count)) if which == "first" else set(range(arr.n - count, arr.n))


_LOCAL_CACHE: dict = {}
#: Where a tooth follows a fanned finger, the side meeting the bus at an acute angle gets a fill this
#: far below the bus (program choice: no notch the DRC flags).
TOOTH_FILLET_UM = 0.5


def _local_fingers(arr: FingerArray) -> dict[tuple[str, int], kdb.Region]:
    """Every finger's shape in the array's local frame (cached per array geometry)."""
    params = tuple(sorted(((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in arr.params.to_json().items()),
                          key=lambda kv: kv[0]))
    key = (tuple(arr.center), arr.angle_deg, arr.n, params)
    got = _LOCAL_CACHE.get(key)
    if got is None:
        got = {(f.column, f.index): geo.region(arr.to_local(f.polygon)) for f in arr.fingers()}
        if len(_LOCAL_CACHE) > 64:
            _LOCAL_CACHE.clear()
        _LOCAL_CACHE[key] = got
    return got


def _tooth_local(arr: FingerArray, column: str, i: int, t: float) -> kdb.Region:
    """Finger ``i``'s first ``t`` µm from the tip (local frame): the tooth follows the finger, also
    where it fans out (a straight rectangle would leave it there)."""
    p = arr.params
    sgn = 1.0 if column == "top" else -1.0
    big = 1e4
    finger = _local_fingers(arr)[(column, i)]
    v1 = p.gap / 2 + t
    tooth = finger & geo.region([(-big, sgn * p.gap / 2), (big, sgn * p.gap / 2), (big, sgn * v1), (-big, sgn * v1)])
    # where the finger fans out, one edge of the tooth meets the bus at an acute angle (a notch the DRC
    # flags): there a near-vertical edge runs from the tooth edge TOOTH_FILLET_UM below the bus up to it
    pieces = geo.to_polys(tooth.merged())
    if len(pieces) == 1 and t > p.straight_at(column, i) + 1e-6:
        pts = pieces[0]
        tops = [k for k, q in enumerate(pts) if abs(abs(q[1]) - v1) < 2e-3]
        if len(tops) == 2:
            for k in tops:
                other = pts[tops[1] if k == tops[0] else tops[0]]
                T = pts[k]
                out = 1.0 if T[0] > other[0] else -1.0
                N = next(pts[j] for j in ((k - 1) % len(pts), (k + 1) % len(pts)) if j not in tops)
                if (N[0] - T[0]) * out <= 1e-4:
                    continue                                  # this side meets the bus at ≥ 90°
                d = min(TOOTH_FILLET_UM, abs(N[1] - T[1]))
                r = d / max(abs(N[1] - T[1]), 1e-9)
                Q = (T[0] + (N[0] - T[0]) * r, T[1] + (N[1] - T[1]) * r)
                e = 0.02 * out
                tooth.insert(geo.polygon([(T[0] - e, T[1]), (Q[0] - e, Q[1]), (Q[0], T[1])]))
    return tooth.merged()


def _teeth_local(arr: FingerArray, column: str, gated, t: float) -> tuple[kdb.Region, float, float]:
    """The teeth of ``gated`` fingers (local frame) and their u extent."""
    reg = kdb.Region()
    for i in gated:
        reg.insert(_tooth_local(arr, column, i, t))
    reg = reg.merged()
    x0, _y0, x1, _y1 = geo.bbox(reg)
    return reg, x0, x1


def tooth_to_keepout(arr: FingerArray, column: str, u_lo: float, u_hi: float, keepout: kdb.Region | None) -> float:
    """The shortest tooth (tip → bus, along v) that puts a bus spanning ``u_lo``…``u_hi`` (local u)
    outside ``keepout`` (the TMD grown by ``cg.bus_tmd_gap_um``): the bus as close to the TMD as that
    allows. At least ``fingers.min_overlap_um`` (a tooth covers at least the tip part the flake must
    overlap — program choice); the preset ``cg.tooth_length_um`` without a keepout."""
    p = arr.params
    if keepout is None or keepout.is_empty():
        return p.cg_tooth
    sgn = 1.0 if column == "top" else -1.0
    far = p.gap / 2 + max(p.length_of("top"), p.length_of("bottom"), p.cg_tooth) + 100.0
    band = geo.region(arr.to_chip([(u_lo, sgn * p.gap / 2), (u_hi, sgn * p.gap / 2), (u_hi, sgn * far), (u_lo, sgn * far)]))
    hit = band & keepout
    t_min = p.min_overlap
    if hit.is_empty():
        return t_min
    v = max(sgn * q[1] for poly in geo.to_polys(hit) for q in arr.to_local(poly))
    return max(t_min, round(v - p.gap / 2 + 0.005, 2))           # rounded up to 10 nm


def contact_gate_comb(arr: FingerArray, columns: tuple[str, ...] = ("top", "bottom"), skip: dict | None = None,
                      dirs: dict | None = None, extension: dict | None = None, n_skip: int = 1, tooth: dict | None = None,
                      keepout: kdb.Region | None = None) -> dict[str, dict]:
    """One comb per column: {"top": {"polygon": …, "terminal": (point, direction), "gated": [indices]}, …}.

    A tooth per gated finger (its width, from the tip outward ``cg.tooth_length_um``), joined behind
    the teeth by a bus ``cg.bus_width_um`` wide (as in the lab's drawings: it crosses only the gated
    fingers) that leads out sideways ``extension`` µm beyond the outer gated tooth on the side
    ``dirs[column]`` (+1 = +u). ``skip[column]`` = "first" / "last" / None: the end whose ``n_skip``
    outer fingers are left ungated (source / drain contacts); the bus never reaches them. Defaults:
    every finger gated, top bus to −u, bottom to +u. Tooth length per column: ``tooth[column]``, else
    as short as ``keepout`` allows (:func:`tooth_to_keepout`: the bus as close to the TMD as the gap
    preset lets it), else ``cg.tooth_length_um`` — the two columns need not match."""
    p = arr.params
    skip = skip or {}
    dirs = dirs or {"top": -1.0, "bottom": 1.0}
    out = {}
    for column, sgn in (("top", 1.0), ("bottom", -1.0)):
        if column not in columns:
            continue
        side = float(dirs.get(column, -1.0 if column == "top" else 1.0))
        ext = float((extension or {}).get(column, p.cg_bus_extension))
        skipped = _skipped(arr, skip.get(column), n_skip)
        gated = [i for i in range(arr.n) if i not in skipped]
        if not gated:
            continue
        lo, hi = arr.u_of(gated[0]) - p.width / 2, arr.u_of(gated[-1]) + p.width / 2
        t = (tooth or {}).get(column)
        if t is not None:
            t = float(t)
            teeth, lo, hi = _teeth_local(arr, column, gated, t)
        else:
            for _ in range(4):                    # fanned teeth widen the bus span, which may push the bus out: settle
                ua, ub = (lo - ext, hi) if side < 0 else (lo, hi + ext)
                t = float(tooth_to_keepout(arr, column, ua, ub, keepout))
                if keepout is not None and t > p.straight_of(column) + 1e-6:
                    t += TOOTH_FILLET_UM                # fanned teeth get a fill that far below the bus: keep it off the TMD too
                teeth, lo2, hi2 = _teeth_local(arr, column, gated, t)
                if abs(lo2 - lo) < 1e-3 and abs(hi2 - hi) < 1e-3:
                    break
                lo, hi = lo2, hi2
        ua, ub = (lo - ext, hi) if side < 0 else (lo, hi + ext)
        reg = kdb.Region()
        reg.insert(teeth)
        v1 = p.gap / 2 + t
        b0, b1 = v1, v1 + p.cg_bus_width                       # the bus behind the teeth
        reg.insert(geo.polygon([(ua, sgn * b0), (ub, sgn * b0), (ub, sgn * b1), (ua, sgn * b1)]))
        reg = arr.region_to_chip(reg)
        u_end = ua if side < 0 else ub
        (tx, ty), = arr.to_chip([(u_end - side * p.electrode_width / 2, sgn * (b0 + b1) / 2)])
        ew = p.electrode_width
        lead_end = arr.to_chip([(u_end, sgn * b0), (u_end - side * ew, sgn * b0), (u_end - side * ew, sgn * b1), (u_end, sgn * b1)])
        out[column] = {"polygon": geo.to_polys(reg)[0], "terminal": ((tx, ty), arr.dir_chip(side, 0.0)), "gated": gated,
                       "bus": arr.to_chip([(ua, sgn * b0), (ub, sgn * b0), (ub, sgn * b1), (ua, sgn * b1)]), "lead_end": lead_end,
                       "tooth": float(t)}
    return out


#: At most this many outer fingers per column are left ungated to make the comb fit on its hBN
#: (program choice; the user left 2 on Device 2026-11-16 when 1 did not fit).
CG_MAX_SKIP = 3


_KEEPOUT_CACHE: dict = {}


def tmd_keepout(tmd_polys: list[Poly] | None, gap: float | None) -> kdb.Region | None:
    """Where the contact-gate bus may not be: the TMD (device layers) grown by ``cg.bus_tmd_gap_um``. Memoized (the auto
    design asks for the same outlines many times); a copy is returned."""
    if not tmd_polys or gap is None:
        return None
    key = (tuple(tuple((round(float(x), 4), round(float(y), 4)) for x, y in q) for q in tmd_polys), round(float(gap), 6))
    reg = _KEEPOUT_CACHE.get(key)
    if reg is None:
        if len(_KEEPOUT_CACHE) > 256:
            _KEEPOUT_CACHE.clear()
        reg = _KEEPOUT_CACHE[key] = geo.grown_round(geo.regions(tmd_polys).merged(), float(gap))
    return reg.dup()


def fit_contact_gate(arr: FingerArray, hbn_polys: list[Poly] | None, margin: float | None, leave_sd: bool,
                     step: float = 0.1, min_width: float | None = None, keepout: kdb.Region | None = None,
                     avoid: kdb.Region | None = None) -> tuple[dict, list[str]]:
    """Choose the contact-gate layout, as the user did by hand on Device 2026-11-16: the bus directions
    from ``CG_LAYOUTS`` (in order) with one source / drain finger left ungated at opposite ends of the
    two columns (``leave_sd``); when a comb does not lie ``margin`` inside the hBN under it
    (``hbn_polys``), each bus lead is shortened (down to one electrode width), then the other layout
    is tried, then one more outer finger per column is left ungated (the same number in both columns,
    up to ``CG_MAX_SKIP``) — fingers whose tooth would leave the hBN become source / drain contacts.
    With ``keepout`` (:func:`tmd_keepout`) each column's teeth are as short as puts its bus that far
    from the TMD (user, 2026-11-17: the bus as close to the WSe2 as possible; the columns differ).
    With ``avoid`` (bubbles / cracks grown by their clearance) no part of the comb may touch it.
    Returns ({"skip", "dirs", "extension", "n_skip", "trim", "margin", "tooth"}, notes); raises
    ValueError when nothing fits."""
    p = arr.params
    layouts = CG_LAYOUTS if leave_sd else ({"skip": {}, "dirs": {"top": 1.0, "bottom": -1.0}}, {"skip": {}, "dirs": {"top": -1.0, "bottom": 1.0}})
    counts = range(1, min(CG_MAX_SKIP, arr.n - 1) + 1) if leave_sd else (0,)
    if not hbn_polys:
        lay = layouts[0]
        ext0 = {"top": p.cg_bus_extension, "bottom": p.cg_bus_extension}
        comb = contact_gate_comb(arr, skip=lay["skip"], dirs=lay["dirs"], extension=ext0, n_skip=1 if leave_sd else 0, keepout=keepout)
        return ({"skip": dict(lay["skip"]), "dirs": dict(lay["dirs"]), "extension": ext0, "n_skip": 1 if leave_sd else 0,
                 "tooth": {c: v["tooth"] for c, v in comb.items()}},
                ["The hBN under the contact gate is not located: the comb is not checked against it (locate it, then Generate again)."])
    allowed = geo.sized(geo.regions(hbn_polys).merged(), -(margin or 0.0))
    fails = []

    def fits(column, lay, e, n_skip, trim):
        comb = contact_gate_comb(arr, (column,), lay["skip"], lay["dirs"], {column: e}, n_skip, keepout=keepout)
        if column not in comb:
            return None
        reg = geo.region(comb[column]["polygon"])
        if avoid is not None and not (reg & avoid).is_empty():
            if not trim:
                return None
            cut = _trim(arr, comb[column], allowed - avoid, column).merged()
            return comb[column]["tooth"] if (cut & avoid).is_empty() and _trimmed_ok(arr, comb[column], None, allowed - avoid,
                                                                                       column, min_width) else None
        if (reg - allowed).is_empty():
            return comb[column]["tooth"]
        if not trim:
            return None
        return comb[column]["tooth"] if _trimmed_ok(arr, comb[column], None, allowed, column, min_width) else None

    for n_skip in counts:
        for n_lay, lay in enumerate(layouts):
            poll()
            ext, trims, teeth, ok = {}, {}, {}, True
            for column in ("top", "bottom"):
                best = None
                for trim in (False, True):                 # trimming the bus's corners (as the user did) before giving up fingers
                    e = p.cg_bus_extension
                    while e >= p.electrode_width - 1e-9:
                        t = fits(column, lay, e, n_skip, trim)
                        if t is not None:
                            best = (e, trim, t)
                            break
                        e = round(e - step, 6)
                    if best is not None:
                        break
                if best is None:
                    ok = False
                    fails.append(f"{n_skip} ungated per column, layout {n_lay + 1}, {column} column")
                    break
                ext[column], trims[column], teeth[column] = best
            if ok:
                notes = []
                if any(trims.values()):
                    notes.append("Bus trimmed to the hBN where its corners would leave it (" + ", ".join(c for c, t in trims.items() if t)
                                 + "): the teeth are whole, the bus at least a finger wide.")
                if any(v < p.cg_bus_extension - 1e-9 for v in ext.values()):
                    notes.append("Bus lead shortened to stay on the hBN: " + ", ".join(f"{c} {v:g} µm" for c, v in ext.items()) + ".")
                if n_lay:
                    notes.append("The first layout did not fit on the hBN; the reversed one is used.")
                if n_skip > 1:
                    notes.append(f"{n_skip} outer fingers per column are left ungated (source / drain): with fewer, a tooth would leave the hBN.")
                return ({"skip": dict(lay["skip"]), "dirs": dict(lay["dirs"]), "extension": ext, "n_skip": n_skip,
                         "trim": {c: bool(t) for c, t in trims.items()}, "margin": margin, "tooth": teeth}, notes)
    raise ValueError(f"The contact gate does not fit {margin or 0:g} µm inside the hBN under it, with leads down to "
                     f"{p.electrode_width:g} µm and up to {max(counts)} ungated fingers per column ({len(fails)} layouts tried). "
                     "UStack does not change the comb's dimensions: draw it (Draw contact gate…), or adjust the presets or the hBN outline.")


def _teeth(arr: FingerArray, column: str, gated, tooth: float | None = None) -> kdb.Region:
    p = arr.params
    t = p.cg_tooth if tooth is None else tooth
    return arr.region_to_chip(_teeth_local(arr, column, gated, t)[0])


def _trim(arr: FingerArray, comb: dict, allowed: kdb.Region, column: str) -> kdb.Region:
    """The comb cut to the allowed area, slivers narrower than a finger opened away, the teeth whole."""
    r = arr.params.width / 2 * 0.99
    cut = (geo.region(comb["polygon"]) & allowed).merged()
    cut = geo.sized(geo.sized(cut, -r), r).merged()
    out = ((cut | _teeth(arr, column, comb["gated"], comb.get("tooth"))).merged() & allowed).merged()
    pieces = geo.to_polys(out)
    if len(pieces) > 1:                                   # specks cut off by the hBN edge: keep the comb itself
        out = geo.region(max(pieces, key=geo.poly_area))
    return out


def _trimmed_ok(arr: FingerArray, comb: dict, trimmed: kdb.Region, allowed: kdb.Region, column: str,
                min_width: float | None = None) -> bool:
    """A comb cut to the allowed area is acceptable when every tooth is whole, it is still one piece
    and nowhere narrower than ``min_width`` (the lab's hard minimum; default a finger width)."""
    teeth = _teeth(arr, column, comb["gated"], comb.get("tooth"))
    if not (teeth - allowed).is_empty():
        return False
    if not (geo.region(comb["lead_end"]) - allowed).is_empty():     # where the electrode lands stays whole
        return False
    merged = _trim(arr, comb, allowed, column).merged()
    if merged.count() != 1 or not (teeth - merged).is_empty():
        return False
    return not geo.width_violations(merged, (min_width or arr.params.width) * 0.999)


def apply_trim(arr: FingerArray, comb: dict[str, dict], layout: dict, hbn_polys: list[Poly] | None,
               avoid: kdb.Region | None = None) -> dict[str, dict]:
    """The combs of ``layout`` with the bus cut to the hBN (and off ``avoid``) where ``fit_contact_gate`` chose to."""
    trim = layout.get("trim") or {}
    if not hbn_polys or not any(trim.values()):
        return comb
    allowed = geo.sized(geo.regions(hbn_polys).merged(), -(layout.get("margin") or 0.0))
    if avoid is not None and not avoid.is_empty():
        allowed = (allowed - avoid).merged()
    for column, c in comb.items():
        if trim.get(column):
            polys = geo.to_polys(_trim(arr, c, allowed, column).merged())
            if polys:
                c["polygon"] = max(polys, key=geo.poly_area)
    return comb


def _segment_ok(seg: kdb.Region, allowed: kdb.Region, blocked: kdb.Region, own: kdb.Region | None = None,
                spacing: float = 0.0) -> bool:
    """The new piece (``seg`` beyond the comb ``own`` already there) lies in ``allowed``, off ``blocked``, and — with
    ``spacing`` — leaves no gap narrower than that against the comb (a thin lead leaving the bus at a shallow angle)."""
    new = (seg - own) if own is not None else seg
    new = geo.sized(geo.sized(new, -0.01), 0.01)          # hairlines along the bus edge (grid rounding) are no lead
    if not (new - allowed).is_empty() or not (blocked.is_empty() or (new & blocked).is_empty()):
        return False
    if spacing > 0 and own is not None:
        return not geo.space_violations((own + seg).merged(), spacing)
    return True


#: A bus lead's width changes gradually — each side at most this steep — where it narrows through a tight stretch and
#: widens again after it (program choice; user, 2026-11-20: as smoothly as possible, no sudden angled neck).
TAPER_MAX_HALF_ANGLE_DEG = 15.0
#: Raster step and sample step (µm) of the lead width search (program choices; the result is checked exactly).
LEAD_RASTER_UM = 0.02
LEAD_SAMPLE_UM = 0.05
#: The width profile is simplified to straight runs within this (µm), then lowered to stay within the room.
LEAD_WIDTH_TOL_UM = 0.03
#: The room at a point is the narrowest within this distance along the lead (smooth sides).
LEAD_SMOOTH_UM = 0.25


def taper_path(a, b, w0: float, w1: float, taper_len: float) -> Poly:
    """A straight lead from ``a`` to ``b``: ``w0`` wide at ``a``, narrowing linearly to ``w1`` over ``taper_len`` µm, then
    ``w1`` wide to ``b`` (flat ends). ``taper_len`` ≥ the length = one taper over the whole lead."""
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    if L <= 0:
        return []
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux

    def at(s_: float, w_: float, side: float) -> tuple[float, float]:
        return (ax + ux * s_ + side * nx * w_ / 2, ay + uy * s_ + side * ny * w_ / 2)

    t = min(max(taper_len, 0.0), L)
    left = [at(0.0, w0, 1.0), at(t, w1, 1.0)] + ([at(L, w1, 1.0)] if t < L - 1e-9 else [])
    right = [at(0.0, w0, -1.0), at(t, w1, -1.0)] + ([at(L, w1, -1.0)] if t < L - 1e-9 else [])
    return left + right[::-1]


def profile_path(a, direction, s: np.ndarray, widths: np.ndarray) -> Poly:
    """A straight lead from ``a`` along ``direction`` whose width at distance ``s[i]`` is ``widths[i]`` (flat ends); points
    where the width does not change direction are dropped."""
    ux, uy = direction
    nx, ny = -uy, ux
    keep = [0]
    for i in range(1, len(s) - 1):
        d0 = widths[i] - widths[keep[-1]]
        d1 = widths[i + 1] - widths[i]
        slope0 = d0 / max(s[i] - s[keep[-1]], 1e-12)
        slope1 = d1 / max(s[i + 1] - s[i], 1e-12)
        if abs(slope0 - slope1) > 1e-6:
            keep.append(i)
    keep.append(len(s) - 1)
    left = [(a[0] + ux * s[i] + nx * widths[i] / 2, a[1] + uy * s[i] + ny * widths[i] / 2) for i in keep]
    right = [(a[0] + ux * s[i] - nx * widths[i] / 2, a[1] + uy * s[i] - ny * widths[i] / 2) for i in keep]
    return left + right[::-1]


def _straight_runs(s: np.ndarray, w: np.ndarray, tol: float | None = None) -> np.ndarray:
    """``w`` (sampled at ``s``) as straight runs (Douglas–Peucker within ``tol``), each run lowered just enough to stay
    at or below ``w`` everywhere — the lead's sides become straight lines instead of a raster staircase."""
    tol = LEAD_WIDTH_TOL_UM if tol is None else tol
    n = len(s)
    if n < 3:
        return w.copy()
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        line = w[i] + (w[j] - w[i]) * (s[i + 1:j] - s[i]) / (s[j] - s[i])
        k = int(np.argmax(np.abs(w[i + 1:j] - line)))
        if abs(w[i + 1 + k] - line[k]) > tol:
            keep.add(i + 1 + k)
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    ks = sorted(keep)
    low = np.zeros(len(ks))
    for a in range(len(ks) - 1):                   # each run's overshoot over the room, taken off both of its ends
        i, j = ks[a], ks[a + 1]
        line = w[i] + (w[j] - w[i]) * (s[i:j + 1] - s[i]) / (s[j] - s[i])
        over = float(np.max(line - w[i:j + 1]))
        low[a], low[a + 1] = max(low[a], over), max(low[a + 1], over)
    return np.interp(s, s[ks], w[ks] - low)


def _raster(reg: kdb.Region, x0: float, y0: float, res: float, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    pts = [np.round((np.asarray(poly, float) - (x0, y0)) / res).astype(np.int32) for poly in geo.to_polys(reg)]
    if pts:
        cv2.fillPoly(mask, pts, 1)
    return mask


def extend_buses(arr: FingerArray, comb: dict[str, dict], layout: dict, hbn_polys: list[Poly] | None, margin: float | None,
                 keepout: kdb.Region | None = None, spacing: float = 0.0,
                 min_width: float | None = None) -> tuple[dict[str, dict], dict[str, tuple]]:
    """Each bus led on from its outer end, slightly turned, to the far corner of the hBN it lies on (user, 2026-11-20):
    the top bus toward (lead side, +v), the bottom one toward (its side, −v) — two opposite corners — ending as close to
    the hBN edge as ``margin`` allows, so a stretch of bus lies outside where the top hBN will go and the electrodes
    can land on it. It stays ``margin`` inside the hBN, off ``keepout`` (the TMD grown by the bus gap) and ``spacing``
    clear of the ungated fingers and of the other column's comb.

    Width (user, 2026-11-20): every part of the lead is as wide as the room there allows, never wider than the bus
    (``cg.bus_width_um``) and never narrower than ``min_width`` (the lab's hard minimum): it keeps the bus width wherever
    that fits and narrows only through a tight stretch, widening again after it; the width changes no steeper than
    ``TAPER_MAX_HALF_ANGLE_DEG`` per side and the lead leaves the bus at the bus width (no neck). The end is the one
    reaching farthest toward the corner; among ends within (bus width − hard minimum) µm of that — what pinching the tip into a corner gains — the lead with the
    widest narrowest part
    (then the widest on average) is kept. A column whose lead cannot go farther out than the bus end keeps its straight
    bus. Returns (comb, {column: (x, y, narrowest width)}); each comb gets "reach", "lead_width" (its narrowest part)
    and "end_width"."""
    if not hbn_polys:
        return comb, {}
    p = arr.params
    W = p.cg_bus_width
    w_min = min(W, float(min_width) if min_width else p.width)
    slope = 2 * math.tan(math.radians(TAPER_MAX_HALF_ANGLE_DEG))      # full-width change per µm of length
    allowed = geo.sized(geo.regions(hbn_polys).merged(), -(margin or 0.0))
    if allowed.is_empty():
        return comb, {}
    res, ds = LEAD_RASTER_UM, LEAD_SAMPLE_UM
    bx0, by0, bx1, by1 = geo.bbox(allowed)
    x0, y0 = bx0 - 1.0, by0 - 1.0
    shape = (int(math.ceil((by1 + 1.0 - y0) / res)) + 1, int(math.ceil((bx1 + 1.0 - x0) / res)) + 1)
    reach = {}
    for column, sgn in (("top", 1.0), ("bottom", -1.0)):
        c = comb.get(column)
        if c is None:
            continue
        side = float(layout["dirs"].get(column, -1.0 if column == "top" else 1.0))
        gated = set(c["gated"])
        blocked = kdb.Region()
        for f in arr.fingers():
            if f.column == column and f.index not in gated:
                blocked.insert(geo.polygon(f.polygon))     # the source / drain fingers stay ungated
        for other, oc in comb.items():
            if other != column:
                blocked.insert(geo.region(oc["polygon"]))
        blocked = geo.sized(blocked.merged(), spacing) if spacing > 0 else blocked.merged()
        if keepout is not None:
            blocked = (blocked + keepout).merged()      # the TMD keep-out already holds its own gap
        bus = np.asarray(arr.to_local(c["bus"]), float)
        u_end = float(bus[:, 0].max() if side > 0 else bus[:, 0].min())
        v_mid = float(bus[:, 1].mean())
        own = geo.region(c["polygon"])
        b_in = float(np.abs(bus[:, 1]).min())         # the bus's inner edge (toward the teeth / the TMD)
        big = 1e4
        # where the lead leaves the bus it comes no nearer the gap than the bus's inner edge (its turned square start
        # would poke out); farther out it may rise
        ua_, ub_ = sorted((u_end - side * 2 * W, u_end + side * 2 * W))
        poke = arr.region_to_chip(geo.region([(ua_, 0.0), (ub_, 0.0), (ub_, sgn * b_in), (ua_, sgn * b_in)]))
        side_box = geo.region([(-big, -big), (big, -big), (big, big), (-big, big)]) - poke
        (sx, sy), = arr.to_chip([(u_end - side * W / 2, v_mid)])
        d = np.array(arr.dir_chip(side, sgn), float)
        d /= np.linalg.norm(d)
        base = float(np.dot(d, (sx, sy)))
        # room for the lead: the distance from each point to the nearest place it may not go (the comb itself is room)
        free = _raster(((allowed - blocked) + own).merged(), x0, y0, res, shape)
        inside_own = _raster(own, x0, y0, res, shape)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res - res / 2

        def lookup(grid, xs, ys):
            """Grid values at the points; 0 off the grid (no room there — the grid covers the allowed area)."""
            ix = np.round((xs - x0) / res).astype(int)
            iy = np.round((ys - y0) / res).astype(int)
            on = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
            out_ = np.zeros(len(xs), grid.dtype)
            out_[on] = grid[iy[on], ix[on]]
            return out_

        def lead_to(q, snap: float, pad: float = 0.0) -> list:
            """The leads toward ``q``: as far as it fits, and — when its end narrows into a corner — stopped where that
            narrowing starts (a wide end). Each is (go, narrowest, mean, end, polygon); ``pad`` µm is kept from every
            obstacle beyond the raster's own (a retry when the exact check finds a rounding overlap)."""
            L = math.hypot(q[0] - sx, q[1] - sy)
            if L <= ds:
                return []
            ux, uy = (q[0] - sx) / L, (q[1] - sy) / L
            s = np.arange(0.0, L + 1e-9, ds)
            xs, ys = sx + ux * s, sy + uy * s
            room = np.minimum(W, 2 * (lookup(dist, xs, ys) - pad))
            room[room >= W - snap] = W                 # raster rounding at the full width (checked exactly below)
            k = int(round(LEAD_SMOOTH_UM / ds))         # the narrowest within ± this: no alternating widths (sawtooth sides)
            if k > 0 and len(room) > 1:
                padded = np.pad(room, k, mode="edge")
                room = np.min(np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1), axis=1)
            on_bus = lookup(inside_own, xs, ys) > 0
            room[on_bus] = W                            # still on the bus
            # it fits up to the first place narrower than the hard minimum, or where it could not leave the bus at its width
            bad = (room < w_min - 1e-9) | (room + slope * s < W - 1e-9)
            n = int(np.argmax(bad)) if bad.any() else len(s)
            ends = [n]
            i = n - 1
            while i > 1 and room[i - 1] > room[i] + 0.005:   # the end narrowing into a corner: also stop before it
                i -= 1
            if i < n - 1:
                ends.append(i + 1)
            out = []
            for m in ends:
                if m < 2:
                    continue
                r = room[:m].copy()
                for j in range(1, m):                  # no steeper than the taper limit, both ways
                    r[j] = min(r[j], r[j - 1] + slope * ds)
                for j in range(m - 2, -1, -1):
                    r[j] = min(r[j], r[j + 1] + slope * ds)
                if r[0] < W - 1e-6 or r.min() < w_min - 1e-9:
                    continue
                r = np.maximum(_straight_runs(s[:m], r), w_min)   # straight sides, never wider than the room
                end = (sx + ux * s[m - 1], sy + uy * s[m - 1])
                part = r[~on_bus[:m]] if (~on_bus[:m]).any() else r
                out.append((float(np.dot(d, end)) - base, float(part.min()), float(part.mean()), end,
                            profile_path((sx, sy), (ux, uy), s[:m], r)))
            return out

        targets = set()                                # where a lead of the hard-minimum, a middle or the bus width can end
        for wt in sorted({w_min, (w_min + W) / 2, W}):
            centres = geo.sized(allowed, -(wt / 2 + 0.005)).merged()
            targets |= {(round(x, 4), round(y, 4)) for poly in geo.to_polys(centres) for x, y in poly}
        ang0 = math.atan2(d[1], d[0])
        for k in range(-40, 41):                       # and a fan about the corner direction, every 2°
            a_ = ang0 + math.radians(2 * k)
            targets.add((round(sx + 200 * math.cos(a_), 4), round(sy + 200 * math.sin(a_), 4)))
        chosen = None
        leads = []
        for q in targets:
            lu, lv = arr.to_local([q])[0]
            if (lu - u_end) * side < 2 * W or lv * sgn < p.gap / 2 + w_min:
                continue                               # not outward along the lead side, or across to the other column's side
            for which, r in enumerate(lead_to(q, 2 * res)):
                if r[0] > W:
                    leads.append((q, which, r))
        if leads:
            best_go = max(t[2][0] for t in leads)
            # the farthest; among those within (bus width − hard minimum) of it — about what pinching the end into a corner
            # gains — the one with the widest narrowest part, then the widest on average
            slack = max(0.25, W - w_min)
            leads.sort(key=lambda t: (t[2][0] < best_go - slack, -round(t[2][1], 3), -round(t[2][2], 3), -t[2][0]))
            for q, which, r in leads[:60]:
                for snap, pad in ((None, None), (0.0, res), (0.0, 2.5 * res), (0.0, 5 * res)):
                    if snap is not None:
                        rs = lead_to(q, snap, pad)
                        r = rs[min(which, len(rs) - 1)] if rs else None
                    if r is not None and _segment_ok(geo.region(r[4]) & side_box, allowed, blocked, own, spacing):
                        chosen = r
                        break
                if chosen is not None:
                    break
        if chosen is None:
            continue
        _go, narrowest, _mean, end, poly = chosen
        lead = ((geo.region(poly) & side_box & allowed) - blocked).merged()     # exactly inside: no rounding slivers
        reg = (own + lead).merged()
        polys = geo.to_polys(reg)
        c["polygon"] = max(polys, key=geo.poly_area)
        ex, ey = end
        c["terminal"] = ((ex, ey), tuple(d))
        c["reach"] = (float(ex), float(ey))
        c["lead_width"] = round(float(narrowest), 3)
        ends = [q for q in poly]
        c["end_width"] = round(float(math.dist(ends[len(ends) // 2 - 1], ends[len(ends) // 2])), 3)
        reach[column] = (float(ex), float(ey), c["lead_width"])
    return comb, reach


def gated_fingers(arr: FingerArray, cg_polys: list[Poly]) -> dict[str, set[int]]:
    """Which fingers a contact gate (generated or drawn) covers: at least half of the finger's tip
    zone (the first ``fingers.min_overlap_um`` from the tip — teeth are fitted to the TMD, so their
    length varies) under it. {"top": {i, …}, "bottom": {…}}."""
    p = arr.params
    zl = min(p.min_overlap, p.cg_tooth)
    cg = geo.regions(cg_polys).merged() if cg_polys else kdb.Region()
    out = {"top": set(), "bottom": set()}
    for column, sgn in (("top", 1.0), ("bottom", -1.0)):
        for i in range(arr.n):
            u = arr.u_of(i)
            zone = geo.region(arr.to_chip([(u - p.width / 2, sgn * p.gap / 2), (u + p.width / 2, sgn * p.gap / 2),
                                           (u + p.width / 2, sgn * (p.gap / 2 + zl)), (u - p.width / 2, sgn * (p.gap / 2 + zl))]))
            if geo.area(zone & cg) >= 0.5 * geo.area(zone):
                out[column].add(i)
    return out


def cg_terminals(arr: FingerArray, cg_polys: list[Poly], electrode_width: float) -> list[dict]:
    """Electrode terminals of a contact gate as it is (generated or drawn): for each piece, its end
    that reaches farthest beyond the array along u (the bus lead), the lead leaving along ±u."""
    out = []
    for k, poly in enumerate(cg_polys):
        loc = np.asarray(arr.to_local(poly), float)
        if len(loc) < 3:
            continue
        v_mid = float(loc[:, 1].mean())
        column = "top" if v_mid > 0 else "bottom"
        lo, hi = float(loc[:, 0].min()), float(loc[:, 0].max())
        side = 1.0 if abs(hi) >= abs(lo) else -1.0
        u_end = hi if side > 0 else lo
        near = loc[np.abs(loc[:, 0] - u_end) < max(electrode_width, 0.5)]
        v = float(near[:, 1].mean()) if len(near) else v_mid
        (tx, ty), = arr.to_chip([(u_end - side * electrode_width / 2, v)])
        out.append({"name": f"CG-{column}" + (f"{k + 1}" if len(cg_polys) > 2 else ""), "start": (tx, ty), "direction": arr.dir_chip(side, 0.0)})
    return out


def back_along(metal: kdb.Region, end: tuple[float, float], length: float, inward: tuple[float, float] | None = None,
               spread_deg: float = 60.0) -> tuple[float, float] | None:
    """The point up to ``length`` µm back from ``end`` along the piece of ``metal`` it ends (a bus or its lead): of the
    directions within ``spread_deg`` of ``inward`` (5° apart), the one whose line stays inside the metal longest (the
    piece's own axis — a turned lead included); shortened where the piece turns or ends. None when nothing lies inside."""
    if length <= 0 or metal.is_empty():
        return None

    def inside(q) -> bool:
        box = geo.region([(q[0] - 0.01, q[1] - 0.01), (q[0] + 0.01, q[1] - 0.01), (q[0] + 0.01, q[1] + 0.01), (q[0] - 0.01, q[1] + 0.01)])
        return (box - metal).is_empty()

    a0 = math.atan2(inward[1], inward[0]) if inward is not None else 0.0
    spread = spread_deg if inward is not None else 180.0
    best, best_len = None, 0.0
    for k in range(-int(spread // 5), int(spread // 5) + 1):
        a = a0 + math.radians(5 * k)
        ux, uy = math.cos(a), math.sin(a)
        good = 0.0
        for j in range(1, 41):                           # walk back in steps, stop where the line leaves the metal
            d = length * j / 40
            if not inside((end[0] + ux * d, end[1] + uy * d)):
                break
            good = d
        if good > best_len + 1e-9 or (abs(good - best_len) < 1e-9 and best is not None and abs(k) < abs(best[2])):
            best, best_len = (ux, uy, k), good
    if best is None or best_len <= 0:
        return None
    return (float(end[0] + best[0] * best_len), float(end[1] + best[1] * best_len))


def gate_contacts(arr: FingerArray, gate_polys: list[Poly], avoid: list[Poly], name: str = "BG") -> tuple[list[dict], list[str]]:
    """Contact squares on a gate flake at its two ends along the array axis (see the module docstring)."""
    p = arr.params
    notes = []
    if not gate_polys:
        return [], [f"{name}: no outline in the chip frame — locate or draw the gate flake first."]
    gate = geo.regions(gate_polys).merged()
    keep_out = geo.sized(geo.regions(avoid).merged(), p.contact_clearance) if avoid else kdb.Region()
    allowed = gate - keep_out
    s = p.contact_size
    found = []
    for side in (-1.0, 1.0):
        # walk along ±u from the array, at every v across the gate, keep the farthest square fully allowed
        loc = [q for poly in gate_polys for q in arr.to_local(poly)]
        us, vs = [q[0] for q in loc], [q[1] for q in loc]
        u_far = max(us) if side > 0 else min(us)
        best = None
        u = u_far - side * s / 2
        while side * u > 0 and best is None:
            for v in np.arange(min(vs) + s / 2, max(vs) - s / 2 + 1e-9, 0.25):
                sq = arr.to_chip([(u - s / 2, v - s / 2), (u + s / 2, v - s / 2), (u + s / 2, v + s / 2), (u - s / 2, v + s / 2)])
                if (geo.region(sq) - allowed).is_empty():
                    if best is None or abs(v) < abs(best[1]):
                        best = (u, float(v), sq)
            u -= side * 0.25
        if best is None:
            notes.append(f"{name}: no {s:g} µm contact fits on the {'+' if side > 0 else '−'}u end of the gate "
                         f"≥ {p.contact_clearance:g} µm from the device.")
            continue
        u, v, sq = best
        (tx, ty), = arr.to_chip([(u, v)])
        found.append({"name": f"{name}{'+' if side > 0 else '-'}", "polygon": sq, "terminal": ((tx, ty), arr.dir_chip(side, 0.0))})
    return found, notes


def electrodes(arr: FingerArray | None, terminals: list[dict], pad_tips: list[tuple[float, float]], width: float,
               used_tips: set[int] | None = None) -> tuple[list[Poly], list[dict], list[str]]:
    """Leads from every terminal to a free pad tip. A terminal is
    {"name", "start": (x, y), "path": [points before leaving] (optional), "direction": (dx, dy)}."""
    from .stage2 import assign_tips
    notes = []
    used = set(used_tips or ())
    free = [i for i in range(len(pad_tips)) if i not in used]
    starts, polys, routes = [], [], []
    for t in terminals:
        last = (t.get("path") or [t["start"]])[-1]
        dx, dy = t["direction"]
        starts.append((last[0] + dx * 2 * width, last[1] + dy * 2 * width))
    centre = arr.center if arr is not None else (sum(s[0] for s in starts) / max(len(starts), 1), sum(s[1] for s in starts) / max(len(starts), 1))
    mapping = assign_tips(starts, [pad_tips[i] for i in free], centre)
    if len(free) < len(terminals):
        notes.append(f"{len(terminals)} electrodes but only {len(free)} free bond-pad leads: some stay unrouted.")
    for t, esc, k in zip(terminals, starts, mapping):
        pts = list(t.get("path") or [t["start"]]) + [esc]
        tip_index = free[k] if k is not None else None
        if tip_index is not None:
            pts.append(tuple(pad_tips[tip_index]))
        clean = [pts[0]] + [q for a, q in zip(pts, pts[1:]) if math.dist(a, q) > 1e-6]
        polys.append(lead_path(clean, width))
        routes.append({"terminal": t["name"], "points_um": [[round(x, 4), round(y, 4)] for x, y in clean],
                       "pad_tip": tip_index, "routed_to_pad": tip_index is not None})
    return [q for q in polys if q], routes, notes


def finger_path(arr: FingerArray, f: Finger, dist_from_tip: float) -> list[tuple[float, float]]:
    """The centre line of a finger from ``dist_from_tip`` (from the tip, along the finger) to its far end."""
    path = [arr.point_at(f, dist_from_tip)]
    if dist_from_tip < arr.params.straight_at(f.column, f.index):
        path.append(f.centreline[1])
    path.append(f.far_end)
    return path


def finger_terminals(arr: FingerArray, fingers: list[Finger] | None = None, land: dict[str, float] | None = None) -> list[dict]:
    """Electrode terminals of the fingers: from the landing point along the finger to its far end. ``land`` = {finger
    name: distance from the tip} where an electrode starts elsewhere than ``land_from_tip`` (off the flakes above)."""
    p = arr.params
    out = []
    for f in fingers or arr.fingers():
        d = (land or {}).get(f.name, p.land_from_tip)
        path = finger_path(arr, f, d)
        out.append({"name": f.name, "start": path[0], "path": path, "direction": f.direction, "land": d})
    return out


# ====================================================================== checks
def check_pattern(shapes: dict[str, list[Poly]], config: LabConfig, others: dict[str, list[Poly]] | None = None) -> list[dict]:
    """Width / spacing of each metal purpose (hard minimum = error, target = warning) and touching
    between different metal purposes of this step and earlier ones (``others``). Returns issue dicts
    {"severity", "code", "message", "where"}."""
    issues = []
    wmin, smin = config.value("ebl.contact_width_min_um"), config.value("ebl.contact_spacing_min_um")
    wt, st = config.value("ebl.electrode_width_target_um"), config.value("ebl.electrode_spacing_target_um")
    rid = {("error", "width"): "ebl.contact_width_min_um", ("error", "space"): "ebl.contact_spacing_min_um",
           ("warning", "width"): "ebl.electrode_width_target_um", ("warning", "space"): "ebl.electrode_spacing_target_um"}
    for purpose, polys in shapes.items():
        if not polys or purpose.startswith("ref_"):
            continue
        reg = geo.regions(polys).merged()
        for limit, sev, kind in ((wmin, "error", "width"), (smin, "error", "space"), (wt, "warning", "width"), (st, "warning", "space")):
            if limit is None:
                continue
            pts = geo.width_violations(reg, limit) if kind == "width" else geo.space_violations(reg, limit)
            if pts:
                what = "narrower than" if kind == "width" else "closer than"
                issues.append({"severity": sev, "code": f"{purpose}_{kind}", "where": pts[:5], "rule": f"rule:{rid[(sev, kind)]}",
                               "message": f"{purpose.replace('_', ' ')}: {len(pts)} place(s) {what} {limit:g} µm "
                                          f"({'hard minimum' if sev == 'error' else 'design target'})."})
                if sev == "error":
                    break
    if wmin is None or smin is None:
        issues.append({"severity": "missing", "code": "rules_missing", "where": [],
                       "message": "Width/spacing minimums (ebl.contact_width_min_um, ebl.contact_spacing_min_um) not set: not checked."})
    allm = dict(others or {})
    for purpose, polys in shapes.items():
        for other, opolys in allm.items():
            if other == purpose or not polys or not opolys:
                continue
            ov = geo.regions(polys).merged() & geo.regions(opolys).merged()
            if not ov.is_empty():
                issues.append({"severity": "info", "code": "overlap", "where": [geo.centroid(q) for q in geo.to_polys(ov)][:5],
                               "message": f"{purpose.replace('_', ' ')} overlaps {other.replace('_', ' ')} "
                                          f"({geo.area(ov):.2f} µm²) — intended where an electrode lands on a finger or gate."})
    return issues
