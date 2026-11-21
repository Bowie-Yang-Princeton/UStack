"""Electrode router (user, 2026-11-20): leads from the device to the bond-pad leads that never cross each other, keep
their spacing, and go around the gate flakes instead of over them.

Each lead (a :class:`Net`) is routed in turn on a grid (``res`` µm) with A* (8 directions, Euclidean steps):

* **hard** obstacles may not be entered: every other lead already routed, the pre-patterned metal of other nets (fingers,
  the contact-gate combs, the bond-pad leads, the landing squares of the other leads, alignment markers) and the
  *uncovered* part of a gate flake (a lead touching it shorts to the gate) — each grown by its clearance + half the lead
  width + half a cell, so a centreline on a free cell keeps the spacing;
* **soft** obstacles are crossed only when there is no way around, at ``soft_penalty`` × the length: gate flakes under
  an insulating flake (leakage through the hBN) and the device (twisted region, device layers).

A net's own metal (what touches its fixed start: its finger, its comb, its gate contact; its target pad lead and landing
square) is not an obstacle for it. The grid path is then pulled taut (the farthest point still visible without entering
a hard cell or more soft cells than the path had) — straight wires at any angle, bent only to go around. A net that
cannot be routed is retried first (rip-up of the others, up to ``retries`` times) before it is reported.

Pure numpy + OpenCV + KLayout; no Qt."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import cv2
import klayout.db as kdb
import numpy as np

from . import geometry as geo

Poly = list[tuple[float, float]]
#: Program choices (not lab values).
ROUTE_RES_UM = 0.5
SOFT_PENALTY = 12.0
#: A hard obstacle relaxed for a lead with no other way costs this many times a soft one (program choice).
RELAX_WEIGHT = 4
#: First the obstacles this close (µm) to a blocked lead's start or end are relaxed for it; all only as a last resort.
RELAX_NEAR_UM = 25.0
#: A lead's own fixed part is an obstacle to its own route beyond this (+ its width) from where the route starts.
SELF_CLEAR_UM = 2.0


@dataclass
class Net:
    name: str
    start: tuple[float, float]               # where the routed part begins (end of the fixed part)
    target: tuple[float, float]              # where it ends (inside the landing square on the pad lead)
    width: float
    own: list[Poly] = field(default_factory=list)       # its fixed metal: obstacles touching it are its own
    group: str = ""
    stub: list[Poly] = field(default_factory=list)      # its fixed wire (along a finger, a bus): never doubled back beside


@dataclass
class Obstacle:
    polys: list[Poly]
    hard: bool
    clearance: float
    owner: str | None = None                 # a net name: not an obstacle for that net
    relax: bool = False                      # hard, but soft for the nets in ``route_nets(relax_nets=…)`` (a last resort)
    ownable: bool = True                     # metal: a net whose fixed part touches it owns it (a bubble never is)
    exclusive: bool = False                  # crossed by at most one net (a stray flake may conduct: two leads over it would
                                             # short; user, 2026-11-20) — once a net touches it, it is hard for the others


def wire(points: list[tuple[float, float]], width: float) -> kdb.Region:
    """A lead of ``width`` along ``points``: straight pieces with flat ends, round joints (no mitre spikes)."""
    reg = kdb.Region()
    h = width / 2
    for a, b in zip(points, points[1:]):
        L = math.dist(a, b)
        if L < 1e-9:
            continue
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx, ny = -uy * h, ux * h
        reg.insert(geo.polygon([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]))
    r = h / math.cos(math.pi / 16)
    for q in points[1:-1]:
        reg.insert(geo.polygon([(q[0] + r * math.cos(2 * math.pi * k / 16), q[1] + r * math.sin(2 * math.pi * k / 16)) for k in range(16)]))
    return reg.merged()


class _Grid:
    def __init__(self, box: tuple[float, float, float, float], res: float):
        self.x0, self.y0 = box[0], box[1]
        self.res = res
        self.nx = int(math.ceil((box[2] - box[0]) / res)) + 1
        self.ny = int(math.ceil((box[3] - box[1]) / res)) + 1

    def cell(self, p) -> tuple[int, int]:
        return int(round((p[1] - self.y0) / self.res)), int(round((p[0] - self.x0) / self.res))

    def point(self, iy: int, ix: int) -> tuple[float, float]:
        return (self.x0 + ix * self.res, self.y0 + iy * self.res)

    def inside(self, iy: int, ix: int) -> bool:
        return 0 <= iy < self.ny and 0 <= ix < self.nx

    def crop_mask(self, polys: list[Poly], grow: float):
        """(iy0, ix0, mask) of ``polys`` grown by ``grow`` µm (disk), cropped to their box; None when off the grid."""
        pts = [np.asarray(p, float) for p in polys if len(p) >= 3]
        if not pts:
            return None
        allp = np.vstack(pts)
        r = max(int(math.ceil(grow / self.res)), 0)
        ix0 = int(math.floor((allp[:, 0].min() - self.x0) / self.res)) - r - 1
        iy0 = int(math.floor((allp[:, 1].min() - self.y0) / self.res)) - r - 1
        ix1 = int(math.ceil((allp[:, 0].max() - self.x0) / self.res)) + r + 2
        iy1 = int(math.ceil((allp[:, 1].max() - self.y0) / self.res)) + r + 2
        cx0, cy0, cx1, cy1 = max(ix0, 0), max(iy0, 0), min(ix1, self.nx), min(iy1, self.ny)
        if cx0 >= cx1 or cy0 >= cy1:
            return None
        h, w = iy1 - iy0, ix1 - ix0
        m = np.zeros((h, w), np.uint8)
        for p in pts:                                 # one at a time: fillPoly of several leaves their overlap empty
            cv2.fillPoly(m, [np.round((p - (self.x0, self.y0)) / self.res - (ix0, iy0)).astype(np.int32)], 1)
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m, k)
        return cy0, cx0, m[cy0 - iy0:cy1 - iy0, cx0 - ix0:cx1 - ix0]


def _add(arr: np.ndarray, crop, sign: int = 1) -> None:
    if crop is None:
        return
    iy, ix, m = crop
    arr[iy:iy + m.shape[0], ix:ix + m.shape[1]] += (m.astype(arr.dtype) * sign)


_STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _astar(hard: np.ndarray, soft: np.ndarray, s: tuple[int, int], t: tuple[int, int], penalty: float, res: float,
           max_expand: int = 3_000_000):
    """A* on the grid (8 directions, octile heuristic), on flat arrays with a blocked border (no bounds tests)."""
    ny, nx = hard.shape
    W = nx + 2
    blocked = np.ones((ny + 2, W), np.uint8)
    blocked[1:-1, 1:-1] = hard
    cost = np.ones((ny + 2, W), np.float64)
    cost[1:-1, 1:-1] += penalty * soft
    B = blocked.ravel().tobytes()
    C = cost.ravel().tolist()
    N = (ny + 2) * W
    g = [math.inf] * N
    came = bytearray(N)
    closed = bytearray(N)
    si = (s[0] + 1) * W + s[1] + 1
    ti = (t[0] + 1) * W + t[1] + 1
    ty, tx = t[0] + 1, t[1] + 1
    r2 = math.sqrt(2.0)
    steps = [(-W, 1.0), (W, 1.0), (-1, 1.0), (1, 1.0), (-W - 1, r2), (-W + 1, r2), (W - 1, r2), (W + 1, r2)]

    def h(i):
        dy, dx = abs(i // W - ty), abs(i % W - tx)
        return (max(dy, dx) + (r2 - 1.0) * min(dy, dx)) * 1.001

    g[si] = 0.0
    heap = [(h(si), 0.0, si)]
    push, pop = heapq.heappush, heapq.heappop
    n = 0
    while heap:
        _f, gc, i = pop(heap)
        if closed[i]:
            continue
        closed[i] = 1
        if i == ti:
            path = [i]
            while i != si:
                i -= steps[came[i]][0]
                path.append(i)
            return [(j // W - 1, j % W - 1) for j in reversed(path)]
        n += 1
        if n > max_expand:
            return None
        for k in range(8):
            d, c = steps[k]
            j = i + d
            if B[j] or closed[j]:
                continue
            ng = gc + c * C[j]
            if ng < g[j]:
                g[j] = ng
                came[j] = k
                push(heap, (ng + h(j), ng, j))
    return None


def _visible(hard, soft, a, b, soft_budget: float) -> bool:
    """The straight line a → b (cells) enters no hard cell and crosses no more soft length (cells) than ``soft_budget``."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = int(L * 2) + 2
    yf, xf = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
    ys, xs = np.rint(yf).astype(int), np.rint(xf).astype(int)
    if hard[ys, xs].any():                          # the nearest cell of each sample (the obstacles carry ¾ cell to spare)
        return False
    soft_len = float(soft[ys, xs].sum()) * L / max(n - 1, 1)
    return soft_len <= soft_budget + 1.0


def _taut(path, hard, soft):
    """The grid path pulled taut: from each corner, the farthest path cell still visible (no hard cell, no more soft
    length than the path itself crosses there)."""
    steps = [0.0] + [math.hypot(b[0] - a[0], b[1] - a[1]) * (1 if soft[b] else 0) for a, b in zip(path, path[1:])]
    softc = np.cumsum(steps)
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = i + 1
        k = i + 2
        while k < n:                                  # walk forward while still visible (first failure stops it)
            if _visible(hard, soft, path[i], path[k], float(softc[k] - softc[i])):
                j = k
                k += 1 if k - i < 8 else max(1, (k - i) // 8)
            else:
                break
        # step back to the last visible one when the stride overshot
        while j < n - 1 and j + 1 < k and _visible(hard, soft, path[i], path[j + 1], float(softc[j + 1] - softc[i])):
            j += 1
        out.append(path[j])
        i = j
    return out


def _soft_len(soft, a, b) -> float:
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = int(L * 2) + 2
    ys = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    xs = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return float(soft[ys, xs].sum()) * L / max(n - 1, 1)


def _drop_corners(corners, hard, soft):
    """Corners whose neighbours see each other (no hard cell, no more soft length) are dropped, until none is."""
    out = list(corners)
    changed = True
    while changed and len(out) > 2:
        changed = False
        i = 1
        while i < len(out) - 1:
            a, b, c = out[i - 1], out[i], out[i + 1]
            if _visible(hard, soft, a, c, _soft_len(soft, a, b) + _soft_len(soft, b, c) - 1.0):
                del out[i]
                changed = True
            else:
                i += 1
    return out


def _part_near(G: "_Grid", crop, points, radius: float):
    """The cells of ``crop`` within ``radius`` µm of one of ``points``."""
    iy, ix, m = crop
    ys, xs = np.mgrid[iy:iy + m.shape[0], ix:ix + m.shape[1]]
    X, Y = G.x0 + xs * G.res, G.y0 + ys * G.res
    near = np.zeros(m.shape, bool)
    for x, y in points:
        near |= (X - x) ** 2 + (Y - y) ** 2 <= radius * radius
    return iy, ix, (m * near).astype(m.dtype)


def _region_of(ob: Obstacle) -> kdb.Region:
    reg = getattr(ob, "_reg", None)
    if reg is None:
        reg = ob._reg = geo.regions(ob.polys).merged()
    return reg


def _part_far(G: "_Grid", crop, point, radius: float):
    """The cells of ``crop`` farther than ``radius`` µm from ``point``."""
    iy, ix, m = crop
    ys, xs = np.mgrid[iy:iy + m.shape[0], ix:ix + m.shape[1]]
    X, Y = G.x0 + xs * G.res, G.y0 + ys * G.res
    far = (X - point[0]) ** 2 + (Y - point[1]) ** 2 > radius * radius
    return iy, ix, (m * far).astype(m.dtype)


def _near(ob: Obstacle, points, radius: float) -> bool:
    """Whether the obstacle comes within ``radius`` µm of one of ``points`` (its box first, then its outline)."""
    xs = [q[0] for p in ob.polys for q in p]
    ys = [q[1] for p in ob.polys for q in p]
    reg = None
    for x, y in points:
        dx = max(min(xs) - x, 0.0, x - max(xs))
        dy = max(min(ys) - y, 0.0, y - max(ys))
        if math.hypot(dx, dy) > radius:
            continue
        reg = reg if reg is not None else geo.regions(ob.polys)
        if geo.distance_to_region(x, y, reg) <= radius:
            return True
    return False


def route_nets(nets: list[Net], obstacles: list[Obstacle], spacing: float, box: tuple[float, float, float, float] | None = None,
               res: float = ROUTE_RES_UM, soft_penalty: float = SOFT_PENALTY, retries: int = 2,
               extra: dict[str, list[Poly]] | None = None, relax_nets: set[str] | None = None,
               relax_radius: float = math.inf, both_ways: bool = True):
    """Route ``nets`` in order. Returns ({net name: [points] or None}, {net name: soft cells crossed (µm)}, notes).
    ``extra`` = net name → metal that belongs to it beyond its wire (landing square, gate contact): an obstacle for the
    others from the start."""
    if not nets:
        return {}, {}, []
    extra = extra or {}
    w = max(n.width for n in nets)
    if box is None:
        pts = [q for n in nets for q in (n.start, n.target)]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        box = (min(xs) - 25, min(ys) - 25, max(xs) + 25, max(ys) + 25)
    G = _Grid(box, res)
    slack = res * 0.75                               # a line between free cell centres stays this clear
    items: list[tuple[Obstacle, object]] = []
    for ob in obstacles:
        crop = G.crop_mask(ob.polys, ob.clearance + w / 2 + slack)
        if crop is not None:
            items.append((ob, crop))
    for n in nets:                                    # each lead's fixed part (along its finger …) is in the others' way
        polys = list(n.own) + list(extra.get(n.name, []))
        crop = G.crop_mask(polys, spacing + w / 2 + slack) if polys else None
        if crop is not None:
            items.append((Obstacle(polys, True, spacing, n.name), crop))
        if n.stub:                                    # … and in its own way, but round where its route starts: a lead
            own_crop = G.crop_mask(list(n.stub), spacing + w / 2 + slack)  # never doubles back beside its own stub
            if own_crop is not None:
                far = _part_far(G, own_crop, n.start, SELF_CLEAR_UM + w)
                items.append((Obstacle(list(n.stub), True, spacing, f"self:{n.name}", ownable=False), far))
    for name, polys in extra.items():
        if any(n.name == name for n in nets):
            continue
        crop = G.crop_mask(polys, spacing + w / 2 + slack)
        if crop is not None:
            items.append((Obstacle(polys, True, spacing, name), crop))
    # each net's own items: owned by it, or touching its fixed metal
    own_of: dict[str, list[int]] = {}
    for n in nets:
        mine = geo.sized(geo.regions(n.own), 0.05) if n.own else kdb.Region()
        ids = []
        for i, (ob, _c) in enumerate(items):
            if ob.owner == n.name or (ob.owner is None and ob.hard and ob.ownable and not mine.is_empty()
                                      and not (geo.regions(ob.polys) & mine).is_empty()):
                ids.append(i)
        own_of[n.name] = ids
    # sweep order round the device, starting after the widest gap between the targets: each lead then runs beside the
    # previous one instead of cutting in front of a neighbour's exit
    cx = sum(n.start[0] for n in nets) / len(nets)
    cy = sum(n.start[1] for n in nets) / len(nets)
    ang = {n.name: math.atan2(n.target[1] - cy, n.target[0] - cx) for n in nets}
    srt = sorted(nets, key=lambda n: ang[n.name])
    if len(srt) > 2:
        gaps = [(ang[srt[(i + 1) % len(srt)].name] - ang[srt[i].name]) % (2 * math.pi) for i in range(len(srt))]
        k = int(np.argmax(gaps)) + 1
        srt = srt[k:] + srt[:k]
    exclusive = [i for i, (ob, _c) in enumerate(items) if ob.exclusive]
    near_cache: dict = {}

    def run(order):
        for attempt in range(retries + 1):
            hard = np.zeros((G.ny, G.nx), np.int16)
            soft = np.zeros((G.ny, G.nx), np.int16)
            for ob, crop in items:
                _add(hard if ob.hard else soft, crop)
            paths: dict[str, list | None] = {}
            crossed: dict[str, float] = {}
            failed = []
            claimed: dict[int, str] = {}                 # exclusive obstacle → the net that touches it
            for n in order:
                h, s_ = hard.copy(), soft.copy()
                for i in own_of[n.name]:
                    ob, crop = items[i]
                    _add(h if ob.hard else s_, crop, -1)
                for i, who in claimed.items():           # touched by another lead already: not for this one at all
                    ob, crop = items[i]
                    if who != n.name and not ob.hard:
                        _add(s_, crop, -1)
                        _add(h, crop, 1)
                if relax_nets and n.name in relax_nets:   # this net may cross the relaxable obstacles (at a cost)
                    for i, (ob, crop) in enumerate(items):
                        if not (ob.hard and ob.relax and ob.owner != n.name) or claimed.get(i, n.name) != n.name:
                            continue
                        if relax_radius == math.inf:
                            part = crop
                        else:
                            ck = (i, n.name, relax_radius)
                            if ck not in near_cache:
                                near_cache[ck] = (_part_near(G, crop, (n.start, n.target), relax_radius)   # only round the ends
                                                  if _near(ob, (n.start, n.target), relax_radius) else None)
                            part = near_cache[ck]
                            if part is None:
                                continue
                        _add(h, part, -1)
                        _add(s_, part, RELAX_WEIGHT)
                hb, sb = h > 0, s_ > 0
                a, b = G.cell(n.start), G.cell(n.target)
                if not (G.inside(*a) and G.inside(*b)):
                    paths[n.name] = None
                    failed.append(n)
                    continue
                # a start / end hemmed in by its own neighbourhood: free a lead-wide disk round it (its own fixed metal is there)
                for c in (a, b):
                    r = max(1, int(math.ceil(w / 2 / res)))
                    y0, y1, x0, x1 = max(c[0] - r, 0), min(c[0] + r + 1, G.ny), max(c[1] - r, 0), min(c[1] + r + 1, G.nx)
                    if hb[c]:
                        hb[y0:y1, x0:x1] = False
                # reachable at all? (one flood fill in C: a hopeless search would explore the whole grid first)
                free = (~hb).astype(np.uint8)
                _n, lab = cv2.connectedComponents(free, connectivity=8)
                weight = np.minimum(s_, RELAX_WEIGHT).astype(np.float64)    # a relaxed obstacle costs more than a covered gate
                cells = _astar(hb, weight, a, b, soft_penalty, res) if lab[a] != 0 and lab[a] == lab[b] else None
                if cells is None:
                    paths[n.name] = None
                    failed.append(n)
                    continue
                corners = _drop_corners(_taut(cells, hb, sb), hb, sb)
                pts = [n.start] + [G.point(*c) for c in corners[1:-1]] + [n.target]
                paths[n.name] = pts
                crossed[n.name] = float(sum(1 for c in cells if sb[c]) * res)
                lead = wire(pts, n.width)
                crop = G.crop_mask(geo.to_polys(lead), spacing + w / 2 + slack)
                _add(hard, crop)
                ly, lx, lm = crop if crop is not None else (0, 0, np.zeros((0, 0)))
                for i in exclusive:                      # the exclusive obstacles it touches are its own from now on
                    if i in claimed:
                        continue
                    ob, (oy, ox, om) = items[i]
                    if oy > ly + lm.shape[0] or ly > oy + om.shape[0] or ox > lx + lm.shape[1] or lx > ox + om.shape[1]:
                        continue                         # boxes apart: cannot touch
                    if not (_region_of(ob) & lead).is_empty():
                        claimed[i] = n.name
            if not failed or attempt == retries:
                notes = [f"{n.name}: no way to its bond-pad lead without crossing another lead or metal." for n in failed]
                return paths, crossed, notes
            order = failed + [n for n in order if n not in failed]       # rip up: the blocked ones first
        return paths, crossed, notes

    best = None
    for start in ((srt, srt[::-1]) if both_ways else (srt,)):      # both ways round; the better result is kept
        got = run(list(start))
        p_, c_, n_ = got
        key = (len(n_), sum(c_.values()), sum(sum(math.dist(a_, b_) for a_, b_ in zip(q, q[1:])) for q in p_.values() if q))
        if best is None or key < best[0]:
            best = (key, got)
        if not n_:                                    # every lead routed: the other way round is not needed
            break
    return best[1]

