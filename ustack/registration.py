"""Place microscope images of a fabricated chip in the chip (GDS) frame, and find already-traced
flakes in them (Qt-free).

Used by the Stage 2 workflow: for each EBL step the user uploads images of the chip at several
magnifications (5x / 10x / 20x / 50x / 100x). The lowest magnification is aligned to the bond-pad
GDS (:func:`align_to_gds`), each higher one is chained onto an aligned lower one
(:func:`align_to_image`) and, when GDS shapes are visible in it, refined on the GDS again
(:func:`align_images` does all of this). Then each flake outline traced earlier on its own
single-flake image (µm, arbitrary origin / rotation, possibly mirrored by the transfer) is found in
an aligned high-magnification image (:func:`locate_outline`), which fixes the flake relative to the
bond pads.

Transforms are :class:`ustack.alignment.Similarity` (image px → chip µm, y flipped). Every
returned similarity carries two extra attributes (not part of ``to_json``):

* ``score`` — the normalised cross-correlation (NCC, −1…1) of the final fit at the finest level;
* ``info`` — a dict with ``method``, ``margin`` (best score − best *different* candidate at the
  first refinement level, see below; with ``init`` it is just the score), ``seconds`` and
  per-level details.

``rms_um`` stays ``None`` (there are no point pairs).

Algorithm of the two aligners (:func:`_align`, :func:`_search`, :func:`_refine`): a
coarse-to-fine search over (angle, scale, translation). At each level the *image* (a working copy
of a given size) is the template and the *reference* (the GDS rendered as a filled mask, or the
aligned reference image) is drawn into a canvas in a frame with the candidate angle and scale, so
that one call of ``cv2.matchTemplate`` (``TM_CCOEFF_NORMED``) scores every translation at once.

1. Global level (image ``LEVEL0_PX`` px on its long side; when the search region is large an
   extra coarser level is put first so that the canvas stays near ``CANVAS0_MAX_PX`` px, but never
   coarser than the typical GDS feature near the site): every angle in the allowed range (step ≈
   1.5 px of arc at half the template diagonal) × every scale within ``scale_tol`` (step ≈ 3 px);
   the ``TOP_K`` best distinct peaks are kept.
2. Each next level (``LEVEL1_PX``, ``LEVEL2_PX``; never finer than the reference's own pixels for
   image → image): every kept candidate is refined by a coordinate search over angle and scale
   (± the previous level's step) with the translation free within a few pixels; the best one
   continues. ``margin`` = best score − the best candidate that puts the image centre more than
   ``DISTINCT_FRAC`` × the field diagonal away or is rotated more than ``DISTINCT_DEG``. Angle,
   scale and translation are interpolated (parabola through the best score and its neighbours).

Image features. Against the GDS the image is turned into a *metal map*: the colour distance of
every pixel from the substrate (substrate colour = the main colour of the image border with the
illumination fall-off removed, :func:`ustack.imaging._substrate_field`). After the middle level the
metal colour is measured under the rendered GDS and the map becomes the closeness to that colour
(projection on the substrate → metal colour axis, damped away from it), which removes most flakes.
Map and GDS mask are compared through their x/y gradients (``GDS_FEATURE = "grad"``): only edges
count, so a large uniform flake weighs little — necessary at 100x, where the markers are small and
the flakes large. Against another image both are reduced to band-pass filtered CIELab (local
contrast, independent of the illumination and of the objective's colour cast).

:func:`locate_outline` is an oriented chamfer matcher, described in its docstring.

Thresholds were checked on the lab's tWSe2 batch-1 images (``tools/registration_benchmark.py``,
``docs/registration_benchmark.md``). They are program choices for *measuring*, not lab values.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .alignment import Similarity
from .imaging import _substrate_field

# ---------------------------------------------------------------------------- constants
LEVEL0_PX = 128          # long side of the image at the global search level
LEVEL1_PX = 560          # middle level
LEVEL2_PX = 1600         # finest level (6000-px images are never used at full size)
CANVAS0_MAX_PX = 480     # the global level's canvas side is kept near this (a coarser level is added for large regions)
TOP_K = 5                # distinct global peaks refined at the next level (also used by locate_outline)
DISTINCT_FRAC = 0.05     # a candidate is "different" if it moves the image centre > this × field diagonal
DISTINCT_DEG = 2.0       # ... or is rotated more than this
MAX_FILL_FRAC = 0.5      # GDS shapes with an area above this fraction of the image field are ignored (chip outline, write field)

GDS_SCORE_MIN = 0.20     # align_to_gds: final NCC below this → ValueError
GDS_MARGIN_MIN = 0.02    # align_to_gds: best − best different candidate (first refinement level) below this → ValueError (ambiguous)
IMG_SCORE_MIN = 0.25     # align_to_image: final NCC below this → ValueError
IMG_COARSE_MAX_K = 1.6   # align_to_image: the extra coarse level is at most this much coarser than LEVEL0_PX
IMG_MARGIN_MIN = 0.03    # align_to_image: ambiguity margin

REFINE_GDS_SCORE_MIN = 0.30   # align_images: a chained image is re-fitted on the GDS only if that fit scores at least this
REFINE_GDS_MAX_SHIFT_FRAC = 0.03  # ... and moves the image centre less than this × field diagonal (else the chain is kept)

OUTLINE_SCORE_MIN = 0.45  # locate_outline: soft fraction of the outline on a matching edge below this → ValueError
OUTLINE_MARGIN_MIN = 0.10  # locate_outline: best − best different pose below this → ValueError (ambiguous)
OUTLINE_STEP0_DEG = 3.0   # coarse rotation step of the outline search
OUTLINE_COARSE_DIV = 64   # coarse raster: outline size / this (µm per px)
OUTLINE_FINE_DIV = 240    # fine raster (edges are found here): outline size / this, never finer than the image
OUTLINE_TAU_PX = 2.5      # chamfer truncation (px of the level's raster)
FRAME_EDGE_FRAC = 0.01   # outline points this close (× field size) to the source image's edge are not scored
OUTLINE_BINS = 6          # edge-orientation bins (mod 180°)


# ---------------------------------------------------------------------------- similarity helpers
def _lin(sim: Similarity) -> np.ndarray:
    a = math.radians(sim.angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    P = np.diag([-1.0 if sim.mirror else 1.0, -1.0])
    return sim.scale * R @ P


def _affine(sim: Similarity) -> np.ndarray:
    """3×3 matrix: chip = M · [x, y, 1]."""
    M = np.eye(3)
    M[:2, :2] = _lin(sim)
    M[:2, 2] = [sim.tx, sim.ty]
    return M


def _rescaled(sim: Similarity, k: float, c: float) -> Similarity:
    """S' with S'(p) = S(k·p + c) (change of pixel grid)."""
    t = sim.apply([[c, c]])[0]
    return Similarity(sim.scale * k, sim.angle_deg, float(t[0]), float(t[1]), sim.mirror)


def _pivot(sim: Similarity, px: tuple[float, float], scale: float, angle: float) -> Similarity:
    """Same mirror, new scale / angle, pixel ``px`` mapped to the same chip point as by ``sim``."""
    X = sim.apply([px])[0]
    s = Similarity(scale, angle, 0.0, 0.0, sim.mirror)
    t = X - s.apply([px])[0]
    return Similarity(scale, angle, float(t[0]), float(t[1]), sim.mirror)


def _shifted(sim: Similarity, d: tuple[float, float]) -> Similarity:
    """S'(p) = S(p + d)."""
    t = sim.apply([d])[0]
    return Similarity(sim.scale, sim.angle_deg, float(t[0]), float(t[1]), sim.mirror)


def _to_full(sim_w: Similarity, f: float) -> Similarity:
    """Working-copy similarity (x_full = (x_w + 0.5)/f − 0.5) → full-image similarity."""
    return _rescaled(sim_w, f, 0.5 * f - 0.5)


def _to_work(sim: Similarity, f: float) -> Similarity:
    return _rescaled(sim, 1.0 / f, 0.5 / f - 0.5)


def _with_meta(sim: Similarity, score: float, info: dict) -> Similarity:
    sim.score = float(score)  # type: ignore[attr-defined]
    sim.info = info           # type: ignore[attr-defined]
    return sim


# ---------------------------------------------------------------------------- images
def _shrink(img: np.ndarray, f: float) -> np.ndarray:
    """Isotropic working copy with the exact pixel mapping x_full = (x_w + 0.5)/f − 0.5 (no per-axis
    rounding of the scale). Anti-aliased by an area resize to about twice the size first."""
    H, W = img.shape[:2]
    w_t, h_t = max(1, int(W * f)), max(1, int(H * f))
    if f >= 1.0:
        a, sx, sy = img, 1.0, 1.0
    else:
        wa, ha = min(W, 2 * w_t), min(H, 2 * h_t)
        a = cv2.resize(img, (wa, ha), interpolation=cv2.INTER_AREA) if (wa, ha) != (W, H) else img
        sx, sy = W / wa, H / ha
    M = np.array([[1 / (f * sx), 0, 0.5 / (f * sx) - 0.5], [0, 1 / (f * sy), 0.5 / (f * sy) - 0.5]])
    return cv2.warpAffine(a, M, (w_t, h_t), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _lab(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _contrast_map(work: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour distance from the substrate (0…1, clipped at the 99.5th percentile), and the
    substrate-flattened Lab image."""
    lab = _lab(work)
    field, _ = _substrate_field(lab)
    diff = lab - field
    d = np.linalg.norm(diff, axis=2)
    top = max(float(np.percentile(d, 99.5)), 1e-3)
    return np.clip(d / top, 0, 1).astype(np.float32), diff


def _metal_map(diff: np.ndarray, metal: np.ndarray) -> np.ndarray:
    """Closeness to the metal colour (Lab difference from the substrate ``metal``): projection on the
    substrate → metal axis (0…1), damped by the distance from that axis."""
    n = float(np.linalg.norm(metal))
    if n < 1e-3:
        return np.zeros(diff.shape[:2], np.float32)
    u = metal / n
    proj = diff @ u
    perp = np.linalg.norm(diff - proj[..., None] * u, axis=2)
    m = np.clip(proj / n, 0, 1.25) * np.exp(-0.5 * (perp / (0.35 * n)) ** 2)
    return m.astype(np.float32)


def _bandpass(work: np.ndarray, hp_sigma: float = 12.0, lp_sigma: float = 1.0) -> np.ndarray:
    """Band-pass filtered CIELab, each channel scaled by its robust spread (3 channels, float32)."""
    lab = _lab(work)
    out = np.empty_like(lab)
    for c in range(3):
        ch = lab[..., c]
        bp = cv2.GaussianBlur(ch, (0, 0), lp_sigma) - cv2.GaussianBlur(ch, (0, 0), hp_sigma)
        s = 1.4826 * float(np.median(np.abs(bp))) + 1e-3
        out[..., c] = np.clip(bp / s, -12, 12)
    return out


# ---------------------------------------------------------------------------- GDS
def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _gds_shapes(gds_polys: dict, layers, max_area: float | None = None) -> list[np.ndarray]:
    out = []
    for key in layers:
        for poly in gds_polys.get(tuple(key), []):
            p = np.asarray(poly, float).reshape(-1, 2)
            if len(p) < 3:
                continue
            if max_area is not None and _poly_area(p) > max_area:
                continue
            out.append(p)
    return out


def _fill(shapes: list[np.ndarray], to_px: Callable[[np.ndarray], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    if not shapes:
        return m
    allp = to_px(np.concatenate(shapes))
    ends = np.cumsum([len(p) for p in shapes])[:-1]
    margin = 4 * (w + h)
    pts = []
    for q in np.split(allp, ends):
        if q[:, 0].max() < -1 or q[:, 1].max() < -1 or q[:, 0].min() > w + 1 or q[:, 1].min() > h + 1:
            continue
        q = np.clip(q, -margin, margin + max(w, h))
        pts.append(np.round(q * 8).astype(np.int32))
    if pts:
        cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_AA, shift=3)
    return m


def render_gds(polys_by_layer: dict, layers: list[tuple[int, int]], sim: Similarity, shape: tuple[int, int]) -> np.ndarray:
    """uint8 mask (0/255, anti-aliased edges) of the given GDS layers drawn in the pixels of an image
    of ``shape`` (h, w[, …]) placed by ``sim`` — for checks and overlays."""
    h, w = shape[:2]
    return _fill(_gds_shapes(polys_by_layer, layers), sim.inverse, (h, w))


# ---------------------------------------------------------------------------- matching core
_NOISE = np.random.default_rng(12345).normal(0.0, 1.0, (4096, 4096)).astype(np.float32)


def _ncc(canvas: np.ndarray, templ: np.ndarray) -> np.ndarray:
    """``TM_CCOEFF_NORMED``. OpenCV returns arbitrary values (up to ±1) for windows of constant
    canvas; a fixed faint noise (1e-4 of the feature range) makes those score ≈ 0 instead."""
    h, w = canvas.shape[:2]
    if h <= _NOISE.shape[0] and w <= _NOISE.shape[1]:
        noise = _NOISE[:h, :w]
    else:
        noise = np.random.default_rng(12345).normal(0.0, 1.0, (h, w)).astype(np.float32)
    canvas = canvas.astype(np.float32, copy=True)
    if canvas.ndim == 3:
        canvas += 1e-4 * noise[..., None]
    else:
        canvas += 1e-4 * noise
    r = cv2.matchTemplate(canvas, templ.astype(np.float32, copy=False), cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _peaks(r: np.ndarray, n: int, radius: int) -> list[tuple[float, int, int]]:
    r = r.copy()
    out = []
    for _ in range(n):
        _, mx, _, (x, y) = cv2.minMaxLoc(r)
        if mx <= 0:
            break
        out.append((float(mx), x, y))
        r[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1] = -1.0
    return out


def _subpix(r: np.ndarray, x: int, y: int) -> tuple[float, float]:
    def para(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-12 else max(-0.5, min(0.5, 0.5 * (a - c) / den))

    dx = para(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    dy = para(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + dx, y + dy


def _frame_over(region: tuple[float, float, float, float], scale: float, angle: float, mirror: bool,
                pad: tuple[int, int]) -> tuple[Similarity, tuple[int, int]]:
    """Canvas frame C (canvas px → chip µm) with the given scale / angle covering the chip box
    ``region`` (x0, y0, x1, y1) plus ``pad`` (w, h) px on every side."""
    x0, y0, x1, y1 = region
    C0 = Similarity(scale, angle, 0.0, 0.0, mirror)
    q = C0.inverse([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    qmin = q.min(0) - np.asarray(pad, float)
    qmax = q.max(0) + np.asarray(pad, float)
    C = _shifted(C0, (float(qmin[0]), float(qmin[1])))
    size = (int(math.ceil(qmax[1] - qmin[1])) + 1, int(math.ceil(qmax[0] - qmin[0])) + 1)
    return C, size


Render = Callable[[Similarity, tuple[int, int]], np.ndarray]


def _search(templ: np.ndarray, render: Render, region, scale_w: float, mirror: bool,
            angles: np.ndarray, scales: np.ndarray, top_k: int) -> list[dict]:
    """Level 0: every (angle, scale) in the grid, the best translations of each. ``scale_w`` =
    nominal µm per template px; ``scales`` are factors on it."""
    th, tw = templ.shape[:2]
    cands = []
    for s in scales:
        for a in angles:
            C, size = _frame_over(region, scale_w * s, float(a), mirror, (tw, th))
            canvas = render(C, size)
            if canvas.shape[0] <= th or canvas.shape[1] <= tw:
                continue
            r = _ncc(canvas, templ)
            for sc, x, y in _peaks(r, 3, max(3, min(th, tw) // 4)):
                cands.append({"score": sc, "angle": float(a), "scale": scale_w * s,
                              "sim_w": _shifted(C, (float(x), float(y)))})
    cands.sort(key=lambda c: -c["score"])
    out = []
    diag = math.hypot(tw, th)
    for c in cands:
        ctr = c["sim_w"].apply([[tw / 2, th / 2]])[0]
        if any(_distinct(c, o, ctr, diag * c["scale"]) is False for o in out):
            continue
        c["centre"] = ctr
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def _distinct(a: dict, b: dict, ctr_a, diag_um: float) -> bool:
    ctr_b = b["centre"]
    d = math.hypot(ctr_a[0] - ctr_b[0], ctr_a[1] - ctr_b[1])
    da = abs((a["angle"] - b["angle"] + 180) % 360 - 180)
    return d > DISTINCT_FRAC * diag_um or da > DISTINCT_DEG


def _refine(templ: np.ndarray, render: Render, sim_w: Similarity, angle_half: float, angle_step: float,
            scale_half: float, scale_step: float, win: int, scale_bounds: tuple[float, float]) -> dict:
    """Local search around ``sim_w`` (template px → chip): a grid of angles × scale factors pivoting
    about the template centre, translation within ±``win`` px; the best is interpolated."""
    th, tw = templ.shape[:2]
    ctr = (tw / 2.0, th / 2.0)
    na = int(round(angle_half / angle_step)) if angle_step > 0 else 0
    ns = int(round(scale_half / scale_step)) if scale_step > 0 else 0
    grid = {}

    def evaluate(a: float, s: float):
        S = _pivot(sim_w, ctr, s, a)
        C = _shifted(S, (-win, -win))
        canvas = render(C, (th + 2 * win, tw + 2 * win))
        r = _ncc(canvas, templ)
        _, mx, _, (x, y) = cv2.minMaxLoc(r)
        xs, ys = _subpix(r, x, y)
        return float(mx), _shifted(C, (xs, ys)), (x, y), r

    def ev(i: int, j: int):
        if (i, j) not in grid:
            s = sim_w.scale * (1 + i * scale_step)
            if abs(i) > ns or abs(j) > na or not (scale_bounds[0] <= s <= scale_bounds[1]):
                grid[(i, j)] = (-2.0, None)
            else:
                sc, S, _, _ = evaluate(sim_w.angle_deg + j * angle_step, s)
                grid[(i, j)] = (sc, S)
        return grid[(i, j)]

    # coordinate search: the angle line, the scale line through the best angle, then hill-climb
    i, j = 0, max(range(-na, na + 1), key=lambda jj: ev(0, jj)[0])
    i = max(range(-ns, ns + 1), key=lambda ii: ev(ii, j)[0])
    for _ in range(2 * (na + ns) + 2):
        nb = max(((i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))), key=lambda k: ev(*k)[0])
        if ev(*nb)[0] <= ev(i, j)[0]:
            break
        i, j = nb
    sc, S = ev(i, j)
    if S is None:
        raise ValueError("Scale outside the allowed range")
    grid = {k: v[0] for k, v in grid.items() if v[1] is not None}

    def para(m, c, p):
        den = m - 2 * c + p
        return 0.0 if abs(den) < 1e-12 else max(-0.5, min(0.5, 0.5 * (m - p) / den))

    di = para(grid[(i - 1, j)], sc, grid[(i + 1, j)]) if (i - 1, j) in grid and (i + 1, j) in grid else 0.0
    dj = para(grid[(i, j - 1)], sc, grid[(i, j + 1)]) if (i, j - 1) in grid and (i, j + 1) in grid else 0.0
    if di or dj:
        a = sim_w.angle_deg + (j + dj) * angle_step
        s = sim_w.scale * (1 + (i + di) * scale_step)
        sc2, S2, _, _ = evaluate(a, s)
        if sc2 >= sc - 0.002:
            sc, S = max(sc, sc2), S2
    edge = abs(i) == ns and ns > 0 or abs(j) == na and na > 0
    return {"score": sc, "sim_w": S, "angle": S.angle_deg, "scale": S.scale, "edge": edge}


class _Pyramid:
    """Working copies of one image at the search levels (``f`` = working px per image px, coarse →
    fine), computed lazily. ``coarse_extra`` prepends a coarser level for a large search region."""

    def __init__(self, image: np.ndarray, um_per_px: float, min_um_per_px: float = 0.0):
        self.image = image
        self.um = float(um_per_px)
        H, W = image.shape[:2]
        self.H, self.W = H, W
        self.field_diag = math.hypot(W, H) * self.um
        self.f = []
        for n in (LEVEL0_PX, LEVEL1_PX, LEVEL2_PX):
            f = min(1.0, n / max(W, H))
            if min_um_per_px > 0:
                f = min(f, self.um / min_um_per_px)
            self.f.append(max(f, self.f[-1]) if self.f else f)
        self._work = {}

    def add_coarse(self, k: float) -> None:
        self.f.insert(0, self.f[0] / k)
        self._work = {i + 1: w for i, w in self._work.items()}

    def work(self, level: int) -> np.ndarray:
        if level not in self._work:
            self._work[level] = _shrink(self.image, self.f[level])
        return self._work[level]

    def kind(self, level: int) -> int:
        """0 = global/coarse levels, 1 = middle, 2 = finest (filter widths depend on it)."""
        return max(0, level - (len(self.f) - 3))


def _scale_grid(tol: float, step: float) -> np.ndarray:
    n = max(0, int(round(tol / step))) if step > 0 else 0
    return np.array([1.0]) if n == 0 else 1 + tol * np.arange(-n, n + 1) / n


def _angle_grid(half: float, step: float) -> np.ndarray:
    if half >= 180:
        return np.arange(-180.0, 180.0, step)
    n = int(math.ceil(half / step))
    return np.arange(-n, n + 1) * step


def _align(pyr: _Pyramid, feats: Callable[[int], np.ndarray], render_at: Callable[[int], Render], region,
           scale_tol: float, mirror: bool, angle_range: float, init: Similarity | None,
           init_tol_um: float | None, refit: Callable[[Similarity], Callable[[int], np.ndarray]] | None,
           method: str, max_coarse_um: float | None = None) -> tuple[Similarity, dict]:
    """Shared coarse-to-fine driver (see module docstring). ``feats(level)`` = template features at
    pyramid level ``level``; ``render_at(level)`` = reference renderer for that level;
    ``refit(sim_full)`` may return a new feature function once a first placement is known (metal
    colour). With ``init`` there is no global search: the first level searches ±``angle_range`` and
    ±``init_tol_um`` around it. ``max_coarse_um`` caps the resolution of an extra coarse level
    (the reference's features must stay visible)."""
    t0 = time.time()
    lo, hi = pyr.um * (1 - scale_tol), pyr.um * (1 + scale_tol)
    info: dict = {"method": method}

    def steps(level):
        h, w = pyr.work(level).shape[:2]
        hd = 0.5 * math.hypot(w, h)
        # ≈ 1.5 px of arc / 3 px of size change at half the diagonal (features blurred by σ ≈ 1.5 px)
        return math.degrees(1.5 / hd), 3.0 / hd

    def bounds(level):
        return lo / pyr.f[level], hi / pyr.f[level]

    def to_level(sim_w: Similarity, src: int, dst: int) -> Similarity:
        return _to_work(_to_full(sim_w, pyr.f[src]), pyr.f[dst])

    if init is None:
        # a large search region gets a coarser first level so that its canvas stays small
        x0, y0, x1, y1 = region
        r0 = pyr.um / pyr.f[0]
        h0, w0 = pyr.work(0).shape[:2]
        area = ((x1 - x0) / r0 + 2 * w0) * ((y1 - y0) / r0 + 2 * h0)
        k = math.sqrt(area) / CANVAS0_MAX_PX
        if max_coarse_um:
            k = min(k, max_coarse_um / r0)
        if k > 1.25:
            pyr.add_coarse(min(k, 4.0))
            info["coarse_extra"] = round(min(k, 4.0), 2)
        a_step, s_step = steps(0)
        T = feats(0)
        cands = _search(T, render_at(0), region, pyr.um / pyr.f[0], mirror, _angle_grid(angle_range, a_step),
                        _scale_grid(scale_tol, s_step), TOP_K)
        if not cands:
            raise ValueError("Nothing to match against in the search region")
        info["level0"] = [{"score": round(c["score"], 3), "angle": round(c["angle"], 2)} for c in cands]
        win = None
    else:
        a_step, s_step = steps(0)
        r0 = pyr.um / pyr.f[0]
        cands = [{"score": 0.0, "angle": init.angle_deg, "sim_w": _to_work(init, pyr.f[0])}]
        T = feats(0)
        res = _refine(T, render_at(0), cands[0]["sim_w"], max(angle_range, a_step), a_step,
                      scale_tol, s_step, int(math.ceil((init_tol_um or 0.0) / r0)) + 3, bounds(0))
        cands = [res]
    info["t_level0"] = round(time.time() - t0, 2)
    n_levels = len(pyr.f)
    best = None
    for level in range(1, n_levels):
        T = feats(level)
        k = pyr.f[level] / pyr.f[level - 1]
        a_prev, s_prev = a_step, s_step
        a_step, s_step = steps(level)
        a_step, s_step = max(a_step, a_prev / 8), max(s_step, s_prev / 8)
        win = int(math.ceil(2 * k)) + 2
        out = []
        for c in cands:
            res = _refine(T, render_at(level), to_level(c["sim_w"], level - 1, level), a_prev, a_step,
                          s_prev, s_step, win, bounds(level))
            res["centre"] = _to_full(res["sim_w"], pyr.f[level]).apply([[pyr.W / 2, pyr.H / 2]])[0]
            out.append(res)
        out.sort(key=lambda c: -c["score"])
        best = out[0]
        info[f"level{level}"] = [{"score": round(c["score"], 3), "angle": round(c["angle"], 2)} for c in out]
        if len(out) > 1 or "margin" not in info:
            others = [c for c in out[1:] if _distinct(c, best, c["centre"], pyr.field_diag)]
            info["margin"] = round(best["score"] - (others[0]["score"] if others else 0.0), 3)
        cands = [best]
        if level == n_levels - 2 and refit is not None:
            feats = refit(_to_full(best["sim_w"], pyr.f[level]))
            T = feats(level)
            res = _refine(T, render_at(level), best["sim_w"], a_step, a_step / 2, s_step, s_step / 2, 3, bounds(level))
            info["refit"] = round(res["score"], 3)
            cands = [res]
        info[f"t_level{level}"] = round(time.time() - t0, 2)
    res = cands[0]
    sim = _to_full(res["sim_w"], pyr.f[-1])
    info.update({"score": round(res["score"], 3), "seconds": round(time.time() - t0, 2),
                 "um_per_px_levels": [round(pyr.um / f, 4) for f in pyr.f]})
    return _with_meta(sim, res["score"], info), info


# ---------------------------------------------------------------------------- image → GDS
#: Features compared with the GDS: "grad" = the x/y gradients of the metal map and of the blurred GDS
#: mask (2 channels; edges only, so large uniform flakes weigh little — needed at 100x, where the
#: markers are small and the flakes large), "contrast" = the maps themselves.
GDS_FEATURE = "grad"


def _gfeat(m: np.ndarray) -> np.ndarray:
    if GDS_FEATURE == "grad":
        gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.merge([gx, gy])
    return m


def _gds_region(shapes: list[np.ndarray], site, radius, field_diag: float):
    if site is not None and radius is not None:
        r = radius + 0.6 * field_diag
        return (site[0] - r, site[1] - r, site[0] + r, site[1] + r)
    if not shapes:
        raise ValueError("No GDS shapes on the chosen layers")
    allp = np.concatenate(shapes)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    if site is not None:
        r = 0.6 * field_diag
        x0, y0, x1, y1 = min(x0, site[0] - r), min(y0, site[1] - r), max(x1, site[0] + r), max(y1, site[1] + r)
    return (float(x0), float(y0), float(x1), float(y1))


def align_to_gds(image, um_per_px: float, gds_polys: dict, layers: list[tuple[int, int]],
                 site: tuple[float, float] | None = None, search_radius_um: float | None = None,
                 scale_tol: float = 0.05, mirror: bool = False, angle_range_deg: float = 20.0,
                 init: Similarity | None = None) -> Similarity:
    """Align an image (BGR array or path) to the GDS shapes of ``layers`` (bond pads / fan-out,
    markers, pre-patterned metal …) and return its :class:`Similarity` (px → chip µm) with
    ``score`` (final NCC) and ``info`` (see the module docstring).

    ``um_per_px`` is the nominal scale (from the objective's field of view); the true scale is
    searched within ±``scale_tol``. Angles within ±``angle_range_deg`` are tried (``180`` = all
    angles; a design with 4-fold symmetric pads is then ambiguous). ``site`` + ``search_radius_um``
    restrict where the image *centre* may lie (needed for high magnifications when the GDS has many
    identical markers); without them the whole extent of the shapes is searched. GDS shapes whose
    area exceeds ``MAX_FILL_FRAC`` of the image field (chip outline, write field) are ignored.
    ``init`` starts from a known placement (e.g. a chained one) and searches only ±``search_radius_um``
    around it and ±``angle_range_deg`` of its angle.

    The image is matched as a *metal map* (see module docstring): metal must differ in colour from
    the substrate. Raises ``ValueError`` if the final NCC is below ``GDS_SCORE_MIN`` or the best
    placement is not clearly better than a different one (``GDS_MARGIN_MIN``)."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        from .imaging import load_image
        image = load_image(image)
    pyr = _Pyramid(image, um_per_px)
    field_area = pyr.W * pyr.H * pyr.um ** 2
    shapes = _gds_shapes(gds_polys, layers, MAX_FILL_FRAC * field_area)
    if not shapes:
        raise ValueError(f"No usable GDS shapes on layers {layers}")
    if init is not None:
        centre = init.apply([[pyr.W / 2, pyr.H / 2]])[0]
        rad = search_radius_um if search_radius_um is not None else 0.05 * pyr.field_diag
        region = (centre[0] - rad, centre[1] - rad, centre[0] + rad, centre[1] + rad)
    else:
        region = _gds_region(shapes, site, search_radius_um, pyr.field_diag)
    # only the shapes near the search region are drawn
    pad = 0.6 * pyr.field_diag * (1 + scale_tol)
    bx0, by0, bx1, by1 = region[0] - pad, region[1] - pad, region[2] + pad, region[3] + pad
    near = [p for p in shapes if p[:, 0].max() >= bx0 and p[:, 0].min() <= bx1 and p[:, 1].max() >= by0 and p[:, 1].min() <= by1]
    if not near:
        raise ValueError("No GDS shapes near the search region")
    blur = (1.5, 1.0, 0.8)  # by pyr.kind(level)
    cache: dict = {}

    def contrast(level):
        if ("c", level) not in cache:
            cache[("c", level)] = _contrast_map(pyr.work(level))
        return cache[("c", level)]

    def feats_contrast(level):
        return _gfeat(cv2.GaussianBlur(contrast(level)[0], (0, 0), blur[pyr.kind(level)]))

    def render_at(level):
        def render(C: Similarity, size):
            m = _fill(near, C.inverse, size).astype(np.float32) / 255.0
            return _gfeat(cv2.GaussianBlur(m, (0, 0), blur[pyr.kind(level)]))
        return render

    def refit(sim_full: Similarity):
        mid = len(pyr.f) - 2
        _, diff = contrast(mid)
        sim_w = _to_work(sim_full, pyr.f[mid])
        m = _fill(near, sim_w.inverse, diff.shape[:2])
        inside = cv2.erode(m, np.ones((3, 3), np.uint8)) > 200
        outside = cv2.dilate(m, np.ones((7, 7), np.uint8)) == 0
        if inside.sum() < 20 or outside.sum() < 20:
            return feats_contrast
        metal = np.median(diff[inside], axis=0) - np.median(diff[outside], axis=0)
        if np.linalg.norm(metal) < 3.0:
            return feats_contrast

        def feats_metal(level):
            if ("m", level) not in cache:
                cache[("m", level)] = _gfeat(cv2.GaussianBlur(_metal_map(contrast(level)[1], metal), (0, 0), blur[pyr.kind(level)]))
            return cache[("m", level)]
        return feats_metal

    # feature size near the site (perimeter-weighted median of 2·area/perimeter) limits the coarse level
    if site is not None:
        rr = 0.6 * pyr.field_diag
        local = [p for p in near if np.abs(p - np.asarray(site, float)).max(axis=1).min() <= rr] or near
    else:
        local = near
    tp = np.array([(2 * _poly_area(p) / max(1e-9, np.hypot(*(np.roll(p, -1, 0) - p).T).sum()),
                    np.hypot(*(np.roll(p, -1, 0) - p).T).sum()) for p in local])
    o = np.argsort(tp[:, 0])
    cum = np.cumsum(tp[o, 1])
    feature_um = float(tp[o, 0][min(len(o) - 1, int(np.searchsorted(cum, cum[-1] / 2)))])
    sim, info = _align(pyr, feats_contrast, render_at, region, scale_tol, mirror, angle_range_deg, init,
                       search_radius_um, refit, "gds", max_coarse_um=feature_um)
    info["feature_um"] = round(feature_um, 2)
    info["layers"] = [list(k) for k in layers]
    if sim.score < GDS_SCORE_MIN:
        raise ValueError(f"No confident match of the image to the GDS (score {sim.score:.2f} < {GDS_SCORE_MIN}); "
                         "check the layers, the magnification and the site, or align by markers")
    if init is None and info["margin"] < GDS_MARGIN_MIN:
        raise ValueError(f"The match to the GDS is ambiguous (margin {info['margin']:.3f} < {GDS_MARGIN_MIN}); "
                         "give the device site and a search radius")
    return sim


# ---------------------------------------------------------------------------- image → image
def align_to_image(image, um_per_px: float, ref_image, ref_sim: Similarity, scale_tol: float = 0.05,
                   mirror: bool = False, angle_range_deg: float = 20.0) -> Similarity:
    """Chain an image onto an already aligned image ``ref_image`` (placed by ``ref_sim``) of the same
    chip region, usually at a lower magnification. Both are compared as band-pass filtered CIELab
    (NCC over the three channels) with the coarse-to-fine search of the module docstring; the scale
    prior is ``um_per_px`` ± ``scale_tol``, angles ± ``angle_range_deg`` relative to the chip
    (``180`` = all). The finest level never goes below the reference's own pixel size, so the
    result is about as accurate as ``ref_sim`` plus a fraction of a reference pixel. The image must
    lie (mostly) inside the reference. Raises ``ValueError`` if the final NCC is below
    ``IMG_SCORE_MIN`` or the match is ambiguous (``IMG_MARGIN_MIN``). Mirror is relative to the chip
    (as in ``ref_sim``)."""
    from .imaging import load_image
    if not hasattr(image, "shape"):
        image = load_image(image)
    if not hasattr(ref_image, "shape"):
        ref_image = load_image(ref_image)
    pyr = _Pyramid(image, um_per_px, min_um_per_px=ref_sim.scale)
    RH, RW = ref_image.shape[:2]
    foot = ref_sim.apply([[0, 0], [RW, 0], [0, RH], [RW, RH]])
    region = (float(foot[:, 0].min()), float(foot[:, 1].min()), float(foot[:, 0].max()), float(foot[:, 1].max()))
    ref_cache: dict = {}
    hp = (6.0, 10.0, 14.0)

    def ref_level(level):
        # reference features at about the level's resolution (never finer than the reference itself)
        if level not in ref_cache:
            r = pyr.um / pyr.f[level]
            fr = min(1.0, ref_sim.scale / r)
            work = _shrink(ref_image, fr)
            # same physical filter as the template (the reference working copy has the level's resolution)
            ref_cache[level] = (_bandpass(work, hp[pyr.kind(level)]), _to_work(ref_sim, fr))
        return ref_cache[level]

    feat_cache: dict = {}

    def feats(level):
        if level not in feat_cache:
            feat_cache[level] = _bandpass(pyr.work(level), hp[pyr.kind(level)])
        return feat_cache[level]

    def render_at(level):
        F, S_ref_w = ref_level(level)
        Mi = np.linalg.inv(_affine(S_ref_w))

        def render(C: Similarity, size):
            M = Mi @ _affine(C)  # canvas px → ref working px
            return cv2.warpAffine(F, M[:2], (size[1], size[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return render

    # an extra coarse level may shrink the image by at most IMG_COARSE_MAX_K (image texture is finer than pads)
    sim, info = _align(pyr, feats, render_at, region, scale_tol, mirror, angle_range_deg, None, None, None, "image",
                       max_coarse_um=IMG_COARSE_MAX_K * pyr.um / pyr.f[0])
    if sim.score < IMG_SCORE_MIN:
        raise ValueError(f"No confident match to the reference image (score {sim.score:.2f} < {IMG_SCORE_MIN})")
    if info["margin"] < IMG_MARGIN_MIN:
        raise ValueError(f"The match to the reference image is ambiguous (margin {info['margin']:.3f} < {IMG_MARGIN_MIN})")
    return sim


# ---------------------------------------------------------------------------- several magnifications
class AlignmentResult(dict):
    """``key → Similarity`` for every image that could be placed, plus

    * ``errors``: ``key → message`` for every image that could not (shown to the user);
    * ``log``: ``key → {"method": "gds" | "chain" | "chain+gds", "ref": key of the reference image,
      "score": …, "gds_score": …, "seconds": …}``.
    """

    def __init__(self):
        super().__init__()
        self.errors: dict[str, str] = {}
        self.log: dict[str, dict] = {}


def align_images(images: dict, gds_polys: dict, layers, site=None, mirror: bool = False,
                 search_radius_um: float | None = None, refine_gds: bool = True,
                 angle_range_deg: float = 20.0) -> AlignmentResult:
    """Place several images of one chip region, ``key → (image, nominal µm/px)`` (e.g. "5x", "20x",
    "100x"). The coarsest image is aligned to the GDS (:func:`align_to_gds` with ``site`` /
    ``search_radius_um``; if that fails the next one is tried, and so on). Every finer image is
    chained onto the finest already placed image that is coarser than it (:func:`align_to_image`;
    the other placed images are tried if that fails) and then, with ``refine_gds``, re-fitted on the
    GDS starting from the chained placement (± 2 % of its field, ± 0.5°): the GDS fit is kept if it
    scores ≥ ``REFINE_GDS_SCORE_MIN`` and moves the image centre less than
    ``REFINE_GDS_MAX_SHIFT_FRAC`` × the field diagonal — so a 100x image is placed on the markers
    themselves, not only through the chain. An image that cannot be chained is tried on the GDS
    directly.

    Returns an :class:`AlignmentResult` (a dict of the placed images with ``errors`` and ``log``).
    Raises ``ValueError`` only if no image could be placed."""
    from .imaging import load_image
    loaded = {}
    for k, (img, um) in images.items():
        loaded[k] = (img if hasattr(img, "shape") else load_image(img), float(um))
    order = sorted(loaded, key=lambda k: -loaded[k][1])
    out = AlignmentResult()
    for key in order:
        img, um = loaded[key]
        t0 = time.time()
        sim = None
        errs = []
        placed = [k for k in order if k in out]
        refs = sorted([k for k in placed if loaded[k][1] >= um], key=lambda k: loaded[k][1])
        refs += [k for k in placed if k not in refs]
        for ref in refs:
            try:
                sim = align_to_image(img, um, loaded[ref][0], out[ref], mirror=mirror, angle_range_deg=angle_range_deg)
                out.log[key] = {"method": "chain", "ref": ref, "score": round(sim.score, 3)}
                break
            except ValueError as e:
                errs.append(f"chain to {ref}: {e}")
                sim = None
        if sim is not None and refine_gds:
            try:
                diag = math.hypot(*img.shape[:2]) * um
                g = align_to_gds(img, sim.scale, gds_polys, layers, search_radius_um=0.02 * diag, scale_tol=0.01,
                                 mirror=mirror, angle_range_deg=0.5, init=sim)
                shift = float(np.hypot(*(g.apply([[img.shape[1] / 2, img.shape[0] / 2]])[0]
                                         - sim.apply([[img.shape[1] / 2, img.shape[0] / 2]])[0])))
                out.log[key]["gds_score"] = round(g.score, 3)
                out.log[key]["gds_shift_um"] = round(shift, 3)
                if g.score >= REFINE_GDS_SCORE_MIN and shift <= REFINE_GDS_MAX_SHIFT_FRAC * diag:
                    g.info["chained_from"] = out.log[key]["ref"]
                    sim = g
                    out.log[key]["method"] = "chain+gds"
            except ValueError as e:
                out.log[key]["gds_error"] = str(e)
        if sim is None:
            try:
                sim = align_to_gds(img, um, gds_polys, layers, site=site, search_radius_um=search_radius_um,
                                   mirror=mirror, angle_range_deg=angle_range_deg)
                out.log[key] = {"method": "gds", "score": round(sim.score, 3)}
            except ValueError as e:
                errs.append(f"GDS: {e}")
        if sim is None:
            out.errors[key] = "; ".join(errs)
            continue
        out[key] = sim
        out.log[key]["seconds"] = round(time.time() - t0, 2)
    if not len(out):
        raise ValueError("No image could be aligned: " + " | ".join(f"{k}: {v}" for k, v in out.errors.items()))
    return out


# ---------------------------------------------------------------------------- flake outlines
def _chip_raster(image: np.ndarray, sim: Similarity, box: tuple[float, float, float, float], r: float):
    """The part of ``image`` over the chip box (x0, y0, x1, y1) resampled at ``r`` µm/px in the chip
    frame (x right, y up → rows down). Returns (raster BGR, valid mask, frame C: raster px → chip)."""
    x0, y0, x1, y1 = box
    C = Similarity(r, 0.0, x0, y1, False)
    w = int(math.ceil((x1 - x0) / r)) + 1
    h = int(math.ceil((y1 - y0) / r)) + 1
    f = min(1.0, 1.25 * sim.scale / r)
    work = _shrink(image, f) if f < 1.0 else image
    M = np.linalg.inv(_affine(_to_work(sim, f) if f < 1.0 else sim)) @ _affine(C)
    ras = cv2.warpAffine(work, M[:2], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full(work.shape[:2], 255, np.uint8), M[:2], (w, h),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return ras, valid > 0, C


def _edge_bins(ras: np.ndarray, valid: np.ndarray, nbins: int, tau: float, pool: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Oriented chamfer maps: for each orientation bin (edge *tangent* direction mod 180°), the
    score map g = max(0, 1 − d/τ), d = distance to the nearest edge pixel of that orientation (an
    edge pixel also counts for the neighbouring bin it is closest to). Edges = Canny on the CIELab
    channel with the strongest gradient, thresholds at the 80th / 92nd percentile of the gradient
    magnitude inside the image. With ``pool`` > 1 the edge maps are max-pooled by that factor before
    the distance transform (edges found at the fine resolution stay visible on the coarse grid).
    Returns (g stack nbins×h×w float32, edge mask at the raster resolution)."""
    lab = _lab(cv2.GaussianBlur(ras, (0, 0), 1.0))
    gx = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)])
    gy = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)])
    mag = gx * gx + gy * gy
    k = np.argmax(mag, axis=0)
    ii = np.indices(k.shape)
    dx, dy = gx[k, ii[0], ii[1]], gy[k, ii[0], ii[1]]
    m = np.sqrt(dx * dx + dy * dy)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vals = m[inner] if inner.any() else m.ravel()
    hi = max(float(np.percentile(vals, 92)), 1e-3)
    lo = max(float(np.percentile(vals, 80)), 0.5 * hi * 0.5)
    sc = 2000.0 / max(hi, 1e-6)
    edges = cv2.Canny(np.clip(dx * sc, -32000, 32000).astype(np.int16), np.clip(dy * sc, -32000, 32000).astype(np.int16),
                      lo * sc, hi * sc, L2gradient=True) > 0
    edges &= inner
    theta = (np.arctan2(dy, dx) + math.pi / 2) % math.pi          # tangent orientation
    pos = theta / (math.pi / nbins)
    b0 = np.floor(pos).astype(int) % nbins
    frac = pos - np.floor(pos)
    b1 = np.where(frac >= 0.5, (b0 + 1) % nbins, (b0 - 1) % nbins)
    h, w = edges.shape
    hp, wp = -(-h // pool), -(-w // pool)
    g = np.zeros((nbins, hp, wp), np.float32)
    for b in range(nbins):
        e = edges & ((b0 == b) | (b1 == b))
        if pool > 1:
            e = np.pad(e, ((0, hp * pool - h), (0, wp * pool - w))).reshape(hp, pool, wp, pool).any(axis=(1, 3))
        d = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
        g[b] = np.clip(1.0 - d / tau, 0, 1)
    return g, edges


def _resample(poly: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` along the closed polygon and the tangent angle (rad) at each."""
    p = np.vstack([poly, poly[:1]])
    seg = np.diff(p, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    keep = L > 1e-9
    p0, seg, L = p[:-1][keep], seg[keep], L[keep]
    cum = np.concatenate([[0], np.cumsum(L)])
    n = max(8, int(cum[-1] / step))
    s = np.linspace(0, cum[-1], n, endpoint=False)
    idx = np.searchsorted(cum, s, side="right") - 1
    t = (s - cum[idx]) / L[idx]
    pts = p0[idx] + seg[idx] * t[:, None]
    ang = np.arctan2(seg[idx, 1], seg[idx, 0])
    return pts, ang


def _poly_centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2
    if abs(a) < 1e-12:
        return p.mean(0)
    return np.array([((x + xn) * cr).sum() / (6 * a), ((y + yn) * cr).sum() / (6 * a)])


def _place(pts: np.ndarray, ang: np.ndarray, angle_deg: float, mirror: bool, scale: float):
    """Outline points (centred, µm) and tangent angles → rotated / mirrored / scaled (µm, chip axes)."""
    q = pts.copy()
    t = ang.copy()
    if mirror:
        q[:, 0] *= -1
        t = math.pi - t
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return scale * (q @ R.T), t + a


def _chamfer(g: np.ndarray, pts_um: np.ndarray, tang: np.ndarray, r: float):
    """Score map (mean of g over the outline points) for every position of the outline centre in
    the raster (y down), and the raster px offset of the centre relative to the result index."""
    nb = g.shape[0]
    qx = pts_um[:, 0] / r
    qy = -pts_um[:, 1] / r
    ix, iy = np.round(qx).astype(int), np.round(qy).astype(int)
    ox, oy = ix.min(), iy.min()
    tw, th = ix.max() - ox + 1, iy.max() - oy + 1
    # tangent orientation in the raster (y down): angle → −angle, mod π
    tb = np.round(((-tang) % math.pi) / (math.pi / nb)).astype(int) % nb
    H, W = g.shape[1:]
    if th >= H or tw >= W:
        return None, (0, 0)
    total = None
    for b in range(nb):
        sel = tb == b
        if not sel.any():
            continue
        T = np.zeros((th, tw), np.float32)
        np.add.at(T, (iy[sel] - oy, ix[sel] - ox), 1.0)
        r_b = cv2.matchTemplate(g[b], T, cv2.TM_CCORR)
        total = r_b if total is None else total + r_b
    total /= len(pts_um)
    return total, (-ox, -oy)


def locate_outline(image, sim: Similarity, outline_um: list[tuple[float, float]], allow_mirror: bool = True,
                   search_center_um: tuple[float, float] | None = None, search_radius_um: float | None = None,
                   scale_tol: float = 0.04, min_score: float | None = None, min_margin: float | None = None,
                   frame_um: tuple[float, float, float, float] | None = None) -> dict:
    """Find a flake whose border ``outline_um`` (traced on another image, µm, any origin / rotation)
    appears in ``image`` (BGR array or path) placed in the chip frame by ``sim``.

    Oriented chamfer matching, robust to partial occlusion by other flakes of the stack:

    1. The image around the search area is resampled in the chip frame and edges are found at the
       fine resolution (outline size / ``OUTLINE_FINE_DIV``), then max-pooled to the coarse grid
       (outline size / ``OUTLINE_COARSE_DIV``): one edge map per tangent orientation
       (``OUTLINE_BINS`` bins mod 180°, :func:`_edge_bins`), each converted to g = max(0, 1 − d/τ) with d the distance to the nearest
       edge of that orientation and τ = ``OUTLINE_TAU_PX``.
    2. The outline (centred on its centroid, resampled about every raster pixel, each point with its
       tangent orientation) is rotated in ``OUTLINE_STEP0_DEG`` steps over 360°, also mirrored if
       ``allow_mirror``; for each pose the mean of g over its points is computed for every
       translation at once (``cv2.matchTemplate`` per orientation bin). A point that lies on an edge
       of the same orientation scores 1, so the score is the (soft) fraction of the border that is
       visible — a flake half hidden under another one still scores about one half where a random
       position scores much less.
    3. The ``TOP_K`` best distinct poses are refined at a finer raster (outline size /
       ``OUTLINE_FINE_DIV``, never finer than the image) over angle (± one coarse step), scale (``1 ± scale_tol``) and
       translation.

    Returns ``{"polygon": [(x, y) chip µm], "center": (x, y), "angle_deg", "mirror", "scale",
    "score" (0…1: mean over the scored border points of max(0, 1 − d/τ) at the fine level, τ =
    ``OUTLINE_TAU_PX`` fine pixels), "fraction" (border points with d ≤ τ/2), "margin" (score − the
    best clearly different pose), "candidates" (refined poses, best first), "seconds"}``. The pose maps an outline point p to
    ``center + scale·R(angle)·M·(p − centroid)``, M = x-mirror when ``mirror``.
    ``frame_um`` = (x0, y0, x1, y1), the field of the image the outline was traced on, in the
    outline's coordinates: border points within ``FRAME_EDGE_FRAC`` of the field size of its edge
    are where the flake left the field, not flake edges, and are not scored.
    Raises ``ValueError`` when the best score is below ``min_score`` (default ``OUTLINE_SCORE_MIN``)
    or another, different pose scores almost as well (margin below ``min_margin``, default
    ``OUTLINE_MARGIN_MIN``); pass 0 / -1 to get the best pose anyway."""
    t0 = time.time()
    from .imaging import load_image
    if not hasattr(image, "shape"):
        image = load_image(image)
    min_score = OUTLINE_SCORE_MIN if min_score is None else min_score
    P = np.asarray(outline_um, float).reshape(-1, 2)
    if len(P) < 3:
        raise ValueError("The outline needs at least three points")
    c0 = _poly_centroid(P)
    P0 = P - c0

    def resample(step):
        pts, ang = _resample(P0, step)
        if frame_um is not None:
            x0, y0, x1, y1 = frame_um
            m = FRAME_EDGE_FRAC * max(x1 - x0, y1 - y0)
            q = pts + c0
            ok = (q[:, 0] > x0 + m) & (q[:, 0] < x1 - m) & (q[:, 1] > y0 + m) & (q[:, 1] < y1 - m)
            if ok.sum() >= 8:
                pts, ang = pts[ok], ang[ok]
        return pts, ang

    D = 2 * float(np.hypot(P0[:, 0], P0[:, 1]).max())
    H, W = image.shape[:2]
    foot = sim.apply([[0, 0], [W, 0], [0, H], [W, H]])
    fx0, fy0 = foot.min(0)
    fx1, fy1 = foot.max(0)
    if search_center_um is None:
        search_center_um = tuple(sim.apply([[W / 2, H / 2]])[0])
    cx, cy = search_center_um
    if search_radius_um is None:
        search_radius_um = 0.5 * math.hypot(fx1 - fx0, fy1 - fy0)
    reach = search_radius_um + 0.6 * D
    box = (max(fx0, cx - reach), max(fy0, cy - reach), min(fx1, cx + reach), min(fy1, cy + reach))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("The search area is outside the image")
    tau = OUTLINE_TAU_PX
    nb = OUTLINE_BINS

    def level(r: float, bx, pool: int = 1):
        ras, valid, C = _chip_raster(image, sim, bx, r / pool)
        g, edges = _edge_bins(ras, valid, nb, tau, pool)
        C = _rescaled(C, pool, (pool - 1) / 2.0)
        pad = int(math.ceil(0.55 * D / r)) + 2
        g = np.pad(g, ((0, 0), (pad, pad), (pad, pad)))
        Cp = _shifted(C, (-pad, -pad))
        return g, Cp, edges

    # ---- coarse
    r1 = max(D / OUTLINE_FINE_DIV, sim.scale)   # fine level (edges are always found at this resolution)
    pool = max(1, int(round(D / OUTLINE_COARSE_DIV / r1)))
    r0 = pool * r1
    g0, C0, _ = level(r0, box, pool)
    pts, ang = resample(r0)
    cands = []
    mirrors = (False, True) if allow_mirror else (False,)
    c_px = C0.inverse([[cx, cy]])[0]
    rad_px = search_radius_um / r0
    for m in mirrors:
        for a in np.arange(0.0, 360.0, OUTLINE_STEP0_DEG):
            q, t = _place(pts, ang, a, m, 1.0)
            S, (ox, oy) = _chamfer(g0, q, t, r0)
            if S is None:
                continue
            # only centres within the search radius
            yy, xx = np.mgrid[0:S.shape[0], 0:S.shape[1]]
            S = np.where((xx + ox - c_px[0]) ** 2 + (yy + oy - c_px[1]) ** 2 <= rad_px ** 2, S, 0)
            for sc, x, y in _peaks(S.astype(np.float32), 2, max(2, int(0.1 * D / r0))):
                ctr = C0.apply([[x + ox, y + oy]])[0]
                cands.append({"score": sc, "angle": float(a), "mirror": m, "center": ctr})
    if not cands:
        raise ValueError("The outline does not fit in the search area")
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(math.hypot(*(c["center"] - k["center"])) > 0.15 * D or abs((c["angle"] - k["angle"] + 180) % 360 - 180) > 15
               or c["mirror"] != k["mirror"] for k in kept):
            kept.append(c)
        if len(kept) >= TOP_K:
            break
    # ---- fine
    pts1, ang1 = resample(r1)
    win_um = 2.5 * r0
    fine = []
    for c in kept:
        half = 0.5 * D * (1 + scale_tol) + win_um + 3 * r1
        bx = (c["center"][0] - half, c["center"][1] - half, c["center"][0] + half, c["center"][1] + half)
        g1, C1, _ = level(r1, bx)
        grid = {}

        def ev(i, j, c=c, g1=g1, C1=C1, grid=grid):
            if (i, j) not in grid:
                s = 1 + i * scale_tol / 2
                a = c["angle"] + j * 0.5
                q, t = _place(pts1, ang1, a, c["mirror"], s)
                S, (ox, oy) = _chamfer(g1, q, t, r1)
                if S is None:
                    grid[(i, j)] = (-1.0, None)
                else:
                    cp = C1.inverse([c["center"]])[0] - (ox, oy)
                    w = int(math.ceil(win_um / r1))
                    xa, ya = int(round(cp[0])) - w, int(round(cp[1])) - w
                    xa, ya = max(0, xa), max(0, ya)
                    sub = S[ya: ya + 2 * w + 1, xa: xa + 2 * w + 1]
                    if sub.size == 0:
                        grid[(i, j)] = (-1.0, None)
                        return grid[(i, j)]
                    _, mx, _, (x, y) = cv2.minMaxLoc(sub)
                    xs, ys = _subpix(sub, x, y)
                    grid[(i, j)] = (float(mx), {"angle": a, "scale": s, "mirror": c["mirror"],
                                                 "center": C1.apply([[xs + xa + ox, ys + ya + oy]])[0]})
            return grid[(i, j)]

        nj = int(round(OUTLINE_STEP0_DEG / 0.5))
        i, j = 0, max(range(-nj, nj + 1), key=lambda jj: ev(0, jj)[0])
        i = max(range(-2, 3), key=lambda ii: ev(ii, j)[0])
        for _ in range(8):
            nbh = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            nbh = [k for k in nbh if abs(k[0]) <= 2 and abs(k[1]) <= nj + 2]
            best_n = max(nbh, key=lambda k: ev(*k)[0])
            if ev(*best_n)[0] <= ev(i, j)[0]:
                break
            i, j = best_n
        sc, pose = ev(i, j)
        if pose is None:
            continue
        pose["score"] = sc
        # fraction of points within τ/2 of a matching edge
        q, t = _place(pts1, ang1, pose["angle"], pose["mirror"], pose["scale"])
        qp = C1.inverse(q + pose["center"])
        tb = np.round(((-t) % math.pi) / (math.pi / nb)).astype(int) % nb
        xi = np.clip(np.round(qp[:, 0]).astype(int), 0, g1.shape[2] - 1)
        yi = np.clip(np.round(qp[:, 1]).astype(int), 0, g1.shape[1] - 1)
        pose["fraction"] = float((g1[tb, yi, xi] >= 0.5).mean())
        pose["coarse_score"] = c["score"]
        fine.append(pose)
    if not fine:
        raise ValueError("No placement of the outline found")
    fine.sort(key=lambda p: -p["score"])
    best = fine[0]
    others = [p for p in fine[1:] if math.hypot(*(p["center"] - best["center"])) > 0.15 * D
              or abs((p["angle"] - best["angle"] + 180) % 360 - 180) > 15 or p["mirror"] != best["mirror"]]
    margin = best["score"] - (others[0]["score"] if others else 0.0)
    q, _ = _place(P0, np.zeros(len(P0)), best["angle"], best["mirror"], best["scale"])
    poly = q + best["center"]
    result = {"polygon": [(float(x), float(y)) for x, y in poly],
              "center": (float(best["center"][0]), float(best["center"][1])),
              "angle_deg": float(best["angle"] % 360), "mirror": bool(best["mirror"]), "scale": float(best["scale"]),
              "score": float(best["score"]), "fraction": best["fraction"], "margin": float(margin),
              "candidates": [{"score": round(p["score"], 3), "angle_deg": round(p["angle"] % 360, 1), "mirror": p["mirror"],
                              "center": (round(float(p["center"][0]), 2), round(float(p["center"][1]), 2))} for p in fine],
              "seconds": round(time.time() - t0, 2)}
    min_margin = OUTLINE_MARGIN_MIN if min_margin is None else min_margin
    if best["score"] < min_score:
        raise ValueError(f"The flake was not found (best score {best['score']:.2f} < {min_score}); "
                         "trace it on this image instead")
    if margin < min_margin:
        raise ValueError(f"The flake position is ambiguous (score {best['score']:.2f}, margin {margin:.2f} < {min_margin}); "
                         "narrow the search area or trace it on this image instead")
    return result


# ====================================================================== flakes added since the previous step
#: Change map: lines narrower than about twice this (µm) — EBL metal written between the two images,
#: e.g. the pre-patterned fingers — are removed by an opening and a closing of this radius, so the
#: flakes dropped since then are what remains (tuned on the lab's tWSe2 Device 2026-11-16).
CHANGE_THIN_UM = 1.2
CHANGE_GAIN = 2.5            # ΔE → grey level of the change image the outline matcher works on
CHANGE_KEEP_FRAC = 0.6       # colour mapping between the two images: refit on this best-fitting fraction


@dataclass
class ChangeMap:
    """What changed on the chip between an earlier aligned image and a later one, in the chip frame:
    ``image`` (grey BGR, brighter = more colour change after the colour mapping), ``sim`` (its px →
    chip µm), ``valid`` (covered by both images), ``r`` (µm per px)."""
    image: np.ndarray
    sim: Similarity
    valid: np.ndarray
    r: float


def change_map(image, sim: Similarity, before, before_sim: Similarity, box: tuple[float, float, float, float],
               r: float | None = None, thin_um: float = CHANGE_THIN_UM) -> ChangeMap:
    """Material added between ``before`` and ``image`` (both aligned to the chip) over the chip box.

    Both images are resampled in the chip frame at ``r`` µm/px (default: 1.5 × the coarser image's
    µm/px, at least 0.1) and converted to CIELab; the earlier one is mapped onto the later one by an
    affine colour transform fitted on the pixels that did not change (least squares, refitted three
    times on the best-fitting ``CHANGE_KEEP_FRAC`` — exposure and white balance differ between
    sessions); the colour distance is then opened and closed with a disk of ``thin_um`` so thin
    written lines vanish and flakes (plateaus: one level per stack of new flakes) remain. A flake
    dropped since the earlier image is then a region whose border is an edge of this map even when
    it is faint and covered in the photograph itself (thin WSe2 under a middle hBN)."""
    from .imaging import load_image
    if not hasattr(image, "shape"):
        image = load_image(image)
    if not hasattr(before, "shape"):
        before = load_image(before)
    r = r or max(0.1, 1.5 * max(sim.scale, before_sim.scale))
    # only where both images are
    fa = before_sim.apply([[0, 0], [before.shape[1], 0], [0, before.shape[0]], [before.shape[1], before.shape[0]]])
    fb = sim.apply([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
    box = (max(box[0], fa[:, 0].min(), fb[:, 0].min()), max(box[1], fa[:, 1].min(), fb[:, 1].min()),
           min(box[2], fa[:, 0].max(), fb[:, 0].max()), min(box[3], fa[:, 1].max(), fb[:, 1].max()))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("The two images do not overlap on the chip")
    a, va, C = _chip_raster(before, before_sim, box, r)
    b, vb, _ = _chip_raster(image, sim, box, r)
    lab = lambda x: cv2.cvtColor(cv2.GaussianBlur(x, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    A, B = lab(a), lab(b)
    m = va & vb
    if m.sum() < 100:
        raise ValueError("The two images do not overlap on the chip")
    X = np.c_[A[m], np.ones(int(m.sum()), np.float32)]
    Y = B[m]
    if len(X) > 300_000:                         # the colour fit needs a sample, not every pixel
        pick = np.random.default_rng(0).choice(len(X), 300_000, replace=False)
        X, Y = X[pick], Y[pick]
    keep = np.ones(len(X), bool)
    for _ in range(4):
        M, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        res = np.sqrt(((X @ M - Y) ** 2).sum(1))
        keep = res < np.percentile(res, 100 * CHANGE_KEEP_FRAC)
    A2 = (np.c_[A.reshape(-1, 3), np.ones(A.shape[0] * A.shape[1], np.float32)] @ M).reshape(A.shape)
    d = np.sqrt(((A2 - B) ** 2).sum(-1)).astype(np.float32)
    d[~m] = 0.0
    k = int(round(thin_um / r)) * 2 + 1
    if k >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        d = cv2.morphologyEx(cv2.morphologyEx(d, cv2.MORPH_OPEN, ker), cv2.MORPH_CLOSE, ker)
    g = np.clip(d * CHANGE_GAIN, 0, 255).astype(np.uint8)
    return ChangeMap(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), C, m, r)


def locate_new_outline(cm: ChangeMap, outline_um, allow_mirror: bool = True, search_center_um=None,
                       search_radius_um: float | None = None, **kw) -> dict:
    """:func:`locate_outline` on a :class:`ChangeMap`: for a flake dropped since the earlier image."""
    out = locate_outline(cm.image, cm.sim, outline_um, allow_mirror, search_center_um, search_radius_um, **kw)
    out["method"] = "change"
    return out


# ====================================================================== patterns written at earlier steps
#: Program choices for finding a written pattern (fingers, contact gate) in a later step's image.
PATTERN_SEARCH_UM = 3.0          # how far it may lie from where it was designed
PATTERN_MAX_TURN_DEG = 1.0       # how much it may be turned
PATTERN_WORK_UM = 0.05           # working scale (µm / px)
PATTERN_MIN_INLIERS = 0.45       # share of its outline that must lie on an edge of the image (lab's 100x: 0.48–0.85)
PATTERN_MAX_UM_PX = 0.25         # images coarser than this (5x, 10x) cannot show a 0.8 µm finger's edges


def _colour_edges(work: np.ndarray) -> np.ndarray:
    """Edges in colour (CIELab): at each pixel the channel that changes most (metal on hBN can have the grey level of the
    hBN), Canny on that gradient with thresholds from the image's own gradient statistics."""
    lab = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)], axis=2)
    gy = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)], axis=2)
    mag = gx * gx + gy * gy
    k = np.argmax(mag, axis=2)[..., None]
    dx = np.take_along_axis(gx, k, 2)[..., 0]
    dy = np.take_along_axis(gy, k, 2)[..., 0]
    m = np.sqrt(dx * dx + dy * dy)
    hi = max(float(np.percentile(m, 92)), 8.0)
    s = 16000.0 / max(float(m.max()), 1e-6)            # into int16 for Canny
    return cv2.Canny(np.round(dx * s).astype(np.int16), np.round(dy * s).astype(np.int16), 0.4 * hi * s, hi * s, L2gradient=True)


def locate_pattern(img: np.ndarray, sim: Similarity, polys: list, search_um: float = PATTERN_SEARCH_UM,
                   max_turn_deg: float = PATTERN_MAX_TURN_DEG) -> dict:
    """Where a pattern written at an earlier step (``polys``, chip µm, as designed) lies in an aligned image of a later
    step: the rigid move (turn about its middle + shift) that puts its outline on the image's edges (oriented chamfer
    matching on the image's edge map, truncated; coarse grid then refined). Returns {"dx", "dy", "angle", "cx", "cy",
    "inliers", "rms_px"} in chip µm / degrees; raises ValueError when the outline does not lie on the image's edges."""
    if sim.scale > PATTERN_MAX_UM_PX:
        raise ValueError(f"too coarse ({sim.scale:.2f} µm/px): use a 20x or finer image")
    pts_chip = np.vstack([np.asarray(p, float) for p in polys if len(p) >= 3])
    cx, cy = pts_chip.mean(0)
    px = sim.inverse(pts_chip)
    m = (search_um + 2.0) / sim.scale
    x0, y0 = np.floor(px.min(0) - m).astype(int)
    x1, y1 = np.ceil(px.max(0) + m).astype(int)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 20 or y1 - y0 < 20:
        raise ValueError("the pattern lies outside this image")
    f = min(1.0, sim.scale / PATTERN_WORK_UM)
    crop = img[y0:y1, x0:x1]
    work = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else crop
    edges = _colour_edges(work)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    cap = 4.0
    um_px = sim.scale / f
    # the pattern's outline as points (every ~1 px), in work px, relative to its middle
    outline = np.zeros(work.shape[:2], np.uint8)
    to_work = lambda q: (sim.inverse(np.asarray(q, float)) - (x0, y0)) * f
    for p in polys:
        if len(p) >= 3:
            cv2.polylines(outline, [np.round(to_work(p)).astype(np.int32)], True, 1, 1)
    ys, xs = np.nonzero(outline)
    if len(xs) < 30:
        raise ValueError("the pattern is too small in this image")
    sel = np.linspace(0, len(xs) - 1, min(len(xs), 4000)).astype(int)
    P = np.c_[xs[sel], ys[sel]].astype(np.float64)
    c_work = to_work([(cx, cy)])[0]
    rel = P - c_work
    H, W = dt.shape

    def score(angle_deg, shifts):
        a = math.radians(-angle_deg)                   # image y is down: a chip turn of +θ is −θ in px
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        q = rel @ R.T + c_work
        qs = q[None, :, :] + shifts[:, None, :]
        xi = np.clip(np.round(qs[..., 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(qs[..., 1]).astype(int), 0, H - 1)
        d = np.minimum(dt[yi, xi], cap)
        return d.mean(1), d

    rng = search_um / um_px
    best = (math.inf, 0.0, np.zeros(2))
    step = max(1.0, rng / 15)
    grid = np.arange(-rng, rng + 1e-9, step)
    shifts = np.array([(a, b) for a in grid for b in grid])
    for ang in np.arange(-max_turn_deg, max_turn_deg + 1e-9, 0.25):
        s_, _d = score(ang, shifts)
        k = int(np.argmin(s_))
        if s_[k] < best[0]:
            best = (float(s_[k]), float(ang), shifts[k])
    for fine_step, fine_ang in ((step / 4, 0.1), (0.5, 0.025)):     # refine round the best
        g = np.arange(-2 * fine_step * 4, 2 * fine_step * 4 + 1e-9, fine_step)
        shifts = best[2] + np.array([(a, b) for a in g for b in g])
        for ang in np.clip(best[1] + np.arange(-4, 5) * fine_ang, -max_turn_deg, max_turn_deg):
            s_, _d = score(ang, shifts)
            k = int(np.argmin(s_))
            if s_[k] < best[0]:
                best = (float(s_[k]), float(ang), shifts[k])
    _s, d = score(best[1], best[2][None, :])
    inliers = float((d[0] <= 1.5).mean())
    if inliers < PATTERN_MIN_INLIERS:
        raise ValueError(f"its outline does not lie on the image's edges (only {inliers:.0%} does)")
    shift_px = best[2] / f                             # full-resolution px
    d_chip = sim.apply([shift_px]) - sim.apply([(0.0, 0.0)])
    angle = -float(best[1]) if sim.mirror else float(best[1])      # a mirrored image turns the other way
    return {"dx": float(d_chip[0][0]), "dy": float(d_chip[0][1]), "angle": angle, "cx": float(cx), "cy": float(cy),
            "inliers": round(inliers, 3), "rms_px": round(float(best[0]), 3)}
