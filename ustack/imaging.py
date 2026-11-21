"""Optical-image analysis: flake borders, alignment-marker squares and defect candidates.

All functions take an image as a BGR ``uint8`` array (OpenCV order) and return pixel
coordinates (x right, y down). The UI converts them to µm with the image's transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read any image OpenCV can (JPEG, PNG, TIFF), also from paths with non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    return image


def _working_copy(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        image = cv2.resize(image, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    return image, f


def simplify(points: np.ndarray, epsilon: float) -> list[tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, epsilon, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx]


# ---------------------------------------------------------------------- flake borders
#: Tracer settings, chosen on the lab's WSe2 batch-1 images (``WSe2 device training/Layer OM Image``:
#: 18 single-flake images of hBN, graphite and monolayer WSe2, bright and dark field) by checking
#: every outline by eye and the agreement between clicks at different points of the same flake
#: (``tools/trace_benchmark.py``; results in ``docs/trace_benchmark.md``). Lengths are pixels of the
#: working copy (longest side ``max_side``).
TRACE_DEFAULTS = {
    "max_side": 1600,     # working copy size (px)
    "edge_frac": 0.35,    # whole flake: the border sits where the contrast to the substrate is 35 % of the click's
    "k_sigma": 4.0,       # never closer to the substrate than 4 × its colour noise
    "open_r": 2,          # removes specks and residue threads thinner than ~4 px
    "neck_r": 6,          # a neighbour joined through a neck narrower than ~12 px is cut off
    "close_r": 2,         # smooths the border
    "detail": 0.0012,     # polygon simplification, fraction of the border length
    "residue": True,      # whole flake: drop soft-edged, textured pieces (tape residue), see _drop_residue
    "residue_sharp": 0.6,  # ... a piece is kept if its border is at least this sharp relative to the flake's
    "residue_flat": 0.9,   # ... or if it is a uniform plateau (75th percentile of its gradient, ΔE/px)
}
TRACE_MODES = {"flake": "Whole flake", "colour": "This colour only"}


@dataclass
class TraceContext:
    """An image prepared for tracing: smoothed, in CIELab (L 0–100), with the illumination fall-off
    removed, the substrate colour and its noise. Reuse it for every click on the same image."""
    lab: np.ndarray
    f: float                    # working copy scale (working px per image px)
    substrate: np.ndarray       # Lab colour of the substrate
    sigma: float                # substrate colour noise (ΔE)
    clusters: tuple | None = None

    @property
    def floor(self) -> float:
        return max(TRACE_DEFAULTS["k_sigma"] * self.sigma, 2.5)


def _poly_terms(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys], axis=-1)


def _substrate_field(lab: np.ndarray, iters: int = 4) -> tuple[np.ndarray, float]:
    """Substrate colour over the field: the main colour of the image border (a flake can fill most
    of the field, rarely most of its edge), refined by a quadratic fit to the pixels of that colour
    (the illumination falls off toward the corners)."""
    h, w = lab.shape[:2]
    k = max(1, int(round(max(h, w) / 320)))
    small = cv2.resize(lab, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    bw = max(2, int(0.08 * min(sh, sw)))
    edge = np.concatenate([small[:bw].reshape(-1, 3), small[-bw:].reshape(-1, 3), small[:, :bw].reshape(-1, 3), small[:, -bw:].reshape(-1, 3)])
    keys, counts = np.unique(np.floor(edge / 3.0).astype(np.int32), axis=0, return_counts=True)
    mode = (keys[np.argmax(counts)] + 0.5) * 3.0
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    A = _poly_terms((xx / sw - 0.5).ravel(), (yy / sh - 0.5).ravel())
    pix = small.reshape(-1, 3)
    mask = np.linalg.norm(pix - mode, axis=1) < 10.0
    coef, sigma = None, 2.0
    for _ in range(iters):
        if mask.sum() < 50:
            break
        coef, *_ = np.linalg.lstsq(A[mask], pix[mask], rcond=None)
        r = np.linalg.norm(pix - A @ coef, axis=1)
        sigma = max(0.6, 1.4826 * float(np.median(r[mask])))
        mask = r < max(3.0 * sigma, 2.0)
    if coef is None:
        return np.broadcast_to(mode.astype(np.float32), lab.shape).copy(), sigma
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (_poly_terms(xx / w - 0.5, yy / h - 0.5) @ coef).astype(np.float32), sigma


def prepare_trace(image: np.ndarray, max_side: int | None = None) -> TraceContext:
    work, f = _working_copy(image, max_side or TRACE_DEFAULTS["max_side"])
    smooth = cv2.bilateralFilter(work, 7, 18, 5)          # keeps edges, removes camera noise
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    field, sigma = _substrate_field(lab)
    mean = field.reshape(-1, 3).mean(axis=0)
    return TraceContext(lab - field + mean, f, mean, sigma)


def _clusters(ctx: TraceContext, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    if ctx.clusters is None:
        h, w = ctx.lab.shape[:2]
        s = max(1, int(round(max(h, w) / 320)))
        small = cv2.resize(ctx.lab, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        cv2.setRNGSeed(7)
        _, labels, centres = cv2.kmeans(small, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 3,
                                        cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        order = np.argsort(-counts)
        ctx.clusters = (centres[order], counts[order] / counts.sum())
    return ctx.clusters


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _component_at(mask: np.ndarray, sx: int, sy: int) -> np.ndarray | None:
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    lab = labels[sy, sx]
    if lab == 0:   # the click sits on a removed speck: take the nearest region within a few px
        win = labels[max(0, sy - 6):sy + 7, max(0, sx - 6):sx + 7]
        ys, xs = np.nonzero(win)
        if len(xs) == 0:
            return None
        k = np.argmin((ys + max(0, sy - 6) - sy) ** 2 + (xs + max(0, sx - 6) - sx) ** 2)
        lab = win[ys[k], xs[k]]
    return labels == lab


def _ring_colour(lab: np.ndarray, region: np.ndarray, width: int) -> np.ndarray | None:
    ring = cv2.dilate(region.astype(np.uint8), _disk(width)).astype(bool) & ~region
    pix = lab[ring]
    if len(pix) < 20:
        return None
    keys, counts = np.unique(np.floor(pix / 3.0).astype(np.int32), axis=0, return_counts=True)
    top = (keys[np.argmax(counts)] + 0.5) * 3.0
    near = pix[np.linalg.norm(pix - top, axis=1) < 4.5]
    return np.median(near, axis=0) if len(near) else top


def _gradient(lab: np.ndarray) -> np.ndarray:
    """Colour gradient magnitude (ΔE per working px)."""
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g += gx * gx + gy * gy
    return np.sqrt(g)


def _drop_residue(ctx: TraceContext, region: np.ndarray, ref: np.ndarray, sx: int, sy: int, p: dict) -> np.ndarray:
    """Remove tape residue from a whole-flake region.

    Residue has colours thin hBN can have, so colour does not tell them apart; its edge does. A flake
    (any thickness: a terrace, a thick rim, a wedge) ends in a sharp step to the substrate; residue
    fades out softly and is patchy inside. The region minus the clicked colour splits into pieces
    (thinner than ~6 px is left to the opening / neck cut); a piece is kept when its outer border is
    at least ``residue_sharp`` × as sharp (peak gradient / the piece's contrast) as the clicked
    colour's own border, or when it is a uniform plateau (gradient 75th percentile ≤ ``residue_flat``),
    or when it does not reach the region's border. Bright field only. Tuned on the lab's WSe2 batch-1
    images (docs/trace_benchmark.md)."""
    lab = ctx.lab
    if ctx.substrate[0] < 20.0:
        return region       # dark field: edges glow, the substrate is black — the edge test does not apply
    C = float(np.linalg.norm(ref - ctx.substrate))
    main = region & (np.linalg.norm(lab - ref, axis=2) < 0.5 * C)
    if main.sum() < 50:
        return region
    ru = region.astype(np.uint8)
    border = (ru - cv2.erode(ru, _disk(1))) > 0
    near_main = cv2.dilate(main.astype(np.uint8), _disk(4)) > 0
    G = _gradient(lab)
    Gpk = cv2.dilate(G, np.ones((5, 5), np.uint8))          # the step's peak gradient near each border px
    mb = border & near_main
    if mb.sum() < 20:
        return region
    s_main = float(np.median(Gpk[mb])) / max(C, 1e-6)
    pieces = cv2.morphologyEx((region & ~main).astype(np.uint8), cv2.MORPH_OPEN, _disk(3))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pieces, connectivity=8)
    drop = np.zeros(region.shape, bool)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < 40:
            continue
        comp = (labels == j).astype(np.uint8)
        outer = border & (cv2.dilate(comp, _disk(4)) > 0) & ~near_main
        if outer.sum() < 6:
            continue                                         # inside the flake
        c = float(np.linalg.norm(np.median(lab[comp > 0], axis=0) - ctx.substrate))
        sharp = float(np.median(Gpk[outer])) / max(c, 1e-6)
        if sharp >= p["residue_sharp"] * s_main:
            continue
        inner = cv2.erode(comp, _disk(2)) > 0
        if inner.sum() < 10:
            inner = comp > 0
        if float(np.percentile(G[inner], 75)) <= p["residue_flat"]:
            continue
        drop |= comp > 0
    if not drop.any():
        return region
    drop = (cv2.dilate(drop.astype(np.uint8), _disk(3)) > 0) & ~main       # with its fringe, never the flake itself
    m = (region & ~drop).astype(np.uint8)
    if p["open_r"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(int(p["open_r"])))
    comp = _component_at(m, sx, sy)
    return region if comp is None else comp


def trace_mask(ctx: TraceContext, seed: tuple[float, float], mode: str = "flake", bridge: int = 0,
               params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Region of the flake under ``seed`` (image px) on the working copy: (mask, outer contour).

    ``mode='flake'``: everything that is not substrate (all thicknesses of the flake), bounded where
    the contrast to the substrate drops to ``edge_frac`` of the clicked colour's.
    ``mode='colour'``: only the clicked colour — closer to it than half-way to every neighbouring
    colour (a monolayer attached to a thick piece; a buried layer seen through the stack).
    Neighbouring colours are learnt from a ring around the region (a flake on another flake), and
    regions joined only through a narrow neck are cut there. ``bridge`` (working px) closes gaps,
    e.g. where metal fingers cross the flake, before the region is chosen."""
    p = {**TRACE_DEFAULTS, **(params or {})}
    lab, f = ctx.lab, ctx.f
    h, w = lab.shape[:2]
    sx, sy = int(round(seed[0] * f)), int(round(seed[1] * f))
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("The seed point is outside the image")
    ref = np.median(lab[max(0, sy - 4):sy + 5, max(0, sx - 4):sx + 5].reshape(-1, 3), axis=0)
    floor = max(p["k_sigma"] * ctx.sigma, 2.5)
    if np.linalg.norm(ctx.substrate - ref) > floor:
        backs = [ctx.substrate]
    else:   # the click is on the main colour: the flake fills the field, the substrate is the next colour
        centres, _ = _clusters(ctx)
        far = [c for c in centres if np.linalg.norm(c - ref) > floor]
        if not far:
            raise ValueError("No contrast around the clicked point: is it on the substrate?")
        backs = [far[0]]
    region = None
    ring_w = max(4, int(0.004 * max(h, w)))
    for step in range(3):
        C = min(float(np.linalg.norm(ref - b)) for b in backs)
        if mode == "colour":
            inside = np.linalg.norm(lab - ref, axis=2) < max(0.5 * C, 0.75 * floor)
        else:
            dmin = np.min(np.stack([np.linalg.norm(lab - b, axis=2) for b in backs]), axis=0)
            inside = dmin > max(floor, p["edge_frac"] * C)
        m = inside.astype(np.uint8)
        if p["open_r"]:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(int(p["open_r"])))
        if bridge:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(int(bridge)))
        comp = None
        if p["neck_r"]:
            core = _component_at(cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(int(p["neck_r"]))), sx, sy)
            if core is not None:
                keep = cv2.dilate(core.astype(np.uint8), _disk(int(p["neck_r"]) + 1))
                comp = _component_at(m & keep, sx, sy)
        if comp is None:
            comp = _component_at(m, sx, sy)
        if comp is None:
            raise ValueError("No region found at the clicked point")
        region = comp
        if step == 2:
            break
        rb = _ring_colour(lab, region, ring_w)
        if rb is None or min(np.linalg.norm(rb - b) for b in backs) < floor or np.linalg.norm(rb - ref) < floor:
            break
        backs.append(rb)     # another flake (or a thicker part) borders the region: it bounds it too
    if mode == "flake" and p["residue"]:
        region = _drop_residue(ctx, region, ref, sx, sy, p)
    m = region.astype(np.uint8) * 255
    if p["close_r"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(int(p["close_r"])))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No region found at the clicked point")
    best = max(contours, key=cv2.contourArea)
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (float(sx), float(sy)), False) >= 0:
            best = cnt
            break
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [best], -1, 255, -1)     # holes (dust, bubbles) belong to the flake
    return filled > 0, best


def _contour_polygon(contour: np.ndarray, f: float, detail: float) -> list[tuple[float, float]]:
    eps = max(1.0, detail * cv2.arcLength(contour, True))
    return [(x / f, y / f) for x, y in simplify(contour.reshape(-1, 2), eps)]


def detect_flake(image: np.ndarray, seed: tuple[float, float], tolerance: float | None = None, max_side: int = 1600,
                 smooth: int = 5, detail: float | None = None, mode: str = "flake", bridge: int = 0,
                 context: TraceContext | None = None) -> list[tuple[float, float]]:
    """Border of the flake under ``seed`` (px) as a polygon whose vertices the user can edit (see
    :func:`trace_mask` for ``mode`` and ``bridge``). Pass ``context`` (from :func:`prepare_trace`)
    when clicking repeatedly on the same image. A numeric ``tolerance`` selects the v0.3 wand (flood
    fill within ``tolerance`` ΔE of the click, ``smooth`` px open/close), kept for comparison."""
    if tolerance is not None:
        return _detect_flake_fixed(image, seed, tolerance, max_side, smooth, detail or 0.004)
    ctx = context or prepare_trace(image, max_side)
    _, contour = trace_mask(ctx, seed, mode, bridge)
    return _contour_polygon(contour, ctx.f, TRACE_DEFAULTS["detail"] if detail is None else detail)


def flake_candidates(image: np.ndarray, prefer: str = "any", context: TraceContext | None = None,
                     min_area_frac: float = 0.004) -> list[dict]:
    """Where the flakes are, best first, for tracing without a click: one seed per region of one
    colour class, scored by size and closeness to the centre of the field, regions cut off by the
    field edge counting less. ``prefer='faint'`` puts low-contrast regions first (monolayers)."""
    ctx = context or prepare_trace(image)
    lab, f = ctx.lab, ctx.f
    h, w = lab.shape[:2]
    floor = ctx.floor
    contrast = np.linalg.norm(lab - ctx.substrate, axis=2)
    centres, _ = _clusters(ctx)
    colours = np.stack([ctx.substrate] + [c for c in centres if np.linalg.norm(c - ctx.substrate) > floor])
    cls = np.argmin(np.stack([np.linalg.norm(lab - c, axis=2) for c in colours]), axis=0)
    R = 0.45 * math.hypot(w, h)
    out = []
    for i in range(1, len(colours)):
        m = cv2.morphologyEx((cls == i).astype(np.uint8), cv2.MORPH_OPEN, _disk(int(TRACE_DEFAULTS["neck_r"])))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=4)
        for j in range(1, n):
            a = stats[j, cv2.CC_STAT_AREA]
            if a < min_area_frac * h * w:
                continue
            comp = (labels == j).astype(np.uint8)
            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            y, x = np.unravel_index(np.argmax(dt), dt.shape)
            d = math.hypot(cents[j][0] - w / 2, cents[j][1] - h / 2)
            c = float(np.median(contrast[labels == j]))
            score = math.sqrt(a / (h * w)) * math.exp(-(d / R) ** 2)
            x0, y0, ww, hh = stats[j, :4]
            score *= 0.6 ** sum((x0 <= 1, y0 <= 1, x0 + ww >= w - 1, y0 + hh >= h - 1))
            if prefer == "faint":
                score /= max(c, floor)
            out.append({"seed": (x / f, y / f), "score": float(score), "area_px": float(a / (f * f)), "contrast": c})
    out.sort(key=lambda d: -d["score"])
    return out


def auto_trace(image: np.ndarray, prefer: str = "any", mode: str | None = None, context: TraceContext | None = None,
               skip: int = 0) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Trace the most likely flake without a click (``skip`` = how many candidates to pass over, for
    "next candidate"). Returns (polygon px, seed px). ``mode`` defaults to 'colour' for
    ``prefer='faint'`` and 'flake' otherwise."""
    ctx = context or prepare_trace(image)
    cands = flake_candidates(image, prefer, ctx)
    if not cands:
        raise ValueError("No flake found in the image")
    seed = cands[skip % len(cands)]["seed"]
    return detect_flake(image, seed, mode=mode or ("colour" if prefer == "faint" else "flake"), context=ctx), seed


def _detect_flake_fixed(image: np.ndarray, seed: tuple[float, float], tolerance: float = 12.0, max_side: int = 1600,
                        smooth: int = 5, detail: float = 0.004) -> list[tuple[float, float]]:
    """The v0.3 wand: flood fill within ``tolerance`` (ΔE, OpenCV 8-bit Lab) of the median colour
    around the click, open/close by ``smooth`` px, outer contour. Unstable on graphite with steps,
    faint flakes and vignetted images (see docs/trace_benchmark.md); kept for comparison."""
    work, f = _working_copy(image, max_side)
    lab = cv2.cvtColor(cv2.GaussianBlur(work, (0, 0), 1.2), cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    sx, sy = int(round(seed[0] * f)), int(round(seed[1] * f))
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("The seed point is outside the image")
    # use the median colour around the seed as reference (robust to a dust speck under the click)
    r = 3
    patch = lab[max(0, sy - r):sy + r + 1, max(0, sx - r):sx + r + 1].reshape(-1, 3)
    ref = np.median(patch, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - ref[None, None, :], axis=2)
    within = (dist <= tolerance).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    region = within.copy()
    cv2.floodFill(region, mask, (sx, sy), 2, loDiff=0, upDiff=0, flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    blob = (region == 2).astype(np.uint8) * 255
    k = max(1, int(smooth))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No region found at the seed point; raise the tolerance")
    best = None
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (float(sx), float(sy)), False) >= 0:
            best = cnt
            break
    if best is None:
        best = max(contours, key=cv2.contourArea)
    eps = max(1.0, detail * cv2.arcLength(best, True))
    poly = simplify(best.reshape(-1, 2), eps)
    return [(x / f, y / f) for x, y in poly]




def segment_colors(image: np.ndarray, n_colors: int = 6, max_side: int = 800, min_area_frac: float = 0.002) -> list[dict]:
    """Split the image into ``n_colors`` colour classes (k-means in CIELab) and return every
    connected region with its polygon (px), class and mean colour — candidates the user can pick
    instead of clicking flakes one by one."""
    work, f = _working_copy(image, max_side)
    lab = cv2.cvtColor(cv2.GaussianBlur(work, (0, 0), 1.5), cv2.COLOR_BGR2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(lab.shape[:2])
    out = []
    min_area = min_area_frac * labels.size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for k in range(n_colors):
        m = cv2.morphologyEx((labels == k).astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < min_area:
                continue
            eps = max(1.0, 0.004 * cv2.arcLength(cnt, True))
            poly = simplify(cnt.reshape(-1, 2), eps)
            bgr = cv2.cvtColor(np.uint8([[centers[k]]]), cv2.COLOR_LAB2BGR)[0, 0]
            out.append({"cls": k, "area_px": a / (f * f), "polygon": [(x / f, y / f) for x, y in poly],
                        "color_rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])]})
    out.sort(key=lambda r: -r["area_px"])
    return out


@dataclass
class MarkerSquare:
    cx: float
    cy: float
    side: float


@dataclass
class MarkerGroup:
    squares: list[MarkerSquare]

    @property
    def center(self) -> tuple[float, float]:
        xs = [s.cx for s in self.squares]
        ys = [s.cy for s in self.squares]
        return float(np.mean(xs)), float(np.mean(ys))

    def to_json(self) -> dict:
        return {"center_px": list(self.center), "squares": [[s.cx, s.cy, s.side] for s in self.squares]}


def _square_like(cnt) -> tuple[float, float, float] | None:
    a = cv2.contourArea(cnt)
    if a <= 0:
        return None
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(cnt)
    if min(rw, rh) <= 0:
        return None
    if max(rw, rh) / min(rw, rh) < 1.35 and a / (rw * rh) > 0.8:
        return cx, cy, (rw + rh) / 2
    return None


def detect_marker_squares(image: np.ndarray, max_side: int = 2000, min_side_px: float = 8.0) -> list[MarkerSquare]:
    """Bright, square metal marks (the gold alignment markers look pale yellow in bright field).
    Thresholds the brightest, yellowish pixels; a blob that is not one square (e.g. two squares
    touching at a corner, as in the lab's diagonal pairs) is eroded step by step until it falls
    apart into squares."""
    work, f = _working_copy(image, max_side)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    thr = max(170.0, float(np.percentile(v, 99.0)) * 0.85)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 45) & (hsv[..., 1] >= 60) & (v >= thr)
    m = yellow.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    out = []
    kernel = np.ones((3, 3), np.uint8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if min(w, h) < min_side_px * f:
            continue
        blob = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        blob = cv2.copyMakeBorder(blob, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        for it in range(0, 9):
            eroded = cv2.erode(blob, kernel, iterations=it) if it else blob
            cs, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [c for c in cs if cv2.contourArea(c) > 4]
            found = [_square_like(c) for c in cs]
            if cs and all(found):
                for cx, cy, side in found:
                    side += 2 * it
                    if side >= min_side_px * f:
                        out.append(MarkerSquare((cx + x - 2) / f, (cy + y - 2) / f, side / f))
                break
    return out


def group_markers(squares: list[MarkerSquare], reach: float = 1.8) -> list[MarkerGroup]:
    """Squares closer than ``reach`` × their size belong to one marker (e.g. the lab's diagonal
    pairs of squares)."""
    groups: list[list[MarkerSquare]] = []
    for s in sorted(squares, key=lambda s: -s.side):
        for g in groups:
            if any(math.hypot(s.cx - o.cx, s.cy - o.cy) <= reach * max(s.side, o.side) for o in g):
                g.append(s)
                break
        else:
            groups.append([s])
    return [MarkerGroup(g) for g in groups]


def detect_markers(image: np.ndarray) -> list[MarkerGroup]:
    squares = detect_marker_squares(image)
    if squares:  # drop specks far smaller than the marker squares
        big = max(s.side for s in squares)
        squares = [s for s in squares if s.side >= 0.35 * big]
    return group_markers(squares)


def defect_candidates(image: np.ndarray, region_px: list[tuple[float, float]] | None = None, max_side: int = 1600,
                      min_d_px: float = 6, max_d_px: float = 120, contrast: float = 6.0) -> list[dict]:
    """Small round features that differ from their surroundings (bubbles, blisters, particles),
    optionally only inside ``region_px``. These are suggestions to review, not a verdict."""
    work, f = _working_copy(image, max_side)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    bg = cv2.medianBlur(L.astype(np.uint8), 31).astype(np.float32)
    chroma = [lab[..., c] - cv2.medianBlur(lab[..., c].astype(np.uint8), 31).astype(np.float32) for c in (1, 2)]
    diff = np.abs(L - bg) + 0.5 * np.hypot(*chroma)
    m = (diff > contrast).astype(np.uint8) * 255
    if region_px:
        roi = np.zeros_like(m)
        cv2.fillPoly(roi, [np.int32([[x * f, y * f] for x, y in region_px])], 255)
        roi = cv2.erode(roi, np.ones((7, 7), np.uint8))  # ignore the flake edge itself
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        d = 2 * r / f
        a = cv2.contourArea(cnt)
        if not (min_d_px <= d <= max_d_px) or a <= 0:
            continue
        roundness = a / (math.pi * r * r) if r > 0 else 0
        kind = "bubble" if roundness > 0.5 else "wrinkle"
        out.append({"kind": kind, "center_px": [cx / f, cy / f], "diameter_px": d, "roundness": round(roundness, 2),
                    "polygon": [(float(x) / f, float(y) / f) for x, y in cnt.reshape(-1, 2)]})
    return out
