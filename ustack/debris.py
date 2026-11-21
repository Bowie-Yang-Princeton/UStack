"""Bubbles, cracks and stray flakes round the device (user, 2026-11-20): found in the aligned images of a step so the
electrode router keeps off them.

Finding (``find_in_image``), in each aligned image, at a working scale of ≥ ``WORK_UM`` µm/px:

1. CIELab, lightly smoothed. Everything the step already knows is masked: the metal (bond pads, markers, landing square,
   the fingers and the contact gate — grown by ``KNOWN_GROW_UM``) and the stack's own flakes (their outlines).
2. **Outside the stack** — the substrate colour is the main colour of the unmasked pixels, its illumination fall-off
   fitted by a quadratic; a pixel differing by more than max(``MIN_DE``, ``SIGMA_K`` × the substrate's noise) belongs to
   something on the substrate: stray flakes, tape residue, particles.
3. **On the stack** (inside its outlines, away from their edges by ``EDGE_BAND_UM``): differences from the local colour
   (median over ``LOCAL_UM``) — bubbles (compact blisters, bright or dark) and cracks / folds / wrinkles (thin lines).
4. Pieces smaller than ``MIN_AREA_UM2`` are dropped (sub-µm specks: noise and dust the lithography does not see).

Each candidate gets features (area, size, elongation, solidity, contrast, sign of the lightness change, texture, edge
sharpness, on the stack or not) and a kind from :class:`Classifier`: a k-nearest-neighbour vote over labelled examples —
the built-in set labelled on the lab's tWSe2 batch-1 images (``data/debris_examples.json``, written by
``tools/debris_training.py``) plus every correction the lab makes in the app (``library/training/debris.json``) — with
rules (below) where there are no examples. Kinds: ``flake``, ``particle``, ``bubble``, ``crack``; ``none`` = not a
feature (dropped).

Pure numpy + OpenCV + KLayout (no Qt)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from .alignment import Similarity
from .paths import PACKAGE, library_dir

Poly = list[tuple[float, float]]
KINDS = ("flake", "particle", "bubble", "crack")
KIND_NAMES = {"flake": "Stray flake", "particle": "Particle / residue", "bubble": "Bubble", "crack": "Crack / fold"}

#: Program choices (not lab values), tuned on the lab's batch-1 images (tools/debris_training.py).
WORK_UM = 0.12               # finest working scale, µm per px
KNOWN_GROW_UM = 1.5          # known metal / outlines grown by this before looking for anything else
EDGE_BAND_UM = 1.5           # the stack's own outline edges are not cracks (located outlines are good to ~0.5–1 µm)
EDGE_HUG_UM = 2.5            # a piece lying mostly within this of a layer's outline is that layer, not a feature
HUG_MAX = 0.6                # … "mostly" = this fraction of its area
STEP_EDGE_SIDES = 0.6        # a long piece on the stack whose two sides differ by this × its contrast is a step edge
STEP_EDGE_ELONG = 2.5
UNLOCATED_LAYER_UM2 = 100.0
CRACK_ELONG = 2.5            # a piece on the stack this elongated …
CRACK_SIDES = 0.3            # … with both sides alike (below this × its contrast) …
CRACK_MAX_WIDTH_UM = 3.0     # … and this thin on average is a crack / fold  # a piece off the stack this large that touches it is a layer not located, not a stray flake
MIN_DE = 6.0                 # colour difference from the substrate (ΔE, CIELab)
SIGMA_K = 5.0
LOCAL_UM = 4.0               # local colour on the stack: median over this
MIN_DE_STACK = 7.0
MIN_AREA_UM2 = 0.6
FEATURES = ("log_area", "elong", "solidity", "de_mean", "de_max", "dl", "texture", "sharp", "on_stack", "thin", "sides")


@dataclass
class Candidate:
    polygon: Poly                                # chip µm
    features: dict
    kind: str = ""
    score: float = 0.0
    image: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": self.kind, "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon], "score": round(self.score, 3),
                "how": "auto", "image": self.image, "features": {k: round(float(v), 4) for k, v in self.features.items()}}


# ====================================================================== masks
def _to_px(sim: Similarity, polys: list[Poly], f: float) -> list[np.ndarray]:
    out = []
    for p in polys:
        if len(p) >= 3:
            out.append(np.round(sim.inverse(np.asarray(p, float)) * f).astype(np.int32))
    return out


def _mask(shape, sim, polys, f, grow_px: int = 0) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for q in _to_px(sim, polys, f):                     # one at a time: fillPoly of several leaves their overlap empty
        cv2.fillPoly(m, [q], 1)
    if grow_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    return m


def _lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _substrate(lab: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, float]:
    """Background colour image (quadratic illumination fit of the substrate's main colour) and its noise (ΔE)."""
    h, w = free.shape
    ys, xs = np.nonzero(free)
    if len(xs) < 500:
        ys, xs = np.mgrid[0:h:8, 0:w:8].reshape(2, -1)
    step = max(1, len(xs) // 60000)
    ys, xs = ys[::step], xs[::step]
    pix = lab[ys, xs]
    q = np.round(pix / 3.0).astype(int)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    mode = keys[np.argmax(counts)] * 3.0
    close = np.linalg.norm(pix - mode, axis=1) < 8.0
    ys, xs, pix = ys[close], xs[close], pix[close]
    X = np.c_[np.ones(len(xs)), xs / w, ys / h, (xs / w) ** 2, (ys / h) ** 2, (xs / w) * (ys / h)]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx /= w
    gy /= h
    G = [np.ones_like(gx), gx, gy, gx * gx, gy * gy, gx * gy]
    bg = np.zeros_like(lab)
    resid = []
    for c in range(3):
        coef, *_ = np.linalg.lstsq(X, pix[:, c], rcond=None)
        bg[..., c] = sum(k * g for k, g in zip(coef, G))
        resid.append(pix[:, c] - X @ coef)
    de = np.linalg.norm(np.c_[resid].T, axis=1) if resid else np.zeros(1)
    sigma = float(1.4826 * np.median(np.abs(de - np.median(de)))) + float(np.median(de))
    return bg, max(sigma, 0.5)


# ====================================================================== candidates
def _sides(comp: np.ndarray, lab: np.ndarray, rect) -> float:
    """Colour difference (ΔE) between the two sides of a piece along its long axis: ~0 for a fold or a crack (the same
    flake on both sides), large for the edge between two layers."""
    (cx, cy), (w, h), ang = rect
    a = math.radians(ang if w >= h else ang + 90.0)
    nx, ny = -math.sin(a), math.cos(a)
    ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - cv2.dilate(comp, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(ring)
    if len(xs) < 8:
        return 0.0
    side = (xs - cx) * nx + (ys - cy) * ny
    one, two = lab[ys[side > 0], xs[side > 0]], lab[ys[side < 0], xs[side < 0]]
    if len(one) < 3 or len(two) < 3:
        return 0.0
    return float(np.linalg.norm(np.median(one, 0) - np.median(two, 0)))


def _features(comp: np.ndarray, de: np.ndarray, dl: np.ndarray, grad: np.ndarray, um_px: float, on_stack: bool,
              cnt: np.ndarray, lab: np.ndarray | None = None) -> dict:
    area = float(comp.sum()) * um_px * um_px
    rect = cv2.minAreaRect(cnt)
    a, b = sorted(rect[1])
    a, b = max(a, 1.0) * um_px, max(b, 1.0) * um_px
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    solidity = float(cv2.contourArea(cnt) / ha) if ha > 0 else 1.0
    vals = de[comp > 0]
    ring = cv2.dilate(comp, np.ones((3, 3), np.uint8)) - cv2.erode(comp, np.ones((3, 3), np.uint8))
    edge = grad[ring > 0]
    de_mean = float(vals.mean()) if len(vals) else 0.0
    thin = area / max(b, 1e-6)                     # mean width (µm) of a strip of that length
    sides = _sides(comp, lab, rect) / max(de_mean, 1e-6) if lab is not None else 0.0
    return {"sides": sides, "log_area": math.log10(max(area, 1e-3)), "elong": b / a, "solidity": solidity, "de_mean": de_mean,
            "de_max": float(np.percentile(vals, 95)) if len(vals) else 0.0, "dl": float(dl[comp > 0].mean()) if len(vals) else 0.0,
            "texture": float(vals.std() / max(de_mean, 1e-6)) if len(vals) else 0.0,
            "sharp": float(np.percentile(edge, 75) / max(de_mean, 1e-6)) if len(edge) else 0.0,
            "on_stack": 1.0 if on_stack else 0.0, "thin": thin, "area_um2": area, "length_um": b}


def _components(mask: np.ndarray, de, dl, grad, um_px, on_stack, f, sim, image_name, min_area, lab=None,
                hug: np.ndarray | None = None) -> list[Candidate]:
    out = []
    n, labels, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_px = min_area / (um_px * um_px)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        x, y, w, h = stats[i, :4]
        pad = 2
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, mask.shape[1]), min(y + h + pad, mask.shape[0])
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        cnts, _h = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        feats = _features(comp, de[y0:y1, x0:x1], dl[y0:y1, x0:x1], grad[y0:y1, x0:x1], um_px, on_stack, cnt,
                          lab[y0:y1, x0:x1] if lab is not None else None)
        if hug is not None:                              # how much of it lies along a located layer's outline
            feats["hug"] = float((hug[y0:y1, x0:x1][comp > 0] > 0).mean())
        approx = cv2.approxPolyDP(cnt, 0.6, True).reshape(-1, 2).astype(float) + (x0, y0)
        if len(approx) < 3:
            continue
        poly = [tuple(map(float, q)) for q in sim.apply(approx / f)]
        out.append(Candidate(poly, feats, image=image_name))
    return out


def find_in_image(img: np.ndarray, sim: Similarity, known_metal: list[Poly], stack: list[Poly], region: list[Poly] | None = None,
                  image_name: str = "", min_area_um2: float = MIN_AREA_UM2) -> list[Candidate]:
    """Candidates (not yet classified) in one aligned image; ``region`` (chip µm) limits where to look."""
    um_px0 = sim.scale
    f = min(1.0, um_px0 / WORK_UM)
    work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else img
    um_px = um_px0 / f
    work = cv2.GaussianBlur(work, (3, 3), 0.7)
    lab = _lab(work)
    shape = work.shape[:2]
    gpx = max(8, int(round(KNOWN_GROW_UM / um_px)))     # ≥ 8 px: at low magnification the metal's glow and the alignment (~1–3 µm)
    metal = _mask(shape, sim, known_metal, f, gpx)
    stack_m = _mask(shape, sim, stack, f, 0)
    look = _mask(shape, sim, region, f, 0) if region else np.ones(shape, np.uint8)
    # outside the stack: difference from the substrate
    known = cv2.dilate(stack_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gpx + 1, 2 * gpx + 1))) | metal
    bg, sigma = _substrate(lab, (known == 0) & (look > 0))
    diff = lab - bg
    de = np.linalg.norm(diff, axis=2)
    dl = diff[..., 0]
    gx = cv2.Sobel(de, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(de, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 8.0
    thr = max(MIN_DE, SIGMA_K * sigma)
    off = ((de > thr) & (known == 0) & (look > 0)).astype(np.uint8)
    off = cv2.morphologyEx(off, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    off = cv2.morphologyEx(off, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hug = np.zeros(shape, np.uint8)                      # a band along every located layer's outline
    hpx = max(2, int(round(EDGE_HUG_UM / um_px)))
    for poly in _to_px(sim, stack, f):
        cv2.polylines(hug, [poly], True, 1, 2 * hpx + 1)
    cands = _components(off, de, dl, grad, um_px, False, f, sim, image_name, min_area_um2, lab, hug)
    # on the stack: difference from the local colour
    if stack_m.any():
        k = max(3, int(round(LOCAL_UM / um_px)) | 1)
        k = min(k, 255 if k % 2 else 254)
        local = np.stack([cv2.medianBlur(np.clip(lab[..., c] + 128, 0, 255).astype(np.uint8), k).astype(np.float32) - 128
                          for c in range(3)], axis=2)
        d2 = lab - local
        de2 = np.linalg.norm(d2, axis=2)
        band = max(3, int(round(EDGE_BAND_UM / um_px)))
        edges = np.zeros(shape, np.uint8)
        for poly in _to_px(sim, stack, f):
            cv2.polylines(edges, [poly], True, 1, 2 * band + 1)
        inside = (stack_m > 0) & (edges == 0) & (metal == 0) & (look > 0)
        gx2 = cv2.Sobel(de2, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(de2, cv2.CV_32F, 0, 1, ksize=3)
        on = ((de2 > MIN_DE_STACK) & inside).astype(np.uint8)
        on = cv2.morphologyEx(on, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        on = cv2.morphologyEx(on, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cands += _components(on, de2, d2[..., 0], np.hypot(gx2, gy2) / 8.0, um_px, True, f, sim, image_name, min_area_um2, lab, hug)
    return cands


# ====================================================================== classification
def rule_kind(feat: dict) -> str:
    """The kind from the features alone (used where no labelled example is near)."""
    if feat["on_stack"] > 0.5:
        if feat["elong"] >= 3.5 or feat["thin"] < 0.6:
            return "crack"
        return "bubble" if feat["solidity"] > 0.6 else "crack"
    if feat.get("area_um2", 10 ** feat["log_area"]) < 4.0:
        return "particle"
    return "flake"


def _vec(feat: dict) -> np.ndarray:
    return np.array([feat.get(k, 0.0) for k in FEATURES], float)


class Classifier:
    """k-nearest neighbours on standardised features, over labelled examples (built in + the lab's corrections)."""

    def __init__(self, examples: list[dict] | None = None, k: int = 5):
        self.k = k
        self.examples = list(examples or [])
        self._fit()

    def _fit(self) -> None:
        if not self.examples:
            self.X = np.zeros((0, len(FEATURES)))
            self.y = []
            return
        X = np.array([_vec(e["features"]) for e in self.examples])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.X = (X - self.mu) / self.sd
        self.y = [e["kind"] for e in self.examples]

    def add(self, examples: list[dict]) -> None:
        self.examples += examples
        self._fit()

    def classify(self, feat: dict) -> tuple[str, float]:
        if len(self.y) < self.k:
            return rule_kind(feat), 0.5
        v = (_vec(feat) - self.mu) / self.sd
        d = np.linalg.norm(self.X - v, axis=1)
        idx = np.argsort(d)[: self.k]
        if d[idx[0]] > 4.0:                      # nothing labelled like it: the rules
            return rule_kind(feat), 0.4
        votes: dict[str, float] = {}
        for i in idx:
            votes[self.y[i]] = votes.get(self.y[i], 0.0) + 1.0 / (0.2 + d[i])
        kind = max(votes, key=votes.get)
        return kind, votes[kind] / sum(votes.values())


def builtin_examples_path() -> Path:
    return PACKAGE / "data" / "debris_examples.json"


def lab_examples_path() -> Path:
    return library_dir() / "training" / "debris.json"


def load_examples() -> list[dict]:
    out = []
    for p in (builtin_examples_path(), lab_examples_path()):
        try:
            out += json.loads(p.read_text(encoding="utf-8")).get("examples", [])
        except (OSError, ValueError):
            pass
    return out


def save_lab_examples(examples: list[dict]) -> Path:
    """Append the lab's corrections (kind + features) to its training set."""
    p = lab_examples_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(p.read_text(encoding="utf-8")).get("examples", [])
    except (OSError, ValueError):
        old = []
    p.write_text(json.dumps({"schema": "ustack.debris_examples/1", "examples": old + examples}, indent=1), encoding="utf-8")
    return p


def layer_veto(feat: dict) -> str | None:
    """Why a piece is the stack's own layers, not a feature (user, 2026-11-20: never a step edge taken for a crack, nor a
    layer for a stray flake): a long piece on the stack whose two sides differ (the edge between two layers), or a
    piece lying mostly along a located layer's outline (that layer, seen past its outline)."""
    if feat.get("on_stack", 0) > 0.5 and feat.get("elong", 0) >= STEP_EDGE_ELONG and feat.get("sides", 0) >= STEP_EDGE_SIDES:
        return "step edge between two layers"
    if feat.get("hug", 0) >= HUG_MAX:
        return "along a located layer's outline"
    if feat.get("on_stack", 0) < 0.5 and feat.get("hug", 0) > 0 and feat.get("area_um2", 0) >= UNLOCATED_LAYER_UM2:
        return "a large piece touching the stack: a layer not located"
    return None


def is_crack_shape(feat: dict) -> bool:
    """A long, thin piece on the stack with the same flake on both sides: a crack / fold, whatever the examples say
    (user, 2026-11-20: obvious cracks on the bottom hBN of Device 2026-11-16 were missed — the labelled set has few long
    ones, and its long pieces were mostly layer edges)."""
    return (feat.get("on_stack", 0) > 0.5 and feat.get("elong", 0) >= CRACK_ELONG and feat.get("sides", 1.0) <= CRACK_SIDES
            and feat.get("thin", 99.0) <= CRACK_MAX_WIDTH_UM)


def classify(cands: list[Candidate], clf: Classifier | None = None) -> list[Candidate]:
    clf = clf or Classifier(load_examples())
    out = []
    for c in cands:
        if layer_veto(c.features):
            continue
        c.kind, c.score = clf.classify(c.features)
        if c.kind == "none" and is_crack_shape(c.features):
            c.kind, c.score = "crack", 0.5
        if c.kind != "none":
            out.append(c)
    return out


# ====================================================================== several images
def merge(per_image: list[tuple[float, list[Candidate], Poly | None]]) -> list[Candidate]:
    """Candidates from several images ((µm/px, candidates, the image's frame in chip µm)): the finest image that saw a
    place decides there — a coarser image's candidate inside a finer image's frame is dropped."""
    import klayout.db as kdb
    kept: list[Candidate] = []
    seen = kdb.Region()
    for _um, cands, frame in sorted(per_image, key=lambda t: t[0]):
        for c in cands:
            r = geo.region(c.polygon)
            if seen.is_empty() or (r - seen).area() > 0.5 * r.area():
                kept.append(c)
        if frame:
            seen = (seen + geo.region(frame)).merged()
    return kept


def find_in_step(images: list[tuple[str, np.ndarray, Similarity]], known_metal: list[Poly], stack: list[Poly],
                 region: list[Poly] | None, clf: Classifier | None = None) -> list[Candidate]:
    """Classified features from every aligned image of a step (finest wins where they overlap)."""
    per = []
    for name, img, sim in images:
        h, w = img.shape[:2]
        frame = [tuple(map(float, q)) for q in sim.apply([(0, 0), (w, 0), (w, h), (0, h)])]
        per.append((sim.scale, find_in_image(img, sim, known_metal, stack, region, image_name=name), frame))
    return classify(merge(per), clf)


def describe(img: np.ndarray, sim: Similarity, polygon: Poly, known_metal: list[Poly], stack: list[Poly]) -> dict | None:
    """Features of a feature drawn by hand (to learn from it): the candidate found round it that overlaps it most."""
    import klayout.db as kdb
    target = geo.region(polygon)
    around = geo.to_polys(geo.sized(target, 3.0))
    best, bo = None, 0.0
    for c in find_in_image(img, sim, known_metal, stack, around, min_area_um2=0.1):
        o = geo.area(geo.region(c.polygon) & target)
        if o > bo:
            best, bo = c, o
    return best.features if best is not None and bo > 0.2 * geo.area(target) else None
