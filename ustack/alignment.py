"""Align a device image to the chip's bond-pad GDS through the alignment markers.

The transform maps image pixels (x right, y down) to chip µm (y up):

    [X, Y] = s · R(θ) · [x, ±y] + [tx, ty]

i.e. a similarity transform with the image y axis flipped (``mirror`` flips x as well, for a
camera that mirrors the image). It is estimated from marker squares found in the image and the
marker polygons of the GDS (automatic), or from point pairs the user clicks (manual).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import MarkerSquare


@dataclass
class Similarity:
    scale: float          # µm per px
    angle_deg: float
    tx: float
    ty: float
    mirror: bool = False
    rms_um: float | None = None
    pairs: list[dict] = field(default_factory=list)

    def _pre(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2).copy()
        xy[:, 1] *= -1.0
        if self.mirror:
            xy[:, 0] *= -1.0
        return xy

    def apply(self, xy) -> np.ndarray:
        a = math.radians(self.angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return (self.scale * (R @ self._pre(xy).T)).T + np.array([self.tx, self.ty])

    def inverse(self, XY) -> np.ndarray:
        a = math.radians(self.angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = (R.T @ ((np.asarray(XY, float).reshape(-1, 2) - [self.tx, self.ty]).T)).T / self.scale
        if self.mirror:
            p[:, 0] *= -1.0
        p[:, 1] *= -1.0
        return p

    def to_json(self) -> dict:
        return {"um_per_px": self.scale, "angle_deg": self.angle_deg, "tx_um": self.tx, "ty_um": self.ty,
                "mirror": self.mirror, "rms_um": self.rms_um, "pairs": self.pairs,
                "convention": "chip_um = s*R(angle)*[x, -y](px, x mirrored if mirror) + t"}

    @classmethod
    def from_json(cls, d: dict) -> "Similarity":
        return cls(d["um_per_px"], d["angle_deg"], d["tx_um"], d["ty_um"], d.get("mirror", False), d.get("rms_um"), d.get("pairs", []))


def fit_similarity(src_px, dst_um, mirror: bool = False, scale: float | None = None) -> Similarity:
    """Least-squares similarity (Umeyama) from ≥ 2 point pairs; with ``scale`` (µm/px) given, only the turn and the shift
    are fitted (one pair is then enough)."""
    src = np.asarray(src_px, float).reshape(-1, 2)
    dst = np.asarray(dst_um, float).reshape(-1, 2)
    if len(src) < (1 if scale else 2):
        raise ValueError("At least two point pairs are needed")
    tmp = Similarity(1, 0, 0, 0, mirror)
    p = tmp._pre(src)
    mp, md = p.mean(0), dst.mean(0)
    pc, dc = p - mp, dst - md
    # complex-number form of the 2D Procrustes problem
    zp = pc[:, 0] + 1j * pc[:, 1]
    zd = dc[:, 0] + 1j * dc[:, 1]
    if scale:
        dot = np.vdot(zp, zd)
        k = scale * (dot / abs(dot) if abs(dot) > 1e-12 else 1.0)
    else:
        k = np.vdot(zp, zd) / np.vdot(zp, zp)
    scale, ang = abs(k), math.degrees(np.angle(k))
    rot = np.array([[math.cos(np.angle(k)), -math.sin(np.angle(k))], [math.sin(np.angle(k)), math.cos(np.angle(k))]])
    t = md - scale * (rot @ mp)
    sim = Similarity(float(scale), float(ang), float(t[0]), float(t[1]), mirror)
    res = sim.apply(src) - dst
    sim.rms_um = float(np.sqrt((res ** 2).sum(1).mean()))
    sim.pairs = [{"px": [float(a), float(b)], "um": [float(c), float(d)], "residual_um": float(math.hypot(*r))}
                 for (a, b), (c, d), r in zip(src, dst, res)]
    return sim


@dataclass
class GdsSquare:
    cx: float
    cy: float
    side: float


def squares_from_polygons(polys: list[list[tuple[float, float]]]) -> list[GdsSquare]:
    out = []
    for poly in polys:
        xs, ys = zip(*poly)
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w <= 0 or h <= 0:
            continue
        if max(w, h) / min(w, h) < 1.35:
            out.append(GdsSquare((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (w + h) / 2))
    return out


def auto_align(img_squares: list[MarkerSquare], gds_squares: list[GdsSquare], site: tuple[float, float] | None = None,
               site_radius: float | None = None, mirror: bool = False, size_tol: float = 0.25,
               scale_range: tuple[float, float] | None = None) -> Similarity:
    """Match image marker squares to GDS marker squares and fit the transform.

    Each pair of image squares, paired with each pair of GDS squares of matching size, proposes a
    transform; the one that brings the most image squares onto a GDS square (within a third of
    the square's side) wins and is refined on all its matches. ``site``/``site_radius`` (µm)
    restrict the GDS squares to one device site — necessary when every site has the same marker
    layout. ``scale_range`` (µm/px) rejects implausible magnifications."""
    if site is not None and site_radius:
        gds_squares = [g for g in gds_squares if math.hypot(g.cx - site[0], g.cy - site[1]) <= site_radius]
    if len(img_squares) < 2 or len(gds_squares) < 2:
        raise ValueError("Need at least two marker squares in the image and in the GDS (near the chosen site)")
    img = np.array([[s.cx, s.cy] for s in img_squares])
    img_side = np.array([s.side for s in img_squares])
    gds = np.array([[g.cx, g.cy] for g in gds_squares])
    gds_side = np.array([g.side for g in gds_squares])
    best = None
    order = np.argsort(-img_side)
    cand_i = [int(i) for i in order[:8]]
    for i, j in itertools.permutations(cand_i, 2):
        d_img = np.hypot(*(img[i] - img[j]))
        if d_img < 1e-6:
            continue
        ratio_ij = img_side[i] / img_side[j]
        for a in range(len(gds)):
            for b in range(len(gds)):
                if a == b:
                    continue
                if abs(gds_side[a] / gds_side[b] - ratio_ij) > size_tol * ratio_ij:
                    continue
                s_pts = np.hypot(*(gds[a] - gds[b])) / d_img
                s_size = gds_side[a] / img_side[i]
                if abs(s_pts / s_size - 1) > size_tol:
                    continue
                if scale_range and not (scale_range[0] <= s_pts <= scale_range[1]):
                    continue
                sim = fit_similarity(img[[i, j]], gds[[a, b]], mirror)
                proj = sim.apply(img)
                dist = np.hypot(*(proj[:, None, :] - gds[None, :, :]).transpose(2, 0, 1))
                nearest = dist.argmin(1)
                ok = dist[np.arange(len(img)), nearest] < gds_side[nearest] / 3
                # one GDS square per image square
                used, matches = set(), []
                for k in np.argsort(dist[np.arange(len(img)), nearest]):
                    if ok[k] and nearest[k] not in used:
                        used.add(int(nearest[k]))
                        matches.append((int(k), int(nearest[k])))
                score = (len(matches), -float(np.mean([dist[k, g] for k, g in matches])) if matches else 0)
                if best is None or score > best[0]:
                    best = (score, matches)
    if best is None or len(best[1]) < 2:
        raise ValueError("No consistent marker match found; pick the marker pairs by hand")
    matches = best[1]
    sim = fit_similarity(img[[k for k, _ in matches]], gds[[g for _, g in matches]], mirror)
    return sim
