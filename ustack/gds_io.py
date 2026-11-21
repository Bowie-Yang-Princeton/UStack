"""GDS reading and writing with the ``klayout`` Python package.

The written layout is in the chip frame of the bond-pad GDS (µm, the same origin), so Beamer /
the EBPG align it with the chip's markers. Layer numbers come only from config/layers.json;
missing ones raise :class:`~ustack.config.MissingValues` before anything is written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import klayout.db as kdb

from . import geometry as geo
from .config import LabConfig, MissingValues

Poly = list[tuple[float, float]]


def read_polygons(path: str | Path, cell: str | None = None) -> dict[tuple[int, int], list[Poly]]:
    """All shapes of the top cell (or ``cell``), flattened, as µm polygons per (layer, datatype)."""
    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.cell(cell) if cell else layout.top_cell()
    if top is None:
        raise ValueError(f"{path}: no cell {cell or '(top)'}")
    out: dict[tuple[int, int], list[Poly]] = {}
    dbu = layout.dbu
    for li in layout.layer_indexes():
        info = layout.get_info(li)
        polys = []
        it = top.begin_shapes_rec(li)
        while not it.at_end():
            shape = it.shape()
            if shape.is_polygon() or shape.is_box() or shape.is_path():
                poly = shape.polygon.transformed(it.trans())
                polys.append([(p.x * dbu, p.y * dbu) for p in poly.each_point_hull()])
            it.next()
        if polys:
            out[(info.layer, info.datatype)] = polys
    return out


def layers_in(path: str | Path) -> list[tuple[int, int, int]]:
    return sorted((l, d, len(p)) for (l, d), p in read_polygons(path).items())


def lead_tips(polys: Iterable[Poly], center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    """The point of each polygon closest to ``center`` — the inner end of each pre-patterned lead —
    for polygons that come within ``radius`` µm of it."""
    cx, cy = center
    tips = []
    for poly in polys:
        best, bd = None, math.inf
        n = len(poly)
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = 0 if L2 == 0 else max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / L2))
            px, py = x0 + t * dx, y0 + t * dy
            d = math.hypot(px - cx, py - cy)
            if d < bd:
                best, bd = (px, py), d
        if best is not None and bd <= radius and not geo.point_in_poly(cx, cy, poly):
            tips.append(best)
    return tips


def lead_ends(polys: Iterable[Poly], center: tuple[float, float], radius: float) -> list[dict]:
    """For the same leads as :func:`lead_tips` (same order): the inner end of each lead — "tip" (its point nearest
    ``center``), "mid" (the middle of its end edge), "dir" (unit vector from the end into the lead), "width" (the end
    edge's length) and "poly" (the lead)."""
    cx, cy = center
    out = []
    for poly in polys:
        n = len(poly)
        best, bd = None, math.inf
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = 0 if L2 == 0 else max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / L2))
            px, py = x0 + t * dx, y0 + t * dy
            d = math.hypot(px - cx, py - cy)
            if d < bd:
                best, bd = (px, py), d
        if best is None or bd > radius or geo.point_in_poly(cx, cy, poly):
            continue
        # the end edge: the edge whose middle is nearest the centre
        k = min(range(n), key=lambda i: math.hypot((poly[i][0] + poly[(i + 1) % n][0]) / 2 - cx,
                                                  (poly[i][1] + poly[(i + 1) % n][1]) / 2 - cy))
        (x0, y0), (x1, y1) = poly[k], poly[(k + 1) % n]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        L = math.hypot(x1 - x0, y1 - y0) or 1.0
        nx, ny = -(y1 - y0) / L, (x1 - x0) / L
        if not geo.point_in_poly(mx + nx * 0.05, my + ny * 0.05, poly):
            nx, ny = -nx, -ny
        out.append({"tip": best, "mid": (mx, my), "dir": (nx, ny), "width": L, "poly": [tuple(q) for q in poly]})
    return out


def write_gds(path: str | Path, top_name: str, shapes: dict[str, list[Poly]], config: LabConfig,
              texts: dict[str, list[tuple[str, float, float]]] | None = None, skip_reference: bool = False) -> Path:
    """Write ``shapes`` (purpose → µm polygons) on the layers config/layers.json gives each purpose."""
    missing = []
    layer_of: dict[str, tuple[int, int]] = {}
    for purpose, polys in shapes.items():
        if not polys:
            continue
        lay = config.layers.get(purpose)
        if skip_reference and lay is not None and lay.reference_only:
            continue
        try:
            layer_of[purpose] = config.gds_layer(purpose)
        except MissingValues as exc:
            missing += exc.items
    if missing:
        raise MissingValues(missing)
    layout = kdb.Layout()
    layout.dbu = geo.DBU
    top = layout.create_cell(top_name)
    for purpose, (ln, dt) in layer_of.items():
        li = layout.layer(kdb.LayerInfo(ln, dt, config.layers[purpose].name))
        for poly in shapes[purpose]:
            if len(poly) >= 3:
                top.shapes(li).insert(geo.polygon(poly))
        for label, x, y in (texts or {}).get(purpose, []):
            top.shapes(li).insert(kdb.Text(label, kdb.Trans(geo.to_dbu(x), geo.to_dbu(y))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(path))
    return path
