"""Polygon helpers on KLayout regions. Public functions take and return µm; regions are kept in
integer database units (1 DBU = 1 nm) so boolean operations and DRC checks are exact."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import klayout.db as kdb

DBU = 0.001  # µm per database unit
Point = tuple[float, float]
Poly = list[Point]


def to_dbu(v: float) -> int:
    return int(round(v / DBU))


def polygon(points: Sequence[Sequence[float]]) -> kdb.Polygon:
    return kdb.DPolygon([kdb.DPoint(float(x), float(y)) for x, y in points]).to_itype(DBU)


def region(*polys: Sequence[Sequence[float]] | None) -> kdb.Region:
    reg = kdb.Region()
    for pts in polys:
        if pts is not None and len(pts) >= 3:
            reg.insert(polygon(pts))
    reg.merge()
    return reg


def regions(polys: Iterable[Sequence[Sequence[float]]]) -> kdb.Region:
    return region(*list(polys))


def rect(x0: float, y0: float, x1: float, y1: float) -> Poly:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def to_polys(reg: kdb.Region) -> list[Poly]:
    """Region → list of µm point lists (holes are resolved into the hull by KLayout's
    ``resolved_holes``, so every polygon is a simple outline)."""
    out = []
    for poly in reg.each_merged():
        simple = poly.resolved_holes() if poly.holes() else poly
        out.append([(p.x * DBU, p.y * DBU) for p in simple.each_point_hull()])
    return out


def area(reg: kdb.Region) -> float:
    return reg.area() * DBU * DBU


def sized(reg: kdb.Region, d_um: float) -> kdb.Region:
    return reg.sized(to_dbu(d_um)) if d_um else kdb.Region(reg)


def rounded(reg: kdb.Region, radius_um: float, points: int = 32) -> kdb.Region:
    if not radius_um or reg.is_empty():
        return kdb.Region(reg)
    r = to_dbu(radius_um)
    return reg.rounded_corners(r, r, points)


def bbox(reg: kdb.Region) -> tuple[float, float, float, float]:
    b = reg.bbox()
    return b.left * DBU, b.bottom * DBU, b.right * DBU, b.top * DBU


def transform(points: Iterable[Sequence[float]], angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> Poly:
    a = math.radians(angle_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    return [(c * x - s * y + dx, s * x + c * y + dy) for x, y in points]


def edge_pairs_to_points(pairs: kdb.EdgePairs, limit: int = 20) -> list[Point]:
    """Midpoints of the first ``limit`` violation markers (µm), for showing where a check failed."""
    out = []
    for i, ep in enumerate(pairs.each()):
        if i >= limit:
            break
        b = ep.bbox()
        out.append(((b.left + b.right) * 0.5 * DBU, (b.bottom + b.top) * 0.5 * DBU))
    return out


#: Grid tolerance for DRC: off-axis edges snapped to the 1 nm grid lose up to ~1 nm of width.
GRID_TOL_UM = 0.002


def width_violations(reg: kdb.Region, min_um: float) -> list[Point]:
    if reg.is_empty() or not min_um:
        return []
    min_um = max(0.0, min_um - GRID_TOL_UM)
    # projection metrics: a 90° corner (a path end, a mitre) is not a width violation
    # projection metrics, edges meeting at ≥ 80° ignored: a path end or a mitre is not a width violation
    return edge_pairs_to_points(reg.width_check(to_dbu(min_um), False, kdb.Region.Projection, 80))


def space_violations(reg: kdb.Region, min_um: float) -> list[Point]:
    if reg.is_empty() or not min_um:
        return []
    min_um = max(0.0, min_um - GRID_TOL_UM)
    # projection metrics, edges meeting at ≥ 80° ignored (as for width): the inner corner of a rotated
    # comb or L is not a spacing violation
    return edge_pairs_to_points(reg.space_check(to_dbu(min_um), False, kdb.Region.Projection, 80))


def separation_violations(a: kdb.Region, b: kdb.Region, min_um: float) -> list[Point]:
    """Places where ``a`` and ``b`` come closer than ``min_um`` (including overlaps)."""
    if a.is_empty() or b.is_empty():
        return []
    out = edge_pairs_to_points(a.separation_check(b, to_dbu(min_um)))
    overlap = a & b
    if not overlap.is_empty():
        out += [centroid(p) for p in to_polys(overlap)[:10]]
    return out


def enclosure_violations(outer: kdb.Region, inner: kdb.Region, margin_um: float) -> list[Point]:
    """Places where ``inner`` is not inside ``outer`` with at least ``margin_um`` to spare."""
    if inner.is_empty():
        return []
    outside = inner - outer
    pts = [centroid(p) for p in to_polys(outside)[:10]]
    if margin_um:
        pts += edge_pairs_to_points(outer.enclosing_check(inner, to_dbu(margin_um)))
    return pts


def centroid(points: Sequence[Sequence[float]]) -> Point:
    pts = list(points)
    a = cx = cy = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-12:
        xs, ys = zip(*pts)
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return (cx / (3 * a), cy / (3 * a))


def poly_area(points: Sequence[Sequence[float]]) -> float:
    pts = list(points)
    return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2


def point_in_poly(x: float, y: float, points: Sequence[Sequence[float]]) -> bool:
    inside = False
    pts = list(points)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def distance_to_region(x: float, y: float, reg: kdb.Region) -> float:
    """Distance (µm) from a point to the nearest edge of ``reg`` (0 inside)."""
    p = kdb.Point(to_dbu(x), to_dbu(y))
    best = math.inf
    for poly in reg.each_merged():
        if poly.inside(p):
            return 0.0
        for e in poly.each_edge():
            best = min(best, e.euclidian_distance(p) if hasattr(e, "euclidian_distance") else _edge_dist(e, p))
    return best * DBU


def _edge_dist(e: kdb.Edge, p: kdb.Point) -> float:
    ax, ay, bx, by = e.p1.x, e.p1.y, e.p2.x, e.p2.y
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p.x - ax) * dx + (p.y - ay) * dy) / L2))
    return math.hypot(ax + t * dx - p.x, ay + t * dy - p.y)


def nominal_rect(width_um: float, length_um: float, cx: float = 0.0, cy: float = 0.0) -> Poly:
    """A flake's nominal footprint when no outline was traced: width along y, length along x."""
    return rect(cx - length_um / 2, cy - width_um / 2, cx + length_um / 2, cy + width_um / 2)


def region_distance(a: kdb.Region, b: kdb.Region) -> float:
    """Smallest distance (µm) between two regions (0 when they touch or overlap): for polygons apart
    it is reached at a vertex of one of them."""
    if a.is_empty() or b.is_empty():
        return math.inf
    if not (a & b).is_empty():
        return 0.0
    best = math.inf
    for src, dst in ((a, b), (b, a)):
        for poly in src.each_merged():
            for pt in poly.each_point_hull():
                best = min(best, distance_to_region(pt.x * DBU, pt.y * DBU, dst))
    return best


def grown_round(reg: kdb.Region, d_um: float, points: int = 64) -> kdb.Region:
    """``reg`` grown by ``d_um`` in every direction (Minkowski sum with a disk, its polygon drawn just
    outside the circle): exact distance, unlike ``sized`` whose corners reach farther."""
    if reg.is_empty() or d_um <= 0:
        return kdb.Region(reg)
    r = d_um / math.cos(math.pi / points)
    disk = kdb.Polygon([kdb.Point(to_dbu(r * math.cos(2 * math.pi * k / points)), to_dbu(r * math.sin(2 * math.pi * k / points)))
                        for k in range(points)])
    out = kdb.Region()
    for poly in reg.each_merged():
        out.insert(poly.minkowski_sum(disk, False))
    return out.merged()
