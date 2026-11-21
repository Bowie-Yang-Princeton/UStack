"""What the 3D preview draws: one extruded footprint per layer (Qt-free, so it can be tested).

Footprints come from the traced outline when there is one, else from the nominal width × length
(centred), and for metal layers from the Hall-bar geometry (contacts, gates, leads). Heights are
the real thicknesses mapped through a log scale so a 0.34 nm graphene sheet and a 285 nm oxide
both stay visible; unknown thicknesses get a placeholder height and are flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import geometry as geo
from .config import LabConfig
from .hallbar import HallBar, lead_path, escape_point
from .model import DeviceSpec, Layer
from .validation import category, thickness_nm

MATERIAL_COLORS = {
    "graphene": "#4a4d55", "graphite": "#7b68a6", "hBN": "#9cc3e6", "WSe2": "#3fae9b", "MoTe2": "#4f9d6b",
    "WS2": "#6bbf8a", "MoS2": "#5aa0a8", "MoSe2": "#7fb36a",
    "Au": "#d8b23a", "Pt": "#b9bcc2", "Pd": "#c7cacf", "Cr": "#8e9196", "Ti": "#a4a9ae", "PdAu": "#d2bd6a",
    "substrate": "#c8c2da",
}


def layer_color(layer: Layer, config: LabConfig) -> str:
    mat = config.material(layer.material) if layer.material else None
    if mat is None:
        return "#bfc3c9"
    if mat.category == "metal":
        return MATERIAL_COLORS.get(mat.components[-1] if mat.components else "", "#c9a63a")
    if mat.category == "substrate":
        return MATERIAL_COLORS["substrate"]
    return MATERIAL_COLORS.get(mat.id, "#9aa0a8")


@dataclass
class Slab:
    uid: str
    name: str
    color: str
    polys: list[geo.Poly]
    z0: float
    height: float
    thickness_nm: float | None
    translucent: bool
    known: bool


def display_height(t_nm: float | None) -> float:
    if t_nm is None:
        return 0.6
    return 0.25 + 0.55 * math.log10(1.0 + t_nm)


def _nominal_hallbar(spec: DeviceSpec) -> HallBar | None:
    try:
        return HallBar.from_spec(spec)
    except Exception:
        return None


def footprints(spec: DeviceSpec, config: LabConfig, layer: Layer, hb: HallBar | None, extent: float) -> list[geo.Poly]:
    if layer.outline and len(layer.outline) >= 3:
        return [[tuple(p) for p in layer.outline]]
    cat = category(config, layer)
    if layer.function == "substrate":
        e = extent * 0.75
        return [geo.rect(-e, -e * 0.7, e, e * 0.7)]
    if cat == "metal" and hb is not None:
        if layer.function == "hall_bar_contacts" and spec.contact_scheme == "prepatterned_bottom" and hb.bent:
            # two pre-patterned electrode columns: the probe fingers and the bent S/D fingers themselves
            return [p for _, p in hb.local_arms()] + [p for _, p in hb.local_legs()]
        if layer.function in ("hall_bar_contacts", "source_drain", "leads"):
            polys = []
            for t in hb.terminals():
                if layer.function == "source_drain" and t.kind == "probe":
                    continue
                inner = -(hb.arm_l if t.kind == "probe" else 0.0) * 0.5
                outer = extent * 0.55 if layer.function == "leads" else hb.arm_l * 1.4 + 1.0
                start = escape_point(t, inner if layer.function != "leads" else hb.arm_l * 1.4 + 1.0)
                w = min(t.width, hb.arm_w) if t.kind != "probe" else hb.arm_w
                polys.append(lead_path([start, escape_point(t, outer)], max(w * 0.8, 0.3)))
            return polys
        if layer.function == "top_gate":
            return [geo.rect(-hb.L / 2, -hb.W / 2 - hb.arm_l * 0.4, hb.L / 2, hb.W / 2 + hb.arm_l * 0.4)]
        if layer.function == "contact_gate":
            xs = hb.probe_x()
            a, b = min(xs) - hb.arm_w, max(xs) + hb.arm_w
            y0, y1 = hb.W / 2 + hb.arm_l * 0.5, hb.W / 2 + hb.arm_l * 1.2
            return [geo.rect(a, y0, b, y1), geo.rect(a, -y1, b, -y0)]
        if layer.function == "split_gate":
            lt = hb.total_length
            return [geo.rect(-lt / 2, hb.W / 2 + 0.3, lt / 2, hb.W / 2 + 0.3 + hb.arm_l * 0.4),
                    geo.rect(-lt / 2, -hb.W / 2 - 0.3 - hb.arm_l * 0.4, lt / 2, -hb.W / 2 - 0.3)]
        if layer.function == "bottom_gate":
            return [geo.rect(-hb.total_length / 2, -hb.W / 2 - 1, hb.total_length / 2, hb.W / 2 + 1)]
    w, l = layer.width_um.value, layer.length_um.value
    if w and l:
        return [geo.nominal_rect(w, l)]
    if hb is not None:
        fl, fw = hb.footprint()
        return [geo.nominal_rect(fw * 1.1, fl * 1.1)]
    return [geo.nominal_rect(extent * 0.5, extent * 0.7)]


def build_scene(spec: DeviceSpec, config: LabConfig, chip: dict[str, list[geo.Poly]] | None = None) -> list[Slab]:
    """One slab per layer. ``chip`` (``ebl_flow.structure_footprints``): footprints from Stage 2 (the flakes where
    they are on the chip, the patterned metal as generated) — those layers use them, the others their usual footprint."""
    hb = _nominal_hallbar(spec)
    sizes = [max(l.width_um.value or 0, l.length_um.value or 0) for l in spec.layers]
    if hb is not None:
        sizes.append(max(hb.footprint()) * 1.6)
    if chip:
        pts = [abs(c) for polys in chip.values() for poly in polys for q in poly for c in q]
        if pts:
            sizes = [2 * max(pts)]
    extent = max(sizes + [10.0])
    slabs: list[Slab] = []
    z = 0.0
    for layer in spec.layers:
        t = thickness_nm(layer)
        h = display_height(t) if layer.function != "substrate" else 1.2
        polys = chip[layer.uid] if chip is not None and layer.uid in chip else footprints(spec, config, layer, hb, extent)
        cat = category(config, layer)
        slabs.append(Slab(layer.uid, layer.label, layer_color(layer, config), polys, z, h, t,
                          translucent=layer.material == "hBN", known=t is not None))
        # every layer sits wholly above the one before (no draping into pre-patterned fingers): the 3D view
        # paints bottom to top, which is only exact when layers do not interpenetrate in height
        z += h
    return slabs
