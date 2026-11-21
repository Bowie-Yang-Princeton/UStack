"""The flakes to look for, from a valid Stage 1 specification."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import LabConfig
from .model import FUNCTIONS, DeviceSpec, Layer
from .validation import Report, category, thickness_nm


class SpecNotValid(Exception):
    pass


@dataclass
class FlakeNeed:
    material: str
    purpose: str
    layers: list[str]                 # layer uids this flake becomes
    target: str                       # "1 layer", "12 nm", ...
    min_size_um: tuple[float, float]  # (length, width) of the piece needed
    pieces: int = 1
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"material": self.material, "purpose": self.purpose, "layers": self.layers, "target": self.target,
                "min_size_um": list(self.min_size_um), "pieces": self.pieces, "notes": self.notes}

    def text(self) -> str:
        l, w = self.min_size_um
        size = f"≥ {l:g} × {w:g} µm" + (f" (enough for {self.pieces} pieces)" if self.pieces > 1 else "")
        return f"{self.material} · {self.target} · {size} — {self.purpose}"


def flake_list(spec: DeviceSpec, config: LabConfig, report: Report) -> list[FlakeNeed]:
    """2D-material flakes to search for, in pick-up / stacking order (bottom → top).
    Only a valid specification has one."""
    if not report.ok:
        raise SpecNotValid(report.summary())
    out: list[FlakeNeed] = []
    skip: set[str] = set()
    layers = spec.layers
    shift = float(config.value("stack.alignment_shift_um") or 0.0)
    for i, layer in enumerate(layers):
        if layer.uid in skip or category(config, layer) not in ("2d_conductor", "2d_semiconductor", "2d_insulator"):
            continue
        group = [layer]
        if layer.pieces > 1:   # tear/cut-and-stack: the next pieces of the same material come from this flake
            for nxt in layers[i + 1:i + layer.pieces]:
                if nxt.material == layer.material and nxt.function == layer.function:
                    group.append(nxt)
                    skip.add(nxt.uid)
        notes: list[str] = []
        if layer.layer_count.value is not None:
            n = int(layer.layer_count.value)
            target = "monolayer" if n == 1 else f"{n} layers"
        else:
            t = thickness_nm(layer)
            target = f"{t:g} nm" if t is not None else "thickness open"
        if layer.outline and len(layer.outline) >= 3:
            # already traced on an image: the flake exists, report its size as traced
            xs, ys = zip(*layer.outline)
            length, width = max(xs) - min(xs), max(ys) - min(ys)
        else:
            # design size plus the stacking misalignment on every side
            length, width = (layer.length_um.value or 0.0) + 2 * shift, (layer.width_um.value or 0.0) + 2 * shift
            if shift:
                notes.append(f"Includes ±{shift:g} µm stacking misalignment (rule stack.alignment_shift_um) around the "
                             f"{layer.length_um.value or 0:g} × {layer.width_um.value or 0:g} µm design size.")
        pieces = len(group)
        if pieces > 1:
            notes.append(f"One flake cut into {pieces} pieces (tear/cut-and-stack); each piece ≥ {length:g} × {width:g} µm.")
            twists = [g.twist_deg.value for g in group[1:] if g.twist_deg.value is not None]
            if twists:
                notes.append("Twist " + ", ".join(f"{t:g}°" for t in twists) + " between pieces.")
        if layer.stacking:
            notes.append(f"{layer.stacking.capitalize()} stacking: confirm the domain (near-field IR / Raman) before pick-up.")
        if layer.role == "bottom_hbn" and spec.family == "xu_tmd":
            notes.append("Must be larger than the middle hBN and the TMD (Pt contacts are written on it).")
        if layer.role == "middle_hbn" and spec.family == "xu_tmd":
            notes.append("Thin; must reach ≥ 1 µm beyond every TMD edge.")
        if spec.family == "xia_graphene" and layer.role == "bottom_hbn":
            notes.append("Larger than the BG on three sides; the BG sticks out on the fourth.")
        if spec.family == "xia_graphene" and layer.role == "tg_hbn":
            notes.append("Covers the good graphene regions (graphene can tear at the top-hBN edge).")
        if layer.function == "device_layer":
            notes.append("Clean: no bubbles, folds or steps over the Hall-bar area.")
        if layer.notes:
            notes.append(layer.notes)
        out.append(FlakeNeed(config.material(layer.material).display_name, FUNCTIONS.get(layer.function, layer.function),
                             [g.uid for g in group], target, (length, width), pieces, notes))
    return out
