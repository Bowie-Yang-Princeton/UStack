"""The device specification (``ustack.device/1``): an ordered layer stack (bottom → top), the
device pattern (v1: one parameterised Hall bar), the contact scheme, gate limits, the explicit
user requirements and, after Stage 2, what was measured and built.

Every number is a :class:`Q` — a value with its provenance — so the program can tell what the
user specified (never to be changed by the program), what a template or the literature
suggested, what was measured and what the program chose itself (placement only).
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "ustack.device/1"

#: Layer functions. The first eight are the ones named in the lab's request; the dielectric /
#: encapsulation / leads entries are needed to describe hBN and connecting metal in a stack.
FUNCTIONS: dict[str, str] = {
    "substrate": "Substrate",
    "device_layer": "Device layer",
    "bottom_gate": "Bottom gate",
    "top_gate": "Top gate",
    "source_drain": "Source / drains",
    "hall_bar_contacts": "Hall bar contacts",
    "contact_gate": "Contact gate",
    "split_gate": "Split gates",
    "bottom_gate_dielectric": "Bottom-gate dielectric",
    "top_gate_dielectric": "Top-gate dielectric",
    "contact_gate_dielectric": "Contact-gate dielectric",
    "encapsulation": "Encapsulation / cap",
    "leads": "Leads (connecting electrodes)",
}
GATE_FUNCTIONS = ("bottom_gate", "top_gate", "contact_gate", "split_gate")
CONTACT_FUNCTIONS = ("source_drain", "hall_bar_contacts")
DIELECTRIC_FUNCTIONS = ("bottom_gate_dielectric", "top_gate_dielectric", "contact_gate_dielectric", "encapsulation")

#: Which functions a material category can take (design logic, not lab values).
FUNCTIONS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "substrate": ("substrate",),
    "2d_conductor": ("device_layer", "bottom_gate", "top_gate", "source_drain", "hall_bar_contacts", "contact_gate", "split_gate"),
    "2d_semiconductor": ("device_layer",),
    "2d_insulator": ("bottom_gate_dielectric", "top_gate_dielectric", "contact_gate_dielectric", "encapsulation"),
    "metal": ("bottom_gate", "top_gate", "source_drain", "hall_bar_contacts", "contact_gate", "split_gate", "leads"),
}

CONTACT_SCHEMES: dict[str, str] = {
    "edge_1d": "1D edge contacts (etched Hall bar, metal on the etched edge)",
    "prepatterned_bottom": "Pre-patterned bottom contacts (metal under the device layer)",
    "top_2d": "2D top contacts (selective etch to the device layer)",
}

FAMILIES: dict[str, str] = {
    "xu_tmd": "TMD · Xiaodong Xu (UW)",
    "mak_tmd": "TMD · Kin Fai Mak / Yiyu Xia (Cornell)",
    "xia_graphene": "Graphene · Liqiao Xia (MIT)",
    "ju_rhombohedral": "Rhombohedral graphene · Long Ju (MIT)",
    "custom": "Custom",
}

SOURCES = ("user", "template", "literature", "measured", "auto", "assistant", "material", "")


def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


@dataclass
class Q:
    """A number with its provenance. ``suggested`` is shown beside an empty field (e.g. the value a
    paper used) but is not a value until someone accepts it."""

    value: float | None = None
    source: str = ""
    suggested: float | None = None
    suggested_source: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None

    def set(self, value: float | None, source: str = "user") -> None:
        self.value = None if value is None else float(value)
        self.source = source if value is not None else ""

    def to_json(self) -> Any:
        if not self.suggested_source and self.suggested is None and not self.note:
            if self.value is None:
                return None
            return {"value": self.value, "source": self.source}
        data = {"value": self.value, "source": self.source}
        if self.suggested is not None:
            data["suggested"] = self.suggested
        if self.suggested_source:
            data["suggested_source"] = self.suggested_source
        if self.note:
            data["note"] = self.note
        return data

    @classmethod
    def from_json(cls, data: Any) -> "Q":
        if data is None:
            return cls()
        if isinstance(data, (int, float)):
            return cls(float(data), "user")
        return cls(data.get("value"), data.get("source", ""), data.get("suggested"), data.get("suggested_source", ""), data.get("note", ""))

    @classmethod
    def suggest(cls, value: float | None, source: str) -> "Q":
        return cls(None, "", value, source)


@dataclass
class MetalPart:
    metal: str
    thickness_nm: Q = field(default_factory=Q)

    def to_json(self) -> dict:
        return {"metal": self.metal, "thickness_nm": self.thickness_nm.to_json()}

    @classmethod
    def from_json(cls, data: dict) -> "MetalPart":
        return cls(data["metal"], Q.from_json(data.get("thickness_nm")))


@dataclass
class Layer:
    uid: str
    name: str
    material: str
    function: str
    thickness_nm: Q = field(default_factory=Q)
    layer_count: Q = field(default_factory=Q)
    width_um: Q = field(default_factory=Q)
    length_um: Q = field(default_factory=Q)
    twist_deg: Q = field(default_factory=Q)
    pieces: int = 1
    metal_parts: list[MetalPart] = field(default_factory=list)
    stacking: str = ""
    role: str = ""
    notes: str = ""
    literature: str = ""
    image: dict | None = None
    outline: list[list[float]] | None = None

    def to_json(self) -> dict:
        data = {
            "uid": self.uid, "name": self.name, "material": self.material, "function": self.function,
            "thickness_nm": self.thickness_nm.to_json(), "layer_count": self.layer_count.to_json(),
            "width_um": self.width_um.to_json(), "length_um": self.length_um.to_json(),
            "twist_deg": self.twist_deg.to_json(), "pieces": self.pieces,
            "metal_parts": [p.to_json() for p in self.metal_parts],
            "stacking": self.stacking, "role": self.role, "notes": self.notes, "literature": self.literature,
            "image": self.image, "outline": self.outline,
        }
        return {k: v for k, v in data.items() if v not in (None, "", [])} | {"uid": self.uid, "material": self.material, "function": self.function}

    @classmethod
    def from_json(cls, data: dict) -> "Layer":
        return cls(
            uid=data.get("uid") or new_uid(), name=data.get("name", ""), material=data.get("material", ""),
            function=data.get("function", ""), thickness_nm=Q.from_json(data.get("thickness_nm")),
            layer_count=Q.from_json(data.get("layer_count")), width_um=Q.from_json(data.get("width_um")),
            length_um=Q.from_json(data.get("length_um")), twist_deg=Q.from_json(data.get("twist_deg")),
            pieces=int(data.get("pieces", 1)), metal_parts=[MetalPart.from_json(p) for p in data.get("metal_parts", [])],
            stacking=data.get("stacking", ""), role=data.get("role", ""), notes=data.get("notes", ""),
            literature=data.get("literature", ""), image=data.get("image"), outline=data.get("outline"))

    @property
    def label(self) -> str:
        return self.name or f"{self.material} {FUNCTIONS.get(self.function, self.function)}"

    def total_metal_nm(self) -> float | None:
        if not self.metal_parts:
            return self.thickness_nm.value
        values = [p.thickness_nm.value for p in self.metal_parts]
        return None if any(v is None for v in values) else float(sum(values))


#: The Hall bar's parameters (v1's one pattern). Lengths in µm.
HALL_BAR_FIELDS: dict[str, tuple[str, str]] = {
    "channel_width_um": ("Channel width W", "um"),
    "channel_length_um": ("Probe span L (first to last voltage probe, centre to centre)", "um"),
    "n_probe_pairs": ("Voltage-probe pairs", ""),
    "arm_width_um": ("Probe arm width", "um"),
    "arm_length_um": ("Probe arm length (from the channel edge)", "um"),
    "sd_extension_um": ("Source/drain extension beyond the outer probes", "um"),
}


#: Geometry of the gates and contacts drawn around the Hall bar in Stage 2 (µm). These are design
#: choices per device, asked for when a step needs them (no lab defaults were supplied).
PATTERN_FIELDS: dict[str, tuple[str, str]] = {
    "tg_length_um": ("Metal top gate: length along the channel", "um"),
    "tg_overhang_um": ("Metal top gate: overhang beyond the graphene edge on the lead side", "um"),
    "cg_inner_um": ("Contact gate: reach from the contact tips toward the channel", "um"),
    "cg_outer_um": ("Contact gate: reach from the contact tips away from the channel", "um"),
    "cg_side_um": ("Contact gate: extension past the outermost contacts", "um"),
    "sg_gap_um": ("Split gate: gap to the top-gate edge", "um"),
    "pt_overlap_um": ("Pre-patterned electrodes: device layer overlap onto each electrode tip", "um"),
}

#: How the source and drain leave the channel. ``straight``: along the channel. ``bent``: each turns
#: 90° near its end, the source to +y and the drain to −y, so their contacts land off a gate strip
#: that runs the whole length of the device (and, for pre-patterned electrodes, both sit in the two
#: electrode columns). ``auto``: straight, unless Stage 2 finds a back gate under the S/D ends.
SD_STYLES = {"auto": "Automatic", "straight": "Straight", "bent": "Bent to the sides"}


@dataclass
class HallBar:
    params: dict[str, Q] = field(default_factory=lambda: {k: Q() for k in HALL_BAR_FIELDS})
    sd_style: str = "auto"          # SD_STYLES

    def get(self, key: str) -> float | None:
        q = self.params.get(key)
        return None if q is None else q.value

    def to_json(self) -> dict:
        return {k: q.to_json() for k, q in self.params.items()} | {"sd_style": self.sd_style}

    @classmethod
    def from_json(cls, data: dict | None) -> "HallBar":
        hb = cls()
        for k, v in (data or {}).items():
            if k == "sd_style":
                hb.sd_style = v if v in SD_STYLES else "auto"
            else:
                hb.params[k] = Q.from_json(v)
        return hb


@dataclass
class Requirement:
    """An explicit user request the design must keep satisfying (e.g. 'graphite back gate',
    'edge contacts', 'W = 2 µm'). ``kind``: layer | contact_scheme | param | material."""

    kind: str
    text: str
    match: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": self.kind, "text": self.text, "match": self.match}

    @classmethod
    def from_json(cls, data: dict) -> "Requirement":
        return cls(data["kind"], data.get("text", ""), dict(data.get("match", {})))


def new_uid() -> str:
    return "L" + uuid.uuid4().hex[:6]


@dataclass
class DeviceSpec:
    name: str = "Untitled device"
    family: str = "custom"
    template_id: str = ""
    description: str = ""
    layers: list[Layer] = field(default_factory=list)   # bottom → top
    pattern: str = "hall_bar"
    hall_bar: HallBar = field(default_factory=HallBar)
    contact_scheme: str = "edge_1d"
    pattern_params: dict[str, Q] = field(default_factory=dict)  # PATTERN_FIELDS, filled when Stage 2 needs them
    gate_limits_V: dict[str, Q] = field(default_factory=dict)  # layer uid → planned max |V|
    requirements: list[Requirement] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    process: list[str] = field(default_factory=list)          # fabrication order notes (template)
    stage2: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = field(default_factory=now_iso)
    modified: str = field(default_factory=now_iso)
    validation: dict = field(default_factory=dict)            # last report (written on save)

    # ------------------------------------------------------------------ access
    def layer(self, uid: str) -> Layer | None:
        return next((l for l in self.layers if l.uid == uid), None)

    def index(self, uid: str) -> int:
        return next(i for i, l in enumerate(self.layers) if l.uid == uid)

    def by_function(self, *functions: str) -> list[Layer]:
        return [l for l in self.layers if l.function in functions]

    def by_role(self, role: str) -> Layer | None:
        return next((l for l in self.layers if l.role == role), None)

    def device_layers(self) -> list[Layer]:
        return self.by_function("device_layer")

    def substrate(self) -> Layer | None:
        return next((l for l in self.layers if l.function == "substrate"), None)

    def move_layer(self, uid: str, new_index: int) -> None:
        i = self.index(uid)
        layer = self.layers.pop(i)
        self.layers.insert(max(0, min(new_index, len(self.layers))), layer)

    # ------------------------------------------------------------------ io
    def to_json(self) -> dict:
        return {
            "schema": SCHEMA, "id": self.id, "name": self.name, "family": self.family, "template_id": self.template_id,
            "description": self.description, "created": self.created, "modified": self.modified,
            "layers_order": "bottom_to_top", "layers": [l.to_json() for l in self.layers],
            "pattern": self.pattern, "hall_bar": self.hall_bar.to_json(), "contact_scheme": self.contact_scheme,
            "pattern_params": {k: q.to_json() for k, q in self.pattern_params.items()},
            "gate_limits_V": {k: q.to_json() for k, q in self.gate_limits_V.items()},
            "requirements": [r.to_json() for r in self.requirements], "references": self.references,
            "process": self.process, "stage2": self.stage2, "history": self.history, "validation": self.validation,
        }

    @classmethod
    def from_json(cls, data: dict) -> "DeviceSpec":
        if data.get("schema") not in (SCHEMA, None):
            raise ValueError(f"Unknown spec schema {data.get('schema')!r}")
        spec = cls(
            name=data.get("name", "Untitled device"), family=data.get("family", "custom"), template_id=data.get("template_id", ""),
            description=data.get("description", ""), layers=[Layer.from_json(l) for l in data.get("layers", [])],
            pattern=data.get("pattern", "hall_bar"), hall_bar=HallBar.from_json(data.get("hall_bar")),
            contact_scheme=data.get("contact_scheme", "edge_1d"),
            pattern_params={k: Q.from_json(v) for k, v in data.get("pattern_params", {}).items()},
            gate_limits_V={k: Q.from_json(v) for k, v in data.get("gate_limits_V", {}).items()},
            requirements=[Requirement.from_json(r) for r in data.get("requirements", [])],
            references=list(data.get("references", [])), process=list(data.get("process", [])),
            stage2=dict(data.get("stage2", {})), history=list(data.get("history", [])),
            validation=dict(data.get("validation", {})))
        spec.id = data.get("id", spec.id)
        spec.created = data.get("created", spec.created)
        spec.modified = data.get("modified", spec.modified)
        return spec

    def copy(self) -> "DeviceSpec":
        return DeviceSpec.from_json(copy.deepcopy(self.to_json()))

    def save(self, path: Path, note: str = "") -> Path:
        self.modified = now_iso()
        if note:
            self.history.append({"time": self.modified, "note": note})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DeviceSpec":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
