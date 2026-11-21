"""The lab configuration: materials (config/materials.csv), rules and approved defaults
(config/rules.json), the GDS layer map (config/layers.json) and the design presets
(config/presets.json: the relations Configure uses to generate dimensions, and the microscope's
field of view).

Every value carries a status:

* ``approved``   stated by the lab (the user) or approved in the Rules editor;
* ``pending``    copied from an attached lab document, waiting for approval;
* ``literature`` from a published Methods section (group templates), waiting for approval;
* ``missing``    not supplied. Code that needs it calls :meth:`LabConfig.require`, which raises
  :class:`MissingValues`; the UI catches it, asks the user and stores the answer as approved.

Nothing here invents a process limit, a material or a layer number.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import paths

STATUSES = ("approved", "pending", "literature", "missing")

#: Device families (the four group structures plus a free-form one) and the rule tags they match.
FAMILY_TAGS: dict[str, set[str]] = {
    "xu_tmd": {"all", "tmd", "xu_tmd"},
    "mak_tmd": {"all", "tmd", "mak_tmd"},
    "xia_graphene": {"all", "graphene", "xia_graphene"},
    "ju_rhombohedral": {"all", "graphene", "rhombohedral", "ju_rhombohedral"},
    "custom": {"all"},
}


#: Design presets: the program's choices for Configure's relations (status ``program``, editable in
#: the Presets dialog) and the microscope field of view stated by the user. config/presets.json
#: overrides these; a file without an entry falls back to the value here.
PRESET_DEFAULTS: dict[str, dict] = {
    "hall_bar.width_per_arm": {
        "value": 3.0, "unit": "×", "group": "Hall bar", "label": "Channel width W / arm width",
        "description": "W = this × arm width (arm width = electrode width target). Narrow probes next to a wide channel barely short the Hall voltage."},
    "hall_bar.pitch_per_width": {
        "value": 2.0, "unit": "×", "group": "Hall bar", "label": "Probe pitch / channel width",
        "description": "Probe pitch = max(this × W, arm width + spacing, arm width + arm-spacing minimum); L = (pairs − 1) × pitch."},
    "hall_bar.n_probe_pairs": {
        "value": 2, "unit": "", "group": "Hall bar", "label": "Voltage-probe pairs",
        "description": "2 is the minimum for a four-probe Rxx and an Rxy."},
    "hall_bar.arm_length_per_width_no_gate": {
        "value": 1.0, "unit": "×", "group": "Hall bar", "label": "Arm length / W (no gate to clear)",
        "description": "Without a gate to clear: arm length = max(this × W, contact overlap + spacing)."},
    "hall_bar.sd_extension_per_width_no_gate": {
        "value": 1.0, "unit": "×", "group": "Hall bar", "label": "S/D extension / W (no gate to clear)",
        "description": "Without a gate to clear: source/drain extension beyond the outer probes = this × W."},
    "gates.margin_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Gates and flakes", "label": "Gate margin",
        "description": "Gate margin g = max(stacking shift, this × electrode spacing): how far a graphite gate is designed beyond the channel."},
    "gates.bg_extra_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Gates and flakes", "label": "Extra BG size under a TG",
        "description": "A bottom gate under a top gate is this × spacing larger on every side."},
    "hbn.margin_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Gates and flakes", "label": "hBN margin",
        "description": "Every hBN = the largest conducting flake + this × spacing on every side."},
    "hbn.top_margin_spacings": {
        "value": 2.0, "unit": "× spacing", "group": "Gates and flakes", "label": "Top hBN margin",
        "description": "The top (cap) hBN = the largest conducting flake + this × spacing on every side."},
    "prepattern.tmd_overlap_spacings": {
        "value": 2.0, "unit": "× spacing", "group": "Pre-patterned contacts", "label": "Device over each electrode tip",
        "description": "Two-column pre-patterned electrodes (Mak): the device layer reaches this × spacing onto each electrode beyond the gap."},
    "tg.length_extra_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Stage 2 gates", "label": "Metal TG beyond the outer probes",
        "description": "Metal top-gate length = L + arm width + 2 × this × spacing."},
    "tg.overhang_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Stage 2 gates", "label": "Metal TG overhang",
        "description": "How far the metal top gate's lead reaches past the graphene edge."},
    "cg.reach_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Stage 2 gates", "label": "Contact-gate reaches",
        "description": "Contact gate: inner reach, outer reach and side extension, each this × spacing."},
    "sg.gap_spacings": {
        "value": 1.0, "unit": "× spacing", "group": "Stage 2 gates", "label": "Split-gate gap",
        "description": "Split gate: gap to the top-gate edge."},
    "fingers.width_um": {
        "value": 0.8, "unit": "µm", "group": "Finger array", "label": "Finger width", "status": "approved",
        "source": "stated by the user on 2026-11-17",
        "description": "Width of every pre-patterned finger (straight part)."},
    "fingers.pitch_um": {
        "value": 2.2, "unit": "µm", "group": "Finger array", "label": "Finger pitch (centre to centre)", "status": "approved",
        "source": "stated by the user on 2026-11-17",
        "description": "Fingers are 2.2 µm apart, centre to centre (the lab's drawings: 2.2–2.3 µm pitch, 1.4–1.5 µm edge spacing)."},
    "fingers.column_gap_um": {
        "value": 4.0, "unit": "µm", "group": "Finger array", "label": "Gap between the two columns", "status": "approved",
        "source": "stated by the user on 2026-11-17",
        "description": "Tip-to-tip distance between the two facing finger columns."},
    "fingers.min_overlap_um": {
        "value": 0.8, "unit": "µm", "group": "Finger array", "label": "Finger tip on the twisted region", "status": "program",
        "source": "UStack program default",
        "description": "A finger pair counts when the channel between its tips and this much of each finger from its tip lie on the twisted region (program choice: one finger width)."},
    "fingers.bbn_margin_um": {
        "value": None, "unit": "µm", "group": "Finger array", "label": "Finger ends inside the bottom hBN by", "status": "missing",
        "source": "asked from the user (2026-11-17: fingers as long as possible, this far from the bottom hBN boundary)",
        "description": "Each column's fingers run from the tip until they are this far inside the bottom hBN boundary (as long as "
                       "possible: less need on the covering hBN). Without a located bottom hBN the fixed finger length is used."},
    "fingers.straight_um": {
        "value": 9.5, "unit": "µm", "group": "Finger array", "label": "Straight part from the tip", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Fingers run straight this far from the tip, then fan out (lab drawings: 5.5–12 µm, median 9.5)."},
    "fingers.length_um": {
        "value": 21.8, "unit": "µm", "group": "Finger array", "label": "Finger length (tip to far end)", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Extent of a finger along its axis (lab drawings: 14.4–30.6 µm, median 21.8). Used only when the bottom hBN "
                       "is not located: otherwise each column is as long as the bottom hBN allows."},
    "fingers.far_pitch_um": {
        "value": 3.8, "unit": "µm", "group": "Finger array", "label": "Pitch at the far ends", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Centre-to-centre pitch of the fanned-out fingers at the preset finger length (lab drawings: 2.9–4.6 µm, median 3.8); it sets the fan angle, which longer fingers keep, so their ends spread wider."},
    "fingers.far_width_um": {
        "value": 1.1, "unit": "µm", "group": "Finger array", "label": "Width beyond the bend", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Finger width on the fanned part (lab drawings: 0.78–1.24 µm, median 1.1)."},
    "cg.tooth_length_um": {
        "value": 6.1, "unit": "µm", "group": "Finger array: contact gate", "label": "Contact-gate tooth length", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Each contact-gate tooth covers its finger from the tip outward this far (lab drawings: 5.3–11.1 µm, median 6.1). "
                       "Used when the TMD is not located; otherwise the teeth are fitted to it (cg.bus_tmd_gap_um)."},
    "cg.bus_tmd_gap_um": {
        "value": None, "unit": "µm", "group": "Finger array: contact gate", "label": "Contact-gate bus from the TMD by", "status": "missing",
        "source": "asked from the user (2026-11-17: the bus joining the teeth as close to the WSe2 as possible; how close is a preset)",
        "description": "The bus joining the contact-gate teeth (and its lead) stays this far from the TMD (device layers, e.g. both WSe2 "
                       "halves); each column's teeth are as short as that allows, so the two columns may differ."},
    "cg.hbn_margin_um": {
        "value": None, "unit": "µm", "group": "Finger array: contact gate", "label": "Contact gate inside its hBN by", "status": "missing",
        "source": "asked from the user (2026-11-17: the contact gate lands fully on the hBN under it)",
        "description": "The whole contact gate (teeth, bus and its lead) stays this far inside the hBN it is written on (the middle "
                       "hBN): off it, it would touch the fingers. The bus lead is shortened to stay inside."},
    "cg.bus_width_um": {
        "value": 1.6, "unit": "µm", "group": "Finger array: contact gate", "label": "Contact-gate bus width", "status": "program",
        "source": "UStack program default",
        "description": "Width of the bar joining the teeth behind them (program choice: the electrode width)."},
    "cg.bus_extension_um": {
        "value": 5.0, "unit": "µm", "group": "Finger array: contact gate", "label": "Bus beyond the last tooth", "status": "program",
        "source": "UStack program default",
        "description": "The bus bar runs this far beyond the outer tooth, where an electrode lands on it (program choice)."},
    "electrodes.width_um": {
        "value": 1.6, "unit": "µm", "group": "Finger array: electrodes", "label": "Electrode width", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Width of the EBL electrodes from the fingers to the bond pads (lab drawings: 1.57–1.7 µm, median 1.64)."},
    "electrodes.land_from_tip_um": {
        "value": 14.2, "unit": "µm", "group": "Finger array: electrodes", "label": "Electrode lands at (from the tip)", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Where an electrode starts on its finger, measured from the finger tip (lab drawings: 13.1–15.3 µm, median 14.2)."},
    "electrodes.bg_contact_um": {
        "value": 3.5, "unit": "µm", "group": "Finger array: electrodes", "label": "Back-gate contact bar", "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs (Stack 1/3/4), docs/wse2_design_report.md",
        "description": "Size of the bar that contacts the back gate at each end of the gap strip (lab drawings: ≈3.5 µm)."},
    "electrodes.gate_contact_clearance_um": {
        "value": 3.0, "unit": "µm", "group": "Finger array: electrodes", "label": "Gate contact clearance", "status": "program",
        "source": "UStack program default",
        "description": "A gate contact stays this far from the twisted region and the fingers (program choice)."},
    "electrodes.cg_overlap_um": {
        "value": 3.7, "unit": "µm", "group": "Finger array: electrodes", "label": "Electrode overlap on the contact-gate bus",
        "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs: layer-7 electrode on the layer-6 contact gate, Stack 1 2.95 / 4.52 / "
                  "7.48 µm, Stack 3 1.57 µm (median 3.7)",
        "description": "Each contact-gate electrode runs this far back along its bus lead from the lead's end (inside the lead), "
                       "for a good contact (user, 2026-11-20: overlap the bus more)."},
    "electrodes.finger_overlap_um": {
        "value": 10.9, "unit": "µm", "group": "Finger array: electrodes", "label": "Electrode overlap on a finger (at most)",
        "status": "pending",
        "source": "measured in the lab's tWSe2-1 GDS designs: layer-7 electrode along a layer-5 finger, median 12.17 µm "
                  "(Stack 1) / 9.69 µm (Stack 3), docs/wse2_design_report.md (median 10.9)",
        "description": "A Hall-bar electrode runs at most this far along its finger, back from the finger's far end (user, "
                       "2026-11-20: overlap a bit, not so far that the gold lands on the top hBN). It starts later where the "
                       "flakes above the fingers need it (electrodes.flake_clearance_um)."},
    "electrodes.flake_clearance_um": {
        "value": None, "unit": "µm", "group": "Finger array: electrodes", "label": "Electrode clearance from the flakes above the fingers",
        "status": "missing",
        "source": "asked from the user (2026-11-20: the electrodes on the fingers must not reach onto the top hBN)",
        "description": "Where a Hall-bar electrode lands on its finger, its metal stays this far outside every flake above the "
                       "fingers on the chip when the electrodes are written (top hBN, middle hBN, the device layers, a "
                       "graphite top gate — for a conducting flake at least ebl.contact_to_graphite_gate_min_um)."},
    "electrodes.pad_landing_um": {
        "value": None, "unit": "µm", "group": "Finger array: electrodes", "label": "Landing square on the bond-pad lead",
        "status": "missing",
        "source": "asked from the user (2026-11-20: each electrode ends in a larger square overlapping the bond-pad lead tip)",
        "description": "Side of the square each electrode ends in, centred on the end of its bond-pad lead (half of it over the "
                       "lead), for a good contact. Larger than the lead's width overlaps it on three sides; it must leave the "
                       "spacing to the neighbouring leads' squares."},
    "microscope.fov_100x_width_um": {
        "value": 143.28, "unit": "µm", "group": "Microscope", "label": "100x image width", "status": "approved",
        "source": "stated by the user on 2026-11-16",
        "description": "Field of view of a 100x image (the whole image, any resolution). Other objectives scale as 100 / magnification."},
    "microscope.fov_100x_height_um": {
        "value": 95.13, "unit": "µm", "group": "Microscope", "label": "100x image height", "status": "approved",
        "source": "stated by the user on 2026-11-16",
        "description": "Field of view of a 100x image (height)."},
}

#: Objectives with images (the field of view scales as 100 / magnification).
OBJECTIVES = ("5x", "20x", "50x", "100x")


@dataclass
class MissingItem:
    """One value the program needs and the lab has not supplied."""

    key: str                 # e.g. "rule:ebl.defect_clearance_um", "layer:hallbar_etch", "spec:L3.thickness_nm"
    label: str               # what to ask
    unit: str = ""
    kind: str = "number"     # number | text | layer (layer/datatype pair) | bool
    hint: str = ""           # e.g. a literature value to show beside the field
    suggested: Any = None


class MissingValues(Exception):
    """Raised when a step needs values nobody has supplied yet (the UI asks for them)."""

    def __init__(self, items: Iterable[MissingItem]):
        self.items = list(items)
        super().__init__(", ".join(i.label for i in self.items))


@dataclass
class Material:
    id: str
    category: str                    # 2d | metal | substrate
    display_name: str
    components: list[str]            # bottom → top (metals, substrates)
    conducting: str                  # yes | no | semiconductor
    sio2_thickness_nm: float | None
    status: str
    source: str
    notes: str = ""

    @property
    def is_metal(self) -> bool:
        return self.category == "metal"

    @property
    def is_2d(self) -> bool:
        return self.category == "2d"


@dataclass
class Rule:
    id: str
    value: Any
    unit: str = ""
    kind: str = "min"
    severity: str = "error"
    status: str = "missing"
    applies_to: list[str] = field(default_factory=lambda: ["all"])
    description: str = ""
    source: dict = field(default_factory=dict)
    optional: bool = False

    def applies(self, family: str, extra_tags: Iterable[str] = ()) -> bool:
        tags = set(FAMILY_TAGS.get(family, {"all"})) | set(extra_tags)
        return bool(tags & set(self.applies_to))

    @property
    def is_missing(self) -> bool:
        return self.value is None or self.status == "missing"

    def source_text(self) -> str:
        doc = self.source.get("doc", "")
        quote = self.source.get("quote", "")
        section = self.source.get("section", "")
        bits = [doc]
        if section:
            bits.append(section)
        text = " › ".join(b for b in bits if b)
        return f"{text}: “{quote}”" if quote else text

    def to_json(self) -> dict:
        data = {"value": self.value, "unit": self.unit, "kind": self.kind, "severity": self.severity,
                "status": self.status, "applies_to": self.applies_to, "description": self.description,
                "source": self.source}
        if self.optional:
            data["optional"] = True
        if not self.unit:
            data.pop("unit")
        return data


@dataclass
class GdsLayer:
    purpose: str
    layer: int | None
    datatype: int | None
    name: str
    status: str
    source: dict = field(default_factory=dict)
    reference_only: bool = False

    @property
    def is_missing(self) -> bool:
        return self.layer is None or self.datatype is None


def _today() -> str:
    return _dt.date.today().isoformat()


class LabConfig:
    """Loaded configuration; ``save`` writes the three files back."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else paths.config_dir()
        self.materials: dict[str, Material] = {}
        self.rules: dict[str, Rule] = {}
        self.defaults: dict[str, dict] = {}
        self.layers: dict[str, GdsLayer] = {}
        self.presets: dict[str, dict] = {}
        self._rules_doc: dict = {}
        self._layers_doc: dict = {}
        self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        self.materials = {}
        with open(self.root / "materials.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("id"):
                    continue
                sio2 = row.get("sio2_thickness_nm") or ""
                self.materials[row["id"]] = Material(
                    id=row["id"].strip(), category=row.get("category", "").strip(),
                    display_name=row.get("display_name", row["id"]).strip(),
                    components=[c.strip() for c in (row.get("components_bottom_to_top") or "").split(";") if c.strip()],
                    conducting=(row.get("conducting") or "").strip(),
                    sio2_thickness_nm=float(sio2) if sio2.strip() else None,
                    status=(row.get("status") or "pending").strip(), source=row.get("source", ""), notes=row.get("notes", ""))
        self._rules_doc = json.loads((self.root / "rules.json").read_text(encoding="utf-8"))
        self.rules = {}
        for rid, data in self._rules_doc.get("rules", {}).items():
            self.rules[rid] = Rule(id=rid, value=data.get("value"), unit=data.get("unit", ""), kind=data.get("kind", "min"),
                                   severity=data.get("severity", "error"), status=data.get("status", "missing"),
                                   applies_to=list(data.get("applies_to", ["all"])), description=data.get("description", ""),
                                   source=dict(data.get("source", {})), optional=bool(data.get("optional", False)))
        self.defaults = dict(self._rules_doc.get("defaults", {}))
        self._layers_doc = json.loads((self.root / "layers.json").read_text(encoding="utf-8"))
        self.layers = {}
        for purpose, data in self._layers_doc.get("layers", {}).items():
            self.layers[purpose] = GdsLayer(purpose, data.get("layer"), data.get("datatype"), data.get("name", purpose),
                                            data.get("status", "missing"), dict(data.get("source", {})),
                                            bool(data.get("reference_only", False)))
        self.presets = {k: dict(v, status=v.get("status", "program"), source=v.get("source", "UStack program default"))
                        for k, v in PRESET_DEFAULTS.items()}
        pfile = self.root / "presets.json"
        if pfile.is_file():
            for key, data in json.loads(pfile.read_text(encoding="utf-8")).get("presets", {}).items():
                self.presets[key] = dict(self.presets.get(key, {}), **data)

    def save(self) -> None:
        fields = ["id", "category", "display_name", "components_bottom_to_top", "conducting", "sio2_thickness_nm", "status", "source", "notes"]
        with open(self.root / "materials.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for m in self.materials.values():
                writer.writerow({"id": m.id, "category": m.category, "display_name": m.display_name,
                                 "components_bottom_to_top": ";".join(m.components), "conducting": m.conducting,
                                 "sio2_thickness_nm": "" if m.sio2_thickness_nm is None else f"{m.sio2_thickness_nm:g}",
                                 "status": m.status, "source": m.source, "notes": m.notes})
        doc = dict(self._rules_doc)
        doc["rules"] = {rid: rule.to_json() for rid, rule in self.rules.items()}
        doc["defaults"] = self.defaults
        (self.root / "rules.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ldoc = dict(self._layers_doc)
        out = {}
        for purpose, lay in self.layers.items():
            entry = {"layer": lay.layer, "datatype": lay.datatype, "name": lay.name, "status": lay.status, "source": lay.source}
            if lay.reference_only:
                entry["reference_only"] = True
            out[purpose] = entry
        ldoc["layers"] = out
        (self.root / "layers.json").write_text(json.dumps(ldoc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.save_presets()

    def save_presets(self) -> None:
        doc = {"description": "Design presets: the relations Configure uses to generate dimensions (program choices, status "
                              "'program', until the lab edits them) and the microscope field of view. Edit in UStack › Presets.",
               "presets": self.presets}
        (self.root / "presets.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ presets
    def preset(self, key: str) -> Any:
        entry = self.presets.get(key) or PRESET_DEFAULTS.get(key) or {}
        return entry.get("value")

    def set_preset(self, key: str, value: Any, who: str = "user") -> None:
        entry = self.presets.setdefault(key, dict(PRESET_DEFAULTS.get(key, {})))
        entry["value"] = value
        entry["status"] = "approved"
        entry["source"] = f"Edited in UStack by {who} on {_today()}"

    def field_of_view_um(self, objective: str) -> tuple[float, float] | None:
        """Width and height (µm) of a whole image taken with ``objective`` ('100x', '50x', …):
        the 100x field of view scaled by 100 / magnification."""
        try:
            mag = float(objective.lower().split("x")[0])
        except ValueError:
            return None
        w, h = self.preset("microscope.fov_100x_width_um"), self.preset("microscope.fov_100x_height_um")
        if not w or not h or mag <= 0:
            return None
        return float(w) * 100.0 / mag, float(h) * 100.0 / mag

    def um_per_px(self, objective: str, width_px: int) -> float | None:
        fov = self.field_of_view_um(objective)
        return None if fov is None or not width_px else fov[0] / width_px
    # ------------------------------------------------------------------ materials
    def material(self, mid: str) -> Material | None:
        if mid in self.materials:
            return self.materials[mid]
        low = mid.strip().lower()
        for m in self.materials.values():
            if m.id.lower() == low or m.display_name.lower() == low:
                return m
        return None

    def materials_in(self, category: str) -> list[Material]:
        return [m for m in self.materials.values() if m.category == category]

    # ------------------------------------------------------------------ rules
    def rule(self, rid: str) -> Rule:
        return self.rules[rid]

    def applicable(self, rid: str, family: str, extra_tags: Iterable[str] = ()) -> bool:
        rule = self.rules.get(rid)
        return rule is not None and rule.applies(family, extra_tags)

    def missing_item(self, rid: str) -> MissingItem:
        rule = self.rules[rid]
        kind = "bool" if rule.kind == "bool" else "number"
        return MissingItem(f"rule:{rid}", rule.description or rid, rule.unit, kind, rule.source_text())

    def require(self, *rids: str) -> list[Any]:
        """The values of ``rids``; raises :class:`MissingValues` listing every one not supplied."""
        missing = [self.missing_item(r) for r in rids if self.rules[r].is_missing]
        if missing:
            raise MissingValues(missing)
        return [self.rules[r].value for r in rids]

    def value(self, rid: str, default: Any = None) -> Any:
        rule = self.rules.get(rid)
        return default if rule is None or rule.is_missing else rule.value

    def set_rule_value(self, rid: str, value: Any, who: str = "user") -> None:
        rule = self.rules[rid]
        rule.value = value
        rule.status = "approved"
        note = f"Entered in UStack by {who} on {_today()}"
        if rule.source.get("doc") in (None, "", "not in the attached documents"):
            rule.source = {"doc": note, "quote": ""}
        else:
            rule.source = dict(rule.source, approved=note)

    def approve(self, rid: str, who: str = "user") -> None:
        rule = self.rules[rid]
        if rule.is_missing:
            raise ValueError(f"{rid} has no value to approve")
        rule.status = "approved"
        rule.source = dict(rule.source, approved=f"Approved in UStack by {who} on {_today()}")

    def approve_layer(self, purpose: str, who: str = "user") -> None:
        lay = self.layers[purpose]
        if lay.is_missing:
            raise ValueError(f"{purpose} has no layer number to approve")
        lay.status = "approved"
        lay.source = dict(lay.source, approved=f"Approved in UStack by {who} on {_today()}")

    def approve_material(self, mid: str, who: str = "user") -> None:
        m = self.materials[mid]
        m.status = "approved"
        m.source = f"{m.source} | Approved in UStack by {who} on {_today()}"

    # ------------------------------------------------------------------ GDS layers
    def gds_layer(self, purpose: str) -> tuple[int, int]:
        lay = self.layers.get(purpose)
        if lay is None or lay.is_missing:
            name = lay.name if lay else purpose
            raise MissingValues([MissingItem(f"layer:{purpose}", f"GDS layer/datatype for {name}", "", "layer")])
        return int(lay.layer), int(lay.datatype)

    def set_gds_layer(self, purpose: str, layer: int, datatype: int, who: str = "user") -> None:
        lay = self.layers.get(purpose)
        if lay is None:
            lay = GdsLayer(purpose, None, None, purpose, "missing")
            self.layers[purpose] = lay
        lay.layer, lay.datatype, lay.status = int(layer), int(datatype), "approved"
        lay.source = {"doc": f"Entered in UStack by {who} on {_today()}"}

    # ------------------------------------------------------------------ defaults
    def default(self, key: str) -> dict | None:
        return self.defaults.get(key)

    def pending_summary(self) -> dict[str, list[str]]:
        """Everything not yet approved, by kind (for the Rules editor badge)."""
        out = {"rules": [], "layers": [], "materials": [], "defaults": []}
        for rid, r in self.rules.items():
            if r.status != "approved":
                out["rules"].append(rid)
        for pid, lay in self.layers.items():
            if lay.status != "approved":
                out["layers"].append(pid)
        for mid, m in self.materials.items():
            if m.status != "approved":
                out["materials"].append(mid)
        for did, d in self.defaults.items():
            if d.get("status") != "approved":
                out["defaults"].append(did)
        return out


def apply_answers(config: LabConfig, answers: dict[str, Any]) -> list[str]:
    """Store answers keyed like :class:`MissingItem.key` (``rule:…`` and ``layer:…``) in the
    configuration (approved, attributed to the user) and save it. ``spec:`` keys are left for the
    caller. Returns the keys that were stored."""
    stored = []
    for key, value in answers.items():
        if key.startswith("rule:"):
            config.set_rule_value(key[5:], value)
            stored.append(key)
        elif key.startswith("layer:"):
            layer, datatype = value
            config.set_gds_layer(key[6:], layer, datatype)
            stored.append(key)
        elif key.startswith("preset:"):
            config.set_preset(key[7:], value)
            stored.append(key)
        elif key.startswith("default:"):
            name, _, sub = key[8:].partition("#")  # "default:microscope.um_per_px#100x"
            entry = config.defaults.setdefault(name, {"status": "missing"})
            if sub:
                entry.setdefault("value", {})[sub] = value
            else:
                entry["value"] = value
            if not isinstance(entry.get("value"), dict) or all(v is not None for v in entry["value"].values()):
                entry["status"] = "approved"
            entry["source"] = {"doc": f"Entered in UStack on {_today()}"}
            stored.append(key)
    if stored:
        config.save()
    return stored
