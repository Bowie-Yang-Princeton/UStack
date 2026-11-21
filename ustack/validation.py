"""Stage 1 validation: the specification against the lab's materials and design rules.

The validator only reports. It never changes a material, a dimension or a requested gate to make
a design pass: a failing check is an ``error`` the user resolves, and a value nobody supplied is
a ``missing`` finding (with a :class:`~ustack.config.MissingItem`) the UI asks for.

Findings from rules whose value is still ``pending`` / ``literature`` say so in their text.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

from . import geometry as geo
from .config import LabConfig, MissingItem
from .model import (CONTACT_FUNCTIONS, DIELECTRIC_FUNCTIONS, FUNCTIONS, FUNCTIONS_BY_CATEGORY, GATE_FUNCTIONS,
                    HALL_BAR_FIELDS, DeviceSpec, Layer, Q)

GRAPHENE_LAYER_NM = 0.335  # interlayer spacing of graphite (physical constant)
#: Interlayer spacing (nm) of the bulk crystals, used to turn a layer count into a thickness for the
#: 3D view and step heights. Approximate crystallographic values (c/2 of the 2H unit cell), not lab rules.
LAYER_SPACING_NM = {"graphene": GRAPHENE_LAYER_NM, "WSe2": 0.65, "MoTe2": 0.70, "WS2": 0.62, "MoS2": 0.62, "MoSe2": 0.65}
SEVERITIES = ("error", "missing", "warning", "info")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    layer: str | None = None
    field: str | None = None
    rule: str | None = None
    rule_status: str | None = None
    where: list[tuple[float, float]] = dc_field(default_factory=list)
    ask: MissingItem | None = None

    def to_json(self) -> dict:
        data = {"severity": self.severity, "code": self.code, "message": self.message}
        for key in ("layer", "field", "rule", "rule_status"):
            if getattr(self, key):
                data[key] = getattr(self, key)
        if self.where:
            data["where_um"] = [list(map(lambda v: round(v, 3), p)) for p in self.where]
        return data


@dataclass
class Report:
    issues: list[Issue] = dc_field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def by(self, severity: str) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def errors(self) -> list[Issue]:
        return self.by("error")

    @property
    def missing(self) -> list[Issue]:
        return self.by("missing")

    @property
    def warnings(self) -> list[Issue]:
        return self.by("warning")

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing

    def missing_items(self) -> list[MissingItem]:
        seen, out = set(), []
        for issue in self.missing:
            if issue.ask and issue.ask.key not in seen:
                seen.add(issue.ask.key)
                out.append(issue.ask)
        return out

    def summary(self) -> str:
        n = {s: len(self.by(s)) for s in SEVERITIES}
        state = "Valid" if self.ok else ("Needs values" if not n["error"] else "Not valid")
        return f"{state} · {n['error']} errors · {n['missing']} missing · {n['warning']} warnings"

    def to_json(self) -> dict:
        return {"ok": self.ok, "summary": self.summary(), "issues": [i.to_json() for i in self.issues]}


# ---------------------------------------------------------------------- helpers
def category(config: LabConfig, layer: Layer) -> str | None:
    """design category of a layer's material: substrate | metal | 2d_conductor | 2d_semiconductor | 2d_insulator"""
    m = config.material(layer.material) if layer.material else None
    if m is None:
        return None
    if m.category in ("substrate", "metal"):
        return m.category
    if m.conducting == "yes":
        return "2d_conductor"
    if m.conducting == "no":
        return "2d_insulator"
    return "2d_semiconductor"


def materials_for_function(config: LabConfig, function: str) -> list[str]:
    """Material ids in the lab list that can take ``function``."""
    return [m.id for m in config.materials.values()
            if function in FUNCTIONS_BY_CATEGORY.get(category(config, Layer("", "", m.id, "")) or "", ())]


def is_conductor(config: LabConfig, layer: Layer) -> bool:
    return category(config, layer) in ("metal", "2d_conductor")


def is_insulator(config: LabConfig, layer: Layer) -> bool:
    return category(config, layer) == "2d_insulator"


def thickness_nm(layer: Layer) -> float | None:
    if layer.metal_parts:
        return layer.total_metal_nm()
    if layer.thickness_nm.value is not None:
        return layer.thickness_nm.value
    if layer.material in LAYER_SPACING_NM and layer.layer_count.value is not None:
        return layer.layer_count.value * LAYER_SPACING_NM[layer.material]
    return None


def outline_or_nominal(layer: Layer) -> list[tuple[float, float]] | None:
    if layer.outline and len(layer.outline) >= 3:
        return [tuple(p) for p in layer.outline]
    w, l = layer.width_um.value, layer.length_um.value
    if w and l:
        return geo.nominal_rect(w, l)
    return None


def _rule_suffix(config: LabConfig, rid: str) -> tuple[str, str | None]:
    rule = config.rules.get(rid)
    if rule is None:
        return "", None
    if rule.status in ("pending", "literature"):
        return f" [rule {rid} is {rule.status} approval]", rule.status
    return "", rule.status


def ask_spec(key: str, label: str, unit: str = "", hint: str = "", suggested: Any = None, kind: str = "number") -> MissingItem:
    return MissingItem(f"spec:{key}", label, unit, kind, hint, suggested)


def same_value(have: Any, want: Any) -> bool:
    """``have`` (a layer attribute, possibly a :class:`Q`) equals ``want`` (a plain value from a
    requirement); numbers compare with a relative tolerance of 1e-6."""
    if isinstance(have, Q):
        have = have.value
    if isinstance(have, (int, float)) and isinstance(want, (int, float)) and not isinstance(want, bool):
        return abs(float(have) - float(want)) <= 1e-6 * max(1.0, abs(float(want)))
    return have == want


def assign_roles(spec: DeviceSpec, config: LabConfig) -> list[str]:
    """Give layers without a role the role their position implies (the family rules look layers up
    by role). Existing roles are kept. Returns the uids that got a role."""
    done: list[str] = []
    taken = {l.role for l in spec.layers if l.role}
    devices = spec.device_layers()

    def give(layer: Layer | None, role: str) -> None:
        if layer is not None and not layer.role and role not in taken:
            layer.role = role
            taken.add(role)
            done.append(layer.uid)

    for d in devices:
        if not d.role:
            d.role = "graphene" if d.material == "graphene" else "tmd"
            done.append(d.uid)
    if not devices:
        return done
    lo, hi = min(spec.index(d.uid) for d in devices), max(spec.index(d.uid) for d in devices)
    below = [l for l in spec.layers[:lo] if is_insulator(config, l)]
    above = [l for l in spec.layers[hi + 1:] if is_insulator(config, l)]
    give(below[-1] if below else None, "bottom_hbn")
    if spec.family == "xu_tmd":
        give(next((l for l in above if l.function == "contact_gate_dielectric"), None), "middle_hbn")
    give(next((l for l in above if l.function == "top_gate_dielectric"), above[0] if above else None), "tg_hbn")
    if len(above) > 1 and spec.family != "xia_graphene":
        give(above[-1], "top_hbn")
    if spec.contact_scheme == "prepatterned_bottom":
        for c in spec.by_function(*CONTACT_FUNCTIONS):
            give(c, "prepattern_contacts")
    return done


def bottom_gates(spec: DeviceSpec, config: LabConfig) -> list[Layer]:
    """Conducting bottom-gate layers (graphite or a pre-patterned metal gate)."""
    return [l for l in spec.by_function("bottom_gate") if is_conductor(config, l)]


class _Ctx:
    def __init__(self, spec: DeviceSpec, config: LabConfig, report: Report):
        self.spec, self.config, self.report = spec, config, report
        self.family = spec.family
        tags = set()
        for d in spec.device_layers():
            if d.material in ("graphene",):
                tags.add("graphene")
            elif d.material:
                tags.add("tmd")
        self.tags = tags

    def applies(self, rid: str) -> bool:
        return self.config.applicable(rid, self.family, self.tags)

    def issue(self, severity: str, code: str, message: str, rule: str | None = None, **kw) -> None:
        status = None
        if rule:
            suffix, status = _rule_suffix(self.config, rule)
            message += suffix
            r = self.config.rules.get(rule)
            if r and severity == "error" and r.severity in ("warning", "info"):
                severity = r.severity
        self.report.add(Issue(severity, code, message, rule=rule, rule_status=status, **kw))

    def need_rule(self, rid: str) -> Any:
        """The rule's value, or None after adding a 'missing' finding."""
        rule = self.config.rules.get(rid)
        if rule is None:
            return None
        if rule.is_missing:
            self.report.add(Issue("missing", "rule_value", f"Lab value needed: {rule.description}", rule=rid,
                                  rule_status="missing", ask=self.config.missing_item(rid)))
            return None
        return rule.value


# ---------------------------------------------------------------------- checks
def _check_structure(c: _Ctx) -> None:
    spec, config = c.spec, c.config
    if not spec.layers:
        c.issue("error", "empty", "The stack has no layers.")
        return
    subs = spec.by_function("substrate")
    if len(subs) != 1:
        c.issue("error", "substrate_count", f"The stack needs exactly one substrate layer (has {len(subs)}).")
    elif spec.layers[0].function != "substrate":
        c.issue("error", "substrate_position", "The substrate must be the bottom layer.", layer=subs[0].uid)
    if not spec.device_layers():
        c.issue("error", "no_device_layer", "No layer has the function 'Device layer'.")
    for layer in spec.layers:
        if layer.function not in FUNCTIONS:
            c.issue("error", "function_unknown", f"{layer.label}: unknown function '{layer.function}'.", layer=layer.uid, field="function")
            continue
        if not layer.material:
            options = materials_for_function(config, layer.function)
            hint = layer.literature or layer.notes
            c.report.add(Issue("missing", "material", f"{layer.label}: choose a material.", layer=layer.uid, field="material",
                               ask=ask_spec(f"{layer.uid}.material", f"Material for '{layer.label}'", "", hint, options, kind="choice")))
            continue
        mat = config.material(layer.material)
        if mat is None:
            c.issue("error", "material_unsupported",
                    f"{layer.label}: '{layer.material}' is not in the lab's materials list (config/materials.csv). "
                    "UStack does not substitute materials: add it to the list or choose another.", layer=layer.uid, field="material")
            continue
        if mat.status != "approved":
            c.issue("warning", "material_pending", f"{layer.label}: material '{mat.id}' is {mat.status} approval ({mat.source}).",
                    layer=layer.uid, field="material")
        cat = category(config, layer)
        if layer.function not in FUNCTIONS_BY_CATEGORY.get(cat or "", ()):
            c.issue("error", "function_material", f"{layer.label}: a {mat.display_name} layer cannot be '{FUNCTIONS[layer.function]}'.",
                    layer=layer.uid, field="function")


def _check_values(c: _Ctx) -> None:
    spec, config = c.spec, c.config
    prev_device: Layer | None = None
    if any(category(config, l) in ("2d_conductor", "2d_semiconductor", "2d_insulator") for l in spec.layers):
        # the flake list adds the stacking misalignment on every side of each flake
        c.need_rule("stack.alignment_shift_um")
    for layer in spec.layers:
        cat = category(config, layer)
        if cat is None:
            continue
        hint_t = layer.thickness_nm.suggested_source or layer.literature
        if cat == "metal":
            mat = config.material(layer.material)
            if not layer.metal_parts and mat is not None:
                for comp in mat.components:
                    c.report.add(Issue("missing", "metal_thickness", f"{layer.label}: {comp} thickness needed.", layer=layer.uid,
                                       field=f"metal:{comp}", ask=ask_spec(f"{layer.uid}.metal.{comp}", f"{comp} thickness in '{layer.label}'", "nm", layer.literature)))
            if mat is not None and layer.metal_parts:
                names = [p.metal for p in layer.metal_parts]
                if names != mat.components:
                    c.issue("error", "metal_parts", f"{layer.label}: the metal stack {'/'.join(names)} does not match the material "
                            f"{mat.display_name} ({'/'.join(mat.components)}).", layer=layer.uid, field="metal_parts")
            for part in layer.metal_parts:
                if part.thickness_nm.value is None:
                    c.report.add(Issue("missing", "metal_thickness", f"{layer.label}: {part.metal} thickness needed.", layer=layer.uid,
                                       field=f"metal:{part.metal}",
                                       ask=ask_spec(f"{layer.uid}.metal.{part.metal}", f"{part.metal} thickness in '{layer.label}'", "nm",
                                                    part.thickness_nm.suggested_source or layer.literature, part.thickness_nm.suggested)))
        elif cat != "substrate":
            needs_count = layer.function == "device_layer"
            if needs_count and layer.layer_count.value is None:
                c.report.add(Issue("missing", "layer_count", f"{layer.label}: number of layers needed.", layer=layer.uid, field="layer_count",
                                   ask=ask_spec(f"{layer.uid}.layer_count", f"Layer count of '{layer.label}'", "layers",
                                                layer.layer_count.suggested_source, layer.layer_count.suggested)))
            if not needs_count and layer.thickness_nm.value is None:
                c.report.add(Issue("missing", "thickness", f"{layer.label}: thickness needed.", layer=layer.uid, field="thickness_nm",
                                   ask=ask_spec(f"{layer.uid}.thickness_nm", f"Thickness of '{layer.label}'", "nm", hint_t, layer.thickness_nm.suggested)))
            if not layer.outline:
                for fld, lab in (("width_um", "width"), ("length_um", "length")):
                    q = getattr(layer, fld)
                    if q.value is None:
                        c.report.add(Issue("missing", fld, f"{layer.label}: lateral {lab} needed (for the flake search and the layout).",
                                           layer=layer.uid, field=fld,
                                           ask=ask_spec(f"{layer.uid}.{fld}", f"Lateral {lab} of '{layer.label}'", "µm", q.suggested_source, q.suggested)))
            if layer.function == "device_layer":
                if prev_device is not None and prev_device.material == layer.material and layer.twist_deg.value is None:
                    c.report.add(Issue("missing", "twist", f"{layer.label}: twist angle relative to the layer below needed.", layer=layer.uid,
                                       field="twist_deg", ask=ask_spec(f"{layer.uid}.twist_deg", f"Twist angle of '{layer.label}'", "°",
                                                                       layer.twist_deg.suggested_source, layer.twist_deg.suggested)))
                prev_device = layer
    if spec.pattern == "hall_bar":
        for key, (label, unit) in HALL_BAR_FIELDS.items():
            q = spec.hall_bar.params.get(key)
            if q is None or q.value is None:
                c.report.add(Issue("missing", "hall_bar", f"Hall bar: {label} needed.", field=f"hall_bar.{key}",
                                   ask=ask_spec(f"hall_bar.{key}", f"Hall bar · {label}", "µm" if unit == "um" else unit,
                                                q.suggested_source if q else "", q.suggested if q else None)))
        n = spec.hall_bar.get("n_probe_pairs")
        if n is not None and (n < 1 or n != int(n)):
            c.issue("error", "hall_bar_pairs", "Hall bar: the number of probe pairs must be a whole number ≥ 1.", field="hall_bar.n_probe_pairs")
        for key in ("channel_width_um", "arm_width_um", "arm_length_um"):
            v = spec.hall_bar.get(key)
            if v is not None and v <= 0:
                c.issue("error", "hall_bar_value", f"Hall bar: {HALL_BAR_FIELDS[key][0]} must be positive.", field=f"hall_bar.{key}")


def _between(spec: DeviceSpec, a: Layer, b: Layer) -> list[Layer]:
    i, j = sorted((spec.index(a.uid), spec.index(b.uid)))
    return spec.layers[i + 1:j]


def _check_electrostatics(c: _Ctx) -> None:
    spec, config = c.spec, c.config
    devices = spec.device_layers()
    if not devices:
        return
    for gate in spec.by_function(*GATE_FUNCTIONS):
        if not is_conductor(config, gate):
            continue
        nearest = min(devices, key=lambda d: abs(spec.index(d.uid) - spec.index(gate.uid)))
        between = _between(spec, gate, nearest)
        if not any(is_insulator(config, l) for l in between):
            c.issue("error", "gate_short", f"{gate.label}: no hBN between this gate and {nearest.label} — it would short to the device.",
                    layer=gate.uid)
    for d in devices:
        idx = spec.index(d.uid)
        below = [l for l in spec.layers[:idx] if l.function != "substrate"]
        above = spec.layers[idx + 1:]
        enc_below = any(is_insulator(config, l) for l in below)
        enc_above = any(is_insulator(config, l) for l in above)
        if not (enc_below and enc_above):
            sev = "warning" if spec.family == "custom" else "error"
            c.issue(sev, "encapsulation", f"{d.label}: not hBN-encapsulated (hBN {'below' if not enc_below else 'above'} missing).", layer=d.uid)
    # pre-patterned bottom contacts only work if they lie directly against the device layer
    if spec.contact_scheme == "prepatterned_bottom":
        for contact in spec.by_function(*CONTACT_FUNCTIONS):
            if not is_conductor(config, contact):
                continue
            nearest = min(devices, key=lambda d: abs(spec.index(d.uid) - spec.index(contact.uid)))
            between = _between(spec, contact, nearest)
            if not between:
                continue
            names = ", ".join(l.label for l in between)
            if spec.hall_bar.sd_style != "bent":
                c.issue("error", "contact_not_touching",
                        f"{contact.label} is separated from {nearest.label} by {names}: pre-patterned contacts must lie directly "
                        "against the device layer, or be two electrode columns beside the gate that the device reaches "
                        "beyond the hBN (Hall bar › source/drain 'Bent to the sides').", layer=contact.uid)
                continue
            # two electrode columns with the gate stack in the gap: the device must be wider than every layer in between
            dw = nearest.width_um.value
            for l in between:
                lw = l.width_um.value
                if dw is not None and lw is not None and dw <= lw:
                    c.issue("error", "contact_not_touching",
                            f"{nearest.label} ({dw:g} µm across) is not wider than {l.label} ({lw:g} µm): it cannot reach past it "
                            f"onto the {contact.label}.", layer=nearest.uid)
            # the gates in the gap must not touch the electrodes
            W = spec.hall_bar.get("channel_width_um")
            es = config.value("ebl.electrode_spacing_target_um") or config.value("ebl.contact_spacing_min_um")
            for l in between:
                lw = l.width_um.value
                if is_conductor(config, l) and None not in (W, es, lw) and lw + 2 * es > W + 1e-6:
                    c.issue("error", "gate_in_gap",
                            f"{l.label} ({lw:g} µm across) does not fit in the {W:g} µm gap between the electrode columns with "
                            f"{es:g} µm to each side: it would touch {contact.label}.", layer=l.uid)
    # contacts / gates stacked directly on each other (lateral separation must be checked in Stage 2)
    for i in range(1, len(spec.layers)):
        a, b = spec.layers[i - 1], spec.layers[i]
        if is_conductor(config, a) and is_conductor(config, b) and a.function != b.function and \
                {a.function, b.function} & set(GATE_FUNCTIONS) and {a.function, b.function} & set(CONTACT_FUNCTIONS):
            c.issue("info", "lateral_separation", f"{a.label} and {b.label} are adjacent conductors: they must not overlap laterally "
                    "(checked on the traced outlines in Stage 2).", layer=b.uid)
    # substrate breakdown (doped Si is a gate whether or not it is used)
    sub = spec.substrate()
    mat = config.material(sub.material) if sub and sub.material else None
    if mat is not None and mat.conducting == "yes" and mat.sio2_thickness_nm:
        vq = spec.gate_limits_V.get(sub.uid)
        field_max = c.need_rule("stack.substrate_breakdown_field_MV_per_cm")
        if vq is None or vq.value is None:
            c.report.add(Issue("missing", "gate_limit", "Planned maximum |voltage| on the doped-Si back gate needed (substrate breakdown check).",
                               layer=sub.uid, field="gate_limit", ask=ask_spec(f"gate.{sub.uid}", "Planned max |V| on the doped-Si back gate", "V")))
        elif field_max is not None:
            e = vq.value / mat.sio2_thickness_nm * 10.0  # V/nm → MV/cm
            if e >= field_max:
                c.issue("error", "substrate_breakdown", f"Si back gate at {vq.value:g} V gives {e:.2f} MV/cm across {mat.sio2_thickness_nm:g} nm SiO2, "
                        f"at or above the breakdown field {field_max:g} MV/cm.", rule="stack.substrate_breakdown_field_MV_per_cm", layer=sub.uid)
            else:
                c.issue("info", "substrate_breakdown_ok", f"Si back gate: {e:.2f} MV/cm at {vq.value:g} V (limit {field_max:g} MV/cm).",
                        rule="stack.substrate_breakdown_field_MV_per_cm", layer=sub.uid)
    # hBN gate dielectrics, only where the user set a gate limit
    for uid, vq in spec.gate_limits_V.items():
        gate = spec.layer(uid)
        if gate is None or gate.function == "substrate" or vq.value is None:
            continue
        nearest = min(devices, key=lambda d: abs(spec.index(d.uid) - spec.index(gate.uid)))
        t = [thickness_nm(l) for l in _between(spec, gate, nearest) if is_insulator(config, l)]
        if not t or any(v is None for v in t):
            continue
        field_max = c.config.value("stack.hbn_breakdown_field_MV_per_cm")
        e = vq.value / sum(t) * 10.0
        if field_max is None:
            c.issue("info", "hbn_field", f"{gate.label}: {e:.2f} MV/cm across {sum(t):g} nm hBN at {vq.value:g} V (no hBN breakdown value in rules.json).", layer=uid)
        elif e >= field_max:
            c.issue("error", "hbn_breakdown", f"{gate.label}: {e:.2f} MV/cm across {sum(t):g} nm hBN at {vq.value:g} V exceeds {field_max:g} MV/cm.",
                    rule="stack.hbn_breakdown_field_MV_per_cm", layer=uid)


def _hall_bar_footprint(spec: DeviceSpec) -> tuple[float, float] | None:
    hb = spec.hall_bar
    W, L, al, sd = (hb.get(k) for k in ("channel_width_um", "channel_length_um", "arm_length_um", "sd_extension_um"))
    if None in (W, L, al, sd):
        return None
    if spec.contact_scheme == "prepatterned_bottom" and hb.sd_style == "bent":
        # two electrode columns: the device spans the gap and reaches onto each electrode tip
        q = spec.pattern_params.get("pt_overlap_um")
        return L + 2 * sd, W + 2 * (q.value if q is not None and q.value is not None else 0.0)
    return L + 2 * sd, W + 2 * al   # (along the channel, across it)


def _fits(dims: tuple[float, float], fw: float, fl: float) -> bool:
    a, b = sorted(dims)
    x, y = sorted((fw, fl))
    return a >= x - 1e-6 and b >= y - 1e-6


def _check_family(c: _Ctx) -> None:
    spec, config = c.spec, c.config
    devices = spec.device_layers()
    # Hall bar must fit in every device layer (nominal size or traced outline) — never shrunk to fit
    fp = _hall_bar_footprint(spec)
    hbp = spec.hall_bar
    if hbp.sd_style == "bent" and None not in (hbp.get("sd_extension_um"), hbp.get("arm_width_um"), hbp.get("channel_width_um")):
        aw = hbp.get("arm_width_um")
        leg = aw if spec.contact_scheme == "prepatterned_bottom" else hbp.get("channel_width_um")
        es = config.value("ebl.electrode_spacing_target_um") or config.value("ebl.contact_spacing_min_um") or 0.0
        gap = hbp.get("sd_extension_um") - leg - aw / 2
        if gap < es - 1e-6:
            c.issue("error", "sd_leg_spacing", f"Bent source/drain: the {leg:g} µm leg is {gap:.2f} µm from the outer probe arm "
                    f"(needs {es:g} µm). S/D extension must be ≥ {leg + aw / 2 + es:g} µm.", field="hall_bar")
    if fp is not None:
        for d in devices:
            w, l = d.width_um.value, d.length_um.value
            if d.outline:
                x0, y0, x1, y1 = geo.bbox(geo.region(d.outline))
                w, l = y1 - y0, x1 - x0
            if w and l and not _fits((w, l), fp[1], fp[0]):
                c.issue("error", "hall_bar_fit", f"The Hall bar needs {fp[0]:.2f} × {fp[1]:.2f} µm but {d.label} is {l:g} × {w:g} µm. "
                        "UStack does not shrink a specified dimension: choose a larger flake or change the Hall bar yourself.", layer=d.uid)
    hbns = [l for l in spec.layers if l.material == "hBN"]
    if spec.family == "xu_tmd":
        bottom, middle, top = spec.by_role("bottom_hbn"), spec.by_role("middle_hbn"), spec.by_role("top_hbn")
        tmds = [d for d in devices if d.material != "graphene"]
        if c.applies("xu.bottom_hbn_larger") and bottom:
            for other in [middle, *tmds]:
                if other is None:
                    continue
                bw, bl, ow, ol = bottom.width_um.value, bottom.length_um.value, other.width_um.value, other.length_um.value
                if None not in (bw, bl, ow, ol) and not (_fits((bw, bl), ow, ol) and bw * bl > ow * ol):
                    c.issue("error", "xu_bottom_larger", f"{bottom.label} ({bl:g} × {bw:g} µm) must be larger than {other.label} ({ol:g} × {ow:g} µm).",
                            rule="xu.bottom_hbn_larger", layer=bottom.uid)
        if middle is not None and c.applies("xu.middle_hbn_max_thickness_nm"):
            tmax = c.need_rule("xu.middle_hbn_max_thickness_nm")
            t = thickness_nm(middle)
            if tmax is not None and t is not None and t > tmax:
                c.issue("error", "xu_middle_thin", f"{middle.label} is {t:g} nm; the lab limit for the thin middle hBN is {tmax:g} nm.",
                        rule="xu.middle_hbn_max_thickness_nm", layer=middle.uid)
        if middle is not None and c.applies("xu.middle_hbn_edge_margin_um"):
            margin = c.need_rule("xu.middle_hbn_edge_margin_um")
            for tmd in tmds:
                mo, to = outline_or_nominal(middle), outline_or_nominal(tmd)
                if margin is None or mo is None or to is None:
                    continue
                pts = geo.enclosure_violations(geo.region(mo), geo.region(to), margin)
                if pts:
                    basis = "traced outlines" if (middle.outline and tmd.outline) else "nominal sizes, flakes assumed centred"
                    c.issue("error", "xu_middle_margin", f"{middle.label} must extend ≥ {margin:g} µm beyond every edge of {tmd.label} ({basis}).",
                            rule="xu.middle_hbn_edge_margin_um", layer=middle.uid, where=pts)
    if spec.family == "mak_tmd" and c.applies("mak.tg_smaller_than_bg"):
        tg = next((l for l in spec.by_function("top_gate") if l.material == "graphite"), None)
        bg = next((l for l in spec.by_function("bottom_gate") if l.material == "graphite"), None)
        if tg and bg:
            to, bo = outline_or_nominal(tg), outline_or_nominal(bg)
            if to and bo and geo.area(geo.region(to)) >= geo.area(geo.region(bo)):
                c.issue("warning", "mak_tg_smaller", f"{tg.label} should be smaller than {bg.label} (the TG defines the channel).",
                        rule="mak.tg_smaller_than_bg", layer=tg.uid)
    if spec.contact_scheme == "edge_1d" and any(d.material == "graphene" for d in devices):
        gr = next(d for d in devices if d.material == "graphene")
        # the metal must be thicker than the step it climbs (top hBN + graphene + bottom-hBN over-etch)
        if c.applies("gr.contact_metal_thicker_than_stack"):
            above = [l for l in spec.layers[spec.index(gr.uid) + 1:] if is_insulator(config, l)]
            t_top = thickness_nm(above[0]) if above else None
            t_gr = thickness_nm(gr)
            over = config.value("etch.contact_bottom_bn_nm")
            over_max = max(over) if isinstance(over, list) else (over or 0.0)
            for contact in spec.by_function(*CONTACT_FUNCTIONS):
                t_metal = thickness_nm(contact)
                if None in (t_top, t_gr, t_metal):
                    continue
                step = t_top + t_gr + over_max
                if t_metal <= step:
                    c.issue("error", "contact_too_thin", f"{contact.label} is {t_metal:g} nm, not thicker than the etched step "
                            f"(top hBN {t_top:g} + graphene {t_gr:.2f} + bottom-hBN over-etch {over_max:g} = {step:.1f} nm).",
                            rule="gr.contact_metal_thicker_than_stack", layer=contact.uid)
        # probe arms must reach past the BG (graphite or pre-patterned metal) for 1D contacts
        bg = next(iter(bottom_gates(spec, config)), None)
        hb = spec.hall_bar
        W, al = hb.get("channel_width_um"), hb.get("arm_length_um")
        beyond = config.value("ebl.device_etch_beyond_bg_um") if c.applies("ebl.device_etch_beyond_bg_um") else None
        bg_dims = None
        if bg is not None and bg.outline:
            x0, y0, x1, y1 = geo.bbox(geo.region(bg.outline))
            bg_dims = (x1 - x0, y1 - y0)
        elif bg is not None and bg.width_um.value and bg.length_um.value:
            bg_dims = (bg.width_um.value, bg.length_um.value)
        if bg_dims and None not in (W, al, beyond):
            reach = W + 2 * al - 2 * beyond
            if min(bg_dims) > reach:
                c.issue("warning", "arms_beyond_bg", f"With W = {W:g} µm and arms of {al:g} µm the probe arms reach {W / 2 + al:g} µm from the "
                        f"channel axis; to end {beyond:g} µm beyond the BG the BG must be ≤ {reach:g} µm across (it is "
                        f"{min(bg_dims):g} µm at its narrowest).", rule="ebl.device_etch_beyond_bg_um", layer=bg.uid)
        # overlaps need hBN between (only checkable on traced outlines)
        if gr.outline:
            grr = geo.region(gr.outline)
            bg_l = bg
            bbn = spec.by_role("bottom_hbn")
            tg = next(iter(spec.by_function("top_gate")), None)
            tbn = spec.by_role("tg_hbn")
            if bg_l and bg_l.outline and bbn and bbn.outline and c.applies("gr.graphene_bg_overlap_needs_bottom_bn"):
                bad = (grr & geo.region(bg_l.outline)) - geo.region(bbn.outline)
                if not bad.is_empty():
                    c.issue("error", "gr_bg_no_bn", "Graphene overlaps the BG where there is no bottom hBN.", rule="gr.graphene_bg_overlap_needs_bottom_bn",
                            where=[geo.centroid(p) for p in geo.to_polys(bad)])
            if tg and tg.outline and tbn and tbn.outline and c.applies("gr.graphene_tg_overlap_needs_top_bn"):
                bad = (grr & geo.region(tg.outline)) - geo.region(tbn.outline)
                if not bad.is_empty():
                    c.issue("error", "gr_tg_no_bn", "Graphene overlaps the TG where there is no top hBN.", rule="gr.graphene_tg_overlap_needs_top_bn",
                            where=[geo.centroid(p) for p in geo.to_polys(bad)])
        if spec.family == "xia_graphene" and c.applies("gr.bottom_bn_larger_than_bg_three_sides"):
            bbn = spec.by_role("bottom_hbn")
            if bg and bbn and bg.outline and bbn.outline:
                bx0, by0, bx1, by1 = geo.bbox(geo.region(bbn.outline))
                gx0, gy0, gx1, gy1 = geo.bbox(geo.region(bg.outline))
                inside = [gx0 > bx0, gy0 > by0, gx1 < bx1, gy1 < by1]
                if sum(inside) != 3:
                    c.issue("warning", "bn_bg_three_sides", f"Bottom hBN covers the BG on {sum(inside)} of 4 sides; the lab rule is 3 "
                            "(the BG sticks out on the fourth for its contact).", rule="gr.bottom_bn_larger_than_bg_three_sides", layer=bbn.uid)
    if spec.family == "ju_rhombohedral":
        for d in devices:
            if d.material == "graphene" and d.stacking != "rhombohedral":
                c.issue("warning", "ju_stacking", f"{d.label}: stacking is not marked rhombohedral.", layer=d.uid)
            if d.material == "graphene" and d.layer_count.value is not None and d.layer_count.value < 3:
                c.issue("warning", "ju_layers", f"{d.label}: rhombohedral stacking needs ≥ 3 layers.", layer=d.uid)


def _check_requirements(c: _Ctx) -> None:
    spec = c.spec
    for req in spec.requirements:
        m = req.match
        ok = True
        if req.kind == "layer":
            ok = any(all(same_value(getattr(l, k, None), v) for k, v in m.items()) for l in spec.layers)
        elif req.kind == "contact_scheme":
            ok = spec.contact_scheme == m.get("value")
        elif req.kind == "param":
            ok = same_value(spec.hall_bar.get(m.get("key", "")), m.get("value"))
        elif req.kind == "material":
            ok = any(l.material == m.get("value") for l in spec.layers)
        if not ok:
            c.issue("error", "requirement", f"Requested but not in the design: {req.text}.")


def validate_stage1(spec: DeviceSpec, config: LabConfig) -> Report:
    report = Report()
    c = _Ctx(spec, config, report)
    _check_structure(c)
    _check_values(c)
    _check_electrostatics(c)
    _check_family(c)
    _check_requirements(c)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    report.issues.sort(key=lambda i: order.get(i.severity, 9))
    return report


def apply_spec_answers(spec: DeviceSpec, answers: dict[str, Any], source: str = "user") -> list[str]:
    """Write ``spec:`` answers (from the missing-values dialog) into the spec."""
    from .model import MetalPart, Q
    done = []
    for key, value in answers.items():
        if not key.startswith("spec:") or value is None or value == "":
            continue
        path = key[5:]
        if path.startswith("hall_bar."):
            spec.hall_bar.params.setdefault(path[9:], Q()).set(float(value), source)
        elif path.startswith("pattern."):
            spec.pattern_params.setdefault(path[8:], Q()).set(float(value), source)
        elif path.startswith("gate."):
            spec.gate_limits_V.setdefault(path[5:], Q()).set(float(value), source)
        else:
            uid, _, fld = path.partition(".")
            layer = spec.layer(uid)
            if layer is None:
                continue
            if fld == "material":
                layer.material = str(value)
            elif fld.startswith("metal."):
                metal = fld[6:]
                part = next((p for p in layer.metal_parts if p.metal == metal), None)
                if part is None:
                    part = MetalPart(metal, Q())
                    layer.metal_parts.append(part)
                part.thickness_nm.set(float(value), source)
            else:
                getattr(layer, fld).set(float(value), source)
        done.append(key)
    return done


def structured_summary(spec: DeviceSpec, config: LabConfig) -> dict:
    """The structured specification shown beside the 3D preview: stack, dimensions, contacts, gates."""
    stack = []
    for i, l in enumerate(reversed(spec.layers)):
        mat = config.material(l.material) if l.material else None
        entry = {"position_from_top": i + 1, "name": l.label, "material": mat.display_name if mat else (l.material or "—"),
                 "function": FUNCTIONS.get(l.function, l.function)}
        t = thickness_nm(l)
        if l.metal_parts:
            entry["metal_nm"] = {p.metal: p.thickness_nm.value for p in l.metal_parts}
        if t is not None:
            entry["thickness_nm"] = round(t, 3)
        if l.layer_count.value is not None:
            entry["layers"] = int(l.layer_count.value)
        if l.width_um.value is not None or l.length_um.value is not None:
            entry["size_um"] = [l.length_um.value, l.width_um.value]
        if l.twist_deg.value is not None:
            entry["twist_deg"] = l.twist_deg.value
        if l.stacking:
            entry["stacking"] = l.stacking
        stack.append(entry)
    gates = [{"name": l.label, "function": FUNCTIONS[l.function], "material": l.material,
              "max_V": spec.gate_limits_V.get(l.uid).value if spec.gate_limits_V.get(l.uid) else None}
             for l in spec.by_function(*GATE_FUNCTIONS)]
    sub = spec.substrate()
    smat = config.material(sub.material) if sub and sub.material else None
    if smat is not None and smat.conducting == "yes":
        gates.append({"name": "Doped Si", "function": "Global back gate", "material": smat.display_name,
                      "max_V": spec.gate_limits_V.get(sub.uid).value if spec.gate_limits_V.get(sub.uid) else None})
    contacts = [{"name": l.label, "function": FUNCTIONS[l.function], "material": l.material,
                 "metal_nm": {p.metal: p.thickness_nm.value for p in l.metal_parts}} for l in spec.by_function(*CONTACT_FUNCTIONS, "leads")]
    return {
        "device": spec.name, "family": spec.family, "pattern": spec.pattern,
        "contact_scheme": spec.contact_scheme,
        "hall_bar_um": {k: q.value for k, q in spec.hall_bar.params.items()},
        "stack_top_to_bottom": stack, "gates": gates, "contacts": contacts,
        "total_stack_nm": round(sum(t for t in (thickness_nm(l) for l in spec.layers if l.function != "substrate"
                                                and not (category(config, l) == "metal")) if t), 2),
    }


# ---------------------------------------------------------------------- parameters behind an issue
HALL_BAR_CODES = {"hall_bar_fit", "sd_leg_spacing", "hall_bar_pairs", "hall_bar_value", "gate_in_gap", "channel_not_gated"}
THICKNESS_CODES = {"xu_middle_thin", "contact_too_thin", "hbn_breakdown", "substrate_breakdown",
                   "thickness_range", "thickness"}
SIZE_CODES = {"hall_bar_fit", "xu_bottom_larger", "xu_middle_margin", "gate_in_gap", "size", "lateral_size"}
LAYER_FIELDS = {"thickness_nm": ("Thickness", "nm"), "width_um": ("Width", "um"), "length_um": ("Length", "um"),
                "layer_count": ("Layers", ""), "twist_deg": ("Twist", "deg")}


@dataclass
class IssueParam:
    """One value an issue depends on, editable where the issue is shown. ``key`` is in the
    ``spec:`` namespace of :func:`apply_spec_answers`."""
    key: str
    label: str
    unit: str
    q: Any            # the model.Q holding the value

    @property
    def value(self):
        return self.q.value


def issue_parameters(spec: DeviceSpec, config: LabConfig, issue: Issue) -> list[IssueParam]:
    """The dimensions behind ``issue``: its field, the layers it names (by uid or by label in the
    message), and for Hall-bar / thickness / size problems the matching values. Structural problems
    (order, material, function) have none: they are fixed in the Stack tab."""
    from .model import HALL_BAR_FIELDS, PATTERN_FIELDS, Q
    out: dict[str, IssueParam] = {}

    def add_layer(layer, fields) -> None:
        cat = category(config, layer)
        if cat == "metal" and "thickness_nm" in fields:
            for part in layer.metal_parts:
                out[f"spec:{layer.uid}.metal.{part.metal}"] = IssueParam(f"spec:{layer.uid}.metal.{part.metal}",
                                                                         f"{layer.label}: {part.metal} thickness", "nm", part.thickness_nm)
            fields = [f for f in fields if f != "thickness_nm"]
        if cat in ("substrate", None):
            return
        for f in fields:
            if f == "layer_count" and layer.layer_count.value is None and f not in (issue.field or ""):
                continue
            if f == "twist_deg" and layer.twist_deg.value is None and f not in (issue.field or ""):
                continue
            lab, unit = LAYER_FIELDS[f]
            out[f"spec:{layer.uid}.{f}"] = IssueParam(f"spec:{layer.uid}.{f}", f"{layer.label}: {lab.lower()}", unit, getattr(layer, f))

    def add_hall_bar(keys=None) -> None:
        for k in keys or HALL_BAR_FIELDS:
            q = spec.hall_bar.params.setdefault(k, Q())
            out[f"spec:hall_bar.{k}"] = IssueParam(f"spec:hall_bar.{k}", f"Hall bar: {HALL_BAR_FIELDS[k][0]}", HALL_BAR_FIELDS[k][1], q)

    fld = issue.field or ""
    named = [l for l in spec.layers if l.uid == issue.layer or (l.label and l.label in issue.message)]
    if fld.startswith("hall_bar."):
        add_hall_bar([fld[9:]] if fld[9:] in HALL_BAR_FIELDS else None)
    elif fld.startswith("pattern.") and fld[8:] in PATTERN_FIELDS:
        k = fld[8:]
        out["spec:" + fld] = IssueParam("spec:" + fld, PATTERN_FIELDS[k][0], PATTERN_FIELDS[k][1], spec.pattern_params.setdefault(k, Q()))
    elif fld in LAYER_FIELDS and named:
        add_layer(named[0], [fld])
    code = issue.code
    fields = []
    if code in THICKNESS_CODES:
        fields.append("thickness_nm")
    if code in SIZE_CODES:
        fields += ["width_um", "length_um"]
    if not fields and code not in HALL_BAR_CODES and not fld:
        fields = ["thickness_nm", "width_um", "length_um", "layer_count", "twist_deg"]
    for layer in named:
        add_layer(layer, fields)
    if code in HALL_BAR_CODES:
        add_hall_bar()
    if code in ("hbn_breakdown", "substrate_breakdown"):
        for layer in named:
            if layer.function in GATE_FUNCTIONS or code == "substrate_breakdown":
                q = spec.gate_limits_V.setdefault(layer.uid, Q())
                out[f"spec:gate.{layer.uid}"] = IssueParam(f"spec:gate.{layer.uid}", f"{layer.label}: planned max |V|", "V", q)
    return list(out.values())
