"""Configure: fill every empty dimension of a stack so that the specification passes validation.

The researcher chooses the layers (materials, functions, order: from a template or by hand) and
the physics (layer counts, twist angles, gate voltages). Configure then fills what is still empty:

* the Hall bar (width, probe span, probe pairs, arm width and length, source/drain extension);
* the lateral design size of every flake (device layers, graphite gates, hBN);
* the thickness of every 2D layer and of every metal in a deposited stack;
* the Stage 2 gate geometry (metal top gate, contact gates, split gates).

Rules it never breaks (see AGENTS.md §2):

* A value someone entered (source ``user``, ``measured``, ``assistant``, ``template`` …) is never
  changed. Only empty values and values Configure wrote itself (source ``auto``) are (re)computed.
* Every number comes from a lab rule or a lab design value in config/, combined by the geometric
  relations written out below. Nothing is guessed: if a lab value is missing, Configure returns
  it as a :class:`~ustack.config.MissingItem` (asked for once, then saved to config/ as approved).
* Each generated value records why (``Q.note``), so the specification shows its derivation.

The design relations (µm; ``ew``/``es`` = electrode width/spacing target, ``shift`` = stacking
misalignment, ``ov`` = 1D-contact overlap, ``beyond`` = arms beyond the BG, ``d_gr`` = contact to
graphite gate; ``k_…`` are the presets in config/presets.json, editable in the Presets dialog):

* arm width = ``ew``; channel width W = k_w·arm width; probe pairs = preset; probe pitch =
  max(k_p·W, arm width + ``es``, arm width + arm-spacing minimum); L = (pairs − 1)·pitch.
* gate margin g = max(``shift``, k_g·``es``): how far a graphite gate is designed beyond the channel.
* 1D contacts next to a gate: arm length = g + max(``beyond``, ``d_gr``) + ``ov`` + ``es``; straight
  S/D extension = arm width/2 + the same, so every contacted end clears the gate; otherwise arm
  length = max(k·W, ``ov`` + ``es``) and S/D extension = k·W.
* bent S/D (each end turns 90°, S to +y, D to −y): S/D extension = leg width + arm width/2 + ``es``
  (leg width = W, or the arm width for pre-patterned electrodes). Style ``auto`` takes the larger
  of the straight and bent extensions so Stage 2 can use either.
* two pre-patterned electrode columns (Mak, bent S/D): the channel width is k_w·arm width, the BG
  is the channel + 2g (+ extra under a TG), the bottom hBN the BG + hBN margin, the electrode gap
  W the bottom hBN + 2·``shift``; the device spans the gap and reaches k_pt·``es`` onto each
  electrode; electrode length = that overlap + 2g.
* device flake = Hall-bar footprint (L + 2·S/D) × (W + 2·arm length); graphite gate = (L + arm
  width + 2g) × (W + 2g), a BG under a TG k_bg·``es`` larger on every side; hBN = the largest
  conducting flake + k_h·``es`` on every side (top hBN k_top; Xu: middle hBN = TMD + 2·margin;
  bottom hBN larger still).
* metal top gate length = L + arm width + 2·k·``es``, overhang past the graphene = k·``es``;
  contact-gate reaches and split-gate gap = k·``es``.

The flake list adds ``shift`` on every side of these design sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import LabConfig, MissingItem
from .model import FUNCTIONS, GATE_FUNCTIONS, PATTERN_FIELDS, DeviceSpec, Layer, MetalPart, Q
from .validation import Report, assign_roles, category, is_conductor, is_insulator, validate_stage1

AUTO = "auto"
#: Sources Configure may overwrite (empty, or written by Configure itself).
FREE_SOURCES = ("", AUTO)
#: Validation findings Configure resolves itself once its own inputs are known.
_FILLED_FIELDS = (".thickness_nm", ".width_um", ".length_um", ".metal.")


@dataclass
class Change:
    target: str
    label: str
    value: float
    unit: str
    reason: str

    def text(self) -> str:
        return f"{self.label}: {self.value:g} {self.unit}".rstrip() + f"  ({self.reason})"


@dataclass
class ConfigureResult:
    changes: list[Change] = field(default_factory=list)
    missing: list[MissingItem] = field(default_factory=list)
    report: Report | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and self.report is not None and self.report.ok


def _free(q: Q) -> bool:
    return q.value is None or q.source in FREE_SOURCES


class _Run:
    def __init__(self, spec: DeviceSpec, config: LabConfig):
        self.spec, self.config = spec, config
        self.changes: list[Change] = []
        self.missing: dict[str, MissingItem] = {}
        self.notes: list[str] = []

    # ------------------------------------------------------------------ helpers
    def put(self, q: Q, value: float, target: str, label: str, unit: str, reason: str) -> None:
        if not _free(q):
            return
        value = round(float(value), 3)
        if q.value == value and q.source == AUTO:
            return
        q.value, q.source, q.note = value, AUTO, reason
        self.changes.append(Change(target, label, value, unit, reason))

    def rule(self, rid: str, required: bool = True) -> float | None:
        r = self.config.rules.get(rid)
        if r is None:
            return None
        if r.is_missing:
            if required:
                self.missing[f"rule:{rid}"] = self.config.missing_item(rid)
            return None
        return r.value

    def design_default(self, name: str, sub: str, label: str, unit: str, suggested: Any = None, hint: str = "") -> float | None:
        entry = self.config.defaults.get(name) or {}
        value = (entry.get("value") or {}).get(sub)
        if value is not None:
            return float(value)
        key = f"default:{name}#{sub}"
        self.missing[key] = MissingItem(key, label, unit, "number", hint or "Lab design value, saved to config/rules.json › defaults "
                                        "and used for every later device", suggested)
        return None

    # ------------------------------------------------------------------ steps
    def hall_bar(self) -> dict[str, float] | None:
        spec, hb, P = self.spec, self.spec.hall_bar, self.config.preset
        ew = self.rule("ebl.electrode_width_target_um", required=False) or self.rule("ebl.contact_width_min_um")
        es = self.rule("ebl.electrode_spacing_target_um", required=False) or self.rule("ebl.contact_spacing_min_um")
        shift = self.rule("stack.alignment_shift_um")
        if None in (ew, es):
            return None
        arm_sp = self.rule("ebl.arm_spacing_min_um", required=False) or 0.0
        k_w, k_p, k_g = P("hall_bar.width_per_arm"), P("hall_bar.pitch_per_width"), P("gates.margin_spacings")
        k_bg, k_h = P("gates.bg_extra_spacings"), P("hbn.margin_spacings")
        p = {k: hb.params.setdefault(k, Q()) for k in ("channel_width_um", "channel_length_um", "n_probe_pairs", "arm_width_um",
                                                       "arm_length_um", "sd_extension_um")}
        style = hb.sd_style
        # two electrode columns pre-patterned beside the gate stack (Mak): W is the gap between the columns
        cols = spec.contact_scheme == "prepatterned_bottom" and style == "bent"
        self.put(p["arm_width_um"], ew, "hall_bar.arm_width_um", "Hall bar · arm width", "µm", f"electrode width target {ew:g} µm")
        aw = p["arm_width_um"].value
        g = max(shift or 0.0, k_g * es)
        has_tg = any(is_conductor(self.config, l) for l in spec.by_function("top_gate"))
        grow = k_bg * es if has_tg else 0.0
        w_ch = k_w * aw
        if cols:
            w_bg = w_ch + 2 * (g + grow)
            w_bbn = w_bg + 2 * k_h * es
            self.put(p["channel_width_um"], w_bbn + 2 * (shift or 0.0), "hall_bar.channel_width_um", "Electrode gap W", "µm",
                     f"bottom hBN {w_bbn:g} µm (channel {w_ch:g} + gate margins + hBN margin) + stacking shift on each side, "
                     "so the gate stack never lands on the electrodes")
        else:
            self.put(p["channel_width_um"], w_ch, "hall_bar.channel_width_um", "Hall bar · channel width W", "µm", f"{k_w:g} × arm width")
        W = p["channel_width_um"].value
        if not cols:
            w_ch = W
        self.put(p["n_probe_pairs"], P("hall_bar.n_probe_pairs"), "hall_bar.n_probe_pairs", "Hall bar · probe pairs", "",
                 "preset (2 = minimum for four-probe Rxx and Rxy)")
        n = int(p["n_probe_pairs"].value)
        pitch = max(k_p * w_ch, aw + es, aw + arm_sp)
        self.put(p["channel_length_um"], (n - 1) * pitch, "hall_bar.channel_length_um", "Hall bar · probe span L", "µm",
                 f"{n - 1} × probe pitch {pitch:g} µm ({k_p:g} × channel width, ≥ arm width + spacing)")
        L = p["channel_length_um"].value
        edge = spec.contact_scheme in ("edge_1d", "top_2d")
        tags = {"graphene"} if any(d.material == "graphene" for d in spec.device_layers()) else {"tmd"}
        gates = [l for l in spec.by_function("bottom_gate", "top_gate") if is_conductor(self.config, l)]
        graphite = [l for l in gates if l.material == "graphite"]
        bg = [l for l in gates if l.function == "bottom_gate"]
        clear = 0.0
        if edge and bg and self.config.applicable("ebl.device_etch_beyond_bg_um", spec.family, tags):
            clear = max(clear, self.rule("ebl.device_etch_beyond_bg_um") or 0.0)
        if edge and graphite and self.config.applicable("ebl.contact_to_graphite_gate_min_um", spec.family, tags):
            clear = max(clear, self.rule("ebl.contact_to_graphite_gate_min_um") or 0.0)
        ov = self.rule("ebl.contact_overlap_um") if edge else 0.0
        if ov is None:
            return None
        ov_pt = P("prepattern.tmd_overlap_spacings") * es
        leg = aw if spec.contact_scheme == "prepatterned_bottom" else W
        sd_bent = leg + aw / 2 + es
        if cols:
            arm_l = ov_pt + 2 * g
            why = f"device overlap {ov_pt:g} µm onto the electrode + 2 × gate margin {g:g} µm (the lead starts off the device)"
            sd, sd_why = sd_bent, f"bent S/D electrode ({leg:g} µm) + half an arm + spacing {es:g} µm from the outer probe"
        else:
            if edge and clear:
                arm_l = g + clear + ov + es
                sd_straight = aw / 2 + g + clear + ov + es
                why = f"gate margin {g:g} + clearance {clear:g} + contact overlap {ov:g} + spacing {es:g} µm"
            else:
                arm_l = max(P("hall_bar.arm_length_per_width_no_gate") * W, ov + es)
                sd_straight = P("hall_bar.sd_extension_per_width_no_gate") * W
                why = "no gate to clear: preset × channel width"
            if style == "straight":
                sd, sd_why = sd_straight, why
            elif style == "bent":
                sd, sd_why = sd_bent, f"bent leg {leg:g} µm + half an arm + spacing {es:g} µm from the outer probe"
            else:
                sd = max(sd_straight, sd_bent)
                sd_why = (f"source/drain style automatic: fits straight ({sd_straight:g} µm: {why}) and bent "
                          f"({sd_bent:g} µm) contacts")
        self.put(p["arm_length_um"], arm_l, "hall_bar.arm_length_um", "Hall bar · arm length", "µm", why)
        self.put(p["sd_extension_um"], sd, "hall_bar.sd_extension_um", "Hall bar · S/D extension", "µm", sd_why)
        if shift is None:
            return None
        return {"W": W, "Wch": w_ch, "L": L, "aw": aw, "al": p["arm_length_um"].value, "sd": p["sd_extension_um"].value, "g": g,
                "grow": grow, "es": es, "shift": shift, "ov": ov, "edge": edge, "cols": cols, "ov_pt": ov_pt}

    def sizes(self, d: dict[str, float]) -> None:
        spec, cfg, P = self.spec, self.config, self.config.preset
        dims: dict[str, tuple[float, float]] = {}   # uid → (length, width) used for the hBN sizes

        def size(layer: Layer, length: float, width: float, why: str) -> None:
            if layer.outline:
                xs, ys = zip(*layer.outline)
                dims[layer.uid] = (max(xs) - min(xs), max(ys) - min(ys))
                return
            self.put(layer.length_um, length, f"{layer.uid}.length_um", f"{layer.label} · length", "µm", why)
            self.put(layer.width_um, width, f"{layer.uid}.width_um", f"{layer.label} · width", "µm", why)
            dims[layer.uid] = (layer.length_um.value, layer.width_um.value)

        es = d["es"]
        k_h, k_top = P("hbn.margin_spacings"), P("hbn.top_margin_spacings")
        if d["cols"]:
            fp = (d["L"] + 2 * d["sd"], d["W"] + 2 * d["ov_pt"])
            fp_why = f"spans the electrode gap and reaches {d['ov_pt']:g} µm onto each electrode"
        else:
            fp = (d["L"] + 2 * d["sd"], d["W"] + 2 * d["al"])
            fp_why = "Hall-bar footprint"
        for dev in spec.device_layers():
            size(dev, *fp, fp_why)
        gates = [l for l in spec.layers if category(cfg, l) == "2d_conductor" and l.function in GATE_FUNCTIONS]
        for gate in gates:
            grow = d["grow"] if gate.function == "bottom_gate" else 0.0
            size(gate, d["L"] + d["aw"] + 2 * (d["g"] + grow), d["Wch"] + 2 * (d["g"] + grow),
                 f"probe span + gate margin {d['g']:g} µm" + (" + extra (BG larger than TG)" if grow else ""))
        if d["cols"]:
            # the bottom hBN covers the BG but stays inside the gap, so the device reaches past it onto the electrodes
            bottom = spec.by_role("bottom_hbn")
            bgs = [dims[l.uid] for l in gates if l.function == "bottom_gate" and l.uid in dims]
            if bottom is not None and bgs:
                bl, bw = max(a for a, _ in bgs), max(b for _, b in bgs)
                size(bottom, bl + 2 * k_h * es, bw + 2 * k_h * es, "covers the bottom gate + hBN margin; narrower than the "
                     "electrode gap so the device reaches past it onto the electrodes")
        conducting = [dims[u] for u in dims if not is_insulator(cfg, spec.layer(u))]
        base = (max((a for a, _ in conducting), default=fp[0]), max((b for _, b in conducting), default=fp[1]))
        insulators = [l for l in spec.layers if is_insulator(cfg, l)]
        if spec.family == "xu_tmd":
            margin = cfg.value("xu.middle_hbn_edge_margin_um") or 0.0
            mid = spec.by_role("middle_hbn")
            if mid is not None:
                size(mid, fp[0] + 2 * margin + 2 * k_h * es, fp[1] + 2 * margin + 2 * k_h * es, f"TMD + {margin:g} µm margin (Xu rule) + hBN margin")
                base = (max(base[0], dims[mid.uid][0]), max(base[1], dims[mid.uid][1]))
            bottom = spec.by_role("bottom_hbn")
            if bottom is not None:
                size(bottom, base[0] + 2 * k_h * es, base[1] + 2 * k_h * es, "larger than the middle hBN and the TMD (Xu rule)")
                base = (max(base[0], dims[bottom.uid][0]), max(base[1], dims[bottom.uid][1]))
        for ins in insulators:
            if ins.uid in dims:
                continue
            k = k_top if ins.role == "top_hbn" else k_h
            size(ins, base[0] + 2 * k * es, base[1] + 2 * k * es, f"covers every conducting flake + {k:g} × spacing")

    def thicknesses(self) -> None:
        spec, cfg = self.spec, self.config
        for layer in spec.layers:
            cat = category(cfg, layer)
            if cat in ("2d_conductor", "2d_semiconductor", "2d_insulator") and layer.function != "device_layer":
                if not _free(layer.thickness_nm):
                    continue
                sub = f"{layer.material}:{layer.function}"
                v = self.design_default("design.thickness_nm", sub,
                                        f"Lab design thickness of {layer.material} used as {FUNCTIONS.get(layer.function, layer.function)}",
                                        "nm", layer.thickness_nm.suggested, layer.thickness_nm.suggested_source)
                if v is not None:
                    self.put(layer.thickness_nm, v, f"{layer.uid}.thickness_nm", f"{layer.label} · thickness", "nm",
                             f"lab design thickness for {sub}")
            elif cat == "metal":
                mat = cfg.material(layer.material)
                if mat is None:
                    continue
                if [p.metal for p in layer.metal_parts] != mat.components:
                    old = {p.metal: p for p in layer.metal_parts}
                    layer.metal_parts = [old.get(c, MetalPart(c, Q())) for c in mat.components]
                for part in layer.metal_parts:
                    if not _free(part.thickness_nm):
                        continue
                    v, why = self._fab_default(layer, part.metal)
                    if v is None:
                        sub = f"{layer.material}:{layer.function}:{part.metal}"
                        v = self.design_default("design.metal_nm", sub, f"Lab design thickness of {part.metal} in {mat.display_name} "
                                                f"used as {FUNCTIONS.get(layer.function, layer.function)}", "nm",
                                                part.thickness_nm.suggested, part.thickness_nm.suggested_source)
                        why = f"lab design thickness for {sub}"
                    if v is not None:
                        self.put(part.thickness_nm, v, f"{layer.uid}.metal.{part.metal}", f"{layer.label} · {part.metal}", "nm", why)

    def _fab_default(self, layer: Layer, metal: str) -> tuple[float | None, str]:
        """A metal thickness from the Fab Manual defaults in rules.json (pending or approved) when the material matches."""
        keys = []
        if layer.function in ("hall_bar_contacts", "source_drain") and self.spec.contact_scheme == "edge_1d":
            keys.append("metal.contacts_1d")
        if layer.function == "top_gate":
            keys.append("metal.top_gate")
        for key in keys:
            entry = self.config.defaults.get(key) or {}
            if entry.get("material") == layer.material and (entry.get("thickness_nm") or {}).get(metal) is not None:
                return float(entry["thickness_nm"][metal]), f"{key} default ({entry.get('status', '')}, {entry.get('source', {}).get('doc', '')})"
        return None, ""

    def pattern(self, d: dict[str, float]) -> None:
        spec, cfg, P = self.spec, self.config, self.config.preset
        es = d["es"]

        def put_p(key: str, value: float, why: str) -> None:
            q = spec.pattern_params.setdefault(key, Q())
            self.put(q, value, f"pattern.{key}", PATTERN_FIELDS[key][0], "µm", why)

        metal_tg = [l for l in spec.by_function("top_gate") if category(cfg, l) == "metal"]
        if metal_tg:
            k = P("tg.length_extra_spacings")
            put_p("tg_length_um", d["L"] + d["aw"] + 2 * k * es, f"probe span + arm width + 2 × {k:g} × spacing")
            put_p("tg_overhang_um", P("tg.overhang_spacings") * es, "preset × electrode spacing")
        if spec.by_function("contact_gate"):
            for key in ("cg_inner_um", "cg_outer_um", "cg_side_um"):
                put_p(key, P("cg.reach_spacings") * es, "preset × electrode spacing")
        if spec.by_function("split_gate"):
            put_p("sg_gap_um", P("sg.gap_spacings") * es, "preset × electrode spacing")
        if d["cols"]:
            put_p("pt_overlap_um", d["ov_pt"], "preset × electrode spacing")


def configure(spec: DeviceSpec, config: LabConfig, apply: bool = True) -> ConfigureResult:
    """Fill the empty dimensions of ``spec`` (in place when ``apply``, else on a copy). Returns what
    was set, what still has to be supplied (lab values and the researcher's physics choices), and
    the validation report of the configured specification."""
    target = spec if apply else spec.copy()
    run = _Run(target, config)
    roles = assign_roles(target, config)
    if roles:
        run.notes.append("Roles assigned to " + ", ".join(target.layer(u).label for u in roles) + ".")
    run.thicknesses()
    d = run.hall_bar()
    if d is not None:
        run.sizes(d)
        run.pattern(d)
    report = validate_stage1(target, config)
    missing = dict(run.missing)
    for item in report.missing_items():
        if item.key in missing:
            continue
        if item.key.startswith("spec:") and (item.key[5:].startswith("hall_bar.") or any(f in item.key for f in _FILLED_FIELDS)):
            if run.missing:
                continue   # filled as soon as Configure's own inputs are answered
        missing[item.key] = item
    return ConfigureResult(run.changes, list(missing.values()), report, run.notes)
