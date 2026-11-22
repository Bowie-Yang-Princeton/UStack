"""Stage 2 as a workflow of EBL steps on one chip (Qt-free).

Steps (``STEPS``): Hall bars (the lab's finger array, pre-patterned), contact gate (comb), bottom
gate (contacts on the back-gate flake), top gate (contact on the top-gate flake, or a metal top
gate), electrodes to the bond pads. Each step keeps its own state (:class:`FlowStep`): images of
the sample up to this point at several magnifications, their alignment to the bond-pad preset
(px → chip µm, :mod:`ustack.registration`), the flake outlines in the chip frame, the generated
pattern (editable) and where it was exported. The whole flow is stored in the device spec under
``stage2["flow"]``.

Flakes: the outlines traced in the *Flake images* tab are in the design frame (µm, relative
positions set in *Flake layout*). The flakes on the chip at a step are located in that step's
highest-magnification aligned image (``registration.locate_outline``); the design-frame → chip
similarity fitted on the located flakes then places every other traced flake (planned ones, not
yet on the chip, are shown dashed). The **twisted region** is the overlap of the device-layer
outlines (two TMD halves) in the chip frame, or a region the user draws.

Output layers: each step's purposes go on the GDS layer config/layers.json gives them; missing
ones are asked for (with the layer the lab's tWSe2 batch 1 drawings use as a suggestion — never
written without the user's answer).
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import klayout.db as kdb
import numpy as np

from . import fingers as fg
from . import geometry as geo
from .bondpads import BondPads
from .config import LabConfig, MissingItem, MissingValues
from .model import DeviceSpec

Poly = list[tuple[float, float]]


@dataclass(frozen=True)
class StepKind:
    key: str
    name: str
    purposes: tuple[str, ...]
    layer_function: str | None      # the spec layer this step makes (its position in the stack = when it is done)
    blurb: str


#: The electrodes, all evaporated together at the end (user, 2026-11-19: for TMD-like samples the top and bottom gates
#: are graphite flakes, so their "gate step" is only an electrode to the flake — merged into the Electrodes step).
#: group → (title, purpose, function of the stack layer it connects). Every purpose goes on the one GDS layer
#: ``electrodes`` (:data:`GDS_PURPOSE`); a group shows only when the stack has that layer (:func:`electrode_groups`).
ELECTRODE_GROUPS: dict[str, tuple[str, str, str]] = {
    "fingers": ("Hall bar electrodes", "el_fingers", "hall_bar_contacts"),
    "contact_gate": ("Contact gate electrodes", "el_cg", "contact_gate"),
    "top_gate": ("Top gate electrodes", "el_tg", "top_gate"),
    "bottom_gate": ("Bottom gate electrodes", "el_bg", "bottom_gate"),
}
ELECTRODE_PURPOSES = tuple(v[1] for v in ELECTRODE_GROUPS.values())
#: purpose of a shape → purpose of its GDS layer (the electrode groups share ``electrodes``).
GDS_PURPOSE = {p: "electrodes" for p in ELECTRODE_PURPOSES}
#: Steps removed on 2026-11-19 (merged into the electrodes); a saved flow's images / outlines of them carry over.
OLD_GATE_STEPS = ("top_gate", "bottom_gate")

STEPS: dict[str, StepKind] = {s.key: s for s in (
    StepKind("hall_bars", "Hall bars (finger array)", ("prepattern_contacts",), "hall_bar_contacts",
             "Two facing columns of fingers on the twisted region; the number of pairs is what fits."),
    StepKind("contact_gate", "Contact gate", ("contact_gates",), "contact_gate",
             "A comb over the fingers: one tooth per finger from its tip, joined by a bus bar."),
    StepKind("electrodes", "Electrodes to the bond pads", ELECTRODE_PURPOSES, "leads",
             "Every electrode in one evaporation at the end, on one GDS layer: from the Hall-bar fingers, the contact gate, the "
             "top gate and the bottom gate to the bond-pad leads (each group generated on its own)."),
)}

#: GDS layers the lab's tWSe2 batch 1 drawings use (user, 2026-11-16; made the suggested output layers by the user,
#: 2026-11-18) — offered as suggestions only, the user confirms them in the dialog.
LAYER_SUGGEST = {"prepattern_contacts": (5, 0), "contact_gates": (6, 0), "electrodes": (7, 0), "bg_contacts": (7, 0),
                 "tg_contacts": (7, 0)}
PURPOSE_NAMES = {"prepattern_contacts": "Fingers (pre-patterned contacts)", "contact_gates": "Contact gate",
                 "electrodes": "Electrodes", "bg_contacts": "Back-gate contacts", "tg_contacts": "Top-gate contact",
                 "top_gate": "Metal top gate", **{v[1]: v[0] for v in ELECTRODE_GROUPS.values()}}


class BlockedByElectrodes(ValueError):
    """A gate contact finds no room only because other electrode groups cross its flake (``groups``): drawing the gate
    electrode first and those groups after it makes room (what Generate all does)."""

    def __init__(self, message: str, groups: list[str]):
        super().__init__(message)
        self.groups = groups


def electrode_groups(spec: DeviceSpec) -> list[str]:
    """The electrode groups this sample needs: those whose structure (fingers, contact gate, gates) is in the stack."""
    fns = {l.function for l in spec.layers}
    return [g for g, (_t, _p, fn) in ELECTRODE_GROUPS.items() if fn in fns]


def gds_shapes(shapes: dict[str, list]) -> dict[str, list]:
    """Shapes keyed by their GDS purpose (the electrode groups merged into ``electrodes``)."""
    out: dict[str, list] = {}
    for purpose, polys in shapes.items():
        out.setdefault(GDS_PURPOSE.get(purpose, purpose), []).extend(polys)
    return out
OBJECTIVE_KEYS = ("5x", "10x", "20x", "50x", "100x")


def mag(key: str) -> float:
    m = re.match(r"(\d+(?:\.\d+)?)x", key)
    return float(m.group(1)) if m else 0.0


# ====================================================================== state
@dataclass
class FlowStep:
    key: str
    images: dict[str, str] = field(default_factory=dict)          # objective → file
    transforms: dict[str, dict] = field(default_factory=dict)     # objective → Similarity json (px → chip µm)
    align_notes: dict[str, str] = field(default_factory=dict)
    outlines: dict[str, list[Poly]] = field(default_factory=dict) # layer uid → chip µm polygons
    located: dict[str, dict] = field(default_factory=dict)        # uid → {"how": located|projected|drawn|edited, "score"}
    twisted: list[Poly] = field(default_factory=list)             # chip µm
    twisted_how: str = ""
    overrides: dict[str, float] = field(default_factory=dict)     # FingerParams field → value (this step only)
    fixed_angle: float | None = None
    max_pairs: int | None = None
    shapes: dict[str, list[Poly]] = field(default_factory=dict)   # purpose → polygons (editable)
    debris: list[dict] = field(default_factory=list)              # bubbles, cracks, stray flakes: {kind, polygon, score, how}
    accepted: list[dict] = field(default_factory=list)            # errors accepted by the user: {code, x, y, message}
    debris_looked: bool = False                                   # this step's own bubbles / cracks / flakes (else inherited)
    found_patterns: dict = field(default_factory=dict)            # earlier step → {dx, dy, angle, cx, cy, ...}: where its
                                                                  # written pattern lies in this step's images
    array: dict | None = None
    routes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    show_image: str | None = None
    exported: str | None = None
    plan: dict[str, dict] = field(default_factory=dict)          # uid → {"cx", "cy", "angle"}: a device layer planned from its traced border
    comb: dict | None = None                                      # contact gate: {"skip", "dirs", "extension"} (fingers.fit_contact_gate)
    group_notes: dict[str, list[str]] = field(default_factory=dict)  # electrodes: group → its notes (each group generated alone)

    FIELDS = ("images", "transforms", "align_notes", "outlines", "located", "twisted", "twisted_how", "overrides", "fixed_angle",
              "max_pairs", "shapes", "array", "routes", "notes", "issues", "show_image", "exported", "plan", "comb", "group_notes", "debris", "accepted", "debris_looked", "found_patterns")

    def to_json(self) -> dict:
        return {"key": self.key, **{k: copy.deepcopy(getattr(self, k)) for k in self.FIELDS}}

    @classmethod
    def from_json(cls, d: dict) -> "FlowStep":
        st = cls(d["key"])
        for k in cls.FIELDS:
            if k in d:
                setattr(st, k, copy.deepcopy(d[k]))
        if "debris_looked" not in d:                     # saved before 2026-11-20: a list it has is its own
            st.debris_looked = bool(st.debris)
        return st

    @property
    def status(self) -> str:
        if self.exported:
            return "exported"
        if self.shapes:
            return "generated"
        if self.images:
            return "images"
        return "not started"


@dataclass
class Flow:
    bondpads: str | None = None
    steps: dict[str, FlowStep] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)       # the steps shown (expanded) in the Stage 2 tab
    order: list[str] = field(default_factory=lambda: list(STEPS))   # process order (the user can reorder)
    canvas_image: list | None = None                          # [step key, objective]: the image on the EBL canvas
    ui: dict = field(default_factory=dict)                    # as saved: {"current", "collapsed", "hidden", "view", "tab"}

    def step(self, key: str) -> FlowStep:
        if key not in self.steps:
            self.steps[key] = FlowStep(key)
        return self.steps[key]

    def ordered(self, keys) -> list[str]:
        """``keys`` in process order."""
        keys = set(keys)
        return [k for k in self.order if k in keys]

    def move(self, key: str, delta: int) -> bool:
        """Move a step earlier (−1) or later (+1) in the process order."""
        if key not in self.order:
            return False
        i = self.order.index(key)
        j = i + delta
        if not 0 <= j < len(self.order):
            return False
        self.order[i], self.order[j] = self.order[j], self.order[i]
        return True

    def to_json(self) -> dict:
        return {"bondpads": self.bondpads, "steps": {k: s.to_json() for k, s in self.steps.items()}, "selected": list(self.selected),
                "order": list(self.order), "canvas_image": list(self.canvas_image) if self.canvas_image else None,
                "ui": copy.deepcopy(self.ui)}

    @classmethod
    def from_json(cls, d: dict | None) -> "Flow":
        d = d or {}
        order = [k for k in d.get("order", []) if k in STEPS]
        order += [k for k in STEPS if k not in order]
        ci = d.get("canvas_image")
        raw = d.get("steps", {})
        steps = {k: FlowStep.from_json(v) for k, v in raw.items() if k in STEPS}
        selected = [k for k in d.get("selected", []) if k in STEPS]
        # the old top / bottom gate steps (merged into the electrodes, 2026-11-19): their images, alignment and flake
        # outlines carry over to the electrodes step when it has none (their tiny contact squares are dropped)
        for old in OLD_GATE_STEPS:
            if old not in raw:
                continue
            if old in d.get("selected", []) and "electrodes" not in selected:
                selected.append("electrodes")
            o = FlowStep.from_json(raw[old])
            el = steps.get("electrodes")
            if o.images and (el is None or not el.images):
                el = steps.setdefault("electrodes", FlowStep("electrodes"))
                for f in ("images", "transforms", "align_notes", "outlines", "located", "show_image"):
                    setattr(el, f, copy.deepcopy(getattr(o, f)))
        el = steps.get("electrodes")
        if el is not None and "electrodes" in el.shapes:          # before the groups: one purpose for every electrode
            el.shapes.setdefault("el_fingers", []).extend(el.shapes.pop("electrodes"))
        if ci and ci[0] not in STEPS:
            ci = None
        return cls(d.get("bondpads"), steps, selected, order, list(ci) if ci else None, dict(d.get("ui") or {}))

    @classmethod
    def of(cls, spec: DeviceSpec) -> "Flow":
        flow = cls.from_json(spec.stage2.get("flow"))
        settle_plans(spec, flow)
        return flow

    def store(self, spec: DeviceSpec) -> None:
        spec.stage2["flow"] = self.to_json()


# ====================================================================== stack helpers
def step_layer(spec: DeviceSpec, key: str):
    fn = STEPS[key].layer_function
    return next((l for l in spec.layers if l.function == fn), None)


def present_layers(spec: DeviceSpec, key: str) -> list[str]:
    """Flakes on the chip when step ``key`` is written: every layer below the step's own layer."""
    own = step_layer(spec, key)
    if own is None:
        return [l.uid for l in spec.layers[1:]]
    idx = spec.layers.index(own)
    return [l.uid for l in spec.layers[1:idx]]


#: A flake carried from an earlier step where it was only planned (or projected): by this step it is on
#: the chip (it lies below the step's own layer in the stack), so it needs locating; the earlier
#: position is shown as the starting guess.
EXPECTED_HOW = "expected"


def settle_plans(spec: DeviceSpec, flow: "Flow") -> bool:
    """A layer can only be planned at a step where it is not on the chip yet (above the step's own
    layer). Plans left on layers that are already there (e.g. copied when ticked steps shared their
    state, before v0.5's per-step cards) become "needs locating", their planned outline kept as the
    starting guess. Returns whether anything changed."""
    changed = False
    for key, st in flow.steps.items():
        if key not in STEPS:
            continue
        present = set(present_layers(spec, key))
        for uid in [u for u in st.plan if u in present]:
            st.plan.pop(uid)
            st.located[uid] = dict(st.located.get(uid) or {}, how=EXPECTED_HOW, score=None)
            changed = True
        for uid in [u for u, v in st.located.items() if v.get("how") == "projected" and u not in present]:
            st.located.pop(uid)                  # not on the chip yet at this step: nothing to show (no auto projection)
            st.outlines.pop(uid, None)
            changed = True
        if changed and not st.plan and st.twisted_how == PLAN_HOW:
            st.twisted_how = "planned at an earlier step"
    return changed


def flake_state(spec: DeviceSpec, st: "FlowStep", key: str, uid: str) -> tuple[str, bool]:
    """What the step knows about flake ``uid``: (text for the list, on the chip at this step?).
    The stack decides: every layer below the step's own layer is done when the step is written."""
    on_chip = uid in present_layers(spec, key)
    info = st.located.get(uid) or {}
    how = info.get("how")
    if not on_chip:
        if how == IDENTIFIED_LATER and info.get("at") in STEPS:
            return f"identified at {STEPS[info['at']].name.split(' (')[0]} (dropped after this step)", False
        return ("planned (projected)" if how == "projected" else "planned"), False
    if how is None or how in (EXPECTED_HOW, "planned") or (how == "projected" and info.get("inherited")):
        return ("needs locating" + (" (earlier position shown)" if st.outlines.get(uid) else "")), True
    text = how + (f" ({info['score']:.2f})" if info.get("score") is not None else "")
    if info.get("inherited") in STEPS:
        text += f" · from {STEPS[info['inherited']].name.split(' (')[0]}"
    return text, True


def hbn_under(spec: DeviceSpec, key: str):
    """The hBN a step's metal is written on: the nearest hBN layer below the step's own layer (the
    middle hBN under the contact gate)."""
    own = step_layer(spec, key)
    if own is None:
        return None
    for layer in reversed(spec.layers[:spec.layers.index(own)]):
        if layer.material.lower().startswith("hbn"):
            return layer
    return None


def tmd_outlines(spec: DeviceSpec, st: "FlowStep") -> list[Poly]:
    """The device layers' (TMD) outlines at a step, every piece (the contact-gate bus keeps off them)."""
    return [q for u in device_layer_uids(spec) for q in (st.outlines.get(u) or [])]


def has_source_drain(spec: DeviceSpec) -> bool:
    """Whether the stack plans source / drain contacts of its own (else two fingers serve as them)."""
    return any(l.function == "source_drain" for l in spec.layers)


def previous_imaged_step(flow: "Flow", key: str) -> str | None:
    """The latest step before ``key`` (process order) with an aligned image: the chip as it was
    before the flakes of this step were dropped (change detection, ``registration.change_map``)."""
    order = flow.order
    if key not in order:
        return None
    for k in reversed(order[:order.index(key)]):
        st = flow.steps.get(k)
        if st is not None and st.transforms:
            return k
    return None


def new_layers(spec: DeviceSpec, key: str, earlier: str | None) -> list[str]:
    """Layers on the chip at step ``key`` that were not yet there at step ``earlier`` (the stack decides)."""
    if earlier is None:
        return []
    before = set(present_layers(spec, earlier))
    return [u for u in present_layers(spec, key) if u not in before]


def device_layer_uids(spec: DeviceSpec) -> list[str]:
    tmds = [l.uid for l in spec.layers if l.role == "tmd"]
    return tmds or [l.uid for l in spec.device_layers()]


def twisted_region(spec: DeviceSpec, outlines: dict[str, list[Poly]]) -> list[Poly]:
    """Overlap of the device-layer outlines (chip µm); the one outline when there is a single piece."""
    pieces = [outlines.get(u) for u in device_layer_uids(spec)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return []
    reg = geo.regions(pieces[0]).merged()
    for p in pieces[1:]:
        reg = reg & geo.regions(p).merged()
    return geo.to_polys(reg)


#: How the twisted region was set: these follow the device-layer outlines automatically
#: (:func:`refresh_twisted`); drawn / edited / cleared by the user, it stays as it is.
TWISTED_AUTO = "overlap of the device-layer outlines"
TWISTED_MANUAL = ("drawn", "edited", "cleared")


IDENTIFIED_LATER = "identified"


def adopt_identified(spec: DeviceSpec, flow: "Flow") -> bool:
    """A layer planned at a step (not on the chip yet there: the WSe2 halves at the Hall-bar step)
    and since found on the chip at a later step takes that actual border there too: the plan is
    dropped, the outline replaced (``how`` = ``IDENTIFIED_LATER``, ``at`` = the step) and the twisted
    region follows the actual overlap. Returns whether anything changed."""
    changed = False
    order = flow.order
    found = {}                                      # uid → (step key, polygons) of the latest step that identified it
    for k in order:
        st = flow.steps.get(k)
        if st is None:
            continue
        present = set(present_layers(spec, k))
        for uid, info in st.located.items():
            if uid in present and info.get("how") in ("located", "edited", "placed", "drawn", "traced") and st.outlines.get(uid):
                found[uid] = (k, st.outlines[uid])
    for k in order:
        st = flow.steps.get(k)
        if st is None:
            continue
        present = set(present_layers(spec, k))
        for uid, (src, polys) in found.items():
            if uid in present or src == k:
                continue
            new = [[tuple(map(float, q)) for q in poly] for poly in polys]
            info = st.located.get(uid) or {}
            if info.get("how") == IDENTIFIED_LATER and info.get("at") == src and st.outlines.get(uid) == new:
                continue
            st.plan.pop(uid, None)
            st.outlines[uid] = new
            st.located[uid] = {"how": IDENTIFIED_LATER, "at": src, "score": None}
            if st.twisted_how in ("", PLAN_HOW, "cleared", TWISTED_AUTO) or st.twisted_how.startswith("from "):
                st.twisted_how = ""                 # the twisted region follows the actual layers (refresh_twisted)
            changed = True
    return changed


def refresh_twisted(spec: DeviceSpec, st: "FlowStep") -> bool:
    """The twisted region = the overlap of the device layers' outlines at this step (both WSe2
    halves), kept up to date whenever they change (located, planned, moved, reshaped) — unless the
    user drew, edited or cleared it. Needs an outline of every device layer. Returns whether it changed."""
    if st.twisted_how in TWISTED_MANUAL:
        return False
    uids = device_layer_uids(spec)
    if not uids or any(not st.outlines.get(u) for u in uids):
        return False
    region = twisted_region(spec, st.outlines)
    if not region:
        return False
    how = PLAN_HOW if any(u in st.plan for u in uids) else TWISTED_AUTO
    new = [[tuple(map(float, q)) for q in poly] for poly in region]
    old = [[tuple(map(float, q)) for q in poly] for poly in st.twisted]
    if new == old and how == st.twisted_how:
        return False
    st.twisted, st.twisted_how = new, how
    return True


# ====================================================================== planning the device layers
PLAN_HOW = "planned from the flake images"


def plan_layers(spec: DeviceSpec) -> list[str]:
    """The layers a Hall-bar step plans before they are on the chip: the device layers (both halves
    of a twisted TMD)."""
    return device_layer_uids(spec)


def default_plan(spec: DeviceSpec, uid: str, site: tuple[float, float]) -> dict:
    """Where a planned layer starts: its traced border's middle at the device site (the middle of the
    bond pads), shifted by where it sits in the design frame — flakes placed on each other in Flake
    layout keep their relative positions; flakes never placed all start on the site."""
    layer = spec.layer(uid)
    c = geo.centroid(layer.outline) if layer is not None and layer.outline else (0.0, 0.0)
    placed = bool((layer.image or {}).get("placed")) if layer is not None else False
    return {"cx": float(site[0] + (c[0] if placed else 0.0)), "cy": float(site[1] + (c[1] if placed else 0.0)), "angle": 0.0}


def planned_outline(outline: Poly, entry: dict) -> Poly:
    """The traced border (design frame) turned by ``entry['angle']`` (degrees, counter-clockwise) about
    its middle (area centroid) and moved so the middle is at (cx, cy) — chip µm. Shape and size are
    the traced ones."""
    pts = np.asarray(outline, float)
    c = np.asarray(geo.centroid(outline), float)
    a = math.radians(float(entry.get("angle", 0.0)))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = (pts - c) @ R.T + [float(entry["cx"]), float(entry["cy"])]
    return [(float(x), float(y)) for x, y in out]


def apply_plan(spec: DeviceSpec, st: "FlowStep") -> None:
    """Planned layers → their chip outlines, and the twisted region (their overlap) unless it was set
    another way (drawn, edited, from located flakes)."""
    present = set(present_layers(spec, st.key)) if st.key in STEPS else set()
    for uid, entry in st.plan.items():
        layer = spec.layer(uid)
        if uid in present:                       # on the chip already at this step: not planned here
            continue
        if layer is None or not layer.outline:
            continue
        st.outlines[uid] = [planned_outline(layer.outline, entry)]
        st.located[uid] = {"how": "planned", "score": None}
    if st.plan and st.twisted_how in ("", PLAN_HOW):
        st.twisted = twisted_region(spec, st.outlines)
        st.twisted_how = PLAN_HOW if st.twisted else ""


def translation_of(old: Poly, new: Poly, tol: float = 1e-6) -> tuple[float, float] | None:
    """(dx, dy) when ``new`` is ``old`` moved as a whole, else None."""
    if len(old) != len(new) or not old:
        return None
    d = np.asarray(new, float) - np.asarray(old, float)
    if np.abs(d - d[0]).max() > tol * max(1.0, float(np.abs(np.asarray(old)).max())):
        return None
    return float(d[0][0]), float(d[0][1])


# ====================================================================== design frame → chip
def fit_design_to_chip(pairs: list[tuple[Poly, Poly]]) -> dict:
    """Similarity (rotation, uniform scale, translation, optional reflection) taking design-frame
    outlines to their located chip polygons (same vertex order). Returns
    {"m": 2×2, "t": 2, "mirror", "angle_deg", "scale", "rms_um"}."""
    src = np.concatenate([np.asarray(a, float) for a, _ in pairs])
    dst = np.concatenate([np.asarray(b, float) for _, b in pairs])
    best = None
    for mirror in (False, True):
        s_ = src * ([-1.0, 1.0] if mirror else [1.0, 1.0])
        ms, md = s_.mean(0), dst.mean(0)
        zs = (s_ - ms) @ [1, 1j]
        zd = (dst - md) @ [1, 1j]
        k = np.vdot(zs, zd) / np.vdot(zs, zs)
        a = float(np.angle(k))
        R = abs(k) * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        m = R @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))
        t = md - (R @ ms)
        res = (src @ m.T + t) - dst
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        if best is None or rms < best["rms_um"]:
            best = {"m": m.tolist(), "t": t.tolist(), "mirror": mirror, "angle_deg": math.degrees(a), "scale": float(abs(k)), "rms_um": rms}
    return best


def apply_design_to_chip(T: dict, poly: Poly) -> Poly:
    m, t = np.asarray(T["m"]), np.asarray(T["t"])
    return [tuple(map(float, q)) for q in (np.asarray(poly, float) @ m.T + t)]


#: The located flakes must agree with their Flake-layout positions to this RMS (µm) before the
#: others are placed through it (program choice).
PROJECTION_MAX_RMS_UM = 3.0


def project_flakes(spec: DeviceSpec, st: FlowStep, uids: list[str] | None = None) -> dict | None:
    # (the Stage 2 panel passes ``uids`` = the layers on the chip at the step: nothing above it is projected)
    """Place every traced flake not located/drawn at this step through the design → chip similarity
    of the located ones. Only flakes placed on each other in Flake layout (``image["placed"]``)
    share one design frame, so only those take part. Returns the similarity (None when no placed
    flake is located; ``T["rejected"]`` when the located ones disagree by more than
    ``PROJECTION_MAX_RMS_UM``)."""
    pairs = []
    placed = lambda l: bool((l.image or {}).get("placed")) or not l.image     # drawn outlines (no image) share the design frame
    for uid, info in st.located.items():
        layer = spec.layer(uid)
        if info.get("how") == "located" and layer is not None and layer.outline and st.outlines.get(uid) and placed(layer):
            poly = st.outlines[uid][0]
            if len(poly) == len(layer.outline):
                pairs.append((layer.outline, poly))
    if not pairs:
        return None
    T = fit_design_to_chip(pairs) if len(pairs) > 1 else _single_pair(pairs[0])
    if T["rms_um"] > PROJECTION_MAX_RMS_UM:
        T["rejected"] = True                    # the located flakes disagree with the Flake layout: place nothing
        return T
    for layer in spec.layers:
        if uids is not None and layer.uid not in uids:
            continue
        if layer.outline and placed(layer) and st.located.get(layer.uid, {}).get("how") not in ("located", "drawn", "edited"):
            st.outlines[layer.uid] = [apply_design_to_chip(T, layer.outline)]
            st.located[layer.uid] = {"how": "projected", "score": None}
    return T


def _single_pair(pair) -> dict:
    return fit_design_to_chip([pair])


# ====================================================================== one shape everywhere
#: Chip outlines that are the traced border moved (vertex for vertex): a reshape of either one is
#: carried to the other (drawn outlines and wand traces on a chip image are shapes of their own).
FOLLOW_HOW = ("located", "projected", "planned", "placed", "edited", "expected", "identified")


def is_rigid_move(old: Poly | None, new: Poly | None, tol: float = 1e-6) -> bool:
    """``new`` is ``old`` moved / turned / mirrored as a whole (same shape and size)."""
    if not old or not new or len(old) != len(new):
        return False
    T = fit_design_to_chip([(old, new)])
    span = max(1.0, float(np.ptp(np.asarray(old, float), axis=0).max()))
    return abs(T["scale"] - 1.0) < tol * 10 and T["rms_um"] < tol * span


def _follows(st: FlowStep, uid: str, design: Poly | None) -> bool:
    polys = st.outlines.get(uid)
    how = (st.located.get(uid) or {}).get("how")
    return bool(design and polys and len(polys[0]) == len(design) and (how in FOLLOW_HOW or uid in st.plan))


def follow_design_edit(spec: DeviceSpec, st: FlowStep, uid: str, old_design: Poly) -> bool:
    """The traced border of ``uid`` was reshaped (Flake images, Flake layout): its chip outline at
    this step takes the new shape through the design → chip similarity of the old pair (the outline
    stays where it is on the chip). Returns whether the step changed."""
    layer = spec.layer(uid)
    if layer is None or not layer.outline or not _follows(st, uid, old_design):
        return False
    T = fit_design_to_chip([(old_design, st.outlines[uid][0])])
    new = apply_design_to_chip(T, layer.outline)
    if uid in st.plan:                    # the plan's middle follows, so the planned outline is exactly ``new``
        cx, cy = geo.centroid(new)
        st.plan[uid]["cx"], st.plan[uid]["cy"] = float(cx), float(cy)
        apply_plan(spec, st)
    else:
        st.outlines[uid] = [new] + list(st.outlines[uid][1:])
        if st.twisted_how == "overlap of the device-layer outlines" and uid in device_layer_uids(spec):
            st.twisted = twisted_region(spec, st.outlines)      # the overlap follows its layers
    return True


def design_from_chip_edit(spec: DeviceSpec, st: FlowStep, uid: str, old_chip: Poly, new_chip: Poly) -> Poly | None:
    """The chip outline of ``uid`` was reshaped (EBL canvas, Draw outline window): the traced border
    (design frame) that takes the same shape, through the chip → design similarity of the old pair;
    None when the outline is not the traced border moved."""
    layer = spec.layer(uid)
    if layer is None or not layer.outline or not old_chip or len(old_chip) != len(layer.outline):
        return None
    how = (st.located.get(uid) or {}).get("how")
    if how not in FOLLOW_HOW and uid not in st.plan:
        return None
    T = fit_design_to_chip([(old_chip, layer.outline)])
    return [list(p) for p in apply_design_to_chip(T, new_chip)]


# ====================================================================== chip → Flake layout
#: Located outlines that still carry the traced border's vertices (one per design vertex, same order).
IDENTIFIED_HOW = ("located", "edited", "placed")


def _rigid(design: Poly, chip: Poly) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (+ reflection) and translation taking a traced border onto its chip outline; the
    scale is kept at 1 (the traced size is the flake's size)."""
    T = fit_design_to_chip([(design, chip)])
    a = math.radians(T["angle_deg"])
    m = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ (np.diag([-1.0, 1.0]) if T["mirror"] else np.eye(2))
    src, dst = np.asarray(design, float), np.asarray(chip, float)
    return m, dst.mean(0) - m @ src.mean(0)


def identified_flakes(spec: DeviceSpec, st: FlowStep) -> list[str]:
    """Flakes of the step found on the chip whose outline is still the traced border moved (bottom → top)."""
    out = []
    for layer in spec.layers:
        info, polys = st.located.get(layer.uid) or {}, st.outlines.get(layer.uid)
        if info.get("how") in IDENTIFIED_HOW and layer.outline and polys and len(polys[0]) == len(layer.outline):
            out.append(layer.uid)
    return out


def _image_after(image: dict, m: np.ndarray, t: np.ndarray) -> dict:
    """``image`` placement (px → design µm, ``canvas.image_transform``) followed by x ↦ m·x + t."""
    s, a = float(image.get("um_per_px") or 0.0), math.radians(float(image.get("angle_deg", 0.0)))
    mx = -1.0 if image.get("mirror") else 1.0
    c, sn = math.cos(a) * s, math.sin(a) * s
    M = m @ np.array([[c * mx, sn], [sn * mx, -c]])
    t2 = m @ np.array([float(image.get("x", 0.0)), float(image.get("y", 0.0))]) + t
    sn2, c2 = M[0, 1], -M[1, 1]
    mirror = (M[0, 0] * c2 + M[1, 0] * sn2) < 0
    out = dict(image)
    out.update({"angle_deg": math.degrees(math.atan2(sn2, c2)), "x": float(t2[0]), "y": float(t2[1]), "mirror": bool(mirror), "placed": True})
    return out


def orientation_from_chip(spec: DeviceSpec, st: FlowStep, key: str = "") -> dict | None:
    """The flakes found on the chip at this step → their relative positions and orientations in the
    design frame (Flake layout, the Structure tab's 3D stack).

    Needs two or more identified flakes. The reference is the lowest identified flake already placed
    in Flake layout (else the lowest identified one): it stays where it is; every other identified
    flake is moved and turned (rigidly — its traced shape and size are kept) so that it sits relative
    to the reference as on the chip; its image follows and counts as placed. Flakes not identified
    keep their positions relative to the reference. Returns the record also stored in
    ``spec.stage2["orientation"]`` (None when fewer than two flakes are identified)."""
    uids = identified_flakes(spec, st)
    if len(uids) < 2:
        return None
    placed = [u for u in uids if (spec.layer(u).image or {}).get("placed") or not spec.layer(u).image]
    ref = (placed or uids)[0]
    m0, t0 = _rigid(spec.layer(ref).outline, st.outlines[ref][0])
    inv = np.linalg.inv(m0)
    layers = {}
    for uid in uids:
        layer = spec.layer(uid)
        mi, ti = _rigid(layer.outline, st.outlines[uid][0])
        m, t = inv @ mi, inv @ (ti - t0)                # design → design: T0⁻¹ ∘ Ti
        turned = math.degrees(math.atan2(m[1, 0], m[0, 0]))
        mirrored = bool(np.linalg.det(m) < 0)
        pts = np.asarray(layer.outline, float)
        new = pts @ m.T + t
        moved = float(np.linalg.norm(new.mean(0) - pts.mean(0)))
        changed = uid != ref and (moved > 1e-3 or abs(turned) > 1e-3 or mirrored)
        if changed:
            layer.outline = [[float(x), float(y)] for x, y in new]
            if layer.image and layer.image.get("um_per_px"):
                layer.image = _image_after(layer.image, m, t)
        layers[uid] = {"turned_deg": 0.0 if uid == ref else turned, "moved_um": 0.0 if uid == ref else moved,
                       "mirrored": mirrored and uid != ref, "changed": changed}
    record = {"step": key, "reference": ref, "layers": layers}
    spec.stage2["orientation"] = record
    return record


# ====================================================================== generate
def params_for(st: FlowStep, config: LabConfig) -> fg.FingerParams:
    return fg.FingerParams.from_config(config, st.overrides)


def _array_of(flow: Flow, key: str | None = None) -> fg.FingerArray | None:
    """The finger array; at step ``key`` where it was found written (Locate the fingers), moved there."""
    st = flow.steps.get("hall_bars")
    if st is None or not st.array:
        return None
    arr = fg.FingerArray.from_json(st.array)
    off = pattern_offset(flow, key, "hall_bars") if key else None
    if off:
        (cx, cy), = move_points([arr.center], off)
        arr = fg.FingerArray((float(cx), float(cy)), arr.angle_deg + off["angle"], arr.n, arr.params)
    return arr


# ====================================================================== patterns as written (found at a later step)
#: Which earlier pattern a step can look for in its images (user, 2026-11-20): the fingers at the contact-gate step, the
#: contact gate at the electrodes step.
LOCATABLE = {"contact_gate": ("hall_bars",), "electrodes": ("contact_gate",)}
PATTERN_TITLES = {"hall_bars": "fingers", "contact_gate": "contact gate"}


def move_points(points, off: dict) -> list[tuple[float, float]]:
    """Points moved as a written pattern was found: turned ``angle`` about (cx, cy), then shifted (dx, dy)."""
    a = math.radians(off.get("angle", 0.0))
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = off.get("cx", 0.0), off.get("cy", 0.0)
    return [(cx + (x - cx) * ca - (y - cy) * sa + off.get("dx", 0.0), cy + (x - cx) * sa + (y - cy) * ca + off.get("dy", 0.0))
            for x, y in points]


def pattern_offset(flow: Flow, key: str | None, source: str) -> dict | None:
    """Where step ``source``'s pattern was found written, as seen by step ``key``: the latest step from ``key`` back (process
    order, after ``source``) that located it; None = as designed."""
    order = flow.order
    if key not in order or source not in order:
        return None
    i, j = order.index(key), order.index(source)
    for k in reversed(order[j + 1:i + 1]):
        s_ = flow.steps.get(k)
        if s_ is not None and s_.found_patterns.get(source):
            r = s_.found_patterns[source]
            return None if r.get("as_designed") else r     # "Use as designed": the design is where it is (user, 2026-11-20)
    return None


def pattern_shapes(flow: Flow, key: str, source: str) -> dict[str, list[Poly]]:
    """Step ``source``'s shapes as step ``key`` sees them: where they were found written, else as designed."""
    st = flow.steps.get(source)
    if st is None:
        return {}
    off = pattern_offset(flow, key, source)
    if not off:
        return st.shapes
    return {purpose: [move_points(q, off) for q in polys] for purpose, polys in st.shapes.items()}


def locate_written(flow: Flow, key: str, source: str, images: list) -> dict:
    """Find step ``source``'s written pattern in step ``key``'s aligned ``images`` [(objective, image, Similarity)] —
    the finest image that holds all of it — and keep where it lies (``FlowStep.found_patterns``). Raises ValueError."""
    from . import registration
    src = flow.steps.get(source)
    polys = [q for polys in (src.shapes.values() if src else []) for q in polys]
    if not polys:
        raise ValueError(f"The {PATTERN_TITLES.get(source, source)} are not generated (or drawn) yet.")
    xs = [q[0] for p in polys for q in p]
    ys = [q[1] for p in polys for q in p]
    box = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    errors = []
    for obj, img, sim in sorted(images, key=lambda t: t[2].scale):
        h, w = img.shape[:2]
        frame = [tuple(map(float, q)) for q in sim.apply([(0, 0), (w, 0), (w, h), (0, h)])]
        if not all(geo.point_in_poly(x, y, frame) for x, y in box):
            continue
        try:
            r = registration.locate_pattern(img, sim, polys)
        except ValueError as exc:
            errors.append(f"{obj}: {exc}")
            continue
        r["image"] = obj
        flow.step(key).found_patterns[source] = r
        return r
    raise ValueError("; ".join(errors) or "No aligned image of this step holds the whole pattern.")


def generate(spec: DeviceSpec, flow: Flow, key: str, config: LabConfig, bondpads: BondPads | None,
             group: str | None = None, run_check: bool = True) -> FlowStep:
    """Draw step ``key``'s pattern into its :class:`FlowStep` (shapes, notes, issues). Raises
    ValueError with a message when it cannot (nothing is changed to make it fit). For the electrodes,
    ``group`` (an :data:`ELECTRODE_GROUPS` key) draws that group only, keeping the others; None draws all."""
    st = flow.step(key)
    if key != "electrodes":
        st.notes, st.routes = [], []
    p = params_for(st, config)
    if key == "hall_bars":
        region = st.twisted or twisted_region(spec, st.outlines)
        if not region:
            raise ValueError("No twisted region at this step: locate the flakes (both device-layer halves) or draw the region.")
        bbn = spec.by_role("bottom_hbn")
        bbn_polys = st.outlines.get(bbn.uid) if bbn is not None else None
        if bbn_polys and p.bbn_margin is None:
            raise MissingValues([MissingItem("preset:fingers.bbn_margin_um", "Finger ends inside the bottom hBN by", "µm",
                                             hint="Each finger column runs as far as the bottom hBN allows, ending this far inside "
                                                  "its boundary (saved as a preset, editable in Preset dimensions).")])
        if not st.twisted:
            st.twisted, st.twisted_how = region, "overlap of the device-layer outlines"
        avoid = defect_region(flow, key, config)
        res = fg.fit_finger_array(region, p, st.fixed_angle, st.max_pairs)
        avoid_notes = []
        if avoid is not None:
            # bubbles / cracks (user, 2026-11-20): the array off as many as it can while as many finger pairs fit
            reg0 = geo.regions(region).merged()

            def fit_without(r):
                clean = geo.to_polys((reg0 - r).merged())
                try:
                    return fg.fit_finger_array(clean, p, st.fixed_angle, st.max_pairs) if clean else None
                except ValueError:
                    return None
            near = [(d, r) for d, r in defect_pieces(flow, key, config) if not (r & reg0).is_empty()]
            if near:
                best = fit_without(avoid)
                skipped = []
                if best is None or best.array.n < res.array.n:   # one at a time
                    kept, best = kdb.Region(), None
                    for d, r in near:
                        trial = fit_without((kept + r).merged())
                        if trial is not None and trial.array.n >= res.array.n:
                            kept, best = (kept + r).merged(), trial
                        else:
                            skipped.append(d)
                if best is not None:
                    res = best
                avoid_notes.append("The finger array keeps off the bubbles and cracks on the twisted region" + (
                    f", but for {len(skipped)} (off them, fewer finger pairs fit: {_kinds_text(skipped)}; see the canvas)."
                    if skipped else "."))
        arr = res.array
        length_notes = []
        if bbn_polys:
            # each finger as long as it can be on its own (user, 2026-11-20: they need not be the same length)
            lengths, length_notes = fg.fit_finger_lengths(arr, bbn_polys, p.bbn_margin)
            per = dict(lengths["fingers"])
            if avoid is not None:                      # each finger ends before a bubble / crack if that still leaves room
                clean, _n = fg.fit_finger_lengths(arr, bbn_polys, p.bbn_margin, avoid=avoid)
                shortened, kept = [], []
                for k, L in per.items():
                    Lc = clean["fingers"].get(k)
                    if Lc is not None and Lc < L - 1e-9:
                        if Lc >= p.land_from_tip + p.width:
                            per[k] = Lc
                            shortened.append(k)
                        else:
                            kept.append(k)
                if shortened:
                    avoid_notes.append(f"{len(shortened)} finger(s) end before a bubble / crack ({_finger_names(shortened)}).")
                if kept:
                    avoid_notes.append(f"{len(kept)} finger(s) cross a bubble / crack: ending before it leaves no room for the "
                                       f"electrodes, so they run on ({_finger_names(kept)}).")
            arr.params.lengths = per
            col_max = {c: max((v for k, v in per.items() if k.startswith(c + ":")), default=None) for c in ("top", "bottom")}
            arr.params.length_top, arr.params.length_bottom = col_max["top"], col_max["bottom"]
            short = [k for k, v in per.items() if v < p.land_from_tip]
            if short:
                length_notes.append(f"{len(short)} finger(s) shorter than where the electrodes land ({p.land_from_tip:g} µm from "
                                    f"the tip): they land on the far ends ({_finger_names(short)}).")
            rng = {c: [v for k, v in per.items() if k.startswith(c + ":")] for c in ("top", "bottom")}
            how = ("each finger as long as the bottom hBN allows, ending " + f"{p.bbn_margin:g} µm inside it: "
                   + ", ".join(f"{c} {min(v):.2f}–{max(v):.2f} µm" if v and max(v) - min(v) > 1e-6 else f"{c} {v[0]:.2f} µm"
                               for c, v in rng.items() if v))
        else:
            how = f"{p.length:g} µm long (preset; locate the bottom hBN to make each column as long as it allows)"
        st.array = arr.to_json()
        fs = arr.fingers()
        st.shapes = {"prepattern_contacts": [f.polygon for f in fs]}
        st.notes = [f"{arr.n} finger pairs fit on the twisted region ({2 * arr.n} fingers), array axis {arr.angle_deg:.2f}°, "
                    f"centre ({arr.center[0]:.2f}, {arr.center[1]:.2f}) µm, smallest margin {res.margin_um:.2f} µm.",
                    f"Finger {p.width:g} µm wide, pitch {p.pitch:g} µm, gap {p.gap:g} µm; straight {p.straight:g} µm, then fanned "
                    f"at the angle that gives {p.far_pitch:g} µm pitch at {p.length:g} µm (a longer finger keeps spreading); "
                    f"{how}."] + res.notes + length_notes + avoid_notes
    else:
        arr = _array_of(flow, key)
        if arr is None and key == "contact_gate":
            raise ValueError("Generate the Hall bars (finger array) step first: the other patterns are placed relative to it.")
        if arr is not None:
            fitted = {a: getattr(arr.params, a) for a in fg.FingerParams.FITTED}
            arr.params = p
            # keep the finger geometry of the Hall-bar step; this step's own parameters change only its own shapes
            hb = params_for(flow.step("hall_bars"), config)
            for attr in ("width", "pitch", "gap", "straight", "length", "far_pitch", "far_width"):
                setattr(arr.params, attr, getattr(hb, attr))
            for attr, v in fitted.items():                 # the finger lengths fitted to the bottom hBN
                setattr(arr.params, attr, v)
        if key == "contact_gate":
            hbn = hbn_under(spec, key)
            hbn_polys = st.outlines.get(hbn.uid) if hbn is not None else None
            if hbn_polys and p.cg_hbn_margin is None:
                raise MissingValues([MissingItem("preset:cg.hbn_margin_um", "Contact gate inside its hBN by", "µm",
                                                 hint="The whole contact gate stays this far inside the hBN it is written on (saved as a "
                                                      "preset, editable in Preset dimensions).")])
            tmd_polys = tmd_outlines(spec, st)
            if tmd_polys and p.cg_tmd_gap is None:
                raise MissingValues([MissingItem("preset:cg.bus_tmd_gap_um", "Contact-gate bus from the TMD by", "µm",
                                                 hint="The bus joining the teeth runs this far from the TMD (both device layers); each "
                                                      "column's teeth are as short as that allows (saved as a preset, editable in Preset "
                                                      "dimensions).")])
            keepout = fg.tmd_keepout(tmd_polys, p.cg_tmd_gap)
            leave_sd = not has_source_drain(spec)
            avoid_note = []
            wmin = config.value("ebl.contact_width_min_um")

            def fits(av):
                try:
                    return fg.fit_contact_gate(arr, hbn_polys, p.cg_hbn_margin, leave_sd, min_width=wmin, keepout=keepout, avoid=av)
                except ValueError:
                    return None
            pieces = defect_pieces(flow, key, config)
            avoid = None
            if pieces:                                 # bubbles / cracks (user, 2026-11-20): the comb keeps off as many as it can
                every = kdb.Region()
                for _d, r in pieces:
                    every = (every + r).merged()
                skipped = []
                if fits(every) is not None:
                    avoid = every
                else:                                  # one at a time: each kept if a comb still fits round it
                    avoid = kdb.Region()
                    for d, r in pieces:
                        trial = (avoid + r).merged()
                        if fits(trial) is not None:
                            avoid = trial
                        else:
                            skipped.append(d)
                avoid_note = ["The comb and its bus leads keep off the bubbles and cracks" + (
                    f", but for {len(skipped)} it cannot fit round ({_kinds_text(skipped)}; see the canvas)." if skipped else ".")]
            layout, cg_notes = fg.fit_contact_gate(arr, hbn_polys, p.cg_hbn_margin, leave_sd, min_width=wmin, keepout=keepout,
                                                   avoid=avoid)
            cg_notes = cg_notes + avoid_note
            st.comb = layout
            comb = fg.contact_gate_comb(arr, skip=layout["skip"], dirs=layout["dirs"], extension=layout["extension"],
                                        n_skip=layout.get("n_skip", 1), tooth=layout.get("tooth"))
            comb = fg.apply_trim(arr, comb, layout, hbn_polys, avoid)
            # each bus led on, slightly turned, to the far corner of its hBN (opposite corners for the two columns): a stretch
            # of bus outside where the top hBN goes, for the electrodes (user, 2026-11-20)
            spacing = config.value("ebl.contact_spacing_min_um") or 0.0
            lead_keepout = keepout if avoid is None or avoid.is_empty() else ((keepout + avoid).merged() if keepout is not None
                                                                                else avoid)
            comb, reach = fg.extend_buses(arr, comb, layout, hbn_polys, p.cg_hbn_margin, lead_keepout, float(spacing),
                                          min_width=config.value("ebl.contact_width_min_um"))
            layout["reach"] = {c: list(q[:2]) for c, q in reach.items()}
            layout["lead_width"] = {c: q[2] for c, q in reach.items()}
            st.comb = layout
            st.shapes = {"contact_gates": [c["polygon"] for c in comb.values()]}
            sd = ""
            if leave_sd:
                k = layout.get("n_skip", 1)
                sd = (" No source / drain in the stack: the " + " and the ".join(
                    f"{'left' if (layout['skip'][c] == 'first') else 'right'}-most {k if k > 1 else ''} {c} finger{'s' if k > 1 else ''}".replace("  ", " ")
                    for c in ("top", "bottom")) + " are left ungated (source and drain).")
            teeth = ", ".join(f"{c} {v['tooth']:.2f} µm" for c, v in comb.items())
            how_long = (f"teeth {teeth} long from the tips (as short as puts the bus {p.cg_tmd_gap:g} µm from the TMD)" if keepout is not None
                        else f"teeth {p.cg_tooth:g} µm long from the tips (preset; locate the TMD to bring the bus close to it)")
            long_teeth = [c for c, v in comb.items() if v["tooth"] > arr.params.straight_of(c) + 1e-9]
            if long_teeth:
                cg_notes = cg_notes + [f"The {' and '.join(long_teeth)} teeth reach past the straight part of the fingers "
                                       f"({arr.params.straight:g} µm): the TMD reaches far out there — check the comb on the fanned fingers."]
            if reach:
                tbn = spec.by_role("top_hbn") or next((l for l in spec.layers if l.function == "top_gate_dielectric"), None)
                tb = next((flow.steps[k].outlines.get(tbn.uid) for k in reversed(flow.order)
                           if tbn is not None and flow.steps.get(k) and flow.steps[k].outlines.get(tbn.uid)), None)
                under = [c for c, q in reach.items() if tb and any(geo.point_in_poly(q[0], q[1], poly) for poly in tb)]
                cg_notes = cg_notes + [
                    "Bus led on to the far corners of " + (hbn.label if hbn is not None else "its hBN") + ", "
                    + ", ".join(f"{c} to ({q[0]:.1f}, {q[1]:.1f})" + (f", {p.cg_bus_width:g} µm wide except where tight (down to {q[2]:.2f} µm)" if q[2] < p.cg_bus_width - 1e-6 else "")
                                for c, q in reach.items())
                    + f" ({p.cg_hbn_margin:g} µm inside it), so the electrodes can land outside the top hBN"
                    + (f"; the {' and '.join(under)} end lies under {tbn.label} as planned — move it" if under else "") + "."]
            st.notes = [f"Contact-gate comb: {sum(len(c['gated']) for c in comb.values())} teeth, {how_long}; "
                        f"bus {p.cg_bus_width:g} µm wide behind the teeth, leading out "
                        + ", ".join(f"{c} {'+u' if layout['dirs'][c] > 0 else '−u'} {layout['extension'][c]:g} µm" for c in comb)
                        + (f"; on {hbn.label}" + (f", ≥ {p.cg_hbn_margin:g} µm inside it" if hbn_polys else " (not located: not checked)")
                           if hbn is not None else "") + "." + sd] + cg_notes
        elif key == "electrodes":
            if bondpads is None:
                raise ValueError("Choose the bond-pad preset first (the electrodes end on its leads).")
            present = electrode_groups(spec)
            if group is not None and group not in present:
                raise ValueError(f"The stack has no layer for the {ELECTRODE_GROUPS[group][0].lower()}.")
            if not present:
                raise ValueError("Nothing to connect: the stack has no Hall-bar contacts, contact gate or gate.")
            landing = config.preset("electrodes.pad_landing_um")
            if landing is None:
                ends = bondpads.pad_ends()
                about = ""
                if ends:
                    wd = sorted(e["width"] for e in ends)[len(ends) // 2]
                    gap = min((math.dist(a["mid"], b["mid"]) for i, a in enumerate(ends) for b in ends[i + 1:]), default=0.0)
                    about = f" On '{bondpads.name}' the leads end {wd:.3g} µm wide, their ends {gap:.3g} µm apart at the closest."
                raise MissingValues([MissingItem("preset:electrodes.pad_landing_um", "Landing square on the bond-pad lead", "µm",
                                                 hint="Each electrode ends in a square this wide, centred on the end of its "
                                                      "bond-pad lead (half of it over the lead) for a good contact; saved as a "
                                                      "preset, editable in Preset dimensions." + about)])
            if debris_at(flow, key) and config.value("ebl.defect_clearance_um") is None:
                config.require("ebl.defect_clearance_um")      # how far from bubbles, cracks, stray flakes: asked
            if group is None:                          # all: start clean
                st.shapes, st.routes, st.group_notes = {}, [], {}
            groups = [group] if group else list(present)
            st.group_notes.pop("_router", None)
            for g in groups:
                st.group_notes.pop(g, None)
            # the gate contacts first (they need a free stretch of uncovered flake), then every lead routed round the
            # others, the gate flakes and the device (user, 2026-11-20: no crossing, no leakage)
            _route_electrodes(spec, flow, st, groups, arr, p, config, bondpads, float(landing))
            st.notes = ([f"Electrodes routed to the bond-pad leads without crossing each other, the gate flakes or the device "
                         f"(straight pieces, bent only to go round); each ends in a {float(landing):g} µm square on its lead."]
                        + st.group_notes.get("_router", [])
                        + [n for g in ELECTRODE_GROUPS for n in st.group_notes.get(g, [])])
            if group is None and not any(st.shapes.get(ELECTRODE_GROUPS[g][1]) for g in present):
                raise ValueError("\n".join(n for g in present for n in st.group_notes.get(g, [])))
    st.exported = None
    if run_check:                                      # (the auto design checks its finalists itself)
        check(spec, flow, key, config)
    return st


def gate_surroundings(spec: DeviceSpec, flow: Flow, st: FlowStep, g: str, config: LabConfig):
    """For a gate electrode: (gate layer, its outline region, the part of it no insulating flake above covers,
    the conductors it must keep away from, the covering flakes not located). The stack decides: a flake above
    the gate covers it when it is on the chip at the electrodes step; conducting flakes above it, the fingers,
    the contact gate, the twisted region and the other electrode groups are kept clear of."""
    gate = next((l for l in spec.layers if l.function == ELECTRODE_GROUPS[g][2]), None)
    if gate is None:
        raise ValueError(f"The stack has no {ELECTRODE_GROUPS[g][2].replace('_', ' ')} layer.")
    outline = st.outlines.get(gate.uid)
    if not outline:
        raise ValueError(f"{gate.label} is not located at this step: locate it (or draw its outline) first.")
    present = set(present_layers(spec, "electrodes"))
    cover, conductors, unlocated = kdb.Region(), kdb.Region(), []
    for l in spec.layers[spec.layers.index(gate) + 1:]:
        mat = config.material(l.material) if l.material else None
        found_here = bool(st.outlines.get(l.uid)) and (st.located.get(l.uid) or {}).get("how") not in NOT_ON_CHIP_HOW
        if (l.uid not in present and not found_here) or mat is None or mat.category in ("metal", "substrate"):
            continue                                   # patterned metal comes from the steps' shapes
        out = st.outlines.get(l.uid)
        if not out:
            unlocated.append(l.label)
            continue
        (conductors if mat.conducting == "yes" else cover).insert(geo.regions(out))
    keep = conductors.merged()
    for k, other in flow.steps.items():
        for purpose, polys in (pattern_shapes(flow, "electrodes", k) if k != "electrodes" else other.shapes).items():
            if k == "electrodes" and purpose == ELECTRODE_GROUPS[g][1]:
                continue
            keep.insert(geo.regions(polys))
    twisted = st.twisted or twisted_region(spec, st.outlines)
    if twisted:
        keep.insert(geo.regions(twisted))
    for d in debris_at(flow, st.key):                               # bubbles, cracks, stray flakes: no contact on them
        if len(d.get("polygon") or []) >= 3:
            dclr = config.value("ebl.defect_clearance_um") or 0.0
            keep.insert(geo.sized(geo.region([tuple(q) for q in d["polygon"]]), float(dclr)))
    gate_reg = geo.regions(outline).merged()
    return gate, gate_reg, gate_reg - cover.merged(), keep.merged(), unlocated


def _nearest_in(polys: list[Poly], pt) -> tuple[float, float] | None:
    best, bd = None, float("inf")
    for poly in polys:
        if geo.point_in_poly(pt[0], pt[1], poly):
            return float(pt[0]), float(pt[1])
        for a, b in zip(poly, poly[1:] + poly[:1]):
            ax, ay, bx, by = a[0], a[1], b[0], b[1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
            q = (ax + t * dx, ay + t * dy)
            d = math.dist(q, pt)
            if d < bd:
                best, bd = q, d
    return best


GATE_NET = {"top_gate": "TG", "bottom_gate": "BG"}
#: An accepted error place matches a later one within this distance (µm): a pattern moved more is checked again.
ACCEPT_TOL_UM = 0.05
#: A finger's electrode leaves along the finger for this many electrode widths before it is routed (program choice).
EXIT_STUB = 2.5
#: Extra length (µm) charged for a bond-pad lead whose landing square lies on a stray flake / bubble / crack (program choice).
PAD_ON_DEBRIS_COST = 2000.0
from .router import RELAX_NEAR_UM  # noqa: E402


def flakes_over_fingers(spec: DeviceSpec, st: FlowStep, config: LabConfig) -> tuple[list[tuple[str, list[Poly], bool]], list[str]]:
    """The flakes above the fingers' layer on the chip when the electrodes are written (top hBN, middle hBN, the device
    layers, a graphite top gate): ([(label, outline, conducting)], labels of those not located)."""
    fl = step_layer(spec, "hall_bars")
    if fl is None:
        return [], []
    present = set(present_layers(spec, "electrodes"))
    out, unlocated = [], []
    for l in spec.layers[spec.layers.index(fl) + 1:]:
        mat = config.material(l.material) if l.material else None
        if mat is None or mat.category in ("metal", "substrate"):
            continue                                   # patterned metal comes from the steps' shapes
        polys = st.outlines.get(l.uid) or []
        how = (st.located.get(l.uid) or {}).get("how")
        if l.uid not in present and not (polys and how not in NOT_ON_CHIP_HOW):
            continue                                   # not on the chip yet
        if not polys or how in NOT_ON_CHIP_HOW:
            unlocated.append(l.label)
            continue
        out.append((l.label, [list(map(tuple, q)) for q in polys], mat.conducting == "yes"))
    return out, unlocated


def finger_keepout(flakes, config: LabConfig, clearance: float) -> "kdb.Region":
    """The flakes above the fingers grown by the clearance (a conducting one by at least the contact-to-graphite rule)."""
    graphite = float(config.value("ebl.contact_to_graphite_gate_min_um") or 0.0)
    keep = kdb.Region()
    for _label, polys, conducting in flakes:
        d = max(clearance, graphite) if conducting else clearance
        keep.insert(geo.sized(geo.regions(polys).merged(), d) if d > 0 else geo.regions(polys))
    return keep.merged()


def finger_landings(spec: DeviceSpec, flow: Flow, st: FlowStep, arr: fg.FingerArray, p: fg.FingerParams,
                    config: LabConfig) -> tuple[dict[str, float], list[str]]:
    """Where each Hall-bar electrode starts on its finger (distance from the tip) (user, 2026-11-20: overlap the finger a
    bit, never reach onto the top hBN): at most ``electrodes.finger_overlap_um`` back from the finger's far end, never
    nearer the tip than ``electrodes.land_from_tip_um``, and only where the electrode's metal (its width, flat end) lies
    ``electrodes.flake_clearance_um`` outside every flake above the fingers (``flakes_over_fingers``). A finger with no
    such stretch gets an electrode on its last electrode width (reported; Check flags it)."""
    from .router import wire
    title = ELECTRODE_GROUPS["fingers"][0]
    flakes, unlocated = flakes_over_fingers(spec, st, config)
    clearance = config.preset("electrodes.flake_clearance_um")
    if flakes and clearance is None:
        raise MissingValues([MissingItem("preset:electrodes.flake_clearance_um", "Electrode clearance from the flakes above the "
                                         "fingers", "µm", hint="Each Hall-bar electrode starts on its finger only where its "
                                         "metal is this far outside " + ", ".join(sorted({f[0] for f in flakes})) + " (so no "
                                         "gold lands on the top hBN); saved as a preset, editable in Preset dimensions.")])
    overlap = config.preset("electrodes.finger_overlap_um")
    w = p.electrode_width
    keep = finger_keepout(flakes, config, float(clearance or 0.0)) if flakes else kdb.Region()
    land, blocked, lengths = {}, [], []

    def clear(f, d):
        return keep.is_empty() or (wire(fg.finger_path(arr, f, d), w) & keep).is_empty()

    for f in arr.fingers():
        L = p.length_at(f.column, f.index)
        last = max(L - w, 0.0)                         # the shortest electrode: one width on the finger's end
        d0 = min(max(p.land_from_tip, L - float(overlap) if overlap else p.land_from_tip), last)
        if clear(f, d0):
            d = d0
        else:                                          # walk out to where the metal clears the flakes, then bisect
            lo, d = d0, None
            x = d0
            while x < last:
                x = min(x + 0.5, last)
                if clear(f, x):
                    d = x
                    break
                lo = x
            if d is None:
                blocked.append(f.name)
                d = last
            else:
                while d - lo > 0.05:
                    m = (lo + d) / 2
                    lo, d = (lo, m) if clear(f, m) else (m, d)
        land[f.name] = round(d, 4)
        lengths.append(L - d)
    above = ", ".join(sorted({f[0] for f in flakes}))
    notes = [f"{title}: {len(land)}, {w:g} µm wide, each on its finger's last {min(lengths):.1f}–{max(lengths):.1f} µm "
             f"(at most {float(overlap):g} µm: electrodes.finger_overlap_um" if lengths and overlap else
             f"{title}: {len(land)}, {w:g} µm wide, landing {p.land_from_tip:g} µm from each finger tip"]
    if lengths and overlap:
        notes[0] += (f"; ≥ {float(clearance):g} µm outside {above}" if flakes else "") + "), then along it to its far end."
    else:
        notes[0] += (f", ≥ {float(clearance):g} µm outside {above}" if flakes else "") + ", then along the finger to its far end."
    if blocked:
        notes.append(f"{title}: finger(s) {', '.join(blocked)} — no stretch of the finger is clear of {above}: the electrode lands on its "
                     f"last {w:g} µm, on the flake (Check flags it).")
    if unlocated:
        notes.append(f"{title}: {', '.join(unlocated)} not located at this step — the electrodes may lie on "
                     f"{'it' if len(unlocated) == 1 else 'them'}; locate (or draw) {'it' if len(unlocated) == 1 else 'them'} and "
                     "Generate again.")
    return land, notes


def _group_terminals(spec: DeviceSpec, flow: Flow, st: FlowStep, g: str, arr: fg.FingerArray | None, p: fg.FingerParams,
                     config: LabConfig, mids: list, used: set) -> tuple[list[dict], list[str]]:
    """The electrodes one group needs, before routing: [{name, group, fixed (points of the part that is not routed: along
    the finger, or the contact), start, extra (metal of its own beyond the wire: a gate contact)}], notes."""
    title = ELECTRODE_GROUPS[g][0]
    if g in ("fingers", "contact_gate") and arr is None:
        raise ValueError("Generate the Hall bars (finger array) step first.")
    if g == "fingers":
        land, notes = finger_landings(spec, flow, st, arr, p, config)
        out = [{"name": t["name"], "group": g, "fixed": [tuple(q) for q in t["path"]], "start": tuple(t["path"][-1]), "extra": [],
                "direction": tuple(t["direction"])} for t in fg.finger_terminals(arr, land=land)]
        return out, notes
    if g == "contact_gate":
        cg = flow.steps.get("contact_gate")
        if cg is None or not cg.shapes.get("contact_gates"):
            raise ValueError("Generate (or draw) the contact gate first.")
        # on the contact gate as it is (generated, edited or drawn): the end of each bus lead
        cg_polys = pattern_shapes(flow, "electrodes", "contact_gate").get("contact_gates", cg.shapes["contact_gates"])
        overlap = float(config.preset("electrodes.cg_overlap_um") or 0.0)
        comb = geo.regions(cg_polys).merged()
        out = []
        for t in fg.cg_terminals(arr, cg_polys, p.electrode_width):
            start = tuple(t["start"])
            dx, dy = t["direction"]
            inner = (fg.back_along(comb, start, overlap - p.electrode_width / 2, (-dx, -dy))
                     if overlap > p.electrode_width / 2 else None)
            fixed = [inner, start] if inner is not None else [start]
            out.append({"name": t["name"], "group": g, "fixed": fixed, "start": start, "extra": []})
        return out, [f"{title}: {len(out)}, one at the end of each bus lead, running {overlap:g} µm back along it "
                     "(electrodes.cg_overlap_um)."]
    c, square, notes = gate_contact(spec, flow, st, g, p, config, mids, used)
    # while routing, the contact counts as the disk any turn of it fits in; it is turned along its wire afterwards
    r = p.contact_size / math.sqrt(2)
    disk = [(c[0] + r * math.cos(2 * math.pi * k / 16), c[1] + r * math.sin(2 * math.pi * k / 16)) for k in range(16)]
    return [{"name": GATE_NET[g], "group": g, "fixed": [c], "start": c, "extra": [disk], "contact": p.contact_size}], notes


def bondpads_for(flow: Flow) -> BondPads | None:
    """The flow's bond-pad preset (by name), or None."""
    from . import bondpads as _bp
    if not flow.bondpads:
        return None
    try:
        return _bp.load_preset(flow.bondpads)
    except KeyError:
        return None


def debris_inputs(spec: DeviceSpec, flow: Flow, key: str, bondpads: BondPads | None):
    """What finding bubbles, cracks and stray flakes needs at step ``key``: (known metal, the stack's outlines, the region
    to look in). Known metal = the bond pads and markers and every earlier step's pattern; the stack = every flake on the
    chip at the step (below its own layer, or found there); the region = inside the bond-pad leads' ends."""
    st = flow.step(key)
    metal: list[Poly] = []
    region = None
    if bondpads is not None:
        metal += bondpads.layer_polys("pads") + bondpads.layer_polys("markers", near=True)
        mids = np.array([e["mid"] for e in bondpads.pad_ends()], float)
        if len(mids) >= 3:
            import cv2
            hull = cv2.convexHull(mids.astype(np.float32)).reshape(-1, 2)
            region = [[tuple(map(float, q)) for q in hull]]
    order = flow.order
    for k in order[:order.index(key)] if key in order else []:
        other = flow.steps.get(k)
        if other is not None:
            metal += [q for polys in pattern_shapes(flow, key, k).values() for q in polys]
    present = set(present_layers(spec, key))
    stack = []
    for l in spec.layers[1:]:
        how = (st.located.get(l.uid) or {}).get("how")
        if st.outlines.get(l.uid) and (l.uid in present or how not in NOT_ON_CHIP_HOW):
            stack += [list(q) for q in st.outlines[l.uid]]
    return metal, stack, region


def set_debris(st: FlowStep, cands: list) -> None:
    """Found features replace the earlier automatic ones; those drawn or edited by hand are kept."""
    st.debris = [d for d in st.debris if d.get("how") != "auto"] + [c.to_json() for c in cands]
    st.debris_looked = True


#: Bubbles and cracks are what the Hall bars and the contact gate keep off (stray flakes and particles lie off the stack).
AVOID_KINDS = ("bubble", "crack")
#: Stray pieces that may conduct (graphite, metal dust): two electrodes over one would short (user, 2026-11-20).
CONDUCTING_DEBRIS = ("flake", "particle")


def defect_region(flow: Flow, key: str, config: LabConfig, kinds: tuple[str, ...] = AVOID_KINDS):
    """The bubbles / cracks holding at step ``key``, grown by ``ebl.defect_clearance_um`` (asked when there are any), or
    None when there are none."""
    items = [d for d in debris_at(flow, key) if d.get("kind") in kinds and len(d.get("polygon") or []) >= 3]
    if not items:
        return None
    clr = config.require("ebl.defect_clearance_um")[0]
    return geo.sized(geo.regions([[tuple(q) for q in d["polygon"]] for d in items]).merged(), float(clr)).merged()


def _finger_names(keys: list[str]) -> str:
    """'top 2, 3 · bottom 7' for finger keys 'top:1', 'top:2', 'bottom:6' (numbered from 1 along the array)."""
    by: dict[str, list[int]] = {}
    for k in keys:
        c, i = k.split(":")
        by.setdefault(c, []).append(int(i) + 1)
    return " · ".join(f"{c} {', '.join(map(str, sorted(v)))}" for c, v in by.items())


def defect_pieces(flow: Flow, key: str, config: LabConfig, kinds: tuple[str, ...] = AVOID_KINDS) -> list[tuple[dict, "kdb.Region"]]:
    """Each bubble / crack holding at step ``key`` with its region grown by the clearance (asked when there are any),
    largest first (kept first when they are added one at a time)."""
    items = [d for d in debris_at(flow, key) if d.get("kind") in kinds and len(d.get("polygon") or []) >= 3]
    if not items:
        return []
    clr = float(config.require("ebl.defect_clearance_um")[0])
    items.sort(key=lambda d: -abs(geo.poly_area([tuple(q) for q in d["polygon"]])))
    return [(d, geo.sized(geo.region([tuple(q) for q in d["polygon"]]), clr).merged()) for d in items]


def _kinds_text(items: list[dict]) -> str:
    from collections import Counter
    n = Counter(d.get("kind") for d in items)
    return " and ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in n.items())


def _defects_under(flow: Flow, key: str, polys: list[Poly], kinds: tuple[str, ...] = AVOID_KINDS) -> str:
    """'2 bubbles and 1 crack' under ``polys`` (or '')."""
    reg = geo.regions(polys).merged()
    from collections import Counter
    n = Counter(d["kind"] for d in debris_at(flow, key) if d.get("kind") in kinds and len(d.get("polygon") or []) >= 3
                and not (geo.region([tuple(q) for q in d["polygon"]]) & reg).is_empty())
    return " and ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in n.items())


def debris_source(flow: Flow, key: str) -> str | None:
    """The step whose bubbles / cracks / stray flakes hold at step ``key`` (user, 2026-11-20: those found at a later step
    replace the earlier ones): ``key`` itself once it has looked, else the latest earlier step (process order) that has."""
    st = flow.steps.get(key)
    if st is not None and st.debris_looked:
        return key
    order = flow.order
    for k in reversed(order[:order.index(key)] if key in order else []):
        s_ = flow.steps.get(k)
        if s_ is not None and s_.debris_looked:
            return k
    return None


def debris_at(flow: Flow, key: str) -> list[dict]:
    """The bubbles / cracks / stray flakes that hold at step ``key`` (its own, else the latest earlier step's)."""
    src = debris_source(flow, key)
    return flow.steps[src].debris if src else []


def landing_square(end: dict, side: float) -> Poly:
    """The square an electrode ends in: ``side`` µm, centred on the end of its bond-pad lead, along the lead."""
    (mx, my), (dx, dy) = end["mid"], end["dir"]
    h = side / 2
    return [(mx + a * dx * h - b * dy * h, my + a * dy * h + b * dx * h) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _route_obstacles(spec: DeviceSpec, flow: Flow, st: FlowStep, config: LabConfig, bondpads: BondPads, ends: list[dict],
                     owner_of_end: dict[int, str], groups: list[str], spacing: float, gclr: float):
    from .router import Obstacle
    obs: list[Obstacle] = []
    for k, other in flow.steps.items():                # the patterned metal: fingers, contact-gate combs (one piece each)
        if k == "electrodes":
            continue
        for polys in pattern_shapes(flow, "electrodes", k).values():
            obs += [Obstacle([q], True, spacing) for q in polys]
    for o, (_t, purpose, _f) in ELECTRODE_GROUPS.items():     # the electrodes of the groups kept
        if o not in groups:
            obs += [Obstacle([q], True, spacing) for q in st.shapes.get(purpose, [])]
    for i, e in enumerate(ends):                       # the bond-pad leads (its own one for each electrode)
        obs.append(Obstacle([e["poly"]], True, spacing, owner_of_end.get(i)))
    obs += [Obstacle([q], True, spacing) for q in bondpads.layer_polys("markers", near=True)]
    # conducting flakes (graphite gates …): not crossed where an insulating flake covers them (leakage), never touched where
    # nothing covers them (a short); the gate's own electrode may
    present = set(present_layers(spec, "electrodes"))
    gate_of = {ELECTRODE_GROUPS[g][2]: g for g in GATE_NET}
    def on_chip(uid):                                  # below the electrodes in the stack, or found on the chip at this step
        how = (st.located.get(uid) or {}).get("how")
        return bool(st.outlines.get(uid)) and (uid in present or how not in NOT_ON_CHIP_HOW)
    for i, l in enumerate(spec.layers):
        mat = config.material(l.material) if l.material else None
        out = st.outlines.get(l.uid)
        if not on_chip(l.uid) or mat is None or mat.conducting != "yes" or mat.category in ("metal", "substrate"):
            continue
        cover = kdb.Region()
        for above in spec.layers[i + 1:]:
            ma = config.material(above.material) if above.material else None
            if on_chip(above.uid) and ma is not None and ma.conducting != "yes":
                cover.insert(geo.regions(st.outlines[above.uid]))
        g = gate_of.get(l.function)
        owner = GATE_NET[g] if g in groups else None
        obs.append(Obstacle([list(q) for q in out], False, gclr, owner))
        exposed = geo.to_polys((geo.regions(out).merged() - cover.merged()).merged())
        if exposed:
            obs.append(Obstacle(exposed, True, gclr, owner))
    device = [q for q in (st.twisted or twisted_region(spec, st.outlines) or [])] + tmd_outlines(spec, st)
    if device:
        obs.append(Obstacle(device, False, spacing))
    # the insulating flakes above the fingers (top hBN, middle hBN): no gold on them where a lead can go round (user,
    # 2026-11-20); the leads that must start on them (the contact gate's) cross them at the soft cost
    fclr = float(config.preset("electrodes.flake_clearance_um") or 0.0)
    insul = [q for _label, polys, conducting in flakes_over_fingers(spec, st, config)[0] if not conducting for q in polys]
    if insul:
        obs.append(Obstacle(insul, False, fclr))
    # bubbles, cracks, stray flakes (user, 2026-11-20): no electrode lands on them
    dclr = config.value("ebl.defect_clearance_um")
    for d in debris_at(flow, st.key):
        if len(d.get("polygon") or []) >= 3:
            # a particle (dust, residue) is avoided where there is room; stray flakes, bubbles and cracks are not crossed
            # (but for a lead with no other way)
            hard = d.get("kind") != "particle"
            obs.append(Obstacle([[tuple(q) for q in d["polygon"]]], hard, float(dclr or 0.0), relax=hard, ownable=False,
                                exclusive=d.get("kind") in CONDUCTING_DEBRIS))   # a flake: one lead at most (two would short)
    return obs


def _route_electrodes(spec: DeviceSpec, flow: Flow, st: FlowStep, groups: list[str], arr: fg.FingerArray | None,
                      p: fg.FingerParams, config: LabConfig, bondpads: BondPads, landing: float) -> None:
    """Terminals of ``groups`` → bond-pad leads (order-preserving round the device), routed without crossings
    (:mod:`ustack.router`); the other groups' electrodes are kept and avoided."""
    from .router import Net, route_nets, wire
    from .stage2 import assign_tips
    ends = bondpads.pad_ends()
    mids = [e["mid"] for e in ends]
    used = {r["pad_tip"] for r in st.routes if r.get("group") not in groups and r.get("pad_tip") is not None}
    terms, errors = [], {}
    for g in sorted(groups, key=lambda g: g not in GATE_NET):          # the gate contacts first: they need free flake
        try:
            ts, notes = _group_terminals(spec, flow, st, g, arr, p, config, mids, used)
            terms += ts
            st.group_notes[g] = notes
        except ValueError as exc:
            if isinstance(exc, BlockedByElectrodes) or len(groups) == 1:
                raise
            errors[g] = str(exc)
            st.group_notes[g] = [f"{ELECTRODE_GROUPS[g][0]}: not drawn — {exc}"]
    free = [i for i in range(len(ends)) if i not in used]
    centre = arr.center if arr is not None else (float(np.mean([t["start"][0] for t in terms])) if terms else 0.0,
                                                  float(np.mean([t["start"][1] for t in terms])) if terms else 0.0)
    # a pad whose landing square would lie on a stray flake / bubble / crack is used only when there is no other
    blocked = kdb.Region()
    dclr = float(config.value("ebl.defect_clearance_um") or 0.0)
    for d in debris_at(flow, st.key):
        if d.get("kind") != "particle" and len(d.get("polygon") or []) >= 3:
            blocked.insert(geo.sized(geo.region([tuple(q) for q in d["polygon"]]), dclr))
    cost = [PAD_ON_DEBRIS_COST if not blocked.is_empty() and not (geo.region(landing_square(ends[i], landing)) & blocked).is_empty()
            else 0.0 for i in free]
    mapping = assign_tips([t["start"] for t in terms], [mids[i] for i in free], centre, cost) if terms else []
    owner_of_end, nets, extra = {}, [], {}
    w = p.electrode_width
    for t, m in zip(terms, mapping):
        t["pad_tip"] = free[m] if m is not None else None
        if t.get("direction"):                         # leave along the finger first (no acute notch at the joint; the
            dx, dy = t["direction"]                    # stub keeps the exit free of the neighbours' wires)
            t["fixed"] = list(t["fixed"]) + [(t["start"][0] + dx * EXIT_STUB * w, t["start"][1] + dy * EXIT_STUB * w)]
            t["start"] = t["fixed"][-1]
        own = [q for q in geo.to_polys(wire(t["fixed"], w))] if len(t["fixed"]) > 1 else [
            geo.rect(t["start"][0] - w / 2, t["start"][1] - w / 2, t["start"][0] + w / 2, t["start"][1] + w / 2)]
        if t["pad_tip"] is None:
            continue
        e = ends[t["pad_tip"]]
        owner_of_end[t["pad_tip"]] = t["name"]
        t["square"] = landing_square(e, landing)
        # the wire enters its landing square straight along the lead, from just outside it
        t["approach"] = (e["mid"][0] - e["dir"][0] * (landing / 2 + w), e["mid"][1] - e["dir"][1] * (landing / 2 + w))
        extra[t["name"]] = [t["square"]] + t["extra"]
        stub = [q for q in geo.to_polys(wire(t["fixed"], w))] if len(t["fixed"]) > 1 else []
        nets.append(Net(t["name"], t["start"], t["approach"], w, own + t["extra"], t["group"], stub))
    target = config.value("ebl.electrode_spacing_target_um")
    hard_min = config.value("ebl.contact_spacing_min_um")
    spacing = float(target or hard_min or 0.0)
    gclr = float(config.value("ebl.contact_to_graphite_gate_min_um") or spacing)
    paths, crossed, notes = {}, {}, []
    spacings = [spacing] + ([float(hard_min)] if hard_min and target and hard_min < target else [])
    relaxed: set[str] = set()
    used = spacing
    every = {n.name for n in nets}
    for sp in spacings:
        obs = _route_obstacles(spec, flow, st, config, bondpads, ends, owner_of_end, groups, sp, max(gclr, 0.0))
        radius = RELAX_NEAR_UM
        paths, crossed, notes = route_nets(nets, obs, sp, extra=extra, relax_nets=relaxed or None, relax_radius=radius)
        used = sp
        # last resort, only for the leads with no way round them: over a bubble / stray flake near where they start or end
        # (at a cost: as little as possible) rather than not at all; the set grows while leads still have no way, then
        # every lead may, then anywhere
        for _round in range(5):
            if not notes or not debris_at(flow, st.key) or (relaxed == every and radius == math.inf):
                break
            failed = {n for n, p_ in paths.items() if p_ is None}
            if failed <= relaxed:
                if relaxed == every:
                    radius = math.inf
                relaxed = every
            else:
                relaxed |= failed
            paths, crossed, notes = route_nets(nets, obs, sp, extra=extra, relax_nets=relaxed, relax_radius=radius)
        if not notes:
            break
    if used < spacing and not notes:
        st.group_notes.setdefault("_router", []).append(
            f"Spacing between electrodes relaxed to the hard minimum {hard_min:g} µm (the {spacing:g} µm target leaves no way through).")
    if relaxed and not notes:
        st.group_notes.setdefault("_router", []).append(
            f"{', '.join(sorted(relaxed))}: no way round the bubbles, cracks or stray flakes — routed over the fewest (see Check).")
    by_group: dict[str, list[Poly]] = {g: [] for g in groups}
    routes = []
    for t in terms:
        pts = paths.get(t["name"])
        full = list(t["fixed"]) + (list(pts[1:]) + [tuple(ends[t["pad_tip"]]["mid"])] if pts else [])
        reg = kdb.Region()
        if len(full) > 1:
            reg.insert(wire(full, w))                  # one wire, round joints (also where the routed part starts)
        if t.get("contact"):                           # the gate contact: a square turned along its wire
            c, h = t["start"], t["contact"] / 2
            nxt = full[1] if len(full) > 1 else (c[0] + 1.0, c[1])
            L = math.dist(c, nxt) or 1.0
            ux, uy = (nxt[0] - c[0]) / L, (nxt[1] - c[1]) / L
            reg.insert(geo.region([(c[0] + a * ux * h - b * uy * h, c[1] + a * uy * h + b * ux * h)
                                   for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
        else:
            for q in t["extra"]:
                reg.insert(geo.region(q))
        if pts:
            reg.insert(geo.region(t["square"]))
        by_group[t["group"]] += geo.to_polys(reg.merged())
        routes.append({"terminal": t["name"], "group": t["group"], "points_um": [[round(x, 4), round(y, 4)] for x, y in full],
                       "pad_tip": t.get("pad_tip") if pts else None, "routed_to_pad": bool(pts),
                       "soft_crossing_um": round(crossed.get(t["name"], 0.0), 2)})
    for g in groups:
        if g in errors:
            continue
        st.shapes[ELECTRODE_GROUPS[g][1]] = by_group[g]
        mine = [r for r in routes if r["group"] == g]
        bad = [r["terminal"] for r in mine if not r["routed_to_pad"]]
        over = [r["terminal"] for r in mine if r["soft_crossing_um"] > 0]
        st.group_notes[g] = st.group_notes.get(g, []) + (
            [f"{len(mine) - len(bad)} of {len(mine)} routed to a bond-pad lead without crossing another electrode."]
            + ([f"Not routed (no way through): {', '.join(bad)} — draw them (Draw electrodes…) or move the others."] if bad else [])
            + ([f"Crossing a covered gate flake or the device (no way around): {', '.join(over)}."] if over else []))
    st.routes = [r for r in st.routes if r.get("group") not in groups] + routes


def gate_contact(spec: DeviceSpec, flow: Flow, st: FlowStep, g: str, p: fg.FingerParams, config: LabConfig,
                 tips: list, used: set) -> tuple[tuple[float, float], Poly, list[str]]:
    """The contact of a gate's electrode: a square (``electrodes.bg_contact_um``) on the exposed part of the gate (no
    insulating flake above covers it), ≥ ``electrodes.gate_contact_clearance_um`` from everything it must not touch,
    as close as allowed to the nearest free bond-pad lead. Returns (centre, square, notes)."""
    gate, _gate_reg, exposed, keep, unlocated = gate_surroundings(spec, flow, st, g, config)
    s, clr, w = p.contact_size, p.contact_clearance, p.electrode_width
    allowed = exposed - (geo.sized(keep, clr) if not keep.is_empty() else kdb.Region())
    centres = geo.sized(allowed, -(s / math.sqrt(2) + 0.01)).merged()      # a square of any orientation fits round a centre here
    if centres.is_empty():
        msg = (f"No {s:g} µm contact fits on the exposed part of {gate.label} (the part no hBN above covers), "
               f"≥ {clr:g} µm from the device, the fingers, the contact gate and the other electrodes.")
        others = [o for o in ELECTRODE_GROUPS if o != g and st.shapes.get(ELECTRODE_GROUPS[o][1])]
        if others:                                     # would it fit without the other electrodes?
            own = kdb.Region()
            for o in others:
                own.insert(geo.regions(st.shapes[ELECTRODE_GROUPS[o][1]]))
            rest = geo.regions([q for k in flow.steps if k != "electrodes" for polys in pattern_shapes(flow, "electrodes", k).values()
                                for q in polys])
            twisted = st.twisted or twisted_region(spec, st.outlines)
            base = (keep - own.merged()) if not own.is_empty() else keep
            base = (base + rest + (geo.regions(twisted) if twisted else kdb.Region())).merged()
            free = exposed - (geo.sized(base, clr) if not base.is_empty() else kdb.Region())
            if not geo.sized(free, -(s / math.sqrt(2) + 0.01)).is_empty():
                names = ", ".join(ELECTRODE_GROUPS[o][0].lower() for o in others)
                raise BlockedByElectrodes(msg + f" The {names} cross it: draw this gate electrode first and those after it "
                                                 "(Generate all does), or move them.", others)
        raise ValueError(msg)
    free = [i for i in range(len(tips)) if i not in used]
    if not free:
        raise ValueError("No free bond-pad lead left.")
    cpolys = geo.to_polys(centres)
    d, i, c = min(((math.dist(q, tips[i]), i, q) for i in free for q in [_nearest_in(cpolys, tips[i])] if q is not None),
                  key=lambda t: t[0])
    tip = tuple(tips[i])
    ux, uy = (tip[0] - c[0]) / max(d, 1e-9), (tip[1] - c[1]) / max(d, 1e-9)
    h = s / 2
    square = [(c[0] + a * ux * h - b * uy * h, c[1] + a * uy * h + b * ux * h) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    name = GATE_NET[g]
    notes = [f"{ELECTRODE_GROUPS[g][0]}: a {s:g} µm contact on the exposed part of {gate.label}, ≥ {clr:g} µm from the device, "
             f"fingers, contact gate and other electrodes, near the bond-pad lead {d:.0f} µm away; its electrode is routed ({name})."]
    if unlocated:
        notes.append(f"Not located at this step, so not taken into account: {', '.join(unlocated)} — locate them so the contact "
                     "is checked against them.")
    return (float(c[0]), float(c[1])), square, notes


# ====================================================================== Flake layout and the Structure tab from Stage 2
#: Outlines that are not (yet) the flake on the chip: a plan, a projection through Flake layout, an earlier position.
NOT_ON_CHIP_HOW = ("planned", "projected", "expected")
#: Patterned metal layers drawn in the Structure tab from Stage 2: stack layer function → (step, purposes).
PATTERN_LAYERS = {"hall_bar_contacts": ("hall_bars", ("prepattern_contacts",)),
                  "contact_gate": ("contact_gate", ("contact_gates",)),
                  "leads": ("electrodes", ELECTRODE_PURPOSES)}
#: Display only (program choice): patterns are shown within the flakes' box grown by this fraction of its size + 5 µm
#: (the electrodes run hundreds of µm to the bond pads).
STRUCTURE_WINDOW = 0.3


def located_here(st: "FlowStep", uid: str) -> bool:
    """Flake ``uid`` has its outline on the chip at this step (found, drawn, edited, placed …; not a plan, a projection, an
    earlier position, or cleared to be looked for again)."""
    info = st.located.get(uid) or {}
    return bool(st.outlines.get(uid)) and info.get("how") not in NOT_ON_CHIP_HOW + (None,) and not info.get("relocate")


#: Set by hand at a step: kept there even when an earlier step has the flake.
HAND_HOW = ("drawn", "edited", "placed", "traced")


def reuse_earlier_locations(spec: DeviceSpec, flow: "Flow", key: str) -> list[str]:
    """Locate flakes at step ``key`` (user, 2026-11-20): a flake on the chip here that was already found at an earlier step
    (process order; there below that step's own layer, its outline the flake) takes that outline — buried under later
    flakes it may no longer be visible. Flakes set by hand at this step, or cleared here to be looked for again, are left
    alone. Returns the uids taken over (their located record says from which step)."""
    st = flow.step(key)
    order = flow.order
    earlier = order[:order.index(key)] if key in order else []
    out = []
    for uid in present_layers(spec, key):
        info = st.located.get(uid) or {}
        if info.get("relocate") or (st.outlines.get(uid) and info.get("how") in HAND_HOW and not info.get("inherited")):
            continue
        src = None
        for k in reversed(earlier):
            s_ = flow.steps.get(k)
            if s_ is not None and uid in present_layers(spec, k) and located_here(s_, uid):
                src = k
                break
        if src is None:
            continue
        s_ = flow.steps[src]
        st.outlines[uid] = [[tuple(map(float, q)) for q in poly] for poly in s_.outlines[uid]]
        rec = {k_: v for k_, v in (s_.located.get(uid) or {}).items() if k_ != "inherited"}
        st.located[uid] = dict(rec, inherited=src)
        st.plan.pop(uid, None)
        out.append(uid)
    return out


def on_chip_outlines(spec: DeviceSpec, flow: Flow) -> dict[str, tuple[str, list[Poly]]]:
    """Flakes already fabricated, as Stage 2 found them (user, 2026-11-20): uid → (step, chip outline) at the latest step
    (process order) where the layer is on the chip (below the step's own layer) and its outline is the flake there
    (located, edited, placed, drawn …), not a plan, a projection or an earlier position."""
    out: dict[str, tuple[str, list[Poly]]] = {}
    for k in flow.order:
        st = flow.steps.get(k)
        if st is None:
            continue
        present = set(present_layers(spec, k))
        for uid, polys in st.outlines.items():
            how = (st.located.get(uid) or {}).get("how")
            if polys and uid in present and how not in NOT_ON_CHIP_HOW and spec.layer(uid) is not None:
                out[uid] = (k, [[tuple(map(float, q)) for q in poly] for poly in polys])
    return out


def layout_frame(spec: DeviceSpec, fixed: dict[str, tuple[str, list[Poly]]]) -> tuple[np.ndarray, np.ndarray] | None:
    """The Flake-layout frame → chip map (rotation ± reflection, translation; scale 1) set by the fabricated flakes: the
    lowest one whose chip outline is its traced border moved (same points) keeps its Flake-layout place; else the lowest
    one with a traced border keeps its middle; else the lowest one sits at the origin."""
    order = [l for l in spec.layers[1:] if l.uid in fixed]
    if not order:
        return None
    for layer in order:
        polys = fixed[layer.uid][1]
        if layer.outline and len(polys) == 1 and len(polys[0]) == len(layer.outline) >= 3:
            return _rigid([tuple(q) for q in layer.outline], polys[0])
    for layer in order:
        if layer.outline:
            c, d = geo.centroid(fixed[layer.uid][1][0]), geo.centroid(layer.outline)
            return np.eye(2), np.array([c[0] - d[0], c[1] - d[1]])
    c = geo.centroid(fixed[order[0].uid][1][0])
    return np.eye(2), np.array(c, float)


def to_layout(frame: tuple[np.ndarray, np.ndarray], poly) -> Poly:
    m, t = frame
    pts = (np.asarray(poly, float) - t) @ np.linalg.inv(m).T
    return [(float(x), float(y)) for x, y in pts]


def layout_view(spec: DeviceSpec, flow: Flow) -> dict:
    """What Flake layout and the Structure tab show of Stage 2: {"fixed": {uid: [polys in the Flake-layout frame]},
    "step": {uid: step}, "frame": (m, t) | None}. Fabricated flakes are fixed there — shape and place from the chip —
    while the flakes still to come stay where the user puts them in Flake layout."""
    fixed = on_chip_outlines(spec, flow)
    frame = layout_frame(spec, fixed)
    if frame is None:
        return {"fixed": {}, "step": {}, "frame": None}
    return {"fixed": {u: [to_layout(frame, poly) for poly in polys] for u, (_k, polys) in fixed.items()},
            "step": {u: k for u, (k, _p) in fixed.items()}, "frame": frame}


def _follow_image(layer, old: Poly | None, new: Poly) -> None:
    """The layer's image follows its border from ``old`` to ``new``: the rigid part (same points: rotation ± reflection
    and shift; else the shift of the middle)."""
    if not (layer.image or {}).get("um_per_px") or not old:
        return
    if len(old) == len(new):
        m, t = _rigid([tuple(q) for q in old], [tuple(q) for q in new])
    else:
        a, b = geo.centroid(old), geo.centroid(new)
        m, t = np.eye(2), np.array([b[0] - a[0], b[1] - a[1]])
    layer.image = _image_after(layer.image, m, t)


def sync_layout_from_chip(spec: DeviceSpec, flow: Flow) -> list[str]:
    """Flakes found on the chip → their traced borders (Flake layout, Flake images, the Structure tab) take the shape and
    place they have on the chip, in the Flake-layout frame (user, 2026-11-20: overwrite with the identified shape). The
    image follows (it stays under the border). Returns the layers changed — the caller marks them as synced, so the
    change is not carried back onto the chip (it came from there)."""
    view = layout_view(spec, flow)
    changed = []
    for uid, polys in view["fixed"].items():
        layer = spec.layer(uid)
        new = max(polys, key=lambda q: abs(geo.poly_area(q)))
        old = layer.outline
        if old and len(old) == len(new) and np.allclose(np.asarray(old, float), np.asarray(new, float), atol=1e-6):
            continue
        _follow_image(layer, old, new)
        layer.outline = [[float(x), float(y)] for x, y in new]
        if layer.image is not None:
            layer.image.setdefault("trace", {})["how"] = "chip"
        changed.append(uid)
    return changed


def _plan_frame(spec: DeviceSpec, flow: Flow) -> tuple[np.ndarray, np.ndarray] | None:
    frame = layout_frame(spec, on_chip_outlines(spec, flow))
    if frame is None or np.linalg.det(frame[0]) < 0:      # a mirrored frame cannot be a plan (turn + shift)
        return None
    return frame


def plan_from_layout(spec: DeviceSpec, flow: Flow, uid: str, skip: "FlowStep | None" = None) -> bool:
    """A flake not fabricated yet, moved / turned in Flake layout → its plan at every step that has one (user, 2026-11-20:
    the two are linked): the planned outline = the Flake-layout border through the frame the fabricated flakes set."""
    layer = spec.layer(uid)
    frame = _plan_frame(spec, flow)
    if layer is None or not layer.outline or frame is None or uid in on_chip_outlines(spec, flow):
        return False
    m, t = frame
    c = np.asarray(geo.centroid(layer.outline), float)
    cx, cy = m @ c + t
    angle = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    done = False
    for st in flow.steps.values():
        if st is skip or uid not in st.plan:
            continue
        e = st.plan[uid]
        same = (abs(e["cx"] - cx) < 1e-9 and abs(e["cy"] - cy) < 1e-9
                and abs(math.remainder(e.get("angle", 0.0) - angle, 360.0)) < 1e-9)
        e.update({"cx": float(cx), "cy": float(cy), "angle": float(angle)})
        apply_plan(spec, st)
        done = done or not same
    return done


def layout_from_plan(spec: DeviceSpec, flow: Flow, st: "FlowStep", uid: str) -> bool:
    """A plan moved / turned on the EBL canvas → the flake's border in Flake layout (and the Structure tab) moves with it;
    the plans at the other steps follow. Shape kept (rigid)."""
    layer = spec.layer(uid)
    frame = _plan_frame(spec, flow)
    if layer is None or not layer.outline or frame is None or uid not in st.plan:
        return False
    new = to_layout(frame, planned_outline(layer.outline, st.plan[uid]))
    old = layer.outline
    if np.allclose(np.asarray(old, float), np.asarray(new, float), atol=1e-9):
        return False
    _follow_image(layer, old, new)
    layer.outline = [[float(x), float(y)] for x, y in new]
    if layer.image is not None:
        layer.image["placed"] = True
    # the plan here, rebuilt from the moved border, is the same outline: express it through the frame
    plan_from_layout(spec, flow, uid)
    return True


def layout_patterns(spec: DeviceSpec, flow: Flow, view: dict | None = None) -> dict[str, list[Poly]]:
    """The patterned metal from Stage 2 (fingers, contact gate, electrodes) in the Flake-layout frame, per stack layer, clipped
    to a window round the flakes (``STRUCTURE_WINDOW``): for the Structure tab and Flake layout's metal parts. Empty when
    nothing is on the chip (no frame)."""
    view = view or layout_view(spec, flow)
    frame = view["frame"]
    if frame is None:
        return {}
    flakes = [poly for polys in view["fixed"].values() for poly in polys]
    flakes += [[tuple(q) for q in l.outline] for l in spec.layers[1:]
               if l.uid not in view["fixed"] and l.outline and l.function not in PATTERN_LAYERS]
    pts = [q for poly in flakes for q in poly]
    if not pts:
        return {}
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    m = STRUCTURE_WINDOW * max(x1 - x0, y1 - y0) + 5.0
    window = geo.region(geo.rect(x0 - m, y0 - m, x1 + m, y1 + m))
    out: dict[str, list[Poly]] = {}
    for layer in spec.layers[1:]:
        if layer.function not in PATTERN_LAYERS:
            continue
        k, purposes = PATTERN_LAYERS[layer.function]
        st = flow.steps.get(k)
        polys = [to_layout(frame, q) for purpose in purposes for q in ((st.shapes.get(purpose) or []) if st else [])]
        if polys:
            out[layer.uid] = geo.to_polys(geo.regions(polys).merged() & window)
    return out


def not_yet_on_chip(spec: DeviceSpec, flow: Flow, key: str) -> dict[str, Poly]:
    """Flakes not on the chip at step ``key`` (above its own layer) with no outline there, at the place Flake layout gives
    them on the chip (through the frame the fabricated flakes set): for the EBL canvas's "planned" layers."""
    st = flow.steps.get(key)
    frame = layout_frame(spec, on_chip_outlines(spec, flow))
    if st is None or frame is None:
        return {}
    present = set(present_layers(spec, key))
    m, t = frame
    out = {}
    for layer in spec.layers[1:]:
        if layer.uid in present or st.outlines.get(layer.uid) or not layer.outline or layer.function in PATTERN_LAYERS:
            continue
        out[layer.uid] = [(float(x), float(y)) for x, y in np.asarray(layer.outline, float) @ m.T + t]
    return out


def structure_footprints(spec: DeviceSpec, flow: Flow) -> tuple[dict[str, list[Poly]], str] | None:
    """The 3D stack of the Structure tab once a flake is on the chip (user, 2026-11-19 / 20), in the Flake-layout frame:
    fabricated flakes as Stage 2 found them, the flakes still to come where Flake layout has them, and the patterned
    metal (fingers, contact gate, electrodes) as generated, clipped to a window round the flakes (not generated = nothing
    drawn). Centred on the twisted region (else the flakes). None while nothing is on the chip."""
    view = layout_view(spec, flow)
    frame = view["frame"]
    if frame is None:
        return None
    flakes: dict[str, list[Poly]] = dict(view["fixed"])
    for layer in spec.layers[1:]:
        if layer.uid not in flakes and layer.outline and layer.function not in PATTERN_LAYERS:
            flakes[layer.uid] = [[tuple(q) for q in layer.outline]]
    pts = [q for polys in flakes.values() for poly in polys for q in poly]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pats = layout_patterns(spec, flow, view)
    tw = next((flow.steps[k].twisted for k in reversed(flow.order) if flow.steps.get(k) and flow.steps[k].twisted), None)
    if tw:
        bx0, by0, bx1, by1 = geo.bbox(geo.regions([to_layout(frame, q) for q in tw]).merged())
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out: dict[str, list[Poly]] = {}
    patterns = []
    for layer in spec.layers[1:]:
        if layer.function in PATTERN_LAYERS:
            polys = pats.get(layer.uid) or []
            if not polys:
                out[layer.uid] = []                    # not designed yet: nothing drawn
                continue
            patterns.append(STEPS[PATTERN_LAYERS[layer.function][0]].name)
        elif layer.uid in flakes:
            polys = flakes[layer.uid]
        else:
            continue
        out[layer.uid] = [[(x - cx, y - cy) for x, y in poly] for poly in polys]
    on = [spec.layer(u).label for u in view["fixed"]]
    text = (f"On the chip (from Stage 2): {', '.join(on)}; the others where Flake layout has them"
            + (f"; patterns: {', '.join(patterns)}" if patterns else ""))
    return out, text


# ====================================================================== the rule report (Check)
def _row(rule, required, found, status, where=None, key=None, unit="", label=None, rule_id=None):
    return {"rule": rule, "required": required, "found": found, "status": status, "where": list(where or []), "key": key,
            "unit": unit, "label": label or rule, "rule_id": rule_id or key}


#: Check rows → the lab value behind them (a MissingItem key: "rule:<id>" / "preset:<key>"), for "Change rule" on an error.
ROW_RULES = (("Fingers inside the bottom hBN", "preset:fingers.bbn_margin_um"),
             ("Contact-gate bus off the TMD", "preset:cg.bus_tmd_gap_um"),
             ("Contact gate on ", "preset:cg.hbn_margin_um"),
             (" off the other conductors", "preset:electrodes.gate_contact_clearance_um"),
             ("Electrodes off bubbles, cracks and stray flakes", "rule:ebl.defect_clearance_um"),
             ("Electrodes land on their bond-pad leads", "preset:electrodes.pad_landing_um"))


def _limit_rows(purpose: str, reg, config: LabConfig) -> list[dict]:
    rows = []
    name = PURPOSE_NAMES.get(purpose, purpose.replace("_", " "))
    for rid, kind, sev in (("ebl.contact_width_min_um", "width", "violated"), ("ebl.electrode_width_target_um", "width", "warning"),
                           ("ebl.contact_spacing_min_um", "space", "violated"), ("ebl.electrode_spacing_target_um", "space", "warning")):
        rule = config.rules.get(rid)
        limit = config.value(rid)
        what = f"{name}: {'width' if kind == 'width' else 'spacing'} ≥ {'hard minimum' if sev == 'violated' else 'design target'}"
        if limit is None:
            rows.append(_row(what, "not set", "—", "missing", key=f"rule:{rid}", unit="µm", label=rule.description if rule else rid))
            continue
        pts = geo.width_violations(reg, limit) if kind == "width" else geo.space_violations(reg, limit)
        rows.append(_row(what, f"{limit:g} µm" + (f" ({rule.status})" if rule and rule.status not in ("approved",) else ""),
                         f"{len(pts)} place(s) below" if pts else "all above", sev if pts else "met", pts[:5], rule_id=f"rule:{rid}"))
    return rows


def rule_report(spec: DeviceSpec, flow: Flow, key: str, config: LabConfig) -> list[dict]:
    """Every rule the step's pattern is held to, with what is required, what was found and a status:
    "met", "violated" (an error), "warning" (below a design target), "missing" (a value nobody gave:
    ``key`` is its MissingItem key) or "unchecked" (an input such as an outline is not there)."""
    st = flow.step(key)
    rows: list[dict] = []
    if not st.shapes:
        return [_row("Pattern generated", "Generate first", "nothing drawn", "unchecked")]
    for purpose, polys in gds_shapes(st.shapes).items():          # the electrode groups together: one GDS layer
        if polys and not purpose.startswith("ref_"):
            rows += _limit_rows(purpose, geo.regions(polys).merged(), config)
    p = params_for(st, config)
    arr = _array_of(flow, key)
    hb = flow.steps.get("hall_bars")
    fingers = geo.regions(pattern_shapes(flow, key, "hall_bars").get("prepattern_contacts", []) if key != "hall_bars"
                          else hb.shapes.get("prepattern_contacts", [])).merged() if hb is not None else kdb.Region()
    if key == "hall_bars" and arr is not None:
        n_drawn = len(st.shapes.get("prepattern_contacts", []))
        rows.append(_row("Finger pairs on the twisted region", f"≥ 1 pair, each tip ≥ {p.min_overlap:g} µm on it",
                         f"{arr.n} pairs" if n_drawn == 2 * arr.n else f"{n_drawn} fingers (edited by hand; the fit had {arr.n} pairs)",
                         "met" if arr.n and n_drawn >= 2 else "violated"))
        bbn = spec.by_role("bottom_hbn")
        polys = st.outlines.get(bbn.uid) if bbn is not None else None
        if p.bbn_margin is None:
            rows.append(_row("Fingers inside the bottom hBN", "margin not set", "—", "missing", key="preset:fingers.bbn_margin_um", unit="µm",
                             label="Finger ends inside the bottom hBN by"))
        elif not polys:
            rows.append(_row("Fingers inside the bottom hBN", f"≥ {p.bbn_margin:g} µm inside", "bottom hBN not located", "unchecked"))
        else:
            out = fingers - geo.sized(geo.regions(polys).merged(), -p.bbn_margin)
            rows.append(_row("Fingers inside the bottom hBN", f"≥ {p.bbn_margin:g} µm inside",
                             "all inside" if out.is_empty() else f"{geo.area(out):.2f} µm² outside", "met" if out.is_empty() else "violated",
                             [geo.centroid(q) for q in geo.to_polys(out)][:5]))
        bg = next((l for l in spec.layers if l.function == "bottom_gate"), None)
        if bg is not None:
            if st.outlines.get(bg.uid):
                gap = geo.region(arr.channel()) - geo.regions(st.outlines[bg.uid]).merged()
                rows.append(_row("Back gate under the whole channel", "channel ⊂ back gate", "covered" if gap.is_empty() else "partly outside",
                                 "met" if gap.is_empty() else "warning", [arr.center] if not gap.is_empty() else []))
            else:
                rows.append(_row("Back gate under the whole channel", "channel ⊂ back gate", "back gate not located", "unchecked"))
    if key == "contact_gate" and arr is None:
        rows.append(_row("Contact gate against the fingers", "the Hall-bar fingers", "Hall bars not generated", "unchecked"))
    if key == "contact_gate" and arr is not None:
        # everything from the contact gate as it is now (generated, edited on the canvas or drawn by hand)
        cg_polys = st.shapes.get("contact_gates", [])
        cg = geo.regions(cg_polys).merged()
        gated = fg.gated_fingers(arr, cg_polys)
        ungated = kdb.Region()
        for f in arr.fingers():
            if f.index not in gated[f.column]:
                ungated.insert(geo.polygon(f.polygon))
        touch = cg & ungated
        rows.append(_row("Contact gate off the ungated (source / drain) fingers", "no overlap",
                         "clear" if touch.is_empty() else f"{geo.area(touch):.2f} µm² on them", "met" if touch.is_empty() else "violated",
                         [geo.centroid(q) for q in geo.to_polys(touch)][:5]))
        rows.append(_row("Gated fingers", "the fingers between the source / drain ones",
                         f"top {len(gated['top'])} of {arr.n}, bottom {len(gated['bottom'])} of {arr.n}",
                         "met" if gated["top"] or gated["bottom"] else "violated"))
        if not has_source_drain(spec):
            ends = {}
            for column in ("top", "bottom"):
                free = sorted(set(range(arr.n)) - gated[column])
                if free and free == list(range(len(free))):
                    ends[column] = ("first", len(free))
                elif free and free == list(range(arr.n - len(free), arr.n)):
                    ends[column] = ("last", len(free))
                else:
                    ends[column] = (None, len(free))
            ok = all(e[0] for e in ends.values()) and ends["top"][0] != ends["bottom"][0]
            found = ", ".join(f"{c}: {n} ungated at the {('left' if e == 'first' else 'right') if e else 'middle / none'}"
                              for c, (e, n) in ends.items())
            rows.append(_row("Source and drain fingers left ungated", "outer finger(s) of each column, at opposite ends", found,
                             "met" if ok else "violated"))
        on_gap = cg & geo.sized(geo.region(arr.channel()), -0.01)    # 10 nm: grid rounding of rotated shapes, not an overlap
        depth = 0.0
        if not on_gap.is_empty():
            loc = [q for poly in geo.to_polys(on_gap) for q in arr.to_local(poly)]
            depth = max(arr.params.gap / 2 - abs(v) for _, v in loc)
        rows.append(_row("Contact gate off the channel between the tips", "no overlap",
                         "clear" if on_gap.is_empty() else f"reaches {depth + 0.01:.2f} µm into the gap", "met" if on_gap.is_empty() else "violated",
                         [geo.centroid(q) for q in geo.to_polys(on_gap)][:5]))
        tmd = tmd_outlines(spec, st)
        rule = "Contact-gate bus off the TMD"
        if p.cg_tmd_gap is None:
            rows.append(_row(rule, "gap not set", "—", "missing", key="preset:cg.bus_tmd_gap_um", unit="µm",
                             label="Contact-gate bus from the TMD by"))
        elif not tmd:
            rows.append(_row(rule, f"≥ {p.cg_tmd_gap:g} µm from it", "TMD not located", "unchecked"))
        else:
            # the bus = the comb off the fingers and outside the channel (the teeth lie on the fingers, over the TMD on purpose)
            on_fingers = kdb.Region()
            for f in arr.fingers():
                on_fingers.insert(geo.polygon(f.polygon))
            bus = cg - on_fingers.merged() - geo.region(arr.channel())
            bus = geo.sized(geo.sized(bus, -0.01), 0.01).merged()     # hairlines along the finger edges (grid rounding) are no bus
            near = bus & geo.grown_round(geo.regions(tmd).merged(), p.cg_tmd_gap - 0.01)
            gap = geo.region_distance(bus, geo.regions(tmd).merged())
            found = (f"closer in {geo.area(near):.2f} µm²" if not near.is_empty()
                     else ("no bus" if bus.is_empty() else f"{gap:.2f} µm"))
            rows.append(_row(rule, f"≥ {p.cg_tmd_gap:g} µm from it (as close as that)", found, "met" if near.is_empty() else "warning",
                             [geo.centroid(q) for q in geo.to_polys(near)][:5]))
        hbn = hbn_under(spec, key)
        hp = st.outlines.get(hbn.uid) if hbn is not None else None
        name = f"Contact gate on {hbn.label}" if hbn is not None else "Contact gate on its hBN"
        if p.cg_hbn_margin is None:
            rows.append(_row(name, "margin not set", "—", "missing", key="preset:cg.hbn_margin_um", unit="µm",
                             label="Contact gate inside its hBN by"))
        elif not hp:
            rows.append(_row(name, f"≥ {p.cg_hbn_margin:g} µm inside", "hBN not located", "unchecked"))
        else:
            out = cg - geo.sized(geo.regions(hp).merged(), -(p.cg_hbn_margin - 0.01))     # 10 nm: grid rounding of the trimmed bus
            rows.append(_row(name, f"≥ {p.cg_hbn_margin:g} µm inside", "all inside" if out.is_empty() else f"{geo.area(out):.2f} µm² outside",
                             "met" if out.is_empty() else "violated", [geo.centroid(q) for q in geo.to_polys(out)][:5]))
    if key == "electrodes":
        for g in electrode_groups(spec):
            title, purpose, _fn = ELECTRODE_GROUPS[g]
            polys = st.shapes.get(purpose, [])
            rows.append(_row(f"{title} drawn", "generated (or drawn)", f"{len(polys)} shape(s)", "met" if polys else "violated"))
            if g not in ("top_gate", "bottom_gate") or not polys:
                continue
            try:
                gate, _gr, exposed, keep, unlocated = gate_surroundings(spec, flow, st, g, config)
            except ValueError as exc:
                rows.append(_row(f"{title} on the exposed gate", "touches the gate where no hBN covers it", str(exc), "unchecked"))
                continue
            mine = geo.regions(polys).merged()
            on = mine & exposed
            rows.append(_row(f"{title} on the exposed {gate.label}", "touches the gate where no hBN covers it",
                             f"{geo.area(on):.2f} µm² of contact" if not on.is_empty() else "not on the exposed gate",
                             "met" if not on.is_empty() else "violated", [] if not on.is_empty() else [geo.centroid(polys[0])]))
            touch = mine & keep
            near = (mine & geo.sized(keep, p.contact_clearance)) - touch if not keep.is_empty() else kdb.Region()
            rows.append(_row(f"{title} off the other conductors", f"no contact; ≥ {p.contact_clearance:g} µm wanted",
                             "touches " + f"{geo.area(touch):.2f} µm²" if not touch.is_empty() else
                             ("closer than wanted" if not near.is_empty() else "clear"),
                             "violated" if not touch.is_empty() else ("warning" if not near.is_empty() else "met"),
                             [geo.centroid(q) for q in geo.to_polys(touch if not touch.is_empty() else near)][:5]))
            if unlocated:
                rows.append(_row(f"{title}: flakes above {gate.label}", "located", "not located: " + ", ".join(unlocated), "unchecked"))
        # an electrode over a gate's uncovered flake touches it: a short to that gate
        for g in ("top_gate", "bottom_gate"):
            if g not in electrode_groups(spec):
                continue
            try:
                gate, _gr, exposed, _keep, _u = gate_surroundings(spec, flow, st, g, config)
            except ValueError:
                continue
            others = geo.regions([q for o, (_t, purpose, _f) in ELECTRODE_GROUPS.items() if o != g
                                  for q in st.shapes.get(purpose, [])]).merged()
            short = others & exposed
            rows.append(_row(f"Other electrodes off the exposed {gate.label}", "no contact (it would short to the gate)",
                             "clear" if short.is_empty() else f"{len(geo.to_polys(short))} crossing(s), {geo.area(short):.2f} µm²",
                             "met" if short.is_empty() else "violated", [geo.centroid(q) for q in geo.to_polys(short)][:5]))
    if key == "electrodes":
        # every electrode a separate piece of metal: none may touch another (user, 2026-11-20: crossings leak)
        pieces = [(g, q) for g, (_t, purpose, _f) in ELECTRODE_GROUPS.items() for q in st.shapes.get(purpose, [])]
        regs = [geo.region(q) for _g, q in pieces]
        hits = []
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                if not (regs[i] & regs[j]).is_empty():
                    hits.append(geo.centroid(geo.to_polys(regs[i] & regs[j])[0]))
        if pieces:
            rows.append(_row("Electrodes do not cross each other", "no two electrodes touch", "none touch" if not hits else
                             f"{len(hits)} crossing(s)", "met" if not hits else "violated", hits[:5]))
        # over a gate flake under an hBN: no short, but leakage through the hBN (routed round it when there is a way)
        for g in ("top_gate", "bottom_gate"):
            if g not in electrode_groups(spec):
                continue
            try:
                gate, gate_reg, exposed, _keep, _u = gate_surroundings(spec, flow, st, g, config)
            except ValueError:
                continue
            others = geo.regions([q for o, (_t, purpose, _f) in ELECTRODE_GROUPS.items() if o != g
                                  for q in st.shapes.get(purpose, [])]).merged()
            over = (others & gate_reg) - exposed
            rows.append(_row(f"Other electrodes off the covered {gate.label}", "not over it (leakage through the hBN)",
                             "clear" if over.is_empty() else f"{geo.area(over):.2f} µm² over it",
                             "met" if over.is_empty() else "warning", [geo.centroid(q) for q in geo.to_polys(over)][:5]))
        if debris_at(flow, st.key) and pieces:
            metal = geo.regions([q for _g, q in pieces]).merged()
            dclr = config.value("ebl.defect_clearance_um")
            on, near = [], []
            for d in debris_at(flow, st.key):
                r = geo.region([tuple(q) for q in d["polygon"]])
                if not (metal & r).is_empty():
                    on.append(geo.centroid(d["polygon"]))
                elif dclr and not (metal & geo.sized(r, float(dclr) - 0.01)).is_empty():
                    near.append(geo.centroid(d["polygon"]))
            shared = []                                # one electrode at most over each flake (two would short through it)
            for d in [d for d in debris_at(flow, st.key) if d.get("kind") in CONDUCTING_DEBRIS]:
                r = geo.region([tuple(q) for q in d["polygon"]])
                if sum(1 for _g, q in pieces if not (geo.region(q) & r).is_empty()) > 1:
                    shared.append(geo.centroid(d["polygon"]))
            rows.append(_row("No two electrodes over the same stray flake", "at most one each (it may conduct)",
                             "none shared" if not shared else f"{len(shared)} shared", "met" if not shared else "violated",
                             shared[:5]))
            rows.append(_row("Electrodes off bubbles, cracks and stray flakes",
                             f"≥ {dclr:g} µm away" if dclr else "not on them",
                             "clear" if not on and not near else (f"{len(on)} on" if on else "") + (", " if on and near else "")
                             + (f"{len(near)} closer than {dclr:g} µm" if near else ""),
                             "violated" if on else ("warning" if near else "met"), (on + near)[:5]))
        landing = config.preset("electrodes.pad_landing_um")
        if bondpads_for(flow) is not None and st.routes and landing:
            ends = bondpads_for(flow).pad_ends()
            metal = geo.regions([q for _g, q in pieces]).merged()
            thin = []
            for r in st.routes:
                i = r.get("pad_tip")
                if i is None or i >= len(ends):
                    continue
                e = ends[i]
                need = 0.9 * min(landing, e["width"]) * landing / 2
                if geo.area(metal & geo.region(e["poly"])) < need:
                    thin.append(tuple(e["mid"]))
            rows.append(_row("Electrodes land on their bond-pad leads", f"{landing:g} µm square over each lead's end",
                             "all" if not thin else f"{len(thin)} with too little overlap", "met" if not thin else "violated", thin[:5]))
    if key == "electrodes" and st.shapes.get(ELECTRODE_GROUPS["fingers"][1]):
        flakes, unlocated = flakes_over_fingers(spec, st, config)
        if flakes:
            clr = config.preset("electrodes.flake_clearance_um")
            metal = geo.regions(st.shapes[ELECTRODE_GROUPS["fingers"][1]]).merged()
            above = ", ".join(sorted({f[0] for f in flakes}))
            bare = finger_keepout(flakes, config, 0.0)
            on = [tuple(map(float, geo.centroid(q))) for q in geo.to_polys((metal & bare).merged())]
            near = [] if clr is None else [tuple(map(float, geo.centroid(q))) for q in geo.to_polys(
                ((metal & finger_keepout(flakes, config, max(float(clr) - 0.01, 0.0))) - bare).merged())]
            rows.append(_row(f"Hall-bar electrodes off {above}", f"≥ {float(clr):g} µm outside" if clr is not None else "not on them",
                             "clear" if not on and not near else ", ".join(t for t in (f"{len(on)} place(s) on" if on else "",
                                                                                         f"{len(near)} closer" if near else "") if t),
                             "violated" if on else "warning" if near else "met", (on + near)[:5],
                             rule_id="preset:electrodes.flake_clearance_um"))
        if unlocated:
            rows.append(_row("Flakes above the fingers located", "all", "not located: " + ", ".join(unlocated), "unchecked"))
    if st.routes:
        bad = [r for r in st.routes if not r.get("routed_to_pad")]
        rows.append(_row("Every electrode reaches a bond-pad lead", "all routed", f"{len(st.routes) - len(bad)} of {len(st.routes)}",
                         "met" if not bad else "violated", [tuple(r["points_um"][0]) for r in bad][:5]))
    if key in ("hall_bars", "contact_gate") and st.shapes:
        items = [d for d in debris_at(flow, key) if d.get("kind") in AVOID_KINDS and len(d.get("polygon") or []) >= 3]
        if items:
            metal = geo.regions([q for polys in st.shapes.values() for q in polys]).merged()
            clr = config.value("ebl.defect_clearance_um")
            on = [geo.centroid(d["polygon"]) for d in items if not (metal & geo.region([tuple(q) for q in d["polygon"]])).is_empty()]
            near = [geo.centroid(d["polygon"]) for d in items if clr and not (metal & geo.sized(
                geo.region([tuple(q) for q in d["polygon"]]), float(clr) - 0.01)).is_empty()]
            near = [q for q in near if q not in on]
            name = "Fingers" if key == "hall_bars" else "Contact gate"
            rows.append(_row(f"{name} off bubbles and cracks", f"≥ {clr:g} µm away, where they can be" if clr else "not on them",
                             "clear" if not on and not near else (f"{len(on)} on" if on else "")
                             + (", " if on and near else "") + (f"{len(near)} closer" if near else ""),
                             "warning" if on or near else "met", (on + near)[:5], rule_id="rule:ebl.defect_clearance_um"))
    for r in rows:
        if not r.get("rule_id"):
            r["rule_id"] = next((k for text, k in ROW_RULES if text in r["rule"]), None)
    return rows


def check(spec: DeviceSpec, flow: Flow, key: str, config: LabConfig) -> list[dict]:
    st = flow.step(key)
    others: dict[str, list[Poly]] = {}
    for k, s in flow.steps.items():
        if k != key:
            for purpose, polys in pattern_shapes(flow, key, k).items():         # where they were found written, if found
                others.setdefault(f"{purpose} ({STEPS[k].name if k in STEPS else k})", []).extend(polys)
    issues = fg.check_pattern(gds_shapes(st.shapes), config, others)
    if key == "hall_bars" and st.array:
        arr = fg.FingerArray.from_json(st.array)
        bg = next((l for l in spec.layers if l.function == "bottom_gate"), None)
        if bg is not None and st.outlines.get(bg.uid):
            gap = geo.region(arr.channel())
            if (gap - geo.regions(st.outlines[bg.uid]).merged()).is_empty():
                issues.append({"severity": "info", "code": "bg_under_gap", "where": [], "message": "The back gate covers the whole channel between the tips."})
            else:
                issues.append({"severity": "warning", "code": "bg_not_under_gap", "where": [arr.center],
                               "message": f"{bg.label} does not cover the whole channel between the finger tips."})
    if key in ("contact_gate", "electrodes"):
        for row in rule_report(spec, flow, key, config):
            if row["status"] == "violated" and not (": width" in row["rule"] or ": spacing" in row["rule"]):   # (those are check_pattern's)
                issues.append({"severity": "error", "code": "cg_rule", "where": row["where"], "rule": row.get("rule_id"),
                               "message": f"{row['rule']}: {row['found']} (required: {row['required']})."})
    for r in st.routes:
        if not r.get("routed_to_pad"):
            issues.append({"severity": "error", "code": "unrouted", "where": [tuple(r["points_um"][0])],
                           "message": f"Electrode {r['terminal']} has no bond-pad lead."})
    st.issues = _apply_accepted(st, issues)
    return st.issues


def _accepted_at(st: FlowStep, code: str, pt) -> bool:
    return any(a.get("code") == code and math.hypot(a["x"] - pt[0], a["y"] - pt[1]) < ACCEPT_TOL_UM for a in st.accepted)


def _apply_accepted(st: FlowStep, issues: list[dict]) -> list[dict]:
    """Places of an error the user accepted (the red circle's pop-up) are taken out of it; an error with none left becomes
    "accepted" (kept in the list, not an error: no red circle, export not stopped)."""
    if not st.accepted:
        return issues
    out = []
    for i in issues:
        where = [tuple(p) for p in (i.get("where") or [])]
        if i.get("severity") == "error" and where:
            left = [p for p in where if not _accepted_at(st, i["code"], p)]
            if len(left) < len(where):
                i = dict(i, where=left, accepted=len(where) - len(left))
                if not left:
                    i["severity"] = "accepted"
                    i["message"] = "Accepted: " + i["message"]
        out.append(i)
    return out


def accept_issue(st: FlowStep, issue: dict, pt) -> None:
    """Accept one place of an error (the user's choice: it is fine there)."""
    st.accepted.append({"code": issue["code"], "x": float(pt[0]), "y": float(pt[1]), "message": issue.get("message", "")})
    st.issues = _apply_accepted(st, st.issues)


# ====================================================================== export
#: Purposes of the traced flake outlines in an exported GDS: ``flake:<layer uid>``, one GDS layer per
#: stack layer, asked once and kept in config/layers.json (user, 2026-11-17: export the borders too).
FLAKE_PREFIX = "flake:"


def flake_shapes(spec: DeviceSpec, flow: Flow, keys: list[str]) -> dict[str, list[Poly]]:
    """The flakes' outlines on the chip for a GDS: each layer's outline at the latest of ``keys``
    (process order) that has one, keyed ``flake:<uid>`` (bottom → top)."""
    order = [k for k in flow.order if k in keys and k in flow.steps]
    out: dict[str, list[Poly]] = {}
    for layer in spec.layers[1:]:
        for k in reversed(order):
            polys = flow.steps[k].outlines.get(layer.uid)
            if polys:
                out[FLAKE_PREFIX + layer.uid] = [list(map(tuple, q)) for q in polys]
                break
    return out


def _purpose_name(spec: DeviceSpec | None, purpose: str) -> str:
    if purpose.startswith(FLAKE_PREFIX):
        layer = spec.layer(purpose[len(FLAKE_PREFIX):]) if spec is not None else None
        return f"flake outline: {layer.label if layer is not None else purpose[len(FLAKE_PREFIX):]}"
    return PURPOSE_NAMES.get(purpose, purpose)


def _check_flake_layers(config: LabConfig, purposes) -> None:
    """A flake outline on a layer another purpose uses would merge with it: refuse."""
    used: dict[tuple, list[str]] = {}
    for p in purposes:
        used.setdefault(config.gds_layer(p), []).append(p)
    clash = [ps for ps in used.values() if len(ps) > 1 and any(p.startswith(FLAKE_PREFIX) for p in ps)]
    if clash:
        raise ValueError("A flake outline shares a GDS layer with another purpose (Rules › GDS layers, or config/layers.json): "
                         + "; ".join(" = ".join(ps) for ps in clash))


#: First GDS layer suggested for the traced flake outlines (program choice, 2026-11-18): 100 + the layer's place in the
#: stack (substrate = 100), moved up past any layer already in use — far from the lab's pattern layers 1–8.
FLAKE_LAYER_BASE = 100
LAYER_HINTS = {"prepattern_contacts": "The lab's tWSe2 batch 1 drawings use 5/0 (InnerContacts)",
               "contact_gates": "The lab's tWSe2 batch 1 drawings use 6/0 (CG)",
               "electrodes": "The lab's tWSe2 batch 1 drawings use 7/0 (OuterContacts)",
               "bg_contacts": "The lab's tWSe2 batch 1 drawings put the back-gate contacts on 7/0, with the electrodes",
               "tg_contacts": "Suggested with the electrodes, 7/0, as the lab's drawings do for the back-gate contacts"}


def _used_layers(config: LabConfig) -> set[tuple[int, int]]:
    return {(g.layer, g.datatype) for g in config.layers.values() if not g.is_missing and g.layer is not None}


def suggested_layer(config: LabConfig, purpose: str, spec: DeviceSpec | None = None,
                    taken: set | None = None) -> tuple[tuple[int, int] | None, str]:
    """(layer, datatype) offered for ``purpose`` when it has none, and why. The lab's tWSe2 batch 1 numbers for the
    fingers / contact gate / electrodes (+ gate contacts with the electrodes); the first free layer for the metal top
    gate; 100 + stack position for a flake outline. Suggestions only: the user confirms them in the dialog."""
    used = _used_layers(config) | set(taken or ())
    if purpose in LAYER_SUGGEST:
        return LAYER_SUGGEST[purpose], LAYER_HINTS.get(purpose, "")
    if purpose.startswith(FLAKE_PREFIX):
        uid = purpose[len(FLAKE_PREFIX):]
        idx = next((i for i, l in enumerate(spec.layers) if l.uid == uid), 0) if spec is not None else 0
        n = FLAKE_LAYER_BASE + idx
        while (n, 0) in used:
            n += 1
        return (n, 0), "Program suggestion: 100 + the flake's place in the stack (no lab number for flake outlines)"
    if purpose == "top_gate":
        busy = used | set(LAYER_SUGGEST.values())
        n = 1
        while (n, 0) in busy:
            n += 1
        return (n, 0), "Program suggestion: the first layer no other purpose uses (no lab number for a metal top gate)"
    return None, ""


def _missing_layers(config: LabConfig, purposes, spec: DeviceSpec | None = None) -> list[MissingItem]:
    out, taken = [], set()
    for purpose in purposes:
        lay = config.layers.get(purpose)
        if lay is None or lay.is_missing:
            sug, hint = suggested_layer(config, purpose, spec, taken)
            if sug is not None and (purpose.startswith(FLAKE_PREFIX) or purpose == "top_gate"):
                taken.add(sug)                       # two flakes asked together get different layers
            out.append(MissingItem(f"layer:{purpose}", f"GDS layer/datatype for {_purpose_name(spec, purpose)}", "", "layer",
                                   hint=hint, suggested=list(sug) if sug else None))
    return out


def _with_flakes(spec: DeviceSpec, flow: Flow, keys: list[str], config: LabConfig, sets: list, flakes: bool) -> list:
    """``sets`` plus the flake outlines (asking for their layers) when ``flakes``."""
    if not flakes:
        return sets
    fl = flake_shapes(spec, flow, keys)
    if not fl:
        raise ValueError("No flake outline on the chip at these steps (locate or draw them first), or untick the flake outlines.")
    return sets + [(fl, config)]


def _prepare(spec: DeviceSpec, sets: list, config: LabConfig) -> None:
    purposes = sorted({p for shapes, _c in sets for p, polys in shapes.items() if polys})
    missing = _missing_layers(config, purposes, spec)
    if missing:
        raise MissingValues(missing)
    _check_flake_layers(config, purposes)


def export_step(spec: DeviceSpec, flow: Flow, key: str, config: LabConfig, out_dir: Path, flakes: bool = False,
                file_name: str | None = None) -> Path:
    """One step's pattern as a GDS (its purposes on their configured layers, chip frame); with
    ``flakes`` also the traced flake outlines at that step, each on its own layer (:func:`flake_shapes`)."""
    st = flow.step(key)
    if not st.shapes:
        raise ValueError("Generate the pattern first.")
    sets = _with_flakes(spec, flow, [key], config, [(gds_shapes(st.shapes), config)], flakes)
    _prepare(spec, sets, config)
    name = f"{_safe(spec.name)}_{key}"
    path = _write(Path(out_dir) / (file_name or f"{name}.gds"), name, sets, None)
    st.exported = str(path)
    return path


def export_steps(spec: DeviceSpec, flow: Flow, keys: list[str], config: LabConfig, out_dir: Path, flakes: bool = False,
                 file_name: str | None = None) -> Path:
    """Several steps written in the same EBL exposure: their patterns in one GDS (no bond pads)."""
    keys = [k for k in STEPS if k in keys]
    if len(keys) == 1:
        return export_step(spec, flow, keys[0], config, out_dir, flakes, file_name)
    chosen = [flow.step(k) for k in keys]
    if not any(s.shapes for s in chosen):
        raise ValueError("Generate the patterns first.")
    sets = _with_flakes(spec, flow, keys, config, [(gds_shapes(s.shapes), config) for s in chosen], flakes)
    _prepare(spec, sets, config)
    name = f"{_safe(spec.name)}_{'+'.join(keys)}"
    path = _write(Path(out_dir) / (file_name or f"{name}.gds"), name, sets, None)
    for s in chosen:
        s.exported = str(path)
    return path


def export_merged(spec: DeviceSpec, flow: Flow, keys: list[str], config: LabConfig, bondpads: BondPads | None, out_path: Path,
                  flakes: bool = False) -> Path:
    """The chosen steps' patterns and the bond-pad preset (its own layers) in one GDS; with ``flakes``
    also the traced flake outlines (each layer's latest outline among the chosen steps)."""
    chosen = [flow.steps[k] for k in keys if k in flow.steps and flow.steps[k].shapes]
    if not chosen:
        raise ValueError("No generated step selected.")
    sets = _with_flakes(spec, flow, keys, config, [(gds_shapes(s.shapes), config) for s in chosen], flakes)
    _prepare(spec, sets, config)
    purposes = sorted({p for shapes, _c in sets for p, polys in shapes.items() if polys})
    if bondpads is not None:
        clash = [p for p in purposes if config.gds_layer(p) in {tuple(v) for v in bondpads.layers.values()}]
        if clash:
            raise ValueError("These purposes use a bond-pad layer: " + ", ".join(f"{p} ({config.gds_layer(p)[0]}/{config.gds_layer(p)[1]})"
                                                                                 for p in clash))
    return _write(out_path, f"{_safe(spec.name)}_merged", sets, bondpads)


def _write(path: Path, top_name: str, sets: list, bondpads: BondPads | None) -> Path:
    layout = kdb.Layout()
    layout.dbu = geo.DBU
    top = layout.create_cell(top_name)
    for shapes, config in sets:
        for purpose, polys in shapes.items():
            if not polys:
                continue
            ln, dt = config.gds_layer(purpose)
            li = layout.layer(kdb.LayerInfo(ln, dt, config.layers[purpose].name if purpose in config.layers else purpose))
            reg = geo.regions(polys)
            top.shapes(li).insert(reg)
    if bondpads is not None:
        src = kdb.Layout()
        src.read(str(bondpads.gds))
        st = src.top_cell()
        scale = src.dbu / layout.dbu
        for li in src.layer_indexes():
            info = src.get_info(li)
            dst = layout.layer(kdb.LayerInfo(info.layer, info.datatype, info.name))
            reg = kdb.Region(st.begin_shapes_rec(li))
            if abs(scale - 1.0) > 1e-12:
                reg = reg.transformed(kdb.ICplxTrans(scale))
            top.shapes(dst).insert(reg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(path))
    return path


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip() or "device")
