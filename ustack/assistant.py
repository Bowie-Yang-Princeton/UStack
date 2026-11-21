"""Optional assistant: a plain-English request → a draft device stack.

With ``ANTHROPIC_API_KEY`` set, the request goes to Claude (Messages API, model from
``USTACK_CLAUDE_MODEL`` or the default below) with the lab's materials and functions and the
instruction to leave every unstated number empty and never substitute a material. Without a
key, a small keyword parser handles the common phrasings. Either way the draft is only a
proposal: it is shown to the user, and Python validation (never the model) decides whether it is
valid; the explicit requests become :class:`~ustack.model.Requirement` entries the validator keeps
checking.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .config import LabConfig
from .model import CONTACT_SCHEMES, FAMILIES, FUNCTIONS, DeviceSpec, Layer, Q, Requirement, new_uid
from .templates import load_template, new_device_from

DEFAULT_MODEL = "claude-opus-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
#: Server-side refusal fallback (the API re-runs a declined request on a fallback model).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Draft:
    spec: DeviceSpec
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "keywords"


SYSTEM = """You turn a researcher's request for a van der Waals heterostructure device into a JSON draft.
Rules:
- Use ONLY these material ids when the request names one of them: {materials}. If the request names a material that is not in this list, put the requested name verbatim in "material" (do NOT substitute).
- Functions (use these keys): {functions}.
- Contact schemes: {schemes}. Families: {families}.
- Layers are listed bottom to top and start with the substrate.
- Put a number only if the request states it. Otherwise use null and list what is missing in "missing".
- Every explicit request (a gate, a contact type, a dimension, a material) also goes into "requirements" as
  {{"kind": "layer"|"contact_scheme"|"param"|"material", "text": "...", "match": {{...}}}}; for kind "layer" the match keys are layer attributes such as "function" and "material".
- Hall bar keys: channel_width_um, channel_length_um, n_probe_pairs, arm_width_um, arm_length_um, sd_extension_um.
Return only JSON: {{"name": str, "family": str, "contact_scheme": str, "layers": [{{"name": str, "material": str, "function": str, "thickness_nm": number|null, "layer_count": number|null, "width_um": number|null, "length_um": number|null, "twist_deg": number|null, "stacking": str}}], "hall_bar": {{...}}, "requirements": [...], "missing": [str], "notes": [str]}}"""


def _call_claude(request: str, config: LabConfig, api_key: str, model: str, timeout: float = 300.0) -> dict:
    system = SYSTEM.format(materials=", ".join(config.materials), functions=", ".join(FUNCTIONS),
                           schemes=", ".join(CONTACT_SCHEMES), families=", ".join(FAMILIES))
    body = json.dumps({"model": model, "max_tokens": 16000, "system": system, "fallbacks": "default",
                       "messages": [{"role": "user", "content": request}]}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                                              "anthropic-beta": FALLBACK_BETA,
                                                              "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("stop_reason") == "refusal":
        raise ValueError("The assistant declined this request; describe the device again or use the templates.")
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("The assistant did not return JSON")
    return json.loads(match.group(0))


def _spec_from_json(d: dict, config: LabConfig | None = None) -> DeviceSpec:
    """The model's JSON as a spec. Numbers it wrote are marked ``assistant`` (not ``user``), so the
    provenance shows they came through the model; like user values, Configure never replaces them."""
    spec = DeviceSpec(name=d.get("name") or "Assistant draft", family=d.get("family") if d.get("family") in FAMILIES else "custom",
                      contact_scheme=d.get("contact_scheme") if d.get("contact_scheme") in CONTACT_SCHEMES else "edge_1d")
    for ld in d.get("layers", []):
        layer = Layer(uid=new_uid(), name=ld.get("name", ""), material=ld.get("material", ""), function=ld.get("function", ""))
        for key in ("thickness_nm", "layer_count", "width_um", "length_um", "twist_deg"):
            if ld.get(key) is not None:
                getattr(layer, key).set(float(ld[key]), "assistant")
        layer.stacking = ld.get("stacking", "") or ""
        spec.layers.append(layer)
    for key, value in (d.get("hall_bar") or {}).items():
        if value is not None and key in spec.hall_bar.params:
            spec.hall_bar.params[key] = Q(float(value), "assistant")
    spec.requirements = [Requirement.from_json(r) for r in d.get("requirements", []) if isinstance(r, dict) and "kind" in r]
    if config is not None:
        from .validation import assign_roles
        assign_roles(spec, config)
    return spec


_NUM = r"(\d+(?:\.\d+)?)"


def parse_offline(request: str, config: LabConfig) -> Draft:
    """Keyword parser for common requests. Builds only what is asked for on top of the matching
    group template's skeleton (substrate + encapsulating hBN), and records each explicit request."""
    text = request.lower()
    notes: list[str] = []
    if any(k in text for k in ("rhombohedral", "pentalayer", "tetralayer", "r5g", "r4g")):
        family = "ju_rhombohedral"
    elif "graphene" in text:
        family = "xia_graphene"
    elif any(m.lower() in text for m in ("wse2", "mote2", "ws2", "mos2", "mose2", "tmd")):
        family = "mak_tmd" if ("mak" in text or "xia" in text or "cornell" in text) else "xu_tmd"
    else:
        family = "custom"
    if family != "custom":
        spec = new_device_from(load_template(family), "Assistant draft")
    else:
        spec = DeviceSpec(name="Assistant draft")
        spec.layers = [Layer("S0", "Substrate", "", "substrate", thickness_nm=Q(285.0, "material"))]
    reqs: list[Requirement] = []
    # gates the request asks for / leaves out
    wants_bg = any(k in text for k in ("back gate", "bottom gate", "backgate", "bg"))
    wants_tg = any(k in text for k in ("top gate", "topgate", "dual gate", "dual-gate", "dual-gated"))
    if wants_bg:
        mat = "graphite" if "graphite back" in text or "graphite bottom" in text else None
        reqs.append(Requirement("layer", f"{mat or ''} bottom gate".strip(), {"function": "bottom_gate", **({"material": mat} if mat else {})}))
    if wants_tg:
        mat = "graphite" if "graphite top" in text else None
        reqs.append(Requirement("layer", f"{mat or ''} top gate".strip(), {"function": "top_gate", **({"material": mat} if mat else {})}))
    if wants_bg and not wants_tg:
        # the request lists its gates and a top gate is not among them
        removed = [l.label for l in spec.layers if l.function == "top_gate"]
        if removed:
            spec.layers = [l for l in spec.layers if l.function != "top_gate"]
            notes.append(f"Left out {', '.join(removed)}: the request names a back gate only.")
            for l in spec.layers:
                if l.function == "top_gate_dielectric":
                    l.function = "encapsulation"
    if "edge contact" in text or "1d contact" in text or "edge-contact" in text:
        spec.contact_scheme = "edge_1d"
        reqs.append(Requirement("contact_scheme", "edge (1D) contacts", {"value": "edge_1d"}))
    if "monolayer" in text or "single-layer" in text or "single layer" in text:
        for d in spec.device_layers():
            d.layer_count.set(1, "user")
        reqs.append(Requirement("layer", "monolayer device layer", {"function": "device_layer", "layer_count": 1}))
    if "encapsulated" in text and "hbn" in text:
        reqs.append(Requirement("material", "hBN encapsulation", {"value": "hBN"}))
    if "hall bar" in text or "hallbar" in text:
        spec.pattern = "hall_bar"
    m = re.search(r"width\s*(?:of|=)?\s*" + _NUM + r"\s*(?:µm|um|micron)", text)
    if m:
        spec.hall_bar.params["channel_width_um"] = Q(float(m.group(1)), "user")
        reqs.append(Requirement("param", f"channel width {m.group(1)} µm", {"key": "channel_width_um", "value": float(m.group(1))}))
    m = re.search(r"length\s*(?:of|=)?\s*" + _NUM + r"\s*(?:µm|um|micron)", text)
    if m:
        spec.hall_bar.params["channel_length_um"] = Q(float(m.group(1)), "user")
        reqs.append(Requirement("param", f"channel length {m.group(1)} µm", {"key": "channel_length_um", "value": float(m.group(1))}))
    spec.requirements = reqs
    spec.description = f"Drafted from the request: “{request.strip()}”"
    return Draft(spec, [], notes, "keywords")


def draft_from_request(request: str, config: LabConfig, api_key: str | None = None, model: str | None = None) -> Draft:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return parse_offline(request, config)
    data = _call_claude(request, config, key, model or os.environ.get("USTACK_CLAUDE_MODEL", DEFAULT_MODEL))
    spec = _spec_from_json(data, config)
    spec.description = f"Drafted by Claude from the request: “{request.strip()}”"
    return Draft(spec, list(data.get("missing", [])), list(data.get("notes", [])), "claude")
