"""Write the four built-in group templates to ustack/data/templates/*.json.

Each template records the published stack and process with its citation. Numbers a paper states
are put in ``suggested`` (with the paper as ``suggested_source``), never in ``value``: the user
accepts them (one click) or enters their own, and anything left empty is asked for on save.
Where a paper used a material that is not in config/materials.csv (e.g. Ti/Pd gates), the
material is left empty with the paper's choice in ``literature`` — the program does not swap in
a lab metal on its own.

Run:  python tools/make_builtin_templates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ustack.model import DeviceSpec, HallBar, Layer, MetalPart, Q  # noqa: E402

OUT = ROOT / "ustack" / "data" / "templates"

PARK24 = "H. Park et al., arXiv:2406.09591 (2024), Methods"
PARK23 = "H. Park et al., Nature 622, 74 (2023), arXiv:2308.02657, Methods"
XIA24 = "Y. Xia et al., Nature 637, 833 (2025), arXiv:2405.14784, Methods"
XIA25 = "Y. Xia et al., arXiv:2508.02662 (2025), Methods"
LXIA23 = "L.-Q. Xia et al., arXiv:2310.12204 (helical trilayer graphene), Methods A"
LAB_FAB = "Fab Manual (Sihong), Graphene Devices - New / Nov 2025"
LAB_DRC = "Design - Autocad DRC.docx (Liqiao)"
HAN23 = "T. Han et al., arXiv:2305.03151 (pentalayer rhombohedral graphene), Methods"
HAN24 = "T. Han et al., arXiv:2408.15233 (chiral superconductivity, rhombohedral graphene), Methods"
LU24 = "Z. Lu et al., Nature 626, 759 (2024), arXiv:2309.17436"


def L(uid, name, material, function, **kw) -> Layer:
    layer = Layer(uid=uid, name=name, material=material, function=function)
    for key, value in kw.items():
        setattr(layer, key, value)
    return layer


def metal(*parts) -> list[MetalPart]:
    return [MetalPart(m, Q.suggest(t, src) if t is not None else Q()) for m, t, src in parts]


def substrate() -> Layer:
    return L("S0", "Substrate", "", "substrate", thickness_nm=Q(285.0, "material"),
             notes="Choose 285 nm SiO2 on doped Si (usable as a global back gate) or on resistive Si.")


def hallbar(**suggested) -> HallBar:
    hb = HallBar()
    for key, (value, src) in suggested.items():
        hb.params[key] = Q.suggest(value, src)
    return hb


def xu_tmd() -> DeviceSpec:
    spec = DeviceSpec(name="TMD Hall bar · Xu structure", family="xu_tmd", template_id="xu_tmd",
                      contact_scheme="prepatterned_bottom")
    spec.description = ("Xiaodong Xu (UW) structure: the bottom gate (graphite + hBN) is stacked first, Pt Hall-bar contacts are "
                        "written on it, the TMD is dropped onto the Pt together with a thin hBN that carries the contact gates, "
                        "and an hBN/graphite/hBN top gate is transferred last.")
    spec.layers = [
        substrate(),
        L("BG", "Graphite bottom gate", "graphite", "bottom_gate", notes="Few-layer graphite; thickness not stated in the papers."),
        L("BBN", "Bottom hBN", "hBN", "bottom_gate_dielectric", role="bottom_hbn",
          notes="Lab rule: larger than the middle hBN and the TMD (the Pt contacts are written on it)."),
        L("PT", "Pt Hall-bar contacts", "Pt/Ti", "hall_bar_contacts", role="prepattern_contacts",
          metal_parts=metal(("Ti", None, ""), ("Pt", 8.0, PARK24)),
          literature="Pt 8 nm by e-beam evaporation on the BG stack, no adhesion layer stated; contact-mode AFM clean, Pt RMS < 300 pm (Park 2024).",
          notes="Lab metal list has Ti/Pt; set the Ti thickness (0 if none)."),
        L("TMD1", "WSe2 (half 1)", "WSe2", "device_layer", layer_count=Q.suggest(1, "monolayer, tear/cut-and-stack"), pieces=2,
          role="tmd", notes="One monolayer cut in half with an AFM tip; this is the first half.",
          literature="Xu group: MoTe2 monolayer cut with an AFM tip in an Ar glovebox (Park 2023/2024)."),
        L("TMD2", "WSe2 (half 2, twisted)", "WSe2", "device_layer", layer_count=Q.suggest(1, "monolayer"), role="tmd",
          notes="Second half, rotated by the twist angle."),
        L("MBN", "Middle hBN (thin, contact-gate dielectric)", "hBN", "contact_gate_dielectric", role="middle_hbn",
          notes="Lab rule: thin; edge ≥ 1 µm beyond every TMD edge so the contact gates sit on hBN.",
          literature="'a thin BN as the contact gate dielectric' (Park 2024); thickness not stated."),
        L("CG", "Pt contact gates", "Pt/Ti", "contact_gate", metal_parts=metal(("Ti", None, ""), ("Pt", 8.0, PARK24)),
          literature="Pt (8 nm) contact gates by EBL + evaporation (Park 2024)."),
        L("LEADS", "Cr/Au leads", "Au/Cr", "leads", metal_parts=metal(("Cr", 5.0, PARK24), ("Au", 70.0, PARK24)),
          literature="'Cr/Au (5/70 nm) connecting electrodes' (Park 2024)."),
        L("TBN", "hBN (top-gate dielectric)", "hBN", "top_gate_dielectric", role="tg_hbn"),
        L("TG", "Graphite top gate", "graphite", "top_gate"),
        L("CAP", "Top-most hBN", "hBN", "encapsulation", role="top_hbn"),
    ]
    spec.hall_bar = hallbar()
    spec.process = [
        "PC dry transfer: pick up the bottom hBN, then the graphite bottom gate; release onto the Si chip.",
        "EBL + e-beam evaporation of the Pt Hall-bar contacts on the BG stack; contact-mode AFM cleaning (Pt RMS < 300 pm).   [UStack Stage 2: prepattern contacts]",
        "Cut the TMD monolayer with an AFM tip (MoTe2: Ar glovebox, O2/H2O < 0.1 ppm). Pick up the thin hBN, half 1, rotate, half 2; release onto the BG stack and Pt.",
        "Wash, AFM-clean; EBL + evaporation of the Pt contact gates and the Cr/Au connecting electrodes; AFM-clean again.   [UStack Stage 2: contact gates + leads]",
        "Transfer the hBN/graphite/hBN top gate.",
    ]
    spec.references = [
        {"cite": PARK24, "url": "https://arxiv.org/abs/2406.09591"},
        {"cite": PARK23, "url": "https://arxiv.org/abs/2308.02657"},
    ]
    return spec


def mak_tmd() -> DeviceSpec:
    spec = DeviceSpec(name="TMD Hall bar · Mak / Xia structure", family="mak_tmd", template_id="mak_tmd",
                      contact_scheme="prepatterned_bottom")
    spec.description = ("Kin Fai Mak / Jie Shan (Cornell), Yiyu Xia's devices. Ti/Pt electrodes are pre-patterned on the chip as two "
                        "columns of parallel fingers with a long thin gap between the columns. The whole stack (cap, TG, thin hBN, "
                        "WSe2, bottom hBN, graphite BG) is released with the BG in the gap, never touching the Pt. The WSe2 is wider "
                        "than the bottom hBN, so it reaches past the hBN edge onto the finger tips of both columns and stays gated in "
                        "between. Source and drain are the end fingers of the two columns (Hall bar › S/D 'Bent to the sides'; W is "
                        "the gap between the columns). Contact gates and split gates are written on top; the top gate is smaller "
                        "than the bottom gate and defines the channel. Layout as described by the lab on 2026-11-16.")
    tipd = "Ti/Pd 5/35 nm (Xia 2025b; 5/40 nm in Xia 2024). Ti/Pd is not in config/materials.csv: choose a lab metal or add it."
    spec.layers = [
        substrate(),
        L("PT", "Ti/Pt electrodes (two columns)", "Pt/Ti", "hall_bar_contacts", role="prepattern_contacts",
          metal_parts=metal(("Ti", 5.0, XIA25), ("Pt", 25.0, XIA25)),
          literature="Pre-patterned Ti/Pt (5 nm/25 nm) electrodes (Xia 2025b).",
          notes="Two columns of fingers on the bare chip; the BG lands in the gap between them and never touches them."),
        L("BG", "Graphite bottom gate (in the gap)", "graphite", "bottom_gate", notes="Few-layer graphite; narrower than the electrode gap."),
        L("BBN", "Bottom hBN", "hBN", "bottom_gate_dielectric", role="bottom_hbn", thickness_nm=Q.suggest(12.0, XIA25),
          notes="Covers the BG; narrower than the WSe2 so the WSe2 reaches past it onto the electrodes."),
        L("TMD1", "WSe2 (half 1)", "WSe2", "device_layer", layer_count=Q.suggest(1, "monolayer"), pieces=2, role="tmd",
          notes="Large monolayer cut in half with an AFM tip; wider than the bottom hBN and the electrode gap."),
        L("TMD2", "WSe2 (half 2, twisted)", "WSe2", "device_layer", layer_count=Q.suggest(1, "monolayer"), role="tmd"),
        L("TBN", "Top hBN (top-gate dielectric)", "hBN", "top_gate_dielectric", role="tg_hbn", thickness_nm=Q.suggest(3.4, XIA25)),
        L("TG", "Graphite top gate (smaller than BG)", "graphite", "top_gate",
          notes="Smaller than the bottom gate; the TG ∩ BG overlap is the channel (Xia 2024/2025)."),
        L("CAP", "hBN cap", "hBN", "encapsulation", role="top_hbn", notes="Thicker cap; thickness not stated."),
        L("CG", "Contact gates", "", "contact_gate", literature=tipd,
          notes="Heavily dope the TMD between the channel and the Pt electrodes."),
        L("SG", "Split gates", "", "split_gate", literature=tipd, notes="Deplete the TMD regions gated only by the BG."),
    ]
    spec.hall_bar = hallbar(channel_length_um=(1.6, XIA25 + ": L ≈ 1.6 µm (centre-to-centre probe distance)"))
    spec.hall_bar.sd_style = "bent"
    spec.process = [
        "EBL + evaporation of the Ti/Pt electrodes on the chip: two columns of fingers with a thin gap between them for the BG.   "
        "[UStack Stage 2: prepattern contacts]",
        "Exfoliate hBN, graphite and monolayer WSe2; cut the WSe2 in half with an AFM tip.",
        "PC on PDMS picks up, in order: hBN cap, TG graphite, thin hBN, half 1, half 2 (twisted), bottom hBN, BG graphite.",
        "Release at 200 °C (Xia 2024; 180 °C for MoTe2, Kang 2025) with the BG in the gap between the electrode columns; the WSe2 "
        "reaches past the bottom hBN onto the finger tips.",
        "EBL + evaporation of the contact gates and split gates.   [UStack Stage 2: contact gates + split gates]",
    ]
    spec.references = [
        {"cite": XIA24, "url": "https://arxiv.org/abs/2405.14784"},
        {"cite": XIA25, "url": "https://arxiv.org/abs/2508.02662"},
    ]
    return spec


def xia_graphene() -> DeviceSpec:
    spec = DeviceSpec(name="Graphene Hall bar · Liqiao Xia structure", family="xia_graphene", template_id="xia_graphene",
                      contact_scheme="edge_1d")
    spec.description = ("Liqiao Xia (MIT) structure, as used in the lab's DRC notes: a graphite bottom gate under hBN, the graphene "
                        "(monolayer here; twisted / helical stacks add pieces) under a top hBN, a metal top gate, and 1D edge "
                        "contacts. Board etch, Hall-bar (device) etch and contacts are separate EBL steps.")
    spec.layers = [
        substrate(),
        L("BG", "Graphite bottom gate", "graphite", "bottom_gate",
          notes="'few-layer graphene strip' in the paper. Lab DRC: sticks out of the bottom BN on one side for its contact.",
          literature="BG stack (hBN + graphite) by PC, released on 285 nm SiO2/Si, 350 °C 12 h vacuum anneal, AFM contact-mode clean (L.-Q. Xia)."),
        L("BBN", "Bottom hBN", "hBN", "bottom_gate_dielectric", role="bottom_hbn",
          notes="Lab DRC: larger than the BG in three directions."),
        L("GR", "Graphene", "graphene", "device_layer", layer_count=Q.suggest(1, "monolayer (Hall-bar example)"), role="graphene",
          literature="Helical trilayer: one monolayer laser-cut into three, each rotated 1.8° at pick-up (room temperature)."),
        L("TBN", "Top hBN", "hBN", "top_gate_dielectric", role="tg_hbn",
          notes="Lab DRC: covers the good graphene regions (graphene may tear at the top-BN edge)."),
        L("TG", "Metal top gate", "Ti/Au", "top_gate",
          metal_parts=[MetalPart("Ti", Q.suggest(5.0, LAB_FAB + " (pending)")), MetalPart("Au", Q.suggest(30.0, LAB_FAB + " (pending)"))],
          literature="Paper: 25-65 nm Au on a 2-5 nm Cr or Ti adhesion layer (thermal evaporation). Lab default Ti/Au 5/30 nm (Fab Manual, pending).",
          notes="Delete this layer for a device without a top gate."),
        L("CT", "1D edge contacts", "Au/Cr", "hall_bar_contacts",
          metal_parts=[MetalPart("Cr", Q.suggest(5.0, LAB_FAB + " (pending)")), MetalPart("Au", Q.suggest(60.0, LAB_FAB + " (pending)"))],
          literature="Paper: 63-75 nm Au on 2-5 nm Cr. Lab: Cr/Au 5/~60 nm, thicker than BN/Gr/BN.",
          notes="Also used for source / drain."),
    ]
    spec.hall_bar = hallbar()
    spec.process = [
        "Bottom stack: PC picks up an hBN flake and a few-layer graphite strip; release on 285 nm SiO2/Si; 350 °C 12 h vacuum anneal; AFM contact-mode clean.",
        "Top stack: a second PC stamp picks up the top hBN and the graphene piece(s) at room temperature (rotate between pieces for twisted stacks).",
        "Release onto the bottom stack at 150-170 °C; find a bubble-free region by AFM.",
        "Board etch (BN-Test, 5-10 nm more than top + bottom BN), with top-BN-only check windows.   [UStack Stage 2: board etch]",
        "Metal top gate (Ti/Au or Cr/Au).   [UStack Stage 2: gate]",
        "Hall-bar (device) etch: arms ≥ 1.5 µm beyond the BG for 1D contacts.   [UStack Stage 2: Hall-bar etch]",
        "1D contacts: etch top BN, graphene 7-10 s, 5-10 nm of bottom BN; Cr/Au 5/~60 nm.   [UStack Stage 2: contacts]",
    ]
    spec.references = [
        {"cite": LXIA23, "url": "https://arxiv.org/abs/2310.12204"},
        {"cite": LAB_DRC, "url": ""},
        {"cite": LAB_FAB, "url": ""},
    ]
    return spec


def ju_rhombohedral() -> DeviceSpec:
    spec = DeviceSpec(name="Rhombohedral graphene Hall bar · Long Ju structure", family="ju_rhombohedral",
                      template_id="ju_rhombohedral", contact_scheme="edge_1d")
    spec.description = ("Long Ju (MIT) structure: a rhombohedral multilayer graphene domain (found by near-field IR / Raman and cut "
                        "out by AFM or femtosecond laser) is picked up with the top hBN, graphite TG and hBN cap on PPC and landed on a "
                        "prepared bottom stack (hBN + graphite BG); EBL + RIE into a Hall bar, Cr/Au contacts.")
    spec.layers = [
        substrate(),
        L("BG", "Graphite bottom gate", "graphite", "bottom_gate"),
        L("BBN", "Bottom hBN", "hBN", "bottom_gate_dielectric", role="bottom_hbn",
          notes="Straight edges misaligned to the graphene unless a moiré is wanted (Han 2024).", literature=HAN23),
        L("RG", "Rhombohedral multilayer graphene", "graphene", "device_layer", stacking="rhombohedral",
          layer_count=Q.suggest(5, "pentalayer (Lu 2024, Han 2024)"), role="graphene",
          literature="Rhombohedral domains identified by IR camera / near-field IR / Raman; isolated by AFM or fs-laser cutting (Han 2024, 2025)."),
        L("TBN", "Top hBN (top-gate dielectric)", "hBN", "top_gate_dielectric", role="tg_hbn",
          twist_deg=Q(None, "", 0.77, LU24 + ": moiré with the top hBN, ~0.77°"),
          notes="Aligned to the graphene only for a moiré device (Lu 2024: ~0.77°, period ~11.5 nm); otherwise misaligned."),
        L("TG", "Graphite top gate", "graphite", "top_gate", literature="Han 2025: top hBN / graphite / middle hBN / ... ; Han 2024 used a NiCr top layer."),
        L("CAP", "hBN cap", "hBN", "encapsulation", role="top_hbn"),
        L("CT", "Cr/Au contacts", "Au/Cr", "hall_bar_contacts",
          metal_parts=[MetalPart("Cr", Q()), MetalPart("Au", Q())],
          literature="'Cr–Au for electrical connections' (Han 2025); thickness not stated."),
    ]
    spec.hall_bar = hallbar()
    spec.process = [
        "Exfoliate multilayer graphene; identify rhombohedral domains (IR camera, near-field IR, Raman).",
        "Isolate the rhombohedral domain by AFM or femtosecond-laser cutting.",
        "PPC film picks up the hBN cap, graphite TG, top hBN and the graphene; land on the prepared bottom stack (hBN + graphite BG).",
        "EBL + RIE into a Hall bar.   [UStack Stage 2: Hall-bar etch]",
        "Cr/Au contacts to source, drain, probes and gates.   [UStack Stage 2: contacts]",
    ]
    spec.references = [
        {"cite": LU24, "url": "https://arxiv.org/abs/2309.17436"},
        {"cite": HAN23, "url": "https://arxiv.org/abs/2305.03151"},
        {"cite": HAN24, "url": "https://arxiv.org/abs/2408.15233"},
    ]
    return spec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (xu_tmd, mak_tmd, xia_graphene, ju_rhombohedral):
        spec = build()
        spec.id = spec.template_id
        spec.created = spec.modified = "2026-11-16T00:00:00"
        data = spec.to_json()
        (OUT / f"{spec.template_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", spec.template_id, len(spec.layers), "layers")


if __name__ == "__main__":
    main()
