"""Stage 2: Hall-bar geometry, image analysis, alignment, pattern generation, DRC, GDS export."""

import math

import cv2
import numpy as np
import pytest

import make_fixtures as mf
from conftest import circle, graphene_answers
from ustack import geometry as geo
from ustack import gds_io, imaging, stage2, templates, validation
from ustack.alignment import Similarity, auto_align, fit_similarity, squares_from_polygons
from ustack.config import MissingValues, apply_answers
from ustack.hallbar import HallBar


def test_hallbar_geometry():
    hb = HallBar(W=2, L=6, n=3, arm_w=1, arm_l=4, sd_ext=2)
    assert len(hb.terminals()) == 2 + 2 * 3
    assert hb.probe_x() == [-3, 0, 3]
    area = geo.area(hb.body_region())
    assert area == pytest.approx(2 * 10 + 6 * 4 * 1, rel=1e-3)
    hb.center, hb.angle_deg = (10, 5), 90
    s = next(t for t in hb.terminals() if t.name == "S")
    assert s.end == pytest.approx((10, 5 - 5))
    assert s.direction == pytest.approx((0, -1))


def synthetic_image() -> np.ndarray:
    """Grey image with the sample's marker squares painted on (pale yellow)."""
    img = np.full((3984, 6000, 3), 128, np.uint8)
    for cx, cy, side in mf.IMAGE_SQUARES:
        h = side / 2
        cv2.rectangle(img, (int(cx - h), int(cy - h)), (int(cx + h), int(cy + h)), (130, 250, 245), -1)
    return img


def test_marker_detection_and_alignment(bondpads):
    img = synthetic_image()
    groups = imaging.detect_markers(img)
    squares = [s for g in groups for s in g.squares]
    assert len(squares) == 8
    polys = gds_io.read_polygons(bondpads)
    gsq = squares_from_polygons(polys[mf.MARKER_LAYER])
    sim = auto_align(squares, gsq, site=mf.site_center_um(), site_radius=150)
    assert sim.scale == pytest.approx(mf.TRUE.scale, rel=2e-3)
    assert sim.angle_deg == pytest.approx(mf.TRUE.angle_deg, abs=0.1)
    assert (sim.tx, sim.ty) == pytest.approx((mf.TRUE.tx, mf.TRUE.ty), abs=0.1)
    assert sim.rms_um < 0.05


def test_similarity_roundtrip():
    t = Similarity(0.05, 30, 10, -4, mirror=True)
    px = np.array([[0, 0], [100, 50], [3000, 2000]])
    assert t.inverse(t.apply(px)) == pytest.approx(px)
    fit = fit_similarity(px, t.apply(px), mirror=True)
    assert fit.scale == pytest.approx(0.05) and fit.angle_deg == pytest.approx(30)


def test_flake_wand_on_synthetic_flake():
    img = np.full((800, 1200, 3), 128, np.uint8)
    cv2.fillPoly(img, [np.int32([[300, 200], [900, 250], [850, 600], [350, 550]])], (200, 150, 90))
    poly = imaging.detect_flake(img, (600, 400))
    assert 4 <= len(poly) <= 12
    assert geo.poly_area(poly) == pytest.approx(cv2.contourArea(np.float32([[300, 200], [900, 250], [850, 600], [350, 550]])), rel=0.05)


def _graphene_step(cfg, bondpads):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "g")
    validation.apply_spec_answers(spec, graphene_answers())
    apply_answers(cfg, {"rule:stack.substrate_breakdown_field_MV_per_cm": 5.0})  # TEST VALUE
    cx, cy = mf.site_center_um()

    def R(w, l):
        return geo.nominal_rect(w, l, cx, cy)

    inp = stage2.StepInputs(purposes=["board_etch", "top_gate", "hallbar_etch", "contacts"])
    inp.outlines = {"GR": [R(16, 26)], "BG": [R(6, 10)], "BBN": [R(24, 34)], "TBN": [R(20, 30)]}
    inp.defects = [{"kind": "bubble", "polygon": circle(cx + 9, cy + 5, 0.6)}]
    inp.measured = {"TBN": {"thickness_nm": 21.5}, "BBN": {"thickness_nm": 28}}
    inp.etch_windows = [geo.rect(cx + 13.3, cy + 8.2, cx + 14.8, cy + 9.7)]
    polys = gds_io.read_polygons(bondpads)
    inp.lead_tips = gds_io.lead_tips(polys[mf.LEAD_LAYER], (cx, cy), 60)
    inp.marker_polys = [p for p in polys[mf.MARKER_LAYER] if math.dist(geo.centroid(p), (cx, cy)) < 150]
    return spec, inp


def test_stage2_asks_once_for_everything(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    with pytest.raises(MissingValues) as exc:
        stage2.design_step(spec, inp, cfg)
    keys = {i.key for i in exc.value.items}
    assert keys == {"rule:ebl.defect_clearance_um", "rule:ebl.contact_overlap_um", "rule:ebl.etch_mask_overhang_um",
                    "rule:ebl.board_etch_keep_margin_um", "spec:pattern.tg_length_um", "spec:pattern.tg_overhang_um"}


def test_stage2_graphene_end_to_end(cfg, bondpads, tmp_path):
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 3.0, "spec:pattern.tg_overhang_um": 1.0})
    design = stage2.design_step(spec, inp, cfg)
    hb = design.hall_bar
    # the specified dimensions are untouched; only the placement was chosen
    assert (hb.W, hb.L, hb.n, hb.arm_w, hb.arm_l, hb.sd_ext) == (2, 6, 2, 1, 4.5, 4)
    assert design.placement_score["gated_fraction"] == 1.0
    gr = geo.region(inp.outlines["GR"][0])
    assert (hb.body_region() - gr).is_empty()
    report = stage2.check_step(spec, inp, design, cfg)
    assert not report.errors, [i.message for i in report.errors]
    assert all(r["routed_to_pad"] for r in design.routes)
    with pytest.raises(MissingValues):
        stage2.export_step(spec, inp, design, report, cfg, tmp_path, "s1")   # hallbar_etch / top_gate / windows have no layers yet
    apply_answers(cfg, {"layer:hallbar_etch": (90, 0), "layer:top_gate": (91, 0), "layer:etch_windows": (92, 0)})  # TEST VALUES
    out = stage2.export_step(spec, inp, design, report, cfg, tmp_path, "s1")
    layers = {(l, d) for l, d, _ in gds_io.layers_in(out["gds"])}
    assert {(3, 0), (7, 0), (8, 0), (90, 0), (91, 0), (92, 0)} <= layers
    updated = out["updated_spec"]
    assert updated.stage2["steps"][0]["hall_bar_as_built"]["center_um"] == pytest.approx(list(hb.center), abs=1e-3)
    assert updated.layer("TBN").thickness_nm.value == 21.5 and updated.layer("TBN").thickness_nm.source == "measured"
    assert out["report"].read_text().startswith("# g")


def test_stage2_reports_defect_and_never_moves_params(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 3.0, "spec:pattern.tg_overhang_um": 1.0})
    cx, cy = mf.site_center_um()
    inp.defects.append({"kind": "bubble", "polygon": circle(cx, cy, 0.5)})
    inp.placement = {"center": [cx, cy], "angle_deg": 0, "source": "user"}   # a placement on top of the bubble
    design = stage2.design_step(spec, inp, cfg)
    report = stage2.check_step(spec, inp, design, cfg)
    assert any(i.code == "defect" for i in report.errors)


def test_hall_bar_that_cannot_fit_is_reported(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 3.0, "spec:pattern.tg_overhang_um": 1.0,
                                         "spec:hall_bar.arm_length_um": 14.0})
    with pytest.raises(ValueError, match="does not shrink"):
        stage2.design_step(spec, inp, cfg)


def test_etch_plan_accepts_single_values_and_missing_rules(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    inp.purposes = ["board_etch", "contacts"]
    cfg.rules["etch.contact_bottom_bn_nm"].value = 7.0          # TEST VALUE: one number instead of a range
    cfg.rules["etch.board_overetch_nm"].value = None
    cfg.rules["etch.board_overetch_nm"].status = "missing"
    lines = stage2.etch_plan(spec, inp, cfg)
    assert any("7–7 nm" in l for l in lines)


def test_top_gate_stays_off_the_contacts_and_other_steps_are_checked(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 7.0, "spec:pattern.tg_overhang_um": 1.0})
    inp.purposes = ["top_gate"]
    design = stage2.design_step(spec, inp, cfg)
    pads = geo.regions(design.shapes["ref_contacts"])          # where the later contacts will land
    assert (geo.regions(design.shapes["top_gate"]) & pads).is_empty()
    assert not any(i.code in ("metal_overlap", "metal_gap") for i in stage2.check_step(spec, inp, design, cfg).errors)
    # a contact written in an earlier step on top of the gate is reported as a short
    hb = design.hall_bar
    spec.stage2["steps"] = [{"step": "old", "patterns_um": {"contacts": [geo.rect(*hb.center, hb.center[0] + 1, hb.center[1] + 1)]}}]
    assert any(i.code == "metal_overlap" for i in stage2.check_step(spec, inp, design, cfg).errors)


def test_metal_back_gate_from_the_gds_is_used(cfg, bondpads):
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 3.0, "spec:pattern.tg_overhang_um": 1.0})
    spec.layer("BG").material = "Ti/PdAu"                        # a pre-patterned metal strip instead of graphite
    spec.layer("BG").metal_parts = []
    cx, cy = mf.site_center_um()
    inp.outlines["BG"] = [geo.rect(cx - 4.5, cy - 2.5, cx + 4.5, cy + 2.5)]   # a 9 × 5 µm metal gate pad
    design = stage2.design_step(spec, inp, cfg)
    assert design.placement_score["gated_fraction"] == 1.0      # the channel is placed over the strip
    report = stage2.check_step(spec, inp, design, cfg)
    assert not any(i.code == "beyond_bg" for i in report.errors), [i.message for i in report.errors]


def test_gate_strip_under_the_whole_device_bends_the_source_and_drain(cfg, bondpads):
    """A back-gate strip that runs the whole length of the graphene (Gr device 1): straight S/D
    contacts would land on it, so the automatic style turns them to the sides and the channel is
    fully gated."""
    spec, inp = _graphene_step(cfg, bondpads)
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0, "rule:ebl.contact_overlap_um": 0.5,
                        "rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    validation.apply_spec_answers(spec, {"spec:pattern.tg_length_um": 3.0, "spec:pattern.tg_overhang_um": 1.0})
    assert spec.hall_bar.sd_style == "auto"
    spec.layer("BG").material = "Ti/PdAu"
    spec.layer("BG").metal_parts = []
    cx, cy = mf.site_center_um()
    inp.outlines["BG"] = [geo.rect(cx - 40, cy - 2.5, cx + 40, cy + 2.5)]     # 80 × 5 µm strip, longer than the graphene
    inp.outlines["GR"] = [geo.rect(cx - 13, cy - 8, cx + 13, cy + 8)]
    inp.defects = []
    design = stage2.design_step(spec, inp, cfg)
    hb = design.hall_bar
    assert hb.bent and design.placement_score["gated_fraction"] == 1.0
    s, d = (t for t in hb.terminals() if t.kind in ("source", "drain"))
    # the source leaves to one side and the drain to the other
    assert s.direction[0] * d.direction[0] + s.direction[1] * d.direction[1] == pytest.approx(-1.0)
    report = stage2.check_step(spec, inp, design, cfg)
    codes = {i.code for i in report.errors}
    assert not codes & {"beyond_bg", "contact_touch", "metal_overlap", "arm_spacing"}, [i.message for i in report.errors]
    assert not any(i.code == "channel_not_gated" for i in report.issues)
    # an explicit 'straight' choice is kept: here no straight placement clears the strip, which is reported
    spec.hall_bar.sd_style = "straight"
    with pytest.raises(ValueError, match="does not fit"):
        stage2.design_step(spec, inp, cfg)
