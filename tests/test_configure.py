"""Configure: every empty dimension generated so the stack passes validation, user values kept.

Lab values the tests type in are TEST VALUES: they live only in the temporary config copy."""

import math

import pytest

import make_fixtures as mf
from ustack import autoconfig, flakes, gds_io, stage2, templates, validation
from ustack import geometry as geo
from ustack.config import apply_answers
from ustack.model import Layer

#: TEST VALUES for the lab inputs Configure asks for (none of these are lab data)
TEST_DEFAULTS = {
    "graphite": 5.0,
    "hBN:bottom_gate_dielectric": 30.0, "hBN:contact_gate_dielectric": 4.0, "hBN:top_gate_dielectric": 15.0,
    "hBN:encapsulation": 40.0,
}
TEST_RULES = {"stack.alignment_shift_um": 2.0, "ebl.contact_overlap_um": 0.5, "stack.substrate_breakdown_field_MV_per_cm": 5.0,
              "xu.middle_hbn_max_thickness_nm": 10.0}


def _test_answer(item):
    key = item.key
    if item.kind == "choice":
        return (item.suggested or [None])[0]
    if key.startswith("rule:"):
        return TEST_RULES.get(key[5:])
    if key.startswith("default:design.thickness_nm#"):
        sub = key.split("#", 1)[1]
        return TEST_DEFAULTS.get(sub, TEST_DEFAULTS.get(sub.split(":")[0], 5.0))
    if key.startswith("default:design.metal_nm#"):
        return item.suggested if isinstance(item.suggested, (int, float)) else 5.0
    if key.endswith(".layer_count"):
        return 1
    if key.endswith(".twist_deg"):
        return 1.1
    if key.startswith("spec:gate."):
        return 10.0
    return item.suggested if isinstance(item.suggested, (int, float)) else None


def configure_with_test_answers(spec, cfg, rounds=4):
    for _ in range(rounds):
        res = autoconfig.configure(spec, cfg, apply=False)
        if not res.missing:
            break
        answers = {i.key: _test_answer(i) for i in res.missing}
        answers = {k: v for k, v in answers.items() if v is not None}
        assert answers, [i.key for i in res.missing]
        apply_answers(cfg, answers)
        validation.apply_spec_answers(spec, answers)
    return autoconfig.configure(spec, cfg, apply=True)


def twisted_trilayer():
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "TTG")
    gr = spec.layer("GR")
    gr.pieces = 3
    for k in (2, 3):
        piece = Layer.from_json(gr.to_json())
        piece.uid, piece.name, piece.pieces = f"GR{k}", f"Graphene piece {k}", 1
        spec.layers.insert(spec.index("GR") + k - 1, piece)
    return spec


@pytest.mark.parametrize("tid", ["xu_tmd", "mak_tmd", "xia_graphene", "ju_rhombohedral"])
def test_every_template_configures_to_a_valid_spec(cfg, tid):
    spec = templates.new_device_from(templates.load_template(tid), tid)
    res = configure_with_test_answers(spec, cfg)
    assert res.ok, [i.message for i in res.report.issues if i.severity in ("error", "missing")]
    assert flakes.flake_list(spec, cfg, res.report)
    assert all(c.reason for c in res.changes)


def test_twisted_trilayer_graphene(cfg):
    spec = twisted_trilayer()
    first = autoconfig.configure(spec, cfg, apply=False)
    keys = {i.key for i in first.missing}
    # physics choices and lab values are asked for, never generated
    assert {"spec:GR2.twist_deg", "spec:GR3.twist_deg", "rule:stack.alignment_shift_um", "rule:ebl.contact_overlap_um"} <= keys
    assert spec.hall_bar.get("channel_width_um") is None            # apply=False leaves the spec alone
    res = configure_with_test_answers(spec, cfg)
    assert res.ok
    needs = flakes.flake_list(spec, cfg, res.report)
    graphene = next(f for f in needs if f.material == "Graphene")
    assert graphene.pieces == 3
    fp = (spec.hall_bar.get("channel_length_um") + 2 * spec.hall_bar.get("sd_extension_um"),
          spec.hall_bar.get("channel_width_um") + 2 * spec.hall_bar.get("arm_length_um"))
    assert graphene.min_size_um == pytest.approx((fp[0] + 4.0, fp[1] + 4.0))   # ± 2 µm TEST shift


def test_user_values_are_never_changed(cfg):
    spec = twisted_trilayer()
    spec.hall_bar.params["channel_width_um"].set(1.7, "user")
    spec.layer("TBN").thickness_nm.set(12.0, "user")
    spec.layer("GR").width_um.set(40.0, "user")
    configure_with_test_answers(spec, cfg)
    assert spec.hall_bar.params["channel_width_um"].value == 1.7 and spec.hall_bar.params["channel_width_um"].source == "user"
    assert spec.layer("TBN").thickness_nm.value == 12.0
    assert spec.layer("GR").width_um.value == 40.0
    assert spec.layer("GR2").width_um.source == "auto"
    # a second Configure recomputes only its own values
    spec.hall_bar.params["arm_width_um"].set(1.0, "user")
    res = autoconfig.configure(spec, cfg)
    assert spec.hall_bar.get("arm_width_um") == 1.0
    assert res.ok


def test_configured_device_runs_through_stage2(cfg, bondpads):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "cfg")
    res = configure_with_test_answers(spec, cfg)
    assert res.ok
    apply_answers(cfg, {"rule:ebl.etch_mask_overhang_um": 1.0, "rule:ebl.board_etch_keep_margin_um": 1.0})  # TEST VALUES
    cx, cy = mf.site_center_um()
    inp = stage2.StepInputs(purposes=["board_etch", "top_gate", "hallbar_etch", "contacts"])
    # the flakes as designed, centred on the site
    for uid in ("BG", "BBN", "GR", "TBN"):
        l = spec.layer(uid)
        inp.outlines[uid] = [geo.nominal_rect(l.width_um.value, l.length_um.value, cx, cy)]
    polys = gds_io.read_polygons(bondpads)
    inp.lead_tips = gds_io.lead_tips(polys[mf.LEAD_LAYER], (cx, cy), 60)
    inp.marker_polys = [p for p in polys[mf.MARKER_LAYER] if math.dist(geo.centroid(p), (cx, cy)) < 150]
    inp.etch_windows = [geo.rect(cx + 9.0, cy + 7.0, cx + 9.6, cy + 7.6)]
    design = stage2.design_step(spec, inp, cfg)            # no question: Configure supplied the pattern geometry
    report = stage2.check_step(spec, inp, design, cfg)
    assert not report.errors, [i.message for i in report.errors]
    # the metal top gate keeps clear of every contact landing zone
    tg = geo.regions(design.shapes["top_gate"])
    assert (tg & geo.regions(design.shapes["contacts"])).is_empty()


def test_mak_two_electrode_columns(cfg, bondpads):
    """Mak: Configure sizes the electrode gap from the BG stack, and the pre-pattern step draws two finger
    columns that never touch the BG landing in the gap."""
    spec = templates.new_device_from(templates.load_template("mak_tmd"), "mak")
    res = configure_with_test_answers(spec, cfg)
    assert res.ok, [i.message for i in res.report.issues if i.severity in ("error", "missing")]
    W = spec.hall_bar.get("channel_width_um")
    es = cfg.value("ebl.electrode_spacing_target_um")
    bg, bbn, tmd = spec.layer("BG"), spec.layer("BBN"), spec.layer("TMD1")
    assert bg.width_um.value + 2 * es <= W                       # the BG fits in the gap
    assert bbn.width_um.value < W                                 # the bottom hBN stays off the electrodes
    assert tmd.width_um.value > W                                 # the WSe2 reaches onto both columns
    cx, cy = mf.site_center_um()
    inp = stage2.StepInputs(purposes=["prepattern_contacts"], site=(cx, cy))
    polys = gds_io.read_polygons(bondpads)
    inp.lead_tips = gds_io.lead_tips(polys[mf.LEAD_LAYER], (cx, cy), 60)
    design = stage2.design_step(spec, inp, cfg)
    hb = design.hall_bar
    assert hb.bent
    fingers = geo.regions(design.shapes["prepattern_contacts"])
    # nothing in the gap: the BG as designed, centred on the channel, does not touch the electrodes
    bg_rect = geo.region(geo.transform(geo.nominal_rect(bg.width_um.value, bg.length_um.value), hb.angle_deg, *hb.center))
    assert (fingers & bg_rect).is_empty()
    inp.outlines["BG"] = [geo.to_polys(bg_rect)[0]]
    report = stage2.check_step(spec, inp, design, cfg)
    assert not {i.code for i in report.errors} & {"finger_bg", "finger_bg_gap"}, [i.message for i in report.errors]


def test_presets_drive_configure(cfg):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "p")
    configure_with_test_answers(spec, cfg)
    aw = spec.hall_bar.get("arm_width_um")
    assert spec.hall_bar.get("channel_width_um") == pytest.approx(3 * aw)
    cfg.set_preset("hall_bar.width_per_arm", 4.0)       # TEST VALUE (the lab's own relation)
    cfg.set_preset("hall_bar.n_probe_pairs", 3)
    res = autoconfig.configure(spec, cfg)
    assert spec.hall_bar.get("channel_width_um") == pytest.approx(4 * aw) and spec.hall_bar.get("n_probe_pairs") == 3
    assert res.ok


def test_field_of_view_scales_with_the_objective(cfg):
    assert cfg.field_of_view_um("100x") == pytest.approx((143.28, 95.13))
    assert cfg.field_of_view_um("20x") == pytest.approx((716.4, 475.65))
    assert cfg.um_per_px("100x", 6096) == pytest.approx(0.0235, abs=1e-4)
    assert cfg.um_per_px("50x", 3048) == pytest.approx(286.56 / 3048)      # half-resolution 50x image
