"""Stage 1: configuration, templates, validation, flake list, assistant."""

import pytest

from conftest import graphene_answers
from ustack import assistant, flakes, templates, validation
from ustack.config import MissingValues, apply_answers
from ustack.model import DeviceSpec, Requirement


def test_config_statuses(cfg):
    # values from the user's request are approved; from the attached documents pending; unknown ones missing
    assert cfg.rules["ebl.electrode_width_target_um"].status == "approved"
    assert cfg.rules["ebl.contact_width_min_um"].status == "pending"
    assert cfg.rules["stack.substrate_breakdown_field_MV_per_cm"].value is None
    assert cfg.gds_layer("contacts") == (8, 0)
    with pytest.raises(MissingValues) as exc:
        cfg.gds_layer("hallbar_etch")
    assert exc.value.items[0].key == "layer:hallbar_etch"
    with pytest.raises(MissingValues):
        cfg.require("ebl.defect_clearance_um", "ebl.contact_width_min_um")


def test_answers_are_saved_as_approved(cfg):
    from ustack.config import LabConfig
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.25, "layer:hallbar_etch": (21, 0)})  # TEST VALUES
    again = LabConfig(cfg.root)
    assert again.rules["ebl.defect_clearance_um"].value == 1.25
    assert again.rules["ebl.defect_clearance_um"].status == "approved"
    assert again.gds_layer("hallbar_etch") == (21, 0)


def test_four_builtin_templates(cfg):
    ids = [t.id for t in templates.list_templates() if t.builtin]
    assert ids == ["xu_tmd", "mak_tmd", "xia_graphene", "ju_rhombohedral"]
    for tid in ids:
        spec = templates.load_template(tid)
        report = validation.validate_stage1(spec, cfg)
        assert not report.errors, (tid, [i.message for i in report.errors])
        assert report.missing, tid                  # nothing is pre-filled from the literature
        assert spec.layers[0].function == "substrate"
        for layer in spec.layers:                   # literature numbers are suggestions only
            for q in (layer.thickness_nm, layer.width_um, layer.length_um):
                assert q.value is None or q.source in ("material", "user")


def _graphene(cfg):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "test")
    validation.apply_spec_answers(spec, graphene_answers())
    return spec


def test_graphene_device_valid_after_values(cfg):
    spec = _graphene(cfg)
    report = validation.validate_stage1(spec, cfg)
    assert sorted(i.ask.key for i in report.missing) == ["rule:stack.alignment_shift_um",
                                                         "rule:stack.substrate_breakdown_field_MV_per_cm"]
    apply_answers(cfg, {"rule:stack.substrate_breakdown_field_MV_per_cm": 5.0, "rule:stack.alignment_shift_um": 2.0})  # TEST VALUES
    report = validation.validate_stage1(spec, cfg)
    assert report.ok, report.summary()
    needs = flakes.flake_list(spec, cfg, report)
    assert [f.material for f in needs] == ["Graphite", "hBN", "Graphene", "hBN"]
    assert needs[2].target == "monolayer"
    # the flake list adds the stacking misalignment on every side of the design size (GR is 26 × 16 µm)
    assert needs[2].min_size_um == (30.0, 20.0)


def test_flake_list_requires_valid_spec(cfg):
    spec = templates.load_template("xu_tmd")
    report = validation.validate_stage1(spec, cfg)
    with pytest.raises(flakes.SpecNotValid):
        flakes.flake_list(spec, cfg, report)


def test_unsupported_material_is_flagged_not_replaced(cfg):
    spec = _graphene(cfg)
    spec.layer("CT").material = "Ti/Pd"
    report = validation.validate_stage1(spec, cfg)
    assert any(i.code == "material_unsupported" for i in report.errors)
    assert spec.layer("CT").material == "Ti/Pd"


def test_hall_bar_never_shrunk(cfg):
    spec = _graphene(cfg)
    spec.hall_bar.params["channel_length_um"].set(40.0, "user")
    report = validation.validate_stage1(spec, cfg)
    assert any(i.code == "hall_bar_fit" for i in report.errors)
    assert spec.hall_bar.get("channel_length_um") == 40.0


def test_requirement_kept(cfg):
    spec = _graphene(cfg)
    spec.requirements = [Requirement("layer", "graphite back gate", {"function": "bottom_gate", "material": "graphite"})]
    assert not any(i.code == "requirement" for i in validation.validate_stage1(spec, cfg).issues)
    spec.layers = [l for l in spec.layers if l.uid != "BG"]
    assert any(i.code == "requirement" for i in validation.validate_stage1(spec, cfg).errors)


def test_gate_without_dielectric_is_an_error(cfg):
    spec = _graphene(cfg)
    spec.layers = [l for l in spec.layers if l.uid != "BBN"]
    codes = {i.code for i in validation.validate_stage1(spec, cfg).errors}
    assert "gate_short" in codes


def test_substrate_breakdown(cfg):
    spec = _graphene(cfg)
    apply_answers(cfg, {"rule:stack.substrate_breakdown_field_MV_per_cm": 1.0})  # TEST VALUE
    spec.gate_limits_V["S0"].set(60.0, "user")   # 60 V / 285 nm = 2.1 MV/cm
    assert any(i.code == "substrate_breakdown" for i in validation.validate_stage1(spec, cfg).errors)


def _xu(cfg):
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "xu")
    ans = {"spec:S0.material": "SiO2_285_doped_Si", "spec:gate.S0": 10,
           "spec:BG.thickness_nm": 6, "spec:BG.width_um": 10, "spec:BG.length_um": 40,
           "spec:BBN.thickness_nm": 30, "spec:BBN.width_um": 40, "spec:BBN.length_um": 50,
           "spec:PT.metal.Ti": 0, "spec:PT.metal.Pt": 8,
           "spec:TMD1.layer_count": 1, "spec:TMD1.width_um": 14, "spec:TMD1.length_um": 24,
           "spec:TMD2.layer_count": 1, "spec:TMD2.width_um": 14, "spec:TMD2.length_um": 24, "spec:TMD2.twist_deg": 3.65,
           "spec:MBN.thickness_nm": 4, "spec:MBN.width_um": 18, "spec:MBN.length_um": 30,
           "spec:CG.metal.Ti": 0, "spec:CG.metal.Pt": 8, "spec:LEADS.metal.Cr": 5, "spec:LEADS.metal.Au": 70,
           "spec:TBN.thickness_nm": 15, "spec:TBN.width_um": 20, "spec:TBN.length_um": 30,
           "spec:TG.thickness_nm": 5, "spec:TG.width_um": 12, "spec:TG.length_um": 30,
           "spec:CAP.thickness_nm": 40, "spec:CAP.width_um": 25, "spec:CAP.length_um": 35,
           "spec:hall_bar.channel_width_um": 2, "spec:hall_bar.channel_length_um": 6, "spec:hall_bar.n_probe_pairs": 2,
           "spec:hall_bar.arm_width_um": 0.8, "spec:hall_bar.arm_length_um": 3, "spec:hall_bar.sd_extension_um": 2}
    validation.apply_spec_answers(spec, ans)
    apply_answers(cfg, {"rule:stack.substrate_breakdown_field_MV_per_cm": 5.0, "rule:xu.middle_hbn_max_thickness_nm": 10,
                        "rule:stack.alignment_shift_um": 2.0})  # TEST VALUES
    return spec


def test_xu_rules(cfg):
    spec = _xu(cfg)
    assert validation.validate_stage1(spec, cfg).ok
    # middle hBN only 0.5 µm larger than the TMD on each side → violates the 1 µm rule
    spec.layer("MBN").width_um.set(15, "user")
    assert any(i.code == "xu_middle_margin" for i in validation.validate_stage1(spec, cfg).errors)
    spec.layer("MBN").width_um.set(18, "user")
    spec.layer("CAP").thickness_nm.set(10, "user")   # thinner than the bottom hBN: no rule on relative hBN thickness
    assert validation.validate_stage1(spec, cfg).ok
    spec.layer("MBN").thickness_nm.set(12, "user")   # above the (test) thin limit
    assert any(i.code == "xu_middle_thin" for i in validation.validate_stage1(spec, cfg).errors)


def test_offline_assistant_example(cfg):
    d = assistant.parse_offline("an hBN-encapsulated monolayer graphene Hall bar with a graphite back gate and edge contacts", cfg)
    spec = d.spec
    assert spec.family == "xia_graphene" and spec.contact_scheme == "edge_1d"
    assert not spec.by_function("top_gate")
    assert spec.device_layers()[0].layer_count.value == 1
    texts = [r.text for r in spec.requirements]
    assert "graphite bottom gate" in texts and "edge (1D) contacts" in texts
    assert not validation.validate_stage1(spec, cfg).errors


def test_spec_roundtrip(tmp_path, cfg):
    spec = _graphene(cfg)
    path = spec.save(tmp_path / "d.json", "test")
    again = DeviceSpec.load(path)
    assert again.to_json()["layers"] == spec.to_json()["layers"]
    assert again.hall_bar.get("channel_width_um") == 2


def test_user_template(cfg):
    spec = _graphene(cfg)
    path = templates.save_as_template(spec, "Lab graphene v1")
    assert any(t.path == path and not t.builtin for t in templates.list_templates())


def test_numeric_requirements_compare_values(cfg):
    spec = _graphene(cfg)
    spec.requirements = [Requirement("layer", "monolayer graphene", {"material": "graphene", "layer_count": 1}),
                         Requirement("param", "W = 2 µm", {"key": "channel_width_um", "value": 2})]
    assert not any(i.code == "requirement" for i in validation.validate_stage1(spec, cfg).issues)
    spec.layer("GR").layer_count.set(3, "user")
    assert any("monolayer graphene" in i.message for i in validation.validate_stage1(spec, cfg).errors)


def test_offline_monolayer_request_pins_layer_count(cfg):
    spec = assistant.parse_offline("an hBN-encapsulated monolayer graphene Hall bar with a graphite back gate", cfg).spec
    assert not any(i.code == "requirement" for i in validation.validate_stage1(spec, cfg).issues)
    spec.device_layers()[0].layer_count.set(3, "user")
    assert any(i.code == "requirement" for i in validation.validate_stage1(spec, cfg).errors)


def test_assistant_draft_provenance_and_roles(cfg):
    data = {"name": "d", "family": "xia_graphene", "contact_scheme": "edge_1d",
            "layers": [{"name": "Si", "material": "SiO2_285_doped_Si", "function": "substrate"},
                       {"name": "BG", "material": "graphite", "function": "bottom_gate"},
                       {"name": "bot", "material": "hBN", "function": "bottom_gate_dielectric", "thickness_nm": 30},
                       {"name": "gr", "material": "graphene", "function": "device_layer", "layer_count": 1},
                       {"name": "top", "material": "hBN", "function": "top_gate_dielectric"}],
            "hall_bar": {"channel_width_um": 2}}
    spec = assistant._spec_from_json(data, cfg)
    assert spec.layers[2].thickness_nm.source == "assistant"
    assert spec.hall_bar.params["channel_width_um"].source == "assistant"
    assert [l.role for l in spec.layers][2:] == ["bottom_hbn", "graphene", "tg_hbn"]


def test_prepatterned_contacts_must_touch_device(cfg):
    """Mak: two electrode columns on the chip, the BG in the gap, the WSe2 reaching past the bottom hBN onto them."""
    spec = templates.load_template("mak_tmd")
    assert [l.uid for l in spec.layers[:4]] == ["S0", "PT", "BG", "BBN"] and spec.hall_bar.sd_style == "bent"
    codes = lambda: {i.code for i in validation.validate_stage1(spec, cfg).errors}
    assert not codes() & {"contact_not_touching", "gate_in_gap"}
    spec.hall_bar.params["channel_width_um"].set(6.0, "user")       # electrode gap
    spec.layer("BG").width_um.set(5.0, "user")                       # too wide for the gap with 0.8 µm each side
    spec.layer("BBN").width_um.set(7.0, "user")
    spec.layer("TMD1").width_um.set(6.5, "user")                     # narrower than the bottom hBN: cannot reach the Pt
    assert {"contact_not_touching", "gate_in_gap"} <= codes()
    spec.layer("BG").width_um.set(4.0, "user")
    spec.layer("TMD1").width_um.set(10.0, "user")
    assert not codes() & {"contact_not_touching", "gate_in_gap"}
    # straight S/D (one finger set under the device) needs the electrodes directly against the device
    spec.hall_bar.sd_style = "straight"
    assert "contact_not_touching" in codes()
