"""v0.5: the finger-array pattern (fit to the twisted region, comb, gate contacts, electrodes), the
Stage 2 step workflow (generate → edit → export, merged export with the bond-pad preset), the
bond-pad presets and the measure tool."""

import math
import os

import numpy as np
import pytest

from ustack import bondpads, ebl_flow as ef, fingers as fg, geometry as geo, templates
from ustack.config import MissingValues, apply_answers
from ustack.gds_io import read_polygons

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FLAKES = {"TMD1": [geo.transform([(-12, -8), (10, -8), (10, 6), (-12, 6)], 10)],
          "TMD2": [geo.transform([(-8, -10), (14, -10), (14, 8), (-8, 8)], 25)],
          "BG": [geo.transform([(-60, -4), (60, -4), (60, 4), (-60, 4)], 15)]}
LAYERS = {"layer:prepattern_contacts": (5, 0), "layer:contact_gates": (6, 0), "layer:electrodes": (7, 0),
          "layer:bg_contacts": (7, 0), "layer:tg_contacts": (7, 0)}   # TEST VALUES (the lab's tWSe2 drawings)


def test_fit_counts_the_pairs_that_fit_and_finds_the_angle():
    p = fg.FingerParams()          # 0.8 µm fingers, 2.2 µm pitch, 4 µm gap, 0.8 µm of each tip on the region
    r = fg.fit_finger_array([[(0, 0), (20, 0), (20, 12), (0, 12)]], p)
    assert r.n == 9 == math.floor((20 - p.width) / p.pitch) + 1
    assert r.array.angle_deg % 180 == pytest.approx(0, abs=0.3)
    assert r.array.center[0] == pytest.approx(10, abs=0.1) and r.array.center[1] == pytest.approx(6, abs=0.1)
    rot = fg.fit_finger_array([geo.transform([(0, 0), (20, 0), (20, 12), (0, 12)], 30, 5, 5)], p)
    assert rot.n == 9 and rot.array.angle_deg % 180 == pytest.approx(30, abs=0.6)
    assert fg.fit_finger_array([[(0, 0), (20, 0), (20, 12), (0, 12)]], p, n_max=4).n == 4
    with pytest.raises(ValueError):                       # 5.6 µm strip needed, 5 µm available: nothing is shrunk
        fg.fit_finger_array([[(0, 0), (20, 0), (20, 5), (0, 5)]], p, angle_deg=0.0)
    turned = fg.fit_finger_array([[(0, 0), (20, 0), (20, 5), (0, 5)]], p)    # free angle: the array turns across the strip
    assert turned.n == 2 and turned.array.angle_deg % 180 == pytest.approx(90, abs=0.6)
    # every pair strip lies inside the region
    reg = geo.region([(0, 0), (20, 0), (20, 12), (0, 12)])
    assert all((geo.region(r.array.strip(i)) - reg).is_empty() for i in range(r.n))


@pytest.mark.parametrize("angle", [0.0, 10.0, 37.0, 97.0])
def test_fingers_and_comb_are_drc_clean_at_any_angle(angle):
    arr = fg.FingerArray((3.0, 4.0), angle, 12, fg.FingerParams())
    fs = arr.fingers()
    assert len(fs) == 24
    fr = geo.regions([f.polygon for f in fs]).merged()
    assert fr.count() == 24                               # no finger touches another
    assert not geo.width_violations(fr, 0.8) and not geo.space_violations(fr, 0.8)
    comb = geo.regions([c["polygon"] for c in fg.contact_gate_comb(arr).values()]).merged()
    assert not geo.width_violations(comb, 0.8) and not geo.space_violations(comb, 0.8)
    # tips 4 µm apart, 0.8 µm wide, 2.2 µm pitch
    t0, b0 = fs[0].tip, fs[12].tip
    assert math.dist(t0, b0) == pytest.approx(4.0, abs=1e-6)
    assert math.dist(fs[0].tip, fs[1].tip) == pytest.approx(2.2, abs=1e-6)


def test_bondpad_preset_and_saving_one(cfg, tmp_path):
    bp = bondpads.default_preset()
    assert bp is not None and bp.builtin and bp.layers == {"landing": (1, 0), "markers": (3, 0), "pads": (4, 0)}
    assert len(bp.pad_tips()) == 48
    saved = bondpads.save_preset("My chip", bp.gds, {"pads": (4, 0), "markers": (3, 0)}, (10.0, -5.0))
    assert saved.json_path.is_file() and bondpads.load_preset("My chip").site == (10.0, -5.0)


def test_design_frame_flakes_are_projected_onto_the_chip():
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "proj")
    design = {"BG": [(-30, -3), (30, -3), (30, 3), (-30, 3)], "TMD1": [(-8, -6), (8, -6), (8, 6), (-8, 6)],
              "BBN": [(-20, -15), (20, -15), (20, 15), (-20, 15)]}
    for uid, poly in design.items():
        spec.layer(uid).outline = poly
        spec.layer(uid).image = {"path": "x.png", "placed": True}    # placed on each other in Flake layout
    st = ef.FlowStep("hall_bars")
    # BG and BBN "located": the design frame rotated by 20° and moved to (100, 50), mirrored
    T = {"m": [[-math.cos(math.radians(20)), -math.sin(math.radians(20))], [-math.sin(math.radians(20)), math.cos(math.radians(20))]],
         "t": [100.0, 50.0]}
    for uid in ("BG", "BBN"):
        st.outlines[uid] = [ef.apply_design_to_chip(T, design[uid])]
        st.located[uid] = {"how": "located", "score": 0.9}
    fit = ef.project_flakes(spec, st)
    assert fit["mirror"] and fit["rms_um"] < 1e-6
    assert st.located["TMD1"]["how"] == "projected"
    for a, b in zip(st.outlines["TMD1"][0], ef.apply_design_to_chip(T, design["TMD1"])):
        assert a == pytest.approx(b, abs=1e-6)
    # a flake never placed in Flake layout is not carried over (its frame is its own)
    spec.layer("TMD1").image = {"path": "x.png"}
    st2 = ef.FlowStep("hall_bars")
    st2.outlines = {u: list(v) for u, v in st.outlines.items() if u != "TMD1"}
    st2.located = {u: st.located[u] for u in ("BG", "BBN")}
    ef.project_flakes(spec, st2)
    assert "TMD1" not in st2.outlines
    # located flakes that disagree with the layout: nothing is projected
    st3 = ef.FlowStep("hall_bars")
    st3.outlines = {"BG": st.outlines["BG"], "BBN": [geo.transform(st.outlines["BBN"][0], 0, 40, 0)]}
    st3.located = {u: {"how": "located"} for u in ("BG", "BBN")}
    spec.layer("TMD1").image = {"path": "x.png", "placed": True}
    assert ef.project_flakes(spec, st3)["rejected"] and "TMD1" not in st3.outlines


def _flow_spec():
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "flow test")
    flow = ef.Flow(bondpads=bondpads.default_preset().name)
    for k in ("hall_bars", "contact_gate", "electrodes"):
        flow.step(k).outlines = {u: [list(p) for p in polys] for u, polys in FLAKES.items()}
    return spec, flow


#: a graphite top gate across the device (the Xu stack has one), located at the electrodes step only
TG_STRIP = [geo.transform([(-60, -12), (60, -12), (60, 12), (-60, 12)], 100)]


def _tg_strip(flow) -> list:
    """A graphite top gate over the channel between the finger columns, running out along the array to one side and
    widening there (where a top gate lies and is contacted: not over the fingers' far ends; the bottom gate stays
    uncovered on the other side)."""
    arr = fg.FingerArray.from_json(flow.step("hall_bars").array)
    return [arr.to_chip([(-20.0, -1.5), (30.0, -1.5), (30.0, -6.0), (70.0, -6.0), (70.0, 6.0), (30.0, 6.0), (30.0, 1.5),
                         (-20.0, 1.5)])]


def test_every_step_generates_checks_and_exports(cfg, tmp_path):
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    arr = fg.FingerArray.from_json(hb.array)
    assert arr.n >= 6 and len(hb.shapes["prepattern_contacts"]) == 2 * arr.n
    assert not [i for i in hb.issues if i["severity"] == "error"]
    assert any(i["code"] in ("bg_under_gap", "bg_not_under_gap") for i in hb.issues)
    with pytest.raises(MissingValues) as exc:                                   # how close the bus comes to the TMD: asked
        ef.generate(spec, flow, "contact_gate", cfg, bp)
    assert exc.value.items[0].key == "preset:cg.bus_tmd_gap_um"
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE
    cg = ef.generate(spec, flow, "contact_gate", cfg, bp)
    assert len(cg.shapes["contact_gates"]) == 2 and not [i for i in cg.issues if i["severity"] == "error"]
    # the electrodes: one group per structure in the stack, each generated on its own, one GDS layer for all
    assert ef.electrode_groups(spec) == ["fingers", "contact_gate", "top_gate", "bottom_gate"]
    assert "bottom_gate" not in ef.STEPS and "top_gate" not in ef.STEPS
    with pytest.raises(MissingValues) as exc:                         # the landing square on the bond-pad leads: asked
        ef.generate(spec, flow, "electrodes", cfg, bp)
    assert exc.value.items[0].key == "preset:electrodes.pad_landing_um"
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})    # TEST VALUE
    with pytest.raises(ValueError, match="not located"):              # the top gate is not located yet
        ef.generate(spec, flow, "electrodes", cfg, bp, group="top_gate")
    flow.step("electrodes").outlines["TG"] = _tg_strip(flow)
    el = ef.generate(spec, flow, "electrodes", cfg, bp)                # all groups (the gates first)
    assert all(el.shapes[ef.ELECTRODE_GROUPS[g][1]] for g in ef.electrode_groups(spec))
    assert len(el.shapes["el_tg"]) == 1
    before = {k: list(v) for k, v in el.shapes.items()}
    el = ef.generate(spec, flow, "electrodes", cfg, bp, group="bottom_gate")
    assert all(el.shapes[k] == before[k] for k in before if k != "el_bg")   # the other groups are kept
    assert len(el.shapes["el_bg"]) == 1                                # one electrode from the bottom gate to a bond pad
    for purpose, gate in (("el_tg", "TG"), ("el_bg", "BG")):          # the contact lies on its gate flake
        assert not (geo.regions(el.shapes[purpose]) & geo.regions(flow.step("electrodes").outlines[gate])).is_empty()
    assert len(el.routes) == 2 * arr.n + 2 + 1 + 1 and all(r["routed_to_pad"] for r in el.routes)
    tips = [r["pad_tip"] for r in el.routes]
    assert len(set(tips)) == len(tips)                             # one pad lead per electrode
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "electrodes", cfg)}
    assert rows["Bottom gate electrodes drawn"]["status"] == "met" and rows["Top gate electrodes drawn"]["status"] == "met"
    assert rows["Bottom gate electrodes on the exposed Graphite bottom gate"]["status"] == "met"
    # routed (user, 2026-11-20): no two electrodes touch, none on another gate's uncovered flake, each ends in its square on
    # its own bond-pad lead, the spacing kept
    assert rows["Electrodes do not cross each other"]["status"] == "met"
    assert rows["Other electrodes off the exposed Graphite bottom gate"]["status"] == "met"
    assert rows["Other electrodes off the exposed Graphite top gate"]["status"] == "met"
    assert rows["Electrodes land on their bond-pad leads"]["status"] == "met"
    assert rows["Electrodes: spacing ≥ hard minimum"]["status"] == "met"
    ends = bp.pad_ends()
    for r in el.routes:                                               # the landing square over its own lead's end
        e = ends[r["pad_tip"]]
        assert r["points_um"][-1] == pytest.approx(list(e["mid"]), abs=1e-3)
        mine = geo.regions([q for purpose in ef.ELECTRODE_PURPOSES for q in el.shapes.get(purpose, [])
                            if geo.point_in_poly(e["mid"][0], e["mid"][1], q)])
        assert geo.area(mine & geo.region(e["poly"])) >= 0.9 * min(10.0, e["width"]) * 5.0
    # the layers are asked for (with the lab's numbers as suggestions only)
    with pytest.raises(MissingValues) as exc:
        ef.export_step(spec, flow, "hall_bars", cfg, tmp_path)
    item = exc.value.items[0]
    assert item.key == "layer:prepattern_contacts" and item.suggested == [5, 0] and cfg.layers.get("prepattern_contacts").is_missing
    apply_answers(cfg, LAYERS)
    out = ef.export_step(spec, flow, "hall_bars", cfg, tmp_path)
    assert set(read_polygons(out)) == {(5, 0)} and flow.step("hall_bars").status == "exported"
    merged = ef.export_merged(spec, flow, list(ef.STEPS), cfg, bp, tmp_path / "merged.gds")
    got = read_polygons(merged)
    assert {(1, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)} <= set(got)
    assert len(got[(5, 0)]) == 2 * arr.n and len(got[(4, 0)]) == 48
    # state round-trips through the spec
    flow.store(spec)
    again = ef.Flow.of(spec)
    assert again.step("electrodes").routes == el.routes and again.bondpads == flow.bondpads


def test_flow_panel_generates_and_edits_on_the_canvas(cfg):
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE (no dialog in a test)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    names = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert names[-1] == "Stage 2" and w.tabs.widget(names.index("Stage 2")) is p
    p.select_step("hall_bars")
    assert p.step_card.title_label.text().lower().startswith("step 1: hall bars")
    p.step.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p.twisted_from_flakes()
    p.generate()
    st = p.step
    key = "pat:hall_bars|prepattern_contacts#0"
    assert key in w.ebl_canvas.polys
    moved = [(x + 0.5, y) for x, y in st.shapes["prepattern_contacts"][0]]
    w.ebl_canvas.polygonEdited.emit(key, moved)                     # dragging the shape on the canvas
    assert st.shapes["prepattern_contacts"][0] == moved
    p.select_step("contact_gate")                                   # its own card opens below; flakes carried over
    assert p.keys == ["hall_bars", "contact_gate"] and p.key == "contact_gate" and set(p.cards) == {"hall_bars", "contact_gate"}
    assert set(p.step.outlines) == set(FLAKES)
    # each ticked step has its own card: its own images, flakes, Generate, issues
    p.set_steps(["hall_bars", "contact_gate", "electrodes"])
    assert p.keys == ["hall_bars", "contact_gate", "electrodes"] and set(p.cards) == set(p.keys)
    assert all(p.cards[k]["flake_list"] is not None and p.cards[k]["pill"] is not None for k in p.keys)
    p.flow.step("electrodes").outlines["TG"] = _tg_strip(p.flow)
    for k in p.keys:
        p._activate(k)                                              # a card's button puts that card in use
        assert p.key == k and p.flake_list is p.cards[k]["flake_list"]
        p.generate()
    assert all(p.flow.step(k).shapes for k in p.keys)
    # the electrodes card: one card per electrode group of this sample, each with its own Generate; one Export
    from ustack.ui.style import QtWidgets as _W
    titles = [c.title_label.text() for c in p.cards["electrodes"]["step_card"].findChildren(type(p.cards["electrodes"]["step_card"]))]
    assert all(t in " ".join(titles) for t in ("Hall Bar Electrodes", "Contact Gate Electrodes", "Top Gate Electrodes",
                                               "Bottom Gate Electrodes")) or \
        all(t.lower() in " ".join(titles).lower() for t in ("hall bar electrodes", "contact gate electrodes", "top gate electrodes",
                                                             "bottom gate electrodes"))
    gens = [b for b in p.cards["electrodes"]["step_card"].findChildren(_W.QPushButton) if b.text() == "Generate"]
    exports = [b for b in p.cards["electrodes"]["step_card"].findChildren(_W.QPushButton) if b.text().startswith("Export GDS")]
    assert len(gens) == 4 and len(exports) == 1
    p._activate("electrodes")
    p.clear_electrodes("top_gate")
    assert not p.flow.step("electrodes").shapes.get("el_tg") and p.flow.step("electrodes").shapes.get("el_fingers")
    # a top gate lying where the Hall-bar electrodes run (60 µm out, among their leads): no room for its contact while
    # they are there
    el_now = p.flow.step("electrodes")
    arr_now = fg.FingerArray.from_json(p.flow.step("hall_bars").array)
    r0 = next(r for r in el_now.routes if r["group"] == "fingers")
    pts = r0["points_um"]
    ox, oy = arr_now.center
    cx = cy = None
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):                   # where the lead is 60 µm from the array's centre
        da, db = math.hypot(ax - ox, ay - oy), math.hypot(bx - ox, by - oy)
        if da <= 60.0 <= db:
            t = (60.0 - da) / max(db - da, 1e-9)
            cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
            break
    assert cx is not None
    el_now.outlines["TG"] = [geo.rect(cx - 5, cy - 5, cx + 5, cy + 5)]
    # the Hall-bar electrodes cross the top gate now: asked, then the top gate is drawn first and the others after it
    asked = []
    orig_q = _W.QMessageBox.question
    _W.QMessageBox.question = staticmethod(lambda *a, **k: asked.append(a[2]) or _W.QMessageBox.StandardButton.Yes)
    try:
        p.generate("top_gate")
    finally:
        _W.QMessageBox.question = orig_q
    assert asked and "cross the gate flake" in asked[0]
    el_st = p.flow.step("electrodes")
    assert len(el_st.shapes["el_tg"]) == 1 and el_st.shapes["el_fingers"] and el_st.shapes["el_cg"]
    assert all(f"pat:{k}|" in " ".join(w.ebl_canvas.polys) for k in p.keys)
    apply_answers(cfg, LAYERS)
    out = ef.export_steps(spec, p.flow, p.keys, cfg, p.app.device_dir() / "stage2" / "combo")
    assert {(5, 0), (6, 0), (7, 0)} == set(read_polygons(out)) and all(p.flow.step(k).exported == str(out) for k in p.keys)
    ticks = {w_.data(0, 0x0100): w_.checkState(0).value for w_ in (p.step_list.topLevelItem(i) for i in range(p.step_list.topLevelItemCount()))}
    assert ticks["contact_gate"] == 2 and ticks["hall_bars"] == 2 and list(ticks)[:3] == ["hall_bars", "contact_gate", "electrodes"]
    # reorder: the electrodes before the contact gate
    row = next(i for i in range(p.step_list.topLevelItemCount()) if p.step_list.topLevelItem(i).data(0, 0x0100) == "electrodes")
    p.step_list.setCurrentItem(p.step_list.topLevelItem(row))
    p.move_step(-1)
    assert p.flow.order == ["hall_bars", "electrodes", "contact_gate"] and p.keys == ["hall_bars", "electrodes", "contact_gate"]
    assert p.cards["electrodes"]["step_card"].title_label.text().startswith("Step 2:")
    assert ef.Flow.of(w.spec).order == ["hall_bars", "electrodes", "contact_gate"]
    # fold a card with its ^, unfold it
    p.toggle_card("electrodes")
    assert p.cards["electrodes"].get("flake_list") is None
    p.toggle_card("electrodes")
    assert p.cards["electrodes"]["flake_list"] is not None
    p.select_step("classic")
    assert p.stack.currentIndex() == 1 and w.stage2_panel.active
    p.select_step(None)
    w.dirty = False
    w.close()


def test_measure_tool_on_the_viewer():
    import numpy as np
    from ustack.app import make_app
    from ustack.ui.style import QtCore, setup_application
    from ustack.ui.viewer import ImageViewerDialog
    app = make_app()
    setup_application(app)
    dlg = ImageViewerDialog(np.zeros((200, 300, 3), np.uint8), "t", "", 0.5)
    dlg.show()
    dlg.tools["measure"].setChecked(True)
    c = dlg.canvas
    assert c.tool == "measure"
    got = []
    c.measured.connect(got.append)
    c._ruler_click(QtCore.QPointF(10, 20))
    c._ruler_click(QtCore.QPointF(13, 24))
    assert got == [pytest.approx(5.0)] and c.ruler_length() == pytest.approx(5.0)
    assert c._ruler_items[-1].text == "5.00 µm"
    c._ruler_click(QtCore.QPointF(0, 0))                            # a third click starts a new measurement
    assert c.ruler_length() is None
    dlg.tools["pointer"].setChecked(True)
    assert not c._ruler_items
    # double-click on the image clears a measurement
    from PyQt6.QtTest import QTest
    dlg.tools["measure"].setChecked(True)
    c._ruler_click(QtCore.QPointF(10, 20))
    c._ruler_click(QtCore.QPointF(13, 24))
    vp = c.mapFromScene(QtCore.QPointF(50, 50))
    QTest.mouseDClick(c.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, vp)
    assert not c._ruler_items and c.ruler_length() is None
    # pointer: click and drag moves the view
    dlg.tools["pointer"].setChecked(True)
    c.zoom_by(4.0)
    before = c.mapToScene(c.viewport().rect().center())
    QTest.mousePress(c.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, QtCore.QPoint(100, 100))
    QTest.mouseMove(c.viewport(), QtCore.QPoint(160, 130))
    QTest.mouseRelease(c.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, QtCore.QPoint(160, 130))
    after = c.mapToScene(c.viewport().rect().center())
    assert after.x() < before.x() - 1e-6 and after.y() > before.y() + 1e-6   # dragged right/down: content followed the mouse
    dlg.close()


def test_undo_and_redo(cfg):
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "undo test")
    w.adopt_spec(spec, "new")
    assert not w.undo_button.isEnabled() and not w.redo_button.isEnabled()
    w.spec.layer("BBN").thickness_nm.set(31.0, "user")
    w.spec_changed("value")
    app.processEvents()
    w.spec.layer("BBN").thickness_nm.set(32.0, "user")
    w.spec_changed("value")
    w.spec_changed("value")                          # several calls in one operation = one step
    app.processEvents()
    assert len(w._undo) == 2 and w.undo_button.isEnabled()
    w.undo()
    assert w.spec.layer("BBN").thickness_nm.value == 31.0 and w.redo_button.isEnabled()
    w.undo()
    assert w.spec.layer("BBN").thickness_nm.value is None or w.spec.layer("BBN").thickness_nm.value != 31.0
    w.redo()
    w.redo()
    assert w.spec.layer("BBN").thickness_nm.value == 32.0 and not w.redo_button.isEnabled()
    # Stage 2 flow edits are undone too (they live in the spec)
    p = w.ebl_flow
    p.select_step("hall_bars")
    app.processEvents()
    p.step.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p.twisted_from_flakes()
    p.generate()
    app.processEvents()
    assert p.step.shapes
    w.undo()
    assert not w.ebl_flow.flow.step("hall_bars").shapes
    w.redo()
    assert w.ebl_flow.flow.step("hall_bars").shapes
    # a new edit clears the redo list
    w.undo()
    w.spec.layer("BBN").thickness_nm.set(40.0, "user")
    w.spec_changed("value")
    app.processEvents()
    assert not w._redo
    w.dirty = False
    w.close()


def test_3d_view_separation_rotation_and_moving():
    from ustack.app import make_app
    from ustack.scene3d import build_scene
    from ustack.config import LabConfig
    from ustack.ui.stack3d import Stack3DView
    from ustack.ui.style import QtCore, QtGui, setup_application
    app = make_app()
    setup_application(app)
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "3d")
    for i, l in enumerate(spec.layers):
        l.width_um.set(20.0 + i, "user")
        l.length_um.set(30.0 - i, "user")
    v = Stack3DView()
    v.resize(900, 600)
    v.set_slabs(build_scene(spec, LabConfig()))
    v.show()
    scale = v._layout()[1]
    v.set_explode(4.0)                                    # Separate Layers: z only
    assert v._layout()[1] == pytest.approx(scale)
    assert v._heights()[-1][0] > v.slabs[-1].z0
    for el in (-60.0, 0.0, 120.0, 179.0):                 # any angle, also from below / over the top
        v.el = math.radians(el)
        v.grab()
    v.el = math.radians(85.0)
    v._drag = QtCore.QPointF(100, 100)
    ev = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(100, 160), QtCore.QPointF(100, 160),
                           QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    v.mouseMoveEvent(ev)
    assert math.degrees(v.el) == pytest.approx(85.0 + 60 * 0.01 * 180 / math.pi)   # past 90°: no clamp
    # two-finger (right-button) drag and trackpad scroll move the stack; zoom is untouched
    press = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress, QtCore.QPointF(300, 300), QtCore.QPointF(300, 300),
                              QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier)
    v.mousePressEvent(press)
    move = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(340, 320), QtCore.QPointF(340, 320),
                             QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.KeyboardModifier.NoModifier)
    v.mouseMoveEvent(move)
    v.mouseReleaseEvent(press)
    assert (v.pan.x(), v.pan.y()) == (40.0, 20.0)
    zoom = v.zoom
    v.wheelEvent(QtGui.QWheelEvent(QtCore.QPointF(300, 300), QtCore.QPointF(300, 300), QtCore.QPoint(5, -10), QtCore.QPoint(40, -80),
                                   QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                                   QtCore.Qt.ScrollPhase.ScrollUpdate, False))
    assert (v.pan.x(), v.pan.y()) == (45.0, 10.0) and v.zoom == zoom
    v.wheelEvent(QtGui.QWheelEvent(QtCore.QPointF(300, 300), QtCore.QPointF(300, 300), QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                                   QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                                   QtCore.Qt.ScrollPhase.NoScrollPhase, False))
    assert v.zoom > zoom                                   # a mouse wheel still zooms
    v.reset_view()
    assert v.pan.isNull()
    v.close()


def test_locate_aligns_new_images_and_falls_back_to_low_magnification(cfg, tmp_path, monkeypatch):
    """A 50x image added after the 5x was aligned is aligned by Locate first; with only a 5x / 10x
    image aligned, Locate still runs on it (coarse, and says so)."""
    import cv2
    import numpy as np
    from ustack import registration
    from ustack.alignment import Similarity
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    for uid, polys in FLAKES.items():
        if spec.layer(uid) is not None:
            spec.layer(uid).outline = [list(q) for q in polys[0]]
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("hall_bars")
    for obj in ("5x", "50x"):
        f = tmp_path / f"{obj}.png"
        cv2.imwrite(str(f), np.full((60, 90, 3), 128, np.uint8))
        p.step.images[obj] = str(f)
    p.step.transforms = {"5x": Similarity(1.0, 0.0, 0.0, 0.0).to_json()}     # the 5x was aligned before the 50x came
    aligned, used, warned = [], [], []

    def fake_align():
        aligned.append(sorted(p.step.images))
        p.step.transforms = {o: Similarity(1.0, 0.0, 0.0, 0.0).to_json() for o in p.step.images}
        return True

    def fake_locate(img, sim, outline, *a, **k):
        used.append(img.shape)
        return {"polygon": outline, "score": 0.9, "margin": 0.5, "mirror": False}

    monkeypatch.setattr(p, "align", fake_align)
    monkeypatch.setattr(registration, "locate_outline", fake_locate)
    monkeypatch.setattr(w, "warn", lambda title, text: warned.append(text))
    p.locate()
    assert aligned == [["50x", "5x"]] and used and not warned
    assert {v["image"] for v in p.step.located.values() if v.get("how") == "located"} == {"50x"}
    # only the 5x aligned (the 50x failed to align): Locate uses the 5x and says the outlines are coarse
    p.step.located = {}
    monkeypatch.setattr(p, "align", lambda: False)
    p.step.transforms = {"5x": Similarity(1.0, 0.0, 0.0, 0.0).to_json()}
    msgs = []
    monkeypatch.setattr(w, "status", lambda text, *a, **k: msgs.append(text))
    p.locate()
    assert {v["image"] for v in p.step.located.values() if v.get("how") == "located"} == {"5x"} and not warned
    assert any("coarse" in m for m in msgs)
    w.dirty = False
    w.close()


def test_locate_reuses_flakes_found_at_earlier_steps(cfg, tmp_path, monkeypatch):
    """Locate at a later step looks only for the flakes not located yet; flakes found at an earlier step keep that place
    (buried, they may not be visible any more). A flake cleared at the step is looked for again."""
    import cv2
    import numpy as np
    from ustack import registration
    from ustack.alignment import Similarity
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    for uid, polys in FLAKES.items():
        if spec.layer(uid) is not None:
            spec.layer(uid).outline = [list(q) for q in polys[0]]
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    looked = []

    def fake_locate(img, sim, outline, *a, **k):
        looked.append(next(l.uid for l in w.spec.layers if l.outline and [tuple(q) for q in l.outline] == list(outline)))
        return {"polygon": [(x + 1.0, y) for x, y in outline], "score": 0.9, "margin": 0.5, "mirror": False}

    monkeypatch.setattr(registration, "locate_outline", fake_locate)
    monkeypatch.setattr(registration, "locate_new_outline", lambda *a, **k: (_ for _ in ()).throw(ValueError("no change map")))
    monkeypatch.setattr(w, "warn", lambda *a, **k: None)
    for key in ("hall_bars", "contact_gate"):
        p.select_step(key)
        p._activate(key)
        f = tmp_path / f"{key}.png"
        cv2.imwrite(str(f), np.full((60, 90, 3), 128, np.uint8))
        p.step.images = {"50x": str(f)}
        p.step.transforms = {"50x": Similarity(1.0, 0.0, 0.0, 0.0).to_json()}
    p._activate("hall_bars")
    p.locate()
    first = [u for u in ef.present_layers(w.spec, "hall_bars") if w.spec.layer(u).outline]
    assert first and sorted(looked) == sorted(first)
    hb = p.flow.step("hall_bars")
    hb.outlines[first[0]] = [[(x + 5.0, y) for x, y in hb.outlines[first[0]][0]]]     # corrected by hand at the Hall-bar step
    looked.clear()
    p._activate("contact_gate")
    p.locate()
    later = [u for u in ef.present_layers(w.spec, "contact_gate") if w.spec.layer(u).outline and u not in first]
    assert later and sorted(looked) == sorted(later)                   # only the flakes new since the Hall-bar step
    cg = p.flow.step("contact_gate")
    for u in first:                                                    # the others: where the Hall-bar step has them
        assert cg.located[u]["inherited"] == "hall_bars" and cg.outlines[u] == hb.outlines[u]
    # nothing left to look for: Locate only refreshes the reused ones
    looked.clear()
    p.locate()
    assert not looked
    # cleared at this step: looked for again in this step's images
    p.flake_list.setCurrentItem(next(p.flake_list.topLevelItem(i) for i in range(p.flake_list.topLevelItemCount())
                                     if p.flake_list.topLevelItem(i).data(0, 0x0100) == first[0]))
    p.clear_outline(first[0])
    p.locate()
    assert looked == [first[0]] and cg.located[first[0]]["how"] == "located" and "inherited" not in cg.located[first[0]]
    w.dirty = False
    w.close()


def test_plan_helpers():
    sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]                # middle (2, 1)
    out = ef.planned_outline(sq, {"cx": 100.0, "cy": 50.0, "angle": 90.0})
    assert out[0] == pytest.approx((101.0, 48.0)) and out[1] == pytest.approx((101.0, 52.0))
    assert geo.poly_area(out) == pytest.approx(8.0)                        # the traced size is kept
    assert ef.translation_of(sq, [(x + 3, y - 1) for x, y in sq]) == pytest.approx((3.0, -1.0))
    assert ef.translation_of(sq, [(0, 0), (5, 0), (4, 2), (0, 2)]) is None


def _ui_with_traced_tmds(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.resize(1300, 850)
    w.show()
    spec, _ = _flow_spec()
    # traced in Flake images (design frame, centred on the origin)
    spec.layer("TMD1").outline = [[-12.0, -8.0], [10.0, -8.0], [10.0, 6.0], [-12.0, 6.0]]
    spec.layer("TMD2").outline = [[-8.0, -10.0], [14.0, -10.0], [14.0, 8.0], [-8.0, 8.0]]
    w.adopt_spec(spec, "new")
    return app, w, spec


def test_plan_hall_bars_from_flake_images(cfg):
    """Hall bars before the device layers are on the chip: their traced borders go on the bond-pad
    site, are moved by dragging their line and turned by their slider; the twisted region follows."""
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    from PyQt6.QtTest import QTest
    from ustack.ui.style import QtCore
    Qt = QtCore.Qt
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    st = p.step
    st.outlines = {"BG": [list(map(tuple, FLAKES["BG"][0]))]}
    site = p.bondpads().site
    p.plan_from_flake_images()
    st = p.step
    assert set(st.plan) == {"TMD1", "TMD2"} and st.twisted_how == ef.PLAN_HOW
    c1 = geo.centroid(st.outlines["TMD1"][0])
    # where Flake layout has it relative to the fabricated bottom gate (linked, 2026-11-20); without a frame: the site
    frame = ef.layout_frame(w.spec, ef.on_chip_outlines(w.spec, p.flow))
    if frame is None:
        assert c1 == pytest.approx(site, abs=1e-6)
    else:
        m_, t_ = frame
        assert c1 == pytest.approx(tuple(m_ @ np.asarray(geo.centroid(w.spec.layer("TMD1").outline)) + t_), abs=1e-6)
    overlap = geo.area(geo.region(st.outlines["TMD1"][0]) & geo.region(st.outlines["TMD2"][0]))
    assert overlap > 0 and geo.poly_area(st.twisted[0]) == pytest.approx(overlap, rel=1e-3)   # overlap of the two
    assert [p.plan_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) for i in range(2)] == ["TMD1", "TMD2"]
    # turn TMD2 by 30°: about its middle, the twisted region follows
    c2 = geo.centroid(st.outlines["TMD2"][0])
    a0 = geo.poly_area(st.twisted[0])
    p.plan_sliders["TMD2"].setValue(30)
    assert p.plan_sliders["TMD2"].value() == pytest.approx(30.0)        # the flake's angle (linked: Flake layout turns too)
    assert geo.centroid(st.outlines["TMD2"][0]) == pytest.approx(c2, abs=1e-6)
    a30 = geo.poly_area(st.twisted[0])
    assert a30 != pytest.approx(a0, rel=1e-3)
    if ef._plan_frame(w.spec, p.flow) is not None:                     # linked: the border turned in Flake layout as well
        m_, t_ = ef._plan_frame(w.spec, p.flow)
        back = np.asarray(w.spec.layer("TMD2").outline) @ m_.T + t_
        assert np.allclose(back, np.asarray(st.outlines["TMD2"][0]), atol=1e-6)
    # click TMD1's line on the canvas and drag it 5 µm to the right
    c = w.ebl_canvas
    app.processEvents()
    c1 = geo.centroid(st.outlines["TMD1"][0])
    c.fit_rect(QtCore.QRectF(c1[0] - 40, c1[1] - 40, 80, 80))
    app.processEvents()
    left = min(st.outlines["TMD1"][0], key=lambda q: q[0])
    nxt = sorted(st.outlines["TMD1"][0], key=lambda q: q[0])[1]
    mid = ((left[0] + nxt[0]) / 2, (left[1] + nxt[1]) / 2)               # a point on the left side
    edge = c.mapFromScene(QtCore.QPointF(*mid))
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, edge)
    assert c.active_poly == "outline:TMD1#0" and p.plan_list.currentItem().data(0, Qt.ItemDataRole.UserRole) == "TMD1"
    cx0 = st.plan["TMD1"]["cx"]
    layout0 = np.asarray(w.spec.layer("TMD1").outline)
    to = c.mapFromScene(QtCore.QPointF(mid[0] + 5.0, mid[1]))
    QTest.mouseMove(c.viewport(), to)
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, to)
    st = p.step
    assert st.plan["TMD1"]["cx"] == pytest.approx(cx0 + 5.0, abs=0.3) and st.located["TMD1"]["how"] == "planned"
    assert geo.poly_area(st.twisted[0]) != pytest.approx(a30, rel=1e-3)
    if ef._plan_frame(w.spec, p.flow) is not None:                     # linked: Flake layout moved with the plan
        m_, _t = ef._plan_frame(w.spec, p.flow)
        shift = np.asarray(w.spec.layer("TMD1").outline) - layout0
        assert np.allclose(m_ @ shift.mean(0), [5.0, 0.0], atol=0.3)
    # Generate fits the fingers on the planned twisted region
    p.generate()
    assert p.step.shapes.get("prepattern_contacts")
    w.dirty = False
    w.close()


def test_outline_window_and_canvas_rotation(cfg):
    from PyQt6.QtTest import QTest
    import numpy as np
    import cv2
    from ustack.alignment import Similarity
    from ustack.ui.style import QtCore, QtGui
    Qt = QtCore.Qt
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    f = p.app.device_dir()
    f.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f / "20x.png"), np.full((300, 400, 3), 128, np.uint8))
    p.step.images["20x"] = str(f / "20x.png")
    p.step.transforms["20x"] = Similarity(0.5, 0.0, -100.0, 75.0).to_json()
    p.step.show_image = "20x"
    p.step.outlines.pop("TMD1", None)
    p._fill_step()
    # Draw outline: its own window over the aligned image, no drawing on the EBL canvas
    for i in range(p.flake_list.topLevelItemCount()):
        if p.flake_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == "TMD1":
            p.flake_list.setCurrentItem(p.flake_list.topLevelItem(i))
    p.draw_outline()
    d = p.outline_dialog
    assert d.isVisible() and d.uid == "TMD1" and "step:20x" in d.canvas.images and d.canvas.tool == "draw"
    assert w.ebl_canvas.tool != "draw"
    d.start_from_trace()
    out = p.step.outlines["TMD1"][0]
    assert geo.poly_area(out) == pytest.approx(22 * 14) and p.step.located["TMD1"]["how"] == "placed"
    moved = [(x + 2.0, y) for x, y in out]
    d.canvas.polygonEdited.emit("outline:TMD1#0", moved)
    assert np.allclose(p.step.outlines["TMD1"][0], moved) and p.step.located["TMD1"]["how"] == "edited"
    d.close()
    # EBL canvas rotation: slider, reset button, twist gesture; two-finger scroll moves the view
    c = w.ebl_canvas
    w.ebl_rotation.setValue(45)
    assert c.view_angle == pytest.approx(45.0)
    v = c.mapFromScene(QtCore.QPointF(10, 0)) - c.mapFromScene(QtCore.QPointF(0, 0))
    assert v.x() > 0 and v.y() < 0                                        # chip +x turned 45° counter-clockwise
    w.ebl_rotation_reset.click()
    assert c.view_angle == 0.0 and w.ebl_rotation.value() == 0
    def gesture(kind, value=0.0):
        return QtGui.QNativeGestureEvent(kind, QtGui.QPointingDevice.primaryPointingDevice(), 2, QtCore.QPointF(100, 100),
                                         QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), value, QtCore.QPointF(0, 0))

    def scroll(dx, phase):
        return QtGui.QWheelEvent(QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), QtCore.QPoint(dx, 0), QtCore.QPoint(0, 0),
                                 Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, phase, False)
    app.sendEvent(c.viewport(), gesture(Qt.NativeGestureType.BeginNativeGesture))
    app.sendEvent(c.viewport(), gesture(Qt.NativeGestureType.RotateNativeGesture, 1.0))     # a wobble: no turn yet
    assert c.view_angle == 0.0
    app.sendEvent(c.viewport(), gesture(Qt.NativeGestureType.RotateNativeGesture, 11.0))    # a twist: turns, panning stops
    assert c.view_angle == pytest.approx(-12.0) and w.ebl_rotation.value() == -12    # opposite sign to the gesture value (user)
    centre = c.mapToScene(c.viewport().rect().center())
    app.sendEvent(c.viewport(), scroll(40, Qt.ScrollPhase.ScrollUpdate))                   # fingers drift while twisting
    assert c.mapToScene(c.viewport().rect().center()) == centre
    app.sendEvent(c.viewport(), gesture(Qt.NativeGestureType.EndNativeGesture))
    app.sendEvent(c.viewport(), scroll(40, Qt.ScrollPhase.ScrollMomentum))                 # momentum right after: still no shift
    assert c.mapToScene(c.viewport().rect().center()) == centre
    app.sendEvent(c.viewport(), scroll(0, Qt.ScrollPhase.ScrollBegin))                     # a new two-finger scroll pans again
    # the angle can be typed
    w.ebl_rotation.field.setText("33.5")
    w.ebl_rotation.field.editingFinished.emit()
    assert c.view_angle == pytest.approx(33.5)
    w.ebl_rotation.field.setText("-12")
    w.ebl_rotation.field.editingFinished.emit()
    assert c.view_angle == pytest.approx(-12.0)
    s0 = c._px_per_unit()
    z = QtGui.QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, QtGui.QPointingDevice.primaryPointingDevice(), 2,
                                  QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), 0.25,
                                  QtCore.QPointF(0, 0))
    app.sendEvent(c.viewport(), z)
    assert c._px_per_unit() == pytest.approx(1.25 * s0)
    before = c.mapToScene(c.viewport().rect().center())
    wheel = QtGui.QWheelEvent(QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), QtCore.QPoint(30, 0), QtCore.QPoint(0, 0),
                              Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
    app.sendEvent(c.viewport(), wheel)
    assert c._px_per_unit() == pytest.approx(1.25 * s0)                  # no zoom: it moved
    assert c.mapToScene(c.viewport().rect().center()) != before
    w.dirty = False
    w.close()


def _img_map(image, x, y):
    s, a = image["um_per_px"], math.radians(image.get("angle_deg", 0.0))
    mx = -1.0 if image.get("mirror") else 1.0
    c, sn = math.cos(a) * s, math.sin(a) * s
    return (c * mx * x + sn * y + image.get("x", 0.0), sn * mx * x - c * y + image.get("y", 0.0))


def test_flakes_found_on_the_chip_update_the_flake_layout_and_3d_stack(cfg):
    """Flakes identified in Stage 2 → their relative position and orientation in the design frame
    (Flake layout, 3D stack); the reference flake stays, shapes and sizes are kept, images follow."""
    import numpy as np
    from ustack.scene3d import build_scene
    spec, flow = _flow_spec()
    st = flow.step("hall_bars")
    st.outlines, st.located = {}, {}
    bg_img = {"path": "bg.png", "um_per_px": 0.1, "angle_deg": 0.0, "x": -30.0, "y": 20.0, "placed": True}
    tmd_img = {"path": "t.png", "um_per_px": 0.05, "angle_deg": 10.0, "x": -5.0, "y": 4.0, "mirror": True}
    bg_px = [(0, 0), (600, 0), (600, 80), (0, 80)]
    tmd_px = [(10, 10), (300, 30), (260, 200), (40, 160), (0, 90)]
    spec.layer("BG").image, spec.layer("TMD1").image = bg_img, tmd_img
    spec.layer("BG").outline = [list(_img_map(bg_img, *q)) for q in bg_px]
    spec.layer("TMD1").outline = [list(_img_map(tmd_img, *q)) for q in tmd_px]
    old_tmd = [tuple(p) for p in spec.layer("TMD1").outline]
    old_bg = [list(p) for p in spec.layer("BG").outline]
    # on the chip: the BG turned 15° and moved; the TMD turned 40° and sits elsewhere relative to it
    st.outlines["BG"] = [geo.transform(spec.layer("BG").outline, 15, 1000, 500)]
    st.outlines["TMD1"] = [geo.transform(old_tmd, 40, 1012, 497)]
    st.located = {"BG": {"how": "located"}, "TMD1": {"how": "located"}}
    assert ef.orientation_from_chip(spec, ef.FlowStep(key="hall_bars"), "hall_bars") is None      # needs two flakes
    rec = ef.orientation_from_chip(spec, st, "hall_bars")
    assert rec["reference"] == "BG" and spec.layer("BG").outline == old_bg       # the placed, lowest flake stays
    assert rec["layers"]["TMD1"]["turned_deg"] == pytest.approx(25.0, abs=1e-6) and rec["layers"]["TMD1"]["changed"]
    # TMD1 relative to the BG in the design frame = as on the chip
    back = geo.transform([(x - 1000, y - 500) for x, y in st.outlines["TMD1"][0]], -15)
    assert np.allclose(spec.layer("TMD1").outline, back, atol=1e-6)
    assert geo.poly_area(spec.layer("TMD1").outline) == pytest.approx(geo.poly_area(old_tmd))
    # the image follows its outline (mirror kept) and counts as placed
    img = spec.layer("TMD1").image
    assert img["placed"] and img["mirror"] and img["um_per_px"] == 0.05
    assert np.allclose([_img_map(img, *q) for q in tmd_px], spec.layer("TMD1").outline, atol=1e-6)
    # the 3D stack draws the new footprint
    slab = next(s for s in build_scene(spec, cfg) if s.uid == "TMD1")
    assert np.allclose(slab.polys[0], spec.layer("TMD1").outline)
    # found again: nothing moves
    rec2 = ef.orientation_from_chip(spec, st, "hall_bars")
    assert not any(v["changed"] for v in rec2["layers"].values())
    assert spec.stage2["orientation"]["reference"] == "BG"


def test_moving_a_located_flake_on_the_ebl_canvas_updates_the_3d_stack(cfg):
    import numpy as np
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    st = p.step
    for u in ("TMD1", "TMD2"):
        st.outlines[u] = [geo.transform(w.spec.layer(u).outline, 0, 1000, 500)]
        st.located[u] = {"how": "located", "score": 0.9}
    st.outlines.pop("BG", None)
    st.located.pop("BG", None)
    p._store()
    moved = geo.transform(w.spec.layer("TMD2").outline, 20, 1003, 501)
    p.set_outline("TMD2", moved, "edited")
    app.processEvents()
    t1 = np.asarray(w.spec.layer("TMD1").outline)
    t2 = np.asarray(w.spec.layer("TMD2").outline)
    chip1, chip2 = np.asarray(p.step.outlines["TMD1"][0]), np.asarray(p.step.outlines["TMD2"][0])
    assert np.allclose(t2 - t1.mean(0), chip2 - chip1.mean(0), atol=1e-6)     # same relative placement as on the chip
    # the 3D stack shows the flakes as on the chip (Stage 2, centred on the device): same relative placement
    s1 = np.asarray(next(s for s in w.view3d.slabs if s.uid == "TMD1").polys[0])
    s2 = np.asarray(next(s for s in w.view3d.slabs if s.uid == "TMD2").polys[0])
    assert np.allclose(s2 - s1.mean(0), chip2 - chip1.mean(0), atol=1e-6)
    assert "turned +20.0°" in w.statusBar().currentMessage()


def test_one_traced_border_everywhere_core():
    """A reshape of the traced border reaches the chip outline (same place on the chip) and back."""
    import numpy as np
    spec, flow = _flow_spec()
    st = flow.step("hall_bars")
    design = [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]]
    spec.layer("TMD1").outline = [list(p) for p in design]
    st.outlines["TMD1"] = [geo.transform(design, 30, 500, 200)]
    st.located["TMD1"] = {"how": "located"}
    assert ef.is_rigid_move(design, geo.transform(design, 30, 5, 2)) and not ef.is_rigid_move(design, [[0, 0], [12, 0], [10, 6], [0, 6]])
    # Flake images: one corner pulled out → the chip outline takes it, the other corners stay put
    spec.layer("TMD1").outline = [[0.0, 0.0], [12.0, -1.0], [10.0, 6.0], [0.0, 6.0]]
    assert ef.follow_design_edit(spec, st, "TMD1", design)
    assert np.allclose(st.outlines["TMD1"][0], geo.transform(spec.layer("TMD1").outline, 30, 500, 200), atol=1e-9)
    # EBL canvas: a point added on the chip → the traced border gets it
    chip = st.outlines["TMD1"][0]
    extra = geo.transform([(5.0, 7.0)], 30, 500, 200)[0]
    new_chip = chip[:3] + [extra] + chip[3:]
    back = ef.design_from_chip_edit(spec, st, "TMD1", chip, new_chip)
    assert np.allclose(back, [[0, 0], [12, -1], [10, 6], [5, 7], [0, 6]], atol=1e-9)
    # a drawn outline is a shape of its own
    st.located["TMD1"] = {"how": "drawn"}
    assert ef.design_from_chip_edit(spec, st, "TMD1", chip, new_chip) is None
    # planned layers follow too, and stay where they are on the chip
    st2 = ef.Flow().step("hall_bars")              # planned where it is not on the chip yet (the WSe2 lies above the fingers)
    st2.plan = {"TMD1": {"cx": 100.0, "cy": 50.0, "angle": 20.0}}
    ef.apply_plan(spec, st2)
    before = [tuple(p) for p in st2.outlines["TMD1"][0]]
    old = [list(p) for p in spec.layer("TMD1").outline]
    spec.layer("TMD1").outline = [[0.0, 0.0], [12.0, -1.0], [10.0, 6.0], [0.0, 9.0]]
    assert ef.follow_design_edit(spec, st2, "TMD1", old)
    assert np.allclose(st2.outlines["TMD1"][0][:3], before[:3], atol=1e-9)


def test_borders_sync_layout_canvas_and_tabs(cfg, tmp_path):
    """Stage 2 ↔ Flake images ↔ Flake layout share one border; the Flake layout shows borders only,
    moves and turns them; a click away deselects; side and centre tabs are independent."""
    import numpy as np
    from ustack.ui.style import QtCore
    from PyQt6.QtTest import QTest
    Qt = QtCore.Qt
    app, w, spec = _ui_with_traced_tmds(cfg)
    spec = w.spec
    p = w.ebl_flow
    p.select_step("hall_bars")
    st = p.step
    d1 = [tuple(q) for q in spec.layer("TMD1").outline]
    st.outlines["TMD1"] = [geo.transform(d1, 0, 1000, 500)]
    st.located["TMD1"] = {"how": "located", "score": 0.9}
    p._store()
    # EBL canvas: the first corner pulled → Flake images' border changes the same way
    chip = list(st.outlines["TMD1"][0])
    chip[0] = (chip[0][0] - 2.0, chip[0][1] - 1.0)
    p.set_outline("TMD1", chip, "edited")
    assert np.allclose(spec.layer("TMD1").outline[0], [d1[0][0] - 2.0, d1[0][1] - 1.0])
    assert np.allclose(spec.layer("TMD1").outline[1:], d1[1:])
    # Flake images: a point moved → the chip outline follows
    pts = [list(q) for q in spec.layer("TMD1").outline]
    pts[2] = [pts[2][0] + 1.0, pts[2][1] + 3.0]
    w.flake_panel.edited("TMD1", pts)
    assert np.allclose(p.step.outlines["TMD1"][0], geo.transform(pts, 0, 1000, 500))
    # Flake layout: borders only (no images)
    w.center_tabs.setCurrentWidget(w.layout_page)
    app.processEvents()
    c = w.layout_canvas
    assert not c.images and "outline:TMD1" in c.polys
    # moving a border whole in Flake layout leaves the chip outline alone, its image moves with it
    chip_before = [tuple(q) for q in p.step.outlines["TMD1"][0]]
    spec.layer("TMD2").image = {"path": "x.png", "um_per_px": 0.1, "angle_deg": 0.0, "x": 0.0, "y": 0.0}
    old2 = [tuple(q) for q in spec.layer("TMD2").outline]
    c.polygonEdited.emit("outline:TMD2", [(x + 5.0, y - 2.0) for x, y in old2])
    assert spec.layer("TMD2").image["x"] == pytest.approx(5.0) and spec.layer("TMD2").image["placed"]
    w.flake_panel.edited("TMD1", [list(q) for q in spec.layer("TMD1").outline])     # unchanged shape: nothing moves
    assert np.allclose(p.step.outlines["TMD1"][0], chip_before)
    # select by clicking the line, turn it with the round rotate handle (the turn slider is gone, 2026-11-21): about its
    # middle, the image turns too
    c.fit()
    app.processEvents()
    x0, y0 = spec.layer("TMD2").outline[0]
    x1, y1 = spec.layer("TMD2").outline[1]
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=c.mapFromScene(QtCore.QPointF((x0 + x1) / 2, (y0 + y1) / 2)))
    assert c.active_poly == "outline:TMD2" and c.rotate_handles and c._rot_handle is not None and c._rot_handle.isVisible()
    assert not hasattr(w, "layout_turn")
    mid = geo.centroid(spec.layer("TMD2").outline)
    assert c.rotate_active(30.0)
    assert spec.layer("TMD2").image["angle_deg"] == pytest.approx(30.0, abs=1e-6) and spec.layer("TMD2").image["placed"]
    assert geo.centroid(spec.layer("TMD2").outline) == pytest.approx(mid)
    # a click on nothing deselects
    far = c.mapFromScene(QtCore.QPointF(mid[0] + 500, mid[1] + 500))
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, pos=far)
    assert c.active_poly is None and not c._rot_handle.isVisible()
    # Export to image: what the canvas shows, as a JPG (the device's folder by default; here the path given)
    out = w.save_layout_image(tmp_path / "layout.jpg")
    assert out is not None and out.is_file() and out.read_bytes()[:2] == b"\xff\xd8"           # a JPEG
    # the same on the EBL canvas
    w.center_tabs.setCurrentWidget(w.ebl_page)
    app.processEvents()
    e = w.ebl_canvas
    e.set_active_polygon("outline:TMD1#0")
    QTest.mouseClick(e.viewport(), Qt.MouseButton.LeftButton, pos=QtCore.QPoint(3, 3))
    assert e.active_poly is None
    # tabs: Flake images beside Flake layout or Structure
    w.tabs.setCurrentWidget(w.flake_panel)
    w.center_tabs.setCurrentWidget(w.layout_page)
    app.processEvents()
    assert w.tabs.currentWidget() is w.flake_panel
    w.tabs.setCurrentWidget(w.stack_panel)
    w.tabs.setCurrentWidget(w.flake_panel)
    assert w.center_tabs.currentWidget() is w.layout_page
    w.center_tabs.setCurrentWidget(w.structure_page)
    assert w.tabs.currentWidget() is w.flake_panel
    w.dirty = False


def test_fingers_as_long_as_the_bottom_hbn_allows(cfg):
    """Each finger runs until it is the preset distance inside the bottom hBN; asked for when unset."""
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    spec, flow = _flow_spec()
    st = flow.step("hall_bars")
    bbn = spec.by_role("bottom_hbn")
    # bottom hBN: 60 µm up from the array centre, 40 µm down (the twisted region sits around the origin, turned 10–25°)
    region = ef.twisted_region(spec, st.outlines)
    cx, cy = geo.centroid(region[0])
    st.outlines[bbn.uid] = [[(cx - 80, cy - 40), (cx + 80, cy - 40), (cx + 80, cy + 60), (cx - 80, cy + 60)]]
    with pytest.raises(MissingValues) as exc:
        ef.generate(spec, flow, "hall_bars", cfg, bondpads.default_preset())
    assert exc.value.items[0].key == "preset:fingers.bbn_margin_um"
    apply_answers(cfg, {"preset:fingers.bbn_margin_um": 2.0})                     # TEST VALUE
    assert cfg.preset("fingers.bbn_margin_um") == 2.0
    hb = ef.generate(spec, flow, "hall_bars", cfg, bondpads.default_preset())
    arr = fg.FingerArray.from_json(hb.array)
    lt, lb = arr.params.length_of("top"), arr.params.length_of("bottom")
    assert lt != lb and lt > 21.8 and lb > 21.8                                   # longer than the fixed preset
    allowed = geo.sized(geo.regions(st.outlines[bbn.uid]).merged(), -2.0)
    fingers = geo.regions([f.polygon for f in arr.fingers()]).merged()
    assert (fingers - allowed).is_empty()                                         # every finger ≥ 2 µm inside
    # each finger as long as it can be on its own (user, 2026-11-20: they need not be the same length): any finger 0.5 µm
    # longer leaves the allowed area
    per = arr.params.lengths
    assert per and len(per) == 2 * arr.n and len({round(v, 2) for v in per.values()}) > 2
    for k in list(per)[:: max(1, len(per) // 5)]:
        column, i = k.split(":")
        f = arr.finger(column, int(i), per[k] + 0.5)
        assert not (geo.region(f.polygon) - allowed).is_empty()
    assert arr.params.width == 0.8 and arr.params.pitch == 2.2                    # the presets are not changed
    # the other steps keep the fitted lengths
    cg = ef.generate(spec, flow, "electrodes", cfg, bondpads.default_preset())
    assert fg.FingerArray.from_json(flow.step("hall_bars").array).params.lengths == per


def test_image_on_canvas_from_any_step(cfg):
    import numpy as np
    import cv2
    from ustack.alignment import Similarity
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    f = p.app.device_dir()
    f.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f / "a.png"), np.full((100, 120, 3), 90, np.uint8))
    cv2.imwrite(str(f / "b.png"), np.full((100, 120, 3), 160, np.uint8))
    p.flow.step("hall_bars").images = {"5x": str(f / "a.png"), "50x": str(f / "b.png")}
    p.flow.step("hall_bars").transforms = {"5x": Similarity(2.0, 0.0, 0.0, 0.0).to_json()}
    p.flow.step("hall_bars").show_image = "50x"
    p.select_step("contact_gate")
    combo = p.canvas_combo
    items = [combo.itemData(i) for i in range(combo.count())]
    assert items[0] is None and ["hall_bars", "5x"] in items and ["hall_bars", "50x"] in items
    combo.setCurrentIndex(items.index(["hall_bars", "5x"]))
    p._canvas_image_chosen()
    assert p.flow.canvas_image == ["hall_bars", "5x"] and "step:hall_bars/5x" in w.ebl_canvas.images
    assert ef.Flow.of(w.spec).canvas_image == ["hall_bars", "5x"]
    combo.setCurrentIndex(0)
    p._canvas_image_chosen()
    assert p.flow.canvas_image is None
    w.dirty = False


def test_finger_ends_on_one_line_and_fan_wider_when_longer():
    p = fg.FingerParams()
    arr = fg.FingerArray((0.0, 0.0), 0.0, 6, p)
    for column, sgn in (("top", 1), ("bottom", -1)):
        far = [f for f in arr.fingers() if f.column == column]
        v_end = sgn * (p.gap / 2 + p.length)
        for f in far:                                   # every far end is cut along u, on one line
            on_line = [q for q in f.polygon if abs(q[1] - v_end) < 2e-3]
            assert len(on_line) == 2 and all(abs(q[1] - v_end) < 2e-3 for q in on_line)
            assert max(q[1] * sgn for q in f.polygon) == pytest.approx(abs(v_end), abs=2e-3)
        ends = sorted(f.far_end[0] for f in far)
        assert ends[1] - ends[0] == pytest.approx(p.far_pitch, abs=1e-6)   # the preset pitch at the preset length
    longer = fg.FingerArray((0.0, 0.0), 0.0, 6, fg.FingerParams(length_top=40.0))
    ends = sorted(f.far_end[0] for f in longer.fingers() if f.column == "top")
    assert ends[1] - ends[0] > p.far_pitch + 1.0                           # same angle, further out: wider
    reg = geo.regions([f.polygon for f in longer.fingers()]).merged()
    assert not geo.space_violations(reg, 0.8)                              # no fingers too close


def test_layers_below_a_step_need_locating_there(cfg):
    """The stack decides: at the contact-gate step the WSe2 halves (below it) are on the chip, even
    when the Hall-bar step only planned them."""
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    app, w, spec = _ui_with_traced_tmds(cfg)
    spec = w.spec
    p = w.ebl_flow
    p.select_step("hall_bars")
    p.step.outlines = {"BG": [list(map(tuple, FLAKES["BG"][0]))]}
    p.plan_from_flake_images()
    st = p.step
    assert ef.flake_state(spec, st, "hall_bars", "TMD1") == ("planned", False)
    p.select_step("contact_gate")
    cg = p.step
    assert cg.located["TMD1"]["how"] == ef.EXPECTED_HOW and cg.outlines["TMD1"]
    text, on_chip = ef.flake_state(spec, cg, "contact_gate", "TMD1")
    assert on_chip and text.startswith("needs locating")
    rows = {p.flake_list.topLevelItem(i).data(0, 0x0100): p.flake_list.topLevelItem(i).text(1) for i in range(p.flake_list.topLevelItemCount())}
    assert rows["TMD1"].startswith("needs locating") and rows["TMD2"].startswith("needs locating")
    # Layers card: hide the Hall-bar fingers while working on the contact gate
    p._activate("hall_bars")
    p.generate()
    p._activate("contact_gate")
    assert any(k.startswith("pat:hall_bars|") for k in w.ebl_canvas.polys)
    w.show_canvas_layers()
    card = w.canvas_layers_card
    assert "pat:hall_bars" in card.boxes and "flake:TMD1" in card.boxes and "bp:pads" in card.boxes
    card.boxes["pat:hall_bars"].setChecked(False)
    assert not any(k.startswith("pat:hall_bars|") for k in w.ebl_canvas.polys)
    card.boxes["flake:TMD1"].setChecked(False)
    assert not any(k.startswith("outline:TMD1") for k in w.ebl_canvas.polys)
    card.show_all()
    assert any(k.startswith("pat:hall_bars|") for k in w.ebl_canvas.polys)
    card.hide_all()                                                    # nothing on the canvas
    assert not w.ebl_canvas.polys and not any(cb.isChecked() for cb in card.boxes.values())
    card.only_image()                                                  # only the image ticked
    assert [k for k, cb in card.boxes.items() if cb.isChecked()] == ["image"] and not w.ebl_canvas.polys
    card.show_all()
    assert any(k.startswith("pat:hall_bars|") for k in w.ebl_canvas.polys)
    card.close()
    w.dirty = False


def test_old_shared_plans_become_needs_locating():
    """A plan left on a layer that is on the chip at that step (copied by the old shared state) is
    turned into 'needs locating' when the flow is read."""
    spec, flow = _flow_spec()
    st = flow.step("contact_gate")
    st.plan = {"TMD1": {"cx": 0.0, "cy": 0.0, "angle": 5.0}}
    st.located["TMD1"] = {"how": "planned", "score": None}
    st.twisted_how = ef.PLAN_HOW
    flow.store(spec)
    again = ef.Flow.of(spec)
    cg = again.step("contact_gate")
    assert not cg.plan and cg.located["TMD1"]["how"] == ef.EXPECTED_HOW and cg.outlines["TMD1"]
    assert ef.flake_state(spec, cg, "contact_gate", "TMD1")[0].startswith("needs locating")
    assert cg.twisted_how != ef.PLAN_HOW


def test_rotate_handle_and_import_border(cfg):
    import numpy as np
    from PyQt6.QtTest import QTest
    from ustack.ui.style import QtCore
    Qt = QtCore.Qt
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    p.step.outlines = {"BG": [list(map(tuple, FLAKES["BG"][0]))]}
    p.step.located = {"BG": {"how": "located"}}
    p._fill_step()
    for i in range(p.flake_list.topLevelItemCount()):
        if p.flake_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == "TMD1":
            p.flake_list.setCurrentItem(p.flake_list.topLevelItem(i))
    p.draw_outline()
    d = p.outline_dialog
    d.resize(900, 700)
    app.processEvents()
    # Import border ▾: this flake's and the other flakes' traced borders
    d._fill_import_menu()
    texts = [a.text() for a in d.import_menu.actions() if a.text()]
    assert texts[0].startswith("This flake's border") and any(t.startswith("WSe2 (half 2") for t in texts)
    next(a for a in d.import_menu.actions() if a.text().startswith("WSe2 (half 2")).trigger()
    got = p.step.outlines["TMD1"][0]
    assert geo.poly_area(got) == pytest.approx(geo.poly_area(spec.layer("TMD2").outline), rel=1e-6)   # TMD2's border, own size
    # the round handle above the outline turns it about its middle (Shift: 15° steps)
    c = d.canvas
    c.fit()
    app.processEvents()
    h = c._rot_handle
    assert h is not None and h.isVisible()
    before = [tuple(q) for q in p.step.outlines["TMD1"][0]]
    mid = geo.centroid(before)
    start = c.mapFromScene(h.pos())
    m = c.mapFromScene(QtCore.QPointF(*mid))
    end = QtCore.QPoint(m.x() + (m.y() - start.y()), m.y())              # a quarter turn, clockwise on screen
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(c.viewport(), QtCore.QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
    QTest.mouseMove(c.viewport(), end)
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    after = p.step.outlines["TMD1"][0]
    assert geo.centroid(after) == pytest.approx(mid, abs=0.05)
    turn = math.degrees(math.atan2(after[0][1] - mid[1], after[0][0] - mid[0]) - math.atan2(before[0][1] - mid[1], before[0][0] - mid[0]))
    assert math.remainder(turn, 360) == pytest.approx(-90.0, abs=4.0)
    assert geo.poly_area(after) == pytest.approx(geo.poly_area(before), rel=1e-6)
    # typed: the canvas API turns the selected outline exactly
    assert c.rotate_active(30.0)
    a2 = p.step.outlines["TMD1"][0]
    turn = math.degrees(math.atan2(a2[0][1] - mid[1], a2[0][0] - mid[0]) - math.atan2(after[0][1] - mid[1], after[0][0] - mid[0]))
    assert math.remainder(turn, 360) == pytest.approx(30.0, abs=0.01)
    d.close()
    # the trace window has the handle too
    from ustack.ui.flake_images import TraceDialog
    t = TraceDialog(w.flake_panel)
    t.set_layer("TMD1")
    t.show()
    app.processEvents()
    assert t.canvas.rotate_handles and t.canvas._rot_handle is not None and t.canvas._rot_handle.isVisible()
    t.close()
    w.dirty = False


def test_twisted_region_follows_the_device_layers_and_nothing_above_is_projected():
    spec, flow = _flow_spec()
    st = flow.step("contact_gate")
    st.twisted, st.twisted_how = [[(0, 0), (1, 0), (1, 1)]], "from Hall bars (finger array)"
    assert ef.refresh_twisted(spec, st)                                       # the overlap of the two WSe2 outlines
    assert st.twisted_how == ef.TWISTED_AUTO
    area = geo.area(geo.regions(st.twisted).merged())
    expect = geo.area(geo.regions(FLAKES["TMD1"]).merged() & geo.regions(FLAKES["TMD2"]).merged())
    assert area == pytest.approx(expect, rel=1e-6)
    st.outlines["TMD2"] = [geo.transform(FLAKES["TMD2"][0], 0, 3.0, 0.0)]    # a WSe2 moved (located again, dragged)
    assert ef.refresh_twisted(spec, st)
    assert geo.area(geo.regions(st.twisted).merged()) != pytest.approx(area, rel=1e-3)
    assert not ef.refresh_twisted(spec, st)                                   # nothing new
    st.twisted, st.twisted_how = [[(0, 0), (5, 0), (5, 5)]], "drawn"          # drawn by hand: left alone
    assert not ef.refresh_twisted(spec, st) and st.twisted == [[(0, 0), (5, 0), (5, 5)]]
    one = flow.step("bottom_gate")
    one.outlines = {"TMD1": FLAKES["TMD1"]}                                    # one half only: not a twisted region
    assert not ef.refresh_twisted(spec, one)
    # a layer above the step (top-gate hBN at the contact gate) is never projected; old projections are dropped
    tbn = spec.by_role("tg_hbn")
    assert tbn.uid not in ef.present_layers(spec, "contact_gate")
    st.located[tbn.uid] = {"how": "projected"}
    st.outlines[tbn.uid] = [[(0, 0), (9, 0), (9, 9)]]
    flow.store(spec)
    again = ef.Flow.of(spec).step("contact_gate")
    assert tbn.uid not in again.outlines and tbn.uid not in again.located
    spec.layer("TMD1").image = {"placed": True}
    spec.layer("TMD1").outline = [list(p) for p in FLAKES["TMD1"][0]]
    tbn.image, tbn.outline = {"placed": True}, [[0, 0], [9, 0], [9, 9]]
    st2 = ef.FlowStep("contact_gate", outlines={"TMD1": [list(map(tuple, FLAKES["TMD1"][0]))]}, located={"TMD1": {"how": "located"}})
    ef.project_flakes(spec, st2, ef.present_layers(spec, "contact_gate"))
    assert tbn.uid not in st2.outlines


def test_outline_window_moves_and_zooms_like_the_ebl_canvas(cfg):
    from ustack.ui.style import QtCore, QtGui
    Qt = QtCore.Qt
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    p.step.outlines = {"TMD1": [list(map(tuple, FLAKES["TMD1"][0]))]}
    p.step.located = {"TMD1": {"how": "located"}}
    p._fill_step()
    p.flake_list.setCurrentItem(p.flake_list.topLevelItem(0))
    p.draw_outline()
    c = p.outline_dialog.canvas
    p.outline_dialog.resize(900, 700)
    app.processEvents()
    assert c.trackpad_gestures and c.drag_pans
    s0 = c._px_per_unit()
    before = c.mapToScene(c.viewport().rect().center())
    wheel = QtGui.QWheelEvent(QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), QtCore.QPoint(30, 0), QtCore.QPoint(0, 0),
                              Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
    app.sendEvent(c.viewport(), wheel)                                        # two-finger scroll: moves, no zoom
    assert c._px_per_unit() == pytest.approx(s0) and c.mapToScene(c.viewport().rect().center()) != before
    z = QtGui.QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, QtGui.QPointingDevice.primaryPointingDevice(), 2,
                                  QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), QtCore.QPointF(100, 100), 0.25, QtCore.QPointF(0, 0))
    app.sendEvent(c.viewport(), z)                                            # pinch: zooms
    assert c._px_per_unit() == pytest.approx(1.25 * s0)
    p.outline_dialog.close()
    w.dirty = False


def test_contact_gate_gates_every_finger_and_stays_on_its_hbn(cfg):
    """Every finger is gated, the source / drain ones too (user, 2026-11-22); the bus lies behind the teeth and its lead,
    shortened to stay on the middle hBN, leads out past the outer tooth; the reversed layout when the first does not fit."""
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    arr = fg.FingerArray.from_json(hb.array)
    assert not ef.has_source_drain(spec)
    st = flow.step("contact_gate")
    mbn = ef.hbn_under(spec, "contact_gate")
    assert mbn is not None and mbn.role == "middle_hbn"
    # a middle hBN reaching 3 µm past the outer teeth on +u, 30 µm on −u
    p = arr.params
    hs = arr.span / 2
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 30, -25), (hs + 3, -25), (hs + 3, 25), (-hs - 30, 25)])]
    with pytest.raises(MissingValues) as exc:
        ef.generate(spec, flow, "contact_gate", cfg, bp)
    assert exc.value.items[0].key == "preset:cg.hbn_margin_um"
    apply_answers(cfg, {"preset:cg.hbn_margin_um": 0.5})                       # TEST VALUE
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    lay = st.comb
    # layout 1 (top bus to +u) needs 5 µm past the top-right tooth: only 2.5 fit → shortened; no finger left ungated
    assert not lay["skip"] and lay["n_skip"] == 0 and lay["dirs"] == {"top": 1.0, "bottom": -1.0}
    assert lay["extension"]["top"] == pytest.approx(2.5, abs=0.11) and lay["extension"]["bottom"] == pytest.approx(5.0)
    # the bottom bus (to −u) is led on, slightly turned, to the hBN's far corner (−u, −v), 0.5 µm inside it (user, 2026-11-20)
    ru, rv = arr.to_local([lay["reach"]["bottom"]])[0]
    w = lay["lead_width"]["bottom"]                                    # as wide as the bus, or narrower where that reaches farther
    assert cfg.value("ebl.contact_width_min_um") - 1e-9 <= w <= p.cg_bus_width + 1e-9
    assert ru == pytest.approx(-hs - 30 + 0.5 + w / 2, abs=0.1) and rv == pytest.approx(-25 + 0.5 + w / 2, abs=0.1)
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "contact_gate", cfg)}
    assert rows["Every finger gated (source / drain too)"]["status"] == "met"
    assert "Source and drain fingers left ungated" not in rows
    if "Contact gate on the bottom hBN" in rows:                      # never over bare back gate (user, 2026-11-22)
        assert rows["Contact gate on the bottom hBN"]["status"] == "met"
    assert rows[f"Contact gate on {mbn.label}"]["status"] == "met"
    assert rows["Contact-gate bus off the TMD"]["status"] == "met"
    fingers = {(f.column, f.index): geo.region(f.polygon) for f in arr.fingers()}
    cg = geo.regions(st.shapes["contact_gates"]).merged()
    assert all(not (cg & fingers[k]).is_empty() for k in fingers)          # the outer (source / drain) fingers too
    assert fg.gated_fingers(arr, st.shapes["contact_gates"]) == {"top": set(range(arr.n)), "bottom": set(range(arr.n))}
    # only 1 µm of hBN past the top-right tooth (plenty beside the bottom column): the reversed layout
    # (top bus to −u, bottom bus to +u)
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 30, -25), (hs + 30, -25), (hs + 30, 0), (hs + 1.0, 0), (hs + 1.0, 25), (-hs - 30, 25)])]
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    assert st.comb["dirs"] == {"top": -1.0, "bottom": 1.0} and any("reversed" in n for n in st.notes)
    # no room on either side: an error, nothing shrunk
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 0.5, -25), (hs + 0.5, -25), (hs + 0.5, 25), (-hs - 0.5, 25)])]
    with pytest.raises(ValueError):
        ef.generate(spec, flow, "contact_gate", cfg, bp)
    # the electrodes land on the generated comb's lead ends
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 30, -25), (hs + 30, -25), (hs + 30, 25), (-hs - 30, 25)])]
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    el = ef.generate(spec, flow, "electrodes", cfg, bp)
    assert sum(r["terminal"].startswith("CG-") for r in el.routes) == 2


def test_layers_found_later_replace_the_plan_at_the_hall_bar_step(cfg):
    spec, flow = _flow_spec()
    hb = flow.step("hall_bars")
    spec.layer("TMD1").outline = [list(p) for p in FLAKES["TMD1"][0]]
    spec.layer("TMD2").outline = [list(p) for p in FLAKES["TMD2"][0]]
    hb.outlines = {"BG": FLAKES["BG"]}
    hb.plan = {u: {"cx": 0.0, "cy": 0.0, "angle": 0.0} for u in ("TMD1", "TMD2")}
    ef.apply_plan(spec, hb)
    assert hb.twisted_how == ef.PLAN_HOW
    cg = flow.step("contact_gate")
    real = {"TMD1": [geo.transform(FLAKES["TMD1"][0], 3, 2.0, 1.0)], "TMD2": [geo.transform(FLAKES["TMD2"][0], -2, 1.0, -1.0)]}
    cg.outlines = {u: [list(map(tuple, q)) for q in v] for u, v in real.items()}
    cg.located = {"TMD1": {"how": "located"}, "TMD2": {"how": "edited"}}
    assert ef.adopt_identified(spec, flow)
    assert not hb.plan and hb.located["TMD1"]["how"] == ef.IDENTIFIED_LATER and hb.located["TMD1"]["at"] == "contact_gate"
    assert hb.outlines["TMD1"] == [list(map(tuple, q)) for q in real["TMD1"]]
    ef.refresh_twisted(spec, hb)
    ov = geo.regions(real["TMD1"]).merged() & geo.regions(real["TMD2"]).merged()
    assert geo.area(geo.regions(hb.twisted).merged() ^ ov) < 1e-3                 # the actual overlap
    assert ef.flake_state(spec, hb, "hall_bars", "TMD1")[0].startswith("identified at Contact gate")
    assert not ef.adopt_identified(spec, flow)                                     # nothing new


def test_check_window_and_reopening_shows_the_canvas_as_saved(cfg, monkeypatch):
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    import numpy as np
    import cv2
    from ustack.alignment import Similarity
    from ustack.ui.main_window import MainWindow
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    f = p.app.device_dir()
    (f / "images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f / "images" / "cg_20x.png"), np.full((300, 400, 3), 120, np.uint8))
    p.select_step("hall_bars")
    p.step.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p.step.located = {u: {"how": "located"} for u in FLAKES}
    p._store()
    p.generate()
    # Check: every rule in a window, missing values entered there
    p.check()
    d = p.check_dialog
    rules = [d.tree.topLevelItem(i).text(0) for i in range(d.tree.topLevelItemCount())]
    assert any("width ≥ hard minimum" in r for r in rules) and "Finger pairs on the twisted region" in rules
    assert "preset:fingers.bbn_margin_um" in d.fields
    d.fields["preset:fingers.bbn_margin_um"].setText("1.5")                      # TEST VALUE
    d.save_and_check()
    assert cfg.preset("fingers.bbn_margin_um") == 1.5
    d.close()
    # the canvas as saved: image on canvas, the step in use, hidden layers, the view
    p.select_step("contact_gate")
    cgs = p.flow.step("contact_gate")
    cgs.images = {"20x": str(f / "images" / "cg_20x.png")}
    cgs.transforms = {"20x": Similarity(0.5, 0.0, -100.0, 75.0).to_json()}
    p.flow.canvas_image = ["contact_gate", "20x"]
    p._store()
    w.center_tabs.setCurrentWidget(w.ebl_page)
    app.processEvents()
    p.redraw()
    p.set_layer_visible("bp:pads", False)
    w.ebl_canvas.set_view_rotation(15.0)
    v = w.ebl_canvas.view_state()
    from ustack.ui import main_window as mw
    asked = []
    monkeypatch.setattr(mw, "prompt_text", lambda parent, title, prompt, text="", ok="": asked.append(text) or text)
    w.save()                                               # the first Save asks the device's name (offered: its name)
    assert asked == [w.spec.name]
    path = w.spec_path
    w.dirty = False
    w2 = MainWindow(cfg)
    w2.resize(1300, 850)
    w2.show()
    w2.open_device(path)
    for _ in range(5):
        app.processEvents()
    assert w2.center_tabs.currentWidget() is w2.ebl_page and w2.ebl_flow.key == "contact_gate"
    assert "step:20x" in w2.ebl_canvas.images and w2.ebl_flow.hidden == {"bp:pads"}
    v2 = w2.ebl_canvas.view_state()
    assert v2["angle"] == pytest.approx(15.0) and v2["scale"] == pytest.approx(v["scale"], rel=1e-3)
    assert math.dist((v2["cx"], v2["cy"]), (v["cx"], v["cy"])) < 2.0 / v["scale"]
    w2.dirty = False


def test_contact_gate_never_drops_a_finger_and_check_reads_drawn_shapes(cfg):
    """Every finger needs its tooth (user, 2026-11-22): when the hBN under the right-most fingers' teeth is missing, the
    generator says so (no finger is left ungated to make it fit); Check reads the shapes as they are, also drawn by hand."""
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    arr = fg.FingerArray.from_json(hb.array)
    p = arr.params
    st = flow.step("contact_gate")
    mbn = ef.hbn_under(spec, "contact_gate")
    hs = arr.span / 2
    # the hBN's lower-right corner cut away under the two right-most bottom fingers' teeth
    u_cut = arr.u_of(arr.n - 2) - p.width / 2 - 0.3
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 30, -25), (u_cut, -25), (u_cut, -1.0), (hs + 30, -1.0), (hs + 30, 25), (-hs - 30, 25)])]
    apply_answers(cfg, {"preset:cg.hbn_margin_um": 0.2})                       # TEST VALUE
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE
    with pytest.raises(ValueError, match="every finger gated"):
        ef.generate(spec, flow, "contact_gate", cfg, bp)
    st.outlines[mbn.uid] = [arr.to_chip([(-hs - 30, -25), (hs + 30, -25), (hs + 30, 25), (-hs - 30, 25)])]
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "contact_gate", cfg)}
    assert all(r["status"] in ("met", "warning") for r in rows.values()), rows
    # drawn by hand without the outer bottom finger's tooth: caught by Check
    lay = st.comb
    st.shapes["contact_gates"] = [c["polygon"] for c in fg.contact_gate_comb(arr, skip={"bottom": "last"}, dirs=lay["dirs"],
                                                                              tooth=lay.get("tooth")).values()]
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "contact_gate", cfg)}
    assert rows["Every finger gated (source / drain too)"]["status"] == "violated"
    assert f"bottom {arr.n}" in rows["Every finger gated (source / drain too)"]["found"]
    ef.check(spec, flow, "contact_gate", cfg)
    assert any(i["code"] == "cg_rule" for i in st.issues)


def test_draw_contact_gate_window(cfg):
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    app, w, spec = _ui_with_traced_tmds(cfg)
    p = w.ebl_flow
    p.select_step("hall_bars")
    p.step.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p._store()
    p.generate()
    p.select_step("contact_gate")
    p.draw_pattern()
    d = p.shape_dialog
    assert d.purpose == "contact_gates" and not d.canvas.polys
    arr = fg.FingerArray.from_json(p.flow.step("hall_bars").array)
    comb = fg.contact_gate_comb(arr, skip={}, dirs={"top": 1.0, "bottom": -1.0}, n_skip=0)
    for c in comb.values():                                                     # drawn with the polygon tool
        d.canvas.polygonDrawn.emit([tuple(q) for q in c["polygon"]])
    st = p.flow.step("contact_gate")
    assert len(st.shapes["contact_gates"]) == 2 and set(d.canvas.polys) == {"shape:0", "shape:1"}
    p.check()
    rows = {r["rule"]: r for r in p.check_dialog.rows}
    assert rows["Every finger gated (source / drain too)"]["status"] == "met"         # checked as drawn
    p.check_dialog.close()
    moved = [(x, y + 5.0) for x, y in st.shapes["contact_gates"][0]]               # dragged off its fingers
    d.canvas.polygonEdited.emit("shape:0", moved)
    assert st.shapes["contact_gates"][0] == moved and not st.issues
    d.canvas.set_active_polygon("shape:1")
    d.delete_shape()
    assert len(st.shapes["contact_gates"]) == 1
    d.close()
    w.dirty = False


def test_contact_gate_bus_as_close_to_the_tmd_as_the_preset_allows(cfg):
    """User, 2026-11-17: the bus joining the teeth runs as close to the WSe2 as ``cg.bus_tmd_gap_um``
    allows; each column's teeth are fitted on their own (the columns need not match)."""
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    ef.generate(spec, flow, "hall_bars", cfg, bp)
    st = flow.step("contact_gate")
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    t05 = dict(st.comb["tooth"])
    assert abs(t05["top"] - t05["bottom"]) > 1.0                                # FLAKES' TMD reaches farther on the bottom side
    row = {r["rule"]: r for r in ef.rule_report(spec, flow, "contact_gate", cfg)}["Contact-gate bus off the TMD"]
    assert row["status"] == "met" and float(row["found"].split()[0]) == pytest.approx(0.5, abs=0.03)
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 1.5})                      # TEST VALUE
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    for c in ("top", "bottom"):
        assert st.comb["tooth"][c] == pytest.approx(t05[c] + 1.0, abs=0.3)
    # a bus dragged onto the TMD by hand: Check says so
    arr = fg.FingerArray.from_json(flow.step("hall_bars").array)
    st.shapes["contact_gates"] = [arr.to_chip([(u, v - 6.0) for u, v in arr.to_local(q)]) if arr.to_local(q)[0][1] > 0 else q
                                  for q in st.shapes["contact_gates"]]
    row = {r["rule"]: r for r in ef.rule_report(spec, flow, "contact_gate", cfg)}["Contact-gate bus off the TMD"]
    assert row["status"] == "warning" and row["found"].startswith("closer")


def test_export_with_the_traced_flake_outlines(cfg, tmp_path):
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    ef.generate(spec, flow, "hall_bars", cfg, bp)
    with pytest.raises(MissingValues) as exc:
        ef.export_step(spec, flow, "hall_bars", cfg, tmp_path, flakes=True)
    keys = {i.key for i in exc.value.items}
    assert {f"layer:flake:{u}" for u in FLAKES} <= keys and "layer:prepattern_contacts" in keys
    assert any("flake outline: " in i.label for i in exc.value.items)
    answers = {"layer:prepattern_contacts": [5, 0]}                             # TEST VALUE
    answers.update({f"layer:flake:{u}": [100 + i, 0] for i, u in enumerate(sorted(FLAKES))})   # TEST VALUE
    apply_answers(cfg, answers)
    path = ef.export_step(spec, flow, "hall_bars", cfg, tmp_path, flakes=True)
    got = read_polygons(path)
    for i, u in enumerate(sorted(FLAKES)):
        assert geo.area(geo.regions(got[(100 + i, 0)])) == pytest.approx(geo.area(geo.regions(FLAKES[u])), rel=1e-3)
    assert (5, 0) in got
    assert ef.flake_shapes(spec, flow, ["hall_bars"]).keys() == {f"flake:{u}" for u in FLAKES}
    # without the option: no flake layers
    path = ef.export_step(spec, flow, "hall_bars", cfg, tmp_path)
    assert set(read_polygons(path)) == {(5, 0)}
    # a flake on a pattern's layer would merge with it: refused
    apply_answers(cfg, {f"layer:flake:{sorted(FLAKES)[0]}": [5, 0]})            # TEST VALUE
    with pytest.raises(ValueError):
        ef.export_step(spec, flow, "hall_bars", cfg, tmp_path, flakes=True)


def test_scale_handles_stretch_an_outline():
    from PyQt6.QtTest import QTest
    from ustack.ui.canvas import Canvas, ScaleHandle
    from ustack.ui.style import QtCore
    from ustack.app import make_app
    Qt = QtCore.Qt
    qapp = make_app()
    c = Canvas()
    c.scale_handles = True
    c.resize(600, 500)
    c.show()
    edits = []
    c.polygonEdited.connect(lambda k, pts: edits.append(pts))
    c.set_polygon("a", [(0, 0), (10, 0), (10, 5), (0, 5)], "#d05a5a", True)
    c.fit()
    c.set_active_polygon("a")
    qapp.processEvents()
    hs = {h.side: h for h in c._scale_items}
    assert len(hs) == 8 and all(h.isVisible() for h in hs.values())

    def drag(side, dx, dy, mods=Qt.KeyboardModifier.NoModifier):
        start = c.mapFromScene(hs[side].pos())
        end = QtCore.QPoint(start.x() + dx, start.y() + dy)
        QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, mods, start)
        QTest.mouseMove(c.viewport(), end)
        QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, mods, end)

    def box():
        pts = c.polys["a"].points
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    px = c.transform().m11()                                                    # screen px per µm
    drag((1, 0), int(round(5 * px)), 0)                                         # right side: 5 µm wider, height kept
    x0, y0, x1, y1 = box()
    assert x0 == pytest.approx(0, abs=1e-6) and x1 == pytest.approx(15, abs=0.2) and (y0, y1) == pytest.approx((0, 5), abs=1e-6)
    assert edits and len(edits[-1]) == 4
    drag((0, -1), 0, -int(round(5 * px)))                                       # top side (screen up = +y): 5 µm taller
    x0, y0, x1, y1 = box()
    assert y0 == pytest.approx(0, abs=1e-6) and y1 == pytest.approx(10, abs=0.2) and x1 == pytest.approx(15, abs=0.2)
    c.scale_active(0.5)                                                         # whole: half the size about its middle
    x0, y0, x1, y1 = box()
    assert x1 - x0 == pytest.approx(7.5, abs=0.2) and y1 - y0 == pytest.approx(5.0, abs=0.2)
    assert isinstance(hs[(1, 1)], ScaleHandle)
    c.close()


def test_suggested_output_layers(cfg):
    """The lab's tWSe2 batch 1 numbers are the suggestions for the fingers / contact gate / electrodes (gate contacts
    with the electrodes); a metal top gate and the flake outlines get free layers — suggestions only, nothing stored."""
    spec, _ = _flow_spec()
    items = {i.key: i for i in ef._missing_layers(cfg, ["prepattern_contacts", "contact_gates", "electrodes", "bg_contacts",
                                                         "tg_contacts", "top_gate"], spec)}
    assert items["layer:prepattern_contacts"].suggested == [5, 0] and items["layer:contact_gates"].suggested == [6, 0]
    assert items["layer:electrodes"].suggested == [7, 0]
    assert items["layer:bg_contacts"].suggested == [7, 0] and items["layer:tg_contacts"].suggested == [7, 0]
    tg = tuple(items["layer:top_gate"].suggested)
    used = {(g.layer, g.datatype) for g in cfg.layers.values() if not g.is_missing}
    assert tg not in used and tg not in set(ef.LAYER_SUGGEST.values()) and "Program suggestion" in items["layer:top_gate"].hint
    flakes = [f"{ef.FLAKE_PREFIX}{l.uid}" for l in spec.layers]
    fl = [tuple(i.suggested) for i in ef._missing_layers(cfg, flakes, spec)]
    assert fl[0] == (ef.FLAKE_LAYER_BASE, 0) and len(set(fl)) == len(fl) and not set(fl) & used
    assert all(cfg.layers.get(p) is None or cfg.layers.get(p).is_missing for p in ["electrodes", "top_gate"] + flakes)


def test_structure_shows_the_stack_as_on_the_chip(cfg):
    """Fabricated flakes (Stage 2) are fixed — shape and place from the chip — in Flake layout and the Structure tab's 3D
    stack; flakes still to come stay where Flake layout has them; generated patterns are drawn too."""
    apply_answers(cfg, {"preset:electrodes.pad_landing_um": 10.0, "preset:electrodes.flake_clearance_um": 1.0})              # TEST VALUE (no dialog in a test)
    from ustack.scene3d import build_scene
    spec, flow = _flow_spec()
    assert ef.structure_footprints(spec, ef.Flow()) is None             # nothing on the chip: the design view stays
    bp = bondpads.default_preset()
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                      # TEST VALUE
    # BG is on the chip from the Hall-bar step on (the TMDs only from the contact-gate step: above the fingers)
    assert set(ef.on_chip_outlines(spec, flow)) == {"BG", "TMD1", "TMD2"}
    assert ef.on_chip_outlines(spec, flow)["BG"][0] == "electrodes"
    spec.layer("BG").outline = [list(q) for q in geo.transform(FLAKES["BG"][0], -15, -100, -50)]   # its traced border, elsewhere
    tg = [[-30.0, 20.0], [30.0, 20.0], [30.0, 26.0], [-30.0, 26.0]]
    spec.layer("TG").outline = [list(q) for q in tg]                   # not on the chip yet: where Flake layout has it
    view = ef.layout_view(spec, flow)
    for a_, b_ in zip(view["fixed"]["BG"][0], spec.layer("BG").outline):    # the reference keeps its Flake-layout place
        assert a_ == pytest.approx(tuple(b_), abs=1e-6)
    # the TMDs sit relative to BG as on the chip
    m, t = view["frame"]
    back = [tuple(np.asarray(q) @ m.T + t) for q in view["fixed"]["TMD1"][0]]
    for a_, b_ in zip(back, FLAKES["TMD1"][0]):
        assert a_ == pytest.approx(tuple(b_), abs=1e-6)
    ef.generate(spec, flow, "hall_bars", cfg, bp)
    chip, text = ef.structure_footprints(spec, flow)
    pt = spec.by_function("hall_bar_contacts")[0].uid
    assert len(chip[pt]) == len(flow.step("hall_bars").shapes["prepattern_contacts"])   # the fingers as generated
    leads = spec.by_function("leads")[0].uid
    assert chip[leads] == [] and "On the chip" in text                 # electrodes not designed yet: nothing drawn
    # TG (not fabricated) is where Flake layout has it relative to the fixed flakes: moving it there moves it here
    off = np.asarray(chip["TG"][0][0]) - np.asarray(tg[0])
    spec.layer("TG").outline = [[x + 5.0, y] for x, y in tg]
    chip2, _t = ef.structure_footprints(spec, flow)
    assert np.asarray(chip2["TG"][0][0]) - np.asarray(spec.layer("TG").outline[0]) == pytest.approx(off, abs=1e-6)
    assert chip2["BG"] == chip["BG"]
    slabs = {s.uid: s for s in build_scene(spec, cfg, chip)}
    assert slabs["BG"].polys == chip["BG"] and slabs[pt].polys == chip[pt]
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    ef.generate(spec, flow, "electrodes", cfg, bp)
    chip, _t = ef.structure_footprints(spec, flow)
    assert chip[leads] and chip[spec.by_function("contact_gate")[0].uid]
    xs = [abs(q[0]) for poly in chip[leads] for q in poly]
    assert max(xs) < 250                                              # the electrodes are clipped round the flakes


def test_flake_layout_locks_the_flakes_on_the_chip(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, flow = _flow_spec()
    spec.layer("TG").outline = [[-30.0, 20.0], [30.0, 20.0], [30.0, 26.0], [-30.0, 26.0]]
    flow.store(spec)
    w.adopt_spec(spec, "new")
    w.refresh_layout_canvas()
    c = w.layout_canvas
    assert not c.polys["outline:BG"].editable and c.polys["outline:TG"].editable    # BG on the chip: locked; TG movable
    w.dirty = False
    w.close()


def test_flakes_found_on_the_chip_overwrite_their_traced_borders():
    spec, flow = _flow_spec()
    bg = spec.layer("BG")
    bg.outline = [list(q) for q in geo.transform([(-50, -3), (50, -3), (50, 3), (-50, 3)], 5, 7, 9)]   # a different traced shape
    bg.image = {"path": "bg.png", "um_per_px": 0.1, "angle_deg": 0.0, "x": 0.0, "y": 0.0}
    spec.layer("TMD1").outline = [list(q) for q in FLAKES["TMD1"][0]]
    changed = ef.sync_layout_from_chip(spec, flow)
    assert set(changed) >= {"BG", "TMD1", "TMD2"}
    view = ef.layout_view(spec, flow)
    for uid in ("BG", "TMD1", "TMD2"):                                 # the shape found on the chip, in the Flake-layout frame
        assert np.allclose(np.asarray(spec.layer(uid).outline), np.asarray(view["fixed"][uid][0]), atol=1e-6)
    assert len(bg.outline) == len(FLAKES["BG"][0]) and bg.image["trace"]["how"] == "chip" and bg.image["x"] != 0.0
    assert ef.sync_layout_from_chip(spec, flow) == []                  # settled: nothing changes the second time
    # relative placement as on the chip
    off = np.asarray(spec.layer("TMD1").outline[0]) - np.asarray(spec.layer("BG").outline[0])
    m, _t = view["frame"]
    assert np.allclose(m @ off, np.asarray(FLAKES["TMD1"][0][0]) - np.asarray(FLAKES["BG"][0][0]), atol=1e-6)


def test_flake_layout_and_the_plan_are_linked():
    spec, flow = _flow_spec()
    for k in ("contact_gate", "electrodes"):                           # the TMDs are not on the chip yet: only BG is
        for u in ("TMD1", "TMD2"):
            flow.step(k).outlines.pop(u)
    spec.layer("BG").outline = [list(q) for q in geo.transform(FLAKES["BG"][0], -15, -100, -50)]
    spec.layer("TMD1").outline = [[-6.0, -4.0], [6.0, -4.0], [6.0, 4.0], [-6.0, 4.0]]
    st = flow.step("hall_bars")
    st.plan["TMD1"] = ef.default_plan(spec, "TMD1", (0.0, 0.0))
    assert ef.plan_from_layout(spec, flow, "TMD1")                      # the plan = where Flake layout has it
    m, t = ef.layout_frame(spec, ef.on_chip_outlines(spec, flow))
    expect = [tuple(m @ np.asarray(q) + t) for q in spec.layer("TMD1").outline]
    assert np.allclose(st.outlines["TMD1"][0], expect, atol=1e-6) and st.located["TMD1"]["how"] == "planned"
    # moved in Flake layout → the plan moves the same way on the chip
    spec.layer("TMD1").outline = [[x + 5.0, y] for x, y in spec.layer("TMD1").outline]
    assert ef.plan_from_layout(spec, flow, "TMD1")
    assert np.allclose(np.asarray(st.outlines["TMD1"][0]) - np.asarray(expect), m @ np.array([5.0, 0.0]), atol=1e-6)
    # the plan dragged / turned on the EBL canvas → Flake layout follows
    before = np.asarray(spec.layer("TMD1").outline)
    st.plan["TMD1"]["cx"] += 3.0
    ef.apply_plan(spec, st)
    assert ef.layout_from_plan(spec, flow, st, "TMD1")
    assert np.allclose(np.asarray(spec.layer("TMD1").outline) - before, np.linalg.inv(m) @ np.array([3.0, 0.0]), atol=1e-6)
    st.plan["TMD1"]["angle"] += 30.0
    ef.apply_plan(spec, st)
    planned = np.asarray(st.outlines["TMD1"][0])
    assert ef.layout_from_plan(spec, flow, st, "TMD1")
    assert np.allclose(np.asarray(st.outlines["TMD1"][0]), planned, atol=1e-6)   # the chip outline stays where it was put
    back = np.asarray(spec.layer("TMD1").outline) @ m.T + t
    assert np.allclose(back, planned, atol=1e-6)
    # a fabricated flake is not planned from Flake layout
    assert not ef.plan_from_layout(spec, flow, "BG")


def test_moving_a_planned_flake_in_flake_layout_moves_its_plan(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, flow = _flow_spec()
    for k in ("contact_gate", "electrodes"):
        for u in ("TMD1", "TMD2"):
            flow.step(k).outlines.pop(u)
    spec.layer("TMD1").outline = [[-6.0, -4.0], [6.0, -4.0], [6.0, 4.0], [-6.0, 4.0]]
    st = flow.step("hall_bars")
    st.plan["TMD1"] = ef.default_plan(spec, "TMD1", (0.0, 0.0))
    ef.plan_from_layout(spec, flow, "TMD1")
    flow.store(spec)
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    before = np.asarray(p.flow.step("hall_bars").outlines["TMD1"][0])
    moved = [[x + 4.0, y + 1.0] for x, y in w.spec.layer("TMD1").outline]
    w._layout_edited("outline:TMD1", moved)                            # dragged on the Flake layout canvas
    after = np.asarray(p.flow.step("hall_bars").outlines["TMD1"][0])
    m, _t = ef.layout_frame(w.spec, ef.on_chip_outlines(w.spec, p.flow))
    assert np.allclose(after - before, m @ np.array([4.0, 1.0]), atol=1e-6)
    w.dirty = False
    w.close()


def test_ebl_canvas_shows_where_flakes_not_on_the_chip_are_planned(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, flow = _flow_spec()
    spec.layer("TG").outline = [[-30.0, 20.0], [30.0, 20.0], [30.0, 26.0], [-30.0, 26.0]]
    flow.store(spec)
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("hall_bars")
    later = ef.not_yet_on_chip(w.spec, p.flow, "hall_bars")
    assert "TG" in later and "BG" not in later
    m, t = ef.layout_frame(w.spec, ef.on_chip_outlines(w.spec, p.flow))
    assert np.allclose(later["TG"], np.asarray(w.spec.layer("TG").outline) @ m.T + t, atol=1e-6)
    keys = [k for _g, _h, items in p.canvas_layers() for k, _t in items]
    assert "plan:TG" in keys and not p.layer_shown("plan:TG")           # off until ticked
    n0 = len(w.ebl_canvas.overlays)
    p.set_layer_visible("plan:TG", True)
    assert p.layer_shown("plan:TG") and len(w.ebl_canvas.overlays) == n0 + 1
    p.remember_ui()
    assert "TG" in ef.Flow.of(w.spec).ui["shown_plans"]
    w.dirty = False
    w.close()


def test_flake_layout_follows_the_ticks_and_shows_metal(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, flow = _flow_spec()
    spec.layer("TG").outline = [[-30.0, 20.0], [30.0, 20.0], [30.0, 26.0], [-30.0, 26.0]]
    ef.generate(spec, flow, "hall_bars", cfg, bondpads.default_preset())
    flow.store(spec)
    w.adopt_spec(spec, "new")
    c = w.layout_canvas
    assert "outline:TG" in c.polys and "outline:BG" in c.polys
    w.view3d.set_hidden({"TG"})                                         # unticked in Flake images
    assert "outline:TG" not in c.polys and "outline:BG" in c.polys
    w.flake_panel.show_all()
    assert "outline:TG" in c.polys
    n0 = len(c.overlays)
    w.layout_metal.setChecked(True)                                     # the fingers, where they are relative to the flakes
    fingers = ef.layout_patterns(w.spec, w.ebl_flow.flow)[w.spec.by_function("hall_bar_contacts")[0].uid]
    assert len(c.overlays) - n0 == len(fingers) > 0
    w.dirty = False
    w.close()


def test_narrow_bus_leads_taper_smoothly():
    """A lead narrower than the bus narrows gradually from the bus width (no sudden neck)."""
    poly = fg.taper_path((0.0, 0.0), (20.0, 0.0), 1.6, 0.6, 20.0)        # one taper over the whole lead
    assert len(poly) == 4 and geo.poly_area(poly) == pytest.approx(20 * (1.6 + 0.6) / 2, rel=1e-6)
    ys = sorted({round(abs(y), 6) for _x, y in poly})
    assert ys == [0.3, 0.8]
    part = fg.taper_path((0.0, 0.0), (20.0, 0.0), 1.6, 0.6, 5.0)         # tapered over 5 µm, then straight
    assert len(part) == 6 and geo.poly_area(part) == pytest.approx(5 * 1.1 + 15 * 0.6, rel=1e-6)
    # the steepest taper allowed: each side at most TAPER_MAX_HALF_ANGLE_DEG
    t_min = (1.6 - 0.6) / 2 / math.tan(math.radians(fg.TAPER_MAX_HALF_ANGLE_DEG))
    steep = fg.taper_path((0.0, 0.0), (20.0, 0.0), 1.6, 0.6, t_min)
    side = [q for q in steep if q[1] > 0][:2]
    assert math.degrees(math.atan2(abs(side[0][1] - side[1][1]), abs(side[1][0] - side[0][0]))) == pytest.approx(
        fg.TAPER_MAX_HALF_ANGLE_DEG, abs=1e-6)
    # a comb whose lead had to narrow: nowhere narrower than the lead's end width (no neck)
    arr = fg.FingerArray((0.0, 0.0), 0.0, 6, fg.FingerParams(cg_tooth=3.0))
    comb = fg.contact_gate_comb(arr, skip={"top": "first", "bottom": "last"}, dirs={"top": 1.0, "bottom": -1.0})
    hs = arr.span / 2
    # an hBN whose bottom-left arm is a 1.4 µm-wide strip: only a narrower lead (0.5 µm margin each side) reaches into it
    b = comb["bottom"]
    bus = np.asarray(arr.to_local(b["bus"]))
    v0 = float(bus[:, 1].mean())
    # room to taper for 9 µm past the bus end, then the strip
    hbn = [arr.to_chip([(-20, -25), (hs + 4, -25), (hs + 4, 25), (-20, 25), (-20, v0 + 0.7),
                        (-40, v0 + 0.7), (-40, v0 - 0.7), (-20, v0 - 0.7)])]
    out, reach = fg.extend_buses(arr, comb, {"dirs": {"top": 1.0, "bottom": -1.0}}, hbn, 0.25, None, 0.4, min_width=0.5)
    assert "bottom" in reach and reach["bottom"][2] < arr.params.cg_bus_width
    lw = reach["bottom"][2]
    assert not geo.width_violations(geo.region(out["bottom"]["polygon"]), lw * 0.999)
    assert lw >= 0.5 - 1e-6                                              # never below the hard minimum


def test_bus_lead_is_as_wide_as_the_room_allows():
    """The lead keeps the bus width wherever it fits, narrows only through a tight stretch (gradually) and widens back
    to the bus width after it — never wider than the bus, never narrower than the hard minimum."""
    arr = fg.FingerArray((0.0, 0.0), 0.0, 6, fg.FingerParams(cg_tooth=3.0))
    comb = fg.contact_gate_comb(arr, skip={"top": "first", "bottom": "last"}, dirs={"top": 1.0, "bottom": -1.0})
    hs = arr.span / 2
    bus = np.asarray(arr.to_local(comb["bottom"]["bus"]))
    v0 = float(bus[:, 1].mean())
    # main hBN, then a 1.4 µm-wide neck (0.25 µm margin: 0.9 µm of room) from u = −20 to −24, then a wide pad to −40
    # whose lower edge continues the neck's (the far corner lies straight on)
    local = [(-20, -25), (hs + 4, -25), (hs + 4, 25), (-20, 25), (-20, v0 + 0.7), (-24, v0 + 0.7), (-24, v0 + 6),
             (-40, v0 + 6), (-40, v0 - 0.7), (-20, v0 - 0.7)]
    hbn = [arr.to_chip(local)]
    out, reach = fg.extend_buses(arr, comb, {"dirs": {"top": 1.0, "bottom": -1.0}}, hbn, 0.25, None, 0.4, min_width=0.5)
    b = out["bottom"]
    W = arr.params.cg_bus_width
    assert reach["bottom"][2] == pytest.approx(0.9, abs=0.12)           # about as wide as the neck allows
    assert b["end_width"] > reach["bottom"][2] + 0.1                     # wider again past the neck
    assert arr.to_local([b["reach"]])[0][0] < -30                         # and on into the pad
    reg = geo.region(b["polygon"])
    assert not geo.width_violations(reg, reach["bottom"][2] * 0.99)      # no neck narrower than that
    # no part wider than the bus: a 1.6 µm square fits nowhere in the lead beyond the neck except as the bus width
    lead = (reg & geo.region(arr.to_chip([(-40, v0 - 6), (-25, v0 - 6), (-25, v0 + 6), (-40, v0 + 6)]))).merged()
    _x0, _y0, _x1, _y1 = geo.bbox(lead)
    assert min(_x1 - _x0, _y1 - _y0) <= W + 0.02
    # the width changes no steeper than the taper limit: every side edge of the lead within it of the lead's axis
    poly = [arr.to_local([q])[0] for q in b["polygon"]]
    for (u1, v1), (u2, v2) in zip(poly, poly[1:] + poly[:1]):
        if min(u1, u2) < -20.5 and abs(u2 - u1) > 0.05:
            ang = math.degrees(math.atan2(abs(v2 - v1), abs(u2 - u1)))
            assert ang <= fg.TAPER_MAX_HALF_ANGLE_DEG + 1.0 or ang >= 89.0


def test_router_goes_round_walls_and_avoids_covered_gates():
    """The electrode router: a lead goes round metal in its way (never through it), keeps the spacing, and goes round a
    soft obstacle (a covered gate flake) when there is a way, crossing it only when there is none."""
    from ustack.router import Net, Obstacle, route_nets, wire
    wall = [(10.0, -30.0), (12.0, -30.0), (12.0, 30.0), (10.0, 30.0)]               # metal between start and target
    net = Net("A", (0.0, 0.0), (40.0, 0.0), 1.6)
    paths, crossed, notes = route_nets([net], [Obstacle([wall], True, 0.8)], 0.8, box=(-10, -50, 50, 50))
    pts = paths["A"]
    assert not notes and pts[0] == (0.0, 0.0) and pts[-1] == (40.0, 0.0) and len(pts) >= 3     # bent round the wall
    lead = wire(pts, 1.6)
    assert (lead & geo.region(wall)).is_empty() and geo.region_distance(lead, geo.region(wall)) >= 0.8 - 1e-3
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 0.6 for a, b in zip(pts, pts[1:]))  # straight pieces, not a staircase
    # a covered gate across the way with a gap at the top: gone round; closed: crossed (and reported)
    gate = [(18.0, -40.0), (22.0, -40.0), (22.0, 20.0), (18.0, 20.0)]
    paths, crossed, _n = route_nets([net], [Obstacle([gate], False, 0.7)], 0.8, box=(-10, -50, 50, 50))
    assert crossed["A"] == 0.0 and (wire(paths["A"], 1.6) & geo.region(gate)).is_empty()
    closed = [(18.0, -50.0), (22.0, -50.0), (22.0, 50.0), (18.0, 50.0)]
    paths, crossed, _n = route_nets([net], [Obstacle([closed], False, 0.7)], 0.8, box=(-10, -50, 50, 50))
    assert paths["A"] and 0 < crossed["A"] < 10
    # two leads whose straight lines would cross: routed without touching
    a, b = Net("A", (0.0, 0.0), (40.0, 20.0), 1.6), Net("B", (0.0, 20.0), (40.0, 0.0), 1.6)
    paths, _c, notes = route_nets([a, b], [], 0.8, box=(-10, -30, 50, 50))
    if not notes:
        assert geo.region_distance(wire(paths["A"], 1.6), wire(paths["B"], 1.6)) >= 0.8 - 1e-3


def test_finger_electrodes_overlap_a_bit_and_stay_off_the_top_hbn(cfg):
    """User, 2026-11-20: the electrodes overlapped the fingers too far and the gold landed on the top hBN. Each starts on
    its finger at most electrodes.finger_overlap_um back from the far end, and only where its metal lies
    electrodes.flake_clearance_um outside every flake above the fingers."""
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5, "preset:electrodes.pad_landing_um": 10.0})   # TEST VALUES
    arr = fg.FingerArray.from_json(ef.generate(spec, flow, "hall_bars", cfg, bp).array)
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    p = arr.params
    reach = p.gap / 2 + p.land_from_tip + 3.0                       # the top hBN covers the fingers past the usual landing
    tbn = [arr.to_chip([(-60.0, -reach), (60.0, -reach), (60.0, reach), (-60.0, reach)])]
    flow.step("electrodes").outlines["TBN"] = tbn
    with pytest.raises(MissingValues) as exc:                       # the clearance is the lab's: asked
        ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    assert exc.value.items[0].key == "preset:electrodes.flake_clearance_um"
    apply_answers(cfg, {"preset:electrodes.flake_clearance_um": 1.0})                                  # TEST VALUE
    el = ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    metal = geo.regions(el.shapes["el_fingers"]).merged()
    assert (metal & geo.sized(geo.regions(tbn), 0.99)).is_empty()   # ≥ 1 µm outside the top hBN
    overlap = cfg.preset("electrodes.finger_overlap_um")
    for f in arr.fingers():                                         # a bit of each finger, never more than the preset
        on = metal & geo.region(f.polygon)
        assert not on.is_empty()
        x0, y0, x1, y1 = geo.bbox(on)
        assert math.hypot(x1 - x0, y1 - y0) <= overlap + p.electrode_width + 0.1
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "electrodes", cfg)}
    row = next(r for k, r in rows.items() if k.startswith("Hall-bar electrodes off"))
    assert row["status"] == "met" and "hBN" in row["rule"]
    # a top hBN over every finger's whole length: the electrode lands on its last width, and Check says so
    tbn_all = [arr.to_chip([(-80.0, -80.0), (80.0, -80.0), (80.0, 80.0), (-80.0, 80.0)])]
    flow.step("electrodes").outlines["TBN"] = tbn_all
    el = ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    assert any("no stretch of the finger is clear" in n for n in el.group_notes["fingers"])
    row = next(r for r in ef.rule_report(spec, flow, "electrodes", cfg) if r["rule"].startswith("Hall-bar electrodes off"))
    assert row["status"] == "violated"
    assert any(i["code"] == "cg_rule" and "Hall-bar electrodes off" in i["message"] for i in el.issues)
