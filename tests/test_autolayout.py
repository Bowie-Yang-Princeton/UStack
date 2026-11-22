"""Flake layout › Auto design (user, 2026-11-21): the traced flakes placed so the rules hold and the Stage 2 patterns fit
(``ustack.autolayout``), on synthetic flakes (not lab data)."""

import math
import os

import numpy as np
import pytest

from ustack import autolayout as al, ebl_flow as ef, fingers as fg, geometry as geo, templates
from ustack.config import MissingValues, apply_answers

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: synthetic traced borders (µm, design frame), roughly the sizes of the lab's tWSe2 flakes
SHAPES = {
    "BG": [(-60, -14), (55, -18), (62, -10), (58, 16), (-58, 12)],
    "BBN": [(-55, -40), (50, -45), (62, 0), (48, 46), (-40, 50), (-62, 5)],
    "TMD1": [(-12, -8), (12, -8), (12, 8), (-12, 8)],
    "TMD2": [(-14, -7), (15, -7), (15, 8), (-14, 8)],
    "MBN": [(-34, -17), (34, -18), (36, 17), (-33, 18)],
    "TBN": [(-32, -20), (33, -19), (32, 20), (-31, 21)],
    "TG": [(-60, -11), (58, -12), (60, 11), (-59, 12)],
    "CAP": [(-40, -24), (40, -25), (41, 25), (-40, 24)],
}
TEST_VALUES = {"rule:stack.alignment_shift_um": 0.0, "preset:fingers.bbn_margin_um": 3.0, "preset:cg.hbn_margin_um": 0.5,
               "preset:cg.bus_tmd_gap_um": 0.5, "preset:electrodes.flake_clearance_um": 1.0}      # TEST VALUES


def _spec(scramble: int | None = 1):
    spec = templates.new_device_from(templates.load_template("xu_tmd"), "auto design test")
    rng = np.random.default_rng(scramble) if scramble is not None else None
    for layer in spec.layers:
        if layer.uid in SHAPES:
            poly = [tuple(map(float, q)) for q in SHAPES[layer.uid]]
            if rng is not None and layer.uid != "TMD1":
                turn = 0.0 if layer.uid == "TMD2" else float(rng.uniform(0, 360))   # the twist kept: TMD2 only shifted
                poly = al._moved(al._turn(poly, turn, geo.centroid(poly)), float(rng.uniform(-30, 30)), float(rng.uniform(-30, 30)))
            layer.outline = [list(q) for q in poly]
    return spec


def test_pair_estimate_agrees_with_the_exact_fit():
    p = fg.FingerParams()
    region = [(0.0, 0.0), (20.0, 0.0), (20.0, 12.0), (0.0, 12.0)]
    exact = fg.fit_finger_array([region], p).n
    res = 0.25
    m = al._mask([region], -2.0, -2.0, res, (int(16 / res), int(24 / res)))
    est = al._pairs(m, res, p, np.arange(0.0, 180.0, 6.0))
    assert est["n"] in (exact, exact - 1)                  # slightly conservative, never more
    assert min(est["angle"], 180 - est["angle"]) < 6.1     # along the long side
    capped = al._pairs(m, res, p, np.arange(0.0, 180.0, 6.0), target=5, union=m)
    assert capped["n"] == 5


def test_placement_search_holds_what_it_must_and_keeps_off_what_it_must_not():
    flake = [(0.0, 0.0), (30.0, 0.0), (30.0, 12.0), (0.0, 12.0)]
    must = [[(50.0, 50.0), (60.0, 50.0), (60.0, 56.0), (50.0, 56.0)]]
    hard = [[(48.0, 40.0), (75.0, 40.0), (75.0, 47.0), (48.0, 47.0)]]       # below the target: never covered
    got = al.search_flake(flake, al.Targets(must=must, margin=1.0, hard=hard), keep=2)
    assert got
    for _s, pose, placed in got:
        reg = geo.region(placed)
        assert (geo.regions(must) - geo.sized(reg, -1.0)).is_empty()        # inside by the margin, exactly
        assert (reg & geo.regions(hard)).is_empty()
        assert abs(geo.poly_area(placed)) == pytest.approx(30 * 12, rel=1e-6)   # shape and size kept


def test_auto_design_places_the_flakes_for_the_patterns(cfg):
    spec = _spec(scramble=2)
    flow = ef.Flow()
    with pytest.raises(MissingValues) as exc:              # the lab values it needs: asked all at once
        al.auto_design(spec, cfg, flow, pairs=6)
    keys = {i.key for i in exc.value.items}
    assert {"preset:fingers.bbn_margin_um", "preset:cg.hbn_margin_um", "preset:electrodes.flake_clearance_um"} <= keys
    apply_answers(cfg, {k: v for k, v in TEST_VALUES.items() if k in keys})
    steps = []
    cands = al.auto_design(spec, cfg, flow, pairs=6, progress=lambda t, f: steps.append(t))
    assert cands and steps
    best = cands[0]
    assert best.pairs == 6 and best.comb and not best.blocked and not best.errors
    assert set(best.zones) == {"bg", "tg"}                # room for both gate contacts
    # the gates stick out on opposite sides (user, 2026-11-22: ≥ 120° apart, ideally 180°), the bottom gate's long side
    # along the channel's (the finger array's axis)
    assert best.gate_angle is not None and best.gate_angle >= al.GATE_MIN_APART_DEG, best.notes
    assert abs(math.remainder(al._long_axis_deg(best.outlines["BG"]) - best.axis, 180.0)) < 1.0
    # the contact gate lies on the bottom hBN too: never over bare bottom-gate graphite (user, 2026-11-22: a gate leak)
    comb = geo.regions(best.comb).merged()
    assert (comb - geo.region(best.outlines["BBN"])).is_empty()
    # every bus has its lead and landing; the middle and top hBN on the bottom hBN where they can be (user, 2026-11-22)
    assert len(best.reach) == len(best.comb) and not best.blocked
    # the moved borders keep their shape and size; the twist between the device layers is kept
    before = {l.uid: [tuple(q) for q in l.outline] for l in spec.layers if l.outline}
    moved = al.apply_candidate(spec, best)
    assert "TMD1" not in moved and set(moved) <= set(before)
    for uid in moved:
        after = spec.layer(uid).outline
        assert abs(geo.poly_area(after)) == pytest.approx(abs(geo.poly_area(before[uid])), rel=1e-6)
    assert best.poses["TMD2"][0] == pytest.approx(0.0, abs=1e-9)
    # its planned metal is kept (user, 2026-11-21): Flake layout and the Structure tab show it while nothing is generated
    pt = spec.by_function("hall_bar_contacts")[0].uid
    cg = spec.by_function("contact_gate")[0].uid
    src = {}
    pats = ef.layout_patterns(spec, ef.Flow(), None, src)
    assert len(pats[pt]) == len(best.fingers) and pats[cg] and src[pt] == src[cg] == "plan"
    chip, text = ef.structure_footprints(spec, ef.Flow())
    assert len(chip[pt]) == len(best.fingers) and "planned by the auto design" in text
    # the layout as applied: the Stage 2 generators place the Hall bars and the contact gate on it
    s, f = al._scratch(spec, {l.uid: [tuple(q) for q in l.outline] for l in spec.layers if l.outline})
    f.step("hall_bars").max_pairs = 6
    hb = ef.generate(s, f, "hall_bars", cfg, None)
    assert fg.FingerArray.from_json(hb.array).n == 6
    src = {}
    ef.layout_patterns(s, f, None, src)
    assert src[pt] == "stage2" and src[cg] == "plan"                  # a generated pattern takes the plan's place
    ef.generate(s, f, "contact_gate", cfg, None)
    src = {}
    pats = ef.layout_patterns(s, f, None, src)
    assert src[cg] == "stage2" and len(pats[pt]) == len(f.step("hall_bars").shapes["prepattern_contacts"])
    # nothing to move: said so
    with pytest.raises(ValueError, match="No flake may move"):
        al.auto_design(spec, cfg, flow, movable=set(), pairs=6)


def test_auto_design_abort_stops_at_once(cfg):
    """Abort (user, 2026-11-21): the window is free at once and the run itself ends within a moment."""
    import time
    from ustack.app import make_app
    from ustack.ui.auto_design_window import AutoDesignDialog
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    apply_answers(cfg, TEST_VALUES)
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.adopt_spec(_spec(scramble=3), "new")
    d = AutoDesignDialog(w)
    d.pairs.setValue(6)
    d.run()
    assert d.abort_btn.isVisibleTo(d) and not d.run_btn.isEnabled()
    t = time.time()
    while time.time() - t < 3:
        app.processEvents()
        time.sleep(0.05)
    worker = d.worker
    assert worker is not None and worker.isRunning()
    d.abort_btn.click()
    assert d.run_btn.isEnabled() and not d.abort_btn.isVisibleTo(d) and "Aborted" in d.state.text() and not d.cands
    t = time.time()
    assert worker.wait(5000), "the run did not stop"
    assert time.time() - t < 5
    app.processEvents()
    assert not d.cands
    d.close()
    w.dirty = False
    w.close()


def test_auto_design_window_runs_and_applies(cfg):
    from ustack.app import make_app
    from ustack.ui.auto_design_window import AutoDesignDialog
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    apply_answers(cfg, TEST_VALUES)
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.adopt_spec(_spec(scramble=4), "new")
    d = AutoDesignDialog(w)
    assert d.list.topLevelItemCount() == len([l for l in w.spec.layers if l.uid in SHAPES])
    d.pairs.setValue(6)
    d.run()
    import time
    t = time.time()
    while d.worker is not None and d.worker.isRunning() and time.time() - t < 600:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    assert d.cands, d.state.text()
    before = [list(map(tuple, w.spec.layer("BBN").outline))]
    d.show()
    d.apply()
    assert not d.isVisible() and d.result() == 1          # Apply closes the window (user, 2026-11-21)
    assert [list(map(tuple, w.spec.layer("BBN").outline))] != before
    # the planned metal: in Flake layout (Show metal parts on), the Structure tab, and the window opened again
    assert ef.auto_plan(w.spec) and w.layout_metal.isChecked()
    n_items = len(w.layout_canvas.scene().items())
    w.layout_metal.setChecked(False)
    assert len(w.layout_canvas.scene().items()) < n_items
    w.layout_metal.setChecked(True)
    app.processEvents()
    w.refresh_structure()
    pt = w.spec.by_function("hall_bar_contacts")[0].uid
    assert next(s_ for s_ in w.view3d.slabs if s_.uid == pt).polys and "auto design" in w.title_label.text()
    d2 = AutoDesignDialog(w)
    assert "Applied layout" in d2.state.text()
    assert len(d2.canvas.scene().items()) > len([l for l in w.spec.layers if l.outline])
    # metal kept as it is (user, 2026-11-22): the applied plan's fingers / contact gate can be kept; keeping the comb keeps
    # the middle hBN and the device layers; Fix all / Free all
    assert set(d2.keep_boxes) == {"hall_bars", "contact_gate"} and not d2.keep_boxes["hall_bars"].isChecked()
    d2.keep_boxes["contact_gate"].setChecked(True)
    assert d2.keep_boxes["hall_bars"].isChecked() and not d2.pairs.isEnabled()
    assert set(d2._fixed()) == {"hall_bars", "contact_gate"}
    assert not d2._movable() & {"MBN", "TMD1", "TMD2"} and "TG" in d2._movable()
    d2.keep_boxes["contact_gate"].setChecked(False)
    assert {"MBN", "TMD1", "TMD2"} <= d2._movable()
    d2._tick_all(False)
    assert not d2._movable()
    d2._tick_all(True)
    assert "BBN" in d2._movable()
    # the layouts of the last run are listed again, the applied one marked; another one can be applied from the list
    assert len(d2.cands) == len(d.cands) and "applied" in d2.results.topLevelItem(0).text(0)
    undos = 1
    if len(d2.cands) > 1:
        d2.show()
        d2.results.setCurrentItem(d2.results.topLevelItem(1))
        d2.apply()
        undos = 2
        app.processEvents()
        for uid, poly in d2.cands[1].outlines.items():
            assert np.allclose(np.asarray(w.spec.layer(uid).outline, float), np.asarray(poly, float), atol=1e-6), uid
        d3 = AutoDesignDialog(w)
        assert "applied" in d3.results.topLevelItem(1).text(0) and d3.results.currentItem() is d3.results.topLevelItem(1)
        d3.close()
    d2.close()
    for _ in range(undos):                                 # one undo step per Apply puts them back
        w.undo()
        app.processEvents()
    assert [list(map(tuple, w.spec.layer("BBN").outline))] == before and ef.auto_plan(w.spec) is None
    d.close()
    w.dirty = False
    w.close()


def _turn_between(a, b) -> float:
    m, _t = ef._rigid([tuple(q) for q in a], [tuple(q) for q in b])
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def test_device_layers_take_their_twist():
    """User, 2026-11-22: the device layers lie as in their pictures, each turned by its twist (relative to the device layer
    below, counter-clockwise); a device layer turned by hand turns the others with it."""
    spec = _spec(scramble=None)
    t1, t2 = spec.layer("TMD1"), spec.layer("TMD2")
    before = [list(q) for q in t2.outline]
    assert ef.apply_device_twist(spec) == {}                       # no twist given: as in the pictures
    t2.twist_deg.set(30.0, "user")                                 # TEST VALUE
    assert ef.apply_device_twist(spec) == pytest.approx({"TMD2": 30.0})
    assert _turn_between(before, t2.outline) == pytest.approx(30.0)
    assert geo.centroid(t2.outline) == pytest.approx(geo.centroid(before))
    assert ef.apply_device_twist(spec) == {}                       # kept with the device: not turned again
    # with images: the picture's angle is the turn
    for layer, ang in ((t1, 10.0), (t2, 70.0)):
        layer.image = {"path": "x.png", "um_per_px": 0.1, "angle_deg": ang, "x": 0.0, "y": 0.0}
    got = ef.apply_device_twist(spec)
    assert got == pytest.approx({"TMD2": -30.0}) and t2.image["angle_deg"] == pytest.approx(40.0)
    # turned by hand: the others follow (anchor)
    o1 = [list(q) for q in t1.outline]
    ef.turn_layer(t2, 15.0, spec)
    assert ef.apply_device_twist(spec, anchor="TMD2") == pytest.approx({"TMD1": 15.0})
    assert _turn_between(o1, t1.outline) == pytest.approx(15.0)
    assert ef.apply_device_twist(spec, skip={"TMD1"}, ref_turn=0.0) == pytest.approx({"TMD2": -25.0})


def test_auto_design_starts_from_the_pictures_turned_by_the_twist(cfg):
    apply_answers(cfg, TEST_VALUES)
    spec = _spec(scramble=5)
    spec.layer("TMD2").twist_deg.set(12.0, "user")                 # TEST VALUE
    cands = al.auto_design(spec, cfg, ef.Flow(), pairs=4)
    assert cands
    best = cands[0]
    assert best.poses["TMD2"][0] == pytest.approx(12.0, abs=1e-6)  # the twist, nothing else (kept)
    assert _turn_between(SHAPES["TMD2"], best.outlines["TMD2"]) - _turn_between(SHAPES["TMD1"], best.outlines["TMD1"]) == \
        pytest.approx(12.0, abs=1e-6)
    al.apply_candidate(spec, best)
    for uid, poly in best.outlines.items():
        assert np.allclose(np.asarray(spec.layer(uid).outline, float), np.asarray(poly, float), atol=1e-6)


def test_flake_layout_shows_the_twist(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.adopt_spec(_spec(scramble=None), "new")
    t1, t2 = w.spec.layer("TMD1"), w.spec.layer("TMD2")
    b1, b2 = [list(q) for q in t1.outline], [list(q) for q in t2.outline]
    t2.twist_deg.set(20.0, "user")                                 # TEST VALUE (typed in the Stack tab)
    w.spec_changed("value")
    assert _turn_between(b2, t2.outline) == pytest.approx(20.0)
    assert np.allclose(np.asarray(w.layout_canvas.polys["outline:TMD2"].points), np.asarray(t2.outline), atol=1e-9)
    chip = __import__("ustack.scene3d", fromlist=["build_scene"]).build_scene(w.spec, cfg)
    assert np.allclose(np.asarray(next(s_ for s_ in chip if s_.uid == "TMD2").polys[0]), np.asarray(t2.outline), atol=1e-6)
    # TMD1 turned with the rotate handle: TMD2 turns with it (the twist kept)
    w.layout_canvas.set_active_polygon("outline:TMD1")
    w.layout_canvas.rotate_active(10.0)
    app.processEvents()
    assert _turn_between(b1, t1.outline) == pytest.approx(10.0, abs=1e-6)
    assert _turn_between(b2, t2.outline) == pytest.approx(30.0, abs=1e-6)
    w.dirty = False
    w.close()


def test_reference_angle_sets_the_lattice_direction(cfg, monkeypatch):
    """User, 2026-11-22: each image has a reference angle (the lattice direction from the image's long side, e.g. by AFM);
    the twist turns the layers so their lattices lie at it; another flake given a twist follows the nearest device layer."""
    spec = _spec(scramble=None)
    t1, t2, tbn = spec.layer("TMD1"), spec.layer("TMD2"), spec.layer("TBN")
    for layer in (t1, t2, tbn):
        layer.image = {"path": "x.png", "um_per_px": 0.1, "angle_deg": 0.0, "x": 0.0, "y": 0.0}
    t2.image["ref_angle_deg"] = 20.0                               # TEST VALUE: TMD2's lattice 20° from its image's long side
    t2.twist_deg.set(5.0, "user")                                  # TEST VALUE
    assert ef.lattice_turn(t2, spec) == pytest.approx(20.0)
    got = ef.apply_device_twist(spec)
    assert got == pytest.approx({"TMD2": -15.0})                    # lattice at 0 + 5°: the image turned by −15°
    assert ef.lattice_turn(t2, spec) - ef.lattice_turn(t1, spec) == pytest.approx(5.0)
    t2.image["mirror"] = True                                      # a mirrored image reverses the reference angle
    assert ef.lattice_turn(t2, spec) == pytest.approx(-15.0 + 160.0)
    t2.image["mirror"] = False
    # an hBN given a twist (moiré with the device): turned relative to the nearest device layer (TMD2 here)
    assert "TBN" not in ef.twist_layer_uids(spec)
    tbn.image["ref_angle_deg"] = -10.0                             # TEST VALUE
    tbn.twist_deg.set(0.77, "user")                                # TEST VALUE
    assert "TBN" in ef.twist_layer_uids(spec)
    ef.apply_device_twist(spec)
    assert ef.lattice_turn(tbn, spec) - ef.lattice_turn(t2, spec) == pytest.approx(0.77)
    # the panel: a reference angle field with a circled question mark that explains it
    from ustack.app import make_app
    from ustack.ui import widgets
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    s2 = _spec(scramble=None)
    s2.layer("TMD2").image = {"path": "x.png", "um_per_px": 0.1, "angle_deg": 0.0, "x": 0.0, "y": 0.0}
    w.adopt_spec(s2, "new")
    w.flake_panel._select("TMD2")
    p = w.flake_panel
    assert p.ref_field.value() in (0.0, None)
    shown = []
    monkeypatch.setattr(widgets, "message", lambda parent, title, text, kind="info": shown.append((title, text)))
    p.ref_help.click()
    assert shown and shown[0][0] == "Reference angle" and "long side" in shown[0][1]
    before = [list(q) for q in w.spec.layer("TMD2").outline]
    p._image_param("TMD2", "ref_angle_deg", 30.0)                  # typed: TMD2's border turns so its lattice is at the twist
    assert w.spec.layer("TMD2").image["ref_angle_deg"] == 30.0
    assert _turn_between(before, w.spec.layer("TMD2").outline) == pytest.approx(-30.0)
    w.dirty = False
    w.close()


def test_auto_design_bond_pad_count_is_a_limit_and_impossible_flakes_are_said_at_once(cfg):
    """Device 2026-11-21 (user, 2026-11-22): with no pair count asked, the bond pads' count is an upper limit (as many pairs
    as the flakes hold); a middle hBN that cannot hold the device layers at any turn is reported at once, with why."""
    import time
    apply_answers(cfg, TEST_VALUES)
    spec = _spec(scramble=2)
    flow = ef.Flow()
    flow.step("hall_bars").max_pairs = 40                          # the "bond pads" take 40: more than the flakes hold
    cands = al.auto_design(spec, cfg, flow)
    assert cands and 0 < cands[0].pairs < 40
    assert any("hold at most" in n for n in cands[0].notes)
    with pytest.raises(ValueError, match="at most"):                # asked for explicitly: an error saying how many fit
        al.auto_design(spec, cfg, flow, pairs=40)
    spec.layer("MBN").outline = [[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0], [-6.0, 6.0]]   # far too small for the device
    t = time.time()
    with pytest.raises(ValueError, match="cannot hold"):
        al.auto_design(spec, cfg, flow, pairs=4)
    assert time.time() - t < 60


def test_auto_design_on_a_partly_made_sample(cfg):
    """User, 2026-11-22: re-run the auto design with the back gate and bottom hBN made and the fingers written (kept as they
    are), the device layers and the later flakes only traced; then with the contact gate made too. The top hBN never covers
    a contact-gate lead's end (its electrode would have nowhere to land)."""
    apply_answers(cfg, TEST_VALUES)
    spec = _spec(scramble=2)
    flow = ef.Flow()
    best = al.auto_design(spec, cfg, flow, pairs=6)[0]
    al.apply_candidate(spec, best)
    made = al.made_patterns(spec, flow)                            # nothing in Stage 2: the applied plan
    assert made["hall_bars"]["source"] == "plan" and not made["hall_bars"]["made"]
    arr = made["hall_bars"]["array"]
    assert arr.n == 6 and made["contact_gate"]["reach"]
    keep = {"BG", "BBN"}
    before = {u: [tuple(q) for q in spec.layer(u).outline] for u in keep}
    rng = np.random.default_rng(7)
    for uid in ("TMD1", "TMD2", "MBN", "TBN", "TG", "CAP"):         # traced, not stacked yet: anywhere
        layer = spec.layer(uid)
        turn = 0.0 if uid.startswith("TMD") else float(rng.uniform(0, 360))
        layer.outline = [list(q) for q in al._moved(al._turn(layer.outline, turn, geo.centroid(layer.outline)), 40.0, -25.0)]
    movable = {l.uid for l in al.roles(spec, cfg).flakes} - keep
    cands = al.auto_design(spec, cfg, flow, movable=movable, fixed={"hall_bars": arr})
    assert cands
    c = cands[0]
    assert c.array["center"] == pytest.approx(list(arr.center), abs=1e-3) and c.array["angle_deg"] == pytest.approx(arr.angle_deg)
    assert c.pairs == 6 and any("as made" in n for n in c.notes)
    for u in keep:                                                  # the made flakes stay
        assert np.allclose(np.asarray(c.outlines[u], float), np.asarray(before[u], float))
    tw = geo.regions(ef.twisted_region(spec, {u: [q] for u, q in c.outlines.items()})).merged()
    assert (geo.regions(al._pair_strips(arr)).merged() - tw).is_empty()           # the device holds every pair's tips
    assert c.poses["TMD2"][0] == pytest.approx(c.poses["TMD1"][0], abs=1e-6)     # turned together: the twist kept

    def cg_landings_free(cand):
        st = al.State(dict(cand.outlines), reach=[tuple(q) for q in cand.reach])
        v = al._values(spec, cfg, al.roles(spec, cfg), flow)
        tbn = geo.region(cand.outlines["TBN"])
        return all((geo.sized(geo.region(z), -(v.clearance + v.shift)) & tbn).is_empty() for z in al._cg_zones(st, v))
    assert cg_landings_free(c)
    # that layout applied and its contact gate made too (written on the middle hBN, over the device layers — those are
    # then kept): only the top hBN, top gate and cap are placed, round the comb as it is
    al.apply_candidate(spec, c)
    made = al.made_patterns(spec, flow)
    comb = made["contact_gate"]
    held = {u: [tuple(q) for q in spec.layer(u).outline] for u in ("MBN", "TMD1", "TMD2")}
    for uid in ("TBN", "TG", "CAP"):
        layer = spec.layer(uid)
        layer.outline = [list(q) for q in al._moved(layer.outline, -35.0, 30.0)]
    cands = al.auto_design(spec, cfg, flow, movable=movable, fixed={"hall_bars": made["hall_bars"]["array"], "contact_gate": comb})
    assert cands
    c2 = cands[0]
    assert [list(map(tuple, q)) for q in c2.comb] == [list(map(tuple, q)) for q in comb["polys"]]
    for u, poly in held.items():                                  # kept with the comb, though ticked as movable
        assert np.allclose(np.asarray(c2.outlines[u], float), np.asarray(poly, float))
    assert set(c2.poses) <= {"TBN", "TG", "CAP"}
    assert cg_landings_free(c2)
    with pytest.raises(ValueError, match="with the fingers"):        # a comb cannot be kept without its fingers
        al.auto_design(spec, cfg, flow, movable=movable, fixed={"contact_gate": comb})


def test_shift_click_chooses_several_outlines_to_move_and_turn_together(cfg):
    """User, 2026-11-22 (as in PowerPoint): Shift-click outlines on the canvas to choose several; dragging one moves them all,
    the rotate handle turns them together about their common middle; a plain click elsewhere drops the group."""
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import QtCore, setup_application
    from PyQt6.QtTest import QTest
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.resize(1400, 900)
    w.show()
    w.adopt_spec(_spec(scramble=None), "new")
    w.center_tabs.setCurrentWidget(w.layout_page)
    app.processEvents()
    c = w.layout_canvas
    c.fit()
    app.processEvents()

    def on_line(uid):
        pts = c.polys[f"outline:{uid}"].points
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return c.mapFromScene(QtCore.QPointF((x0 + x1) / 2, (y0 + y1) / 2))
    Qt = QtCore.Qt
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, on_line("BG"))
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, on_line("TG"))
    app.processEvents()
    assert c.multi == {"outline:BG", "outline:TG"} and len(c.selection()) == 2
    before = {u: [tuple(q) for q in w.spec.layer(u).outline] for u in ("BG", "TG", "BBN")}
    # turned together: each border turned by 30° about the group's middle (not its own)
    assert c.rotate_active(30.0)
    app.processEvents()
    for u in ("BG", "TG"):
        m, t = ef._rigid(before[u], [tuple(q) for q in w.spec.layer(u).outline])
        assert math.degrees(math.atan2(m[1, 0], m[0, 0])) == pytest.approx(30.0, abs=1e-6)
    gc = np.mean([geo.centroid(before["BG"]), geo.centroid(before["TG"])], axis=0)   # roughly the group's middle
    moved = [math.dist(geo.centroid(before[u]), geo.centroid(w.spec.layer(u).outline)) for u in ("BG", "TG")]
    assert max(moved) > 1.0                                       # about the common middle, so they moved too
    assert [tuple(q) for q in w.spec.layer("BBN").outline] == before["BBN"]
    assert c.multi == {"outline:BG", "outline:TG"}                # still chosen together after the redraw
    # dragging one of them moves both
    mid = {u: geo.centroid(w.spec.layer(u).outline) for u in ("BG", "TG")}
    p0 = on_line("BG")
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    QTest.mouseMove(c.viewport(), p0 + QtCore.QPoint(40, 0))
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0 + QtCore.QPoint(40, 0))
    app.processEvents()
    shift = [np.subtract(geo.centroid(w.spec.layer(u).outline), mid[u]) for u in ("BG", "TG")]
    assert np.hypot(*shift[0]) > 1.0 and np.allclose(shift[0], shift[1], atol=1e-6)
    # Shift-click again takes one out
    QTest.mouseClick(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, on_line("TG"))
    assert c.multi == {"outline:BG"} and len(c.selection()) == 1
    w.dirty = False
    w.close()
