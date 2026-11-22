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
    # the top gate sticks out on the side away from the bottom gate's contact (user, 2026-11-21: they cannot touch)
    assert not any("also sticks out" in n for n in best.notes), best.notes
    bg, tg = np.asarray(geo.centroid(best.zones["bg"])), np.asarray(geo.centroid(best.zones["tg"]))
    mid = np.asarray(geo.centroid(best.outlines["TMD1"]))
    assert float(np.dot(bg - mid, tg - mid)) < 0
    # the moved borders keep their shape and size; the twist between the device layers is kept
    before = {l.uid: [tuple(q) for q in l.outline] for l in spec.layers if l.outline}
    moved = al.apply_candidate(spec, best)
    assert "TMD1" not in moved and set(moved) <= set(before)
    for uid in moved:
        after = spec.layer(uid).outline
        assert abs(geo.poly_area(after)) == pytest.approx(abs(geo.poly_area(before[uid])), rel=1e-6)
    assert best.poses["TMD2"][0] == pytest.approx(0.0, abs=1e-9)
    # the layout as applied: the Stage 2 generators place the Hall bars and the contact gate on it
    s, f = al._scratch(spec, {l.uid: [tuple(q) for q in l.outline] for l in spec.layers if l.outline})
    f.step("hall_bars").max_pairs = 6
    hb = ef.generate(s, f, "hall_bars", cfg, None)
    assert fg.FingerArray.from_json(hb.array).n == 6
    ef.generate(s, f, "contact_gate", cfg, None)
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
    d.apply()
    assert [list(map(tuple, w.spec.layer("BBN").outline))] != before
    w.undo()                                               # one undo step puts them back
    app.processEvents()
    assert [list(map(tuple, w.spec.layer("BBN").outline))] == before
    d.close()
    w.dirty = False
    w.close()
