"""Patterns written at an earlier step found in a later step's images (fingers at the contact-gate step, the contact gate at
the electrodes step), the drawing windows' box selection, and the tick boxes of the lists."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ustack import bondpads, ebl_flow as ef, fingers as fg, geometry as geo, registration as reg
from ustack.alignment import Similarity
from ustack.config import apply_answers

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "WSe2 device training"


def _draw(polys, sim, shape=(1600, 1900)):
    img = np.full((*shape, 3), (175, 120, 95), np.uint8)               # hBN-ish blue
    for p in polys:
        cv2.fillPoly(img, [np.round(sim.inverse(np.asarray(p))).astype(np.int32)], (150, 120, 190))   # pink metal, same grey-ish
    return img


@pytest.mark.parametrize("mirror", [False, True])
def test_locate_pattern_finds_the_move(mirror):
    arr = fg.FingerArray((0.0, 0.0), 12.0, 6, fg.FingerParams())
    polys = [f.polygon for f in arr.fingers()]
    pts = np.vstack(polys)
    true = {"dx": 0.8, "dy": -0.5, "angle": 0.6, "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean())}
    sim = Similarity(0.047, 3.0, 45.0 if mirror else -45.0, 35.0, mirror)
    r = reg.locate_pattern(_draw([ef.move_points(p, true) for p in polys], sim), sim, polys)
    assert r["dx"] == pytest.approx(0.8, abs=0.06) and r["dy"] == pytest.approx(-0.5, abs=0.06)
    assert r["angle"] == pytest.approx(0.6, abs=0.1) and r["inliers"] > 0.8
    with pytest.raises(ValueError, match="coarse"):
        reg.locate_pattern(_draw(polys, sim), Similarity(0.47, 3.0, -45.0, 35.0), polys)


def test_contact_gate_follows_the_fingers_found_written(cfg, tmp_path):
    from test_fingers_flow import _flow_spec
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                    # TEST VALUE
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    cg0 = [list(map(tuple, q)) for q in ef.generate(spec, flow, "contact_gate", cfg, bp).shapes["contact_gates"]]
    polys = hb.shapes["prepattern_contacts"]
    pts = np.vstack(polys)
    true = {"dx": 0.6, "dy": 0.4, "angle": 0.3, "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean())}
    sim = Similarity(0.047, 0.0, float(pts[:, 0].mean()) - 45.0, float(pts[:, 1].mean()) + 37.0)
    img = _draw([ef.move_points(p, true) for p in polys], sim)
    with pytest.raises(ValueError):                                        # nothing written at the electrodes step's images
        ef.locate_written(flow, "electrodes", "contact_gate", [("50x", np.full_like(img, 128), sim)])
    r = ef.locate_written(flow, "contact_gate", "hall_bars", [("50x", img, sim)])
    assert r["dx"] == pytest.approx(0.6, abs=0.06) and r["dy"] == pytest.approx(0.4, abs=0.06)
    assert flow.step("contact_gate").found_patterns["hall_bars"] is r
    # the comb is placed on the fingers as found; the electrodes step sees them there too; the Hall-bar step as designed
    cg = ef.generate(spec, flow, "contact_gate", cfg, bp)
    c0 = np.mean(np.vstack(cg0), 0)
    c1 = np.mean(np.vstack(cg.shapes["contact_gates"]), 0)
    want = np.array(ef.move_points([tuple(c0)], r)[0])
    assert np.linalg.norm(c1 - want) < 0.3
    moved = ef.pattern_shapes(flow, "electrodes", "hall_bars")["prepattern_contacts"][0]
    assert moved[0] == pytest.approx(ef.move_points(polys[0], r)[0])
    assert ef.pattern_shapes(flow, "hall_bars", "hall_bars")["prepattern_contacts"][0] == polys[0]
    assert ef.Flow.from_json(flow.to_json()).step("contact_gate").found_patterns["hall_bars"]["dx"] == r["dx"]


@pytest.mark.skipif(not (LAB / "tWSe2-1_KLayout_session" / "tWSe2-1.lys").is_file(), reason="the lab's batch-1 data is not here")
def test_lab_fingers_and_contact_gate_found_where_written():
    """The lab's Stack 1 / 3 images before the electrodes (placed by its session): the written fingers (layer 5) and the
    contact gate (layer 6) lie within half a µm of the lab's own GDS."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import debris_training as T
    from ustack.gds_io import read_polygons
    from ustack.lab_designs import load_session
    views = load_session(T.SESSION)
    for view in (0, 1):
        im = next(i for i in views[view].images if "before_electrodes" in i.path.name and "_20x_" in i.path.name)
        img = cv2.imread(str(im.path))
        sim = T._sim(im, img.shape[1], img.shape[0])
        P = read_polygons(views[view].gds)
        for layer in ((5, 0), (6, 0)):
            polys = [p for p in P[layer] if max(abs(q[0]) for q in p) < 120 and max(abs(q[1]) for q in p) < 120]
            r = reg.locate_pattern(img, sim, polys)
            assert math.hypot(r["dx"], r["dy"]) < 0.5 and abs(r["angle"]) < 0.5 and r["inliers"] > 0.55


def test_drawing_windows_box_select_and_never_turn(cfg):
    """In the Draw … windows a drag on nothing boxes every shape inside it (Delete removes them all), dragging one of them
    moves them all; the trackpad twist does not turn the view there; the step card reads Generate · Draw · Check, then
    the export."""
    from test_fingers_flow import _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import QtCore, QtWidgets, setup_application
    from ustack.ui.shape_window import ShapeDialog
    from PyQt6.QtTest import QTest
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("hall_bars")
    p._activate("hall_bars")
    card = p.cards["hall_bars"]["step_card"]
    texts = [b.text() for b in card.findChildren(QtWidgets.QPushButton)]
    i_gen, i_draw, i_check, i_exp = (texts.index("Generate"), texts.index("Draw fingers…"), texts.index("Check"),
                                     texts.index("Export GDS…"))
    assert i_gen < i_draw < i_check < i_exp
    st = p.step
    st.shapes = {"prepattern_contacts": [geo.rect(0, 0, 2, 2), geo.rect(5, 0, 7, 2), geo.rect(40, 40, 42, 42)]}
    dlg = ShapeDialog(p, "hall_bars")
    dlg.resize(900, 700)
    dlg.show()
    c = dlg.canvas
    assert c.box_select_outlines and not c.allow_twist
    c.fit_rect(QtCore.QRectF(-10, -10, 60, 60))
    QtWidgets.QApplication.processEvents()
    a = c.mapFromScene(QtCore.QPointF(-1, 3))
    b = c.mapFromScene(QtCore.QPointF(8, -1))
    vp = c.viewport()
    QTest.mousePress(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(vp, b)
    QTest.mouseRelease(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, b)
    assert c.multi == {"shape:0", "shape:1"}
    # drag one of them: both move
    m = c.mapFromScene(QtCore.QPointF(1.0, 0.0))                          # on shape 0's lower edge
    m2 = c.mapFromScene(QtCore.QPointF(4.0, 3.0))
    QTest.mousePress(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, m)
    QTest.mouseMove(vp, m2)
    QTest.mouseRelease(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, m2)
    xs0 = [q[0] for q in st.shapes["prepattern_contacts"][0]]
    xs1 = [q[0] for q in st.shapes["prepattern_contacts"][1]]
    assert min(xs0) == pytest.approx(3.0, abs=0.3) and min(xs1) == pytest.approx(8.0, abs=0.3)
    assert st.shapes["prepattern_contacts"][2] == geo.rect(40, 40, 42, 42)
    c.set_multi(["shape:0", "shape:1"])
    dlg.delete_shape()
    assert st.shapes["prepattern_contacts"] == [geo.rect(40, 40, 42, 42)]
    dlg.close()
    w.dirty = False
    w.close()


def test_list_rows_have_tick_boxes(cfg):
    """A click on a row's tick box adds / removes that row (any set of rows); Shift still chooses a run."""
    from ustack.app import make_app
    from ustack.ui.style import QtCore, QtWidgets, setup_application
    from ustack.ui.widgets import SelectBoxDelegate, clickable_rows
    from PyQt6.QtTest import QTest
    app = make_app()
    setup_application(app)
    tree = clickable_rows(QtWidgets.QTreeWidget())
    tree.setHeaderLabels(["Name"])
    for n in "abcde":
        QtWidgets.QTreeWidgetItem(tree, [n])
    tree.resize(300, 300)
    tree.show()
    QtWidgets.QApplication.processEvents()
    assert isinstance(tree.itemDelegate(), SelectBoxDelegate)
    for r in (0, 2, 4):
        rect = tree.visualItemRect(tree.topLevelItem(r))
        QTest.mouseClick(tree.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         QtCore.QPoint(rect.left() + 8, rect.center().y()))
    assert [tree.topLevelItem(i).isSelected() for i in range(5)] == [True, False, True, False, True]
    rect = tree.visualItemRect(tree.topLevelItem(2))                    # its box again: off
    QTest.mouseClick(tree.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                     QtCore.QPoint(rect.left() + 8, rect.center().y()))
    assert [tree.topLevelItem(i).isSelected() for i in range(5)] == [True, False, False, False, True]
    tree.close()


def test_use_all_suggestions_fills_layers_and_numbers(cfg):
    """Missing values › Use all suggestions fills every empty field that has one — GDS layers too (the GDS export's
    dialog did nothing before)."""
    from ustack.app import make_app
    from ustack.config import MissingItem
    from ustack.ui.dialogs import MissingValuesDialog
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    items = [MissingItem("layer:electrodes", "GDS layer/datatype for electrodes", "", "layer", suggested=[7, 0]),
             MissingItem("preset:electrodes.pad_landing_um", "Landing square", "µm", suggested=10.0),
             MissingItem("rule:ebl.defect_clearance_um", "Clearance", "µm")]                 # no suggestion: stays empty
    dlg = MissingValuesDialog(items, None, "Values needed", "")
    dlg._use_all()
    assert dlg.answers() == {"layer:electrodes": (7, 0), "preset:electrodes.pad_landing_um": 10.0}
    dlg.close()


def test_use_as_designed_reaches_the_bubble_finder(cfg):
    from test_fingers_flow import _flow_spec
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5})                    # TEST VALUE
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    pts = np.vstack(hb.shapes["prepattern_contacts"])
    flow.step("contact_gate").found_patterns["hall_bars"] = {"dx": 1.0, "dy": 0.0, "angle": 0.0, "cx": float(pts[:, 0].mean()),
                                                             "cy": float(pts[:, 1].mean())}
    moved, _s, _r = ef.debris_inputs(spec, flow, "electrodes", bp)
    designed = hb.shapes["prepattern_contacts"][0]
    assert any(q[0] == pytest.approx(designed[0][0] + 1.0) for q in [p[0] for p in moved])
    flow.step("electrodes").found_patterns["hall_bars"] = {"as_designed": True}            # Use as designed, later
    assert ef.pattern_offset(flow, "electrodes", "hall_bars") is None
    metal, _s, _r = ef.debris_inputs(spec, flow, "electrodes", bp)
    assert designed in [list(map(tuple, p)) for p in metal] or designed in metal
    assert ef.pattern_offset(flow, "contact_gate", "hall_bars")["dx"] == 1.0              # the earlier step keeps its own


def test_export_gds_where_the_user_chooses(cfg, tmp_path, monkeypatch):
    """Export GDS… asks where to write it: the device's folder or another; the file goes there."""
    from test_fingers_flow import LAYERS, _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import QtWidgets, setup_application
    from ustack.ui import save_gds
    from ustack.gds_io import read_polygons
    app = make_app()
    setup_application(app)
    apply_answers(cfg, LAYERS)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    def _no_warn(title, text):
        raise AssertionError(text)
    monkeypatch.setattr(w, "warn", _no_warn)
    p.select_step("hall_bars")
    p._activate("hall_bars")
    from test_fingers_flow import FLAKES
    p.step.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p.twisted_from_flakes()
    p.generate()
    # the dialog itself: the device's folder by default, another folder when chosen
    dlg = save_gds.SaveGdsDialog(None, tmp_path / "dev", "x.gds")
    assert dlg.path() == tmp_path / "dev" / "x.gds"
    dlg.in_other.setChecked(True)
    dlg.folder.setText(str(tmp_path / "elsewhere"))
    dlg.name.setText("mine")
    assert dlg.path() == tmp_path / "elsewhere" / "mine.gds"
    dlg.close()
    target = tmp_path / "elsewhere" / "fingers.gds"
    monkeypatch.setattr(save_gds, "ask_gds_path", lambda *a, **k: target)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(w, "save", lambda *a, **k: None)
    p.export()
    assert target.is_file() and (5, 0) in read_polygons(target)
    w.dirty = False
    w.close()


def test_two_leads_never_over_the_same_stray_flake():
    """A wall of metal with gaps that only stray flakes fill: with one flake, one lead crosses it and the other is not
    routed (reported); with two, each lead takes its own (user, 2026-11-20: two leads over one flake would short)."""
    from ustack.router import Net, Obstacle, route_nets, wire
    a, b = Net("A", (-6.0, -20.0), (-6.0, 20.0), 1.6), Net("B", (6.0, -20.0), (6.0, 20.0), 1.6)
    left = [(-60.0, -1.0), (-4.0, -1.0), (-4.0, 1.0), (-60.0, 1.0)]
    right = [(4.0, -1.0), (60.0, -1.0), (60.0, 1.0), (4.0, 1.0)]
    flake = [(-4.0, -1.0), (4.0, -1.0), (4.0, 1.0), (-4.0, 1.0)]
    obs = [Obstacle([left], True, 0.8), Obstacle([right], True, 0.8),
           Obstacle([flake], True, 0.5, relax=True, ownable=False, exclusive=True)]
    paths, _c, notes = route_nets([a, b], obs, 0.8, box=(-30, -30, 30, 30), relax_nets={"A", "B"})
    over = [k for k, p in paths.items() if p and not (wire(p, 1.6) & geo.region(flake)).is_empty()]
    assert len(over) == 1 and len(notes) == 1                         # one over it, the other not routed (and said so)
    # two flakes in two gaps: each lead over its own
    left2 = [(-60.0, -1.0), (-10.0, -1.0), (-10.0, 1.0), (-60.0, 1.0)]
    mid = [(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)]
    right2 = [(10.0, -1.0), (60.0, -1.0), (60.0, 1.0), (10.0, 1.0)]
    f1 = [(-10.0, -1.0), (-2.0, -1.0), (-2.0, 1.0), (-10.0, 1.0)]
    f2 = [(2.0, -1.0), (10.0, -1.0), (10.0, 1.0), (2.0, 1.0)]
    obs = [Obstacle([left2], True, 0.8), Obstacle([mid], True, 0.8), Obstacle([right2], True, 0.8),
           Obstacle([f1], True, 0.5, relax=True, ownable=False, exclusive=True),
           Obstacle([f2], True, 0.5, relax=True, ownable=False, exclusive=True)]
    paths, _c, notes = route_nets([a, b], obs, 0.8, box=(-30, -30, 30, 30), relax_nets={"A", "B"})
    assert not notes
    for f in (f1, f2):
        assert sum(1 for p in paths.values() if not (wire(p, 1.6) & geo.region(f)).is_empty()) == 1


def test_contact_gate_electrodes_overlap_the_bus():
    """A contact-gate electrode runs electrodes.cg_overlap_um back along its bus (inside it), along the bus's own axis."""
    bus = geo.region([(0.0, 0.0), (20.0, 0.0), (20.0, 1.6), (0.0, 1.6)])
    end = (19.2, 0.8)
    q = fg.back_along(bus, end, 3.0, (-1.0, 0.0))
    assert q == pytest.approx((16.2, 0.8), abs=0.1)
    turned = geo.region([(0.0, 0.0), (8.0, 0.0), (14.0, 4.0), (13.0, 5.4), (7.5, 1.6), (0.0, 1.6)])   # a lead turned up 34°
    q = fg.back_along(turned, (13.4, 4.6), 4.0, (-1.0, 0.0))
    assert q is not None and math.dist(q, (13.4, 4.6)) > 3.0 and q[1] < 4.0
