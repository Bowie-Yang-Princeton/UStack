"""Bubbles, cracks and stray flakes (ustack.debris), the router keeping the electrodes off them, and the Manual alignment
and Draw / edit windows. Synthetic images (+ the lab's batch-1 images when ``WSe2 device training/`` is there)."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ustack import bondpads, debris, ebl_flow as ef, fingers as fg, geometry as geo
from ustack.alignment import Similarity, fit_similarity
from ustack.config import apply_answers

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "WSe2 device training"


def _synthetic():
    """A 0.1 µm/px image (chip origin at its centre): substrate with a gentle illumination fall-off, a stray flake, a
    'stack' flake with a bubble and a crack on it, a metal pad (known)."""
    h, w = 800, 1200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - 0.08 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img = np.zeros((h, w, 3), np.float32)
    img[...] = np.array([150, 120, 140], np.float32)                  # substrate (BGR)
    img *= shade[..., None]
    img = img.astype(np.uint8)
    cv2.fillPoly(img, [np.array([[100, 100], [260, 90], [280, 230], [120, 240]], np.int32)], (120, 200, 90))      # stray flake
    stack_px = np.array([[500, 250], [900, 250], [900, 600], [500, 600]], np.int32)
    cv2.fillPoly(img, [stack_px], (190, 150, 110))                     # the stack (known)
    cv2.circle(img, (620, 380), 14, (120, 100, 70), -1)                # a bubble on it
    cv2.line(img, (560, 520), (860, 470), (230, 200, 160), 4)          # a crack / fold on it
    cv2.rectangle(img, (1000, 650), (1150, 760), (80, 230, 230), -1)   # metal (known)
    sim = Similarity(0.1, 0.0, -60.0, 40.0)                            # px → chip: 120 × 80 µm centred on 0
    stack = [[tuple(map(float, q)) for q in sim.apply(stack_px)]]
    metal = [[tuple(map(float, q)) for q in sim.apply([(1000, 650), (1150, 650), (1150, 760), (1000, 760)])]]
    return img, sim, stack, metal


def test_finds_stray_flakes_bubbles_and_cracks_on_a_synthetic_image():
    img, sim, stack, metal = _synthetic()
    cands = debris.find_in_image(img, sim, metal, stack)
    found = debris.classify(cands, debris.Classifier([]))               # the rules (no examples)
    by = {}
    for c in found:
        by.setdefault(c.kind, []).append(c)
    flake = sim.apply([(190, 165)])[0]
    bubble = sim.apply([(620, 380)])[0]
    crack = sim.apply([(710, 495)])[0]
    assert any(geo.point_in_poly(*flake, c.polygon) for c in by.get("flake", []))
    assert any(geo.point_in_poly(*bubble, c.polygon) for c in by.get("bubble", []))
    assert any(geo.distance_to_region(*crack, geo.region(c.polygon)) < 1.0 for c in by.get("crack", []))
    # the known metal and the stack itself are not features; the substrate's fall-off is not either
    pad = sim.apply([(1075, 705)])[0]
    assert not any(geo.point_in_poly(*pad, c.polygon) for c in found)
    assert not any(abs(geo.poly_area(c.polygon)) > 1000 for c in found)
    assert len(found) <= 6


def test_built_in_training_set_and_lab_corrections(tmp_path, monkeypatch):
    """The labelled set built from the lab's images is shipped; the lab's corrections are added to it."""
    monkeypatch.setenv("USTACK_HOME", str(tmp_path / "library"))
    data = json.loads(debris.builtin_examples_path().read_text(encoding="utf-8"))["examples"]
    kinds = {e["kind"] for e in data}
    assert len(data) >= 150 and {"flake", "particle", "bubble", "crack", "none"} <= kinds
    clf = debris.Classifier(data)
    ok = sum(debris.Classifier(data[:i] + data[i + 1:]).classify(e["features"])[0] == e["kind"] for i, e in enumerate(data))
    assert ok / len(data) >= 0.75                                      # leave-one-out (docs/debris_training.md)
    n0 = len(debris.load_examples())
    debris.save_lab_examples([{"kind": "none", "features": data[0]["features"]}])
    assert len(debris.load_examples()) == n0 + 1
    assert clf.classify(data[0]["features"])[0] in kinds


def test_router_keeps_the_electrodes_off_stray_flakes(cfg):
    """A stray flake where an electrode ran: the next Generate goes round it (Check: off them); without the clearance
    set it is asked."""
    from test_fingers_flow import _flow_spec
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5, "preset:electrodes.pad_landing_um": 10.0,
                        "preset:electrodes.flake_clearance_um": 1.0})   # TEST VALUES
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    ef.generate(spec, flow, "hall_bars", cfg, bp)
    ef.generate(spec, flow, "contact_gate", cfg, bp)
    st = ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    r0 = next(r for r in st.routes if r["group"] == "fingers")
    pts = r0["points_um"]
    (ax, ay), (bx, by) = pts[-3], pts[-2]                                # on the routed part, half way
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    st.debris = [{"kind": "flake", "polygon": [list(q) for q in geo.rect(cx - 4, cy - 4, cx + 4, cy + 4)], "how": "drawn"}]
    st.debris_looked = True
    from ustack.config import MissingValues
    with pytest.raises(MissingValues) as exc:
        ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    assert exc.value.items[0].key == "rule:ebl.defect_clearance_um"
    apply_answers(cfg, {"rule:ebl.defect_clearance_um": 1.0})          # TEST VALUE
    st = ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    assert all(r["routed_to_pad"] for r in st.routes)
    flake = geo.region(st.debris[0]["polygon"])
    metal = geo.regions(st.shapes["el_fingers"]).merged()
    assert (metal & flake).is_empty() and geo.region_distance(metal, flake) >= 1.0 - 0.05
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "electrodes", cfg)}
    assert rows["Electrodes off bubbles, cracks and stray flakes"]["status"] == "met"
    # a pad whose landing square lies on a stray flake is avoided when another will do
    ends = bp.pad_ends()
    used = {r["pad_tip"] for r in st.routes}
    e = ends[next(iter(used))]
    sq = ef.landing_square(e, 10.0)
    st.debris.append({"kind": "flake", "polygon": [list(q) for q in geo.to_polys(geo.sized(geo.region(sq), 2.0))[0]], "how": "drawn"})
    st = ef.generate(spec, flow, "electrodes", cfg, bp, group="fingers")
    assert all(r["routed_to_pad"] for r in st.routes)
    assert next(i for i, e2 in enumerate(ends) if e2 is e) not in {r["pad_tip"] for r in st.routes}


def test_find_and_edit_in_the_electrodes_card(cfg, tmp_path, monkeypatch):
    """Electrodes card › Find bubbles, cracks, flakes on an aligned image; Draw / edit…: delete one (a 'none' example for
    the finder), draw one (kept by Find again)."""
    from test_fingers_flow import _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    from ustack.ui.style import QtWidgets as _W
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("electrodes")
    p._activate("electrodes")
    texts = [b.text() for b in p.cards["electrodes"]["step_card"].findChildren(_W.QPushButton)]
    assert "Find" in texts and "Manual alignment" in texts
    assert "Pick pairs" not in texts and "Fit pairs" not in texts
    assert not any(t.startswith("Use the images") for t in texts)
    img, sim, _stack, _metal = _synthetic()
    f = tmp_path / "e50x.png"
    cv2.imwrite(str(f), img)
    p.step.images = {"50x": str(f)}
    p.step.transforms = {"50x": sim.to_json()}
    p.step.outlines = {}
    warned = []
    monkeypatch.setattr(w, "warn", lambda title, text: warned.append(text))
    p.find_debris()
    assert warned and "Not located at this step" in warned[0]               # a layer not located can look like a flake
    st = p.step
    assert st.debris and all(d["how"] == "auto" for d in st.debris)
    p.edit_debris()
    dlg = p._debris_dlg
    n = len(st.debris)
    dlg.canvas.set_active_polygon("d:0")
    dlg.delete_selected()
    assert len(st.debris) == n - 1
    lab = json.loads(debris.lab_examples_path().read_text(encoding="utf-8"))["examples"]
    assert lab and lab[-1]["kind"] == "none"                           # learned: not a feature
    dlg.kind_combo.setCurrentIndex(dlg.kind_combo.findData("bubble"))
    dlg._drawn([(-5.0, -5.0), (-3.0, -5.0), (-3.0, -3.0), (-5.0, -3.0)])
    assert st.debris[-1]["how"] == "drawn" and st.debris[-1]["kind"] == "bubble"
    dlg.find_again()
    assert any(d["how"] == "drawn" for d in st.debris)                # kept by Find again
    dlg.close()
    w.dirty = False
    w.close()


def test_manual_alignment_window(cfg, tmp_path):
    """Manual alignment: turning / moving the image frame sets the step's alignment (a similarity); Auto scale keeps the
    field of view's scale; pairs picked in the window are fitted."""
    from test_fingers_flow import _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("hall_bars")
    p._activate("hall_bars")
    img = np.full((600, 900, 3), 128, np.uint8)
    f = tmp_path / "h20x.png"
    cv2.imwrite(str(f), img)
    p.step.images = {"20x": str(f)}
    p.manual_alignment()
    d = p._align_dlg
    assert d.obj == "20x" and d.auto_scale.isChecked()                # not aligned yet: the field of view's scale
    ns = d.nominal_scale()
    sim0 = d.transform()
    pts = d._frame_points(sim0)
    cx, cy = sum(q[0] for q in pts) / 4, sum(q[1] for q in pts) / 4
    a = math.radians(10.0)
    turned = [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a) + 20.0, cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a))
              for x, y in pts]
    big = [(cx + (x - cx) * 1.5, cy + (y - cy) * 1.5) for x, y in turned]   # also scaled by hand …
    d._frame_edited("frame", big)
    t = Similarity.from_json(p.flow.step("hall_bars").transforms["20x"])
    assert t.angle_deg == pytest.approx(sim0.angle_deg + 10.0, abs=1e-6)
    assert t.scale == pytest.approx(ns, rel=1e-9)                       # … but Auto scale keeps the field of view's
    d.auto_scale.setChecked(False)
    d._frame_edited("frame", big)
    assert Similarity.from_json(p.flow.step("hall_bars").transforms["20x"]).scale == pytest.approx(ns * 1.5, rel=1e-6)
    # pairs: image px → chip µm of a known transform
    true = Similarity(0.12, 30.0, 15.0, -8.0)
    d.start_pairs()
    item = d.canvas.images[f"align:{d.obj}"]
    for px in [(100.0, 100.0), (800.0, 120.0), (450.0, 500.0)]:
        x, y = d.transform().apply([px])[0]                             # where the pixel is shown now
        d._picked(float(x), float(y))
        assert d.pairs[-1]["px"] == pytest.approx(px, abs=1e-6)
        X, Y = true.apply([px])[0]
        d._picked(float(X), float(Y))
    d.fit_pairs()
    got = Similarity.from_json(p.flow.step("hall_bars").transforms["20x"])
    assert got.angle_deg == pytest.approx(30.0, abs=1e-6) and got.scale == pytest.approx(0.12, rel=1e-6)
    d.close()
    w.dirty = False
    w.close()


@pytest.mark.skipif(not (LAB / "tWSe2-1_KLayout_session" / "tWSe2-1.lys").is_file(), reason="the lab's batch-1 data is not here")
def test_lab_images_before_the_electrodes():
    """On the lab's Stack 1 image at 20x (placed by its KLayout session): stray flakes and particles round the device,
    none on the bond pads, every one inside the pads' region."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import debris_training as T
    case, img, sim, cands = T.candidates()[0]
    found = debris.classify(cands)
    kinds = {c.kind for c in found}
    assert {"flake", "particle"} <= kinds and len(found) > 50
    for c in found:
        x, y = geo.centroid(c.polygon)
        assert abs(x) <= 200 and abs(y) <= 200


def test_step_edges_and_layers_are_not_features():
    """Two stack layers whose located outlines are ~0.6 µm off the real edges: the edge between them is not a crack, the
    sliver of a layer outside its outline not a stray flake; a real fold on the upper layer still is."""
    h, w = 800, 1200
    img = np.zeros((h, w, 3), np.uint8)
    img[...] = (150, 120, 140)
    lower = np.array([[300, 200], [1000, 200], [1000, 650], [300, 650]], np.int32)
    upper = np.array([[500, 300], [900, 300], [900, 560], [500, 560]], np.int32)
    cv2.fillPoly(img, [lower], (190, 150, 110))
    cv2.fillPoly(img, [upper], (120, 190, 160))
    cv2.line(img, (560, 330), (860, 520), (175, 225, 200), 4)               # a fold on the upper layer
    sim = Similarity(0.1, 0.0, -60.0, 40.0)
    off = np.array([6, -5])                                                 # the located outlines: 0.6 / 0.5 µm off
    stack = [[tuple(map(float, q)) for q in sim.apply(p + off)] for p in (lower, upper)]
    found = debris.classify(debris.find_in_image(img, sim, [], stack))
    fold = sim.apply([(710, 425)])[0]
    assert any(c.kind == "crack" and geo.distance_to_region(*fold, geo.region(c.polygon)) < 1.0 for c in found)
    edge_pts = [sim.apply([q])[0] for q in [(700, 300), (900, 430), (700, 560), (500, 430), (650, 200), (1000, 420)]]
    for c in found:
        r = geo.region(c.polygon)
        assert all(geo.distance_to_region(x, y, r) > 1.0 for x, y in edge_pts), (c.kind, geo.centroid(c.polygon))
    assert debris.layer_veto({"on_stack": 1.0, "elong": 6.0, "sides": 1.5}) is not None      # one colour each side: an edge
    assert debris.layer_veto({"on_stack": 1.0, "elong": 6.0, "sides": 0.1}) is None           # the same flake both sides
    assert debris.layer_veto({"on_stack": 0.0, "hug": 0.9, "area_um2": 3.0}) is not None       # along a layer's outline
    assert debris.layer_veto({"on_stack": 0.0, "hug": 0.1, "area_um2": 400.0}) is not None     # a layer not located


def test_later_steps_replace_earlier_bubbles_and_cracks():
    from test_fingers_flow import _flow_spec
    spec, flow = _flow_spec()
    a = {"kind": "bubble", "polygon": [[0, 0], [1, 0], [1, 1]], "how": "drawn"}
    b = {"kind": "crack", "polygon": [[5, 5], [9, 5], [9, 5.5]], "how": "drawn"}
    hb, cg = flow.step("hall_bars"), flow.step("contact_gate")
    hb.debris, hb.debris_looked = [a], True
    assert ef.debris_source(flow, "electrodes") == "hall_bars" and ef.debris_at(flow, "contact_gate") == [a]
    cg.debris, cg.debris_looked = [b], True                               # found again later: replaces the earlier ones
    assert ef.debris_at(flow, "electrodes") == [b] and ef.debris_at(flow, "hall_bars") == [a]
    cg.debris, cg.debris_looked = [], True                                # looked, none: none hold any more
    assert ef.debris_at(flow, "electrodes") == []
    again = ef.FlowStep.from_json({"key": "hall_bars", "debris": [a]})     # saved before the flag: its own
    assert again.debris_looked


def test_hall_bars_and_contact_gate_keep_off_bubbles_and_cracks(cfg):
    from test_fingers_flow import _flow_spec
    apply_answers(cfg, {"preset:cg.bus_tmd_gap_um": 0.5, "preset:fingers.bbn_margin_um": 0.5,
                        "preset:cg.hbn_margin_um": 0.5, "rule:ebl.defect_clearance_um": 0.5})   # TEST VALUES
    spec, flow = _flow_spec()
    bp = bondpads.default_preset()
    bbn = spec.by_role("bottom_hbn")
    for k in ("hall_bars", "contact_gate"):
        flow.step(k).outlines[bbn.uid] = [geo.rect(-60, -60, 60, 60)]
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    arr = fg.FingerArray.from_json(hb.array)
    # a bubble on one top finger's far part: that finger ends before it (still long enough for the electrodes)
    f = next(f for f in arr.fingers() if f.column == "top")
    L = arr.params.length_at("top", f.index)
    x, y = arr.point_at(f, L - 2.0)
    st = flow.step("hall_bars")
    st.debris = [{"kind": "bubble", "polygon": [list(q) for q in geo.rect(x - 0.5, y - 0.5, x + 0.5, y + 0.5)], "how": "drawn"}]
    st.debris_looked = True
    hb = ef.generate(spec, flow, "hall_bars", cfg, bp)
    arr2 = fg.FingerArray.from_json(hb.array)
    key = f"top:{f.index}"
    assert arr2.params.lengths[key] < arr.params.lengths[key] - 1.0          # that finger ends before it …
    assert any(arr2.params.lengths[k] == pytest.approx(v) for k, v in arr.params.lengths.items() if k != key)   # … not all
    assert any("before a bubble" in n for n in hb.notes)
    metal = geo.regions(hb.shapes["prepattern_contacts"]).merged()
    assert (metal & geo.region(st.debris[0]["polygon"])).is_empty()
    rows = {r["rule"]: r for r in ef.rule_report(spec, flow, "hall_bars", cfg)}
    assert rows["Fingers off bubbles and cracks"]["status"] == "met"
    # the contact gate: a crack where the bottom bus would run; the comb keeps off it (from the Hall-bar step's list)
    cg = ef.generate(spec, flow, "contact_gate", cfg, bp)
    comb0 = fg.contact_gate_comb(arr2, skip=cg.comb["skip"], dirs=cg.comb["dirs"], extension=cg.comb["extension"],
                                 n_skip=cg.comb.get("n_skip", 1), tooth=cg.comb.get("tooth"))
    bus = np.asarray(arr2.to_local(comb0["bottom"]["bus"]))
    u0, v_out = float(bus[:, 0].mean()), float(bus[:, 1].min())             # the bottom bus's outer edge
    crack_poly = arr2.to_chip([(u0 - 1.5, v_out - 0.9), (u0 + 1.5, v_out - 0.9), (u0 + 1.5, v_out - 0.5), (u0 - 1.5, v_out - 0.5)])
    assert not (geo.regions(cg.shapes["contact_gates"]).merged() - geo.sized(geo.region(crack_poly), 0.5)).is_empty()
    flow.step("contact_gate").debris = [{"kind": "crack", "polygon": [list(q) for q in crack_poly], "how": "drawn"}]
    flow.step("contact_gate").debris_looked = True
    cg = ef.generate(spec, flow, "contact_gate", cfg, bp)
    comb = geo.regions(cg.shapes["contact_gates"]).merged()
    crack = geo.region(flow.step("contact_gate").debris[0]["polygon"])
    assert (comb & crack).is_empty()
    assert any("keep off the bubbles and cracks" in n for n in cg.notes)


def test_error_circles_pop_up_accept_and_change_rule(cfg, monkeypatch):
    """A red circle: its pop-up names the error and the rule; Accept clears it (and keeps it accepted on Check again);
    Change rule… edits the value behind it."""
    from test_fingers_flow import _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    from ustack.ui import ebl_flow_panel as efp
    from ustack.ui.canvas import IssueMarker
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    p = w.ebl_flow
    p.select_step("hall_bars")
    p._activate("hall_bars")
    st = p.step
    st.shapes = {"prepattern_contacts": [geo.rect(0, 0, 0.3, 5)]}            # narrower than the hard minimum
    ef.check(w.spec, p.flow, "hall_bars", cfg)
    errs = [i for i in st.issues if i["severity"] == "error"]
    assert errs and errs[0]["rule"] == "rule:ebl.contact_width_min_um"
    p.redraw()
    markers = [i for i in w.ebl_canvas.scene().items() if isinstance(i, IssueMarker)]
    assert markers
    p.issue_popup(markers[0].key, markers[0].pos().x(), markers[0].pos().y())
    pop = p._issue_popup
    assert "ebl.contact_width_min_um" in pop._rule_text()
    pop._accept()
    assert not [i for i in st.issues if i["severity"] == "error"] and st.accepted
    assert not [i for i in w.ebl_canvas.scene().items() if isinstance(i, IssueMarker)]
    ef.check(w.spec, p.flow, "hall_bars", cfg)                              # still accepted on Check again
    assert not [i for i in st.issues if i["severity"] == "error"]
    # Change rule…: the value behind it, saved to the (test) config
    class FakeDialog:
        def __init__(self, items, *a, **k):
            self.items = items

        def exec(self):
            return True

        def answers(self):
            return {self.items[0].key: 0.25}                              # TEST VALUE
    monkeypatch.setattr(efp, "MissingValuesDialog", FakeDialog)
    p.change_rule("hall_bars", "rule:ebl.contact_width_min_um")
    assert cfg.value("ebl.contact_width_min_um") == 0.25
    w.dirty = False
    w.close()


def test_long_thin_pieces_on_the_stack_are_cracks():
    """A long, thin piece on the stack with the same flake on both sides is a crack / fold even where the examples say
    otherwise (user, 2026-11-20: obvious cracks on the bottom hBN were missed)."""
    assert debris.is_crack_shape({"on_stack": 1.0, "elong": 5.4, "sides": 0.02, "thin": 1.4})
    assert not debris.is_crack_shape({"on_stack": 1.0, "elong": 5.4, "sides": 1.2, "thin": 1.4})      # a step edge
    assert not debris.is_crack_shape({"on_stack": 0.0, "elong": 5.4, "sides": 0.02, "thin": 1.4})     # off the stack
    assert not debris.is_crack_shape({"on_stack": 1.0, "elong": 1.3, "sides": 0.02, "thin": 1.4})     # round: a bubble
    c = debris.Candidate([(0, 0), (10, 0), (10, 1), (0, 1)], {"on_stack": 1.0, "elong": 5.4, "sides": 0.02, "thin": 1.4,
                                                            "log_area": 1.9, "solidity": 0.9, "de_mean": 9.0, "de_max": 14.0,
                                                            "dl": 4.0, "texture": 0.3, "sharp": 0.6, "hug": 0.1, "area_um2": 80.0})
    class AlwaysNone:
        def classify(self, feat):
            return "none", 0.9
    assert [x.kind for x in debris.classify([c], AlwaysNone())] == ["crack"]


def test_several_rows_can_be_chosen(cfg):
    """Every list takes Shift / Ctrl-click to choose several rows; Delete removes every chosen layer, Clear every chosen
    flake outline."""
    from test_fingers_flow import _flow_spec
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    from ustack.ui.style import QtWidgets as _W
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    spec, _ = _flow_spec()
    w.adopt_spec(spec, "new")
    ext = _W.QAbstractItemView.SelectionMode.ExtendedSelection
    assert w.library.dev.selectionMode() == ext and w.library.tpl.selectionMode() == ext
    lst = w.stack_panel.list
    assert lst.selectionMode() == ext
    n = len(w.spec.layers)
    lst.clearSelection()
    picked = []
    for r in range(lst.count()):
        uid = lst.item(r).data(0x0100)
        if w.spec.layer(uid).function in ("encapsulation", "top_gate"):
            lst.item(r).setSelected(True)
            picked.append(uid)
    assert len(picked) == 2
    w.stack_panel._delete()
    assert len(w.spec.layers) == n - 2 and not any(w.spec.layer(u) for u in picked)
    p = w.ebl_flow
    p.select_step("hall_bars")
    p._activate("hall_bars")
    st = p.step
    from test_fingers_flow import FLAKES
    st.outlines = {u: [list(q) for q in polys] for u, polys in FLAKES.items()}
    p._fill_step()
    fl = p.flake_list
    assert fl.selectionMode() == ext
    keys = [fl.topLevelItem(i).data(0, 0x0100) for i in range(fl.topLevelItemCount())]
    chosen = [k for k in keys if st.outlines.get(k)][:2]
    for i in range(fl.topLevelItemCount()):
        fl.topLevelItem(i).setSelected(fl.topLevelItem(i).data(0, 0x0100) in chosen)
    p.clear_outline()
    assert len(chosen) == 2 and not any(st.outlines.get(k) for k in chosen)
    w.dirty = False
    w.close()
