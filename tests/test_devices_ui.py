"""v0.4: rename / delete saved devices, the Configure problems window, the Stage 2 image viewer and
trash button, and the Flake images tab (upload → automatic border → 3D stack, show only selected)."""

import json
import os

import cv2
import numpy as np
import pytest

from test_configure import configure_with_test_answers
from ustack import devices, paths, templates, validation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _saved_device(name="Dev A"):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), name)
    folder = paths.devices_dir() / paths.safe_name(name)
    (folder / "images").mkdir(parents=True)
    img = folder / "images" / "GR_flake.png"
    img.write_bytes(b"x")
    spec.layer("GR").image = {"path": str(img.resolve()), "um_per_px": 0.1}
    spec.save(folder / "spec.json", "test")
    return folder / "spec.json"


def test_rename_moves_the_folder_and_rewrites_paths(cfg):
    path = _saved_device()
    new = devices.rename_device(path, "Renamed device")
    assert not path.exists() and new.is_file()
    data = json.loads(new.read_text(encoding="utf-8"))
    assert data["name"] == "Renamed device"
    gr = next(l for l in data["layers"] if l["uid"] == "GR")
    assert gr["image"]["path"].startswith(str(new.parent.resolve())) and os.path.isfile(gr["image"]["path"])
    other = _saved_device("Other")
    with pytest.raises(devices.DeviceError):
        devices.rename_device(other, "Renamed device")      # the name is taken


def test_delete_removes_only_the_device_folder(cfg, tmp_path):
    path = _saved_device()
    devices.delete_device(path)
    assert not path.parent.exists() and paths.devices_dir().is_dir()
    stray = tmp_path / "elsewhere" / "spec.json"
    stray.parent.mkdir()
    stray.write_text("{}")
    with pytest.raises(devices.DeviceError):
        devices.delete_device(stray)


def test_issue_parameters_point_at_the_values_behind_an_error(cfg):
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "bad")
    spec.layer("GR").width_um.set(3.0, "user")              # too narrow for the Hall bar
    res = configure_with_test_answers(spec, cfg)
    fit = next(i for i in res.report.issues if i.code == "hall_bar_fit")
    keys = {p.key for p in validation.issue_parameters(spec, cfg, fit)}
    assert {"spec:GR.width_um", "spec:GR.length_um", "spec:hall_bar.channel_width_um"} <= keys


def _flake_png(path, w=900, h=600):
    a = np.full((h, w, 3), (150, 150, 140), np.uint8)
    cv2.fillPoly(a, [np.array([[300, 200], [620, 230], [600, 420], [320, 400]])], (190, 110, 70))
    cv2.imwrite(str(path), a)


def test_new_windows(cfg, tmp_path):
    from ustack.app import make_app
    from ustack.ui.configure_window import ConfigureProblemsDialog
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.show()
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "UI dev")
    w.adopt_spec(spec, "new")
    # Configure problems: errors with their values, editable in place; Configure again reruns
    spec.layer("GR").width_um.set(3.0, "user")
    res = configure_with_test_answers(spec, cfg)
    assert any(i.code == "hall_bar_fit" for i in res.report.issues)
    dlg = ConfigureProblemsDialog(w, res)
    assert dlg.tabs.count() == 3 and not res.ok
    spec.layer("GR").width_um.set(40.0, "user")
    dlg.configure_again()
    assert not any(i.code == "hall_bar_fit" for i in dlg.result.report.issues)
    dlg.close()
    # Flake images (side tab between Stack and Pattern): upload, then Trace all (whole flake, no click)
    png = tmp_path / "flake.png"
    _flake_png(png)
    page = w.flake_panel
    names = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert names.index("Flake images") == names.index("Stack") + 1 == names.index("Pattern") - 1
    page.objective.setCurrentIndex(page.objective.findData("100x"))
    assert page.upload("GR", str(png))
    gr = spec.layer("GR")
    assert gr.outline is None                                # upload does not trace any more
    page.trace_all()
    assert gr.outline and len(gr.outline) >= 4 and gr.image["trace"] == {**gr.image["trace"], "how": "auto", "mode": "flake"}
    cx = sum(p[0] for p in gr.outline) / len(gr.outline)
    assert abs(cx) < 0.5                                     # centred on the design origin
    slab = next(s for s in w.view3d.slabs if s.uid == "GR")
    assert slab.polys[0] == [tuple(p) for p in gr.outline]
    gr.image["trace"]["how"] = "edited"                      # a border you corrected is kept by Trace all
    kept = [list(p) for p in gr.outline]
    page.trace_all()
    assert gr.outline == kept
    # Trace flake… opens the tracing window on the image
    page.trace("GR")
    dlg = page.trace_dialog
    assert dlg.isVisible() and dlg.uid == "GR" and dlg.canvas.active_image == "GR"
    dlg.close()
    page.list.clearSelection()
    page.current = "GR"
    page.show_selected()
    assert w.view3d.hidden == {s.uid for s in w.view3d.slabs} - {"GR"}
    page.show_all()
    assert not w.view3d.hidden
    # Stage 2: View opens a window with the image (also after it is only on disk); the trash forgets it
    p = w.stage2_panel
    p.inputs.images["5x"] = str(png)
    p.images.clear()
    p.view_image("5x")
    assert p._viewers and p._viewers[-1].isVisible()
    p._viewers[-1].close()
    p.remove_image("5x")
    assert "5x" not in p.inputs.images and png.exists()     # a file outside the device folder is not deleted
    w.dirty = False
    w.close()


def test_trace_window_point_editing(cfg, tmp_path):
    """Trace window: add a point on the border, box-select several points, move them together, delete
    them (trash button or Delete) — the neighbours join up; a border keeps at least 3 points."""
    from PyQt6.QtTest import QTest
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import QtCore, setup_application
    Qt = QtCore.Qt
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.show()
    spec = templates.new_device_from(templates.load_template("xia_graphene"), "Points dev")
    w.adopt_spec(spec, "new")
    png = tmp_path / "flake.png"
    _flake_png(png)
    page = w.flake_panel
    page.objective.setCurrentIndex(page.objective.findData("100x"))
    assert page.upload("GR", str(png))
    gr = spec.layer("GR")
    gr.outline = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    page.trace("GR")
    dlg = page.trace_dialog
    c = dlg.canvas
    c.fitInView(QtCore.QRectF(-5, -5, 20, 20))
    c._zoomed = True
    vp = lambda x, y: c.mapFromScene(QtCore.QPointF(x, y))          # noqa: E731

    def click(x, y, mods=Qt.KeyboardModifier.NoModifier):
        QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, mods, vp(x, y))
        QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, mods, vp(x, y))

    # add a point: a click on the border line inserts it there; off the line nothing happens
    dlg.tools["addpoint"].setChecked(True)
    click(5.0, 3.0)
    assert len(gr.outline) == 4 and "border" in dlg.hint.text()
    click(5.0, 0.02)
    assert len(gr.outline) == 5
    assert gr.outline[1] == pytest.approx([5.0, 0.0], abs=0.1)
    # box-select two points, then drag one of them: both move
    dlg.tools["select"].setChecked(True)
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(8, 12))
    QTest.mouseMove(c.viewport(), vp(12, -2))
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(12, -2))
    poly = c.polys[c.active_poly]
    assert sorted(poly.selected_indices()) == [2, 3] and dlg.trash.isEnabled()
    QTest.mousePress(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(10, 10))
    QTest.mouseMove(c.viewport(), vp(11, 11))
    QTest.mouseRelease(c.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(11, 11))
    assert gr.outline[2] == pytest.approx([11.0, 1.0], abs=0.15) and gr.outline[3] == pytest.approx([11.0, 11.0], abs=0.15)
    poly = c.polys[c.active_poly]                                   # the window redrew: the selection is kept
    assert sorted(poly.selected_indices()) == [2, 3]
    # Shift-click adds a point to the selection; the trash deletes all selected, the rest join up
    dlg.tools["pointer"].setChecked(True)
    click(0.0, 10.0, Qt.KeyboardModifier.ShiftModifier)
    assert sorted(c.polys[c.active_poly].selected_indices()) == [2, 3, 4]
    dlg.trash.click()
    assert len(gr.outline) == 5, "a border keeps at least 3 points"
    assert "at least 3" in dlg.hint.text()
    # one point selected, Delete key: removed
    app.processEvents()                                             # (each edit is its own undo step)
    click(5.0, 0.0)
    assert c.polys[c.active_poly].selected_indices() == [1]
    QTest.keyClick(c.viewport(), Qt.Key.Key_Delete)
    assert len(gr.outline) == 4 and [5.0, 0.0] not in [list(map(round, p)) for p in gr.outline]
    # undo brings the point back
    app.processEvents()
    w.undo()
    assert len(w.spec.layer("GR").outline) == 5
    dlg.close()
    w.dirty = False
    w.close()


def test_stack_layer_list_rows_are_clickable(cfg):
    from ustack.app import make_app
    from ustack.ui.main_window import MainWindow
    from ustack.ui.style import QtCore, setup_application
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.adopt_spec(templates.new_device_from(templates.load_template("xu_tmd"), "Rows"), "new")
    lst = w.stack_panel.list
    assert lst.viewport().cursor().shape() == QtCore.Qt.CursorShape.PointingHandCursor
    assert lst.hasMouseTracking() and "QListView::item:hover" in lst.styleSheet() and "QListView::item:selected" in lst.styleSheet()
    w.dirty = False
    w.close()
