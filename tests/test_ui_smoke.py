"""Offscreen smoke test of the window: every template loads, validates and draws."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")


def test_main_window_templates(cfg):
    from ustack.app import make_app
    from ustack.ui.style import setup_application
    from ustack.ui.main_window import MainWindow
    from ustack.ui.dialogs import MissingValuesDialog, RulesDialog
    from ustack import templates, validation
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.show()
    for info in templates.list_templates():
        w.adopt_spec(templates.new_device_from(templates.load_template(info), info.id), "new")
        app.processEvents()
        assert w.view3d.slabs, info.id
        assert w.check_panel.report is not None
        items = validation.validate_stage1(w.spec, cfg).missing_items()
        dlg = MissingValuesDialog(items, w)
        assert len(dlg.fields) == len([i for i in items if i.kind != "action"])
        dlg.close()
    RulesDialog(cfg, w).close()
    # the 100x LUT shares the BF alignment, scaled to its own resolution (here half)
    import numpy as np
    from ustack.alignment import Similarity
    p = w.stage2_panel
    p.images = {"100x": np.zeros((3984, 6000, 3), np.uint8), "100x_lut": np.zeros((1992, 3000, 3), np.uint8)}
    p.transform = Similarity(0.0235, 0.2, -8.0, 2.0)
    assert p._sim("100x_lut").scale == pytest.approx(0.047)
    assert p._sim("100x_lut").apply([[1500, 996]])[0] == pytest.approx(p._sim("100x").apply([[3000, 1992]])[0])
    p.images, p.transform = {}, None
    w.dirty = False
    w.close()


def test_new_device_presets_rules_and_scale_bar(cfg, tmp_path, monkeypatch):
    from ustack.app import make_app
    from ustack.ui.style import QtCore, QtWidgets, setup_application
    from ustack.ui.main_window import MainWindow
    from ustack.ui.dialogs import PresetsDialog, RulesDialog
    from ustack.ui.new_device import NewDeviceDialog
    from ustack import templates
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.show()
    # New device › Build from scratch: substrate, bottom gate, two graphene pieces, top gate
    dlg = NewDeviceDialog(cfg, w)
    dlg.name.setText("scratch")
    dlg.tabs.setCurrentIndex(1)
    dlg._add_device_row("graphene")
    dlg.tg.setCurrentIndex(dlg.tg.findData("Ti/Au"))
    dlg._create()
    spec = dlg.spec
    assert [l.function for l in spec.layers] == ["substrate", "bottom_gate", "bottom_gate_dielectric", "device_layer", "device_layer",
                                                 "top_gate_dielectric", "top_gate", "hall_bar_contacts"]
    assert all(l.width_um.value is None for l in spec.layers)           # no dimension asked for
    # … or from a template
    dlg = NewDeviceDialog(cfg, w)
    dlg._create()
    assert dlg.spec is not None and dlg.spec.template_id
    # … or from a template edited first: Edit Template is the next step in the same window (user, 2026-11-21); the
    # template file itself is not changed
    dlg = NewDeviceDialog(cfg, w)
    assert dlg.edit_btn.isVisibleTo(dlg) and not dlg.back_btn.isVisibleTo(dlg)
    tpl_path = dlg.tpl.currentItem().data(0, QtCore.Qt.ItemDataRole.UserRole)
    n_tpl = len(templates.load_template(tpl_path).layers)
    dlg.edit_template()
    assert dlg.steps.currentWidget() is dlg.edit_page and dlg.back_btn.isVisibleTo(dlg) and not dlg.edit_btn.isVisibleTo(dlg)
    dlg.editor.stack.list.setCurrentRow(0)
    dlg.editor.stack._delete()
    dlg.back()
    dlg.edit_template()                                                 # the edit is kept while the same template is chosen
    assert len(dlg.editor.spec.layers) == n_tpl - 1
    dlg._create()
    assert len(dlg.spec.layers) == n_tpl - 1 and dlg.spec.name == dlg.name.text()
    assert len(templates.load_template(tpl_path).layers) == n_tpl
    w.adopt_spec(spec, "new")
    app.processEvents()
    # the first Save asks the device's name (user, 2026-11-21); files copied in before move to that folder
    from ustack import paths
    from ustack.ui import main_window as mw
    early = w.device_dir() / "images" / "early.png"
    early.parent.mkdir(parents=True, exist_ok=True)
    early.write_bytes(b"png")
    spec.layers[3].image = {"path": str(early), "um_per_px": 0.1, "angle_deg": 0.0, "x": 0.0, "y": 0.0}
    monkeypatch.setattr(mw, "prompt_text", lambda *a, **k: None)
    assert not w.save() and w.spec_path is None                         # Cancel: not saved
    monkeypatch.setattr(mw, "prompt_text", lambda *a, **k: "scratch renamed")
    assert w.save()
    # Save device puts it in the devices list, selected
    assert w.library.dev.currentItem() is not None and "scratch renamed" in w.library.dev.currentItem().text(0)
    moved = paths.devices_dir() / paths.safe_name("scratch renamed") / "images" / "early.png"
    assert moved.is_file() and spec.layers[3].image["path"] == str(moved.resolve()) and not early.exists()
    monkeypatch.setattr(mw, "prompt_text", lambda *a, **k: pytest.fail("asked again"))
    assert w.save()                                                     # later saves do not ask
    # the 3D scale bar re-chooses its length with the zoom
    w.view3d.resize(900, 600)
    w.view3d.grab()
    first = w.view3d.scalebar_um
    w.view3d.zoom = 6.0
    w.view3d.grab()
    assert first and w.view3d.scalebar_um < first
    assert w.view3d.nice_length(37) == 20 and w.view3d.nice_length(0.7) == 0.5
    # Rules: read-only until Edit; presets editable there and in their own dialog
    r = RulesDialog(cfg, w)
    flags = r.rules_table.item(0, 1).flags()
    assert not flags & QtCore.Qt.ItemFlag.ItemIsEditable
    r._toggle_edit()
    assert r.rules_table.item(0, 1).flags() & QtCore.Qt.ItemFlag.ItemIsEditable
    r.close()
    pd = PresetsDialog(cfg, w)
    row = pd.table.keys.index("hall_bar.width_per_arm")
    assert pd.table.item(row, 2).flags() & QtCore.Qt.ItemFlag.ItemIsEditable          # editable at once (finger presets too)
    frow = pd.table.keys.index("fingers.width_um")
    assert pd.table.item(frow, 2).flags() & QtCore.Qt.ItemFlag.ItemIsEditable
    pd.table.item(row, 2).setText("4")
    assert pd._save()
    assert cfg.preset("hall_bar.width_per_arm") == 4.0 and cfg.presets["hall_bar.width_per_arm"]["status"] == "approved"
    pd.close()
    # pointing hand over buttons
    b = w.findChildren(QtWidgets.QPushButton)[0]
    assert b.cursor().shape() == QtCore.Qt.CursorShape.PointingHandCursor
    w.dirty = False
    w.close()


def test_3d_paint_order_is_depth_correct(cfg):
    """Layers never interpenetrate in height (bottom-to-top painting is exact), and the shapes of one
    layer (electrode fingers) are painted far → near at every orientation."""
    import math
    from ustack.app import make_app
    from ustack.ui.style import setup_application
    from ustack.ui.stack3d import Stack3DView
    from ustack.scene3d import build_scene
    from ustack import templates
    app = make_app()
    setup_application(app)
    spec = templates.new_device_from(templates.load_template("mak_tmd"), "m")
    for k, v in {"channel_width_um": 13.6, "channel_length_um": 4.8, "n_probe_pairs": 2, "arm_width_um": 0.8,
                 "arm_length_um": 5.6, "sd_extension_um": 2.0}.items():
        spec.hall_bar.params[k].set(v, "user")
    slabs = build_scene(spec, cfg)
    for a, b in zip(slabs, slabs[1:]):
        assert b.z0 >= a.z0 + a.height - 1e-9, (a.name, b.name)
    fingers = next(s for s in slabs if s.uid == "PT")
    assert len(fingers.polys) == 6                       # 2 probe pairs × 2 sides + bent S and D
    view = Stack3DView()
    view.resize(800, 500)
    view.set_slabs(slabs)
    for az in range(0, 360, 30):
        view.az = math.radians(az)
        depths = [view._poly_depth(p, 1.0, 1.0, 0.0) for p in sorted(fingers.polys, key=lambda p: -view._poly_depth(p, 1.0, 1.0, 0.0))]
        assert depths == sorted(depths, reverse=True)
        view.grab()                                      # paints without error; scale bar inside the view
        assert view.scalebar_um


def test_library_buttons_edit_template_and_new_device_from_description(cfg, monkeypatch):
    from ustack.app import make_app
    from ustack.ui.style import QtWidgets, setup_application
    from ustack.ui.main_window import MainWindow
    from ustack.ui.new_device import NewDeviceDialog
    from ustack.ui import main_window as mw
    from ustack import templates
    app = make_app()
    setup_application(app)
    w = MainWindow(cfg)
    w.show()
    lib = w.library
    texts = {b.text() for b in lib.findChildren(QtWidgets.QPushButton)}
    assert {"Edit template", "New device", "Delete"} <= texts
    assert not texts & {"Save device", "Open", "Describe…"}             # Save device is in the title bar; Describe is a New device tab
    # New device: three ways to start
    dlg = NewDeviceDialog(cfg, w, start="description")
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == ["Start from a template", "Build from scratch",
                                                                       "Build from description"]
    assert dlg.tabs.currentIndex() == 2
    dlg.name.setText("described")
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dlg._create()
    assert dlg.spec is None and "Draft" in warned[0]                     # nothing drafted yet
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)              # the offline keyword parser
    dlg.describe.text.setPlainText("an hBN-encapsulated monolayer graphene Hall bar with a graphite back gate and edge contacts")
    dlg.describe.run()
    assert dlg.describe.draft is not None and dlg.describe.out.topLevelItemCount() > 2
    dlg._create()
    assert dlg.spec is not None and dlg.spec.name == "described"
    # Edit template: its own window (the open device untouched); Save overwrites the template (user, 2026-11-21) — for
    # a built-in, the lab's edit takes its place (the shipped file is kept; Restore built-in brings it back)
    from ustack.ui.template_editor import TemplateEditorDialog
    builtin = next(t for t in templates.list_templates() if t.builtin)
    before = builtin.path.read_text(encoding="utf-8")
    n_list = len(templates.list_templates())
    w.adopt_spec(templates.new_device_from(templates.load_template(builtin), "dev"), "new")
    dev_layers = [l.uid for l in w.spec.layers]
    d = TemplateEditorDialog(cfg, builtin, w)
    n = len(d.spec.layers)
    d.stack.refresh()
    d.stack.list.setCurrentRow(0)                                       # top layer selected
    d.stack._delete()
    assert len(d.spec.layers) == n - 1 and d.dirty
    assert [l.uid for l in w.spec.layers] == dev_layers                 # the device is not touched
    d.family.setCurrentIndex(d.family.findData("custom"))
    d._options_changed()
    assert d.save() and not d.dirty                                     # no name asked
    assert builtin.path.read_text(encoding="utf-8") == before
    infos = templates.list_templates()
    assert len(infos) == n_list                                         # in the built-in's place, not a new entry
    lab = next(t for t in infos if t.id == builtin.id)
    tpl = templates.load_template(lab)
    assert not lab.builtin and lab.overrides == builtin.id and len(tpl.layers) == n - 1 and tpl.family == "custom"
    assert len(templates.load_template(builtin.id).layers) == n - 1     # by id too
    d.desc.setPlainText("edited description")
    assert d.save()
    assert templates.load_template(lab).description == "edited description"
    assert len([t for t in templates.list_templates() if not t.builtin]) == 1
    d.close()
    # the Edit template button opens that window (exec stubbed)
    opened = []
    monkeypatch.setattr(TemplateEditorDialog, "exec", lambda self: opened.append(self.info.path) or 0)
    lib.refresh()
    lib.select_template(lab.path)
    lib.edit_template()
    assert opened == [lab.path]
    # Restore built-in (right-click › Delete on the lab's edit) brings the original back
    monkeypatch.setattr(mw, "confirm", lambda *a, **k: True)
    lib.select_template(lab.path)
    lib.delete_template()
    back = next(t for t in templates.list_templates() if t.id == builtin.id)
    assert back.builtin and len(templates.load_template(builtin.id).layers) == n
    monkeypatch.setattr(mw, "prompt_text", lambda *a, **k: "dev")
    w.save()
    # Delete at the bottom of Devices deletes the selected device (after confirming)
    w.save()
    monkeypatch.setattr(mw, "confirm", lambda *a, **k: True)
    assert lib.dev.currentItem() is not None
    lib.delete_selected_device()
    assert lib.dev.topLevelItemCount() == 0 and w.spec is None
    w.dirty = False
    w.close()
