"""UStack's main window, laid out like UAnalysis: frameless title bar, the device library on the
left, the workspace (Structure / Flake layout / EBL design) in the middle, and the settings tabs
(Stack, Pattern, Check, Stage 2) on the right."""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .. import devices, ebl_flow as ef, geometry as geo, imaging, paths, templates
from ..config import LabConfig, MissingItem, apply_answers
from ..model import FUNCTIONS, DeviceSpec
from ..scene3d import build_scene, layer_color
from ..validation import category, structured_summary, validate_stage1
from .canvas import Canvas
from .flake_images import FlakeImagesPanel
from .dialogs import AssistantDialog, MissingValuesDialog, PresetsDialog, RulesDialog
from .new_device import NewDeviceDialog
from .panels import CheckPanel, PatternPanel
from .stack3d import Stack3DView
from .stack_panel import StackPanel
from .ebl_flow_panel import EBLFlowPanel
from .style import (ASSET_DIR, THEME, Backdrop, Banner, Card, CheckBox, ComboBox, FramelessMainWindow, IconButton, PALETTE, Panel, Pill,
                    QtCore, QtGui, QtWidgets, SearchField, Slider, Splitter, TabWidget, ToolStrip, button, label, tool_strip_button)
from .widgets import AngleControl, buttons_row, clickable_rows, confirm, coord_label, install_hand_cursors, modernize, open_file, prompt_text, row

Qt = QtCore.Qt


class LibraryPanel(Panel):
    """Templates (the four group structures and the lab's own) and saved devices."""

    def __init__(self, app):
        super().__init__(kind="side")
        self.app = app
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)
        lay.addWidget(label("Templates", "cardTitle"))
        self.tpl = clickable_rows(QtWidgets.QTreeWidget())
        self.tpl.setHeaderHidden(True)
        self.tpl.setRootIsDecorated(False)
        self.tpl.itemDoubleClicked.connect(lambda *_: self.new_from_template())
        self.tpl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tpl.customContextMenuRequested.connect(self._template_menu)
        self.tpl.setToolTip("Double-click to start a device from it; right-click to edit or delete")
        lay.addWidget(self.tpl, 3)
        self.edit_tpl_button = button("Edit template", self.edit_template, "primary",
                                      "Edit the selected template in its own window (layer order, materials, rules …)")
        lay.addLayout(buttons_row(self.edit_tpl_button))      # left, in line with New device below
        lay.addWidget(label("Devices", "cardTitle"))
        self.search = SearchField("Filter devices")
        self.search.textChanged.connect(self.refresh_devices)
        lay.addWidget(self.search)
        self.dev = clickable_rows(QtWidgets.QTreeWidget())
        self.dev.setHeaderHidden(True)
        self.dev.setRootIsDecorated(False)
        self.dev.itemDoubleClicked.connect(lambda it, _c: self.app.open_device(Path(it.data(0, Qt.ItemDataRole.UserRole))))
        self.dev.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dev.customContextMenuRequested.connect(self._device_menu)
        self.dev.setToolTip("Double-click to open; right-click to rename or delete")
        lay.addWidget(self.dev, 4)
        # (Save device is in the title bar; Describe… is New device › Build from description — AssistantDialog is kept)
        lay.addLayout(buttons_row(button("New device", self.new_from_template, "primary",
                                         "New device: from a template, built from scratch, or from a description", "plus"),
                                  button("Delete", self.delete_selected_device, "danger", "Delete the selected device")))
        self.refresh()

    def refresh(self) -> None:
        self.tpl.clear()
        for info in templates.list_templates():
            it = QtWidgets.QTreeWidgetItem(self.tpl, [info.name + ("" if info.builtin else "  · edited" if info.overrides else "  · lab")])
            it.setData(0, Qt.ItemDataRole.UserRole, str(info.path))
            it.setToolTip(0, info.description)
        if self.tpl.topLevelItemCount():
            self.tpl.setCurrentItem(self.tpl.topLevelItem(0))
        self.refresh_devices()

    def refresh_devices(self) -> None:
        self.dev.clear()
        needle = self.search.text().strip().lower() if hasattr(self, "search") else ""
        for spec_path in sorted(paths.devices_dir().glob("*/spec.json"), key=lambda p: -p.stat().st_mtime):
            try:
                data = json.loads(spec_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            name = data.get("name", spec_path.parent.name)
            if needle and needle not in name.lower():
                continue
            ok = data.get("validation", {}).get("ok")
            it = QtWidgets.QTreeWidgetItem(self.dev, [f"{name}  {'✓' if ok else ''}"])
            it.setData(0, Qt.ItemDataRole.UserRole, str(spec_path))
            it.setToolTip(0, f"{data.get('family', '')} · modified {data.get('modified', '')}")

    def select_device(self, spec_path: Path) -> None:
        for i in range(self.dev.topLevelItemCount()):
            it = self.dev.topLevelItem(i)
            if Path(it.data(0, Qt.ItemDataRole.UserRole)) == Path(spec_path):
                self.dev.setCurrentItem(it)
                return

    def select_template(self, path) -> None:
        for i in range(self.tpl.topLevelItemCount()):
            it = self.tpl.topLevelItem(i)
            if Path(it.data(0, Qt.ItemDataRole.UserRole)) == Path(path):
                self.tpl.setCurrentItem(it)
                return

    def delete_selected_device(self) -> None:
        items = self.dev.selectedItems() or ([self.dev.currentItem()] if self.dev.currentItem() is not None else [])
        paths = [Path(it.data(0, Qt.ItemDataRole.UserRole)) for it in items if it.data(0, Qt.ItemDataRole.UserRole)]
        if not paths:
            self.app.status("Select a device in the list first")
            return
        if len(paths) == 1:
            self.delete_device(paths[0])
            return
        names = [self._device_name(p) for p in paths]           # several (Shift / Ctrl-click): one question for all
        if not confirm(self.app, "Delete devices?", f"Delete these {len(paths)} devices? Their folders in the library are removed. "
                       "This cannot be undone.\n\n" + "\n".join(names), "Delete", danger=True):
            return
        for path in paths:
            self.delete_device(path, ask=False)

    def _template_info(self, it=None):
        it = it or self.tpl.currentItem()
        if it is None:
            return None
        return next((t for t in templates.list_templates() if str(t.path) == it.data(0, Qt.ItemDataRole.UserRole)), None)

    def edit_template(self) -> None:
        info = self._template_info()
        if info is None:
            self.app.status("Select a template first")
            return
        self.app.edit_template(info)

    def _template_menu(self, pos) -> None:
        it = self.tpl.itemAt(pos)
        if it is None:
            return
        self.tpl.setCurrentItem(it)
        info = self._template_info(it)
        menu = QtWidgets.QMenu(self.tpl)
        menu.addAction("Edit template", self.edit_template)
        menu.addAction("New device from it…", self.new_from_template)
        act = menu.addAction("Restore built-in…" if info is not None and info.overrides else "Delete template…", self.delete_template)
        act.setEnabled(info is not None and not info.builtin)
        menu.exec(self.tpl.viewport().mapToGlobal(pos))

    def _device_menu(self, pos) -> None:
        it = self.dev.itemAt(pos)
        if it is None:
            return
        self.dev.setCurrentItem(it)
        path = Path(it.data(0, Qt.ItemDataRole.UserRole))
        menu = QtWidgets.QMenu(self.dev)
        menu.addAction("Rename…", lambda: self.rename_device(path))
        menu.addAction("Delete…", lambda: self.delete_device(path))
        menu.exec(self.dev.viewport().mapToGlobal(pos))

    @staticmethod
    def _device_name(path: Path) -> str:
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("name", path.parent.name)
        except Exception:
            return path.parent.name

    def rename_device(self, path: Path) -> None:
        old = self._device_name(path)
        name = prompt_text(self.app, "Rename device", f"New name for '{old}':", old, "Rename")
        if not name or name == old:
            return
        app = self.app
        is_open = app.spec is not None and app.spec_path is not None and Path(app.spec_path).resolve() == path.resolve()
        try:
            new_path = devices.rename_device(path, name)
        except devices.DeviceError as exc:
            app.warn("Rename device", str(exc))
            return
        if is_open:
            # the open device follows: new name and folder; in-memory paths into the old folder are rewritten
            dirty = app.dirty
            a, b = str(path.parent.resolve()), str(new_path.parent.resolve())
            app.spec.name = name
            for layer in app.spec.layers:
                if layer.image:
                    layer.image = devices.rewrite_paths(layer.image, a, b)
            app.spec.stage2 = devices.rewrite_paths(app.spec.stage2, a, b)
            app.spec_path = new_path
            app.stage2_panel.inputs.images = devices.rewrite_paths(app.stage2_panel.inputs.images, a, b)
            app.layer_images.clear()
            app.refresh_all()
            app.dirty = dirty
        self.refresh_devices()
        self.select_device(new_path)
        app.status(f"Renamed '{old}' to '{name}'")

    def delete_device(self, path: Path, ask: bool = True) -> None:
        name = self._device_name(path)
        if ask and not confirm(self.app, "Delete device?", f"Delete '{name}'? Its folder in the library (spec, copied images and "
                               f"exported Stage 2 files) is removed. This cannot be undone.\n\n{path.parent}", "Delete", danger=True):
            return
        app = self.app
        is_open = app.spec is not None and app.spec_path is not None and Path(app.spec_path).resolve() == path.resolve()
        try:
            devices.delete_device(path)
        except (devices.DeviceError, OSError) as exc:
            app.warn("Delete device", str(exc))
            return
        if is_open:
            app.close_device()
        self.refresh_devices()
        app.status(f"Deleted '{name}'")

    def new_from_template(self) -> None:
        """Always ask: a template (the one selected here is preselected) or a stack built from scratch."""
        it = self.tpl.currentItem()
        dlg = NewDeviceDialog(self.app.config, self.app, it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None)
        if dlg.exec() and dlg.spec is not None:
            self.app.adopt_spec(dlg.spec, "new")

    def delete_template(self) -> None:
        info = self._template_info()
        if info is None:
            return
        if info.builtin:
            self.app.warn("Templates", "Built-in templates cannot be deleted.")
            return
        if info.overrides:
            if not confirm(self.app, "Restore built-in?", f"Discard the lab's edit of '{info.name}' and bring the built-in template "
                           f"back? This cannot be undone.\n\n{info.path}", "Restore", danger=True):
                return
        elif not confirm(self.app, "Delete template?", f"Delete the lab template '{info.name}'? This cannot be undone.\n\n{info.path}",
                         "Delete", danger=True):
            return
        templates.delete_template(info)
        self.refresh()


class MainWindow(FramelessMainWindow):
    def __init__(self, config: LabConfig | None = None):
        super().__init__()
        install_hand_cursors()
        self.config = config or LabConfig()
        self.spec: DeviceSpec | None = None
        self.spec_path: Path | None = None
        self.dirty = False
        self.layer_images: dict[str, object] = {}
        self.ebl_view_key = "100x"
        # undo / redo: snapshots of the whole device specification (stage 2 flow included)
        self._undo: list[str] = []
        self._redo: list[str] = []
        self._snap: str | None = None
        self._restoring = False
        self._snap_timer = QtCore.QTimer(self)
        self._snap_timer.setSingleShot(True)
        self._snap_timer.setInterval(0)                 # one operation (many spec_changed calls) = one step
        self._snap_timer.timeout.connect(self._take_snapshot)
        self.setWindowTitle("UStack")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 720)
        self._build()
        THEME.changed.connect(self._theme_changed)
        self.statusBar().showMessage("Ready — choose a template and press New device, or open a saved device")

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        bar = self.install_chrome("UStack", self._logo())
        bar.left.addWidget(button("Save device", self.save, "primary", "Save the device to the devices list (Ctrl+S)", "check"))
        bar.left.addWidget(button("Save as template…", self.save_template, None, "Keep this stack as a lab template", "layers"))
        bar.left.addWidget(button("Configure", lambda: self.check_panel.configure(), None,
                                  "Generate every empty dimension so the stack passes validation"))
        bar.left.addWidget(button("Edit preset dimensions", self.presets, None,
                                  "The relations Configure uses to generate dimensions, and the microscope field of view"))
        self.undo_button = IconButton("undo", "Undo (Ctrl+Z / ⌘Z)", size=28, icon_size=18)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = IconButton("redo", "Redo (Ctrl+Y or Ctrl+Shift+Z / ⇧⌘Z)", size=28, icon_size=18)
        self.redo_button.clicked.connect(self.redo)
        bar.actions.addWidget(self.undo_button)
        bar.actions.addWidget(self.redo_button)
        self.rules_button = button("Rules…", self.rules, None, "Lab rules, GDS layers and materials (approve pending values)")
        bar.actions.addWidget(self.rules_button)
        self._update_rules_badge()
        root = Backdrop()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(4, 0, 4, 2)
        outer.setSpacing(0)
        split = Splitter(Qt.Orientation.Horizontal)
        self.library = LibraryPanel(self)
        self.library.setMinimumWidth(240)
        self.library.setMaximumWidth(420)
        split.addWidget(self.library)
        # centre: workspace tabs
        centre = Panel(kind="workspace")
        cl = QtWidgets.QVBoxLayout(centre)
        cl.setContentsMargins(8, 4, 8, 8)
        self.center_tabs = TabWidget()
        self.structure_page = self._structure_page()
        self.layout_page = self._layout_page()
        self.ebl_page = self._ebl_page()
        self.center_tabs.addTab(self.structure_page, "Structure")
        self.center_tabs.addTab(self.layout_page, "Flake layout")
        self.center_tabs.addTab(self.ebl_page, "EBL design")
        self.center_tabs.currentChanged.connect(self._center_changed)
        cl.addWidget(self.center_tabs)
        split.addWidget(centre)
        # right: settings
        side = Panel(flat=True)
        side.setMinimumWidth(400)
        side.setMaximumWidth(620)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 2, 0, 0)
        self.tabs = TabWidget()
        self.stack_panel = StackPanel(self)
        self.flake_panel = FlakeImagesPanel(self)
        self.view3d.hiddenChanged.connect(lambda _h: (self.flake_panel.sync_checks(), self.refresh_layout_canvas()))
        self.pattern_panel = PatternPanel(self)
        self.check_panel = CheckPanel(self)
        self.ebl_flow = EBLFlowPanel(self)
        self.stage2_panel = self.ebl_flow.classic           # the classic graphene Hall-bar step
        self.tabs.addTab(self.stack_panel, "Stack")
        self.tabs.addTab(self.flake_panel, "Flake images")
        self.tabs.addTab(self.pattern_panel, "Pattern")
        self.tabs.currentChanged.connect(self._side_changed)
        self.tabs.addTab(self.check_panel, "Check")
        self.tabs.addTab(self.ebl_flow, "Stage 2")
        sl.addWidget(self.tabs)
        split.addWidget(side)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([290, 960, 430])
        outer.addWidget(split, 1)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self, activated=self.save)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self, activated=self.redo)
        if QtGui.QKeySequence("Ctrl+Shift+Z") not in QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Redo):
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        if QtGui.QKeySequence("Ctrl+Y") not in QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Redo):
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self.check_panel.run)
        modernize(self)
        self.refresh_all()

    def _logo(self) -> QtGui.QPixmap | None:
        path = ASSET_DIR / "UStack_mark.png"
        if path.is_file():
            return QtGui.QPixmap(str(path)).scaled(26, 26, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return None

    def _structure_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        top = QtWidgets.QHBoxLayout()
        self.title_label = label("No device", "itemTitle")
        top.addWidget(self.title_label)
        top.addStretch(1)
        top.addWidget(button("Show all layers", lambda: self.view3d.set_hidden(set()), None,
                             "Show every layer again (click a swatch in the legend to hide or show one)"))
        top.addWidget(label("Separate Layers", "muted"))
        self.explode = Slider(0, 100, 0)
        self.explode.setFixedWidth(140)
        self.explode.valueChanged.connect(lambda v: self.view3d.set_explode(v / 25.0))
        top.addWidget(self.explode)
        lay.addLayout(top)
        split = Splitter(Qt.Orientation.Vertical, gap=6)
        self.view3d = Stack3DView()
        self.view3d.layerClicked.connect(self.select_layer)
        split.addWidget(self.view3d)
        spec_box = QtWidgets.QWidget()
        sb = QtWidgets.QVBoxLayout(spec_box)
        sb.setContentsMargins(0, 0, 0, 0)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label("Structured Specification", "cardTitle"))
        self.valid_pill = Pill("Not checked", "neutral")
        head.addWidget(self.valid_pill)
        head.addStretch(1)
        head.addWidget(button("Copy JSON", self.copy_spec_json, None, "Copy the full specification as JSON", "copy"))
        sb.addLayout(head)
        self.spec_tree = QtWidgets.QTreeWidget()
        self.spec_tree.setHeaderLabels(["Item", "Value"])
        self.spec_tree.setColumnWidth(0, 300)
        sb.addWidget(self.spec_tree)
        split.addWidget(spec_box)
        split.setSizes([560, 300])
        lay.addWidget(split, 1)
        return page

    def _toolstrip(self, canvas: Canvas, tools: list[tuple[str, str, str]]) -> tuple[QtWidgets.QHBoxLayout, dict]:
        lay = QtWidgets.QHBoxLayout()
        lay.setSpacing(2)
        strip = ToolStrip()
        sl = QtWidgets.QHBoxLayout(strip)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        group = QtWidgets.QButtonGroup(strip)
        group.setExclusive(True)
        buttons = {}
        for tool, icon, tip in tools:
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            strip.register(b)
            sl.addWidget(b)
            b.toggled.connect(lambda on, t=tool, c=canvas: on and c.set_tool(t))
            buttons[tool] = b
        buttons[tools[0][0]].setChecked(True)
        lay.addWidget(strip)
        for icon, tip, fn in (("fit", "Fit (hold H or the middle button to pan)", canvas.fit), ("zoomIn", "Zoom in", lambda: canvas.zoom_by(1.25)), ("zoomOut", "Zoom out", lambda: canvas.zoom_by(0.8))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            strip.register(b, secondary=True)
            lay.addWidget(b)
        lay.addStretch(1)
        return lay, buttons

    def _layout_page(self) -> QtWidgets.QWidget:
        """Flake layout: the traced borders only (no images), placed on each other — click a border's line
        to select it, drag to move it, turn it with the round handle above it (Shift: 15° steps); the view moves, zooms and turns like the
        EBL canvas."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        self.layout_canvas = c = Canvas()
        c.drag_outlines = True
        c.scale_handles = True                       # PowerPoint-style stretch / scale handles on the selected outline
        c.rotate_handles = True                      # … and the round rotate handle above it (replaces the turn slider, 2026-11-21)
        c.trackpad_gestures = True
        c.empty_text = "Trace the flakes first (Flake images tab), then place their borders on each other here"
        bar, self.layout_tools = self._toolstrip(c, [("pointer", "pointer", "Click a border's line to select it and drag it to move it; "
                                                      "drag its points to reshape it; drag elsewhere (or two-finger scroll) to move the "
                                                      "view; pinch to zoom, twist to rotate"),
                                                     ("addpoint", "addPoint", "Add a point: click on the selected border (keep the button down to drag it)"),
                                                     ("select", "selectPoints", "Select points: drag a box (Shift adds); drag one to move them all; "
                                                      "Delete removes them"),
                                                     ("draw", "annotate", "Draw the selected layer's border: click its corners, double-click to close"),
                                                     ("measure", "measure", "Measure a length: click the start, then click the end (double-click or Esc clears)")])
        self.layout_trash = tool_strip_button("trash", "Delete the selected points (Delete / Backspace)")
        self.layout_trash.setEnabled(False)
        self.layout_trash.clicked.connect(c.delete_selected_vertices)
        bar.insertWidget(bar.count() - 1, self.layout_trash)
        self.layout_info = coord_label("x -00000.00 µm   y -00000.00 µm")
        bar.addWidget(self.layout_info)
        lay.addLayout(bar)
        lay.addWidget(c, 1)
        rot, self.layout_rotation, self.layout_rotation_text, self.layout_rotation_reset = self._view_rotation_row(c)
        turn = QtWidgets.QHBoxLayout()
        turn.setContentsMargins(6, 0, 6, 2)
        self.layout_metal = CheckBox("Show metal parts")
        self.layout_metal.setToolTip("The fingers, contact gate and electrodes generated in Stage 2, where they are relative to the flakes "
                                     "— before that, the metal parts the auto design planned (dashed). Not editable here.")
        self.layout_metal.toggled.connect(lambda _on: self.refresh_layout_canvas())
        turn.addWidget(self.layout_metal)
        turn.addStretch(1)
        turn.addWidget(button("Export to image…", self.export_layout_image, None,
                              "Save what the Flake layout canvas shows as a JPG image (the device's folder by default)"))
        turn.addSpacing(6)
        turn.addWidget(button("Auto design…", self.auto_design, "primary",
                              "Place the traced flakes automatically: the stack's rules met, room for the Hall bars, contact gate, "
                              "gate contacts and electrode landings (turn and shift only)"))
        lay.addLayout(rot)
        lay.addLayout(turn)
        c.polygonEdited.connect(self._layout_edited)
        c.polygonDrawn.connect(self._layout_drawn)
        c.polygonSelected.connect(self._layout_selected)
        c.verticesSelected.connect(lambda n: self.layout_trash.setEnabled(n > 0))
        c.message.connect(self.status)
        c.cursorMoved.connect(lambda x, y: self.layout_info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        return page

    def _view_rotation_row(self, canvas: Canvas):
        """'View rotation' slider (±180°) + back to 0°, kept in step with the canvas (trackpad twist)."""
        rot = QtWidgets.QHBoxLayout()
        rot.setContentsMargins(6, 4, 6, 2)
        rot.addWidget(label("View rotation", "muted"))
        slider = AngleControl(0.0, "Turn the view (the coordinates do not change); double-click the slider for 0°, or type the angle")
        slider.setMaximumWidth(340)
        rot.addWidget(slider, 1)
        text = slider
        reset = tool_strip_button("reset", "Back to 0° (x to the right)")
        reset.clicked.connect(lambda: canvas.set_view_rotation(0.0))
        rot.addWidget(reset)
        rot.addStretch(1)
        slider.valueChanged.connect(lambda v: canvas.set_view_rotation(float(v)))

        def _rotated(deg: float) -> None:
            slider.blockSignals(True)
            slider.setValue(deg)
            slider.blockSignals(False)
        canvas.rotationChanged.connect(_rotated)
        return rot, slider, text, reset

    def _ebl_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        self.ebl_canvas = Canvas()
        self.ebl_canvas.drag_outlines = True
        self.ebl_canvas.scale_handles = True
        self.ebl_canvas.trackpad_gestures = True
        bar, self.ebl_tools = self._toolstrip(self.ebl_canvas, [("pointer", "pointer", "Click an outline's line to select it and drag "
                                                                 "it to move it; drag its points to reshape it; drag elsewhere (or "
                                                                 "two-finger scroll) to move the view; pinch to zoom, twist to rotate"),
                                                                ("wand", "mask", "Detect a flake border on the 100x image"),
                                                                ("draw", "annotate", "Draw a polygon (outline, defect, window): click corners, double-click to close"),
                                                                ("pick", "cursor", "Pick a point (site, alignment pairs, Hall-bar centre)"),
                                                                ("measure", "measure", "Measure a length: click the start, then click the end (double-click or Esc clears)")])
        self.ebl_info = coord_label()
        bar.addWidget(self.ebl_info)
        self.ebl_layers = tool_strip_button("layers", "Layers on the canvas: show or hide the bond pads, the image, each flake "
                                                      "and each step's pattern")
        self.ebl_layers.clicked.connect(self.show_canvas_layers)
        bar.addWidget(self.ebl_layers)
        lay.addLayout(bar)
        lay.addWidget(self.ebl_canvas, 1)
        self.ebl_canvas.cursorMoved.connect(lambda x, y: self.ebl_info.setText(f"chip x {x:9.2f} µm   y {y:9.2f} µm"))
        # view rotation: slider (±180°), back to 0°; a two-finger twist on the trackpad turns it too
        rot, self.ebl_rotation, self.ebl_rotation_text, self.ebl_rotation_reset = self._view_rotation_row(self.ebl_canvas)
        lay.addLayout(rot)
        return page

    def show_canvas_layers(self) -> None:
        """The Layers card under the EBL canvas's top-right button."""
        from .ebl_flow_panel import CanvasLayersCard
        self.canvas_layers_card = CanvasLayersCard(self.ebl_flow, self)
        self.canvas_layers_card.popup_under(self.ebl_layers)

    def set_ebl_tool(self, tool: str) -> None:
        b = self.ebl_tools.get(tool)
        if b is not None:
            b.setChecked(True)
        self.ebl_canvas.set_tool(tool)
        self.center_tabs.setCurrentWidget(self.ebl_page)

    # ------------------------------------------------------------------ device life cycle
    def device_dir(self) -> Path:
        name = paths.safe_name(self.spec.name if self.spec else "untitled")
        d = paths.devices_dir() / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def adopt_spec(self, spec: DeviceSpec, why: str = "") -> None:
        self.spec = spec
        self.spec_path = None if why == "new" else self.spec_path
        self.layer_images = {}
        self.dirty = why == "new"
        self.stage2_panel.design = None
        self.stage2_panel.report = None
        if why == "new":
            self.stage2_panel.inputs.outlines.clear()
        self.refresh_all()
        self._twist_on_open()
        self.check_panel.run()
        if why == "new":
            self.reset_history()
        else:
            self._snap_timer.start()                       # e.g. a Stage 2 export or an assistant draft: one undo step

    def open_device(self, path: Path) -> None:
        try:
            spec = DeviceSpec.load(path)
        except Exception as exc:
            self.warn("Open", f"{path}\n{exc}")
            return
        self.spec, self.spec_path, self.dirty = spec, path, False
        self.layer_images = {}
        from ..stage2 import StepInputs
        self.stage2_panel.inputs = StepInputs(purposes=[])
        self.stage2_panel.design = self.stage2_panel.report = None
        self.refresh_all()
        if self._twist_on_open():
            self.dirty = True
        self.check_panel.run()
        self.reset_history()
        # the EBL canvas as saved (image on canvas, flakes, patterns, view); a saved view is applied when it is on screen
        if (self.ebl_flow.flow.ui or {}).get("tab") == "ebl" and self.ebl_flow.keys and self.center_tabs.currentWidget() is not self.ebl_page:
            self.center_tabs.setCurrentWidget(self.ebl_page)
        else:
            self.ebl_flow.redraw(fit=True)
        self.status(f"Opened {path}")

    def _twist_on_open(self) -> dict:
        """A device opened / made: its device layers take their twists (user, 2026-11-22), said in the status bar."""
        turned = self._sync_twist()
        if turned:
            self._reset_outline_cache()
            self.flake_panel.refresh()
            self.refresh_structure()
            self.refresh_layout_canvas()
            self.status("Device layers turned by their twist: " + ", ".join(
                f"{self.spec.layer(u).label} {d:+.2f}°" for u, d in turned.items()))
        return turned

    def close_device(self) -> None:
        """Nothing open (after the open device was deleted from the library)."""
        from ..stage2 import StepInputs
        self.spec, self.spec_path, self.dirty = None, None, False
        self.layer_images = {}
        self.stage2_panel.inputs = StepInputs(purposes=[])
        self.stage2_panel.images = {}
        self.stage2_panel.design = self.stage2_panel.report = None
        self.refresh_all()
        self.check_panel.report = None
        self.reset_history()

    # ------------------------------------------------------------------ templates
    def window_title(self) -> str:
        if self.spec is None:
            return "UStack"
        return f"UStack — {self.spec.name}" + (" •" if self.dirty else "")

    def edit_template(self, info) -> None:
        """Library › Edit template: the template in its own window (order, materials, functions, rules …); the open
        device and the Structure / Stack tabs are not touched."""
        from .template_editor import TemplateEditorDialog
        dlg = TemplateEditorDialog(self.config, info, self)
        dlg.exec()
        if dlg.saved_path is not None:
            self.library.refresh()
            self.library.select_template(dlg.saved_path)

    def _name_new_device(self) -> bool:
        """The first Save of a device asks its name (user, 2026-11-21). Files copied into the device's folder
        before (Flake images, Stage 2 images) move to the folder of the name chosen; their paths follow."""
        name = self.spec.name
        while True:
            name = prompt_text(self, "Save device", "Name of the device (its folder in the library):", name, "Save")
            if not name:
                return False
            new_dir = paths.devices_dir() / paths.safe_name(name)
            if not (new_dir / "spec.json").exists():
                break
            self.warn("Save device", f"A device named '{new_dir.name}' is in the library already. Choose another name.")
        old_dir = paths.devices_dir() / paths.safe_name(self.spec.name)
        self.spec.name = name
        if old_dir.resolve() != new_dir.resolve() and old_dir.is_dir() and not (old_dir / "spec.json").exists():
            from ..devices import rewrite_paths
            if new_dir.exists():
                for item in old_dir.iterdir():
                    if not (new_dir / item.name).exists():
                        shutil.move(str(item), str(new_dir / item.name))
            else:
                shutil.move(str(old_dir), str(new_dir))
            old, new = str(old_dir.resolve()), str(new_dir.resolve())
            self.ebl_flow.remember_ui()
            for layer in self.spec.layers:
                if layer.image:
                    layer.image = rewrite_paths(layer.image, old, new)
            self.spec.stage2 = rewrite_paths(self.spec.stage2, old, new)
            self.ebl_flow.load_from_spec()
            self.ebl_flow.build()
        return True

    def save(self) -> bool:
        if self.spec is None:
            return False
        if self.spec_path is None and not self._name_new_device():
            return False
        report = validate_stage1(self.spec, self.config)
        self.spec.validation = report.to_json()
        self.ebl_flow.remember_ui()                       # reopening shows the EBL canvas as it is now
        path = self.device_dir() / "spec.json"
        self.spec.save(path, "Saved")
        self.spec_path, self.dirty = path, False
        self.titlebar.set_title(f"UStack — {self.spec.name}")
        self.library.refresh_devices()
        self.library.select_device(path)
        self.refresh_structure(report)
        self.check_panel.run()
        # saving is allowed at any stage: missing values are listed in the Check tab, not asked for here
        more = f" {len(report.missing_items())} values still missing (Configure or Check › Fill in missing)." if report.missing else ""
        self.status(f"Device saved to {path} — {report.summary()}.{more}")
        return True

    def after_configure(self, result) -> None:
        """Refresh everything after Configure changed the spec."""
        self.spec_changed("layers")
        self.pattern_panel.refresh()
        self.stage2_panel.build()
        self.check_panel.run()
        self.status(f"Configure: {len(result.changes)} values generated · {result.report.summary() if result.report else ''}")

    def save_template(self) -> None:
        if self.spec is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save as template", "Template name:", text=self.spec.name)
        if ok and name.strip():
            path = templates.save_as_template(self.spec, name.strip())
            self.library.refresh()
            self.status(f"Template saved: {path}")

    def assistant(self) -> None:
        dlg = AssistantDialog(self.config, self)
        if dlg.exec() and dlg.draft is not None:
            name, ok = QtWidgets.QInputDialog.getText(self, "New device", "Device name:", text=dlg.draft.spec.name)
            if ok and name.strip():
                dlg.draft.spec.name = name.strip()
                self.adopt_spec(dlg.draft.spec, "new")

    def presets(self) -> None:
        PresetsDialog(self.config, self).exec()
        self.config.load()

    def rules(self) -> None:
        RulesDialog(self.config, self).exec()
        self.config.load()
        self._update_rules_badge()
        self.spec_changed("rules")

    def _update_rules_badge(self) -> None:
        pend = self.config.pending_summary()
        n = sum(len(v) for v in pend.values())
        self.rules_button.setText(f"Rules ({n} to review)" if n else "Rules")

    # ------------------------------------------------------------------ refresh
    def refresh_all(self) -> None:
        self._reset_outline_cache()
        self.flake_panel.refresh()
        self.stack_panel.refresh()
        self.pattern_panel.refresh()
        self.stage2_panel.build()
        self.ebl_flow.load_from_spec()
        self.ebl_flow.build()
        self.refresh_structure()
        self.refresh_layout_canvas(fit=True)
        self.titlebar.set_title(self.window_title())

    # ------------------------------------------------------------------ undo / redo
    UNDO_LIMIT = 200

    def _spec_json(self) -> str | None:
        return json.dumps(self.spec.to_json(), sort_keys=True, default=str) if self.spec is not None else None

    def reset_history(self) -> None:
        self._snap_timer.stop()
        self._undo, self._redo = [], []
        self._snap = self._spec_json()
        self._update_undo_buttons()

    def _take_snapshot(self) -> None:
        if self._restoring or self.spec is None:
            return
        cur = self._spec_json()
        if cur != self._snap and self._snap is not None:
            self._undo.append(self._snap)
            del self._undo[:-self.UNDO_LIMIT]
            self._redo.clear()
        self._snap = cur
        self._update_undo_buttons()

    def _update_undo_buttons(self) -> None:
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(bool(self._undo))
            self.redo_button.setEnabled(bool(self._redo))

    def undo(self) -> None:
        self._step_history(self._undo, self._redo, "Undone")

    def redo(self) -> None:
        self._step_history(self._redo, self._undo, "Redone")

    def _step_history(self, src: list, dst: list, word: str) -> None:
        if self._snap_timer.isActive():
            self._snap_timer.stop()
            self._take_snapshot()
        if self.spec is None or not src:
            self.status(f"Nothing to {word[:-1].lower()}")
            return
        dst.append(self._snap)
        self._restore(src.pop())
        self.status(f"{word} the last change")

    def _restore(self, snap: str) -> None:
        old = self.spec
        spec = DeviceSpec.from_json(json.loads(snap))
        for layer in spec.layers:                         # keep the images already read when their file is the same
            prev = old.layer(layer.uid) if old is not None else None
            if prev is None or (prev.image or {}).get("path") != (layer.image or {}).get("path"):
                self.layer_images.pop(layer.uid, None)
        current = self.stack_panel.current
        self._restoring = True
        try:
            self.spec = spec
            self.dirty = True
            self.refresh_all()
            if current and spec.layer(current) is not None:
                self.stack_panel.refresh(current)
            if self.center_tabs.currentWidget() is self.ebl_page:
                self.ebl_flow.redraw()
        finally:
            self._restoring = False
        self._snap_timer.stop()
        self._snap = snap
        self.titlebar.set_title(self.window_title())
        self._update_undo_buttons()

    def spec_changed(self, what: str = "") -> None:
        self.dirty = True
        twisted = {}
        if not self._restoring:
            twisted = self._sync_twist()                  # a twist typed / a device layer traced: the borders turn
            self._snap_timer.start()
            self._sync_outlines()
        if self.spec is not None:
            self.titlebar.set_title(self.window_title())
        if what in ("name", "family", "contact_scheme", "rules"):
            self.pattern_panel.refresh()
            self.stage2_panel.build()
        if what in ("layers", "order", "material", "function"):
            self.stack_panel.refresh(self.stack_panel.current)
            self.pattern_panel.refresh()
        if what in ("layers", "order", "material", "function", "name", "values", "value", "rules") or twisted:
            self.flake_panel.refresh()
        self.refresh_structure()
        if what in ("layers", "order", "image", "material", "outline", "stage2") or twisted:
            self.refresh_layout_canvas()

    # ------------------------------------------------------------------ one traced border everywhere
    def _reset_outline_cache(self) -> None:
        self._outline_cache = {l.uid: copy.deepcopy(l.outline) for l in self.spec.layers} if self.spec is not None else {}

    def outline_synced(self, uid: str) -> None:
        """``uid``'s traced border was changed together with its Stage 2 outlines (nothing to carry over)."""
        layer = self.spec.layer(uid) if self.spec else None
        if layer is not None:
            self._outline_cache[uid] = copy.deepcopy(layer.outline)

    def _sync_outlines(self) -> None:
        """A traced border reshaped anywhere (Flake images, Flake layout) → the same shape in every
        Stage 2 step where that flake's outline is the border moved. Moving / turning a border as a
        whole (placing it in Flake layout) does not move it on the chip."""
        if self.spec is None:
            return
        cache = getattr(self, "_outline_cache", None)
        if cache is None:
            self._reset_outline_cache()
            return
        for layer in self.spec.layers:
            old, new = cache.get(layer.uid), layer.outline
            if old == new:
                continue
            cache[layer.uid] = copy.deepcopy(new)
            if old and new and not ef.is_rigid_move(old, new):
                self.ebl_flow.design_outline_changed(layer.uid, old)
            elif old and new:                          # moved / turned whole: a flake still to come takes its plan along
                self.ebl_flow.layout_moved(layer.uid)

    def traced_borders_changed(self, uids) -> None:
        """Traced borders changed from Stage 2: redraw where they are shown (Flake images, its trace
        window, Flake layout; the 3D stack follows spec_changed)."""
        self.flake_panel.refresh()
        dlg = self.flake_panel.trace_dialog
        if dlg is not None and dlg.isVisible() and dlg.uid in uids:
            dlg.redraw()
        self.refresh_layout_canvas()

    def report_changed(self, report) -> None:
        tone = "ok" if report.ok else "warn"
        self.valid_pill.set_tone(tone, report.summary())

    def refresh_structure(self, report=None) -> None:
        spec = self.spec
        if spec is None:
            self.view3d.set_slabs([], "Choose a template and press New device")
            self.title_label.setText("No device")
            self.spec_tree.clear()
            return
        # once Stage 2 knows where the flakes are on the chip (and has drawn patterns), the 3D stack shows them there
        chip = ef.structure_footprints(spec, self.ebl_flow.flow) if getattr(self, "ebl_flow", None) is not None else None
        tag = ("  ·  with Stage 2" if chip[1].startswith("On the chip") or "patterns:" in chip[1] else "  ·  with the auto design") if chip else ""
        self.title_label.setText(f"{spec.name}  ·  {spec.family.replace('_', ' ')}" + tag)
        self.title_label.setToolTip(chip[1] if chip else "")
        self.view3d.set_slabs(build_scene(spec, self.config, chip[0] if chip else None))
        summary = structured_summary(spec, self.config)
        self.spec_tree.clear()
        self._fill_tree(self.spec_tree.invisibleRootItem(), summary)
        self.spec_tree.expandToDepth(0)

    def _fill_tree(self, parent, data, key: str = "") -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    it = QtWidgets.QTreeWidgetItem(parent, [str(k).replace("_", " "), ""])
                    self._fill_tree(it, v)
                else:
                    QtWidgets.QTreeWidgetItem(parent, [str(k).replace("_", " "), "—" if v is None else str(v)])
        elif isinstance(data, list):
            for i, v in enumerate(data, 1):
                if isinstance(v, dict):
                    title = v.get("name", str(i))
                    it = QtWidgets.QTreeWidgetItem(parent, [f"{i}. {title}", v.get("material", "") if isinstance(v.get("material"), str) else ""])
                    self._fill_tree(it, {k: x for k, x in v.items() if k != "name"})
                else:
                    QtWidgets.QTreeWidgetItem(parent, [str(i), str(v)])

    def copy_spec_json(self) -> None:
        if self.spec is not None:
            QtWidgets.QApplication.clipboard().setText(json.dumps(self.spec.to_json(), indent=2, ensure_ascii=False))
            self.status("Specification copied as JSON")

    def layer_selected(self, uid: str) -> None:
        self.view3d.set_highlight(uid)
        self.layout_canvas.set_active_polygon(f"outline:{uid}" if f"outline:{uid}" in self.layout_canvas.polys else None)
        self._layout_lock_status(uid if f"outline:{uid}" in self.layout_canvas.polys else None)

    def select_layer(self, uid: str) -> None:
        if self.tabs.currentWidget() is self.flake_panel:
            self.flake_panel.select(uid)
            return
        self.tabs.setCurrentWidget(self.stack_panel)
        self.stack_panel.select_uid(uid)

    def _center_changed(self, i: int) -> None:
        # Structure and Flake layout go with any side tab (Stack, Flake images, …); EBL design needs Stage 2
        if self.center_tabs.widget(i) is self.ebl_page:
            self.tabs.setCurrentWidget(self.ebl_flow)
            if not (self.ebl_canvas.images or self.ebl_canvas.polys) or getattr(self.ebl_flow, "_pending_view", None):
                QtCore.QTimer.singleShot(0, lambda: self.ebl_flow.redraw(fit=True))   # first look after opening a device

    def _theme_changed(self) -> None:
        for c in (self.layout_canvas, self.ebl_canvas):
            c.setBackgroundBrush(QtGui.QColor(PALETTE["surface"]))
        self.view3d.update()

    # ------------------------------------------------------------------ Stage 1 flake layout
    def refresh_layout_canvas(self, fit: bool = False) -> None:
        """The traced borders on each other (design frame), without their images."""
        c = self.layout_canvas
        active = c.active_poly
        c.clear_all()
        if self.spec is None:
            return
        # flakes already on the chip: shape and place from Stage 2, locked here (dashed); the others stay movable
        view = ef.layout_view(self.spec, self.ebl_flow.flow) if getattr(self, "ebl_flow", None) is not None else {"fixed": {}}
        self._layout_fixed = view.get("step", {})
        hidden = set(self.view3d.hidden) if getattr(self, "view3d", None) is not None else set()   # the Flake images ticks
        if getattr(self, "layout_metal", None) is not None and self.layout_metal.isChecked() and getattr(self, "ebl_flow", None) is not None:
            sources: dict = {}                          # Stage 2's patterns, else the auto design's plan (dashed)
            for uid, polys in ef.layout_patterns(self.spec, self.ebl_flow.flow, view, sources).items():
                layer = self.spec.layer(uid)
                if uid not in hidden and layer is not None:
                    c.add_overlay_polys(polys, layer_color(layer, self.config), 0.45, z=95 + self.spec.layers.index(layer), width=0.8,
                                        dash=sources.get(uid) == "plan")
        for z, layer in enumerate(self.spec.layers):
            if layer.uid in hidden:
                continue
            fixed = view["fixed"].get(layer.uid)
            if fixed:
                c.set_polygon(f"outline:{layer.uid}", fixed[0], layer_color(layer, self.config), False, 100 + z, 0.15, dash=True)
                for i, poly in enumerate(fixed[1:], 1):
                    c.set_polygon(f"outline:{layer.uid}#{i}", poly, layer_color(layer, self.config), False, 100 + z, 0.15, dash=True)
            elif layer.outline:
                c.set_polygon(f"outline:{layer.uid}", layer.outline, layer_color(layer, self.config), True, 100 + z, 0.15)
        if active in c.polys:
            c.set_active_polygon(active)
        self._layout_lock_status(active.split(":", 1)[1] if active in c.polys else None)
        if fit:
            c.fit()

    def export_layout_image(self) -> Path | None:
        """Flake layout › Export to image… (user, 2026-11-21): what the canvas shows, as a JPG, in the device's folder
        (default) or another folder, under a name the user gives."""
        if self.spec is None:
            return None
        from .save_gds import ask_save_path
        name = f"{paths.safe_name(self.spec.name)}_flake_layout.jpg"
        path = ask_save_path(self, self.device_dir(), name, "Export to image", ".jpg", "image", "image_dir")
        if path is None:
            return None
        return self.save_layout_image(path)

    def save_layout_image(self, path: Path) -> Path | None:
        c = self.layout_canvas
        img = c.viewport().grab().toImage()              # as shown (full resolution on a Retina screen)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not img.save(str(path), "JPG", 95):
            self.warn("Export to image", f"Could not write {path}.")
            return None
        self.status(f"Flake layout saved to {path}")
        return path

    def auto_design(self) -> None:
        """Flake layout › Auto design…: its own window (:class:`ustack.ui.auto_design_window.AutoDesignDialog`)."""
        if self.spec is None:
            return
        from .auto_design_window import AutoDesignDialog
        dlg = AutoDesignDialog(self)
        dlg.exec()

    def layout_auto_applied(self, moved: list, cand) -> None:
        """The auto design moved ``moved`` flakes' borders (and images) in Flake layout: one undo step."""
        self.layout_metal.setChecked(True)              # the planned metal parts shown with the flakes
        self.spec_changed("image")
        self.flake_panel.refresh()
        self.refresh_layout_canvas(fit=True)
        self.refresh_structure()
        self.status(f"Auto design applied: {len(moved)} flake(s) placed — {cand.summary()}")

    def _layout_selected(self, key: str) -> None:
        uid = key.split(":", 1)[1] if key.startswith("outline:") else None
        self._layout_lock_status(uid)
        if uid and self.spec is not None and self.spec.layer(uid) is not None:
            self.select_layer(uid)

    @property
    def _layout_turns(self) -> dict:
        """Turn of borders without an image (kept with the device: ``ebl_flow.layout_turns``)."""
        return ef.layout_turns(self.spec) if self.spec is not None else {}

    def _border_turn(self, layer) -> float:
        return ef.border_turn(layer, self.spec)

    def _sync_twist(self, anchor: str | None = None) -> dict:
        """The device layers turned by their twists (user, 2026-11-22): as they lie in their pictures, each turned by its
        twist from the Stack tab relative to the device layer below; ``anchor`` (a device layer the user turned) keeps its
        new turn and the others follow it. Layers on the chip (Stage 2) are left alone."""
        if self.spec is None:
            return {}
        flow = self.ebl_flow.flow if getattr(self, "ebl_flow", None) is not None else None
        fixed = set(ef.layout_view(self.spec, flow)["fixed"]) if flow is not None else set()
        return ef.apply_device_twist(self.spec, anchor=anchor, skip=fixed)

    def _layout_lock_status(self, uid: str | None) -> None:
        layer = self.spec.layer(uid) if (self.spec is not None and uid) else None
        step = getattr(self, "_layout_fixed", {}).get(uid) if uid else None
        if step and layer is not None:
            self.status(f"{layer.label} is on the chip ({ef.STEPS[step].name}): its shape and place come from Stage 2 (locked here)")

    def turn_border(self, layer, delta: float) -> None:
        """Turn ``layer``'s traced border by ``delta``° about its middle; its image turns with it (or the kept turn)."""
        ef.turn_layer(layer, delta, self.spec)
        if layer.uid in ef.twist_layer_uids(self.spec):
            self._sync_twist(anchor=layer.uid)            # the other device layers keep their twist to it

    def layer_image(self, uid: str):
        """The flake image of layer ``uid`` (read from its file the first time)."""
        arr = self.layer_images.get(uid)
        layer = self.spec.layer(uid) if self.spec else None
        if arr is None and layer is not None and layer.image and layer.image.get("path") and Path(layer.image["path"]).is_file():
            try:
                arr = self.layer_images[uid] = imaging.load_image(layer.image["path"])
            except Exception:
                arr = None
        return arr

    def _side_changed(self, i: int) -> None:
        if self.tabs.widget(i) is self.flake_panel:
            self.flake_panel.refresh()
            if self.center_tabs.currentWidget() is self.ebl_page:    # Structure / Flake layout stay as they are
                self.center_tabs.setCurrentWidget(self.structure_page)

    def open_flake_images(self, uid: str | None = None) -> None:
        self.tabs.setCurrentWidget(self.flake_panel)
        if uid:
            self.flake_panel.select(uid)

    def load_layer_image(self, uid: str) -> None:
        layer = self.spec.layer(uid)
        path = open_file(self, f"OM image for {layer.label}", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not path:
            return
        try:
            arr = imaging.load_image(path)
        except Exception as exc:
            self.warn("Image", str(exc))
            return
        scale = self._ask_scale(arr.shape[1])
        if scale is None:
            return
        dest = self.device_dir() / "images" / f"{uid}_{Path(path).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)
        h, w = arr.shape[:2]
        layer.image = {"path": str(dest), "um_per_px": scale, "angle_deg": 0.0, "x": -w * scale / 2, "y": h * scale / 2,
                       "opacity": 0.6, "visible": True, "mirror": False}
        self.layer_images[uid] = arr
        self.spec_changed("image")
        self.refresh_layout_canvas(fit=True)
        self.stack_panel.refresh(uid)
        self.center_tabs.setCurrentWidget(self.layout_page)

    def _ask_scale(self, width_px: int | None = None) -> float | None:
        """µm per pixel of a flake image: the objective's field of view (Presets › Microscope; the 100x
        image is 143.28 × 95.13 µm and other objectives scale as 100 / magnification) over the image
        width, or a scale entered by hand."""
        from ..config import OBJECTIVES
        options = [f"{o}  ({self.config.field_of_view_um(o)[0]:g} µm wide)" if self.config.field_of_view_um(o) else o for o in OBJECTIVES]
        obj_i = OBJECTIVES.index("100x")
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Image scale", "Objective used for this image:",
                                                    options + ["Other (enter µm/px)"], obj_i, False)
        if not ok:
            return None
        obj = choice.split()[0]
        if obj in OBJECTIVES and width_px:
            scale = self.config.um_per_px(obj, width_px)
            if scale:
                return scale
        item = MissingItem("spec:image_scale", "Image scale", "µm/px")
        dlg = MissingValuesDialog([item], self, "Image scale")
        if not dlg.exec():
            return None
        v = next(iter(dlg.answers().values()), None)
        return None if v is None else float(v)

    @staticmethod
    def _image_qtransform(img: dict):
        from .canvas import image_transform
        return image_transform(img["um_per_px"], img.get("angle_deg", 0.0), img.get("x", 0.0), img.get("y", 0.0), img.get("mirror", False))

    def set_image_param(self, uid: str, key: str, value, redraw: bool = True) -> None:
        layer = self.spec.layer(uid)
        if layer is None or layer.image is None or value is None:
            return
        if key in ("um_per_px", "angle_deg", "mirror") and layer.image.get("um_per_px") and uid in self.layer_images:
            # scale / rotate / mirror about the image centre; the traced outline follows its image
            h, w = self.layer_images[uid].shape[:2]
            old = self._image_qtransform(layer.image)
            centre = old.map(QtCore.QPointF(w / 2, h / 2))
            inv, ok = old.inverted()
            px = [inv.map(QtCore.QPointF(x, y)) for x, y in (layer.outline or [])] if ok else []
            layer.image[key] = value
            layer.image["x"], layer.image["y"] = 0.0, 0.0
            c0 = self._image_qtransform(layer.image).map(QtCore.QPointF(w / 2, h / 2))
            layer.image["x"], layer.image["y"] = centre.x() - c0.x(), centre.y() - c0.y()
            if px:
                new = self._image_qtransform(layer.image)
                layer.outline = [[q.x(), q.y()] for q in (new.map(p) for p in px)]
            self.refresh_layout_canvas()
            self.dirty = True
            return
        layer.image[key] = value
        if key == "ref_angle_deg":                        # the lattice direction changed: the twist turns the border
            self.spec_changed("image")
            return
        item = self.layout_canvas.images.get(uid)
        if key == "opacity" and item is not None:
            item.setOpacity(float(value))
        elif key == "visible" and item is not None:
            item.setVisible(bool(value))
        elif redraw:
            self.refresh_layout_canvas()
        self.dirty = True

    def _layout_edited(self, key: str, pts: list) -> None:
        """A border edited on the Flake layout canvas: moved whole → its image moves with it (placed);
        reshaped → the traced border changes (and with it Flake images, the 3D stack, Stage 2)."""
        uid = key.split(":", 1)[1]
        layer = self.spec.layer(uid)
        if layer is None:
            return
        d = ef.translation_of(layer.outline, pts) if layer.outline else None
        if d is None and layer.outline and ef.is_rigid_move(layer.outline, pts, 1e-4):
            m, t = ef._rigid(layer.outline, pts)
            if np.linalg.det(m) > 0:              # turned with the rotate handle: the image turns with it (placed)
                ang = math.degrees(math.atan2(m[1, 0], m[0, 0]))
                if (layer.image or {}).get("um_per_px"):
                    layer.image = ef._image_after(layer.image, m, t)
                else:
                    self._layout_turns[uid] = self._border_turn(layer) + ang
                    if layer.image is not None:
                        layer.image["placed"] = True
                layer.outline = [list(p) for p in pts]
                if uid in ef.twist_layer_uids(self.spec):
                    self._sync_twist(anchor=uid)  # the other device layers turn with it (their twists kept)
                self.spec_changed("image")
                self.flake_panel.refresh()
                return
        if d is not None and layer.image is not None:
            layer.image["x"] = layer.image.get("x", 0.0) + d[0]
            layer.image["y"] = layer.image.get("y", 0.0) + d[1]
            layer.image["placed"] = True          # placed by hand: tracing again no longer re-centres it
        elif d is None and layer.image:
            layer.image.setdefault("trace", {})["how"] = "edited"
        layer.outline = [list(p) for p in pts]
        self.spec_changed("image" if d is not None else "outline")
        self.flake_panel.refresh()

    def _layout_drawn(self, pts: list) -> None:
        if self.center_tabs.currentWidget() is not self.layout_page:
            return
        uid = self.stack_panel.current
        layer = self.spec.layer(uid) if uid else None
        if layer is not None:
            layer.outline = [list(p) for p in pts]
            self.spec_changed("outline")
            self.refresh_layout_canvas()

    def clear_outline(self, uid: str) -> None:
        layer = self.spec.layer(uid)
        if layer is not None:
            layer.outline = None
            self.spec_changed("outline")
            self.refresh_layout_canvas()

    # ------------------------------------------------------------------ messages
    def status(self, text: str) -> None:
        self.statusBar().showMessage(text, 12000)

    def warn(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, text)

    def closeEvent(self, e):  # noqa: N802 - Qt API
        if self.dirty and self.spec is not None:
            ans = QtWidgets.QMessageBox.question(self, "UStack", "Save the device before closing?",
                                                 QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                                                 | QtWidgets.QMessageBox.StandardButton.Cancel)
            if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
            if ans == QtWidgets.QMessageBox.StandardButton.Save and not self.save():
                e.ignore()
                return
        super().closeEvent(e)
