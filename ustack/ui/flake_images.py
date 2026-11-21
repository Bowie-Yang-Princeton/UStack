"""Flake images: one optical image per flake, its border traced, and the stack built from the traced
borders. Three windows, so none is cramped:

- **the panel** (right-hand tab "Flake images", between Stack and Pattern): every layer (top → bottom)
  with its image state; Upload all… / Trace all, and for the selected flake Upload, View, Trace
  flake…, Clear and the image scale. The ticks choose the layers shown in the 3D stack.
- **the image**: View opens the image alone in its own window (zoom, scale bar).
- **the trace**: Trace flake… opens :class:`TraceDialog`, the image with its border — auto trace,
  next candidate, the wand (click the flake), drawing by hand, dragging points.

Trace all traces every flake that has an image, *whole flake* mode, without a click; borders you
traced by clicking, drew or edited are kept. The Structure tab's 3D view stacks the real borders
(each centred on the design origin until it is placed in the Flake layout tab)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .. import imaging
from ..config import OBJECTIVES
from ..scene3d import layer_color
from ..validation import category
from .canvas import Canvas, image_transform
from .style import (Card, CheckBox, ComboBox, Dialog, QtCore, QtGui, QtWidgets, button, clear_layout, label, scroll,
                    tool_strip_button)
from .widgets import CheckDelegate, NumberField, PickTree, coord_label, buttons_row, clickable_rows, modernize, open_file, row

Qt = QtCore.Qt
STATUS_TEXT = {"auto": "auto-traced", "click": "traced", "drawn": "drawn", "edited": "edited"}
HAND_MADE = ("click", "drawn", "edited", "chip")      # chip: taken from the flake found on the chip (Stage 2)


def is_flake(config, layer) -> bool:
    return category(config, layer) not in ("metal", "substrate", None)


def prefers_faint(config, layer) -> bool:
    """Device layers of graphene or a semiconductor are usually a thin piece attached to thicker ones:
    look for the faintest region first."""
    cat = category(config, layer)
    thin = layer.layer_count.value is None or layer.layer_count.value <= 3
    return layer.function == "device_layer" and cat in ("2d_semiconductor", "2d_conductor") and thin and layer.material != "graphite"


class FlakeImagesPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.current: str | None = None
        self._ctx: dict[str, tuple] = {}            # uid → (image array, TraceContext)
        self._skip: dict[str, int] = {}             # uid → candidate shown by "Next candidate"
        self._viewers = []
        self.trace_dialog: TraceDialog | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)
        up = Card("Flake Images", "Upload one image per flake, then trace the borders: Trace all (whole flakes, no click) or "
                                  "Trace flake… for one flake in its own window.")
        self.objective = ComboBox()
        for o in OBJECTIVES:
            fov = self.app.config.field_of_view_um(o)
            self.objective.addItem(f"{o}  ({fov[0]:g} µm wide)" if fov else o, o)
        self.objective.addItem("Other (enter µm/px)", "")
        self.objective.setCurrentIndex(self.objective.findData("100x"))
        self.objective.setToolTip("Objective of the images you upload (sets µm per pixel from the field of view)")
        up.layout.addLayout(row("Objective", self.objective, width=70, stretch=False))
        up.layout.addLayout(buttons_row(button("Upload all…", self.upload_all, "primary",
                                               "Ask for an image of every flake that has none yet, bottom to top", "image"),
                                        button("Trace all", self.trace_all, None,
                                               "Trace every flake that has an image (whole flake, no click). Borders you traced by "
                                               "clicking, drew or edited are kept.")))
        v.addWidget(up)
        self.list = clickable_rows(PickTree(), boxes=False)   # its ticks show / hide the flakes
        self.list.setItemDelegate(CheckDelegate(self.list))
        self.list.setHeaderLabels(["Layer (top → bottom)", "Image"])
        self.list.setRootIsDecorated(False)
        self.list.header().setStretchLastSection(False)
        self.list.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.list.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMinimumHeight(220)
        self.list.setToolTip("Tick = shown in the 3D stack. Shift- or Ctrl-click adds or removes a row (any combination), then 'Only selected'. "
                             "Double-click a flake to trace it.")
        self.list.currentItemChanged.connect(lambda it, _p: self._select(it.data(0, Qt.ItemDataRole.UserRole) if it else None))
        self.list.itemChanged.connect(self._checked)
        self.list.itemDoubleClicked.connect(lambda it, _c: self.trace(it.data(0, Qt.ItemDataRole.UserRole)))
        v.addWidget(self.list, 2)
        v.addLayout(buttons_row(button("Show all", self.show_all, None, "Show every layer in the 3D stack"),
                                button("Only selected", self.show_selected, None, "Show only the layers selected in the list"),
                                button("Only flakes", self.show_flakes, None, "Hide the substrate and the metal layers")))
        self.details = Card("Selected Flake")
        v.addWidget(self.details)
        v.addStretch(1)
        outer.addWidget(scroll(content), 1)
        modernize(self)

    @property
    def view3d(self):
        return self.app.view3d

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        spec = self.app.spec
        d = self.trace_dialog
        if d is not None and d.isVisible() and (d.spec is not spec or spec.layer(d.uid) is None):
            d.close()                                   # another device was opened, or the layer was removed
        self.list.blockSignals(True)
        self.list.clear()
        if spec is None:
            self.list.blockSignals(False)
            self._show_details(None)
            return
        uids = {l.uid for l in spec.layers}
        self.view3d.hidden &= uids
        for layer in reversed(spec.layers):
            it = QtWidgets.QTreeWidgetItem(self.list, [layer.label, self._status(layer)])
            it.setData(0, Qt.ItemDataRole.UserRole, layer.uid)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Unchecked if layer.uid in self.view3d.hidden else Qt.CheckState.Checked)
            if not is_flake(self.app.config, layer):
                it.setForeground(1, QtGui.QBrush(self.list.palette().placeholderText().color()))
        self.list.blockSignals(False)
        if self.current not in uids:
            first = next((l.uid for l in spec.layers if is_flake(self.app.config, l) and not (l.image or {}).get("path")), None)
            self.current = first or next((l.uid for l in spec.layers if is_flake(self.app.config, l)), None)
        self._select_row(self.current)

    def _status(self, layer) -> str:
        if not is_flake(self.app.config, layer):
            return "—"
        img = layer.image or {}
        if not img.get("path"):
            return "needs image"
        how = (img.get("trace") or {}).get("how")
        if layer.outline and how:
            return STATUS_TEXT.get(how, "traced")
        return "not traced" if not layer.outline else "border set"

    def refresh_3d(self) -> None:
        self.app.refresh_structure()
        self.view3d.set_highlight(self.current)

    def _select_row(self, uid: str | None) -> None:
        for i in range(self.list.topLevelItemCount()):
            it = self.list.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == uid:
                self.list.setCurrentItem(it)
                break
        self._select(uid)

    def select(self, uid: str) -> None:
        self._select_row(uid)

    def _select(self, uid: str | None) -> None:
        self.current = uid
        self.view3d.set_highlight(uid)
        layer = self.app.spec.layer(uid) if (self.app.spec and uid) else None
        self._show_details(layer)

    def _show_details(self, layer) -> None:
        body = self.details.layout
        clear_layout(body)
        if layer is None:
            body.addWidget(label("No layer selected", "faint"))
            return
        body.addWidget(label(layer.label, "strong"))
        if not is_flake(self.app.config, layer):
            body.addWidget(label("Not a flake (substrate or metal): no image needed. Its shape comes from the design.", "hint", wrap=True))
            modernize(self.details)
            return
        img = layer.image or {}
        name = Path(img.get("path", "")).name
        body.addWidget(label((name + " · " + self._status(layer)) if name else "No image yet", "muted", wrap=True))
        upload = button("Upload image…" if not name else "Replace…", lambda: self.upload(layer.uid), "primary" if not name else None,
                        "Load this flake's optical image", "image")
        if not name:
            body.addLayout(buttons_row(upload))
        else:
            body.addLayout(buttons_row(upload, button("View", lambda: self.view(layer.uid), None, "Open the image in its own window"),
                                       button("Trace flake…", lambda: self.trace(layer.uid), "primary" if not layer.outline else None,
                                              "Open the tracing window for this flake")))
            if layer.outline:
                xs = [p[0] for p in layer.outline]
                ys = [p[1] for p in layer.outline]
                body.addWidget(label(f"Border: {len(layer.outline)} points · {max(xs) - min(xs):.1f} × {max(ys) - min(ys):.1f} µm (x × y)",
                                     "muted", wrap=True))
                body.addLayout(buttons_row(button("Clear border", lambda: self.clear(layer.uid))))
            scale = NumberField(img.get("um_per_px"), "µm/px", 88, 6)
            scale.valueEdited.connect(lambda v: self._image_param(layer.uid, "um_per_px", v))
            body.addLayout(row("Scale", scale, label("µm / px", "muted"), width=70))
            rot = NumberField(img.get("angle_deg", 0.0), "0", 88)
            rot.valueEdited.connect(lambda v: self._image_param(layer.uid, "angle_deg", v or 0.0))
            body.addLayout(row("Rotation", rot, label("°", "muted"), width=70))
            flip = CheckBox("Mirror (image taken from the other side)")
            flip.setChecked(img.get("mirror", False))
            flip.toggled.connect(lambda v: self._image_param(layer.uid, "mirror", bool(v)))
            body.addLayout(row("", flip, width=70))
        modernize(self.details)

    # ------------------------------------------------------------------ check boxes / visibility
    def _checked(self, it, col) -> None:
        if col != 0:
            return
        uid = it.data(0, Qt.ItemDataRole.UserRole)
        hidden = set(self.view3d.hidden)
        if it.checkState(0) == Qt.CheckState.Checked:
            hidden.discard(uid)
        else:
            hidden.add(uid)
        self.view3d.set_hidden(hidden)

    def sync_checks(self) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.topLevelItemCount()):
            it = self.list.topLevelItem(i)
            uid = it.data(0, Qt.ItemDataRole.UserRole)
            it.setCheckState(0, Qt.CheckState.Unchecked if uid in self.view3d.hidden else Qt.CheckState.Checked)
        self.list.blockSignals(False)

    def show_all(self) -> None:
        self.view3d.set_hidden(set())
        self.sync_checks()

    def show_selected(self) -> None:
        uids = [it.data(0, Qt.ItemDataRole.UserRole) for it in self.list.selectedItems()] or ([self.current] if self.current else [])
        self.view3d.show_only(uids)
        self.sync_checks()

    def show_flakes(self) -> None:
        self.view3d.show_only([l.uid for l in self.app.spec.layers if is_flake(self.app.config, l)] if self.app.spec else [])
        self.sync_checks()

    # ------------------------------------------------------------------ images
    def _scale_for(self, width_px: int) -> float | None:
        obj = self.objective.currentData()
        if obj:
            s = self.app.config.um_per_px(obj, width_px)
            if s:
                return s
        return self.app._ask_scale(width_px)

    def upload(self, uid: str, path: str | None = None) -> bool:
        spec = self.app.spec
        layer = spec.layer(uid) if spec else None
        if layer is None:
            return False
        if path is None:
            path = open_file(self, f"Image of {layer.label}", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not path:
            return False
        try:
            arr = imaging.load_image(path)
        except Exception as exc:
            self.app.warn("Image", str(exc))
            return False
        scale = self._scale_for(arr.shape[1])
        if scale is None:
            return False
        dest = self.app.device_dir() / "images" / f"{uid}_{Path(path).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)
        h, w = arr.shape[:2]
        layer.image = {"path": str(dest), "um_per_px": scale, "angle_deg": 0.0, "x": -w * scale / 2, "y": h * scale / 2,
                       "opacity": 0.6, "visible": True, "mirror": False, "objective": self.objective.currentData() or None}
        layer.outline = None
        self.app.layer_images[uid] = arr
        self._ctx.pop(uid, None)
        self._skip[uid] = 0
        self.current = uid
        self.app.spec_changed("image")
        self.refresh()
        self.refresh_3d()
        if self.trace_dialog is not None and self.trace_dialog.uid == uid:
            self.trace_dialog.redraw(fit=True)
        self.app.status(f"{layer.label}: image loaded. Trace flake… or Trace all to find its border.")
        return True

    def upload_all(self) -> None:
        spec = self.app.spec
        if spec is None:
            return
        todo = [l for l in spec.layers if is_flake(self.app.config, l) and not (l.image or {}).get("path")]
        if not todo:
            self.app.status("Every flake has an image. Use Replace… on a flake to change one.")
            return
        for k, layer in enumerate(todo, 1):
            self._select_row(layer.uid)
            path = open_file(self, f"Image {k} of {len(todo)}: {layer.label} (Cancel stops)",
                             "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
            if not path:
                break
            self.upload(layer.uid, path)

    def view(self, uid: str) -> None:
        layer = self.app.spec.layer(uid)
        arr = self.app.layer_image(uid)
        if arr is None:
            self.app.status("No image for this flake yet")
            return
        from .viewer import ImageViewerDialog
        img = layer.image or {}
        dlg = ImageViewerDialog(arr, f"{layer.label} — {Path(img.get('path', '')).name}",
                                f"{arr.shape[1]} × {arr.shape[0]} px · {img.get('um_per_px', 0):.5g} µm/px", img.get("um_per_px"), self.app)
        self._viewers = [v for v in self._viewers if v.isVisible()] + [dlg]
        dlg.show()

    def trace(self, uid: str | None) -> None:
        """Open (or point) the tracing window at this flake; trace it at once if it has no border."""
        layer = self.app.spec.layer(uid) if (self.app.spec and uid) else None
        if layer is None or not is_flake(self.app.config, layer):
            return
        if self.app.layer_image(uid) is None:
            self.app.status(f"{layer.label}: upload an image first")
            return
        self._select_row(uid)
        if self.trace_dialog is None:
            self.trace_dialog = TraceDialog(self)
        self.trace_dialog.set_layer(uid)
        self.trace_dialog.show()
        self.trace_dialog.raise_()
        if not layer.outline:
            self.auto_trace(uid, mode=self.trace_dialog.mode.currentData())

    def trace_all(self) -> None:
        """Whole-flake trace of every flake with an image, without a click; hand-made borders stay."""
        spec = self.app.spec
        if spec is None:
            return
        done, kept, failed = [], [], []
        flakes = [l for l in spec.layers if is_flake(self.app.config, l) and (l.image or {}).get("path")]
        if not flakes:
            self.app.status("No flake has an image yet: Upload all… first.")
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for layer in flakes:
                if layer.outline and ((layer.image or {}).get("trace") or {}).get("how") in HAND_MADE:
                    kept.append(layer.label)
                    continue
                if self.app.layer_image(layer.uid) is None:
                    failed.append(f"{layer.label} (image file not found)")
                    continue
                ok = self.auto_trace(layer.uid, mode="flake", refresh=False)
                (done if ok else failed).append(layer.label)
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.app.spec_changed("outline")
        self.app.refresh_layout_canvas()
        if self.app.stack_panel.current:
            self.app.stack_panel.refresh(self.app.stack_panel.current)
        self.refresh()
        self.refresh_3d()
        if self.trace_dialog is not None and self.trace_dialog.isVisible():
            self.trace_dialog.redraw()
        msg = f"Traced {len(done)} flake{'s' * (len(done) != 1)}"
        if kept:
            msg += f"; kept your border for {', '.join(kept)}"
        if failed:
            msg += f"; no flake found in {', '.join(failed)} (Trace flake… and click it)"
        self.app.status(msg + ". Check each in Trace flake…; Next candidate if a wrong flake was picked.")

    # ------------------------------------------------------------------ tracing
    def _context(self, uid: str):
        arr = self.app.layer_image(uid)
        if arr is None:
            return None, None
        got = self._ctx.get(uid)
        if got is None or got[0] is not arr:
            got = self._ctx[uid] = (arr, imaging.prepare_trace(arr))
        return got

    def _to_design(self, layer, poly_px) -> list[list[float]]:
        img = layer.image
        t = image_transform(img["um_per_px"], img.get("angle_deg", 0.0), img.get("x", 0.0), img.get("y", 0.0), img.get("mirror", False))
        return [[q.x(), q.y()] for q in (t.map(QtCore.QPointF(x, y)) for x, y in poly_px)]

    def _set_outline(self, layer, pts, how: str, mode: str | None = None, seed=None) -> None:
        img = layer.image or {}
        if how in ("auto", "click") and not img.get("placed"):
            # stack the flakes on the design origin until they are placed in the Flake layout tab
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            pts = [[x - cx, y - cy] for x, y in pts]
            img["x"] = img.get("x", 0.0) - cx
            img["y"] = img.get("y", 0.0) - cy
        layer.outline = pts
        img["trace"] = {"how": how, "mode": mode, "seed_px": [round(v, 1) for v in seed] if seed else None}
        layer.image = img

    def auto_trace(self, uid: str, mode: str = "flake", nxt: bool = False, refresh: bool = True) -> bool:
        layer = self.app.spec.layer(uid)
        arr, ctx = self._context(uid)
        if arr is None:
            return False
        faint = prefers_faint(self.app.config, layer)
        self._skip[uid] = self._skip.get(uid, 0) + 1 if nxt else 0
        try:
            poly, seed = imaging.auto_trace(arr, "faint" if faint else "any", mode, ctx, self._skip[uid])
        except ValueError as exc:
            if refresh:
                self.app.status(f"{layer.label}: {exc} — click the flake with the wand")
            return False
        self._set_outline(layer, self._to_design(layer, poly), "auto", mode, seed)
        if refresh:
            self._done(layer, f"{layer.label}: border traced automatically ({len(poly)} points)"
                       + (f", candidate {self._skip[uid] + 1}" if nxt else "") + ". Wrong flake? Next candidate, or click it with the wand.")
        return True

    def wand(self, uid: str, px: float, py: float, mode: str) -> bool:
        layer = self.app.spec.layer(uid) if self.app.spec else None
        if layer is None:
            return False
        arr, ctx = self._context(uid)
        try:
            poly = imaging.detect_flake(arr, (px, py), mode=mode, context=ctx)
        except ValueError as exc:
            self.app.status(str(exc))
            return False
        self._set_outline(layer, self._to_design(layer, poly), "click", mode, (px, py))
        self._done(layer, f"{layer.label}: border traced ({len(poly)} points). Drag the points to correct it.")
        return True

    def edited(self, uid: str, pts: list) -> None:
        layer = self.app.spec.layer(uid)
        if layer is not None:
            layer.outline = [list(p) for p in pts]
            if layer.image:
                layer.image.setdefault("trace", {})["how"] = "edited"
            self.app.spec_changed("outline")
            self.app.refresh_layout_canvas()
            self.refresh_3d()
            self._update_row(layer)

    def drawn(self, uid: str, pts: list) -> None:
        layer = self.app.spec.layer(uid) if self.app.spec else None
        if layer is None or not is_flake(self.app.config, layer):
            return
        layer.outline = [list(p) for p in pts]
        if layer.image is not None:
            layer.image["trace"] = {"how": "drawn"}
        self._done(layer, f"{layer.label}: border drawn ({len(pts)} points)")

    def clear(self, uid: str) -> None:
        layer = self.app.spec.layer(uid)
        layer.outline = None
        if layer.image:
            layer.image.pop("trace", None)
        self._done(layer, f"{layer.label}: border cleared")

    def _done(self, layer, message: str) -> None:
        self.app.spec_changed("outline")
        if self.app.stack_panel.current == layer.uid:
            self.app.stack_panel.refresh(layer.uid)
        self.app.refresh_layout_canvas()
        self._update_row(layer)
        if self.current == layer.uid:
            self._show_details(layer)
        self.refresh_3d()
        if self.trace_dialog is not None and self.trace_dialog.uid == layer.uid and self.trace_dialog.isVisible():
            self.trace_dialog.redraw()
        self.app.status(message)

    def _update_row(self, layer) -> None:
        for i in range(self.list.topLevelItemCount()):
            it = self.list.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) == layer.uid:
                it.setText(1, self._status(layer))

    def _image_param(self, uid: str, key: str, value) -> None:
        self.app.set_image_param(uid, key, value)
        self.refresh_3d()
        if self.trace_dialog is not None and self.trace_dialog.uid == uid and self.trace_dialog.isVisible():
            self.trace_dialog.redraw()


class TraceDialog(Dialog):
    """One flake's image with its border, in its own window: auto trace / next candidate, the wand,
    drawing by hand, dragging points. Every change goes straight into the device."""

    def __init__(self, panel: FlakeImagesPanel):
        super().__init__(panel.app)
        self.panel = panel
        self.app = panel.app
        self.uid: str | None = None
        self.spec = None
        self.setModal(False)
        self.resize(1100, 820)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        self.canvas = Canvas()
        self.canvas.rotate_handles = True                     # drag the round handle above the border to turn it
        self.canvas.scale_handles = True                      # square handles on its box: stretch / scale it
        self.canvas.trackpad_gestures = True                  # as on the EBL canvas: two-finger scroll moves, pinch zooms, twist turns
        self.canvas.empty_text = "No image for this flake"
        self.canvas.fit_on_resize = True
        self.tools = {}
        group = QtWidgets.QButtonGroup(self)
        for tool, icon, tip in (("pointer", "pointer", "Edit the border: click a point to select it (Shift-click adds more), drag to move the "
                                 "selected points, double-click an edge to add a point; drag elsewhere moves the view"),
                                ("addpoint", "addPoint", "Add a point: click on the border line (keep the button down to drag it into place)"),
                                ("select", "selectPoints", "Select points: drag a box over them (Shift adds to the selection); drag a selected "
                                 "point to move them all; Delete / Backspace or the trash button removes them"),
                                ("wand", "mask", "Trace by clicking the flake"),
                                ("draw", "annotate", "Draw the border: click its corners, double-click to close"),
                                ("measure", "measure", "Measure a length: click the start, then click the end (double-click or Esc clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self.canvas.set_tool(t))
            bar.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        self.trash = tool_strip_button("trash", "Delete the selected points (Delete / Backspace); the points on either side join up")
        self.trash.clicked.connect(self.canvas.delete_selected_vertices)
        self.trash.setEnabled(False)
        bar.addWidget(self.trash)
        bar.addSpacing(6)
        for icon, tip, fn in (("fit", "Fit", self.canvas.fit), ("zoomIn", "Zoom in", lambda: self.canvas.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: self.canvas.zoom_by(0.8)),
                              ("reset", "View back to 0° (after a two-finger twist)", lambda: self.canvas.set_view_rotation(0.0))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(10)
        bar.addWidget(label("Trace", "muted"))
        self.mode = ComboBox()
        for k, v in imaging.TRACE_MODES.items():
            self.mode.addItem(v, k)
        self.mode.setToolTip("Whole flake: everything that is not substrate (all its thicknesses). This colour only: the clicked "
                             "region alone (a monolayer attached to a thick piece).")
        self.mode.setMinimumWidth(150)
        bar.addWidget(self.mode)
        bar.addWidget(button("Auto trace", lambda: self.uid and panel.auto_trace(self.uid, self.mode.currentData()), None,
                             "Find the flake and trace it (no click)"))
        bar.addWidget(button("Next candidate", lambda: self.uid and panel.auto_trace(self.uid, self.mode.currentData(), nxt=True), None,
                             "Another flake in the image, if the first guess is not the one"))
        bar.addWidget(button("Clear", lambda: self.uid and panel.clear(self.uid)))
        bar.addStretch(1)
        self.info = coord_label("x -00000.00 µm   y -00000.00 µm")
        bar.addWidget(self.info)
        self.body.addLayout(bar)
        self.body.addSpacing(4)
        self.body.addWidget(self.canvas, 1)
        foot = QtWidgets.QHBoxLayout()
        self.hint = label("", "hint", wrap=True)
        foot.addWidget(self.hint, 1)
        foot.addWidget(button("Previous flake", lambda: self._step(-1), None, "The flake below in the stack"))
        foot.addWidget(button("Next flake", lambda: self._step(1), None, "The flake above in the stack"))
        foot.addWidget(button("Close", self.close))
        self.body.addSpacing(4)
        self.body.addLayout(foot)
        self.canvas.wandClicked.connect(self._wand)
        self.canvas.polygonEdited.connect(lambda key, pts: panel.edited(key.split(":", 1)[1], pts))
        self.canvas.polygonDrawn.connect(self._drawn)
        self.canvas.verticesSelected.connect(self._points_selected)
        self.canvas.message.connect(self.hint.setText)
        self.canvas.cursorMoved.connect(lambda x, y: self.info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        modernize(self)

    def set_layer(self, uid: str) -> None:
        self.uid = uid
        self.spec = self.app.spec
        layer = self.app.spec.layer(uid)
        img = layer.image or {}
        cur = (img.get("trace") or {}).get("mode") or ("colour" if prefers_faint(self.app.config, layer) else "flake")
        self.mode.setCurrentIndex(self.mode.findData(cur))
        self.setWindowTitle(f"Trace flake — {layer.label}")
        self.tools["pointer"].setChecked(True)
        self.redraw(fit=True)

    def redraw(self, fit: bool = False) -> None:
        c = self.canvas
        c.clear_all()
        layer = self.app.spec.layer(self.uid) if (self.app.spec and self.uid) else None
        if layer is None:
            return
        img = layer.image or {}
        arr = self.app.layer_image(layer.uid)
        if arr is not None and img.get("um_per_px"):
            c.set_image(layer.uid, arr, (img["um_per_px"], img.get("angle_deg", 0.0), img.get("x", 0.0), img.get("y", 0.0),
                                         img.get("mirror", False)), 1.0, 0)
            c.active_image = layer.uid
        if layer.outline:
            c.set_polygon(f"outline:{layer.uid}", layer.outline, layer_color(layer, self.app.config), True, 100, 0.12)
            c.set_active_polygon(f"outline:{layer.uid}")
        how = (img.get("trace") or {}).get("how")
        self.hint.setText((f"Border: {STATUS_TEXT.get(how, 'set')}. " if layer.outline else "No border yet. ")
                          + "Wrong flake? Next candidate, or pick the wand and click it. Drag points to correct the border; "
                          "add points on the border or box-select several to move or delete them.")
        if fit:
            c.fit()
            QtCore.QTimer.singleShot(0, c.fit)

    def _points_selected(self, n: int) -> None:
        self.trash.setEnabled(n > 0)
        self.trash.setToolTip(f"Delete the {n} selected point{'s' if n != 1 else ''} (Delete / Backspace)" if n else
                              "Delete the selected points (select them first)")

    def _wand(self, key: str, px: float, py: float) -> None:
        if self.panel.wand(key, px, py, self.mode.currentData()):
            self.tools["pointer"].setChecked(True)

    def _drawn(self, pts: list) -> None:
        if self.uid:
            self.panel.drawn(self.uid, pts)
            self.tools["pointer"].setChecked(True)

    def _step(self, d: int) -> None:
        spec = self.app.spec
        flakes = [l.uid for l in spec.layers if is_flake(self.app.config, l) and (l.image or {}).get("path")]
        if not flakes:
            return
        i = flakes.index(self.uid) if self.uid in flakes else -1
        self.panel.trace(flakes[(i + d) % len(flakes)])
