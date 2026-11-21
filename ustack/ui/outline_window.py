"""Stage 2 › Draw outline: a flake's outline on the chip, edited in its own window over an aligned
image of the step (as the Flake images trace window does for the flake's own image).

The window shows one aligned image (the one on the EBL canvas to start with) in chip µm, the other
flakes' outlines faintly, and the chosen flake's outline, editable: drag the line to move it, drag
points, add points on the line, box-select points to move or delete them, draw it anew, trace it
with the wand on the image, or start from the border traced in Flake images. Every change goes
straight into the step (and the EBL canvas)."""

from __future__ import annotations

from .. import ebl_flow as ef
from .. import geometry as geo
from .. import imaging
from ..scene3d import layer_color
from .canvas import Canvas
from .style import ComboBox, Dialog, QtCore, QtWidgets, button, label, tool_strip_button
from .widgets import coord_label, modernize

Qt = QtCore.Qt


class OutlineDialog(Dialog):
    def __init__(self, panel):
        super().__init__(panel.app)
        self.panel = panel
        self.app = panel.app
        self.uid: str | None = None
        self.obj: str | None = None
        self._ctx: dict[str, object] = {}                 # objective → imaging.TraceContext (wand)
        self.setModal(False)
        self.resize(1100, 820)
        self.canvas = Canvas()
        self.canvas.empty_text = "Load and align an image of this step first"
        self.canvas.fit_on_resize = True
        self.canvas.drag_outlines = True
        self.canvas.rotate_handles = True                     # drag the round handle above the outline to turn it
        self.canvas.scale_handles = True                      # square handles on its box: stretch / scale it
        self.canvas.trackpad_gestures = True                  # two-finger scroll moves, pinch zooms (no twist here)
        self.canvas.allow_twist = False
        self.canvas.box_select_outlines = True                # drag on nothing: a box chooses every outline inside
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        self.tools = {}
        group = QtWidgets.QButtonGroup(self)
        for tool, icon, tip in (("pointer", "pointer", "Drag the outline's line to move it, drag a point to reshape it (Shift-click "
                                 "selects several); drag on nothing: a box chooses every outline inside (moved / deleted together); two fingers move the view"),
                                ("addpoint", "addPoint", "Add a point: click on the outline (keep the button down to drag it)"),
                                ("select", "selectPoints", "Select points: drag a box (Shift adds); drag one to move them all; "
                                 "Delete or the trash removes them"),
                                ("wand", "mask", "Trace the flake on this image: click it"),
                                ("draw", "annotate", "Draw the outline anew: click its corners, double-click to close"),
                                ("measure", "measure", "Measure a length: click the start, then the end (double-click clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self.canvas.set_tool(t))
            bar.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        self.trash = tool_strip_button("trash", "Delete the selected points (Delete / Backspace)")
        self.trash.clicked.connect(self.canvas.delete_selected_vertices)
        self.trash.setEnabled(False)
        bar.addWidget(self.trash)
        bar.addSpacing(6)
        for icon, tip, fn in (("fit", "Fit", self.canvas.fit), ("zoomIn", "Zoom in", lambda: self.canvas.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: self.canvas.zoom_by(0.8))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(10)
        bar.addWidget(label("Image", "muted"))
        self.image_combo = ComboBox()
        self.image_combo.setMinimumWidth(90)
        self.image_combo.activated.connect(lambda _i: self._image_changed())
        bar.addWidget(self.image_combo)
        bar.addWidget(label("Wand", "muted"))
        self.mode = ComboBox()
        for k, v in imaging.TRACE_MODES.items():
            self.mode.addItem(v, k)
        self.mode.setMinimumWidth(130)
        bar.addWidget(self.mode)
        self.from_trace = button("Import border", None, None, "Bring in a border: this flake's (or another flake's) border traced in "
                                 "Flake images, at its own shape and size, or this flake's outline at another step; then drag it "
                                 "into place and turn it with the round handle")
        self.import_menu = QtWidgets.QMenu(self.from_trace)
        self.from_trace.clicked.connect(self._show_import_menu)
        bar.addWidget(self.from_trace)
        bar.addWidget(button("Clear", self.clear))
        bar.addStretch(1)
        self.info = coord_label("x -00000.00 µm   y -00000.00 µm")
        bar.addWidget(self.info)
        self.body.addLayout(bar)
        self.body.addSpacing(4)
        self.body.addWidget(self.canvas, 1)
        foot = QtWidgets.QHBoxLayout()
        self.hint = label("", "hint", wrap=True)
        foot.addWidget(self.hint, 1)
        foot.addWidget(button("Close", self.close))
        self.body.addSpacing(4)
        self.body.addLayout(foot)
        c = self.canvas
        c.polygonEdited.connect(lambda key, pts: self._changed(pts, "edited"))
        c.polygonDrawn.connect(self._drawn)
        c.wandClicked.connect(self._wand)
        c.verticesSelected.connect(lambda n: self.trash.setEnabled(n > 0))
        c.message.connect(self.hint.setText)
        c.cursorMoved.connect(lambda x, y: self.info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        modernize(self)
        self.from_trace.setStyleSheet("min-width: 126px;")     # the style sheet sets button widths on show: the label is never cut

    # ------------------------------------------------------------------ state
    @property
    def step(self):
        return self.panel.step

    def set_flake(self, uid: str) -> None:
        self.uid = uid
        st = self.step
        layer = self.app.spec.layer(uid)
        self.setWindowTitle(f"Outline on the chip — {layer.label}")
        aligned = [o for o in ef.OBJECTIVE_KEYS if o in st.images and o in st.transforms]
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for o in aligned:
            self.image_combo.addItem(o, o)
        self.obj = st.show_image if st.show_image in aligned else (aligned[-1] if aligned else None)
        if self.obj:
            self.image_combo.setCurrentIndex(self.image_combo.findData(self.obj))
        self.image_combo.blockSignals(False)
        self.from_trace.setEnabled(any(l.outline for l in self.app.spec.layers)
                                   or any(s.outlines.get(uid) for k, s in self.panel.flow.steps.items() if k != st.key))
        self.tools["pointer" if st.outlines.get(uid) else "draw"].setChecked(True)
        self.redraw(fit=True)

    def _image_changed(self) -> None:
        self.obj = self.image_combo.currentData()
        self.redraw(fit=True)

    # ------------------------------------------------------------------ drawing
    def redraw(self, fit: bool = False) -> None:
        c, st = self.canvas, self.step
        if st is None or self.uid is None or self.app.spec is None:
            return
        c.clear_all()
        img = self.panel._image(self.obj) if self.obj else None
        if img is not None:
            c.set_image(f"step:{self.obj}", img, self.panel._placement(self.obj, img), 1.0, 0)
            c.active_image = f"step:{self.obj}"
        for layer in self.app.spec.layers:
            if layer.uid == self.uid:
                continue
            polys = st.outlines.get(layer.uid) or []
            if polys:
                c.add_overlay_polys(polys, layer_color(layer, self.app.config), 0.04, z=40, width=0.8)
        layer = self.app.spec.layer(self.uid)
        mine = st.outlines.get(self.uid) or []
        key = f"outline:{self.uid}#0"
        if mine:
            c.set_polygon(key, mine[0], layer_color(layer, self.app.config), True, 100, 0.12, width=2.0)
            c.set_active_polygon(key)
        how = ef.flake_state(self.app.spec, st, st.key, self.uid)[0] if st.located.get(self.uid) else None
        self.hint.setText((f"Outline: {how}. " if mine and how else "No outline yet: draw it, trace it with the wand, or Import border. ")
                          + "Drag the line to move it, the round handle above it to turn it (Shift: 15° steps); drag points to "
                          "reshape; add points on the line; box-select points to move or delete several.")
        if fit:
            if mine:
                xs, ys = [p[0] for p in mine[0]], [p[1] for p in mine[0]]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                m = max(w, h, 10.0)
                rect = QtCore.QRectF(min(xs) - m, min(ys) - m, w + 2 * m, h + 2 * m)
                c.fit_rect(rect)
                c._zoomed = True                          # keep this view when the window is shown / resized
                QtCore.QTimer.singleShot(0, lambda r=rect: c.fit_rect(r))
            else:
                c.fit()

    # ------------------------------------------------------------------ edits
    def _changed(self, pts: list, how: str) -> None:
        if self.uid is None or len(pts) < 3:
            return
        self.panel.set_outline(self.uid, [tuple(map(float, p)) for p in pts], how)
        self.redraw()

    def _drawn(self, pts: list) -> None:
        self._changed(pts, "drawn")
        self.tools["pointer"].setChecked(True)

    def _wand(self, key: str, px: float, py: float) -> None:
        if not key.startswith("step:"):
            return
        obj = key.split(":", 1)[1]
        img = self.panel._image(obj)
        item = self.canvas.images.get(key)
        if img is None or item is None:
            return
        ctx = self._ctx.get(obj)
        if ctx is None:
            ctx = self._ctx[obj] = imaging.prepare_trace(img)
        try:
            poly = imaging.detect_flake(img, (px, py), mode=self.mode.currentData(), context=ctx)
        except ValueError as exc:
            self.hint.setText(str(exc))
            return
        pts = [(q.x(), q.y()) for q in (item.px_to_scene(x, y) for x, y in poly)]
        self._changed(pts, "traced")
        self.tools["pointer"].setChecked(True)

    def _show_import_menu(self) -> None:
        self._fill_import_menu()
        self.import_menu.exec(self.from_trace.mapToGlobal(QtCore.QPoint(0, self.from_trace.height())))

    def _fill_import_menu(self) -> None:
        """Import border ▾: every border traced in Flake images (this flake first), and this flake's
        outline at the other steps."""
        m, spec, uid = self.import_menu, self.app.spec, self.uid
        m.clear()
        if uid is None:
            return
        traced = [l for l in spec.layers if l.outline]
        traced.sort(key=lambda l: l.uid != uid)
        for l in traced:
            act = m.addAction(("This flake's border" if l.uid == uid else l.label) + " — traced in Flake images")
            act.triggered.connect(lambda _=False, u=l.uid: self.start_from_trace(u))
        steps = [(k, s) for k, s in self.panel.flow.steps.items() if k != self.step.key and s.outlines.get(uid)]
        if steps:
            m.addSeparator()
            for k, s in steps:
                act = m.addAction(f"This flake's outline at '{ef.STEPS[k].name}' ({(s.located.get(uid) or {}).get('how') or 'set'})")
                act.triggered.connect(lambda _=False, key=k: self.import_from_step(key))
        if m.isEmpty():
            m.addAction("Nothing traced yet (Flake images)").setEnabled(False)

    def import_from_step(self, key: str) -> None:
        """This flake's outline at another step, where it is on the chip (same chip frame)."""
        polys = self.panel.flow.step(key).outlines.get(self.uid)
        if polys:
            self._changed([tuple(map(float, q)) for q in polys[0]], "placed")
            self.redraw(fit=True)
            self.tools["pointer"].setChecked(True)

    def start_from_trace(self, source: str | None = None) -> None:
        """A border traced in Flake images (this flake's, or ``source``'s), its middle where the
        outline is now (or at the plan / the middle of the view), unturned: drag it into place and turn
        it with the round handle."""
        layer = self.app.spec.layer(source or self.uid) if (source or self.uid) else None
        if layer is None or not layer.outline:
            self.hint.setText("This flake has no traced border: trace it in Flake images first.")
            return
        st = self.step
        cur = st.outlines.get(self.uid)
        if cur:
            cx, cy = geo.centroid(cur[0])
        elif self.uid in st.plan:
            cx, cy = st.plan[self.uid]["cx"], st.plan[self.uid]["cy"]
        else:
            p = self.canvas.mapToScene(self.canvas.viewport().rect().center())
            cx, cy = p.x(), p.y()
        angle = st.plan.get(self.uid, {}).get("angle", 0.0) if layer.uid == self.uid else 0.0
        self._changed(ef.planned_outline(layer.outline, {"cx": cx, "cy": cy, "angle": angle}), "placed")
        self.redraw(fit=True)
        self.tools["pointer"].setChecked(True)

    def clear(self) -> None:
        if self.uid is not None:
            self.panel.clear_outline(self.uid)
            self.redraw()
