"""Stage 2 › Draw contact gate… (and Draw fingers… / electrodes / gate contacts): a step's pattern
drawn or adjusted by hand in its own window, over an aligned image of the step.

The window shows the image in chip µm, the flakes of the step faintly, the patterns of the other
steps (the fingers under a contact gate) for reference, and this step's shapes, editable: the
polygon tool draws a new shape (click its corners, double-click to close), the pointer moves a
shape by its line or reshapes it by its points, the round handle above it turns it, points are added
on the line or box-selected to move or delete them, and Delete shape removes the selected shape.
Every change goes straight into the step (and the EBL canvas); Check then checks the pattern as it
is now (``ebl_flow.rule_report``)."""

from __future__ import annotations

from .. import ebl_flow as ef
from ..scene3d import layer_color
from .canvas import Canvas
from .style import ComboBox, Dialog, QtCore, QtWidgets, button, label, tool_strip_button
from .widgets import coord_label, modernize

Qt = QtCore.Qt
DRAW_NAMES = {"hall_bars": "Draw fingers…", "contact_gate": "Draw contact gate…", "electrodes": "Draw electrodes…"}


class ShapeDialog(Dialog):
    def __init__(self, panel, key: str):
        super().__init__(panel.app)
        self.panel, self.app, self.key = panel, panel.app, key
        self.setModal(False)
        self.resize(1100, 820)
        self.canvas = c = Canvas()
        c.empty_text = "Load and align an image of this step first"
        c.fit_on_resize = True
        c.drag_outlines = True
        c.scale_handles = True                       # PowerPoint-style stretch / scale handles on the selected outline
        c.rotate_handles = True
        c.trackpad_gestures = True                        # two-finger scroll moves, pinch zooms (no twist here)
        c.allow_twist = False
        c.box_select_outlines = True                      # drag on nothing: a box chooses every outline inside
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        self.tools = {}
        group = QtWidgets.QButtonGroup(self)
        for tool, icon, tip in (("pointer", "pointer", "Drag a shape's line to move it, drag a point to reshape it; drag on nothing: a box "
                                 "chooses every shape inside (moved / deleted together); two fingers move the view"),
                                ("draw", "annotate", "Draw a new shape: click its corners, double-click to close"),
                                ("addpoint", "addPoint", "Add a point: click on the selected shape's line"),
                                ("select", "selectPoints", "Select points: drag a box (Shift adds); drag one to move them all"),
                                ("measure", "measure", "Measure a length: click the start, then the end (double-click clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self.canvas.set_tool(t))
            bar.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        self.trash = tool_strip_button("trash", "Delete the selected points (Delete / Backspace)")
        self.trash.clicked.connect(c.delete_selected_vertices)
        self.trash.setEnabled(False)
        bar.addWidget(self.trash)
        bar.addSpacing(6)
        for icon, tip, fn in (("fit", "Fit", c.fit), ("zoomIn", "Zoom in", lambda: c.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: c.zoom_by(0.8))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(10)
        bar.addWidget(label("Image", "muted"))
        self.image_combo = ComboBox()
        self.image_combo.setMinimumWidth(90)
        self.image_combo.activated.connect(lambda _i: self._image_changed())
        bar.addWidget(self.image_combo)
        bar.addWidget(label("Pattern", "muted"))
        self.purpose_combo = ComboBox()
        purposes = ef.STEPS[key].purposes
        if key == "electrodes":                      # only the groups this sample has
            purposes = [ef.ELECTRODE_GROUPS[g][1] for g in ef.electrode_groups(panel.app.spec)] or list(purposes)
        for purpose in purposes:
            self.purpose_combo.addItem(ef.PURPOSE_NAMES.get(purpose, purpose), purpose)
        self.purpose_combo.setMinimumWidth(170)
        self.purpose_combo.activated.connect(lambda _i: self.redraw())
        bar.addWidget(self.purpose_combo)
        self.delete_btn = button("Delete shape", self.delete_shape, None, "Remove the selected shape(s) (drag a box round several)")
        self.delete_btn.setStyleSheet("min-width: 128px;")
        bar.addWidget(self.delete_btn)
        bar.addStretch(1)
        self.info = coord_label("x -00000.00 µm   y -00000.00 µm")
        bar.addWidget(self.info)
        self.body.addLayout(bar)
        self.body.addSpacing(4)
        self.body.addWidget(c, 1)
        foot = QtWidgets.QHBoxLayout()
        self.hint = label("", "hint", wrap=True)
        foot.addWidget(self.hint, 1)
        foot.addWidget(button("Check", self._check, None, "Check the pattern as it is now"))
        foot.addWidget(button("Close", self.close))
        self.body.addSpacing(4)
        self.body.addLayout(foot)
        c.polygonEdited.connect(self._edited)
        c.polygonDrawn.connect(self._drawn)
        c.verticesSelected.connect(lambda n: self.trash.setEnabled(n > 0))
        c.message.connect(self.hint.setText)
        c.cursorMoved.connect(lambda x, y: self.info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        self.setWindowTitle(f"{DRAW_NAMES.get(key, 'Draw…').rstrip('…')} — {ef.STEPS[key].name}")
        self._fill_images()
        modernize(self)
        self.redraw(fit=True)

    # ------------------------------------------------------------------ state
    @property
    def step(self) -> ef.FlowStep:
        return self.panel.flow.step(self.key)

    @property
    def purpose(self) -> str:
        return self.purpose_combo.currentData() or ef.STEPS[self.key].purposes[0]

    def _fill_images(self) -> None:
        st = self.step
        aligned = [o for o in ef.OBJECTIVE_KEYS if o in st.images and o in st.transforms]
        self.image_combo.clear()
        for o in aligned:
            self.image_combo.addItem(o, o)
        self.obj = st.show_image if st.show_image in aligned else (aligned[-1] if aligned else None)
        if self.obj:
            self.image_combo.setCurrentIndex(self.image_combo.findData(self.obj))

    def _image_changed(self) -> None:
        self.obj = self.image_combo.currentData()
        self.redraw(fit=True)

    # ------------------------------------------------------------------ drawing
    def redraw(self, fit: bool = False) -> None:
        c, st, spec = self.canvas, self.step, self.app.spec
        active = c.active_poly
        c.clear_all()
        img = self.panel._image(self.obj, self.key) if self.obj else None
        if img is not None:
            c.set_image(f"step:{self.obj}", img, self.panel._placement(self.obj, img, self.key), 1.0, 0)
        for layer in spec.layers:
            polys = st.outlines.get(layer.uid) or []
            if polys:
                c.add_overlay_polys(polys, layer_color(layer, self.app.config), 0.03, z=30, width=0.8)
        from .ebl_flow_panel import SHAPE_COLORS
        for k, other in self.panel.flow.steps.items():                       # the other steps' patterns, for reference
            if k == self.key:
                continue
            for purpose, polys in other.shapes.items():
                c.add_overlay_polys(polys, SHAPE_COLORS.get(purpose, "#90a4ae"), 0.12, z=40, width=0.6)
        for purpose, polys in st.shapes.items():
            mine = purpose == self.purpose
            for i, poly in enumerate(polys):
                if mine:
                    c.set_polygon(f"shape:{i}", poly, SHAPE_COLORS.get(purpose, "#90a4ae"), True, 90, 0.55, width=2.2)
                else:
                    c.add_overlay_polys([poly], SHAPE_COLORS.get(purpose, "#90a4ae"), 0.2, z=45, width=0.8)
        if active in c.polys:
            c.set_active_polygon(active)
        n = len(st.shapes.get(self.purpose, []))
        self.hint.setText(f"{n} shape(s). Draw a new one with the polygon tool; drag a line to move a shape, its points to reshape it, "
                          "the round handle to turn it; Delete shape removes the selected one.")
        if fit:
            pts = [q for poly in st.shapes.get(self.purpose, []) for q in poly] or [q for poly in st.twisted for q in poly]
            if pts:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.25 + 5
                rect = QtCore.QRectF(min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m)
                c.fit_rect(rect)
                c._zoomed = True
                QtCore.QTimer.singleShot(0, lambda r=rect: c.fit_rect(r))
            else:
                c.fit()

    # ------------------------------------------------------------------ edits
    def _edited(self, key: str, pts: list) -> None:
        if key.startswith("shape:"):
            self.panel.set_shape(self.key, self.purpose, int(key.split(":")[1]), pts)
            self.redraw()

    def _drawn(self, pts: list) -> None:
        if len(pts) >= 3:
            i = self.panel.add_shape(self.key, self.purpose, pts)
            self.redraw()
            self.canvas.set_active_polygon(f"shape:{i}")
        self.tools["pointer"].setChecked(True)

    def _chosen(self) -> list[int]:
        """The shapes chosen: a box selection, else the one clicked."""
        keys = self.canvas.multi or ({self.canvas.active_poly} if self.canvas.active_poly else set())
        return sorted((int(k.split(":")[1]) for k in keys if k and k.startswith("shape:")), reverse=True)

    def delete_shape(self) -> None:
        idx = self._chosen()
        if not idx:
            self.hint.setText("Select a shape first: click its line, or drag a box round several.")
            return
        for i in idx:                                   # last first: the indices of the others stay right
            self.panel.delete_shape(self.key, self.purpose, i)
        self.canvas.set_multi([])
        self.canvas.set_active_polygon(None)
        self.redraw()

    def keyPressEvent(self, e):  # noqa: N802 - Qt API
        item = self.canvas.polys.get(self.canvas.active_poly) if self.canvas.active_poly else None
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._chosen() and (item is None or not item.selected_indices()):
            self.delete_shape()
            return
        super().keyPressEvent(e)

    def _check(self) -> None:
        self.panel._activate(self.key)
        self.panel.check()
