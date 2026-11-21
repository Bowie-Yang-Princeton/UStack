"""Stage 2 › Electrodes › Bubbles, cracks and stray flakes › Draw / edit…: the features found (or drawn) in their own
window, over an aligned image of the step.

Each feature is an outline coloured by its kind (stray flake, particle, bubble, crack / fold). Click one to select it
(the Kind box shows its kind; choose another to change it), drag its line or points to reshape it, Delete removes it; the
polygon tool draws a new one of the kind in the Kind box. Every change is the step's at once (the router keeps off them
on the next Generate), and every correction is also saved as a training example (``debris.save_lab_examples``):
a feature deleted = "not a feature", a kind changed or a feature drawn = that kind — the finder learns from them."""

from __future__ import annotations

from .. import debris
from .. import ebl_flow as ef
from ..scene3d import layer_color
from .canvas import Canvas
from .style import ComboBox, Dialog, QtCore, QtWidgets, button, label, tool_strip_button
from .widgets import coord_label, modernize

Qt = QtCore.Qt


class DebrisDialog(Dialog):
    def __init__(self, panel, key: str):
        super().__init__(panel.app)
        self.panel, self.app, self.key = panel, panel.app, key
        self.setModal(False)
        self.resize(1150, 840)
        self.setWindowTitle(f"Bubbles, cracks and stray flakes — {ef.STEPS[key].name}")
        self.canvas = c = Canvas()
        c.empty_text = "Load and align an image of this step first"
        c.fit_on_resize = True
        c.drag_outlines = True
        c.trackpad_gestures = True                        # two-finger scroll moves, pinch zooms (no twist here)
        c.allow_twist = False
        c.box_select_outlines = True                      # drag on nothing: a box chooses every outline inside
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        self.tools = {}
        group = QtWidgets.QButtonGroup(self)
        for tool, icon, tip in (("pointer", "pointer", "Click a feature's line to select it; drag the line to move it, a point to "
                                 "reshape it; drag on nothing: a box chooses every outline inside (moved / deleted together); two fingers move the view"),
                                ("draw", "annotate", "Draw a feature: click its corners, double-click to close (kind: the Kind box)"),
                                ("addpoint", "addPoint", "Add a point: click on the selected feature's line"),
                                ("measure", "measure", "Measure a length: click the start, then the end (double-click clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self.canvas.set_tool(t))
            bar.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        for icon, tip, fn in (("fit", "Fit", c.fit), ("zoomIn", "Zoom in", lambda: c.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: c.zoom_by(0.8))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(10)
        bar.addWidget(label("Kind", "muted"))
        self.kind_combo = ComboBox()
        for k in debris.KINDS:
            self.kind_combo.addItem(debris.KIND_NAMES[k], k)
        self.kind_combo.setMinimumWidth(150)
        self.kind_combo.activated.connect(lambda _i: self._kind_changed())
        bar.addWidget(self.kind_combo)
        self.delete_btn = button("Delete", self.delete_selected, None, "Remove the selected feature (it is not one)")
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
        foot.addWidget(button("Find again", self.find_again, None, "Look again (the ones drawn or changed by hand are kept)"))
        foot.addWidget(button("Close", self.close))
        self.body.addSpacing(4)
        self.body.addLayout(foot)
        c.polygonEdited.connect(self._edited)
        c.polygonDrawn.connect(self._drawn)
        c.polygonSelected.connect(self._selected)
        c.message.connect(self.hint.setText)
        c.cursorMoved.connect(lambda x, y: self.info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        self._fill_images()
        modernize(self)
        self.redraw(fit=True)

    # ------------------------------------------------------------------ state
    @property
    def step(self) -> ef.FlowStep:
        return self.panel.flow.step(self.key)

    def _fill_images(self) -> None:
        """Every aligned image of the step, the finest on top (the features were found in all of them)."""
        self.images = {o: (img, sim) for o, img, sim in self.panel._debris_images(self.key)}
        self.obj = max(self.images, key=ef.mag) if self.images else None

    def _image_for(self, polygon) -> str | None:
        """The finest image that shows a feature (to learn its features from)."""
        from .. import geometry as geo
        cx, cy = geo.centroid([tuple(q) for q in polygon])
        for o in sorted(self.images, key=ef.mag, reverse=True):
            img, sim = self.images[o]
            h, w = img.shape[:2]
            frame = [tuple(map(float, q)) for q in sim.apply([(0, 0), (w, 0), (w, h), (0, h)])]
            if geo.point_in_poly(cx, cy, frame):
                return o
        return None

    def _index(self, key: str | None) -> int | None:
        if key and key.startswith("d:"):
            i = int(key[2:])
            return i if 0 <= i < len(self.step.debris) else None
        return None

    # ------------------------------------------------------------------ drawing
    def redraw(self, fit: bool = False) -> None:
        from .ebl_flow_panel import DEBRIS_COLORS
        c, st, spec = self.canvas, self.step, self.app.spec
        active = c.active_poly
        c.clear_all()
        for z, o in enumerate(sorted(self.images, key=ef.mag)):         # coarse below, fine on top
            img, sim = self.images[o]
            c.set_image(f"debris:{o}", img, (sim.scale, sim.angle_deg, sim.tx, sim.ty, sim.mirror), 1.0, z)
        for layer in spec.layers:
            polys = st.outlines.get(layer.uid) or []
            if polys:
                c.add_overlay_polys(polys, layer_color(layer, self.app.config), 0.02, z=30, width=0.6)
        for i, d in enumerate(st.debris):
            c.set_polygon(f"d:{i}", [tuple(q) for q in d["polygon"]], DEBRIS_COLORS.get(d.get("kind"), "#90a4ae"), True, 80, 0.12,
                          width=1.6)
        if active in c.polys:
            c.set_active_polygon(active)
        self.hint.setText(f"{len(st.debris)} feature(s): {self.panel._debris_text(self.key)}. Click one to select it; Delete if it is not "
                          "one; the Kind box changes its kind; the polygon tool draws a missing one.")
        if fit:
            bp = self.panel.bondpads()
            if bp is not None:
                x, y = bp.site
                rect = QtCore.QRectF(x - 210, y - 210, 420, 420)     # the bond-pad leads' ends and inside
                c.fit_rect(rect)
                c._zoomed = True
                QtCore.QTimer.singleShot(0, lambda r=rect: c.fit_rect(r))
            else:
                c.fit()

    # ------------------------------------------------------------------ edits (each one also a training example)
    def _learn(self, d: dict, kind: str) -> None:
        feats = d.get("features")
        obj = self._image_for(d["polygon"])
        if not feats and obj:
            img, sim = self.images[obj]
            metal, stack, _r = ef.debris_inputs(self.app.spec, self.panel.flow, self.key, self.panel.bondpads())
            feats = debris.describe(img, sim, [tuple(q) for q in d["polygon"]], metal, stack)
            if feats:
                d["features"] = {k: round(float(v), 4) for k, v in feats.items()}
        if feats:
            debris.save_lab_examples([{"kind": kind, "features": feats, "source": f"{self.app.spec.name}|{self.key}|{obj}"}])

    def _commit(self) -> None:
        self.panel._store()
        self.panel._fill_step()
        self.panel.redraw()
        self.redraw()

    def _selected(self, key: str) -> None:
        i = self._index(key)
        if i is not None:
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(self.step.debris[i].get("kind"))))

    def _chosen(self) -> list[int]:
        """The features chosen: a box selection, else the one clicked (indices, last first)."""
        keys = self.canvas.multi or ({self.canvas.active_poly} if self.canvas.active_poly else set())
        return sorted((i for i in (self._index(k) for k in keys) if i is not None), reverse=True)

    def _kind_changed(self) -> None:
        kind = self.kind_combo.currentData()
        changed = False
        for i in self._chosen():                        # every chosen one takes the kind
            d = self.step.debris[i]
            if d.get("kind") != kind:
                d["kind"], d["how"] = kind, "edited"
                self._learn(d, kind)
                changed = True
        if changed:
            self._commit()

    def _edited(self, key: str, pts: list) -> None:
        i = self._index(key)
        if i is not None:
            d = self.step.debris[i]
            d["polygon"] = [[float(x), float(y)] for x, y in pts]
            d["how"] = "edited"
            self._commit()

    def _drawn(self, pts: list) -> None:
        if len(pts) >= 3:
            kind = self.kind_combo.currentData()
            d = {"kind": kind, "polygon": [[float(x), float(y)] for x, y in pts], "score": 1.0, "how": "drawn",
                 "image": self._image_for(pts) or ""}
            self.step.debris.append(d)
            self._learn(d, kind)
            self._commit()
            self.canvas.set_active_polygon(f"d:{len(self.step.debris) - 1}")
        self.tools["pointer"].setChecked(True)

    def delete_selected(self) -> None:
        idx = self._chosen()
        if not idx:
            self.hint.setText("Select a feature first: click its line, or drag a box round several.")
            return
        for i in idx:                                   # last first: the indices of the others stay right
            d = self.step.debris.pop(i)
            if d.get("how") == "auto":
                self._learn(d, "none")                  # found but not a feature
        self.canvas.set_multi([])
        self.canvas.set_active_polygon(None)
        self._commit()

    def keyPressEvent(self, e):  # noqa: N802 - Qt API
        item = self.canvas.polys.get(self.canvas.active_poly) if self.canvas.active_poly else None
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._chosen() \
                and (item is None or not item.selected_indices()):
            self.delete_selected()
            return
        super().keyPressEvent(e)

    def find_again(self) -> None:
        self.panel.find_debris()
        self.redraw()
