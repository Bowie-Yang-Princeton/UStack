"""Stage 2 › Manual alignment: an image of the step placed on the bond-pad pattern by hand, in its own window.

The canvas shows the bond-pad pattern (pads and fan-out, markers, landing square) and the chosen image, half
transparent. The image's frame is an outline with PowerPoint-like handles: drag its line to move the image, the round
handle to turn it, the square handles to scale it (an image stays a similarity: no stretch — the frame is refitted).
Auto scale (bottom) keeps the scale the microscope's field of view gives (µm/px = field width / image width) and leaves
only the turn and the position to the hand / the pairs. Pick pairs: click a feature in the image, then the same feature
in the pattern, as many pairs as wanted; Fit pairs places the image on them (least squares). Every change is the step's
alignment at once (the EBL canvas follows)."""

from __future__ import annotations

import math

import numpy as np

from .. import ebl_flow as ef
from ..alignment import Similarity, fit_similarity
from .canvas import Canvas
from .style import PALETTE, CheckBox, ComboBox, Dialog, QtCore, QtWidgets, Slider, button, label, tool_strip_button
from .widgets import coord_label, modernize

Qt = QtCore.Qt
FRAME = "frame"


class AlignDialog(Dialog):
    def __init__(self, panel, key: str):
        super().__init__(panel.app)
        self.panel, self.app, self.key = panel, panel.app, key
        self.obj: str | None = None
        self.pairs: list[dict] = []
        self.pick: str | None = None                   # "img" | "gds" while picking pairs
        self.setModal(False)
        self.resize(1150, 860)
        self.setWindowTitle(f"Manual alignment — {ef.STEPS[key].name}")
        self.canvas = c = Canvas()
        c.empty_text = "Load an image of this step first"
        c.fit_on_resize = True
        c.drag_outlines = True                         # drag the frame's line: move the image
        c.rotate_handles = True                        # the round handle: turn it
        c.scale_handles = True                         # the square handles: scale it
        c.trackpad_gestures = True
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        bar.addWidget(label("Image", "muted"))
        self.image_combo = ComboBox()
        self.image_combo.setMinimumWidth(90)
        self.image_combo.activated.connect(lambda _i: self._image_changed())
        bar.addWidget(self.image_combo)
        bar.addSpacing(8)
        self.tools = {}
        group = QtWidgets.QButtonGroup(self)
        for tool, icon, tip in (("pointer", "pointer", "Drag the image frame's line to move the image, the round handle to turn it, "
                                 "the square handles to scale it; drag elsewhere moves the view"),
                                ("measure", "measure", "Measure a length: click the start, then the end (double-click clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self._set_tool(t))
            bar.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        for icon, tip, fn in (("fit", "Fit", c.fit), ("zoomIn", "Zoom in", lambda: c.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: c.zoom_by(0.8)),
                              ("reset", "View back to 0° (after a two-finger twist)", lambda: c.set_view_rotation(0.0))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(10)
        self.pick_btn = button("Pick pairs", self.start_pairs, None, "Click a feature in the image, then the same feature in the "
                               "bond-pad pattern; repeat (two or more pairs, one with Auto scale)")
        self.pick_btn.setCheckable(True)
        bar.addWidget(self.pick_btn)
        bar.addWidget(button("Fit pairs", self.fit_pairs, None, "Place the image on the pairs picked (least squares)"))
        bar.addWidget(button("Clear pairs", self.clear_pairs))
        bar.addSpacing(10)
        bar.addWidget(label("Opacity", "muted"))
        self.opacity = Slider(10, 100, 60)
        self.opacity.setFixedWidth(90)
        self.opacity.valueChanged.connect(lambda v: self._set_opacity(v / 100))
        bar.addWidget(self.opacity)
        bar.addStretch(1)
        self.info = coord_label("x -00000.00 µm   y -00000.00 µm")
        bar.addWidget(self.info)
        self.body.addLayout(bar)
        self.body.addSpacing(4)
        self.body.addWidget(c, 1)
        self.hint = label("", "hint", wrap=True)
        self.body.addSpacing(4)
        self.body.addWidget(self.hint)
        foot = QtWidgets.QHBoxLayout()
        self.auto_scale = CheckBox("Auto scale (from the microscope's field of view)")
        self.auto_scale.setToolTip("Keep the scale the microscope's field of view gives (µm/px = field width / image width): "
                                   "only the turn and the position are set by hand or by the pairs")
        self.auto_scale.toggled.connect(lambda on: self._auto_scale_toggled(on))
        foot.addWidget(self.auto_scale)
        self.mirror = CheckBox("Image mirrored")
        self.mirror.setChecked(bool(panel.mirror.isChecked()) if getattr(panel, "mirror", None) is not None else False)
        self.mirror.toggled.connect(lambda _on: self._mirror_toggled())
        foot.addWidget(self.mirror)
        foot.addStretch(1)
        foot.addWidget(button("Close", self.close))
        self.body.addSpacing(4)
        self.body.addLayout(foot)
        c.polygonEdited.connect(self._frame_edited)
        c.pointPicked.connect(self._picked)
        c.message.connect(self.hint.setText)
        c.cursorMoved.connect(lambda x, y: self.info.setText(f"x {x:8.2f} µm   y {y:8.2f} µm"))
        self._fill_images()
        self._default_auto_scale()
        modernize(self)
        self.redraw(fit=True)

    # ------------------------------------------------------------------ state
    @property
    def step(self) -> ef.FlowStep:
        return self.panel.flow.step(self.key)

    def _img(self):
        return self.panel._image(self.obj, self.key) if self.obj else None

    def nominal_scale(self) -> float | None:
        img = self._img()
        return self.panel._nominal_scale(self.obj, img) if img is not None else None

    def transform(self) -> Similarity | None:
        img = self._img()
        if img is None:
            return None
        st = self.step
        if self.obj in st.transforms:
            return Similarity.from_json(st.transforms[self.obj])
        s, ang, tx, ty, mir = self.panel._placement(self.obj, img, self.key)
        return Similarity(s, ang, tx, ty, mir)

    def _fill_images(self) -> None:
        st = self.step
        objs = [o for o in ef.OBJECTIVE_KEYS if o in st.images and self.panel._image(o, self.key) is not None]
        self.image_combo.clear()
        for o in objs:
            self.image_combo.addItem(o + ("" if o in st.transforms else "  (not aligned)"), o)
        want = st.show_image if st.show_image in objs else next((o for o in objs if o not in st.transforms), objs[0] if objs else None)
        self.obj = want
        if want:
            self.image_combo.setCurrentIndex(self.image_combo.findData(want))

    def _default_auto_scale(self) -> None:
        """On for an image not aligned yet (its scale is then the field of view's), off for one already aligned (its
        fitted scale is kept)."""
        if getattr(self, "auto_scale", None) is None:
            return
        self.auto_scale.blockSignals(True)
        self.auto_scale.setChecked(bool(self.obj) and self.obj not in self.step.transforms and self.nominal_scale() is not None)
        self.auto_scale.blockSignals(False)

    def _image_changed(self) -> None:
        self.obj = self.image_combo.currentData()
        self._default_auto_scale()
        self.clear_pairs()
        self.redraw(fit=True)

    # ------------------------------------------------------------------ drawing
    def _frame_points(self, sim: Similarity) -> list[tuple[float, float]]:
        img = self._img()
        h, w = img.shape[:2]
        return [tuple(map(float, q)) for q in sim.apply([(0, 0), (w, 0), (w, h), (0, h)])]

    def redraw(self, fit: bool = False) -> None:
        c = self.canvas
        c.clear_all()
        bp = self.panel.bondpads()
        if bp is not None:
            for role, col, alpha in (("landing", "#c3c6ca", 0.0), ("pads", "#b0b4ba", 0.35), ("markers", "#c9a227", 0.5)):
                polys = bp.layer_polys(role, near=(role == "markers")) if role != "landing" else \
                    [p for p in bp.layer_polys("landing") if abs(_area(p)) < 1e6]
                c.add_overlay_polys(polys, col, alpha, z=60, width=1.0)
        img, sim = self._img(), self.transform()
        if img is not None and sim is not None:
            c.set_image(f"align:{self.obj}", img, (sim.scale, sim.angle_deg, sim.tx, sim.ty, sim.mirror), self.opacity.value() / 100, 1)
            c.set_polygon(FRAME, self._frame_points(sim), PALETTE["accent"], True, 90, 0.0, width=2.0)
        for i, p in enumerate(self.pairs):
            if "px" in p and sim is not None:
                q = sim.apply([p["px"]])[0]
                c.add_markers([tuple(q)], "#2e7d32", 7)
            if "um" in p:
                c.add_markers([tuple(p["um"])], "#c62828", 7)
        self._hint()
        if fit:
            if img is not None and sim is not None:            # the image, with some of the pattern round it
                pts = self._frame_points(sim)
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.35
                rect = QtCore.QRectF(min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m)
                c.fit_rect(rect)
                c._zoomed = True                                # keep this view when the window is laid out
                QtCore.QTimer.singleShot(0, lambda r=rect: c.fit_rect(r))
            elif bp is not None:
                x, y = bp.site
                c.fit_rect(QtCore.QRectF(x - 450, y - 450, 900, 900))
            else:
                c.fit()

    def _hint(self) -> None:
        sim = self.transform()
        ns = self.nominal_scale()
        state = "aligned" if self.obj in self.step.transforms else "not aligned"
        if sim is None:
            self.hint.setText("Load an image of this step first.")
            return
        text = (f"{self.obj}: {state} · {sim.scale:.5f} µm/px" + (f" (field of view: {ns:.5f})" if ns else "")
                + f" · {sim.angle_deg:.2f}°")
        if sim.rms_um is not None:
            text += f" · pairs fit ±{sim.rms_um:.2f} µm"
        if self.pick == "img":
            text += " — click a feature in the IMAGE"
        elif self.pick == "gds":
            text += " — now click the same feature in the BOND-PAD PATTERN"
        elif self.pairs:
            text += f" — {len([p for p in self.pairs if 'um' in p])} pair(s): Fit pairs"
        self.hint.setText(text)

    def _set_opacity(self, a: float) -> None:
        item = self.canvas.images.get(f"align:{self.obj}")
        if item is not None:
            item.setOpacity(a)

    def _set_tool(self, tool: str) -> None:
        if tool != "pick" and getattr(self, "pick_btn", None) is not None:
            self.pick = None
            self.pick_btn.setChecked(False)
        self.canvas.set_tool(tool)

    # ------------------------------------------------------------------ edits
    def _commit(self, sim: Similarity, note: str) -> None:
        st = self.step
        st.transforms[self.obj] = sim.to_json()
        st.align_notes[self.obj] = note
        self.panel._store()
        self.panel._fill_step()
        self.panel.redraw()
        self.redraw()
        idx = self.image_combo.findData(self.obj)
        if idx >= 0:
            self.image_combo.setItemText(idx, self.obj)

    def _frame_edited(self, key: str, pts: list) -> None:
        if key != FRAME or self._img() is None:
            return
        img = self._img()
        h, w = img.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        scale = self.nominal_scale() if self.auto_scale.isChecked() else None
        sim = fit_similarity(corners, pts, self.mirror.isChecked(), scale=scale)
        sim.rms_um, sim.pairs = None, []
        self._commit(sim, "placed by hand (Manual alignment)")

    def _auto_scale_toggled(self, on: bool) -> None:
        sim, ns = self.transform(), self.nominal_scale()
        if on and sim is not None and ns and abs(sim.scale - ns) > 1e-12:
            img = self._img()
            h, w = img.shape[:2]
            centre = sim.apply([(w / 2, h / 2)])[0]
            new = Similarity(ns, sim.angle_deg, 0.0, 0.0, sim.mirror)
            off = centre - new.apply([(w / 2, h / 2)])[0]
            new.tx, new.ty = float(off[0]), float(off[1])
            self._commit(new, "scale from the field of view (Manual alignment)")
        else:
            self._hint()

    def _mirror_toggled(self) -> None:
        if getattr(self.panel, "mirror", None) is not None:
            self.panel.mirror.setChecked(self.mirror.isChecked())
        sim = self.transform()
        if sim is not None and sim.mirror != self.mirror.isChecked():
            img = self._img()
            h, w = img.shape[:2]
            centre = sim.apply([(w / 2, h / 2)])[0]
            new = Similarity(sim.scale, sim.angle_deg, 0.0, 0.0, self.mirror.isChecked())
            off = centre - new.apply([(w / 2, h / 2)])[0]
            new.tx, new.ty = float(off[0]), float(off[1])
            self._commit(new, "mirrored (Manual alignment)")

    # ------------------------------------------------------------------ pairs
    def start_pairs(self) -> None:
        if self._img() is None:
            self.pick_btn.setChecked(False)
            return
        if self.pick is not None and not self.pick_btn.isChecked():
            self.pick = None
            self.canvas.set_tool("pointer")
            self.tools["pointer"].setChecked(True)
            self._hint()
            return
        self.pick_btn.setChecked(True)
        if self.pairs and "um" not in self.pairs[-1]:
            self.pairs.pop()
        self.pick = "img"
        self.canvas.set_tool("pick")
        self._hint()

    def _picked(self, x: float, y: float) -> None:
        if self.pick is None:
            return
        if self.pick == "img":
            item = self.canvas.images.get(f"align:{self.obj}")
            if item is None:
                return
            px, py = item.scene_to_px(QtCore.QPointF(x, y))
            self.pairs.append({"px": (float(px), float(py))})
            self.pick = "gds"
        else:
            self.pairs[-1]["um"] = (float(x), float(y))
            self.pick = "img"
        self.redraw()

    def fit_pairs(self) -> None:
        done = [p for p in self.pairs if "um" in p]
        scale = self.nominal_scale() if self.auto_scale.isChecked() else None
        need = 1 if scale else 2
        if len(done) < need:
            self.hint.setText(f"Pick at least {need} complete pair{'s' if need > 1 else ''} (image point, then the same point in the "
                              "bond-pad pattern) first.")
            return
        sim = fit_similarity([p["px"] for p in done], [p["um"] for p in done], self.mirror.isChecked(), scale=scale)
        self.pick = None
        self.pick_btn.setChecked(False)
        self.canvas.set_tool("pointer")
        self.tools["pointer"].setChecked(True)
        self._commit(sim, f"{len(done)} pairs picked by hand, ±{sim.rms_um:.2f} µm")

    def clear_pairs(self) -> None:
        self.pairs = []
        if self.pick is not None:
            self.pick = "img"
        self.redraw()


def _area(p) -> float:
    a = np.asarray(p, float)
    return 0.5 * float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))
