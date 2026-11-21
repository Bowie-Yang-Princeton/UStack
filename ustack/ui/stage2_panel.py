"""The Stage 2 tab: one EBL step on a fabricated device, drawn on the EBL canvas.

1. Images (5x, 50x, 100x bright field, 100x LUT) → 2. bond-pad GDS (marker and lead layers, the
device site) → 3. align the 100x image to the GDS markers → 4. trace flakes / defects / etch
windows → 5. measured thicknesses → 6. purposes and Hall-bar placement → Generate → Export.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from .. import geometry as geo
from .. import imaging
from ..alignment import Similarity, auto_align, fit_similarity, squares_from_polygons
from ..config import MissingItem, MissingValues, apply_answers
from ..gds_io import lead_tips, read_polygons
from ..model import FUNCTIONS, GATE_FUNCTIONS
from ..scene3d import layer_color
from ..stage2 import DEFECT_KINDS, PURPOSES, PURPOSES_BY_SCHEME, StepInputs, check_step, design_step, etch_plan, export_step
from ..validation import apply_spec_answers, category
from .dialogs import MissingValuesDialog
from .panels import IssueList
from .style import Banner, Card, CheckBox, ComboBox, IconButton, PALETTE, Pill, QtCore, QtGui, QtWidgets, Slider, button, clear_layout, label, scroll
from .widgets import NumberField, buttons_row, modernize, open_file, row

Qt = QtCore.Qt
IMAGE_KEYS = (("5x", "5x bright field"), ("20x", "20x bright field"), ("50x", "50x bright field"), ("100x", "100x bright field"),
              ("100x_lut", "100x LUT"))
PATTERN_COLORS = {"board_etch": "#e57373", "hallbar_etch": "#ffb74d", "contacts": "#d4af37", "top_gate": "#7986cb",
                  "prepattern_contacts": "#b0bec5", "contact_gates": "#9575cd", "split_gates": "#4db6ac", "etch_windows": "#81c784",
                  "ref_hallbar": "#ffffff", "ref_tg_body": "#3949ab", "ref_top_gate": "#7986cb", "ref_contact_gates": "#9575cd",
                  "ref_split_gates": "#4db6ac", "ref_board_keep": "#66bb6a"}


class Stage2Panel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.inputs = StepInputs(purposes=[])
        self.images: dict[str, object] = {}         # key → BGR array
        self.gds_polys: dict = {}
        self.gds_path: str | None = None
        self.transform: Similarity | None = None
        self.design = None
        self.report = None
        self.pick_mode: str | None = None
        self.pairs: list[dict] = []
        self.trace_mode = "outline"                  # outline | defect | window
        self._trace_ctx: dict = {}                   # image key → (image, prepared for the wand)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.content = QtWidgets.QWidget()
        self.lay = QtWidgets.QVBoxLayout(self.content)
        self.lay.setContentsMargins(6, 0, 6, 6)
        self.lay.setSpacing(0)
        outer.addWidget(scroll(self.content))
        canvas = app.ebl_canvas
        canvas.wandClicked.connect(self._wand)
        canvas.polygonDrawn.connect(self._drawn)
        canvas.polygonEdited.connect(self._edited)
        canvas.pointPicked.connect(self._picked)
        self.build()

    # ------------------------------------------------------------------ layout
    def build(self) -> None:
        clear_layout(self.lay)
        spec = self.app.spec
        if spec is None:
            self.lay.addWidget(label("No device", "faint"))
            return
        # step + purposes
        step = Card("EBL Step")
        self.step_name = QtWidgets.QLineEdit(f"step{len(spec.stage2.get('steps', [])) + 1}")
        step.layout.addLayout(row("Name", self.step_name, stretch=False))
        self.purpose_boxes = {}
        offered = PURPOSES_BY_SCHEME.get(spec.contact_scheme, list(PURPOSES))
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        for i, key in enumerate(list(dict.fromkeys(offered + list(PURPOSES)))):
            cb = CheckBox(PURPOSES[key])
            cb.setChecked(key in self.inputs.purposes)
            if key not in offered:
                cb.setToolTip("Not usual for this contact scheme")
            cb.toggled.connect(lambda v, k=key: self._purpose(k, v))
            self.purpose_boxes[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        step.layout.addLayout(grid)
        self.lay.addWidget(step)
        # images
        imgs = Card("Device Images", "Upload the current images before this EBL step.")
        self.image_labels = {}
        for key, cap in IMAGE_KEYS:
            name = label(Path(self.inputs.images.get(key, "")).name or "—", "muted")
            name.setMinimumWidth(120)
            self.image_labels[key] = name
            b = button("Load…", lambda _=False, k=key: self.load_image(k), None, f"Load the {cap} image", "image")
            v = button("View", lambda _=False, k=key: self.view_image(k), None, "Open this image in its own window")
            t = IconButton("trash", f"Delete the uploaded {cap} image", size=26, icon_size=16)
            t.clicked.connect(lambda _=False, k=key: self.remove_image(k))
            imgs.layout.addLayout(row(cap.split(" ")[0] + (" LUT" if "lut" in key else ""), b, v, t, name, width=70))
        self.lay.addWidget(imgs)
        # bond pads + alignment
        al = Card("Bond Pads / Alignment", "The pattern is written in the bond-pad GDS frame, relative to the chip's markers.")
        al.layout.addLayout(buttons_row(button("Load bond-pad GDS…", self.load_gds, None, "", "file"),
                                        lab := label(Path(self.gds_path).name if self.gds_path else "—", "muted")))
        self.gds_label = lab
        self.marker_combo = ComboBox()
        self.lead_combo = ComboBox()
        self._fill_layer_combos()
        self.marker_combo.activated.connect(lambda _i: self._layers_chosen())
        self.lead_combo.activated.connect(lambda _i: self._layers_chosen())
        al.layout.addLayout(row("Marker layer", self.marker_combo, stretch=False))
        al.layout.addLayout(row("Lead layer", self.lead_combo, stretch=False))
        self.site_label = label(self._site_text(), "muted")
        al.layout.addLayout(row("Device site", button("Pick on canvas", lambda: self._start_pick("site"), None,
                                                      "Click near the device's markers in the GDS"), self.site_label))
        self.mirror = CheckBox("Image mirrored")
        al.layout.addLayout(row("", self.mirror))
        al.layout.addLayout(buttons_row(button("Auto align", self.auto_align, "primary", "Find the marker squares in the 100x image and match them to the GDS"),
                                        button("Pick pairs", lambda: self._start_pick("pair_img"), None,
                                               "Click a marker in the image, then the same marker in the GDS; two or more pairs"),
                                        button("Fit pairs", self.fit_pairs)))
        self.align_label = Banner(self._align_text(), "neutral")
        al.layout.addWidget(self.align_label)
        self.lay.addWidget(al)
        # tracing
        tr = Card("Trace", "Wand: click a flake on the 100x image. Draw: click corners, double-click to close. "
                           "Drag the square handles to edit; double-click an edge to add a point; Delete removes the selected point.")
        self.trace_layer = ComboBox()
        for l in reversed(spec.layers):
            cat = category(self.app.config, l)
            if cat in ("2d_conductor", "2d_semiconductor", "2d_insulator") or (cat == "metal" and l.function in GATE_FUNCTIONS):
                done = "✓ " if self.inputs.outlines.get(l.uid) else ""
                self.trace_layer.addItem(done + l.label, l.uid)
        self.trace_layer.activated.connect(lambda _i: self._activate_trace_layer())
        tr.layout.addLayout(row("Layer", self.trace_layer, stretch=False))
        self.import_combo = ComboBox()
        self._fill_import_combo()
        tr.layout.addLayout(row("From GDS", self.import_combo,
                                button("Use", self.import_outline, None, "Use this bond-pad GDS layer's shapes near the device site as the "
                                       "selected layer's outline (e.g. a pre-patterned metal gate or a flake traced in KLayout)"), stretch=False))
        self.trace_kind = ComboBox()
        for k, v in imaging.TRACE_MODES.items():
            self.trace_kind.addItem(v, k)
        self.trace_kind.setCurrentIndex(self.trace_kind.findData(getattr(self, "_trace_kind", "colour")))
        self.trace_kind.setToolTip("This colour only: one layer seen through the stack (the BG under the hBN, a monolayer "
                                   "beside a thick piece). Whole flake: everything that is not substrate.")
        self.trace_kind.activated.connect(lambda _i: setattr(self, "_trace_kind", self.trace_kind.currentData()))
        tr.layout.addLayout(row("Wand", self.trace_kind, stretch=False))
        self.bridge = Slider(0, 25, 0)
        self.bridge.setFixedWidth(160)
        self.bridge.setToolTip("Close gaps up to this size (working pixels), e.g. across metal fingers that cross a flake")
        tr.layout.addLayout(row("Bridge gaps", self.bridge))
        tr.layout.addLayout(buttons_row(button("Wand", lambda: self._trace("outline", "wand"), "primary"),
                                        button("Draw", lambda: self._trace("outline", "draw")),
                                        button("Edit", lambda: self.app.set_ebl_tool("pointer")),
                                        button("Clear", self._clear_outline, "danger")))
        self.defect_kind = ComboBox()
        for k in DEFECT_KINDS:
            self.defect_kind.addItem(k, k)
        tr.layout.addLayout(row("Defect", self.defect_kind, button("Draw", lambda: self._trace("defect", "draw")),
                                button("Suggest", self.suggest_defects, None, "Find bubble / wrinkle candidates inside the device layer"),
                                button("Clear", self._clear_defects, "danger"), stretch=False))
        tr.layout.addLayout(row("Etch window", button("Draw", lambda: self._trace("window", "draw"), None, "Top-hBN-only check window"),
                                button("Clear", self._clear_windows, "danger")))
        self.lay.addWidget(tr)
        # measured
        ms = Card("Measured", "Thicknesses (AFM) or layer counts measured on this device. Empty = use the Stage 1 value.")
        self.measure_fields = {}
        for l in reversed(spec.layers):
            cat = category(self.app.config, l)
            if cat not in ("2d_conductor", "2d_semiconductor", "2d_insulator"):
                continue
            m = self.inputs.measured.get(l.uid, {})
            if l.function == "device_layer":
                f = NumberField(m.get("layer_count"), f"{l.layer_count.value:g}" if l.layer_count.value is not None else "", 72)
                f.valueEdited.connect(lambda v, u=l.uid: self._measure(u, "layer_count", v))
                ms.layout.addLayout(row(l.label[:20], f, label("layers", "muted"), width=150))
            else:
                f = NumberField(m.get("thickness_nm"), f"{l.thickness_nm.value:g}" if l.thickness_nm.value is not None else "", 72)
                f.valueEdited.connect(lambda v, u=l.uid: self._measure(u, "thickness_nm", v))
                ms.layout.addLayout(row(l.label[:20], f, label("nm", "muted"), width=150))
        self.lay.addWidget(ms)
        # placement + run
        run = Card("Design")
        self.placement = ComboBox()
        for k, v in (("auto", "Automatic"), ("previous step", "From the previous step"), ("user", "Manual")):
            self.placement.addItem(v, k)
        if spec.stage2.get("hall_bar_placement"):
            self.placement.setCurrentIndex(1)
        run.layout.addLayout(row("Hall bar", self.placement, stretch=False))
        prev = spec.stage2.get("hall_bar_placement") or {}
        self.px = NumberField(prev.get("center", [None, None])[0], "x µm", 80)
        self.py = NumberField(prev.get("center", [None, None])[1], "y µm", 80)
        self.pa = NumberField(prev.get("angle_deg"), "angle °", 64)
        run.layout.addLayout(row("Manual", self.px, self.py, self.pa, button("Pick", lambda: self._start_pick("centre"))))
        run.layout.addLayout(buttons_row(button("Generate", self.generate, "primary", "Place the Hall bar and draw every purpose", "check"),
                                         button("Re-check", self.recheck, None, "Run the design-rule check again after edits"),
                                         button("Export GDS…", self.export, None, "Write the GDS and the updated specification", "export")))
        self.drc_pill = Pill("Not generated", "neutral")
        run.layout.addWidget(self.drc_pill)
        self.drc = IssueList()
        self.drc.setMinimumHeight(260)
        self.drc.itemClicked.connect(self._drc_clicked)
        run.layout.addWidget(self.drc)
        self.lay.addWidget(run)
        self.lay.addStretch(1)
        modernize(self.content)

    # ------------------------------------------------------------------ helpers
    def _site_text(self) -> str:
        s = self.inputs.site
        return f"({s[0]:.1f}, {s[1]:.1f}) µm" if s else "not chosen"

    def _align_text(self) -> str:
        t = self.transform
        if t is None:
            return "Not aligned. Load the 100x image and the bond-pad GDS, pick the device site, then Auto align."
        return (f"Aligned: {t.scale * 1000:.2f} nm/px, rotation {t.angle_deg:.2f}°, RMS {t.rms_um * 1000 if t.rms_um is not None else float('nan'):.0f} nm "
                f"over {len(t.pairs)} marker squares.")

    def _fill_layer_combos(self) -> None:
        for combo, key in ((self.marker_combo, "marker_layer"), (self.lead_combo, "lead_layer")):
            combo.clear()
            combo.addItem("—", None)
            for (l, d), polys in sorted(self.gds_polys.items()):
                combo.addItem(f"{l}/{d}  ({len(polys)} shapes)", f"{l}/{d}")
            cur = getattr(self.inputs, key)
            if cur is not None:
                i = combo.findData(f"{cur[0]}/{cur[1]}")
                combo.setCurrentIndex(max(0, i))

    def _fill_import_combo(self) -> None:
        if not hasattr(self, "import_combo"):
            return
        self.import_combo.clear()
        self.import_combo.addItem("—", None)
        for (l, d), polys in sorted(self.gds_polys.items()):
            self.import_combo.addItem(f"{l}/{d}  ({len(polys)} shapes)", f"{l}/{d}")

    def import_outline(self) -> None:
        uid = self.trace_layer.currentData()
        ld = self._ld(self.import_combo.currentData())
        if uid is None or ld is None:
            self.app.warn("From GDS", "Load the bond-pad GDS, then choose the layer to trace and the GDS layer to take its outline from.")
            return
        polys = self.gds_polys.get(ld, [])
        if self.inputs.site:
            radius = self._site_radius()
            polys = [p for p in polys if math.dist(geo.centroid(p), self.inputs.site) <= radius]
        if not polys:
            self.app.warn("From GDS", f"GDS layer {ld[0]}/{ld[1]} has no shapes near the device site.")
            return
        self.inputs.outlines[uid] = [list(map(tuple, p)) for p in polys]
        self.redraw()
        self.app.status(f"{self.app.spec.layer(uid).label}: outline from GDS layer {ld[0]}/{ld[1]} ({len(polys)} shapes).")

    def _purpose(self, key: str, on: bool) -> None:
        ps = [p for p in self.inputs.purposes if p != key]
        if on:
            ps.append(key)
        self.inputs.purposes = [p for p in PURPOSES if p in ps]

    def _measure(self, uid: str, key: str, v) -> None:
        m = self.inputs.measured.setdefault(uid, {})
        if v is None:
            m.pop(key, None)
        else:
            m[key] = v

    def device_dir(self) -> Path:
        return self.app.device_dir()

    # ------------------------------------------------------------------ images / GDS
    def load_image(self, key: str) -> None:
        path = open_file(self, "Load image", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not path:
            return
        try:
            img = imaging.load_image(path)
        except Exception as exc:
            self.app.warn("Image", str(exc))
            return
        dest = self.device_dir() / "images" / f"{self.step_name.text().strip() or 'step'}_{key}{Path(path).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)
        self.inputs.images[key] = str(dest)
        self.images[key] = img
        self.image_labels[key].setText(Path(path).name)
        if key in ("100x", "100x_lut"):
            if self.transform is not None:
                self._store_transforms()
            self.redraw(fit=self.transform is None)


    def _image(self, key: str):
        """The image for ``key``: in memory, or read again from the file this step stored."""
        if key not in self.images and self.inputs.images.get(key) and Path(self.inputs.images[key]).is_file():
            try:
                self.images[key] = imaging.load_image(self.inputs.images[key])
            except Exception:
                return None
        return self.images.get(key)

    def view_image(self, key: str) -> None:
        """Open the uploaded image in its own window (µm from the objective's field of view)."""
        img = self._image(key)
        cap = dict(IMAGE_KEYS).get(key, key)
        if img is None:
            self.app.status(f"No {cap} image loaded for this step — press Load… first")
            return
        obj = "100x" if key == "100x_lut" else key
        s = self.app.config.um_per_px(obj, img.shape[1])
        fov = self.app.config.field_of_view_um(obj)
        name = Path(self.inputs.images.get(key, "")).name
        caption = f"{name} · {img.shape[1]} × {img.shape[0]} px" + (f" · {fov[0]:g} × {fov[1]:g} µm field of view" if s and fov else " · pixel units")
        if key in ("100x", "100x_lut") and self.transform is not None:
            caption += " · aligned to the chip on the EBL canvas"
        from .viewer import ImageViewerDialog
        dlg = ImageViewerDialog(img, f"{cap} — {name}", caption, s, self.app)
        self._viewers = [v for v in getattr(self, "_viewers", []) if v.isVisible()] + [dlg]
        dlg.show()

    def remove_image(self, key: str) -> None:
        """Forget this step's image (the copy in the device folder is deleted; the original file is not touched)."""
        path = self.inputs.images.pop(key, None)
        self.images.pop(key, None)
        if key in getattr(self, "image_labels", {}):
            self.image_labels[key].setText("—")
        if path:
            p = Path(path)
            try:
                if p.is_file() and p.resolve().parent == (self.device_dir() / "images").resolve():
                    p.unlink()
            except OSError:
                pass
        if key in ("100x", "100x_lut"):
            self.redraw()
        self.app.status(f"{dict(IMAGE_KEYS).get(key, key)} image removed from this step")

    def load_gds(self) -> None:
        path = open_file(self, "Bond-pad GDS", "GDS (*.gds *.gds2 *.gdsii *.oas)")
        if not path:
            return
        try:
            self.gds_polys = read_polygons(path)
        except Exception as exc:
            self.app.warn("GDS", str(exc))
            return
        self.gds_path = path
        self.inputs.bondpad_gds = path
        self.gds_label.setText(Path(path).name)
        self._fill_layer_combos()
        self._fill_import_combo()
        self.redraw(fit=True)

    @staticmethod
    def _ld(text):
        if not text:
            return None
        a, b = str(text).split("/")
        return int(a), int(b)

    def _layers_chosen(self) -> None:
        self.inputs.marker_layer = self._ld(self.marker_combo.currentData())
        self.inputs.lead_layer = self._ld(self.lead_combo.currentData())
        self._update_site_dependents()
        self.redraw()

    def _update_site_dependents(self) -> None:
        site = self.inputs.site
        radius = self._site_radius()
        if site and self.inputs.marker_layer in self.gds_polys:
            self.inputs.marker_polys = [p for p in self.gds_polys[self.inputs.marker_layer] if math.dist(geo.centroid(p), site) <= radius]
        if site and self.inputs.lead_layer in self.gds_polys:
            self.inputs.lead_tips = lead_tips(self.gds_polys[self.inputs.lead_layer], site, radius)

    def _site_radius(self) -> float:
        region = self.app.config.value("ebl.alignment_region_um")
        return (region / 2) if region else 400.0

    # ------------------------------------------------------------------ alignment
    def auto_align(self) -> None:
        if "100x" not in self.images:
            self.app.warn("Align", "Load the 100x bright-field image first.")
            return
        if self.inputs.marker_layer is None:
            self.app.warn("Align", "Load the bond-pad GDS and choose its marker layer.")
            return
        squares = [s for g in imaging.detect_markers(self.images["100x"]) for s in g.squares]
        gsq = squares_from_polygons(self.gds_polys.get(self.inputs.marker_layer, []))
        s = self.app.config.um_per_px("100x", self.images["100x"].shape[1])
        # the 100x field of view gives the scale to within the camera's calibration: reject other magnifications
        rng = (0.85 * s, 1.15 * s) if s else None
        try:
            self.transform = auto_align(squares, gsq, self.inputs.site, self._site_radius() if self.inputs.site else None,
                                        mirror=self.mirror.isChecked(), scale_range=rng)
        except ValueError as exc:
            self.app.warn("Align", f"{exc}\n\nFound {len(squares)} marker squares in the image.")
            return
        self._store_transforms()
        if self.inputs.site is None:
            (x, y), = self.transform.apply([(self.images['100x'].shape[1] / 2, self.images['100x'].shape[0] / 2)])
            self.inputs.site = (float(x), float(y))
            self.site_label.setText(self._site_text())
            self._update_site_dependents()
        self.align_label.set_tone("ok", self._align_text())
        self.redraw(fit=True)

    def fit_pairs(self) -> None:
        if len(self.pairs) < 2 or any("um" not in p for p in self.pairs):
            self.app.warn("Align", "Pick at least two complete pairs (image point, then GDS point).")
            return
        self.transform = fit_similarity([p["px"] for p in self.pairs], [p["um"] for p in self.pairs], self.mirror.isChecked())
        self._store_transforms()
        self.align_label.set_tone("ok", self._align_text())
        self.pick_mode = None
        self.app.set_ebl_tool("pointer")
        self.redraw()

    def _start_pick(self, mode: str) -> None:
        self.pick_mode = mode
        if mode == "pair_img":
            self.pairs = []
        self.app.set_ebl_tool("pick")
        self.app.status({"site": "Click near the device site in the GDS", "pair_img": "Click a marker corner/centre in the IMAGE",
                         "centre": "Click the Hall-bar centre"}.get(mode, ""))

    def _picked(self, x: float, y: float) -> None:
        if not getattr(self, "active", True):
            return
        mode = self.pick_mode
        if mode == "site":
            self.inputs.site = (x, y)
            self.site_label.setText(self._site_text())
            self._update_site_dependents()
            self.pick_mode = None
            self.app.set_ebl_tool("pointer")
            self.redraw()
        elif mode == "pair_img":
            item = self.app.ebl_canvas.images.get("100x")
            if item is None:
                return
            px, py = item.scene_to_px(QtCore.QPointF(x, y))
            self.pairs.append({"px": (px, py)})
            self.pick_mode = "pair_gds"
            self.app.status("Now click the same marker in the GDS")
        elif mode == "pair_gds":
            self.pairs[-1]["um"] = (x, y)
            self.pick_mode = "pair_img"
            self.app.status(f"{len(self.pairs)} pair(s). Click the next image marker, or press Fit pairs.")
        elif mode == "centre":
            self.px.set_value(x)
            self.py.set_value(y)
            self.placement.setCurrentIndex(2)
            self.pick_mode = None
            self.app.set_ebl_tool("pointer")

    def _store_transforms(self) -> None:
        self.inputs.transforms["100x"] = self.transform.to_json()
        if "100x_lut" in self.images:
            self.inputs.transforms["100x_lut"] = self._sim("100x_lut").to_json()

    def _lut_factor(self) -> float:
        """BF pixels per LUT pixel: the LUT covers the same field of view, often at a lower resolution."""
        bf, lut = self.images.get("100x"), self.images.get("100x_lut")
        if bf is None or lut is None:
            return 1.0
        return bf.shape[1] / lut.shape[1]

    def _placement_of_image(self, key: str = "100x"):
        if self.transform is not None:
            t = self._sim(key)
            return (t.scale, t.angle_deg, t.tx, t.ty, t.mirror)
        img = self.images.get("100x")
        if img is None:
            return None
        if key == "100x_lut":
            s, a, tx, ty, m = self._placement_of_image("100x")
            return (s * self._lut_factor(), a, tx, ty, m)
        # not aligned yet: show the image over the site at the 100x field-of-view scale so pairs can be picked
        s = self.app.config.um_per_px("100x", img.shape[1])
        if not s:
            s = self._site_radius() * 2 * 0.25 / img.shape[1]
        span = s * img.shape[1]
        cx, cy = self.inputs.site or (0.0, 0.0)
        return (s, 0.0, cx - span / 2, cy + img.shape[0] * s / 2, False)

    # ------------------------------------------------------------------ tracing
    def _trace(self, mode: str, tool: str) -> None:
        self.trace_mode = mode
        if mode == "outline":
            self._activate_trace_layer()
        self.app.set_ebl_tool(tool)

    def _activate_trace_layer(self) -> None:
        uid = self.trace_layer.currentData()
        self.app.ebl_canvas.set_active_polygon(f"outline:{uid}" if uid else None)

    def _wand(self, key: str, px: float, py: float) -> None:
        if not getattr(self, "active", True):
            return
        uid = self.trace_layer.currentData()
        img = self.images.get(key)
        if uid is None or img is None or self.transform is None and key != "100x":
            return
        try:
            ctx = self._trace_ctx.get(key)
            if ctx is None or ctx[0] is not img:
                ctx = self._trace_ctx[key] = (img, imaging.prepare_trace(img))
            poly_px = imaging.detect_flake(img, (px, py), mode=self.trace_kind.currentData(), bridge=int(self.bridge.value()),
                                           context=ctx[1])
        except ValueError as exc:
            self.app.status(str(exc))
            return
        t = self._sim(key)
        pts = [tuple(map(float, p)) for p in t.apply(poly_px)]
        self.inputs.outlines[uid] = [pts]
        self.redraw()
        self.app.ebl_canvas.set_active_polygon(f"outline:{uid}")
        self.app.status(f"Outline of {self.app.spec.layer(uid).label}: {len(pts)} points. Drag points to correct it.")

    def _sim(self, key: str = "100x") -> Similarity:
        """px → chip µm for image ``key`` (the LUT shares the BF alignment, scaled to its resolution)."""
        if self.transform is not None:
            t = self.transform
            if key == "100x_lut":
                return Similarity(t.scale * self._lut_factor(), t.angle_deg, t.tx, t.ty, t.mirror, t.rms_um)
            return t
        s, a, tx, ty, m = self._placement_of_image(key)
        return Similarity(s, a, tx, ty, m)

    def _drawn(self, pts: list) -> None:
        if not getattr(self, "active", True):
            return
        if self.app.center_tabs.currentWidget() is not self.app.ebl_page:
            return
        if self.trace_mode == "defect":
            self.inputs.defects.append({"kind": self.defect_kind.currentData(), "polygon": pts})
        elif self.trace_mode == "window":
            self.inputs.etch_windows.append(pts)
        else:
            uid = self.trace_layer.currentData()
            if uid:
                self.inputs.outlines[uid] = [pts]
        self.redraw()

    def _edited(self, key: str, pts: list) -> None:
        if not getattr(self, "active", True):
            return
        kind, _, rest = key.partition(":")
        if kind == "outline":
            self.inputs.outlines[rest] = [pts]
        elif kind == "defect":
            self.inputs.defects[int(rest)]["polygon"] = pts
        elif kind == "window":
            self.inputs.etch_windows[int(rest)] = pts

    def _clear_outline(self) -> None:
        uid = self.trace_layer.currentData()
        self.inputs.outlines.pop(uid, None)
        self.redraw()

    def _clear_defects(self) -> None:
        self.inputs.defects = []
        self.redraw()

    def _clear_windows(self) -> None:
        self.inputs.etch_windows = []
        self.redraw()

    def suggest_defects(self) -> None:
        img = self.images.get("100x")
        dev = [p for l in self.app.spec.device_layers() for p in self.inputs.outlines.get(l.uid, [])]
        if img is None or not dev:
            self.app.warn("Defects", "Load the 100x image and trace the device layer first.")
            return
        t = self._sim()
        region_px = [tuple(p) for p in t.inverse(dev[0])]
        found = imaging.defect_candidates(img, region_px)
        for d in found:
            self.inputs.defects.append({"kind": d["kind"], "polygon": [tuple(map(float, p)) for p in t.apply(d["polygon"])], "suggested": True})
        self.redraw()
        self.app.status(f"{len(found)} defect candidates added (review them: delete the false ones with Clear and redraw, or edit).")

    # ------------------------------------------------------------------ drawing
    def redraw(self, fit: bool = False) -> None:
        if not getattr(self, "active", True):
            return
        c = self.app.ebl_canvas
        spec = self.app.spec
        active = c.active_poly
        c.clear_all()
        c.empty_text = "Load the 100x image and the bond-pad GDS"
        for (l, d), polys in self.gds_polys.items():
            col = "#c9a227" if (l, d) == self.inputs.marker_layer else ("#8d99ae" if (l, d) == self.inputs.lead_layer else "#c3c6ca")
            c.add_overlay_polys(polys, col, 0.35, z=5)
        key = getattr(self.app, "ebl_view_key", "100x")
        if key not in self.images:
            key = "100x"
        place = self._placement_of_image(key)
        if key in self.images and place:
            c.set_image(key, self.images[key], place, 1.0, 1)
        for l in spec.layers:
            for i, poly in enumerate(self.inputs.outlines.get(l.uid, [])):
                c.set_polygon(f"outline:{l.uid}", poly, layer_color(l, self.app.config), True, 50, 0.10)
        for i, d in enumerate(self.inputs.defects):
            c.set_polygon(f"defect:{i}", d["polygon"], "#d05a5a", True, 60, 0.25, dash=d.get("suggested", False))
        for i, w in enumerate(self.inputs.etch_windows):
            c.set_polygon(f"window:{i}", w, "#2e8b57", True, 60, 0.2)
        if self.inputs.site:
            c.add_markers([self.inputs.site], PALETTE["accent"], 6)
        for x, y in self.inputs.lead_tips:
            c.add_markers([(x, y)], "#5c6bc0", 3)
        if self.design is not None:
            for k, polys in self.design.shapes.items():
                if k in ("ref_flakes", "align_marks", "ref_defects"):
                    continue
                col = PATTERN_COLORS.get(k, "#90a4ae")
                alpha = 0.0 if k in ("ref_hallbar", "ref_board_keep", "ref_tg_body") else (0.25 if k.startswith("ref_") else 0.5)
                c.add_overlay_polys(polys, col, alpha, z=90 if not k.startswith("ref_") else 85, width=1.2)
            if self.design.hall_bar is not None:
                for t in self.design.hall_bar.terminals():
                    if t.kind != "probe" or t.name.endswith("T"):
                        c.add_label(t.name if t.kind != "probe" else t.name[1:-1], t.end[0], t.end[1], PALETTE["text"], small=True)
        if self.report is not None:
            pts = [p for i in self.report.issues if i.severity == "error" for p in i.where]
            c.add_markers(pts, PALETTE["danger"], 9)
        if active:
            c.set_active_polygon(active)
        if fit:
            img = c.images.get(key)
            if img is not None:
                c.fit_rect(img.sceneBoundingRect())
            else:
                c.fit()

    # ------------------------------------------------------------------ run
    def _collect_placement(self) -> None:
        mode = self.placement.currentData()
        prev = self.app.spec.stage2.get("hall_bar_placement")
        if mode == "previous step" and prev:
            self.inputs.placement = {"center": prev["center"], "angle_deg": prev["angle_deg"], "source": "previous step",
                                     "sd_style": prev.get("sd_style")}
        elif mode == "user" and None not in (self.px.value(), self.py.value(), self.pa.value()):
            self.inputs.placement = {"center": [self.px.value(), self.py.value()], "angle_deg": self.pa.value(), "source": "user"}
        else:
            self.inputs.placement = None

    def generate(self) -> None:
        spec = self.app.spec
        if not self.inputs.purposes:
            self.app.warn("Generate", "Choose at least one purpose for this step.")
            return
        self._collect_placement()
        for _ in range(4):
            try:
                self.design = design_step(spec, self.inputs, self.app.config)
                break
            except MissingValues as exc:
                actions = [i for i in exc.items if i.kind == "action"]
                values = [i for i in exc.items if i.kind != "action"]
                if not values:
                    self.app.warn("Generate", "\n".join(a.label for a in actions))
                    return
                dlg = MissingValuesDialog(exc.items, self.app, "Values needed for this step")
                if not dlg.exec():
                    return
                ans = dlg.answers()
                apply_answers(self.app.config, ans)
                apply_spec_answers(spec, ans)
                self.app.spec_changed("values")
                if actions:
                    return
            except ValueError as exc:
                self.app.warn("Generate", str(exc))
                return
        else:
            return
        self.recheck()

    def recheck(self) -> None:
        if self.design is None:
            return
        self.report = check_step(self.app.spec, self.inputs, self.design, self.app.config)
        self.drc.clear()
        stage1_missing = [i for i in self.report.issues if i.code.startswith("stage1:") and i.severity == "missing"]
        for i in self.report.issues:
            if i.code.startswith("stage1:") and i.severity in ("info", "missing"):
                continue
            self.drc.add(i.severity, i.message, i.where[:1])
        if stage1_missing:
            self.drc.add("missing", f"{len(stage1_missing)} Stage 1 values are still missing (Check tab › Fill in missing…).")
        for line in etch_plan(self.app.spec, self.inputs, self.app.config) + self.design.notes:
            self.drc.add("note", line)
        self.drc_pill.set_tone("ok" if self.report.ok else "warn", self.report.summary())
        self.redraw()

    def _drc_clicked(self, item) -> None:
        where = item.data(Qt.ItemDataRole.UserRole)
        if where:
            x, y = where[0]
            self.app.ebl_canvas.center_on(x, y, 12.0)

    def export(self) -> None:
        if self.design is None or self.report is None:
            self.app.warn("Export", "Generate the pattern first.")
            return
        if not self.report.ok:
            ans = QtWidgets.QMessageBox.question(self, "Export", "The design-rule check reports errors or missing values. Export anyway "
                                                 "(the report lists them)?")
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.step_name.text().strip() or "step")
        out_dir = self.device_dir() / "stage2" / name
        for _ in range(3):
            try:
                out = export_step(self.app.spec, self.inputs, self.design, self.report, self.app.config, out_dir, name)
                break
            except MissingValues as exc:
                dlg = MissingValuesDialog(exc.items, self.app, "GDS layers needed",
                                          "These pattern purposes have no GDS layer/datatype in config/layers.json yet.")
                if not dlg.exec():
                    return
                apply_answers(self.app.config, dlg.answers())
        else:
            return
        self.app.adopt_spec(out["updated_spec"], f"Stage 2 step '{name}' exported")
        self.app.status(f"Wrote {out['gds']} and {out['spec'].name}")
        QtWidgets.QMessageBox.information(self, "Exported", f"GDS: {out['gds']}\nSpecification: {out['spec']}\nReport: {out['report']}")
