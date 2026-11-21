"""The Stage 2 tab: a workflow of EBL steps on one chip (see :mod:`ustack.ebl_flow`).

Top: the bond-pad preset (saved once, reused). Then the steps in process order — Hall bars (finger
array), contact gate, bottom gate, top gate, electrodes to the bond pads (drag a step or use the
arrows to reorder them), and the classic graphene Hall-bar step. Every ticked step opens its own
card (collapsible with the ^ at its top right): images of the sample up to this point (any of 5x /
10x / 20x / 50x / 100x; at least one of 5x/10x/20x fixes the chip alignment), Align to bond pads,
Locate flakes, the twisted region (Hall bars), Generate / Check / Export GDS. The card last used is
the one the EBL canvas edits. *Image on canvas* chooses the image the canvas shows, from any step
and magnification. Last: Merge & Export writes the chosen steps and the bond pads into one GDS.
"""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path

from .. import ebl_flow as ef
from .. import geometry as geo
from .. import imaging
from ..alignment import Similarity, fit_similarity
from ..bondpads import ROLES, BondPads, list_presets, save_preset
from ..config import MissingValues, apply_answers
from ..gds_io import read_polygons
from ..scene3d import layer_color
from .dialogs import MissingValuesDialog, PresetsDialog
from .outline_window import OutlineDialog
from .panels import IssueList
from .stage2_panel import Stage2Panel
from .style import (Banner, Card, CheckBox, ComboBox, Dialog, IconButton, PALETTE, Pill, QtCore, QtGui, QtWidgets, Slider, button,
                    clear_layout, label, scroll)
from .widgets import AngleControl, CheckDelegate, NumberField, buttons_row, clickable_rows, modernize, open_file, row

Qt = QtCore.Qt
CLASSIC = "classic"
LOCATE_MIN_MAG = 20.0          # flakes are looked for in 20x and finer images
LOCATE_RADIUS_UM = 150.0       # … within this distance of the device site
SHAPE_COLORS = {"prepattern_contacts": "#d4af37", "contact_gates": "#9575cd", "electrodes": "#e0883a", "bg_contacts": "#4db6ac",
                "tg_contacts": "#7986cb", "top_gate": "#7986cb",
                **{purpose: "#e0883a" for purpose in ef.ELECTRODE_PURPOSES}}      # one metal, one GDS layer
DEBRIS_COLORS = {"flake": "#ef6c00", "particle": "#ad1457", "bubble": "#2e7d32", "crack": "#00838f"}
DEBRIS_HINTS = {
    "hall_bars": "Found in this step's aligned images after the flakes are located. The finger array is placed and its fingers "
                 "are ended to keep off the bubbles and cracks where it can (as many finger pairs as without them).",
    "contact_gate": "Found in this step's aligned images after the flakes are located. The comb and its bus leads keep off the "
                    "bubbles and cracks where they can.",
    "electrodes": "Found in this step's aligned images inside the bond-pad leads, after the flakes are located; the electrodes "
                  "are routed round them."}
STATUS_TEXT = {"not started": "—", "images": "images", "generated": "generated", "exported": "exported ✓"}
#: Widgets of a step card that the panel's methods use; they are bound to ``self`` for the card in use.
CARD_ATTRS = ("step_card", "image_labels", "mirror", "align_banner", "flake_list", "fixed_angle", "max_pairs", "pill", "issues",
              "plan_list", "plan_sliders")


def _alive(w) -> bool:
    try:
        w.isChecked()
        return True
    except RuntimeError:                         # the card was rebuilt
        return False


class _StepList(QtWidgets.QTreeWidget):
    """The step list; dragging a step to another row reorders the process (``orderChanged``)."""

    orderChanged = QtCore.pyqtSignal()

    def dropEvent(self, e):  # noqa: N802 - Qt API
        super().dropEvent(e)
        self.orderChanged.emit()


class EBLFlowPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.flow = ef.Flow()
        self._just_toggled = False
        self.images: dict[tuple[str, str], object] = {}      # (step, objective) → BGR
        self.draw_target: str | None = None                  # "twisted" | "outline:<uid>"
        self.pick: str | None = None
        self.pairs: list[dict] = []
        self._viewers = []
        self.outline_dialog: OutlineDialog | None = None
        self.plan_sliders: dict[str, AngleControl] = {}
        self.plan_list = None
        self.flake_list = None
        self.current: str | None = None                      # the step whose card is in use (the canvas edits it)
        self.cards: dict[str, dict] = {}                     # step → its card's widgets (CARD_ATTRS)
        self.collapsed: set[str] = set()                     # cards folded with their ^ button
        self._canvas_shown: tuple[str, str] | None = None    # (step, objective) of the image on the canvas
        self.hidden: set[str] = set()                        # canvas layers switched off (Layers card, top right)
        self.export_flakes = False                           # Export GDS: also the traced flake outlines (each on its own layer)
        self.shown_plans: set[str] = set()                   # Layers card: flakes not on the chip yet, shown where planned
        self._pending_view: dict | None = None               # the saved EBL view, applied on the first redraw after opening
        self._bp_cache: dict[str, BondPads] = {}
        self.classic = Stage2Panel(app)
        self.classic.active = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.stack = QtWidgets.QStackedWidget()
        self.content = QtWidgets.QWidget()
        self.lay = QtWidgets.QVBoxLayout(self.content)
        self.lay.setContentsMargins(6, 0, 6, 6)
        self.lay.setSpacing(0)
        self.stack.addWidget(scroll(self.content))
        classic_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(classic_page)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addLayout(buttons_row(button("← EBL steps", lambda: self.select_step(None), None, "Back to the step list")))
        cl.addWidget(self.classic, 1)
        self.stack.addWidget(classic_page)
        outer.addWidget(self.stack)
        c = app.ebl_canvas
        c.wandClicked.connect(lambda *a: None)
        c.polygonDrawn.connect(self._drawn)
        c.polygonEdited.connect(self._edited)
        c.pointPicked.connect(self._picked)
        c.markerClicked.connect(self.issue_popup)
        c.polygonSelected.connect(self._polygon_selected)
        self.build()

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self.stack.currentIndex() == 0

    def load_from_spec(self) -> None:
        spec = self.app.spec
        self.flow = ef.Flow.of(spec) if spec is not None else ef.Flow()
        if self.flow.bondpads is None and list_presets():
            self.flow.bondpads = list_presets()[0].name
        self.images = {}
        ui = self.flow.ui or {}                          # the Stage 2 tab and the EBL canvas as they were saved
        self.current = ui.get("current") if ui.get("current") in self.flow.selected else None
        self.collapsed = set(ui.get("collapsed") or [])
        self.hidden = set(ui.get("hidden") or [])
        self.export_flakes = bool(ui.get("export_flakes"))
        self.shown_plans = set(ui.get("shown_plans") or [])   # flakes not on the chip yet, shown where Flake layout plans them
        self._pending_view = ui.get("view")
        if spec is not None and hasattr(self.app, "_outline_cache"):
            self._sync_layout()                          # flakes on the chip: their borders as found there

    def remember_ui(self) -> None:
        """Before saving: what the Stage 2 tab and the EBL canvas show, so reopening shows the same."""
        if self.app.spec is None:
            return
        c = self.app.ebl_canvas
        self.flow.ui = {"current": self.current, "collapsed": sorted(self.collapsed), "hidden": sorted(self.hidden),
                        "view": c.view_state() if (c.images or c.polys) else (self._pending_view if getattr(self, "_pending_view", None) else None),
                        "tab": "ebl" if self.app.center_tabs.currentWidget() is self.app.ebl_page else None,
                        "export_flakes": bool(getattr(self, "export_flakes", False)),
                        "shown_plans": sorted(getattr(self, "shown_plans", set()))}
        self.flow.store(self.app.spec)

    @property
    def keys(self) -> list[str]:
        """The steps shown (ticked in the list), each in its own card, in process order."""
        return self.flow.ordered(self.flow.selected)

    @property
    def key(self) -> str | None:
        """The step whose card is in use: its images, flakes and pattern are on the EBL canvas."""
        ks = self.keys
        if self.current in ks:
            return self.current
        return ks[0] if ks else None

    def _store(self) -> None:
        if self.app.spec is not None:
            ef.adopt_identified(self.app.spec, self.flow)   # layers found on the chip replace their plan at earlier steps
            for st in self.flow.steps.values():         # the twisted region follows the device-layer outlines
                ef.refresh_twisted(self.app.spec, st)
            self.flow.store(self.app.spec)
            self._sync_layout()
            self.app.spec_changed("stage2")

    def _sync_layout(self) -> list[str]:
        """Flakes found on the chip → their traced borders take the chip shape and place (Flake layout, Flake images,
        the 3D stack). Marked as synced: the change came from the chip, it is not carried back."""
        changed = ef.sync_layout_from_chip(self.app.spec, self.flow)
        for uid in changed:
            self.app.outline_synced(uid)
        if changed:
            self.app.traced_borders_changed(changed)
        return changed

    def _layout_follows_plan(self, uid: str) -> None:
        """A plan moved / turned here → the flake's border in Flake layout moves with it (linked)."""
        if ef.layout_from_plan(self.app.spec, self.flow, self.step, uid):
            self.app.outline_synced(uid)
            self.app.traced_borders_changed([uid])

    def layout_moved(self, uid: str) -> None:
        """A flake not fabricated yet moved / turned in Flake layout → its plans follow (linked)."""
        if ef.plan_from_layout(self.app.spec, self.flow, uid):
            for st in self.flow.steps.values():
                ef.refresh_twisted(self.app.spec, st)
            self.flow.store(self.app.spec)
            if self.active:
                self._fill_step()
                self.redraw()

    def bondpads(self) -> BondPads | None:
        name = self.flow.bondpads
        if not name:
            return None
        if name not in self._bp_cache:
            bp = next((b for b in list_presets() if b.name == name), None)
            if bp is None:
                return None
            self._bp_cache[name] = bp
        return self._bp_cache[name]

    @property
    def step(self) -> ef.FlowStep | None:
        return self.flow.step(self.key) if self.key in ef.STEPS else None

    # ------------------------------------------------------------------ layout
    def build(self) -> None:
        clear_layout(self.lay)
        spec = self.app.spec
        if spec is None:
            self.lay.addWidget(label("No device", "faint"))
            return
        # bond pads
        bp_card = Card("Bond Pads", "The chip pattern every step is written on: bond pads + fan-out, markers, landing square. "
                                    "Saved once as a preset.")
        self.bp_combo = ComboBox()
        for b in list_presets():
            self.bp_combo.addItem(b.name + ("" if not b.builtin else "  (built in)"), b.name)
        i = self.bp_combo.findData(self.flow.bondpads)
        self.bp_combo.setCurrentIndex(max(0, i))
        self.bp_combo.activated.connect(lambda _i: self._choose_bondpads())
        bp_card.layout.addLayout(row("Preset", self.bp_combo, stretch=False))
        bp = self.bondpads()
        if bp is not None:
            bp_card.layout.addWidget(label(f"Layers: " + ", ".join(f"{ROLES.get(k, k)} {v[0]}/{v[1]}" for k, v in bp.layers.items())
                                           + f" · device site ({bp.site[0]:g}, {bp.site[1]:g}) µm · {len(bp.pad_tips())} lead tips",
                                           "muted", wrap=True))
        bp_card.layout.addLayout(buttons_row(button("Save a GDS as preset…", self.save_bondpad_preset, None,
                                                    "Use another bond-pad GDS from now on (copied into library/bondpads)", "file"),
                                             button("Show on canvas", lambda: self.redraw(fit=True), None, "Draw the bond pads")))
        self.lay.addWidget(bp_card)
        # steps
        steps = Card("EBL Steps", "Tick the steps to work on: each opens its own card below. The order is the process order: "
                                  "drag a step to another row, or use the arrows.")
        self.step_list = clickable_rows(_StepList(), boxes=False)   # its ticks are the steps in use
        self.step_list.setItemDelegate(CheckDelegate(self.step_list))
        self.step_list.setHeaderLabels(["Step (process order)", "State"])
        self.step_list.setRootIsDecorated(False)
        self.step_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.step_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.step_list.header().setStretchLastSection(False)
        self.step_list.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.step_list.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for n, k in enumerate(self.flow.order, 1):
            kind = ef.STEPS[k]
            st = self.flow.steps.get(k)
            it = QtWidgets.QTreeWidgetItem(self.step_list, [f"{n}. {kind.name}", STATUS_TEXT[st.status] if st else "—"])
            it.setData(0, Qt.ItemDataRole.UserRole, k)
            it.setFlags((it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsDropEnabled)
            it.setCheckState(0, Qt.CheckState.Checked if k in self.flow.selected else Qt.CheckState.Unchecked)
            it.setToolTip(0, kind.blurb + " Tick to open its card; drag to reorder.")
        it = QtWidgets.QTreeWidgetItem(self.step_list, ["Classic Hall bar (graphene etch / contacts)", ""])
        it.setData(0, Qt.ItemDataRole.UserRole, CLASSIC)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsDragEnabled & ~Qt.ItemFlag.ItemIsDropEnabled)
        it.setToolTip(0, "The parameterised Hall bar of v0.1–v0.4 (board etch, Hall-bar etch, 1D contacts, metal top gate)")
        self.step_list.setFixedHeight(32 + 28 * self.step_list.topLevelItemCount())
        self.step_list.itemChanged.connect(self._step_ticked)
        self.step_list.itemClicked.connect(self._step_clicked)
        self.step_list.orderChanged.connect(self._list_reordered)
        steps.layout.addWidget(self.step_list)
        up = IconButton("arrowUp", "Move the selected step earlier in the process", size=26, icon_size=16)
        down = IconButton("arrowDown", "Move the selected step later in the process", size=26, icon_size=16)
        up.clicked.connect(lambda: self.move_step(-1))
        down.clicked.connect(lambda: self.move_step(1))
        steps.layout.addLayout(buttons_row(up, down, label("Reorder (or drag a row)", "hint")))
        self.lay.addWidget(steps)
        # the image on the canvas: any step, any magnification
        self.canvas_card = Card("Image On Canvas", "The image the EBL canvas shows under the pattern: from any step, at any "
                                                   "magnification (aligned images sit on the chip; others are centred on the site).")
        self.canvas_combo = ComboBox()
        self.canvas_combo.setMinimumWidth(220)
        self.canvas_combo.activated.connect(lambda _i: self._canvas_image_chosen())
        self.canvas_card.layout.addLayout(row("Image", self.canvas_combo, stretch=False, width=50))
        self.lay.addWidget(self.canvas_card)
        # one card per ticked step
        self.cards_widget = QtWidgets.QWidget()
        self.cards_box = QtWidgets.QVBoxLayout(self.cards_widget)
        self.cards_box.setContentsMargins(0, 0, 0, 0)
        self.cards_box.setSpacing(0)
        self.lay.addWidget(self.cards_widget)
        self.merge_card = Card("Merge & Export", "Chosen steps on their own GDS layers + the bond-pad preset, in one file.")
        self.lay.addWidget(self.merge_card)
        self.lay.addStretch(1)
        self._fill_step()
        self._fill_merge()
        modernize(self.content)

    def _status_rows(self) -> None:
        self.step_list.blockSignals(True)
        for i in range(self.step_list.topLevelItemCount()):
            it = self.step_list.topLevelItem(i)
            k = it.data(0, Qt.ItemDataRole.UserRole)
            if k in ef.STEPS:
                st = self.flow.steps.get(k)
                it.setText(0, f"{self.flow.order.index(k) + 1}. {ef.STEPS[k].name}")
                it.setText(1, STATUS_TEXT[st.status] if st else "—")
                it.setCheckState(0, Qt.CheckState.Checked if k in self.flow.selected else Qt.CheckState.Unchecked)
            it.setSelected(k == self.key)
        self.step_list.blockSignals(False)

    def _step_ticked(self, item, col) -> None:
        k = item.data(0, Qt.ItemDataRole.UserRole)
        if col != 0 or k not in ef.STEPS:
            return
        self._just_toggled = True
        on = item.checkState(0) == Qt.CheckState.Checked
        if on:
            self.current = k
        self.set_steps(self.keys + [k] if on else [x for x in self.keys if x != k])

    def _step_clicked(self, item, _col) -> None:
        if self._just_toggled:                  # the click was on the check box: handled by _step_ticked
            self._just_toggled = False
            return
        self.select_step(item.data(0, Qt.ItemDataRole.UserRole))

    def _list_reordered(self) -> None:
        order = [self.step_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) for i in range(self.step_list.topLevelItemCount())]
        order = [k for k in order if k in ef.STEPS]
        if sorted(order) == sorted(ef.STEPS) and order != self.flow.order:
            self.flow.order = order
            self._store()
        QtCore.QTimer.singleShot(0, self._rebuild_keep_scroll)   # (the list is mid-drop: rebuild after it)

    def move_step(self, delta: int) -> None:
        """The selected step one place earlier (−1) or later (+1) in the process."""
        it = self.step_list.currentItem()
        k = it.data(0, Qt.ItemDataRole.UserRole) if it is not None else self.key
        if k in ef.STEPS and self.flow.move(k, delta):
            self._store()
            self._rebuild_keep_scroll()
            for i in range(self.step_list.topLevelItemCount()):
                if self.step_list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == k:
                    self.step_list.setCurrentItem(self.step_list.topLevelItem(i))

    def _rebuild_keep_scroll(self) -> None:
        area = self.stack.widget(0)
        v = area.verticalScrollBar().value() if isinstance(area, QtWidgets.QScrollArea) else 0
        self.build()
        if isinstance(area, QtWidgets.QScrollArea):
            QtCore.QTimer.singleShot(0, lambda: area.verticalScrollBar().setValue(v))

    def set_steps(self, keys: list[str]) -> None:
        """Show these steps, each in its own card (in process order). A step opened for the first
        time starts from the flakes of the latest earlier step that has them."""
        self.classic.active = False
        self.stack.setCurrentIndex(0)
        keys = self.flow.ordered(keys)
        for k in keys:
            if k not in self.flow.selected:
                self._inherit_flakes(k)
        self.flow.selected = keys
        self.draw_target = None
        if self.current not in keys:
            self.current = keys[0] if keys else None
        self._store()
        self._fill_step()
        self._status_rows()
        if keys:
            self.app.center_tabs.setCurrentWidget(self.app.ebl_page)
            self.redraw(fit=True)

    def select_step(self, key: str | None) -> None:
        """Work on step ``key`` (its card opens, the others stay); ``None``: close every card;
        ``"classic"``: the classic graphene step."""
        if key == CLASSIC:
            self.flow.selected = []
            self.classic.active = True
            self.classic.build()
            self.stack.setCurrentIndex(1)
            self.app.center_tabs.setCurrentWidget(self.app.ebl_page)
            self.classic.redraw(fit=True)
            return
        if key not in ef.STEPS:
            self.current = None
            self.set_steps([])
            return
        self.current = key
        self.collapsed.discard(key)
        self.set_steps(self.keys + ([key] if key not in self.keys else []))
        card = self.cards.get(key, {}).get("step_card")
        area = self.stack.widget(0)
        if card is not None and isinstance(area, QtWidgets.QScrollArea):
            QtCore.QTimer.singleShot(0, lambda: area.ensureWidgetVisible(card, 0, 0))

    def _inherit_flakes(self, key: str) -> None:
        """Flakes do not move between EBL steps: a step without outlines starts from the latest
        earlier step (process order) that has them (relocate or edit if the sample changed)."""
        st = self.flow.step(key)
        if st.outlines:
            return
        order = self.flow.order
        src = next((self.flow.steps[k] for k in reversed(order[:order.index(key)]) if self.flow.steps.get(k) and self.flow.steps[k].outlines),
                   None) or next((s for k, s in self.flow.steps.items() if k != key and s.outlines), None)
        if src is None:
            return
        st.outlines = {u: [list(map(tuple, q)) for q in polys] for u, polys in src.outlines.items()}
        # the stack decides what is on the chip now: a layer only planned at the earlier step but below this step's
        # own layer has been dropped since — it needs locating (the planned position is the starting guess)
        present = set(ef.present_layers(self.app.spec, key))
        st.located = {u: dict(v, how=(ef.EXPECTED_HOW if u in present and v.get("how") in ("planned", "projected") else v.get("how")),
                              inherited=src.key) for u, v in src.located.items()}
        if key != "hall_bars" and not st.twisted and src.twisted:
            st.twisted, st.twisted_how = [list(map(tuple, q)) for q in src.twisted], f"from {ef.STEPS[src.key].name}"
        self.flow.store(self.app.spec)

    # ------------------------------------------------------------------ the card in use
    def _on(self, k: str, fn, *args):
        """A card's button: put that card in use, then run ``fn``."""
        def run(*_):
            self._activate(k)
            return fn(*args)
        return run

    def _activate(self, k: str) -> None:
        if k not in self.keys:
            return
        changed = k != self.current
        self.current = k
        self._bind(k)
        if changed:
            self._mark_current()
            self._status_rows()
            self.redraw()

    def _bind(self, k: str | None) -> None:
        w = self.cards.get(k) or {}
        for name in CARD_ATTRS:
            setattr(self, name, w.get(name, {} if name in ("image_labels", "plan_sliders") else None))

    def _mark_current(self) -> None:
        many = len(self.keys) > 1
        for k, w in self.cards.items():
            card = w.get("step_card")
            if card is None:
                continue
            n = self.flow.order.index(k) + 1
            card.title_label.setText(f"Step {n}: {ef.STEPS[k].name}" + ("   ◀ on the canvas" if many and k == self.key else ""))

    # ------------------------------------------------------------------ the step cards
    def _fill_step(self) -> None:
        """(Re)build one card per shown step, in process order; bind the card in use."""
        if self.app.spec is None or not hasattr(self, "cards_box"):
            return
        try:
            self.cards_box.count()
        except RuntimeError:                      # the panel was rebuilt without a device
            return
        while self.cards_box.count():
            w = self.cards_box.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self.cards = {}
        keys = self.keys
        if self.current not in keys:
            self.current = keys[0] if keys else None
        if not keys:
            empty = Card("Steps")
            empty.layout.addWidget(label("Tick a step above to open its card.", "faint"))
            self.cards_box.addWidget(empty)
            self._bind(None)
            self.step_card = empty
        for k in keys:
            self._build_card(k)
        self._bind(self.key)
        self._mark_current()
        self._fill_canvas_images()

    def _build_card(self, k: str) -> None:
        st = self.flow.step(k)
        kind = ef.STEPS[k]
        folded = k in self.collapsed
        fold = IconButton("chevronDown" if folded else "chevronUp", "Expand this step" if folded else "Collapse this step",
                          size=26, icon_size=16)
        card = Card(f"Step {self.flow.order.index(k) + 1}: {kind.name}", kind.blurb, fold)
        fold.clicked.connect(lambda _=False, key=k: self.toggle_card(key))
        self.cards_box.addWidget(card)
        w = {"step_card": card, "image_labels": {}, "plan_sliders": {}}
        self.cards[k] = w
        if folded:
            card.layout.addWidget(label(f"{STATUS_TEXT[st.status]} · {len(st.images)} image(s) · "
                                        f"{sum(1 for v in st.located.values() if v.get('how') == 'located')} flake(s) located", "muted"))
            modernize(card)
            return
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        card.layout.addWidget(inner)
        on = lambda fn, *a: self._on(k, fn, *a)
        body.addWidget(label(kind.blurb, "muted", wrap=True))
        # images
        body.addWidget(label("Images Of The Sample Up To This Point", "cardTitle"))
        body.addWidget(label("Load what you have; one of 5x / 10x / 20x is needed to fix the chip alignment, higher "
                             "magnifications make it more precise.", "hint", wrap=True))
        for obj in ef.OBJECTIVE_KEYS:
            name = label(Path(st.images.get(obj, "")).name or "—", "muted")
            name.setMinimumWidth(80)
            w["image_labels"][obj] = name
            t = IconButton("trash", f"Delete the {obj} image of this step", size=26, icon_size=16)
            t.clicked.connect(on(self.remove_image, obj))
            body.addLayout(row(obj, button("Load…", on(self.load_image, obj), None, f"Load the {obj} image", "image"),
                               button("View", on(self.view_image, obj), None, "Open the image in its own window"),
                               t, name, width=44))
        # alignment
        body.addWidget(label("Alignment To The Bond Pads", "cardTitle"))
        w["mirror"] = CheckBox("Images mirrored")
        body.addLayout(row("", w["mirror"], width=44))
        body.addLayout(buttons_row(button("Align to bond pads", on(self.align), "primary", "Low magnification to the pads and markers, "
                                          "then each higher magnification onto the previous one"),
                                   button("Manual alignment", on(self.manual_alignment), None, "Place an image on the bond-pad "
                                          "pattern by hand in its own window: move, turn and scale it with handles, or pick "
                                          "pairs of points and fit them")))
        w["align_banner"] = Banner(self._align_text(k), "ok" if st.transforms else "neutral")
        body.addWidget(w["align_banner"])
        # flakes
        body.addWidget(label("Flakes On The Chip", "cardTitle"))
        fl = clickable_rows(QtWidgets.QTreeWidget())
        fl.setHeaderLabels(["Layer", "Where"])
        fl.setRootIsDecorated(False)
        fl.header().setStretchLastSection(False)
        fl.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        fl.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        fl.itemClicked.connect(lambda *_a, key=k: self._activate(key))
        w["flake_list"] = fl
        self._fill_flakes(k, fl)
        fl.setFixedHeight(min(260, 32 + 28 * max(3, fl.topLevelItemCount())))
        body.addWidget(fl)
        body.addLayout(buttons_row(button("Locate flakes", on(self.locate), "primary", "Find each traced flake in the aligned images; "
                                          "the others are placed through the Flake layout"),
                                   button("Draw outline", on(self.draw_outline), None, "Draw or adjust the selected flake's outline "
                                          "on an aligned image, in its own window"),
                                   button("Clear", on(self.clear_outline))))
        # the pattern written at the step before, where it really lies in this step's images (user, 2026-11-20)
        for src in ef.LOCATABLE.get(k, ()):
            what = ef.PATTERN_TITLES.get(src, src)
            body.addWidget(label(self._found_text(k, src), "faint", wrap=True))
            body.addLayout(buttons_row(button(f"Locate the {what}", on(self.locate_pattern, src), "primary",
                                              f"Find the {what} written at '{ef.STEPS[src].name}' in this step's aligned images: this "
                                              "step's pattern is then placed on them as they are"),
                                       button("Use as designed", on(self.unlocate_pattern, src), None,
                                              f"Forget where the {what} were found: use them as designed")))
        # bubbles, cracks, stray flakes: found after the flakes are located; the pattern keeps off them (user, 2026-11-20)
        body.addWidget(label("Bubbles, Cracks And Stray Flakes", "cardTitle"))
        body.addWidget(label(DEBRIS_HINTS.get(k, ""), "muted", wrap=True))
        body.addWidget(label(self._debris_text(k), "faint", wrap=True))
        body.addLayout(buttons_row(button("Find", on(self.find_debris), "primary",
                                          "Look for bubbles, cracks and folds on the stack, stray flakes and particles round it, in "
                                          "every aligned image of this step (they replace those of the earlier steps)"),
                                   button("Draw / edit…", on(self.edit_debris), None, "Check, draw, change or delete them in their "
                                          "own window"),
                                   button("Clear", on(self.clear_debris), None, "Forget the ones of this step (the earlier step's, "
                                          "if any, hold again)")))
        if k == "hall_bars":
            body.addWidget(label("Twisted Region", "cardTitle"))
            how = st.twisted_how or ("not set" if not st.twisted else "")
            body.addWidget(label(f"{len(st.twisted)} polygon(s) · {how}" if st.twisted else "Not set: plan it from the flake images, "
                                 "or take the overlap of the located device-layer halves.", "muted", wrap=True))
            body.addLayout(buttons_row(button("Plan from flake images", on(self.plan_from_flake_images), "primary" if not st.twisted else None,
                                              "The device layers are not on the chip yet: put their traced borders (Flake images) on the "
                                              "bond-pad site, then move and turn them to plan the device; the twisted region is their "
                                              "overlap")))
            body.addLayout(buttons_row(button("Clear", on(self.clear_twisted), None, "Remove the plan and the twisted region")))
            if st.plan or (getattr(self, "_planning", False) and k == self.current):
                self._build_plan_list(body, st, w)
            body.addWidget(label("Finger Array", "cardTitle"))
            p = ef.params_for(st, self.app.config)
            length = (f"each column as long as the bottom hBN allows, ending {p.bbn_margin:g} µm inside it" if p.bbn_margin is not None
                      else "each column as long as the bottom hBN allows (the distance from its boundary is asked on Generate)")
            body.addWidget(label(f"Finger {p.width:g} µm, pitch {p.pitch:g} µm, gap {p.gap:g} µm, {p.min_overlap:g} µm of each tip on "
                                 f"the twisted region; straight {p.straight:g} µm then fanned out ({p.far_pitch:g} µm pitch at {p.length:g} µm, wider on longer fingers; the far ends of a column on one line); {length} "
                                 f"(without a located bottom hBN: {p.length:g} µm).", "muted", wrap=True))
            w["fixed_angle"] = NumberField(st.fixed_angle, "auto", 80)
            w["fixed_angle"].valueEdited.connect(lambda v, s_=st: (setattr(s_, "fixed_angle", v), self._store()))
            w["max_pairs"] = NumberField(st.max_pairs, "all that fit", 80, 0)
            w["max_pairs"].valueEdited.connect(lambda v, s_=st: (setattr(s_, "max_pairs", int(v) if v else None), self._store()))
            body.addLayout(row("Array axis", w["fixed_angle"], label("°", "muted"), width=90))
            body.addLayout(row("Pairs at most", w["max_pairs"], width=90))
        elif k == "contact_gate":
            p = ef.params_for(st, self.app.config)
            body.addWidget(label("Contact-Gate Comb", "cardTitle"))
            gap = f"{p.cg_tmd_gap:g} µm" if p.cg_tmd_gap is not None else "a preset distance (asked)"
            teeth = (st.comb or {}).get("tooth") or {}
            fitted = (" Now: " + ", ".join(f"{c} {v:.2f} µm" for c, v in teeth.items()) + ".") if teeth else ""
            body.addWidget(label(f"One tooth per finger from its tip; the bus ({p.cg_bus_width:g} µm wide, {p.cg_bus_extension:g} µm past "
                                 f"the outer tooth) runs {gap} from the WSe2 (TMD), so each column's teeth are as short as that "
                                 f"allows ({p.cg_tooth:g} µm without a located TMD).{fitted}", "muted", wrap=True))
        body.addLayout(buttons_row(button("Preset dimensions…", on(self.edit_presets), None, "Finger, contact-gate and electrode "
                                          "presets (Edit preset dimensions)")))
        # run
        body.addWidget(label("Pattern", "cardTitle"))
        from .shape_window import DRAW_NAMES
        draw = button(DRAW_NAMES.get(k, "Draw…"), on(self.draw_pattern), None,
                      "Draw or adjust this step's pattern by hand, with the polygon tool, in its own window (Check then checks it "
                      "as drawn)")
        check = button("Check", on(self.check), None, "Every rule of this pattern, met or not, in a window")
        if k == "electrodes":                           # Generate (all) · Draw · Check, then the export (user, 2026-11-20)
            self._electrode_cards(k, st, body, on, draw, check)
        else:
            body.addLayout(buttons_row(button("Generate", on(self.generate), "primary", "Draw this step's pattern on the EBL canvas",
                                              "check"), draw, check))
        body.addWidget(self._flakes_box())
        body.addLayout(buttons_row(button("Export GDS…", on(self.export), "primary" if k == "electrodes" else None,
                                          "Write this step's GDS" if k != "electrodes" else "Write every electrode in one GDS, on one "
                                          "layer", "export")))
        w["pill"] = Pill(self._pill_text(k), "ok" if st.shapes and not any(i["severity"] == "error" for i in st.issues) else "neutral")
        body.addWidget(w["pill"])
        w["issues"] = IssueList()
        w["issues"].setMinimumHeight(140)
        w["issues"].itemClicked.connect(self._issue_clicked)
        body.addWidget(w["issues"])
        self._fill_issues(k, w["issues"])
        modernize(card)

    ELECTRODE_HINTS = {
        "fingers": "On the last part of each finger (at most {overlap:g} µm, starting only where it clears the top hBN and the "
                   "other flakes above the fingers), then to a bond-pad lead. Every electrode is routed round the others, the "
                   "gate flakes and the device, and ends in a square over its lead.",
        "contact_gate": "From the end of each contact-gate bus lead to a bond-pad lead.",
        "top_gate": "A {size:g} µm contact on the top-gate flake where nothing covers it, then an electrode to a bond-pad lead.",
        "bottom_gate": "A {size:g} µm contact on the bottom-gate flake where no hBN covers it, then an electrode to a bond-pad lead.",
    }

    def _electrode_cards(self, k: str, st: ef.FlowStep, body, on, *extra) -> None:
        """Electrodes › Pattern: one card per electrode group the sample has, each with its own Generate (all on one layer)."""
        groups = ef.electrode_groups(self.app.spec)
        if not groups:
            body.addWidget(label("The stack has no Hall-bar contacts, contact gate or gate to connect.", "muted", wrap=True))
            return
        p = ef.params_for(st, self.app.config)
        for g in groups:
            title, purpose, _fn = ef.ELECTRODE_GROUPS[g]
            n = len(st.shapes.get(purpose, []))
            routed = [r for r in st.routes if r.get("group") == g]
            state = (f"{n} shape(s), {sum(r.get('routed_to_pad', False) for r in routed)} to a bond-pad lead" if n else "not generated")
            sub = Card(title)
            sub.layout.addWidget(label(self.ELECTRODE_HINTS[g].format(land=p.land_from_tip, size=p.contact_size, overlap=float(self.app.config.preset("electrodes.finger_overlap_um") or 0)), "muted", wrap=True))
            sub.layout.addWidget(label(state, "faint"))
            sub.layout.addLayout(buttons_row(button("Generate", on(self.generate, g), "primary", f"Draw the {title.lower()} (the other "
                                                    "electrodes stay)", "check"),
                                             button("Clear", on(self.clear_electrodes, g), None, f"Remove the {title.lower()}")))
            body.addWidget(sub)
        body.addLayout(buttons_row(button("Generate all", on(self.generate), "primary", "Draw every group again (the gate electrodes first, "
                                          "so their contacts get a free stretch of the flake)"), *extra))

    def clear_electrodes(self, g: str) -> None:
        st = self.flow.step("electrodes")
        st.shapes.pop(ef.ELECTRODE_GROUPS[g][1], None)
        st.routes = [r for r in st.routes if r.get("group") != g]
        st.group_notes.pop(g, None)
        st.notes = [n for gg in ef.ELECTRODE_GROUPS for n in st.group_notes.get(gg, [])]
        st.exported = None
        if st.shapes:
            ef.check(self.app.spec, self.flow, "electrodes", self.app.config)
        else:
            st.issues = []
        self._store()
        self._fill_step()
        self._fill_merge()
        self._status_rows()
        self.redraw()

    def toggle_card(self, k: str) -> None:
        """Fold / unfold a step's card (the ^ at its top right)."""
        if k in self.collapsed:
            self.collapsed.discard(k)
        else:
            self.collapsed.add(k)
        self._fill_step()

    # ------------------------------------------------------------------ the image on the canvas
    def _fill_canvas_images(self) -> None:
        cb = getattr(self, "canvas_combo", None)
        if cb is None:
            return
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("Automatic (step in use)", None)
        for k in self.flow.order:
            st = self.flow.steps.get(k)
            if st is None:
                continue
            for obj in ef.OBJECTIVE_KEYS:
                if obj in st.images:
                    cb.addItem(f"{ef.STEPS[k].name} — {obj}" + ("" if obj in st.transforms else "  (not aligned)"), [k, obj])
        want = self.flow.canvas_image
        i = next((n for n in range(cb.count()) if cb.itemData(n) == want), 0) if want else 0
        cb.setCurrentIndex(i)
        cb.blockSignals(False)

    def _canvas_image_chosen(self) -> None:
        d = self.canvas_combo.currentData()
        self.flow.canvas_image = list(d) if d else None
        self._store()
        self.redraw(fit=True)

    def _canvas_choice(self) -> tuple[str, str] | None:
        """(step, objective) of the image the canvas shows."""
        ci = self.flow.canvas_image
        if ci and ci[0] in self.flow.steps and ci[1] in self.flow.steps[ci[0]].images:
            return ci[0], ci[1]
        st = self.step
        if st is not None and st.show_image in st.images:
            return st.key, st.show_image
        return None

    def _pill_text(self, k: str | None = None) -> str:
        st = self.flow.step(k or self.key) if (k or self.key) else None
        if st is None or not st.shapes:
            return "Not generated"
        n = {sev: sum(1 for i in st.issues if i["severity"] == sev) for sev in ("error", "warning")}
        out = f"{sum(len(v) for v in st.shapes.values())} shapes · {n['error']} errors · {n['warning']} warnings"
        return out + (f" · exported {Path(st.exported).name}" if st.exported else "")

    def _fill_issues(self, k: str | None = None, issues=None) -> None:
        k = k or self.key
        issues = issues if issues is not None else self.issues
        if issues is None or k is None:
            return
        issues.clear()
        st = self.flow.step(k)
        for i in st.issues:
            issues.add(i["severity"], i["message"], i.get("where"))
        for n in st.notes:
            issues.add("note", n)

    def _fill_flakes(self, k: str | None = None, flake_list=None) -> None:
        k = k or self.key
        fl = flake_list if flake_list is not None else self.flake_list
        if fl is None or k is None:
            return
        st, spec = self.flow.step(k), self.app.spec
        fl.clear()
        for layer in reversed(spec.layers[1:]):
            if not layer.outline and layer.uid not in st.outlines:
                continue
            how, _on_chip = ef.flake_state(spec, st, k, layer.uid)
            it = QtWidgets.QTreeWidgetItem(fl, [layer.label, how])
            it.setData(0, Qt.ItemDataRole.UserRole, layer.uid)

    def _fill_merge(self) -> None:
        body = self.merge_card.layout
        clear_layout(body)
        self.merge_boxes = {}
        for k in self.flow.order:
            kind = ef.STEPS[k]
            st = self.flow.steps.get(k)
            cb = CheckBox(kind.name + ("" if st and st.shapes else "  (not generated)"))
            cb.setChecked(bool(st and st.shapes))
            cb.setEnabled(bool(st and st.shapes))
            self.merge_boxes[k] = cb
            body.addWidget(cb)
        self.merge_pads = CheckBox("Include the bond pads, markers and landing square")
        self.merge_pads.setChecked(True)
        body.addWidget(self.merge_pads)
        body.addWidget(self._flakes_box())
        body.addLayout(buttons_row(button("Export merged GDS…", self.export_merged, "primary", "One GDS with every chosen step on its "
                                          "own layer", "export")))
        modernize(self.merge_card)

    def _align_text(self, k: str | None = None) -> str:
        st = self.flow.steps.get(k or self.key) if (k or self.key) else None
        if st is None or not st.transforms:
            return "Not aligned. Load the images, then Align to bond pads (or Manual alignment)."
        parts = []
        for obj, t in sorted(st.transforms.items(), key=lambda kv: ef.mag(kv[0])):
            rms = t.get("rms_um")
            parts.append(f"{obj}: {t['um_per_px'] * 1000:.2f} nm/px, {t['angle_deg']:.2f}°" + (f", ±{rms:.2f} µm" if rms is not None else "")
                         + (f" ({st.align_notes[obj]})" if obj in st.align_notes else ""))
        for obj, note in st.align_notes.items():
            if obj not in st.transforms:
                parts.append(f"{obj}: {note}")
        return "Aligned · " + " · ".join(parts)

    # ------------------------------------------------------------------ bond pads
    def _choose_bondpads(self) -> None:
        self.flow.bondpads = self.bp_combo.currentData()
        self._store()
        self.build()
        self.redraw(fit=True)

    def save_bondpad_preset(self) -> None:
        path = open_file(self, "Bond-pad GDS", "GDS (*.gds *.gds2 *.gdsii *.oas)")
        if not path:
            return
        try:
            polys = read_polygons(path)
        except Exception as exc:
            self.app.warn("Bond pads", str(exc))
            return
        dlg = Dialog(self.app)
        dlg.setWindowTitle("Save bond-pad preset")
        dlg.resize(520, 320)
        name = QtWidgets.QLineEdit(Path(path).stem)
        dlg.body.addLayout(row("Name", name))
        combos = {}
        for role, cap in ROLES.items():
            cb = ComboBox()
            cb.addItem("—", None)
            for (l, d), ps in sorted(polys.items()):
                cb.addItem(f"{l}/{d}  ({len(ps)} shapes)", (l, d))
            combos[role] = cb
            dlg.body.addLayout(row(cap, cb, stretch=False, width=150))
        sx, sy = NumberField(0.0, "x µm", 80), NumberField(0.0, "y µm", 80)
        dlg.body.addLayout(row("Device site", sx, sy, width=150))
        dlg.body.addStretch(1)
        dlg.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Cancel", dlg.reject), button("Save", dlg.accept, "primary")))
        modernize(dlg)
        if not dlg.exec():
            return
        layers = {r: cb.currentData() for r, cb in combos.items() if cb.currentData() is not None}
        if "pads" not in layers:
            self.app.warn("Bond pads", "Choose the layer with the bond pads and fan-out.")
            return
        bp = save_preset(name.text().strip() or Path(path).stem, path, layers, (sx.value() or 0.0, sy.value() or 0.0),
                         source=f"Saved in UStack from {Path(path).name}")
        self.flow.bondpads = bp.name
        self._bp_cache.pop(bp.name, None)
        self._store()
        self.build()
        self.redraw(fit=True)

    # ------------------------------------------------------------------ images
    def _image(self, obj: str, key: str | None = None):
        key = key or self.key
        st = self.flow.steps.get(key)
        if st is None or obj not in st.images:
            return None
        k = (key, obj)
        if k not in self.images and Path(st.images[obj]).is_file():
            try:
                self.images[k] = imaging.load_image(st.images[obj])
            except Exception:
                return None
        return self.images.get(k)

    def _nominal_scale(self, obj: str, img) -> float | None:
        return self.app.config.um_per_px(obj, img.shape[1])

    def load_image(self, obj: str, path: str | None = None) -> bool:
        st = self.step
        path = path or open_file(self, f"{obj} image", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not path or st is None:
            return False
        try:
            img = imaging.load_image(path)
        except Exception as exc:
            self.app.warn("Image", str(exc))
            return False
        dest = self.app.device_dir() / "images" / f"{self.key}_{obj}{Path(path).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)
        st.images[obj] = str(dest)
        st.transforms.pop(obj, None)
        self.images[(self.key, obj)] = img
        if not st.show_image or ef.mag(obj) > ef.mag(st.show_image):
            st.show_image = obj
        self._store()
        self._fill_step()
        self._status_rows()
        self.redraw(fit=True)
        return True

    def view_image(self, obj: str) -> None:
        img = self._image(obj)
        if img is None:
            self.app.status(f"No {obj} image in this step — press Load… first")
            return
        from .viewer import ImageViewerDialog
        st = self.step
        s = self._nominal_scale(obj, img)
        dlg = ImageViewerDialog(img, f"{ef.STEPS[self.key].name} — {obj}", f"{Path(st.images[obj]).name} · {img.shape[1]} × {img.shape[0]} px",
                                s, self.app)
        self._viewers = [v for v in self._viewers if v.isVisible()] + [dlg]
        dlg.show()

    def remove_image(self, obj: str) -> None:
        st = self.step
        path = st.images.pop(obj, None)
        st.transforms.pop(obj, None)
        self.images.pop((self.key, obj), None)
        if path:
            p = Path(path)
            others = [s for k, s in self.flow.steps.items() if k != self.key and path in s.images.values()]
            try:
                if not others and p.is_file() and p.resolve().parent == (self.app.device_dir() / "images").resolve():
                    p.unlink()
            except OSError:
                pass
        if st.show_image == obj:
            st.show_image = max(st.images, key=ef.mag) if st.images else None
        self._store()
        self._fill_step()
        self._status_rows()
        self.redraw()

    # ------------------------------------------------------------------ alignment
    def align(self) -> bool:
        """Align every loaded image of the step to the bond pads. True when at least one is aligned."""
        st, bp = self.step, self.bondpads()
        if st is None or not st.images:
            self.app.warn("Align", "Load the images of this step first.")
            return False
        if bp is None:
            self.app.warn("Align", "Choose a bond-pad preset first.")
            return False
        if not any(ef.mag(o) <= 20 for o in st.images):
            self.app.warn("Align", "Load a 5x, 10x or 20x image: it fixes the position on the chip (the bond pads and markers).")
            return False
        from .. import registration
        imgs = {}
        for obj in st.images:
            img = self._image(obj)
            s = self._nominal_scale(obj, img) if img is not None else None
            if img is not None and s:
                imgs[obj] = (img, s)
        layers = [tuple(v) for r, v in bp.layers.items() if r in ("pads", "markers")]
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = registration.align_images(imgs, bp.polys, layers, bp.site, self.mirror.isChecked())
        except ValueError as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.app.warn("Align", f"{exc}\n\nUse Manual alignment to align by hand.")
            return False
        finally:
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()
        st.transforms = {k: v.to_json() for k, v in out.items()}
        notes = {k: str(v) for k, v in out.errors.items()}
        for k, v in out.items():
            log = out.log.get(k, {})
            how = {"gds": "on the bond pads", "chain": f"on the {log.get('ref')} image", "chain+gds": f"on the {log.get('ref')} image, refined on the bond pads"}
            notes[k] = f"{how.get(log.get('method'), log.get('method') or '')}, match {getattr(v, 'score', float('nan')):.2f}"
        st.align_notes = notes
        self._store()
        self._fill_step()
        self.redraw(fit=True)
        if out.errors:
            self.app.warn("Align", f"Aligned {len(out)} of {len(imgs)} images.\n\n" + "\n".join(f"{k}: {v}" for k, v in out.errors.items())
                          + "\n\nAlign those with Manual alignment, or remove them.")
        else:
            self.app.status(f"Aligned all {len(out)} images to the bond pads.")
        return len(out) > 0

    # ------------------------------------------------------------------ patterns written at the step before
    def _found_text(self, k: str, src: str) -> str:
        what = ef.PATTERN_TITLES.get(src, src).capitalize()
        r = self.flow.step(k).found_patterns.get(src)
        if r and r.get("as_designed"):
            return f"{what} (written at '{ef.STEPS[src].name}'): used as designed (your choice) — the bubble finder and this step's " \
                   "pattern take them where they were designed."
        if not r:
            off = ef.pattern_offset(self.flow, k, src)
            if off:
                return f"{what}: as found at an earlier step ({math.hypot(off['dx'], off['dy']):.2f} µm from the design)."
            return f"{what} (written at '{ef.STEPS[src].name}'): as designed — not located in this step's images."
        return (f"{what} found in the {r.get('image', '')} image: {math.hypot(r['dx'], r['dy']):.2f} µm from the design "
                f"(x {r['dx']:+.2f}, y {r['dy']:+.2f} µm), turned {r['angle']:+.2f}° — {r.get('inliers', 0):.0%} of the outline on "
                "the image's edges. This step's pattern is placed on them (the move includes that image's own alignment "
                "error: align a 50x / 100x image well first).")

    def locate_pattern(self, src: str) -> None:
        st = self.step
        if st is None:
            return
        imgs = self._debris_images()
        if not imgs:
            self.app.warn("Locate", "Align at least one image of this step to the bond pads first.")
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ef.locate_written(self.flow, self.key, src, imgs)
        except ValueError as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.app.warn("Locate", f"The {ef.PATTERN_TITLES.get(src, src)} were not found: {exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._store()
        self._fill_step()
        self.redraw()
        self.app.status(self._found_text(self.key, src) + " Generate again to place this step's pattern on them.")

    def unlocate_pattern(self, src: str) -> None:
        """Use as designed: the pattern is taken to lie where it was designed — at this step and after it (the bubble finder
        masks it there, the next patterns are placed on it there), whatever an earlier step found."""
        st = self.step
        if st is None:
            return
        st.found_patterns[src] = {"as_designed": True}
        self._store()
        self._fill_step()
        self.redraw()
        self.app.status(f"The {ef.PATTERN_TITLES.get(src, src)} are taken as designed (Find the bubbles again to use that).")

    # ------------------------------------------------------------------ an error's red circle
    def issue_popup(self, key: str, x: float, y: float) -> None:
        """A click on a red circle: what error it is, in a small pop-up — Close, Change rule… (the lab value behind it),
        or Accept (this place is fine: the circle goes, the error no longer stops the export)."""
        from .issue_popup import IssuePopup
        try:
            step_key, n, m = key.split("|")
            st = self.flow.step(step_key)
            issue = st.issues[int(n)]
            pt = tuple(issue["where"][int(m)])
        except (ValueError, IndexError, KeyError):
            return
        old = getattr(self, "_issue_popup", None)
        if old is not None:
            old.close()
        self._issue_popup = IssuePopup(self, step_key, issue, pt)
        self._issue_popup.show_at(QtGui.QCursor.pos())

    def accept_issue(self, step_key: str, issue: dict, pt) -> None:
        st = self.flow.step(step_key)
        ef.accept_issue(st, issue, pt)
        self._store()
        self._fill_step()
        self.redraw()
        self.app.status(f"Accepted at ({pt[0]:.1f}, {pt[1]:.1f}) µm: {issue.get('message', '')}")

    def change_rule(self, step_key: str, item_key: str) -> None:
        """Edit the lab value behind an error (saved to config/, as yours), then check the step again."""
        from ..config import MissingItem
        kind, _, name = item_key.partition(":")
        cfg = self.app.config
        if kind == "rule":
            rule = cfg.rules.get(name)
            label_, unit, now = (rule.description if rule else name), (rule.unit if rule else ""), cfg.value(name)
        else:
            meta = cfg.presets.get(name, {}) if hasattr(cfg, "presets") else {}
            label_, unit, now = meta.get("label", name), meta.get("unit", ""), cfg.preset(name)
        item = MissingItem(item_key, f"{label_} ({name})", unit.replace("um", "µm"), hint=f"Now: {now if now is not None else 'not set'}",
                           suggested=now)
        dlg = MissingValuesDialog([item], self.app, "Change rule", "The value is saved to config/ as yours; the step is checked again.")
        if not dlg.exec():
            return
        apply_answers(cfg, dlg.answers())
        ef.check(self.app.spec, self.flow, step_key, cfg)
        self._store()
        self._fill_step()
        self.redraw()

    # ------------------------------------------------------------------ bubbles, cracks, stray flakes
    def _debris_text(self, key: str | None = None) -> str:
        key = key or self.key
        src = ef.debris_source(self.flow, key)
        if src is None:
            return "None looked for yet."
        items = self.flow.steps[src].debris
        from collections import Counter
        from ..debris import KIND_NAMES
        n = Counter(d.get("kind") for d in items)
        hand = sum(d.get("how") != "auto" for d in items)
        text = (" · ".join(f"{KIND_NAMES.get(k, k)}: {v}" for k, v in sorted(n.items())) or "None found") + \
            (f" ({hand} by hand)" if hand else "")
        if src != key:
            text += f" — from '{ef.STEPS[src].name}' (not looked for at this step yet)"
        return text

    def _debris_images(self, key: str | None = None):
        st = self.flow.step(key or self.key)
        out = []
        for o in sorted(st.transforms, key=ef.mag):
            img = self._image(o, st.key)
            if img is not None:
                out.append((o, img, Similarity.from_json(st.transforms[o])))
        return out

    def find_debris(self) -> None:
        from .. import debris
        st = self.step
        if st is None:
            return
        imgs = self._debris_images()
        if not imgs:
            self.app.warn("Bubbles, cracks, flakes", "Align at least one image of this step to the bond pads first.")
            return
        metal, stack, region = ef.debris_inputs(self.app.spec, self.flow, self.key, self.bondpads())
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.app.status("Looking for bubbles, cracks and stray flakes…")
            QtWidgets.QApplication.processEvents()
            found = debris.find_in_step(imgs, metal, stack, region)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        ef.set_debris(st, found)
        self._store()
        self._fill_step()
        self.redraw()
        msg = f"Found {len(found)}: {self._debris_text()}. The pattern keeps off them (Generate again)."
        missing = [l.label for l in self.app.spec.layers[1:] if l.uid in ef.present_layers(self.app.spec, self.key)
                   and not st.outlines.get(l.uid)]
        if missing:                                      # a layer on the chip but not located looks like a stray flake
            self.app.warn("Bubbles, cracks, flakes", msg + "\n\nNot located at this step: " + ", ".join(missing)
                          + ". A layer that is not located can be taken for stray flakes or cracks: locate it (or draw its "
                          "outline), then Find again.")
        else:
            self.app.status(msg)

    def edit_debris(self) -> None:
        from .debris_window import DebrisDialog
        if self.step is None or not self._debris_images():
            self.app.warn("Bubbles, cracks, flakes", "Align at least one image of this step to the bond pads first.")
            return
        st = self.step
        if not st.debris_looked:                         # edited here: this step's own from now on (from the earlier step's)
            st.debris = copy.deepcopy(ef.debris_at(self.flow, self.key))
            st.debris_looked = True
            self._store()
        dlg = getattr(self, "_debris_dlg", None)
        if dlg is not None and dlg.isVisible() and dlg.key == self.key:
            dlg.raise_()
            return
        self._debris_dlg = DebrisDialog(self, self.key)
        self._debris_dlg.show()

    def clear_debris(self) -> None:
        st = self.step
        if st is None or not (st.debris or st.debris_looked):
            return
        st.debris = []
        st.debris_looked = False
        self._store()
        self._fill_step()
        self.redraw()

    def manual_alignment(self) -> None:
        """Manual alignment window (user, 2026-11-20): the bond-pad pattern and one image of the step, placed by hand."""
        from .align_window import AlignDialog
        if self.step is None or not self.step.images:
            self.app.warn("Manual alignment", "Load an image of this step first.")
            return
        dlg = getattr(self, "_align_dlg", None)
        if dlg is not None and dlg.isVisible() and dlg.key == self.key:
            dlg.raise_()
            return
        self._align_dlg = AlignDialog(self, self.key)
        self._align_dlg.show()

    def start_pairs(self) -> None:
        shown = self._canvas_choice()
        if not self.step or shown is None or shown[0] != self.key:
            self.app.warn("Pick pairs", "Show one of this step's images on the canvas first (Image on canvas).")
            return
        self._pair_obj = shown[1]
        self.pairs = []
        self.pick = "pair_img"
        self.app.set_ebl_tool("pick")
        self.app.status("Click a feature in the IMAGE, then the same feature in the bond-pad GDS")

    def fit_pairs(self) -> None:
        st = self.step
        if len([p for p in self.pairs if "um" in p]) < 2:
            self.app.warn("Pick pairs", "Pick at least two complete pairs (image point, then GDS point).")
            return
        obj = getattr(self, "_pair_obj", None) or st.show_image
        sim = fit_similarity([p["px"] for p in self.pairs if "um" in p], [p["um"] for p in self.pairs if "um" in p], self.mirror.isChecked())
        st.transforms[obj] = sim.to_json()
        st.align_notes.pop(obj, None)
        self.pick = None
        self.app.set_ebl_tool("pointer")
        self._store()
        self._fill_step()
        self.redraw()

    def _placement(self, obj: str, img, key: str | None = None):
        st = self.flow.step(key or self.key)
        if obj in st.transforms:
            t = Similarity.from_json(st.transforms[obj])
            return (t.scale, t.angle_deg, t.tx, t.ty, t.mirror)
        s = self._nominal_scale(obj, img) or 0.1
        bp = self.bondpads()
        cx, cy = bp.site if bp else (0.0, 0.0)
        return (s, 0.0, cx - img.shape[1] * s / 2, cy + img.shape[0] * s / 2, False)

    # ------------------------------------------------------------------ flakes
    def _best_aligned(self):
        st = self.step
        objs = sorted((o for o in st.transforms if self._image(o) is not None), key=ef.mag, reverse=True)
        return objs[0] if objs else None

    def locate(self) -> None:
        st, spec = self.step, self.app.spec
        if st is None:
            return
        # images loaded since the last alignment (e.g. a 50x added to an aligned 5x): align them first
        unaligned = [o for o in st.images if o not in st.transforms and self._image(o) is not None]
        if unaligned:
            self.app.status(f"Aligning the {', '.join(unaligned)} image{'s' if len(unaligned) > 1 else ''} first…")
            QtWidgets.QApplication.processEvents()
            self.align()
            st = self.step
        obj = self._best_aligned()
        if obj is None:
            self.app.warn("Locate flakes", "Align at least one image of this step to the bond pads first.")
            return
        from .. import registration
        # flakes need resolution: 20x and finer first (a graphite strip longer than the 100x field is found at 20x);
        # with only 5x / 10x images aligned, those are used (coarse: a flake is a few tens of px there)
        aligned = [o for o in st.transforms if self._image(o) is not None]
        objs = sorted((o for o in aligned if ef.mag(o) >= LOCATE_MIN_MAG), key=ef.mag, reverse=True)
        coarse = not objs
        if coarse:
            objs = sorted(aligned, key=ef.mag, reverse=True)
        present = ef.present_layers(spec, self.key)
        # flakes already found at an earlier step keep that position (user, 2026-11-20: a buried flake may no longer be
        # visible); only the flakes not located yet are looked for — Clear a flake to look for it again
        reused = ef.reuse_earlier_locations(spec, self.flow, self.key)
        traced = [l for l in spec.layers if l.uid in present and l.outline and l.uid not in reused
                  and not ef.located_here(st, l.uid)]
        if not traced:
            if reused:
                self._sync_orientation()
                self._store()
                self._fill_step()
                self.redraw()
                self.app.status(f"Every flake on the chip is located; reused from earlier steps: "
                                f"{', '.join(spec.layer(u).label for u in reused)}.")
            elif any(ef.located_here(st, u) for u in present):
                self.app.status("Every flake on the chip at this step is located already (Clear one to locate it again).")
            else:
                self.app.warn("Locate flakes", "No flake of this step has a traced border: trace them in the Flake images tab.")
            return
        bp = self.bondpads()
        found, failed = 0, []
        # flakes dropped since the previous step: found in what changed between its images and these
        # (works when they are thin and covered — WSe2 under a middle hBN — where the photograph alone fails)
        prev = ef.previous_imaged_step(self.flow, self.key)
        new = set(ef.new_layers(spec, self.key, prev))
        cm, cm_note = None, ""
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for layer in traced:
                if layer.uid in new:
                    if cm is None and not cm_note:
                        self.app.status(f"Comparing with the images of '{ef.STEPS[prev].name}'…")
                        QtWidgets.QApplication.processEvents()
                        cm, cm_note = self._change_map(prev, [spec.layer(u) for u in new if spec.layer(u).outline])
                    if cm is not None:
                        self.app.status(f"Locating {layer.label} in what changed since '{ef.STEPS[prev].name}'… ({found} found so far)")
                        QtWidgets.QApplication.processEvents()
                        try:
                            r = registration.locate_new_outline(cm[0], [tuple(p) for p in layer.outline], True, bp.site if bp else None,
                                                                LOCATE_RADIUS_UM, frame_um=self._flake_frame(layer))
                            st.outlines[layer.uid] = [[tuple(map(float, q)) for q in r["polygon"]]]
                            st.located[layer.uid] = {"how": "located", "score": float(r["score"]), "image": cm[1],
                                                     "method": f"change since {ef.STEPS[prev].name}"}
                            found += 1
                            continue
                        except ValueError:
                            pass                          # fall back to the photograph itself
                # finest image first; a flake larger than its field (graphite strips) is found at lower magnification
                best, errs = None, []
                for o in objs:
                    self.app.status(f"Locating {layer.label} in the {o} image… ({found} found so far)")
                    QtWidgets.QApplication.processEvents()
                    try:
                        r = registration.locate_outline(self._image(o), Similarity.from_json(st.transforms[o]),
                                                        [tuple(p) for p in layer.outline], True, bp.site if bp else None, LOCATE_RADIUS_UM,
                                                        frame_um=self._flake_frame(layer))
                    except ValueError as exc:
                        errs.append(f"{o}: {exc}")
                        continue
                    if best is None or r["score"] > best[0]["score"]:
                        best = (r, o)
                    if r["score"] >= 0.6:
                        break
                if best is None:
                    failed.append(f"{layer.label}: " + "; ".join(errs))
                    continue
                r, o = best
                st.outlines[layer.uid] = [[tuple(map(float, q)) for q in r["polygon"]]]
                st.located[layer.uid] = {"how": "located", "score": float(r["score"]), "image": o}
                found += 1
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        rec = self._sync_orientation()
        # only layers on the chip at this step and not found are placed through the Flake layout (dashed, "projected");
        # layers above this step's own layer are not there yet: never projected
        T = ef.project_flakes(spec, st, [u for u in ef.present_layers(spec, self.key)])
        self._store()
        self._fill_step()
        self.redraw()
        msg = f"Located {found} of {len(traced)} flakes"
        if reused:
            msg += f" (reused from earlier steps: {', '.join(spec.layer(u).label for u in reused)})"
        if T is not None and T.get("rejected"):
            msg += (f"; the located flakes do not agree with their Flake layout positions (±{T['rms_um']:.1f} µm), so the others "
                    "were not placed — check Flake layout, or draw them")
        elif T is not None:
            msg += f"; the other flakes placed in Flake layout follow (fit ±{T['rms_um']:.2f} µm)"
        else:
            msg += "; the others are not placed (place the flakes on each other in Flake layout to carry them over)"
        if rec is not None:
            msg += ". " + self._orientation_text(rec)
        if coarse and found:
            msg += (f". Found in the {', '.join(objs)} image only: at that magnification the outlines are coarse (5x: off by ~1–2 µm on the lab's Stack 1) — "
                    f"check them, or add a {LOCATE_MIN_MAG:g}x or finer image and Locate again")
        if failed:
            self.app.warn("Locate flakes", msg + ".\n\nNot found:\n" + "\n".join(failed) + "\n\nDraw those outlines by hand (Draw outline).")
        else:
            self.app.status(msg + ".")

    def _sync_orientation(self) -> dict | None:
        """Flakes found on the chip → their relative positions / orientations in Flake layout and the 3D stack."""
        rec = ef.orientation_from_chip(self.app.spec, self.step, self.key)
        if rec is not None and any(v["changed"] for v in rec["layers"].values()):
            self.app.spec_changed("image")
        return rec

    def _orientation_text(self, rec: dict) -> str:
        spec = self.app.spec
        ref = spec.layer(rec["reference"])
        moves = [f"{spec.layer(u).label} turned {v['turned_deg']:+.1f}°, moved {v['moved_um']:.1f} µm" + (" (mirrored)" if v["mirrored"] else "")
                 for u, v in rec["layers"].items() if v["changed"]]
        if not moves:
            return f"Flake layout already matches the chip ({len(rec['layers'])} flakes)"
        return (f"Flake layout and the 3D stack updated from the chip, relative to {ref.label}: " + "; ".join(moves))

    def _change_map(self, prev: str, layers) -> tuple[tuple | None, str]:
        """The change map between step ``prev``'s image and this step's, at the finest magnification
        ≥ 20x both have aligned (else the finest of each). Returns ((ChangeMap, objective), "") or (None, why)."""
        from .. import registration
        st, ps = self.step, self.flow.step(prev)
        mine = [o for o in st.transforms if self._image(o) is not None]
        theirs = [o for o in ps.transforms if self._image(o, prev) is not None]
        if not mine or not theirs:
            return None, "no aligned image"
        common = sorted((o for o in mine if o in theirs and ef.mag(o) >= LOCATE_MIN_MAG), key=ef.mag, reverse=True)
        o_now, o_prev = (common[0], common[0]) if common else (max(mine, key=ef.mag), max(theirs, key=ef.mag))
        bp = self.bondpads()
        cx, cy = bp.site if bp else Similarity.from_json(st.transforms[o_now]).apply([[0, 0]])[0]
        reach = LOCATE_RADIUS_UM + 0.6 * max([2 * max(math.hypot(x - geo.centroid(l.outline)[0], y - geo.centroid(l.outline)[1])
                                                       for x, y in l.outline) for l in layers] or [0.0])
        try:
            cm = registration.change_map(self._image(o_now), Similarity.from_json(st.transforms[o_now]), self._image(o_prev, prev),
                                         Similarity.from_json(ps.transforms[o_prev]), (cx - reach, cy - reach, cx + reach, cy + reach))
        except ValueError as exc:
            return None, str(exc)
        return (cm, o_now), ""

    def _flake_frame(self, layer):
        """The field of the flake's own image in the design frame (x0, y0, x1, y1): border points on
        that image's edge are not the flake's edge."""
        img = layer.image or {}
        arr = self.app.layer_image(layer.uid)
        if arr is None or not img.get("um_per_px"):
            return None
        h, w = arr.shape[:2]
        s, a = img["um_per_px"], math.radians(img.get("angle_deg", 0.0))
        sx = -1.0 if img.get("mirror") else 1.0
        pts = [(img.get("x", 0.0) + s * (math.cos(a) * sx * x + math.sin(a) * y), img.get("y", 0.0) + s * (math.sin(a) * sx * x - math.cos(a) * y))
               for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
        return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))

    def _selected_flake(self) -> str | None:
        it = self.flake_list.currentItem() if self.flake_list is not None else None
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def draw_outline(self) -> None:
        uid = self._selected_flake()
        if uid is None:
            self.app.warn("Draw outline", "Select the flake in the list first.")
            return
        if self.outline_dialog is None:
            self.outline_dialog = OutlineDialog(self)
        self.outline_dialog.set_flake(uid)
        self.outline_dialog.show()
        self.outline_dialog.raise_()

    def set_outline(self, uid: str, pts: list, how: str) -> None:
        """A flake's outline on the chip changed (outline window, canvas): store it. A planned layer
        moved as a whole stays planned (its plan follows); reshaped, it is an outline of its own."""
        st, spec = self.step, self.app.spec
        old = (st.outlines.get(uid) or [None])[0]
        d = ef.translation_of(old, pts) if old else None
        # reshaped (not moved): the traced border takes the same shape (Flake images, Flake layout, the 3D stack, other steps)
        design = ef.design_from_chip_edit(spec, st, uid, old, pts) if (how == "edited" and old and d is None
                                                                                    and not ef.is_rigid_move(old, pts)) else None
        if uid in st.plan:
            if d is not None:
                st.plan[uid]["cx"] += d[0]
                st.plan[uid]["cy"] += d[1]
            elif design is not None:
                cx, cy = geo.centroid(pts)            # the planned outline built from the new border is exactly ``pts``
                st.plan[uid]["cx"], st.plan[uid]["cy"] = float(cx), float(cy)
            else:
                st.plan.pop(uid)
        st.outlines[uid] = [list(map(tuple, pts))]
        if uid not in st.plan:
            st.located[uid] = {"how": how, "score": None}
        if design is not None:
            layer = spec.layer(uid)
            old_design = [list(p) for p in layer.outline]
            layer.outline = design
            if layer.image:
                layer.image.setdefault("trace", {})["how"] = "edited"
            self.app.outline_synced(uid)
            for other in self.flow.steps.values():
                if other is not st:
                    ef.follow_design_edit(spec, other, uid, old_design)
        ef.apply_plan(spec, st)
        if uid in st.plan and d is not None:
            self._layout_follows_plan(uid)            # the plan moved: Flake layout follows (linked)
        rec = self._sync_orientation() if how in ef.IDENTIFIED_HOW else None
        self._store()
        if rec is not None and any(v["changed"] for v in rec["layers"].values()):
            self.app.status(self._orientation_text(rec))
        if design is not None:
            self.app.traced_borders_changed([uid])
        if self.flake_list is not None:
            self._fill_flakes()
        self.redraw()

    def design_outline_changed(self, uid: str, old_design: list) -> None:
        """``uid``'s traced border was reshaped elsewhere: its outline at every step follows."""
        spec = self.app.spec
        changed = [ef.follow_design_edit(spec, st, uid, old_design) for st in self.flow.steps.values()]
        if any(changed):
            self.flow.store(spec)
            if self.active:
                if self.flake_list is not None:
                    self._fill_flakes()
                self.redraw()
            if self.outline_dialog is not None and self.outline_dialog.isVisible() and self.outline_dialog.uid == uid:
                self.outline_dialog.redraw()

    def _start_draw(self, target: str) -> None:
        self.draw_target = target
        self.app.set_ebl_tool("draw")
        self.app.status("Click the corners on the EBL canvas; double-click to close")

    def _selected_flakes(self) -> list[str]:
        """The flakes chosen in the list (Shift / Ctrl-click: several)."""
        if self.flake_list is None:
            return []
        uids = [it.data(0, Qt.ItemDataRole.UserRole) for it in self.flake_list.selectedItems()]
        return [u for u in uids if u] or ([u] if (u := self._selected_flake()) else [])

    def clear_outline(self, uid: str | None = None) -> None:
        uids = [uid] if uid else self._selected_flakes()
        if uids:
            for u in uids:
                self.step.outlines.pop(u, None)
                self.step.located[u] = {"relocate": True}    # Locate looks for it in this step's images, not at an earlier step's
                self.step.plan.pop(u, None)
            self._store()
            self._fill_step()
            self.redraw()

    def twisted_from_flakes(self) -> None:
        st = self.step
        region = ef.twisted_region(self.app.spec, st.outlines)
        if not region:
            self.app.warn("Twisted region", "Locate (or draw) both device-layer halves first.")
            return
        st.twisted, st.twisted_how = region, "overlap of the device-layer outlines"
        self._store()
        self._fill_step()
        self.redraw()

    def clear_twisted(self) -> None:
        st = self.step
        for uid in list(st.plan):                       # a plan goes with its twisted region
            st.outlines.pop(uid, None)
            st.located.pop(uid, None)
        st.plan = {}
        self._planning = False
        st.twisted, st.twisted_how = [], "cleared"        # stays empty until From located flakes / a plan
        self._store()
        self._fill_step()
        self.redraw()

    # ------------------------------------------------------------------ planning from the flake images
    def plan_from_flake_images(self) -> None:
        """Hall bars go on the bottom gate before the device layers are dropped: plan them from their
        traced borders. Each device layer's border goes on the bond-pad site (relative positions from
        Flake layout kept); move it by its line on the canvas, turn it with its slider."""
        st, spec = self.step, self.app.spec
        bp = self.bondpads()
        site = bp.site if bp is not None else (0.0, 0.0)
        uids = ef.plan_layers(spec)
        if not uids:
            self.app.warn("Plan from flake images", "This device has no device layer.")
            return
        traced = [u for u in uids if spec.layer(u).outline]
        for uid in traced:
            if uid not in st.plan:
                st.plan[uid] = ef.default_plan(spec, uid, site)
                ef.plan_from_layout(spec, self.flow, uid)   # where Flake layout has it, once the chip sets the frame (linked)
        if st.twisted_how not in ("", ef.PLAN_HOW, ef.TWISTED_AUTO) and traced:
            st.twisted_how = ""                          # the plan sets the twisted region from now on
        ef.apply_plan(spec, st)
        self._planning = True
        self._store()
        self._fill_step()
        self.redraw()
        c = self.app.ebl_canvas
        pts = [q for u in traced for q in st.outlines.get(u, [[]])[0]]
        if pts:
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6 + 10
            c.fit_rect(QtCore.QRectF(min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m))
        self.app.center_tabs.setCurrentWidget(self.app.ebl_page)
        missing = [spec.layer(u).label for u in uids if u not in traced]
        if not traced:
            self.app.warn("Plan from flake images", "No device layer has a traced border yet: trace them in the Flake images tab.")
        else:
            self.app.status("Click a layer's line on the canvas (or its row) and drag it; turn it with its slider."
                            + (f" Not traced yet: {', '.join(missing)}." if missing else ""))

    def _build_plan_list(self, body, st, w: dict) -> None:
        spec = self.app.spec
        body.addWidget(label("Planned device layers: click a row or a layer's line on the canvas, drag the line to move it, turn it "
                             "with the slider (about its middle). The twisted region follows their overlap.", "hint", wrap=True))
        self.plan_list = w["plan_list"] = clickable_rows(QtWidgets.QTreeWidget())
        self.plan_list.setHeaderLabels(["Layer", "Rotation", ""])
        self.plan_list.setRootIsDecorated(False)
        h = self.plan_list.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        h.resizeSection(1, 200)
        h.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.plan_sliders = w["plan_sliders"]
        for uid in ef.plan_layers(spec):
            layer = spec.layer(uid)
            it = QtWidgets.QTreeWidgetItem(self.plan_list, [layer.label + ("" if layer.outline else "  (not traced)"), ""])
            it.setData(0, Qt.ItemDataRole.UserRole, uid)
            it.setSizeHint(0, QtCore.QSize(0, 32))
            cell = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(cell)
            hl.setContentsMargins(2, 0, 2, 0)
            sl = AngleControl(self.plan_angle(st, uid) if uid in st.plan else 0.0, "Turn this layer about its middle (−180° … 180°); "
                              "double-click for 0°, or type the angle. Turns it in Flake layout too (linked)", field_width=52)
            sl.setEnabled(uid in st.plan)
            sl.valueChanged.connect(lambda v, u=uid, key=st.key: (self._activate(key), self._plan_rotated(u, v)))
            sl.sliderReleased.connect(lambda u=uid: self._plan_released(u))
            hl.addWidget(sl, 1)
            self.plan_list.setItemWidget(it, 1, cell)
            self.plan_sliders[uid] = sl
            b = IconButton("image", "Open its traced image (Flake images)", size=26, icon_size=16)
            b.clicked.connect(lambda _=False, u=uid: self.app.flake_panel.trace(u))
            self.plan_list.setItemWidget(it, 2, b)
        self.plan_list.setFixedHeight(min(200, 30 + 34 * max(1, self.plan_list.topLevelItemCount())))
        self.plan_list.itemClicked.connect(lambda item, col, key=st.key: (self._activate(key), self._plan_row_clicked(item, col)))
        body.addWidget(self.plan_list)

    def _plan_row_clicked(self, item, _col=0) -> None:
        uid = item.data(0, Qt.ItemDataRole.UserRole)
        if uid in self.step.plan:
            self.app.ebl_canvas.set_active_polygon(f"outline:{uid}#0")

    def _polygon_selected(self, key: str) -> None:
        if not self.active or self.plan_list is None:
            return
        uid = key.split(":", 1)[1].split("#")[0] if key.startswith("outline:") else None
        try:
            for i in range(self.plan_list.topLevelItemCount()):
                it = self.plan_list.topLevelItem(i)
                if it.data(0, Qt.ItemDataRole.UserRole) == uid:
                    self.plan_list.setCurrentItem(it)
                    return
            self.plan_list.clearSelection()
        except RuntimeError:          # the list was rebuilt
            pass

    def _plan_linked(self, uid: str) -> bool:
        """The plan follows Flake layout (a fabricated flake sets the frame, and this one is not on the chip yet)."""
        spec = self.app.spec
        return (ef._plan_frame(spec, self.flow) is not None and uid not in ef.on_chip_outlines(spec, self.flow)
                and bool(spec.layer(uid) and spec.layer(uid).outline))

    def plan_angle(self, st: ef.FlowStep, uid: str) -> float:
        """What a plan's slider shows: linked, the flake's angle on the chip (the frame's turn + how far its border is
        turned in Flake layout); else the plan's own turn of the traced border."""
        if self._plan_linked(uid):
            m, _t = ef._plan_frame(self.app.spec, self.flow)
            frame_deg = math.degrees(math.atan2(m[1, 0], m[0, 0]))
            return math.remainder(frame_deg + self.app._border_turn(self.app.spec.layer(uid)), 360.0)
        return float(st.plan.get(uid, {}).get("angle", 0.0))

    def _plan_rotated(self, uid: str, deg: float, text=None) -> None:
        st = self.step
        if st is None or uid not in st.plan:
            return
        spec = self.app.spec
        if self._plan_linked(uid):                    # linked: the border turns in Flake layout, the plan follows it
            delta = math.remainder(float(deg) - self.plan_angle(st, uid), 360.0)
            if abs(delta) > 1e-9:
                self.app.turn_border(spec.layer(uid), delta)
                self.app.outline_synced(uid)
                ef.plan_from_layout(spec, self.flow, uid)
        else:
            st.plan[uid]["angle"] = float(deg)
            ef.apply_plan(spec, st)
        self._draw_plan_shapes()
        sl = self.plan_sliders.get(uid)
        if sl is None or not sl.isSliderDown():
            self._plan_released(uid)

    def _plan_released(self, uid: str | None = None) -> None:
        if uid is not None and self._plan_linked(uid):
            self.app.traced_borders_changed([uid])
        self._store()

    def _draw_plan_shapes(self) -> None:
        """Redraw only the planned outlines and the twisted region (the image stays): smooth turning."""
        c, st, spec = self.app.ebl_canvas, self.step, self.app.spec
        for uid in st.plan:
            layer = spec.layer(uid)
            for i, poly in enumerate(st.outlines.get(uid, [])):
                c.set_polygon(f"outline:{uid}#{i}", poly, layer_color(layer, self.app.config), True, 50, 0.08, dash=True)
        c.remove_polygons("twisted:")
        for i, poly in enumerate(st.twisted):
            c.set_polygon(f"twisted:{i}", poly, PALETTE["accent"], st.twisted_how != ef.PLAN_HOW, 60, 0.18, width=2.0)

    def edit_presets(self) -> None:
        PresetsDialog(self.app.config, self.app).exec()
        if self.step is not None:
            self._fill_step()

    # ------------------------------------------------------------------ canvas
    def _drawn(self, pts: list) -> None:
        if not self.active or self.app.center_tabs.currentWidget() is not self.app.ebl_page or self.step is None:
            return
        st = self.step
        if self.draw_target == "twisted":
            st.twisted, st.twisted_how = [pts], "drawn"
        elif self.draw_target and self.draw_target.startswith("outline:"):
            uid = self.draw_target.split(":", 1)[1]
            st.outlines[uid] = [pts]
            st.located[uid] = {"how": "drawn", "score": None}
        else:
            return
        self.draw_target = None
        self.app.set_ebl_tool("pointer")
        self._store()
        self._fill_step()
        self.redraw()

    def _edited(self, key: str, pts: list) -> None:
        if not self.active or self.step is None:
            return
        st = self.step
        kind, _, rest = key.partition(":")
        pts = [tuple(p) for p in pts]
        if kind == "outline":
            uid, _, i = rest.partition("#")
            k = int(i or 0)
            if k == 0:
                self.set_outline(uid, pts, "edited")
                return
            polys = st.outlines.setdefault(uid, [])
            if k < len(polys):
                polys[k] = pts
            st.located[uid] = {"how": "edited", "score": None}
        elif kind == "twisted":
            st.twisted[int(rest)] = pts
            st.twisted_how = "edited"
        elif kind == "pat":
            step_key, _, rest = rest.partition("|")
            purpose, _, i = rest.rpartition("#")
            target = self.flow.step(step_key)
            target.shapes[purpose][int(i)] = pts
            target.exported = None
            target.issues = []                           # findings about the old shape: Check again
            pill = self.cards.get(step_key, {}).get("pill")
            if pill is not None:
                pill.setText(self._pill_text(step_key) + " · edited (Check again)")
        self._store()

    def _picked(self, x: float, y: float) -> None:
        if not self.active or self.pick is None:
            return
        st = self.step
        if self.pick == "pair_img":
            item = self.app.ebl_canvas.images.get(f"step:{getattr(self, '_pair_obj', None) or st.show_image}")
            if item is None:
                return
            px, py = item.scene_to_px(QtCore.QPointF(x, y))
            self.pairs.append({"px": (px, py)})
            self.pick = "pair_gds"
            self.app.status("Now click the same feature in the GDS")
        elif self.pick == "pair_gds":
            self.pairs[-1]["um"] = (x, y)
            self.pick = "pair_img"
            self.app.status(f"{len(self.pairs)} pair(s). Click the next image feature, or press Fit pairs.")

    def redraw(self, fit: bool = False) -> None:
        if not self.active:
            return
        c = self.app.ebl_canvas
        spec = self.app.spec
        c.clear_all()
        c.empty_text = "Choose a bond-pad preset and a step"
        bp = self.bondpads()
        hide = self.hidden
        if bp is not None:
            for role, col, alpha in (("landing", "#c3c6ca", 0.0), ("pads", "#b0b4ba", 0.35), ("markers", "#c9a227", 0.5)):
                if f"bp:{role}" in hide:
                    continue
                c.add_overlay_polys(bp.layer_polys(role, near=(role == "markers")) if role != "landing" else
                                    [p for p in bp.layer_polys("landing") if geo.poly_area(p) < 1e6], col, alpha, z=5)
        st = self.step
        focus = None
        shown = self._canvas_choice()
        self._canvas_shown = None
        if shown is not None and "image" not in hide:
            sk, obj = shown
            img = self._image(obj, sk)
            if img is not None:
                # this step's images are keyed by objective (the wand, pick pairs); another step's by step too
                item = c.set_image(f"step:{obj}" if sk == self.key else f"step:{sk}/{obj}", img, self._placement(obj, img, sk), 1.0, 1)
                focus = item.sceneBoundingRect() if item is not None else None
                self._canvas_shown = (sk, obj)
        if st is not None:
            present = set(ef.present_layers(spec, self.key))
            for layer in spec.layers:
                if f"flake:{layer.uid}" in hide:
                    continue
                for i, poly in enumerate(st.outlines.get(layer.uid, [])):
                    how = st.located.get(layer.uid, {}).get("how")
                    planned = layer.uid not in present or how == ef.EXPECTED_HOW
                    c.set_polygon(f"outline:{layer.uid}#{i}", poly, layer_color(layer, self.app.config), True, 50, 0.06 if planned else 0.10,
                                  dash=planned or how == "projected")
            # flakes not on the chip yet, ticked in the Layers card: where Flake layout puts them (dashed, not editable)
            for uid, poly in ef.not_yet_on_chip(spec, self.flow, self.key).items():
                if uid in getattr(self, "shown_plans", set()):
                    c.add_overlay_polys([poly], layer_color(spec.layer(uid), self.app.config), 0.05, z=45, width=1.2, dash=True)
            for i, poly in enumerate(st.twisted if "twisted" not in hide else []):   # a planned region follows its layers
                c.set_polygon(f"twisted:{i}", poly, PALETTE["accent"], st.twisted_how != ef.PLAN_HOW, 60, 0.18, width=2.0)
            for k, other in self.flow.steps.items():
                if k in self.keys or f"pat:{k}" in hide:
                    continue
                for purpose, polys in other.shapes.items():
                    c.add_overlay_polys(polys, SHAPE_COLORS.get(purpose, "#90a4ae"), 0.18, z=70, width=0.8)
            mine = [self.flow.step(k) for k in self.keys]
            for s_ in mine:
                if f"pat:{s_.key}" in hide:
                    continue
                for purpose, polys in s_.shapes.items():
                    for i, poly in enumerate(polys):
                        c.set_polygon(f"pat:{s_.key}|{purpose}#{i}", poly, SHAPE_COLORS.get(purpose, "#90a4ae"), True, 90, 0.45, width=1.2)
            allp = [q for polys in st.shapes.values() for poly in polys for q in poly] or [q for poly in st.twisted for q in poly]
            if allp:
                xs, ys = [q[0] for q in allp], [q[1] for q in allp]
                focus = QtCore.QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            for src, found in st.found_patterns.items():   # earlier patterns where they were found written (dashed)
                if f"pat:{src}" in hide or found.get("as_designed"):
                    continue
                for purpose, polys in ef.pattern_shapes(self.flow, self.key, src).items():
                    c.add_overlay_polys(polys, SHAPE_COLORS.get(purpose, "#90a4ae"), 0.10, z=72, width=1.4, dash=True)
            if "debris" not in hide and ef.debris_at(self.flow, self.key):    # bubbles, cracks, stray flakes (dashed, by kind)
                for kind, col in DEBRIS_COLORS.items():
                    polys = [[tuple(q) for q in d["polygon"]] for d in ef.debris_at(self.flow, self.key) if d.get("kind") == kind]
                    if polys:
                        c.add_overlay_polys(polys, col, 0.10, z=65, width=1.0, dash=True)
            errs, keys = [], []
            for s_ in mine:                              # each error place: a red circle (click: what it is)
                for n, i in enumerate(s_.issues):
                    if i["severity"] != "error":
                        continue
                    for m, p_ in enumerate(i.get("where") or []):
                        errs.append(tuple(p_))
                        keys.append(f"{s_.key}|{n}|{m}")
            if "errors" not in hide:
                c.add_markers(errs, PALETTE["danger"], 9, keys=keys)
        pending = getattr(self, "_pending_view", None)
        if pending:                                      # reopened: the view as it was saved (once the canvas is on screen)
            if c.isVisible():
                self._pending_view = None
                c.set_view_state(pending)
        elif fit:
            if focus is not None and not focus.isEmpty():
                c.fit_rect(focus)
            elif bp is not None:
                x, y = bp.site
                c.fit_rect(QtCore.QRectF(x - 450, y - 450, 900, 900))
            else:
                c.fit()

    def canvas_layers(self) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """What the EBL canvas can show, grouped: [(group, hint, [(key, label), …]), …] (the Layers card)."""
        spec, st = self.app.spec, self.step
        groups = [("Chip", "The bond-pad preset and the image under the pattern",
                   [("bp:pads", "Bond pads and fan-out"), ("bp:markers", "Alignment markers"), ("bp:landing", "Landing square"),
                    ("image", "Image on canvas")])]
        if spec is not None and st is not None:
            flakes = [(f"flake:{l.uid}", l.label) for l in reversed(spec.layers[1:]) if st.outlines.get(l.uid)]
            if st.twisted:
                flakes.append(("twisted", "Twisted region"))
            if ef.debris_at(self.flow, self.key):
                flakes.append(("debris", "Bubbles, cracks, stray flakes"))
            if flakes:
                groups.append(("Flakes", f"Outlines at {ef.STEPS[self.key].name}", flakes))
            later = [(f"plan:{u}", spec.layer(u).label) for u in reversed(list(ef.not_yet_on_chip(spec, self.flow, self.key)))]
            if later:
                groups.append(("Not On The Chip Yet", "Flakes still to come, where Flake layout plans them relative to the flakes "
                               "already on the chip (dashed; move them in Flake layout)", later))
        pats = [(f"pat:{k}", ef.STEPS[k].name) for k in self.flow.order if self.flow.steps.get(k) and self.flow.steps[k].shapes]
        if pats:
            groups.append(("Patterns", "Generated EBL patterns", pats + [("errors", "Check errors (red marks)")]))
        return groups

    def layer_shown(self, key: str) -> bool:
        """Ticked in the Layers card: most things are shown unless hidden; planned flakes only when chosen."""
        if key.startswith("plan:"):
            return key[5:] in getattr(self, "shown_plans", set())
        return key not in self.hidden

    def set_layer_visible(self, key: str, on: bool) -> None:
        if key.startswith("plan:"):
            (self.shown_plans.add if on else self.shown_plans.discard)(key[5:])
        else:
            (self.hidden.discard if on else self.hidden.add)(key)
        self.redraw()

    def _issue_clicked(self, item) -> None:
        where = item.data(Qt.ItemDataRole.UserRole)
        if where:
            x, y = where[0]
            self.app.ebl_canvas.center_on(x, y, 15.0)

    # ------------------------------------------------------------------ run
    def generate(self, group: str | None = None) -> None:
        """Draw the pattern of the step in use (its card's Generate); for the electrodes, ``group`` = one group's Generate."""
        k = self.key
        if k is None:
            return
        what = ef.ELECTRODE_GROUPS[group][0] if group else ef.STEPS[k].name
        for _ in range(3):
            try:
                ef.generate(self.app.spec, self.flow, k, self.app.config, self.bondpads(), group=group)
                break
            except MissingValues as exc:              # e.g. how far inside the bottom hBN the fingers end
                dlg = MissingValuesDialog(exc.items, self.app, "Values needed", "Generate needs these values (saved as presets).")
                if not dlg.exec():
                    return
                apply_answers(self.app.config, dlg.answers())
            except ef.BlockedByElectrodes as exc:       # other groups' electrodes leave the gate flake no room for its contact
                names = ", ".join(ef.ELECTRODE_GROUPS[g][0].lower() for g in exc.groups)
                if QtWidgets.QMessageBox.question(
                        self, "Generate", f"{what}: the {names} cross the gate flake, so its contact has no room.\n\n"
                        f"Draw the {what.lower()} first and the {names} again after it?") \
                        != QtWidgets.QMessageBox.StandardButton.Yes:
                    break
                self._regenerate_after(group, exc.groups)
                break
            except ValueError as exc:
                self.app.warn("Generate", f"{what}: {exc}")
                break
        self._store()
        self._fill_step()
        self._fill_merge()
        self._status_rows()
        self.redraw(fit=True)
        self.app.status(f"{what}: drawn — drag the points to adjust, then Check and Export GDS.")

    def _regenerate_after(self, group: str, others: list[str]) -> None:
        """Clear ``others``, draw ``group``, then draw ``others`` again (each keeps clear of what is already drawn)."""
        st = self.flow.step("electrodes")
        for g in others:
            st.shapes.pop(ef.ELECTRODE_GROUPS[g][1], None)
            st.routes = [r for r in st.routes if r.get("group") != g]
        failed = []
        for g in [group] + others:
            try:
                ef.generate(self.app.spec, self.flow, "electrodes", self.app.config, self.bondpads(), group=g)
            except ValueError as exc:
                failed.append(f"{ef.ELECTRODE_GROUPS[g][0]}: {exc}")
        if failed:
            self.app.warn("Generate", "\n\n".join(failed))

    def check(self) -> None:
        """Check (the card's button): run the checks and show every rule of the pattern in a window."""
        k = self.key
        if k is None:
            return
        self._run_check()
        from .check_window import PatternCheckDialog
        self.check_dialog = PatternCheckDialog(self, k)
        self.check_dialog.show()
        self.check_dialog.raise_()

    def _run_check(self) -> None:
        k = self.key
        if k is not None and self.flow.step(k).shapes:
            ef.check(self.app.spec, self.flow, k, self.app.config)
        self._after_check(k)

    def _after_check(self, k: str | None) -> None:
        self._store()
        pill = self.cards.get(k, {}).get("pill") if k else None
        if pill is not None:
            pill.setText(self._pill_text(k))
        issues = self.cards.get(k, {}).get("issues") if k else None
        if issues is not None:
            self._fill_issues(k, issues)
        self.redraw()

    # ------------------------------------------------------------------ the pattern drawn by hand
    def draw_pattern(self) -> None:
        """Draw contact gate… (etc.): the step's pattern in its own window, editable with the polygon tool."""
        from .shape_window import ShapeDialog
        k = self.key
        if k is None:
            return
        old = getattr(self, "shape_dialog", None)
        if old is not None:
            old.close()
        self.shape_dialog = ShapeDialog(self, k)
        self.shape_dialog.show()
        self.shape_dialog.raise_()

    def _shapes_changed(self, key: str) -> None:
        st = self.flow.step(key)
        st.exported = None
        st.issues = []                                   # the old findings were about the old shapes: Check again
        self._store()
        pill = self.cards.get(key, {}).get("pill")
        if pill is not None:
            pill.setText(self._pill_text(key) + " · edited (Check again)")
        issues = self.cards.get(key, {}).get("issues")
        if issues is not None:
            self._fill_issues(key, issues)
        self._status_rows()
        self._fill_merge()
        self.redraw()

    def set_shape(self, key: str, purpose: str, i: int, pts: list) -> None:
        polys = self.flow.step(key).shapes.setdefault(purpose, [])
        if 0 <= i < len(polys):
            polys[i] = [tuple(map(float, q)) for q in pts]
            self._shapes_changed(key)

    def add_shape(self, key: str, purpose: str, pts: list) -> int:
        polys = self.flow.step(key).shapes.setdefault(purpose, [])
        polys.append([tuple(map(float, q)) for q in pts])
        self._shapes_changed(key)
        return len(polys) - 1

    def delete_shape(self, key: str, purpose: str, i: int) -> None:
        polys = self.flow.step(key).shapes.get(purpose, [])
        if 0 <= i < len(polys):
            polys.pop(i)
            if not polys:
                self.flow.step(key).shapes.pop(purpose, None)
            self._shapes_changed(key)

    def _flakes_box(self) -> CheckBox:
        """'Include the traced flake outlines' (every card and the merge card show the same choice)."""
        cb = CheckBox("Include the traced flake outlines (one GDS layer per flake)")
        cb.setToolTip("Each flake's outline on the chip at this step (merged: its latest outline among the chosen steps), on a GDS "
                      "layer of its own — asked once, kept in config/layers.json")
        cb.setChecked(bool(getattr(self, "export_flakes", False)))
        cb.toggled.connect(self._set_export_flakes)
        self._flake_boxes = [b for b in getattr(self, "_flake_boxes", []) if _alive(b)] + [cb]
        return cb

    def _set_export_flakes(self, on: bool) -> None:
        self.export_flakes = bool(on)
        for b in getattr(self, "_flake_boxes", []):
            if _alive(b) and b.isChecked() != self.export_flakes:
                b.blockSignals(True)
                b.setChecked(self.export_flakes)
                b.blockSignals(False)

    def _ask_layers(self, exc: MissingValues) -> bool:
        dlg = MissingValuesDialog(exc.items, self.app, "GDS layers needed",
                                  "These patterns have no GDS layer/datatype in config/layers.json yet.")
        if not dlg.exec():
            return False
        apply_answers(self.app.config, dlg.answers())
        return True

    def export(self) -> None:
        keys = [self.key] if self.key else []
        sts = [self.flow.step(k) for k in keys]
        if not sts or not all(s.shapes for s in sts):
            self.app.warn("Export", "Generate this step's pattern first.")
            return
        self._run_check()
        errors = [i for s in sts for i in s.issues if i["severity"] == "error"]
        if errors and QtWidgets.QMessageBox.question(self, "Export", f"The check reports {len(errors)} error(s). Export anyway?") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        from .save_gds import ask_gds_path
        target = ask_gds_path(self, self.app.device_dir() / "stage2" / keys[0], f"{ef._safe(self.app.spec.name)}_{keys[0]}.gds",
                              f"Export GDS — {ef.STEPS[keys[0]].name}")
        if target is None:
            return
        out_dir = target.parent
        for _ in range(3):
            try:
                path = ef.export_steps(self.app.spec, self.flow, keys, self.app.config, out_dir, self.export_flakes, target.name)
                break
            except MissingValues as exc:
                if not self._ask_layers(exc):
                    return
            except ValueError as exc:
                self.app.warn("Export", str(exc))
                return
        else:
            return
        self._store()
        self.app.save()
        self._fill_step()
        self._fill_merge()
        self._status_rows()
        self.app.status(f"Wrote {path}")
        QtWidgets.QMessageBox.information(self, "Exported", f"GDS: {path}")

    def export_merged(self) -> None:
        keys = [k for k, cb in self.merge_boxes.items() if cb.isChecked()]
        if not keys:
            self.app.warn("Merge", "Choose at least one generated step.")
            return
        from .save_gds import ask_gds_path
        out = ask_gds_path(self, self.app.device_dir() / "stage2", f"{ef._safe(self.app.spec.name)}_merged.gds", "Merged GDS")
        if out is None:
            return
        bp = self.bondpads() if self.merge_pads.isChecked() else None
        for _ in range(3):
            try:
                path = ef.export_merged(self.app.spec, self.flow, keys, self.app.config, bp, out, self.export_flakes)
                break
            except MissingValues as exc:
                if not self._ask_layers(exc):
                    return
            except ValueError as exc:
                self.app.warn("Merge", str(exc))
                return
        else:
            return
        self.app.status(f"Wrote {path}")
        QtWidgets.QMessageBox.information(self, "Merged GDS", f"{path}\n\nSteps: " + ", ".join(ef.STEPS[k].name for k in keys)
                                          + ("\nWith the bond pads (their own layers)." if bp else "")
                                          + ("\nWith the traced flake outlines (one layer each)." if self.export_flakes else ""))


class CanvasLayersCard(QtWidgets.QFrame):
    """The Layers card of the EBL canvas (its top-right button): a tick per thing on the canvas — bond
    pads, markers, the image, each flake outline, the twisted region, each step's pattern. Unticked
    ones are not drawn (e.g. hide the generated Hall bars while drawing the contact gate)."""

    def __init__(self, panel: "EBLFlowPanel", parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.panel = panel
        self.setObjectName("CanvasLayersCard")
        self.setStyleSheet(f"#CanvasLayersCard {{ background: {PALETTE['card']}; border: 1px solid {PALETTE['line']}; border-radius: 3px; }}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)
        lay.addWidget(label("Layers On The Canvas", "cardTitle"))
        self.boxes: dict[str, CheckBox] = {}
        if not panel.active or panel.app.spec is None:
            lay.addWidget(label("Open a Stage 2 step first.", "faint"))
        for group, hint, items in panel.canvas_layers() if panel.active else []:
            g = label(group, "muted")
            g.setToolTip(hint)
            lay.addSpacing(4)
            lay.addWidget(g)
            for key, text in items:
                cb = CheckBox(text)
                cb.setChecked(panel.layer_shown(key))
                cb.toggled.connect(lambda on, k=key: self.panel.set_layer_visible(k, on))
                lay.addWidget(cb)
                self.boxes[key] = cb
        lay.addSpacing(6)
        lay.addLayout(buttons_row(button("Show all", self.show_all), button("Hide all", self.hide_all),
                                  button("Only image", self.only_image, None, "Hide everything but the image on the canvas")))
        modernize(self)

    def _set(self, shown: set[str] | None) -> None:
        """Tick the keys in ``shown`` (None: all) and untick the others."""
        panel = self.panel
        keys = [k for _g, _h, items in panel.canvas_layers() for k, _t in items] if panel.active else list(self.boxes)
        plans = {k[5:] for k in keys if k.startswith("plan:")}
        if shown is None:
            panel.hidden.clear()
            panel.shown_plans |= plans
        else:
            panel.hidden = {k for k in keys if not k.startswith("plan:") and k not in shown}
            panel.shown_plans = {u for u in panel.shown_plans if f"plan:{u}" in shown}
        for k, cb in self.boxes.items():
            cb.blockSignals(True)
            cb.setChecked(panel.layer_shown(k))
            cb.blockSignals(False)
        panel.redraw()

    def show_all(self) -> None:
        self._set(None)

    def hide_all(self) -> None:
        self._set(set())

    def only_image(self) -> None:
        self._set({"image"})

    def popup_under(self, anchor: QtWidgets.QWidget) -> None:
        self.adjustSize()
        p = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height() + 4))
        self.move(p.x() - self.width(), p.y())
        self.show()
