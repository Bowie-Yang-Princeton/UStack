"""The Stack tab: the layer list (drag to reorder, top layer first) and the selected layer's
material, function, dimensions and flake image."""

from __future__ import annotations

from ..model import FUNCTIONS, Layer, MetalPart, Q, new_uid
from ..scene3d import layer_color
from ..validation import category, materials_for_function
from .style import (Banner, Card, CheckBox, ComboBox, PALETTE, QtCore, QtGui, QtWidgets, Signal, Slider, button, clear_layout, label,
                    scroll)
from .widgets import NumberField, buttons_row, clickable_rows, modernize, row

Qt = QtCore.Qt
OTHER = "__other__"


class LayerList(QtWidgets.QListWidget):
    orderChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setIconSize(QtCore.QSize(14, 14))
        self.setMinimumHeight(330)
        clickable_rows(self)          # rows darken under the mouse and when selected, pointing hand over them

    def dropEvent(self, e):  # noqa: N802 - Qt API
        super().dropEvent(e)
        self.orderChanged.emit()


def swatch(color: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(14, 14)
    pm.fill(QtGui.QColor(color))
    return QtGui.QIcon(pm)


class StackPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.current: str | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(6, 0, 6, 6)
        lay.setSpacing(0)
        card = Card("Layers", "Top layer first. Drag a row to change the order; the substrate stays at the bottom.")
        self.list = LayerList()
        self.list.currentRowChanged.connect(self._select_row)
        self.list.orderChanged.connect(self._reordered)
        card.layout.addWidget(self.list)
        card.layout.addLayout(buttons_row(
            button("Add", self._add, None, "Add a layer above the selected one", "plus"),
            button("Duplicate", self._duplicate, None, "Copy the selected layer", "copy"),
            button("Delete", self._delete, "danger", "Remove the selected layer", "trash"),
            button("Up", lambda: self._move(+1), None, "Move the layer up"),
            button("Down", lambda: self._move(-1), None, "Move the layer down")))
        lay.addWidget(card)
        self.props = Card("Selected Layer")
        lay.addWidget(self.props)
        lay.addStretch(1)
        outer.addWidget(scroll(content))

    # ------------------------------------------------------------------ list
    def refresh(self, keep: str | None = None) -> None:
        spec = self.app.spec
        keep = keep or self.current
        self.list.blockSignals(True)
        self.list.clear()
        if spec is None:
            self.list.blockSignals(False)
            self._show_props(None)
            return
        for layer in reversed(spec.layers):
            mat = self.app.config.material(layer.material) if layer.material else None
            text = f"{layer.label}\n{mat.display_name if mat else (layer.material or 'material?')} · {FUNCTIONS.get(layer.function, layer.function)}"
            it = QtWidgets.QListWidgetItem(swatch(layer_color(layer, self.app.config)), text)
            it.setData(Qt.ItemDataRole.UserRole, layer.uid)
            if layer.function == "substrate":
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            self.list.addItem(it)
        self.list.blockSignals(False)
        uids = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        if keep in uids:
            self.list.setCurrentRow(uids.index(keep))
        elif uids:
            self.list.setCurrentRow(0)
        else:
            self._show_props(None)

    def select_uid(self, uid: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == uid:
                self.list.setCurrentRow(i)
                return

    def _select_row(self, r: int) -> None:
        if r < 0:
            self._show_props(None)
            return
        uid = self.list.item(r).data(Qt.ItemDataRole.UserRole)
        self.current = uid
        self._show_props(self.app.spec.layer(uid))
        self.app.layer_selected(uid)

    def _reordered(self) -> None:
        spec = self.app.spec
        uids = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        order = list(reversed(uids))
        sub = spec.substrate()
        if sub is not None and order[0] != sub.uid:
            order.remove(sub.uid)
            order.insert(0, sub.uid)
        spec.layers = [spec.layer(u) for u in order]
        self.app.spec_changed("order")
        self.refresh(self.current)

    def _add(self) -> None:
        spec = self.app.spec
        if spec is None:
            return
        layer = Layer(new_uid(), "New layer", "hBN", "encapsulation")
        idx = spec.index(self.current) + 1 if self.current else len(spec.layers)
        spec.layers.insert(max(1, idx), layer)
        self.current = layer.uid
        self.app.spec_changed("layers")
        self.refresh(layer.uid)

    def selected_uids(self) -> list[str]:
        """The layers chosen in the list (Shift / Ctrl-click: several), else the current one."""
        uids = [it.data(Qt.ItemDataRole.UserRole) for it in self.list.selectedItems()]
        return uids or ([self.current] if self.current else [])

    def _duplicate(self) -> None:
        spec = self.app.spec
        if spec is None:
            return
        last = None
        for uid in self.selected_uids():
            layer = spec.layer(uid)
            if layer is None or layer.function == "substrate":
                continue
            copy = Layer.from_json(layer.to_json())
            copy.uid, copy.name = new_uid(), layer.label + " (copy)"
            copy.image, copy.outline = None, None
            spec.layers.insert(spec.index(layer.uid) + 1, copy)
            last = copy.uid
        if last is None:
            return
        self.app.spec_changed("layers")
        self.refresh(last)

    def _delete(self) -> None:
        spec = self.app.spec
        gone = set(self.selected_uids()) if spec else set()
        if not gone:
            return
        spec.layers = [l for l in spec.layers if l.uid not in gone]
        self.current = None
        self.app.spec_changed("layers")
        self.refresh()

    def _move(self, d: int) -> None:
        spec = self.app.spec
        if not spec or not self.current:
            return
        i = spec.index(self.current)
        j = i + d
        if j < 1 or j >= len(spec.layers) or spec.layers[i].function == "substrate":
            return
        spec.layers[i], spec.layers[j] = spec.layers[j], spec.layers[i]
        self.app.spec_changed("order")
        self.refresh(self.current)

    # ------------------------------------------------------------------ properties
    def _show_props(self, layer: Layer | None) -> None:
        body = self.props.layout
        clear_layout(body)
        if layer is None:
            body.addWidget(label("No layer selected", "faint"))
            return
        cfg = self.app.config
        name = QtWidgets.QLineEdit(layer.name)
        name.editingFinished.connect(lambda: self._set(layer, "name", name.text()))
        body.addLayout(row("Name", name, stretch=False))
        func = ComboBox()
        for k, v in FUNCTIONS.items():
            func.addItem(v, k)
        func.setCurrentIndex(max(0, func.findData(layer.function)))
        func.activated.connect(lambda _i: self._set(layer, "function", func.currentData()))
        body.addLayout(row("Function", func, stretch=False))
        mat = ComboBox()
        mat.addItem("— choose —", "")
        options = materials_for_function(cfg, layer.function) if layer.function else list(cfg.materials)
        for mid in options:
            m = cfg.material(mid)
            mat.addItem(m.display_name + ("" if m.status == "approved" else f"  ({m.status})"), mid)
        if layer.material and mat.findData(layer.material) < 0:
            mat.addItem(f"{layer.material}  (not in lab list)", layer.material)
        mat.addItem("Other (type a name)…", OTHER)
        mat.setCurrentIndex(max(0, mat.findData(layer.material)))
        mat.activated.connect(lambda _i: self._set_material(layer, mat.currentData()))
        body.addLayout(row("Material", mat, stretch=False))
        cat = category(cfg, layer)
        if cat == "metal":
            m = cfg.material(layer.material)
            if not layer.metal_parts and m is not None:
                layer.metal_parts = [MetalPart(c, Q()) for c in m.components]
            for part in layer.metal_parts:
                q = part.thickness_nm
                f = NumberField(q.value, f"{q.suggested:g}" if q.suggested is not None else "")
                f.setToolTip(q.suggested_source or "")
                f.valueEdited.connect(lambda v, q=q: self._set_q(q, v))
                extra = [f, label("nm", "muted")]
                if q.suggested is not None and q.value is None:
                    extra.append(button("Use", lambda _=False, q=q: (self._set_q(q, q.suggested), self._show_props(layer)), None,
                                        f"Use {q.suggested:g} nm ({q.suggested_source})"))
                body.addLayout(row(f"{part.metal}", *extra))
        elif cat != "substrate" and cat is not None:
            for fld, cap, unit in (("thickness_nm", "Thickness", "nm"), ("layer_count", "Layers", ""),
                                   ("width_um", "Width", "µm"), ("length_um", "Length", "µm"), ("twist_deg", "Twist", "°")):
                if fld == "layer_count" and layer.material not in ("graphene",) and cat != "2d_semiconductor":
                    continue
                q = getattr(layer, fld)
                f = NumberField(q.value, f"{q.suggested:g}" if q.suggested is not None else "")
                tip = q.suggested_source or ""
                if q.source:
                    tip = f"Value from: {q.source}" + (f" · suggestion: {tip}" if tip else "")
                f.setToolTip(tip)
                f.valueEdited.connect(lambda v, q=q: self._set_q(q, v))
                extra = [f, label(unit, "muted")]
                if q.suggested is not None and q.value is None:
                    extra.append(button("Use", lambda _=False, q=q: (self._set_q(q, q.suggested), self._show_props(layer)), None,
                                        f"Use {q.suggested:g} ({q.suggested_source})"))
                body.addLayout(row(cap, *extra))
            pieces = QtWidgets.QSpinBox()
            pieces.setRange(1, 6)
            pieces.setValue(layer.pieces)
            pieces.setFixedWidth(72)
            pieces.valueChanged.connect(lambda v: self._set(layer, "pieces", int(v), refresh=False))
            pieces.setToolTip("Pieces cut from one flake (2 for tear/cut-and-stack bilayers)")
            body.addLayout(row("Pieces", pieces))
            if layer.material == "graphene":
                st = ComboBox()
                for s_ in ("", "rhombohedral", "Bernal"):
                    st.addItem(s_ or "—", s_)
                st.setCurrentIndex(max(0, st.findData(layer.stacking)))
                st.activated.connect(lambda _i: self._set(layer, "stacking", st.currentData()))
                body.addLayout(row("Stacking", st, stretch=False))
        else:
            t = label(f"{layer.thickness_nm.value:g} nm SiO2" if layer.thickness_nm.value else "", "muted")
            body.addLayout(row("Oxide", t))
        if layer.literature:
            b = Banner("Literature: " + layer.literature, "neutral")
            body.addWidget(b)
        notes = QtWidgets.QPlainTextEdit(layer.notes)
        notes.setFixedHeight(64)
        notes.setPlaceholderText("Notes")
        notes.textChanged.connect(lambda: setattr(layer, "notes", notes.toPlainText()))
        body.addWidget(notes)
        if cat not in ("metal", "substrate", None) and getattr(self.app, "open_flake_images", None):   # (not for a template)
            img = layer.image or {}
            state = ("no image yet" if not img.get("path") else ("border traced" if layer.outline else "image, border not traced"))
            body.addLayout(row("Flake image", label(state, "muted"),
                               button("Flake images tab", lambda: self.app.open_flake_images(layer.uid), None,
                                      "Upload this flake's image and trace its border", "image")))
        modernize(self.props)

    def _set(self, layer: Layer, attr: str, value, refresh: bool = True) -> None:
        if getattr(layer, attr) == value:
            return
        setattr(layer, attr, value)
        if attr == "function":
            layer.metal_parts = layer.metal_parts if category(self.app.config, layer) == "metal" else []
        self.app.spec_changed(attr)
        if refresh:
            self.refresh(layer.uid)

    def _set_material(self, layer: Layer, mid: str) -> None:
        if mid == OTHER:
            from .widgets import ask_text
            name = ask_text(self, "Material", "Material name (it will be flagged until it is added to config/materials.csv):", layer.material)
            if not name:
                self._show_props(layer)
                return
            mid = name
        layer.material = mid
        m = self.app.config.material(mid)
        if m is not None and m.category == "metal":
            old = {p.metal: p for p in layer.metal_parts}
            layer.metal_parts = [old.get(c, MetalPart(c, Q())) for c in m.components]
        else:
            layer.metal_parts = []
        self.app.spec_changed("material")
        self.refresh(layer.uid)

    def _set_q(self, q: Q, v) -> None:
        if v == q.value:
            return
        q.set(v, "user")
        self.app.spec_changed("value")
