"""Small house-style helpers shared by the UStack panels (ported from UAnalysis' workbench)."""

from __future__ import annotations

import math
from typing import Callable

from .style import (H_CONTROL, PALETTE, Card, CheckBox, ComboBox, QtCore, QtGui, QtWidgets, Signal, button, frame_fields,
                    label, watch)

Qt = QtCore.Qt
CAPTION_WIDTH = 96


class ComboChevron(QtWidgets.QWidget):
    """The small chevron drawn over a combo (the QSS hides the platform arrow)."""

    def __init__(self, combo: QtWidgets.QComboBox):
        super().__init__(combo)
        self.combo = combo
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        watch(combo, self)
        self.sync_geometry()
        self.show()

    def sync_geometry(self):
        self.setGeometry(max(self.combo.width() - 31, 0), 0, 31, self.combo.height())
        self.raise_()
        self.update()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if watched is self.combo and event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show, QtCore.QEvent.Type.FocusIn,
                                                     QtCore.QEvent.Type.FocusOut, QtCore.QEvent.Type.EnabledChange):
            self.sync_geometry()
        return False

    def paintEvent(self, _event):  # noqa: N802 - Qt API
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        color = PALETTE["accent"] if self.combo.hasFocus() else PALETTE["text2"]
        if not self.combo.isEnabled():
            color = PALETTE["faint"]
        pen = QtGui.QPen(QtGui.QColor(color), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        cx, cy = self.width() / 2, self.height() / 2
        painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(cx - 4, cy - 2), QtCore.QPointF(cx, cy + 2), QtCore.QPointF(cx + 4, cy - 2)]))


def modernize(root: QtWidgets.QWidget) -> None:
    """Spin boxes without arrows, combo chevrons, field frames — as UAnalysis does."""
    for spin in root.findChildren(QtWidgets.QAbstractSpinBox):
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
    for combo in root.findChildren(QtWidgets.QComboBox):
        if not hasattr(combo, "_modern_chevron"):
            combo._modern_chevron = ComboChevron(combo)
    for row in root.findChildren(QtWidgets.QWidget):
        if isinstance(row, CheckBox):
            row.setFixedHeight(H_CONTROL)
    frame_fields(root)


def coord_label(sample: str = "chip x -00000.00 µm   y -00000.00 µm") -> QtWidgets.QLabel:
    """A cursor-position readout of fixed width: its text changes on every mouse move, and a label
    that grows and shrinks with it makes the layout (and the canvas beside it) jump."""
    item = label("", "muted")
    item.setFixedWidth(item.fontMetrics().horizontalAdvance(sample) + 12)
    item.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
    return item


def caption(text: str, width: int = CAPTION_WIDTH) -> QtWidgets.QLabel:
    item = label(text, "fieldLabel")
    item.setFixedWidth(width)
    return item


def row(caption_text: str, *widgets: QtWidgets.QWidget, width: int = CAPTION_WIDTH, stretch: bool = True) -> QtWidgets.QHBoxLayout:
    lay = QtWidgets.QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    if caption_text is not None:
        lay.addWidget(caption(caption_text, width))
    for w in widgets:
        if isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QComboBox)) and w.maximumWidth() > 10000 and type(w).__name__ != "NumberField":
            w.setMaximumWidth(260)
        lay.addWidget(w)
    if stretch:
        lay.addStretch(1)
    return lay


def combo(items: list[tuple[str, object]] | list[str], current=None) -> ComboBox:
    box = ComboBox()
    for it in items:
        if isinstance(it, tuple):
            box.addItem(it[0], it[1])
        else:
            box.addItem(it, it)
    if current is not None:
        i = box.findData(current)
        if i >= 0:
            box.setCurrentIndex(i)
    return box


class NumberField(QtWidgets.QLineEdit):
    """A number field that may be empty (None = not specified). ``placeholder`` shows a suggestion."""

    valueEdited = Signal(object)

    def __init__(self, value: float | None = None, placeholder: str = "", width: int = 88, decimals: int = 4):
        super().__init__()
        self.decimals = decimals
        self.setFixedWidth(width)
        self.setPlaceholderText(placeholder)
        self.set_value(value)
        self.editingFinished.connect(lambda: self.valueEdited.emit(self.value()))

    def value(self) -> float | None:
        text = self.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def set_value(self, value: float | None) -> None:
        self.blockSignals(True)
        self.setText("" if value is None else f"{value:.{self.decimals}g}")
        self.blockSignals(False)


def section(title: str, hint: str | None = None, trailing: QtWidgets.QWidget | None = None) -> Card:
    return Card(title, hint, trailing)


def buttons_row(*items: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
    lay = QtWidgets.QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for it in items:
        lay.addWidget(it)
    lay.addStretch(1)
    return lay


def message(parent, title: str, text: str, kind: str = "info") -> None:
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon({"info": QtWidgets.QMessageBox.Icon.Information, "warn": QtWidgets.QMessageBox.Icon.Warning,
                 "error": QtWidgets.QMessageBox.Icon.Critical}.get(kind, QtWidgets.QMessageBox.Icon.Information))
    box.exec()


def ask_text(parent, title: str, prompt: str, text: str = "") -> str | None:
    value, ok = QtWidgets.QInputDialog.getText(parent, title, prompt, text=text)
    return value.strip() if ok and value.strip() else None


def open_file(parent, title: str, filters: str, start: str = "") -> str | None:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, start, filters)
    return path or None


def save_file(parent, title: str, filters: str, start: str = "") -> str | None:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, start, filters)
    return path or None


def act(text: str, callback: Callable, variant: str | None = None, tooltip: str | None = None, icon: str | None = None):
    return button(text, callback, variant, tooltip, icon)


# ---------------------------------------------------------------------- clickable feedback
#: Widgets that get the pointing-hand cursor (anything clicked to act; text fields keep the I-beam).
_CLICKABLE = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QTabBar, QtWidgets.QAbstractSlider)


class HandCursors(QtCore.QObject):
    """Application event filter: every button, combo, tab, check box and slider shows the pointing
    hand when the mouse is over it (unless the widget chose its own cursor)."""

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if event.type() in (QtCore.QEvent.Type.Polish, QtCore.QEvent.Type.Show) and isinstance(obj, _CLICKABLE):
            if not obj.testAttribute(Qt.WidgetAttribute.WA_SetCursor):
                obj.setCursor(Qt.CursorShape.PointingHandCursor)
        return False


_HAND_CURSORS: HandCursors | None = None


def install_hand_cursors() -> None:
    global _HAND_CURSORS
    app = QtWidgets.QApplication.instance()
    if app is not None and _HAND_CURSORS is None:
        _HAND_CURSORS = HandCursors(app)
        app.installEventFilter(_HAND_CURSORS)


def _row_style() -> str:
    return (f"QTreeView::item, QListView::item {{ padding: 4px 6px; border-radius: 3px; background: transparent; color: {PALETTE['text']}; }}"
            f"QTreeView::item:hover, QListView::item:hover {{ background: {PALETTE['flat_hover']}; }}"
            f"QTreeView::item:selected, QTreeView::item:selected:active, QTreeView::item:selected:!active, "
            f"QListView::item:selected, QListView::item:selected:active, QListView::item:selected:!active "
            f"{{ background: {PALETTE['flat_pressed']}; color: {PALETTE['text']}; }}"
            # the check box is painted by CheckDelegate: no native indicator (no dark frame on any platform)
            "QTreeView::indicator { width: 13px; height: 13px; image: none; border: none; background: transparent; }")


def clickable_rows(view: QtWidgets.QAbstractItemView, boxes: bool = True) -> QtWidgets.QAbstractItemView:
    """A list whose rows darken under the mouse (until it leaves) and stay darker while selected,
    with the pointing hand over the rows. Several rows can be chosen (user, 2026-11-20): Shift-click
    selects the rows from the last one clicked, Ctrl / ⌘-click adds or removes one (a PickTree keeps
    its own: Shift-click toggles one row); with ``boxes`` every row has a tick box at its left that
    adds / removes that row (any set of rows). Lists whose ticks mean something else pass boxes=False."""
    from .style import THEME
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    if boxes:
        selection_boxes(view)
    view.setMouseTracking(True)
    view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
    view.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
    view.setStyleSheet(_row_style())
    THEME.changed.connect(lambda v=view: v.setStyleSheet(_row_style()))
    return view


# ---------------------------------------------------------------------- small house-style prompts
def prompt_text(parent, title: str, prompt: str, text: str = "", ok_text: str = "Confirm") -> str | None:
    """A styled one-line prompt with Confirm / Cancel; returns the stripped text, or None."""
    from .style import Dialog
    dlg = Dialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(440, 150)
    dlg.body.addWidget(label(prompt, "fieldLabel", wrap=True))
    field = QtWidgets.QLineEdit(text)
    field.selectAll()
    dlg.body.addSpacing(4)
    dlg.body.addWidget(field)
    dlg.body.addStretch(1)
    ok = button(ok_text, dlg.accept, "primary")
    ok.setDefault(True)
    dlg.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Cancel", dlg.reject), ok))
    field.returnPressed.connect(dlg.accept)
    field.textChanged.connect(lambda t: ok.setEnabled(bool(t.strip())))
    modernize(dlg)
    field.setFocus()
    if dlg.exec() and field.text().strip():
        return field.text().strip()
    return None


def confirm(parent, title: str, text: str, ok_text: str = "Confirm", danger: bool = False) -> bool:
    """A styled yes/no question with Confirm (or ``ok_text``) / Cancel; Cancel is the default."""
    from .style import Dialog
    dlg = Dialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(460, 160)
    dlg.body.addWidget(label(text, None, wrap=True))
    dlg.body.addStretch(1)
    cancel = button("Cancel", dlg.reject)
    cancel.setDefault(True)
    dlg.body.addLayout(buttons_row(QtWidgets.QWidget(), cancel, button(ok_text, dlg.accept, "danger" if danger else "primary")))
    modernize(dlg)
    return bool(dlg.exec())


# ---------------------------------------------------------------------- check boxes in item views
class CheckDelegate(QtWidgets.QStyledItemDelegate):
    """Paints an item's check box as a plain square outline with a tick inside when checked,
    whatever the row's state (selected / hovered rows only darken their background). The native
    indicator is switched off by the row style sheet (``_row_style``)."""

    SIZE = 13

    def paint(self, painter, option, index) -> None:  # noqa: D401 - Qt API
        super().paint(painter, option, index)
        state = index.data(Qt.ItemDataRole.CheckStateRole)
        if state is None:
            return
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        r = QtCore.QRectF(style.subElementRect(QtWidgets.QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, widget))
        if r.isEmpty():
            return
        s = self.SIZE
        box = QtCore.QRectF(round(r.center().x() - s / 2) + 0.5, round(r.center().y() - s / 2) + 0.5, s - 1, s - 1)
        checked = Qt.CheckState(state) == Qt.CheckState.Checked
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        pen = QtGui.QPen(QtGui.QColor(PALETTE["faint"]), 1.0)          # a thin grey square line
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)
        if checked:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            tp = QtGui.QPen(QtGui.QColor(PALETTE["text"]), 1.6)
            tp.setCapStyle(Qt.PenCapStyle.RoundCap)
            tp.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(tp)
            tick = QtGui.QPainterPath()
            tick.moveTo(box.left() + s * 0.22, box.top() + s * 0.52)
            tick.lineTo(box.left() + s * 0.42, box.top() + s * 0.72)
            tick.lineTo(box.left() + s * 0.78, box.top() + s * 0.28)
            painter.drawPath(tick)
        painter.restore()


class SelectBoxDelegate(QtWidgets.QStyledItemDelegate):
    """A tick box at the left of each row's first column showing whether the row is chosen (selected); the row's own
    content (drawn by ``inner``) moves right by the box's width. Clicking the box adds / removes the row
    (:func:`selection_boxes`)."""

    W = 22
    SIZE = 13

    def __init__(self, view: QtWidgets.QAbstractItemView, inner: QtWidgets.QAbstractItemDelegate | None):
        super().__init__(view)
        self.view = view
        self.inner = inner if inner is not None and inner is not self else QtWidgets.QStyledItemDelegate(view)

    def paint(self, painter, option, index) -> None:  # noqa: D401 - Qt API
        if index.column() != 0:
            self.inner.paint(painter, option, index)
            return
        widget = option.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        style.drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_PanelItemViewItem, option, painter, widget)  # the whole row's tint
        opt = QtWidgets.QStyleOptionViewItem(option)
        opt.rect = option.rect.adjusted(self.W, 0, 0, 0)
        self.inner.paint(painter, opt, index)
        sm = self.view.selectionModel()
        checked = sm is not None and sm.isSelected(index)
        s = self.SIZE
        r = option.rect
        box = QtCore.QRectF(r.left() + (self.W - s) // 2 + 0.5, round(r.center().y() - s / 2) + 0.5, s - 1, s - 1)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["faint"]), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)
        if checked:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            tp = QtGui.QPen(QtGui.QColor(PALETTE["text"]), 1.6)
            tp.setCapStyle(Qt.PenCapStyle.RoundCap)
            tp.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(tp)
            tick = QtGui.QPainterPath()
            tick.moveTo(box.left() + s * 0.22, box.top() + s * 0.52)
            tick.lineTo(box.left() + s * 0.42, box.top() + s * 0.72)
            tick.lineTo(box.left() + s * 0.78, box.top() + s * 0.28)
            painter.drawPath(tick)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 - Qt API
        sz = self.inner.sizeHint(option, index)
        return QtCore.QSize(sz.width() + (self.W if index.column() == 0 else 0), sz.height())

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API
        return self.inner.createEditor(parent, option, index)

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        self.inner.setEditorData(editor, index)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API
        self.inner.setModelData(editor, model, index)

    def editorEvent(self, event, model, option, index):  # noqa: N802 - Qt API
        return self.inner.editorEvent(event, model, option, index)


class _BoxClicks(QtCore.QObject):
    """A press on a row's tick box adds / removes that row (no other row changes)."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def eventFilter(self, obj, e):  # noqa: N802 - Qt API
        if e.type() in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonDblClick) \
                and e.button() == Qt.MouseButton.LeftButton:
            pos = e.position().toPoint()
            idx = self.view.indexAt(pos)
            if idx.isValid():
                first = idx.sibling(idx.row(), 0)
                rect = self.view.visualRect(first)
                if rect.left() <= pos.x() < rect.left() + SelectBoxDelegate.W:
                    if e.type() == QtCore.QEvent.Type.MouseButtonPress:
                        sm = self.view.selectionModel()
                        sm.select(first, QtCore.QItemSelectionModel.SelectionFlag.Toggle
                                  | QtCore.QItemSelectionModel.SelectionFlag.Rows)
                        sm.setCurrentIndex(first, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
                        self.view.viewport().update()
                    return True
        return False


def selection_boxes(view: QtWidgets.QAbstractItemView) -> None:
    """A tick box at the left of every row (user, 2026-11-20: choose any set of rows; Shift still chooses a run)."""
    if isinstance(view.itemDelegate(), SelectBoxDelegate):
        return
    view.setItemDelegate(SelectBoxDelegate(view, view.itemDelegate()))
    f = _BoxClicks(view)
    view.viewport().installEventFilter(f)
    view._box_clicks = f
    sm = view.selectionModel()
    if sm is not None:
        sm.selectionChanged.connect(lambda *_a, v=view: v.viewport().update())


class PickTree(QtWidgets.QTreeWidget):
    """A tree where Shift-click (like Ctrl/Cmd-click) adds or removes one row from the selection, so
    any combination of rows can be picked, with unselected rows in between."""

    def selectionCommand(self, index, event=None):  # noqa: N802 - Qt API
        if isinstance(event, QtGui.QMouseEvent) and event.modifiers() & (Qt.KeyboardModifier.ShiftModifier
                                                                         | Qt.KeyboardModifier.ControlModifier):
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                return QtCore.QItemSelectionModel.SelectionFlag.Toggle | QtCore.QItemSelectionModel.SelectionFlag.Rows
            return QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
        return super().selectionCommand(index, event)


class AngleControl(QtWidgets.QWidget):
    """An angle: a slider (−180° … 180°) and a number box beside it that can be typed into (any
    value, wrapped to −180 … 180). ``valueChanged(float)`` on every change; ``sliderReleased`` when a
    slider drag ends. Mirrors the slider API the panels use (value, setValue, isSliderDown)."""

    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(self, value: float = 0.0, tip: str = "", minimum: int = -180, maximum: int = 180, field_width: int = 58):
        super().__init__()
        from .style import Slider, label
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.slider = Slider(minimum, maximum, int(round(value)))
        self.slider.setMinimumWidth(80)
        self.field = NumberField(value, "0", field_width, 5)
        self.field.setToolTip("Type an angle in degrees and press Enter")
        if tip:
            self.slider.setToolTip(tip)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.field)
        lay.addWidget(label("°", "muted"))
        self._value = float(value)
        self.slider.valueChanged.connect(lambda v: self._set(float(v)) if self.slider.isSliderDown() or abs(v - self._value) >= 0.5 else None)
        self.slider.sliderReleased.connect(self.sliderReleased.emit)
        self.field.valueEdited.connect(self._typed)

    def _typed(self, v) -> None:
        if v is None:
            self._show()
            return
        self._set(math.remainder(float(v), 360.0) if self.slider.minimum() == -180 else float(v))

    def _show(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(self._value)))
        self.slider.blockSignals(False)
        self.field.set_value(round(self._value, 3))

    def _set(self, v: float, emit: bool = True) -> None:
        changed = abs(v - self._value) > 1e-9
        self._value = v
        self._show()
        if emit and changed:
            self.valueChanged.emit(v)

    def value(self) -> float:
        return self._value

    def setValue(self, v: float) -> None:  # noqa: N802 - Qt-style API
        self._set(float(v), emit=not self.signalsBlocked())

    def isSliderDown(self) -> bool:  # noqa: N802 - Qt-style API
        return self.slider.isSliderDown()

    def text(self) -> str:
        return f"{self._value:g}°"


class BoxedCellDelegate(QtWidgets.QStyledItemDelegate):
    """Editable table cells drawn as text boxes (a field frame inside the cell), so a value that can
    be typed into looks like one; read-only cells are drawn as usual."""

    def paint(self, painter, option, index) -> None:  # noqa: N802 - Qt API
        super().paint(painter, option, index)
        if not (index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
            return
        from .style import R_FIELD, control_frame, pixel_frame
        r = QtCore.QRectF(option.rect).adjusted(3, 3, -3, -3)
        if r.height() > 30:                      # wrapped rows: a box one control high, centred
            r = QtCore.QRectF(r.x(), r.center().y() - 13, r.width(), 26)
        painter.save()
        hover = bool(option.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        pixel_frame(painter, r, R_FIELD, control_frame(1.0 if hover else 0.0, focus=False, enabled=True))
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 - Qt API
        s = super().sizeHint(option, index)
        return QtCore.QSize(s.width() + 12, max(s.height(), 32))


def boxed_cells(table: QtWidgets.QAbstractItemView) -> QtWidgets.QAbstractItemView:
    """Draw the editable cells of ``table`` as text boxes and edit them with one click."""
    table.setItemDelegate(BoxedCellDelegate(table))
    table.setMouseTracking(True)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
                          | QtWidgets.QAbstractItemView.EditTrigger.CurrentChanged
                          | QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
                          | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
                          | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed)
    return table


def help_button(title: str, text: str):
    """A small circled question mark next to a field (user, 2026-11-22); clicking it says what the field is."""
    from .style import IconButton
    b = IconButton("help", "What is this?", size=22, icon_size=16)
    b.help_title, b.help_text = title, text
    b.clicked.connect(lambda: _show_help(b))
    return b


def _show_help(b) -> None:
    import sys
    sys.modules[__name__].message(b.window(), b.help_title, b.help_text)
