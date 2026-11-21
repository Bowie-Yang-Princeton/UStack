"""Dialogs: ask for missing values, edit / approve the lab rules, the request assistant."""

from __future__ import annotations

import json
from typing import Any

from ..config import LabConfig, MissingItem
from .style import (Banner, Card, CheckBox, ComboBox, Dialog, PALETTE, QtCore, QtGui, QtWidgets, TabWidget, button, label,
                    scroll)
from .widgets import NumberField, boxed_cells, buttons_row, modernize

Qt = QtCore.Qt


class MissingValuesDialog(Dialog):
    """One field per missing value. Lab rules and GDS layers entered here are saved to config/
    as approved, attributed to the user; device values go into the spec with source 'user'.
    Suggestions (e.g. a paper's value) are shown beside the field and only used if accepted."""

    def __init__(self, items: list[MissingItem], parent=None, title: str = "Values needed", intro: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, min(860, 220 + 46 * len(items)))
        self.items = [i for i in items if i.kind != "action"]
        self.actions = [i for i in items if i.kind == "action"]
        self.mf = MissingFields()
        self.fields = self.mf.fields
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        lay.addWidget(Banner(intro or "UStack needs these values to continue. Nothing is guessed: leave a field empty to "
                                      "decide later. Lab rules and layer numbers you enter are saved to the config folder as approved.", "info"))
        if self.actions:
            lay.addWidget(Banner("To do first: " + "; ".join(a.label for a in self.actions), "warn"))
        groups: dict[str, list[MissingItem]] = {}
        for it in self.items:
            kind = it.key.split(":", 1)[0]
            groups.setdefault({"rule": "Lab rules (config/rules.json)", "layer": "GDS layers (config/layers.json)",
                               "default": "Lab defaults (config/rules.json)"}.get(kind, "This device"), []).append(it)
        for title_, its in groups.items():
            card = Card(title_)
            for it in its:
                card.layout.addLayout(self._field_row(it))
            lay.addWidget(card)
        lay.addStretch(1)
        self.body.addWidget(scroll(content), 1)
        ok = button("Continue", self.accept, "primary")
        use_all = button("Use all suggestions", self._use_all, None, "Fill every empty field that has a suggestion")
        cancel = button("Cancel", self.reject)
        self.body.addLayout(buttons_row(use_all, QtWidgets.QWidget(), cancel, ok))
        modernize(self)

    def _field_row(self, it: MissingItem) -> QtWidgets.QLayout:
        return self.mf.row(it)

    def _use_all(self) -> None:
        self.mf.use_all()

    def answers(self) -> dict[str, Any]:
        return self.mf.answers()


class MissingFields:
    """One field per missing value (number, choice, GDS layer, yes/no) with its suggestion and
    source; shared by the missing-values dialog and the Configure problems window."""

    def __init__(self):
        self.fields: dict[str, Any] = {}

    def row(self, it: MissingItem) -> QtWidgets.QLayout:
        box = QtWidgets.QVBoxLayout()
        box.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        cap = label(it.label + (f" ({it.unit})" if it.unit else ""), "fieldLabel")
        cap.setWordWrap(True)
        cap.setMinimumWidth(320)
        top.addWidget(cap, 1)
        if it.kind == "choice":
            w = ComboBox()
            w.addItem("—", None)
            for opt in (it.suggested or []):
                w.addItem(str(opt), opt)
            w.setFixedWidth(220)
        elif it.kind == "layer":
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            w.layer = NumberField(None, "layer", 64, 6)
            w.datatype = NumberField(None, "datatype", 64, 6)
            h.addWidget(w.layer)
            h.addWidget(label("/"))
            h.addWidget(w.datatype)
            sug = it.suggested if isinstance(it.suggested, (list, tuple)) and len(it.suggested) == 2 else None
            if sug is not None:
                w.layer.setPlaceholderText(str(sug[0]))
                w.datatype.setPlaceholderText(str(sug[1]))
                top.addWidget(button("Use", lambda _=False, f=w, v=sug: (f.layer.set_value(v[0]), f.datatype.set_value(v[1])), None,
                                     f"Use the suggested layer {sug[0]}/{sug[1]}"))
        elif it.kind == "bool":
            w = CheckBox("yes")
        else:
            sug = it.suggested if isinstance(it.suggested, (int, float)) else None
            w = NumberField(None, f"{sug:g}" if sug is not None else "", 100)
            if sug is not None:
                use = button("Use", lambda _=False, f=w, v=sug: f.set_value(v), None, f"Use the suggested value {sug:g}")
                top.addWidget(use)
        top.addWidget(w)
        self.fields[it.key] = (it, w)
        box.addLayout(top)
        if it.hint:
            hint = label(f"Source / note: {it.hint}", "hint")
            hint.setWordWrap(True)
            box.addWidget(hint)
        return box

    def use_all(self) -> int:
        """Fill every empty field that has a suggestion — numbers and GDS layers (user, 2026-11-20: the layer fields of the
        GDS export were left empty). Returns how many were filled."""
        n = 0
        for it, w in self.fields.values():
            sug = it.suggested
            if it.kind == "layer":
                if isinstance(sug, (list, tuple)) and len(sug) == 2 and w.layer.value() is None and w.datatype.value() is None:
                    w.layer.set_value(int(sug[0]))
                    w.datatype.set_value(int(sug[1]))
                    n += 1
            elif isinstance(w, NumberField) and w.value() is None and isinstance(sug, (int, float)) and not isinstance(sug, bool):
                w.set_value(sug)
                n += 1
            elif isinstance(w, CheckBox) and isinstance(sug, bool):
                w.setChecked(sug)
                n += 1
        return n

    def answers(self) -> dict[str, Any]:
        out = {}
        for key, (it, w) in self.fields.items():
            if isinstance(w, ComboBox):
                v = w.currentData()
            elif it.kind == "layer":
                a, b = w.layer.value(), w.datatype.value()
                v = (int(a), int(b)) if a is not None and b is not None else None
            elif isinstance(w, CheckBox):
                v = w.isChecked()
            else:
                v = w.value()
            if v is not None:
                out[key] = v
        return out



class PresetsTable(QtWidgets.QTableWidget):
    """The design presets (config/presets.json) as a table. Values are read-only until
    :meth:`set_editable`; :meth:`collect` stores the changed ones (approved, attributed to the user)."""

    HEADERS = ["Group", "Preset", "Value", "Unit", "Status", "What it does"]

    def __init__(self, config: LabConfig, editable: bool = False):
        super().__init__(0, len(self.HEADERS))
        self.config = config
        self.editable = editable
        boxed_cells(self)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        self.setWordWrap(True)
        self.fill()

    def fill(self) -> None:
        items = sorted(self.config.presets.items(), key=lambda kv: kv[1].get("group", ""))
        self.keys = [k for k, _ in items]
        self.setRowCount(len(items))
        for r, (key, e) in enumerate(items):
            self.setItem(r, 0, _ro(e.get("group", "")))
            name = _ro(e.get("label", key))
            name.setToolTip(key)
            self.setItem(r, 1, name)
            v = e.get("value")
            self.setItem(r, 2, QtWidgets.QTableWidgetItem("" if v is None else f"{v:g}" if isinstance(v, (int, float)) else str(v)))
            self.setItem(r, 3, _ro(e.get("unit", "")))
            st = e.get("status", "program")
            it = _ro(st)
            it.setToolTip(str(e.get("source", "")))
            if st == "approved":
                it.setForeground(QtGui.QColor(PALETTE["success"]))
            self.setItem(r, 4, it)
            self.setItem(r, 5, _ro(e.get("description", "")))
        self.resizeColumnsToContents()
        self.setColumnWidth(1, 240)
        self.setColumnWidth(5, 420)
        self.set_editable(self.editable)

    def set_editable(self, on: bool) -> None:
        self.editable = on
        for r in range(self.rowCount()):
            it = self.item(r, 2)
            flags = it.flags() | Qt.ItemFlag.ItemIsEditable if on else it.flags() & ~Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
        self.viewport().update()

    def collect(self) -> list[str]:
        changed = []
        for r, key in enumerate(self.keys):
            text = self.item(r, 2).text().strip().replace(",", ".")
            if not text:
                continue
            value = float(text)
            if value != self.config.preset(key):
                if key == "hall_bar.n_probe_pairs":
                    value = int(round(value))
                self.config.set_preset(key, value)
                changed.append(key)
        return changed


def _ro(text: str) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    it.setToolTip(text)
    return it


class PresetsDialog(Dialog):
    """Edit preset dimensions: the relations Configure uses and the microscope field of view."""

    def __init__(self, config: LabConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Preset dimensions")
        self.resize(1100, 620)
        self.body.addWidget(Banner("Configure and Stage 2 derive their dimensions from these presets (finger array, contact "
                                   "gate, electrodes, Hall bar relations, …). Click a value box to change it; Save writes "
                                   "config/presets.json, the edited values marked approved.", "info"))
        self.table = PresetsTable(config, editable=True)
        self.body.addWidget(self.table, 1)
        save = button("Save", self._save, "primary", "Write config/presets.json")
        self.body.addLayout(buttons_row(save, button("Close", self._close)))

    def _save(self) -> bool:
        try:
            self.table.collect()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Not saved", f"A value is not a number: {exc}")
            return False
        self.config.save_presets()
        self.table.fill()
        return True

    def _close(self) -> None:
        if self._changed() and not self._save():
            return
        self.accept()

    def _changed(self) -> bool:
        for r, key in enumerate(self.table.keys):
            text = self.table.item(r, 2).text().strip()
            v = self.config.preset(key)
            if text != ("" if v is None else f"{v:g}" if isinstance(v, (int, float)) else str(v)):
                return True
        return False


class RulesDialog(Dialog):
    """The lab configuration as tables: value, status and source for every rule, GDS layer and
    material, and the Configure presets, with Approve buttons. Values are read-only until Edit
    (bottom left) is pressed. Edits are saved to config/ (attributed to the user)."""

    def __init__(self, config: LabConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Lab rules, layers and materials")
        self.resize(1180, 760)
        self.body.addWidget(Banner("Values copied from the attached lab documents are 'pending' and literature values are "
                                   "'literature' until you approve them; checks still run but say so. Missing values are asked "
                                   "for when a step needs them.", "info"))
        tabs = TabWidget()
        self.rules_table = self._table(["Rule", "Value", "Unit", "Status", "Applies to", "Description", "Source"])
        self.layers_table = self._table(["Purpose", "Layer", "Datatype", "Name", "Status", "Source"])
        self.mat_table = self._table(["Id", "Category", "Name", "Components (bottom→top)", "Status", "Source"])
        tabs.addTab(self.rules_table, "Rules")
        tabs.addTab(self.layers_table, "GDS layers")
        tabs.addTab(self.mat_table, "Materials")
        self.presets_table = PresetsTable(config)
        tabs.addTab(self.presets_table, "Configure presets")
        self.body.addWidget(tabs, 1)
        self.editing = False
        self.edit = button("Edit", self._toggle_edit, None, "Make the values editable (rules, GDS layers, presets)")
        approve = button("Approve selected", self._approve, "primary", "Mark the selected rows as approved by the lab")
        save = button("Save", self._save, None, "Write config/rules.json, layers.json, materials.csv and presets.json")
        close = button("Close", self._close)
        self.tabs = tabs
        self.body.addLayout(buttons_row(self.edit, approve, save, QtWidgets.QWidget(), close))
        self._fill()

    def _toggle_edit(self) -> None:
        self.editing = not self.editing
        self.edit.setText("Done editing" if self.editing else "Edit")
        self._apply_editable()
        self.presets_table.set_editable(self.editing)

    def _apply_editable(self) -> None:
        for table, cols in ((self.rules_table, (1,)), (self.layers_table, (1, 2))):
            for r in range(table.rowCount()):
                for col in cols:
                    it = table.item(r, col)
                    if it is None:
                        continue
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable if self.editing else it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.viewport().update()

    def _close(self) -> None:
        if self.editing:
            ans = QtWidgets.QMessageBox.question(self, "Rules", "Save your edits before closing?",
                                                 QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard
                                                 | QtWidgets.QMessageBox.StandardButton.Cancel)
            if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if ans == QtWidgets.QMessageBox.StandardButton.Save and not self._save():
                return
        self.accept()

    def _table(self, headers: list[str]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setStretchLastSection(True)
        t.setWordWrap(True)
        boxed_cells(t)
        return t

    def _status_item(self, status: str) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem(status)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        color = {"approved": PALETTE["success"], "pending": PALETTE["warning"], "literature": PALETTE["warning"], "missing": PALETTE["danger"]}.get(status)
        if color:
            it.setForeground(QtGui.QColor(color))
        return it

    def _ro(self, text: str) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        it.setToolTip(text)
        return it

    def _fill(self) -> None:
        c = self.config
        t = self.rules_table
        t.setRowCount(len(c.rules))
        for r, rule in enumerate(c.rules.values()):
            t.setItem(r, 0, self._ro(rule.id))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem("" if rule.value is None else json.dumps(rule.value)))
            t.setItem(r, 2, self._ro(rule.unit))
            t.setItem(r, 3, self._status_item(rule.status))
            t.setItem(r, 4, self._ro(", ".join(rule.applies_to)))
            t.setItem(r, 5, self._ro(rule.description))
            t.setItem(r, 6, self._ro(rule.source_text()))
        t.resizeColumnsToContents()
        t.setColumnWidth(5, 320)
        t = self.layers_table
        t.setRowCount(len(c.layers))
        for r, lay in enumerate(c.layers.values()):
            t.setItem(r, 0, self._ro(lay.purpose))
            t.setItem(r, 1, QtWidgets.QTableWidgetItem("" if lay.layer is None else str(lay.layer)))
            t.setItem(r, 2, QtWidgets.QTableWidgetItem("" if lay.datatype is None else str(lay.datatype)))
            t.setItem(r, 3, self._ro(lay.name))
            t.setItem(r, 4, self._status_item(lay.status))
            t.setItem(r, 5, self._ro(f"{lay.source.get('doc', '')}: {lay.source.get('quote', '')}".strip(": ")))
        t.resizeColumnsToContents()
        if hasattr(self, "editing"):
            self._apply_editable()
        t = self.mat_table
        t.setRowCount(len(c.materials))
        for r, m in enumerate(c.materials.values()):
            t.setItem(r, 0, self._ro(m.id))
            t.setItem(r, 1, self._ro(m.category))
            t.setItem(r, 2, self._ro(m.display_name))
            t.setItem(r, 3, self._ro(" / ".join(m.components)))
            t.setItem(r, 4, self._status_item(m.status))
            t.setItem(r, 5, self._ro(m.source))
        t.resizeColumnsToContents()

    def _collect(self) -> None:
        c = self.config
        for r, rule in enumerate(list(c.rules.values())):
            text = self.rules_table.item(r, 1).text().strip()
            new = None if not text else json.loads(text) if text[0] in "[{" or text in ("true", "false") else float(text)
            if new != rule.value:
                if new is None:
                    rule.value, rule.status = None, "missing"
                else:
                    c.set_rule_value(rule.id, new)
        for r, lay in enumerate(list(c.layers.values())):
            a, b = self.layers_table.item(r, 1).text().strip(), self.layers_table.item(r, 2).text().strip()
            if a and b and (int(a), int(b)) != (lay.layer, lay.datatype):
                c.set_gds_layer(lay.purpose, int(a), int(b))
        self.presets_table.collect()

    def _approve(self) -> None:
        self._collect()
        idx = self.tabs.currentIndex()
        table = [self.rules_table, self.layers_table, self.mat_table][idx]
        rows = sorted({i.row() for i in table.selectedIndexes()})
        keys = [table.item(r, 0).text() for r in rows]
        problems = []
        for k in keys:
            try:
                if idx == 0:
                    self.config.approve(k)
                elif idx == 1:
                    self.config.approve_layer(k)
                else:
                    self.config.approve_material(k)
            except ValueError as exc:
                problems.append(str(exc))
        self.config.save()
        self._fill()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Not approved", "\n".join(problems))

    def _save(self) -> bool:
        try:
            self._collect()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Not saved", f"A value could not be read: {exc}")
            return False
        self.config.save()
        self._fill()
        self.presets_table.fill()
        return True


class DescribePage(QtWidgets.QWidget):
    """Type the device you want; get a draft stack to review (the assistant: Claude with an API key, else the keyword
    parser). Used by New device › Build from description and by :class:`AssistantDialog`. ``drafted`` fires with the
    draft; nothing is applied here."""

    drafted = QtCore.pyqtSignal(object)

    def __init__(self, config: LabConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.draft = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 0)
        import os
        mode = "Claude (ANTHROPIC_API_KEY found)" if os.environ.get("ANTHROPIC_API_KEY") else "the built-in keyword parser (set ANTHROPIC_API_KEY to use Claude)"
        lay.addWidget(Banner(f"Drafts use {mode}. Unstated numbers stay empty and unknown materials are kept as written, "
                             "so validation can ask for or flag them.", "info"))
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlaceholderText("e.g. an hBN-encapsulated monolayer graphene Hall bar with a graphite back gate and edge contacts")
        self.text.setFixedHeight(90)
        lay.addWidget(self.text)
        lay.addLayout(buttons_row(button("Draft", self.run, "primary", "Draft the stack from the description")))
        self.out = QtWidgets.QTreeWidget()
        self.out.setHeaderLabels(["Layer (bottom → top)", "Material", "Function", "Values"])
        self.out.setRootIsDecorated(False)
        lay.addWidget(self.out, 1)
        self.notes = Banner("Describe the device, then press Draft.", "neutral")
        lay.addWidget(self.notes)

    def run(self) -> None:
        from ..assistant import draft_from_request
        req = self.text.toPlainText().strip()
        if not req:
            self.notes.set_tone("warn", "Describe the device first.")
            return
        try:
            self.draft = draft_from_request(req, self.config)
        except Exception as exc:  # network / API errors are shown, never swallowed
            self.notes.set_tone("warn", f"The assistant failed: {exc}")
            return
        self.out.clear()
        for l in self.draft.spec.layers:
            vals = ", ".join(f"{k}={getattr(l, k).value:g}" for k in ("thickness_nm", "layer_count", "width_um", "length_um", "twist_deg")
                             if getattr(l, k).value is not None)
            QtWidgets.QTreeWidgetItem(self.out, [l.label, l.material or "—", l.function, vals])
        for c in range(4):
            self.out.resizeColumnToContents(c)
        notes = [f"Source: {self.draft.source}"] + [f"Requested: {r.text}" for r in self.draft.spec.requirements] + self.draft.notes
        if self.draft.missing:
            notes.append("Not stated: " + ", ".join(self.draft.missing))
        self.notes.set_tone("neutral", "\n".join(notes))
        self.drafted.emit(self.draft)


class AssistantDialog(Dialog):
    """Type the device you want; get a draft stack to review. Nothing is applied until you press
    'Use this draft', and the draft then goes through the same validation as any stack. (No button opens it at
    present — New device › Build from description uses the same :class:`DescribePage`; kept for later use.)"""

    def __init__(self, config: LabConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Describe a device")
        self.resize(820, 640)
        self.page = DescribePage(config, self)
        self.body.addWidget(self.page, 1)
        self.use = button("Use this draft", self.accept, "primary")
        self.use.setEnabled(False)
        self.page.drafted.connect(lambda _d: self.use.setEnabled(True))
        self.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Cancel", self.reject), self.use))

    @property
    def draft(self):
        return self.page.draft

    @property
    def text(self):
        return self.page.text

    def _run(self) -> None:
        self.page.run()


class ConfigureDialog(Dialog):
    """What Configure set (value and derivation for each), and the validation result."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure")
        self.resize(900, 620)
        report = result.report
        tone = "ok" if result.ok else "warn"
        head = report.summary() if report is not None else "Not validated"
        if result.missing:
            head += f" · {len(result.missing)} values still needed (Configure again to enter them)"
        self.body.addWidget(Banner(head, tone))
        self.body.addWidget(label(f"{len(result.changes)} values generated. Values you or a measurement entered were kept as they are; "
                                  "generated values are marked 'auto' and are recomputed by the next Configure.", "muted", wrap=True))
        table = QtWidgets.QTreeWidget()
        table.setHeaderLabels(["Item", "Value", "Derived from"])
        table.setRootIsDecorated(False)
        for c in result.changes:
            it = QtWidgets.QTreeWidgetItem(table, [c.label, f"{c.value:g} {c.unit}".strip(), c.reason])
            it.setToolTip(2, c.reason)
        for col in range(3):
            table.resizeColumnToContents(col)
        self.body.addWidget(table, 1)
        for note in result.notes:
            self.body.addWidget(label(note, "hint", wrap=True))
        if report is not None:
            problems = [i.message for i in report.issues if i.severity in ("error", "warning")]
            if problems:
                self.body.addWidget(Banner("\n".join(problems[:12]), "warn" if report.errors else "neutral"))
        self.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Close", self.accept, "primary")))
        modernize(self)
