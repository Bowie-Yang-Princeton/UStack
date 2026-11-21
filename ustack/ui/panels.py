"""The Pattern tab (Hall bar, contacts, gate limits, gate geometry) and the Check tab
(validation findings, missing values, flake list)."""

from __future__ import annotations

import csv
from pathlib import Path

from .. import autoconfig
from ..config import apply_answers
from ..flakes import SpecNotValid, flake_list
from ..model import CONTACT_SCHEMES, FAMILIES, GATE_FUNCTIONS, HALL_BAR_FIELDS, PATTERN_FIELDS, SD_STYLES, Q
from ..validation import apply_spec_answers, validate_stage1
from .dialogs import ConfigureDialog, MissingValuesDialog
from .style import Banner, Card, ComboBox, PALETTE, Pill, QtCore, QtGui, QtWidgets, button, clear_layout, label, scroll
from .widgets import NumberField, buttons_row, modernize, row, save_file

Qt = QtCore.Qt


class PatternPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.content = QtWidgets.QWidget()
        self.lay = QtWidgets.QVBoxLayout(self.content)
        self.lay.setContentsMargins(6, 0, 6, 6)
        self.lay.setSpacing(0)
        outer.addWidget(scroll(self.content))

    def refresh(self) -> None:
        clear_layout(self.lay)
        spec = self.app.spec
        if spec is None:
            self.lay.addWidget(label("No device", "faint"))
            return
        dev = Card("Device")
        name = QtWidgets.QLineEdit(spec.name)
        name.editingFinished.connect(lambda: self._set_name(name.text()))
        dev.layout.addLayout(row("Name", name, stretch=False))
        fam = ComboBox()
        for k, v in FAMILIES.items():
            fam.addItem(v, k)
        fam.setCurrentIndex(max(0, fam.findData(spec.family)))
        fam.activated.connect(lambda _i: self._set_attr("family", fam.currentData()))
        fam.setToolTip("Which group's rules apply")
        dev.layout.addLayout(row("Structure", fam, stretch=False))
        cs = ComboBox()
        for k, v in CONTACT_SCHEMES.items():
            cs.addItem(v, k)
        cs.setCurrentIndex(max(0, cs.findData(spec.contact_scheme)))
        cs.activated.connect(lambda _i: self._set_attr("contact_scheme", cs.currentData()))
        dev.layout.addLayout(row("Contacts", cs, stretch=False))
        self.lay.addWidget(dev)
        hb = Card("Hall Bar", "The v1 pattern. Every value you enter is kept as given; placement is the only thing UStack chooses.")
        for key, (cap, unit) in HALL_BAR_FIELDS.items():
            q = spec.hall_bar.params.setdefault(key, Q())
            f = NumberField(q.value, f"{q.suggested:g}" if q.suggested is not None else "")
            f.setToolTip(cap + (f"\nSuggestion: {q.suggested_source}" if q.suggested_source else ""))
            f.valueEdited.connect(lambda v, q=q: self._set_q(q, v))
            extra = [f, label("µm" if unit == "um" else unit, "muted")]
            if q.suggested is not None and q.value is None:
                extra.append(button("Use", lambda _=False, q=q: (self._set_q(q, q.suggested), self.refresh()), None, q.suggested_source))
            r = row(cap.split(" (")[0], *extra, width=150)
            hb.layout.addLayout(r)
        sd = ComboBox()
        for k, v in SD_STYLES.items():
            sd.addItem(v, k)
        sd.setCurrentIndex(max(0, sd.findData(spec.hall_bar.sd_style)))
        sd.setToolTip("Straight: S/D contacts at the channel ends. Bent: the source turns to +y and the drain to −y, so their "
                      "contacts land beside the channel (a gate strip along the whole device; two pre-patterned electrode columns). "
                      "Automatic: straight unless Stage 2 finds the back gate under the ends.")
        sd.setMinimumWidth(190)
        sd.activated.connect(lambda _i: self._set_sd_style(sd.currentData()))
        hb.layout.addLayout(row("Source / drain", sd, width=150))
        if spec.contact_scheme == "prepatterned_bottom" and spec.hall_bar.sd_style == "bent":
            hb.layout.addWidget(label("Two electrode columns: W is the gap between the columns (the bottom gate lands in it).", "hint", wrap=True))
        self.lay.addWidget(hb)
        gates = Card("Gate Limits", "Planned maximum |voltage| per gate, for the breakdown checks (optional except the doped-Si back gate).")
        sub = spec.substrate()
        targets = [l for l in spec.layers if l.function in GATE_FUNCTIONS]
        if sub is not None:
            m = self.app.config.material(sub.material) if sub.material else None
            if m is not None and m.conducting == "yes":
                targets.insert(0, sub)
        for l in targets:
            q = spec.gate_limits_V.setdefault(l.uid, Q())
            f = NumberField(q.value, "")
            f.valueEdited.connect(lambda v, q=q: self._set_q(q, v))
            gates.layout.addLayout(row("Doped Si" if l.function == "substrate" else l.label[:22], f, label("V", "muted"), width=150))
        if not targets:
            gates.layout.addWidget(label("No gates in the stack", "faint"))
        self.lay.addWidget(gates)
        geo = Card("Gate / Contact Geometry", "Used in Stage 2 when the step draws a metal top gate, contact gates or split gates. Asked for when needed.")
        for key, (cap, unit) in PATTERN_FIELDS.items():
            q = spec.pattern_params.setdefault(key, Q())
            f = NumberField(q.value, "")
            f.setToolTip(cap)
            f.valueEdited.connect(lambda v, q=q: self._set_q(q, v))
            geo.layout.addLayout(row(cap.split(": ")[-1][:24], f, label("µm", "muted"), width=150))
        self.lay.addWidget(geo)
        if spec.process:
            proc = Card("Process (Template)")
            for i, step in enumerate(spec.process, 1):
                l = label(f"{i}. {step}", "muted", wrap=True)
                proc.layout.addWidget(l)
            for ref in spec.references:
                proc.layout.addWidget(label(ref["cite"] + (f" — {ref['url']}" if ref.get("url") else ""), "hint", wrap=True))
            self.lay.addWidget(proc)
        self.lay.addStretch(1)
        modernize(self.content)

    def _set_name(self, text: str) -> None:
        if text.strip() and text != self.app.spec.name:
            self.app.spec.name = text.strip()
            self.app.spec_changed("name")

    def _set_attr(self, attr: str, value) -> None:
        setattr(self.app.spec, attr, value)
        self.app.spec_changed(attr)

    def _set_sd_style(self, style: str) -> None:
        if style != self.app.spec.hall_bar.sd_style:
            self.app.spec.hall_bar.sd_style = style
            self.app.spec_changed("rules")
            self.refresh()

    def _set_q(self, q: Q, v) -> None:
        if v != q.value:
            q.set(v, "user")
            self.app.spec_changed("value")


SEV_COLOR = {"error": "danger", "missing": "warning", "warning": "warning", "info": "text2", "note": "text2"}


class IssueList(QtWidgets.QListWidget):
    """Findings as wrapped lines, coloured by severity; each row carries data for a click."""

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setUniformItemSizes(False)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)   # Shift / Ctrl-click: several

    def add(self, severity: str, text: str, data=None, tip: str = "") -> QtWidgets.QListWidgetItem:
        it = QtWidgets.QListWidgetItem(f"{severity.upper()} · {text}")
        it.setForeground(QtGui.QColor(PALETTE[SEV_COLOR.get(severity, "text")]))
        it.setData(Qt.ItemDataRole.UserRole, data)
        it.setToolTip(tip or text)
        self.addItem(it)
        return it


class CheckPanel(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.report = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(6, 0, 6, 6)
        lay.setSpacing(0)
        self.pill = Pill("Not checked", "neutral")
        card = Card("Validation", "Python checks the spec against config/materials.csv and config/rules.json.", self.pill)
        card.layout.addLayout(buttons_row(button("Configure", self.configure, "primary", "Generate every empty dimension so the stack "
                                                 "passes validation (asks once for lab values and your physics choices)", "check"),
                                          button("Validate", self.run, None, "Check the specification (Ctrl+R)"),
                                          button("Fill in missing…", self.fill_missing, None, "Enter the values the checks need")))
        self.tree = IssueList()
        self.tree.setMinimumHeight(300)
        self.tree.itemClicked.connect(self._clicked)
        card.layout.addWidget(self.tree)
        lay.addWidget(card)
        fcard = Card("Flakes To Find", "Produced only by a valid specification.")
        self.flakes = QtWidgets.QTreeWidget()
        self.flakes.setHeaderLabels(["Material", "Target", "Min. size (µm)", "Purpose"])
        self.flakes.setRootIsDecorated(True)
        self.flakes.setMinimumHeight(180)
        fcard.layout.addWidget(self.flakes)
        fcard.layout.addLayout(buttons_row(button("Export list…", self.export_flakes, None, "Save the flake list as CSV", "export")))
        lay.addWidget(fcard)
        lay.addStretch(1)
        outer.addWidget(scroll(content))
        self._flake_list = []

    def run(self) -> None:
        spec = self.app.spec
        if spec is None:
            return
        self.report = validate_stage1(spec, self.app.config)
        spec.validation = self.report.to_json()
        self.tree.clear()
        for i in self.report.issues:
            self.tree.add(i.severity, i.message, i.layer, i.message + (f"\nRule: {i.rule} ({i.rule_status})" if i.rule else ""))
        tone = "ok" if self.report.ok else "warn"
        self.pill.set_tone(tone, self.report.summary())
        self.flakes.clear()
        self._flake_list = []
        if self.report.ok:
            try:
                self._flake_list = flake_list(spec, self.app.config, self.report)
            except SpecNotValid:
                self._flake_list = []
            for f in self._flake_list:
                l, w = f.min_size_um
                it = QtWidgets.QTreeWidgetItem(self.flakes, [f.material, f.target, f"{l:g} × {w:g}" + (f" ×{f.pieces}" if f.pieces > 1 else ""), f.purpose])
                for n in f.notes:
                    QtWidgets.QTreeWidgetItem(it, ["", n])
                it.setExpanded(False)
            for c in range(4):
                self.flakes.resizeColumnToContents(c)
        else:
            QtWidgets.QTreeWidgetItem(self.flakes, ["—", "Resolve the errors and missing values first"])
        self.app.report_changed(self.report)

    def configure(self) -> None:
        """Ask once for what only the lab or the researcher can give, then generate every empty dimension."""
        spec = self.app.spec
        if spec is None:
            return
        cfg = self.app.config
        cancelled = False
        for _ in range(4):
            result = autoconfig.configure(spec, cfg, apply=False)
            if not result.missing:
                break
            dlg = MissingValuesDialog(result.missing, self.app, "Configure: values needed",
                                      "Configure generates every dimension from the lab rules. It needs these first: lab values "
                                      "(saved to config/ as approved and reused for later devices) and your choices for this device "
                                      "(layer counts, twist angles, gate voltages). Leave a field empty to decide later.")
            if not dlg.exec():
                cancelled = True
                break
            answers = dlg.answers()
            if not answers:
                break
            apply_answers(cfg, answers)
            apply_spec_answers(spec, answers)
        result = autoconfig.configure(spec, cfg, apply=True)
        self.app.after_configure(result)
        if result.ok:
            ConfigureDialog(result, self.app).exec()
        else:
            # not valid: a window with every error, the values behind it (editable) and what is still needed
            from .configure_window import ConfigureProblemsDialog
            ConfigureProblemsDialog(self.app, result, cancelled).exec()

    def fill_missing(self) -> None:
        if self.report is None:
            self.run()
        items = self.report.missing_items() if self.report else []
        if not items:
            self.app.status("Nothing missing")
            return
        dlg = MissingValuesDialog(items, self.app)
        if dlg.exec():
            ans = dlg.answers()
            apply_answers(self.app.config, ans)
            apply_spec_answers(self.app.spec, ans)
            self.app.spec_changed("values")
            self.run()

    def _clicked(self, item) -> None:
        uid = item.data(Qt.ItemDataRole.UserRole)
        if uid:
            self.app.select_layer(uid)

    def export_flakes(self) -> None:
        if not self._flake_list:
            self.app.status("No flake list: the specification is not valid yet")
            return
        path = save_file(self, "Save flake list", "CSV (*.csv)", str(self.app.device_dir() / "flakes.csv"))
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["material", "target", "min_length_um", "min_width_um", "pieces", "purpose", "layers", "notes"])
            for f in self._flake_list:
                w.writerow([f.material, f.target, f.min_size_um[0], f.min_size_um[1], f.pieces, f.purpose, " ".join(f.layers), " | ".join(f.notes)])
        self.app.status(f"Saved {path}")
