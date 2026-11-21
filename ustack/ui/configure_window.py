"""The Configure window when Configure cannot produce a valid specification: every error with the
values behind it, editable in place, the values still needed, and the values Configure generated.
Edits are the researcher's own (source ``user``: Configure keeps them); "Configure again" reruns
Configure with them, "Validate" only checks. The window stays open until it is closed."""

from __future__ import annotations

from .. import autoconfig
from ..config import apply_answers
from ..validation import apply_spec_answers, issue_parameters, validate_stage1
from .dialogs import MissingFields
from .style import Banner, Card, Dialog, QtCore, QtWidgets, TabWidget, button, clear_layout, label, scroll
from .widgets import NumberField, buttons_row, modernize

Qt = QtCore.Qt
SOURCE_TEXT = {"user": "yours", "auto": "Configure", "template": "template", "measured": "measured", "assistant": "assistant",
               "literature": "literature", "material": "material"}


class ConfigureProblemsDialog(Dialog):
    def __init__(self, app, result, cancelled: bool = False):
        super().__init__(app)
        self.app = app
        self.result = result
        self.setWindowTitle("Configure: fix these")
        self.resize(1000, 800)
        self.banner = Banner("", "warn")
        self.body.addWidget(self.banner)
        intro = ("Configure could not make the specification valid. Each problem below lists the values it depends on: edit "
                 "them here (your values are kept by Configure) and press Configure again, or Validate to check only.")
        if cancelled:
            intro = "Configure stopped before every value was entered. " + intro
        self.body.addWidget(label(intro, "muted", wrap=True))
        self.tabs = TabWidget()
        self.problems = QtWidgets.QWidget()
        self.plist = QtWidgets.QVBoxLayout(self.problems)
        self.plist.setContentsMargins(2, 4, 2, 4)
        self.plist.setSpacing(6)
        self.tabs.addTab(scroll(self.problems), "Problems")
        self.generated = QtWidgets.QTreeWidget()
        self.generated.setHeaderLabels(["Item", "Value", "Derived from"])
        self.generated.setRootIsDecorated(False)
        self.tabs.addTab(self.generated, "Generated values")
        self.warnings = QtWidgets.QListWidget()
        self.tabs.addTab(self.warnings, "Warnings")
        self.body.addWidget(self.tabs, 1)
        self.rerun = button("Configure again", self.configure_again, "primary", "Apply the values entered here and run Configure again")
        self.body.addLayout(buttons_row(self.rerun, button("Validate", self.validate, None, "Check the specification with the values entered here"),
                                        QtWidgets.QWidget(), button("Close", self.accept)))
        self.missing = MissingFields()
        self._fill()
        modernize(self)

    # ------------------------------------------------------------------ content
    def _fill(self) -> None:
        res, spec, cfg = self.result, self.app.spec, self.app.config
        report = res.report
        clear_layout(self.plist)
        self.missing = MissingFields()
        errors = [i for i in report.issues if i.severity == "error"] if report else []
        missing = list(res.missing)
        if report is not None:
            keys = {m.key for m in missing}
            missing += [m for m in report.missing_items() if m.key not in keys]
        if res.ok:
            self.banner.set_tone("ok", f"Valid · {len(res.changes)} values generated. You can close this window.")
        else:
            self.banner.set_tone("warn", f"{len(errors)} errors · {len(missing)} values still needed"
                                 + (f" · {report.summary()}" if report else ""))
        for n, issue in enumerate(errors, 1):
            card = Card(f"Error {n}")
            msg = label(issue.message, None, wrap=True)
            msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card.layout.addWidget(msg)
            if issue.rule:
                card.layout.addWidget(label(f"Lab rule {issue.rule} ({issue.rule_status or 'approved'}) — change it in Rules.", "hint", wrap=True))
            params = issue_parameters(spec, cfg, issue)
            for p in params:
                card.layout.addLayout(self._param_row(p))
            if not params:
                card.layout.addWidget(label("No dimension behind this one: change the layer order, material or function in the "
                                            "Stack tab.", "hint", wrap=True))
            if issue.layer and spec.layer(issue.layer) is not None:
                card.layout.addLayout(buttons_row(button("Show layer", lambda _=False, u=issue.layer: self.app.select_layer(u), None,
                                                         "Select this layer in the Stack tab")))
            self.plist.addWidget(card)
        if missing:
            card = Card("Values Still Needed", "Lab values are saved to config/ as approved; device values go into this device.")
            for it in missing:
                if it.kind == "action":
                    card.layout.addWidget(label("To do: " + it.label, "hint", wrap=True))
                else:
                    card.layout.addLayout(self.missing.row(it))
            self.plist.addWidget(card)
        if not errors and not missing:
            self.plist.addWidget(label("No errors and nothing missing.", "muted"))
        self.plist.addStretch(1)
        self.generated.clear()
        for c in res.changes:
            it = QtWidgets.QTreeWidgetItem(self.generated, [c.label, f"{c.value:g} {c.unit}".strip(), c.reason])
            it.setToolTip(2, c.reason)
        for col in range(3):
            self.generated.resizeColumnToContents(col)
        self.warnings.clear()
        for i in (report.issues if report else []):
            if i.severity in ("warning", "info"):
                self.warnings.addItem(f"{i.severity}: {i.message}")
        for note in res.notes:
            self.warnings.addItem(f"note: {note}")
        self.tabs.setTabText(2, f"Warnings ({self.warnings.count()})")
        self.tabs.setTabText(0, f"Problems ({len(errors) + len(missing)})")
        modernize(self.problems)

    def _param_row(self, p) -> QtWidgets.QLayout:
        h = QtWidgets.QHBoxLayout()
        h.setSpacing(8)
        cap = label(p.label, "fieldLabel")
        cap.setWordWrap(True)
        cap.setMinimumWidth(360)
        h.addWidget(cap, 1)
        f = NumberField(p.q.value, f"{p.q.suggested:g}" if p.q.suggested is not None else "", 100)
        f.valueEdited.connect(lambda v, q=p.q: self._edit(q, v))
        h.addWidget(f)
        unit = label({"um": "µm", "deg": "°"}.get(p.unit, p.unit), "muted")
        unit.setFixedWidth(26)
        h.addWidget(unit)
        src = SOURCE_TEXT.get(p.q.source, p.q.source or "empty")
        tip = label(f"from: {src}", "hint")
        tip.setMinimumWidth(120)
        if p.q.note:
            tip.setToolTip(p.q.note)
        h.addWidget(tip)
        return h

    def _edit(self, q, v) -> None:
        if v == q.value:
            return
        q.set(v, "user")
        self.app.spec_changed("value")

    # ------------------------------------------------------------------ actions
    def _apply_missing(self) -> None:
        answers = self.missing.answers()
        if answers:
            apply_answers(self.app.config, answers)
            apply_spec_answers(self.app.spec, answers)

    def configure_again(self) -> None:
        self._apply_missing()
        self.result = autoconfig.configure(self.app.spec, self.app.config, apply=True)
        self.app.after_configure(self.result)
        self._fill()

    def validate(self) -> None:
        self._apply_missing()
        report = validate_stage1(self.app.spec, self.app.config)
        self.result = autoconfig.ConfigureResult(self.result.changes, [], report, self.result.notes)
        self.app.check_panel.run()
        self.app.refresh_structure(report)
        self._fill()
