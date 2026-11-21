"""The pop-up of an error's red circle on the EBL canvas (user, 2026-11-20): what the error is, the rule behind it, and
Close / Change rule… / Accept (this place is fine — the circle goes and the error no longer stops the export)."""

from __future__ import annotations

from .. import ebl_flow as ef
from .style import PALETTE, QtCore, QtWidgets, button, label
from .widgets import buttons_row, modernize

Qt = QtCore.Qt


class IssuePopup(QtWidgets.QFrame):
    def __init__(self, panel, step_key: str, issue: dict, pt):
        super().__init__(panel.app, Qt.WindowType.Popup)
        self.panel, self.step_key, self.issue, self.pt = panel, step_key, issue, pt
        self.setObjectName("IssuePopup")
        self.setStyleSheet(f"#IssuePopup {{ background: {PALETTE['card']}; border: 1px solid {PALETTE['line']}; border-radius: 3px; }}")
        self.setMaximumWidth(440)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        lay.addWidget(label(f"Error · {ef.STEPS[step_key].name if step_key in ef.STEPS else step_key}", "cardTitle"))
        lay.addWidget(label(issue.get("message", ""), None, wrap=True))
        lay.addWidget(label(f"At ({pt[0]:.2f}, {pt[1]:.2f}) µm", "faint"))
        self.rule = issue.get("rule")
        if self.rule:
            lay.addWidget(label(f"Rule: {self._rule_text()}", "muted", wrap=True))
        change = button("Change rule…", self._change, None,
                        "Edit the lab value behind this error (saved to config/), then check again" if self.rule else
                        "No lab value is behind this error (a design condition)")
        change.setEnabled(bool(self.rule))
        lay.addLayout(buttons_row(button("Close", self.close), change,
                                  button("Accept", self._accept, "primary", "Accept the error at this place: the red circle goes "
                                         "and it no longer stops the export (Check again keeps it accepted while the pattern "
                                         "stays here)")))
        modernize(self)

    def _rule_text(self) -> str:
        kind, _, name = (self.rule or "").partition(":")
        cfg = self.panel.app.config
        if kind == "rule":
            v = cfg.value(name)
            r = cfg.rules.get(name)
            unit = (r.unit if r else "").replace("um", "µm")
            return f"{name} = {v if v is not None else 'not set'} {unit}".strip()
        v = cfg.preset(name)
        return f"{name} (preset) = {v if v is not None else 'not set'}"

    def show_at(self, global_pos) -> None:
        self.adjustSize()
        self.move(global_pos + QtCore.QPoint(12, 12))
        self.show()

    def _accept(self) -> None:
        self.close()
        self.panel.accept_issue(self.step_key, self.issue, self.pt)

    def _change(self) -> None:
        self.close()
        self.panel.change_rule(self.step_key, self.rule)
