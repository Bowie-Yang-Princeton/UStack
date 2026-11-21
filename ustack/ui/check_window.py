"""Stage 2 › Check: every rule the step's pattern is held to, in its own window — what each rule
requires, what the pattern has, and whether it is met, violated, below a design target, missing a
value (enter it here) or not checked (an outline it needs is not on the chip yet). Clicking a row
with a place shows it on the EBL canvas (:func:`ustack.ebl_flow.rule_report`)."""

from __future__ import annotations

from .. import ebl_flow as ef
from ..config import apply_answers
from .style import PALETTE, Banner, Dialog, QtCore, QtGui, QtWidgets, button, label
from .widgets import NumberField, buttons_row, clickable_rows, modernize

Qt = QtCore.Qt
STATUS = {"met": ("✓ met", "success"), "violated": ("✗ violated", "danger"), "warning": ("! below target", "warning"),
          "missing": ("? value missing", "warning"), "unchecked": ("— not checked", "faint")}


class PatternCheckDialog(Dialog):
    def __init__(self, panel, key: str):
        super().__init__(panel.app)
        self.panel, self.app, self.key = panel, panel.app, key
        self.setModal(False)
        self.resize(980, 560)
        self.setWindowTitle(f"Check — {ef.STEPS[key].name}")
        self.banner = Banner("", "neutral")
        self.body.addWidget(self.banner)
        self.tree = clickable_rows(QtWidgets.QTreeWidget())
        self.tree.setHeaderLabels(["Rule", "Required", "Found", "Status", "Enter"])
        self.tree.setRootIsDecorated(False)
        self.tree.setWordWrap(True)
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            h.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemClicked.connect(self._clicked)
        self.body.addWidget(self.tree, 1)
        self.body.addWidget(label("Values entered here are saved to config/ (presets / rules, approved, as yours). Click a row "
                                  "with a place to show it on the EBL canvas.", "hint", wrap=True))
        self.apply = button("Save values and check again", self.save_and_check, "primary")
        self.body.addLayout(buttons_row(self.apply, button("Check again", self.refresh), QtWidgets.QWidget(),
                                        button("Close", self.close)))
        self.fields: dict[str, NumberField] = {}
        self.refresh()
        modernize(self)

    def refresh(self) -> None:
        spec, cfg = self.app.spec, self.app.config
        if self.panel.flow.step(self.key).shapes:
            ef.check(spec, self.panel.flow, self.key, cfg)
        self.rows = ef.rule_report(spec, self.panel.flow, self.key, cfg)
        self.tree.clear()
        self.fields = {}
        for row in self.rows:
            text, tone = STATUS.get(row["status"], (row["status"], "text"))
            it = QtWidgets.QTreeWidgetItem(self.tree, [row["rule"], str(row["required"]), str(row["found"]), text, ""])
            it.setForeground(3, QtGui.QColor(PALETTE.get(tone, PALETTE["text"])))
            it.setData(0, Qt.ItemDataRole.UserRole, row.get("where"))
            it.setToolTip(0, row["rule"])
            if row["status"] == "missing" and row.get("key"):
                f = NumberField(None, row.get("unit") or "", 90)
                f.setToolTip(row.get("label") or row["rule"])
                self.tree.setItemWidget(it, 4, f)
                self.fields[row["key"]] = f
        n = {s: sum(r["status"] == s for r in self.rows) for s in STATUS}
        tone = "warn" if (n["violated"] or n["missing"] or n["warning"]) else "ok"
        self.banner.set_tone(tone, f"{n['met']} met · {n['violated']} violated · {n['warning']} below target · {n['missing']} missing · "
                                   f"{n['unchecked']} not checked")
        self.apply.setEnabled(bool(self.fields))
        modernize(self)

    def save_and_check(self) -> None:
        answers = {k: f.value() for k, f in self.fields.items() if f.value() is not None}
        if answers:
            apply_answers(self.app.config, answers)
        self.refresh()
        self.panel._after_check(self.key)

    def _clicked(self, item, _col=0) -> None:
        where = item.data(0, Qt.ItemDataRole.UserRole)
        if where:
            x, y = where[0]
            self.app.ebl_canvas.center_on(x, y, 15.0)
