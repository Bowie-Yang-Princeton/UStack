"""New device: always asks how to start — from a template (group structures and the lab's own),
built from scratch (substrate, bottom gate, device layers, top gate, then the hBN and contacts), or
from a plain-English description (the assistant's draft, reviewed before it is used).
Dimensions are never asked here: Configure or the user fills them afterwards.

From a template, **Edit Template** (user, 2026-11-21) is a second step in the same window: the chosen template's layers
(add, remove, reorder, materials, functions …), structure rules, contacts and S/D style edited before the device is made
from it (:class:`ustack.ui.template_editor.TemplateEditor` on a copy); the template file is untouched unless "Save as lab
template…" is used. Back returns to the choice (the edit is kept while the same template stays chosen)."""

from __future__ import annotations

from pathlib import Path

from .. import templates
from ..config import LabConfig
from ..model import CONTACT_SCHEMES, FAMILIES, SD_STYLES, DeviceSpec
from .dialogs import DescribePage
from .style import Banner, Card, CheckBox, ComboBox, Dialog, QtCore, QtWidgets, TabWidget, button, label
from .widgets import buttons_row, clickable_rows, modernize, row

Qt = QtCore.Qt
NONE = "— none —"


class NewDeviceDialog(Dialog):
    PAGES = ("template", "scratch", "description")

    def __init__(self, config: LabConfig, parent=None, template_path: str | None = None, start: str = "template"):
        super().__init__(parent)
        self.config = config
        self.spec: DeviceSpec | None = None
        self.setWindowTitle("New device")
        self.resize(780, 820)
        self.name = QtWidgets.QLineEdit(f"Device {QtCore.QDate.currentDate().toString('yyyy-MM-dd')}")
        self.body.addLayout(row("Device name", self.name, width=110))
        self.tabs = TabWidget()
        self.tabs.addTab(self._template_page(template_path), "Start from a template")
        self.tabs.addTab(self._scratch_page(), "Build from scratch")
        self.describe = DescribePage(config)
        self.tabs.addTab(self.describe, "Build from description")
        self.tabs.setCurrentIndex(self.PAGES.index(start) if start in self.PAGES else 0)
        # step 1: how to start; step 2 (from a template): Edit Template
        self.steps = QtWidgets.QStackedWidget()
        choose = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(choose)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(label("How to start", "cardTitle"))
        cl.addWidget(self.tabs, 1)
        self.steps.addWidget(choose)
        self.edit_page = QtWidgets.QWidget()
        self.edit_lay = QtWidgets.QVBoxLayout(self.edit_page)
        self.edit_lay.setContentsMargins(0, 0, 0, 0)
        self.edit_title = label("Edit Template", "cardTitle")
        self.edit_lay.addWidget(self.edit_title)
        self.edit_lay.addWidget(label("Adjust the layers (add, remove, drag to reorder, material, function …), the structure rules, "
                                      "contacts and source / drain style for this device. The template itself is not changed "
                                      "unless you save the edit as a lab template.", "muted", wrap=True))
        self.steps.addWidget(self.edit_page)
        self.editor = None                           # TemplateEditor of the chosen template (made on Edit Template)
        self._edit_path: str | None = None
        self.body.addWidget(self.steps, 1)
        self.create = button("Create device", self._create, "primary")
        self.edit_btn = button("Edit Template", self.edit_template, None,
                               "Next step: adjust the chosen template's layers and options before creating the device")
        self.back_btn = button("Back", self.back, None, "Back to choosing how to start (your edit is kept)")
        self.save_tpl_btn = button("Save as lab template…", self.save_template, None,
                                   "Keep this edit as a new lab template too (the original template is not changed)")
        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(8)
        for b in (self.back_btn, self.edit_btn):
            foot.addWidget(b)
        foot.addStretch(1)
        for b in (self.save_tpl_btn, button("Cancel", self.reject), self.create):
            foot.addWidget(b)
        self.body.addLayout(foot)
        self.tabs.currentChanged.connect(lambda _i: self._update_buttons())
        self._update_buttons()
        modernize(self)

    # ------------------------------------------------------------------ steps
    def _update_buttons(self) -> None:
        editing = self.steps.currentWidget() is self.edit_page
        self.edit_btn.setVisible(not editing and self.PAGES[self.tabs.currentIndex()] == "template")
        self.edit_btn.setEnabled(self.tpl.currentItem() is not None)
        self.back_btn.setVisible(editing)
        self.save_tpl_btn.setVisible(editing)

    def edit_template(self) -> None:
        """Step 2: the chosen template in the layer editor (a copy; kept while the same template stays chosen)."""
        from .template_editor import TemplateEditor
        it = self.tpl.currentItem()
        if it is None:
            QtWidgets.QMessageBox.warning(self, "New device", "Select a template first.")
            return
        path = it.data(0, Qt.ItemDataRole.UserRole)
        if self.editor is None or self._edit_path != path:
            if self.editor is not None:
                self.editor.setParent(None)
                self.editor.deleteLater()
            self.editor = TemplateEditor(self.config, templates.load_template(Path(path)), None, self.edit_page, text_fields=False)
            self.edit_lay.addWidget(self.editor, 1)
            self._edit_path = path
            modernize(self.editor)
        self.edit_title.setText(f"Edit Template — {it.text(0).strip()}")
        self.steps.setCurrentWidget(self.edit_page)
        self.resize(max(self.width(), 1180), max(self.height(), 860))
        self._update_buttons()

    def back(self) -> None:
        self.steps.setCurrentIndex(0)
        self._update_buttons()

    def save_template(self) -> None:
        from .widgets import prompt_text
        if self.editor is None:
            return
        name = prompt_text(self, "Save as lab template", "Save this edit as a lab template named:",
                           f"{self.editor.spec.name} (lab)", "Save")
        if name:
            path = templates.save_as_template(self.editor.spec, name.strip())
            if self.parent() is not None and hasattr(self.parent(), "status"):
                self.parent().status(f"Template saved to {path}")

    # ------------------------------------------------------------------ template
    def _template_page(self, template_path: str | None) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        self.tpl = clickable_rows(QtWidgets.QTreeWidget(), boxes=False)   # one template to start from
        self.tpl.setHeaderHidden(True)
        self.tpl.setRootIsDecorated(False)
        for info in templates.list_templates():
            it = QtWidgets.QTreeWidgetItem(self.tpl, [info.name + ("" if info.builtin else "  · edited" if info.overrides else "  · lab")])
            it.setData(0, Qt.ItemDataRole.UserRole, str(info.path))
            it.setToolTip(0, info.description)
            if template_path and str(info.path) == str(template_path):
                self.tpl.setCurrentItem(it)
        if self.tpl.currentItem() is None and self.tpl.topLevelItemCount():
            self.tpl.setCurrentItem(self.tpl.topLevelItem(0))
        self.tpl.itemDoubleClicked.connect(lambda *_: self._create())
        self.tpl.currentItemChanged.connect(lambda *_: self._update_buttons() if hasattr(self, "edit_btn") else None)
        lay.addWidget(self.tpl, 1)
        self.tpl_desc = label("", "muted", wrap=True)
        self.tpl.currentItemChanged.connect(lambda *_: self._show_description())
        lay.addWidget(self.tpl_desc)
        self._show_description()
        return page

    def _show_description(self) -> None:
        it = self.tpl.currentItem()
        self.tpl_desc.setText(it.toolTip(0) if it is not None else "")

    # ------------------------------------------------------------------ from scratch
    def _materials(self, *categories: str, extra: tuple[str, ...] = ()) -> list[tuple[str, str]]:
        out = []
        for m in self.config.materials.values():
            if m.category in categories or m.id in extra:
                out.append((m.display_name, m.id))
        return out

    def _combo(self, items: list[tuple[str, str]], none: bool = True, current: str | None = None) -> ComboBox:
        box = ComboBox()
        if none:
            box.addItem(NONE, "")
        for text, data in items:
            box.addItem(text, data)
        if current is not None:
            box.setCurrentIndex(max(0, box.findData(current)))
        box.setMinimumWidth(220)
        return box

    def _scratch_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.addWidget(Banner("Choose the layers from the bottom up. No dimension is needed: press Configure afterwards to "
                               "generate them. hBN dielectrics are added where ticked.", "info"))
        cfg = self.config
        twod = [m for m in cfg.materials.values() if m.category == "2d"]
        conductors_2d = [(m.display_name, m.id) for m in twod if m.conducting == "yes"]
        metals = self._materials("metal")
        gate_choices = conductors_2d + metals
        stack = Card("Stack (bottom → top)")
        self.sub = self._combo(self._materials("substrate"), none=False)
        stack.layout.addLayout(row("1. Substrate", self.sub, width=130))
        self.bg = self._combo(gate_choices, current="graphite")
        stack.layout.addLayout(row("2. Bottom gate", self.bg, width=130))
        self.bbn = CheckBox("hBN under the device (bottom-gate dielectric)")
        self.bbn.setChecked(True)
        stack.layout.addLayout(row("", self.bbn, width=130))
        stack.layout.addWidget(label("3. Device layers (one row per flake piece, bottom → top)", "fieldLabel"))
        self.dev_box = QtWidgets.QVBoxLayout()
        self.dev_box.setSpacing(4)
        self.dev_rows: list[QtWidgets.QWidget] = []
        stack.layout.addLayout(self.dev_box)
        self.dev_items = [(m.display_name, m.id) for m in twod if m.id not in ("hBN", "graphite")]
        self._add_device_row("graphene")
        stack.layout.addLayout(buttons_row(button("Add device layer", lambda: self._add_device_row(None), None,
                                                  "Another flake piece (e.g. the next layer of a twisted stack)", "plus")))
        self.tbn = CheckBox("hBN over the device (top-gate dielectric)")
        self.tbn.setChecked(True)
        stack.layout.addLayout(row("", self.tbn, width=130))
        self.tg = self._combo(gate_choices)
        stack.layout.addLayout(row("4. Top gate", self.tg, width=130))
        self.cap = CheckBox("hBN cap over a graphite top gate")
        self.cap.setChecked(True)
        stack.layout.addLayout(row("", self.cap, width=130))
        outer.addWidget(stack)
        con = Card("Contacts and Rules")
        self.scheme = self._combo([(v, k) for k, v in CONTACT_SCHEMES.items()], none=False)
        con.layout.addLayout(row("Contacts", self.scheme, width=130))
        self.metal = self._combo(metals, current="Au/Cr")
        con.layout.addLayout(row("Contact metal", self.metal, width=130))
        self.sd = self._combo([(v, k) for k, v in SD_STYLES.items()], none=False)
        self.sd.setToolTip("Bent: source and drain turn to the sides (for a gate strip along the whole device, "
                           "or two pre-patterned electrode columns)")
        con.layout.addLayout(row("Source / drain", self.sd, width=130))
        self.family = self._combo([(v, k) for k, v in FAMILIES.items()], none=False, current="custom")
        self.family.setToolTip("Which group's rules apply")
        con.layout.addLayout(row("Structure rules", self.family, width=130))
        outer.addWidget(con)
        outer.addStretch(1)
        wrap = QtWidgets.QScrollArea()
        wrap.setWidgetResizable(True)
        wrap.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        wrap.setWidget(page)
        return wrap

    def _add_device_row(self, material: str | None) -> None:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        prev = self.dev_rows[-1].combo.currentData() if self.dev_rows else None
        combo = self._combo(self.dev_items, none=False, current=material or prev or "graphene")
        w.combo = combo
        h.addWidget(label(f"{len(self.dev_rows) + 1}.", "muted"))
        h.addWidget(combo)
        rm = button("Remove", lambda _=False, ww=w: self._remove_device_row(ww), None, "Remove this device layer")
        h.addWidget(rm)
        h.addStretch(1)
        self.dev_rows.append(w)
        self.dev_box.addWidget(w)
        modernize(w)

    def _remove_device_row(self, w: QtWidgets.QWidget) -> None:
        if len(self.dev_rows) <= 1:
            return
        self.dev_rows.remove(w)
        w.setParent(None)
        w.deleteLater()

    # ------------------------------------------------------------------ result
    def _create(self) -> None:
        name = self.name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "New device", "Give the device a name.")
            return
        page = self.PAGES[self.tabs.currentIndex()]
        if self.steps.currentWidget() is self.edit_page and self.editor is not None:     # the edited template
            self.spec = templates.new_device_from(self.editor.spec, name)
        elif page == "description":
            if self.describe.draft is None:
                QtWidgets.QMessageBox.warning(self, "New device", "Describe the device and press Draft first.")
                return
            self.spec = self.describe.draft.spec
            self.spec.name = name
        elif page == "template":
            it = self.tpl.currentItem()
            if it is None:
                QtWidgets.QMessageBox.warning(self, "New device", "Select a template, or build from scratch.")
                return
            self.spec = templates.new_device_from(templates.load_template(Path(it.data(0, Qt.ItemDataRole.UserRole))), name)
        else:
            family = self.family.currentData()
            devices = [w.combo.currentData() for w in self.dev_rows]
            self.spec = templates.build_from_scratch(
                name, self.config, self.sub.currentData(), self.bg.currentData() or None, devices, self.tg.currentData() or None,
                self.scheme.currentData(), self.metal.currentData() or None, self.bbn.isChecked(), self.tbn.isChecked(),
                self.cap.isChecked(), family, self.sd.currentData())
        self.accept()
