"""Library › Edit template: a template edited in its own window — name, description, structure rules,
contacts, source / drain style, and the layers (order, material, function, thicknesses … — the Stack
tab's layer editor working on the template's copy), with a 3D preview and its validation summary.

Nothing touches the open device. Save overwrites the template (user, 2026-11-21;
:func:`ustack.templates.update_template`): a lab template's file, or — for a built-in — the lab's edit of it,
which takes the built-in's place (Library › right-click › Restore built-in brings the original back).
New device › Edit Template is different: it never changes the template."""

from __future__ import annotations

from pathlib import Path

from .. import templates
from ..config import LabConfig
from ..model import CONTACT_SCHEMES, FAMILIES, SD_STYLES
from ..scene3d import build_scene
from ..validation import validate_stage1
from .stack3d import Stack3DView
from .stack_panel import StackPanel
from .style import Banner, ComboBox, Dialog, QtCore, QtWidgets, Splitter, button, label
from .widgets import confirm, modernize, prompt_text, row

Qt = QtCore.Qt


class _TemplateHost:
    """What the layer editor (:class:`StackPanel`) needs from its host, for a template being edited."""

    def __init__(self, editor: "TemplateEditor"):
        self.editor = editor

    @property
    def spec(self):
        return self.editor.spec

    @property
    def config(self):
        return self.editor.config

    def spec_changed(self, what: str = "") -> None:
        self.editor.changed()

    def layer_selected(self, uid: str) -> None:
        self.editor.view3d.set_highlight(uid)

    open_flake_images = None                         # templates carry no flake images


class TemplateEditor(QtWidgets.QWidget):
    """Name, description, structure rules, contacts, S/D style, the layers (the Stack tab's editor on ``spec``) and a 3D
    preview with the validation summary. ``on_change`` is called after every edit. Used by Library › Edit template and by
    New device › Edit template (user, 2026-11-21: adjust the template's layers before the device is created)."""

    def __init__(self, config: LabConfig, spec, on_change=None, parent=None, text_fields: bool = True):
        super().__init__(parent)
        self.config, self.spec, self.on_change = config, spec, on_change
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.name = QtWidgets.QLineEdit(self.spec.name)
        self.name.textEdited.connect(lambda _t: self.changed(False))
        self.name.setMinimumWidth(460)
        self.name.setMaximumWidth(520)
        self.desc = QtWidgets.QPlainTextEdit(self.spec.description)
        self.desc.setFixedHeight(62)
        self.desc.textChanged.connect(lambda: self.changed(False))
        if text_fields:                              # (New device has its own name; the template's name is asked on saving)
            lay.addLayout(row("Name", self.name, width=110))
            lay.addLayout(row("Description", self.desc, width=110, stretch=False))
        self.family = self._combo(FAMILIES, self.spec.family, "Which group's rules apply")
        self.scheme = self._combo(CONTACT_SCHEMES, self.spec.contact_scheme)
        self.sd = self._combo(SD_STYLES, self.spec.hall_bar.sd_style)
        opts = QtWidgets.QHBoxLayout()
        opts.setSpacing(14)
        for text, box in (("Structure rules", self.family), ("Contacts", self.scheme), ("Source / drain", self.sd)):
            opts.addLayout(row(text, box, width=110 if text == "Structure rules" else (70 if text == "Contacts" else 100)))
        opts.addStretch(1)
        lay.addLayout(opts)
        split = Splitter(Qt.Orientation.Horizontal)
        self.stack = StackPanel(_TemplateHost(self))
        self.stack.setMinimumWidth(420)
        split.addWidget(self.stack)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        self.view3d = Stack3DView()
        rl.addWidget(self.view3d, 1)
        self.summary = label("", "muted", wrap=True)
        rl.addWidget(self.summary)
        split.addWidget(right)
        split.setSizes([480, 640])
        lay.addWidget(split, 1)
        self.stack.refresh()
        self._preview()

    def _combo(self, items: dict, current: str, tip: str = "") -> ComboBox:
        box = ComboBox()
        for k, v in items.items():
            box.addItem(v, k)
        box.setCurrentIndex(max(0, box.findData(current)))
        box.setMinimumWidth(230)
        if tip:
            box.setToolTip(tip)
        box.activated.connect(lambda _i: self._options_changed())
        return box

    def _options_changed(self) -> None:
        self.spec.family = self.family.currentData()
        self.spec.contact_scheme = self.scheme.currentData()
        self.spec.hall_bar.sd_style = self.sd.currentData()
        self.changed()

    def changed(self, preview: bool = True) -> None:
        if preview:
            self._preview()
        if self.on_change is not None:
            self.on_change()

    def _preview(self) -> None:
        self.view3d.set_slabs(build_scene(self.spec, self.config) if self.spec.layers else [], "No layers")
        report = validate_stage1(self.spec, self.config)
        self.summary.setText(f"{len(self.spec.layers)} layers · {report.summary()}")

    def take_text(self) -> None:
        """Name and description from their fields into the spec."""
        self.spec.name = self.name.text().strip() or self.spec.name
        self.spec.description = self.desc.toPlainText().strip()


class TemplateEditorDialog(Dialog):
    def __init__(self, config: LabConfig, info: templates.TemplateInfo, parent=None):
        super().__init__(parent)
        self.config, self.info = config, info
        self.spec = templates.load_template(info)
        self.saved_path: Path | None = None
        self.dirty = False
        self.setWindowTitle(f"Edit template — {info.name}")
        self.resize(1240, 880)
        self.body.addWidget(Banner(
            "Save overwrites this template (a built-in's original can be restored: Library › right-click › Restore built-in)."
            if (info.builtin or info.overrides) else "Save overwrites this template.", "info"))
        self.editor = TemplateEditor(config, self.spec, self._edited, self)
        for attr in ("name", "desc", "family", "scheme", "sd", "stack", "view3d", "summary"):
            setattr(self, attr, getattr(self.editor, attr))
        self.body.addWidget(self.editor, 1)
        self.save_btn = button("Save template", self.save, "primary")
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(button("Close", self.close))
        foot.addWidget(self.save_btn)
        self.body.addLayout(foot)
        modernize(self)

    # ------------------------------------------------------------------ edits
    def _options_changed(self) -> None:
        self.editor._options_changed()

    def _edited(self) -> None:
        self.dirty = True
        self.setWindowTitle(f"Edit template — {self.info.name} •")

    def changed(self, preview: bool = True) -> None:
        self.editor.changed(preview)

    # ------------------------------------------------------------------ save / close
    def save(self) -> bool:
        self.editor.take_text()
        self.spec.name = self.spec.name or self.info.name
        path = templates.update_template(self.info, self.spec)         # id kept, even with a new name
        self.saved_path = Path(path)
        self.info = next((t for t in templates.list_templates() if Path(t.path) == self.saved_path), self.info)
        self.dirty = False
        self.save_btn.setText("Save template")
        self.setWindowTitle(f"Edit template — {self.info.name}")
        if self.parent() is not None and hasattr(self.parent(), "status"):
            self.parent().status(f"Template saved to {path}")
        return True

    def closeEvent(self, e):  # noqa: N802 - Qt API
        if self.dirty and confirm(self, "Save the template?", "Save your changes to the template before closing?", "Save"):
            if not self.save():
                e.ignore()
                return
        super().closeEvent(e)
