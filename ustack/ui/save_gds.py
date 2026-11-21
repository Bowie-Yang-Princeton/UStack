"""Stage 2 › Export GDS… / Merge & Export: where to write the file (user, 2026-11-20) — the device's own folder
(``library/devices/<device>/stage2/…``, the default) or any other folder, and the file's name."""

from __future__ import annotations

from pathlib import Path

from .style import CheckBox, Dialog, QtCore, QtWidgets, button, label
from .widgets import buttons_row, modernize

SETTINGS = ("Uri Lab", "UStack")


class SaveGdsDialog(Dialog):
    def __init__(self, parent, device_dir: Path, file_name: str, title: str = "Export GDS"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 260)
        self.device_dir = Path(device_dir)
        settings = QtCore.QSettings(*SETTINGS)
        other = str(settings.value("gds_dir", "") or "")
        self.body.addWidget(label("Where to write the GDS file", "cardTitle"))
        group = QtWidgets.QButtonGroup(self)
        self.in_device = QtWidgets.QRadioButton("In the device's folder")
        self.in_other = QtWidgets.QRadioButton("In another folder")
        group.addButton(self.in_device)
        group.addButton(self.in_other)
        self.in_device.setChecked(True)
        self.body.addWidget(self.in_device)
        dev = label(str(self.device_dir), "hint", wrap=True)
        dev.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(dev)
        self.body.addWidget(self.in_other)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(other)
        self.folder.setPlaceholderText("Choose a folder…")
        self.folder.textEdited.connect(lambda _t: self.in_other.setChecked(True))
        row.addWidget(self.folder, 1)
        row.addWidget(button("Browse…", self._browse, None, "Choose the folder"))
        self.body.addLayout(row)
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(label("File name", "fieldLabel"))
        self.name = QtWidgets.QLineEdit(file_name)
        name_row.addWidget(self.name, 1)
        self.body.addLayout(name_row)
        self.body.addStretch(1)
        self.ok = button("Save", self.accept, "primary")
        self.ok.setDefault(True)
        self.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Cancel", self.reject), self.ok))
        modernize(self)

    def _browse(self) -> None:
        start = self.folder.text().strip() or str(self.device_dir)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the GDS file", start)
        if path:
            self.folder.setText(path)
            self.in_other.setChecked(True)

    def path(self) -> Path | None:
        """The file to write (None when no folder is chosen for "another folder")."""
        name = self.name.text().strip() or "pattern.gds"
        if not name.lower().endswith(".gds"):
            name += ".gds"
        if self.in_device.isChecked():
            return self.device_dir / name
        folder = self.folder.text().strip()
        if not folder:
            return None
        QtCore.QSettings(*SETTINGS).setValue("gds_dir", folder)       # offered again next time
        return Path(folder).expanduser() / name


def ask_gds_path(parent, device_dir: Path, file_name: str, title: str = "Export GDS") -> Path | None:
    dlg = SaveGdsDialog(parent, device_dir, file_name, title)
    while dlg.exec():
        p = dlg.path()
        if p is not None:
            return p
        dlg.folder.setFocus()
    return None
