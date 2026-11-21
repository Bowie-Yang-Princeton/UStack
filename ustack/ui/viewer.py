"""A window that shows one image on its own (zoom, pan, scale bar in µm when the scale is known)."""

from __future__ import annotations

import numpy as np

from .canvas import Canvas
from .style import Dialog, QtWidgets, button, label, tool_strip_button
from .widgets import buttons_row, modernize


class ImageViewerDialog(Dialog):
    def __init__(self, image: np.ndarray, title: str, caption: str = "", um_per_px: float | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 780)
        self.setModal(False)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        self.canvas = Canvas()
        self.canvas.empty_text = ""
        self.canvas.fit_on_resize = True
        group = QtWidgets.QButtonGroup(self)
        self.tools = {}
        for tool, icon, tip in (("pointer", "pointer", "Drag to move the image around"),
                                ("measure", "measure", "Measure a length: click the start, then click the end (double-click or Esc clears)")):
            b = tool_strip_button(icon, tip, checkable=True)
            group.addButton(b)
            b.toggled.connect(lambda on, t=tool: on and self.canvas.set_tool(t))
            top.addWidget(b)
            self.tools[tool] = b
        self.tools["pointer"].setChecked(True)
        for icon, tip, fn in (("fit", "Fit (hold H or the middle button to pan)", self.canvas.fit),
                              ("zoomIn", "Zoom in", lambda: self.canvas.zoom_by(1.25)),
                              ("zoomOut", "Zoom out", lambda: self.canvas.zoom_by(0.8))):
            b = tool_strip_button(icon, tip)
            b.clicked.connect(fn)
            top.addWidget(b)
        self.info = label(caption, "muted", wrap=True)
        top.addWidget(self.info, 1)
        self.body.addLayout(top)
        self.body.addWidget(self.canvas, 1)
        h, w = image.shape[:2]
        s = um_per_px or 1.0
        self.canvas.unit = "µm" if um_per_px else "px"
        self.canvas.set_image("view", image, (s, 0.0, 0.0, float(h) * s), 1.0, 0)
        self.body.addLayout(buttons_row(QtWidgets.QWidget(), button("Close", self.accept, "primary")))
        modernize(self)

    def showEvent(self, e):  # noqa: N802 - Qt API
        super().showEvent(e)
        self.canvas.fit()
