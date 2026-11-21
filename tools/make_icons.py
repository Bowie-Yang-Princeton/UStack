"""Draw UStack's app mark and icon (three stacked sheets) into ustack/assets."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "ustack" / "assets"


def sheet(p, cx, cy, w, h, color, edge):
    poly = QtGui.QPolygonF([QtCore.QPointF(cx - w / 2, cy), QtCore.QPointF(cx, cy - h / 2), QtCore.QPointF(cx + w / 2, cy), QtCore.QPointF(cx, cy + h / 2)])
    p.setBrush(QtGui.QColor(color))
    p.setPen(QtGui.QPen(QtGui.QColor(edge), w * 0.025))
    p.drawPolygon(poly)


def draw(size: int, tile: bool) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    s = size
    if tile:
        p.setBrush(QtGui.QColor("#f2f3f4"))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(s * 0.06, s * 0.06, s * 0.88, s * 0.88), s * 0.2, s * 0.2)
    w, h = s * (0.74 if tile else 0.92), s * (0.37 if tile else 0.46)
    cx = s / 2
    for dy, col in ((0.18, "#9cc3e6"), (0.0, "#3fae9b"), (-0.18, "#1f3d6b")):
        sheet(p, cx, s / 2 + dy * s * (0.85 if tile else 1.0), w, h, col, "#193156")
    p.end()
    return img


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    OUT.mkdir(parents=True, exist_ok=True)
    draw(256, False).save(str(OUT / "UStack_mark.png"))
    draw(1024, True).save(str(OUT / "UStack_icon_1024.png"))
    print("icons written to", OUT)
