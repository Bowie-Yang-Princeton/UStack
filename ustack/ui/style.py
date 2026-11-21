"""House style for UStack (PyQt6 first, PySide6 fallback).

Vendored from the AFM data viewer (UAnalysis) ``afm_style.py`` so UStack looks and behaves the
same; only the app name, settings key, About text and asset folder differ. Keep it in sync by
copying the newer afm_style.py over this file and re-applying those four edits (see AGENTS.md).


The Uri Lab instrument look, ported from the lab's Transfer Stage software
(``transfer_stage/ui/style.py`` and ``chrome.py``) and pared down: greys and hairlines
carry the structure, one navy accent is kept for what is selected or primary,
headings are charcoal, corners are 3 px, every control sits on a 26 px row, the text
is Windows' own Segoe UI at 9 pt throughout (section titles in its bold); only the plots
use another face. Nothing glows, glides or floats: no
shadows, no gloss, no slide-in pages; state changes are immediate, and hover feedback
is a short fade of a grey key or of a frame. Day and night palettes.

Everything a stylesheet cannot do well is painted with QPainter:

* ``pixel_frame`` — 1 px rounded frames snapped to whole device pixels, so sides and
  corners carry the same weight at 125 / 150 % display scaling (Qt's QSS
  ``border-radius`` draws the corner arcs heavier than the straight edges).
* ``Panel`` / ``Card`` — regions and settings sections; the panel style (``inset`` by
  default) decides whether they are white blocks, groups under a title, or cards.
* ``Button`` / ``FieldFrame`` — one frame for buttons and fields (``control_frame``);
  primary = accent fill, a toggle that is on = a pressed-in grey key.
* ``IconButton`` / ``ToolStrip`` — flat tool buttons; one glyph colour in every state.
* ``TabBar`` (underline tabs), ``SegmentedControl`` (grey track, white key), ``CheckBox``,
  ``Slider`` and ``paint_knob`` (a flat fader cap), ``BusyRing`` (the one loading indicator).
* ``TitleBar`` / ``FramelessMainWindow`` — frameless chrome: day/night toggle, centred
  title, painted minimise / maximise / close, native move and edge resize (Aero Snap). On a
  Mac: ``TrafficLights`` (red / yellow / green on the left), resize by hand, and a menu bar
  (``install_mac_menu``).

All colours come from :data:`PALETTE`, which :data:`THEME` rewrites in place when the
appearance flips; painted widgets read it at paint time.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

try:
    from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets

    QT_API = "PyQt6"
    Signal = QtCore.pyqtSignal
except Exception:  # pragma: no cover - depends on the installed binding
    try:
        from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

        QT_API = "PySide6"
        Signal = QtCore.Signal
    except Exception as exc:  # pragma: no cover
        raise ImportError("UStack requires PyQt6 or PySide6. See README.md") from exc

Qt = QtCore.Qt
QColor, QPainter, QPen, QBrush = QtGui.QColor, QtGui.QPainter, QtGui.QPen, QtGui.QBrush
QPointF, QRectF, QRect, QSize, QPoint = QtCore.QPointF, QtCore.QRectF, QtCore.QRect, QtCore.QSize, QtCore.QPoint
QRadialGradient, QLinearGradient = QtGui.QRadialGradient, QtGui.QLinearGradient
QVariantAnimation, QEasingCurve, QTimer, QEvent = QtCore.QVariantAnimation, QtCore.QEasingCurve, QtCore.QTimer, QtCore.QEvent

NO_PEN = Qt.PenStyle.NoPen
NO_BRUSH = Qt.BrushStyle.NoBrush
ANTIALIAS = QPainter.RenderHint.Antialiasing
TEXT_ANTIALIAS = QPainter.RenderHint.TextAntialiasing
ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
ALIGN_LEFT_VCENTER = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
ALIGN_RIGHT_VCENTER = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
USER_ROLE = int(Qt.ItemDataRole.UserRole)

ASSET_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "assets"

#: macOS: the window chrome follows the Mac (red / yellow / green lights on the left, ⌘ shortcuts, a
#: menu bar) and the text sizes are converted, as Qt counts 72 points per inch there (96 on Windows).
MAC = sys.platform == "darwin"
#: The point sizes in this program are Windows points (96 per inch); on a Mac they are multiplied by
#: this, so every text keeps its size in pixels (the 9 pt interface text is 12 px on both).
PT_SCALE = 96 / 72 if MAC else 1.0


# --------------------------------------------------------------------------- palette / theme
# The Transfer Stage "Steel" skin: truer greys, crisp text, one navy accent. ``accent`` is the
# ink (titles, tabs, marks); ``accent_fill`` / ``_hover`` / ``_pressed`` sit under white text
# (primary buttons, ticks, selections) — at night a deeper navy than the ink, so white reads on it.
_DAY = dict(
    app_bg="#f2f3f4", surface="#ffffff", card="#ffffff", win_border="#c5c8cf",
    heading="#000000", text="#000000", text2="#000000", muted="#000000", faint="#8b9098",  # body text all black
    line="#dadcdf", line2="#c3c6ca", hover="#ebeced", pressed="#e3e4e6",
    flat_hover="#e3e4e6", flat_pressed="#d4d6d9", flat_checked="#dcdee1",
    win_glyph="#3f444c", track="#c3c6ca",
    accent="#1f3d6b", accent_fill="#1f3d6b", accent_hover="#35507a", accent_pressed="#193156",
    btn_dis_bg="#f5f5f5", btn_dis_fg="#b3b3b3", btn_dis_border="#e3e3e3",
    danger="#d05a5a", success="#2e8b57", warning="#9a6a2e",
)
_NIGHT = dict(
    app_bg="#222428", surface="#2d3035", card="#2d3035", win_border="#3f434a",
    heading="#f2f3f5", text="#f2f3f5", text2="#f2f3f5", muted="#f2f3f5", faint="#6f747d",  # one body ink, as by day
    line="#3e4148", line2="#4c5058", hover="#373a40", pressed="#41454c",
    flat_hover="#3a3d43", flat_pressed="#464a51", flat_checked="#42464e",
    win_glyph="#c4c7ce", track="#4c5058",
    accent="#6f93c9", accent_fill="#3b5f94", accent_hover="#46699f", accent_pressed="#314f7c",
    btn_dis_bg="#33363d", btn_dis_fg="#71767f", btn_dis_border="#43474f",
    danger="#e07a7a", success="#6cc28f", warning="#cf9c5c",
)
# Window-control hover (rgb, alpha) and the cards' shadow (rgb, rest alpha, hover alpha).
_DAY_FX = dict(win_hover=(0, 0, 0, 22), shadow_rgb=(15, 23, 42), shadow_a=38, shadow_a_hover=58)
_NIGHT_FX = dict(win_hover=(255, 255, 255, 30), shadow_rgb=(0, 0, 0), shadow_a=95, shadow_a_hover=125)


def mix(a: str, b: str, t: float) -> str:
    """``a`` blended toward ``b`` by ``t`` (0 → a, 1 → b), as ``#rrggbb``."""
    ca, cb = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t),
                  round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t)).name()


def _derive(base: dict, dark: bool) -> dict:
    """The full palette: the skin plus the tints the controls use."""
    p = dict(base)
    k = 1.6 if dark else 1.0
    p.update(
        accent2=p["accent_hover"], onAccent="#ffffff",
        accentBg=mix(p["surface"], p["accent"], 0.11 * k), accentDim=mix(p["line2"], p["accent"], 0.45),
        field_ro=mix(p["surface"], p["app_bg"], 0.7),
        successBg=mix(p["surface"], p["success"], 0.12 * k), successBorder=mix(p["line2"], p["success"], 0.5),
        warningBg=mix(p["surface"], p["warning"], 0.12 * k), warningBorder=mix(p["line2"], p["warning"], 0.5),
        dangerBg=mix(p["surface"], p["danger"], 0.10 * k), dangerBorder=mix(p["line2"], p["danger"], 0.5),
        info=p["accent"], infoBg=mix(p["surface"], p["accent"], 0.08 * k), infoBorder=mix(p["line2"], p["accent"], 0.35),
        graphite=p["text2"],
    )
    p.update(_NIGHT_FX if dark else _DAY_FX)
    return p


PALETTE: dict = _derive(_DAY, False)

if MAC:  # the Mac's system typeface (SF), like the Finder and System Settings
    UI_FONT = ".AppleSystemUIFont"
    UI_FONT_FALLBACKS = ("Helvetica Neue", "Helvetica", "Arial")
else:
    UI_FONT = "Segoe UI"  # the Windows system typeface, like Explorer and Settings
    UI_FONT_FALLBACKS = ("Arial", "Helvetica")
#: The interface face as matplotlib knows it (plot labels in the interface's type): Arial on a Mac,
#: whose system face is no font file matplotlib can load or embed in a PDF.
PLOT_UI_FONT = "Arial" if MAC else UI_FONT
UI_FONT_SIZE = 9  # every interface text, section titles included; only the plots use another face

# Corner radii: tight, instrument panelling rather than consumer-app softness.
R_FIELD = 3
R_BTN = 3
R_TOOL = 3
R_ROW = 2
R_CARD = 3
R_CHIP = 3
PANEL_RADIUS = CARD_RADIUS = CONTROL_RADIUS = R_CARD
#: Panel treatment (set it, or UANALYSIS_PANELS, before the window is created):
#:   "card"    — elevated white cards with a soft shadow that lifts on hover
#:   "divider" — one flat surface, regions and sections separated by hairlines
#:   "tonal"   — no boxes: the workspace is a white block, the side columns sit on the grey
#:   "flat"    — flat white rounded tiles, no shadow, no frame
#:   "outline" — white panels with a 1 px frame, no shadow
#:   "inset"   — like "tonal", but each settings section is a white group under its title
PANEL_STYLE = os.environ.get("USTACK_PANELS", os.environ.get("UANALYSIS_PANELS", "inset")).strip().lower()


def divider_panels() -> bool:
    return PANEL_STYLE == "divider"


#: Row height shared by every control that can sit on one line (fields, buttons, toggles).
H_CONTROL = 26
#: A field's QSS min-height is its CONTENT height: + 2 × 3 px padding + 2 × 1 px border = H_CONTROL.
H_FIELD_CONTENT = H_CONTROL - 8
TOOLTIP_DELAY_MS = 700


def _font_stack(*names: str) -> str:
    return ", ".join(f"'{name}'" for name in names) + ", sans-serif"


def build_stylesheet() -> str:
    """Stylesheet for the plain Qt controls; the painted widgets read PALETTE themselves."""
    p = dict(PALETTE, ui=_font_stack(UI_FONT, *UI_FONT_FALLBACKS), value_pt=f"{12 * PT_SCALE:g}", rf=R_FIELD, rb=R_BTN, rr=R_ROW, rc=R_CHIP, hfc=H_FIELD_CONTENT, win_key=WindowButton.WIDTH)
    return """
* { font-family: %(ui)s; }
QMainWindow, QDialog { background: %(app_bg)s; }
QWidget#root { background: transparent; }
QToolTip { background: %(surface)s; color: %(text)s; border: 1px solid %(line2)s; padding: 2px 7px; }
QLabel { background: transparent; color: %(text)s; }
QLabel[role="title"] { color: %(heading)s; }
QLabel[role="subtitle"] { color: %(muted)s; }
/* Section titles: the body face and size, set apart by weight and the accent ink only. Every bold
   is Segoe UI Bold with Windows' own hinting, like the scan names (see bold_font / label). */
QLabel[role="cardTitle"] { font-weight: 700; color: %(accent)s; }
/* The name of a thing (the open file, a scan group) rather than a section: heading ink. */
QLabel[role="itemTitle"] { font-weight: 700; color: %(heading)s; }
/* One text size across the interface (the app font, 10 pt): secondary text differs by ink only.
   Captions and values in the main ink; descriptive notes one step lighter; only counts and
   other incidental text use the faint grey. */
QLabel[role="fieldLabel"] { color: %(text)s; }
QLabel[role="muted"] { color: %(text2)s; }
QLabel[role="faint"] { color: %(muted)s; }  /* body text is one ink; faint stays for placeholders and disabled */
QLabel[role="hint"] { color: %(text2)s; }
QLabel[role="mono"] { color: %(text2)s; }
QLabel[role="value"] { font-size: %(value_pt)spt; font-weight: 700; color: %(heading)s; }
QLabel[role="num"] { color: %(text)s; }
QLabel[role="strong"] { font-weight: 600; color: %(text)s; }
QLabel[role="strongAccent"] { font-weight: 600; color: %(accent)s; }
/* Status words beside a title: plain small text, no badge. */
QLabel[pill="neutral"], QLabel[pill="accent"] { padding: 0px 2px; background: transparent; border: none; color: %(muted)s; }
QLabel[pill="ok"] { padding: 0px 2px; background: transparent; border: none; color: %(success)s; }
QLabel[pill="warn"] { padding: 0px 2px; background: transparent; border: none; color: %(warning)s; }
/* Notes: no tinted box, a 2 px rule on the left and the text in the secondary ink. */
QLabel[banner="info"], QLabel[banner="neutral"] { background: transparent; border: none; border-left: 2px solid %(line2)s; border-radius: 0px; color: %(text2)s; padding: 2px 0px 2px 10px; }
QLabel[banner="warn"] { background: transparent; border: none; border-left: 2px solid %(warning)s; border-radius: 0px; color: %(warning)s; padding: 2px 0px 2px 10px; }
QLabel[banner="ok"] { background: transparent; border: none; border-left: 2px solid %(success)s; border-radius: 0px; color: %(success)s; padding: 2px 0px 2px 10px; }
/* Plain push buttons (message boxes, file dialogs); the house Button paints itself. */
QPushButton { background: %(surface)s; color: %(text)s; border: 1px solid %(line2)s; border-radius: %(rb)spx; padding: 4px 12px; min-height: 16px; }
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed { background: %(pressed)s; }
QPushButton:disabled { background: %(btn_dis_bg)s; color: %(btn_dis_fg)s; border-color: %(btn_dis_border)s; }
QPushButton[house="true"], QPushButton[house="true"]:hover, QPushButton[house="true"]:pressed, QPushButton[house="true"]:checked, QPushButton[house="true"]:disabled { background: transparent; border: none; padding: 0; min-height: 0; }
QToolButton[flat="true"], QToolButton[flat="true"]:hover, QToolButton[flat="true"]:pressed, QToolButton[flat="true"]:checked { background: transparent; border: none; padding: 0; }
/* Fields: the QSS draws only the fill. The 1 px frame is painted by FieldFrame exactly like the
   buttons' (pixel-snapped, same colours / hover / focus); a QSS border-radius rasterises the corner
   arcs heavier than the sides and left stray corner pixels. The transparent border keeps the box. */
QLineEdit, QComboBox, QPlainTextEdit, QAbstractSpinBox { background: %(surface)s; color: %(text)s; border: 1px solid transparent; border-radius: %(rf)spx; padding: 3px 8px; min-height: %(hfc)spx; selection-background-color: %(accent_fill)s; selection-color: %(onAccent)s; }
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled { background: %(btn_dis_bg)s; color: %(btn_dis_fg)s; }
QAbstractSpinBox QLineEdit, QComboBox QLineEdit { border: none; padding: 0; background: transparent; }
QLineEdit:read-only { background: %(field_ro)s; color: %(text2)s; }
/* combobox-popup: 0 — a plain list that drops below the combo (above when there is no room) and
   honours maxVisibleItems, instead of Fusion's menu laid over the combo that can fill the screen. */
QComboBox { combobox-popup: 0; }  /* no padding-right: the 26 px drop-down already keeps the text clear of the chevron */
QComboBox::drop-down { border: none; width: 26px; background: transparent; }
QComboBox::down-arrow { image: none; width: 0px; height: 0px; }
/* Combo popups: one hairline frame on the container, 4 px inside it; the rows are painted by
   PopupItemDelegate (grey key under the pointer, the current entry in the accent with a tick). */
QComboBoxPrivateContainer { background: %(surface)s; border: 1px solid %(line2)s; border-radius: 0px; padding: 0px; }
QComboBox QAbstractItemView { background: %(surface)s; color: %(text)s; border: none; outline: 0; padding: 3px; selection-background-color: transparent; selection-color: %(text)s; }
QComboBox QAbstractItemView::item { background: transparent; border: none; padding: 0px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 0px; height: 0px; border: 0px; }
QListWidget { padding: 2px; outline: 0; background: %(surface)s; border: 1px solid %(line)s; border-radius: %(rf)spx; }
QListWidget::item { padding: 0px; border-radius: %(rr)spx; margin: 1px; color: %(text)s; background: transparent; }
QListWidget::item:hover, QListWidget::item:selected { background: transparent; color: %(text)s; }
/* The browser delegate paints its own row key: Qt must not fill the indent / branch column with
   the palette highlight (it showed as a blue block left of the selected row). */
QTreeView { background: transparent; border: none; outline: 0; selection-background-color: transparent; show-decoration-selected: 0; }
QTreeView::item, QTreeView::item:selected, QTreeView::item:hover, QTreeView::item:selected:active, QTreeView::item:selected:!active { border: none; padding: 0px; background: transparent; color: %(text)s; }
QTreeView::branch, QTreeView::branch:selected, QTreeView::branch:hover { background: transparent; border-image: none; image: none; }
QTableWidget, QTableView { background: %(surface)s; color: %(text)s; border: 1px solid %(line)s; border-radius: %(rf)spx; gridline-color: %(hover)s; alternate-background-color: %(field_ro)s; selection-background-color: %(app_bg)s; selection-color: %(text)s; outline: 0; }
QTableWidget::item { padding: 3px 8px; border: none; }
QHeaderView::section { background: %(surface)s; color: %(muted)s; border: none; border-bottom: 1px solid %(line)s; padding: 4px 8px; font-weight: 600; }
QTableCornerButton::section { background: %(surface)s; border: none; }
QTabWidget::pane { border: none; background: transparent; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; border: none; padding: 6px 12px; }
QTabBar QToolButton { background: transparent; border: none; }
QScrollArea, QAbstractScrollArea, QStackedWidget { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
/* Thin scroll bars: a 2 px hairline at rest that fills the 6 px slot on hover / drag (no reflow). */
/* Scroll bars: a 4 px rounded bar in a 10 px gutter, clear of the window edge; under the pointer
   it darkens and widens by a pixel each side (no jump from a hairline to a pill). */
/* Scroll bars show only while the pointer is over their area (or a bar is being dragged): the
   gutter keeps its room, so nothing reflows; _ScrollBarReveal sets the "shown" property. */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: transparent; min-height: 32px; border-radius: 2px; margin: 3px 3px; }
QScrollBar[shown="true"]::handle:vertical { background: %(line2)s; }
QScrollBar[shown="true"]::handle:vertical:hover, QScrollBar[shown="true"]::handle:vertical:pressed { background: %(faint)s; border-radius: 3px; margin: 3px 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: transparent; min-width: 32px; border-radius: 2px; margin: 3px 3px; }
QScrollBar[shown="true"]::handle:horizontal { background: %(line2)s; }
QScrollBar[shown="true"]::handle:horizontal:hover, QScrollBar[shown="true"]::handle:horizontal:pressed { background: %(faint)s; border-radius: 3px; margin: 2px 3px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: %(accentBg)s; border-radius: 2px; }
/* A border-radius makes Qt draw the menu itself (no native drop shadow); DWM rounds the window. */
QMenu { background: %(surface)s; color: %(text)s; border: 1px solid %(line)s; border-radius: %(rf)spx; padding: 4px; }
QMenu::item { padding: 5px 22px 5px 14px; border-radius: %(rr)spx; }
QMenu::item:selected { background: %(hover)s; color: %(text)s; }
QMenu::item:disabled { color: %(faint)s; }
QMenu::separator { height: 1px; background: %(line)s; margin: 4px 8px; }
QStatusBar { background: %(app_bg)s; color: %(muted)s; border-top: 1px solid %(line)s; padding-left: 8px; }
QStatusBar::item { border: none; }
QMessageBox QLabel { color: %(text)s; }
QDialog QPushButton { min-width: 76px; }
QPushButton[chrome="key"] { min-width: %(win_key)spx; max-width: %(win_key)spx; }
QPushButton[chrome="toggle"] { min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px; padding: 0px; }  /* an IconButton's square key */
QCheckBox { spacing: 8px; color: %(text)s; background: transparent; }
QCheckBox:disabled { color: %(faint)s; }
QProgressBar { background: transparent; border: none; }
""" % p


def _qpalette() -> QtGui.QPalette:
    """A QPalette matching the active appearance, so Fusion-painted surfaces follow day/night."""
    p = QtGui.QPalette()
    role, group = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    for r, key in ((role.Window, "app_bg"), (role.WindowText, "text"), (role.Base, "surface"), (role.AlternateBase, "hover"),
                   (role.Text, "text"), (role.Button, "surface"), (role.ButtonText, "text"), (role.ToolTipBase, "surface"),
                   (role.ToolTipText, "text"), (role.PlaceholderText, "faint"), (role.Highlight, "accent_fill"),
                   (role.HighlightedText, "onAccent"), (role.Link, "accent")):
        p.setColor(r, QColor(PALETTE[key]))
    p.setColor(group.Disabled, role.Text, QColor(PALETTE["btn_dis_fg"]))
    p.setColor(group.Disabled, role.ButtonText, QColor(PALETTE["btn_dis_fg"]))
    p.setColor(group.Disabled, role.WindowText, QColor(PALETTE["faint"]))
    return p


# --------------------------------------------------------------------------- colour / geometry helpers
def rgba(color: str, alpha: float) -> QColor:
    result = QColor(color)
    result.setAlphaF(max(0.0, min(1.0, alpha)))
    return result


def rounded_path(rect: QRectF, radius: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    radius = max(0.0, min(radius, rect.width() / 2, rect.height() / 2))
    path.addRoundedRect(rect, radius, radius)
    return path


def pixel_frame(p: QPainter, rect, radius: float, color, fill=None) -> None:
    """Stroke a rounded 1-unit frame just inside ``rect``, snapped to whole DEVICE pixels so all
    four sides and corners weigh the same at fractional display scales. ``fill`` paints the same
    snapped box first (a surface and its frame share one edge); ``color=None`` skips the stroke."""
    dt = p.deviceTransform()
    dpr = max(1.0, dt.m11())
    d = dt.mapRect(QRectF(rect))
    left, top = math.ceil(d.left() - 1e-6), math.ceil(d.top() - 1e-6)  # a widget owns only whole pixels
    right, bottom = math.floor(d.right() + 1e-6), math.floor(d.bottom() + 1e-6)
    if right - left < 2 or bottom - top < 2:
        return
    w = max(1, round(dpr))
    p.save()
    p.setWorldTransform(dt.inverted()[0], True)  # draw in device pixels
    p.setRenderHint(ANTIALIAS, True)
    if fill is not None:
        p.setPen(NO_PEN)
        p.setBrush(QColor(fill))
        p.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius * dpr, radius * dpr)
    if color is not None:
        p.setPen(QPen(QColor(color), w))
        p.setBrush(NO_BRUSH)
        half = w / 2.0
        rad = max(0.0, radius * dpr - half)
        p.drawRoundedRect(QRectF(left + half, top + half, right - left - w, bottom - top - w), rad, rad)
    p.restore()


def _shadow_stops(col: QColor, n: int = 9) -> list:
    """Gaussian-edge stops: full ``col`` at 0, clear at 1, the shape's true edge at 0.5."""
    out = []
    for i in range(n):
        s = i / (n - 1)
        c = QColor(col)
        c.setAlphaF(col.alphaF() * 0.5 * math.erfc((s - 0.5) * 2.0 * 1.4142))
        out.append((s, c))
    return out


def paint_soft_shadow(p: QPainter, rect: QRectF, blur: float, col: QColor) -> None:
    """Fill ``rect`` blurred by ``blur`` px: a solid core, four edge strips and four corners with a
    Gaussian falloff (painted directly — no QGraphicsEffect re-rendering the whole card)."""
    if col.alpha() <= 0 or blur <= 0:
        return
    b = blur
    core = rect.adjusted(b, b, -b, -b)
    if core.width() <= 0 or core.height() <= 0:
        return
    stops = _shadow_stops(col)
    p.setPen(NO_PEN)
    p.setBrush(col)
    p.drawRect(core)

    def strip(area: QRectF, x0, y0, x1, y1) -> None:
        g = QLinearGradient(x0, y0, x1, y1)
        g.setStops(stops)
        p.setBrush(g)
        p.drawRect(area)

    left, top, right, bottom = core.left(), core.top(), core.right(), core.bottom()
    strip(QRectF(left, top - 2 * b, core.width(), 2 * b), 0, top, 0, top - 2 * b)
    strip(QRectF(left, bottom, core.width(), 2 * b), 0, bottom, 0, bottom + 2 * b)
    strip(QRectF(left - 2 * b, top, 2 * b, core.height()), left, 0, left - 2 * b, 0)
    strip(QRectF(right, top, 2 * b, core.height()), right, 0, right + 2 * b, 0)
    for cx, cy, x, y in ((left, top, left - 2 * b, top - 2 * b), (right, top, right, top - 2 * b),
                         (left, bottom, left - 2 * b, bottom), (right, bottom, right, bottom)):
        g = QRadialGradient(cx, cy, 2 * b)
        g.setStops(stops)
        p.setBrush(g)
        p.drawRect(QRectF(x, y, 2 * b, 2 * b))


def scaled_font(font: QtGui.QFont, delta_pt: float, bold: bool | None = None) -> QtGui.QFont:
    """Copy of ``font`` ``delta_pt`` points larger/smaller — works for pixel-sized fonts too.
    ``bold`` gives the house bold (:func:`bold_font`)."""
    result = QtGui.QFont(font)
    if result.pointSizeF() > 0:
        result.setPointSizeF(max(6.0 * PT_SCALE, result.pointSizeF() + delta_pt * PT_SCALE))
    else:
        result.setPixelSize(max(8, int(round(result.pixelSize() + delta_pt * 4 / 3))))
    if bold:
        result = bold_font(result)
    elif bold is not None:
        result.setWeight(QtGui.QFont.Weight.Normal)
    return result


def bold_font(font: QtGui.QFont) -> QtGui.QFont:
    """The house bold: Segoe UI Bold drawn with Windows' own hinting. The app font is unhinted
    (true outlines for the body text), but unhinted Bold at 9 pt smears into a heavy, soft weight;
    hinted it is the crisp bold of the scan names, which every bold label, tab and row now shares."""
    result = QtGui.QFont(font)
    result.setWeight(QtGui.QFont.Weight.Bold)
    result.setHintingPreference(QtGui.QFont.HintingPreference.PreferDefaultHinting)
    return result


#: Label roles drawn in the house bold (the QSS sets their weight; label() their hinting).
BOLD_ROLES = ("cardTitle", "itemTitle", "value")


def _animation(owner: QtCore.QObject, duration: int, curve=QEasingCurve.Type.OutCubic) -> QVariantAnimation:
    anim = QVariantAnimation(owner)
    anim.setDuration(duration)
    anim.setEasingCurve(curve)
    return anim


def _retarget(anim: QVariantAnimation, start: float, end: float, duration: int | None = None) -> None:
    anim.stop()
    if duration is not None:
        anim.setDuration(max(1, int(duration)))
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.start()


class HoverFade:
    """A 0 → 1 hover progress that eases in and out (an interrupted fade only runs for the
    distance it still has to cover). Calls ``widget.update()`` on every step."""

    def __init__(self, widget: QtWidgets.QWidget, in_ms: int = 140, out_ms: int = 180, curve=QEasingCurve.Type.OutCubic,
                 on_step: Callable[[], None] | None = None) -> None:
        self.value = 0.0
        self._in, self._out = in_ms, out_ms
        self._step = on_step or widget.update
        self._anim = _animation(widget, in_ms, curve)
        self._anim.valueChanged.connect(self._set)

    def _set(self, value) -> None:
        self.value = float(value)
        self._step()

    def to(self, target: float) -> None:
        span = abs(target - self.value)
        _retarget(self._anim, self.value, target, (self._in if target > self.value else self._out) * span)

    def reset(self) -> None:
        self._anim.stop()
        self.value = 0.0
        self._step()


# --------------------------------------------------------------------------- icons
_SVG = {
    # a plain arrow head: tip at the upper left, a short notched tail — no hand, no outline tricks
    "pointer": "<path d='M4.4 4.4l11.8 4.9-4.6 1.4c-.4.1-.7.4-.8.8l-1.4 4.6z'/>",
    "pan": "<path d='M6.6 11.6V6.2a1.25 1.25 0 012.5 0v4.3'/><path d='M9.1 10.5V4.4a1.25 1.25 0 012.5 0v6.1'/><path d='M11.6 10.5V5.6a1.25 1.25 0 012.5 0v5.6'/><path d='M14.1 11.2V7.8a1.25 1.25 0 012.5 0v4.4c0 3.3-2.3 5.4-5.5 5.4h-.6c-2.1 0-3.4-.8-4.4-2.4l-3-4.6a1.2 1.2 0 011.9-1.4l1.6 2'/>",
    "crop": "<path d='M6 2.5V14h11.5'/><path d='M2.5 6H14v11.5'/>",
    # UStack: border editing — a point added on an edge (with a plus), points caught in a dashed box
    "addPoint": "<path d='M2.5 15.5l9-9'/><rect x='5.4' y='10.1' width='3' height='3' fill='{c}'/><path d='M14.5 11v6M11.5 14h6'/>",
    "selectPoints": "<path d='M3 3h14v14H3z' stroke-dasharray='2 2'/><rect x='6' y='6.5' width='3' height='3' fill='{c}'/><rect x='11' y='10.5' width='3' height='3' fill='{c}'/>",
    # the tool's own mark: a line between two open end points, as measurement lines are drawn
    "measure": "<path d='M5.7 14.3l8.6-8.6'/><circle cx='4.4' cy='15.6' r='1.8'/><circle cx='15.6' cy='4.4' r='1.8'/>",
    # the tool's own marks: an arrow (narrow open head) and a cross marker, as annotations are drawn
    "annotate": "<path d='M4 16L14.6 5.4'/><path d='M9.2 4.6h6.2v6.2'/><path d='M13.2 13.2l3.6 3.6M16.8 13.2l-3.6 3.6'/>",
    # the mask tool: a drawn region, its inside lightly filled (as the mask lies over the image)
    "mask": "<path d='M4.5 6.5l6.5-3 5 4.5-1.5 8-9.5-1.5z' fill='{c}' fill-opacity='.28'/>",
    "fit": "<path d='M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4'/>",
    "zoomIn": "<circle cx='9' cy='9' r='5.5'/><path d='M13.2 13.2L17.5 17.5'/><path d='M9 6.5v5M6.5 9h5'/>",
    "zoomOut": "<circle cx='9' cy='9' r='5.5'/><path d='M13.2 13.2L17.5 17.5'/><path d='M6.5 9h5'/>",
    "undo": "<path d='M7.5 5.5L4 9l3.5 3.5'/><path d='M4 9h8.5a3.5 3.5 0 010 7H9'/>",
    "redo": "<path d='M12.5 5.5L16 9l-3.5 3.5'/><path d='M16 9H7.5a3.5 3.5 0 000 7H11'/>",
    "reset": "<path d='M5 10a5 5 0 105-5H7.5'/><path d='M9.5 2.8L7.3 5l2.2 2.2'/>",
    "eye": "<path d='M2.5 10s3-5 7.5-5 7.5 5 7.5 5-3 5-7.5 5-7.5-5-7.5-5z'/><circle cx='10' cy='10' r='2.3'/>",
    "eyeOff": "<path d='M2.5 10s3-5 7.5-5 7.5 5 7.5 5-3 5-7.5 5-7.5-5-7.5-5z'/><circle cx='10' cy='10' r='2.3'/><path d='M4 4l12 12'/>",
    "trash": "<path d='M4.5 6h11M8 6V4.5h4V6M6 6l.7 10h6.6L14 6'/><path d='M8.5 9v4.5M11.5 9v4.5'/>",
    "copy": "<rect x='7' y='7' width='9.5' height='9.5' rx='1.6'/><path d='M13 7V5a1.5 1.5 0 00-1.5-1.5H5A1.5 1.5 0 003.5 5v6.5A1.5 1.5 0 005 13h2'/>",
    "folder": "<path d='M3 5.5A1.5 1.5 0 014.5 4h3.6l1.8 2h5.6A1.5 1.5 0 0117 7.5v7a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 14.5z'/>",
    "file": "<path d='M5 3h7l4 4v10H5z'/><path d='M12 3v4h4'/>",
    "image": "<rect x='3' y='4' width='14' height='12' rx='1.8'/><path d='M3 13.5l4-4 3 3 2.5-2.5 4.5 4.5'/><circle cx='13' cy='7.5' r='1.2' fill='{c}' stroke='none'/>",
    "layers": "<path d='M10 3.5l7 3.5-7 3.5-7-3.5z'/><path d='M3 10.5l7 3.5 7-3.5M3 14l7 3.5 7-3.5'/>",
    "chart": "<path d='M3.5 16.5h13'/><path d='M5.5 13.5v-4M9 13.5V6M12.5 13.5V9.5M16 13.5V4'/>",
    "export": "<path d='M10 3v9'/><path d='M6.5 8.5L10 12l3.5-3.5'/><path d='M4 14.5v1A1.5 1.5 0 005.5 17h9a1.5 1.5 0 001.5-1.5v-1'/>",
    "search": "<circle cx='9' cy='9' r='5.5'/><path d='M13.2 13.2L17.5 17.5'/>",
    "close": "<path d='M5 5l10 10M15 5L5 15'/>",
    "check": "<path d='M4.5 10.5l3.5 3.5 7.5-8'/>",
    "chevronDown": "<path d='M5 8l5 5 5-5'/>",
    "chevronRight": "<path d='M8 5l5 5-5 5'/>",
    "chevronUp": "<path d='M5 12l5-5 5 5'/>",
    "info": "<circle cx='10' cy='10' r='7.25'/><path d='M10 9v5'/><circle cx='10' cy='6.4' r='0.9' fill='{c}' stroke='none'/>",
    "sidebarLeft": "<rect x='3' y='4' width='14' height='12' rx='1.8'/><path d='M7.5 4v12'/>",
    "sidebarRight": "<rect x='3' y='4' width='14' height='12' rx='1.8'/><path d='M12.5 4v12'/>",
    "sidebarBottom": "<rect x='3' y='4' width='14' height='12' rx='1.8'/><path d='M3 11.5h14'/>",
    "wave": "<path d='M2.5 10c2 0 2-5 3.75-5s1.75 10 3.75 10 1.75-10 3.75-10 1.75 5 3.75 5'/>",
    "cursor": "<path d='M10 3v14'/><path d='M7 3h6M7 17h6'/><path d='M3.5 10h4M12.5 10h4'/>",
    "plus": "<path d='M10 4.5v11M4.5 10h11'/>",
    "minus": "<path d='M4.5 10h11'/>",
    "arrowLeft": "<path d='M16 10H4.5'/><path d='M9 5.5L4.5 10 9 14.5'/>",
    "arrowRight": "<path d='M4 10h11.5'/><path d='M11 5.5l4.5 4.5-4.5 4.5'/>",
    "arrowUp": "<path d='M10 16V4.5'/><path d='M5.5 9L10 4.5 14.5 9'/>",
    "arrowDown": "<path d='M10 4v11.5'/><path d='M5.5 11l4.5 4.5 4.5-4.5'/>",
    # Solid glyphs (folder picker): a filled shape, a lighter back part where there is one.
    "folderFill": "<path d='M2.5 5.2A1.7 1.7 0 014.2 3.5h3.6c.4 0 .8.2 1.1.5l1.3 1.5h5.6a1.7 1.7 0 011.7 1.7v.8h-15z' fill='{c}' stroke='none' opacity='.55'/>"
                  "<path d='M2.5 8.2h15v6.6a1.7 1.7 0 01-1.7 1.7H4.2a1.7 1.7 0 01-1.7-1.7z' fill='{c}' stroke='none'/>",
    "homeFill": "<path d='M10 2.8l7.2 6.1v7.1a1.5 1.5 0 01-1.5 1.5H12.6v-4.6a1 1 0 00-1-1H8.4a1 1 0 00-1 1v4.6H4.3A1.5 1.5 0 012.8 16V8.9z' fill='{c}' stroke='none'/>",
    "desktopFill": "<rect x='2.5' y='3.5' width='15' height='10' rx='1.6' fill='{c}' stroke='none'/><path d='M7 16.8h6M10 13.5v3.3'/>",
    "documentFill": "<path d='M5.7 2.5h5.6l4 4v9.3a1.7 1.7 0 01-1.7 1.7H5.7A1.7 1.7 0 014 15.8V4.2a1.7 1.7 0 011.7-1.7z' fill='{c}' stroke='none' opacity='.55'/>"
                    "<path d='M11.3 2.5l4 4h-2.6a1.4 1.4 0 01-1.4-1.4z' fill='{c}' stroke='none'/>",
    "downloadFill": "<path d='M10 3v8.3M6.6 8.2l3.4 3.4 3.4-3.4'/><path d='M3 13.5h14v1.8a1.7 1.7 0 01-1.7 1.7H4.7A1.7 1.7 0 013 15.3z' fill='{c}' stroke='none'/>",
    "pcFill": "<rect x='3.5' y='4' width='13' height='9.5' rx='1.4' fill='{c}' stroke='none'/><path d='M2 16h16'/>",
    "cloudFill":"<path d='M6 15.5a3.5 3.5 0 01-.5-7 4.7 4.7 0 019.1.6 3.2 3.2 0 01-.2 6.4z' fill='{c}' stroke='none'/>",
    "driveFill": "<path d='M4.4 5h11.2l1.9 6.5h-15z' fill='{c}' stroke='none' opacity='.55'/>"
                 "<rect x='2.5' y='11' width='15' height='5' rx='1.4' fill='{c}' stroke='none'/>",
}


#: Button icons drawn for a 16 px square: 1 px strokes on half-pixel coordinates with square caps,
#: so straight lines land on whole device pixels at 100 % (the 20 px set scaled to 16 came out as
#: 1.3 px grey smears). Dots and knobs are filled whole-pixel squares.
_SVG16 = {
    "copy": "<path d='M5.5 5.5h8v8h-8z'/><path d='M10.5 5.5v-3h-8v8h3'/>",
    "export": "<path d='M7.5 2.5v7'/><path d='M4.5 6.5l3 3 3-3'/><path d='M2.5 10.5v3h10v-3'/>",
    "file": "<path d='M3.5 1.5h6l3 3v10h-9z'/><path d='M9.5 1.5v3h3'/>",
    "trash": "<path d='M2.5 3.5h11'/><path d='M6.5 3.5v-2h3v2'/><path d='M4.5 3.5v10h7v-10'/><path d='M6.5 6.5v4M9.5 6.5v4'/>",
    "folder": "<path d='M1.5 2.5h4l2 2h7v9h-13z'/>",
    "plus": "<path d='M7.5 3.5v8M3.5 7.5h8'/>",
    "check": "<path d='M3 8l3 3 6.5-6.5'/>",
    "chart": "<path d='M1.5 13.5h13'/><path d='M3.5 11.5v-4M6.5 11.5v-7M9.5 11.5v-5M12.5 11.5v-9'/>",
    "layers": "<path d='M8 2.5l6 3-6 3-6-3z'/><path d='M2 9l6 3 6-3'/>",
    # the reciprocal-space editor: a framed spot pattern
    "spectrum": "<path d='M1.5 1.5h13v13h-13z'/>"
                "<path d='M7 7h2v2h-2zM3 7h2v2h-2zM11 7h2v2h-2zM7 3h2v2h-2zM7 11h2v2h-2z' fill='{c}' stroke='none'/>",
    # pick a reference line on the image: a frame crossed by a line with its grip
    "pickLine": "<path d='M1.5 2.5h13v11h-13z'/><path d='M1.5 7.5h13'/><path d='M6 6h4v3h-4z' fill='{c}' stroke='none'/>",
    # options that fold out (Exclude features): two slider tracks with their knobs
    "sliders": "<path d='M1.5 4.5h13M1.5 11.5h13'/><path d='M4 3h3v3h-3zM9 10h3v3h-3z' fill='{c}' stroke='none'/>",
}


def _svg_bytes(name: str, color: str, crisp: bool = False) -> bytes:
    if crisp and name in _SVG16:
        body = _SVG16[name].replace("{c}", color)
        return (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' width='16' height='16'>"
            f"<g fill='none' stroke='{color}' stroke-width='1' stroke-linecap='square' stroke-linejoin='miter'>{body}</g></svg>"
        ).encode("utf-8")
    body = _SVG[name].replace("{c}", color)
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' width='20' height='20'>"
        f"<g fill='none' stroke='{color}' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'>{body}</g></svg>"
    ).encode("utf-8")


_pixmap_cache: dict[tuple[str, str, int, float], QtGui.QPixmap] = {}


def device_ratio() -> float:
    screen = QtGui.QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen else 1.0


def icon_pixmap(name: str, color: str, size: int, crisp: bool = False) -> QtGui.QPixmap:
    """Icon ``name`` rasterised in ``color`` straight to ``size × dpr`` device pixels (blitted 1:1).
    ``crisp``: the 16 px pixel-grid drawing where there is one (button icons)."""
    ratio = device_ratio()
    key = (name, color, size, ratio, crisp)
    cached = _pixmap_cache.get(key)
    if cached is not None:
        return cached
    pixels = max(1, int(round(size * ratio)))
    image = QtGui.QImage(pixels, pixels, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(_svg_bytes(name, color, crisp)))
    painter = QPainter(image)
    painter.setRenderHint(ANTIALIAS)
    renderer.render(painter, QRectF(0, 0, pixels, pixels))
    painter.end()
    pixmap = QtGui.QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(ratio)
    _pixmap_cache[key] = pixmap
    return pixmap


def _draw_icon(painter: QPainter, name: str, color: str, size: int, centre: QPointF, crisp: bool = False) -> None:
    painter.drawPixmap(QPointF(round(centre.x() - size / 2), round(centre.y() - size / 2)), icon_pixmap(name, color, size, crisp))


# --------------------------------------------------------------------------- theme manager
#: How long the old appearance lingers while the new one comes up underneath.
THEME_FADE_MS = 260


class _ThemeVeil(QtWidgets.QWidget):
    """A frozen snapshot of a window's OLD appearance, dissolved over the new one, so day and
    night cross-fade instead of the whole UI inverting between two frames."""

    def __init__(self, window: QtWidgets.QWidget, shot: QtGui.QPixmap) -> None:
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._shot = shot
        self.setGeometry(window.rect())
        self.raise_()
        self.show()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._shot)

    def dissolve(self) -> None:
        effect = QtWidgets.QGraphicsOpacityEffect(self)
        effect.setOpacity(1.0)
        self.setGraphicsEffect(effect)
        anim = QtCore.QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(THEME_FADE_MS)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.finished.connect(self.deleteLater)
        anim.start()
        self._anim = anim


class _Theme(QtCore.QObject):
    """Global day/night controller: rewrites PALETTE in place, restyles the app and repaints."""

    changed = Signal()
    SETTINGS = ("Uri Lab", "UStack")

    def __init__(self) -> None:
        super().__init__()
        self.dark = False

    def saved_dark(self) -> bool:
        return str(QtCore.QSettings(*self.SETTINGS).value("appearance/theme", "light")).lower() == "dark"

    def set_dark(self, dark: bool, animate: bool = True, persist: bool = True) -> None:
        dark = bool(dark)
        changed = dark != self.dark
        app = QtWidgets.QApplication.instance()
        veils = []
        if animate and changed and app is not None:
            for window in app.topLevelWidgets():
                if window.isVisible() and window.width() > 2 and window.windowType() != Qt.WindowType.Popup and not isinstance(window, _ThemeVeil):
                    shot = window.grab()
                    if not shot.isNull():
                        veils.append(_ThemeVeil(window, shot))
        self.dark = dark
        self.apply()
        for veil in veils:
            veil.dissolve()
        if persist:
            QtCore.QSettings(*self.SETTINGS).setValue("appearance/theme", "dark" if dark else "light")

    def toggle(self) -> None:
        self.set_dark(not self.dark)

    def apply(self) -> None:
        PALETTE.clear()
        PALETTE.update(_derive(_NIGHT if self.dark else _DAY, self.dark))
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet())
            app.setPalette(_qpalette())
            for widget in app.allWidgets():
                widget.update()
            for window in app.topLevelWidgets():
                if window.isVisible():
                    apply_window_chrome(window)
        self.changed.emit()


THEME = _Theme()


# --------------------------------------------------------------------------- application / window helpers
class HintStyle(QtWidgets.QProxyStyle):
    """Application style: short tooltip delay, scroll bars in their own gutter (never over the controls)."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802 - Qt API
        if hint == QtWidgets.QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return TOOLTIP_DELAY_MS
        if hint == QtWidgets.QStyle.StyleHint.SH_ToolTip_FallAsleepDelay:
            return 400
        if hint == QtWidgets.QStyle.StyleHint.SH_ScrollBar_Transient:
            return 0
        return super().styleHint(hint, option, widget, returnData)


def _dwm_attribute(window: QtWidgets.QWidget, attribute: int, value: int) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        data = ctypes.c_int(value)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(int(window.winId())), ctypes.c_uint(attribute), ctypes.byref(data), ctypes.sizeof(data))
    except Exception:
        pass


def _colorref(color: str) -> int:
    """A Win32 COLORREF (0x00BBGGRR) for ``color``."""
    c = QColor(color)
    return c.red() | (c.green() << 8) | (c.blue() << 16)


def apply_window_chrome(window: QtWidgets.QWidget) -> None:
    """Windows 11 frame details: a dark native title bar at night (dialogs) and rounded corners
    for floating windows (square when maximised, so a maximised window fills the screen)."""
    _dwm_attribute(window, 20, 1 if THEME.dark else 0)  # DWMWA_USE_IMMERSIVE_DARK_MODE
    _dwm_attribute(window, 33, 1 if window.isMaximized() else 2)  # DWMWA_WINDOW_CORNER_PREFERENCE


#: Tooltip geometry: side padding, the gap above the capitals and below the baseline, the share of
#: the descent added below (so descenders do not crowd the edge), the widest line before the text
#: wraps, and the line pitch as a multiple of the cap height.
TIP_PAD_X, TIP_PAD_Y, TIP_DESCENT_ROOM, TIP_MAX_WIDTH, TIP_LINE_PITCH = 9, 6, 0.55, 380, 1.9


def _tooltip_lines(tip: QtWidgets.QLabel) -> tuple[list[str], QtGui.QFontMetricsF]:
    """The tooltip's text as plain lines, word-wrapped at ``TIP_MAX_WIDTH``."""
    text = tip.text()
    if Qt.mightBeRichText(text):
        document = QtGui.QTextDocument(); document.setHtml(text); text = document.toPlainText()
    text = shortcut_text(text)  # ⌘O rather than Ctrl+O on a Mac
    metrics = QtGui.QFontMetricsF(tip.font())
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}" if line else word
            if line and metrics.horizontalAdvance(trial) > TIP_MAX_WIDTH:
                lines.append(line); line = word
            else:
                line = trial
        lines.append(line)
    return lines, metrics


def _ink_metrics(metrics: QtGui.QFontMetricsF) -> tuple[float, float]:
    """(cap height, descender depth) measured on the glyph outlines: the font's own figures
    differ between platforms (Windows reports a taller cap and a deeper descent than drawn)."""
    return -metrics.tightBoundingRect("H").top(), metrics.tightBoundingRect("gjpqy()").bottom()


def _tooltip_size(tip: QtWidgets.QLabel) -> QSize:
    """Avenir's line box is half again its size, so Qt's own layout leaves a tall, lopsided
    popup; the house tooltip is sized on the ink instead: pad, cap height (+ part of the
    descenders), pad, per line."""
    lines, metrics = _tooltip_lines(tip)
    cap, descent = _ink_metrics(metrics)
    width = max(metrics.horizontalAdvance(line) for line in lines) if lines else 0.0
    tail = TIP_DESCENT_ROOM * descent  # p, g and brackets hang below the baseline
    height = 2 * TIP_PAD_Y + cap + tail + (len(lines) - 1) * round(cap * TIP_LINE_PITCH)
    return QSize(math.ceil(width) + 2 * TIP_PAD_X, int(round(height)))


def _paint_tooltip(tip: QtWidgets.QLabel) -> None:
    lines, metrics = _tooltip_lines(tip)
    painter = QPainter(tip)
    painter.setRenderHint(TEXT_ANTIALIAS)
    bounds = QRectF(tip.rect())
    painter.fillRect(bounds, QColor(PALETTE["surface"]))
    painter.setPen(QPen(QColor(PALETTE["line2"]), 1))
    painter.drawRect(bounds.adjusted(0.5, 0.5, -0.5, -0.5))
    painter.setFont(tip.font())
    painter.setPen(QColor(PALETTE["text"]))
    ratio = tip.devicePixelRatioF()
    cap = _ink_metrics(metrics)[0]
    pitch = round(cap * TIP_LINE_PITCH)
    for number, line in enumerate(lines):
        baseline = round((TIP_PAD_Y + cap + number * pitch) * ratio) / ratio  # on a whole device pixel
        painter.drawText(QPointF(TIP_PAD_X, baseline), line)
    painter.end()


class _PopupCorners(QtCore.QObject):
    """Round Qt's tooltips and menus on Windows 11 through DWM (a QSS radius on a top-level popup
    shows square black corners), and build tooltips without the grey native drop shadow."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        try:
            if isinstance(obj, QtWidgets.QWidget):
                name = obj.metaObject().className()
                if name == "QTipLabel":
                    if not obj.testAttribute(Qt.WidgetAttribute.WA_WState_Created) and not (obj.windowFlags() & Qt.WindowType.NoDropShadowWindowHint):
                        obj.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
                    kind = event.type()
                    if kind in (QEvent.Type.Show, QEvent.Type.Resize):  # Qt resizes it on every new text
                        size = _tooltip_size(obj)
                        if obj.size() != size:
                            obj.resize(size)
                    if kind == QEvent.Type.Paint:
                        _paint_tooltip(obj)
                        return True
                    if kind == QEvent.Type.Show:
                        _dwm_attribute(obj, 33, 3)  # DWMWCP_ROUNDSMALL
                elif name in ("QMenu", "QComboBoxPrivateContainer") and event.type() == QEvent.Type.Show:
                    _dwm_attribute(obj, 33, 3)  # DWMWCP_ROUNDSMALL, like the 3 px controls
                    _dwm_attribute(obj, 34, _colorref(PALETTE["line2"]))  # DWMWA_BORDER_COLOR: the field hairline
        except Exception:
            pass
        return False


_filters: list[tuple[QtCore.QObject, QtCore.QObject]] = []  # (watched object, filter)


def watch(target: QtCore.QObject, item: QtCore.QObject) -> None:
    """Install ``item`` as an event filter on ``target`` and remove it again on ``aboutToQuit``.
    A Python ``eventFilter`` still installed while Qt tears the widgets down is called after the
    interpreter has started finalising, which crashes the process on exit — so every Python
    filter goes through here."""
    app = QtWidgets.QApplication.instance()
    target.installEventFilter(item)
    if not _filters and app is not None:
        app.aboutToQuit.connect(_remove_filters)
    _filters.append((target, item))


def install_app_filter(item: QtCore.QObject) -> None:
    """An application-wide event filter (see :func:`watch`)."""
    watch(QtWidgets.QApplication.instance(), item)


def _remove_filters() -> None:
    while _filters:
        target, item = _filters.pop()
        try:
            target.removeEventFilter(item)
        except RuntimeError:  # already deleted
            pass


def _report_exception(exc_type, exc_value, exc_traceback) -> None:
    """PyQt6 aborts the process when a Python exception escapes a slot or a paint handler and
    ``sys.excepthook`` is the default; logging it instead keeps the app alive."""
    import traceback

    traceback.print_exception(exc_type, exc_value, exc_traceback)


class _ScrollBarReveal(QtCore.QObject):
    """Scroll bars appear while the pointer is over their scroll area and go when it leaves (a bar
    held in a drag stays until it is let go). Only the handle's paint changes (the "shown"
    property, see the QSS), never the layout, so the content does not move."""

    WATCHED = (QEvent.Type.Enter, QEvent.Type.Leave, QEvent.Type.MouseButtonRelease, QEvent.Type.Show)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in self.WATCHED and isinstance(obj, QtWidgets.QWidget):
            widget = obj
            while widget is not None:  # every scroll area the event's widget sits in (they nest)
                if isinstance(widget, QtWidgets.QAbstractScrollArea):
                    self.update(widget)
                widget = widget.parentWidget()
        return False

    @staticmethod
    def update(area: QtWidgets.QAbstractScrollArea) -> None:
        try:
            bars = (area.verticalScrollBar(), area.horizontalScrollBar())
            inside = area.isVisible() and area.rect().contains(area.mapFromGlobal(QtGui.QCursor.pos()))
            pointed = inside or any(bar.isSliderDown() for bar in bars)
            for bar in bars:
                shown = pointed and bar.maximum() > bar.minimum()  # an always-on bar with nothing to scroll stays blank
                if bar.property("shown") != shown:
                    bar.setProperty("shown", shown)
                    bar.style().unpolish(bar); bar.style().polish(bar); bar.update()
        except RuntimeError:  # deleted in the meantime
            pass


class _FocusGuard(QtCore.QObject):
    """Keeps text fields from holding or taking focus nobody gave them.

    * Buttons, toggles, sliders and check boxes take no focus, so a field once clicked kept its
      accent frame and caret through every later click — and came back lit whenever the window was
      re-activated, the wheel then changing its value. A click on anything that does not take focus
      itself now ends the edit (the value is committed) and parks the focus on the window.
    * When the focused widget is hidden or disabled (a tab or page switch, a section folding) Qt
      hands focus on as if Tab had been pressed, and a text field focused that way selects all its
      text. Only a real Tab key may do that; any other such hand-over is undone the same way.
    * A window opening with nothing focused gives its first field the focus (the Open window's
      search box lit up on arrival); that first, unasked focus is parked too."""

    TAB_WINDOW_S = 0.5

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._tab_at = -1e9
        self._press_pending = False

    @staticmethod
    def _field_of(widget: QtWidgets.QWidget | None) -> QtWidgets.QWidget | None:
        """The text field (line edit or spin box) ``widget`` is, or is part of."""
        while widget is not None and not widget.isWindow():
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                return widget
            if isinstance(widget, QtWidgets.QLineEdit):
                parent = widget.parentWidget()
                return parent if isinstance(parent, QtWidgets.QAbstractSpinBox) else widget
            widget = widget.parentWidget()
        return None

    def _after_press(self, target: QtWidgets.QWidget) -> None:
        self._press_pending = False
        try:
            field = self._field_of(QtWidgets.QApplication.focusWidget())
            if field is None or self._field_of(target) is field or field.window() is not target.window():
                return  # no field holds focus, the click was on it, or in another window
            self._park(field)
        except RuntimeError:  # deleted in the meantime
            pass

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and isinstance(obj, QtWidgets.QWidget) and not self._press_pending:
            # once per click (the press propagates up the parents); judged after Qt has moved focus
            self._press_pending = True
            QTimer.singleShot(0, lambda target=obj: self._after_press(target))
        elif kind == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            self._tab_at = time.monotonic()
        elif kind == QEvent.Type.FocusIn and isinstance(obj, QtWidgets.QWidget):
            window = obj.window()
            first = False
            if obj is not window:  # a dialog takes focus itself on showing, before handing it to a child
                first = not window.property("focusSeen")  # a window's first focus goes to whatever comes first
                window.setProperty("focusSeen", True)
            if isinstance(obj, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
                reason = event.reason()
                unasked_tab = (reason in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
                               and time.monotonic() - self._tab_at > self.TAB_WINDOW_S)
                if unasked_tab or (first and reason == Qt.FocusReason.ActiveWindowFocusReason):
                    QTimer.singleShot(0, lambda field=obj: self._park(field))
        return False

    @staticmethod
    def _park(field: QtWidgets.QWidget) -> None:
        try:
            if not field.hasFocus():
                return
            if isinstance(field, QtWidgets.QAbstractSpinBox):
                field.interpretText()  # commit what was typed, as leaving the field would
            edit = field.lineEdit() if isinstance(field, QtWidgets.QAbstractSpinBox) else field
            edit.deselect()
            field.window().setFocus(Qt.FocusReason.OtherFocusReason)
        except RuntimeError:  # deleted in the meantime
            pass


def setup_application(app: QtWidgets.QApplication) -> None:
    """Fusion (it honours the QSS radii fully), the house typeface, the saved day/night choice."""
    sys.excepthook = _report_exception
    app.setStyle(HintStyle("Fusion"))
    font = QtGui.QFont()
    font.setFamilies([UI_FONT, *UI_FONT_FALLBACKS])
    font.setPointSizeF(UI_FONT_SIZE * PT_SCALE)  # 12 px on Windows and on a Mac alike
    # True outlines: with grid-fitting on, Windows snaps letter heights to whole pixels at UI sizes.
    font.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    QtWidgets.QToolTip.setFont(font)  # Windows gives tooltips its own 9 pt system font otherwise
    install_app_filter(_PopupCorners(app))
    install_app_filter(_FocusGuard(app))
    install_app_filter(_ScrollBarReveal(app))
    THEME.dark = THEME.saved_dark()
    THEME.apply()


def repolish(widget: QtWidgets.QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


# --------------------------------------------------------------------------- surfaces
def window_fill() -> str:
    """The body colour: the grey the cards sit on, or the one flat surface of the divider style."""
    return PALETTE["surface"] if divider_panels() else PALETTE["app_bg"]


class Backdrop(QtWidgets.QWidget):
    """The window background every card sits on."""

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        QPainter(self).fillRect(self.rect(), QColor(window_fill()))


class Panel(QtWidgets.QFrame):
    """Elevated card: a white surface with a 1 px rim and a soft two-layer drop shadow, painted in
    the panel's own margin (``PAD``). On hover one 0 → 1 progress eases every parameter together —
    both shadows grow, drop and deepen while the rim warms toward the accent — so nothing pops.
    ``flat=True`` is a transparent region (no card): its children sit straight on the backdrop."""

    PAD, PAD_B = 6, 9
    #: (y-offset, blur, share of the palette's shadow alpha) at rest and on hover
    _CONTACT = ((0.5, 1.5, 0.55), (1.0, 2.5, 0.50))
    _AMBIENT = ((1.5, 4.5, 0.55), (3.0, 6.5, 0.85))

    def __init__(self, flat: bool = False, lift: bool = True, parent: QtWidgets.QWidget | None = None, kind: str = "panel") -> None:
        super().__init__(parent)
        # ``kind`` — "workspace" (image / sections), "side" (browser, rail), "section" (a settings
        # card) or "panel" (anything else) — lets the panel styles treat the regions differently.
        # In every style the margins stay the same, so the spacing does not change between them.
        self.kind = kind
        self.flat = flat
        self.lift = lift and not flat and PANEL_STYLE == "card"
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        if not flat:
            self.setContentsMargins(self.PAD, self.PAD, self.PAD, self.PAD_B)
        self._hover = HoverFade(self, 380, 520, QEasingCurve.Type.InOutSine, self._repaint_rim)

    def card_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(self.PAD, self.PAD, -self.PAD, -self.PAD_B)

    def _repaint_rim(self) -> None:
        # Only the margin (shadows) and the 1 px rim change; the card's contents do not redraw.
        card = self.card_rect().toRect()
        self.update(QtGui.QRegion(self.rect()).subtracted(QtGui.QRegion(card.adjusted(2, 2, -2, -2))))

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.lift:
            self._hover.to(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.lift:
            self._hover.to(0.0)
        super().leaveEvent(event)

    def look(self) -> tuple[str, str | None, float] | None:
        """(fill, frame, radius) of this panel in the current style, or None for no box."""
        style, kind = PANEL_STYLE, self.kind
        if self.flat or style == "divider":
            return None
        if style == "flat":
            return PALETTE["surface"], None, float(R_CARD)
        if style == "outline":
            return PALETTE["surface"], PALETTE["line"], 4.0
        if style in ("tonal", "inset"):
            return None if kind in ("side", "section") else (PALETTE["surface"], None, float(R_CARD))
        return PALETTE["card"], PALETTE["line"], float(R_CARD)  # card

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        look = self.look()
        if look is None:
            return
        card = self.card_rect()
        if card.width() < 8 or card.height() < 8:
            return
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        fill, frame, radius = look
        if PANEL_STYLE != "card":
            pixel_frame(painter, card, radius, frame, fill=fill)
            return
        t = self._hover.value
        peak = PALETTE["shadow_a"] + (PALETTE["shadow_a_hover"] - PALETTE["shadow_a"]) * t
        for rest, hover in (self._AMBIENT, self._CONTACT):
            dy, blur, share = (a + (b - a) * t for a, b in zip(rest, hover))
            paint_soft_shadow(painter, card.translated(0, dy), blur, QColor(*PALETTE["shadow_rgb"], round(peak * share)))
        rim = mix(PALETTE["line"], mix(PALETTE["line2"], PALETTE["accent"], 0.15), t)
        pixel_frame(painter, card, R_CARD, rim, fill=PALETTE["card"])


class Dialog(QtWidgets.QDialog):
    """Dialog on the window background with the main window's frameless chrome (the same
    :class:`TitleBar`, without minimise); add cards to ``self.body``."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)  # the 1 px window border (painted below)
        outer.setSpacing(0)
        owner = parent.window() if parent is not None else None
        self.titlebar = TitleBar(self, self.windowTitle(), getattr(getattr(owner, "titlebar", None), "logo", None), minimize=False)
        self.windowTitleChanged.connect(self.titlebar.set_title)
        outer.addWidget(self.titlebar)
        self.backdrop = Backdrop(self)
        outer.addWidget(self.backdrop, 1)
        self.body = QtWidgets.QVBoxLayout(self.backdrop)
        self.body.setContentsMargins(8, 6, 8, 6)
        self.body.setSpacing(0)
        self._resizer = _EdgeResizer(self)
        install_app_filter(self._resizer)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        apply_window_chrome(self)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            margin = 0 if self.isMaximized() else 1
            self.layout().setContentsMargins(margin, margin, margin, margin)
            self.titlebar.sync_state()
            apply_window_chrome(self)
        elif event.type() == QEvent.Type.ActivationChange:
            self.titlebar.sync_active()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(window_fill()))
        if not self.isMaximized():
            painter.setPen(QColor(PALETTE["win_border"]))
            painter.setBrush(NO_BRUSH)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class _SplitterHandle(QtWidgets.QSplitterHandle):
    """A gap between cards, or (divider style) the hairline between two regions; tinted on hover."""

    def __init__(self, orientation, parent) -> None:
        super().__init__(orientation, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def event(self, event) -> bool:
        if event.type() in (QEvent.Type.HoverEnter, QEvent.Type.HoverLeave):
            self.update()
        return super().event(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        rect = self.rect()
        if self.underMouse():
            painter.fillRect(rect, QColor(PALETTE["accentBg"]))
        if divider_panels():
            color = QColor(PALETTE["accent"] if self.underMouse() else PALETTE["line"])
            if self.orientation() == Qt.Orientation.Horizontal:
                painter.fillRect(QRect(rect.width() // 2, 0, 1, rect.height()), color)
            else:
                painter.fillRect(QRect(0, rect.height() // 2, rect.width(), 1), color)


class Splitter(QtWidgets.QSplitter):
    """Splitter whose handles are the narrow gaps between the cards (hairlines in the divider style)."""

    def __init__(self, orientation, parent: QtWidgets.QWidget | None = None, gap: int = 4) -> None:
        super().__init__(orientation, parent)
        self.setHandleWidth(max(gap, 5) if divider_panels() else gap)
        self.setChildrenCollapsible(False)
        self.setOpaqueResize(True)

    def createHandle(self) -> QtWidgets.QSplitterHandle:  # noqa: N802 - Qt API
        return _SplitterHandle(self.orientation(), self)


class Divider(QtWidgets.QWidget):
    """Hairline divider (horizontal by default) in the palette's line colour."""

    def __init__(self, vertical: bool = False, inset: int = 4, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.vertical, self.inset = vertical, inset
        if vertical:
            self.setFixedWidth(1 + 2 * inset)
        else:
            self.setFixedHeight(1 + 2 * inset)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        if self.vertical:
            painter.fillRect(QRect(self.inset, 2, 1, self.height() - 4), QColor(PALETTE["line"]))
        else:
            painter.fillRect(QRect(4, self.inset, self.width() - 8, 1), QColor(PALETTE["line"]))


class SearchField(QtWidgets.QLineEdit):
    """Search / filter field: a magnifier glyph in front (re-tinted with the theme), a short
    placeholder and Qt's clear button."""

    def __init__(self, placeholder: str = "Search", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._glyph = self.addAction(QtGui.QIcon(), QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        self._tint()
        THEME.changed.connect(self._tint)

    def _tint(self) -> None:
        self._glyph.setIcon(QtGui.QIcon(icon_pixmap("search", PALETTE["win_glyph"], 16)))


# --------------------------------------------------------------------------- flat tool buttons and the rail
class IconButton(QtWidgets.QToolButton):
    """Flat, square icon button: a hover fill that fades in, and a checked state painted as a
    quiet grey key; the glyph never changes colour (a tool strip draws its own sliding key)."""

    def __init__(self, name: str, tooltip: str = "", checkable: bool = False, size: int = 28, icon_size: int = 18, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.icon_name = name
        self.icon_px = icon_size
        self.rail_managed = False  # True when a ToolStrip paints the checked key
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setProperty("flat", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedSize(size, size)
        self._hover = HoverFade(self)
        self.toggled.connect(lambda _checked: self.update())

    def set_icon_name(self, name: str) -> None:
        self.icon_name = name
        self.update()

    def event(self, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.HoverEnter, QEvent.Type.Enter):
            self._hover.to(1.0)
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave):
            self._hover.to(0.0)
        elif kind == QEvent.Type.EnabledChange and not self.isEnabled():
            self._hover.reset()
        return super().event(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        rect = QRectF(self.rect())
        enabled = self.isEnabled()
        checked = self.isCheckable() and self.isChecked()
        # The glyph keeps one colour in every state; only the grey key behind it shows hover / checked
        # (a click does not flash a darker fill or recolour the icon).
        if checked and not self.rail_managed:
            pixel_frame(painter, rect, R_TOOL, None, fill=PALETTE["flat_checked"])
        elif self._hover.value > 0.01 and enabled and not checked:
            painter.setPen(NO_PEN)
            painter.setBrush(rgba(PALETTE["flat_hover"], self._hover.value))
            painter.drawRoundedRect(rect, R_TOOL, R_TOOL)
        color = PALETTE["win_glyph"] if enabled else PALETTE["btn_dis_fg"]
        _draw_icon(painter, self.icon_name, color, self.icon_px, rect.center())


def tool_strip_button(name: str, tooltip: str, checkable: bool = False) -> IconButton:
    """Square tool button for the tool strip, on the control row height."""
    return IconButton(name, tooltip, checkable, size=30, icon_size=18)


class ToolStrip(QtWidgets.QWidget):
    """Horizontal tool bar along the top of the image card: the buttons sit in its layout and a
    plain grey key marks the checked tool."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: list[IconButton] = []
        self._secondary: list[IconButton] = []
        self._key_x: float | None = None

    def register(self, item: IconButton, secondary: bool = False) -> None:
        if secondary:
            self._secondary.append(item)
            return
        self._buttons.append(item)
        if item.isCheckable():
            item.rail_managed = True
            item.toggled.connect(lambda checked, b=item: checked and self.select(b))

    def refresh(self, item: IconButton) -> None:
        """Repaint a secondary toggle after its state was changed with signals blocked."""
        item.update()

    def _checked(self) -> IconButton | None:
        return next((b for b in self._buttons if b.isCheckable() and b.isChecked()), None)

    def select(self, item: IconButton, animate: bool = False) -> None:
        self._key_x = float(item.geometry().left())
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        checked = self._checked()
        if checked is not None:
            self.select(checked, animate=False)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        checked = self._checked()
        if checked is not None:
            QTimer.singleShot(0, lambda b=checked: self.select(b, animate=False))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        checked = self._checked()
        if checked is None or self._key_x is None:
            return
        painter = QPainter(self)
        geometry = QRectF(checked.geometry())
        key = QRectF(self._key_x, geometry.top(), geometry.width(), geometry.height())
        pixel_frame(painter, key, R_TOOL, None, fill=PALETTE["flat_checked"])


# --------------------------------------------------------------------------- buttons
class Button(QtWidgets.QPushButton):
    """House push button. Variants: default, primary, danger, ok, warn, info, ghost.

    The base fill comes from the variant; on top a hover light — the accent (red for danger) at
    low opacity — fades in and out rather than snapping, slightly deeper while pressed, and the
    1 px pixel-snapped frame warms toward the same tone. A checkable button shows its checked
    state as the "active" key: a pressed-in grey fill with the label in the heading colour."""

    def __init__(self, text: str = "", variant: str | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.variant = variant or "default"
        self.icon_name: str | None = None
        self.setProperty("house", True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # A mouse-driven button has no use for keyboard focus (and disabling a focused button
        # would bounce the focus into a field and select-all it).
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lit = HoverFade(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)  # may shrink (label elides)
        self.toggled.connect(lambda _checked: self.update())

    def set_variant(self, variant: str | None) -> None:
        self.variant = variant or "default"
        self.update()

    def set_icon_name(self, name: str | None) -> None:
        self.icon_name = name
        self.updateGeometry()
        self.update()

    def event(self, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.HoverEnter, QEvent.Type.Enter):
            self._lit.to(1.0)
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave):
            self._lit.to(0.0)
        elif kind == QEvent.Type.EnabledChange and not self.isEnabled():
            self._lit.reset()
        return super().event(event)

    def _kind(self) -> str:
        if self.isCheckable() and self.isChecked():
            return "active"
        return self.variant

    def _label_font(self) -> QtGui.QFont:
        font = QtGui.QFont(self.font())
        font.setWeight(QtGui.QFont.Weight.DemiBold if self._kind() in ("primary", "danger", "active") else QtGui.QFont.Weight.Normal)
        return font

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        metrics = QtGui.QFontMetrics(self._label_font())
        width = metrics.horizontalAdvance(self.text()) + 26
        if self.icon_name:
            width += 16 + (7 if self.text() else 0)
        return QSize(max(width, 34), H_CONTROL)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = self.sizeHint()
        return QSize(min(hint.width(), 90), hint.height())  # the label elides when a row is squeezed

    def _colours(self) -> tuple[str, str, str, str | None]:
        """(fill, frame, label, hover tone)"""
        p, kind, lit, down = PALETTE, self._kind(), self._lit.value, self.isDown()
        if not self.isEnabled():
            return p["btn_dis_bg"], p["btn_dis_border"], p["btn_dis_fg"], None
        if kind == "primary":
            fill = p["accent_pressed"] if down else mix(p["accent_fill"], p["accent_hover"], lit)
            return fill, fill, p["onAccent"], None
        if kind == "active":  # a toggle that is on: pressed-in grey key, the label in the heading colour
            return p["flat_checked"], control_frame(lit, down), p["heading"], None
        if kind == "ghost":
            return p["surface"], mix(p["surface"], p["line2"], lit), p["accent"], p["accent"]
        if kind in ("ok", "warn", "info"):
            key = {"ok": "success", "warn": "warning", "info": "info"}[kind]
            return p[key + "Bg"], p[key + "Border"], p[key], p[key]
        if kind == "danger":
            frame = mix(p["line2"], p["danger"], 0.75 if down else 0.5 * lit)
            return p["surface"], frame, p["danger"], p["danger"]
        return p["surface"], control_frame(lit, down), p["text"], p["accent"]  # the same frame as the fields

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        painter.setRenderHint(TEXT_ANTIALIAS)
        rect = QRectF(self.rect())
        fill, frame, ink, tone = self._colours()
        pixel_frame(painter, rect, R_BTN, None, fill=fill)
        if tone is not None:
            k = 1.35 if THEME.dark else 1.0
            light = QColor(tone)
            light.setAlphaF(min(1.0, (0.15 if self.isDown() else 0.06 * self._lit.value) * k))
            if light.alpha():
                painter.setPen(NO_PEN)
                painter.setBrush(light)
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), R_BTN, R_BTN)
        pixel_frame(painter, rect, R_BTN, frame)
        painter.setPen(QColor(ink))
        painter.setFont(self._label_font())
        metrics = painter.fontMetrics()
        icon_w = 16 if self.icon_name else 0
        spacing = 7 if self.icon_name and self.text() else 0
        available = rect.width() - 20 - icon_w - spacing
        text = metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, int(max(10, available))) if self.text() else ""
        total = icon_w + spacing + metrics.horizontalAdvance(text)
        x = rect.center().x() - total / 2
        if self.icon_name:
            _draw_icon(painter, self.icon_name, ink, 16, QPointF(round(x) + 8, rect.center().y()), crisp=True)
            x += icon_w + spacing
        painter.drawText(QRectF(x, rect.top(), max(1.0, rect.right() - x), rect.height()), int(ALIGN_LEFT_VCENTER), text)


# --------------------------------------------------------------------------- tabs, segmented control
class TabBar(QtWidgets.QTabBar):
    """Underline tabs: muted labels, the current one in the navy accent over a 2 px accent
    underline, on a hairline across the bar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._line: tuple[float, float] | None = None  # (left, width)
        self._hover_index = -1
        self.tab_height = 34
        self._glide = _animation(self, TAB_SWITCH_MS)
        self._glide.valueChanged.connect(self._glide_step)
        self._glide_from: tuple[float, float] | None = None
        self.currentChanged.connect(self._slide)

    def set_tab_height(self, height: int) -> None:
        """Make the bar ``height`` tall (e.g. so its hairline lines up with a neighbour's)."""
        if height != self.tab_height:
            self.tab_height = height
            if self.count():
                self.setTabText(0, self.tabText(0))  # drops Qt's cached tab sizes and relays the bar out

    def _target(self, index: int) -> tuple[float, float]:
        rect = self.tabRect(index)
        return float(rect.left() + 10), float(rect.width() - 20)

    def _slide(self, index: int) -> None:
        if index < 0:
            return
        if self._line is None or not self.isVisible():
            self._line = self._target(index)
            self.update()
            return
        self._glide_from = self._line  # the underline glides from where it is to the new tab
        _retarget(self._glide, 0.0, 1.0)

    def _glide_step(self, value) -> None:
        if self._glide_from is None or self.currentIndex() < 0:
            return
        t = float(value)
        (l0, w0), (l1, w1) = self._glide_from, self._target(self.currentIndex())
        self._line = (l0 + (l1 - l0) * t, w0 + (w1 - w0) * t)
        self.update()

    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802 - Qt API
        font = bold_font(self.font())  # sized for the bold current label, so tabs never shift
        return QSize(QtGui.QFontMetrics(font).horizontalAdvance(self.tabText(index)) + 26, self.tab_height)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self.currentIndex() >= 0:
            self._glide.stop()
            self._line = self._target(self.currentIndex())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        index = self.tabAt(event.position().toPoint())
        if index != self._hover_index:
            self._hover_index = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._hover_index = -1
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        painter.setRenderHint(TEXT_ANTIALIAS)
        painter.fillRect(QRect(0, self.height() - 1, self.width(), 1), QColor(PALETTE["line"]))
        current = self.currentIndex()
        if self._line is None and current >= 0:
            self._line = self._target(current)
        for index in range(self.count()):
            rect = QRectF(self.tabRect(index))
            on = index == current
            painter.setFont(bold_font(self.font()) if on else self.font())
            color = PALETTE["accent"] if on else PALETTE["text"] if index == self._hover_index else PALETTE["text2"]
            painter.setPen(QColor(color))
            painter.drawText(rect.adjusted(0, 0, 0, -2), int(ALIGN_CENTER), self.tabText(index))
        if current >= 0 and self._line is not None:
            left, width = self._line
            painter.fillRect(QRectF(left, self.height() - 2, width, 2), QColor(PALETTE["accent"]))


class _PageShot(QtWidgets.QWidget):
    """A picture of the page being left, over the new one: it fades and drifts out while the new
    page drifts in underneath (mouse events pass through to the new page at once)."""

    def __init__(self, parent: QtWidgets.QWidget, shot: QtGui.QPixmap, rect: QRect) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._shot, self.opacity, self.offset = shot, 1.0, 0.0
        self.setGeometry(rect)
        self.raise_()
        self.show()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setOpacity(self.opacity)
        painter.drawPixmap(QtCore.QPointF(self.offset, 0.0), self._shot)


#: Tab switch: the underline glide and the page cross-fade (ms), and how far the pages drift (px).
TAB_SWITCH_MS, TAB_DRIFT = 200, 14
SEGMENT_GLIDE_MS = 240  # a choice bar's key moving to the new option


class TabWidget(QtWidgets.QTabWidget):
    """The tab bar's hairline runs the full width, under a corner widget too (the style's own
    document-mode base line is not drawn). Switching tabs cross-fades the pages with a short
    sideways drift in the direction of the new tab."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTabBar(TabBar())
        self.setDocumentMode(True)
        self._page = None  # the page shown before the switch
        self._shot: _PageShot | None = None
        self._slot = QtCore.QPoint()  # the page's place in the tab widget's page stack
        self._direction = 1.0
        self._switch = _animation(self, TAB_SWITCH_MS)
        self._switch.valueChanged.connect(self._switch_step)
        self._switch.finished.connect(self._switch_done)
        self.currentChanged.connect(self._switch_page)

    def _switch_page(self, index: int) -> None:
        old, new = self._page, self.widget(index)
        self._page = new
        if self._shot is not None and old is not None:
            old.move(self._slot)  # a switch cut short: the page that was drifting in goes back first
        self._switch_done()
        if old is None or new is None or old is new or not self.isVisible():
            return
        old_index = self.indexOf(old)
        self._direction = 1.0 if old_index < 0 or index > old_index else -1.0
        self._slot = new.pos()
        self._shot = _PageShot(self, old.grab(), QRect(new.mapTo(self, QtCore.QPoint(0, 0)), new.size()))
        self._switch_step(0.0)
        _retarget(self._switch, 0.0, 1.0)

    def _switch_step(self, value) -> None:
        t, page = float(value), self.currentWidget()
        if self._shot is None:
            return
        self._shot.opacity, self._shot.offset = 1.0 - t, -self._direction * TAB_DRIFT * t
        self._shot.update()
        if page is not None:
            page.move(self._slot.x() + round(self._direction * TAB_DRIFT * (1.0 - t)), self._slot.y())

    def _switch_done(self) -> None:
        if self._shot is not None:
            page = self.currentWidget()
            if page is not None:
                page.move(self._slot)  # back in its place
            self._shot.deleteLater()
            self._shot = None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(QRect(0, self.tabBar().geometry().bottom(), self.width(), 1), QColor(PALETTE["line"]))


class SegmentedControl(QtWidgets.QWidget):
    """Square choice bar (the Dry DR segmented control, without its shadow): a flat grey track, the
    chosen option a white key inset 3 px with a hairline frame and its label in the accent, bold.
    Unchosen options are plain labels on the track; under the pointer they get a slightly deeper grey
    key. A new choice glides the key across in SEGMENT_GLIDE_MS, the labels' ink following it; a
    click is reported (``currentIndexChanged`` …, ``activated``) once the key has arrived.

    It also speaks the part of QComboBox the settings code uses (``addItems``, ``add_values``,
    ``currentText`` / ``setCurrentText``, ``currentIndex`` / ``setCurrentIndex`` and the
    ``currentIndexChanged`` / ``currentTextChanged`` signals), so a short combo can be swapped for
    one without touching the code that reads it. ``index`` may be -1 (no option highlighted);
    ``activated`` fires on user clicks only."""

    activated = Signal(int)
    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)

    INSET = 3   # track to key
    GAP = 2     # between neighbouring keys

    def __init__(self, labels: list[str] | None = None, parent: QtWidgets.QWidget | None = None, height: int = H_CONTROL) -> None:
        super().__init__(parent)
        self.labels: list[str] = list(labels or [])
        self.values: list[str] = list(self.labels)
        self._index = -1
        self._hover = -1
        self._key = -1.0  # where the key is, in cells (fractional while it glides)
        self._glide = _animation(self, SEGMENT_GLIDE_MS)
        self._glide.valueChanged.connect(self._glide_step)
        self._glide.finished.connect(self._glide_done)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    # ---- the combo-box part
    def addItems(self, labels) -> None:  # noqa: N802 - QComboBox API
        self.labels += [str(text) for text in labels]
        self.values += [str(text) for text in labels]
        if self._index < 0 and self.labels:
            self.set_index(0)
        self.updateGeometry(); self.update()

    def add_values(self, values) -> None:
        """Options shown with a capital first letter while ``currentText`` speaks the values (as ComboBox.add_values)."""
        self.labels += [value[:1].upper() + value[1:] for value in values]
        self.values += [str(value) for value in values]
        if self._index < 0 and self.labels:
            self.set_index(0)
        self.updateGeometry(); self.update()

    def count(self) -> int:
        return len(self.labels)

    def itemText(self, index: int) -> str:  # noqa: N802 - QComboBox API
        return self.labels[index] if 0 <= index < len(self.labels) else ""

    def currentIndex(self) -> int:  # noqa: N802 - QComboBox API
        return self._index

    def currentText(self) -> str:  # noqa: N802 - QComboBox API
        return self.values[self._index] if 0 <= self._index < len(self.values) else ""

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - QComboBox API
        self.set_index(index)

    def setCurrentText(self, text: str) -> None:  # noqa: N802 - QComboBox API
        for options in (self.values, self.labels):
            if text in options:
                self.set_index(options.index(text))
                return

    # ---- the bar
    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(len(self.labels) * self._cell_width() + 2 * self.INSET, self.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """As narrow as a combo may go, so a bar never prises a settings grid's columns apart."""
        return QSize(min(self.sizeHint().width(), ComboBox.MIN_WIDTH), self.height())

    def _cell_width(self) -> int:
        metrics = QtGui.QFontMetrics(bold_font(self.font()))
        return max((metrics.horizontalAdvance(text) for text in self.labels), default=0) + 14

    def index(self) -> int:
        return self._index

    def set_index(self, index: int, animate: bool = False) -> None:
        """Select ``index`` (-1: none). On screen the key glides from the old choice to the new one
        (``animate`` is accepted for compatibility; a hidden bar or one with no choice jumps)."""
        index = int(index) if index is not None else -1
        if index >= len(self.labels):
            index = -1
        if index == self._index:
            return
        old = self._index
        self._index = index
        self._pending = None  # a newer choice speaks for itself
        if old >= 0 and index >= 0 and self.isVisible():
            _retarget(self._glide, self._key if self._key >= 0 else old, index)
        else:
            self._glide.stop(); self._key = float(index)
        self.update()
        self.currentIndexChanged.emit(index)
        self.currentTextChanged.emit(self.currentText())

    def _click(self, index: int) -> None:
        """A choice by the user: the key glides there first, and only then does the bar say so. A
        choice sets off a redraw of the image that holds the event loop for a good part of a second's
        tenth and then shares it; said at once, the glide would be left two or three frames."""
        if index == self._index or self._index < 0 or not self.isVisible():
            self.set_index(index)
            self.activated.emit(index)
            return
        old = self._index
        self._index = index
        _retarget(self._glide, self._key if self._key >= 0 else old, index)
        self._pending = index
        self.update()

    _pending = None

    def _glide_done(self) -> None:
        index, self._pending = self._pending, None
        if index is not None and index == self._index:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())
            self.activated.emit(index)

    def _glide_step(self, value) -> None:
        self._key = float(value)
        self.update()

    def _cells(self) -> list[QRectF]:
        count = max(len(self.labels), 1)
        inner = QRectF(self.rect()).adjusted(self.INSET, self.INSET, -self.INSET, -self.INSET)
        width = (inner.width() - (count - 1) * self.GAP) / count
        return [QRectF(inner.left() + i * (width + self.GAP), inner.top(), width, inner.height()) for i in range(count)]

    def _segment_at(self, x: float) -> int:
        if not self.labels:
            return -1
        width = self.width() / len(self.labels)
        return max(0, min(len(self.labels) - 1, int(x // max(width, 1))))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        hovered = self._segment_at(event.position().x())
        if hovered != self._hover:
            self._hover = hovered
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._click(self._segment_at(event.position().x()))
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        painter.setRenderHint(TEXT_ANTIALIAS)
        p = PALETTE
        night = QColor(p["surface"]).lightness() < 128
        enabled = self.isEnabled()
        # by day a grey track under a white key; by night a track darker than the panel under a lifted key
        track = (p["app_bg"] if night else p["pressed"]) if enabled else p["btn_dis_bg"]
        key_fill = p["pressed"] if night else p["surface"]
        hover_fill = mix(track, p["line2"], 0.45)
        pixel_frame(painter, QRectF(self.rect()), R_BTN, None, fill=track)
        cells = self._cells()
        key = self._key if self._index >= 0 else -1.0
        # how much of the key covers each cell (1 under a key at rest, shared while it glides)
        cover = [max(0.0, 1.0 - abs(key - i)) if key >= 0 else 0.0 for i in range(len(cells))]
        for i, cell in enumerate(cells):
            if i == self._hover and enabled and i != self._index and cover[i] < 0.01:
                pixel_frame(painter, cell, R_BTN - 1, None, fill=hover_fill)
        if key >= 0 and cells:
            lo = min(int(key), len(cells) - 1); hi = min(lo + 1, len(cells) - 1); t = key - lo
            a, b = cells[lo], cells[hi]
            rect = QRectF(a.left() + (b.left() - a.left()) * t, a.top(), a.width() + (b.width() - a.width()) * t, a.height())
            pixel_frame(painter, rect, R_BTN - 1, p["line2"] if enabled else p["btn_dis_border"], fill=key_fill)
        plain = QtGui.QFont(self.font()); bold = bold_font(self.font())
        for i, (cell, text) in enumerate(zip(cells, self.labels)):
            on = cover[i]
            painter.setFont(bold if on > 0.5 else plain)
            painter.setPen(QColor(p["btn_dis_fg"] if not enabled else mix(p["text"], p["accent"], on)))
            painter.drawText(cell, int(ALIGN_CENTER), painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(cell.width())))


# --------------------------------------------------------------------------- combo box
class PopupItemDelegate(QtWidgets.QStyledItemDelegate):
    """Rows of a combo popup in the language of every other selection in the interface: 26 px
    (the control row), a light grey key under the pointer, the current entry on the deeper grey key
    with its label in the heading colour — no accent, no tick. The text starts at exactly the x of
    the combo's own text (combo: 1 px frame + 8 px padding; popup: 1 px frame + 3 px view padding +
    INSET)."""

    ROW = H_CONTROL
    INSET = 5   # 1 + 8 = 1 + 3 + 5
    TICK = 12   # right padding

    def __init__(self, combo: QtWidgets.QComboBox) -> None:
        super().__init__(combo)
        self.combo = combo

    def sizeHint(self, option, index) -> QSize:  # noqa: N802 - Qt API
        metrics = QtGui.QFontMetrics(scaled_font(option.font, 0, bold=True))
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        extra = (self.combo.iconSize().width() + 8) if icon is not None and not QtGui.QIcon(icon).isNull() else 0
        return QSize(metrics.horizontalAdvance(text) + self.TICK + extra + 22, self.ROW)

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setRenderHint(ANTIALIAS)
        painter.setRenderHint(TEXT_ANTIALIAS)
        state = option.state
        enabled = bool(index.flags() & Qt.ItemFlag.ItemIsEnabled)
        rect = QRectF(option.rect).adjusted(0, 1, 0, -1)
        current = index.row() == self.combo.currentIndex()
        under_pointer = bool(state & (QtWidgets.QStyle.StateFlag.State_Selected | QtWidgets.QStyle.StateFlag.State_MouseOver))
        if enabled and (current or under_pointer):
            painter.setPen(NO_PEN)
            painter.setBrush(QColor(PALETTE["flat_checked"] if current else PALETTE["hover"]))
            painter.drawRoundedRect(rect, R_ROW, R_ROW)
        color = PALETTE["faint"] if not enabled else PALETTE["heading"] if current else PALETTE["text"]
        x = rect.left() + self.INSET
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and not QtGui.QIcon(icon).isNull():
            size = self.combo.iconSize()
            pixmap = QtGui.QIcon(icon).pixmap(size)
            painter.drawPixmap(QPointF(x, round(rect.center().y() - size.height() / 2)), pixmap)
            x += size.width() + 4  # the gap the combo leaves between its icon and its text
        row_font = QtGui.QFont(option.font)
        row_font.setWeight(QtGui.QFont.Weight.DemiBold if current else QtGui.QFont.Weight.Normal)
        painter.setFont(row_font)
        painter.setPen(QColor(color))
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        width = rect.right() - x - self.TICK
        painter.drawText(QRectF(x, rect.top(), width, rect.height()), int(ALIGN_LEFT_VCENTER),
                         painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, int(width)))
        painter.restore()


class ComboBox(QtWidgets.QComboBox):
    """Combo that never forces its panel wider (the current item is clipped when squeezed)
    while its popup always shows the full names, in house-style rows."""

    MIN_WIDTH = 72  # as narrow as a settings grid's number field, so two combos share a 400 px panel row

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(4)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setItemDelegate(PopupItemDelegate(self))
        self.setMaxVisibleItems(12)
        self.view().setMouseTracking(True)
        # The popup draws no native drop shadow (set before its window exists); DWM rounds its corners
        # and draws the hairline border (see _PopupCorners).
        self.view().window().setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        """The wheel never changes a combo's value (it scrolls the panel underneath instead)."""
        event.ignore()

    VALUE_ROLE = Qt.ItemDataRole.UserRole + 77

    def add_values(self, values: list[str], icons: list[QtGui.QIcon] | None = None) -> None:
        """Items shown with a capital first letter while the code keeps working with the values
        themselves (``nearest``, ``white`` …): ``currentText`` and ``setCurrentText`` speak values.
        ``currentTextChanged`` still carries the shown label."""
        for position, value in enumerate(values):
            text = value[:1].upper() + value[1:]
            if icons is not None:
                self.addItem(icons[position], text)
            else:
                self.addItem(text)
            self.setItemData(self.count() - 1, value, self.VALUE_ROLE)

    def currentText(self) -> str:  # noqa: N802 - Qt API
        value = self.currentData(self.VALUE_ROLE)
        return value if isinstance(value, str) else super().currentText()

    def setCurrentText(self, text: str) -> None:  # noqa: N802 - Qt API
        index = self.findData(text, self.VALUE_ROLE)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), self.MIN_WIDTH), hint.height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().sizeHint()
        return QSize(max(self.MIN_WIDTH, min(hint.width(), 260)), hint.height())

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        metrics = self.fontMetrics()
        widest = 0
        for index in range(self.count()):
            width = metrics.horizontalAdvance(self.itemText(index))
            if not self.itemIcon(index).isNull():
                width += self.iconSize().width() + 10
            widest = max(widest, width)
        self.view().setMinimumWidth(max(self.width(), widest + PopupItemDelegate.TICK + 52, self.view().minimumWidth()))
        super().showPopup()


# --------------------------------------------------------------------------- field frames
def control_frame(hover: float, down: bool = False, focus: bool = False, enabled: bool = True) -> str:
    """The one frame colour of every bordered control (buttons, fields, segmented toggles):
    hairline at rest, warming toward the accent under the pointer, the accent with focus."""
    if not enabled:
        return PALETTE["btn_dis_border"]
    if focus:
        return PALETTE["accent"]
    return mix(PALETTE["line2"], PALETTE["accent"], 0.75 if down else 0.5 * hover)


class FieldFrame(QtWidgets.QWidget):
    """Paints a field's 1 px frame on top of it (a mouse-transparent overlay), identical to the
    house buttons' frame; the hover warmth fades like theirs."""

    def __init__(self, field: QtWidgets.QWidget) -> None:
        super().__init__(field)
        self.field = field
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        field.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hover = HoverFade(self)
        watched = [field, *field.findChildren(QtWidgets.QLineEdit)]  # a spin box / editable combo's inner editor
        for widget in watched:
            watch(widget, self)
        self.setGeometry(field.rect())
        self.show()
        self.raise_()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        kind = event.type()
        if obj is self.field and kind in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self.field.rect())
            self.raise_()
        elif kind in (QEvent.Type.HoverEnter, QEvent.Type.Enter) and obj is self.field:
            self._hover.to(1.0)
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave) and obj is self.field:
            self._hover.to(0.0)
        elif kind in (QEvent.Type.FocusIn, QEvent.Type.FocusOut, QEvent.Type.EnabledChange, QEvent.Type.ReadOnlyChange):
            self.update()
        return False

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        focus_widget = QtWidgets.QApplication.focusWidget()
        focus = focus_widget is not None and (focus_widget is self.field or self.field.isAncestorOf(focus_widget))
        editable = not (isinstance(self.field, QtWidgets.QLineEdit) and self.field.isReadOnly())
        color = control_frame(self._hover.value, focus=focus and editable, enabled=self.field.isEnabled())
        pixel_frame(QPainter(self), QRectF(self.rect()), R_FIELD, color)


class _WheelGuard(QtCore.QObject):
    """Spin boxes take the wheel only while they have the keyboard focus (clicked into): otherwise
    the wheel scrolls the panel under them, so scrolling past a field never changes its value."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            scroll_panel(obj, event)
            return True
        return False


def scroll_panel(widget: QtWidgets.QWidget, event) -> None:
    """Hand a wheel event to the scroll area that holds ``widget`` (the panel scrolls instead)."""
    area = widget.parentWidget()
    while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
        area = area.parentWidget()
    if area is not None:
        QtWidgets.QApplication.sendEvent(area.viewport(), event)


_wheel_guard: _WheelGuard | None = None


class _SpinEditFix(QtCore.QObject):
    """Qt 6.12 lays out a style-sheeted spin box without buttons with a 1 px wide editor, so the
    number never shows (Qt 6.11 gave it the padded width). After every resize / show / restyle the
    editor is put back where 6.11 put it: inside the 1 px border and the 8 px side padding."""

    PAD_X, PAD_Y = 9, 4

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.StyleChange, QEvent.Type.LayoutRequest):
            QTimer.singleShot(0, lambda spin=obj: self.place(spin))  # after Qt's own layout of the editor
        return False

    @classmethod
    def place(cls, spin: QtWidgets.QAbstractSpinBox) -> None:
        try:
            target = spin.rect().adjusted(cls.PAD_X, cls.PAD_Y, -cls.PAD_X, -cls.PAD_Y)
            edit = spin.lineEdit()
            if target.width() > 0 and edit.geometry() != target:
                edit.setGeometry(target)
        except RuntimeError:  # deleted in the meantime
            pass


_spin_edit_fix: _SpinEditFix | None = None


def frame_fields(root: QtWidgets.QWidget) -> None:
    """Give every field under ``root`` (line edits, spin boxes, combo boxes) its painted frame."""
    global _wheel_guard, _spin_edit_fix
    for field in root.findChildren(QtWidgets.QWidget):
        if getattr(field, "_field_frame", None) is not None:
            continue
        if isinstance(field, QtWidgets.QAbstractSpinBox):
            if _wheel_guard is None:
                _wheel_guard = _WheelGuard(QtWidgets.QApplication.instance())
                _spin_edit_fix = _SpinEditFix(QtWidgets.QApplication.instance())
            field.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            watch(field, _wheel_guard)
            watch(field, _spin_edit_fix)
            _SpinEditFix.place(field)
        elif isinstance(field, (QtWidgets.QComboBox, QtWidgets.QPlainTextEdit)):
            pass
        elif isinstance(field, QtWidgets.QLineEdit):
            if isinstance(field.parent(), (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                continue  # the inner editor: its owner carries the frame
        else:
            continue
        if not isinstance(field, QtWidgets.QPlainTextEdit):
            # one row height for fields and buttons, whatever the typeface's line height (Avenir's is taller)
            field.setFixedHeight(H_CONTROL)
        field._field_frame = FieldFrame(field)


# --------------------------------------------------------------------------- slider
KNOB_W, KNOB_H = 16, 10


def paint_knob(painter: QPainter, centre: QPointF, hover: float, pressed: bool = False, enabled: bool = True) -> None:
    """The house knob (sliders, histogram handles): a small capsule of the surface colour with a
    1 px hairline rim. No shadow, no gloss, no grip; the rim darkens to the heading ink under the
    pointer and while held."""
    painter.save()
    painter.setRenderHint(ANTIALIAS)
    # whole pixels + 0.5, so the 1 px rim lands on one device row instead of smearing over two
    rect = QRectF(round(centre.x() - KNOB_W / 2) + 0.5, round(centre.y() - KNOB_H / 2) + 0.5, KNOB_W - 1, KNOB_H - 1)
    if not enabled:
        rim = PALETTE["btn_dis_border"]
    else:
        rim = PALETTE["heading"] if pressed else mix(PALETTE["line2"], PALETTE["heading"], hover)
    painter.setPen(QPen(QColor(rim), 1))
    painter.setBrush(QColor(PALETTE["surface"] if enabled else PALETTE["btn_dis_bg"]))
    painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
    painter.restore()


class Slider(QtWidgets.QSlider):
    """Horizontal slider: a 2 px track, an accent fill up to the value and the house capsule knob.
    Click anywhere on the track to jump there; a double-click returns to the default value. The
    wheel scrolls the panel instead (scrolling past a slider never moves it)."""

    def __init__(self, minimum: int = 0, maximum: int = 100, value: int = 0, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(minimum, maximum)
        self.setValue(value)
        self.default = value
        self.setMinimumHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._dragging = False
        self._hover = HoverFade(self, 150, 150)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(160, 22)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(60, 22)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._hover.to(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._dragging:
            self._hover.to(0.0)
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        scroll_panel(self, event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.setValue(self.default)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _track(self) -> QRectF:
        margin = KNOB_W / 2 + 1
        return QRectF(margin, round(self.height() / 2) - 1, max(self.width() - 2 * margin, 1.0), 2)

    def _value_at(self, x: float) -> int:
        track = self._track()
        ratio = min(max((x - track.left()) / track.width(), 0.0), 1.0)
        return int(round(self.minimum() + ratio * (self.maximum() - self.minimum())))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._dragging = True
            self.setSliderDown(True)
            self.setValue(self._value_at(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging:
            self.setValue(self._value_at(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging:
            self._dragging = False
            self.setSliderDown(False)
            if not self.underMouse():
                self._hover.to(0.0)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        track = self._track()
        span = max(self.maximum() - self.minimum(), 1)
        ratio = (self.value() - self.minimum()) / span
        enabled = self.isEnabled()
        painter.setRenderHint(ANTIALIAS, False)  # a hairline track on whole pixels
        painter.fillRect(track, QColor(PALETTE["line"]))
        filled = QRectF(track.left(), track.top(), round(track.width() * ratio), track.height())
        if filled.width() >= 1:
            painter.fillRect(filled, QColor(PALETTE["accent"] if enabled else PALETTE["btn_dis_border"]))
        centre = QPointF(track.left() + track.width() * ratio, track.center().y())
        paint_knob(painter, centre, self._hover.value, pressed=self._dragging, enabled=enabled)


# --------------------------------------------------------------------------- check box
class CheckBox(QtWidgets.QCheckBox):
    """Painted check box: a 16 px key that fills with the accent while the tick grows in;
    long labels are elided."""

    INDICATOR = 16
    SPACING = 8

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._check = 1.0 if self.isChecked() else 0.0
        self._check_anim = _animation(self, 170, QEasingCurve.Type.OutCubic)
        self._check_anim.valueChanged.connect(self._set_check)
        self._hover = HoverFade(self, 120, 160)
        self.toggled.connect(self._animate)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)

    def _animate(self, checked: bool) -> None:
        if self.isVisible():
            _retarget(self._check_anim, self._check, 1.0 if checked else 0.0)
        else:
            self._check = 1.0 if checked else 0.0
            self.update()

    def _set_check(self, value) -> None:
        self._check = float(value)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._hover.to(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._hover.to(0.0)
        super().leaveEvent(event)

    def hitButton(self, pos) -> bool:  # noqa: N802 - Qt API
        return self.rect().contains(pos)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        metrics = QtGui.QFontMetrics(self.font())
        return QSize(self.INDICATOR + self.SPACING + metrics.horizontalAdvance(self.text()) + 4, max(24, metrics.height() + 6))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self.INDICATOR + self.SPACING + 40, self.sizeHint().height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        painter.setRenderHint(TEXT_ANTIALIAS)
        enabled = self.isEnabled()
        check = min(1.0, max(0.0, self._check))
        size = self.INDICATOR
        box = QRectF(0, round((self.height() - size) / 2), size, size)
        if enabled:
            fill = mix(PALETTE["surface"], PALETTE["accent_fill"], check)
            frame = mix(mix(PALETTE["line2"], PALETTE["accent"], self._hover.value), PALETTE["accent_fill"], check)
        else:
            fill, frame = PALETTE["btn_dis_bg"], PALETTE["btn_dis_border"]
        pixel_frame(painter, box, R_FIELD, frame, fill=fill)
        if check > 0.02:
            pen = QPen(QColor("#ffffff" if enabled else PALETTE["btn_dis_fg"]), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(NO_BRUSH)
            painter.save()
            painter.translate(box.center())
            painter.scale(check, check)
            painter.translate(-box.center())
            tick = QtGui.QPainterPath()
            tick.moveTo(box.left() + size * 0.26, box.top() + size * 0.53)
            tick.lineTo(box.left() + size * 0.44, box.top() + size * 0.71)
            tick.lineTo(box.left() + size * 0.76, box.top() + size * 0.33)
            painter.drawPath(tick)
            painter.restore()
        painter.setPen(QColor(PALETTE["text"] if enabled else PALETTE["faint"]))
        painter.setFont(self.font())
        text_rect = QRectF(size + self.SPACING, 0, max(1, self.width() - size - self.SPACING), self.height())
        painter.drawText(text_rect, int(ALIGN_LEFT_VCENTER), painter.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, int(text_rect.width())))


# --------------------------------------------------------------------------- progress
class BusyRing(QtWidgets.QWidget):
    """Small spinning arc for background work. Idle it paints nothing but keeps its place, so the
    row it sits in never shifts when work starts or stops."""

    def __init__(self, size: int = 16, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._busy = False
        self._angle = 0.0
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(900)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(self._set_angle)

    def _set_angle(self, value) -> None:
        self._angle = float(value)
        self.update()

    def set_busy(self, on: bool) -> None:
        self._busy = bool(on)
        if on and self._spin.state() != QtCore.QAbstractAnimation.State.Running:
            self._spin.start()
        elif not on:
            self._spin.stop()
        self.update()

    def is_busy(self) -> bool:
        return self._busy

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self._busy:
            return
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        stroke = 2.0
        bounds = QRectF(self.rect()).adjusted(stroke / 2 + 1, stroke / 2 + 1, -stroke / 2 - 1, -stroke / 2 - 1)
        pen = QPen(QColor(PALETTE["line"]), stroke)
        painter.setPen(pen)
        painter.drawEllipse(bounds)
        pen.setColor(QColor(PALETTE["accent"]))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(bounds, int(-self._angle * 16), 100 * 16)  # Qt angles: 1/16 degree, counter-clockwise


# --------------------------------------------------------------------------- window chrome
class WindowButton(QtWidgets.QPushButton):
    """Painted window control: thin line glyphs on a key as tall as the title bar, lit edge to edge
    on hover (muted brick red for close), like the Windows 11 caption buttons.
    ``kind`` ∈ min / max / restore / close."""

    WIDTH = 46

    def __init__(self, kind: str, parent: QtWidgets.QWidget | None = None, height: int = 42) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setProperty("house", True)
        self.setProperty("chrome", "key")  # title-bar keys keep their size in dialogs too (QSS)
        self.setFixedSize(self.WIDTH, height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        close = self.kind == "close"
        hover = self.underMouse()
        if hover:
            painter.setPen(NO_PEN)
            painter.setBrush(QColor("#c0392b") if close else QColor(*PALETTE["win_hover"]))
            painter.drawRect(self.rect())
        pen = QPen(QColor("#ffffff") if close and hover else QColor(PALETTE["win_glyph"]))
        pen.setWidthF(1.3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(NO_BRUSH)
        cx, cy, s = self.width() // 2, self.height() // 2, 5
        if self.kind == "min":
            painter.drawLine(cx - s, cy, cx + s, cy)
        elif self.kind == "max":
            painter.drawRoundedRect(QRectF(cx - s, cy - s, 2 * s, 2 * s), 1.5, 1.5)
        elif self.kind == "restore":
            r, k = 1.5, 2.5  # Win11 style: a front window plus the top + right edge of the one behind
            front = QRectF(cx - s, cy - s + k, 2 * s - k, 2 * s - k)
            painter.drawRoundedRect(front, r, r)
            back = QtGui.QPainterPath()
            back.moveTo(cx - s + k, front.top())
            back.lineTo(cx - s + k, cy - s + r)
            back.arcTo(QRectF(cx - s + k, cy - s, 2 * r, 2 * r), 180, -90)
            back.lineTo(cx + s - r, cy - s)
            back.arcTo(QRectF(cx + s - 2 * r, cy - s, 2 * r, 2 * r), 90, -90)
            back.lineTo(cx + s, cy + s - k)
            back.lineTo(front.right(), cy + s - k)
            painter.drawPath(back)
        else:
            painter.drawLine(cx - s, cy - s, cx + s, cy + s)
            painter.drawLine(cx - s, cy + s, cx + s, cy - s)


class ThemeToggle(QtWidgets.QPushButton):
    """Day/night switch: a sun by day, a crescent moon at night. Flips :data:`THEME`."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("house", True)
        self.setProperty("chrome", "toggle")
        self.setFixedSize(28, 28)  # the same square key as an IconButton
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setToolTip("Night mode")
        self._hover = HoverFade(self)
        self.clicked.connect(THEME.toggle)

    def event(self, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.HoverEnter, QEvent.Type.Enter):
            self._hover.to(1.0)
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave):
            self._hover.to(0.0)
        return super().event(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        if self._hover.value > 0.01:  # the house key: the whole square, fading in like IconButton's
            painter.setPen(NO_PEN)
            painter.setBrush(rgba(PALETTE["flat_hover"], self._hover.value))
            painter.drawRoundedRect(QRectF(self.rect()), R_TOOL, R_TOOL)
        color = QColor(PALETTE["win_glyph"])
        cx, cy = self.width() / 2.0, self.height() / 2.0
        if THEME.dark:
            full = QtGui.QPainterPath()
            full.addEllipse(QPointF(cx, cy), 6.2, 6.2)
            bite = QtGui.QPainterPath()
            bite.addEllipse(QPointF(cx + 3.2, cy - 2.4), 5.8, 5.8)
            painter.setPen(NO_PEN)
            painter.setBrush(color)
            painter.drawPath(full.subtracted(bite))
        else:
            pen = QPen(color)
            pen.setWidthF(1.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx, cy), 3.4, 3.4)
            for i in range(8):
                a = i * math.pi / 4.0
                painter.drawLine(QPointF(cx + 5.4 * math.cos(a), cy + 5.4 * math.sin(a)), QPointF(cx + 7.4 * math.cos(a), cy + 7.4 * math.sin(a)))


_MAC_MODIFIERS = (("Ctrl+Shift+", "⇧⌘"), ("Ctrl+Alt+", "⌥⌘"), ("Ctrl+", "⌘"), ("Alt+", "⌥"), ("Shift+", "⇧"))
_MAC_ARROWS = {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓"}


def shortcut_text(text: str) -> str:
    """``text`` with its shortcuts written the platform's way: on a Mac Ctrl+O reads ⌘O (Qt binds
    Ctrl to the Command key there) and Alt+Up ⌥↑. Elsewhere the text comes back as it is."""
    if not MAC or "+" not in text:
        return text
    for name, mark in _MAC_MODIFIERS:
        text = text.replace(name, mark)
    return re.sub(r"([⌘⌥⇧])(Left|Right|Up|Down)\b", lambda m: m.group(1) + _MAC_ARROWS[m.group(2)], text)


def zoom_window(window: QtWidgets.QWidget) -> None:
    """The Mac's zoom for a frameless window (the green light, a double-click on the title bar,
    Window › Zoom): fill the screen's work area, menu bar and Dock left clear, or go back to the
    place and size it had. Done by hand: a frameless window has no native zoom state to rely on."""
    if window.isFullScreen() or window.isMaximized():
        window.showNormal()
        return
    screen = window.screen() or QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return
    area = screen.availableGeometry()
    current = window.geometry()
    if all(abs(a - b) <= 2 for a, b in zip(current.getRect(), area.getRect())):
        normal = window.property("_zoom_normal")
        if not isinstance(normal, QRect) or not normal.isValid():
            size = QSize(int(area.width() * 0.8), int(area.height() * 0.8))
            normal = QRect(area.center() - QPoint(size.width() // 2, size.height() // 2), size)
        window.setGeometry(normal)
    else:
        window.setProperty("_zoom_normal", current)
        window.setGeometry(area)


class TrafficLights(QtWidgets.QWidget):
    """A Mac's window controls on the frameless title bar: close, minimise and zoom as the red,
    yellow and green discs at its left end. Their marks (×, −, the zoom arrows) show while the
    pointer is over the three; in a background window, and a dialog's minimise, the discs are
    grey. A press beside the discs moves the window, like the rest of the bar."""

    D, PITCH, INSET = 12.0, 20.0, 14.0  # disc diameter, centre to centre, the first disc's left edge
    #: (fill, rim, mark) for close, minimise and zoom: the Mac's own inks
    INKS = (("#ff5f57", "#e0443e", "#4d0000"), ("#febc2e", "#dea123", "#985712"), ("#28c840", "#1aab29", "#0b650d"))

    def __init__(self, bar: "TitleBar", minimize: bool = True) -> None:
        super().__init__(bar)
        self.bar = bar
        self.keys_enabled = (True, bool(minimize), True)
        self._hover = False
        self._pressed = -1
        self.setFixedSize(int(self.INSET + 2 * self.PITCH + self.D + 10), bar.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _centre(self, index: int) -> QPointF:
        return QPointF(self.INSET + self.D / 2 + index * self.PITCH, self.height() / 2.0)

    def _key_at(self, pos) -> int:
        for index in range(3):
            centre = self._centre(index)
            if self.keys_enabled[index] and math.hypot(pos.x() - centre.x(), pos.y() - centre.y()) <= self.D / 2 + 2:
                return index
        return -1

    def event(self, event) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.HoverEnter, QEvent.Type.Enter):
            self._hover = True
            self.update()
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave):
            self._hover = False
            self.update()
        return super().event(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        index = self._key_at(event.position()) if event.button() == Qt.MouseButton.LeftButton else -1
        if index < 0:
            event.ignore()  # beside the discs: the title bar moves the window
            return
        self._pressed = index
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._key_at(event.position()) < 0:
            event.ignore()  # the title bar's double-click zooms
        # on a disc the first click has acted; the second does nothing

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        index, self._pressed = self._pressed, -1
        self.update()
        if index < 0 or index != self._key_at(event.position()):
            return
        window = self.bar.window_ref
        if index == 0:
            window.close()
        elif index == 1:
            window.showMinimized()
        else:
            self.bar.toggle_max()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIAS)
        awake = self.isActiveWindow() or self._hover
        off_fill, off_rim = ("#4b4e54", "#585c63") if THEME.dark else ("#d5d6d9", "#c2c4c8")
        radius = self.D / 2
        for index, (fill, rim, mark) in enumerate(self.INKS):
            lit = awake and self.keys_enabled[index]
            centre = self._centre(index)
            body = QColor(fill if lit else off_fill)
            if lit and index == self._pressed:
                body = body.darker(125)
            painter.setPen(QPen(QColor(rim if lit else off_rim), 1.0))
            painter.setBrush(body)
            painter.drawEllipse(centre, radius - 0.5, radius - 0.5)
            if not (lit and self._hover):
                continue
            ink = QColor(mark)
            cx, cy = centre.x(), centre.y()
            if index == 2:  # zoom: two arrowheads into opposite corners
                painter.setPen(NO_PEN)
                painter.setBrush(ink)
                a, b = 3.2, 1.4
                painter.drawPolygon(QtGui.QPolygonF([QPointF(cx - a, cy - a), QPointF(cx + b, cy - a), QPointF(cx - a, cy + b)]))
                painter.drawPolygon(QtGui.QPolygonF([QPointF(cx + a, cy + a), QPointF(cx - b, cy + a), QPointF(cx + a, cy - b)]))
                continue
            pen = QPen(ink, 1.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            if index == 0:  # close: ×
                s = 2.6
                painter.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
                painter.drawLine(QPointF(cx - s, cy + s), QPointF(cx + s, cy - s))
            else:  # minimise: −
                s = 3.3
                painter.drawLine(QPointF(cx - s, cy), QPointF(cx + s, cy))


def _about(parent: QtWidgets.QWidget) -> None:
    QtWidgets.QMessageBox.about(parent, "About UStack", "UStack\n\nDesign van der Waals heterostructure stacks and their "
                                "e-beam lithography patterns.\n\nAviram Uri Lab, Department of Physics, Princeton University")


def install_mac_menu(window: QtWidgets.QWidget) -> QtWidgets.QMenuBar | None:
    """The Mac's menu bar (the program has no menus of its own): About and Quit under UAnalysis,
    Close Window (⌘W) under File, Night Mode under View, Minimise (⌘M) and Zoom under Window,
    each acting on the window in front (a dialog included). Nothing elsewhere."""
    if not MAC:
        return None
    bar = QtWidgets.QMenuBar()  # parentless: the one menu bar at the top of the screen

    def front() -> QtWidgets.QWidget:
        return QtWidgets.QApplication.activeWindow() or window

    def zoom() -> None:
        target = front()
        titlebar = getattr(target, "titlebar", None)
        if titlebar is not None:
            titlebar.toggle_max()
        else:
            zoom_window(target)

    file_menu = bar.addMenu("File")
    about = file_menu.addAction("About UStack")
    about.setMenuRole(QtGui.QAction.MenuRole.AboutRole)  # Qt files it under the UAnalysis menu
    about.triggered.connect(lambda: _about(front()))
    close = file_menu.addAction("Close Window")
    close.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close))
    close.triggered.connect(lambda: front().close())
    view = bar.addMenu("View")
    night = view.addAction("Night Mode")
    night.setCheckable(True)
    night.setChecked(THEME.dark)
    night.triggered.connect(lambda checked: THEME.set_dark(bool(checked)))

    def follow_theme() -> None:
        try:
            night.setChecked(THEME.dark)
        except RuntimeError:  # the menu is gone (quitting)
            pass

    THEME.changed.connect(follow_theme)
    windows = bar.addMenu("Window")
    minimise = windows.addAction("Minimise")
    minimise.setShortcut(QtGui.QKeySequence("Ctrl+M"))
    minimise.triggered.connect(lambda: front().showMinimized())
    windows.addAction("Zoom").triggered.connect(zoom)
    window._mac_menu_bar = bar  # kept with the window
    return bar


class _FileOpens(QtCore.QObject):
    """macOS hands a file opened from the Finder (Open With, a double-click, a drop on the Dock
    icon) to the program as an event rather than on the command line, often before the window
    exists: the paths wait in ``pending`` until :meth:`serve` names who opens them."""

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[str] = []
        self.handler: Callable[[str], None] | None = None

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.FileOpen and event.file():
            if self.handler is None:
                self.pending.append(event.file())
            else:
                self.handler(event.file())
            return True
        return False

    def serve(self, handler: Callable[[str], None]) -> None:
        self.handler = handler
        while self.pending:
            handler(self.pending.pop(0))


#: Installed on the application at its start on a Mac (ustack.app.make_app).
FILE_OPENS = _FileOpens()


class TitleBar(QtWidgets.QWidget):
    """The frameless window's title bar, seamless with the window background: the day/night
    toggle and the app mark on the left (then the ``left`` slot), the centred title, the
    ``actions`` slot and the window controls on the right. Press-drag moves the window natively
    (Aero Snap keeps working); a double-click maximises. On a Mac the window controls are the
    red / yellow / green lights at the left end (:class:`TrafficLights`) and maximise is zoom."""

    HEIGHT = 42

    def __init__(self, window: QtWidgets.QWidget, title: str, logo: QtGui.QPixmap | None = None, minimize: bool = True) -> None:
        super().__init__(window)
        self.window_ref = window
        self.logo = logo
        self.setFixedHeight(self.HEIGHT)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0 if MAC else 8, 0, 0, 0)  # the close key runs to the window edge
        layout.setSpacing(6)
        self.lights: TrafficLights | None = None
        if MAC:  # a Mac's window controls lead the bar
            self.lights = TrafficLights(self, minimize)
            layout.addWidget(self.lights, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(ThemeToggle(), 0, Qt.AlignmentFlag.AlignVCenter)
        self.mark: QtWidgets.QLabel | None = None
        if logo is not None and not logo.isNull():
            self.mark = QtWidgets.QLabel()
            self.mark.setPixmap(logo)
            self.mark.setFixedSize(26, 26)
            self.mark.setAlignment(ALIGN_CENTER)
            layout.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignVCenter)
        self.left = QtWidgets.QHBoxLayout()
        self.left.setContentsMargins(4, 0, 0, 0)
        self.left.setSpacing(6)
        layout.addLayout(self.left)
        layout.addStretch(1)
        self.title = QtWidgets.QLabel(title)
        self.title.setProperty("role", "title")
        self.title.setAlignment(ALIGN_CENTER)
        self.title.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.title.setMinimumWidth(80)
        layout.addWidget(self.title, 2, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        self.actions = QtWidgets.QHBoxLayout()
        self.actions.setContentsMargins(0, 0, 12 if MAC else 8, 0)
        self.actions.setSpacing(6)
        layout.addLayout(self.actions)
        self.min_button: WindowButton | None = None
        self.max_button: WindowButton | None = None
        self.close_button: WindowButton | None = None
        if MAC:  # the lights at the left end are the window controls
            return
        # parented from the start: a parentless key made visible below would flash up as its own little window
        self.min_button = WindowButton("min", self, height=self.HEIGHT)
        self.max_button = WindowButton("max", self, height=self.HEIGHT)
        self.close_button = WindowButton("close", self, height=self.HEIGHT)
        self.min_button.clicked.connect(window.showMinimized)
        self.max_button.clicked.connect(self.toggle_max)
        self.close_button.clicked.connect(window.close)
        keys = QtWidgets.QHBoxLayout(); keys.setContentsMargins(0, 0, 0, 0); keys.setSpacing(0)
        for item in (self.min_button, self.max_button, self.close_button):
            keys.addWidget(item)
        self.min_button.setVisible(minimize)  # dialogs: no minimise (an owned window has no taskbar entry)
        layout.addLayout(keys)

    def set_title(self, text: str) -> None:
        self.title.setText(text)

    def set_logo(self, logo: QtGui.QPixmap) -> None:
        """Swap the app mark (a day / night version of it)."""
        self.logo = logo
        if self.mark is not None and logo is not None and not logo.isNull():
            self.mark.setPixmap(logo)

    def toggle_max(self) -> None:
        if MAC:  # the Mac's zoom: the work area, or back again
            zoom_window(self.window_ref)
            return
        if self.window_ref.isMaximized():
            self.window_ref.showNormal()
        else:
            self.window_ref.showMaximized()

    def sync_state(self) -> None:
        if self.max_button is not None:
            self.max_button.set_kind("restore" if self.window_ref.isMaximized() else "max")

    def sync_active(self) -> None:
        """The lights go grey while the window is in the background (as a Mac's do)."""
        if self.lights is not None:
            self.lights.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(PALETTE["app_bg"]))
        if divider_panels():  # the bar is grey, the body is the surface: a hairline between them
            painter.fillRect(QRect(0, self.height() - 1, self.width(), 1), QColor(PALETTE["line"]))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window_ref.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_max()


class _EdgeResizer(QtCore.QObject):
    """Native edge resize for a frameless window (``startSystemResize`` keeps Aero Snap). Watches
    the application's mouse events, so the 6 px hot zone works over child widgets too. macOS has
    no system resize for a frameless window: there the edge follows the pointer by hand."""

    ZONE = 6

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self._cursor = False
        self._drag: tuple | None = None  # a resize by hand: (edges, pointer at the press, geometry at the press)
        # The arrow is an application-wide override, and a pointer that leaves the window fast (to
        # the desktop, a dialog, a popup) sends no move event here to drop it; while it is set, a
        # light poll of the real pointer position drops it once the pointer is off the edge.
        self._watch = QtCore.QTimer(self); self._watch.setInterval(120)
        self._watch.timeout.connect(self._recheck)

    def _recheck(self) -> None:
        if self._drag is not None:  # the arrow stays while an edge is dragged
            return
        over = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        edges = self._edges() if over is None or over.window() is self.window else None
        self._set_cursor(self._shape(edges) if edges is not None else None)

    def _edges(self) -> QtCore.Qt.Edge | None:
        if self.window.isMaximized() or self.window.isFullScreen():
            return None
        pos = self.window.mapFromGlobal(QtGui.QCursor.pos())
        w, h, z = self.window.width(), self.window.height(), self.ZONE
        if not (-1 <= pos.x() <= w and -1 <= pos.y() <= h):
            return None
        edges = Qt.Edge(0)
        if pos.x() <= z:
            edges |= Qt.Edge.LeftEdge
        if pos.x() >= w - z:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= z:
            edges |= Qt.Edge.TopEdge
        if pos.y() >= h - z:
            edges |= Qt.Edge.BottomEdge
        return edges if edges != Qt.Edge(0) else None

    @staticmethod
    def _shape(edges) -> Qt.CursorShape:
        left, right = bool(edges & Qt.Edge.LeftEdge), bool(edges & Qt.Edge.RightEdge)
        top, bottom = bool(edges & Qt.Edge.TopEdge), bool(edges & Qt.Edge.BottomEdge)
        if (top and left) or (bottom and right):
            return Qt.CursorShape.SizeFDiagCursor
        if (top and right) or (bottom and left):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def _set_cursor(self, shape) -> None:
        if shape is None:
            if self._cursor:
                QtWidgets.QApplication.restoreOverrideCursor()
                self._cursor = False
                self._watch.stop()
            return
        if self._cursor:
            QtWidgets.QApplication.changeOverrideCursor(QtGui.QCursor(shape))
        else:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(shape))
            self._cursor = True
            self._watch.start()

    def _drag_to(self, pos: QPoint) -> None:
        edges, start, geometry = self._drag
        dx, dy = pos.x() - start.x(), pos.y() - start.y()
        least = self.window.minimumSize()  # the window's own minimum, else its layout's
        if least.isEmpty():
            least = self.window.minimumSizeHint()
        rect = QRect(geometry)
        if edges & Qt.Edge.LeftEdge:
            rect.setLeft(min(geometry.left() + dx, geometry.right() - least.width() + 1))
        if edges & Qt.Edge.RightEdge:
            rect.setRight(max(geometry.right() + dx, geometry.left() + least.width() - 1))
        if edges & Qt.Edge.TopEdge:
            rect.setTop(min(geometry.top() + dy, geometry.bottom() - least.height() + 1))
        if edges & Qt.Edge.BottomEdge:
            rect.setBottom(max(geometry.bottom() + dy, geometry.top() + least.height() - 1))
        if rect != self.window.geometry():
            self.window.setGeometry(rect)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        kind = event.type()
        if self._drag is not None:  # an edge dragged by hand: the pointer's moves are the resize's
            if kind == QEvent.Type.MouseMove:
                self._drag_to(event.globalPosition().toPoint())
                return True
            if kind == QEvent.Type.MouseButtonRelease:
                self._drag = None
                self._set_cursor(None)
                return True
        if kind not in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.HoverMove):
            return False
        if not isinstance(obj, QtWidgets.QWidget):
            return False
        if obj.window() is not self.window:
            if kind != QEvent.Type.MouseButtonPress:
                self._set_cursor(None)  # the pointer is over another window of the app
            return False
        if kind in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
            if QtWidgets.QApplication.mouseButtons() == Qt.MouseButton.NoButton:
                edges = self._edges()
                self._set_cursor(self._shape(edges) if edges is not None else None)
            return False
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._edges()
            handle = self.window.windowHandle()
            if edges is not None and handle is not None:
                if not MAC:
                    self._set_cursor(None)
                    if handle.startSystemResize(edges):
                        return True
                # no system resize for a frameless window here (macOS): the edge follows the pointer
                self._drag = (edges, event.globalPosition().toPoint(), self.window.geometry())
                self._set_cursor(self._shape(edges))
                return True
        return False


class FramelessMainWindow(QtWidgets.QMainWindow):
    """QMainWindow with the lab's frameless chrome: :class:`TitleBar` as the menu widget, a 1 px
    window border, rounded Win11 corners while floating and square ones when maximised."""

    def install_chrome(self, title: str, logo: QtGui.QPixmap | None = None) -> TitleBar:
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.titlebar = TitleBar(self, title, logo)
        self.setMenuWidget(self.titlebar)
        self.setContentsMargins(1, 1, 1, 1)
        self._resizer = _EdgeResizer(self)
        install_app_filter(self._resizer)
        return self.titlebar

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, "titlebar"):
            margin = 0 if self.isMaximized() else 1
            self.setContentsMargins(margin, margin, margin, margin)
            self.titlebar.sync_state()
            apply_window_chrome(self)
        elif event.type() == QEvent.Type.ActivationChange and hasattr(self, "titlebar"):
            self.titlebar.sync_active()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(window_fill()))
        if hasattr(self, "titlebar") and not self.isMaximized():
            painter.setPen(QColor(PALETTE["win_border"]))
            painter.setBrush(NO_BRUSH)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


# --------------------------------------------------------------------------- small helpers
def title_case(text: str) -> str:
    """Every word's first letter capitalised, the rest left alone (so FFT, 2D, µm keep theirs)."""
    return " ".join(word[:1].upper() + word[1:] for word in text.split(" "))


def label(text: str = "", role: str | None = None, wrap: bool = False) -> QtWidgets.QLabel:
    """A label in a QSS ``role``; section titles (``cardTitle``) are title-cased."""
    item = QtWidgets.QLabel(title_case(text) if role == "cardTitle" else text)
    if role:
        item.setProperty("role", role)
    if role in BOLD_ROLES:
        item.setFont(bold_font(item.font()))  # the hinting; the weight comes from the QSS
    if wrap:
        item.setWordWrap(True)
    return item


def set_role(widget: QtWidgets.QWidget, role: str) -> None:
    widget.setProperty("role", role)
    repolish(widget)


def button(text: str, callback: Callable | None = None, variant: str | None = None, tooltip: str | None = None, icon_name: str | None = None) -> Button:
    item = Button(text, variant)
    if tooltip:
        item.setToolTip(tooltip)
    if icon_name:
        item.set_icon_name(icon_name)
    if callback is not None:
        item.clicked.connect(callback)
    return item


class Pill(QtWidgets.QLabel):
    """Status word beside a title, as plain small text (no badge). ``tone`` is neutral / accent /
    ok / warn; only ok and warn colour the text."""

    def __init__(self, text: str = "", tone: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.setProperty("pill", tone)
        self.setAlignment(ALIGN_LEFT_VCENTER)

    def set_tone(self, tone: str, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        if self.property("pill") != tone:
            self.setProperty("pill", tone)
            repolish(self)


class Banner(QtWidgets.QLabel):
    """A note: wrapped text in the secondary ink behind a 2 px rule on the left (no tinted box)."""

    def __init__(self, text: str = "", tone: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.setProperty("banner", tone)
        self.setWordWrap(True)

    def set_tone(self, tone: str, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        if self.property("banner") != tone:
            self.setProperty("banner", tone)
            repolish(self)


class Card(Panel):
    """Settings section with a title row; ``layout`` receives the body widgets. A ``hint`` becomes
    the title's tooltip (no ⓘ glyph)."""

    def __init__(self, title: str, hint: str | None = None, trailing: QtWidgets.QWidget | None = None) -> None:
        super().__init__(kind="section")
        inset = PANEL_STYLE == "inset"
        outer = QtWidgets.QVBoxLayout(self)
        # "inset": the title sits on the background flush with the white group's left edge; the
        # group starts just below it and only the body (``layout``) is indented inside it.
        outer.setContentsMargins(0 if inset else 12, 10, 0 if inset else 12, 12)
        outer.setSpacing(8)
        heading = QtWidgets.QHBoxLayout()
        heading.setContentsMargins(0, 0, 0, 12 if inset else 2)
        heading.setSpacing(6)
        self.title_label = label(title, "cardTitle")
        heading.addWidget(self.title_label)
        if hint:
            self.title_label.setToolTip(hint)
        heading.addStretch()
        if trailing is not None:
            heading.addWidget(trailing)
        outer.addLayout(heading)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(12 if inset else 0, 0, 12 if inset else 0, 0)
        self.layout.setSpacing(8)
        outer.addLayout(self.layout)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if divider_panels():
            # Divider style: a section of the column, closed by a hairline across the full width.
            QPainter(self).fillRect(QRect(0, self.height() - 1, self.width(), 1), QColor(PALETTE["line"]))
            return
        if PANEL_STYLE == "inset":
            # Inset style: the title on the background, the body in a white group below it.
            group = self.card_rect()
            group.setTop(self.title_label.geometry().bottom() + 6)
            pixel_frame(QPainter(self), group, R_CARD, None, fill=PALETTE["surface"])
            return
        super().paintEvent(event)


def scroll(content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # The bar's gutter is always kept: a card opening past the bottom then shows the bar without
    # narrowing the column (an as-needed bar squeezed every field by its width when it appeared).
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    area.verticalScrollBar().rangeChanged.connect(lambda *_: _ScrollBarReveal.update(area))
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidget(content)
    return area


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())
