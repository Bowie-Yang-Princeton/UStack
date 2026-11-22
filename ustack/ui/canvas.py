"""The design canvas: optical images, flake outlines, defects and EBL patterns in one µm frame.

The world is µm with y up (the GDS / chip convention); the view flips y so images look upright.
An image is placed by a similarity transform px → µm (scale, rotation, translation, y flipped).

Tools: pointer (drag outline vertices; double-click an edge to add one; Delete removes the
selected vertex), pan (or the middle button / H), wand (click a flake: its border is detected),
draw (click points, double-click to close: defects, windows, manual outlines), move image
(drag the selected layer's image), pick (report clicked points: alignment pairs, the device site).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .style import PALETTE, QtCore, QtGui, QtWidgets, Signal, mix

Qt = QtCore.Qt
QPointF = QtCore.QPointF

TOOLS = ("pointer", "pan", "wand", "draw", "move", "pick", "measure", "addpoint", "select")


def image_transform(scale: float, angle_deg: float, tx: float, ty: float, mirror: bool = False) -> QtGui.QTransform:
    """px (x right, y down) → µm (y up)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    mx = -1.0 if mirror else 1.0
    # X = c*mx*x + s*y + tx ; Y = s*mx*x - c*y + ty
    return QtGui.QTransform(c * mx, s * mx, s, -c, tx, ty)


def qimage_from_bgr(bgr: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return img.copy()


_PIXMAPS: dict[int, tuple[np.ndarray, QtGui.QPixmap, float]] = {}


def _pixmap_for(bgr: np.ndarray, max_side: int) -> tuple[QtGui.QPixmap, float]:
    """Downsampled pixmap of an image array, cached per array (redraws reuse it)."""
    hit = _PIXMAPS.get(id(bgr))
    if hit is not None and hit[0] is bgr:
        return hit[1], hit[2]
    h, w = bgr.shape[:2]
    f = min(1.0, max_side / max(h, w))
    small = bgr
    if f < 1.0:
        import cv2
        small = cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    pm = QtGui.QPixmap.fromImage(qimage_from_bgr(small))
    if len(_PIXMAPS) > 12:
        _PIXMAPS.clear()
    _PIXMAPS[id(bgr)] = (bgr, pm, f)
    return pm, f


class ImageItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, key: str, bgr: np.ndarray, max_side: int = 3000):
        h, w = bgr.shape[:2]
        pm, f = _pixmap_for(bgr, max_side)
        self.full_size = (w, h)
        self.display_factor = f
        super().__init__(pm)
        self.key = key
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setShapeMode(QtWidgets.QGraphicsPixmapItem.ShapeMode.BoundingRectShape)

    def set_placement(self, scale: float, angle_deg: float, tx: float, ty: float, mirror: bool = False) -> None:
        """``scale`` is µm per ORIGINAL pixel; the pixmap may be downsampled."""
        t = QtGui.QTransform.fromScale(1 / self.display_factor, 1 / self.display_factor) * image_transform(scale, angle_deg, tx, ty, mirror)
        self.setTransform(t)

    def scene_to_px(self, p: QPointF) -> tuple[float, float]:
        q = self.mapFromScene(p)
        return q.x() / self.display_factor, q.y() / self.display_factor

    def px_to_scene(self, x: float, y: float) -> QPointF:
        return self.mapToScene(QPointF(x * self.display_factor, y * self.display_factor))


class Handle(QtWidgets.QGraphicsRectItem):
    SIZE = 7

    def __init__(self, owner: "PolygonItem", index: int):
        s = self.SIZE
        super().__init__(-s / 2, -s / 2, s, s)
        self.owner, self.index = owner, index
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(1000)
        self._style(False)

    def _style(self, hot: bool) -> None:
        self.setBrush(QtGui.QColor(PALETTE["accent"] if hot or self.isSelected() else PALETTE["surface"]))
        self.setPen(QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.2))

    def hoverEnterEvent(self, e):  # noqa: N802 - Qt API
        self._style(True)

    def hoverLeaveEvent(self, e):  # noqa: N802 - Qt API
        self._style(False)

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._style(False)
        return super().itemChange(change, value)

    def mousePressEvent(self, e):  # noqa: N802 - Qt API
        e.accept()
        if e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self.setSelected(not self.isSelected())        # Shift / Ctrl-click adds or removes this point
            return
        if not self.isSelected():
            self.owner.select_only([self.index])
        self.owner.begin_drag(e.scenePos())                 # dragging moves every selected point

    def mouseMoveEvent(self, e):  # noqa: N802 - Qt API
        self.owner.drag_to(e.scenePos())

    def mouseReleaseEvent(self, e):  # noqa: N802 - Qt API
        self.owner.end_drag()


class IssueMarker(QtWidgets.QGraphicsEllipseItem):
    """A clickable circle marking an error on the canvas (same size at any zoom); a click shows what it is."""

    def __init__(self, key: str, r: float):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.key = key
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click: what this error is (close, change the rule, or accept it)")

    def shape(self):                                 # the whole disc is clickable, not only the ring
        path = QtGui.QPainterPath()
        path.addEllipse(self.rect().adjusted(-3, -3, 3, 3))
        return path


class RotateHandle(QtWidgets.QGraphicsEllipseItem):
    """The round handle above the selected outline (as in PowerPoint): drag it around to turn the
    outline about its middle; Shift snaps to 15° steps. Same size at any zoom."""

    SIZE = 12

    def __init__(self, canvas: "Canvas"):
        s = self.SIZE
        super().__init__(-s / 2, -s / 2, s, s)
        self.canvas = canvas
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setZValue(1001)
        self.setToolTip("Drag to rotate the outline about its middle (Shift: 15° steps)")
        self._style(False)

    def _style(self, hot: bool) -> None:
        self.setBrush(QtGui.QColor(PALETTE["accent"] if hot else PALETTE["surface"]))
        self.setPen(QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.6))

    def paint(self, painter, option, widget=None):  # a circle with a small turning arrow
        super().paint(painter, option, widget)
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["surface"] if self.brush().color() == QtGui.QColor(PALETTE["accent"])
                                               else PALETTE["accent"]), 1.2))
        r = QtCore.QRectF(-3.2, -3.2, 6.4, 6.4)
        painter.drawArc(r, 30 * 16, 270 * 16)

    def hoverEnterEvent(self, e):  # noqa: N802 - Qt API
        self._style(True)
        self.update()

    def hoverLeaveEvent(self, e):  # noqa: N802 - Qt API
        self._style(False)
        self.update()

    def mousePressEvent(self, e):  # noqa: N802 - Qt API
        e.accept()
        self.canvas.begin_rotate(e.scenePos())

    def mouseMoveEvent(self, e):  # noqa: N802 - Qt API
        self.canvas.rotate_to(e.scenePos(), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))

    def mouseReleaseEvent(self, e):  # noqa: N802 - Qt API
        self.canvas.end_rotate()


class _NoHitPolygon(QtWidgets.QGraphicsPolygonItem):
    """The dashed box around the selected outline: drawn only, never hit (a polygon item's shape is
    its whole inside, so ``itemAt`` would return it instead of the point handles under it)."""

    def shape(self):  # noqa: D102 - Qt API
        return QtGui.QPainterPath()


class ScaleHandle(QtWidgets.QGraphicsRectItem):
    """One of the eight square handles on the selected outline's box (as in PowerPoint): a side
    handle stretches the outline across the screen (horizontal / vertical as seen), a corner handle
    both ways; Shift on a corner keeps the proportions, Alt / Option scales about the middle. Same size
    at any zoom."""

    SIZE = 9
    CURSORS = {(-1, -1): Qt.CursorShape.SizeFDiagCursor, (1, 1): Qt.CursorShape.SizeFDiagCursor,
               (1, -1): Qt.CursorShape.SizeBDiagCursor, (-1, 1): Qt.CursorShape.SizeBDiagCursor,
               (-1, 0): Qt.CursorShape.SizeHorCursor, (1, 0): Qt.CursorShape.SizeHorCursor,
               (0, -1): Qt.CursorShape.SizeVerCursor, (0, 1): Qt.CursorShape.SizeVerCursor}

    def __init__(self, canvas: "Canvas", side: tuple[int, int]):
        s = self.SIZE
        super().__init__(-s / 2, -s / 2, s, s)
        self.canvas, self.side = canvas, side            # side: (−1 / 0 / +1 along screen x, along screen y; −1 = top)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1001)
        self.setToolTip("Drag to stretch the outline" + (" (Shift: keep proportions, Alt: about the middle)" if all(side)
                                                          else " (Alt: about the middle)"))
        self._style(False)

    def _style(self, hot: bool) -> None:
        self.setBrush(QtGui.QColor(PALETTE["accent"] if hot else PALETTE["accentBg"]))
        self.setPen(QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.4))

    def hoverEnterEvent(self, e):  # noqa: N802 - Qt API
        self._style(True)
        self.update()

    def hoverLeaveEvent(self, e):  # noqa: N802 - Qt API
        self._style(False)
        self.update()


class PolygonItem(QtWidgets.QGraphicsPolygonItem):
    """An outline the user can edit: vertex handles appear while it is the active outline."""

    def __init__(self, key: str, points: list[tuple[float, float]], color: str, editable: bool = True, fill_alpha: float = 0.12,
                 on_change: Callable[[str, list[tuple[float, float]]], None] | None = None, width: float = 1.6, dash: bool = False):
        super().__init__()
        self.key, self.color, self.editable = key, color, editable
        self.points = [tuple(p) for p in points]
        self.on_change = on_change
        self.handles: list[Handle] = []
        pen = QtGui.QPen(QtGui.QColor(color), width)
        pen.setCosmetic(True)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        c = QtGui.QColor(color)
        c.setAlphaF(fill_alpha)
        self.setBrush(c)
        self._refresh()

    def _refresh(self) -> None:
        self.setPolygon(QtGui.QPolygonF([QPointF(x, y) for x, y in self.points]))
        sc = self.scene()
        if sc is not None and self.handles:
            for v in sc.views():
                if hasattr(v, "place_rotate_handle"):
                    v.place_rotate_handle()

    def set_marked(self, on: bool) -> None:
        """Part of a box selection: a heavier line (the handles stay on one outline at a time)."""
        pen = self.pen()
        base = getattr(self, "_base_width", None)
        if base is None:
            base = self._base_width = pen.widthF()
        pen.setWidthF(base * 2.2 if on else base)
        self.setPen(pen)

    def set_active(self, active: bool) -> None:
        for h in self.handles:
            if h.scene():
                h.scene().removeItem(h)
        self.handles = []
        if active and self.editable and self.scene():
            for i, (x, y) in enumerate(self.points):
                h = Handle(self, i)
                h.setPos(x, y)
                self.scene().addItem(h)
                self.handles.append(h)

    def move_vertex(self, i: int, pos: QPointF) -> None:
        self.points[i] = (pos.x(), pos.y())
        self.handles[i].setPos(pos)
        self._refresh()

    # ---- selection of points (several can be moved or deleted together)
    def selected_indices(self) -> list[int]:
        return [h.index for h in self.handles if h.isSelected()]

    def select_only(self, indices) -> None:
        keep = set(indices)
        for h in self.handles:
            h.setSelected(h.index in keep)

    def select_in_rect(self, rect: QtCore.QRectF, add: bool = False) -> int:
        """Select the points inside ``rect`` (scene units); ``add`` keeps the points already selected."""
        r = rect.normalized()
        for h in self.handles:
            inside = r.contains(h.pos())
            h.setSelected(inside or (add and h.isSelected()))
        return len(self.selected_indices())

    def begin_drag(self, pos: QPointF) -> None:
        self._drag_from = QPointF(pos)
        self._drag_orig = {i: self.points[i] for i in self.selected_indices()}
        self._drag_moved = False

    def drag_to(self, pos: QPointF) -> None:
        if getattr(self, "_drag_from", None) is None:
            return
        dx, dy = pos.x() - self._drag_from.x(), pos.y() - self._drag_from.y()
        for i, (x, y) in self._drag_orig.items():
            self.points[i] = (x + dx, y + dy)
            self.handles[i].setPos(x + dx, y + dy)
        self._drag_moved = True
        self._refresh()

    def end_drag(self) -> None:
        moved = getattr(self, "_drag_moved", False)
        self._drag_from, self._drag_orig, self._drag_moved = None, {}, False
        if moved:
            self.commit()

    # ---- moving the whole outline (click on its line and drag)
    def begin_move(self, pos: QPointF) -> None:
        self._move_from = QPointF(pos)
        self._move_orig = list(self.points)
        self._move_moved = False

    def move_to(self, pos: QPointF) -> None:
        if getattr(self, "_move_from", None) is None:
            return
        dx, dy = pos.x() - self._move_from.x(), pos.y() - self._move_from.y()
        self.points = [(x + dx, y + dy) for x, y in self._move_orig]
        for h, (x, y) in zip(self.handles, self.points):
            h.setPos(x, y)
        self._move_moved = True
        self._refresh()

    def end_move(self) -> None:
        moved = getattr(self, "_move_moved", False)
        self._move_from, self._move_orig, self._move_moved = None, [], False
        if moved:
            self.commit()

    def nearest_edge(self, pos: QPointF) -> tuple[float, int, QPointF]:
        """(distance, index of the edge's first point, closest point on it) for the edge nearest ``pos``."""
        best = (math.inf, -1, QPointF(pos))
        n = len(self.points)
        for i in range(n):
            (x0, y0), (x1, y1) = self.points[i], self.points[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy or 1e-12
            t = max(0.0, min(1.0, ((pos.x() - x0) * dx + (pos.y() - y0) * dy) / L2))
            q = QPointF(x0 + t * dx, y0 + t * dy)
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < best[0]:
                best = (d, i, q)
        return best

    def insert_vertex_near(self, pos: QPointF, commit: bool = True, on_edge: bool = False) -> int | None:
        """Add a point on the edge nearest ``pos`` (at ``pos``, or at its closest point on the edge when
        ``on_edge``); it becomes the selected point. Returns its index."""
        d, i, q = self.nearest_edge(pos)
        if i < 0:
            return None
        p = q if on_edge else pos
        self.points.insert(i + 1, (p.x(), p.y()))
        self._refresh()
        self.set_active(True)
        self.select_only([i + 1])
        if commit:
            self.commit()
        return i + 1

    def delete_selected_vertices(self) -> bool:
        """Delete the selected points; their neighbours join up. A border keeps at least 3 points."""
        sel = self.selected_indices()
        if not sel or len(self.points) - len(sel) < 3:
            return False
        for i in sorted(sel, reverse=True):
            del self.points[i]
        self._refresh()
        self.set_active(True)
        self.commit()
        return True

    delete_selected_vertex = delete_selected_vertices

    def commit(self) -> None:
        if self.on_change:
            self.on_change(self.key, list(self.points))


class RulerLabel(QtWidgets.QGraphicsItem):
    """The length of a measurement, on a small plate that keeps its size at any zoom."""

    def __init__(self, text: str = ""):
        super().__init__()
        self.text = text
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(620)

    def set_text(self, text: str) -> None:
        self.prepareGeometryChange()
        self.text = text
        self.update()

    def _rect(self) -> QtCore.QRectF:
        fm = QtGui.QFontMetricsF(self._font())
        return QtCore.QRectF(8, -fm.height() - 12, fm.horizontalAdvance(self.text) + 12, fm.height() + 6)

    @staticmethod
    def _font() -> QtGui.QFont:
        f = QtGui.QFont()
        f.setBold(True)
        return f

    def boundingRect(self):  # noqa: N802 - Qt API
        return self._rect().adjusted(-1, -1, 1, 1)

    def paint(self, painter, option, widget=None):
        r = self._rect()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1))
        painter.setBrush(QtGui.QColor(PALETTE["surface"]))
        painter.drawRoundedRect(r, 3, 3)
        painter.setFont(self._font())
        painter.setPen(QtGui.QColor(PALETTE["text"]))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text)


class Canvas(QtWidgets.QGraphicsView):
    """See the module docstring."""

    pointPicked = Signal(float, float)
    wandClicked = Signal(str, float, float)          # image key, px x, px y
    polygonDrawn = Signal(list)                       # list of (x, y) µm
    polygonEdited = Signal(str, list)                 # key, points
    imageMoved = Signal(str, float, float)            # key, dx, dy (µm)
    cursorMoved = Signal(float, float)
    measured = Signal(float)                          # length of a finished measurement (scene unit)
    verticesSelected = Signal(int)                    # number of selected points of the active outline
    message = Signal(str)                             # a short note for the window's hint line
    polygonSelected = Signal(str)                     # an outline was picked by clicking its line
    outlinesSelected = Signal(list)                   # outlines chosen together by a box (keys)
    rotationChanged = Signal(float)                   # view rotation (degrees, counter-clockwise)
    markerClicked = Signal(str, float, float)         # a clickable marker (an error's red circle): its key, where (µm)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.scale(1, -1)
        self.scene().setSceneRect(-1e6, -1e6, 2e6, 2e6)  # µm: room to pan anywhere on a chip
        self.tool = "pointer"
        self.images: dict[str, ImageItem] = {}
        self.polys: dict[str, PolygonItem] = {}
        self.overlays: list[QtWidgets.QGraphicsItem] = []
        self.active_image: str | None = None
        self.active_poly: str | None = None
        self._pan: QtCore.QPoint | None = None
        self._move_from: QPointF | None = None
        self._draft: list[tuple[float, float]] = []
        self._draft_item: QtWidgets.QGraphicsPathItem | None = None
        self.empty_text = "No images"
        self.unit = "µm"                                   # scene unit shown by the scale bar
        self.fit_on_resize = False                         # keep the content fitted until the user zooms
        self.drag_pans = True                              # pointer tool: drag (not on a point) moves the view
        self.box_select_outlines = False                   # pointer drag on nothing: box-select every outline inside (pop-ups)
        self.allow_twist = True                            # a trackpad twist turns the view
        self.multi: set[str] = set()                       # outlines chosen together (box selection): moved / deleted together
        self._group_move: list | None = None
        self._band_outlines = False
        self._zoomed = False
        self._ruler_start: QPointF | None = None
        self._ruler_done = False
        self._ruler_items: list = []
        self._band_from: QtCore.QPoint | None = None
        self._add_drag: PolygonItem | None = None
        self._pan_start: QtCore.QPoint | None = None
        self.drag_outlines = False                         # pointer: click an outline's line to select it, drag to move it
        self.trackpad_gestures = False                     # two-finger scroll pans, pinch zooms, twist rotates the view
        self.view_angle = 0.0
        self._poly_move: PolygonItem | None = None
        self._band: QtWidgets.QRubberBand | None = None
        self._keep_sel: dict[str, tuple[int, list[int]]] = {}   # outline key -> (point count, selected) across a redraw
        self.rotate_handles = False                        # a PowerPoint-style rotate handle above the selected outline
        self._rot_handle: RotateHandle | None = None
        self._rot_line: QtWidgets.QGraphicsLineItem | None = None
        self._rot: dict | None = None
        self.scale_handles = False                         # PowerPoint-style stretch / scale handles on the selected outline's box
        self._scale_items: list[ScaleHandle] = []
        self._scale_box: QtWidgets.QGraphicsPolygonItem | None = None
        self._scl: dict | None = None
        self.scene().selectionChanged.connect(self._selection_changed)
        self.setBackgroundBrush(QtGui.QColor(PALETTE["surface"]))

    # ------------------------------------------------------------------ content
    def clear_all(self) -> None:
        self._remember_selection()
        self._multi_keep = set(self.multi)              # chosen again when the same outlines are drawn again
        self.multi = set()
        self._rot_handle, self._rot_line, self._rot = None, None, None
        self._scale_items, self._scl, self._scale_box = [], None, None
        self.scene().clear()
        self.scene().setSceneRect(-1e6, -1e6, 2e6, 2e6)
        self.images, self.polys, self.overlays = {}, {}, []
        self._draft_item = None
        self._draft = []
        self._ruler_items, self._ruler_start, self._ruler_done = [], None, False

    def set_image(self, key: str, bgr: np.ndarray | None, placement: tuple | None = None, opacity: float = 1.0, z: float = 0.0,
                  visible: bool = True) -> ImageItem | None:
        old = self.images.pop(key, None)
        if old is not None:
            self.scene().removeItem(old)
        if bgr is None:
            return None
        item = ImageItem(key, bgr)
        self.scene().addItem(item)
        if placement:
            item.set_placement(*placement)
        item.setOpacity(opacity)
        item.setZValue(z)
        item.setVisible(visible)
        self.images[key] = item
        return item

    def set_polygon(self, key: str, points: list | None, color: str, editable: bool = True, z: float = 50.0, fill_alpha: float = 0.12,
                    width: float = 1.6, dash: bool = False) -> PolygonItem | None:
        old = self.polys.pop(key, None)
        if old is not None:
            if old.handles:
                self._keep_sel[key] = (len(old.points), old.selected_indices())
            old.set_active(False)
            self.scene().removeItem(old)
        if not points or len(points) < 3:
            return None
        item = PolygonItem(key, points, color, editable, fill_alpha, self._edited, width, dash)
        item.setZValue(z)
        self.scene().addItem(item)
        self.polys[key] = item
        if self.active_poly == key:
            item.set_active(True)
            self._restore_selection(item)
        if key in getattr(self, "_multi_keep", set()):      # it was part of a box selection before the redraw
            self.multi.add(key)
            if len(self._multi_keep) > 1:
                item.set_marked(True)
        return item

    # ------------------------------------------------------------------ selected points
    def _remember_selection(self) -> None:
        item = self.polys.get(self.active_poly) if self.active_poly else None
        if item is not None and item.handles:
            self._keep_sel[item.key] = (len(item.points), item.selected_indices())

    def _restore_selection(self, item: "PolygonItem") -> None:
        """A redraw after an edit rebuilds the outline: keep its selected points (same point count)."""
        n, sel = self._keep_sel.pop(item.key, (None, []))
        if n == len(item.points) and sel:
            item.select_only(sel)

    def _selection_changed(self) -> None:
        try:
            n = sum(isinstance(it, Handle) for it in self.scene().selectedItems())
        except RuntimeError:          # the scene is being torn down
            return
        self.verticesSelected.emit(n)

    def selected_vertex_count(self) -> int:
        item = self.polys.get(self.active_poly) if self.active_poly else None
        return len(item.selected_indices()) if item is not None else 0

    def delete_selected_vertices(self) -> bool:
        """Delete the selected points of the active outline (Delete / Backspace, or a trash button)."""
        item = self.polys.get(self.active_poly) if self.active_poly else None
        if item is None or not item.selected_indices():
            return False
        if item.delete_selected_vertices():
            return True
        self.message.emit("A border needs at least 3 points: not deleted.")
        return False

    def select_all_vertices(self) -> None:
        item = self.polys.get(self.active_poly) if self.active_poly else None
        if item is not None:
            item.select_only(range(len(item.points)))

    def _px_per_unit(self) -> float:
        t = self.transform()
        return math.sqrt(abs(t.m11() * t.m22() - t.m12() * t.m21())) or 1e-12

    def _near_px(self, px: float = 8.0) -> float:
        """``px`` screen px in scene units (how close a click must be to a line)."""
        return px / max(self._px_per_unit(), 1e-12)

    def _edge_hit(self, pos: QPointF) -> "PolygonItem | None":
        """The editable outline whose line passes within 6 screen px of ``pos`` (topmost, then nearest)."""
        tol, best = self._near_px(6.0), None
        for item in self.polys.values():
            if not item.editable or not item.isVisible():
                continue
            d = item.nearest_edge(pos)[0]
            if d <= tol and (best is None or (item.zValue(), -d) > (best[0], -best[1])):
                best = (item.zValue(), d, item)
        return best[2] if best else None

    # ------------------------------------------------------------------ view rotation
    def set_view_rotation(self, deg: float) -> None:
        """Turn the view (degrees, counter-clockwise on screen) about the middle of the window; the
        scale stays."""
        deg = math.remainder(float(deg), 360.0)
        if abs(deg - self.view_angle) < 1e-9:
            return
        centre = self.mapToScene(self.viewport().rect().center())
        m = self._px_per_unit()
        self.setTransform(QtGui.QTransform().rotate(deg) * QtGui.QTransform.fromScale(m, -m))
        self.centerOn(centre)
        self.view_angle = deg
        self.place_rotate_handle()
        self.rotationChanged.emit(deg)

    def view_state(self) -> dict:
        """Centre (scene units), scale (px per unit) and rotation of the view: saved with the device."""
        c = self.mapToScene(self.viewport().rect().center())
        return {"cx": float(c.x()), "cy": float(c.y()), "scale": float(self._px_per_unit()), "angle": float(self.view_angle)}

    def set_view_state(self, v: dict) -> None:
        m = float(v.get("scale") or self._px_per_unit())
        deg = float(v.get("angle", 0.0))
        self.setTransform(QtGui.QTransform().rotate(deg) * QtGui.QTransform.fromScale(m, -m))
        self.view_angle = deg
        self.centerOn(QPointF(float(v.get("cx", 0.0)), float(v.get("cy", 0.0))))
        self._zoomed = True
        self.rotationChanged.emit(deg)
        self.place_rotate_handle()

    def remove_polygons(self, prefix: str) -> None:
        for key in [k for k in self.polys if k.startswith(prefix)]:
            item = self.polys.pop(key)
            item.set_active(False)
            self.scene().removeItem(item)

    def _edited(self, key: str, pts: list) -> None:
        self.polygonEdited.emit(key, pts)

    def set_multi(self, keys) -> None:
        """Outlines chosen together (a box): drawn heavier; dragging one moves them all; the windows delete them all."""
        old = self.multi
        self.multi = {k for k in keys if k in self.polys}
        for k in old | self.multi:
            it = self.polys.get(k)
            if it is not None:
                it.set_marked(k in self.multi and len(self.multi) > 1)

    def selection(self) -> list:
        """The outlines the handles act on: the outlines chosen together (Shift-click / a box) when there are several,
        else the selected one."""
        if len(self.multi) > 1:
            items = [self.polys[k] for k in sorted(self.multi) if k in self.polys]
            items = [it for it in items if it.editable and it.isVisible() and len(it.points) >= 3]
            if len(items) > 1:
                return items
        item = self.polys.get(self.active_poly) if self.active_poly else None
        return [item] if item is not None and item.editable and item.isVisible() and len(item.points) >= 3 else []

    def set_active_polygon(self, key: str | None) -> None:
        if key is not None and key not in self.multi and self.multi:
            self.set_multi([])                         # another outline chosen alone: the group is dropped
        for k, item in self.polys.items():
            if k == key and item.handles:
                continue                      # already active: keep its handles (and selected points)
            item.set_active(k == key)
        self.active_poly = key
        self.place_rotate_handle()

    # ------------------------------------------------------------------ rotate handle
    def place_rotate_handle(self) -> None:
        """Put the rotate handle 26 screen px above the selected outline's box (hidden without one),
        and the scale handles on that box (``place_scale_handles``)."""
        self.place_scale_handles()
        if not getattr(self, "rotate_handles", False):
            return
        sel = self.selection()
        if not sel:
            for it in (self._rot_handle, self._rot_line):
                if it is not None:
                    it.setVisible(False)
            return
        if self._rot_handle is None or self._rot_handle.scene() is None:
            self._rot_handle = RotateHandle(self)
            self.scene().addItem(self._rot_handle)
            self._rot_line = QtWidgets.QGraphicsLineItem()
            pen = QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.0)
            pen.setCosmetic(True)
            self._rot_line.setPen(pen)
            self._rot_line.setZValue(1000)
            self.scene().addItem(self._rot_line)
        if self._rot is not None:                     # while turning, the handle follows the pointer
            return
        scr = [self.mapFromScene(QPointF(x, y)) for it in sel for x, y in it.points]
        xs, ys = [p.x() for p in scr], [p.y() for p in scr]
        top = QtCore.QPointF((min(xs) + max(xs)) / 2, min(ys))
        handle = self.mapToScene(QtCore.QPoint(int(round(top.x())), int(round(top.y() - 26))))
        anchor = self.mapToScene(QtCore.QPoint(int(round(top.x())), int(round(top.y()))))
        self._rot_handle.setPos(handle)
        self._rot_line.setLine(QtCore.QLineF(anchor, handle))
        self._rot_handle.setVisible(True)
        self._rot_line.setVisible(True)

    # ------------------------------------------------------------------ scale handles
    def _screen_box(self, pts) -> tuple[float, float, float, float]:
        t = self.viewportTransform()
        scr = [t.map(QPointF(x, y)) for x, y in pts]
        xs, ys = [p.x() for p in scr], [p.y() for p in scr]
        return min(xs), min(ys), max(xs), max(ys)

    def place_scale_handles(self) -> None:
        """Eight square handles on the selected outline's box, as seen on screen (hidden without one)."""
        if not getattr(self, "scale_handles", False):
            return
        sel = self.selection()
        show = bool(sel)
        if show and (not self._scale_items or self._scale_items[0].scene() is None):
            self._scale_items = []
            for side in ScaleHandle.CURSORS:
                h = ScaleHandle(self, side)
                h.setCursor(ScaleHandle.CURSORS[side])
                self.scene().addItem(h)
                self._scale_items.append(h)
            self._scale_box = _NoHitPolygon()
            pen = QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._scale_box.setPen(pen)
            self._scale_box.setZValue(999)
            self._scale_box.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene().addItem(self._scale_box)
        if not show:
            for h in self._scale_items + ([self._scale_box] if self._scale_box is not None else []):
                if h.scene() is not None:
                    h.setVisible(False)
            return
        x0, y0, x1, y1 = self._screen_box([q for it in sel for q in it.points])
        inv = self.viewportTransform().inverted()[0]
        off = 10.0                                  # screen px outside the box: clear of the point handles
        for h in self._scale_items:
            sx, sy = h.side
            px = (x0 + x1) / 2 + sx * ((x1 - x0) / 2 + off)
            py = (y0 + y1) / 2 + sy * ((y1 - y0) / 2 + off)
            h.setPos(inv.map(QPointF(px, py)))
            h.setVisible(True)
        if self._scale_box is not None:
            corners = [(x0 - off, y0 - off), (x1 + off, y0 - off), (x1 + off, y1 + off), (x0 - off, y1 + off)]
            self._scale_box.setPolygon(QtGui.QPolygonF([inv.map(QPointF(x, y)) for x, y in corners]))
            self._scale_box.setVisible(True)

    def begin_scale(self, handle: ScaleHandle, pos: QPointF | None = None) -> None:
        sel = self.selection()
        if not sel:
            return
        t = QtGui.QTransform(self.viewportTransform())
        box = self._screen_box([q for it in sel for q in it.points])
        g = t.map(pos if pos is not None else handle.pos())
        x0, y0, x1, y1 = box
        hx, hy = handle.side
        # where the handle was grabbed relative to the box edge it moves (it sits outside the box): no jump on grab
        grab = (g.x() - (x1 if hx > 0 else x0) if hx else 0.0, g.y() - (y1 if hy > 0 else y0) if hy else 0.0)
        self._scl = {"items": sel, "orig": [list(it.points) for it in sel], "box": box, "side": handle.side, "t": t,
                     "grab": grab, "sx": 1.0, "sy": 1.0}

    def scale_to(self, pos: QPointF, keep_ratio: bool = False, about_middle: bool = False) -> None:
        r = self._scl
        if r is None:
            return
        t = r["t"]
        p = t.map(pos) - QPointF(*r.get("grab", (0.0, 0.0)))
        x0, y0, x1, y1 = r["box"]
        hx, hy = r["side"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        def factor(h, q, lo, hi, c):
            if h == 0:
                return 1.0, c
            edge, anchor = (hi, lo) if h > 0 else (lo, hi)
            if about_middle:
                anchor = c
            span = edge - anchor
            if abs(span) < 1e-9:
                return 1.0, anchor
            return max((q - anchor) / span, 0.02), anchor      # never flipped through the anchor

        sx, ax = factor(hx, p.x(), x0, x1, cx)
        sy, ay = factor(hy, p.y(), y0, y1, cy)
        if keep_ratio and hx and hy:
            sx = sy = max(sx, sy)
        r["sx"], r["sy"] = sx, sy
        inv = t.inverted()[0]
        for item, orig in zip(r["items"], r["orig"]):
            pts = []
            for x, y in orig:
                q = t.map(QPointF(x, y))
                q2 = inv.map(QPointF(ax + (q.x() - ax) * sx, ay + (q.y() - ay) * sy))
                pts.append((q2.x(), q2.y()))
            item.points = pts
            for h, (x, y) in zip(item.handles, item.points):
                h.setPos(x, y)
            item.setPolygon(QtGui.QPolygonF([QPointF(x, y) for x, y in item.points]))
        self.message.emit(f"Stretched {sx * 100:.0f} % across × {sy * 100:.0f} % up–down"
                          + ("" if keep_ratio or not (hx and hy) else "  (Shift: keep proportions)")
                          + ("" if about_middle else "  (Alt: about the middle)"))

    def end_scale(self) -> None:
        r, self._scl = self._scl, None
        if r is None:
            return
        if abs(r["sx"] - 1.0) > 1e-9 or abs(r["sy"] - 1.0) > 1e-9:
            self._commit_all(r["items"])
        self.place_rotate_handle()

    def _commit_all(self, items) -> None:
        """Each changed outline committed (``polygonEdited`` per outline); the group stays chosen."""
        keys = [it.key for it in items]
        group = len(items) > 1
        for it in items:
            it.commit()
        if group:
            self.set_multi([k for k in keys if k in self.polys])
            self.place_rotate_handle()

    def scale_active(self, sx: float, sy: float | None = None) -> bool:
        """Stretch the selected outline by ``sx`` across and ``sy`` up–down (as seen on screen; ``sy``
        defaults to ``sx``) about its middle, and commit it."""
        sel = self.selection()
        if not sel:
            return False
        sy = sx if sy is None else sy
        t = QtGui.QTransform(self.viewportTransform())
        x0, y0, x1, y1 = self._screen_box([q for it in sel for q in it.points])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inv = t.inverted()[0]
        for item in sel:
            pts = []
            for x, y in item.points:
                q = t.map(QPointF(x, y))
                q2 = inv.map(QPointF(cx + (q.x() - cx) * sx, cy + (q.y() - cy) * sy))
                pts.append((q2.x(), q2.y()))
            item.points = pts
            item._refresh()
            if len(sel) == 1:
                item.set_active(True)
        self._commit_all(sel)
        self.place_rotate_handle()
        return True

    def begin_rotate(self, pos: QPointF) -> None:
        sel = self.selection()
        if not sel:
            return
        tot, sx_, sy_ = 0.0, 0.0, 0.0                 # turn about the area centroid (of all the chosen outlines together)
        for item in sel:
            pts = np.asarray(item.points, float)
            x, y = pts[:, 0], pts[:, 1]
            a = x * np.roll(y, -1) - np.roll(x, -1) * y
            A = a.sum() / 2
            if abs(A) > 1e-12:
                tot += abs(A)
                sx_ += abs(A) * float(((x + np.roll(x, -1)) * a).sum() / (6 * A))
                sy_ += abs(A) * float(((y + np.roll(y, -1)) * a).sum() / (6 * A))
        if tot > 1e-12:
            c = (sx_ / tot, sy_ / tot)
        else:
            allp = np.asarray([q for it in sel for q in it.points], float)
            c = (float(allp[:, 0].mean()), float(allp[:, 1].mean()))
        self._rot = {"items": sel, "orig": [list(it.points) for it in sel], "c": c,
                     "a0": math.atan2(pos.y() - c[1], pos.x() - c[0]), "deg": 0.0}

    def rotate_to(self, pos: QPointF, snap: bool = False) -> None:
        r = self._rot
        if r is None:
            return
        cx, cy = r["c"]
        deg = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx) - r["a0"])
        deg = math.remainder(deg, 360.0)
        if snap:
            deg = round(deg / 15.0) * 15.0
        r["deg"] = deg
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        for item, orig in zip(r["items"], r["orig"]):
            item.points = [(cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)) for x, y in orig]
            for h, (x, y) in zip(item.handles, item.points):
                h.setPos(x, y)
            item.setPolygon(QtGui.QPolygonF([QPointF(x, y) for x, y in item.points]))
        if self._rot_handle is not None:
            self._rot_handle.setPos(pos)
            self._rot_line.setLine(QtCore.QLineF(QPointF(cx, cy), pos))
        self.message.emit(f"Rotated {deg:+.1f}°" + ("" if snap else "  (Shift: 15° steps)"))

    def end_rotate(self) -> None:
        r, self._rot = self._rot, None
        if r is None:
            return
        if abs(r["deg"]) > 1e-9:
            self._commit_all(r["items"])
        self.place_rotate_handle()

    def rotate_active(self, deg: float) -> bool:
        """Turn the selected outline (or the outlines chosen together) by ``deg`` (counter-clockwise) about the middle and
        commit it."""
        if not self.selection():
            return False
        self.begin_rotate(QPointF(0, 0))
        cx, cy = self._rot["c"]
        self._rot["a0"] = 0.0
        t = math.radians(deg)
        self.rotate_to(QPointF(cx + math.cos(t), cy + math.sin(t)))
        self.end_rotate()
        return True

    def clear_overlays(self) -> None:
        for it in self.overlays:
            if it.scene():
                self.scene().removeItem(it)
        self.overlays = []

    def add_overlay_polys(self, polys: list, color: str, fill_alpha: float = 0.45, z: float = 80.0, width: float = 1.0,
                          dash: bool = False) -> None:
        pen = QtGui.QPen(QtGui.QColor(mix(color, "#000000", 0.35)), width)
        pen.setCosmetic(True)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        fill = QtGui.QColor(color)
        fill.setAlphaF(fill_alpha)
        for poly in polys:
            if len(poly) < 3:
                continue
            it = self.scene().addPolygon(QtGui.QPolygonF([QPointF(x, y) for x, y in poly]), pen, QtGui.QBrush(fill))
            it.setZValue(z)
            self.overlays.append(it)

    def add_markers(self, points: list, color: str = "#d05a5a", radius_px: float = 9.0, z: float = 200.0,
                    keys: list[str] | None = None) -> None:
        """Circles of a fixed size on screen; with ``keys`` (one per point) they are clickable (``markerClicked``)."""
        for n, (x, y) in enumerate(points):
            if keys is not None:
                it = IssueMarker(keys[n], radius_px)
            else:
                it = QtWidgets.QGraphicsEllipseItem(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px)
            it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            it.setPen(pen)
            it.setPos(x, y)
            it.setZValue(z)
            self.scene().addItem(it)
            self.overlays.append(it)

    def add_label(self, text: str, x: float, y: float, color: str | None = None, z: float = 210.0, small: bool = False) -> None:
        it = QtWidgets.QGraphicsSimpleTextItem(text)
        if small:
            f = it.font()
            f.setPointSizeF(max(6.0, f.pointSizeF() * 0.8))
            it.setFont(f)
        it.setBrush(QtGui.QColor(color or PALETTE["text"]))
        it.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        it.setPos(x, y)
        it.setZValue(z)
        self.scene().addItem(it)
        self.overlays.append(it)

    # ------------------------------------------------------------------ view
    def fit(self) -> None:
        rect = QtCore.QRectF()
        for it in self.scene().items():
            if it.isVisible() and not (it.flags() & QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations):
                rect = rect.united(it.sceneBoundingRect())
        if rect.isEmpty():
            return
        m = max(rect.width(), rect.height()) * 0.04
        self.fitInView(rect.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        self.place_rotate_handle()
        self._zoomed = False

    def fit_rect(self, rect: QtCore.QRectF) -> None:
        if rect.isEmpty():
            return
        m = max(rect.width(), rect.height()) * 0.03
        self.fitInView(rect.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        self.place_rotate_handle()

    def zoom_by(self, f: float) -> None:
        self._zoomed = True
        self.scale(f, f)
        self.place_rotate_handle()

    def scrollContentsBy(self, dx, dy):  # noqa: N802 - Qt API
        super().scrollContentsBy(dx, dy)
        self.place_rotate_handle()

    def center_on(self, x: float, y: float, span: float | None = None) -> None:
        if span:
            self.fitInView(QtCore.QRectF(x - span / 2, y - span / 2, span, span), Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(x, y)

    def set_tool(self, tool: str) -> None:
        if tool != self.tool and self.tool == "measure":
            self.clear_ruler()
        self._end_band()
        self.tool = tool
        self._cancel_draft()
        cursor = {"measure": Qt.CursorShape.CrossCursor, "pan": Qt.CursorShape.OpenHandCursor, "wand": Qt.CursorShape.CrossCursor, "draw": Qt.CursorShape.CrossCursor,
                  "move": Qt.CursorShape.SizeAllCursor, "pick": Qt.CursorShape.CrossCursor,
                  "addpoint": Qt.CursorShape.CrossCursor, "select": Qt.CursorShape.CrossCursor}.get(tool, Qt.CursorShape.ArrowCursor)
        self.viewport().setCursor(cursor)

    def drawForeground(self, painter, rect):  # noqa: N802 - Qt API
        if not self.scene().items():
            painter.save()
            painter.resetTransform()
            painter.setPen(QtGui.QColor(PALETTE["faint"]))
            f = painter.font()
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            painter.restore()
        self._draw_scale_bar(painter)

    def _draw_scale_bar(self, painter) -> None:
        t = self.transform()
        px_per_um = math.hypot(t.m11(), t.m12())
        if px_per_um <= 0 or not self.scene().items():
            return
        target = self.viewport().width() / 6 / px_per_um
        exp = 10 ** math.floor(math.log10(max(target, 1e-9)))
        length = max((m * exp for m in (1, 2, 5) if m * exp <= target), default=exp)
        w = length * px_per_um
        painter.save()
        painter.resetTransform()
        x1 = self.viewport().width() - 18
        y = self.viewport().height() - 18
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["text"]), 2))
        painter.drawLine(QPointF(x1 - w, y), QPointF(x1, y))
        painter.drawText(QtCore.QRectF(x1 - w - 40, y - 20, w + 40, 16), Qt.AlignmentFlag.AlignRight, f"{length:g} {self.unit}")
        painter.restore()

    # ------------------------------------------------------------------ measure tool
    def clear_ruler(self) -> None:
        for it in self._ruler_items:
            if it.scene():
                self.scene().removeItem(it)
        self._ruler_items, self._ruler_start, self._ruler_done = [], None, False

    def _length_text(self, a: QPointF, b: QPointF) -> str:
        d = math.hypot(b.x() - a.x(), b.y() - a.y())
        if self.unit == "µm":
            return f"{d:.2f} µm" if d < 1000 else f"{d / 1000:.4f} mm"
        return f"{d:.1f} {self.unit}"

    def _draw_ruler(self, a: QPointF, b: QPointF) -> None:
        if not self._ruler_items:
            pen = QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 2)
            pen.setCosmetic(True)
            line = self.scene().addLine(QtCore.QLineF(a, b), pen)
            line.setZValue(600)
            dots = []
            for _ in range(2):
                d = QtWidgets.QGraphicsEllipseItem(-4, -4, 8, 8)
                d.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                d.setPen(QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.5))
                d.setBrush(QtGui.QColor(PALETTE["surface"]))
                d.setZValue(610)
                self.scene().addItem(d)
                dots.append(d)
            lab = RulerLabel()
            self.scene().addItem(lab)
            self._ruler_items = [line, *dots, lab]
        line, d0, d1, lab = self._ruler_items
        line.setLine(QtCore.QLineF(a, b))
        d0.setPos(a)
        d1.setPos(b)
        lab.setPos(b)
        lab.set_text(self._length_text(a, b))

    def ruler_length(self) -> float | None:
        """Length of the finished measurement on the canvas, or None."""
        if not self._ruler_done or not self._ruler_items:
            return None
        return self._ruler_items[0].line().length()

    def _ruler_click(self, pos: QPointF) -> None:
        if self._ruler_start is None or self._ruler_done:
            self.clear_ruler()
            self._ruler_start = pos
            self._draw_ruler(pos, pos)
            return
        self._draw_ruler(self._ruler_start, pos)
        self._ruler_done = True
        self.measured.emit(math.hypot(pos.x() - self._ruler_start.x(), pos.y() - self._ruler_start.y()))

    # ------------------------------------------------------------------ draft polygon (draw tool)
    def _cancel_draft(self) -> None:
        self._draft = []
        if self._draft_item is not None and self._draft_item.scene():
            self.scene().removeItem(self._draft_item)
        self._draft_item = None

    def _update_draft(self, cursor: QPointF | None = None) -> None:
        if not self._draft:
            return
        path = QtGui.QPainterPath(QPointF(*self._draft[0]))
        for x, y in self._draft[1:]:
            path.lineTo(x, y)
        if cursor is not None:
            path.lineTo(cursor)
        if self._draft_item is None:
            pen = QtGui.QPen(QtGui.QColor(PALETTE["accent"]), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._draft_item = self.scene().addPath(path, pen)
            self._draft_item.setZValue(500)
        else:
            self._draft_item.setPath(path)

    def finish_draft(self) -> None:
        if len(self._draft) >= 3:
            pts = list(self._draft)
            self._cancel_draft()
            self.polygonDrawn.emit(pts)
        else:
            self._cancel_draft()

    # ------------------------------------------------------------------ events
    #: A twist turns the view only once the fingers have turned this far (degrees) in one gesture; from
    #: then on until the fingers lift, two-finger scrolling does not move the view (as on an iPhone).
    TWIST_START_DEG = 3.0
    TWIST_LOCK_AFTER_S = 0.35

    def _pan_locked(self) -> bool:
        import time
        return getattr(self, "_twisting", False) or time.monotonic() < getattr(self, "_twist_lock_until", 0.0)

    def wheelEvent(self, e):  # noqa: N802 - Qt API
        trackpad = not e.pixelDelta().isNull() or e.phase() != Qt.ScrollPhase.NoScrollPhase
        if self.trackpad_gestures and trackpad and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if self._pan_locked():
                if e.phase() == Qt.ScrollPhase.ScrollBegin and not getattr(self, "_twisting", False):
                    self._twist_lock_until = 0.0          # a new scroll after the twist ended: pan again
                else:
                    return                                # the fingers are twisting: no shift
            d = e.pixelDelta() if not e.pixelDelta().isNull() else e.angleDelta() / 8
            self._zoomed = True                     # two-finger scroll moves the view
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        step = e.angleDelta().y() / 120.0 if e.angleDelta().y() else e.pixelDelta().y() / 60.0
        self.zoom_by(1.2 ** step)

    def viewportEvent(self, e):  # noqa: N802 - Qt API
        if not getattr(self, "trackpad_gestures", False):        # (also called while the view is being built)
            return super().viewportEvent(e)
        if e.type() == QtCore.QEvent.Type.NativeGesture:
            import time
            g = e.gestureType()
            if g == Qt.NativeGestureType.BeginNativeGesture:
                self._twist_sum, self._twisting = 0.0, False
                return True
            if g == Qt.NativeGestureType.EndNativeGesture:
                if getattr(self, "_twisting", False):     # momentum scroll after a twist must not shift the view
                    self._twist_lock_until = time.monotonic() + self.TWIST_LOCK_AFTER_S
                self._twist_sum, self._twisting = 0.0, False
                return True
            if g == Qt.NativeGestureType.ZoomNativeGesture:       # pinch: fingers apart / together
                self.zoom_by(max(0.2, 1.0 + e.value()))
                return True
            if g == Qt.NativeGestureType.RotateNativeGesture and not self.allow_twist:
                return True                                   # (pop-up drawing windows: the view never turns)
            if g == Qt.NativeGestureType.RotateNativeGesture:     # twist: the view turns with the fingers
                if getattr(self, "_twisting", False):
                    self.set_view_rotation(self.view_angle - e.value())
                else:
                    self._twist_sum = getattr(self, "_twist_sum", 0.0) + e.value()
                    if abs(self._twist_sum) >= self.TWIST_START_DEG:   # a twist, not a wobble while scrolling
                        self._twisting = True
                        self.set_view_rotation(self.view_angle - self._twist_sum)
                return True
        return super().viewportEvent(e)

    def _on_point(self, e) -> bool:
        """Whether a border point's handle is under the mouse (also when a stretch handle overlaps it)."""
        return any(isinstance(i, Handle) for i in self.items(e.position().toPoint()))

    def mousePressEvent(self, e):  # noqa: N802 - Qt API
        pos = self.mapToScene(e.position().toPoint())
        hit = self.itemAt(e.position().toPoint()) if e.button() == Qt.MouseButton.LeftButton else None
        if isinstance(hit, RotateHandle):
            self.begin_rotate(pos)                   # the round handle above the outline: turn it
            return
        if isinstance(hit, ScaleHandle) and not self._on_point(e):
            self.begin_scale(hit, pos)               # a square handle on the outline's box (a border point under it wins)
            return
        marker = next((i for i in self.items(e.position().toPoint()) if isinstance(i, IssueMarker)), None) \
            if e.button() == Qt.MouseButton.LeftButton and self.tool in ("pointer", "pan") else None
        if marker is not None:                       # a red circle: what error it is (the window's pop-up)
            self.markerClicked.emit(marker.key, marker.pos().x(), marker.pos().y())
            return
        if e.button() == Qt.MouseButton.MiddleButton or (self.tool == "pan" and e.button() == Qt.MouseButton.LeftButton):
            self._pan = e.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            if self.tool == "draw":
                self._cancel_draft()
            return super().mousePressEvent(e)
        if self.tool == "pick":
            self.pointPicked.emit(pos.x(), pos.y())
            return
        if self.tool == "measure":
            self._ruler_click(pos)
            return
        if self.tool == "wand":
            key = self._image_under(pos)
            if key is not None:
                px, py = self.images[key].scene_to_px(pos)
                self.wandClicked.emit(key, px, py)
            return
        if self.tool == "draw":
            self._draft.append((pos.x(), pos.y()))
            self._update_draft()
            return
        if self.tool == "move" and self.active_image in self.images:
            self._move_from = pos
            return
        on_handle = self._on_point(e)
        if self.tool == "addpoint" and not on_handle:
            item = self.polys.get(self.active_poly) if self.active_poly else None
            if item is None:
                item = self._edge_hit(pos)
                if item is not None:
                    self.set_active_polygon(item.key)
                    self.polygonSelected.emit(item.key)
            if item is None or not item.editable:
                self.message.emit("No border to add a point to.")
                return
            d, _, _ = item.nearest_edge(pos)
            if d > self._near_px():
                self.message.emit("Click on the border line to add a point there.")
                return
            item.insert_vertex_near(pos, commit=False, on_edge=True)
            item.begin_drag(pos)                     # keep the button down to drag the new point into place
            item._drag_moved = True                  # the release commits the new point even without a drag
            self._add_drag = item
            return
        if self.tool == "select" and not on_handle:
            self._band_from = e.position().toPoint()
            self._band_add = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                                   | Qt.KeyboardModifier.MetaModifier))
            if self._band is None:
                self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.viewport())
            self._band.setGeometry(QtCore.QRect(self._band_from, QtCore.QSize()))
            self._band.show()
            return
        if self.tool == "pointer" and not on_handle:
            item = self._edge_hit(pos)
            shift = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                          | Qt.KeyboardModifier.MetaModifier))
            if item is not None and shift and item.editable:
                # Shift / Ctrl / ⌘-click on a line (user, 2026-11-22, as in PowerPoint): add it to the outlines chosen
                # together, or take it out; dragging one of them then moves all, the handles turn / stretch all
                keys = set(self.multi) or ({self.active_poly} if self.active_poly in self.polys else set())
                keys ^= {item.key}
                self.set_multi(keys)
                if item.key in keys:
                    self.set_active_polygon(item.key)
                    self.polygonSelected.emit(item.key)
                elif keys:
                    self.set_active_polygon(sorted(keys)[0])
                else:
                    self.set_active_polygon(None)
                self.outlinesSelected.emit(sorted(self.multi))
                self.place_rotate_handle()
                if self.drag_outlines and item.key in self.multi and len(self.multi) > 1:
                    self._group_move = [self.polys[k] for k in self.multi if k in self.polys]
                    for it in self._group_move:
                        it.begin_move(pos)
                    self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            if item is not None and not self.drag_outlines:   # on an outline's line: select it
                if self.active_poly != item.key:
                    self.set_active_polygon(item.key)
                    self.polygonSelected.emit(item.key)
            elif item is not None and item.key in self.multi and len(self.multi) > 1:   # one of a box selection: all move
                self._group_move = [self.polys[k] for k in self.multi if k in self.polys]
                for it in self._group_move:
                    it.begin_move(pos)
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            elif item is not None:                   # on an outline's line: select it, drag moves it whole
                self.set_multi([item.key])
                if self.active_poly != item.key:
                    self.set_active_polygon(item.key)
                self.polygonSelected.emit(item.key)
                item.begin_move(pos)
                self._poly_move = item
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            if self.box_select_outlines:             # on nothing: a box chooses every outline inside it
                self._band_from = e.position().toPoint()
                self._band_outlines = True
                self._band_add = bool(e.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
                                                       | Qt.KeyboardModifier.MetaModifier))
                if self._band is None:
                    self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.viewport())
                self._band.setGeometry(QtCore.QRect(self._band_from, QtCore.QSize()))
                self._band.show()
                return
        if self.tool == "pointer" and self.drag_pans and not on_handle:
            # drag anywhere but on a point handle to move the view around (a plain click drops the point selection)
            self._pan = e.position().toPoint()
            self._pan_start = self._pan
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):  # noqa: N802 - Qt API
        pos = self.mapToScene(e.position().toPoint())
        if self.tool == "measure":
            self.clear_ruler()                 # double-click clears the measurement
            return
        if self.tool == "pointer" and self.drag_pans and self.active_poly in self.polys:
            self._pan = None
            self.set_tool(self.tool)
        if self.tool == "draw":
            self.finish_draft()
            return
        if self.tool == "pointer" and self.active_poly in self.polys:
            self.polys[self.active_poly].insert_vertex_near(pos)
            return
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e):  # noqa: N802 - Qt API
        pos = self.mapToScene(e.position().toPoint())
        self.cursorMoved.emit(pos.x(), pos.y())
        if self._rot is not None:
            self.rotate_to(pos, bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            return
        if self._scl is not None:
            self.scale_to(pos, bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                          bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))
            return
        if getattr(self, "_add_drag", None) is not None:
            self._add_drag.drag_to(pos)
            return
        if self._poly_move is not None:
            self._poly_move.move_to(pos)
            return
        if self._group_move is not None:
            for it in self._group_move:
                it.move_to(pos)
            return
        if (self.drag_outlines and self.tool == "pointer" and self._pan is None and not e.buttons()
                and not self._on_point(e)):
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self._edge_hit(pos) is not None else Qt.CursorShape.ArrowCursor)
        if self._band_from is not None and self._band is not None:
            self._band.setGeometry(QtCore.QRect(self._band_from, e.position().toPoint()).normalized())
            return
        if self._pan is not None:
            d = e.position().toPoint() - self._pan
            self._pan = e.position().toPoint()
            self._zoomed = True                            # the user moved the view: stop refitting on resize
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        if self.tool == "draw" and self._draft:
            self._update_draft(pos)
        if self.tool == "measure" and self._ruler_start is not None and not self._ruler_done:
            self._draw_ruler(self._ruler_start, pos)
        if self._move_from is not None and self.active_image in self.images:
            d = pos - self._move_from
            self._move_from = pos
            self.imageMoved.emit(self.active_image, d.x(), d.y())
            return
        super().mouseMoveEvent(e)

    def _end_band(self) -> None:
        self._band_from = None
        if self._band is not None:
            self._band.hide()

    def mouseReleaseEvent(self, e):  # noqa: N802 - Qt API
        if self._rot is not None:
            self.end_rotate()
            return
        if self._scl is not None:
            self.end_scale()
            return
        if self._poly_move is not None:
            item, self._poly_move = self._poly_move, None
            self.set_tool(self.tool)
            item.end_move()
            return
        if self._group_move is not None:
            items, self._group_move = self._group_move, None
            self.set_tool(self.tool)
            keys = [it.key for it in items]
            for it in items:
                it.end_move()                         # each commits (polygonEdited)
            self.set_multi([k for k in keys if k in self.polys])
            return
        if self._band_from is not None and self._band_outlines:
            a, b = self._band_from, e.position().toPoint()
            self._band_outlines = False
            self._end_band()
            if (b - a).manhattanLength() < 3:          # a plain click on nothing: nothing chosen
                if not self._band_add:
                    self.set_multi([])
                    self.set_active_polygon(None)
                    self.polygonSelected.emit("")
                return
            box = self.mapToScene(QtCore.QRect(a, b).normalized())
            inside = [k for k, it in self.polys.items() if it.editable and it.isVisible() and it.points
                      and all(box.containsPoint(QPointF(x, y), Qt.FillRule.OddEvenFill) for x, y in it.points)]
            keys = sorted(set(inside) | (self.multi if self._band_add else set()))
            self.set_multi(keys)
            if len(keys) == 1:
                self.set_active_polygon(keys[0])
                self.polygonSelected.emit(keys[0])
            else:
                self.set_active_polygon(None)
            self.outlinesSelected.emit(keys)
            return
        if getattr(self, "_add_drag", None) is not None:
            item, self._add_drag = self._add_drag, None
            item.end_drag()
            return
        if self._band_from is not None:
            a, b = self._band_from, e.position().toPoint()
            self._end_band()
            item = self.polys.get(self.active_poly) if self.active_poly else None
            if item is None:                          # nothing selected: the one editable outline, if there is one
                editable = [p for p in self.polys.values() if p.editable and p.isVisible()]
                if len(editable) == 1 and (b - a).manhattanLength() >= 3:
                    item = editable[0]
                    self.set_active_polygon(item.key)
                    self.polygonSelected.emit(item.key)
            if item is not None:
                if (b - a).manhattanLength() < 3:       # a plain click on nothing: drop the selection
                    if not self._band_add:
                        item.select_only([])
                else:
                    item.select_in_rect(QtCore.QRectF(self.mapToScene(a), self.mapToScene(b)), add=self._band_add)
            return
        if self._pan is not None:
            start = getattr(self, "_pan_start", None)
            if start is not None and (e.position().toPoint() - start).manhattanLength() < 3 and self.active_poly in self.polys:
                # a plain click away from every outline: nothing stays selected
                if self._edge_hit(self.mapToScene(start)) is None:
                    self.polys[self.active_poly].select_only([])
                    self.set_multi([])                 # the outlines chosen together too
                    self.set_active_polygon(None)
                    self.polygonSelected.emit("")
            self._pan, self._pan_start = None, None
            self.set_tool(self.tool)
            return
        self._move_from = None
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):  # noqa: N802 - Qt API
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.active_poly in self.polys:
            if self.delete_selected_vertices():
                return
        if e.matches(QtGui.QKeySequence.StandardKey.SelectAll) and self.active_poly in self.polys and self.tool in ("pointer", "select"):
            self.select_all_vertices()
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.tool == "draw":
            self.finish_draft()
            return
        if e.key() == Qt.Key.Key_Escape:
            self._cancel_draft()
            self.clear_ruler()
            if self.active_poly in self.polys:
                self.polys[self.active_poly].select_only([])
            return
        if e.key() == Qt.Key.Key_H and not e.isAutoRepeat():
            self._tool_before_pan = self.tool
            self.set_tool("pan")
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):  # noqa: N802 - Qt API
        if e.key() == Qt.Key.Key_H and not e.isAutoRepeat() and getattr(self, "_tool_before_pan", None):
            self.set_tool(self._tool_before_pan)
            self._tool_before_pan = None
            return
        super().keyReleaseEvent(e)

    def _image_under(self, pos: QPointF) -> str | None:
        if self.active_image in self.images and self.images[self.active_image].isVisible():
            item = self.images[self.active_image]
            if item.contains(item.mapFromScene(pos)):
                return self.active_image
        for key, item in sorted(self.images.items(), key=lambda kv: -kv[1].zValue()):
            if item.isVisible() and item.contains(item.mapFromScene(pos)):
                return key
        return None

    def resizeEvent(self, e):  # noqa: N802 - Qt API
        super().resizeEvent(e)
        if self.fit_on_resize and not self._zoomed:
            self.fit()
        self.viewport().update()
