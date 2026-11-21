"""The device preview: the layer stack drawn as extruded footprints with QPainter.

Drag to rotate (any angle, also from below), two-finger / right / middle drag to move, wheel or pinch to zoom, double-click resets; the Separate Layers slider moves the layers apart in z only. An x–y scale bar lies on the
chip surface: it turns and tilts with the device and its length is re-chosen (1, 2 or 5 × 10ⁿ µm)
as you zoom. Heights follow the real
thicknesses on a log scale (see scene3d.display_height); layers with no thickness yet are drawn
hatched. Painter's algorithm: layers bottom to top, each prism's visible sides by depth, then
its top face.
"""

from __future__ import annotations

import math

from ..scene3d import Slab
from .style import PALETTE, QtCore, QtGui, QtWidgets, Signal, mix

Qt = QtCore.Qt
QPointF = QtCore.QPointF


class Stack3DView(QtWidgets.QWidget):
    layerClicked = Signal(str)
    hiddenChanged = Signal(object)          # set of hidden layer uids

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slabs: list[Slab] = []
        self.az = math.radians(-35.0)
        self.el = math.radians(32.0)
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)          # screen offset of the stack (px)
        self._pan_from: QtCore.QPointF | None = None
        self.explode = 0.0
        self.z_scale = 1.0
        self.highlight: str | None = None
        self.hidden: set[str] = set()       # layers not drawn (they keep their place in height)
        self._swatch_rects: list[tuple[QtCore.QRectF, str]] = []
        self.message = "No device"
        self._drag: QtCore.QPointF | None = None
        self._legend_rects: list[tuple[QtCore.QRectF, str]] = []
        self.scalebar_um: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(260)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)   # two-finger / right drag pans
        self.grabGesture(Qt.GestureType.PinchGesture)

    # ------------------------------------------------------------------ data
    def set_slabs(self, slabs: list[Slab], message: str = "") -> None:
        self.slabs = slabs
        self.message = message
        self.update()

    def set_highlight(self, uid: str | None) -> None:
        self.highlight = uid
        self.update()

    def set_hidden(self, uids) -> None:
        """Show only some layers: ``uids`` are hidden (empty = show all)."""
        new = set(uids or ())
        changed = new != self.hidden
        self.hidden = new
        self.update()
        if changed:
            self.hiddenChanged.emit(set(self.hidden))     # (Flake layout hides the same layers)

    def show_only(self, uids) -> None:
        keep = set(uids)
        self.set_hidden({s.uid for s in self.slabs if s.uid not in keep} if keep else set())

    def toggle_layer(self, uid: str) -> None:
        self.hidden ^= {uid}
        self.update()
        self.hiddenChanged.emit(set(self.hidden))

    def set_explode(self, value: float) -> None:
        self.explode = value
        self.update()

    def reset_view(self) -> None:
        self.az, self.el, self.zoom = math.radians(-35.0), math.radians(32.0), 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)
        self.update()

    # ------------------------------------------------------------------ projection
    def _extent(self) -> float:
        shown = [s for s in self.slabs if s.uid not in self.hidden] or self.slabs
        xs = [abs(c) for s in shown for p in s.polys for pt in p for c in pt]
        return max(xs + [1.0])

    def _project(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        ca, sa = math.cos(self.az), math.sin(self.az)
        xr, yr = x * ca - y * sa, x * sa + y * ca
        ce, se = math.cos(self.el), math.sin(self.el)
        sx = xr
        sy = -(yr * se + z * ce)
        depth = yr * ce - z * se
        return sx, sy, depth

    def _layout(self):
        legend_w = 230 if self.width() > 640 else 0
        area = QtCore.QRectF(12, 12, self.width() - legend_w - 24, self.height() - 24)
        ext = self._extent()
        # the height scale ignores the separation: separating the layers moves them apart, it never thins them
        total_h = (self.slabs[-1].z0 + self.slabs[-1].height) if self.slabs else 1.0
        scale = 0.42 * min(area.width(), area.height() * 1.4) / ext * self.zoom
        zs = self._zscale(ext, total_h)
        return area, scale, zs, legend_w

    def _zscale(self, ext: float, total_h: float) -> float:
        # the stack's display height is about a third of its lateral extent
        return (ext * 0.55) / max(total_h, 1e-6) * self.z_scale

    def _poly_depth(self, poly, z: float, zs: float, mid_z: float) -> float:
        """View depth of a footprint's centre at height ``z`` (larger = farther from the viewer)."""
        if not poly:
            return 0.0
        cx = sum(x for x, _ in poly) / len(poly)
        cy = sum(y for _, y in poly) / len(poly)
        return self._project(cx, cy, z * zs - mid_z)[2]

    def _heights(self) -> list[tuple[float, float]]:
        out = []
        n = len(self.slabs)
        base = (self.slabs[-1].z0 + self.slabs[-1].height) if self.slabs else 0.0
        gap = (self.explode / 4.0) * 1.5 * base / max(n - 1, 1)     # at most 1.5 × the stack height in all
        for i, s in enumerate(self.slabs):
            z0 = s.z0 + i * gap
            out.append((z0, z0 + s.height))
        return out

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event):  # noqa: N802 - Qt API
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(PALETTE["surface"]))
        if not self.slabs:
            p.setPen(QtGui.QColor(PALETTE["faint"]))
            f = p.font()
            f.setBold(True)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message or "No device")
            return
        area, scale, zs, legend_w = self._layout()
        heights = self._heights()
        cx, cy = area.center().x() + self.pan.x(), area.center().y() + area.height() * 0.08 + self.pan.y()
        # centre the stack vertically
        mid_z = heights[-1][1] * 0.5 * zs

        def P(x, y, z):
            sx, sy, d = self._project(x, y, z * zs - mid_z)
            return QPointF(cx + sx * scale, cy + sy * scale), d

        light = (0.45, -0.75, 0.5)
        se_, ce_ = math.sin(self.el), math.cos(self.el)
        from_below = se_ < 0
        order = list(zip(self.slabs, heights))
        if from_below:
            order.reverse()                       # seen from underneath: the top layer is the farthest
        for slab, (z0, z1) in order:
            if slab.uid in self.hidden:
                continue
            base = QtGui.QColor(slab.color)
            hl = self.highlight == slab.uid
            alpha = 0.55 if slab.translucent else 0.96
            if self.highlight and not hl:
                alpha *= 0.45
            # several shapes in one layer (electrode fingers, leads): draw the farthest first, so a
            # nearer prism always paints over the ones behind it at any orientation
            zmid = (z0 + z1) / 2
            for poly in sorted(slab.polys, key=lambda pl: -self._poly_depth(pl, zmid, zs, mid_z)):
                pts = list(poly)
                if len(pts) < 3:
                    continue
                if _signed_area(pts) < 0:
                    pts.reverse()
                faces = []
                for i in range(len(pts)):
                    (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
                    nx, ny = (y1 - y0), -(x1 - x0)
                    ca, sa = math.cos(self.az), math.sin(self.az)
                    nyr = nx * sa + ny * ca
                    if nyr * ce_ >= 0:
                        continue  # faces away from the viewer (or edge-on, seen straight from above/below)
                    q = [P(x0, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y0, z1)]
                    depth = sum(d for _, d in q) / 4
                    shade = 0.62 + 0.25 * max(0.0, (nx * light[0] + ny * light[1]) / (math.hypot(nx, ny) or 1))
                    faces.append((depth, [pt for pt, _ in q], shade))
                faces.sort(key=lambda f: -f[0])
                for _, quad, shade in faces:
                    c = QtGui.QColor(base)
                    c = c.darker(int(100 / shade))
                    c.setAlphaF(min(1.0, alpha + 0.1))
                    p.setPen(QtGui.QPen(QtGui.QColor(mix(slab.color, "#000000", 0.35)), 0.6))
                    p.setBrush(c)
                    p.drawPolygon(QtGui.QPolygonF(quad))
                top = QtGui.QPolygonF([P(x, y, z0 if from_below else z1)[0] for x, y in pts])
                c = QtGui.QColor(base)
                c.setAlphaF(alpha)
                pen = QtGui.QPen(QtGui.QColor(PALETTE["accent"] if hl else mix(slab.color, "#000000", 0.4)), 1.6 if hl else 0.7)
                p.setPen(pen)
                p.setBrush(c)
                p.drawPolygon(top)
                if not slab.known:
                    brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 60), Qt.BrushStyle.BDiagPattern)
                    p.setBrush(brush)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.drawPolygon(top)
        self._paint_scalebar(p, P, scale, heights)
        self._paint_legend(p, legend_w)
        self._paint_axes(p)

    @staticmethod
    def nice_length(target_um: float) -> float:
        """The largest 1, 2 or 5 × 10ⁿ µm not above ``target_um``."""
        if target_um <= 0:
            return 1.0
        exp = math.floor(math.log10(target_um))
        for m in (5, 2, 1):
            if m * 10 ** exp <= target_um:
                return m * 10 ** exp
        return 10 ** exp

    def _paint_scalebar(self, p: QtGui.QPainter, P, scale: float, heights) -> None:
        """An L-shaped x–y scale bar lying on the chip surface, tucked into the bottom-right corner of
        the view (away from the sample). Its arms run along the device's x and y axes (so it turns and
        tilts with the device), and its length is re-chosen as 1, 2 or 5 × 10ⁿ µm (about 110 px) at
        every zoom."""
        if scale <= 0 or not self.slabs:
            return
        z = heights[0][1]
        area, _, zs, _ = self._layout()
        cx, cy = area.center().x() + self.pan.x(), area.center().y() + area.height() * 0.08 + self.pan.y()
        mid_z = heights[-1][1] * 0.5 * zs
        zc = z * zs - mid_z
        ce, se = math.cos(self.el), math.sin(self.el)
        ca, sa = math.cos(self.az), math.sin(self.az)

        def under(pt: QPointF) -> tuple[float, float]:
            """The chip-surface point drawn at screen point ``pt`` (the projection inverted on the plane z)."""
            xr = (pt.x() - cx) / scale
            yr = (-(pt.y() - cy) / scale - zc * ce) / (se if abs(se) > 1e-3 else math.copysign(1e-3, se or 1.0))
            return xr * ca + yr * sa, -xr * sa + yr * ca

        length = self.nice_length(110.0 / scale)
        f = QtGui.QFont(p.font())
        f.setBold(True)
        fm = QtGui.QFontMetrics(f)
        # provisional corner; each arm points left on screen (or up, when it is nearly vertical)
        ox, oy = under(QPointF(self.width() / 2, self.height() / 2))
        o = P(ox, oy, z)[0]
        ends: dict[str, tuple[float, float]] = {}
        pts = [o]
        for axis, (ux, uy) in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
            e = P(ox + ux * length, oy + uy * length, z)[0]
            dx, dy = e.x() - o.x(), e.y() - o.y()
            n = math.hypot(dx, dy) or 1.0
            sign = (-1.0 if dy > 0 else 1.0) if abs(dx) < 0.25 * n else (-1.0 if dx > 0 else 1.0)
            ends[axis] = (sign * ux, sign * uy)
            ex, ey = o.x() + sign * dx, o.y() + sign * dy
            w, h = fm.horizontalAdvance(f"{axis}  {length:g} µm") + 8, fm.height()
            lx, ly = ex + sign * dx / n * (w / 2 + 6), ey + sign * dy / n * (h / 2 + 6)
            pts += [QPointF(ex, ey), QPointF(lx - w / 2, ly - h / 2), QPointF(lx + w / 2, ly + h / 2)]
        # move the whole bar (arms and labels) into the bottom-right corner
        shift = QPointF(self.width() - 14 - max(q.x() for q in pts), self.height() - 34 - max(q.y() for q in pts))   # above the hint line
        ox, oy = under(QPointF(o.x() + shift.x(), o.y() + shift.y()))
        o = P(ox, oy, z)[0]
        color = QtGui.QColor(PALETTE["text"])
        pen = QtGui.QPen(color, 2.2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        tick = length * 0.08
        for axis, (ux, uy) in ends.items():
            e = P(ox + ux * length, oy + uy * length, z)[0]
            p.setPen(pen)
            p.drawLine(o, e)
            vx, vy = -uy, ux   # a tick across the arm's end, in the chip plane
            p.setPen(QtGui.QPen(color, 1.4))
            p.drawLine(P(ox + ux * length + vx * tick, oy + uy * length + vy * tick, z)[0],
                       P(ox + ux * length - vx * tick, oy + uy * length - vy * tick, z)[0])
            # the label just past the arm's end
            d = QPointF(e.x() - o.x(), e.y() - o.y())
            n = math.hypot(d.x(), d.y()) or 1.0
            text = f"{axis}  {length:g} µm"
            w, h = fm.horizontalAdvance(text) + 8, fm.height()
            c = QPointF(e.x() + d.x() / n * (w / 2 + 6), e.y() + d.y() / n * (h / 2 + 6))
            r = QtCore.QRectF(c.x() - w / 2, c.y() - h / 2, w, h)
            bg = QtGui.QColor(PALETTE["surface"])
            bg.setAlphaF(0.8)
            p.fillRect(r, bg)
            p.setFont(f)
            p.setPen(color)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        f.setBold(False)
        p.setFont(f)
        self.scalebar_um = length

    def _paint_legend(self, p: QtGui.QPainter, legend_w: int) -> None:
        self._legend_rects = []
        self._swatch_rects = []
        if not legend_w:
            return
        x0 = self.width() - legend_w
        y = 16
        f = p.font()
        fm = QtGui.QFontMetrics(f)
        p.setPen(QtGui.QColor(PALETTE["accent"]))
        bold = QtGui.QFont(f)
        bold.setBold(True)
        p.setFont(bold)
        p.drawText(QtCore.QRectF(x0, y, legend_w - 12, 18), Qt.AlignmentFlag.AlignLeft, "Stack (top → bottom)")
        p.setFont(f)
        y += 24
        for s in reversed(self.slabs):
            r = QtCore.QRectF(x0, y, legend_w - 12, 20)
            if self.highlight == s.uid:
                p.fillRect(r, QtGui.QColor(PALETTE["hover"]))
            sw = QtCore.QRectF(x0 + 4, y + 4, 12, 12)
            off = s.uid in self.hidden
            c = QtGui.QColor(s.color)
            p.setBrush(Qt.BrushStyle.NoBrush if off else c)       # hollow swatch = hidden
            p.setPen(QtGui.QPen(QtGui.QColor(mix(s.color, "#000000", 0.4)), 0.8))
            p.drawRect(sw)
            self._swatch_rects.append((sw.adjusted(-4, -4, 4, 4), s.uid))
            p.setPen(QtGui.QColor(PALETTE["faint"] if off else PALETTE["text"]))
            t = "?" if s.thickness_nm is None else (f"{s.thickness_nm:.3g} nm")
            name = fm.elidedText(s.name, Qt.TextElideMode.ElideRight, legend_w - 90)
            p.drawText(QtCore.QRectF(x0 + 22, y, legend_w - 90, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
            p.setPen(QtGui.QColor(PALETTE["faint"] if s.thickness_nm is None else PALETTE["text2"]))
            p.drawText(QtCore.QRectF(x0 + legend_w - 80, y, 64, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, t)
            self._legend_rects.append((r, s.uid))
            y += 21

    def _paint_axes(self, p: QtGui.QPainter) -> None:
        o = QPointF(34, self.height() - 30)
        p.setPen(QtGui.QPen(QtGui.QColor(PALETTE["faint"]), 1.2))
        for (dx, dy, dz), name in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")):
            sx, sy, _ = self._project(dx, dy, dz)
            end = QPointF(o.x() + sx * 18, o.y() + sy * 18)
            p.drawLine(o, end)
            p.drawText(end + QPointF(2, 4), name)
        p.drawText(QtCore.QRectF(60, self.height() - 22, max(200, self.width() - 80), 18), Qt.AlignmentFlag.AlignLeft,
                   "Drag to rotate · two-finger drag to move · pinch / wheel to zoom · double-click resets · click a swatch to hide / show a layer")

    # ------------------------------------------------------------------ interaction
    def mousePressEvent(self, e):  # noqa: N802 - Qt API
        if e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from = e.position()           # two-finger click-drag (right button) or middle button: move
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        for r, uid in self._swatch_rects:
            if r.contains(e.position()):
                self.toggle_layer(uid)
                return
        for r, uid in self._legend_rects:
            if r.contains(e.position()):
                self.layerClicked.emit(uid)
                return
        self._drag = e.position()

    def mouseMoveEvent(self, e):  # noqa: N802 - Qt API
        if self._pan_from is not None:
            self.pan += e.position() - self._pan_from
            self._pan_from = e.position()
            self.update()
            return
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            d = e.position() - self._drag
            self._drag = e.position()
            self.az += d.x() * 0.01
            # no limit: any elevation, from above, edge-on, from below, over the top
            self.el = math.remainder(self.el + d.y() * 0.01, 2 * math.pi)
            self.update()
            return
        hit = next((uid for r, uid in self._legend_rects if r.contains(e.position())), None)
        self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.OpenHandCursor)
        if hit != self.highlight and (hit or self.highlight):
            self.highlight = hit
            self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802 - Qt API
        self._drag = None
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, e):  # noqa: N802 - Qt API
        self.reset_view()

    def wheelEvent(self, e):  # noqa: N802 - Qt API
        trackpad = not e.pixelDelta().isNull() or e.phase() != Qt.ScrollPhase.NoScrollPhase
        if trackpad and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # two fingers sliding on a trackpad: move the stack (pinch zooms)
            d = e.pixelDelta() if not e.pixelDelta().isNull() else e.angleDelta() / 8
            self.pan += QtCore.QPointF(d.x(), d.y())
            self.update()
            return
        step = e.angleDelta().y() / 120.0
        self.zoom = max(0.2, min(20.0, self.zoom * (1.15 ** step)))
        self.update()

    def event(self, e):  # noqa: N802 - Qt API
        if e.type() == QtCore.QEvent.Type.NativeGesture and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom = max(0.2, min(20.0, self.zoom * (1.0 + e.value())))     # trackpad pinch
            self.update()
            return True
        if e.type() == QtCore.QEvent.Type.Gesture:
            g = e.gesture(Qt.GestureType.PinchGesture)
            if g is not None:
                self.zoom = max(0.2, min(20.0, self.zoom * g.scaleFactor()))
                self.update()
                return True
        return super().event(e)


def _signed_area(pts) -> float:
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts))) / 2
