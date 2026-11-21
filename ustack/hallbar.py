"""The v1 pattern: one Hall bar with adjustable parameters.

Local frame: the channel runs along +x, centred on the origin; probe pairs sit at
x_i = -L/2 + i·L/(n-1) with arms going to +y (top, "T") and -y (bottom, "B"); the source is the
-x end and the drain the +x end. With ``sd_style == "bent"`` the source and drain turn 90° at the
channel ends: a leg of width ``leg_w`` (flush with the channel end) runs ``arm_l`` to +y for the
source and to -y for the drain, so their contacts land beside the channel like the probes (off a
gate strip that runs the whole length of the device). :meth:`HallBar.placed` returns the same shapes rotated by
``angle_deg`` and moved to ``center`` (µm, chip frame).

The generator never changes a parameter: if a parameter is missing it raises
:class:`~ustack.config.MissingValues`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geometry as geo
from .config import MissingItem, MissingValues
from .model import HALL_BAR_FIELDS, DeviceSpec


@dataclass
class Terminal:
    name: str                  # S, D, V1T, V1B, ...
    kind: str                  # source | drain | probe
    end: tuple[float, float]   # where the arm / channel ends
    direction: tuple[float, float]  # unit vector pointing away from the device
    width: float               # arm (or channel) width at the end

    def moved(self, angle_deg: float, dx: float, dy: float) -> "Terminal":
        (ex, ey), = geo.transform([self.end], angle_deg, dx, dy)
        (ux, uy), = geo.transform([self.direction], angle_deg)
        return Terminal(self.name, self.kind, (ex, ey), (ux, uy), self.width)

    def to_json(self) -> dict:
        return {"name": self.name, "kind": self.kind, "end_um": [round(v, 4) for v in self.end],
                "direction": [round(v, 6) for v in self.direction], "width_um": self.width}


@dataclass
class HallBar:
    W: float
    L: float
    n: int
    arm_w: float
    arm_l: float
    sd_ext: float
    center: tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    corner_radius: float = 0.0
    shapes: dict = field(default_factory=dict)
    sd_style: str = "straight"      # straight | bent ("auto" is resolved by the caller)
    leg_w: float | None = None      # width of a bent S/D leg (default: W)

    @classmethod
    def from_spec(cls, spec: DeviceSpec) -> "HallBar":
        values, missing = {}, []
        for key, (label, unit) in HALL_BAR_FIELDS.items():
            v = spec.hall_bar.get(key)
            if v is None:
                q = spec.hall_bar.params.get(key)
                missing.append(MissingItem(f"spec:hall_bar.{key}", f"Hall bar · {label}", "µm" if unit == "um" else unit,
                                           q.suggested_source if q else "", q.suggested if q else None))
            values[key] = v
        if missing:
            raise MissingValues(missing)
        style = getattr(spec.hall_bar, "sd_style", "auto")
        hb = cls(values["channel_width_um"], values["channel_length_um"], int(values["n_probe_pairs"]),
                 values["arm_width_um"], values["arm_length_um"], values["sd_extension_um"],
                 sd_style="bent" if style == "bent" else "straight")
        # pre-patterned electrodes: the bent S/D are electrode fingers like the probes
        hb.leg_w = values["arm_width_um"] if spec.contact_scheme == "prepatterned_bottom" else values["channel_width_um"]
        return hb

    @property
    def bent(self) -> bool:
        return self.sd_style == "bent"

    @property
    def leg_width(self) -> float:
        return self.leg_w or self.W

    # ------------------------------------------------------------------ local geometry
    @property
    def total_length(self) -> float:
        return self.L + 2 * self.sd_ext

    def probe_x(self) -> list[float]:
        if self.n == 1:
            return [0.0]
        return [-self.L / 2 + i * self.L / (self.n - 1) for i in range(self.n)]

    def local_channel(self) -> geo.Poly:
        lt = self.total_length
        return geo.rect(-lt / 2, -self.W / 2, lt / 2, self.W / 2)

    def local_arms(self) -> list[tuple[str, geo.Poly]]:
        out = []
        h = self.arm_w / 2
        for i, x in enumerate(self.probe_x(), 1):
            out.append((f"V{i}T", geo.rect(x - h, self.W / 2 - 1e-3, x + h, self.W / 2 + self.arm_l)))
            out.append((f"V{i}B", geo.rect(x - h, -self.W / 2 - self.arm_l, x + h, -self.W / 2 + 1e-3)))
        return out

    def leg_x(self) -> tuple[float, float]:
        """Centres (x) of the bent source and drain legs."""
        lt, lw = self.total_length, self.leg_width
        return -lt / 2 + lw / 2, lt / 2 - lw / 2

    def local_legs(self) -> list[tuple[str, geo.Poly]]:
        if not self.bent:
            return []
        lt, lw = self.total_length, self.leg_width
        return [("S", geo.rect(-lt / 2, self.W / 2 - 1e-3, -lt / 2 + lw, self.W / 2 + self.arm_l)),
                ("D", geo.rect(lt / 2 - lw, -self.W / 2 - self.arm_l, lt / 2, -self.W / 2 + 1e-3))]

    def local_terminals(self) -> list[Terminal]:
        lt = self.total_length
        if self.bent:
            xs, xd = self.leg_x()
            out = [Terminal("S", "source", (xs, self.W / 2 + self.arm_l), (0.0, 1.0), self.leg_width),
                   Terminal("D", "drain", (xd, -self.W / 2 - self.arm_l), (0.0, -1.0), self.leg_width)]
        else:
            out = [Terminal("S", "source", (-lt / 2, 0.0), (-1.0, 0.0), self.W),
                   Terminal("D", "drain", (lt / 2, 0.0), (1.0, 0.0), self.W)]
        for i, x in enumerate(self.probe_x(), 1):
            out.append(Terminal(f"V{i}T", "probe", (x, self.W / 2 + self.arm_l), (0.0, 1.0), self.arm_w))
            out.append(Terminal(f"V{i}B", "probe", (x, -self.W / 2 - self.arm_l), (0.0, -1.0), self.arm_w))
        return out

    # ------------------------------------------------------------------ placed geometry (chip frame)
    def _place(self, pts: geo.Poly) -> geo.Poly:
        return geo.transform(pts, self.angle_deg, *self.center)

    def channel(self) -> geo.Poly:
        return self._place(self.local_channel())

    def probe_span(self) -> geo.Poly:
        """The channel between the outermost probe centres (what the gates must cover)."""
        return self._place(geo.rect(-self.L / 2, -self.W / 2, self.L / 2, self.W / 2))

    def arms(self) -> list[tuple[str, geo.Poly]]:
        return [(name, self._place(p)) for name, p in self.local_arms()]

    def legs(self) -> list[tuple[str, geo.Poly]]:
        return [(name, self._place(p)) for name, p in self.local_legs()]

    def body_region(self):
        reg = geo.region(self.channel(), *[p for _, p in self.arms()], *[p for _, p in self.legs()])
        return geo.rounded(reg, self.corner_radius) if self.corner_radius else reg

    def terminals(self) -> list[Terminal]:
        return [t.moved(self.angle_deg, *self.center) for t in self.local_terminals()]

    def terminal_end_box(self, t: Terminal, depth: float) -> geo.Poly:
        """The last ``depth`` µm of the arm (or channel) at terminal ``t`` — where an edge contact lands."""
        ux, uy = t.direction
        vx, vy = -uy, ux
        ex, ey = t.end
        h = t.width / 2
        return [(ex - ux * depth + vx * h, ey - uy * depth + vy * h), (ex + vx * h, ey + vy * h),
                (ex - vx * h, ey - vy * h), (ex - ux * depth - vx * h, ey - uy * depth - vy * h)]

    def footprint(self) -> tuple[float, float]:
        return self.total_length, self.W + 2 * self.arm_l

    def to_json(self) -> dict:
        return {"W_um": self.W, "L_um": self.L, "n_probe_pairs": self.n, "arm_width_um": self.arm_w, "arm_length_um": self.arm_l,
                "sd_extension_um": self.sd_ext, "center_um": [round(c, 4) for c in self.center], "angle_deg": round(self.angle_deg, 4),
                "corner_radius_um": self.corner_radius, "sd_style": self.sd_style, "terminals": [t.to_json() for t in self.terminals()]}


def lead_path(points: list[tuple[float, float]], width: float) -> geo.Poly:
    """A lead of ``width`` along ``points`` (µm) as one polygon (square ends, mitred joints)."""
    import klayout.db as kdb
    path = kdb.DPath([kdb.DPoint(x, y) for x, y in points], width, 0, 0)
    poly = path.polygon()
    return [(p.x, p.y) for p in poly.each_point_hull()]


def escape_point(t: Terminal, distance: float) -> tuple[float, float]:
    return (t.end[0] + t.direction[0] * distance, t.end[1] + t.direction[1] * distance)


def angle_of(p: tuple[float, float], c: tuple[float, float]) -> float:
    return math.atan2(p[1] - c[1], p[0] - c[0])
