"""SYNTHETIC test fixtures — not lab data.

Builds a bond-pad-like GDS whose marker squares sit where the markers of the sample 100x image
``WSe2_device_Batch1_before_electrodes_stack1_100x_BF.jpg`` land under a made-up transform
(0.02 µm/px, 1.5°), repeated at four device sites, with fan-out leads around each site. The
layer numbers used here (10/0 markers, 11/0 leads) are fixture choices, not the lab's.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import klayout.db as kdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from ustack import geometry as geo  # noqa: E402
from ustack.alignment import Similarity  # noqa: E402

TRUE = Similarity(0.02, 1.5, -60.0, 40.0)            # px → µm for the sample image (synthetic)
#: marker squares measured on the sample image (px): (cx, cy, side)
IMAGE_SQUARES = [(1247, 299, 217), (1412, 456, 110), (4713, 365, 110), (4816, 255, 110),
                 (4803, 3666, 110), (4912, 3771, 110), (1384, 3862, 110), (1490, 3752, 110)]
SITES = [(0.0, 0.0), (2000.0, 0.0), (0.0, 1500.0), (2000.0, 1500.0)]
MARKER_LAYER = (10, 0)
LEAD_LAYER = (11, 0)


def site_center_um() -> tuple[float, float]:
    (x, y), = TRUE.apply([(3000, 2000)])
    return float(x), float(y)


def build(path: Path) -> Path:
    layout = kdb.Layout()
    layout.dbu = geo.DBU
    top = layout.create_cell("SYNTHETIC_BONDPADS")
    lm = layout.layer(*MARKER_LAYER)
    ll = layout.layer(*LEAD_LAYER)
    cx0, cy0 = site_center_um()
    for sx, sy in SITES:
        for (px, py, side) in IMAGE_SQUARES:
            (x, y), = TRUE.apply([(px, py)])
            s = side * TRUE.scale
            sq = geo.rect(-s / 2, -s / 2, s / 2, s / 2)
            sq = geo.transform(sq, TRUE.angle_deg, x + sx, y + sy)
            top.shapes(lm).insert(geo.polygon(sq))
        # 24 fan-out leads ending 32-36 µm from the site centre
        for k in range(24):
            a = 2 * math.pi * (k + 0.5) / 24
            r0, r1 = 34.0, 90.0
            p0 = (cx0 + sx + r0 * math.cos(a), cy0 + sy + r0 * math.sin(a))
            p1 = (cx0 + sx + r1 * math.cos(a), cy0 + sy + r1 * math.sin(a))
            path_ = kdb.DPath([kdb.DPoint(*p0), kdb.DPoint(*p1)], 2.0)
            top.shapes(ll).insert(path_.polygon().to_itype(geo.DBU))
    path.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(path))
    return path


if __name__ == "__main__":
    print(build(HERE / "synthetic_bondpads.gds"))
