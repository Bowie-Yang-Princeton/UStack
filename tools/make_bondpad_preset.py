"""Write the built-in bond-pad preset from the lab's tWSe2 batch 1 KLayout session.

Layers 1/0 (landing square, chip and write field), 3/0 (alignment markers) and 4/0 (bond pads and
fan-out) of ``Design/Stack 4.gds`` are identical in Stack 1/3/4; they are copied unchanged into
``ustack/data/bondpads/tWSe2_batch1.gds``. Layer meanings: the user's message of 2026-11-16.

    python tools/make_bondpad_preset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import klayout.db as kdb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "WSe2 device training" / "tWSe2-1_KLayout_session" / "Design" / "Stack 4.gds"
OUT = ROOT / "ustack" / "data" / "bondpads"
KEEP = {(1, 0): "Landing", (3, 0): "Markers", (4, 0): "BondPads"}


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC}")
        return 1
    src = kdb.Layout()
    src.read(str(SRC))
    top = src.top_cell()
    out = kdb.Layout()
    out.dbu = src.dbu
    cell = out.create_cell("BondPads_tWSe2_batch1")
    for li in src.layer_indexes():
        info = src.get_info(li)
        if (info.layer, info.datatype) not in KEEP:
            continue
        reg = kdb.Region(top.begin_shapes_rec(li))
        dst = out.layer(kdb.LayerInfo(info.layer, info.datatype, KEEP[(info.layer, info.datatype)]))
        cell.shapes(dst).insert(reg)
    OUT.mkdir(parents=True, exist_ok=True)
    gds = OUT / "tWSe2_batch1.gds"
    out.write(str(gds))
    meta = {"schema": "ustack.bondpads/1", "name": "tWSe2 batch 1 chip", "gds": gds.name,
            "layers": {"landing": [1, 0], "markers": [3, 0], "pads": [4, 0]}, "site": [0.0, 0.0], "site_radius_um": 400.0,
            "source": "Layers 1/3/4 of tWSe2-1_KLayout_session/Design/Stack 4.gds (identical in Stack 1/3/4); layer meanings "
                      "stated by the user on 2026-11-16."}
    (OUT / "tWSe2_batch1.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {gds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
