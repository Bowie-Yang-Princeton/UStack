"""Bond-pad presets: the chip pattern every EBL step is written onto (bond pads + fan-out, alignment
markers, landing square), saved once and reused by every device.

A preset is a GDS file plus a small JSON description: which GDS layer holds what, and where the
device site is (chip µm). Built-in presets live in ``ustack/data/bondpads/`` (regenerate with
``tools/make_bondpad_preset.py``); the lab's own are saved in ``library/bondpads/<name>/``.

The built-in preset is the chip of the lab's tWSe2 batch 1 (layers 1/3/4 of
``tWSe2-1_KLayout_session/Design/Stack 4.gds``, identical in Stack 1/3/4), with the layer meanings
the user gave on 2026-11-16: 1 landing square, 3 markers, 4 bond pads and fan-out.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import geometry as geo
from .gds_io import lead_tips, read_polygons
from .paths import PACKAGE, library_dir, safe_name

Poly = list[tuple[float, float]]
ROLES = {"landing": "Landing square", "markers": "Alignment markers", "pads": "Bond pads and fan-out"}


def builtin_dir() -> Path:
    return PACKAGE / "data" / "bondpads"


def user_dir() -> Path:
    path = library_dir() / "bondpads"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class BondPads:
    name: str
    gds: Path
    layers: dict[str, tuple[int, int]]            # role (landing / markers / pads) → (layer, datatype)
    site: tuple[float, float] = (0.0, 0.0)        # device site, chip µm
    site_radius_um: float = 400.0                 # markers / pad tips used around the site
    builtin: bool = False
    source: str = ""
    json_path: Path | None = None
    _polys: dict | None = field(default=None, repr=False)

    @property
    def polys(self) -> dict[tuple[int, int], list[Poly]]:
        if self._polys is None:
            self._polys = read_polygons(self.gds)
        return self._polys

    def layer_polys(self, role: str, near: bool = False) -> list[Poly]:
        ld = self.layers.get(role)
        polys = self.polys.get(tuple(ld), []) if ld else []
        if near:
            polys = [p for p in polys if math.dist(geo.centroid(p), self.site) <= self.site_radius_um
                     and geo.poly_area(p) < (2 * self.site_radius_um) ** 2]
        return polys

    def pad_tips(self, radius: float | None = None) -> list[tuple[float, float]]:
        """Inner ends of the fan-out leads around the device site (where EBL electrodes connect)."""
        return lead_tips(self.layer_polys("pads"), self.site, radius or self.site_radius_um)

    def pad_ends(self, radius: float | None = None) -> list[dict]:
        """The inner end of each fan-out lead (same order as :meth:`pad_tips`): tip, mid, dir (into the lead), width, poly."""
        from .gds_io import lead_ends
        return lead_ends(self.layer_polys("pads"), self.site, radius or self.site_radius_um)

    def to_json(self) -> dict:
        return {"schema": "ustack.bondpads/1", "name": self.name, "gds": self.gds.name,
                "layers": {k: list(v) for k, v in self.layers.items()}, "site": list(self.site),
                "site_radius_um": self.site_radius_um, "source": self.source}


def _read(path: Path, builtin: bool) -> BondPads | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return BondPads(d["name"], path.parent / d["gds"], {k: tuple(v) for k, v in d["layers"].items()},
                        tuple(d.get("site", (0.0, 0.0))), float(d.get("site_radius_um", 400.0)), builtin, d.get("source", ""), path)
    except (OSError, KeyError, ValueError):
        return None


def list_presets() -> list[BondPads]:
    out = []
    for folder, builtin in ((builtin_dir(), True), (user_dir(), False)):
        if folder.is_dir():
            for p in sorted(folder.glob("**/*.json")):
                bp = _read(p, builtin)
                if bp is not None and bp.gds.is_file():
                    out.append(bp)
    return out


def load_preset(name: str) -> BondPads:
    for bp in list_presets():
        if bp.name == name:
            return bp
    raise KeyError(f"No bond-pad preset named {name!r}")


def default_preset() -> BondPads | None:
    presets = list_presets()
    return presets[0] if presets else None


def save_preset(name: str, gds_path: str | Path, layers: dict[str, tuple[int, int]], site: tuple[float, float] = (0.0, 0.0),
                site_radius_um: float = 400.0, source: str = "") -> BondPads:
    """Save a bond-pad GDS as a lab preset (copied into ``library/bondpads/<name>/``)."""
    folder = user_dir() / safe_name(name)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / Path(gds_path).name
    if Path(gds_path).resolve() != dest.resolve():
        shutil.copy2(gds_path, dest)
    bp = BondPads(name, dest, {k: tuple(v) for k, v in layers.items()}, tuple(site), site_radius_um, False, source)
    bp.json_path = folder / "bondpads.json"
    bp.json_path.write_text(json.dumps(bp.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bp
