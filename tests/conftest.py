"""Test helpers. Tests never touch the real config/ or library/: each gets a copy of the seeded
configuration (tests/fixtures/config_seed) in a temp dir.

Values that tests enter where the lab has not supplied one are marked TEST VALUE; they exist
only in the temporary config copy.
"""

import math
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    d = tmp_path / "config"
    # a frozen copy of the seeded configuration: values the lab enters in config/ never change the tests
    shutil.copytree(ROOT / "tests" / "fixtures" / "config_seed", d)
    monkeypatch.setenv("USTACK_CONFIG", str(d))
    monkeypatch.setenv("USTACK_HOME", str(tmp_path / "library"))
    from ustack.config import LabConfig
    return LabConfig(d)


@pytest.fixture(scope="session")
def bondpads(tmp_path_factory):
    import make_fixtures
    return make_fixtures.build(tmp_path_factory.mktemp("gds") / "synthetic_bondpads.gds")


def graphene_answers():
    return {"spec:S0.material": "SiO2_285_doped_Si", "spec:gate.S0": 20,
            "spec:BG.thickness_nm": 5, "spec:BG.width_um": 6, "spec:BG.length_um": 10,
            "spec:BBN.thickness_nm": 30, "spec:BBN.width_um": 24, "spec:BBN.length_um": 34,
            "spec:GR.layer_count": 1, "spec:GR.width_um": 16, "spec:GR.length_um": 26,
            "spec:TBN.thickness_nm": 20, "spec:TBN.width_um": 20, "spec:TBN.length_um": 30,
            "spec:TG.metal.Ti": 5, "spec:TG.metal.Au": 30, "spec:CT.metal.Cr": 5, "spec:CT.metal.Au": 60,
            "spec:hall_bar.channel_width_um": 2, "spec:hall_bar.channel_length_um": 6, "spec:hall_bar.n_probe_pairs": 2,
            "spec:hall_bar.arm_width_um": 1, "spec:hall_bar.arm_length_um": 4.5, "spec:hall_bar.sd_extension_um": 4}


def circle(cx, cy, r, n=16):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
