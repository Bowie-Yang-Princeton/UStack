"""lab_designs: measuring two-column electrode designs and reading KLayout sessions.

(a) a synthetic GDS built here (geometry chosen for the test, not lab values);
(b) the lab's tWSe2-1 session, skipped when the folder is not present.
"""

from pathlib import Path

import numpy as np
import pytest

import klayout.db as kdb

from ustack import lab_designs as ld

ROOT = Path(__file__).resolve().parent.parent
SESSION = ROOT / "WSe2 device training" / "tWSe2-1_KLayout_session" / "tWSe2-1.lys"
ROLES = {"landing": (1, 0), "markers": (3, 0), "pads": (4, 0), "fingers": (5, 0),
         "contact_gate": (6, 0), "electrodes": (7, 0)}

# synthetic geometry (µm)
W_F, PITCH, GAP, LEN, N = 0.8, 2.4, 3.0, 10.0, 4


def _box(cell, layer, x0, y0, x1, y1):
    cell.shapes(layer).insert(kdb.DBox(x0, y0, x1, y1))


def _finger_x(i):
    return (i - (N - 1) / 2) * PITCH


def _synthetic(path: Path) -> Path:
    ly = kdb.Layout()
    ly.dbu = 0.001
    top = ly.create_cell("TOP")
    L = {k: ly.layer(*v) for k, v in ROLES.items()}
    _box(top, L["landing"], -200, -200, 200, 200)
    for i in range(N):
        x = _finger_x(i)
        _box(top, L["fingers"], x - W_F / 2, GAP / 2, x + W_F / 2, GAP / 2 + LEN)       # top column
        _box(top, L["fingers"], x - W_F / 2, -GAP / 2 - LEN, x + W_F / 2, -GAP / 2)     # bottom column
        _box(top, L["contact_gate"], x - W_F / 2, GAP / 2, x + W_F / 2, GAP / 2 + 3.0)  # CG teeth, top only
        # electrodes: 1.5 µm wide, landing 7 µm from the tip, running out to the pads
        _box(top, L["electrodes"], x - 0.75, GAP / 2 + 7.0, x + 0.75, 60.0)
        if i < N - 1:  # last bottom finger left floating
            _box(top, L["electrodes"], x - 0.75, -60.0, x + 0.75, -GAP / 2 - 7.0)
    xl = _finger_x(0) - W_F / 2
    _box(top, L["contact_gate"], xl, GAP / 2 + 3.0, -xl + 3.0, GAP / 2 + 4.0)           # CG bus bar, led out
    _box(top, L["electrodes"], -xl + 2.0, GAP / 2 + 3.0, 60.0, GAP / 2 + 4.5)            # CG electrode
    _box(top, L["electrodes"], -60.0, -1.0, -30.0, 1.0)                                  # along the gap, far
    _box(top, L["pads"], -80, 58, 80, 70)
    _box(top, L["pads"], -80, -70, 80, -58)
    _box(top, L["pads"], 58, -50, 70, 50)
    _box(top, L["pads"], -70, -50, -58, 50)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _box(top, L["markers"], sx * 40 - 1.25, sy * 40 - 1.25, sx * 40 + 1.25, sy * 40 + 1.25)
    ly.write(str(path))
    return path


def test_synthetic_design(tmp_path):
    a = ld.analyse_design(_synthetic(tmp_path / "synthetic.gds"), ROLES)
    assert abs(a.frame["finger_axis_deg"] - 90.0) < 0.01
    t, b = a.columns["top"], a.columns["bottom"]
    assert t["n"] == N and b["n"] == N
    assert t["tip_width_um"] == pytest.approx(W_F, abs=0.005)
    assert t["pitch_at_tips_um"] == pytest.approx(PITCH, abs=0.005)
    assert t["spacing_at_tips_um"] == pytest.approx(PITCH - W_F, abs=0.005)
    assert a.array["column_gap_um"] == pytest.approx(GAP, abs=0.005)
    assert a.array["tips_facing"]["max_abs_offset_um"] < 0.005
    assert t["straight_length_um"] == pytest.approx(LEN, abs=0.1)
    assert t["centreline_length_um"] == pytest.approx(LEN, abs=0.05)
    assert a.array["span_at_tips_um"] == pytest.approx((N - 1) * PITCH + W_F, abs=0.01)
    # contact gate: teeth on the 4 top fingers, flush with the tips, 3 + 1 µm along, not into the gap
    cg = a.contact_gate
    assert cg["n_shapes"] == 1 and cg["top_fingers_covered"] == N and cg["bottom_fingers_covered"] == 0
    assert cg["top_overlap_length_um"] == pytest.approx(4.0, abs=0.01)  # 3 µm tooth + the 1 µm bus bar over the finger
    assert cg["top_overlap_from_tip_um"] == pytest.approx(0.0, abs=0.01)
    assert cg["top_cg_width_on_finger_um"] == pytest.approx(W_F, abs=0.01)
    assert abs(cg["top_clearance_to_gap_um"]) < 0.01
    # electrodes
    es = a.electrode_summary
    assert es["by_kind"] == {"finger": 2 * N - 1, "contact_gate": 1, "neither": 1}
    assert es["finger_electrodes_per_column"] == {"top": N, "bottom": N - 1}
    assert es["floating_fingers"] == {"top": [], "bottom": [N - 1]}
    assert es["finger_electrode_width_um"] == pytest.approx(1.5, abs=0.02)
    assert es["min_spacing_near_device_um"] == pytest.approx(PITCH - 1.5, abs=0.01)
    assert es["overlap_from_tip_um"] == pytest.approx(7.0, abs=0.01)
    assert es["overlap_length_along_finger_um"] == pytest.approx(LEN - 7.0, abs=0.02)
    assert es["all_reach_pads"] is True
    (u, v), = es["neither_inner_ends_local_um"]
    assert u == pytest.approx(-30.0, abs=0.01) and abs(v) <= 1.0
    # landing square and markers
    assert a.landing["smallest"]["size_um"] == [400.0, 400.0]
    assert a.markers["by_size"]["2.5x2.5"] and len(a.markers["items"]) == 4
    assert a.to_json()["status"].startswith("measured")


def test_rotated_design_measures_the_same(tmp_path):
    src = _synthetic(tmp_path / "s.gds")
    ly = kdb.Layout()
    ly.read(str(src))
    ly.top_cell().transform(kdb.DCplxTrans(1.0, 7.0, False, 3.0, -2.0))
    ly.write(str(tmp_path / "r.gds"))
    a = ld.analyse_design(tmp_path / "r.gds", ROLES)
    assert abs(a.frame["finger_axis_deg"] - 97.0) < 0.02
    assert a.columns["top"]["n"] == N and a.columns["bottom"]["n"] == N
    assert a.columns["top"]["pitch_at_tips_um"] == pytest.approx(PITCH, abs=0.01)
    assert a.array["column_gap_um"] == pytest.approx(GAP, abs=0.01)
    assert a.electrode_summary["n_floating"] == {"top": 0, "bottom": 1}


def test_session_image_round_trip():
    img = ld.SessionImage(file="x.jpg", path=Path("x.jpg"), exists=False,
                          matrix=[[0.0234, 0.0001, -3.3], [-0.0002, 0.0235, 1.5], [2.7e-7, -1.2e-6, 1.0]])
    W, H = 6000, 4000
    cols, rows = np.array([0.0, 1234.5, 5999.0]), np.array([0.0, 2000.0, 3999.0])
    x, y = img.px_to_chip(cols, rows, W, H)
    c2, r2 = img.chip_to_px(x, y, W, H)
    assert np.allclose(c2, cols, atol=1e-6) and np.allclose(r2, rows, atol=1e-6)
    assert img.um_per_px == pytest.approx(0.02345, rel=0.01)


# ------------------------------------------------------------------------------- real lab data
needs_data = pytest.mark.skipif(not SESSION.is_file(), reason="tWSe2-1 KLayout session not present")


@needs_data
def test_session_views_and_markers_project_into_100x_image():
    cv2 = pytest.importorskip("cv2")
    views = ld.load_session(SESSION)
    assert [v.stem for v in views] == ["Stack 1", "Stack 3", "Stack 4"]
    assert all(v.gds and v.gds.is_file() for v in views)
    assert any(not i.exists for v in views for i in v.images)  # missing Picture*.png are kept, flagged
    v = views[0]
    img = next(i for i in v.images if i.exists and i.magnification == "100x" and "AFM clean" in i.file)
    assert img.um_per_px == pytest.approx(0.0234, abs=0.0005)
    im = cv2.imread(str(img.path))
    H, W = im.shape[:2]
    a = ld.analyse_design(v.gds, ld.WSE2_BATCH1_LAYERS)
    near = [m for m in a.markers["items"] if max(abs(m["dx_um"]), abs(m["dy_um"])) < 60]
    assert len(near) >= 4
    cx, cy = a.frame["device_centre_chip_um"]
    cols, rows = img.chip_to_px([cx + m["dx_um"] for m in near], [cy + m["dy_um"] for m in near], W, H)
    assert np.all((cols > 0) & (cols < W) & (rows > 0) & (rows < H))


@needs_data
def test_real_stacks():
    folder = SESSION.parent / "Design"
    s1 = ld.analyse_design(folder / "Stack 1.gds", ld.WSE2_BATCH1_LAYERS)
    assert s1.columns["top"]["n"] >= 2 and s1.columns["bottom"]["n"] >= 2
    assert s1.array["column_gap_um"] > 0
    assert s1.electrode_summary["by_kind"].get("finger", 0) > 0
    s3 = ld.analyse_design(folder / "Stack 3.gds", ld.WSE2_BATCH1_LAYERS)
    assert s3.electrode_summary["n_floating"] == {"top": 1, "bottom": 3}  # as the user described
    s4 = ld.analyse_design(folder / "Stack 4.gds", ld.WSE2_BATCH1_LAYERS)
    assert s4.electrode_summary["n_near_device"] == 0
    assert s4.columns["top"]["n"] > 0
