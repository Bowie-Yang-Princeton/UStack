"""Image registration (ustack.registration): synthetic chips rendered with a known transform, and the
lab's tWSe2 batch-1 images when the training folder is present (skipped otherwise; thresholds
checked with tools/registration_benchmark.py, see docs/registration_benchmark.md)."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ustack import registration as reg
from ustack.alignment import Similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "WSe2 device training"
SESSION = DATA / "tWSe2-1_KLayout_session" / "tWSe2-1.lys"
SUBSTRATE = (120, 105, 125)    # BGR, purple-grey SiO2
METAL = (40, 200, 225)         # BGR, gold


# ---------------------------------------------------------------------------- synthetic chip
def _rect(cx, cy, w, h):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def _fake_gds():
    """Bond pads around a centre, fan-out leads of different lengths (no symmetry), markers."""
    pads, leads, marks = [], [], []
    for side in range(4):
        n = 5 + side            # a different number per side: no 90° symmetry
        for k in range(n):
            t = (k - (n - 1) / 2) * 90.0
            a = side * math.pi / 2
            ca, sa = math.cos(a), math.sin(a)
            px, py = 600 * ca - t * sa, 600 * sa + t * ca
            pads.append(_rect(px, py, 60, 60))
            # lead from the pad toward the centre, ending at a varied radius
            r_end = 120 + 25 * k + 10 * side
            ex, ey = r_end * ca - t * 0.25 * sa, r_end * sa + t * 0.25 * ca
            d = np.array([ex - px, ey - py])
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d) * 6
            leads.append([tuple(np.array([px, py]) + nrm), tuple(np.array([ex, ey]) + nrm),
                          tuple(np.array([ex, ey]) - nrm), tuple(np.array([px, py]) - nrm)])
    for x, y, s in [(-40, -40, 3), (40, -40, 3), (-40, 40, 3), (40, 40, 3), (-195, 195, 10), (195, -195, 10), (195, 195, 10)]:
        marks.append(_rect(x, y, s, s))
        marks.append(_rect(x + s, y - s, s, s))
    marks.append(_rect(-300, 260, 25, 25))
    return {(4, 0): pads + leads, (3, 0): marks, (1, 0): [_rect(0, 0, 22000, 15000)]}


def _render_chip(gds, sim: Similarity, shape, flakes=(), seed=0, noise=3.0):
    """BGR image of the chip seen through ``sim`` (px → chip µm): substrate, GDS metal, flakes on top."""
    h, w = shape
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = SUBSTRATE
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1 - 0.15 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img *= vign[..., None]
    m = reg.render_gds(gds, [(3, 0), (4, 0)], sim, (h, w)).astype(np.float32) / 255
    img = img * (1 - m[..., None]) + np.array(METAL, np.float32) * m[..., None]
    for poly, colour, alpha in flakes:
        fm = np.zeros((h, w), np.uint8)
        q = sim.inverse(np.asarray(poly, float))
        cv2.fillPoly(fm, [np.round(q * 8).astype(np.int32)], 255, cv2.LINE_AA, shift=3)
        a = (fm.astype(np.float32) / 255 * alpha)[..., None]
        img = img * (1 - a) + np.array(colour, np.float32) * a
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _site_error(sim, truth, X=0.0, Y=0.0):
    p = truth.inverse([[X, Y]])
    q = sim.apply(p)[0]
    return math.hypot(q[0] - X, q[1] - Y)


def _blob(rng, cx, cy, r, n=9):
    a = np.sort(rng.uniform(0, 2 * math.pi, n))
    rr = r * rng.uniform(0.6, 1.0, n)
    return [(cx + rr[i] * math.cos(a[i]), cy + rr[i] * math.sin(a[i])) for i in range(n)]


def _clutter(rng, n, span, rmax):
    cols = [(200, 160, 60), (90, 200, 120), (160, 90, 140), (230, 190, 80)]
    return [(_blob(rng, *rng.uniform(-span, span, 2), rng.uniform(5, rmax)), cols[i % 4], 0.8) for i in range(n)]


# ---------------------------------------------------------------------------- tests
def test_render_gds_and_ignore_huge_shapes():
    gds = _fake_gds()
    sim = Similarity(2.0, 0.0, -800.0, 600.0)      # 2 µm/px, px (0, 0) at chip (−800, 600)
    m = reg.render_gds(gds, [(3, 0)], sim, (600, 800))
    c, r = sim.inverse([[195.0, 195.0]])[0]        # a 10 µm marker square
    assert m[int(round(r)), int(round(c))] == 255
    assert m.mean() < 5                              # sparse
    shapes = reg._gds_shapes(gds, [(1, 0), (3, 0)], max_area=1e6)
    assert len(shapes) == len(gds[(3, 0)])           # the 22×15 mm chip outline is dropped


def test_align_to_gds_low_mag_with_clutter_and_wrong_nominal_scale():
    gds = _fake_gds()
    rng = np.random.default_rng(1)
    truth = Similarity(1.6, 4.0, -780.0, 560.0)
    img = _render_chip(gds, truth, (720, 1080), _clutter(rng, 25, 600, 60))
    nominal = truth.scale * 1.025                   # field of view 2.5 % off
    sim = reg.align_to_gds(img, nominal, gds, [(3, 0), (4, 0)])
    assert _site_error(sim, truth) < 0.5             # < 1/3 px (measured 0.05 µm)
    assert abs(sim.angle_deg - truth.angle_deg) < 0.15
    assert abs(sim.scale / truth.scale - 1) < 0.004
    assert sim.score > 0.5 and sim.info["margin"] > 0.1
    assert sim.info["seconds"] < 10


def test_align_to_gds_without_metal_raises():
    gds = _fake_gds()
    img = np.full((500, 750, 3), SUBSTRATE, np.uint8)
    img = (img + np.random.default_rng(0).normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    with pytest.raises(ValueError):
        reg.align_to_gds(img, 1.5, gds, [(3, 0), (4, 0)])


def test_chain_and_align_images():
    gds = _fake_gds()
    rng = np.random.default_rng(2)
    flakes = _clutter(rng, 40, 450, 40)
    low_t = Similarity(1.6, -2.0, -800.0, 560.0)
    high_t = Similarity(0.32, -1.2, -110.0, 40.0)   # 5x the magnification, near the centre
    low = _render_chip(gds, low_t, (720, 1080), flakes, seed=1)
    high = _render_chip(gds, high_t, (500, 750), flakes, seed=2)
    ref = reg.align_to_gds(low, 1.6 * 1.02, gds, [(3, 0), (4, 0)])
    s = reg.align_to_image(high, 0.32 * 1.02, low, ref)
    assert _site_error(s, high_t, -30.0, -40.0) < 1.0
    res = reg.align_images({"5x": (low, 1.6 * 1.02), "20x": (high, 0.32 * 1.02)}, gds, [(3, 0), (4, 0)])
    assert set(res) == {"5x", "20x"} and not res.errors
    assert res.log["5x"]["method"] == "gds" and res.log["20x"]["method"].startswith("chain")
    assert _site_error(res["20x"], high_t, -30.0, -40.0) < 0.6


def _flake_shape():
    # an irregular, not mirror-symmetric flake (µm, its own frame)
    return [(0, 0), (30, -4), (44, 6), (40, 22), (24, 30), (12, 26), (6, 34), (-6, 20)]


@pytest.mark.parametrize("mirror", [False, True])
def test_locate_outline_rotated_mirrored_and_partly_covered(mirror):
    shape = np.asarray(_flake_shape(), float)
    angle, centre = 67.0, np.array([12.0, -8.0])
    q = shape - reg._poly_centroid(shape)
    if mirror:
        q[:, 0] *= -1
    a = math.radians(angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    placed = q @ R.T + centre
    rng = np.random.default_rng(5)
    cover = _rect(centre[0] + 14, centre[1] + 6, 24, 30)          # an opaque flake over ~1/3 of it
    flakes = [(placed.tolist(), (190, 150, 90), 0.7), (cover, (110, 60, 120), 1.0)] + _clutter(rng, 6, 70, 12)
    sim = Similarity(0.1, 1.5, -70.0, 45.0)
    img = _render_chip({}, sim, (900, 1350), flakes, seed=3, noise=2.0)
    outline = (shape + [100.0, -50.0]).tolist()                    # arbitrary origin in its own image
    r = reg.locate_outline(img, sim, outline)
    assert r["mirror"] is mirror
    assert math.hypot(*(np.asarray(r["center"]) - centre)) < 0.5
    assert abs((r["angle_deg"] - angle + 180) % 360 - 180) < 1.0
    assert 0.45 < r["score"] < 0.95                                 # part of the border is hidden
    assert len(r["polygon"]) == len(outline)
    assert r["seconds"] < 10


def test_locate_outline_absent_flake_raises():
    sim = Similarity(0.1, 0.0, -70.0, 45.0)
    rng = np.random.default_rng(9)
    img = _render_chip({}, sim, (900, 1350), _clutter(rng, 8, 60, 10), seed=4)
    with pytest.raises(ValueError):
        reg.locate_outline(img, sim, _flake_shape(), search_radius_um=40.0)


# ---------------------------------------------------------------------------- the lab's data
lab = pytest.mark.skipif(not SESSION.is_file(), reason="WSe2 device training/ not present")


@pytest.fixture(scope="module")
def stack1():
    from ustack.gds_io import read_polygons
    from ustack.imaging import load_image
    from ustack.lab_designs import load_session
    view = load_session(SESSION)[0]
    gds = read_polygons(view.gds)
    ims = {}
    for im in view.images:
        if im.exists and ("TS/" in im.file or "1 - BS" in im.file):
            key = ("BS " if "1 - BS" in im.file else "TS ") + im.magnification
            ims[key] = (load_image(im.path), im)
    return gds, ims


def _err(sim, si, W=6000, H=3984):
    c, r = si.chip_to_px(0.0, 0.0, W, H)
    p = sim.apply([[float(c), float(r)]])[0]
    return math.hypot(*p)


def _nominal(mag, W=6000):
    return 143.28 * 100 / mag / W


@lab
@pytest.mark.parametrize("mag,limit", [("5x", 1.0), ("20x", 0.5), ("100x", 0.3)])
def test_lab_align_to_gds(stack1, mag, limit):
    # measured (docs/registration_benchmark.md): 5x 0.25 µm, 20x 0.07 µm, 100x 0.07 µm at the device site
    gds, ims = stack1
    img, si = ims["TS " + mag]
    kw = {"site": (0.0, 0.0), "search_radius_um": 150.0} if mag == "100x" else {}
    sim = reg.align_to_gds(img, _nominal(int(mag[:-1])), gds, [(3, 0), (4, 0)], **kw)
    assert _err(sim, si) < limit
    assert sim.info["seconds"] < 12


@lab
def test_lab_flakes_found(stack1):
    # top hBN in the TS 100x image, back gate in the 1-BS 100x image (both checked by eye on the
    # benchmark overlays; scores there 0.73 and 0.86); the BG must run through the finger column gap
    from ustack import imaging
    seeds = {e["file"]: e for e in json.loads((ROOT / "tools" / "trace_benchmark_seeds.json").read_text())["images"]}
    gds, ims = stack1

    def outline(name):
        e = seeds[name]
        img = imaging.load_image(DATA / "Layer OM Image" / name)
        H, W = img.shape[:2]
        s = 143.28 / W
        p = np.asarray(imaging.detect_flake(img, e["seeds"][0], mode=e["mode"]))
        return np.c_[(p[:, 0] - W / 2) * s, -(p[:, 1] - H / 2) * s], (-W / 2 * s, -H / 2 * s, W / 2 * s, H / 2 * s)

    res = {}
    for name, key in (("Stack 1/Stack 1 Top hBN.png", "TS 100x"), ("Stack 1/Stack 1 graphite back gate.png", "BS 100x")):
        img, _ = ims[key]
        sim = reg.align_to_gds(img, _nominal(100), gds, [(3, 0), (4, 0)], site=(0.0, 0.0), search_radius_um=150.0)
        P, frame = outline(name)
        r = reg.locate_outline(img, sim, P, frame_um=frame, search_center_um=(0.0, 0.0), search_radius_um=60.0)
        assert r["score"] > 0.6 and r["margin"] > 0.2
        assert not r["mirror"]
        res[name] = r
    bg = res["Stack 1/Stack 1 graphite back gate.png"]["polygon"]
    # the column gap of Stack 1 is centred near (0.7, −1.7) µm, 5 µm wide, fingers along y
    assert cv2.pointPolygonTest(np.asarray(bg, np.float32), (0.7, -1.7), False) > 0


# ---------------------------------------------------------------------- flakes dropped since the previous step
def test_change_map_finds_a_faint_covered_flake_under_thin_lines():
    """A faint flake dropped between two images, under a thicker cover flake and crossed by thin
    written lines, with a different exposure: found in what changed, not in the photograph."""
    rng = np.random.default_rng(3)
    H, W = 900, 1200
    before = np.full((H, W, 3), (150, 140, 135), np.float32)
    cv2.fillPoly(before, [np.array([[100, 150], [1100, 120], [1050, 800], [150, 760]], np.int32)], (200, 150, 110))   # old hBN
    after = before.copy()
    cover = np.array([[300, 250], [950, 260], [900, 650], [280, 620]], np.int32)
    cv2.fillPoly(after, [cover], (210, 165, 115))                                                  # new thick cover, faint
    flake = np.array([[480, 360], [700, 330], [740, 520], [610, 545], [570, 470], [480, 540], [440, 440]], np.float64)   # a notch: no mirror twin
    cv2.fillPoly(after, [flake.astype(np.int32)], (222, 178, 118))                                 # new thin flake under it
    for x in range(380, 860, 40):                                                                  # thin metal lines
        cv2.line(after, (x, 200), (x + 30, 720), (170, 120, 230), 6)
    after = after * 1.08 + 6                                                                       # another exposure
    noise = lambda a: np.clip(a + rng.normal(0, 2.0, a.shape), 0, 255).astype(np.uint8)
    before, after = noise(before), noise(after)
    sim = Similarity(0.05, 0.0, -30.0, 22.5)                                                       # 0.05 µm/px, chip centred
    outline = sim.apply(flake)
    traced = [tuple(p) for p in (np.asarray(outline) - outline.mean(0)) @ np.array([[0, -1], [1, 0]]).T]   # turned 90°
    cm = reg.change_map(after, sim, before, sim, (-30, -22.5, 30, 22.5))
    r = reg.locate_new_outline(cm, traced, True, (0.0, 0.0), 20.0)
    from ustack import geometry as geo
    assert r["method"] == "change" and r["score"] > 0.5 and not r["mirror"]
    assert math.dist(geo.centroid(r["polygon"]), geo.centroid([tuple(p) for p in outline])) < 0.4


LAB_DEVICE = Path(__file__).resolve().parents[1] / "library" / "devices" / "Device_2026-11-16" / "spec.json"


@pytest.mark.skipif(not LAB_DEVICE.is_file(), reason="the lab's Device 2026-11-16 is not in library/")
def test_lab_wse2_and_middle_hbn_found_since_the_hall_bars_step():
    """The lab's tWSe2 device: at the contact-gate step the two WSe2 halves and the middle hBN (thin,
    covered, crossed by the fingers) are found in what changed since the Hall-bar step's images; the
    WSe2 halves land where the user planned them (≤ 1.5 µm)."""
    from ustack import ebl_flow as ef, geometry as geo, imaging
    from ustack.model import DeviceSpec
    spec = DeviceSpec.from_json(json.loads(LAB_DEVICE.read_text()))
    flow = ef.Flow.of(spec)
    st, hb = flow.step("contact_gate"), flow.step("hall_bars")
    assert ef.previous_imaged_step(flow, "contact_gate") == "hall_bars"
    assert set(ef.new_layers(spec, "contact_gate", "hall_bars")) >= {"TMD1", "TMD2", "MBN"}
    cm = reg.change_map(imaging.load_image(st.images["50x"]), Similarity.from_json(st.transforms["50x"]),
                        imaging.load_image(hb.images["50x"]), Similarity.from_json(hb.transforms["50x"]), (-80, -80, 80, 80))
    found = {}
    for uid in ("TMD1", "TMD2", "MBN"):
        found[uid] = reg.locate_new_outline(cm, [tuple(p) for p in spec.layer(uid).outline], True, (0.0, 0.0), 60.0)
        assert found[uid]["score"] >= reg.OUTLINE_SCORE_MIN
    planned = {"TMD1": (-0.39, -1.72), "TMD2": (0.19, -2.19)}          # the user's plan in the Hall-bar step
    for uid, c in planned.items():
        assert math.dist(geo.centroid(found[uid]["polygon"]), c) < 1.5
    # the WSe2 halves lie under the middle hBN
    mbn = geo.region(found["MBN"]["polygon"])
    for uid in ("TMD1", "TMD2"):
        t = geo.region(found[uid]["polygon"])
        assert geo.area(t & mbn) > 0.9 * geo.area(t)
