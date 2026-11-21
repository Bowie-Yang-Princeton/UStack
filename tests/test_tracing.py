"""Flake tracing (imaging.detect_flake / auto_trace): synthetic cases, and the lab's WSe2 batch-1
images when the training folder is present (skipped otherwise)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ustack import imaging

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "WSe2 device training" / "Layer OM Image"
SEEDS = json.loads((ROOT / "tools" / "trace_benchmark_seeds.json").read_text(encoding="utf-8"))


def _substrate(h=600, w=900, vignette=0.0, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
    base = np.array([150, 150, 140], np.float32) * (1 - vignette * r2)[..., None]
    return base + rng.normal(0, noise, (h, w, 3))


def _mask_of(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
    return m.astype(bool)


def _iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def _img(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def test_two_tone_flake_is_traced_whole_and_stable():
    """A graphite flake with a thicker step (two colours) under strong vignetting: 'whole flake'
    gives the full outline from a click on either tone (the v0.3 wand stopped at the step)."""
    a = _substrate(vignette=0.5)
    flake = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(flake, [np.array([[200, 150], [700, 180], [650, 450], [250, 420]])], 1)
    step = np.zeros_like(flake)
    cv2.fillPoly(step, [np.array([[450, 150], [700, 180], [650, 450], [470, 440]])], 1)
    a[flake == 1] = (150, 90, 120)
    a[(flake & step) == 1] = (170, 70, 95)
    img = _img(a)
    truth = flake.astype(bool)
    ctx = imaging.prepare_trace(img)
    for seed in ((300, 300), (600, 300)):
        poly = imaging.detect_flake(img, seed, context=ctx)
        assert _iou(_mask_of(poly, img.shape), truth) > 0.97


def test_monolayer_beside_thick_piece_in_colour_mode():
    """A faint monolayer (ΔE ≈ 8) attached to a thick flake: 'This colour only' separates it."""
    a = _substrate()
    mono = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(mono, [np.array([[250, 200], [450, 200], [450, 420], [250, 420]])], 1)
    thick = np.zeros_like(mono)
    cv2.fillPoly(thick, [np.array([[450, 150], [650, 150], [650, 450], [450, 450]])], 1)
    a[mono == 1] = (155, 140, 140)
    a[thick == 1] = (190, 110, 60)
    img = _img(a)
    whole = _mask_of(imaging.detect_flake(img, (350, 300)), img.shape)
    assert _iou(whole, (mono | thick).astype(bool)) > 0.95
    only = _mask_of(imaging.detect_flake(img, (350, 300), mode="colour"), img.shape)
    assert _iou(only, mono.astype(bool)) > 0.95
    # auto: the faint one when a monolayer is wanted, the big one otherwise
    poly, _ = imaging.auto_trace(img, "faint")
    assert _iou(_mask_of(poly, img.shape), mono.astype(bool)) > 0.9


def test_flake_filling_the_field_and_clicks_on_substrate():
    a = _substrate()
    big = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(big, [np.array([[40, 20], [880, 30], [870, 599], [60, 599]])], 1)
    a[big == 1] = (200, 140, 90)
    img = _img(a)
    m = _mask_of(imaging.detect_flake(img, (450, 300)), img.shape)
    assert _iou(m, big.astype(bool)) > 0.97
    with pytest.raises(ValueError):
        imaging.detect_flake(_img(_substrate()), (450, 300))


def test_legacy_wand_still_available():
    a = _substrate(noise=0.5)
    cv2.rectangle(a, (300, 200), (600, 400), (100, 60, 140), -1)
    poly = imaging.detect_flake(_img(a), (450, 300), tolerance=12.0)
    assert len(poly) >= 4


def test_tape_residue_is_left_out_but_a_thin_terrace_is_kept():
    """Residue (soft-edged, patchy, a colour thin hBN can have) attached to a sharp-edged flake is not
    part of the flake; a uniform thinner terrace with a sharp edge is."""
    rng = np.random.default_rng(3)
    a = _substrate(noise=1.0)
    flake = np.zeros(a.shape[:2], np.uint8)
    cv2.fillPoly(flake, [np.array([[300, 180], [600, 200], [590, 420], [320, 400]])], 1)
    terrace = np.zeros_like(flake)
    cv2.fillPoly(terrace, [np.array([[590, 260], [700, 270], [690, 380], [592, 370]])], 1)
    # residue: a blotchy, blurred blue film below the flake, fading out
    blot = cv2.GaussianBlur((rng.random(a.shape[:2]) < 0.5).astype(np.float32), (0, 0), 2.5)
    blot = (blot - blot.min()) / (blot.max() - blot.min())
    halo = np.zeros(a.shape[:2], np.float32)
    cv2.ellipse(halo, (450, 430), (150, 45), 0, 0, 360, 1.0, -1)
    halo = cv2.GaussianBlur(halo, (0, 0), 12) * (0.1 + 1.8 * blot)
    a = a * (1 - np.clip(halo, 0, 1)[..., None] * 0.25) + np.clip(halo, 0, 1)[..., None] * np.array([60, 5, 0]) * 0.6
    a[terrace.astype(bool)] = (175, 140, 120)
    a[flake.astype(bool)] = (200, 210, 60)
    img = _img(cv2.GaussianBlur(_img(a), (3, 3), 0))
    m = _mask_of(imaging.detect_flake(img, (450, 300)), img.shape)
    truth = (flake | terrace).astype(bool)
    assert m[320, 650] and m[300, 450]                       # terrace and flake
    assert not m[455, 450] and not m[445, 380]               # the residue below the flake
    assert _iou(m, truth) > 0.95
    # without the filter the residue would be traced in (the case the filter is for)
    old, _ = imaging.trace_mask(imaging.prepare_trace(img), (450, 300), params={"residue": False})
    assert old[445, 450]


# ---------------------------------------------------------------------- the lab's images
lab_images = pytest.mark.skipif(not DATA.is_dir(), reason="WSe2 training images not present")


@lab_images
@pytest.mark.parametrize("entry", SEEDS["images"], ids=lambda e: e["file"])
def test_lab_images_click_stability_and_auto_pick(entry):
    """Each flake: clicks at five points inside give the same border (IoU ≥ 0.95), and tracing
    without a click finds the same flake (IoU ≥ 0.9)."""
    img = imaging.load_image(DATA / entry["file"])
    ctx = imaging.prepare_trace(img)
    mode = entry["mode"]
    refs = []
    for seed in entry["seeds"]:
        _, cnt = imaging.trace_mask(ctx, seed, mode)
        refs.append(_mask_of(imaging._contour_polygon(cnt, ctx.f, imaging.TRACE_DEFAULTS["detail"]), img.shape))
    ref = refs[0]
    er = cv2.erode(ref.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    ys, xs = np.nonzero(er)
    rng = np.random.default_rng(3)
    for k in rng.choice(len(xs), size=5, replace=False):
        got = _mask_of(imaging.detect_flake(img, (xs[k], ys[k]), mode=mode, context=ctx), img.shape)
        assert _iou(got, ref) > entry.get("min_iou", 0.95)
    poly, _ = imaging.auto_trace(img, entry["prefer"], mode, ctx)
    auto = _mask_of(poly, img.shape)
    assert max(_iou(auto, r) for r in refs) > 0.9


@lab_images
def test_lab_top_hbn_tape_residue():
    """Stack 1 top hBN (under the top gate): the blue tape residue at its lower-left corner and along
    its lower edge is not part of the border (it was, before the residue filter)."""
    img = imaging.load_image(DATA / "Stack 1" / "Stack 1 Top hBN.png")
    m = _mask_of(imaging.detect_flake(img, (1057, 476)), img.shape)
    assert m[476, 1057] and m[700, 1000]                     # the flake
    assert not m[700, 858] and not m[690, 852]               # residue hook at the lower-left corner
    assert m.sum() > 0.97 * _mask_of(imaging.detect_flake(img, (1057, 476)), img.shape).sum()
