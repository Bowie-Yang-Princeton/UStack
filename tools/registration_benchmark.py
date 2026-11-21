"""Benchmark of ustack.registration on the lab's tWSe2 batch-1 data (``WSe2 device training/``).

Ground truth = the lab's own hand alignment in the KLayout session ``tWSe2-1.lys`` (a projective
3×3 matrix per image; errors are measured at the device site (0, 0) of the chip).

  1. Low magnification → GDS (layers 3/0 markers + 4/0 pads / fan-out): error at the site,
     rotation and scale error, NCC score, ambiguity margin, time. Also how far the session's own
     projective matrix is from the best similarity (a limit for any similarity model).
  2. Chain 5x → 10x → 20x → 100x (``align_images``) and the AFM-clean / 1-BS 100x images chained
     onto the TS images or aligned on the GDS directly.
  3. Flakes: each single-flake image (``Layer OM Image``) traced with its seed
     (``tools/trace_benchmark_seeds.json``), converted to µm (100x field of view, or 50x) and
     located in the aligned TS 100x, TS 20x and 1-BS 100x images. No hand-labelled truth: scores
     plus overlays to judge by eye, and the physical check that the back gate runs through the
     column gap of the layer-5 fingers.

Writes docs/registration_benchmark.md and docs/training/registration_*.jpg.

    python tools/registration_benchmark.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ustack import imaging, registration as reg  # noqa: E402
from ustack.alignment import Similarity, fit_similarity  # noqa: E402
from ustack.gds_io import read_polygons  # noqa: E402
from ustack.lab_designs import analyse_design, load_session, local_to_chip  # noqa: E402

DATA = ROOT / "WSe2 device training"
SESSION = DATA / "tWSe2-1_KLayout_session" / "tWSe2-1.lys"
FLAKES = DATA / "Layer OM Image"
OUT = ROOT / "docs" / "training"
DOC = ROOT / "docs" / "registration_benchmark.md"
FOV100 = 143.28              # µm, whole 100x image (presets.microscope.fov_100x_width_um, approved)
LAYERS = [(3, 0), (4, 0)]    # markers + bond pads / fan-out (user's meaning of the lab's files)
STACKS = {"Stack 1": 0, "Stack 3": 1}   # session view index
SITE = (0.0, 0.0)

#: My judgement of the overlays of the run of 2026-11-17 (re-check it after changing the algorithm).
BY_EYE = """
### Checked by eye (overlays of 2026-11-17)

- **Correct:** Stack 1 bottom hBN (TS 100x, TS 20x, 1-BS 100x: same pose, 23°), Stack 1 top hBN (TS 100x, TS 20x, 61°),
  Stack 1 back gate (TS 20x and 1-BS 100x, 229°; in the 1-BS image the outline lies on the dark graphite band),
  Stack 1 top gate (TS 20x, 19.5°), Stack 3 top hBN (TS 100x, TS 20x, 102°). Stack 3 bottom hBN (dark-field source
  image; 14.5–15° in all three images, accepted only on the 20x) looks plausible but is not certain.
- **Not found:** the middle hBN and WSe2 flakes of both stacks (faint, under other layers; scores ≤ 0.44 with margins
  ≤ 0.16, all rejected), the back gate and top gate in the TS 100x images (most of each graphite strip lies outside
  the 143 µm field there), Stack 3 back gate in every image (its single-flake image shows a straight strip crossing
  the whole field: two parallel edges and no end, so its position along the strip is undetermined), and Stack 3 top
  gate (the best 0.49 in the 1-BS image is a false match — that image was taken before the top gate was added — and
  is rejected by the margin rule).
- **Magnification of the single-flake images:** every accepted flake scores far higher with the 100x field of view
  (143.28 µm across the image width, also for the 3000-px dark-field images) than with the 50x one (e.g. Stack 1
  bottom hBN 0.59 vs 0.11, top hBN 0.73 vs 0.23), and the fitted scale is 0.98 (the configured field of view is
  ≈ 2 % larger than the session's), so they are 100x images.
- **Back gate vs fingers:** where the back gate was found (Stack 1, two independent images) it covers the whole column
  gap, but its outline also covers ≈ 124 µm² of finger area within ±15 µm of the device centre, i.e. this graphite is
  wider than the 5 µm gap and lies across the finger tips in the images. Reported as measured; whether that is the
  intended layout is for the lab to say.
"""


def nominal(mag: int, width_px: int) -> float:
    return FOV100 * 100.0 / mag / width_px


def session_image(view, *keys: str):
    for im in view.images:
        if all(k in im.file for k in keys) and im.exists:
            return im
    return None


def site_error(sim: Similarity, si, W: int, H: int, X: float = 0.0, Y: float = 0.0) -> float:
    c, r = si.chip_to_px(X, Y, W, H)
    p = sim.apply([[float(c), float(r)]])[0]
    return float(math.hypot(p[0] - X, p[1] - Y))


def session_angle(si) -> float:
    m = np.asarray(si.matrix)
    return math.degrees(math.atan2(m[1, 0], m[0, 0]))


def session_similarity(si, W: int, H: int) -> Similarity:
    cs, rs = np.meshgrid(np.linspace(0, W, 9), np.linspace(0, H, 7))
    x, y = si.px_to_chip(cs.ravel(), rs.ravel(), W, H)
    return fit_similarity(np.c_[cs.ravel(), rs.ravel()], np.c_[x, y])


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def gds_overlay(img, sim, polys, layers, colour, thick=3):
    m = reg.render_gds(polys, layers, sim, img.shape)
    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, colour, thick)
    return img


def fit_width(img, width=1600):
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def trace_outlines(entry: dict, fov: float):
    img = imaging.load_image(FLAKES / entry["file"])
    H, W = img.shape[:2]
    s = fov / W
    out = []
    for seed in entry["seeds"]:
        poly = np.asarray(imaging.detect_flake(img, seed, mode=entry["mode"]), float)
        out.append(np.c_[(poly[:, 0] - W / 2) * s, -(poly[:, 1] - H / 2) * s])
    return out, (-W / 2 * s, -H / 2 * s, W / 2 * s, H / 2 * s), (W, H)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    views = load_session(SESSION)
    md = ["# Image registration benchmark (WSe2 batch 1)", "",
          "Generated by `tools/registration_benchmark.py` for `ustack/registration.py`. Ground truth = the lab's hand "
          "alignment in the KLayout session `tWSe2-1.lys` (projective matrices). Errors are at the device site (0, 0). "
          f"Nominal scales come from the configured 100x field of view ({FOV100} µm wide, other objectives × 100/mag), "
          "which is ≈ 1.6–1.9 % larger than the session's scales, so every run also has to find the scale. "
          "GDS layers used for alignment: 3/0 (markers) and 4/0 (bond pads, fan-out).", ""]
    seeds = json.loads((ROOT / "tools" / "trace_benchmark_seeds.json").read_text(encoding="utf-8"))["images"]
    sims: dict = {}
    images: dict = {}

    # ------------------------------------------------------------------ 1. low magnification → GDS
    md += ["## 1. One image → GDS (`align_to_gds`)", "",
           "Low magnification without a site (whole GDS searched, ±20°, scale ±5 %); 100x with site (0, 0) and a 300 µm "
           "search radius. *Session vs similarity* = distance at the site between the session's projective matrix and "
           "the best similarity to it over the whole image (how much of the 'truth' a similarity cannot reproduce).", "",
           row("Stack", "Image", "Error at site (µm)", "Rotation error (°)", "Scale error (%)", "Score (NCC)", "Margin",
               "Time (s)", "Session vs similarity (µm)"),
           row(*["---"] * 9)]
    for stack, vi in STACKS.items():
        view = views[vi]
        gds = read_polygons(view.gds)
        for mag in ("5x", "10x", "20x", "100x"):
            si = session_image(view, "TS/", f"_{mag}_")
            img = imaging.load_image(si.path)
            H, W = img.shape[:2]
            images[(stack, "TS " + mag)] = (img, si)
            kw = {"site": SITE, "search_radius_um": 300.0} if mag == "100x" else {}
            t = time.time()
            try:
                sim = reg.align_to_gds(img, nominal(int(mag[:-1]), W), gds, LAYERS, **kw)
            except ValueError as e:
                md.append(row(stack, "TS " + mag, f"failed: {e}", "", "", "", "", f"{time.time() - t:.1f}", ""))
                continue
            dt = time.time() - t
            sims[(stack, "TS " + mag)] = sim
            ss = session_similarity(si, W, H)
            md.append(row(stack, "TS " + mag, f"{site_error(sim, si, W, H):.2f}", f"{sim.angle_deg - session_angle(si):+.3f}",
                          f"{100 * (sim.scale / si.um_per_px - 1):+.2f}", f"{sim.score:.3f}", f"{sim.info['margin']:.3f}",
                          f"{dt:.1f}", f"{site_error(ss, si, W, H):.2f}"))
            print(stack, mag, "gds done", f"{dt:.1f}s")
        # overlay: 5x / 20x / 100x with GDS outlines
        tiles = []
        for mag, crop in (("5x", None), ("20x", None), ("100x", None)):
            k = (stack, "TS " + mag)
            if k not in sims:
                continue
            img = images[k][0].copy()
            gds_overlay(img, sims[k], gds, [(4, 0)], (0, 0, 255), 6)
            gds_overlay(img, sims[k], gds, [(3, 0)], (255, 0, 255), 6)
            gds_overlay(img, sims[k], gds, [(5, 0)], (255, 255, 0), 4)
            tiles.append(cv2.resize(img, (800, 531), interpolation=cv2.INTER_AREA))
        if tiles:
            sheet = np.hstack(tiles) if len(tiles) < 3 else np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], np.zeros_like(tiles[2])])])
            cv2.imwrite(str(OUT / f"registration_{stack.replace(' ', '').lower()}_gds.jpg"), fit_width(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
    md += ["", "Overlays: `docs/training/registration_stack1_gds.jpg`, `registration_stack3_gds.jpg` (red = layer 4, "
           "magenta = layer 3, cyan = layer 5 fingers, drawn with the found transform on 5x, 20x and 100x).", ""]

    # ------------------------------------------------------------------ 2. chains
    md += ["## 2. Several magnifications (`align_images`)", "",
           "TS 5x, 10x, 20x, 100x of one stack with site (0, 0), search radius 1500 µm: the 5x is aligned to the GDS, each "
           "next one chained onto the previous one and re-fitted on the GDS (kept if NCC ≥ "
           f"{reg.REFINE_GDS_SCORE_MIN} and the shift ≤ {100 * reg.REFINE_GDS_MAX_SHIFT_FRAC:.0f} % of the field diagonal).", "",
           row("Stack", "Image", "Method", "Error at site (µm)", "Rotation error (°)", "Chain NCC", "GDS NCC", "Time (s)"),
           row(*["---"] * 8)]
    chain_sims = {}
    for stack, vi in STACKS.items():
        view = views[vi]
        gds = read_polygons(view.gds)
        ims = {m: (images[(stack, "TS " + m)][0], nominal(int(m[:-1]), 6000)) for m in ("5x", "10x", "20x", "100x")}
        t = time.time()
        res = reg.align_images(ims, gds, LAYERS, site=SITE, search_radius_um=1500.0)
        total = time.time() - t
        for m, sim in res.items():
            img, si = images[(stack, "TS " + m)]
            lg = res.log[m]
            md.append(row(stack, "TS " + m, lg["method"], f"{site_error(sim, si, 6000, 3984):.2f}",
                          f"{sim.angle_deg - session_angle(si):+.3f}", lg.get("score", ""), lg.get("gds_score", ""), lg.get("seconds", "")))
            chain_sims[(stack, "TS " + m)] = sim
        for m, e in res.errors.items():
            md.append(row(stack, "TS " + m, "failed", e, "", "", "", ""))
        md.append(row(stack, "**all four**", "", "", "", "", "", f"{total:.1f}"))
        # chain only (no GDS refit) for the 100x, from our 20x and from the session's 20x
        img100, si100 = images[(stack, "TS 100x")]
        img20, si20 = images[(stack, "TS 20x")]
        for label, ref in (("our 20x", res.get("20x")), ("session 20x", session_similarity(si20, 6000, 3984))):
            if ref is None:
                continue
            try:
                s = reg.align_to_image(img100, nominal(100, 6000), img20, ref)
            except ValueError as e:
                md.append(row(stack, "TS 100x", f"chain only, onto {label}", f"failed ({e})", "", "", "", ""))
                continue
            md.append(row(stack, "TS 100x", f"chain only, onto {label}", f"{site_error(s, si100, 6000, 3984):.2f}",
                          f"{s.angle_deg - session_angle(si100):+.3f}", f"{s.score:.3f}", "", f"{s.info['seconds']:.1f}"))
        print(stack, "chain done", f"{total:.1f}s")
    md += ["", "### AFM-clean and 1-BS 100x images", "",
           "These have no lower-magnification partners. Tried: chaining onto the TS 20x / TS 100x of the same stack (placed "
           "by our GDS alignment), and the GDS directly (site (0, 0), 300 µm radius).", "",
           row("Stack", "Image", "Method", "Error at site (µm)", "Rotation error (°)", "Score", "Time (s)"), row(*["---"] * 7)]
    for stack, vi in STACKS.items():
        view = views[vi]
        gds = read_polygons(view.gds)
        num = stack.split()[1]
        for label, key in (("AFM clean 100x", "AFM clean"), ("1-BS 100x", "1 - BS")):
            si = session_image(view, key)
            img = imaging.load_image(si.path)
            images[(stack, label)] = (img, si)
            for refk in ("TS 20x", "TS 100x"):
                t = time.time()
                try:
                    s = reg.align_to_image(img, nominal(100, 6000), images[(stack, refk)][0], sims[(stack, refk)])
                    md.append(row(stack, label, f"chain onto {refk}", f"{site_error(s, si, 6000, 3984):.2f}",
                                  f"{s.angle_deg - session_angle(si):+.3f}", f"{s.score:.3f}", f"{time.time() - t:.1f}"))
                except ValueError as e:
                    md.append(row(stack, label, f"chain onto {refk}", f"failed ({e})", "", "", f"{time.time() - t:.1f}"))
            t = time.time()
            try:
                s = reg.align_to_gds(img, nominal(100, 6000), gds, LAYERS, site=SITE, search_radius_um=300.0)
                sims[(stack, label)] = s
                md.append(row(stack, label, "GDS", f"{site_error(s, si, 6000, 3984):.2f}", f"{s.angle_deg - session_angle(si):+.3f}",
                              f"{s.score:.3f}", f"{time.time() - t:.1f}"))
            except ValueError as e:
                md.append(row(stack, label, "GDS", f"failed ({e})", "", "", f"{time.time() - t:.1f}"))
        print(stack, num, "AFM/BS done")

    # ------------------------------------------------------------------ 3. flakes
    md += ["", "## 3. Flakes (`locate_outline`)", "",
           "Each single-flake image traced with its seed (`tools/trace_benchmark_seeds.json`; a WSe2 image with two seeds "
           "gives two pieces), converted to µm with the 100x field of view (and, for the magnification check, with the 50x "
           "one), searched within 60 µm of the device site with all rotations and mirror. Points on the edge of the "
           "single-flake image are not scored (`frame_um`). Score = soft fraction of the border on a matching edge; "
           f"accepted when score ≥ {reg.OUTLINE_SCORE_MIN} and margin ≥ {reg.OUTLINE_MARGIN_MIN}. No hand-labelled truth: see "
           "*Checked by eye* below the table.", ""]
    targets = [("TS 100x", sims), ("TS 20x", sims), ("1-BS 100x", sims)]
    flake_rows = [row("Stack", "Flake", "Image", "FOV (µm)", "Score", "Fraction", "Margin", "Angle (°)", "Mirror", "Scale",
                      "Accepted", "Time (s)"), row(*["---"] * 12)]
    found: dict = {}
    for stack in STACKS:
        for e in seeds:
            if not e["file"].startswith(stack + "/"):
                continue
            for fov_label, fov in (("100x", FOV100), ("50x", 2 * FOV100)):
                outlines, frame, _ = trace_outlines(e, fov)
                for k, P in enumerate(outlines):
                    name = Path(e["file"]).stem + (f" piece {k + 1}" if len(outlines) > 1 else "")
                    for tgt, sd in targets:
                        if fov_label == "50x" and tgt != "TS 100x":
                            continue
                        if (stack, tgt) not in sd:
                            continue
                        img = images[(stack, tgt)][0]
                        try:
                            r = reg.locate_outline(img, sd[(stack, tgt)], P, frame_um=frame, search_center_um=SITE,
                                                   search_radius_um=60.0, min_score=0.0, min_margin=-1.0)
                        except ValueError as ex:
                            flake_rows.append(row(stack, name, tgt, f"{fov:.1f}", f"failed ({ex})", "", "", "", "", "", "", ""))
                            continue
                        ok = r["score"] >= reg.OUTLINE_SCORE_MIN and r["margin"] >= reg.OUTLINE_MARGIN_MIN
                        flake_rows.append(row(stack, name, tgt, f"{fov:.1f}", f"{r['score']:.3f}", f"{r['fraction']:.2f}",
                                              f"{r['margin']:.3f}", f"{r['angle_deg']:.1f}", "yes" if r["mirror"] else "no",
                                              f"{r['scale']:.2f}", "**yes**" if ok else "no", f"{r['seconds']:.1f}"))
                        if fov_label == "100x":
                            found[(stack, name, tgt)] = (r, ok)
                        print(stack, name, tgt, fov_label, f"{r['score']:.3f} {r['margin']:.3f} {r['seconds']:.1f}s")
    md += flake_rows
    # overlays per stack and image
    colours = [(0, 0, 255), (0, 200, 0), (255, 80, 0), (0, 220, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255), (0, 128, 255)]
    for stack, vi in STACKS.items():
        gds = read_polygons(views[vi].gds)
        for tgt in ("TS 100x", "TS 20x", "1-BS 100x"):
            if (stack, tgt) not in sims:
                continue
            img = images[(stack, tgt)][0]
            R = 100.0 if tgt == "TS 20x" else 72.0
            ras, valid, C = reg._chip_raster(img, sims[(stack, tgt)], (-R, -R * 0.66, R, R * 0.66), max(sims[(stack, tgt)].scale, 2 * R / 1600))
            vis = ras.copy()
            fm = reg.render_gds(gds, [(5, 0)], C, ras.shape)
            cnts, _ = cv2.findContours((fm > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 140, 255), 1)
            n = 0
            for (st, name, t2), (r, ok) in found.items():
                if st != stack or t2 != tgt:
                    continue
                col = colours[n % len(colours)]
                q = C.inverse(np.asarray(r["polygon"]))
                cv2.polylines(vis, [np.round(q).astype(np.int32)], True, col, 3 if ok else 1, cv2.LINE_AA)
                cv2.putText(vis, f"{name[:30]}  {r['score']:.2f}{'' if ok else ' (rejected)'}", (12, 34 + 32 * n),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
                n += 1
            fn = f"registration_{stack.replace(' ', '').lower()}_{tgt.replace(' ', '').replace('-', '').lower()}_flakes.jpg"
            cv2.imwrite(str(OUT / fn), fit_width(vis), [cv2.IMWRITE_JPEG_QUALITY, 85])
    md += ["", "Overlays (chip frame, x right / y up, orange = layer-5 fingers; thick line = accepted, thin = rejected): "
           "`docs/training/registration_<stack>_<image>_flakes.jpg`.", ""]

    # ------------------------------------------------------------------ physical check
    md += ["### Back gate vs the column gap", "",
           "The gap between the two finger columns (layer 5) in the lab's design: a rectangle (array span at the tips) × "
           "(column gap), from `lab_designs.analyse_design`. *Gap covered* = fraction of it inside the located back-gate "
           "outline (the BG should run through the gap). *On fingers* = finger area inside the BG outline within ±15 µm of "
           "the device centre.", "",
           row("Stack", "Image", "BG score", "Gap covered", "On fingers (µm²)"), row(*["---"] * 5)]
    import klayout.db as kdb
    for stack, vi in STACKS.items():
        a = analyse_design(views[vi].gds)
        gap, span = a.array["column_gap_um"], a.array["span_at_tips_um"]
        rect = [local_to_chip(a, u, v) for u, v in ((-span / 2, -gap / 2), (span / 2, -gap / 2), (span / 2, gap / 2), (-span / 2, gap / 2))]

        def region(poly):
            return kdb.Region(kdb.Polygon([kdb.Point(int(round(x * 1000)), int(round(y * 1000))) for x, y in poly]))

        fingers = kdb.Region()
        for p in read_polygons(views[vi].gds).get((5, 0), []):
            fingers += region(p)
        cx, cy = a.frame["device_centre_chip_um"]
        near = region([(cx - 15, cy - 15), (cx + 15, cy - 15), (cx + 15, cy + 15), (cx - 15, cy + 15)])
        g = region(rect)
        for tgt in ("TS 20x", "1-BS 100x", "TS 100x"):
            key = [k for k in found if k[0] == stack and k[2] == tgt and "back gate" in k[1]]
            if not key:
                continue
            r, ok = found[key[0]]
            bg = region(r["polygon"])
            cov = (g & bg).area() / max(1, g.area())
            onf = (fingers & bg & near).area() * 1e-6
            md.append(row(stack, tgt, f"{r['score']:.3f}{'' if ok else ' (rejected)'}", f"{cov:.2f}", f"{onf:.1f}"))
    md.append(BY_EYE)
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", DOC)


if __name__ == "__main__":
    main()
