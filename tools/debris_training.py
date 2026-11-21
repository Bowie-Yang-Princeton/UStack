"""Training set for ustack.debris on the lab's tWSe2 batch-1 images taken before the electrodes (``WSe2 device training/
OM images/2026-11-02``, placed on the chip by the lab's KLayout session ``tWSe2-1.lys``).

    python tools/debris_training.py sheets   # candidates → numbered contact sheets (docs/training/debris_sheet_*.jpg)
    python tools/debris_training.py build    # labels (tools/debris_labels.json) → ustack/data/debris_examples.json
                                             #   + docs/debris_training.md (leave-one-out accuracy, k-NN vs the rules)

The sheet index keeps each candidate's place, and ``build`` matches the labels by place (a changed finder renumbers the
candidates; an index without places — the first one — needs the finder as it was when its sheets were made).

The known metal is the lab's own GDS (layers 3 markers, 4 bond pads, 5 fingers, 6 contact gate; layer 1 holds reference
frames, not metal). The stack is the large piece at the device site (the session has no flake outlines; in the app every
located layer is known, and the edges between them are not looked at). Labels are given by eye on the sheets: flake,
particle, bubble, crack, none (not a feature: a layer edge, a misaligned metal edge, illumination)."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ustack import debris, geometry as geo  # noqa: E402
from ustack.alignment import fit_similarity  # noqa: E402
from ustack.gds_io import read_polygons  # noqa: E402
from ustack.lab_designs import load_session  # noqa: E402

DATA = ROOT / "WSe2 device training"
SESSION = DATA / "tWSe2-1_KLayout_session" / "tWSe2-1.lys"
OUT = ROOT / "docs" / "training"
LABELS = ROOT / "tools" / "debris_labels.json"
EXAMPLES = ROOT / "ustack" / "data" / "debris_examples.json"
DOC = ROOT / "docs" / "debris_training.md"
CASES = [(0, "20x"), (0, "100x"), (1, "20x"), (1, "100x")]
METAL = [(3, 0), (4, 0), (5, 0), (6, 0)]
REGION = [[(-200.0, -200.0), (200.0, -200.0), (200.0, 200.0), (-200.0, 200.0)]]
PER_CASE = 48                     # candidates sampled per image for labelling
SHEET = 6                         # 6 × 6 crops per sheet


def _sim(im, W, H):
    g = np.array([(x, y) for x in np.linspace(0, W, 5) for y in np.linspace(0, H, 5)])
    X, Y = im.px_to_chip(g[:, 0], g[:, 1], W, H)
    return fit_similarity(g, np.c_[X, Y])


def candidates():
    """[(case id, image, Similarity, [Candidate])] for every case (stack = the piece at the site)."""
    views = load_session(SESSION)
    out = []
    for view, mag in CASES:
        im = next(i for i in views[view].images if "before_electrodes" in i.path.name and f"_{mag}_" in i.path.name)
        img = cv2.imread(str(im.path))
        H, W = img.shape[:2]
        sim = _sim(im, W, H)
        P = read_polygons(views[view].gds)
        metal = [p for L in METAL for p in P.get(L, [])]
        first = debris.find_in_image(img, sim, metal, [], REGION, image_name=im.path.name)
        stack = [c.polygon for c in first if geo.point_in_poly(0.0, 0.0, c.polygon)]
        stack = [max(stack, key=lambda p: abs(geo.poly_area(p)))] if stack else []
        cands = debris.find_in_image(img, sim, metal, stack, REGION, image_name=im.path.name)
        out.append((f"{views[view].name}|{mag}", img, sim, cands))
    return out


def _sample(cands, n, seed):
    """A sample spread over the rule kinds and sizes (every kind seen, big pieces not drowned by specks)."""
    rnd = random.Random(seed)
    by = {}
    for i, c in enumerate(cands):
        by.setdefault((debris.rule_kind(c.features), c.features["log_area"] > 1.0), []).append(i)
    picked = []
    groups = list(by.values())
    while len(picked) < n and any(groups):
        for g in groups:
            if g and len(picked) < n:
                picked.append(g.pop(rnd.randrange(len(g))))
    return sorted(picked)


def sheets() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for k, (case, img, sim, cands) in enumerate(candidates()):
        for i in _sample(cands, PER_CASE, k):
            index.append((case, i, img, sim, cands[i]))
    tile = 220
    for s in range(0, len(index), SHEET * SHEET):
        page = np.full((SHEET * (tile + 22), SHEET * tile, 3), 255, np.uint8)
        for j, (case, i, img, sim, c) in enumerate(index[s:s + SHEET * SHEET]):
            px = sim.inverse(np.asarray(c.polygon, float))
            x0, y0 = px.min(0)
            x1, y1 = px.max(0)
            side = max(x1 - x0, y1 - y0, 6.0 / sim.scale) * 2.2           # ≥ 6 µm of context
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            a = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
            b = int(min(cx + side / 2, img.shape[1])), int(min(cy + side / 2, img.shape[0]))
            crop = img[a[1]:b[1], a[0]:b[0]].copy()
            if crop.size == 0:
                continue
            f = tile / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
            pts = np.round((px - a) * f).astype(np.int32)
            cv2.polylines(crop, [pts], True, (0, 0, 255), 1)
            r, q = divmod(j, SHEET)
            y, x = r * (tile + 22), q * tile
            page[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
            cv2.putText(page, f"{s + j}", (x + 3, y + tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(OUT / f"debris_sheet_{s // (SHEET * SHEET)}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT / "debris_sheet_index.json").write_text(json.dumps([[case, i, *map(float, geo.centroid(c.polygon))]
                                                             for case, i, _img, _sim, c in index]), encoding="utf-8")
    print(f"{len(index)} candidates on {math.ceil(len(index) / SHEET / SHEET)} sheets in {OUT}")


def build() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["labels"]          # sheet number → kind
    index = json.loads((OUT / "debris_sheet_index.json").read_text(encoding="utf-8"))
    cases = {case: cands for case, _img, _sim, cands in candidates()}
    examples = []
    for n, kind in labels.items():
        if kind == "skip":
            continue
        case, i, *xy = index[int(n)]
        c = cases[case][i] if i < len(cases[case]) else None
        if xy:                                          # matched by place: a changed finder renumbers its candidates
            c = min(cases[case], key=lambda c_: math.dist(geo.centroid(c_.polygon), xy))
            if math.dist(geo.centroid(c.polygon), xy) > 1.0:
                continue
        if c is None:
            continue
        examples.append({"kind": kind, "features": {k: round(float(v), 4) for k, v in c.features.items()}, "source": f"{case}#{i}"})
    EXAMPLES.write_text(json.dumps({"schema": "ustack.debris_examples/1",
                                    "source": "tools/debris_training.py on the lab's tWSe2 batch-1 images before the electrodes "
                                              "(2026-11-02), labelled by eye", "examples": examples}, indent=1), encoding="utf-8")
    # leave-one-out: k-NN on the others vs the rules
    knn_ok = rule_ok = 0
    confusion = Counter()
    for j, e in enumerate(examples):
        clf = debris.Classifier(examples[:j] + examples[j + 1:])
        got, _s = clf.classify(e["features"])
        knn_ok += got == e["kind"]
        rule_ok += debris.rule_kind(e["features"]) == e["kind"]
        confusion[(e["kind"], got)] += 1
    n = len(examples)
    kinds = sorted({e["kind"] for e in examples})
    lines = ["# Bubbles, cracks and stray flakes — training set", "",
             "Written by `tools/debris_training.py build` (do not edit by hand).", "",
             f"{n} candidates found by `ustack.debris.find_in_image` in the lab's Stack 1 / Stack 3 images before the electrodes "
             "(20x and 100x, 2026-11-02, placed by the lab's KLayout session), labelled by eye on the contact sheets "
             "(`docs/training/debris_sheet_*.jpg`).", "",
             "| Kind | Examples |", "|---|---|"] + [f"| {k} | {sum(e['kind'] == k for e in examples)} |" for k in kinds] + [
             "", f"Leave-one-out: k-NN (k = 5) **{knn_ok}/{n} ({100 * knn_ok / n:.0f} %)**, the rules alone {rule_ok}/{n} "
             f"({100 * rule_ok / n:.0f} %).", "", "Confusion (k-NN; rows = label, columns = predicted):", "",
             "| | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
    for a in kinds:
        lines.append(f"| {a} | " + " | ".join(str(confusion[(a, b)]) for b in kinds) + " |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{n} examples; k-NN {knn_ok}/{n}, rules {rule_ok}/{n}")


if __name__ == "__main__":
    {"sheets": sheets, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "sheets"]()
