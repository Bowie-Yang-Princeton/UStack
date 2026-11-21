# Bubbles, cracks and stray flakes — training set

Written by `tools/debris_training.py build` (do not edit by hand).

180 candidates found by `ustack.debris.find_in_image` in the lab's Stack 1 / Stack 3 images before the electrodes (20x and 100x, 2026-11-02, placed by the lab's KLayout session), labelled by eye on the contact sheets (`docs/training/debris_sheet_*.jpg`).

| Kind | Examples |
|---|---|
| bubble | 16 |
| crack | 11 |
| flake | 34 |
| none | 43 |
| particle | 76 |

Leave-one-out: k-NN (k = 5) **149/180 (83 %)**, the rules alone 99/180 (55 %).

Confusion (k-NN; rows = label, columns = predicted):

| | bubble | crack | flake | none | particle |
|---|---|---|---|---|---|
| bubble | 13 | 0 | 0 | 2 | 1 |
| crack | 0 | 4 | 0 | 6 | 1 |
| flake | 0 | 0 | 24 | 1 | 9 |
| none | 4 | 0 | 1 | 36 | 2 |
| particle | 0 | 0 | 2 | 2 | 72 |
