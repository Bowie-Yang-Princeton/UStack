# UStack

Design van der Waals heterostructure devices and their e-beam lithography patterns — Aviram Uri Lab.

UStack (v0.1) builds a device stack layer by layer (or from a template), checks it against the
lab's materials and design rules, lists the flakes to look for, lays the flakes out on their optical
images, and — after fabrication, before each EBL step — aligns the device images to the chip's
bond-pad GDS and generates the EBL pattern for one parameterised **Hall bar**, re-checks it with the
measured values and writes a GDS with KLayout plus an updated specification.

The look follows UAnalysis (the AFM data viewer): its house style module is vendored as
`ustack/ui/style.py`.

## Run it

- **Mac:** double-click `run_ustack.command` (the first start sets up `~/Library/Caches/UStack-build`, a few
  minutes). If the Finder will not open it, right-click it › **Open** › **Open** (once), or run it from the
  Terminal: `bash ~/Library/CloudStorage/Dropbox-Princeton/Software/UStack/run_ustack.command`.
  It needs a Python 3.10–3.14 from python.org, Homebrew or Anaconda (it can reuse the one UAnalysis uses).
- **Windows:** double-click `run_ustack.bat`.
- **Any terminal:** `pip install -r requirements.txt`, then `python -m ustack` in this folder.
- Tests: `python -m pytest -q tests`.

Optional: set `ANTHROPIC_API_KEY` (and `USTACK_CLAUDE_MODEL`, default `claude-opus-5-5`) to let
**Describe…** draft a stack from a plain-English request with Claude. Without a key a keyword parser
handles phrasings like "an hBN-encapsulated monolayer graphene Hall bar with a graphite back gate
and edge contacts".

## The window

- **Left:** Templates (the four group structures and the lab's own) and saved Devices (rows darken
  under the mouse and stay darker when selected); **Save device** puts the current device in the list.
  Right-click a device to **Rename…** (its folder and the image paths in it follow) or **Delete…**
  (asks first; removes the device's folder in the library, not the original image files).
- **Middle:** *Structure* (3D preview — built from the traced flake borders — + structured
  specification), *Flake layout* (place the flakes on each other), *EBL design* (Stage 2).
- **Right:** *Stack* (layers — drag to reorder; material, function, dimensions), *Flake images*
  (one image per flake, Trace all / Trace flake…; the tracing opens in its own window),
  *Pattern* (Hall bar, contact scheme, gate limits, gate geometry), *Check* (validation, missing
  values, flake list), *Stage 2*.
- Title bar: day/night, **Save device**, Save as template, **Configure**, **Edit preset dimensions**
  (the relations Configure uses, and the microscope field of view), **Rules** (the lab configuration,
  with the number of values waiting for approval; values are read-only until **Edit**, bottom left).
- The 3D preview: drag to rotate it to any angle (also from below); two-finger (right-button) drag or a
  two-finger trackpad slide moves it; wheel or pinch zooms; double-click resets; *Separate Layers*
  only spreads the layers in height.
- The 3D preview has an x–y scale bar on the chip surface that turns with the device and re-scales
  as you zoom. Click a layer's swatch in its legend to hide / show it (**Show all layers** brings them back).

## Stage 1 — before fabrication

1. **New device** always asks: *Start from a template* (the one selected in the list is preselected)
   or *Build from scratch* (substrate → bottom gate → device layers, one row per flake piece → top
   gate, with tick boxes for the hBN dielectrics, plus the contact scheme and metal). Or **Describe…**.
2. Edit the stack in **Stack**: add / delete / drag layers; each layer has a material (from
   `config/materials.csv` only — anything else is flagged, never replaced) and a function.
3. Enter dimensions in **Stack** and **Pattern**. Literature numbers appear as grey placeholders
   with a **Use** button; nothing is filled in for you.
4. Or press **Configure** (title bar or Check tab): leave the dimensions empty and UStack generates
   them — Hall bar, flake sizes, thicknesses, gate geometry — so the stack passes validation. It asks
   once for what only the lab or you can give (stacking misalignment, contact overlap, the lab's
   design thickness per hBN/graphite role, layer counts, twist angles, gate voltages); lab values are
   saved to `config/` and reused. Values you typed are never changed; generated ones are marked
   *auto* with their derivation and are recomputed by the next Configure. The relations themselves
   (W = 3 × arm width, pitch = 2 × W, 2 probe pairs, hBN and gate margins, …) are presets: change
   them in **Edit preset dimensions** (or Rules › *Configure presets*).
   Pattern › **Source / drain**: *Straight*, *Bent to the sides* (the source turns to +y, the drain to
   −y, so their contacts land beside the channel: for a back-gate strip that runs the whole length
   of the device, and for the Mak two-column electrodes) or *Automatic* (straight, unless Stage 2
   finds the back gate under the ends).
   If Configure cannot make the stack valid, a **Configure: fix these** window stays open with every
   error, the values behind it (editable there: your values are kept), the values still needed, the
   generated values and the warnings; **Configure again** reruns it, **Validate** only checks.
5. **Save device** at any stage (or Check › **Validate**). The 3D preview and the structured specification update; the
   Check tab lists errors, warnings and **missing values** — *Fill in missing…* asks for them.
6. A valid specification produces the **Flakes To Find** list (export as CSV); each minimum size is
   the design size plus the stacking misalignment (`stack.alignment_shift_um`) on every side.
7. **Flake images** (right-hand tab, between Stack and Pattern): the tab lists every layer; each
   flake asks for an image (**Upload all…** walks through the flakes without one, bottom to top; the
   objective sets the scale from its field of view: a whole 100x image is 143.28 × 95.13 µm, other
   objectives scale as 100 / magnification — edit in *Edit preset dimensions* › Microscope).
   **Trace all** then traces every flake with an image as a *whole flake*, without a click (borders
   you traced by clicking, drew or edited are kept). **Trace flake…** (or double-click a flake)
   opens the tracing window: the image and its border, **Auto trace**, **Next candidate** (wrong
   flake), the wand (click the flake), the pen (draw by hand), drag points to edit (double-click an
   edge to add one); **Add point** (click on the border line; hold to drag the new point),
   **Select points** (drag a box; Shift adds; Shift-click a point also adds it) — drag any selected
   point to move them all, the trash button or Delete / Backspace removes them and the points on
   either side join up; Ctrl/⌘+Z undoes. Whole-flake tracing leaves out tape residue (soft-edged,
   patchy film next to the flake) on bright-field images; *Trace*: **Whole flake** or **This colour only** (a
   monolayer attached to a thick piece; a buried layer); Previous / Next flake. **View** opens the
   image alone. Each flake is centred on the design origin and the Structure tab's 3D view stacks
   the real borders; tick layers to show them, or **Only selected** / **Only flakes** / **Show
   all**. The tracer was tuned on the lab's WSe2 batch-1 images: see `docs/trace_benchmark.md`.
8. **Flake layout:** move / rotate / fade the images to place the flakes on each other (a flake moved
   here keeps its place when traced again). Outlines replace the nominal sizes in the checks.

## Stage 2 — before each EBL step

The **Stage 2** tab is a workflow of EBL steps on one chip (the EBL design view opens with it):

1. **Bond Pads**: the chip pattern every step is written on — bond pads + fan-out, alignment
   markers, landing square — saved once as a preset. Built in: the lab's tWSe2 batch 1 chip (layers
   1/3/4 of the KLayout session's Stack GDS). **Save a GDS as preset…** adds another (copied into
   `library/bondpads/`, with which layer holds what and the device site).
2. **EBL Steps**: *Hall bars (finger array)*, *Contact gate*, *Bottom gate*, *Top gate*, *Electrodes
   to the bond pads*, and the classic graphene Hall-bar step. Tick the steps written in the same EBL
   exposure (they share the images, alignment and flakes; Generate draws all of them, Export GDS
   writes them into one file); clicking a name works on that step alone. The card:
   - **Images of the sample up to this point**: 5x / 10x / 20x / 50x / 100x (at least one of 5x, 10x,
     20x — it fixes the position on the chip; more magnifications make it more precise). **View**
     opens one in its own window; the trash removes it; *Use the images of …* starts from another step's.
   - **Align to bond pads**: the lowest magnification is matched to the pads and markers, each higher
     one to the previous and re-fitted on the pads; or **Pick pairs** (image point, then GDS point;
     ≥ 2) and **Fit pairs**. On the lab's Stack 1/3 images this lands within 0.1 µm of the lab's own
     KLayout alignment at 100x (`docs/registration_benchmark.md`).
   - **Flakes on the chip**: **Locate flakes** finds each border traced in *Flake images* in the
     aligned 20x–100x images — images loaded after the last Align are aligned first; with only a 5x /
     10x image it uses that one, coarser (~1–2 µm at 5x) — (any rotation, mirrored or not, partly covered; ~5–20 s per flake). hBN
     and graphite are usually found; thin covered layers (middle hBN, WSe2) usually are not — draw
     them. Flakes you placed on each other in *Flake layout* follow the located ones (dashed =
     planned, not on the chip yet). **Draw outline** opens the selected flake's outline in its own
     window over an aligned image (choose which): draw it, trace it with the wand, or start from the
     border traced in *Flake images*; drag its line to move it, edit points as in the trace window.
     On the EBL canvas too, click any outline's line to select it and drag it to move it (its points
     then show for reshaping). A step without outlines starts from the previous step's.
   - *Hall bars*: the **twisted region**. The fingers go on the bottom gate before the device layers
     are dropped, so **Plan from flake images** puts the device layers' traced borders (*Flake
     images*) on the bond-pad site — at the middle of the pads, or where Flake layout put them
     relative to each other — and lists them: click a row (or a layer's line on the canvas) and drag
     the line to move the layer; its slider turns it about its middle (−180°…180°); the image button
     opens its traced image. The twisted region is their overlap and follows every move. Or **From
     located flakes** (the overlap of the located halves). Then the
     array axis (empty = automatic) and a cap on the pairs. The fingers are the presets (0.8 µm wide,
     2.2 µm pitch, 4 µm between the columns — *Preset dimensions…*); UStack counts how many pairs fit
     on the twisted region (each pair: the channel between the tips and 0.8 µm of each finger on the
     region) and places the most, at the best angle. Nothing is shrunk to fit.
   - EBL canvas: two-finger scroll moves the view, pinch zooms, a two-finger twist turns it (a mouse
     wheel zooms); **View rotation** slider (±180°, double-click = 0°) and the reset button below the
     canvas. The chip coordinates do not change.
   - **Generate** draws the step on the EBL canvas — fingers (straight, then fanned out), the
     contact-gate comb (one tooth per finger + bus), back-gate contacts at both ends of the gate strip,
     the top-gate contact or metal gate, the electrodes from every finger / comb / gate contact to the
     bond-pad leads. Every shape is editable on the canvas; **Check** reruns the width / spacing
     check; **Export GDS…** writes `library/devices/<device>/stage2/<step>/<device>_<step>.gds`. GDS
     layers come from `config/layers.json`; missing ones are asked for (the lab's 5/6/7 offered).
3. **Merge & Export**: the chosen steps on their own layers plus the bond pads in one GDS.

The measure tool (ruler icon) works on the EBL canvas, in Flake layout, in the trace window and in
every image window: click the start, click the end; double-click (or Esc) clears. **Undo / Redo**
(top right, or Ctrl/⌘+Z and Ctrl+Y / Ctrl(⌘)+Shift+Z) step back and forth through every change to the
device, Stage 2 patterns included (not the lab configuration, not files already written). With the pointer,
click and drag moves the view (drag a square handle to edit a point).

## Lab training data (`WSe2 device training/`)

The lab's WSe2 batch-1 data (stacks 1, 3, 4: one image per flake, images before / after each step,
and the KLayout session with the three designs) is used two ways:

- `tools/trace_benchmark.py` — the flake tracer on the 18 single-flake images (click stability, the
  flake found without a click, overlays) → `docs/trace_benchmark.md`; `tests/test_tracing.py` keeps it so.
- `tools/learn_lab_designs.py` (with `ustack/lab_designs.py`) — reads the KLayout session (each
  image's pixel → chip transform) and measures the lab's own GDS designs (fingers, contact-gate comb,
  electrodes, markers) → `docs/wse2_design_report.md`. These are measurements, not approved values.

## Lab configuration (`config/`)

| File | What |
|---|---|
| `materials.csv` | Materials the lab has. |
| `rules.json` | Design rules and approved defaults, each with value, unit, severity, status and source. |
| `layers.json` | GDS layer/datatype per pattern purpose. |
| `presets.json` | Configure's design relations (program choices until edited) and the microscope field of view. |

Every value has a **status**: *approved* (stated by the lab or approved in Rules), *pending* (copied
from the attached Fab Manual / DRC notes, waiting for approval), *literature* (from a paper, for a
group template) or *missing* (asked for the first time a step needs it, then saved as approved).

### Values still needed from the lab

Missing in `rules.json`: stacking misalignment (flake list / Configure); the lab's design thicknesses
(asked by Configure per hBN/graphite role); SiO2 breakdown field; defect clearance; 1D-contact overlap; etch-mask
overhang; board-etch keep margin; etch-window size; the "thin" middle-hBN limit (Xu); optionally the
hBN breakdown field. Missing in `layers.json`: Hall-bar
etch, etch windows, top gate, bottom gate, pre-pattern contacts, contact gates, split gates and the
two reference layers. At first start 49 entries are not yet approved (pending, literature or
missing); review them in **Rules**.

See `AGENTS.md` for the architecture and `docs/device_structures.md` for the research behind the
templates.
