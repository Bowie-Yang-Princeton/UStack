# AGENTS.md — handoff for AI agents working on UStack

Read this first. It tells you what UStack is, how it is put together, the rules you must keep,
what is done, what is not, and what the lab still has to supply. Last updated 2026-11-21 (v0.5:
Stage 2 rebuilt as a workflow of EBL steps on a saved bond-pad preset, finger-array pattern (Hall bars,
contact-gate comb, gate contacts, electrodes) fitted to the twisted region, multi-magnification
alignment and flake location (`registration`), merged GDS export, measure tool; v0.4:
Flake images tab, new flake tracer tuned on the lab's WSe2 data, lab-design analysis, Configure
problems window, device rename/delete, Stage 2 image viewer; v0.3: bent S/D, Mak two-column electrodes, presets, New device dialog, UI items; v0.2: Configure + bug
fixes; v0.1 first version, built by Claude for Yanxing Li, Uri Lab, Princeton).

## 1. What UStack is

A PyQt6 desktop app for a 2D-materials transport lab. A researcher designs a van der Waals
heterostructure device and its e-beam lithography (EBL) patterns:

- **Stage 1 (before fabrication):** layer stack (material + function per layer, drag to reorder),
  templates (4 built-in group structures + lab-saved), 3D preview + structured spec, validation
  against lab rules, list of flakes to find, flake layout on OM images (wand border detection,
  editable polygons, image move/rotate/opacity).
- **Stage 2 (before each EBL step), v0.5:** a bond-pad preset (saved chip pattern) and a list of
  EBL steps — Hall bars (the lab's finger array), contact gate, electrodes to the bond pads (with the
  top / bottom gate electrodes, since 2026-11-19) — each with its own card: images at 5x–100x, alignment to the bond pads, flakes
  located in the images, Generate (editable shapes on the EBL canvas), Check, Export GDS; plus Merge
  & Export (chosen steps + bond pads in one GDS). The v0.1–v0.4 graphene Hall-bar step (board etch,
  Hall-bar etch, 1D contacts, metal TG, …) stays as the "Classic Hall bar" entry (`stage2_panel`).

Patterns: the parameterised Hall bar (classic step) and the finger array (`fingers.py`).

## 2. Non-negotiable rules (from the user)

1. **Never invent lab values** — process limits, materials, GDS layer numbers, approved defaults.
   A value not supplied is `null` with status `missing`; code that needs it calls
   `LabConfig.require(...)` → `MissingValues` → the UI asks the user → `config.apply_answers`
   stores it as `approved` with attribution. Do not add "sensible defaults".
2. **Never change the user's design to pass validation**: no material substitution, no dropping a
   requested gate, no shrinking/growing a specified dimension. Validators *report*; generators place
   (position/angle only) or fail with a message. `Requirement` objects record explicit requests and
   are re-checked.
3. **Ask for crucial missing parameters** — through `MissingItem`s, all at once where possible
   (`stage2.required_values`).
4. Config lives in `config/materials.csv`, `config/rules.json`, `config/layers.json`, and (v0.3)
   `config/presets.json` — Configure's design relations (status `program` until the lab edits them)
   and the microscope field of view.
5. UI style follows the AFM data viewer (UAnalysis).

Values in `config/` were **seeded from the user's attached documents** (Fab Manual by Sihong;
"Design – Autocad DRC.docx") with status `pending` and the exact quote as `source`, by the user's
choice. Values the user stated in the request are `approved`. Paper values used by templates are
`literature` (rules) or `suggested` (template fields), never `value`.

User decisions recorded on 2026-11-16:
- Seed config from the docs as *pending* (approve in the Rules dialog).
- Electrode width/spacing: **0.8 µm is the design target** (warning below it, rules
  `ebl.electrode_*_target_um`), the DRC doc's **0.5 / 0.4 µm are the hard minimum** (error,
  `ebl.contact_width_min_um`, `ebl.contact_spacing_min_um`).
- AI step: deterministic Python core; optional Claude assistant drafts stacks from plain English;
  Python validation is always the authority.
- **Configure** (v0.2, user request): the researcher picks layers (materials, functions, order); the
  program generates every empty dimension (Hall bar, flake sizes, thicknesses, gate geometry) so
  the spec validates. It only writes empty or `auto` values, derives each number from lab rules /
  lab design values by documented relations (`autoconfig.py` docstring), and asks once for what
  only the lab (rules, design thicknesses) or the researcher (layer counts, twists, gate voltages)
  can give. Generated values carry source `auto` and a `note` with their derivation.

User decisions recorded on 2026-11-16 (v0.3):
- **Mak layout** (stated by the user): Ti/Pt is pre-patterned on the chip as **two columns** of
  parallel fingers with a long thin gap between them; the BG graphite lands in the gap and never
  touches them; the WSe2 is wider than the bottom hBN and reaches past it onto the finger tips.
  Template order: substrate, PT, BG, BBN, WSe2 ×2, TBN, TG, CAP, CG, SG; `hall_bar.sd_style =
  "bent"`; W = the gap between the columns.
- **Bent source/drain** (user): when a back-gate strip runs the whole device length, the S/D turn
  90° near the ends, S to +y and D to −y (`HallBar.sd_style`: auto | straight | bent; auto bends
  only when Stage 2 finds the BG under the ends).
- Configure's relations live in `config/presets.json`, editable in **Edit preset dimensions** and
  Rules › *Configure presets*; the Rules dialog is read-only until **Edit** (bottom left).
- 100x images are **143.28 × 95.13 µm** (whole image; `presets.microscope.fov_100x_*`, approved);
  other objectives scale as 100 / magnification. µm/px = field width / image width in px (so a
  half-resolution LUT works). `defaults.microscope.um_per_px` in rules.json is no longer used.
- Lead router: the finger-array electrodes are routed since 2026-11-20 (`router.py`); the classic step keeps straight leads.

User decisions / data (v0.4, 2026-11-16):
- **Flake images** (revised 2026-11-17): a right-hand tab between Stack and Pattern (the panel);
  View shows the image in its own window; Trace flake… opens the tracing in its own window
  (`TraceDialog`), so panel, image and trace are three windows. Upload does not trace; **Trace all**
  traces every flake with an image, whole-flake mode, no click, keeping hand-made borders. The 3D
  stack of the real borders is the Structure tab's view (layer ticks / Only selected act on it).
  Flake layout stays for placing the flakes on each other. The Stack tab only links to it.
- **No rule on the relative thickness of hBN layers** (user, 2026-11-17): `xu.top_hbn_thickest`
  was removed from rules.json, the seed, validation and the templates. Do not add one back.
- When Configure ends invalid, a window stays open with every error and the values behind it,
  editable there (`validation.issue_parameters`), plus the values still needed.
- Devices: right-click → Rename (folder moves, paths rewritten) / Delete (confirm; removes the
  library folder only).
- The lab's WSe2 batch-1 data is in `WSe2 device training/` (not ours to change): "Layer OM Image"
  (one image per flake, stacks 1/3/4), dated OM folders, and `tWSe2-1_KLayout_session` (Stack 1/3/4
  GDS + .lys image placements). The user's GDS layer meanings for these files: 1 landing square,
  3 markers, 4 bond pads + fan-out, 5 pre-patterned fingers ("hall bars", two columns), 6 contact
  gate, 7 electrodes (`lab_designs.WSE2_BATCH1_LAYERS`; NOT entered in config/layers.json — ask
  before using them as the app's output layers). The structure is Xu TMD without the top
  encapsulating hBN.

User decisions (v0.5, 2026-11-17):
- **Finger array**: finger width 0.8 µm, "2.2 µm apart", 4 µm between the columns (user). "Apart"
  is implemented as the centre-to-centre **pitch** (the lab's drawings: pitch 2.2–2.3, edge spacing
  1.4–1.5) — reported to the user; flip `fingers.pitch_um` if they meant the edge gap. The number of
  pairs = what fits on the twisted region (`fingers.fit_finger_array`). A user-set column gap per
  device is a later feature. Other finger / comb / electrode numbers are presets measured from the
  lab's drawings (status *pending*, source = the design report) or program choices (status
  *program*) — the lab approves them in Rules › presets.
- **Stage 2 workflow**: pick an EBL step → its own drawing card; upload images up to this point
  (several magnifications, at least one of 5x/10x/20x); align to the bond pads; locate the flakes
  traced in Flake images; choose the image shown on the canvas; Generate draws the pattern on the
  canvas, editable; Export GDS per step; a global Merge & Export with the bond pads. Steps
  implemented: Hall bars, contact gate, electrodes to the bond pads (top / bottom gate merged into it, 2026-11-19).
- **Bond pads are a saved preset** (`bondpads.py`): built in = layers 1/3/4 of the tWSe2-1 session's
  Stack GDS (`tools/make_bondpad_preset.py`); the lab can save others (`library/bondpads/`).
- Output GDS layers for the new purposes (`prepattern_contacts`, `contact_gates`, `electrodes`,
  `bg_contacts`, `tg_contacts`) stay *missing* in config/layers.json until the user answers; the
  export asks, offering the tWSe2 batch 1 numbers (5/6/7) as a suggestion only.
- Measure tool (ruler) on the EBL canvas, Flake layout, the trace window and every image window;
  double-click clears it. Pointer tool: click-drag pans the view (not on a point handle).
- Border points (any editable outline, `canvas.PolygonItem`): click selects a point, Shift/Ctrl-click
  adds, dragging a selected point moves all selected; Delete/Backspace deletes them (≥ 3 points
  kept), Esc / a plain click on nothing deselects, Ctrl/⌘+A selects all. Tools `addpoint` (click on
  the border line, within 8 screen px, adds a point there — hold to drag it) and `select` (box
  selection, Shift adds). The trace window has both plus a trash button. A redraw after an edit
  keeps the selection (`Canvas._keep_sel`).
- Cursor readouts have a fixed width (`widgets.coord_label`): a label resizing with its text made
  the Flake layout canvas re-centre on every mouse move (the image "shook").
- Every list (incl. the Stack tab's layer list) uses `widgets.clickable_rows`: hover darkens a row, selection darker,
  pointing hand; `_row_style` covers QTreeView and QListView.
- List check boxes (`widgets.CheckDelegate`) are a thin grey square with a tick; the native indicator
  is switched off in `_row_style` (it drew a dark frame).
- (2026-11-17, user) **Any canvas**: a plain click away from every outline deselects it (`Canvas` pointer release
  → `set_active_polygon(None)`, `polygonSelected("")`); clicking a line selects it on every canvas (drag-moving it only
  where `drag_outlines`). The trackpad twist turns the view opposite to the gesture value (`view_angle − value`; user:
  the first sign was the wrong way). Slider: positive = counter-clockwise.
- (2026-11-17, user) **One traced border everywhere**: reshaping a flake's border in Flake images / Flake layout reshapes
  its Stage 2 outline at every step (`MainWindow._sync_outlines` keeps a cache of the borders and calls
  `EBLFlowPanel.design_outline_changed` → `ebl_flow.follow_design_edit`, through the old design → chip similarity;
  planned layers move their plan centre so they stay put); reshaping it on the EBL canvas / Draw outline window
  reshapes the border (`ebl_flow.design_from_chip_edit`, then the other steps). Only outlines that are the border moved
  vertex for vertex follow (`FOLLOW_HOW`, same point count); drawn / wand-traced chip outlines are shapes of their own.
  Moving / turning a border whole (`is_rigid_move`) never moves it on the chip.
- (2026-11-17, user) **Flake layout** shows the traced borders only (no images, no wand / image move / opacity): click a
  line to select (selects the layer), drag to move (its image moves with it, placed), points as in the trace window,
  the round rotate handle (since 2026-11-21, replacing the "Turn the selected border" slider; `_layout_edited` sees a rigid
  turn and turns the image with it, or keeps a session value without an image), the same view controls as the EBL canvas (`_view_rotation_row`).
- (2026-11-17, user) **Tabs**: Structure and Flake layout go with any side tab; only EBL design ↔ Stage 2 are coupled
  (Flake images moves the centre to Structure only from EBL design).
- (2026-11-17, user; replaces "ticked steps share one exposure") **Stage 2 cards**: every ticked step (`Flow.selected`)
  has its own card (`EBLFlowPanel._build_card`): its own images, alignment, flakes, twisted region (Hall bars),
  Generate / Check / Export GDS — steps no longer share state. The ^ at a card's top right folds it
  (`toggle_card`, session only). The card whose button was used last is "in use" (`current`, `_activate` binds its
  widgets — `CARD_ATTRS` — to the panel's attributes); the EBL canvas shows its flakes and edits them; every shown
  step's pattern is drawn editable. Clicking a step name opens its card (others stay) and scrolls to it. A step opened
  the first time inherits the flakes of the latest earlier step (process order). One GDS of several steps: Merge &
  Export (`ebl_flow.export_steps` still writes several steps for scripts).
- (2026-11-17, user) **Finger shape**: the fan keeps the presets' angle (pitch `far_pitch` at `fingers.length_um`) all the
  way out, so fingers fitted longer to the bottom hBN fan out wider (`FingerArray.far_u`); the last `far_width` µm of
  each finger is a stub along v joined by a convex hull, so the ends are square and those of a column lie on one line
  (a slanted cut left acute tips that failed the 0.5 µm width minimum).
- (2026-11-17, user) **Layers card** on the EBL canvas (layers button, top right of its toolbar,
  `MainWindow.show_canvas_layers` → `ebl_flow_panel.CanvasLayersCard`, a popup): ticks for bond pads / markers /
  landing square / image, each flake outline and the twisted region of the step in use, each step's pattern and the
  check marks; unticked keys go in `EBLFlowPanel.hidden` (session) and are not drawn (`canvas_layers`).
- (2026-11-17, user) **The stack decides what is on the chip at a step**: every layer below the step's own layer
  (`present_layers`) is done. A flake only planned (or projected) at an earlier step but present now is inherited as
  `EXPECTED_HOW` ("needs locating", earlier position shown dashed, Locate looks for it); `ebl_flow.flake_state` gives
  the list text.
- (2026-11-17, user report: WSe2 ×2 and the middle hBN not found at the contact-gate step of Device 2026-11-16)
  **Flakes dropped since the previous step are found in what changed** (`registration.change_map` /
  `locate_new_outline`): the previous step's aligned image and this step's are resampled in the chip frame (≥ 20x
  pair at the same objective, else the finest of each), the earlier one mapped onto the later by an affine Lab colour
  fit refitted on the best 60 % (exposure / white balance differ), ΔE opened + closed with a 1.2 µm disk
  (`CHANGE_THIN_UM`: the written fingers vanish), then the usual chamfer matcher on that map. Used first for
  `ebl_flow.new_layers(spec, key, previous_imaged_step)`; falls back to the photograph. On the lab device: WSe2 0.58 /
  0.62, middle hBN 0.56 (raw photo: 0.39 / 0.49 / 0.43, wrong places); the WSe2 land ≤ 0.4 µm from the user's plan
  (`tests/test_registration.py::test_lab_wse2_and_middle_hbn_found_since_the_hall_bars_step`, skipped without the
  device in `library/`). `located[uid]["method"]` records it.
- (2026-11-17) **Stale plans**: `ebl_flow.settle_plans` (run by `Flow.of`) turns plans on layers already on the chip at
  that step (left by the old shared state) into `EXPECTED_HOW`; `apply_plan` ignores them; `flake_state` reports
  "needs locating" for any "planned" layer that is present.
- (2026-11-17, user) **Rotate handle** (PowerPoint-like) in the trace window and the Draw outline window
  (`Canvas.rotate_handles`): a round handle 26 screen px above the selected outline's box (`RotateHandle`, ignores
  transformations, re-placed on zoom / pan / view rotation / edits); drag = turn about the area centroid, Shift = 15°
  steps; commits once on release (`polygonEdited`); `Canvas.rotate_active(deg)` for code. Canvas mouse events route a
  press on it before anything else.
- (2026-11-17, user) **Draw outline › Import border**: any flake's border traced in Flake images (this flake first; at
  its own size, centred where the outline is / the plan / the view) or this flake's chip outline at another step.
  `QDialog QPushButton { min-width: 76px }` in the house style sheet overrides code-set widths — set `min-width`
  with a style sheet on the button itself.
- (2026-11-17, user) **Twisted region = overlap of the device layers, kept up to date**: `ebl_flow.refresh_twisted` (run on
  every `EBLFlowPanel._store`, for every step) sets it to the overlap of all device-layer outlines whenever each has one
  (`TWISTED_AUTO`, or `PLAN_HOW` while planned); drawn / edited / cleared (`TWISTED_MANUAL`; Clear sets "cleared") are
  left alone until From located flakes or a plan.
- (2026-11-17, user) **No projection of layers not yet on the chip**: Locate projects (`project_flakes(..., uids=
  present_layers)`) only layers below the step's own layer; `settle_plans` drops old projections of layers above it.
- (2026-11-17, user) Pop-up canvases (Draw outline, trace window) use the EBL canvas's trackpad logic
  (`trackpad_gestures`) and have a back-to-0° button next to the zoom buttons.
- (2026-11-17, user) **Reopening shows the EBL canvas as saved**: Save stores `Flow.ui` = {current step, folded cards,
  hidden layers, canvas view (centre, px/µm, rotation: `Canvas.view_state`), tab}; `load_from_spec` restores them and the
  view is applied (`set_view_state`) on the first redraw with the canvas on screen; `open_device` and switching to EBL
  design redraw the canvas (before, nothing redrew it after opening).
- (2026-11-17, user) **Hall-bar card**: "From located flakes" removed. `ebl_flow.adopt_identified` (every `_store`): a layer
  planned at a step and found on the chip at a later one (located / edited / placed / drawn / traced) takes that border
  at the earlier step too (`IDENTIFIED_LATER`, plan dropped), and that step's twisted region follows the actual overlap.
- (2026-11-17, user; revised after the user's own fix on Device 2026-11-16) **Contact gate**: "touching the Hall bar"
  = touching an *ungated* (source / drain) finger. The bus lies behind the teeth (as in the lab's drawings and the user's
  edit) and crosses only gated fingers. Without a `source_drain` layer the outer finger(s) of each column at opposite
  ends stay ungated and each bus leads away from them (`CG_LAYOUTS`: top-left / bottom-right first, as the lab's Stack 1,
  then reversed). `fit_contact_gate` copes as the user did by hand: for 1, 2, … (`CG_MAX_SKIP` = 3) ungated fingers per
  column (same number in both), each layout, each lead from `cg.bus_extension_um` down to one electrode width — first
  untrimmed, then with the bus cut to the hBN (`_trim`: opened by half a finger width, teeth whole, one piece, ≥ the
  hard width minimum, the electrode landing end whole) — so the comb is `cg.hbn_margin_um` (preset, asked) inside the hBN
  under it (`ebl_flow.hbn_under`). On the user's device (1 µm margin) it reproduces their design: 2 + 2 ungated, top lead
  to +u, bottom to −u, bottom bus corner trimmed (their own drawing is 0.6 µm inside). `FlowStep.comb` records the
  layout; the Electrodes step lands on the contact gate as it is (`fingers.cg_terminals`, also when drawn).
- (2026-11-17, user) **Check reads the pattern as it is** (generated, edited on the canvas or drawn): contact-gate rules
  use `fingers.gated_fingers` (≥ half of a finger's tip zone covered) — off the ungated fingers, S/D at opposite ends,
  off the channel (10 nm rounding tolerance), on its hBN. Editing a shape clears the old findings ("edited — Check again").
- (2026-11-17, user) **Draw contact gate… / Draw fingers… / …** (`ui/shape_window.ShapeDialog`, a button in each step
  card): the step's pattern over an aligned image, the other steps' patterns for reference; polygon tool adds a shape,
  pointer moves / reshapes, rotate handle, Delete shape, Check (`EBLFlowPanel.add_shape / set_shape / delete_shape`).
- (2026-11-17, user) **Check opens a rule list** (`ui/check_window.PatternCheckDialog` ← `ebl_flow.rule_report`): rule,
  required, found, status (met / violated / below target / value missing / not checked); missing values are typed in
  and saved to config/ (`apply_answers`, incl. `preset:` keys); rows with a place centre the canvas. Contact-gate rule
  violations are also errors in `check`. Export runs the checks without the window (`_run_check`).
- (2026-11-17, user) **Process order** is the user's: `Flow.order` (saved), drag a row in the step list or the ↑/↓
  buttons (`move_step`); numbering, inheritance, merge list and "previous images" follow it.
- (2026-11-17, user) **Image on canvas** is its own card after EBL Steps: any step's image at any magnification
  (`Flow.canvas_image` = [step, objective]; None = the finest image of the step in use, `FlowStep.show_image`).
  Another step's image is keyed `step:<step>/<obj>` on the canvas (Manual alignment has its own window since 2026-11-20).
- (2026-11-17, user) **Finger length** = as long as possible: each column (top / bottom) runs until every finger is
  `fingers.bbn_margin_um` inside the located bottom hBN (`fingers.fit_finger_lengths`, coarse 1 µm walk + bisection to
  0.05 µm; `FingerParams.length_top/length_bottom`, saved with the array, kept by the later steps). The margin is a
  preset with no value (status missing): Generate asks for it (`preset:` MissingItem key → `config.set_preset`).
  Without a located bottom hBN the fixed `fingers.length_um` is used. Width, pitch, straight part, fan unchanged.
- (2026-11-17, user) **Angles can be typed**: `widgets.AngleControl` (slider + number box, wraps to ±180) for the view
  rotation (EBL, Flake layout), Turn selected border and the plan rows. **Editable numbers look editable**: table
  cells that can be edited are drawn as text boxes (`widgets.boxed_cells` / `BoxedCellDelegate`; presets, Rules in Edit
  mode). Preset dimensions opens editable (no Edit button; Save / Close asks to save). The Rules dialog keeps Edit.
- (2026-11-17, user) **Trackpad**: a twist turns the view only after 3° in one gesture (`TWIST_START_DEG`); from then
  until the fingers lift (+0.35 s of momentum) two-finger scrolling does not move the view (as on an iPhone).
- (2026-11-18, user) **Contact-gate bus as close to the WSe2 as allowed**: the bus (and its lead) keeps
  `cg.bus_tmd_gap_um` (preset, missing → asked on Generate) from the TMD (all device-layer outlines at the step,
  `ebl_flow.tmd_outlines`, grown with `geometry.grown_round` — `sized` corners reach too far); each column's teeth are as
  short as that allows (`fingers.tooth_to_keepout`, ≥ `fingers.min_overlap_um`), so the columns differ; `cg.tooth_length_um`
  only without a located TMD. Teeth follow the finger (`_tooth_local` = finger ∩ band) also where it fans; there the side
  meeting the bus at an acute angle gets a fill `TOOTH_FILLET_UM` below the bus (the tooth grows by it so the fill keeps
  the gap). `FlowStep.comb["tooth"]` records the lengths. `gated_fingers` now tests the first `min_overlap` µm of each
  finger. Check row "Contact-gate bus off the TMD" (comb − fingers − channel, opened 10 nm; warning when closer). On
  Device 2026-11-16 (gap 0.5, margin 0.5): teeth top 7.19 / bottom 5.83 µm, every rule met.
- (2026-11-18, user) **GDS export with the flake outlines**: a tick box in every step card and the merge card ("Include the
  traced flake outlines", one shared choice, `Flow.ui["export_flakes"]`); `ebl_flow.flake_shapes` = each layer's outline at
  the latest chosen step, purpose `flake:<uid>`; its GDS layer is asked once (`layer:flake:<uid>`, suggested 100 + stack place since 2026-11-18) and kept
  in config/layers.json; a flake layer shared with another purpose is refused (`_check_flake_layers`).
- (2026-11-18, user) **Stretch / scale handles** (PowerPoint-like) on the selected outline in every canvas (EBL, Flake
  layout, trace window, Draw outline, Draw contact gate…: `Canvas.scale_handles`): eight squares (`ScaleHandle`) 10 px outside
  the outline's box as seen on screen (a border point under one wins the click; the stretch counts from where the handle was grabbed, no jump); a side handle stretches across / up–down as seen (with a turned view, along the
  screen axes), a corner both; Shift on a corner keeps the proportions, Alt / Option scales about the middle; one commit on
  release (`polygonEdited`, so the border change follows everywhere); `Canvas.scale_active(sx, sy)` for code. Placed
  together with the rotate handle (`place_rotate_handle` → `place_scale_handles`).
- (2026-11-18, user answers) **Finger "2.2 µm apart" = centre-to-centre pitch** (as implemented). **100x field of view =
  143.28 µm** (measured in the microscope; keep it, not the ~140.6 µm of the lab's KLayout alignments). **Suggested output
  layers** (`ebl_flow.suggested_layer`, offered in the layer dialog — the user confirms, nothing is written by itself):
  fingers 5/0, contact gate 6/0, electrodes 7/0 (the lab's tWSe2 batch 1), back-gate and top-gate contacts 7/0 with the
  electrodes (the lab's drawings put the BG contacts on 7); a metal top gate = the first layer no purpose uses (program
  suggestion); a flake outline = `FLAKE_LAYER_BASE` (100) + its place in the stack, moved past layers in use (program
  suggestion; flakes asked together get different numbers).
- (2026-11-18, user) **Library panel**: Templates has only **Edit template** (primary, bottom left, in line with New device; right-click a template:
  Edit / New device from it / Delete for lab templates). Devices has **New device** and **Delete** (the selected device,
  after confirming) at its bottom; Save device (the title bar has it), Open and Describe… are gone from the panel.
  **Edit template** (revised 2026-11-19, user: not in the Structure / Stack tabs) opens its own window
  (`ui/template_editor.TemplateEditorDialog`): name, description, structure rules, contacts, S/D style, the layers (the Stack
  tab's `StackPanel` on the template's copy through `_TemplateHost`; no flake-image button), 3D preview + validation summary.
  The open device is never touched. Save overwrites the template (revised 2026-11-21, user; `templates.update_template`, id
  kept, no name asked): a lab template's own file; a built-in's edit is written to `library/templates/<built-in id>.json`,
  which takes the built-in's place in `list_templates` (`TemplateInfo.overrides`, list label "· edited") and in
  `load_template(id)`; the shipped file (tools/make_builtin_templates.py) is never changed; right-click › **Restore built-in…**
  deletes the edit. (New device › Edit Template never changes the template.) Closing with changes asks to save. **New device** has three tabs: from a template, from scratch, **from
  description** (`dialogs.DescribePage`, the assistant; `AssistantDialog` is kept, unused, for later — the user's wish).

- (2026-11-19, user) **Top / bottom gate steps removed; their electrodes are part of "Electrodes to the bond pads"** (TMD-like
  samples: the gates are graphite flakes, so a gate step was only a contact, evaporated with the other electrodes at the end).
  `ebl_flow.ELECTRODE_GROUPS`: Hall bar electrodes (`el_fingers`), Contact gate electrodes (`el_cg`), Top gate electrodes
  (`el_tg`), Bottom gate electrodes (`el_bg`); a group shows only when the stack has its layer (`electrode_groups`: functions
  hall_bar_contacts / contact_gate / top_gate / bottom_gate). Electrodes card › Pattern: one sub-card per group with its own
  Generate + Clear (`generate(..., group=g)`, the other groups and their pad leads kept; `FlowStep.group_notes`), Generate all
  (primary; gates first), then one Check and one Export GDS. Every group goes on the one GDS layer `electrodes` (`GDS_PURPOSE`,
  `gds_shapes` — export, Merge, width / spacing checks). A gate electrode (`gate_contact` since 2026-11-20, then routed): a `electrodes.bg_contact_um`
  square on the gate where no insulating flake above it (on the chip at the electrodes step, located) covers it, ≥
  `electrodes.gate_contact_clearance_um` from conducting flakes above it, the fingers, the comb, the twisted region and the
  other electrodes, as close as allowed to the nearest free pad tip, + a straight electrode to it; one per gate. Blocked only
  by other groups' leads → `BlockedByElectrodes`; the panel asks, then draws the gate first and those groups after it.
  Check rows: each group drawn; gate electrode on the exposed gate / off the other conductors / flakes above not located;
  "Other electrodes off the exposed <gate>" (an electrode over uncovered graphite shorts to it). Saved flows: the old gate
  steps' images / alignment / outlines move to the electrodes step when it has none (`Flow.from_json`); old one-purpose
  electrode shapes become `el_fingers`. A metal top gate is no longer drawn in the finger-array workflow (classic step only).
- (2026-11-19 / 20, user) **Flake layout and Structure follow Stage 2 for what is fabricated**: `ebl_flow.on_chip_outlines` =
  each flake on the chip at the latest step (process order) where it is below the step's own layer and its outline is the
  flake there (how not in `NOT_ON_CHIP_HOW`: planned / projected / expected); `layout_frame` = Flake-layout → chip map
  (rotation ± reflection, scale 1) from the lowest such flake whose chip outline is its traced border moved (it keeps its
  Flake-layout place; else centroid shift; else origin); `layout_view` = those flakes in the Flake-layout frame. Flake
  layout draws them dashed and locked (not editable, no turn slider, status says why); the flakes still to come stay
  movable there. `structure_footprints` (Structure tab, `build_scene(..., chip=)`): the fixed flakes, the others where Flake
  layout has them (so moving one there moves it in 3D), the patterned metal from Stage 2 (`PATTERN_LAYERS`: fingers,
  contact gate, electrodes; not generated = nothing drawn) mapped into that frame and clipped to the flakes' box +
  `STRUCTURE_WINDOW` (display only), centred on the twisted region. Title "· with Stage 2" (tooltip lists the fixed flakes).
  (2026-11-20, user) **The traced border takes the chip shape**: `ebl_flow.sync_layout_from_chip` (every
  `EBLFlowPanel._store` and on load, `_sync_layout`) overwrites `Layer.outline` of each fabricated flake with its chip
  outline in the Flake-layout frame (so Flake images, Flake layout and the 3D stack show the shape found on the chip); the
  image follows the rigid part (`_follow_image`) and its trace is marked how = "chip" (in `flake_images.HAND_MADE`: Trace
  all keeps it); the layers are marked synced (`MainWindow.outline_synced`) so the change is not carried back onto the
  chip. **Flake layout ↔ plan linked** (when a fabricated flake sets an unmirrored frame, `_plan_frame`): a flake not on the
  chip yet moved / turned in Flake layout → its plan at every step = its border through the frame (`plan_from_layout`, from
  `MainWindow._sync_outlines` on a rigid move → `EBLFlowPanel.layout_moved`; a new plan starts there too); a plan dragged
  on the EBL canvas → the border moves (`layout_from_plan`, from `set_outline`); the plan slider then shows / sets the
  flake's angle on the chip (frame turn + the border's turn, `plan_angle`) and turns the border (`MainWindow.turn_border`),
  the plan following. Without a frame (nothing on the chip, or mirrored) plans keep their own turn and position as before.
- (2026-11-20, user) **EBL canvas › Layers › "Not On The Chip Yet"**: a tick per flake above the step's own layer with no
  outline at the step (`ebl_flow.not_yet_on_chip`: its Flake-layout border through the frame), off by default
  (`EBLFlowPanel.shown_plans`, saved in `Flow.ui["shown_plans"]`, `layer_shown`); drawn dashed, not editable.
- (2026-11-20, user) **Flake layout follows the Flake images ticks** (the 3D stack's hidden set: `Stack3DView.set_hidden` now
  emits `hiddenChanged` → `refresh_layout_canvas`) and has **Show metal parts** (`MainWindow.layout_metal`): the Stage 2
  fingers / contact gate / electrodes in the Flake-layout frame (`ebl_flow.layout_patterns`, also used by the Structure tab),
  not editable, hidden with their layer's tick.
- (2026-11-20, user) **Contact-gate bus led to the middle hBN's far corners** (`fingers.extend_buses`, after `apply_trim`):
  from each bus's outer end a bus-wide straight lead, slightly turned, toward the hBN corner in direction (lead side, ±v
  outward) — top and bottom columns at opposite corners (on Device 2026-11-16: top → top-right, bottom → bottom-left) —
  ending as close to the edge as `cg.hbn_margin_um` allows; it keeps off the TMD keep-out (gap already in it), `ebl.contact_
  spacing_min_um` from the ungated fingers and the other comb, and near the bus end never nearer the gap than the bus's inner
  edge (a turned square end would poke out). Candidates = vertices of the allowed region (margin + half the lead width
  inside the hBN) by distance along the corner direction; the corner, else the farthest point toward it that is clear.
  `FlowStep.comb["reach"]`; the note says where each lead ends and warns when one lies under a located top hBN.
  (2026-11-20, user: width as large as possible, narrower only where needed; smooth, no neck) The lead is a straight line
  whose **width varies along it**: every part is as wide as the room there allows, at most `cg.bus_width_um`, never below
  `ebl.contact_width_min_um` (the hard minimum) — it keeps the bus width wherever that fits, narrows only through a tight
  stretch and widens again after it. Room = distance transform of the free area (allowed − blocked + own comb, raster
  `LEAD_RASTER_UM` 0.02, sampled every `LEAD_SAMPLE_UM` 0.05 along the line, narrowest within ±`LEAD_SMOOTH_UM`), width
  changes limited to `TAPER_MAX_HALF_ANGLE_DEG` (15°) per side both ways, the profile simplified to straight runs lowered
  under the room (`_straight_runs`, `profile_path`); it leaves the bus at the bus width (no neck). Targets = vertices of the
  allowed area shrunk by half the minimum / middle / bus width + a ±80° fan about the corner direction; each ray is taken as
  far as it fits, and also stopped where its end starts narrowing into a corner (a wide end). Choice: the farthest; among
  ends within max(0.25, bus width − hard minimum) µm of it, the widest narrowest part, then widest on average. Exact check
  (`_segment_ok`, with spacing to its own comb), retried with a 0.02–0.1 µm pad; the lead is clipped exactly to the allowed
  area. `comb["lead_width"]` = narrowest part, `comb["end_width"]`. Below the design target → Check's "below target"
  warning. On Device 2026-11-16 (margin 0.5, gap 0.5): top to (38.3, 18.7), 1.6 µm until the hBN's thin tip (0.5 there);
  bottom to (−21.6, −11.0), 1.6 µm except 1.32 between the WSe2's left tail and the hBN edge.
- (2026-11-20, user) **Layers card**: Show all / **Hide all** / **Only image** (`CanvasLayersCard._set`). The step cards no
  longer offer "Use the images of '<step>'" (each step loads its own images; `copy_images` removed).
- (2026-11-20, user) **Locate reuses earlier steps' flakes**: `ebl_flow.reuse_earlier_locations` (run first by Locate) gives
  each flake on the chip at the step the outline of the latest earlier step (process order) where it was on the chip and
  found (`located_here`: an outline whose how is not planned / projected / expected; record copied, `inherited` = that
  step) — a flake buried by later ones may no longer be visible. Locate then looks only for the flakes still without a
  location (e.g. top hBN and top gate at the electrodes step). Kept: flakes drawn / edited / placed / traced by hand at this
  step. **Clear** at a step marks the flake `{"relocate": True}`: the next Locate looks for it in this step's images.
- (2026-11-20, user) **Electrode router** (`ustack/router.py`, used by every electrode Generate): no electrode crosses another
  or a gate flake. `ebl_flow._route_electrodes`: each group's terminals (`_group_terminals`: a finger's fixed part from the
  landing point along the finger + one electrode width out along it; a contact-gate bus end; a gate contact from
  `gate_contact`, counted as the disk any turn of it fits in while routing, then turned along its wire) → pad leads by the
  order-preserving `stage2.assign_tips` → `route_nets`: A* on a 0.5 µm grid (`ROUTE_RES_UM`), 8 directions, net by net;
  **hard** = other nets' wires and fixed parts, fingers, combs (one obstacle per piece), bond-pad leads, landing squares,
  markers, the *uncovered* part of every conducting flake on the chip (a short), each grown by its clearance + half the
  width + ¾ cell; **soft** (×`SOFT_PENALTY` 60) = conducting flakes under an insulating one (leakage) and the device
  (twisted region + device layers). A net's own items: owned by it (its pad lead, its gate) or hard ones touching its fixed
  part. Path pulled taut (`_taut`, `_drop_corners`: no hard cell, no more soft length) → straight pieces at any angle, round
  joints (`router.wire`). Blocked nets are routed first on a retry (rip-up, 3×); still blocked → not drawn, reported.
  Spacing: `ebl.electrode_spacing_target_um`, else the hard minimum (noted); gate flakes: `ebl.contact_to_graphite_gate_min_um`.
  A conducting flake counts when below the electrodes in the stack **or found at the electrodes step** (a top gate above
  LEADS in the template is still there when its electrodes are written; also in `gate_surroundings`).
  **Landing square** (user): every routed electrode enters, straight along the lead, a square of `electrodes.pad_landing_um`
  (preset, *missing* → asked on Generate, hint with the preset's lead width / spacing) centred on the end of its bond-pad
  lead (`gds_io.lead_ends` / `BondPads.pad_ends`: mid of the end edge, direction into the lead; same order as `pad_tips`).
  Check rows: "Electrodes do not cross each other", "Other electrodes off the covered <gate>" (warning), "Electrodes land on
  their bond-pad leads". Routes record `soft_crossing_um`. On Device 2026-11-16: 23 electrodes in ~2 s, spacing met, no
  crossing; the contact-gate leads start on their combs over the covered bottom gate, so they cross it briefly (warning).
- (2026-11-20, user) **Manual alignment** (white button, replaces Pick pairs / Fit pairs; `ui/align_window.AlignDialog`): its own
  window with the bond-pad pattern and one image of the step (combo; opacity slider). The image's frame is an editable outline
  with the rotate / scale handles: every edit is refitted to a similarity (`fit_similarity` of the image corners → the edited
  frame) and is the step's alignment at once. **Auto scale** (bottom; on for an image not aligned yet, off for an aligned one)
  keeps µm/px = field of view / width (`fit_similarity(..., scale=)`: only the turn and the shift fitted, one pair enough).
  Pick pairs / Fit pairs / Clear pairs work inside the window (green = image point, red = pattern point). "Image mirrored"
  follows the card's tick. The panel's old canvas pair-picking code (`start_pairs`, `fit_pairs`, `_picked`) is unused.
- (2026-11-20, user) **Bubbles, cracks and stray flakes** (`ustack/debris.py`; Electrodes card, after Locate flakes: **Find
  bubbles, cracks, flakes** (primary), **Draw / edit…** (`ui/debris_window.DebrisDialog`), Clear). `find_in_image`: CIELab at ≥
  `WORK_UM` µm/px; known = metal (bond pads, markers, earlier steps' patterns; grown max(`KNOWN_GROW_UM`, 8 px) — the pads' glow
  at 5x) + the stack (every flake on the chip at the step, `ebl_flow.debris_inputs`); off the stack: ΔE from the substrate
  (main colour, quadratic illumination fit) > max(`MIN_DE`, `SIGMA_K`·σ); on the stack (away from its outlines' edges): ΔE from
  the local median (`LOCAL_UM`); pieces ≥ `MIN_AREA_UM2`; region = inside the bond-pad leads' ends (hull). Features per piece
  (area, elongation, solidity, contrast, lightness sign, texture, edge sharpness, on stack, mean width, **sides** = colour
  difference across a thin piece: a fold has the same flake on both sides, a layer edge does not). **Classifier** = k-NN (k 5)
  over labelled examples: built in = `ustack/data/debris_examples.json`, 180 candidates from the lab's Stack 1 / 3 images before
  the electrodes (20x, 100x; 2026-11-02; placed by the KLayout session), labelled by eye on contact sheets
  (`tools/debris_training.py sheets|build`, labels `tools/debris_labels.json`, report `docs/debris_training.md`: leave-one-out
  83 %, rules alone 55 %; cracks are the weak class, 11 examples) + the lab's corrections (`library/training/debris.json`,
  written by the window: deleting a found one = "none", changing a kind or drawing one = that kind, features via `describe`);
  `rule_kind` where nothing labelled is near. Kinds flake / particle / bubble / crack; "none" dropped. Several images:
  `merge` (the finest image decides inside its frame). Stored in `FlowStep.debris` [{kind, polygon, score, how auto|drawn|edited,
  image, features}]; Find again keeps the hand-made ones (`set_debris`). EBL canvas: dashed by kind, Layers › "Bubbles,
  cracks, stray flakes".
- (2026-11-20, user) **Router keeps off them**: `_route_obstacles` adds each one with `ebl.defect_clearance_um` (the lab rule,
  *missing* → asked on Generate when there are any): stray flakes / bubbles / cracks hard but relaxable, particles soft.
  A lead with no way round (A* target unreachable) is relaxed first only near its ends (`RELAX_NEAR_UM` 25, only the part of
  the obstacle there: `_part_near`), the relaxed set grows, then every lead, then anywhere — a relaxed obstacle costs
  `RELAX_WEIGHT` (4) × a soft one. Pads whose landing square lies on one cost `PAD_ON_DEBRIS_COST` in `assign_tips`
  (`tip_cost`). The gate contact keeps off them (`gate_surroundings` keep). Check row "Electrodes off bubbles, cracks and stray
  flakes" (on = violated, within the clearance = warning). Router speed-ups: flat-array A* with an octile heuristic, a flood-fill
  reachability test before each search, both sweep orders round the device (the better kept), `SOFT_PENALTY` 12, 2 rip-up
  retries, finger exits reserved `EXIT_STUB` (2.5) electrode widths. On Device 2026-11-16 (clearance 1 µm, landing 10 µm —
  test values in a scratch config): 123 found, 23 electrodes routed in ~25 s; two crossings left, reported: a fold on the hBN
  across the bottom finger exits and a dark patch on the stack classified as a flake.
- (2026-11-20, user) **Error circles are clickable** (`canvas.IssueMarker`, `Canvas.markerClicked`, keys `step|issue|place`):
  `EBLFlowPanel.issue_popup` → `ui/issue_popup.IssuePopup` (a small pop-up at the cursor: the message, the place, the rule
  behind it) with **Close**, **Change rule…** (the value behind it in a MissingValuesDialog, saved to config/ as the user's,
  then Check again; disabled when no lab value is behind the error) and **Accept** (`ebl_flow.accept_issue`: the place goes
  into `FlowStep.accepted`; `check` → `_apply_accepted` takes accepted places out of the errors — an error with none left
  becomes severity "accepted": no red circle, export not stopped; a place moved more than `ACCEPT_TOL_UM` is checked again).
  Issues carry `rule` (a MissingItem key): `check_pattern` width / spacing, and rule-report rows via `_row(rule_id=)` /
  `ROW_RULES` (text → key).
- (2026-11-20, user) **Layers are never features**: `debris.layer_veto` (before the classifier): a long piece on the stack whose
  two sides differ (`sides` ≥ `STEP_EDGE_SIDES` 0.6, elongation ≥ 2.5) is a step edge between layers; a piece with ≥ `HUG_MAX`
  (0.6) of its area within `EDGE_HUG_UM` (2.5) of a located outline is that layer seen past its outline; a piece off the stack
  ≥ `UNLOCATED_LAYER_UM2` (100 µm²) touching it is a layer not located. `EDGE_BAND_UM` 1.5 (≥ 3 px). Find warns about layers on
  the chip at the step that are not located (they look like stray flakes). Masks are filled polygon by polygon (OpenCV
  `fillPoly` with several leaves their overlap empty — that also hit `router._Grid.crop_mask`). The training tool's sheet
  index keeps each candidate's place; `build` matches labels by place.
- (2026-11-20, user) **Every step card** has the Bubbles, cracks and stray flakes section (`DEBRIS_HINTS`). **Later steps
  replace earlier ones**: `FlowStep.debris_looked` (set by Find / editing; Clear unsets it), `ebl_flow.debris_source` /
  `debris_at` = the step's own list once it has looked, else the latest earlier step's (process order) — used by the router,
  the gate contact, the checks, the canvas; editing an inherited list copies it into the step first. Saved flows: a list
  present = looked.
- (2026-11-20, user) **Hall bars and contact gate keep off bubbles and cracks** (`AVOID_KINDS`; `defect_pieces` / `defect_region`
  grown by `ebl.defect_clearance_um`, asked when there are any): the finger array is fitted on the twisted region minus them —
  all at once, else one at a time (largest first) while as many finger pairs fit as without them; each column ends before
  one (`fit_finger_lengths(avoid=)`) when that still leaves `land_from_tip` + a finger width. The contact gate:
  `fit_contact_gate(avoid=)` — no part of the comb may touch them (trimmed bus too: `apply_trim(avoid=)`; the device keep-out
  still sets the teeth length), all at once, else one at a time; the bus leads keep off them (`extend_buses` keepout + avoid).
  Notes say which could not be avoided; Check rows "Fingers / Contact gate off bubbles and cracks" (warning).
- (2026-11-20, user) **Several rows in every list**: `widgets.clickable_rows` sets ExtendedSelection (Shift-click = a range,
  Ctrl / ⌘-click = one more; `PickTree` keeps Shift = toggle); the issue list too (tick boxes since 2026-11-20, below). Acting on several: layer list Delete /
  Duplicate (`StackPanel.selected_uids`), Devices › Delete (one question for all), Stage 2 flake list › Clear
  (`EBLFlowPanel._selected_flakes`); single-row actions (Draw outline, Edit template, …) use the current row.
- (2026-11-20, user: obvious cracks on Device 2026-11-16's bottom hBN were missed — found as candidates, then called "not a
  feature" by the k-NN, whose few long examples were mostly layer edges) **`debris.is_crack_shape`**: a piece on the stack,
  elongation ≥ `CRACK_ELONG` (2.5), both sides alike (`sides` ≤ `CRACK_SIDES` 0.3), mean width ≤ `CRACK_MAX_WIDTH_UM` (3) is a
  crack whatever the examples say (`classify`, after the layer vetoes).
- (2026-11-20, user) **Fingers need not be the same length**: `FingerParams.lengths` {"top:i": µm} (saved with the array, kept by
  the later steps), `length_at` / `straight_at`; `FingerArray.finger(column, i, length=)` builds one finger (`fingers()` = all).
  `fit_finger_lengths` fits each finger on its own (walk + bisection, single-finger geometry) inside the bottom hBN by the
  margin and off `avoid`; returns {"top", "bottom" (longest of each column), "fingers"}. The Hall-bar step stops each finger
  before a bubble / crack when that leaves `land_from_tip` + a finger width (notes name the fingers, `_finger_names`). The fan
  angle per finger does not depend on its length, so fingers of different lengths never meet. `length_top` / `length_bottom`
  = the longest of each column (display, keep-out reach). This replaces "the far ends of a column on one line" (2026-11-17).
  On Device 2026-11-16: top 18.3–25.2 µm, bottom 23.3–38.2 µm (most fingers stop before the folds on the bottom hBN).
- (2026-11-20, user) **Step card buttons**: Generate · Draw … · Check in one row (electrodes: Generate all · Draw electrodes… ·
  Check), then "Include the traced flake outlines" and Export GDS… below. The Bubbles section's button is "Find".
- (2026-11-20, user) **Tick box on every list row** (`widgets.SelectBoxDelegate` + `_BoxClicks`, set by `clickable_rows(boxes=True)`):
  a box at the left of the first column shows whether the row is chosen; clicking it adds / removes that row (any set);
  Shift / Ctrl-click as before. Not on lists whose ticks mean something else (step list = steps in use, Flake images =
  visibility) nor on New device's template list (one to start from): `boxes=False`.
- (2026-11-20, user) **Drawing windows** (Draw outline, Draw fingers / contact gate / electrodes, Bubbles › Draw / edit):
  `Canvas.box_select_outlines` — a pointer drag on nothing draws a box; every editable outline wholly inside is chosen
  (`Canvas.multi`, drawn heavier; Shift adds; `outlinesSelected`); dragging one of them moves all (`_group_move`, each commits);
  Delete / Delete shape / the Kind box act on all. `Canvas.allow_twist = False` there (two-finger scroll / pinch still move /
  zoom; the back-to-0° button is gone). The EBL canvas keeps drag-to-pan and the twist.
- (2026-11-20, user) **Patterns found written at the next step**: contact-gate card › "Locate the fingers", electrodes card ›
  "Locate the contact gate" (`ebl_flow.LOCATABLE`), "Use as designed" undoes. `registration.locate_pattern`: oriented chamfer
  match of the pattern's outline on colour edges (`_colour_edges`: per pixel the CIELab channel that changes most — metal on
  hBN can have the hBN's grey level), ±`PATTERN_SEARCH_UM` 3 µm, ±`PATTERN_MAX_TURN_DEG` 1°, coarse grid then refined;
  ≥ `PATTERN_MIN_INLIERS` (0.45) of the outline on an edge; images coarser than `PATTERN_MAX_UM_PX` (0.25: 5x / 10x) refused;
  mirrored images handled. `locate_written` (finest image holding the whole pattern) → `FlowStep.found_patterns[source]` =
  {dx, dy, angle, cx, cy, inliers, image}. `pattern_offset(flow, key, source)` = the latest step ≤ key that found it;
  `pattern_shapes` / `_array_of(flow, key)` / `move_points` give the pattern as written; used by the contact gate (comb on the
  fingers as found), the electrodes (finger landing points, comb terminals), gate contacts, the router's obstacles, the bubble
  finder's known metal, Check (`others`) and the canvas (dashed). Lab data: Stack 1 / 3 before the electrodes, 20x / 100x:
  fingers and contact gate within 0.05–0.35 µm of the lab's GDS (the same offset for both in one image: mostly the image's own
  alignment error, which the card's note says); on Device 2026-11-16 nothing is found — its saved finger design is not the
  pattern that was written (straight parts agree, the fan does not).
- (2026-11-20, user) "Locate the fingers / contact gate" is blue (primary). **Use as designed** is a choice kept at the step:
  `found_patterns[src] = {"as_designed": True}`; `pattern_offset` stops there and returns None, so from that step on the
  pattern is where it was designed — the bubble finder masks it there (`debris_inputs` → `pattern_shapes`), the next patterns
  are placed on it — whatever an earlier step found (that step keeps its own). The card says which holds.
- (2026-11-20, user: "Use all suggestions" did nothing in the GDS export's dialog) `dialogs.MissingFields.use_all` fills GDS
  layer fields (both numbers) too, besides numbers and yes / no; it is the only "use all" button (Configure's window has none).
- (2026-11-20, user) **Export GDS… / Merge & Export ask where** (`ui/save_gds.SaveGdsDialog`, `ask_gds_path`): the device's folder
  (`library/devices/<device>/stage2/<step>` — the default) or another folder (Browse…; remembered in QSettings "gds_dir"), and
  the file name; `export_step(s)(..., file_name=)`.
- (2026-11-20, user) **One lead at most over a stray flake** (`router.Obstacle.exclusive`, set for `ebl_flow.CONDUCTING_DEBRIS`
  = flake, particle — bubbles and cracks do not conduct): once a routed lead touches one, it is hard for every other lead
  (`claimed` in `route_nets`; bounding-box prefilter); a lead with no other way is rerouted (rip-up) or reported unrouted,
  never shorted. Check row "No two electrodes over the same stray flake". A lead's own fixed wire (`Net.stub`) is an obstacle
  to its own route beyond `SELF_CLEAR_UM` (2) + its width from where the route starts (no doubling back beside its own stub —
  the notch the spacing check flags). On Device 2026-11-16 (fingers ending before the cracks) Generate all takes ~45 s: the
  router needs its relax rounds there.
- (2026-11-20, user) **Contact-gate electrodes overlap the bus**: preset `electrodes.cg_overlap_um` = 3.7 µm (*pending*; measured in
  the lab's tWSe2-1 GDS: layer 7 on layer 6, Stack 1 2.95 / 4.52 / 7.48, Stack 3 1.57 µm). The electrode's fixed part runs
  that far back from the bus lead's end (`fingers.back_along`: of the directions within 60° of straight back, the one that stays
  inside the comb longest — the lead's own axis, turned or not; shortened where it ends).
- (2026-11-20, user: the finger electrodes overlapped the fingers too far and the gold landed on the top hBN) **Finger
  electrodes overlap a bit, off the flakes above**: `ebl_flow.finger_landings` → `fg.finger_terminals(arr, land=)`: each
  starts at max(`land_from_tip`, finger length − `electrodes.finger_overlap_um`) (preset 10.9 µm, *pending*: the lab's
  layer-7-on-5 overlap, median 12.17 / 9.69 µm in Stack 1 / 3), then further out until its metal (`router.wire`, flat end)
  lies `electrodes.flake_clearance_um` (preset, *missing* → asked on Generate) outside every flake above the fingers' layer on
  the chip at the electrodes step (`flakes_over_fingers`: top / middle hBN, the device layers, a graphite top gate — that one by
  at least `ebl.contact_to_graphite_gate_min_um`; `finger_keepout`); walk 0.5 µm + bisection to 0.05. No clear stretch →
  the last electrode width of the finger, noted. Insulating flakes above the fingers are soft router obstacles. Check row
  "Hall-bar electrodes off <flakes>" (on = violated → error, within the clearance = warning) and "Flakes above the fingers
  located". On Device 2026-11-16 (clearance 1 µm, test value): gold on the top hBN 22.5 → 0 µm², overlaps 3.8–10.9 µm.
- (2026-11-21, user) **New device › Edit Template**: from a template, "Edit Template" (bottom left, only on that tab) is the
  next step in the same window (`NewDeviceDialog.steps`: choose → `edit_page`): the chosen template's layers, structure
  rules, contacts and S/D style in `ui/template_editor.TemplateEditor` (the widget Library › Edit template's dialog now
  wraps; `text_fields=False` here — the device has its own name) on a copy; Back keeps the edit while the same template
  stays chosen; Create device makes the device from the edited copy; "Save as lab template…" also saves it (the template
  file is never changed by this step).
- (2026-11-21, user) **Flake layout › Auto design…** (`ustack/autolayout.py`, window `ui/auto_design_window.AutoDesignDialog`,
  button under the Flake layout canvas): places the traced flakes (turn + shift only, never mirrored or resized) so the rules
  hold and the Stage 2 patterns fit. Finger-array stacks with the fingers written on an hBN under the device (Xu); others
  get a ValueError (Mak pre-patterned: not yet). Inputs: finger pairs wanted (default = Stage 2 Hall bars' count; without
  one, what the bond pads take), the flakes it may move (on-chip flakes locked), keep the twist (device layers only
  shifted), thorough (wider beams). Lab values asked at once (`required_values`: `stack.alignment_shift_um` — added to
  every margin / clearance —, `fingers.bbn_margin_um`, `cg.hbn_margin_um`, `cg.bus_tmd_gap_um`, `electrodes.flake_
  clearance_um`, …). **Stages** (each fixes what the next must hold): device layers (first one = reference; the overlap
  must hold the pairs, most compact first: `_pairs` raster estimate — region turned, eroded by a pair strip, longest run —
  plus how far the device layers reach out along the fingers = contact-gate teeth; 1 µm shift grid pruned by one overlap
  correlation, refined 0.25 µm / 2°, exact `fit_finger_array` + `_teeth_reach`) → bottom hBN (must: twisted region +
  straight finger parts by `bbn_margin`; want: fingers at preset length, the device layers) then the real Hall-bar
  generator (`ef.generate(..., run_check=False)` on a scratch flow) → bottom gate (must: the channel; a free contact square
  outside the hBN, bare graphite beyond 2× that strongly penalised) → top gate (must: twisted region; never over a finger /
  contact-gate electrode landing, the BG contact or another gate's exposed graphite; crossing fingers discouraged; room for
  its contact) → middle hBN (must: the contact-gate teeth (`_teeth_estimate`, 1–3 ungated per column) + twisted region
  (+ TMD by `xu.middle_hbn_edge_margin_um` when it applies) by `cg.hbn_margin_um`; off the finger landings; mildly off the
  top gate) then the real contact-gate generator, kept only if its electrodes land off the top gate and the top gate's
  contact stays clear → top-gate hBN (must: twisted region + wherever the top gate crosses a conductor; off every landing
  and contact) → cap / other flakes (cover the twisted region, off the contacts). **Search per flake** (`search_flake`):
  FFT correlation with the turned flake as kernel (must-hold covered by the flake shrunk by its margin, want / avoid area,
  never-cover, exposure) — 10° on 0.5 µm (tolerant: up to `COARSE_SLACK` 8 % may stick out), ±5° / 1° on 0.25 µm around
  diverse coarse picks (≥ 15° or 5 µm apart), exact KLayout check; must / never targets rasterised ¾ px wider (thin teeth).
  **Backtracking**: a stage with no result sends the search back one stage, which keeps 4× the alternatives (each stage
  once). Finalists: `validate_stage1`, `finger_landings` (blocked fingers), both gate contacts → up to 3 `Candidate`s;
  preview in the window (fingers, comb, contact places); Apply = `apply_candidate` (borders + images, placed) and one undo
  step (`MainWindow.layout_auto_applied`). Order matters: middle hBN before the top gate failed far more often (the comb's
  leads end at the middle hBN's corners, where the top gate must pass). On the lab's Device 2026-11-16 flakes scrambled 6
  ways (10 pairs; test values in a scratch config): all six valid, 12–51 s. `fingers.tmd_keepout` is memoized (copy returned).
- (2026-11-21, user) **Gates stick out on opposite sides** (auto design): the top gate avoids the half-plane beyond the twisted
  region (+ contact-to-graphite clearance + shift + 2 µm) on the bottom gate's side (direction: twisted middle → the bottom
  gate's contact place; `_bg_side`), soft (`GATE_SIDE_WEIGHT` 0.25 × AVOID_WEIGHT per µm²) with a small reward on the other
  side (`GATE_OTHER_SIDE_WANT`); a layout where it could not (> `GATE_SIDE_NOTE_UM2` 25 µm²) is noted ("also sticks out on
  … side") and ranked below those that could (`State.side_over` in the candidate score, after the contacts). The bottom-gate
  stage also offers placements sticking out one way only (4 directions: ± along / across the fingers, `_half_planes`
  avoided on the other side) so a long bottom gate centred on the device does not leave the top gate no side. On the lab
  flakes scrambled 6 ways: all six opposite, 18–28 s, one 148 s (it backtracks to the bottom hBN; 74 s before).
- (2026-11-21, user) **Abort** (auto design window, red, shown while running): the window is free at once (the worker's
  signals are cut, its result dropped); the run stops within a moment — `ustack/cancel.py` (per-thread check set by
  `auto_design`, `poll()` in the search loops, `fingers._fit_at_angle` and `fit_contact_gate`).
- (2026-11-21, user) **Flake layout › Export to image…**: the canvas as shown (`viewport().grab()`, Retina resolution) as a JPG
  (quality 95); where / name in `save_gds.ask_save_path` (the GDS window made general: device folder by default or another
  folder, remembered under "image_dir"). `MainWindow.save_layout_image(path)` for code.
- (2026-11-21, user) **First Save asks the device's name** (`MainWindow._name_new_device`, when `spec_path` is None): a name
  already in the library is refused (asked again); files copied into the folder of the old name before (Flake images,
  Stage 2 images) move to the new folder and the spec's paths follow (`devices.rewrite_paths`, then the Stage 2 flow reloads).
  `save()` returns False when cancelled (closing then stays open).

## 3. Layout of the repository

```
UStack/
  AGENTS.md  README.md  requirements.txt  run_ustack.command  run_ustack.bat
  config/            materials.csv, rules.json, layers.json (lab configuration, see §5)
  docs/device_structures.md   research notes + citations for the 4 templates
  library/           user data (USTACK_HOME overrides): templates/ (lab templates), devices/<name>/
                     spec.json, images/, stage2/<step>/{<step>.gds,<step>_spec.json,<step>_report.md}
  tools/make_builtin_templates.py   regenerates ustack/data/templates/*.json (edit templates HERE)
  tools/make_icons.py               draws ustack/assets/UStack_mark.png, UStack_icon_1024.png
  tools/trace_benchmark.py          tracer on the lab's flake images → docs/trace_benchmark.md (seeds: trace_benchmark_seeds.json)
  tools/learn_lab_designs.py        measures the lab's GDS designs → docs/wse2_design_report.md
  tools/make_bondpad_preset.py      writes the built-in bond-pad preset from the lab's session
  tools/registration_benchmark.py   alignment / flake location on the lab's session → docs/registration_benchmark.md
  tools/debris_training.py   contact sheets of candidates on the lab's images → labels (debris_labels.json) → data/debris_examples.json
  docs/training/     overlays written by the two tools
  tests/             pytest; fixtures/make_fixtures.py builds a SYNTHETIC bond-pad GDS (not lab data)
  ustack/
    config.py        LabConfig, Rule, Material, GdsLayer, MissingItem/MissingValues, apply_answers
    model.py         DeviceSpec (schema ustack.device/1), Layer, Q (value+provenance), MetalPart,
                     HallBar params, Requirement, FUNCTIONS, CONTACT_SCHEMES, FAMILIES, PATTERN_FIELDS
    templates.py     list/load built-in + lab templates, new_device_from, save_as_template, build_from_scratch
    validation.py    validate_stage1 → Report[Issue]; apply_spec_answers; structured_summary
    flakes.py        flake_list (only from a valid report; design size + stacking shift on every side)
    autoconfig.py    configure(spec, config, apply) → ConfigureResult(changes, missing, report)
    geometry.py      KLayout Region helpers in µm (DBU 1 nm): booleans, sizing, DRC helpers
    hallbar.py       HallBar geometry (local frame → placed), Terminal, lead_path
    imaging.py       OpenCV: load_image, prepare_trace/trace_mask/detect_flake/flake_candidates/auto_trace,
                     segment_colors, marker squares, defect_candidates
    lab_designs.py   KLayout session reader (image px ↔ chip) and measurements of the lab's own GDS designs
    devices.py       rename / delete a saved device (folder + paths in its spec)
    bondpads.py      bond-pad presets (GDS + role → layer, device site, pad lead tips); data/bondpads/ built in
    fingers.py       finger array: FingerParams (presets), FingerArray (local u/v frame), fit_finger_array,
                     contact_gate_comb, gate_contacts, electrodes (+ finger_terminals), check_pattern
    router.py        electrode router: A* on a grid round hard / soft obstacles, taut straight pieces, round joints
    debris.py        bubbles / cracks / stray flakes in the aligned images (candidates, features, k-NN on labelled examples)
    cancel.py        per-thread stop request (Abort) polled by the auto design and the finger generators
    autolayout.py    Flake layout › Auto design: staged placement of the traced flakes (FFT correlation search, beam + backtracking)
    ebl_flow.py      Stage 2 workflow state (Flow/FlowStep in spec.stage2["flow"]), STEPS, present_layers,
                     twisted_region, design→chip projection of traced flakes, generate/check, export_step/export_merged
    registration.py  image ↔ chip alignment (low mag to the bond-pad GDS, chaining higher mags) and locating
                     traced flake outlines in an aligned image
    alignment.py     Similarity (px→chip µm, y flipped), fit_similarity (Umeyama), auto_align (RANSAC-like)
    gds_io.py        read_polygons (flattened), layers_in, lead_tips, write_gds (layers from config)
    stage2.py        StepInputs, required_values, place_hall_bar, design_step, check_step, etch_plan, export_step
    scene3d.py       Qt-free 3D scene (footprints + log-scaled heights) for the preview
    assistant.py     Claude Messages API call (urllib) or offline keyword parser → Draft
    app.py, __main__.py   entry point (`python -m ustack`)
    data/templates/  built-in templates (generated JSON)
    assets/          app mark and icon
    ui/
      style.py       VENDORED from "AFM data viewer/afm_style.py" (house style; see §8)
      widgets.py     ComboChevron, modernize, row(), NumberField (empty = None), dialogs helpers, CheckDelegate (readable ticks on selected rows), PickTree (Shift/Ctrl-click toggles a row)
      main_window.py MainWindow (frameless), LibraryPanel (right-click rename/delete), Stage 1 flake-layout logic
      flake_images.py FlakeImagesPanel (side tab: upload, Trace all, layer visibility) + TraceDialog (trace window)
      configure_window.py ConfigureProblemsDialog (errors + editable values behind them, values still needed)
      viewer.py      ImageViewerDialog (one image, zoom/pan, scale bar)
      stack_panel.py layer list (drag reorder) + selected-layer form
      panels.py      PatternPanel (incl. Source / drain style), CheckPanel, IssueList
      ebl_flow_panel.py Stage 2 tab: bond-pad preset, step list, per-step card, merge & export
      outline_window.py OutlineDialog (Stage 2 › Draw outline: a flake's chip outline over an aligned image)
      align_window.py AlignDialog (Stage 2 › Manual alignment: frame handles, auto scale, pick / fit pairs)
      debris_window.py DebrisDialog (each step card › Draw / edit… bubbles, cracks, stray flakes; corrections → training examples)
      issue_popup.py IssuePopup (an error's red circle: Close / Change rule… / Accept)
      auto_design_window.py AutoDesignDialog (Flake layout › Auto design…: options, background run, candidates, Apply)
      save_gds.py    SaveGdsDialog / ask_save_path (Export GDS…, Export to image…: the device's folder or another, the file name)
      stage2_panel.py the classic graphene Hall-bar step (embedded in the Stage 2 tab)
      canvas.py      QGraphicsView canvas in µm (y up): ImageItem, PolygonItem (editable), tools
      stack3d.py     QPainter 3D stack view (drag = rotate at any elevation — from below the layers are drawn top → bottom with
                     their bottom faces, side faces culled by sign(cos el); right/middle drag or trackpad two-finger scroll = move
                     (`pan`), wheel / pinch = zoom; "Separate Layers" = gaps up to 1.5× the stack height in z only, x–y scale fixed; x–y scale bar in the bottom-right corner above the hint line; shapes of a layer painted far → near, layers never interpenetrate in height)
      dialogs.py     MissingValuesDialog, RulesDialog (Edit mode, presets tab), PresetsTable/PresetsDialog,
                     AssistantDialog, ConfigureDialog
      new_device.py  NewDeviceDialog: from a template (+ Edit Template step), built from scratch, or from a description
      template_editor.py TemplateEditor (widget: layers, rules, 3D preview) + TemplateEditorDialog (Library › Edit template)
```

Core modules (everything outside `ui/`) import without Qt; tests exercise them directly.

## 4. Data model essentials

- `DeviceSpec.layers` is **bottom → top**; layer 0 must be the substrate.
- Every number is a `Q(value, source, suggested, suggested_source, note)`. `source` ∈ user,
  template, literature, measured, auto, assistant, material. The program never overwrites a `user`
  value. `Q.suggest()` makes a placeholder-only value.
- Metals: `Layer.metal_parts` = list of `MetalPart(metal, thickness Q)` in deposition order; must
  match the material's `components` in materials.csv (e.g. `Au/Cr` → Cr; Au).
- `Layer.role` tags template-specific layers: `bottom_hbn`, `middle_hbn`, `top_hbn`, `tg_hbn`,
  `tmd`, `graphene`, `prepattern_contacts`. Family rules look layers up by role.
  `validation.assign_roles` fills missing roles from position (assistant drafts, Configure).
- Sources: `user` and `assistant` (numbers the model wrote from the request) are never changed by
  the program; `auto` values are Configure's and are recomputed by the next Configure.
- `Layer.outline` (µm, Stage 1 design frame) replaces nominal width×length in checks/3D.
  Stage 2 outlines are in **chip µm** in `StepInputs.outlines[uid]` (list of polygons).
- `Layer.image` = {path, um_per_px, angle_deg, x, y, opacity, visible, mirror}: px→µm is
  `X = s·R(θ)·[x·(±1), −y] + (x0, y0)` (see `ui/canvas.image_transform`).
- `spec.hall_bar.params` (HALL_BAR_FIELDS): W, L (first→last probe centre), n pairs, arm width,
  arm length, S/D extension. `spec.pattern_params` (PATTERN_FIELDS): top-gate / contact-gate /
  split-gate geometry, asked for when a step needs them.
- `spec.stage2.steps[]` records every exported step (inputs incl. transforms and outlines, as-built
  Hall bar, placement, routes, patterns, DRC). `spec.stage2.hall_bar_placement` lets the next step
  reuse the placement ("From the previous step").
- `MissingItem.key` namespaces: `rule:<id>`, `layer:<purpose>`, `default:<name>#<sub>`,
  `spec:<uid>.<field>`, `spec:<uid>.metal.<metal>`, `spec:hall_bar.<k>`, `spec:pattern.<k>`,
  `spec:gate.<uid>`, `trace:*` (an action, not a value).

## 5. Configuration files

`rules.json` → `rules{id: {value, unit, kind(min|max|bool|range|info), severity, status,
applies_to[tags], description, source{doc, section, quote}}}` and `defaults{}` (metal stacks from
the Fab Manual, microscope µm/px). Tags: `all`, `graphene`, `rhombohedral`, `tmd`, and family ids
(`config.FAMILY_TAGS`). A rule whose status is pending/literature still runs; its findings say so.
`layers.json` → purpose → {layer, datatype, name, status, source, reference_only}.
Known purposes: align_marks 3/0, board_etch 7/0, contacts 8/0 (pending, from the Fab Manual);
hallbar_etch, etch_windows, top_gate, bottom_gate, prepattern_contacts, contact_gates, split_gates,
ref_flakes, ref_defects (missing).

`defaults.design.thickness_nm` (per `material:function`) and `defaults.design.metal_nm` (per
`material:function:metal`) hold the lab's design thicknesses for Configure; they start empty and
fill as Configure asks. Metal stacks use the Fab Manual defaults (`metal.contacts_1d`,
`metal.top_gate`) first when the material matches.

Still **missing** (ask the lab, do not guess): `stack.alignment_shift_um` (stacking misalignment,
used by the flake list and Configure), `stack.substrate_breakdown_field_MV_per_cm`,
`ebl.defect_clearance_um`, `ebl.contact_overlap_um`, `ebl.etch_mask_overhang_um`,
`ebl.board_etch_keep_margin_um`, `ebl.etch_window_size_um` (defined, not used yet — windows are drawn
by hand), `xu.middle_hbn_max_thickness_nm`, `stack.hbn_breakdown_field_MV_per_cm` (optional),
and the missing GDS layers above. (The lab has since entered some of these in its own config/ —
read the file, not this list.)

## 6. Algorithms worth knowing

- **Validation** (`validation.validate_stage1`): structure (substrate, device layer, function vs
  material category), values (thickness / layer count / lateral size / metal thicknesses / twist /
  Hall bar), electrostatics (dielectric between every conducting gate and the device; encapsulation;
  SiO2 breakdown from the doped-Si gate limit; hBN field if gate limits given), family rules (Xu,
  Mak, graphene 1D-contact rules, Ju), requirements. Nominal sizes assume centred flakes; traced
  outlines are used when present.
- **Configure** (`autoconfig.configure`): roles → thicknesses (lab design values / Fab Manual
  defaults) → Hall bar (electrode targets, gate margin max(shift, spacing), clearances
  beyond-BG / contact-to-graphite, contact overlap) → flake sizes (device = Hall-bar footprint,
  graphite gate = probe span + margin, hBN covers all) → Stage 2 gate geometry → validate. Missing
  inputs come back as `MissingItem`s; validation's own missing physics choices are added. Every
  ratio / margin multiplier comes from `config.preset(...)`. S/D extension: straight, bent
  (leg + arm/2 + spacing) or, for `auto`, the larger of the two. Mak (pre-patterned + bent): channel
  = k_w·arm, BG = channel + 2g (+ extra under a TG), bottom hBN = BG + margin, gap W = bottom hBN +
  2·shift, device = (L + 2·S/D) × (W + 2·`pt_overlap_um`), electrode length = overlap + 2g.
- **Bent S/D** (`hallbar.HallBar.sd_style == "bent"`): legs of width `leg_w` (W; the arm width for
  pre-patterned electrodes) flush with the channel ends, S to +y and D to −y, `arm_l` long; the S/D
  terminals sit at the leg tips. `stage2.design_step` with style `auto` places the straight bar,
  and if a traced BG exists and the gated fraction is < 0.99 (or nothing fits) places a bent copy
  and keeps it when it gates more. The TG lead avoids the source leg; contact gates cover the legs
  (no end bands). The as-built style is stored with the placement for later steps.
- **Two-column electrodes** (validation): electrodes separated from the device by other layers are
  fine only with bent S/D, if the device is wider than each layer in between and every conducting
  layer in between fits the gap with spacing (`contact_not_touching`, `gate_in_gap`). Stage 2:
  electrodes laid before the BG must not touch it at all (`finger_bg`, `finger_bg_gap`).
- **Placement** (`stage2.place_hall_bar`): grid × angle search (3° steps, then refined) of the Hall
  bar body inside usable = device layers ∩ encapsulating hBN − (defects ⊕ clearance); for 1D
  contacts each terminal's contact box must lie ≥ `ebl.device_etch_beyond_bg_um` outside the BG.
  Score = (fraction of the probe span over BG∩TG, margin level). Manual and previous-step placements
  bypass the search.
- **Leads** (`stage2.route`): every terminal (Hall-bar ends, gates) leaves straight (≥ 2 lead
  widths), escapes the device keep-out, then goes straight to a pad tip; tips are assigned by an
  order-preserving DP around the device (no crossings in a radial fan-out). Gate leads go on the
  gate's own layer.
- **Metal top gate** (`stage2.design_step`): body = channel × `tg_length` only (graphene under
  metal survives the Hall-bar etch, so a gate over the arms would join them); its lead leaves through
  the widest gap between probe arms (`_tg_exit`, ≥ spacing to every arm and the S/D contact) and
  ends `tg_overhang` past the graphene. The Hall-bar etch mask follows the gate. Fails with a
  message if no gap is wide enough (dimensions are never changed).
- **Back gate**: any traced `bottom_gate` layer (graphite or a pre-patterned metal; the Trace card's
  "From GDS" copies a bond-pad GDS layer's shapes near the site as an outline) counts for placement,
  beyond-BG and gating checks; `channel_not_gated` warns when the probe span is not over it.
- **DRC** (`stage2.check_step`): KLayout width (projection metrics, ≥80° corners ignored, 2 nm grid
  tolerance) and space checks per metal layer (hard min = error, target = warning), metal overlaps
  and gaps between metal layers on the same stack level (this step against the as-built metal of earlier
  steps and the planned metal of later ones: `ref_contacts`, `ref_top_gate`, …), arm spacing, contact–graphite distance, terminals beyond BG, contacts touch
  graphene, defects, board etch keeps the graphene (and is larger), etch windows on top-hBN only,
  metal TG width / reaching beyond graphene / lead start off the graphene, Xu middle-hBN margin + contact gate
  never on TMD, fingers vs BG, Mak TG ⊂ BG, nothing written on alignment markers, unrouted leads.
- **Alignment** (`alignment.auto_align`): pairs of image squares × pairs of GDS squares with matching
  size ratio propose a similarity; best = most squares matched within side/3; refined on all
  matches. Needs the device site when every site has identical markers.
- **Flake tracer** (`imaging`, v0.4; tuned on the lab's 18 WSe2 batch-1 flake images, see
  `docs/trace_benchmark.md`): bilateral smoothing → CIELab (L 0–100) → substrate colour = main
  colour of the image *border*, refined by a quadratic fit (illumination fall-off removed). From the
  click (median 9×9): *whole flake* = farther than max(4σ, 0.35·contrast) from every background
  colour; *this colour* = closer to the click than half-way to every competing colour. The
  background set grows with the dominant colour of a ring around the region (a flake on a flake).
  Open r=2, a coarser open r=6 decides connectivity (cuts necks to neighbours), close r=2, holes
  filled, polygon at 0.12 % of the perimeter. `flake_candidates`/`auto_trace`: one seed per region
  of a k-means colour class, scored √area × centre weight, ×0.6 per field edge touched,
  ÷contrast for monolayers (`prefer='faint'`). `tolerance=` keeps the v0.3 flood-fill wand.
  Mean click stability 0.99 (v0.3: 0.79); auto pick right on 18/18 images. Weak spot: dark-field
  images where the edge is a soft gradient (worst click 0.8).
  **Tape residue** (v0.5, user report on Stack 1 top hBN: residue has thin-hBN colours): whole-flake
  mode, bright field only, `_drop_residue` — region minus the clicked colour (within half its
  contrast) → pieces (open r=3); a piece is dropped when its outer border is softer than
  `residue_sharp` (0.6) × the clicked colour's border (peak gradient / contrast) AND its interior is
  textured (gradient 75th pct > `residue_flat` 0.9 ΔE/px). Sharp-edged thick rims / wedges and
  uniform terraces are kept. On the 18 images only Stack 1 top hBN (residue hook + halo removed,
  IoU 0.988 with the old trace) and Stack 4 top hBN (IoU 0.997) change. `params={"residue": False}`
  switches it off.
- **Flake images**: Trace all / opening Trace flake… on an untraced flake → `imaging.auto_trace`
  (faint seed for thin graphene/TMD device layers; Trace all always whole-flake mode) → the image is shifted so the border's centroid is the design origin, unless the image
  was moved in Flake layout (`image["placed"]`). `image["trace"]` records how/mode/seed.

- **Finger array** (`fingers.fit_finger_array`): rasterise the twisted region in the array's local
  frame (step = pitch / an integer ≈ 0.05 µm), erode by the pair strip (finger width ×
  (gap + 2·min_overlap)), then for every angle (2°, refined 0.25°), pitch phase and gap position the
  longest run of consecutive valid positions = the pair count; ties → largest edge distance (then
  most central). Verified with KLayout booleans. Shapes are built in the local frame, merged, then
  rotated once (no grid slivers); the fan piece has square ends and a convex-hull joint.
- **Flow steps** (`ebl_flow.generate`): Hall bars = fit + fingers; contact gate = comb (one tooth per
  finger, bus beyond the outer tooth: top column to −u, bottom to +u); electrodes = one group at a time
  (`ELECTRODE_GROUPS`, see §2 2026-11-19): leads from the finger landing points (along the finger to its far end),
  from the comb bus ends (`stage2.assign_tips`, tips used by the other groups excluded), and `gate_contact` per gate;
  all routed together by `router.route_nets` (§2, 2026-11-20), each ending in its landing square.
- **Flakes in Stage 2**: located outlines keep the design outline's vertex order, so
  `ebl_flow.fit_design_to_chip` (Procrustes with optional reflection) maps the Flake-layout frame to
  the chip and `project_flakes` places the flakes not located (dashed). Steps inherit outlines from
  the previous step.
- **Chip → Flake layout** (v0.5, user request): flakes identified at a step (`ebl_flow.IDENTIFIED_HOW`: located, or
  that outline moved / edited with the traced border's vertex count) update the design frame —
  `ebl_flow.orientation_from_chip`: rigid fit (rotation ± reflection, scale 1) of each traced border onto its chip outline;
  reference = the lowest identified flake already placed in Flake layout (else the lowest identified); every other
  identified flake gets T_ref⁻¹ ∘ T_i applied to its outline and image (`_image_after`, image marked placed). Shapes and
  sizes are never changed; unidentified flakes keep their place relative to the reference. Needs ≥ 2 flakes. Runs after
  Locate (before `project_flakes`) and on `EBLFlowPanel.set_outline` with how in IDENTIFIED_HOW; record in
  `spec.stage2["orientation"]` {step, reference, layers{uid: turned_deg, moved_um, mirrored, changed}}; the 3D stack
  (Structure) and Flake layout redraw from the outlines. The spec's twist angles (user values) are not touched.
- **Planning the device layers** (v0.5, user request: Hall bars are written on the BG before the TMD
  halves are dropped): `FlowStep.plan` uid → {cx, cy, angle}; `ebl_flow.planned_outline` = the traced
  border (design frame) turned about its area centroid and moved to (cx, cy) — shape and size never
  changed; `default_plan` = bond-pad site (+ the design-frame centroid if the image was placed in
  Flake layout); `apply_plan` writes `outlines` (located how = "planned") and, while
  `twisted_how ∈ {"", PLAN_HOW}`, the twisted region = overlap. A planned outline dragged whole moves
  its plan (`translation_of`); reshaped, it leaves the plan. The planned twisted region is not
  editable on the canvas (it follows the layers). Panel: "Plan from flake images", list with a
  rotation slider per layer (`plan_sliders`), canvas `polygonSelected` ↔ row.
- **Canvas outlines** (`Canvas.drag_outlines`, on for the EBL canvas and the outline window): pointer
  press within 6 screen px of an editable outline's line selects it (handles appear) and drags it whole
  (`PolygonItem.begin_move/move_to/end_move`). **Draw outline** = `ui/outline_window.OutlineDialog`
  (an aligned step image in chip µm, the outline editable, wand on that image, "From the traced
  border"); edits go through `EBLFlowPanel.set_outline`.
- **EBL view** (`Canvas.trackpad_gestures`): wheel events with pixelDelta / a scroll phase pan (Ctrl =
  zoom), NativeGesture zoom / rotate; `set_view_rotation` (transform = rotate·scale(m, −m), centre
  kept), `rotationChanged` ↔ the "View rotation" slider + reset under the canvas. QPinchGesture is not
  grabbed: Qt also turns native twists into pinch gestures (double handling, seen in a test). Not
  tried on a real trackpad (offscreen tests with synthetic events).
- **Registration** (`registration.py`, built and benchmarked on the lab's session —
  `docs/registration_benchmark.md`): coarse-to-fine NCC template matching over angle × scale ×
  translation. Low magnification ↔ bond-pad GDS on gradients of a learned metal map vs. the rendered
  layers 3/0 + 4/0 (not layer 5: fingers under flakes break it); finer images chained onto the
  previous one (band-pass Lab) and re-fitted on the GDS. Error at the device vs. the lab's KLayout
  alignment: 5x 0.2 µm, 10x 0.4–0.5, 20x 0.07–0.85 (Stack 3's session matrix itself is 0.66 off a
  similarity), chained 100x 0.07–0.08 µm; ~19 s for four images. Flakes: oriented chamfer matching
  of the traced border (360°, mirror, ±4 % scale), robust to partial cover; score ≥ 0.45 and margin
  ≥ 0.10. In the panel 20x and finer images, within 150 µm of the site (finest first; a graphite
  strip longer than the 100x field is found at 20x); Locate first aligns images loaded since the last
  Align (user report: a 50x added to an aligned 5x was ignored), and with no ≥ 20x image aligned it
  falls back to the 5x / 10x images and says the outlines are coarse (Stack 1 top hBN: 5x 1.4 µm,
  10x 0.2 µm off the 100x result). Found on Stack 1/3: bottom/top hBN, back gate,
  top gate (not always); not found: middle hBN and WSe2 (faint, covered) — drawn by hand.
  `ebl_flow.project_flakes` carries other flakes only if they were placed in Flake layout
  (`image["placed"]`) and the located ones agree with it to `PROJECTION_MAX_RMS_UM` (3 µm).
- **DRC spacing** (`geometry.space_violations`) now uses projection metrics and ignores corners
  ≥ 80°, like the width check (a rotated comb's inner corners were false errors).

## 7. Testing

`python -m pytest -q tests` (see the count in the last run; `test_fingers_flow.py` = finger fit (count,
angle, no shrinking), DRC-clean shapes at any angle, bond-pad presets, design→chip projection, every
step generated → exported → merged GDS read back, the Stage 2 panel (generate, edit on the canvas,
flakes carried to the next step, classic step), the measure tool; `test_registration.py` = alignment and
flake location (synthetic + the lab's session); `test_tracing.py` = synthetic tracer cases + the
lab's 18 flake images (click stability, auto pick; skipped without `WSe2 device training/`);
`test_lab_designs.py` = GDS design analysis (synthetic + the lab's session); `test_devices_ui.py` =
rename/delete, issue parameters, Configure problems window, Flake images upload → 3D, Stage 2
viewer/trash; `test_configure.py` covers Configure on all four
templates, the twisted trilayer, user values kept, a configured device through Stage 2, the Mak
two-column electrodes, presets and the field of view; `test_stage2.py` the BG strip that bends the
S/D; `test_ui_smoke.py` New device / Save device / scale bar / Rules Edit / presets): config statuses and persistence, the 4 templates,
validation rules (Xu, breakdown, requirements, no substitution, no shrinking), flake list, assistant
parser, spec round-trip, Hall-bar geometry, marker detection + alignment on a synthetic image,
wand, full Stage 2 graphene run (generate → DRC clean → export → GDS layers read back), defect
detection, "does not fit" reporting, and an offscreen UI smoke test. Tests copy the **seed config**
`tests/fixtures/config_seed/` (the configuration as seeded, before the lab entered values; presets
come from `config.PRESET_DEFAULTS`) to a temp dir — never the live `config/`, which the lab edits.
When you add a rule to `config/rules.json`, add it (as seeded) to the seed too. Values tests enter
are marked `# TEST VALUE` and never reach the real config.

UI screenshots: set `QT_QPA_PLATFORM=offscreen`, build `MainWindow`, `window.grab().save(...)`.
Wait ~400 ms after switching tabs (they cross-fade).

## 8. Style (UAnalysis house style)

`ustack/ui/style.py` is a copy of `AFM data viewer/afm_style.py` with a few edits: module docstring,
ImportError text, `ASSET_DIR` (→ `ustack/assets`), settings key `("Uri Lab", "UStack")`, About text
/ menu label, the `USTACK_PANELS` env var, a `redo` icon (mirrored `undo`) and the `addPoint` / `selectPoints` / `arrowDown` icons in `_SVG`. To pick up a newer UAnalysis style, copy the file
over and re-apply these edits. Use its widgets (`Card`, `Panel(kind=…)`, `button(...)`,
`TabWidget`, `ComboBox`, `CheckBox`, `Slider`, `Banner`, `Pill`, `label(role=…)`) and call
`widgets.modernize(root)` after building a panel (frames fields, adds combo chevrons). One navy
accent, 3 px corners, 26 px control rows, title-case section titles, no shadows.

## 9. Known limitations / next steps

- **Finger array (v0.5)** reproduces the lab's design in its main lines (two columns, comb, gate
  contacts, electrodes) but simpler: one fan piece per finger ending in a short stub (the lab's fingers have one
  or two bends), every finger contacted (the lab leaves fingers the flake does not reach floating —
  here only fingers over the twisted region are drawn), one comb per column, leads are straight to
  the pad tips (they cross and bunch; DRC reports it). The measured presets are *pending*.
- The fit only looks at the twisted region; it does not prefer the back-gate direction (a warning
  says when the BG does not cover the channel).
- The configured 100x field of view (143.28 µm wide, approved by the user) is ~1.9 % larger than
  what the lab's own KLayout alignments give for 6000-px 100x images (0.02343 µm/px → 140.6 µm).
  Reported to the user; the user confirmed 143.28 µm (measured in the microscope, 2026-11-18).
- Only the Hall bar. Add a pattern type by: a geometry class like `hallbar.HallBar` (terminals,
  body), its params in `model`, a branch in `stage2.design_step`/`place_hall_bar`, its checks.
- Contact-gate and split-gate shapes are simple bands/regions. The Mak two-column electrodes are
  the Hall-bar probe arms plus the bent S/D legs (one finger per terminal); other finger counts or
  shapes need their own parameters.
- Finger-array electrodes are routed (`router.py`, 2026-11-20). The classic graphene step's leads (`stage2.route`) are
  still straight two-segment routes whose crossings / spacing are reported, not avoided.
- Classic step only: Stage 1 flake outlines are not carried into it (the v0.5 workflow projects them).
- Alignment-mark clearance rules ("leave clearance for 2 of 12", outer sets), local capacitance,
  stitching patches and scanning-probe rules from the DRC doc are not encoded.
- Etch windows are drawn by hand (`ebl.etch_window_size_um` unused).
- Undo / redo (v0.5): `MainWindow` keeps JSON snapshots of the whole spec (the Stage 2 flow lives in
  `spec.stage2`); `spec_changed` starts a 0 ms timer, so one operation = one step; Ctrl/⌘+Z,
  Ctrl+Y / Ctrl(⌘)+Shift+Z, buttons at the top right. Not undone: config/ edits (rules, presets,
  layers), files written (exports), the classic Stage 2 panel's own inputs. History resets when another
  device is opened or created. (Save writes `library/devices/<name>/spec.json`; right-click › Rename
  moves the folder.)
- Classic step only: 5x/20x/50x images are not aligned (the v0.5 workflow aligns every magnification).
- The Claude model id defaults to `claude-opus-5-5`; override with `USTACK_CLAUDE_MODEL`.

## 10. Open questions for the lab (ask the user)

0. (Answered 2026-11-18: pitch; 5/6/7 are the suggested output layers.) Still open: approve the measured finger /
   comb / electrode presets (Rules › presets, group *Finger array*).

1. The missing rule values and GDS layers in §5.
2. Confirm the Xu middle-hBN rule interpretation (middle hBN encloses the TMD with ≥ 1 µm margin).
3. Which bond-pad GDS layers hold the alignment markers and the leads to connect to (the app asks
   per file); marker size convention (`pos 10` / `pos 20` in the EBPG notes).
4. Mak layout: settled by the user (two electrode columns, BG in the gap; see §2). Open: whether
   the Configure presets for it (device overlap onto each electrode = 2 × spacing, electrode length
   = overlap + 2 × gate margin) match the lab's practice.
5. Whether Ti/Pd (Mak contact/split gates), Pt without adhesion (Xu) and Ti/PdAu should be added to
   / approved in `materials.csv`.
6. Whether "spacing between Hall bars ≥ 0.8 µm" means arm-to-arm spacing (as implemented) or
   spacing between separate Hall bars on one flake.
7. The lab-design measurements (`docs/wse2_design_report.md`): which become presets/rules for a
   finger-array pattern (layers 5/6/7: answered, suggested output layers).
8. (Answered 2026-11-18: 143.28 µm, measured.)
