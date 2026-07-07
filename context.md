# Schematic-to-Netlist AI — Progress & Aim

## Aim
Commercial-grade pipeline: upload a digital logic schematic image → correct
gate-level netlist + Boolean equations. Must be **accurate** (exact gate types,
exact wire connectivity, exact input/output names) and **robust** to real-world
image variety (clean line-art, hand-drawn, scanned/tinted, colored/watermarked
web images) — not just the training corpus.

## Repo / environment
- GitHub (private): https://github.com/yaswanthsree-exe/Schematic-To-Netlist-Generation
- Core pipeline: `production_v2/predict.py` (single file, ~10 stages: YOLO gate
  detection → CNN reclassify → preprocess/erase → skeletonize → skeleton graph
  → pin assignment → net construction (Union-Find) → OCR naming → gate graph →
  netlist/equations).
- Netlist-only app: `production_v2/app.py` (Streamlit, port 8501). Restart the
  server after every `predict.py` edit — Streamlit keeps the old module in
  memory otherwise (bit us once).
- **Critical env rule:** only `opencv-contrib-python` may be installed (never
  alongside plain `opencv-python`/`opencv-python-headless` — that combo
  silently deletes `cv2.ximgproc.thinning` and degrades every result). Verify:
  `python -c "import cv2; print(hasattr(cv2.ximgproc,'thinning'))"`.
- Test corpus: `Digital_train_data/` (full_half_adder 23, half_full_sub 32,
  random 122 — 177 total). User re-supplied the same dataset from Downloads;
  confirmed byte-identical, no new images to add.
- Regression harness: `production_v2/_reg_bgfix.py` — runs current predict.py
  vs `production_v2_backup_31may/predict.py` (frozen pre-session baseline)
  over all 177 images, reports gained/lost gate-output propagation and input
  counts. **Caveat:** its "gates-with-output" metric counts only TERMINAL
  outputs and goes DOWN as real gate-to-gate links form — judge fixes by the
  named ground-truth benchmarks below + phantom-input counts, not that number.

## Backups (3 independent layers)
1. Git history (5 commits so far, see below).
2. `BACKUP_WORKING_02jul2026_full_pipeline/` — code + both model weights
   (`best.pt` YOLO detector, `newmodel.pth` CNN classifier) + restore guide +
   frozen package versions. Refresh after every accepted commit.
3. `BACKUP_WORKING_02jul2026_full_pipeline.zip` — same, zipped, integrity-tested.

Workflow after every accepted fix: commit → push → refresh backup folder + zip
→ restart Streamlit → confirm app is live.

## Ground-truth benchmark circuits (must stay exact after every change)
1. **AND→NOT→AND** (`Digital_train_data/random/Screenshot 2026-03-26 231843.png`)
   → `E = (A&B) & (~B)`, inputs [A,B]
2. **Full adder** (`full_half_adder/Screenshot 2026-03-26 223101.png`)
   → `S = A^B^Cin`, `Cout = AB+BCin+ACin`
3. **20-combinational** (`20-combinational_circuit.png`, project root)
   → `Y = ~((A&B) | ~C)`, inputs [A,B,C]
4. **Half adder** (image from debug session, OCR-named Sum/Carry)
   → `Sum=(A&~B)|(B&~A)`, `Carry=A&B`
5. **223303 full-adder variant** → majority Cout
6. **Colored XOR (electronicsarea.com web image)** → correct topology
   `OR(AND(·,~A), AND(~B,·))`; 2 input letters remain phantom (see Known
   Limitations — labels drawn ON the wires, not a pipeline bug)
7. **6-input NAND tree** (`Screens...235319.png`, debug folder
   `tmpso1hs3jr`) → all 6 inputs A–F correctly wired through 6 NAND gates

## Commit history (chronological, each regression-validated)
1. `1d60fdc` — initial verified baseline commit (repo created, gitignore, models excluded).
2. `faeae3d` — tight gate-proximity guard: don't override a pin with a real
   traced wire (fixed half-adder `AND(B,~B)` becoming `AND(~B,~B)`).
3. `36eb64b` — out-of-domain **colored** image handling: run YOLO on both the
   original and an Otsu-binarized line-art rendering, keep higher-confidence
   set (gated to ≥2% saturated pixels so the B/W corpus is untouched). Plus
   one-driver-one-pin rule, OCR name length cap (≤6 chars, kills
   watermark/caption theft).
4. `4104f78` — phantom-pin prune: drop a pin whose net is a tiny isolated
   stroke (label glyph overbar) when the gate already has full fan-in.
   Documents 3 reverted FragMerge-relaxation attempts for on-wire labels.
5. `53c6f2f` — **biggest fix**: `_net_path_px` was counting RDP polyline
   VERTICES not pixels (a straight 21px wire measured "2px"), blinding every
   substantial-wire guard — root cause of the 6-NAND C/E-input hijack. Fixed
   to euclidean length. Plus: contact-based pin assignment (adapted from an
   old `predict_v2` prototype — read the net at the pixels where a wire stub
   touches the gate's erased boundary, endpoint-snap as fallback); raw-ink
   evidence bridge (restores 1px wires that MORPH_OPEN erased entirely, when
   the original image proves ≥90% ink on the straight line and no gate is
   crossed); dangling-output rule for the wide-proximity pass (a gate output
   that already drives something can never steal another pin — only a
   dangling, OCR-unnamed orphan wire may be claimed).
6. `81a33d1` — **fixed a real regression** in #5's dangling-output rule: its
   escape hatch ("protected pin may be re-linked to a DANGLING source
   output") was unsound — a gate's legitimate FINAL circuit output (e.g. a
   3-input XOR's Sum) is ALSO "dangling" by definition (nothing consumes a
   terminal output), so the escape hatch was indistinguishable from "this is
   just the answer" and hijacked a real wire (full-adder benchmark's XOR Sum
   output stole the neighbouring AND's real 'A' input, merging 2 outputs
   into 1 garbled equation). Discovered while testing an external tool's
   repackaging of this pipeline (see "External tool episode" below) — it
   faithfully reproduced this same bug, which is what surfaced it. Fix: a
   protected pin (substantial traced wire) is now NEVER overridden, full
   stop — a truly broken fragment can't itself already be "protected"
   (fragments are short by construction), so no legitimate case needed the
   escape hatch. Result: MORE gains (20, up from 16) than the rule it fixed.
7. `5c1e3f4` — widened domain-normalized detection (#3) to also trigger on
   **gray-tinted scanned images** (modal background gray < 240; a real scan
   measured 232 vs ≥245 for genuine white-paper corpus images) — colour
   alone missed this since gray has zero saturation. Fixed a scanned
   NAND/NOT/AND/NOR/OR circuit from garbage classification (XOR/XNOR + a
   detected logo) to 100% correct gate types and topology. ALSO tried and
   REVERTED an OCR change (always-run permissive/magnified retry merged
   into partial standard-pass results) — it recovered missing labels but
   snapped them to the WRONG stub on small gates (input/output only a few
   px apart), regressing 2 verified benchmarks. Reverted to the original
   zero-results-only retry trigger. Residual on the tinted scan: gate
   types/topology are exact; 3 of 4 input NAMES are auto-lettered (not
   matching the schematic's true A/B/C) because OCR still can't read those
   specific low-contrast glyphs — a cosmetic naming gap, not a logic error.

## External tool episode (informational, not part of the pipeline)
User fed the `BACKUP_WORKING_02jul2026_full_pipeline.zip` backup to an
external tool ("Manus AI"), which produced two zip bundles in
`C:\Users\yaswa\Downloads\` proposing a "SchematicAI Pro v6.0 / v6.1"
production wrapper (FastAPI, Docker, Streamlit, config.json). Both were
extracted and LIVE-TESTED (not just read) in throwaway dirs
`manus_pro_test/` and `manus_v2_test/` at the project root (~390MB, left
in place, safe to delete anytime — nothing in them is used by the real
pipeline). Findings:
- v6.0's `predict_refactored.py` was a non-functional stub (literal
  "placeholder" comment, `predict()` always returned an empty
  `CircuitResult`) — confirmed by running it end-to-end, real models
  supplied, on a benchmark image: empty netlist, empty equations.
- v6.1 fixed this by delegating to the real `predict.py` (`from predict
  import predict_circuit`) — a legitimate integration. Running IT is what
  surfaced the dangling-output-rule regression fixed in commit `81a33d1`.
- No malicious content in either bundle. Not adopted; the real
  `production_v2/predict.py` + `app.py` remain the only pipeline in use.

## Known limitations (accepted, not bugs)
- **Labels drawn ON wires** (e.g. electronicsarea.com XOR image): topology and
  gate types extract correctly, but 2 input letters stay phantom because the
  label glyph physically merges with the wire ink and no geometric bridging
  guard can fix it without breaking other circuits (3 variants tried and
  reverted in commit 4104f78). Real fix would need OCR-guided glyph removal or
  retraining the detector on this style — a dataset/training task, not a
  pipeline patch.
