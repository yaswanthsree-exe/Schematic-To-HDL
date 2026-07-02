# Night Agent Report — Schematic-to-Netlist Redesign

## What I did
Built a new robust pipeline (`predict_v2.py`) and replaced `predict.py`'s usage in `app.py`. Original `predict.py` is preserved (only fix: `CNN_CONF_THRESHOLD` constant that was undefined and crashed every run).

## Key redesign decisions

### 1. Skeleton-based wire tracing (replaces CCL-with-crossing-breaks)
- Skeletonize the wire binary to a 1-px graph.
- Junction pixels (skeleton pixel with ≥3 neighbors) are detected and grouped into "junction cores".
- Each core is classified by checking if 4 perpendicular arms exist within 4–14 px of the center on the **un-broken** binary.
- 4-arm core without a dot ⇒ true crossing ⇒ split it; relink `L↔R` and `T↔B` via Union-Find.
- 3-arm (T) or 2-arm (L) cores ⇒ all arms join.

### 2. Distance-transform-based dot detection
Junction dots are local thickenings of the wire. Detection threshold = `1.6 × median wire thickness`. Dots are detected on the **raw** threshold image (before any morphological smoothing destroys them).

### 3. Pin-first wire-to-gate matching
Pins are detected from where wires actually touch the gate's bbox edge (not from fixed `0.33/0.67` fractions). For each detected pin, the dominant wire label in a tolerance window becomes that pin's connection. Variable fan-in is supported (e.g. 3-input XOR/AND/OR).

### 4. Multi-threshold binarization
Otsu + dark-pixel + adaptive-Gaussian, OR'd together. Handles clean screenshots, faint scans, and gray-line schematics in one pass.

### 5. Per-image debug imagery
Every run writes to `debug_intermediates/<image-stem>/`:
- `01_original.png`, `02_binary.png`, `03_dots.png`
- `04_broken.png`, `05_wires.png` (color-coded nets)
- `06_annotated.png` (gate boxes + pins)
- `07_wires_with_gates.png` (overlay)

## Results on full test corpus

```
Processed 177 images, 0 crashes.
Zero-warning results: 65 / 177  (37%)
"no inputs detected": 83 occurrences across 177 runs
"only 1 input": 199 occurrences
```

The pipeline never crashes — every test image produces a netlist + Boolean equation. Detection (YOLO) is solid; the dominant remaining failure mode is wire-net merging on schematics that omit junction dots, where convention is genuinely ambiguous from pixels alone.

## Files modified / added
- `predict.py`: added missing `CNN_CONF_THRESHOLD` constant.
- `predict_v2.py`: NEW — full redesigned pipeline.
- `app.py`: now imports `predict_v2`.
- `batch_test.py`: NEW — runs the corpus and writes `batch_results.txt`.
- `batch_results.txt`: full per-image summary.
- `debug_intermediates/`: per-image intermediate visualizations.
- `NIGHT_AGENT_REPORT.md`: this file.

## Known limitations / next steps
1. **Dot-less schematics**: T-junction-vs-pass-through is genuinely ambiguous without a dot. Some schematics route parallel buses without dots; my heuristic picks one interpretation (T = merge). Where this is wrong, the result over-merges (single net for whole bus).
2. **3-input gates** (e.g. 3-input XOR in full adder): YOLO trained on 2-input gates only. Pin detection finds N pins from wire contact, so the netlist can show 3 inputs even if YOLO labels XOR — but the underlying class hint (e.g. "XOR") still expects 2.
3. **Logo / watermark interference**: green watermarks ("alaCAR" etc.) survive thresholding sometimes. Adaptive threshold helps but not perfectly.
4. **Dot detection still misses small or anti-aliased dots** in compressed PNGs. A circle-template-match Hough pass would help.

## Manual verification spot-check
- `Digital_train_data/full_half_adder/Screenshot 2026-03-26 223101.png`:
  - Detected: 1×XOR, 3×AND, 1×OR ✓ (matches schematic)
  - Inputs: A, B ✗ (should also have C_in — wire-merging issue)
  - The XOR is detected as 2-input but in the schematic it's 3-input; this is a YOLO training-data limitation
- `Digital_train_data/random/Screenshot 2026-03-05 213814.png`:
  - Detected: OR, NAND, AND ✓
  - Inputs: A, B ✓
  - Pin positions correct on annotated image
- `Digital_train_data/random/Screenshot 2026-03-14 205506.png`:
  - Detected: 2×NOT, NAND, AND, OR ✓
  - Wires now correctly separated into Y / X buses (was previously one giant net before the threshold widening)

## How to use
```
# CLI
python predict_v2.py "<path/to/schematic.png>"

# Streamlit UI
streamlit run app.py
```
Annotated images and intermediates land in `debug_intermediates/<image-stem>/`.
