"""
Batch test against Digital_train_data/random directory.
Reports per-image: PASS/PARTIAL/FAIL, gate count, warnings, equation summary.
"""
import sys, os, glob, logging, time
sys.path.insert(0, os.path.dirname(__file__))
logging.disable(logging.CRITICAL)

from predict import find_best_model, predict_circuit, find_gate_classifier, load_gate_classifier
from ultralytics import YOLO

IMG_DIR = r"C:\Users\yaswa\.gemini\antigravity\scratch\schematic_to_netlist\Digital_train_data\random"

model = YOLO(find_best_model())
clf_path = find_gate_classifier()
classifier = load_gate_classifier(clf_path) if clf_path else None

imgs = sorted(glob.glob(IMG_DIR + r"\*.png") +
              glob.glob(IMG_DIR + r"\*.jpg") +
              glob.glob(IMG_DIR + r"\*.jpeg"))

print(f"Testing {len(imgs)} images from: {IMG_DIR}")
print("=" * 90)
print(f"{'#':>3}  {'Status':8}  {'G':>2}  {'Inputs':12}  {'Output':10}  {'Warnings / Equations'}")
print("-" * 90)

totals = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
warn_types = {}

for idx, img in enumerate(imgs, 1):
    name = os.path.basename(img)
    try:
        t0 = time.time()
        r = predict_circuit(img, model=model, classifier=classifier)
        elapsed = time.time() - t0

        n_gates = len(r.gates)
        inputs  = ",".join(sorted(r.global_inputs)[:4]) or "—"
        outputs = ",".join(sorted(r.global_outputs)[:2]) or "—"
        warns   = r.warnings

        # Classify result
        has_unconn = "UNCONNECTED" in (r.equations or "")
        if n_gates == 0:
            status = "FAIL"
        elif warns or has_unconn:
            status = "PARTIAL"
        else:
            status = "PASS"

        totals[status] += 1

        # Collect warning type stats
        for w in warns:
            wtype = w.split(":")[1].strip()[:35] if ":" in w else w[:35]
            warn_types[wtype] = warn_types.get(wtype, 0) + 1

        # Equation snippet (first line, truncated)
        eq_snip = (r.equations or "").splitlines()[0][:50] if r.equations else "—"

        warn_str = " | ".join(warns[:2]) if warns else eq_snip
        print(f"{idx:>3}  {status:8}  {n_gates:>2}  {inputs:12}  {outputs:10}  {warn_str[:60]}")

    except Exception as e:
        totals["ERROR"] += 1
        print(f"{idx:>3}  {'ERROR':8}   0  {'—':12}  {'—':10}  {type(e).__name__}: {str(e)[:50]}")

total = len(imgs)
print("=" * 90)
print(f"\nSUMMARY  ({total} images)")
print(f"  PASS    : {totals['PASS']:>4}  ({100*totals['PASS']/total:.1f}%)")
print(f"  PARTIAL : {totals['PARTIAL']:>4}  ({100*totals['PARTIAL']/total:.1f}%)")
print(f"  FAIL    : {totals['FAIL']:>4}  ({100*totals['FAIL']/total:.1f}%)")
print(f"  ERROR   : {totals['ERROR']:>4}  ({100*totals['ERROR']/total:.1f}%)")

if warn_types:
    print(f"\nTop warning types:")
    for wt, cnt in sorted(warn_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cnt:>4}x  {wt}")
