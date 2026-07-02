"""
Comprehensive test: 15 diverse circuit images, checking for correctness.
Reports gates, equations, warnings, and suspicious spurious inputs.
"""
import sys, os, glob, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

from predict import find_best_model, predict_circuit, find_gate_classifier, load_gate_classifier
from ultralytics import YOLO

model      = YOLO(find_best_model())
clf_path   = find_gate_classifier()
classifier = load_gate_classifier(clf_path) if clf_path else None

BASE = r"C:\Yaswanth\Yash\schematic_to_netlist_backupexp"

# ── Pick a diverse set of 15 test images ─────────────────────────────────────
# Group 1: Half-adder (the circuit where "C" spurious input was seen)
half_adder_imgs = sorted(glob.glob(
    BASE + r"\Digital_train_data\full_half_adder\*.png"))[:5]

# Group 2: Half/full subtractor
subtractor_imgs = sorted(glob.glob(
    BASE + r"\Digital_train_data\half_full_sub\*.png"))[:3]

# Group 3: Random combinational circuits
random_imgs = sorted(glob.glob(
    BASE + r"\Digital_train_data\random\*.png"))[:4]

# Group 4: Internet images (most varied)
internet_imgs = sorted(glob.glob(BASE + r"\internet_images\*.png"))[:3]

all_tests = (
    [("HA",  p) for p in half_adder_imgs]
  + [("SUB", p) for p in subtractor_imgs]
  + [("RND", p) for p in random_imgs]
  + [("INT", p) for p in internet_imgs]
)

print(f"Testing {len(all_tests)} images\n" + "=" * 80)

pass_count = 0
warn_count = 0
err_count  = 0

for tag, img_path in all_tests:
    name = os.path.basename(img_path)
    try:
        r = predict_circuit(img_path, model=model, classifier=classifier)
        n_gates = len(r.gates)
        n_inputs = len(r.global_inputs)
        n_outputs = len(r.global_outputs)

        # Flag suspicious spurious inputs: more inputs than gates would need
        # (e.g. 5 primary inputs for a 2-input half-adder is suspicious)
        typical_max = max(n_gates + 2, 3)
        suspicious = n_inputs > typical_max

        eq_short = r.equations.replace('\n', ' | ')[:90]

        if r.warnings:
            warn_count += 1
            status = "WARN"
        elif suspicious:
            status = "SUSP"  # suspicious (too many primary inputs)
        else:
            pass_count += 1
            status = "PASS"

        print(f"[{status}][{tag}] {name}")
        print(f"       gates={n_gates}  inputs={sorted(r.global_inputs)}  "
              f"outputs={sorted(r.global_outputs)}")
        print(f"       eq: {eq_short}")
        if r.warnings:
            for w in r.warnings:
                print(f"       WARN: {w}")
        print()

    except Exception as e:
        err_count += 1
        import traceback
        print(f"[ERR ][{tag}] {name}")
        print(f"       {e}")
        print()

print("=" * 80)
print(f"PASS={pass_count}  WARN/SUSP={warn_count}  ERR={err_count}"
      f"  (out of {len(all_tests)} images)")
