"""Quick test: run predict on a handful of images and report NOT-gate results."""
import sys, os, glob
sys.path.insert(0, '.')
from predict import find_best_model, predict_circuit, find_gate_classifier, load_gate_classifier
from ultralytics import YOLO
import logging
logging.disable(logging.CRITICAL)  # suppress INFO spam

model = YOLO(find_best_model())
clf_path = find_gate_classifier()
classifier = load_gate_classifier(clf_path) if clf_path else None

test_dirs = [r'C:\Yaswanth\Yash\schematic_to_netlist_backupexp']
all_imgs = []
for d in test_dirs:
    all_imgs += glob.glob(d + r'\**\*.png', recursive=True)
    all_imgs += glob.glob(d + r'\**\*.jpg', recursive=True)

filtered = [i for i in all_imgs
            if 'runs' not in i and 'debug' not in i and 'generated' not in i]

print(f"Scanning {len(filtered)} images...")
not_tested = 0
not_fixed  = 0
not_still_broken = 0

for img in filtered[:80]:
    try:
        r = predict_circuit(img, model=model, classifier=classifier)
        not_gates = [g for g in r.gates if g['cls'] in ('NOT','BUF')]
        if not not_gates:
            continue
        not_tested += 1
        warn = [w for w in r.warnings if 'no input' in w]
        name = os.path.basename(img)
        if warn:
            not_still_broken += 1
            print(f"  STILL BROKEN: {name:50s}  warns={warn}")
        else:
            not_fixed += 1
            print(f"  OK:           {name:50s}  NOT gates={len(not_gates)}  eq={r.equations[:80].strip()}")
    except Exception as e:
        pass

print(f"\nImages with NOT/BUF gates: {not_tested}")
print(f"  Fixed (no warning):    {not_fixed}")
print(f"  Still missing input:   {not_still_broken}")
