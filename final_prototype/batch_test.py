"""Run predict_v2 on every test image and write a summary report."""
import os, glob, sys, traceback, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_v2 import predict_circuit, find_best_model
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "Digital_train_data")
OUT  = os.path.join(ROOT, "batch_results.txt")

def main(limit_per_dir: int = 6):
    model = YOLO(find_best_model())
    lines = []
    for sub in sorted(os.listdir(DATA)):
        d = os.path.join(DATA, sub)
        if not os.path.isdir(d): continue
        files = [f for f in sorted(glob.glob(os.path.join(d, "*.png")))
                   if not f.endswith("_result.png")][:limit_per_dir]
        for f in files:
            stem = os.path.basename(f)
            try:
                t = time.time()
                r = predict_circuit(f, model=model)
                dt = time.time() - t
                line = (
                    f"\n[OK {dt:.1f}s] {sub}/{stem}\n"
                    f"  gates: {[(g['cls'], round(g['conf'], 2)) for g in r.gates]}\n"
                    f"  inputs: {sorted(r.global_inputs)}  outputs: {sorted(r.global_outputs)}\n"
                    f"  netlist:\n    " + r.netlist.replace("\n", "\n    ") + "\n"
                    f"  warnings: {r.warnings}"
                )
            except Exception as e:
                line = f"\n[ERR] {sub}/{stem}: {type(e).__name__}: {e}"
            print(line); lines.append(line)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n=== Wrote {OUT} ===")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(n)
