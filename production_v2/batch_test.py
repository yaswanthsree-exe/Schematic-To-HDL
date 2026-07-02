"""
batch_test.py — Run every image in Digital_train_data through production_v2/predict.py
and produce a structured pass/fail/error report.

Usage:
    python batch_test.py

Pass criteria (auto-checkable):
  PASS   — at least 1 gate detected, ≥1 primary input, ≥1 output, no exception
  PARTIAL— gates detected but 0 primary inputs OR 0 outputs (connectivity gap)
  ERROR  — Python exception / predict.py crashed
  NO_GATE— 0 gates detected (YOLO missed everything)
"""

import os
import subprocess
import sys
import re
from collections import defaultdict

PREDICT = os.path.join(os.path.dirname(__file__), "predict.py")
DATA_ROOT = r"C:\Yaswanth\Yash\schematic_to_netlist_backupexp\Digital_train_data"

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_output(stdout: str, stderr: str):
    """Extract structured info from predict.py combined output."""
    text = stdout + stderr

    # Gate count
    m = re.search(r"Detected\s+(\d+)\s+gate", text)
    gates = int(m.group(1)) if m else 0

    # Primary inputs list
    m = re.search(r"Primary inputs:\s*\[([^\]]*)\]", text)
    inputs_raw = m.group(1) if m else ""
    inputs = [x.strip().strip("'\"") for x in inputs_raw.split(",") if x.strip().strip("'\"")] if inputs_raw.strip() else []

    # Primary outputs list
    m = re.search(r"Outputs:\s*\[([^\]]*)\]", text)
    outputs_raw = m.group(1) if m else ""
    outputs = [x.strip().strip("'\"") for x in outputs_raw.split(",") if x.strip().strip("'\"")] if outputs_raw.strip() else []

    # Equations (one per output)
    equations = re.findall(r"^\s*(\w[\w+' ]*)\s*=\s*(.+)$", text, re.MULTILINE)
    # Keep only non-assign lines (raw equations)
    eqs = {k.strip(): v.strip() for k, v in equations
           if not k.strip().startswith("assign") and "=" not in k}

    # EQUATIONS block (cleaner)
    eq_block_m = re.search(r"=== EQUATIONS ===\n(.*?)(?:===|\Z)", text, re.DOTALL)
    eq_block = eq_block_m.group(1).strip() if eq_block_m else ""

    # Warnings
    warnings = re.findall(r"WARNING.*", text)

    return {
        "gates": gates,
        "inputs": inputs,
        "outputs": outputs,
        "eq_block": eq_block,
        "warnings": warnings,
    }


def classify(info: dict, crashed: bool) -> str:
    if crashed:
        return "ERROR"
    if info["gates"] == 0:
        return "NO_GATE"
    if not info["inputs"] or not info["outputs"]:
        return "PARTIAL"
    return "PASS"


# ── collect images ────────────────────────────────────────────────────────────

categories = {}
for cat in os.listdir(DATA_ROOT):
    cat_path = os.path.join(DATA_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    imgs = sorted([
        os.path.join(cat_path, f)
        for f in os.listdir(cat_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])
    categories[cat] = imgs

all_images = [(cat, p) for cat, imgs in categories.items() for p in imgs]
total = len(all_images)
print(f"\nFound {total} images across {len(categories)} categories: {list(categories.keys())}\n")
print("=" * 80)

# ── run tests ─────────────────────────────────────────────────────────────────

results = []     # list of dicts
stats = defaultdict(lambda: defaultdict(int))   # stats[cat][status]

for idx, (cat, img_path) in enumerate(all_images):
    short = os.path.basename(img_path)
    print(f"[{idx+1:3d}/{total}] {cat}/{short}  ", end="", flush=True)

    try:
        proc = subprocess.run(
            [sys.executable, PREDICT, img_path],
            capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
        crashed = proc.returncode != 0
        info = parse_output(proc.stdout, proc.stderr)
        status = classify(info, crashed)

        # Capture any traceback for ERROR cases
        tb = ""
        if crashed:
            tb_lines = [l for l in (proc.stdout + proc.stderr).splitlines()
                        if "Error" in l or "Traceback" in l or "Exception" in l]
            tb = " | ".join(tb_lines[:3])

    except subprocess.TimeoutExpired:
        crashed = True
        info = {"gates": 0, "inputs": [], "outputs": [], "eq_block": "", "warnings": []}
        status = "ERROR"
        tb = "TIMEOUT (>120s)"

    except Exception as e:
        crashed = True
        info = {"gates": 0, "inputs": [], "outputs": [], "eq_block": "", "warnings": []}
        status = "ERROR"
        tb = str(e)

    results.append({
        "cat": cat, "img": short, "status": status,
        "gates": info["gates"],
        "inputs": info["inputs"],
        "outputs": info["outputs"],
        "eq_block": info["eq_block"],
        "tb": tb,
    })
    stats[cat][status] += 1

    # One-line summary
    tag = {"PASS": "OK", "PARTIAL": "~~", "NO_GATE": "NG", "ERROR": "ER"}[status]
    detail = f"gates={info['gates']} inputs={info['inputs']} outputs={info['outputs']}"
    print(f"{tag}  {detail}")
    if status == "ERROR":
        print(f"      ERR: {tb[:120]}")

# ── per-category summary ──────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY BY CATEGORY")
print("=" * 80)
for cat in sorted(categories.keys()):
    n = len(categories[cat])
    s = stats[cat]
    print(f"\n  {cat}  ({n} images)")
    print(f"    PASS    : {s['PASS']:3d} / {n}  ({100*s['PASS']//n if n else 0}%)")
    print(f"    PARTIAL : {s['PARTIAL']:3d} / {n}")
    print(f"    NO_GATE : {s['NO_GATE']:3d} / {n}")
    print(f"    ERROR   : {s['ERROR']:3d} / {n}")

# ── failures detail ───────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("FAILURES / PARTIALS DETAIL")
print("=" * 80)
for r in results:
    if r["status"] in ("PARTIAL", "NO_GATE", "ERROR"):
        print(f"\n  [{r['status']}] {r['cat']}/{r['img']}")
        print(f"         gates={r['gates']}  inputs={r['inputs']}  outputs={r['outputs']}")
        if r["eq_block"]:
            for line in r["eq_block"].splitlines():
                print(f"         {line}")
        if r["tb"]:
            print(f"         ERR: {r['tb'][:200]}")

# ── overall totals ────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
total_pass = sum(stats[c]["PASS"] for c in stats)
total_partial = sum(stats[c]["PARTIAL"] for c in stats)
total_no_gate = sum(stats[c]["NO_GATE"] for c in stats)
total_error = sum(stats[c]["ERROR"] for c in stats)
print(f"OVERALL: {total}/{total} images processed")
print(f"  PASS    : {total_pass} ({100*total_pass//total if total else 0}%)")
print(f"  PARTIAL : {total_partial}")
print(f"  NO_GATE : {total_no_gate}")
print(f"  ERROR   : {total_error}")
print("=" * 80)

# ── write CSV report ──────────────────────────────────────────────────────────

csv_path = os.path.join(os.path.dirname(__file__), "test_report.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("category,image,status,gates,n_inputs,n_outputs,inputs,outputs,equations\n")
    for r in results:
        eq = r["eq_block"].replace("\n", " | ").replace(",", ";")
        f.write(
            f"{r['cat']},{r['img']},{r['status']},{r['gates']},"
            f"{len(r['inputs'])},{len(r['outputs'])},"
            f"\"{';'.join(r['inputs'])}\",\"{';'.join(r['outputs'])}\","
            f"\"{eq}\"\n"
        )
print(f"\nCSV report written: {csv_path}")
