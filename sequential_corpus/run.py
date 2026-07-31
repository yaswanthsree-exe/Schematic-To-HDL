"""Sequential corpus runner: score the pipeline against hand-labelled truth.

The 177-image regression corpus is entirely COMBINATIONAL, so it proves only
that combinational circuits still work.  Every sequential claim was resting on
synthetic graphs plus ad-hoc images.  This runs the real pipeline over labelled
flip-flop and latch schematics and reports what was recognised versus what is
actually drawn.

Usage:
    python sequential_corpus/run.py [out.json]

Exit status is 0 always -- this is a measurement, not a gate.  Compare two runs
with --compare to see whether a change helped or hurt.
"""
import contextlib
import io
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "production_v2"))
logging.disable(logging.CRITICAL)

IMAGES = os.path.join(HERE, "images")
MANIFEST = os.path.join(HERE, "manifest.json")


def _load_pipeline():
    from ultralytics import YOLO

    import predict as P
    pv2 = os.path.join(ROOT, "production_v2")
    model = YOLO(P.find_best_model(pv2))
    cpath = P.find_gate_classifier(pv2)
    clf = P.load_gate_classifier(cpath) if cpath else None
    return P, model, clf


def _block_reader():
    """Lazily build the OCR-backed block reader used by app_final."""
    import cv2
    import easyocr
    from pattern_engine.block_form import blocks_from_ocr, graph_from_blocks
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(path):
        bgr = cv2.imread(path)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or gray is None:
            return {}
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            results = reader.readtext(bgr)

        def reocr(x, y, w, h):
            pad = int(0.25 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            crop = bgr[y0:min(bgr.shape[0], y + h + pad),
                       x0:min(bgr.shape[1], x + w + pad)]
            if crop.size == 0:
                return []
            scale = max(1.0, 900.0 / max(crop.shape[:2]))
            big = cv2.resize(crop, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                found = reader.readtext(big, text_threshold=0.5, low_text=0.3)
            return [([[p[0] / scale + x0, p[1] / scale + y0] for p in poly], t, c)
                    for poly, t, c in found]

        return graph_from_blocks(blocks_from_ocr(gray, results, reocr=reocr))
    return read


def _degenerate(graph):
    """No gate drives another: nothing for the pattern engine to work with."""
    return not any(src in graph
                   for node in graph.values()
                   for src in node.get("inputs", ()))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    truth = json.load(open(MANIFEST, encoding="utf-8"))
    truth.pop("_comment", None)

    P, model, clf = _load_pipeline()
    read_blocks = _block_reader()
    from pattern_engine import compress

    rows, buf = {}, io.StringIO()
    for name in sorted(truth):
        path = os.path.join(IMAGES, name)
        if not os.path.exists(path):
            rows[name] = {"ok": False, "error": "image missing"}
            continue
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                r = P.predict_circuit(path, model=model, classifier=clf,
                                      debug_root=None)
                got = [m.cls for m in compress(r.graph).matches]
                used_block = False
                # Fall back to the block path whenever the gate path recognised
                # nothing.  Gating on "no gates wired together" was too strict:
                # a block symbol's edge-trigger triangles and box edges are
                # detected as a few stray gates that happen to be wired, so a
                # two-symbol master-slave never reached the block reader.
                if not got:
                    bg = read_blocks(path)
                    if bg:
                        r.graph, used_block = bg, True
                        got = [n["cls"] for n in bg.values()]
            rows[name] = {
                "ok": True, "path": "block" if used_block else "gate",
                "got": sorted(got), "expect": sorted(truth[name]["expect"]),
                "pass": sorted(got) == sorted(truth[name]["expect"]),
                "gates": len(r.graph),
                "gate_edges": sum(1 for n in r.graph.values()
                                  for s in n.get("inputs", ()) if s in r.graph),
            }
        except Exception as exc:                       # noqa: BLE001
            rows[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    npass = sum(1 for v in rows.values() if v.get("pass"))
    gate_rows = {k: v for k, v in rows.items() if truth[k]["form"] == "gate"}
    block_rows = {k: v for k, v in rows.items() if truth[k]["form"] == "block"}
    gp = sum(1 for v in gate_rows.values() if v.get("pass"))
    bp = sum(1 for v in block_rows.values() if v.get("pass"))

    print(f"{'file':18s} {'form':6s} {'expected':34s} {'got':34s} result")
    print("-" * 108)
    for name in sorted(rows, key=lambda n: (truth[n]["form"], n)):
        v, t = rows[name], truth[name]
        if not v["ok"]:
            print(f"{name:18s} {t['form']:6s} {'':34s} {'':34s} ERROR {v['error'][:40]}")
            continue
        mark = "PASS" if v["pass"] else "fail"
        print(f"{name:18s} {t['form']:6s} {str(v['expect']):34s} "
              f"{str(v['got'] or '-'):34s} {mark}")

    print()
    print(f"TOTAL   {npass}/{len(rows)}    "
          f"gate-level {gp}/{len(gate_rows)}    block-form {bp}/{len(block_rows)}")

    if out_path:
        json.dump(rows, open(out_path, "w", encoding="utf-8"),
                  indent=1, sort_keys=True)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
