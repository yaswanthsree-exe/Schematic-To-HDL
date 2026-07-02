"""
debug_why_fail.py
=================
Systematically diagnoses WHY gates still get 'no input found' warnings.

For each failing gate, reports EXACTLY where in the pipeline the assignment
breaks down:

  PASS1/2 : directional search (tight + wide radius)
  PASS3   : pixel proximity on binary
  PASS4   : post_correct_missing wider search
  FINAL   : what build_gate_graph sees after net construction

Run:
    cd production_v2
    python debug_why_fail.py [optional_image_path]
"""
import sys, os, glob, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.CRITICAL)

import numpy as np
import predict as _p
from predict import (
    find_best_model, find_gate_classifier, load_gate_classifier,
    gate_pin_centers, _find_nearest_wire, _build_edge_label_img,
    _edge_euclidean_len, GATE_N_IN, POST_CORRECT_R, MIN_EDGE_LEN, MIN_PRIMARY_PIX,
    SNAP_R_TIGHT, SNAP_R_WIDE, SNAP_R_PIXEL, PIN_MARGIN,
)
from ultralytics import YOLO
from collections import defaultdict

# ── Monkey-patch assign_endpoints and post_correct_missing to capture state ────

_captured = {}  # will store state for post-analysis

_orig_assign = _p.assign_endpoints
_orig_post   = _p.post_correct_missing

def _patched_assign(skel_graph, boxes, binary):
    result = _orig_assign(skel_graph, boxes, binary)
    _captured['assignment']  = dict(result)
    _captured['skel_graph']  = skel_graph
    _captured['binary']      = binary
    _captured['boxes']       = boxes
    return result

def _patched_post(boxes, skel_graph, assignment, binary, radius=POST_CORRECT_R):
    result = _orig_post(boxes, skel_graph, assignment, binary, radius)
    _captured['assignment_after_post'] = dict(result)
    return result

_p.assign_endpoints      = _patched_assign
_p.post_correct_missing  = _patched_post


def _diagnose_image(img_path: str, model, classifier) -> None:
    _captured.clear()
    r = _p.predict_circuit(img_path, model=model, classifier=classifier)

    if not r.warnings:
        print(f"  ✓ PASS — no warnings")
        return

    # Only care about "no input" warnings
    no_input_warns = [w for w in r.warnings if 'no input' in w]
    if not no_input_warns:
        print(f"  ≈ Other warnings: {r.warnings}")
        return

    print(f"  ✗ FAIL — {no_input_warns}")

    skel_graph = _captured.get('skel_graph')
    binary     = _captured.get('binary')
    boxes_list = _captured.get('boxes', [])
    assign_pre = _captured.get('assignment', {})
    assign_post= _captured.get('assignment_after_post', {})

    if skel_graph is None:
        print("    [no captured state — pipeline may have errored]")
        return

    # Build edge label images (same as pipeline uses)
    lbl_pass3  = _build_edge_label_img(skel_graph, binary.shape, min_len=MIN_EDGE_LEN)
    lbl_pass4  = _build_edge_label_img(skel_graph, binary.shape, min_len=0)

    # Map gate id → box
    box_by_id = {b['id']: b for b in boxes_list}

    # Which gates have "no input" warnings?
    failing_gates = {}
    for w in no_input_warns:
        # Format: "G3 (NOT): no input found"
        parts = w.split()
        gid   = parts[0]          # e.g. "G3"
        cls   = parts[1].strip('():')  # e.g. "NOT"
        failing_gates[gid] = cls

    for gid, cls in failing_gates.items():
        b = box_by_id.get(gid)
        if b is None:
            print(f"\n    [{gid}] box not found in captured state")
            continue

        n_in     = b.get('fan_in', GATE_N_IN.get(cls, 2))
        centers  = gate_pin_centers(b)
        gate_cx  = b['x'] + b['w'] / 2.0
        gate_cy  = b['y'] + b['h'] / 2.0
        is_single = cls in ('NOT', 'BUF')

        # What was assigned to output?
        out_key = (gid, 'out', 0)
        out_eid = assign_post.get(out_key, (None,))[0]

        print(f"\n    ── {gid} ({cls})  bbox=({b['x']},{b['y']},{b['w']}×{b['h']})  "
              f"gate_cx={gate_cx:.0f}  out_eid={out_eid}")

        for pidx, (px, py) in enumerate(centers['in'][:n_in]):
            pkey_pre  = (gid, 'in', pidx)
            pkey_post = (gid, 'in', pidx)

            pre_assigned  = pkey_pre  in assign_pre
            post_assigned = pkey_post in assign_post

            print(f"      Pin {pidx}: center=({px},{py})")
            print(f"        Pass1/2/3: {'ASSIGNED' if pre_assigned else 'MISSED'}", end='')
            if pre_assigned:
                eid, end = assign_pre[pkey_pre]
                edge = skel_graph.edges[eid]
                elen = _edge_euclidean_len(edge)
                ax, ay = edge.path[0]; bx, by = edge.path[-1]
                print(f" → eid={eid} end={end} len={elen:.0f}  "
                      f"path[0]=({ax},{ay}) path[-1]=({bx},{by})")
            else:
                print()

            print(f"        Pass4:     {'ASSIGNED' if post_assigned else 'MISSED'}", end='')
            if post_assigned:
                eid, end = assign_post[pkey_post]
                edge = skel_graph.edges[eid]
                elen = _edge_euclidean_len(edge)
                ax, ay = edge.path[0]; bx, by = edge.path[-1]
                print(f" → eid={eid} end={end} len={elen:.0f}  "
                      f"path[0]=({ax},{ay}) path[-1]=({bx},{by})")
            else:
                print()

            if not post_assigned:
                # Diagnose WHY pass4 also failed
                search_r = POST_CORRECT_R * 2 if is_single else POST_CORRECT_R
                lbl = _find_nearest_wire(px, py, lbl_pass4, search_r)

                # Also scan with a very large radius to find ANYTHING
                lbl_huge = _find_nearest_wire(px, py, lbl_pass4, 300)

                if lbl == 0:
                    print(f"        → _find_nearest_wire({search_r}px) returned 0")
                    if lbl_huge == 0:
                        print(f"          Even with 300px radius: NOTHING FOUND (skeleton gap?)")
                        # Count skeleton pixels near this pin
                        h, w = binary.shape
                        x0 = max(0, px - 150); x1 = min(w, px + 151)
                        y0 = max(0, py - 150); y1 = min(h, py + 151)
                        region = lbl_pass4[y0:y1, x0:x1]
                        n_edge_pix = int(np.count_nonzero(region))
                        # Find nearest labelled pixel
                        ys2, xs2 = np.where(region > 0)
                        if len(xs2) > 0:
                            dists2 = np.sqrt((xs2+x0-px)**2 + (ys2+y0-py)**2)
                            min_d  = float(dists2.min())
                            print(f"          300px region edge-pixels={n_edge_pix}  nearest={min_d:.0f}px")
                        else:
                            print(f"          300px region has ZERO skeleton pixels!")
                            # Check the raw binary near the pin
                            bin_r = binary[y0:y1, x0:x1]
                            n_bin = int(np.count_nonzero(bin_r))
                            print(f"          Binary pixels in 300px region: {n_bin}")
                    else:
                        eid_huge = lbl_huge - 1
                        edge_h = skel_graph.edges[eid_huge]
                        elen_h = _edge_euclidean_len(edge_h)
                        ax, ay = edge_h.path[0]; bx, by = edge_h.path[-1]
                        # How far is the nearest pixel?
                        h, w = binary.shape
                        x0 = max(0, px-300); x1 = min(w, px+301)
                        y0 = max(0, py-300); y1 = min(h, py+301)
                        region = lbl_pass4[y0:y1, x0:x1]
                        ys2, xs2 = np.where(region > 0)
                        dists2 = np.sqrt((xs2+x0-px)**2 + (ys2+y0-py)**2)
                        min_d = float(dists2.min())
                        print(f"          Nearest edge with 300px: eid={eid_huge} len={elen_h:.0f}px  "
                              f"nearest-pixel-dist={min_d:.0f}px")
                else:
                    eid_fb = lbl - 1
                    if eid_fb == out_eid:
                        print(f"        → Found eid={eid_fb} but it IS the output edge (blocked)")
                        # What's the distance to the output wire pixel?
                        edge_out = skel_graph.edges[eid_fb]
                        ax, ay = edge_out.path[0]; bx, by = edge_out.path[-1]
                        d_a = ((ax-px)**2 + (ay-py)**2)**0.5
                        d_b = ((bx-px)**2 + (by-py)**2)**0.5
                        print(f"          Output edge path[0]=({ax},{ay}) d={d_a:.0f}  "
                              f"path[-1]=({bx},{by}) d={d_b:.0f}")
                        # Is there truly NOTHING else?
                        ys2, xs2 = np.where(lbl_pass4 > 0)
                        if len(xs2) > 0:
                            dists2 = np.sqrt((xs2-px)**2 + (ys2-py)**2)
                            # exclude output edge pixels
                            mask_notout = lbl_pass4[ys2, xs2] != (eid_fb + 1)
                            if mask_notout.any():
                                d_nonout = float(dists2[mask_notout].min())
                                print(f"          Nearest non-output edge pixel: {d_nonout:.0f}px away")
                            else:
                                print(f"          NO non-output edge pixels anywhere!")
                    else:
                        eid_fb = lbl - 1
                        edge_fb = skel_graph.edges[eid_fb]
                        elen_fb = _edge_euclidean_len(edge_fb)
                        print(f"        → Found eid={eid_fb} len={elen_fb:.0f}px BUT post4 didn't assign it??")
                        print(f"          [BUG: should have been assigned]")


# ── Main ───────────────────────────────────────────────────────────────────────

IMG_DIR = r"C:\Users\yaswa\.gemini\antigravity\scratch\schematic_to_netlist\Digital_train_data\random"

model      = YOLO(find_best_model())
clf_path   = find_gate_classifier()
classifier = load_gate_classifier(clf_path) if clf_path else None

# Use command-line argument or default to known failing images
if len(sys.argv) > 1:
    test_imgs = sys.argv[1:]
else:
    # Known failing cases from previous batch runs
    known_fail = [
        "20-combinational_circuit_result.png",
        "test_curve_routing.png",
        "test_fixed_decoder.png",
        "stamp_schem_1.png",
        "stamp_schem_2.png",
        "stamp_schem_3.png",
        "stamp_schem_4.png",
        "stamp_schem_1001.png",
        "stamp_schem_1002.png",
        "stamp_schem_102.png",
        "stamp_schem_1004.png",
    ]
    # Also add random dir images
    all_random = sorted(glob.glob(IMG_DIR + r"\*.png"))
    test_imgs  = []
    for nm in known_fail:
        # Search in random dir first, then sibling dirs
        for root in [IMG_DIR,
                     os.path.dirname(IMG_DIR),
                     os.path.join(os.path.dirname(IMG_DIR), "stamp_schem"),
                     os.path.dirname(os.path.abspath(__file__))]:
            p = os.path.join(root, nm)
            if os.path.isfile(p):
                test_imgs.append(p); break
    # Fill up with random images that previously had warnings
    for p in all_random[:30]:
        if p not in test_imgs:
            test_imgs.append(p)

print(f"Diagnosing {len(test_imgs)} images\n" + "="*70)

fail_reasons = defaultdict(int)

for img_path in test_imgs:
    print(f"\n{os.path.basename(img_path)}")
    try:
        _diagnose_image(img_path, model, classifier)
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

print("\n" + "="*70)
print("Done.")
