"""Debug why split_merged_input_buses doesn't fix 'only 1 input' cases."""
import sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))

# Patch split_merged_input_buses with verbose logging
import predict as _p
import predict

# Temporarily set logging to DEBUG for predict module
logging.basicConfig(level=logging.WARNING, format='%(message)s')
log_split = logging.getLogger('predict.split')
log_split.setLevel(logging.DEBUG)

# Monkey-patch to add extra debug
_orig_split = predict.split_merged_input_buses

def _debug_split(skel_graph, assignment, edge_net, net_edges):
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for eid, edge in enumerate(skel_graph.edges):
        adj[edge.a].append((eid, edge.b))
        adj[edge.b].append((eid, edge.a))

    assigned_nodes = set()
    for (eid, end) in assignment.values():
        if eid < len(skel_graph.edges):
            e = skel_graph.edges[eid]
            assigned_nodes.add(e.a if end == 'a' else e.b)

    MIN_SPLIT_SHORT = 6
    MIN_PRIMARY_PIX = predict.MIN_PRIMARY_PIX

    net_free_eps_long  = defaultdict(list)
    net_free_eps_short = defaultdict(list)

    for nidx, (nx, ny) in enumerate(skel_graph.node_xy):
        if skel_graph.node_type[nidx] != 'endpoint':
            continue
        if nidx in assigned_nodes:
            continue
        for eid, _ in adj[nidx]:
            edge_len = predict._edge_euclidean_len(skel_graph.edges[eid])
            nid_e = edge_net[eid]
            if edge_len >= MIN_PRIMARY_PIX:
                net_free_eps_long[nid_e].append(nidx)
            elif edge_len >= MIN_SPLIT_SHORT:
                net_free_eps_short[nid_e].append(nidx)
            break

    net_input_pins = defaultdict(int)
    for (gid, side, pidx), (eid, _) in assignment.items():
        if side == 'in' and eid < len(edge_net):
            net_input_pins[edge_net[eid]] += 1

    # Find nets with n_inp >= 2
    print("\n=== BUS SPLIT DEBUG ===")
    print(f"Total nets with >=2 gate input pins:")
    for nid, count in sorted(net_input_pins.items(), key=lambda x: -x[1]):
        if count < 2:
            continue
        long_eps  = net_free_eps_long.get(nid, [])
        short_eps = net_free_eps_short.get(nid, [])
        n_long = len(long_eps)
        n_short = len(short_eps)
        print(f"  net {nid:4d}: gate_inputs={count}  free_long={n_long}  free_short={n_short}")
        if n_long >= 2:
            print(f"           → WOULD SPLIT (long eps sufficient)")
        elif count == 2 and (n_long + n_short) >= 2:
            print(f"           → WOULD SPLIT (supplemented with short eps)")
        elif n_long + n_short < 2:
            print(f"           → CANNOT SPLIT (< 2 free endpoints even with short threshold)")
        else:
            print(f"           → SKIPPED by guard")

    # Also print pin→edge assignments for inputs
    print("\n=== PIN ASSIGNMENTS (inputs only) ===")
    by_net = defaultdict(list)
    for (gid, side, pidx), (eid, end) in assignment.items():
        if side == 'in':
            nid = edge_net[eid]
            elen = predict._edge_euclidean_len(skel_graph.edges[eid])
            by_net[nid].append((gid, pidx, eid, end, elen))

    for nid, pins in sorted(by_net.items()):
        if len(set(p[0] for p in pins)) + len(pins) > 2:  # multiple pins on same net
            print(f"  net {nid}: {[(p[0], p[1], f'edge{p[2]}(len={p[4]:.0f})') for p in pins]}")

    return _orig_split(skel_graph, assignment, edge_net, net_edges)

predict.split_merged_input_buses = _debug_split

# Now run on the target image
from predict import find_best_model, predict_circuit, find_gate_classifier, load_gate_classifier
from ultralytics import YOLO
import glob

model = YOLO(find_best_model())
clf_path = find_gate_classifier()
classifier = load_gate_classifier(clf_path) if clf_path else None
logging.disable(logging.CRITICAL)

IMG_DIR = r"C:\Users\yaswa\.gemini\antigravity\scratch\schematic_to_netlist\Digital_train_data\random"
imgs = sorted(glob.glob(IMG_DIR + r"\*.png"))

# Test image 16 (idx 15, 0-based) — the 5-gate circuit showing OR(B,B)
for i in [15, 7, 11]:   # index 16, 8, 12 in 1-based
    img = imgs[i]
    print(f"\n{'='*60}")
    print(f"Image {i+1}: {os.path.basename(img)}")
    r = predict_circuit(img, model=model, classifier=classifier)
    print(f"Warnings: {r.warnings}")
    print(f"Equations: {r.equations[:150]}")
