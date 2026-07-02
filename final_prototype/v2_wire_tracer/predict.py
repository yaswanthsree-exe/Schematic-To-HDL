import cv2
import numpy as np
import sys
import os
from collections import defaultdict
from ultralytics import YOLO

# 7 classes from Roboflow dataset (alphabetical order)
CLASSES = ["AND", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]

def find_best_model():
    """Find the most recent best.pt in runs/ directory."""
    import glob
    model_paths = glob.glob("runs/**/best.pt", recursive=True)
    if not model_paths:
        return None
    return max(model_paths, key=os.path.getctime)

def detect_gates(image_path, model):
    """Stage 1: Run YOLO to detect all logic gates."""
    results = model(image_path, conf=0.25)[0]
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return None, None
        
    h_img, w_img = img.shape[:2]
    
    boxes = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w, h = x2 - x1, y2 - y1
        
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"UNK_{cls_id}"
        boxes.append({
            "id": f"Gate_{i}", 
            "cls": cls_name, 
            "conf": conf,
            "x": x1, "y": y1, "w": w, "h": h
        })
        
    if not boxes:
        print("YOLO found no logic gates.")
        return None, None
        
    # Sort left-to-right for topological ordering
    boxes.sort(key=lambda b: b['x'])
    
    print(f"Detected {len(boxes)} gates:")
    for b in boxes:
        print(f"  {b['cls']} (conf={b['conf']:.2f}) at ({b['x']},{b['y']})")
    
    return boxes, img

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def trace_wires(img, boxes):
    """Stage 2: Isolate wires using connected components with crossing separation."""
    h_img, w_img = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Mask out all detected gate bounding boxes
    for b in boxes:
        pad = 2
        cv2.rectangle(thresh, 
                       (max(0, b['x']-pad), max(0, b['y']-pad)), 
                       (min(w_img, b['x']+b['w']+pad), min(h_img, b['y']+b['h']+pad)), 
                       0, -1)
                       
    # Dilate slightly to bridge any 1-2 pixel gaps in drawing
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Handle crossings mathematically by breaking intersections without large dots
    kernel_dot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dots = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_dot)
    
    kernel_h = np.ones((1, 15), np.uint8)
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    
    kernel_v = np.ones((15, 1), np.uint8)
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    
    intersections = cv2.bitwise_and(horiz, vert)
    dot_mask = cv2.dilate(dots, np.ones((5,5), np.uint8))
    crossings = cv2.bitwise_and(intersections, cv2.bitwise_not(dot_mask))
    
    # Break crossings
    cross_dilate = cv2.dilate(crossings, np.ones((5,5), np.uint8))
    thresh_broken = cv2.bitwise_and(thresh, cv2.bitwise_not(cross_dilate))
                       
    # Connected Components on the broken topology
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh_broken)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 10:
            labels_im[labels_im == i] = 0
            
    # Re-link corresponding endpoints spanning across the gap of broken crossings 
    uf = UnionFind(num_labels)
    cross_ccs, cross_labels = cv2.connectedComponents(crossings)
    
    def get_id(x, y):
        x, y = int(x), int(y)
        window = labels_im[max(0, y-2):min(h_img, y+3), max(0, x-2):min(w_img, x+3)]
        ids = np.unique(window)
        ids = ids[ids > 0]
        if len(ids) > 0: return ids[0]
        return 0
        
    for c in range(1, cross_ccs):
        ys, xs = np.where(cross_labels == c)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        
        l_id = get_id(cx - 5, cy)
        r_id = get_id(cx + 5, cy)
        t_id = get_id(cx, cy - 5)
        b_id = get_id(cx, cy + 5)
        
        # Pass-through for True Crossings (4-way)
        if l_id > 0 and r_id > 0: uf.union(l_id, r_id)
        if t_id > 0 and b_id > 0: uf.union(t_id, b_id)
        
        # Corners (L-junctions)
        if l_id > 0 and t_id > 0 and r_id == 0 and b_id == 0: uf.union(l_id, t_id)
        if r_id > 0 and t_id > 0 and l_id == 0 and b_id == 0: uf.union(r_id, t_id)
        if l_id > 0 and b_id > 0 and r_id == 0 and t_id == 0: uf.union(l_id, b_id)
        if r_id > 0 and b_id > 0 and l_id == 0 and t_id == 0: uf.union(r_id, b_id)
        
        # T-Junctions
        if (l_id > 0 and r_id > 0) and t_id > 0 and b_id == 0: uf.union(l_id, t_id)
        if (l_id > 0 and r_id > 0) and b_id > 0 and t_id == 0: uf.union(l_id, b_id)
        if (t_id > 0 and b_id > 0) and l_id > 0 and r_id == 0: uf.union(t_id, l_id)
        if (t_id > 0 and b_id > 0) and r_id > 0 and l_id == 0: uf.union(t_id, r_id)
        
    mapping = np.zeros(num_labels, dtype=np.int32)
    for i in range(num_labels):
        mapping[i] = uf.find(i)
        
    labels_im = mapping[labels_im]
    return num_labels, labels_im

def build_graph(boxes, num_labels, labels_im, img_shape):
    """Stage 3+4: Proximity analysis -> dependency graph."""
    h_img, w_img = img_shape[:2]
    
    wire_touches_input_of = defaultdict(list)
    wire_touches_output_of = defaultdict(list)
    
    print("Computing distance-based pin assignments...")
    wire_pixels = {}
    for w_id in range(1, num_labels):
        ys, xs = np.where(labels_im == w_id)
        if len(xs) > 0:
            wire_pixels[w_id] = np.column_stack((xs, ys))
            
    MAX_SNAP_DIST = 45.0  # Search radius for disconnected wire endpoints
    
    for b in boxes:
        out_x = b['x'] + b['w']
        out_y = b['y'] + b['h'] // 2
        in_x = b['x']
        
        for w_id, pts in wire_pixels.items():
            # Check Output Pin (Right side of gate)
            valid_out_pts = pts[pts[:, 0] >= out_x - 10]
            if len(valid_out_pts) > 0:
                dist_out = np.min(np.sqrt((valid_out_pts[:, 0] - out_x)**2 + (valid_out_pts[:, 1] - out_y)**2))
                if dist_out <= MAX_SNAP_DIST:
                    wire_touches_output_of[w_id].append(b['id'])
                    
            # Check Input Pins (Left side of gate, any Y along the height)
            valid_in_pts = pts[pts[:, 0] <= in_x + 10]
            if len(valid_in_pts) > 0:
                dx = np.abs(valid_in_pts[:, 0] - in_x)
                dy = np.maximum(0, np.maximum(b['y'] - valid_in_pts[:, 1], valid_in_pts[:, 1] - (b['y'] + b['h'])))
                dist_in = np.min(np.sqrt(dx**2 + dy**2))
                if dist_in <= MAX_SNAP_DIST:
                    wire_touches_input_of[w_id].append(b['id'])

    # Build logical dependency graph
    graph = {b['id']: {"cls": b['cls'], "inputs": [], "outputs": []} for b in boxes}
    
    global_inputs = set()
    global_outputs = set()
    input_counter = 65  # 'A'
    out_counter = 1
    
    for w_id in range(1, num_labels):
        sources = wire_touches_output_of.get(w_id, [])
        targets = wire_touches_input_of.get(w_id, [])
        
        # Wire enters a gate but didn't exit any -> Global input
        if not sources and targets:
            name = chr(input_counter)
            input_counter = input_counter + 1 if input_counter < 90 else 65
            global_inputs.add(name)
            for t in targets:
                graph[t]["inputs"].append(name)
                
        # Wire exits a gate but doesn't enter another -> Global output
        elif sources and not targets:
            for s in set(sources): # Outputs usually don't duplicate
                name = f"OUT_{out_counter}"
                out_counter += 1
                graph[s]["outputs"].append(name)
                global_outputs.add(name)
                
        # Wire connects two gates -> Internal net
        elif sources and targets:
            for s in set(sources):
                for t in targets:
                    if s != t: # Prevent self-loops
                        graph[t]["inputs"].append(s)

    # Post-process: NOT gates and BUF gates can only have 1 input!
    for b in boxes:
        node = graph[b['id']]
        if node["cls"] in ["NOT", "BUF"]:
            if len(node["inputs"]) > 1:
                # Prioritize distinct external inputs, or just take the first
                unique_inputs = list(dict.fromkeys(node["inputs"]))
                node["inputs"] = unique_inputs[:1]
        else:
            # Deduplicate inputs
            node["inputs"] = list(dict.fromkeys(node["inputs"]))

    return graph, global_inputs, global_outputs

def generate_netlist(boxes, graph, global_outputs):
    """Stage 5: Generate netlist and Boolean equations."""
    new_names = {b['id']: f"G{i+1}" for i, b in enumerate(boxes)}
    netlist_lines = []
    final_eqs = {}
    
    for i, b in enumerate(boxes):
        old_id = b['id']
        node = graph[old_id]
        gate_type = node["cls"]
        new_id = new_names[old_id]
        
        # Netlist line
        resolved_inputs = [new_names.get(inp, inp) for inp in node["inputs"]]
        if not resolved_inputs: 
            resolved_inputs = ["UNKNOWN_IN"]
            
        args = ", ".join(resolved_inputs)
        netlist_lines.append(f"{new_id} = {gate_type}({args})")
        
        # Boolean equation (recursive substitution)
        eq_args = [final_eqs.get(inp, inp) for inp in node["inputs"]]
        
        if len(eq_args) == 1:
            if gate_type in ["NOT"]:
                eq_str = f"(NOT {eq_args[0]})"
            else:
                eq_str = f"({gate_type}({eq_args[0]}))"
        elif len(eq_args) > 1:
            op = f" {gate_type} "
            eq_str = f"({op.join(eq_args)})"
        else:
            eq_str = "UNKNOWN_IN"
            
        final_eqs[old_id] = eq_str
        
        for out_name in node["outputs"]:
            netlist_lines.append(f"{out_name} = {new_id}")
            
    netlist_str = "; ".join(netlist_lines)
    
    # Build final output equations
    if global_outputs:
        out_str = []
        for node_id, node in graph.items():
            if node["outputs"]:
                for out_name in node["outputs"]:
                    out_str.append(f"{out_name} = {final_eqs[node_id]}")
        eq_final = " ; ".join(out_str)
    else:
        # Fallback: assume rightmost gate is the output
        last_gate = boxes[-1]['id']
        eq_final = f"Q = {final_eqs[last_gate]}"
        netlist_str += f"; Q = {new_names[last_gate]}"
        
    return netlist_str, eq_final

def predict_circuit(image_path):
    """Full end-to-end pipeline: Image → Gates → Wires → Netlist + Equations"""
    
    # Find model
    model_path = find_best_model()
    if not model_path:
        print("Error: No trained model found in runs/ directory.")
        print("Run train_yolo.py first to train on the Roboflow dataset.")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Stage 1: Gate Detection
    print(f"\n{'='*50}")
    print(f"STAGE 1: Detecting gates in {os.path.basename(image_path)}")
    print(f"{'='*50}")
    boxes, img = detect_gates(image_path, model)
    if boxes is None:
        return
    
    # Stage 2: Wire Tracing
    print(f"\n{'='*50}")
    print(f"STAGE 2: Tracing wires")
    print(f"{'='*50}")
    num_labels, labels_im = trace_wires(img, boxes)
    print(f"Found {num_labels - 1} wire segments")
    
    # Stage 3+4: Graph Construction
    print(f"\n{'='*50}")
    print(f"STAGE 3: Building circuit graph")
    print(f"{'='*50}")
    graph, global_inputs, global_outputs = build_graph(boxes, num_labels, labels_im, img.shape)
    print(f"Global inputs:  {global_inputs}")
    print(f"Global outputs: {global_outputs}")
    
    # Stage 5: Netlist + Equations
    print(f"\n{'='*50}")
    print(f"STAGE 4: Generating netlist & equations")
    print(f"{'='*50}")
    netlist_str, eq_final = generate_netlist(boxes, graph, global_outputs)
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"NETLIST:   {netlist_str}")
    print(f"EQUATION:  {eq_final}")
    print(f"{'='*50}\n")
    
    # Save annotated image
    out_path = os.path.splitext(image_path)[0] + "_result.png"
    annotated = img.copy()
    for b in boxes:
        cv2.rectangle(annotated, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), (0, 255, 0), 2)
        cv2.putText(annotated, f"{b['cls']}", (b['x'], b['y']-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(out_path, annotated)
    print(f"Annotated image saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_schematic_image>")
    else:
        predict_circuit(sys.argv[1])
