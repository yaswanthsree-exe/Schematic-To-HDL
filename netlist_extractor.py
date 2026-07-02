import cv2
import numpy as np
import os
import glob
from collections import defaultdict

IMAGES_DIR = "internet_dataset/images/train"
LABELS_DIR = "internet_dataset/labels/train"
CLASSES = ["AND", "BUFFER", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]

def parse_labels(txt_path, img_w, img_h):
    boxes = []
    if not os.path.exists(txt_path): return boxes
    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                abs_w, abs_h = int(w*img_w), int(h*img_h)
                abs_x, abs_y = int(xc*img_w - abs_w/2), int(yc*img_h - abs_h/2)
                boxes.append({"id": f"Gate_{i}", "cls": CLASSES[cls_id], "x": abs_x, "y": abs_y, "w": abs_w, "h": abs_h})
                
    # Sort boxes horizontally (left to right) for naive topological sort G1, G2, etc.
    boxes.sort(key=lambda b: b['x'])
    return boxes

def extract_netlist(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    h_img, w_img = img.shape[:2]
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    lbl_path = os.path.join(LABELS_DIR, f"{name}.txt")
    
    boxes = parse_labels(lbl_path, w_img, h_img)
    if not boxes: return None

    # Step 1: Isolate wires as continuous black pixel blobs
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Mask out the bounding boxes so wires break completely at gate boundaries
    for b in boxes:
        pad = 10
        cv2.rectangle(thresh, (max(0, b['x']-pad), max(0, b['y']-pad)), 
                       (min(w_img, b['x']+b['w']+pad), min(h_img, b['y']+b['h']+pad)), 0, -1)

    # Step 2: Use Connected Components to assign an ID to each individual wire
    num_labels, labels_im = cv2.connectedComponents(thresh)
    
    wire_touches_input_of = defaultdict(list)
    wire_touches_output_of = defaultdict(list)
    
    # Step 3: Proximity Analysis (Which wire touches which box edges?)
    for b in boxes:
        margin = 15
        # Look around the left edge for incoming wires
        in_region = labels_im[max(0, b['y']-10):min(h_img, b['y']+b['h']+10), max(0, b['x']-margin):max(0, b['x']+5)]
        # Look around the right edge for outgoing wires
        out_region = labels_im[max(0, b['y']-10):min(h_img, b['y']+b['h']+10), min(w_img, b['x']+b['w']-5):min(w_img, b['x']+b['w']+margin)]
        
        for w_id in np.unique(in_region):
            if w_id > 0: wire_touches_input_of[w_id].append(b['id'])
        for w_id in np.unique(out_region):
            if w_id > 0: wire_touches_output_of[w_id].append(b['id'])

    # Step 4: Build Logical Dependency Graph
    graph = {b['id']: {"cls": b['cls'], "inputs": [], "outputs": []} for b in boxes}
    
    global_inputs = set()
    global_outputs = set()
    
    input_counter = 65 # 'A'
    out_counter = 1
    
    for w_id in range(1, num_labels):
        sources = wire_touches_output_of.get(w_id, [])
        targets = wire_touches_input_of.get(w_id, [])
        
        # Wire enters a gate but didn't emerge from any gate -> It's a global initial input
        if not sources and targets:
            name = chr(input_counter)
            input_counter = input_counter + 1 if input_counter < 90 else 65
            global_inputs.add(name)
            for t in targets:
                if name not in graph[t]["inputs"]:
                    graph[t]["inputs"].append(name)
                
        # Wire emerges from a gate but never enters another gate -> It's a global final output
        elif sources and not targets:
            for s in sources:
                name = f"OUT_{out_counter}"
                out_counter += 1
                graph[s]["outputs"].append(name)
                global_outputs.add(name)
                
        # Wire emerges from a gate AND enters another gate -> Intermediate trace
        elif sources and targets:
            for s in sources:
                for t in targets:
                    if s not in graph[t]["inputs"]:
                        graph[t]["inputs"].append(s)

    # Step 5: Assign G1, G2 names sequentially and build Equations
    new_names = {b['id']: f"G{i+1}" for i, b in enumerate(boxes)}
    netlist_lines = []
    final_eqs = {}
    
    for i, b in enumerate(boxes):
        old_id = b['id']
        node = graph[old_id]
        gate_type = node["cls"]
        new_id = new_names[old_id]
        
        # 5a. Netlist Generation
        resolved_inputs = [new_names.get(inp, inp) for inp in node["inputs"]]
        if not resolved_inputs: resolved_inputs = ["UNKNOWN"]
            
        args = ", ".join(resolved_inputs)
        netlist_lines.append(f"{new_id} = {gate_type}({args})")
        
        # 5b. Equation Generation (Recursion)
        eq_args = [final_eqs.get(inp, inp) for inp in node["inputs"]]
            
        if len(eq_args) == 1:
            eq_str = f"(NOT {eq_args[0]})" if gate_type in ["NOT", "BUFFER"] else f"({gate_type}({eq_args[0]}))"
        elif len(eq_args) > 1:
            op = f" {gate_type} "
            eq_str = f"({op.join(eq_args)})"
        else:
            eq_str = "UNKNOWN"
            
        final_eqs[old_id] = eq_str
        
        # If this gate is explicitly traced off-screen to a global output
        for out_name in node["outputs"]:
            netlist_lines.append(f"{out_name} = {new_id}")
            
    netlist_str = "; ".join(netlist_lines)
    
    # Check if we naturally found outputs
    if global_outputs:
        out_str = []
        for node_id, node in graph.items():
            if node["outputs"]:
                for out_name in node["outputs"]:
                    out_str.append(f"{out_name} = {final_eqs[node_id]}")
        eq_final = " AND ".join(out_str)
    else:
        # Fallback: Assume the very last right-most gate is the output 'Q'
        last_gate = boxes[-1]['id']
        eq_final = f"Q = {final_eqs[last_gate]}"
        netlist_str += f"; Q = {new_names[last_gate]}"
        
    print(f'{filename},"{eq_final}","{netlist_str}"')

if __name__ == "__main__":
    imgs = glob.glob(os.path.join(IMAGES_DIR, "*.png")) + glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    print("Processing sample of annotated images...")
    for i in imgs[:8]: 
        extract_netlist(i)
