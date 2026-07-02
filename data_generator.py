import os
import json
import random
import uuid
import numpy as np
import cv2
import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels") # YOLO format annotations
METADATA_DIR = os.path.join(DATASET_DIR, "metadata") # Netlists and extra info

# Ensure directories exist
for d in [IMAGES_DIR, LABELS_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

# List of all supported gates for generation
import sympy

def generate_half_adder(d):
    """Draws a cleanly routed Half Adder with spatial variation."""
    d.config(fontsize=12, lw=random.uniform(1.2, 2.5))
    
    dx_in = random.uniform(2.0, 4.0)
    dy_and = random.uniform(3.0, 5.0)
    
    xor_g = d.add(logic.Xor().right().label("G1", "bottom"))
    
    in_a = d.add(logic.Line().left().at(xor_g.in1).length(dx_in).label('A', 'left'))
    dot_a = d.add(logic.Dot().at((in_a.end[0] + dx_in * 0.4, in_a.end[1])))
    
    in_b = d.add(logic.Line().left().at(xor_g.in2).length(dx_in).label('B', 'left'))
    dot_b = d.add(logic.Dot().at((in_b.end[0] + dx_in * 0.2, in_b.end[1])))
    
    d.add(logic.Line().right().at(xor_g.out).length(1.5).label('Sum', 'right'))
    
    and_g = d.add(logic.And().right().at((xor_g.in1[0], xor_g.in1[1] - dy_and)).label("G2", "bottom"))
    
    d.add(logic.Wire('|-').at(dot_a.center).to(and_g.in1))
    d.add(logic.Wire('|-').at(dot_b.center).to(and_g.in2))
    
    d.add(logic.Line().right().at(and_g.out).length(1.5).label('Carry', 'right'))
    
    netlist = ["xor G1 (Sum, A, B);", "and G2 (Carry, A, B);"]
    return d, {"circuit_type": "Half Adder", "netlist": "\n".join(netlist)}, "Half_Adder"


def generate_full_adder(d):
    """Draws a cleanly routed Full Adder with spatial variation."""
    d.config(fontsize=12, lw=random.uniform(1.2, 2.5))
    
    dx_in = random.uniform(2.5, 4.0)
    dy_and1 = random.uniform(3.0, 5.0)
    dx_xor2 = random.uniform(3.0, 4.5)
    
    # HA 1
    xor1 = d.add(logic.Xor().right().label("G1", "bottom"))
    in_a = d.add(logic.Line().left().at(xor1.in1).length(dx_in).label('A', 'left'))
    dot_a = d.add(logic.Dot().at((in_a.end[0] + dx_in*0.4, in_a.end[1])))
    in_b = d.add(logic.Line().left().at(xor1.in2).length(dx_in).label('B', 'left'))
    dot_b = d.add(logic.Dot().at((in_b.end[0] + dx_in*0.2, in_b.end[1])))
    
    and1 = d.add(logic.And().right().at((xor1.in1[0], xor1.in1[1] - dy_and1)).label("G2", "bottom"))
    d.add(logic.Wire('|-').at(dot_a.center).to(and1.in1))
    d.add(logic.Wire('|-').at(dot_b.center).to(and1.in2))
    
    # HA 2
    xor2 = d.add(logic.Xor().right().at((xor1.out[0] + dx_xor2, xor1.out[1])).label("G3", "bottom"))
    d.add(logic.Wire('|-').at(xor1.out).to(xor2.in1))
    
    cin_x = in_a.end[0]
    cin_y = and1.in1[1] - random.uniform(2.0, 3.5)
    in_cin = d.add(logic.Line().right().at((cin_x, cin_y)).length(xor2.in2[0] - cin_x - 1.0).label('Cin', 'left'))
    dot_cin = d.add(logic.Dot().at(in_cin.end))
    d.add(logic.Wire('|-').at(dot_cin.center).to(xor2.in2))
    
    and2_y = xor2.in1[1] - random.uniform(3.0, 5.0)
    and2 = d.add(logic.And().right().at((xor2.in1[0], and2_y)).label("G4", "bottom"))
    
    dot_ha1_out = d.add(logic.Dot().at((xor1.out[0] + dx_xor2/2, xor1.out[1])))
    d.add(logic.Wire('|-').at(dot_ha1_out.center).to(and2.in1))
    d.add(logic.Wire('|-').at(dot_cin.center).to(and2.in2))
    
    # OR gate
    or_y = (and1.out[1] + and2.out[1]) / 2
    or1 = d.add(logic.Or().right().at((and2.out[0] + random.uniform(2.0, 3.5), or_y)).label("G5", "bottom"))
    d.add(logic.Wire('|-').at(and1.out).to(or1.in1))
    d.add(logic.Wire('|-').at(and2.out).to(or1.in2))
    
    d.add(logic.Line().right().at(xor2.out).length(1.5).label('Sum', 'right'))
    d.add(logic.Line().right().at(or1.out).length(1.5).label('Cout', 'right'))
    
    netlist = [
        "xor G1 (n1, A, B);", "and G2 (n2, A, B);", "xor G3 (Sum, n1, Cin);",
        "and G4 (n3, n1, Cin);", "or G5 (Cout, n2, n3);"
    ]
    return d, {"circuit_type": "Full Adder", "netlist": "\n".join(netlist)}, "Full_Adder"


def generate_2x1_mux(d):
    """Draws a cleanly routed 2x1 Mux."""
    d.config(fontsize=12, lw=random.uniform(1.2, 2.5))
    
    dy_and2 = random.uniform(4.0, 6.0)
    dx_s = random.uniform(2.5, 4.0)
    
    and1 = d.add(logic.And().right().label("G1", "bottom"))
    d.add(logic.Line().left().at(and1.in1).length(3).label('D0', 'left'))
    
    and2 = d.add(logic.And().right().at((and1.in1[0], and1.in1[1] - dy_and2)).label("G2", "bottom"))
    d.add(logic.Line().left().at(and2.in2).length(3).label('D1', 'left'))
    
    s_x = and1.in2[0] - dx_s
    s_y = and2.in1[1] - 1.5
    sel_dot = d.add(logic.Dot().at((s_x, s_y)))
    d.add(logic.Line().down().at(sel_dot.center).length(1).label('S', 'bottom'))
    
    d.add(logic.Line().right().at((s_x, and2.in1[1])).to(and2.in1))
    d.add(logic.Dot().at((s_x, and2.in1[1])))
    
    d1_y = and2.in2[1]
    not_y = d1_y + random.uniform(0.5, dy_and2 - 2.5)
    not1 = d.add(logic.Not().up().at((s_x, not_y)))
    d.add(elm.Label().at((not1.center[0] - 0.4, not1.center[1])).label("G3"))
    
    # Bridge over D1
    l1 = d.add(logic.Line().up().at(sel_dot.center).length(abs(d1_y - s_y) - 0.2))
    d.add(elm.Arc2(k=0.8).at(l1.end).to((s_x, d1_y + 0.2)))
    d.add(logic.Line().up().to(not1.in1))
    
    d.add(logic.Line().up().at(not1.out).toy(and1.in2))
    d.add(logic.Line().right().to(and1.in2))
    
    or_y = (and1.out[1] + and2.out[1]) / 2 
    or1 = d.add(logic.Or().right().at((and1.out[0] + random.uniform(2.5, 4.0), or_y)).label("G4", "bottom"))
    
    d.add(logic.Wire('|-').at(and1.out).to(or1.in1))
    d.add(logic.Wire('|-').at(and2.out).to(or1.in2))
    d.add(logic.Line().right().at(or1.out).length(1.5).label('Y', 'right'))
    
    netlist = ["not G3 (n_s, S);", "and G1 (n1, D0, n_s);", "and G2 (n2, D1, S);", "or G4 (Y, n1, n2);"]
    return d, {"circuit_type": "2x1 Mux", "netlist": "\n".join(netlist)}, "Mux_2x1"


def generate_2to4_decoder(d):
    """Draws a strictly routed 2-to-4 Decoder."""
    d.config(fontsize=12, lw=random.uniform(1.2, 2.5))
    
    # Inputs
    in_a = d.add(logic.Line().right().length(1.5).at((0, 0)).label('A', 'left'))
    dot_a = d.add(logic.Dot())
    not_a = d.add(logic.Not().right().at(dot_a.center))
    
    in_b = d.add(logic.Line().right().length(1.5).at((0, -4.0)).label('B', 'left'))
    dot_b = d.add(logic.Dot())
    not_b = d.add(logic.Not().right().at(dot_b.center))
    
    col_a = dot_a.center[0]
    col_na = not_a.out[0] + 0.5
    col_b = dot_b.center[0]
    col_nb = not_b.out[0] + 0.5
    
    d.add(logic.Line().right().at(not_a.out).tox(col_na))
    dot_na_top = d.add(logic.Dot())
    
    d.add(logic.Line().right().at(not_b.out).tox(col_nb))
    dot_nb_top = d.add(logic.Dot())
    
    x_and = col_nb + random.uniform(2.5, 4.0)
    y_start = 2.0
    dy_out = random.uniform(2.0, 3.0)
    
    gates = []
    out_labels = ['Y0', 'Y1', 'Y2', 'Y3']
    
    for i in range(4):
        y = y_start - i * dy_out
        g = d.add(logic.And().right().at((x_and, y)).label(f"G{i+3}", "bottom"))
        gates.append(g)
        d.add(logic.Line().right().at(g.out).length(1.5).label(out_labels[i], 'right'))
        
    # Y0 = ~A & ~B
    d.add(logic.Line().down().at(dot_na_top.center).toy(gates[0].in1[1]))
    dot_y0a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[0].in1))
    
    d.add(logic.Line().up().at(dot_nb_top.center).toy(gates[0].in2[1]))
    dot_y0b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[0].in2))
    
    # Y1 = ~A & B
    d.add(logic.Line().down().at(dot_y0a.center).toy(gates[1].in1[1]))
    dot_y1a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[1].in1))
    
    d.add(logic.Line().up().at(dot_b.center).toy(gates[1].in2[1]))
    dot_y1b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[1].in2))
    
    # Y2 = A & ~B
    d.add(logic.Line().down().at(dot_a.center).toy(gates[2].in1[1]))
    dot_y2a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[2].in1))
    
    d.add(logic.Line().down().at(dot_nb_top.center).toy(gates[2].in2[1]))
    dot_y2b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[2].in2))
    
    # Y3 = A & B
    d.add(logic.Line().down().at(dot_y2a.center).toy(gates[3].in1[1]))
    d.add(logic.Line().right().to(gates[3].in1))
    
    d.add(logic.Line().down().at(dot_b.center).toy(gates[3].in2[1]))
    d.add(logic.Line().right().to(gates[3].in2))
    
    # Extend bottom trunks slightly
    d.add(logic.Line().down().at(gates[3].in1).at((col_na, gates[3].in1[1])).length(1.0))
    d.add(logic.Line().down().at(gates[3].in2).at((col_nb, gates[3].in2[1])).length(1.0))
    
    netlist = [
        "not G1 (nA, A);", "not G2 (nB, B);",
        "and G3 (Y0, nA, nB);", "and G4 (Y1, nA, B);",
        "and G5 (Y2, A, nB);", "and G6 (Y3, A, B);"
    ]
    return d, {"circuit_type": "2-to-4 Decoder", "netlist": "\n".join(netlist)}, "Decoder_2to4"

def generate_random_circuit():
    """Picks a clean template circuit to generate."""
    generators = [generate_half_adder, generate_2x1_mux, generate_full_adder, generate_2to4_decoder]
    generator = random.choice(generators)
    with schemdraw.Drawing(show=False) as d:
        return generator(d)


def apply_domain_randomization(image_path):
    """Applies noise, blur, and warping to make data robust."""
    # Ensure file exists before trying to read it
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        return
        
    # Random brightness/contrast
    alpha = random.uniform(0.7, 1.3) # Contrast
    beta = random.uniform(-30, 30)   # Brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    cv2.imwrite(image_path, img)

def generate_dataset(num_samples=10):
    print(f"Generating {num_samples} schematic samples...")
    for i in range(num_samples):
        file_id = str(uuid.uuid4())[:8]
        img_path = os.path.join(IMAGES_DIR, f"schem_{file_id}.png")
        
        # 2. Generate Circuit
        # We also receive the metadata containing netlist & equation
        d, metadata, gate_name = generate_random_circuit()
        
        # 3. Save Image (Need tight=True for YOLO Bbox Math)
        # `bbox` calculates the tightly cropped image bounds
        d.save(img_path, transparent=False, dpi=100)
        
        # 4. Save Metadata (Netlist & Equation)
        meta_path = os.path.join(METADATA_DIR, f"schem_{file_id}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # 5. Extract Bounding Boxes for YOLO
        find_bounding_boxes_via_cv(img_path, file_id)
        
        # 6. Domain Randomization 
        apply_domain_randomization(img_path)
        
        plt.close('all') # Prevent memory leak when generating thousands of images
        
        print(f"Generated {i+1}/{num_samples}: {gate_name}")

def find_bounding_boxes_via_cv(img_path, file_id):
    """Uses OpenCV to find the bounding boxes of the drawn shapes & generate YOLO labels."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    # Invert image (black on white -> white on black)
    img_inv = cv2.bitwise_not(img)
    
    # Threshold to make it strict binary
    _, thresh = cv2.threshold(img_inv, 50, 255, cv2.THRESH_BINARY)
    
    # Optional: Small morphological close to join broken lines slightly
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = img.shape
    
    label_path = os.path.join(LABELS_DIR, f"schem_{file_id}.txt")
    with open(label_path, "w") as f:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter out tiny noise contours and full-image contours
            if w < 10 or h < 10 or (w > w_img*0.9 and h > h_img*0.9):
                continue
                
            # Class 0 = "Generic Component" (Gate or Terminal)
            class_id = 0
            
            # YOLO Format: class x_center y_center width height (Normalized 0.0 to 1.0)
            x_center = (x + w/2) / w_img
            y_center = (y + h/2) / h_img
            norm_w = w / w_img
            norm_h = h / h_img
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")


if __name__ == "__main__":
    generate_dataset(2000)
    print("Generation complete.")
