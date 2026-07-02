import os
import random
import yaml
import logging
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURATION ---
DATASET_DIR = "custom_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

IMG_WIDTH = 1200
IMG_HEIGHT = 800

GATE_TYPES = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'NOT']
CLASS_MAP = {gt: i for i, gt in enumerate(GATE_TYPES)}

REF_DIR = "reference_library"


def setup_directories():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    yaml_config = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(GATE_TYPES),
        'names': GATE_TYPES
    }
    with open(os.path.join(DATASET_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

def load_stamps():
    stamps = {}
    for gt in GATE_TYPES:
        path = os.path.join(REF_DIR, f"{gt}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
            
        img = Image.open(path).convert("RGBA")
        
        # Crop exactly to visible pixels
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        target_h = 60
        w, h = img.size
        target_w = int((target_h / h) * w)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        stamps[gt] = img
    return stamps

class LogicNode:
    def __init__(self, n_id, g_type, layer):
        self.n_id = n_id
        self.g_type = g_type
        self.layer = layer
        self.inputs = []
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.in1_pos = (0, 0)
        self.in2_pos = (0, 0)
        self.out_pos = (0, 0)

def generate_logical_dag():
    num_inputs = random.randint(2, 4)
    num_gates = random.randint(4, 9)
    
    nodes = {}
    layers = {0: []}
    
    for i in range(num_inputs):
        n_id = f"IN_{i}"
        nodes[n_id] = LogicNode(n_id, "INPUT", 0)
        layers[0].append(n_id)
        
    for i in range(num_gates):
        g_type = random.choice(GATE_TYPES)
        n_id = f"G_{i}"
        
        available_sources = list(nodes.keys())
        src1 = random.choice(available_sources)
        
        if g_type == 'NOT':
            srcs = [src1]
            layer = nodes[src1].layer + 1
        else:
            src2 = random.choice(available_sources)
            if src1 == src2 and len(available_sources) > 1:
                while src2 == src1:
                    src2 = random.choice(available_sources)
            srcs = [src1, src2]
            layer = max(nodes[src1].layer, nodes[src2].layer) + 1
            
        new_node = LogicNode(n_id, g_type, layer)
        new_node.inputs = srcs
        
        nodes[n_id] = new_node
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(n_id)
        
    return nodes, layers

def generate_schematic_image(img_id, stamps):
    canvas = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    
    nodes, layers = generate_logical_dag()
    labels = []
    
    COL_SPACING = random.randint(220, 300)
    ROW_SPACING = 140
    
    # 1. Placement
    for layer_id, n_ids in layers.items():
        base_x = 80 + layer_id * COL_SPACING
        
        total_height = len(n_ids) * ROW_SPACING
        start_y = (IMG_HEIGHT - total_height) // 2
        
        for i, n_id in enumerate(n_ids):
            node = nodes[n_id]
            node.x = base_x + random.randint(-10, 10)
            node.y = start_y + i * ROW_SPACING + random.randint(-15, 15)
            
            # Constrain to screen
            node.x = max(20, min(IMG_WIDTH - 150, node.x))
            node.y = max(20, min(IMG_HEIGHT - 100, node.y))
            
            if node.g_type != "INPUT":
                stamp = stamps[node.g_type]
                node.w, node.h = stamp.size
                canvas.paste(stamp, (node.x, node.y), stamp)
                
                # Perfect YOLO bbox
                xc = (node.x + node.w / 2) / IMG_WIDTH
                yc = (node.y + node.h / 2) / IMG_HEIGHT
                w_norm = node.w / IMG_WIDTH
                h_norm = node.h / IMG_HEIGHT
                class_id = CLASS_MAP[node.g_type]
                labels.append(f"{class_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Pins: Overlap into the box to hide the white space
                node.out_pos = (node.x + node.w, node.y + node.h // 2)
                
                if node.g_type == 'NOT':
                    node.in1_pos = (node.x + 5, node.y + node.h // 2)
                    node.in2_pos = (node.x + 5, node.y + node.h // 2)
                else: # For OR/XOR, bringing the pin slightly inward connects to the curve
                    node.in1_pos = (node.x + 8, int(node.y + node.h * 0.28))
                    node.in2_pos = (node.x + 8, int(node.y + node.h * 0.72))
            else:
                r = 6
                out_x = node.x + 20
                out_y = int(node.y + 30)
                draw.ellipse([out_x-r, out_y-r, out_x+r, out_y+r], fill="black")
                draw.text((out_x - 50, out_y - 5), n_id, fill="black")
                node.out_pos = (out_x, out_y)

    # 2. Wire Routing (Avoids slicing through nodes)
    # We assign a dedicated horizontal routing channel randomly above or below the node
    # to avoid the wire hitting completely unrelated gates on the same row.
    for n_id, node in nodes.items():
        if node.g_type == "INPUT": continue
        
        for idx, src_id in enumerate(node.inputs):
            src_node = nodes[src_id]
            src_pt = src_node.out_pos
            dst_pt = node.in1_pos if idx == 0 else node.in2_pos
            
            draw.ellipse([src_pt[0]-4, src_pt[1]-4, src_pt[0]+4, src_pt[1]+4], fill="black")
            
            # 5-Segment Manhattan Router
            # src -> go right -> go up/down to channel -> go right -> go up/down to dst -> dst
            
            x1 = src_pt[0] + random.randint(15, 30)
            x2 = dst_pt[0] - random.randint(15, 30)
            
            # If the source and dest are very close, skip the channel step
            if src_node.layer == node.layer - 1:
                # Direct route
                pts = [src_pt, (x1, src_pt[1]), (x1, dst_pt[1]), dst_pt]
            else:
                # Requires jumping layers. We use a dedicated routing channel Y to avoid gates.
                # Choose a Y coordinate that is either significantly above or below the source node
                direction = random.choice([-1, 1])
                route_y = src_pt[1] + direction * random.randint(50, 80)
                
                # Constrain route_y
                route_y = max(20, min(IMG_HEIGHT-20, route_y))
                
                pts = [
                    src_pt,
                    (x1, src_pt[1]),
                    (x1, route_y),
                    (x2, route_y),
                    (x2, dst_pt[1]),
                    dst_pt
                ]
            
            draw.line(pts, fill="black", width=2)

    img_path = os.path.join(IMAGES_DIR, f"stamp_schem_{img_id}.png")
    lbl_path = os.path.join(LABELS_DIR, f"stamp_schem_{img_id}.txt")
    
    canvas.save(img_path)
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    setup_directories()
    logging.info("Loading reference gate stamps...")
    stamps = load_stamps()
    
    num_samples = 5
    logging.info(f"Generating {num_samples} schematic canvases...")
    for i in range(num_samples):
        if i % 100 == 0:
            logging.info(f"Progress: {i}/{num_samples}")
        generate_schematic_image(i, stamps)
        
    logging.info("Dataset Generation Complete.")
