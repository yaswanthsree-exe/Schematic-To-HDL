import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm
import random
import sympy

def generate_nice_random_circuit(d, min_inputs=3, max_inputs=4, min_gates=4, max_gates=7):
    """Draws a random combinational logic DAG with strict, clean routing logic."""
    d.config(fontsize=12, lw=random.uniform(1.0, 3.0))
    
    num_inputs = random.randint(min_inputs, max_inputs)
    num_gates = random.randint(min_gates, max_gates)
    
    gate_types = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
    
    nodes = []
    
    # Layering for layout
    node_layers = {}
    
    input_names = [chr(65+i) for i in range(num_inputs)] # A, B, C...
    
    for i in range(num_inputs):
        nodes.append({"type": "INPUT", "name": input_names[i], "expr": sympy.Symbol(input_names[i]), "layer": 0})
        node_layers[0] = node_layers.get(0, []) + [len(nodes)-1]
        
    for i in range(num_gates):
        g_type = random.choice(gate_types)
        
        available_nodes = list(range(len(nodes)))
        in1_idx = random.choice(available_nodes)
        in2_idx = random.choice(available_nodes)
        while in1_idx == in2_idx and len(available_nodes) > 1:
            in2_idx = random.choice(available_nodes)
            
        in1_node = nodes[in1_idx]
        in2_node = nodes[in2_idx]
        
        layer = max(in1_node["layer"], in2_node["layer"]) + 1
        
        # Calculate sympy expression
        if g_type == 'AND': expr = in1_node["expr"] & in2_node["expr"]
        elif g_type == 'OR': expr = in1_node["expr"] | in2_node["expr"]
        elif g_type == 'XOR': expr = in1_node["expr"] ^ in2_node["expr"]
        elif g_type == 'NAND': expr = ~(in1_node["expr"] & in2_node["expr"])
        elif g_type == 'NOR': expr = ~(in1_node["expr"] | in2_node["expr"])
        else: expr = in1_node["expr"]
        
        nodes.append({
            "type": "GATE", 
            "g_type": g_type, 
            "in1": in1_idx, 
            "in2": in2_idx, 
            "name": f"G{i+1}", 
            "expr": expr,
            "layer": layer
        })
        node_layers[layer] = node_layers.get(layer, []) + [len(nodes)-1]

    # --- DRAWING LAYOUT ---
    dx_layer = 4.0 # Base Horizontal spacing between layers
    dy_gate = 2.0  # Base Vertical spacing between gates
    
    drawn_elements = {}
    
    # 1. Place all nodes first so we know their strict coordinates
    max_layer = max(node_layers.keys())
    
    # Calculate Y positions to center each layer
    for layer, indices in node_layers.items():
        y_center_offset = (len(indices) - 1) * dy_gate / 2
        for i, n_idx in enumerate(indices):
            nodes[n_idx]['x'] = layer * dx_layer + random.uniform(-0.5, 0.5)
            nodes[n_idx]['y'] = y_center_offset - i * dy_gate + random.uniform(-0.2, 0.2)
            
    # 2. Draw Inputs
    for n_idx in node_layers[0]:
        node = nodes[n_idx]
        dot = d.add(logic.Dot().at((node['x'], node['y'])))
        d.add(elm.Label().at((node['x'], node['y'])).label(node["name"], "left"))
        drawn_elements[n_idx] = {"out_pos": (node['x'], node['y'])}
        
    # 3. Draw Gates
    for layer in range(1, max_layer + 1):
        if layer not in node_layers: continue
        for n_idx in node_layers[layer]:
            node = nodes[n_idx]
            x, y = node['x'], node['y']
            
            if node["g_type"] == 'AND': gate = logic.And()
            elif node["g_type"] == 'OR': gate = logic.Or()
            elif node["g_type"] == 'XOR': gate = logic.Xor()
            elif node["g_type"] == 'NAND': gate = logic.Nand()
            elif node["g_type"] == 'NOR': gate = logic.Nor()
            
            g_elm = d.add(gate.right().at((x, y)).label(node["name"], "bottom"))
            drawn_elements[n_idx] = {"out_pos": g_elm.out, "in1_pos": g_elm.in1, "in2_pos": g_elm.in2}
            
    # 4. Route Wires carefully to prevent intersections across gates
    # We use vertical drop channels before each layer.
    
    for layer in range(1, max_layer + 1):
        if layer not in node_layers: continue
        
        # Assign a unique X coordinate for the vertical routing trunk for each input connection to this layer
        # so lines don't stack on top of each other perfectly.
        connections = []
        for n_idx in node_layers[layer]:
            node = nodes[n_idx]
            connections.append((node['in1'], n_idx, 'in1'))
            connections.append((node['in2'], n_idx, 'in2'))
            
        # Give each connection in this layer a slightly different X trunk coordinate
        base_trunk_x = layer * dx_layer - 1.5
        
        for i, (src_idx, dst_idx, pin) in enumerate(connections):
            src_pos = drawn_elements[src_idx]["out_pos"]
            # To fix the overlapping dots bug, only draw a dot if it's not the very end of an output pin
            if nodes[src_idx]['type'] == 'INPUT' or ('routed' in drawn_elements[src_idx] and drawn_elements[src_idx]['routed'] > 0):
                d.add(logic.Dot().at(src_pos))
                
            drawn_elements[src_idx]['routed'] = drawn_elements[src_idx].get('routed', 0) + 1
            
            dst_pos = drawn_elements[dst_idx][f"{pin}_pos"]
            
            trunk_x = base_trunk_x - (i * 0.15) # Offset each trunk line slightly to the left
            
            # Route: source -> Go Right to trunk_x -> Go Up/Down to dst_pos Y -> Go Right to dst_pos X
            # We must break it down into explicit rigid lines so schemdraw's auto-router doesn't cut corners
            
            # Segment 1: Right to trunk
            d.add(logic.Line().right().at(src_pos).tox(trunk_x))
            # Segment 2: Vertical to target Y
            d.add(logic.Line().up().toy(dst_pos[1]))
            # Segment 3: Right to target in-pin
            d.add(logic.Line().right().to(dst_pos))

    # Add output lines
    used_as_input = set()
    for n in nodes:
        if n["type"] == "GATE":
            used_as_input.add(n["in1"])
            used_as_input.add(n["in2"])
            
    final_outputs = {}
    for i, n in enumerate(nodes):
        if i not in used_as_input and n["type"] == "GATE":
            output_name = f"OUT_{n['name']}"
            d.add(logic.Line().right().at(drawn_elements[i]["out_pos"]).length(1.5).label(output_name, "right"))
            final_outputs[output_name] = str(n["expr"])
            
    return d

if __name__ == "__main__":
    for i in range(5):
        with schemdraw.Drawing(show=False) as d:
            generate_nice_random_circuit(d)
            d.save(f"test_nice_dag_{i}.png", transparent=False, dpi=100)
            print(f"Generated test_nice_dag_{i}.png")
