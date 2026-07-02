import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm
import random
import sympy

def generate_random_dag(d, min_inputs=2, max_inputs=4, min_gates=3, max_gates=8):
    d.config(fontsize=12, lw=random.uniform(1.0, 3.0))
    
    num_inputs = random.randint(min_inputs, max_inputs)
    num_gates = random.randint(min_gates, max_gates)
    
    # Define gates available
    gate_types = ['AND', 'OR', 'XOR', 'NAND', 'NOR']
    
    # Store nodes: inputs are nodes 0 to num_inputs-1
    # gates are nodes num_inputs to num_inputs+num_gates-1
    nodes = []
    
    # Layering for layout
    node_layers = {}
    
    input_names = [chr(65+i) for i in range(num_inputs)] # A, B, C...
    
    for i in range(num_inputs):
        nodes.append({"type": "INPUT", "name": input_names[i], "expr": sympy.Symbol(input_names[i]), "layer": 0})
        node_layers[0] = node_layers.get(0, []) + [i]
        
    for i in range(num_gates):
        g_type = random.choice(gate_types)
        
        # Pick 2 distinct inputs from previously created nodes (guarantees DAG)
        available_nodes = list(range(len(nodes)))
        in1_idx = random.choice(available_nodes)
        
        # Avoid same input for both if possible, but allow it for variety
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
        
        node_idx = len(nodes)
        nodes.append({
            "type": "GATE", 
            "g_type": g_type, 
            "in1": in1_idx, 
            "in2": in2_idx, 
            "name": f"G{i+1}", 
            "expr": expr,
            "layer": layer
        })
        node_layers[layer] = node_layers.get(layer, []) + [node_idx]

    # --- DRAWING LAYOUT ---
    dx = 4.0 # Horizontal spacing between layers
    dy = 2.0 # Vertical spacing between gates
    
    drawn_elements = {}
    
    # Draw inputs at x=0
    for idx, n_idx in enumerate(node_layers[0]):
        node = nodes[n_idx]
        y_pos = -idx * dy
        # Just create a dot and label for the input
        dot = d.add(logic.Dot().at((0, y_pos)))
        d.add(elm.Label().at((0, y_pos)).label(node["name"], "left"))
        drawn_elements[n_idx] = {"out_pos": (0, y_pos)}
        
    # Draw gates layer by layer
    max_layer = max(node_layers.keys())
    
    netlist = []
    for layer in range(1, max_layer + 1):
        if layer not in node_layers: continue
        
        layer_nodes = node_layers[layer]
        
        # To avoid vertical clutter, offset y positions
        # Center the layer somewhat
        y_offset_base = - (len(layer_nodes) - 1) * dy / 2
        
        for idx, n_idx in enumerate(layer_nodes):
            node = nodes[n_idx]
            
            x_pos = layer * dx
            # Add some randomness to position for variation
            y_pos = y_offset_base - idx * dy + random.uniform(-0.5, 0.5)
            x_pos += random.uniform(-0.5, 0.5)
            
            if node["g_type"] == 'AND': gate = logic.And()
            elif node["g_type"] == 'OR': gate = logic.Or()
            elif node["g_type"] == 'XOR': gate = logic.Xor()
            elif node["g_type"] == 'NAND': gate = logic.Nand()
            elif node["g_type"] == 'NOR': gate = logic.Nor()
            
            g_elm = d.add(gate.right().at((x_pos, y_pos)).label(node["name"], "bottom"))
            
            # Route inputs
            src1_pos = drawn_elements[node["in1"]]["out_pos"]
            src2_pos = drawn_elements[node["in2"]]["out_pos"]
            
            # Using |- (orthogonal horizontal-first) for clean routing
            # But we want to avoid all wires bunching up on the same vertical line.
            # Add a dot at the source if it's branching? schemdraw handles dots later if we want.
            
            # Draw wire 1
            d.add(logic.Wire('|-').at(src1_pos).to(g_elm.in1))
            # Draw wire 2
            d.add(logic.Wire('|-').at(src2_pos).to(g_elm.in2))
            
            # Draw dots at source if it's already used? (Simplification: just draw dots at all sources that are used >1 time)
            
            drawn_elements[n_idx] = {"out_pos": g_elm.out}
            
            in1_name = nodes[node["in1"]]["name"]
            in2_name = nodes[node["in2"]]["name"]
            out_name = node["name"]
            netlist.append(f"{node['g_type'].lower()} {node['name']} ({out_name}, {in1_name}, {in2_name});")

    # Add output lines to the very last layer gates (or any gate not used as input)
    used_as_input = set()
    for n in nodes:
        if n["type"] == "GATE":
            used_as_input.add(n["in1"])
            used_as_input.add(n["in2"])
            
    final_outputs = {}
    for i, n in enumerate(nodes):
        if i not in used_as_input and n["type"] == "GATE":
            d.add(logic.Line().right().at(drawn_elements[i]["out_pos"]).length(1).label(f"OUT_{n['name']}", "right"))
            final_outputs[f"OUT_{n['name']}"] = str(n["expr"])
            
    metadata = {
        "circuit_type": "Random_DAG",
        "netlist": "\n".join(netlist),
        "equations": final_outputs
    }
    return d, metadata, "Random_DAG"

if __name__ == "__main__":
    for i in range(3):
        with schemdraw.Drawing(show=False) as d:
            d, meta, name = generate_random_dag(d)
            d.save(f"test_random_dag_{i}.png", transparent=False, dpi=100)
            print(f"Generated {i}: {meta['equations']}")
