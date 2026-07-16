"""
Schematic to HDL Generator
Converts visual circuit designs from the designer into structural Verilog/VHDL
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re


class ICPinDatabase:
    """Database of IC pin configurations with proper naming and pin mapping"""
    
    # Pin definitions for each IC type
    # Format: pin_number -> (port_name, port_type: 'input'|'output'|'power')
    IC_PIN_DEFINITIONS = {
        '7400': {  # Quad 2-input NAND gate (A,B,Y pattern)
            1: ('A1', 'input'), 2: ('B1', 'input'), 3: ('Y1', 'output'),
            4: ('A2', 'input'), 5: ('B2', 'input'), 6: ('Y2', 'output'),
            7: ('GND', 'power'),
            8: ('Y3', 'output'), 9: ('B3', 'input'), 10: ('A3', 'input'),
            11: ('Y4', 'output'), 12: ('B4', 'input'), 13: ('A4', 'input'),
            14: ('VCC', 'power'),
            'function': 'nand',
            'gate_groups': [(('A1', 'B1'), 'Y1'), (('A2', 'B2'), 'Y2'), 
                           (('A3', 'B3'), 'Y3'), (('A4', 'B4'), 'Y4')]
        },
        '7402': {  # Quad 2-input NOR gate (Y,A,B pattern - OUTPUT FIRST!)
            1: ('Y1', 'output'), 2: ('A1', 'input'), 3: ('B1', 'input'),
            4: ('Y2', 'output'), 5: ('A2', 'input'), 6: ('B2', 'input'),
            7: ('GND', 'power'),
            8: ('A3', 'input'), 9: ('B3', 'input'), 10: ('Y3', 'output'),
            11: ('A4', 'input'), 12: ('B4', 'input'), 13: ('Y4', 'output'),
            14: ('VCC', 'power'),
            'function': 'nor',
            'gate_groups': [(('A1', 'B1'), 'Y1'), (('A2', 'B2'), 'Y2'), 
                           (('A3', 'B3'), 'Y3'), (('A4', 'B4'), 'Y4')]
        },
        '7404': {  # Hex Inverter
            1: ('A1', 'input'), 2: ('Y1', 'output'),
            3: ('A2', 'input'), 4: ('Y2', 'output'),
            5: ('A3', 'input'), 6: ('Y3', 'output'),
            7: ('GND', 'power'),
            8: ('Y4', 'output'), 9: ('A4', 'input'),
            10: ('Y5', 'output'), 11: ('A5', 'input'),
            12: ('Y6', 'output'), 13: ('A6', 'input'),
            14: ('VCC', 'power'),
            'function': 'not',
            'gate_groups': [(('A1',), 'Y1'), (('A2',), 'Y2'), (('A3',), 'Y3'),
                           (('A4',), 'Y4'), (('A5',), 'Y5'), (('A6',), 'Y6')]
        },
        '7408': {  # Quad 2-input AND gate
            1: ('A1', 'input'), 2: ('B1', 'input'), 3: ('Y1', 'output'),
            4: ('A2', 'input'), 5: ('B2', 'input'), 6: ('Y2', 'output'),
            7: ('GND', 'power'),
            8: ('Y3', 'output'), 9: ('B3', 'input'), 10: ('A3', 'input'),
            11: ('Y4', 'output'), 12: ('B4', 'input'), 13: ('A4', 'input'),
            14: ('VCC', 'power'),
            'function': 'and',
            'gate_groups': [(('A1', 'B1'), 'Y1'), (('A2', 'B2'), 'Y2'), 
                           (('A3', 'B3'), 'Y3'), (('A4', 'B4'), 'Y4')]
        },
        '7432': {  # Quad 2-input OR gate
            1: ('A1', 'input'), 2: ('B1', 'input'), 3: ('Y1', 'output'),
            4: ('A2', 'input'), 5: ('B2', 'input'), 6: ('Y2', 'output'),
            7: ('GND', 'power'),
            8: ('Y3', 'output'), 9: ('B3', 'input'), 10: ('A3', 'input'),
            11: ('Y4', 'output'), 12: ('B4', 'input'), 13: ('A4', 'input'),
            14: ('VCC', 'power'),
            'function': 'or',
            'gate_groups': [(('A1', 'B1'), 'Y1'), (('A2', 'B2'), 'Y2'), 
                           (('A3', 'B3'), 'Y3'), (('A4', 'B4'), 'Y4')]
        },
        '7486': {  # Quad 2-input XOR gate
            1: ('A1', 'input'), 2: ('B1', 'input'), 3: ('Y1', 'output'),
            4: ('A2', 'input'), 5: ('B2', 'input'), 6: ('Y2', 'output'),
            7: ('GND', 'power'),
            8: ('Y3', 'output'), 9: ('B3', 'input'), 10: ('A3', 'input'),
            11: ('Y4', 'output'), 12: ('B4', 'input'), 13: ('A4', 'input'),
            14: ('VCC', 'power'),
            'function': 'xor',
            'gate_groups': [(('A1', 'B1'), 'Y1'), (('A2', 'B2'), 'Y2'), 
                           (('A3', 'B3'), 'Y3'), (('A4', 'B4'), 'Y4')]
        },
    }
    
    @classmethod
    def get_pin_info(cls, ic_type: str, pin_number: int) -> Optional[Tuple[str, str]]:
        """Get pin name and type for a given IC and pin number"""
        ic_def = cls.IC_PIN_DEFINITIONS.get(ic_type)
        if ic_def and pin_number in ic_def:
            return ic_def[pin_number]
        return None
    
    @classmethod
    def get_pin_name_from_number(cls, ic_type: str, pin_number: int) -> str:
        """Convert pin number to valid Verilog port name"""
        info = cls.get_pin_info(ic_type, pin_number)
        if info:
            return info[0]
        # Fallback: create a valid name like P1, P2, etc.
        return f"P{pin_number}"
    
    @classmethod
    def get_ic_ports(cls, ic_type: str) -> Dict[str, List[str]]:
        """Get all input, output, and power ports for an IC"""
        ic_def = cls.IC_PIN_DEFINITIONS.get(ic_type, {})
        ports = {'inputs': [], 'outputs': [], 'power': []}
        
        for pin_num, pin_info in ic_def.items():
            if isinstance(pin_num, int):  # Skip metadata keys like 'function'
                name, ptype = pin_info
                if ptype == 'input':
                    ports['inputs'].append(name)
                elif ptype == 'output':
                    ports['outputs'].append(name)
                elif ptype == 'power':
                    ports['power'].append(name)
        
        return ports
    
    @classmethod
    def get_function(cls, ic_type: str) -> str:
        """Get the logic function type for an IC"""
        ic_def = cls.IC_PIN_DEFINITIONS.get(ic_type, {})
        return ic_def.get('function', 'unknown')


class SchematicHDLGenerator:
    """Generates HDL from schematic circuit data"""
    
    def __init__(self, circuit_data: Dict[str, Any]):
        self.circuit_data = circuit_data
        self.components = []  # ICs
        self.io_components = []  # Switches, LEDs
        self.wires = []
        self.inputs = []
        self.outputs = []
        self.net_counter = 0
        self.net_connections = {}  # Maps (component_id, pin_name) to net names
        self.position_to_port = {}  # Maps position keys to port info
        self.parse_circuit()
    
    def _make_valid_identifier(self, name: str) -> str:
        """Convert a name to a valid Verilog/VHDL identifier"""
        if not name:
            return "unnamed"
        # If starts with a digit, prefix with 'IC_'
        if name[0].isdigit():
            name = f"IC_{name}"
        # Replace any non-alphanumeric characters with underscore
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return name
    
    def _make_valid_port_name(self, name: str) -> str:
        """Convert a pin name to a valid Verilog port name"""
        if not name:
            return "port"
        # Handle names like "1A", "1B", "1Y" -> "A1", "B1", "Y1"
        if name and name[0].isdigit():
            # Find where letters start
            for i, c in enumerate(name):
                if c.isalpha():
                    return f"{name[i:]}{name[:i]}"
            # All digits - prefix with P
            return f"P{name}"
        # Replace special characters
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return name
    
    def _pos_key(self, left: float, top: float, tolerance: float = 20) -> str:
        """Create a position key for matching ports (with tolerance)"""
        return f"{round(left/tolerance)*tolerance}_{round(top/tolerance)*tolerance}"
    
    def _generate_net_name(self) -> str:
        """Generate a unique net name"""
        name = f"net_{self.net_counter}"
        self.net_counter += 1
        return name
    
    def parse_circuit(self):
        """Parse the circuit JSON data"""
        objects = self.circuit_data.get('objects', {}).get('objects', [])
        
        component_id = 0
        
        for obj in objects:
            obj_left = obj.get('left', 0)
            obj_top = obj.get('top', 0)
            
            # Check if it's an I/O component (Switch or LED)
            if obj.get('isIO'):
                io_type = obj.get('ioType')
                # Prefer the Fabric object's id so portId matching in _get_port_info_from_wire works
                fabric_id = obj.get('id')
                io_id = fabric_id if fabric_id else f"io_{component_id}"
                component_id += 1

                # Use user-set label if it's not the default placeholder
                user_label = obj.get('label', '')
                default_labels = {'IN', 'OUT', ''}
                if user_label and user_label not in default_labels:
                    safe = re.sub(r'[^a-zA-Z0-9_]', '_', user_label)
                    io_name = safe if safe else f"{'sw' if io_type == 'input' else 'led'}_{len([x for x in self.io_components if x['type'] == io_type])}"
                else:
                    io_name = f"{'sw' if io_type == 'input' else 'led'}_{len([x for x in self.io_components if x['type'] == io_type])}"

                io_info = {
                    'id': io_id,
                    'type': io_type,
                    'position': (obj_left, obj_top),
                    'name': io_name
                }
                self.io_components.append(io_info)
                
                # Register position for wire matching
                pos_key = self._pos_key(obj_left, obj_top)
                self.position_to_port[pos_key] = {
                    'component_id': io_id,
                    'port_name': io_name,
                    'is_io': True,
                    'io_type': io_type
                }
                
                # Add to inputs/outputs list
                if io_type == 'input':
                    self.inputs.append(io_name)
                else:
                    self.outputs.append(io_name)
            
            # Check if it's an IC group
            elif obj.get('type') == 'group':
                ic_objects = obj.get('objects', [])
                ic_name = None
                
                for sub_obj in ic_objects:
                    if sub_obj.get('type') == 'text':
                        text = sub_obj.get('text', '')
                        if text and (text[0].isdigit() or text.startswith('74') or text.startswith('IC')):
                            ic_name = text
                            break
                
                if ic_name:
                    instance_name = f"U{len(self.components)}_{self._make_valid_identifier(ic_name)}"
                    ic_info = {
                        'id': f"ic_{component_id}",
                        'type': ic_name,
                        'valid_type': self._make_valid_identifier(ic_name),
                        'instance_name': instance_name,
                        'position': (obj_left, obj_top),
                        'pin_connections': {}  # pin_name -> net_name
                    }
                    component_id += 1
                    self.components.append(ic_info)
        
        # Parse wires and build netlist
        self._parse_wires()
    
    def _parse_wires(self):
        """Parse wire connections and build the netlist"""
        raw_wires = self.circuit_data.get('wires', [])
        
        for wire in raw_wires:
            start = wire.get('start', {})
            end = wire.get('end', {})
            
            start_key = self._pos_key(start.get('left', 0), start.get('top', 0))
            end_key = self._pos_key(end.get('left', 0), end.get('top', 0))
            
            # Get port information from wire data
            start_info = self._get_port_info_from_wire(start, start_key)
            end_info = self._get_port_info_from_wire(end, end_key)
            
            # Determine net name
            # If either end is an I/O, use that name as the net
            # Otherwise, generate a new net name
            if start_info.get('is_io') and start_info.get('io_type') == 'input':
                net_name = start_info['port_name']
            elif end_info.get('is_io') and end_info.get('io_type') == 'input':
                net_name = end_info['port_name']
            elif start_info.get('is_io') and start_info.get('io_type') == 'output':
                net_name = start_info['port_name']
            elif end_info.get('is_io') and end_info.get('io_type') == 'output':
                net_name = end_info['port_name']
            else:
                # Check if either position already has a net assigned
                existing_start_net = self.net_connections.get(start_key)
                existing_end_net = self.net_connections.get(end_key)
                
                if existing_start_net:
                    net_name = existing_start_net
                elif existing_end_net:
                    net_name = existing_end_net
                else:
                    net_name = self._generate_net_name()
            
            # Store the connections
            self.net_connections[start_key] = net_name
            self.net_connections[end_key] = net_name
            
            # Store wire info
            self.wires.append({
                'start': start,
                'end': end,
                'start_key': start_key,
                'end_key': end_key,
                'start_info': start_info,
                'end_info': end_info,
                'net': net_name
            })
    
    def _get_port_info_from_wire(self, port_data: Dict, pos_key: str) -> Dict:
        """Extract port information from wire endpoint data"""
        info = {
            'is_io': port_data.get('isIO', False),
            'io_type': port_data.get('ioType'),
            'pin_name': port_data.get('pinName'),
            'pin_index': port_data.get('pinIndex'),
            'position': (port_data.get('left', 0), port_data.get('top', 0))
        }
        
        # Generate valid port name
        if info['pin_name']:
            info['port_name'] = self._make_valid_port_name(info['pin_name'])
        elif info['pin_index']:
            info['port_name'] = f"P{info['pin_index']}"
        elif info['is_io']:
            # Prefer portId (Fabric object id) over position for matching
            port_id = port_data.get('portId')
            matched = False
            if port_id:
                for io in self.io_components:
                    if io['id'] == port_id:
                        info['port_name'] = io['name']
                        info['io_type'] = io['type']
                        matched = True
                        break
            if not matched:
                # Fallback: position-based matching (for legacy / position-rich data)
                for io in self.io_components:
                    io_pos_key = self._pos_key(io['position'][0], io['position'][1])
                    if io_pos_key == pos_key:
                        info['port_name'] = io['name']
                        info['io_type'] = io['type']
                        matched = True
                        break
            if not matched:
                # Last resort: use ioType from wire data + sequential count
                io_type_hint = port_data.get('ioType')
                if io_type_hint:
                    info['io_type'] = io_type_hint
                    candidates = [io for io in self.io_components if io['type'] == io_type_hint]
                    info['port_name'] = candidates[0]['name'] if candidates else f"io_{pos_key}"
                else:
                    info['port_name'] = f"io_{pos_key}"
        else:
            info['port_name'] = f"port_{pos_key}"
        
        return info
    
    def _get_component_pin_connections(self, component: Dict) -> Dict[str, str]:
        """Get all pin-to-net connections for a component"""
        connections = {}
        ic_type = component['type']
        
        # Find all wires connected to this component's pins
        for wire in self.wires:
            for endpoint_key in ['start', 'end']:
                endpoint = wire[endpoint_key]
                endpoint_info = wire[f'{endpoint_key}_info']
                
                # Check if this wire endpoint is connected to an IC pin (not I/O)
                if not endpoint_info.get('is_io'):
                    # Get IC type from wire data if available
                    wire_ic_type = endpoint.get('icType')
                    
                    # Match by IC type (if available) or by pin name pattern
                    if wire_ic_type == ic_type or (not wire_ic_type and endpoint_info.get('pin_name')):
                        pin_name = endpoint_info.get('pin_name')
                        pin_index = endpoint_info.get('pin_index')
                        
                        # Use the pin name directly if available (it's already in correct format like A1, B1)
                        if pin_name:
                            # The pin name from frontend is already correct (A1, B1, Y1, etc.)
                            connections[pin_name] = wire['net']
                        elif pin_index:
                            # Get proper pin name from database using pin number
                            db_pin_name = ICPinDatabase.get_pin_name_from_number(ic_type, pin_index)
                            if db_pin_name:
                                connections[db_pin_name] = wire['net']
        
        return connections
    
    def _get_unique_internal_nets(self) -> List[str]:
        """Get list of unique internal nets (not directly I/O names)"""
        io_names = set(self.inputs + self.outputs)
        all_nets = set(self.net_connections.values())
        internal_nets = [net for net in all_nets if net not in io_names and net.startswith('net_')]
        return sorted(internal_nets)
    
    def generate_verilog(self, module_name: str = "circuit_design") -> str:
        """Generate Verilog code from the circuit"""
        
        module_name = self._make_valid_identifier(module_name)
        
        lines = []
        lines.append("// Generated from Circuit Designer")
        lines.append(f"// Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Build module port list
        port_list = []
        for inp in self.inputs:
            port_list.append(f"    input wire {inp}")
        for out in self.outputs:
            port_list.append(f"    output wire {out}")
        
        lines.append(f"module {module_name}(")
        if port_list:
            lines.append(",\n".join(port_list))
        else:
            lines.append("    // No I/O ports defined - add switches and LEDs to your circuit")
        lines.append(");")
        lines.append("")
        
        # Internal wire declarations
        internal_nets = self._get_unique_internal_nets()
        if internal_nets:
            lines.append("    // Internal wires")
            for net in internal_nets:
                lines.append(f"    wire {net};")
            lines.append("")
        
        # Generate IC module instantiations
        if self.components:
            lines.append("    // Component instantiations")
            lines.append("")
            
            for comp in self.components:
                ic_type = comp['type']
                instance_name = comp['instance_name']
                valid_type = comp['valid_type']
                
                # Get pin connections for this component
                pin_connections = self._get_component_pin_connections(comp)
                
                lines.append(f"    // {ic_type} - {self._get_ic_description(ic_type)}")
                lines.append(f"    {valid_type} {instance_name} (")
                
                # Get IC port definitions
                ic_ports = ICPinDatabase.get_ic_ports(ic_type)
                port_lines = []
                
                # Connect inputs
                for port_name in ic_ports.get('inputs', []):
                    net_name = pin_connections.get(port_name, "1'bz")  # High-Z if unconnected
                    port_lines.append(f"        .{port_name}({net_name})")
                
                # Connect outputs
                for port_name in ic_ports.get('outputs', []):
                    net_name = pin_connections.get(port_name)
                    if net_name:
                        port_lines.append(f"        .{port_name}({net_name})")
                    else:
                        port_lines.append(f"        .{port_name}()")  # Leave unconnected
                
                if port_lines:
                    lines.append(",\n".join(port_lines))
                
                lines.append("    );")
                lines.append("")
        
        lines.append("endmodule")
        
        # Add IC module definitions
        lines.append(self._generate_ic_modules())
        
        return "\n".join(lines)
    
    def _get_ic_description(self, ic_type: str) -> str:
        """Get a description for common ICs"""
        descriptions = {
            '7400': 'Quad 2-input NAND gate',
            '7402': 'Quad 2-input NOR gate (Output-first pinout)',
            '7404': 'Hex inverter',
            '7408': 'Quad 2-input AND gate',
            '7432': 'Quad 2-input OR gate',
            '7486': 'Quad 2-input XOR gate',
            '7474': 'Dual D flip-flop',
            '7476': 'Dual JK flip-flop',
            '74138': '3-to-8 line decoder',
            '74139': 'Dual 2-to-4 line decoder',
            '74147': '10-to-4 priority encoder',
            '74153': 'Dual 4-to-1 multiplexer',
            '7447': 'BCD to 7-segment decoder',
            '7485': '4-bit magnitude comparator',
            '7490': 'Decade counter',
            '7493': '4-bit binary counter',
            '74121': 'Monostable multivibrator',
            '74245': 'Octal bus transceiver',
            '555': 'Timer IC',
            '4017': 'Decade counter/divider'
        }
        return descriptions.get(ic_type, 'IC component')
    
    def _generate_ic_modules(self) -> str:
        """Generate Verilog module definitions for ICs used"""
        if not self.components:
            return ""
        
        lines = []
        lines.append("")
        lines.append("")
        lines.append("// ============================================")
        lines.append("// IC Module Definitions")
        lines.append("// ============================================")
        lines.append("")
        
        generated = set()
        
        for comp in self.components:
            ic_type = comp['type']
            valid_type = comp['valid_type']
            
            if valid_type in generated:
                continue
            generated.add(valid_type)
            
            ic_ports = ICPinDatabase.get_ic_ports(ic_type)
            function = ICPinDatabase.get_function(ic_type)
            
            lines.append(f"// {self._get_ic_description(ic_type)}")
            lines.append(f"module {valid_type} (")
            
            port_lines = []
            for inp in ic_ports.get('inputs', []):
                port_lines.append(f"    input wire {inp}")
            for out in ic_ports.get('outputs', []):
                port_lines.append(f"    output wire {out}")
            
            if port_lines:
                lines.append(",\n".join(port_lines))
            
            lines.append(");")
            lines.append("")
            
            # Generate logic based on function type
            if function == 'nand':
                lines.append("    // NAND gate implementation")
                lines.append("    assign Y1 = ~(A1 & B1);")
                lines.append("    assign Y2 = ~(A2 & B2);")
                lines.append("    assign Y3 = ~(A3 & B3);")
                lines.append("    assign Y4 = ~(A4 & B4);")
            elif function == 'nor':
                lines.append("    // NOR gate implementation")
                lines.append("    assign Y1 = ~(A1 | B1);")
                lines.append("    assign Y2 = ~(A2 | B2);")
                lines.append("    assign Y3 = ~(A3 | B3);")
                lines.append("    assign Y4 = ~(A4 | B4);")
            elif function == 'and':
                lines.append("    // AND gate implementation")
                lines.append("    assign Y1 = A1 & B1;")
                lines.append("    assign Y2 = A2 & B2;")
                lines.append("    assign Y3 = A3 & B3;")
                lines.append("    assign Y4 = A4 & B4;")
            elif function == 'or':
                lines.append("    // OR gate implementation")
                lines.append("    assign Y1 = A1 | B1;")
                lines.append("    assign Y2 = A2 | B2;")
                lines.append("    assign Y3 = A3 | B3;")
                lines.append("    assign Y4 = A4 | B4;")
            elif function == 'xor':
                lines.append("    // XOR gate implementation")
                lines.append("    assign Y1 = A1 ^ B1;")
                lines.append("    assign Y2 = A2 ^ B2;")
                lines.append("    assign Y3 = A3 ^ B3;")
                lines.append("    assign Y4 = A4 ^ B4;")
            elif function == 'not':
                lines.append("    // Inverter implementation")
                lines.append("    assign Y1 = ~A1;")
                lines.append("    assign Y2 = ~A2;")
                lines.append("    assign Y3 = ~A3;")
                lines.append("    assign Y4 = ~A4;")
                lines.append("    assign Y5 = ~A5;")
                lines.append("    assign Y6 = ~A6;")
            else:
                lines.append("    // TODO: Implement logic based on datasheet")
            
            lines.append("")
            lines.append("endmodule")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_vhdl(self, entity_name: str = "circuit_design") -> str:
        """Generate VHDL code from the circuit"""
        
        entity_name = self._make_valid_identifier(entity_name)
        
        lines = []
        lines.append("-- Generated from Circuit Designer")
        lines.append(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("library IEEE;")
        lines.append("use IEEE.std_logic_1164.all;")
        lines.append("")
        
        # Entity declaration
        lines.append(f"entity {entity_name} is")
        lines.append("    port(")
        
        port_lines = []
        for inp in self.inputs:
            port_lines.append(f"        {inp} : in std_logic")
        for out in self.outputs:
            port_lines.append(f"        {out} : out std_logic")
        
        if port_lines:
            lines.append(";\n".join(port_lines))
        else:
            lines.append("        -- No I/O defined - add switches and LEDs")
        
        lines.append("    );")
        lines.append(f"end {entity_name};")
        lines.append("")
        
        # Architecture
        lines.append(f"architecture structural of {entity_name} is")
        lines.append("")
        
        # Component declarations
        generated_components = set()
        if self.components:
            lines.append("    -- Component declarations")
            for comp in self.components:
                if comp['valid_type'] not in generated_components:
                    generated_components.add(comp['valid_type'])
                    ic_type = comp['type']
                    ic_ports = ICPinDatabase.get_ic_ports(ic_type)
                    
                    lines.append(f"    component {comp['valid_type']}")
                    lines.append("        port(")
                    
                    comp_port_lines = []
                    for inp in ic_ports.get('inputs', []):
                        comp_port_lines.append(f"            {inp} : in std_logic")
                    for out in ic_ports.get('outputs', []):
                        comp_port_lines.append(f"            {out} : out std_logic")
                    
                    if comp_port_lines:
                        lines.append(";\n".join(comp_port_lines))
                    
                    lines.append("        );")
                    lines.append("    end component;")
                    lines.append("")
        
        # Signal declarations
        internal_nets = self._get_unique_internal_nets()
        if internal_nets:
            lines.append("    -- Internal signals")
            for net in internal_nets:
                lines.append(f"    signal {net} : std_logic;")
            lines.append("")
        
        lines.append("begin")
        lines.append("")
        
        # Component instantiations
        if self.components:
            lines.append("    -- Component instantiations")
            for comp in self.components:
                instance_name = comp['instance_name']
                ic_type = comp['type']
                valid_type = comp['valid_type']
                
                pin_connections = self._get_component_pin_connections(comp)
                ic_ports = ICPinDatabase.get_ic_ports(ic_type)
                
                lines.append(f"    {instance_name}: {valid_type}")
                lines.append("        port map(")
                
                port_map_lines = []
                for inp in ic_ports.get('inputs', []):
                    net_name = pin_connections.get(inp, "'Z'")
                    port_map_lines.append(f"            {inp} => {net_name}")
                for out in ic_ports.get('outputs', []):
                    net_name = pin_connections.get(out, "open")
                    port_map_lines.append(f"            {out} => {net_name}")
                
                if port_map_lines:
                    lines.append(",\n".join(port_map_lines))
                
                lines.append("        );")
                lines.append("")
        
        lines.append(f"end structural;")
        
        return "\n".join(lines)


def generate_testbench_from_circuit(circuit_data: Dict[str, Any],
                                     module_name: str = "circuit_design") -> str:
    """Generate a Verilog testbench template for the circuit."""
    gen = SchematicHDLGenerator(circuit_data)
    mod = gen._make_valid_identifier(module_name)

    inputs = gen.inputs
    outputs = gen.outputs

    n_in = len(inputs)
    n_vectors = min(2 ** n_in, 16) if n_in > 0 else 4

    lines = [
        "`timescale 1ns/1ps",
        "",
        f"module tb_{mod};",
        "",
    ]

    if inputs:
        lines.append("  // DUT inputs")
        for p in inputs:
            lines.append(f"  reg {p};")
        lines.append("")

    if outputs:
        lines.append("  // DUT outputs")
        for p in outputs:
            lines.append(f"  wire {p};")
        lines.append("")

    conn = [f"    .{p}({p})" for p in inputs + outputs]
    lines += [
        "  // Instantiate DUT",
        f"  {mod} dut (",
        ",\n".join(conn),
        "  );",
        "",
        "  initial begin",
        f'    $dumpfile("tb_{mod}.vcd");',
        f'    $dumpvars(0, tb_{mod});',
        "",
    ]

    if n_in == 0:
        lines.append("    // No inputs — connect switches to your circuit")
        lines.append("    #100;")
    else:
        lines.append("    // Exhaustive input combinations")
        for i in range(n_vectors):
            parts = []
            for j, p in enumerate(inputs):
                bit = (i >> (n_in - 1 - j)) & 1
                parts.append(f"{p} = {bit}")
            lines.append(f"    {'; '.join(parts)}; #10;")

    lines += [
        "",
        "    $finish;",
        "  end",
        "",
        "endmodule",
    ]
    return "\n".join(lines)


def generate_hdl_from_circuit(circuit_data: Dict[str, Any],
                               language: str = "verilog",
                               module_name: str = "circuit_design") -> str:
    """
    Main entry point for generating HDL from circuit data
    
    Args:
        circuit_data: The circuit design data from the canvas
        language: 'verilog' or 'vhdl'
        module_name: Name for the generated module/entity
    
    Returns:
        Generated HDL code as string
    """
    generator = SchematicHDLGenerator(circuit_data)
    
    if language.lower() == 'vhdl':
        return generator.generate_vhdl(module_name)
    else:
        return generator.generate_verilog(module_name)
