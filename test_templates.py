import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm
import random

def generate_fixed_2to4_decoder(d):
    """Draws a strictly routed 2-to-4 Decoder."""
    d.config(fontsize=12, lw=2.0)
    
    # Inputs
    in_a = d.add(logic.Line().right().length(1.5).at((0, 0)).label('A', 'left'))
    dot_a = d.add(logic.Dot())
    not_a = d.add(logic.Not().right().at(dot_a.center))
    
    in_b = d.add(logic.Line().right().length(1.5).at((0, -4.0)).label('B', 'left'))
    dot_b = d.add(logic.Dot())
    not_b = d.add(logic.Not().right().at(dot_b.center))
    
    # Vertical Trunks (X coordinates)
    # To cleanly route lines down without overlapping, we define specific X columns
    col_a = dot_a.center[0]
    col_na = not_a.out[0] + 0.5
    col_b = dot_b.center[0]
    col_nb = not_b.out[0] + 0.5
    
    # Establish the top of the trunks
    d.add(logic.Line().right().at(not_a.out).tox(col_na))
    dot_na_top = d.add(logic.Dot())
    
    d.add(logic.Line().right().at(not_b.out).tox(col_nb))
    dot_nb_top = d.add(logic.Dot())
    
    # AND Gates
    x_and = col_nb + 3.0
    y_start = 2.0
    dy_out = 2.5
    
    gates = []
    out_labels = ['Y0', 'Y1', 'Y2', 'Y3']
    
    for i in range(4):
        y = y_start - i * dy_out
        g = d.add(logic.And().right().at((x_and, y)).label(f"G{i+3}", "bottom"))
        gates.append(g)
        d.add(logic.Line().right().at(g.out).length(1.5).label(out_labels[i], 'right'))
        
    # ROUTING STRATEGY:
    # We must explicitly draw the vertical lines down the columns, place a dot, and draw horizontal lines to the gate.
    # No `toy()` shortcuts that skip explicit segments.
    
    # Y0 = ~A & ~B
    # Route ~A to G3 in1
    d.add(logic.Line().down().at(dot_na_top.center).toy(gates[0].in1[1]))
    dot_y0a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[0].in1))
    
    # Route ~B to G3 in2
    d.add(logic.Line().up().at(dot_nb_top.center).toy(gates[0].in2[1]))
    dot_y0b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[0].in2))
    
    # Y1 = ~A & B
    # Route ~A to G4 in1
    d.add(logic.Line().down().at(dot_y0a.center).toy(gates[1].in1[1]))
    dot_y1a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[1].in1))
    
    # Route B to G4 in2
    d.add(logic.Line().up().at(dot_b.center).toy(gates[1].in2[1]))
    dot_y1b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[1].in2))
    
    # Y2 = A & ~B
    # Route A to G5 in1
    d.add(logic.Line().down().at(dot_a.center).toy(gates[2].in1[1]))
    dot_y2a = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[2].in1))
    
    # Route ~B to G5 in2
    d.add(logic.Line().down().at(dot_nb_top.center).toy(gates[2].in2[1]))
    dot_y2b = d.add(logic.Dot())
    d.add(logic.Line().right().to(gates[2].in2))
    
    # Y3 = A & B
    # Route A to G6 in1
    d.add(logic.Line().down().at(dot_y2a.center).toy(gates[3].in1[1]))
    d.add(logic.Line().right().to(gates[3].in1))
    
    # Route B to G6 in2
    d.add(logic.Line().down().at(dot_b.center).toy(gates[3].in2[1]))
    d.add(logic.Line().right().to(gates[3].in2))
    
    # Extend bottom-most trunk lines purely for visual completeness
    # (Just slightly past the last connection)
    d.add(logic.Line().down().at(gates[3].in1).at((col_na, gates[3].in1[1])).length(1.0))
    d.add(logic.Line().down().at(gates[3].in2).at((col_nb, gates[3].in2[1])).length(1.0))
    
    return d

if __name__ == "__main__":
    with schemdraw.Drawing(show=False) as d:
        generate_fixed_2to4_decoder(d)
        d.save(f"test_fixed_decoder.png", transparent=False, dpi=100)
        print("Generated test_fixed_decoder.png")
