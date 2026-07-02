import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic

with schemdraw.Drawing(show=False) as d:
    s_x, s_y = 0, 0
    d1_y_line = 1
    d.add(logic.Dot().at((s_x, s_y)))
    d.add(logic.Line().right().at((-1, d1_y_line)).length(2).label('D1'))
    
    start_bp = (s_x, d1_y_line - 0.2)
    end_bp = (s_x, d1_y_line + 0.2)
    
    l1 = d.add(logic.Line().up().at((s_x, s_y)).to(start_bp))
    
    # Try an Arc
    # theta1=-90, theta2=90 draws a right-facing semicircle
    arc = d.add(elm.Arc(theta1=-90, theta2=90, width=0.4, height=0.4).at(l1.end))
    
    # Since Arc might not perfectly set anchor 'end', manually continue
    # wait, schemdraw 0.15 uses radius for Arc?
    # Actually, Jumper is the official element. Let's try Jumper again.
    d.add(logic.Line().up().at(end_bp).length(0.5))

    # Test Jumper on the left
    d.add(logic.Line().right().at((-2, 1)).length(0.5))
    d.add(logic.Line().up().at((-1.5, 0)).length(0.8))
    # No, Jumper is complicated.
    
    d.save('test_arc.png', dpi=100)
    print("Test arc generated")
