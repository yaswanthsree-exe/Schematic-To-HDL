"""Pre-existing Part1->Part2 handoff bug, found while testing an SR flip-flop.

predict.generate_netlist synthesises a 'Q' output for the last gate in
topological order when no output was detected, and predict_circuit merges it
into global_outputs -- but never writes it into graph[last]["outputs"].  Part 2
builds its producer map from the graph, so Q had no driver and was emitted as
`assign Q = 1'b0;` -- a dead output, in both Verilog and VHDL.

Mirrors predict.py's own rule: the fallback applies only when NO declared
output has a producer, so a partially-driven circuit is left alone.
"""
from hdl_gen.graph_to_hdl import generate_all
from pattern_engine import compress


def sr_flipflop_graph():
    """What predict.py actually produced for the SR flip-flop image: two gating
    ANDs plus a cross-coupled NOR pair, and no gate declaring an output."""
    return {
        "G1": {"cls": "AND", "inputs": ["B", "Pulse"], "outputs": []},
        "G2": {"cls": "AND", "inputs": ["Pulse", "A"], "outputs": []},
        "G3": {"cls": "NOR", "inputs": ["G2", "G4"],   "outputs": []},
        "G4": {"cls": "NOR", "inputs": ["G1", "G3"],   "outputs": []},
    }


PORTS = ({"A", "B", "Pulse"}, {"Q"})


def _assign_for(hdl_text, port, sep):
    for line in hdl_text.splitlines():
        s = line.strip()
        if s.startswith(f"assign {port} {sep}") or s.startswith(f"{port} {sep}"):
            return s.split(sep, 1)[1].strip(" ;")
    raise AssertionError(f"no assignment for {port}")


def test_undriven_output_is_not_tied_to_zero_verilog():
    gi, go = PORTS
    hdl = generate_all(sr_flipflop_graph(), gi, go, "sr")
    assert _assign_for(hdl["verilog_behavioral"], "Q", "=") != "1'b0"


def test_undriven_output_is_not_tied_to_zero_vhdl():
    gi, go = PORTS
    hdl = generate_all(sr_flipflop_graph(), gi, go, "sr")
    assert _assign_for(hdl["vhdl"], "Q", "<=") != "'0'"


def test_undriven_output_still_fixed_after_compression():
    gi, go = PORTS
    r = compress(sr_flipflop_graph())
    # Recognised as a GATED SR latch: the two ANDs gate S and R with the clock,
    # which the bare-latch pattern used to miss.
    assert [m.cls for m in r.matches] == ["GATED_SRLATCH"]
    hdl = generate_all(r.graph, gi, go, "sr")
    assert _assign_for(hdl["verilog_behavioral"], "Q", "=") != "1'b0"


def test_driven_outputs_are_left_alone():
    """The benchmark drives OUT_1 properly -- the fallback must not fire."""
    g = {
        "G1": {"cls": "AND", "inputs": ["A", "B"],   "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["C"],        "outputs": []},
        "G3": {"cls": "OR",  "inputs": ["G1", "G2"], "outputs": []},
        "G4": {"cls": "NOT", "inputs": ["G3"],       "outputs": ["OUT_1"]},
    }
    hdl = generate_all(g, {"A", "B", "C"}, {"OUT_1"}, "bench")
    assert _assign_for(hdl["verilog_behavioral"], "OUT_1", "=") == "w_G4"


def test_partially_driven_circuit_does_not_get_a_guessed_driver():
    """One output is real, one is not -- guessing a driver for the second would
    invent connectivity, so it must stay 1'b0."""
    g = {
        "G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": ["P"]},
        "G2": {"cls": "OR",  "inputs": ["A", "B"], "outputs": []},
    }
    hdl = generate_all(g, {"A", "B"}, {"P", "MISSING"}, "part")
    assert _assign_for(hdl["verilog_behavioral"], "P", "=") == "w_G1"
    assert _assign_for(hdl["verilog_behavioral"], "MISSING", "=") == "1'b0"
