"""HDL emission for the sequential macro classes.

Guards the operand-resolution bug these emitters were born with: port names
were matched to argument wires BY POSITION, but ``ports["inputs"]`` and the
node's ``inputs`` list are in unrelated orders, so a JK clocked on CLK emitted
`always @(posedge K)`.  Every test here asserts on the *named* signal.
"""
import pytest

from hdl_gen.graph_to_hdl import generate_all
from pattern_engine import compress


def jk_graph():
    return {
        "G1": {"cls": "NAND", "inputs": ["J", "CLK", "G4"], "outputs": []},
        "G2": {"cls": "NAND", "inputs": ["K", "CLK", "G3"], "outputs": []},
        "G3": {"cls": "NAND", "inputs": ["G1", "G4"],       "outputs": ["Q"]},
        "G4": {"cls": "NAND", "inputs": ["G2", "G3"],       "outputs": ["Qbar"]},
    }


def gated_sr_graph():
    return {
        "G1": {"cls": "NAND", "inputs": ["S", "CLK"], "outputs": []},
        "G2": {"cls": "NAND", "inputs": ["R", "CLK"], "outputs": []},
        "G3": {"cls": "NAND", "inputs": ["G1", "G4"], "outputs": ["Q"]},
        "G4": {"cls": "NAND", "inputs": ["G2", "G3"], "outputs": ["Qbar"]},
    }


def nand_latch_graph():
    return {
        "G1": {"cls": "NAND", "inputs": ["Sbar", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["Rbar", "G1"], "outputs": ["Qbar"]},
    }


def _hdl(graph, gi, go, name):
    return generate_all(compress(graph).graph, gi, go, name)


def test_jk_clocks_on_the_clock_not_a_data_input():
    v = _hdl(jk_graph(), {"J", "K", "CLK"}, {"Q", "Qbar"}, "jk")["verilog_behavioral"]
    assert "posedge CLK" in v
    assert "posedge J" not in v and "posedge K" not in v


def test_jk_cases_use_j_and_k():
    v = _hdl(jk_graph(), {"J", "K", "CLK"}, {"Q", "Qbar"}, "jk")["verilog_behavioral"]
    assert "case ({J, K})" in v


def test_jk_toggles_on_11():
    """J=K=1 toggling is what makes it a JK rather than a gated SR."""
    v = _hdl(jk_graph(), {"J", "K", "CLK"}, {"Q", "Qbar"}, "jk")["verilog_behavioral"]
    assert "2'b11" in v
    toggle = [l for l in v.splitlines() if "2'b11" in l][0]
    assert "~" in toggle


def test_gated_sr_gates_both_inputs_with_the_clock():
    v = _hdl(gated_sr_graph(), {"S", "R", "CLK"}, {"Q", "Qbar"},
             "gsr")["verilog_behavioral"]
    assert "S & CLK" in v or "CLK & S" in v
    assert "R & CLK" in v or "CLK & R" in v


def test_nand_latch_is_active_low():
    v = _hdl(nand_latch_graph(), {"Sbar", "Rbar"}, {"Q", "Qbar"},
             "nl")["verilog_behavioral"]
    assert "Sbar &" in v and "Rbar &" in v


def test_q_and_qbar_are_distinct_signals():
    """A macro's outputs are named by gate id alone, so without the suffix
    table every output port collapses onto one wire and Q == Qbar."""
    for g, gi in ((jk_graph(), {"J", "K", "CLK"}),
                  (gated_sr_graph(), {"S", "R", "CLK"}),
                  (nand_latch_graph(), {"Sbar", "Rbar"})):
        v = _hdl(g, gi, {"Q", "Qbar"}, "m")["verilog_behavioral"]
        q = [l for l in v.splitlines() if l.strip().startswith("assign Q ")]
        qb = [l for l in v.splitlines() if l.strip().startswith("assign Qbar ")]
        assert q and qb
        assert q[0].split("=")[1].strip() != qb[0].split("=")[1].strip()


@pytest.mark.parametrize("graph,gi", [
    (jk_graph(), {"J", "K", "CLK"}),
    (gated_sr_graph(), {"S", "R", "CLK"}),
    (nand_latch_graph(), {"Sbar", "Rbar"}),
])
def test_vhdl_is_emitted_too(graph, gi):
    """generate_all always produces both languages; a macro registered in only
    one table would raise for every circuit containing it."""
    out = _hdl(graph, gi, {"Q", "Qbar"}, "m")
    assert out["vhdl"].strip()
    assert out["verilog_behavioral"].strip()


@pytest.mark.parametrize("graph,gi", [
    (jk_graph(), {"J", "K", "CLK"}),
    (gated_sr_graph(), {"S", "R", "CLK"}),
])
def test_children_still_expand_to_gates_for_the_bom(graph, gi):
    """Structural output and the IC BOM must still see the original NANDs."""
    out = _hdl(graph, gi, {"Q", "Qbar"}, "m")
    assert any(b["gate"] == "NAND" for b in out["ic_bom"])
    assert out["total_physical_gates"] == 4
