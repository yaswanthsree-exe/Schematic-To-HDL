import pytest

from hdl_gen.graph_to_hdl import UnknownGateClass, flatten, generate_all
from pattern_engine import compress


def xor_graph():
    return {
        "G1": {"cls": "NOT", "inputs": ["A"],        "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["B"],        "outputs": []},
        "G3": {"cls": "AND", "inputs": ["A", "G2"],  "outputs": []},
        "G4": {"cls": "AND", "inputs": ["G1", "B"],  "outputs": []},
        "G5": {"cls": "OR",  "inputs": ["G3", "G4"], "outputs": ["Y"]},
    }


def sr_graph():
    return {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": ["Qbar"]},
    }


def test_unknown_class_raises_instead_of_defaulting_to_and():
    """Regression: _OP.get(cls, ("&", False, False)) silently emitted AND."""
    bad = {"G1": {"cls": "MYSTERY", "inputs": ["A", "B"], "outputs": ["Y"]}}
    with pytest.raises(UnknownGateClass):
        generate_all(bad, {"A", "B"}, {"Y"}, "bad")


def test_compressed_xor_emits_xor_operator():
    r = compress(xor_graph())
    hdl = generate_all(r.graph, {"A", "B"}, {"Y"}, "x")
    assert "^" in hdl["verilog_behavioral"]
    assert hdl["total_gates"] == 1


def test_compressed_sr_latch_emits_cross_coupled_form():
    r = compress(sr_graph())
    hdl = generate_all(r.graph, {"R", "S"}, {"Q", "Qbar"}, "sr")
    v = hdl["verilog_behavioral"]
    assert "SRLATCH" in v
    assert "~(R |" in v or "~(S |" in v


def test_compressed_sr_latch_emits_vhdl_without_raising():
    """generate_all always produces VHDL, so a macro missing from the VHDL
    table would make the whole call raise."""
    r = compress(sr_graph())
    hdl = generate_all(r.graph, {"R", "S"}, {"Q", "Qbar"}, "sr")
    v = hdl["vhdl"]
    assert "SRLATCH" in v
    assert "not (" in v


def test_sr_latch_q_and_qbar_are_distinct_signals():
    """Both macro outputs are primary outputs, and the graph contract names a
    gate's output by gate id alone — so both resolved to the same wire and the
    latch emitted Q == Qbar.  Functionally wrong HDL with no error."""
    r = compress(sr_graph())
    for artifact, assign in (("verilog_behavioral", "assign"), ("vhdl", "<=")):
        lines = [l.strip() for l in r and generate_all(
            r.graph, {"R", "S"}, {"Q", "Qbar"}, "sr")[artifact].splitlines()]
        q = next(l for l in lines if l.startswith(f"Q {assign}")
                 or l.startswith(f"{assign} Q ") or l.startswith("assign Q ="))
        qb = next(l for l in lines if l.startswith(f"Qbar {assign}")
                  or l.startswith("assign Qbar ="))
        q_src = q.split("=" if assign == "assign" else "<=")[-1].strip(" ;")
        qb_src = qb.split("=" if assign == "assign" else "<=")[-1].strip(" ;")
        assert q_src != qb_src, f"{artifact}: Q and Qbar both driven by {q_src}"


def test_flatten_restores_the_gate_level_view():
    r = compress(xor_graph())
    flat = flatten(r.graph)
    assert set(flat) == {"G1", "G2", "G3", "G4", "G5"}
    assert flat["G5"]["cls"] == "OR"


def test_flatten_is_identity_on_an_uncompressed_graph():
    g = xor_graph()
    assert flatten(g) == g


def test_ic_bom_counts_physical_gates_not_macros():
    r = compress(xor_graph())
    hdl = generate_all(r.graph, {"A", "B"}, {"Y"}, "x")
    parts = {b["gate"]: b["count"] for b in hdl["ic_bom"]}
    assert parts == {"NOT": 2, "AND": 2, "OR": 1}


def test_structural_verilog_is_gate_level_not_macro_level():
    r = compress(xor_graph())
    hdl = generate_all(r.graph, {"A", "B"}, {"Y"}, "x")
    s = hdl["verilog_structural"]
    assert "not1" in s and "and2" in s and "or2" in s
    assert "SRLATCH" not in s
