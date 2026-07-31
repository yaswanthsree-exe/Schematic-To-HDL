"""Port binding when a matched gate has SEVERAL external inputs.

The first implementation resolved a commutative attach point to the node's
*unique* external input and rejected anything else.  That is fine for XOR and a
bare SR latch, but every clocked device breaks it: a JK's input NAND takes J,
CLK and the Qbar feedback, so the node has two external sources and one
internal.  Binding therefore has to solve an assignment -- each port picks one
external source, ports on the same node must pick different ones -- rather than
read a single value off each node.

Also guards the rule that every external input must be claimed by some port,
without which a latch with an async reset would match the plain latch pattern
and emit HDL that silently drops the reset.
"""
import pytest

from pattern_engine.library import Pattern, load_library
from pattern_engine.matcher import candidates

LIB = {p.name: p for p in load_library()}


def _pat(**kw):
    """A 2-NAND cross-coupled latch pattern with configurable port wiring."""
    from pattern_engine.library import InputPort, OutputPort, PatternEdge
    base = dict(
        name="t", cls="T", level=1,
        nodes=(("n1", "NAND"), ("n2", "NAND")),
        edges=(PatternEdge("n1", "n2", (1,)), PatternEdge("n2", "n1", (1,))),
        inputs=(InputPort("A", (("n1", 0),)), InputPort("B", (("n2", 0),))),
        outputs=(OutputPort("Q", "n1"), OutputPort("Qbar", "n2")),
        validator="sr_latch_nand", expected=None, allow_shared_inputs=False,
    )
    base.update(kw)
    return Pattern(**base)


def test_two_external_inputs_on_one_node_can_bind():
    """n1 has TWO external sources (S and CLK) plus the feedback."""
    from pattern_engine.library import InputPort
    g = {
        "G1": {"cls": "NAND", "inputs": ["S", "CLK", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["R", "G1"],        "outputs": ["Qb"]},
    }
    p = _pat(inputs=(InputPort("S", (("n1", 0),)),
                     InputPort("CLK", (("n1", 0),)),
                     InputPort("R", (("n2", 0),))))
    ms = candidates(g, p)
    assert len(ms) == 1
    assert set(ms[0].inputs) == {"S", "CLK", "R"}
    # S and CLK must land on DIFFERENT sources, both from G1
    assert ms[0].inputs["S"] != ms[0].inputs["CLK"]
    assert {ms[0].inputs["S"], ms[0].inputs["CLK"]} == {"S", "CLK"}
    assert ms[0].inputs["R"] == "R"


def test_shared_clock_across_two_nodes_resolves_by_intersection():
    """CLK attaches to BOTH nodes, so it must resolve to the source common to
    both -- which is what separates the clock from the data inputs.

    J and K themselves are NOT determined: a cross-coupled NAND pair is
    structurally symmetric, so which gate is the "J side" is a mirror image the
    topology cannot distinguish (same reason S/R and Q/Qbar can mirror).  Only
    the clock is pinned down here, and that is the property under test.
    """
    from pattern_engine.library import InputPort
    g = {
        "G1": {"cls": "NAND", "inputs": ["J", "CLK", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["K", "CLK", "G1"], "outputs": ["Qb"]},
    }
    p = _pat(inputs=(InputPort("J", (("n1", 0),)),
                     InputPort("K", (("n2", 0),)),
                     InputPort("CLK", (("n1", 0), ("n2", 0)))))
    ms = candidates(g, p)
    assert len(ms) == 1
    assert ms[0].inputs["CLK"] == "CLK"
    assert {ms[0].inputs["J"], ms[0].inputs["K"]} == {"J", "K"}


def test_no_consistent_assignment_is_rejected():
    """Three ports on a node that only offers two external sources."""
    from pattern_engine.library import InputPort
    g = {
        "G1": {"cls": "NAND", "inputs": ["S", "CLK", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["R", "G1"],        "outputs": ["Qb"]},
    }
    p = _pat(inputs=(InputPort("A", (("n1", 0),)),
                     InputPort("B", (("n1", 0),)),
                     InputPort("C", (("n1", 0),)),
                     InputPort("D", (("n2", 0),))))
    assert candidates(g, p) == []


def test_unclaimed_external_input_rejects_the_match():
    """G1 has an extra async reset nobody declared.  Matching anyway would emit
    HDL that drops it -- a silently wrong circuit."""
    from pattern_engine.library import InputPort
    g = {
        "G1": {"cls": "NAND", "inputs": ["S", "RESET", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["R", "G1"],          "outputs": ["Qb"]},
    }
    p = _pat(inputs=(InputPort("S", (("n1", 0),)),
                     InputPort("R", (("n2", 0),))))
    assert candidates(g, p) == []


def test_existing_sr_latch_still_binds():
    """The single-external-input case must keep working unchanged."""
    sr = LIB["sr_latch_nor"]
    g = {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": ["Qbar"]},
    }
    ms = candidates(g, sr)
    assert len(ms) == 1
    assert set(ms[0].inputs.values()) == {"R", "S"}


def test_existing_xor_still_binds():
    xor = LIB["xor_and_or_not"]
    g = {
        "G1": {"cls": "NOT", "inputs": ["A"],        "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["B"],        "outputs": []},
        "G3": {"cls": "AND", "inputs": ["A", "G2"],  "outputs": []},
        "G4": {"cls": "AND", "inputs": ["G1", "B"],  "outputs": []},
        "G5": {"cls": "OR",  "inputs": ["G3", "G4"], "outputs": ["Y"]},
    }
    ms = candidates(g, xor)
    assert len(ms) == 1
    assert set(ms[0].inputs.values()) == {"A", "B"}


def test_binding_is_deterministic():
    from pattern_engine.library import InputPort
    g = {
        "G1": {"cls": "NAND", "inputs": ["S", "CLK", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NAND", "inputs": ["R", "G1"],        "outputs": ["Qb"]},
    }
    p = _pat(inputs=(InputPort("S", (("n1", 0),)),
                     InputPort("CLK", (("n1", 0),)),
                     InputPort("R", (("n2", 0),))))
    first = candidates(g, p)[0].inputs
    for _ in range(5):
        assert candidates(g, p)[0].inputs == first
