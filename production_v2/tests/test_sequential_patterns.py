"""Recognition of the sequential family: NAND latch, gated SR, JK.

The discriminations that matter here are the ones topology alone does not
give you.  A gated SR and a JK are the SAME four NAND gates in the same
arrangement; the only difference is whether the core outputs feed back into
the input gates.  Confusing them would emit a circuit that cannot toggle.
"""
from pattern_engine import compress


def jk_graph():
    """J/K gated by CLK, cross-coupled NAND core, outputs fed back."""
    return {
        "G1": {"cls": "NAND", "inputs": ["J", "CLK", "G4"], "outputs": []},
        "G2": {"cls": "NAND", "inputs": ["K", "CLK", "G3"], "outputs": []},
        "G3": {"cls": "NAND", "inputs": ["G1", "G4"],       "outputs": ["Q"]},
        "G4": {"cls": "NAND", "inputs": ["G2", "G3"],       "outputs": ["Qbar"]},
    }


def gated_sr_graph():
    """Same shape as the JK but WITHOUT output feedback."""
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


def _one(graph):
    r = compress(graph)
    assert len(r.matches) == 1, [m.cls for m in r.matches]
    return r.matches[0]


def test_jk_recognised():
    m = _one(jk_graph())
    assert m.cls == "JKFF"
    assert sorted(m.absorbed) == ["G1", "G2", "G3", "G4"]


def test_gated_sr_recognised():
    m = _one(gated_sr_graph())
    assert m.cls == "GATED_SRLATCH"
    assert sorted(m.absorbed) == ["G1", "G2", "G3", "G4"]


def test_jk_is_not_reported_as_gated_sr():
    """Feedback present -> must be the JK, never the gated SR."""
    assert _one(jk_graph()).cls == "JKFF"


def test_gated_sr_is_not_reported_as_jk():
    """No feedback -> must NOT claim toggle behaviour."""
    assert _one(gated_sr_graph()).cls == "GATED_SRLATCH"


def test_nand_latch_recognised():
    m = _one(nand_latch_graph())
    assert m.cls == "SRLATCH_NAND"


def test_nand_latch_distinct_from_nor_latch():
    nor = {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": ["Qbar"]},
    }
    assert _one(nor).cls == "SRLATCH"
    assert _one(nand_latch_graph()).cls == "SRLATCH_NAND"


def test_composite_wins_over_its_own_core():
    """The 4-gate device must absorb all four gates, not degrade into a bare
    2-gate latch plus two loose NANDs."""
    for g in (jk_graph(), gated_sr_graph()):
        m = _one(g)
        assert len(m.absorbed) == 4


def test_latch_with_undeclared_reset_is_refused():
    """An async reset nobody modelled must not match the plain latch -- the
    emitted HDL would silently drop the reset."""
    g = {
        "G1": {"cls": "NOR", "inputs": ["R", "G2", "RST"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"],        "outputs": ["Qbar"]},
    }
    assert compress(g).matches == []


def test_clock_is_the_shared_input():
    """CLK must bind to the signal common to both gating gates.

    J and K may come out mirrored -- a cross-coupled pair is symmetric, so
    which side is "J" is not recoverable from topology -- but the clock is
    pinned down, because it is the only source both gating gates share.
    """
    r = compress(jk_graph())
    macro = next(v for v in r.graph.values() if v["cls"] == "JKFF")
    assert macro["ports"]["inputs"]["CLK"] == "CLK"
    assert {macro["ports"]["inputs"]["J"],
            macro["ports"]["inputs"]["K"]} == {"J", "K"}
    assert set(macro["outputs"]) == {"Q", "Qbar"}


def test_benchmark_still_untouched():
    g = {
        "G1": {"cls": "AND", "inputs": ["A", "B"],   "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["C"],        "outputs": []},
        "G3": {"cls": "OR",  "inputs": ["G1", "G2"], "outputs": []},
        "G4": {"cls": "NOT", "inputs": ["G3"],       "outputs": ["OUT_1"]},
    }
    assert compress(g).unchanged is True
