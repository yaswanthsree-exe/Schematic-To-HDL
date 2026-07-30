from pattern_engine.library import load_library
from pattern_engine.matcher import candidates

LIB = {p.name: p for p in load_library()}
XOR = LIB["xor_and_or_not"]
SR = LIB["sr_latch_nor"]


def xor_graph(out="Y"):
    return {
        "G1": {"cls": "NOT", "inputs": ["A"],        "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["B"],        "outputs": []},
        "G3": {"cls": "AND", "inputs": ["A", "G2"],  "outputs": []},
        "G4": {"cls": "AND", "inputs": ["G1", "B"],  "outputs": []},
        "G5": {"cls": "OR",  "inputs": ["G3", "G4"], "outputs": [out]},
    }


def sr_graph():
    return {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": ["Qbar"]},
    }


def test_finds_the_xor():
    ms = candidates(xor_graph(), XOR)
    assert len(ms) == 1
    assert ms[0].nodes == frozenset({"G1", "G2", "G3", "G4", "G5"})


def test_xor_ports_bound_to_external_sources():
    """XOR is symmetric, so which external signal lands on port A vs port B is
    not determined by topology.  Assert the binding is complete and correct as
    a set, not the particular assignment."""
    m = candidates(xor_graph(), XOR)[0]
    assert set(m.inputs) == {"A", "B"}
    assert set(m.inputs.values()) == {"A", "B"}
    assert m.outputs == {"Y": "G5"}


def test_no_match_when_a_gate_class_differs():
    g = xor_graph()
    g["G5"]["cls"] = "AND"
    assert candidates(g, XOR) == []


def test_no_match_when_an_internal_signal_escapes():
    """G3's term is also read by an unrelated gate -> cannot compress."""
    g = xor_graph()
    g["G9"] = {"cls": "NOT", "inputs": ["G3"], "outputs": ["Z"]}
    assert candidates(g, XOR) == []


def test_no_match_when_an_internal_node_is_a_primary_output():
    g = xor_graph()
    g["G3"]["outputs"] = ["LEAK"]
    assert candidates(g, XOR) == []


def test_no_match_when_port_a_binds_two_different_sources():
    """NOT reads A but the AND reads C -> not an XOR of one signal."""
    g = xor_graph()
    g["G3"]["inputs"] = ["C", "G2"]
    assert candidates(g, XOR) == []


def test_no_match_when_two_ports_share_a_source():
    g = xor_graph()
    g["G2"]["inputs"] = ["A"]
    g["G4"]["inputs"] = ["G1", "A"]
    assert candidates(g, XOR) == []


def test_finds_the_sr_latch():
    """The cross-coupled NOR pair is structurally symmetric, so S-vs-R (and
    Q-vs-Qbar) cannot be told apart from topology alone.  Assert the binding
    is complete, not which way round it landed."""
    ms = candidates(sr_graph(), SR)
    assert len(ms) == 1
    assert ms[0].nodes == frozenset({"G1", "G2"})
    assert set(ms[0].inputs) == {"R", "S"}
    assert set(ms[0].inputs.values()) == {"R", "S"}


def test_no_sr_match_without_cross_coupling():
    g = {
        "G1": {"cls": "NOR", "inputs": ["R", "X"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "Y"], "outputs": ["Qbar"]},
    }
    assert candidates(g, SR) == []


def test_rejects_multi_output_macro_with_two_fanning_outputs():
    """Both Q and Qbar drive other gates -> cannot be expressed by one id."""
    g = sr_graph()
    g["G3"] = {"cls": "NOT", "inputs": ["G1"], "outputs": ["P"]}
    g["G4"] = {"cls": "NOT", "inputs": ["G2"], "outputs": ["N"]}
    assert candidates(g, SR) == []


def test_allows_multi_output_macro_with_one_fanning_output():
    g = sr_graph()
    g["G3"] = {"cls": "NOT", "inputs": ["G1"], "outputs": ["P"]}
    assert len(candidates(g, SR)) == 1


def test_result_is_deterministic():
    g = xor_graph()
    first = [sorted(m.nodes) for m in candidates(g, XOR)]
    for _ in range(5):
        assert [sorted(m.nodes) for m in candidates(g, XOR)] == first


def test_commutative_inputs_match_in_either_order():
    """Swapping an AND's input order must not prevent the match."""
    g = xor_graph()
    g["G3"]["inputs"] = ["G2", "A"]
    g["G5"]["inputs"] = ["G4", "G3"]
    assert len(candidates(g, XOR)) == 1
