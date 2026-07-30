from pattern_engine.evaluator import (MAX_TT_INPUTS, subgraph_is_acyclic,
                                      truth_table)


def _xor_graph():
    """XOR(A,B) = (A & ~B) | (~A & B) — five gates."""
    return {
        "G1": {"cls": "NOT", "inputs": ["A"],          "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["B"],          "outputs": []},
        "G3": {"cls": "AND", "inputs": ["A", "G2"],    "outputs": []},
        "G4": {"cls": "AND", "inputs": ["G1", "B"],    "outputs": []},
        "G5": {"cls": "OR",  "inputs": ["G3", "G4"],   "outputs": ["Y"]},
    }


ALL = {"G1", "G2", "G3", "G4", "G5"}
PORTS_IN = {"A": "A", "B": "B"}
PORTS_OUT = {"Y": "G5"}


def test_acyclic_subgraph_detected():
    assert subgraph_is_acyclic(_xor_graph(), ALL) is True


def test_cyclic_subgraph_detected():
    sr = {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": []},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": []},
    }
    assert subgraph_is_acyclic(sr, {"G1", "G2"}) is False


def test_xor_truth_table():
    assert truth_table(_xor_graph(), ALL, PORTS_IN, PORTS_OUT) == "0110"


def test_and_truth_table():
    g = {"G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": []}}
    assert truth_table(g, {"G1"}, {"A": "A", "B": "B"}, {"Y": "G1"}) == "0001"


def test_nand_truth_table():
    g = {"G1": {"cls": "NAND", "inputs": ["A", "B"], "outputs": []}}
    assert truth_table(g, {"G1"}, {"A": "A", "B": "B"}, {"Y": "G1"}) == "1110"


def test_not_truth_table():
    g = {"G1": {"cls": "NOT", "inputs": ["A"], "outputs": []}}
    assert truth_table(g, {"G1"}, {"A": "A"}, {"Y": "G1"}) == "10"


def test_input_ports_sorted_lexicographically_msb_first():
    """B is MSB if ports are (A,B)? No — A sorts first, so A is MSB."""
    g = {"G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": []},
         "G2": {"cls": "NOT", "inputs": ["A"],      "outputs": []}}
    # Y = ~A ignores B entirely -> A is MSB -> "1100"
    assert truth_table(g, {"G2"}, {"A": "A"}, {"Y": "G2"}) == "10"
    assert truth_table(g, {"G1", "G2"}, {"A": "A", "B": "B"},
                       {"Y": "G2"}) == "1100"


def test_multiple_outputs_concatenated_per_assignment():
    g = {"G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": []},
         "G2": {"cls": "OR",  "inputs": ["A", "B"], "outputs": []}}
    # per assignment: P (AND) then Q (OR), ports sorted -> P,Q
    assert truth_table(g, {"G1", "G2"}, {"A": "A", "B": "B"},
                       {"P": "G1", "Q": "G2"}) == "00" "01" "01" "11"


def test_returns_none_for_cyclic_subgraph():
    sr = {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": []},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": []},
    }
    assert truth_table(sr, {"G1", "G2"}, {"R": "R", "S": "S"},
                       {"Q": "G1"}) is None


def test_returns_none_for_undeclared_external_input():
    """C drives the subgraph but is not a declared port."""
    g = {"G1": {"cls": "AND", "inputs": ["A", "C"], "outputs": []}}
    assert truth_table(g, {"G1"}, {"A": "A"}, {"Y": "G1"}) is None


def test_returns_none_above_input_cap():
    ports = {f"P{i}": f"P{i}" for i in range(MAX_TT_INPUTS + 1)}
    g = {"G1": {"cls": "AND", "inputs": list(ports.values()), "outputs": []}}
    assert truth_table(g, {"G1"}, ports, {"Y": "G1"}) is None
