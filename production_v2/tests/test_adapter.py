from pattern_engine.graph_adapter import COMMUTATIVE, INPUT_CLS, to_nx


def _and_graph():
    """G1 = AND(A, B), driving primary output Y."""
    return {"G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": ["Y"]}}


def test_gates_become_nodes_with_cls():
    g = to_nx(_and_graph())
    assert g.nodes["G1"]["cls"] == "AND"


def test_primary_inputs_become_input_nodes():
    g = to_nx(_and_graph())
    assert g.nodes["A"]["cls"] == INPUT_CLS
    assert g.nodes["B"]["cls"] == INPUT_CLS


def test_edges_carry_pin_positions():
    g = to_nx(_and_graph())
    assert g.edges["A", "G1"]["pins"] == (0,)
    assert g.edges["B", "G1"]["pins"] == (1,)


def test_edges_carry_destination_class():
    g = to_nx(_and_graph())
    assert g.edges["A", "G1"]["dst_cls"] == "AND"


def test_repeated_source_collapses_to_one_edge_with_both_pins():
    """AND(x, x) must not need a MultiDiGraph."""
    g = to_nx({"G1": {"cls": "AND", "inputs": ["A", "A"], "outputs": []}})
    assert g.edges["A", "G1"]["pins"] == (0, 1)
    assert g.number_of_edges() == 1


def test_outputs_are_preserved_on_the_node():
    g = to_nx(_and_graph())
    assert g.nodes["G1"]["outputs"] == ("Y",)


def test_gate_to_gate_edge():
    graph = {
        "G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["G1"], "outputs": ["Y"]},
    }
    g = to_nx(graph)
    assert g.has_edge("G1", "G2")
    assert g.edges["G1", "G2"]["pins"] == (0,)
    assert "G1" not in [n for n, d in g.nodes(data=True) if d["cls"] == INPUT_CLS]


def test_commutative_set_contents():
    assert COMMUTATIVE == {"AND", "OR", "XOR", "NAND", "NOR", "XNOR"}
