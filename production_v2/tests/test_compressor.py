import copy

from pattern_engine import CompressionResult, compress


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


def test_xor_compresses_to_one_macro():
    r = compress(xor_graph())
    assert isinstance(r, CompressionResult)
    assert len(r.graph) == 1
    macro = next(iter(r.graph.values()))
    assert macro["cls"] == "XOR"
    assert sorted(macro["inputs"]) == ["A", "B"]
    assert macro["outputs"] == ["Y"]


def test_macro_retains_children():
    r = compress(xor_graph())
    macro = next(iter(r.graph.values()))
    assert set(macro["children"]) == {"G1", "G2", "G3", "G4", "G5"}
    assert macro["children"]["G5"]["cls"] == "OR"


def test_macro_carries_provenance():
    r = compress(xor_graph())
    macro = next(iter(r.graph.values()))
    assert macro["pattern"] == "xor_and_or_not"
    assert macro["level"] == 1


def test_match_records_describe_what_happened():
    r = compress(xor_graph())
    assert len(r.matches) == 1
    rec = r.matches[0]
    assert rec.pattern == "xor_and_or_not"
    assert rec.cls == "XOR"
    assert sorted(rec.absorbed) == ["G1", "G2", "G3", "G4", "G5"]
    assert rec.macro_id in r.graph


def test_sr_latch_compresses():
    r = compress(sr_graph())
    assert len(r.graph) == 1
    macro = next(iter(r.graph.values()))
    assert macro["cls"] == "SRLATCH"
    assert sorted(macro["inputs"]) == ["R", "S"]
    assert sorted(macro["outputs"]) == ["Q", "Qbar"]


def test_consumers_are_rewired_to_the_macro():
    g = xor_graph()
    g["G5"]["outputs"] = []
    g["G6"] = {"cls": "NOT", "inputs": ["G5"], "outputs": ["Z"]}
    r = compress(g)
    assert len(r.graph) == 2
    macro_id = r.matches[0].macro_id
    assert r.graph["G6"]["inputs"] == [macro_id]


def test_input_graph_is_not_mutated():
    g = xor_graph()
    before = copy.deepcopy(g)
    compress(g)
    assert g == before


def test_no_match_leaves_graph_unchanged():
    g = {"G1": {"cls": "AND", "inputs": ["A", "B"], "outputs": ["Y"]}}
    r = compress(g)
    assert r.unchanged is True
    assert r.matches == []
    assert r.graph == g


def test_compression_is_idempotent():
    once = compress(xor_graph())
    twice = compress(once.graph)
    assert twice.unchanged is True
    assert twice.graph == once.graph


def test_terminates_and_reports_passes():
    r = compress(xor_graph())
    assert 1 <= r.passes <= 10


def test_macro_ids_do_not_collide_with_existing_ids():
    g = xor_graph()
    g["M1"] = {"cls": "NOT", "inputs": ["Q"], "outputs": ["W"]}
    r = compress(g)
    assert "M1" in r.graph
    assert r.graph["M1"]["cls"] == "NOT"
    assert r.matches[0].macro_id != "M1"


def test_two_independent_xors_both_compress():
    """Two disjoint XORs, written out explicitly rather than generated."""
    g = {
        # XOR #1 over A, B
        "G1": {"cls": "NOT", "inputs": ["A"],        "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["B"],        "outputs": []},
        "G3": {"cls": "AND", "inputs": ["A", "G2"],  "outputs": []},
        "G4": {"cls": "AND", "inputs": ["G1", "B"],  "outputs": []},
        "G5": {"cls": "OR",  "inputs": ["G3", "G4"], "outputs": ["Y1"]},
        # XOR #2 over C, D
        "H1": {"cls": "NOT", "inputs": ["C"],        "outputs": []},
        "H2": {"cls": "NOT", "inputs": ["D"],        "outputs": []},
        "H3": {"cls": "AND", "inputs": ["C", "H2"],  "outputs": []},
        "H4": {"cls": "AND", "inputs": ["H1", "D"],  "outputs": []},
        "H5": {"cls": "OR",  "inputs": ["H3", "H4"], "outputs": ["Y2"]},
    }
    r = compress(g)
    assert len(r.matches) == 2
    assert all(m.cls == "XOR" for m in r.matches)
    assert len(r.graph) == 2
    assert {tuple(sorted(m.absorbed)) for m in r.matches} == {
        ("G1", "G2", "G3", "G4", "G5"),
        ("H1", "H2", "H3", "H4", "H5"),
    }
