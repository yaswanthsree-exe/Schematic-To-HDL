import pytest

from pattern_engine.library import Pattern, load_library, load_pattern


def test_library_loads_both_patterns():
    lib = load_library()
    names = [p.name for p in lib]
    assert "xor_and_or_not" in names
    assert "sr_latch_nor" in names


def test_library_is_sorted_by_level_then_name():
    lib = load_library()
    keys = [(p.level, p.name) for p in lib]
    assert keys == sorted(keys)


def test_xor_pattern_shape():
    lib = {p.name: p for p in load_library()}
    xor = lib["xor_and_or_not"]
    assert xor.cls == "XOR"
    assert xor.level == 1
    assert xor.node_map == {"n1": "NOT", "n2": "NOT",
                            "n3": "AND", "n4": "AND", "n5": "OR"}
    assert len(xor.edges) == 4
    assert xor.validator == "truth_table"
    assert xor.expected == "0110"
    assert xor.allow_shared_inputs is False


def test_xor_ports():
    lib = {p.name: p for p in load_library()}
    xor = lib["xor_and_or_not"]
    a = next(p for p in xor.inputs if p.name == "A")
    assert a.attach == (("n1", 0), ("n3", 0))
    assert [o.name for o in xor.outputs] == ["Y"]
    assert xor.outputs[0].node == "n5"


def test_sr_latch_pattern_shape():
    lib = {p.name: p for p in load_library()}
    sr = lib["sr_latch_nor"]
    assert sr.cls == "SRLATCH"
    assert sr.node_map == {"n1": "NOR", "n2": "NOR"}
    assert sr.validator == "sr_latch_nor"
    assert sr.expected is None
    assert {o.name for o in sr.outputs} == {"Q", "Qbar"}


def test_pattern_is_hashable_and_frozen():
    lib = load_library()
    assert len({p for p in lib}) == len(lib)
    with pytest.raises(Exception):
        lib[0].name = "changed"


def test_rejects_edge_referencing_unknown_node(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("""
    {"name":"bad","cls":"X","level":1,
     "nodes":{"n1":"AND"},
     "edges":[{"from":"n1","to":"nope","pins":[0]}],
     "ports":{"inputs":[],"outputs":[{"name":"Y","node":"n1"}]},
     "validator":"truth_table","expected":"01"}
    """, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown node"):
        load_pattern(str(bad))


def test_rejects_port_referencing_unknown_node(tmp_path):
    bad = tmp_path / "bad2.json"
    bad.write_text("""
    {"name":"bad2","cls":"X","level":1,
     "nodes":{"n1":"AND"},
     "edges":[],
     "ports":{"inputs":[{"name":"A","attach":[["nope",0]]}],
              "outputs":[{"name":"Y","node":"n1"}]},
     "validator":"truth_table","expected":"01"}
    """, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown node"):
        load_pattern(str(bad))


def test_rejects_missing_required_field(tmp_path):
    bad = tmp_path / "bad3.json"
    bad.write_text('{"name":"bad3","cls":"X"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_pattern(str(bad))
