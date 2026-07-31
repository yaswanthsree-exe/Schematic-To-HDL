"""Master-slave flip-flop: a composite recognised from other macros.

Every other pattern is FLAT -- it lists raw gates -- but a master-slave device
is recognised in terms of two already-compressed latches.  That works because
the compressor runs to a fixpoint: the D latches are found in an early pass,
and the level-4 pattern then matches the macros they left behind.  This is the
hierarchy the design aimed at (primitives -> modules -> registers).
"""
from hdl_gen.graph_to_hdl import generate_all
from pattern_engine import compress


def master_slave_graph():
    """Two gated D latches in series on complementary clocks."""
    return {
        "NI": {"cls": "NOT",  "inputs": ["CLK"],       "outputs": []},
        "MI": {"cls": "NOT",  "inputs": ["D"],         "outputs": []},
        "M1": {"cls": "NAND", "inputs": ["D", "CLK"],  "outputs": []},
        "M2": {"cls": "NAND", "inputs": ["MI", "CLK"], "outputs": []},
        "M3": {"cls": "NAND", "inputs": ["M1", "M4"],  "outputs": []},
        "M4": {"cls": "NAND", "inputs": ["M2", "M3"],  "outputs": []},
        "SI": {"cls": "NOT",  "inputs": ["M3"],        "outputs": []},
        "S1": {"cls": "NAND", "inputs": ["M3", "NI"],  "outputs": []},
        "S2": {"cls": "NAND", "inputs": ["SI", "NI"],  "outputs": []},
        "S3": {"cls": "NAND", "inputs": ["S1", "S4"],  "outputs": ["Q"]},
        "S4": {"cls": "NAND", "inputs": ["S2", "S3"],  "outputs": ["Qbar"]},
    }


def test_recognised_as_a_single_dff():
    r = compress(master_slave_graph())
    assert [n["cls"] for n in r.graph.values()] == ["DFF"]


def test_built_hierarchically_from_two_latches():
    """The two D latches must be found first, then composed."""
    r = compress(master_slave_graph())
    kinds = [m.cls for m in r.matches]
    assert kinds.count("DLATCH") == 2
    assert kinds[-1] == "DFF"
    assert r.passes >= 2                     # composition needs a later pass


def test_absorbs_every_gate():
    r = compress(master_slave_graph())
    node = next(iter(r.graph.values()))
    assert node["inputs"] == ["D", "CLK"]
    assert node["outputs"] == ["Q", "Qbar"]


def test_outputs_are_not_duplicated():
    """Both ports name the SAME slave stage, which previously made the macro
    report that stage's outputs once per port."""
    r = compress(master_slave_graph())
    node = next(iter(r.graph.values()))
    assert node["outputs"] == list(dict.fromkeys(node["outputs"]))


def test_q_and_qbar_remain_distinct_signals():
    r = compress(master_slave_graph())
    v = generate_all(r.graph, {"D", "CLK"}, {"Q", "Qbar"}, "dff")["verilog_behavioral"]
    q = [l for l in v.splitlines() if l.strip().startswith("assign Q ")][0]
    qb = [l for l in v.splitlines() if l.strip().startswith("assign Qbar ")][0]
    assert q.split("=")[1].strip() != qb.split("=")[1].strip()


def test_bom_still_counts_the_real_gates():
    """Compression is a view, not a deletion: the physical parts list must
    still reflect the 11 gates actually drawn."""
    r = compress(master_slave_graph())
    bom = generate_all(r.graph, {"D", "CLK"}, {"Q", "Qbar"}, "dff")["ic_bom"]
    parts = {b["part"]: b["count"] for b in bom}
    assert parts.get("7400") == 8            # eight NANDs
    assert parts.get("7404") == 3            # three inverters


def test_same_clock_on_both_stages_is_not_a_flipflop():
    """Without the inverter the stages are transparent together -- that is a
    latch, and claiming edge-triggered behaviour would be wrong."""
    g = master_slave_graph()
    g["S1"]["inputs"] = ["M3", "CLK"]
    g["S2"]["inputs"] = ["SI", "CLK"]
    del g["NI"]
    assert not any(m.cls == "DFF" for m in compress(g).matches)
