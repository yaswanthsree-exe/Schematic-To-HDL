"""Guards that the engine leaves alone the circuits the pipeline already
handles correctly.  A false positive here would be worse than no engine."""
from hdl_gen.graph_to_hdl import generate_all
from pattern_engine import compress


def benchmark_20_combinational():
    """The verified ground truth: OUT_1 = ~((A & B) | ~C).

    Mirrors exactly what predict.py produces for 20-combinational_circuit.png.
    """
    return {
        "G1": {"cls": "AND", "inputs": ["A", "B"],   "outputs": []},
        "G2": {"cls": "NOT", "inputs": ["C"],        "outputs": []},
        "G3": {"cls": "OR",  "inputs": ["G1", "G2"], "outputs": []},
        "G4": {"cls": "NOT", "inputs": ["G3"],       "outputs": ["OUT_1"]},
    }


def test_benchmark_graph_is_not_compressed():
    g = benchmark_20_combinational()
    r = compress(g)
    assert r.unchanged is True
    assert r.graph == g


def test_benchmark_hdl_is_byte_identical_after_engine():
    g = benchmark_20_combinational()
    before = generate_all(g, {"A", "B", "C"}, {"OUT_1"}, "bench")
    after = generate_all(compress(g).graph, {"A", "B", "C"}, {"OUT_1"}, "bench")
    assert after["verilog_structural"] == before["verilog_structural"]
    assert after["ic_bom"] == before["ic_bom"]


def test_half_adder_is_not_compressed():
    """Sum = A^B, Carry = A&B — an XOR primitive, not an XOR built from gates."""
    g = {
        "G1": {"cls": "XOR", "inputs": ["A", "B"], "outputs": ["Sum"]},
        "G2": {"cls": "AND", "inputs": ["A", "B"], "outputs": ["Carry"]},
    }
    assert compress(g).unchanged is True


def test_plain_or_chain_is_not_compressed():
    g = {
        "G1": {"cls": "OR",  "inputs": ["A", "B"],  "outputs": []},
        "G2": {"cls": "OR",  "inputs": ["G1", "C"], "outputs": ["Y"]},
    }
    assert compress(g).unchanged is True


def test_two_uncoupled_nors_are_not_an_sr_latch():
    g = {
        "G1": {"cls": "NOR", "inputs": ["A", "B"], "outputs": ["P"]},
        "G2": {"cls": "NOR", "inputs": ["C", "D"], "outputs": ["Q"]},
    }
    assert compress(g).unchanged is True


def test_empty_graph_is_safe():
    r = compress({})
    assert r.unchanged is True
    assert r.graph == {}
