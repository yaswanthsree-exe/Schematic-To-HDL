"""Repeated runs must produce identical results.

A symmetric pattern has several valid isomorphisms over the same gates -- a
cross-coupled pair maps either way round -- and VF2 does not promise a stable
order between them.  Keeping whichever arrived first made a JK emit
`case ({J, K})` on some runs and `case ({K, J})` on others, so regenerating
HDL from an unchanged schematic produced a different file.  The matcher now
picks a canonical isomorphism per gate set.

These run the pipeline repeatedly IN ONE PROCESS.  Cross-process stability
additionally depends on PYTHONHASHSEED, which is why the canonical choice is
made from sorted data rather than from iteration order.
"""
from hdl_gen.graph_to_hdl import generate_all
from pattern_engine import compress

REPEATS = 8


def jk_graph():
    return {
        "G1": {"cls": "NAND", "inputs": ["J", "CLK", "G4"], "outputs": []},
        "G2": {"cls": "NAND", "inputs": ["K", "CLK", "G3"], "outputs": []},
        "G3": {"cls": "NAND", "inputs": ["G1", "G4"],       "outputs": ["Q"]},
        "G4": {"cls": "NAND", "inputs": ["G2", "G3"],       "outputs": ["Qbar"]},
    }


def sr_graph():
    return {
        "G1": {"cls": "NOR", "inputs": ["R", "G2"], "outputs": ["Q"]},
        "G2": {"cls": "NOR", "inputs": ["S", "G1"], "outputs": ["Qbar"]},
    }


def _strip_timestamp(hdl: str) -> str:
    """Drop the generated-on line in either language.

    Verilog comments with `//`, VHDL with `--`; stripping only the Verilog form
    left the VHDL timestamp in and made this test fail whenever two calls
    straddled a second boundary.
    """
    return "\n".join(l for l in hdl.splitlines()
                     if not (l.startswith("// 2") or l.startswith("-- 2")))


def test_jk_port_binding_is_stable():
    first = compress(jk_graph()).graph
    macro = next(v for v in first.values() if v["cls"] == "JKFF")
    for _ in range(REPEATS):
        again = compress(jk_graph()).graph
        m2 = next(v for v in again.values() if v["cls"] == "JKFF")
        assert m2["ports"] == macro["ports"]
        assert m2["inputs"] == macro["inputs"]


def test_sr_port_binding_is_stable():
    first = compress(sr_graph()).graph
    macro = next(v for v in first.values() if v["cls"] == "SRLATCH")
    for _ in range(REPEATS):
        m2 = next(v for v in compress(sr_graph()).graph.values()
                  if v["cls"] == "SRLATCH")
        assert m2["ports"] == macro["ports"]


def test_generated_verilog_is_reproducible():
    def gen():
        return _strip_timestamp(generate_all(
            compress(jk_graph()).graph, {"J", "K", "CLK"},
            {"Q", "Qbar"}, "jk")["verilog_behavioral"])
    first = gen()
    for _ in range(REPEATS):
        assert gen() == first


def test_generated_vhdl_is_reproducible():
    def gen():
        return _strip_timestamp(generate_all(
            compress(jk_graph()).graph, {"J", "K", "CLK"},
            {"Q", "Qbar"}, "jk")["vhdl"])
    first = gen()
    for _ in range(REPEATS):
        assert gen() == first
