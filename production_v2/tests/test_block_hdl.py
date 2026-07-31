"""HDL emission for block-form devices.

A block symbol carries no gate-level detail -- the schematic never drew any --
so these nodes have neither `children` nor a `ports` map.  Both of the
mechanisms the compressed macros rely on are therefore absent, and each one
needed its own fallback:

  * operand lookup by port NAME, because there is no ports map; without it an
    SR block was clocked on S rather than CLK.
  * output suffixes read from the node's own `outputs`, because there are no
    children to walk; without it Q and Qbar both resolved to the base wire.
"""
import pytest

from hdl_gen.graph_to_hdl import generate_all

CASES = [
    ("SRFF_BLOCK", ["S", "R", "CLK"], "74279"),
    ("JKFF_BLOCK", ["J", "K", "CLK"], "7476"),
    ("DFF_BLOCK",  ["D", "CLK"],      "7474"),
    ("TFF_BLOCK",  ["T", "CLK"],      "7476"),
]


def _gen(cls, ins, name="blk"):
    g = {"B1": {"cls": cls, "inputs": list(ins), "outputs": ["Q", "Qbar"]}}
    return generate_all(g, set(ins), {"Q", "Qbar"}, name)


@pytest.mark.parametrize("cls,ins,_part", CASES)
def test_clocked_on_clk_not_a_data_pin(cls, ins, _part):
    v = _gen(cls, ins)["verilog_behavioral"]
    assert "posedge CLK" in v
    for data in ins:
        if data != "CLK":
            assert f"@(posedge {data})" not in v


@pytest.mark.parametrize("cls,ins,_part", CASES)
def test_q_and_qbar_are_distinct(cls, ins, _part):
    v = _gen(cls, ins)["verilog_behavioral"]
    q = [l for l in v.splitlines() if l.strip().startswith("assign Q ")][0]
    qb = [l for l in v.splitlines() if l.strip().startswith("assign Qbar ")][0]
    assert q.split("=")[1].strip() != qb.split("=")[1].strip()


@pytest.mark.parametrize("cls,ins,part", CASES)
def test_bom_lists_the_real_sequential_part(cls, ins, part):
    """A block device has nothing to flatten, so it would vanish from the bill
    of materials unless costed as the part it represents."""
    bom = _gen(cls, ins)["ic_bom"]
    assert any(b["part"] == part for b in bom), bom


@pytest.mark.parametrize("cls,ins,_part", CASES)
def test_both_languages_emitted(cls, ins, _part):
    out = _gen(cls, ins)
    assert out["verilog_behavioral"].strip()
    assert out["vhdl"].strip()


@pytest.mark.parametrize("cls,ins,_part", CASES)
def test_structural_instantiates_a_black_box(cls, ins, _part):
    """Structural Verilog must not invent a gate decomposition for a symbol
    whose internals were never drawn."""
    s = _gen(cls, ins)["verilog_structural"]
    assert cls in s
    assert "not decomposed" in s


def test_async_pins_only_appear_when_the_symbol_had_them():
    with_async = _gen("SRFF_BLOCK", ["S", "R", "CLK", "PR", "CLR"])["verilog_behavioral"]
    assert "CLR" in with_async and "PR" in with_async
    plain = _gen("SRFF_BLOCK", ["S", "R", "CLK"])["verilog_behavioral"]
    assert "CLR" not in plain and "posedge PR" not in plain


def test_jk_block_toggles():
    v = _gen("JKFF_BLOCK", ["J", "K", "CLK"])["verilog_behavioral"]
    assert "2'b11" in v and "~" in [l for l in v.splitlines() if "2'b11" in l][0]


def test_t_block_toggles_on_t():
    v = _gen("TFF_BLOCK", ["T", "CLK"])["verilog_behavioral"]
    assert "if (T)" in v
