"""Latches must not be emitted as flip-flops.

A latch is transparent for as long as its enable is asserted; a flip-flop
samples on a clock edge.  Both were mapped onto the same block class, so a box
clearly labelled "D LATCH" produced

    always @(posedge CLK) w_G1_r <= D;

which describes different hardware, with no warning.  These tests pin the
distinction at every layer it has to survive: classification, pin naming, HDL
sensitivity, and the bill of materials.
"""
import pytest

from hdl_gen.graph_to_hdl import generate_all
from pattern_engine.block_form import classify_device, pins_for


def _emit(cls, ins):
    g = {"B1": {"cls": cls, "inputs": list(ins), "outputs": ["Q", "Qbar"]}}
    return generate_all(g, set(ins), {"Q", "Qbar"}, cls.lower())


class TestClassification:
    def test_d_latch_is_level_sensitive_class(self):
        assert classify_device(["D", "LATCH"])[0] == "DLATCH_BLOCK"

    def test_sr_latch_is_level_sensitive_class(self):
        assert classify_device(["SR", "LATCH"])[0] == "SRLATCH_BLOCK"

    def test_d_flipflop_stays_edge_triggered(self):
        assert classify_device(["D", "FLIP", "FLOP"])[0] == "DFF_BLOCK"

    def test_jk_and_t_have_no_latch_form(self):
        """Both depend on the previous state, which a transparent latch cannot
        hold, so 'JK latch' still means the edge-triggered device."""
        assert classify_device(["JK", "LATCH"])[0] == "JKFF_BLOCK"
        assert classify_device(["T", "LATCH"])[0] == "TFF_BLOCK"


class TestPinNaming:
    def test_latch_pin_is_an_enable_not_a_clock(self):
        ins, _outs = pins_for("DLATCH_BLOCK")
        assert "EN" in ins and "CLK" not in ins

    def test_flipflop_pin_is_a_clock(self):
        ins, _outs = pins_for("DFF_BLOCK")
        assert "CLK" in ins and "EN" not in ins


class TestEmission:
    @pytest.mark.parametrize("cls,ins", [("DLATCH_BLOCK", ["D", "EN"]),
                                         ("SRLATCH_BLOCK", ["S", "R", "EN"])])
    def test_latch_is_not_edge_triggered(self, cls, ins):
        v = _emit(cls, ins)["verilog_behavioral"]
        assert "posedge" not in v
        assert "always @(*)" in v

    @pytest.mark.parametrize("cls,ins", [("DLATCH_BLOCK", ["D", "EN"]),
                                         ("SRLATCH_BLOCK", ["S", "R", "EN"])])
    def test_latch_is_transparent_only_while_enabled(self, cls, ins):
        v = _emit(cls, ins)["verilog_behavioral"]
        assert "if (EN)" in v

    @pytest.mark.parametrize("cls,ins", [("DFF_BLOCK", ["D", "CLK"]),
                                         ("JKFF_BLOCK", ["J", "K", "CLK"]),
                                         ("TFF_BLOCK", ["T", "CLK"])])
    def test_flipflop_is_edge_triggered(self, cls, ins):
        assert "posedge CLK" in _emit(cls, ins)["verilog_behavioral"]

    @pytest.mark.parametrize("cls,ins", [("DLATCH_BLOCK", ["D", "EN"]),
                                         ("SRLATCH_BLOCK", ["S", "R", "EN"])])
    def test_both_languages_still_emitted(self, cls, ins):
        out = _emit(cls, ins)
        assert out["verilog_behavioral"].strip() and out["vhdl"].strip()


class TestBillOfMaterials:
    def test_d_latch_costs_a_latch_part_not_a_flipflop(self):
        parts = {b["part"] for b in _emit("DLATCH_BLOCK", ["D", "EN"])["ic_bom"]}
        assert "7475" in parts          # quad D latch
        assert "7474" not in parts      # dual D flip-flop

    def test_d_flipflop_still_costs_a_flipflop_part(self):
        parts = {b["part"] for b in _emit("DFF_BLOCK", ["D", "CLK"])["ic_bom"]}
        assert "7474" in parts
