"""Wiring BETWEEN block symbols.

Recognising the box was only half the job.  The wires running to its pins were
never traced, so two pins driven by one signal looked independent and two
symbols wired together looked like two unrelated devices:

  * a JK symbol with T tied to both J and K -- a T flip-flop -- emitted
    independent J and K ports;
  * a master-slave pair emitted two disconnected flip-flops, losing the
    connection that makes it edge-triggered.

These tests drive graph_from_blocks with hand-written pin->net maps, so they
exercise the wiring rules without needing OCR or an image.
"""
import pytest

from hdl_gen.graph_to_hdl import generate_all
from pattern_engine.block_form import Block, graph_from_blocks


def _jk(name="JK FLIP FLOP", bbox=(0, 0, 10, 10)):
    return Block(cls="JKFF_BLOCK", bbox=bbox, name=name,
                 inputs=["J", "K", "CLK"], outputs=["Q", "Qbar"])


def test_unwired_blocks_are_unchanged():
    """Without a net map the old behaviour must be preserved exactly."""
    g = graph_from_blocks([_jk()])
    assert g["B1"]["inputs"] == ["J", "K", "CLK"]
    assert g["B1"]["outputs"] == ["Q", "Qbar"]


class TestSharedInput:
    """T tied to both J and K."""

    def setup_method(self):
        nets = {(0, "J"): 3, (0, "K"): 3, (0, "CLK"): 9, (0, "Q"): 4}
        self.g = graph_from_blocks([_jk()], nets)

    def test_two_pins_on_one_wire_become_one_signal(self):
        assert self.g["B1"]["inputs"] == ["J", "CLK"]

    def test_pin_map_still_resolves_the_merged_pin(self):
        """`inputs` is de-duplicated but the emitter asks for pins BY NAME, so
        the mapping has to survive -- otherwise K resolves positionally and
        picks up the clock."""
        assert self.g["B1"]["ports"]["inputs"] == {"J": "J", "K": "J",
                                                   "CLK": "CLK"}

    def test_emitted_hdl_toggles_like_a_t_flipflop(self):
        gi = {"J", "CLK"}
        v = generate_all(self.g, gi, {"Q", "Qbar"}, "t")["verilog_behavioral"]
        assert "case ({J, J})" in v
        assert "case ({J, CLK})" not in v      # the bug this guards


class TestMasterSlave:
    """Master Q drives slave J."""

    def setup_method(self):
        nets = {(0, "J"): 1, (0, "K"): 2, (0, "CLK"): 9, (0, "Q"): 3,
                (1, "J"): 3, (1, "K"): 5, (1, "CLK"): 9, (1, "Q"): 7}
        self.g = graph_from_blocks([_jk("master"), _jk("slave")], nets)

    def test_slave_input_references_the_master(self):
        assert "B1" in self.g["B2"]["inputs"]

    def test_masters_consumed_output_is_not_a_circuit_port(self):
        """The master's Q goes to the slave; it is an internal wire, not an
        output of the design."""
        assert "Q" not in self.g["B1"]["outputs"]

    def test_slave_output_is_still_a_port(self):
        assert "Q" in self.g["B2"]["outputs"]

    def test_shared_clock_is_one_signal(self):
        assert self.g["B1"]["ports"]["inputs"]["CLK"] == \
               self.g["B2"]["ports"]["inputs"]["CLK"]


def test_untraced_pin_keeps_its_own_name():
    """OCR misses small pin letters; a pin with no located label must stay a
    plain input rather than being dropped."""
    nets = {(0, "J"): 1}                      # K and CLK never located
    g = graph_from_blocks([_jk()], nets)
    assert set(g["B1"]["inputs"]) == {"J", "K", "CLK"}


def test_block_does_not_reference_itself():
    """A pin sharing a net with the block's OWN output is feedback inside the
    device, not an input from elsewhere."""
    nets = {(0, "J"): 4, (0, "Q"): 4, (0, "CLK"): 9}
    g = graph_from_blocks([_jk()], nets)
    assert "B1" not in g["B1"]["inputs"]
