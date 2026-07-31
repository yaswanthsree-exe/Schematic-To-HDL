"""Identifying a block device when the box carries no device NAME.

Textbook symbols frequently label only the pins and mark the clock with a
triangle instead of the word CLK, so there is nothing to read but J/K/Q or
T/Q/Q'.  Measured on real symbols: a T flip-flop OCR'd to just ['T', "Q'"] and
a JK to ['J', 'K', 'CLK', 'Q'] -- decisive pin sets, but no name at all, so
name-based classification returned nothing for both.

Also covers the "D-type flip-flop" spelling, which normalises to DTYPE and so
never matched the bare "D" key.
"""
import pytest

from pattern_engine.block_form import classify_by_pins, classify_device


class TestPinSignature:
    def test_jk_from_pins(self):
        assert classify_by_pins(["J", "K", "CLK", "Q"])[0] == "JKFF_BLOCK"

    def test_t_from_pins_without_a_clock_label(self):
        """The clock is drawn as a triangle, so CLK is never written."""
        assert classify_by_pins(["T", "Q'"])[0] == "TFF_BLOCK"

    def test_d_from_pins(self):
        assert classify_by_pins(["D", "CLK", "Q", "Qbar"])[0] == "DFF_BLOCK"

    def test_sr_from_pins(self):
        assert classify_by_pins(["S", "R", "CLK", "Q"])[0] == "SRFF_BLOCK"

    def test_jk_wins_over_a_bare_letter(self):
        """J and K present must not be read as a D or T device."""
        assert classify_by_pins(["J", "K", "D", "Q"])[0] == "JKFF_BLOCK"

    def test_requires_a_q_output(self):
        """Without a Q-like pin this is just a box with letters in it."""
        assert classify_by_pins(["J", "K", "CLK"]) is None
        assert classify_by_pins(["HELLO", "WORLD"]) is None

    def test_unknown_pin_set_is_rejected(self):
        assert classify_by_pins(["X", "Y", "Q"]) is None

    @pytest.mark.parametrize("qtoken", ["Q", "Q'", "QBAR", "QN"])
    def test_q_spellings_accepted(self, qtoken):
        assert classify_by_pins(["T", qtoken]) is not None


class TestDeviceName:
    def test_d_type_hyphenated(self):
        assert classify_device(["D-type", "flip-flop"])[0] == "DFF_BLOCK"

    def test_t_type_hyphenated(self):
        assert classify_device(["T-type", "flip-flop"])[0] == "TFF_BLOCK"

    def test_plain_name_still_works(self):
        assert classify_device(["SR", "FLIP", "FLOP"])[0] == "SRFF_BLOCK"

    def test_latch_gets_a_level_sensitive_class(self):
        cls, name = classify_device(["D", "LATCH"])
        assert cls == "DLATCH_BLOCK" and "LATCH" in name

    def test_jk_latch_stays_edge_triggered(self):
        """JK and T depend on the previous state, which a transparent latch
        cannot hold, so there is no level-sensitive form to fall back to."""
        assert classify_device(["JK", "LATCH"])[0] == "JKFF_BLOCK"

    def test_pin_letters_alone_are_not_a_device_name(self):
        """An S and an R floating in a box are pins, not a name -- the name
        path must stay strict, since the pin path handles that case and
        requires a Q to do so."""
        assert classify_device(["S", "R"]) is None
