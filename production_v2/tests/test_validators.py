import pytest

from pattern_engine.library import load_library
from pattern_engine.matcher import candidates
from pattern_engine.validators import VALIDATORS, validate

LIB = {p.name: p for p in load_library()}
XOR = LIB["xor_and_or_not"]
SR = LIB["sr_latch_nor"]


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


def test_both_validators_registered():
    assert set(VALIDATORS) == {"truth_table", "sr_latch_nor"}


def test_real_xor_passes_truth_table():
    g = xor_graph()
    m = candidates(g, XOR)[0]
    assert validate(g, m) is True


def test_truth_table_mismatch_is_rejected():
    """The validator must actually compare, not rubber-stamp whatever VF2 found.

    Note: for XOR specifically, port binding is already so constraining that a
    structural match plus a successful binding implies the function.  The truth
    table is defence in depth here, and becomes load-bearing for looser patterns
    added later — so it must be proven to reject.
    """
    from dataclasses import replace
    g = xor_graph()
    m = candidates(g, XOR)[0]
    wrong = replace(m.pattern, expected="0000")
    m2 = type(m)(pattern=wrong, mapping=m.mapping,
                 inputs=m.inputs, outputs=m.outputs)
    assert validate(g, m2) is False


def test_truth_table_validator_rejects_pattern_without_expected():
    from dataclasses import replace
    g = xor_graph()
    m = candidates(g, XOR)[0]
    empty = replace(m.pattern, expected=None)
    m2 = type(m)(pattern=empty, mapping=m.mapping,
                 inputs=m.inputs, outputs=m.outputs)
    assert validate(g, m2) is False


def test_real_sr_latch_passes_predicate():
    g = sr_graph()
    m = candidates(g, SR)[0]
    assert validate(g, m) is True


def test_truth_table_validator_rejects_cyclic_subgraph():
    """A cyclic subgraph can never satisfy the truth_table validator."""
    from dataclasses import replace
    g = sr_graph()
    m = candidates(g, SR)[0]
    forced = replace(m.pattern, validator="truth_table", expected="0110")
    m2 = type(m)(pattern=forced, mapping=m.mapping,
                 inputs=m.inputs, outputs=m.outputs)
    assert validate(g, m2) is False


def test_unknown_validator_raises():
    from dataclasses import replace
    g = xor_graph()
    m = candidates(g, XOR)[0]
    bogus = replace(m.pattern, validator="does_not_exist")
    m2 = type(m)(pattern=bogus, mapping=m.mapping,
                 inputs=m.inputs, outputs=m.outputs)
    with pytest.raises(KeyError):
        validate(g, m2)
