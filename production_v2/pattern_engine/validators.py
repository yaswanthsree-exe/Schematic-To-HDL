"""Validator registry.

Structural matching alone accepts subgraphs that look right but compute
something else.  Every candidate must pass its pattern's validator before it
is compressed.
"""
from __future__ import annotations

from typing import Callable, Dict

from .evaluator import truth_table
from .graph_adapter import GateGraph
from .matcher import Match

Validator = Callable[[GateGraph, Match], bool]

VALIDATORS: Dict[str, Validator] = {}


def register(name: str) -> Callable[[Validator], Validator]:
    def deco(fn: Validator) -> Validator:
        VALIDATORS[name] = fn
        return fn
    return deco


@register("truth_table")
def _truth_table(graph: GateGraph, match: Match) -> bool:
    """Exhaustive combinational equivalence — decisive, and independent of how
    the circuit happens to be drawn."""
    expected = match.pattern.expected
    if not expected:
        return False
    actual = truth_table(graph, set(match.nodes), match.inputs, match.outputs)
    return actual is not None and actual == expected


@register("sr_latch_nor")
def _sr_latch_nor(graph: GateGraph, match: Match) -> bool:
    """Functional constraints for a cross-coupled NOR SR latch."""
    ids = sorted(match.nodes)
    if len(ids) != 2:
        return False
    if any(graph[i]["cls"] != "NOR" for i in ids):
        return False
    a, b = ids
    if a not in graph[b].get("inputs", ()):
        return False
    if b not in graph[a].get("inputs", ()):
        return False
    if len(set(match.inputs.values())) != 2:
        return False
    if len(match.outputs) != 2:
        return False
    return True


def validate(graph: GateGraph, match: Match) -> bool:
    """Run the pattern's validator.  Raises KeyError if it is not registered —
    a typo in a pattern file must fail loudly, not silently accept."""
    return VALIDATORS[match.pattern.validator](graph, match)
