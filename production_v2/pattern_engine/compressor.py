"""Iterative hierarchical compression.

Smallest pattern level first, greedy and non-overlapping, to a fixpoint.  Each
accepted match becomes a macro node that retains the gates it replaced, so the
gate-level view (structural Verilog, IC BOM) stays recoverable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .graph_adapter import GateGraph
from .library import Pattern, load_library
from .matcher import Match, candidates
from .validators import validate

#: Upper bound on rewrite passes; guards against a pathological loop.
MAX_PASSES = 10


@dataclass
class MatchRecord:
    pattern:  str
    cls:      str
    level:    int
    macro_id: str
    absorbed: List[str]


@dataclass
class CompressionResult:
    graph:     GateGraph
    matches:   List[MatchRecord] = field(default_factory=list)
    passes:    int = 0
    unchanged: bool = True


def _next_macro_id(graph: GateGraph, counter: int) -> Tuple[str, int]:
    while f"M{counter}" in graph:
        counter += 1
    return f"M{counter}", counter + 1


def _apply(graph: GateGraph, match: Match, macro_id: str) -> GateGraph:
    """Return a new graph with *match* replaced by a macro node."""
    absorbed: Set[str] = set(match.nodes)
    pattern = match.pattern

    children = {gid: copy.deepcopy(graph[gid]) for gid in sorted(absorbed)}

    primary_outputs: List[str] = []
    for port in pattern.outputs:
        primary_outputs.extend(graph[match.outputs[port.name]].get("outputs", ()))

    new: GateGraph = {}
    for gid, node in graph.items():
        if gid in absorbed:
            continue
        clone = copy.deepcopy(node)
        clone["inputs"] = [macro_id if s in absorbed else s
                           for s in clone.get("inputs", ())]
        new[gid] = clone

    new[macro_id] = {
        "cls":      pattern.cls,
        "inputs":   [match.inputs[p.name] for p in pattern.inputs],
        "outputs":  primary_outputs,
        "children": children,
        "pattern":  pattern.name,
        "level":    pattern.level,
        "ports": {
            "inputs":  dict(match.inputs),
            "outputs": dict(match.outputs),
        },
    }
    return new


def compress(graph: GateGraph,
             library: Optional[List[Pattern]] = None,
             max_passes: int = MAX_PASSES) -> CompressionResult:
    """Recognize and compress functional motifs.  The input graph is not
    mutated; the result holds a new graph."""
    if library is None:
        library = load_library()

    current: GateGraph = copy.deepcopy(graph)
    records: List[MatchRecord] = []
    counter = 1
    passes = 0

    for _ in range(max_passes):
        passes += 1
        changed = False
        for pattern in sorted(library, key=lambda p: (p.level, p.name)):
            while True:
                accepted: Optional[Match] = None
                for match in candidates(current, pattern):
                    if validate(current, match):
                        accepted = match
                        break
                if accepted is None:
                    break
                macro_id, counter = _next_macro_id(current, counter)
                absorbed = sorted(accepted.nodes)
                current = _apply(current, accepted, macro_id)
                records.append(MatchRecord(
                    pattern=pattern.name, cls=pattern.cls, level=pattern.level,
                    macro_id=macro_id, absorbed=absorbed,
                ))
                changed = True
        if not changed:
            break

    return CompressionResult(graph=current, matches=records,
                             passes=passes, unchanged=not records)
