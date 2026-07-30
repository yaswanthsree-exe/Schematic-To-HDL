"""Combinational simulator for matched subgraphs.

Used by the truth_table validator to confirm that a structurally-matching
subgraph actually computes the pattern's function.  This is also the seed of
the future verification engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from hdl_gen.graph_to_hdl import GATE_SEMANTICS

from .graph_adapter import GateGraph

#: Exhaustive simulation is 2**n; refuse anything larger than this.
MAX_TT_INPUTS = 12


def subgraph_is_acyclic(graph: GateGraph, nodes: Set[str]) -> bool:
    """True when the induced subgraph over *nodes* has no directed cycle."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {n: WHITE for n in nodes}

    def visit(n: str) -> bool:
        colour[n] = GREY
        for src in graph[n].get("inputs", ()):
            if src not in colour:
                continue
            if colour[src] == GREY:
                return False
            if colour[src] == WHITE and not visit(src):
                return False
        colour[n] = BLACK
        return True

    return all(colour[n] != WHITE or visit(n) for n in nodes)


def _topo(graph: GateGraph, nodes: Set[str]) -> List[str]:
    """Producers before consumers, over the induced subgraph."""
    order: List[str] = []
    seen: Set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        for src in graph[n].get("inputs", ()):
            if src in nodes:
                visit(src)
        order.append(n)

    for n in sorted(nodes):
        visit(n)
    return order


def _eval_gate(cls: str, args: List[int]) -> int:
    op, inv, unary = GATE_SEMANTICS[cls]
    if unary:
        value = args[0]
        return 1 - value if inv else value
    value = args[0]
    for a in args[1:]:
        if op == "&":
            value &= a
        elif op == "|":
            value |= a
        elif op == "^":
            value ^= a
    return 1 - value if inv else value


def truth_table(graph: GateGraph,
                nodes: Set[str],
                inputs: Dict[str, str],
                outputs: Dict[str, str]) -> Optional[str]:
    """Exhaustively simulate the induced subgraph.

    *inputs* maps port name -> external source id, *outputs* maps port name ->
    internal node id.  Ports are ordered lexicographically, first port most
    significant.  Per assignment the output bits are appended in sorted port
    order.

    Returns None (meaning "not evaluable, reject the match") when the subgraph
    is cyclic, when a node is driven by an external signal that is not a
    declared port, when a gate class has no known semantics, or when there are
    more than MAX_TT_INPUTS ports.
    """
    if len(inputs) > MAX_TT_INPUTS:
        return None
    if not subgraph_is_acyclic(graph, nodes):
        return None

    in_ports = sorted(inputs)
    out_ports = sorted(outputs)
    source_of = {inputs[p]: p for p in in_ports}

    for n in nodes:
        if graph[n]["cls"] not in GATE_SEMANTICS:
            return None
        for src in graph[n].get("inputs", ()):
            if src not in nodes and src not in source_of:
                return None

    order = _topo(graph, nodes)
    width = len(in_ports)
    bits: List[str] = []

    for vector in range(2 ** width):
        value: Dict[str, int] = {}
        for i, port in enumerate(in_ports):
            value[inputs[port]] = (vector >> (width - 1 - i)) & 1
        for n in order:
            args = [value[src] for src in graph[n].get("inputs", ())]
            if not args:
                return None
            value[n] = _eval_gate(graph[n]["cls"], args)
        bits.extend(str(value[outputs[p]]) for p in out_ports)

    return "".join(bits)
