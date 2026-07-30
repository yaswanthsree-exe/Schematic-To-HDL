"""Gate-graph dict -> networkx.DiGraph projection.

The gate graph produced by predict.build_gate_graph is:

    graph[gate_id] = {"cls": str, "inputs": [...], "outputs": [...]}

where each element of "inputs" is either another gate id or a primary-input
name.  Edges are implicit and pin position is list position.  VF2 needs
explicit typed edges, which is what to_nx builds.

This is a read-only projection: compression rewrites the dict directly.
"""
from __future__ import annotations

from typing import Any, Dict, Set

import networkx as nx

GateGraph = Dict[str, Dict[str, Any]]

#: Gate classes whose input order carries no meaning.  AND(A,B) == AND(B,A),
#: so pin position must never constrain matching for these.
COMMUTATIVE: Set[str] = {"AND", "OR", "XOR", "NAND", "NOR", "XNOR"}

#: Synthetic class for primary-input nodes, so patterns can require a port to
#: be primary or stay agnostic.
INPUT_CLS = "INPUT"


def to_nx(graph: GateGraph) -> nx.DiGraph:
    """Project a gate graph into a typed DiGraph.

    Nodes carry ``cls`` and ``outputs``.  Edges carry ``pins`` (every position
    the source occupies in the target's input list, as a sorted tuple) and
    ``dst_cls`` (the target's class, so edge_match can consult commutativity
    without node access — networkx passes edge_match only the edge data).
    """
    g = nx.DiGraph()

    for gid, node in graph.items():
        g.add_node(gid, cls=node["cls"], outputs=tuple(node.get("outputs", ())))

    for gid, node in graph.items():
        for src in node.get("inputs", ()):
            if src not in graph and not g.has_node(src):
                g.add_node(src, cls=INPUT_CLS, outputs=())

    for gid, node in graph.items():
        pins: Dict[str, list] = {}
        for pin, src in enumerate(node.get("inputs", ())):
            pins.setdefault(src, []).append(pin)
        for src, positions in pins.items():
            g.add_edge(src, gid,
                       pins=tuple(sorted(positions)),
                       dst_cls=node["cls"])

    return g
