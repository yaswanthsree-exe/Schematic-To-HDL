"""VF2 candidate generation plus the acceptance preconditions.

A candidate must survive, in order:
  1. structural isomorphism (node class + pin-aware edges)
  2. the escape rule  — no internal signal consumed outside the match
  3. the fan-out rule — at most one output node consumed outside the match
  4. port binding     — every port resolves to exactly one external source
Validators (validators.py) run afterwards on whatever survives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from .graph_adapter import COMMUTATIVE, GateGraph, to_nx
from .library import Pattern


@dataclass
class Match:
    pattern: Pattern
    mapping: Dict[str, str]   # pattern local id -> graph id
    inputs:  Dict[str, str]   # port name -> external source id
    outputs: Dict[str, str]   # port name -> graph id

    @property
    def nodes(self) -> frozenset:
        return frozenset(self.mapping.values())


def _pattern_nx(pattern: Pattern) -> nx.DiGraph:
    g = nx.DiGraph()
    node_map = pattern.node_map
    for local, cls in node_map.items():
        g.add_node(local, cls=cls)
    for e in pattern.edges:
        g.add_edge(e.src, e.dst, pins=tuple(sorted(e.pins)),
                   dst_cls=node_map[e.dst])
    return g


def _node_match(host: dict, pat: dict) -> bool:
    return host.get("cls") == pat.get("cls")


def _edge_match(host: dict, pat: dict) -> bool:
    # Pin position is meaningless for commutative targets (AND(A,B)==AND(B,A)).
    if pat.get("dst_cls") in COMMUTATIVE:
        return True
    return tuple(host.get("pins", ())) == tuple(pat.get("pins", ()))


def _consumers_outside(graph: GateGraph, gid: str, inside: Set[str]) -> List[str]:
    return [other for other, node in graph.items()
            if other not in inside and gid in node.get("inputs", ())]


def _escapes(graph: GateGraph, mapping: Dict[str, str],
             pattern: Pattern) -> bool:
    """True when an internal (non-output) signal is visible outside the match."""
    inside = set(mapping.values())
    output_nodes = {mapping[p.node] for p in pattern.outputs}
    for gid in inside:
        if gid in output_nodes:
            continue
        if _consumers_outside(graph, gid, inside):
            return True
        if graph[gid].get("outputs"):
            return True
    return False


def _too_many_fanning_outputs(graph: GateGraph, mapping: Dict[str, str],
                              pattern: Pattern) -> bool:
    """A macro is referenced by a single id, so at most one of its output
    nodes may be consumed by a gate outside the match."""
    inside = set(mapping.values())
    fanning = [mapping[p.node] for p in pattern.outputs
               if _consumers_outside(graph, mapping[p.node], inside)]
    return len(fanning) > 1


def _external_source(graph: GateGraph, gid: str, pin: int,
                     inside: Set[str]) -> Optional[str]:
    """Resolve one attach point to the external signal driving it.

    For a commutative target the declared pin is nominal, so the attach point
    resolves to that node's unique external input; ambiguity (zero or two or
    more external inputs) rejects the match.  For a non-commutative target the
    declared pin is exact.
    """
    ins = list(graph[gid].get("inputs", ()))
    if graph[gid]["cls"] in COMMUTATIVE:
        external = [s for s in ins if s not in inside]
        return external[0] if len(external) == 1 else None
    if pin >= len(ins):
        return None
    src = ins[pin]
    return None if src in inside else src


def _bind_ports(graph: GateGraph, mapping: Dict[str, str],
                pattern: Pattern) -> Optional[Dict[str, str]]:
    inside = set(mapping.values())
    bound: Dict[str, str] = {}
    for port in pattern.inputs:
        resolved: Set[str] = set()
        for local, pin in port.attach:
            src = _external_source(graph, mapping[local], pin, inside)
            if src is None:
                return None
            resolved.add(src)
        if len(resolved) != 1:          # attach points disagree
            return None
        bound[port.name] = resolved.pop()
    if not pattern.allow_shared_inputs and len(set(bound.values())) != len(bound):
        return None
    return bound


def candidates(graph: GateGraph, pattern: Pattern) -> List[Match]:
    """Every acceptable match of *pattern* in *graph*, in deterministic order."""
    host = to_nx(graph)
    pat = _pattern_nx(pattern)

    matcher = DiGraphMatcher(host, pat,
                             node_match=_node_match,
                             edge_match=_edge_match)

    found: List[Match] = []
    seen: Set[frozenset] = set()

    for iso in matcher.subgraph_isomorphisms_iter():
        mapping = {local: gid for gid, local in iso.items()}
        key = frozenset(mapping.values())
        if key in seen:
            continue
        if _escapes(graph, mapping, pattern):
            continue
        if _too_many_fanning_outputs(graph, mapping, pattern):
            continue
        bound = _bind_ports(graph, mapping, pattern)
        if bound is None:
            continue
        seen.add(key)
        found.append(Match(
            pattern=pattern,
            mapping=mapping,
            inputs=bound,
            outputs={p.name: mapping[p.node] for p in pattern.outputs},
        ))

    found.sort(key=lambda m: tuple(sorted(m.nodes)))
    return found
