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


def _external_inputs(graph: GateGraph, gid: str, inside: Set[str]) -> List[str]:
    """Sources driving *gid* from outside the match, in pin order."""
    return [s for s in graph[gid].get("inputs", ()) if s not in inside]


def _attach_candidates(graph: GateGraph, gid: str, pin: int,
                       inside: Set[str]) -> Optional[Set[str]]:
    """Which external sources could satisfy one attach point.

    A commutative gate has no meaningful pin order, so ANY of its external
    inputs is a candidate and the choice is settled later by the assignment.
    A non-commutative gate pins the answer down exactly.
    """
    ins = list(graph[gid].get("inputs", ()))
    if graph[gid]["cls"] in COMMUTATIVE:
        return set(_external_inputs(graph, gid, inside)) or None
    if pin >= len(ins):
        return None
    src = ins[pin]
    return None if src in inside else {src}


def _bind_ports(graph: GateGraph, mapping: Dict[str, str],
                pattern: Pattern) -> Optional[Dict[str, str]]:
    """Assign every declared input port to a distinct external source.

    This is an assignment problem, not a lookup.  A clocked device puts several
    external signals on one gate -- a JK's input NAND takes J, CLK and the Qbar
    feedback -- so "the external input of this node" is not well defined.  Each
    port instead gets a candidate set (intersected across its attach points, so
    a port touching two gates must resolve to a source common to both, which is
    exactly what identifies a shared clock), and backtracking finds a
    consistent choice.  Most-constrained port first, candidates in sorted
    order, so the result is deterministic.

    Every external input must end up claimed by some port.  Without that a
    latch with an async reset would match the plain latch pattern and the
    emitted HDL would silently drop the reset.
    """
    inside = set(mapping.values())

    cand: List[Tuple[str, Set[str]]] = []
    for port in pattern.inputs:
        allowed: Optional[Set[str]] = None
        for local, pin in port.attach:
            here = _attach_candidates(graph, mapping[local], pin, inside)
            if not here:
                return None
            allowed = here if allowed is None else (allowed & here)
            if not allowed:
                return None            # attach points cannot agree
        cand.append((port.name, allowed or set()))

    order = sorted(range(len(cand)), key=lambda i: (len(cand[i][1]), cand[i][0]))
    bound: Dict[str, str] = {}
    used: Set[str] = set()

    def _covers_everything() -> bool:
        """Every external signal must be claimed by some port, or compressing
        would silently drop it."""
        if pattern.allow_unbound_inputs:
            return True
        claimed = set(bound.values())
        return all(src in claimed
                   for gid in inside
                   for src in _external_inputs(graph, gid, inside))

    def _solve(k: int) -> bool:
        if k == len(order):
            # Coverage is checked HERE, inside the search, so a complete but
            # incomplete-covering assignment is rejected and the solver keeps
            # looking.  Checking it afterwards meant the first valid-looking
            # answer won and the match was then thrown away: a T flip-flop with
            # T tied to J and K let all three ports pick T, leaving the clock
            # unclaimed, and the device failed to match at all.
            return _covers_everything()
        name, choices = cand[order[k]]
        for src in sorted(choices):
            if not pattern.allow_shared_inputs and src in used:
                continue
            bound[name] = src
            used.add(src)
            if _solve(k + 1):
                return True
            del bound[name]
            used.discard(src)
        return False

    if not _solve(0):
        return None

    return dict(bound)


def candidates(graph: GateGraph, pattern: Pattern) -> List[Match]:
    """Every acceptable match of *pattern* in *graph*, in deterministic order."""
    host = to_nx(graph)
    pat = _pattern_nx(pattern)

    matcher = DiGraphMatcher(host, pat,
                             node_match=_node_match,
                             edge_match=_edge_match)

    # A symmetric pattern yields SEVERAL valid isomorphisms over the same set
    # of gates -- a cross-coupled pair can be mapped either way round.  Keeping
    # whichever VF2 happened to emit first made port binding vary between runs
    # (a JK emitted `case ({J, K})` or `case ({K, J})` at random), so the
    # generated HDL was not reproducible.  Collect every accepted isomorphism
    # per gate set and keep a canonical one.
    by_nodes: Dict[frozenset, List[Match]] = {}

    for iso in matcher.subgraph_isomorphisms_iter():
        mapping = {local: gid for gid, local in iso.items()}
        if _escapes(graph, mapping, pattern):
            continue
        if _too_many_fanning_outputs(graph, mapping, pattern):
            continue
        bound = _bind_ports(graph, mapping, pattern)
        if bound is None:
            continue
        by_nodes.setdefault(frozenset(mapping.values()), []).append(Match(
            pattern=pattern,
            mapping=mapping,
            inputs=bound,
            outputs={p.name: mapping[p.node] for p in pattern.outputs},
        ))

    def _canonical(m: Match) -> tuple:
        return (tuple(sorted(m.mapping.items())),
                tuple(sorted(m.inputs.items())))

    found = [min(ms, key=_canonical) for ms in by_nodes.values()]
    found.sort(key=lambda m: tuple(sorted(m.nodes)))
    return found
