"""Validator registry.

Structural matching alone accepts subgraphs that look right but compute
something else.  Every candidate must pass its pattern's validator before it
is compressed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

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


def _cross_coupled(graph: GateGraph, ids: List[str], cls: str) -> bool:
    """Two gates of *cls*, each driving an input of the other."""
    if len(ids) != 2 or any(graph[i]["cls"] != cls for i in ids):
        return False
    a, b = ids
    return a in graph[b].get("inputs", ()) and b in graph[a].get("inputs", ())


@register("sr_latch_nand")
def _sr_latch_nand(graph: GateGraph, match: Match) -> bool:
    """Cross-coupled NAND latch (active-low Sbar/Rbar).

    Same shape as the NOR latch but the inverting inputs mean the asserted
    level is low, which is why it is a distinct pattern rather than a redrawing
    of the same one.
    """
    if not _cross_coupled(graph, sorted(match.nodes), "NAND"):
        return False
    return len(set(match.inputs.values())) == 2 and len(match.outputs) == 2


def _mutual_pairs(graph: GateGraph, ids: List[str]) -> List[List[str]]:
    """Node pairs that each drive an input of the other.

    Identifying the latch core by "is this node in a cycle" does not work: in a
    JK the output feedback puts ALL four gates on a cycle, so that test returns
    everything.  Direct mutual coupling is the property that actually singles
    out the storage pair.
    """
    out: List[List[str]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (a in graph[b].get("inputs", ())
                    and b in graph[a].get("inputs", ())):
                out.append([a, b])
    return out


def _split_core_and_gating(graph: GateGraph, match: Match,
                           cls: str) -> Optional[Tuple[List[str], List[str]]]:
    """Separate a 4-gate latch into its cross-coupled core and its two input
    gates, or None if it does not have that shape."""
    ids = sorted(match.nodes)
    if len(ids) != 4:
        return None
    # Only the CORE pair must be of *cls*.  The gating gates differ by family:
    # a NAND latch is gated by NANDs, but a NOR latch is gated by ANDs, so
    # requiring all four to match rejected every NOR-based gated latch.
    cores = _mutual_pairs(graph, ids)
    if len(cores) != 1:
        return None
    core = cores[0]
    if any(graph[i]["cls"] != cls for i in core):
        return None
    gating = [i for i in ids if i not in core]
    return (core, gating) if len(gating) == 2 else None


def _shared_external(graph: GateGraph, gating: List[str],
                     ids: List[str]) -> bool:
    """The two input gates must share an external signal -- the enable/clock.

    Without this, two input gates driven by unrelated signals would qualify,
    which is a latch with some logic in front of it, not a gated latch.
    """
    shared = (set(graph[gating[0]].get("inputs", ()))
              & set(graph[gating[1]].get("inputs", ())))
    return bool(shared - set(ids))


def _gated_latch(graph: GateGraph, match: Match, cls: str) -> bool:
    """Cross-coupled *cls* core fed by two gating gates sharing one enable."""
    split = _split_core_and_gating(graph, match, cls)
    if split is None:
        return False
    core, gating = split
    if not _shared_external(graph, gating, sorted(match.nodes)):
        return False
    # No feedback from the core into the gating gates -- that would be a JK.
    return not any(c in graph[g].get("inputs", ())
                   for g in gating for c in core)


@register("gated_sr_latch_nand")
def _gated_sr_nand(graph: GateGraph, match: Match) -> bool:
    return _gated_latch(graph, match, "NAND")


@register("gated_sr_latch_nor")
def _gated_sr_nor(graph: GateGraph, match: Match) -> bool:
    return _gated_latch(graph, match, "NOR")


@register("d_latch")
def _d_latch(graph: GateGraph, match: Match) -> bool:
    """Gated latch whose two data inputs are complements of one signal.

    The inverter is exactly what makes it a D latch rather than a gated SR:
    S and R can never be asserted together, so the forbidden state is
    unreachable by construction.  So the checks are: a real cross-coupled core,
    an inverter driven by the same D that feeds the other gating gate, and a
    clock shared by both gating gates.
    """
    ids = sorted(match.nodes)
    if len(ids) != 5:
        return False
    cores = _mutual_pairs(graph, ids)
    if len(cores) != 1:
        return False
    core = cores[0]
    rest = [i for i in ids if i not in core]
    inverters = [i for i in rest if len(graph[i].get("inputs", ())) == 1]
    if len(inverters) != 1:
        return False
    inv = inverters[0]
    gating = [i for i in rest if i != inv]
    if len(gating) != 2:
        return False
    # one gating gate is fed by the inverter, the other directly by D
    fed_by_inv = [g for g in gating if inv in graph[g].get("inputs", ())]
    if len(fed_by_inv) != 1:
        return False
    direct = next(g for g in gating if g not in fed_by_inv)
    d_src = graph[inv].get("inputs", (None,))[0]
    if d_src is None or d_src not in graph[direct].get("inputs", ()):
        return False                        # not the SAME data signal
    return _shared_external(graph, gating, ids)


@register("jk_flipflop")
def _jk_flipflop(graph: GateGraph, match: Match) -> bool:
    """Gated latch plus output feedback into the input gates.

    The feedback is precisely what separates JK from gated SR: it is what makes
    J=K=1 toggle rather than forbidden.  So each gating gate must be fed by a
    core output, and the two must still share a clock.
    """
    split = _split_core_and_gating(graph, match, "NAND")
    if split is None:
        return False
    core, gating = split
    if not all(any(c in graph[g].get("inputs", ()) for c in core)
               for g in gating):
        return False
    return _shared_external(graph, gating, sorted(match.nodes))


#: Latch macro classes a master-slave stage may be built from.
_LATCH_CLASSES = {"DLATCH", "GATED_SRLATCH", "SRLATCH", "SRLATCH_NAND"}


@register("master_slave")
def _master_slave(graph: GateGraph, match: Match) -> bool:
    """Two latch stages in series on COMPLEMENTARY enables.

    The inversion between the two clocks is the edge-triggering mechanism: if
    both stages were transparent at once the data would race straight through
    and it would be a latch, not a flip-flop.  So the checks are that both
    stages really are latches, that the first feeds the second, and that one
    stage's clock is the inverse of the other's.
    """
    ids = sorted(match.nodes)
    if len(ids) != 3:
        return False
    latches = [i for i in ids if graph[i]["cls"] in _LATCH_CLASSES]
    invs = [i for i in ids if graph[i]["cls"] == "NOT"]
    if len(latches) != 2 or len(invs) != 1:
        return False
    inv = invs[0]
    # one latch feeds the other: that ordering is master -> slave
    a, b = latches
    if a in graph[b].get("inputs", ()):
        master, slave = a, b
    elif b in graph[a].get("inputs", ()):
        master, slave = b, a
    else:
        return False
    # the slave must be clocked by the inverter, the master by its source
    if inv not in graph[slave].get("inputs", ()):
        return False
    inv_src = graph[inv].get("inputs", (None,))[0]
    return inv_src is not None and inv_src in graph[master].get("inputs", ())


def validate(graph: GateGraph, match: Match) -> bool:
    """Run the pattern's validator.  Raises KeyError if it is not registered —
    a typo in a pattern file must fail loudly, not silently accept."""
    return VALIDATORS[match.pattern.validator](graph, match)
