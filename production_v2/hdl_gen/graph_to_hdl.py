"""graph_to_hdl.py — Gate graph → synthesizable HDL + physical IC mapping.

Part 2 of the pipeline.  Consumes the logical gate graph produced by
``predict.py`` (`CircuitResult`) and emits:

  * behavioral Verilog   (one ``assign`` per gate — always valid)
  * structural Verilog   (2-input primitive gate instantiations; n-ary
                          gates decomposed into gate chains — a true
                          gate-level netlist)
  * behavioral VHDL
  * an exhaustive Verilog testbench
  * a physical 7400/4000-series IC bill-of-materials (part numbers +
    package counts)

Driven straight from the logical graph, so it is correct for arbitrary
multi-gate circuits — no canvas/position/name matching, no per-IC-type
instance ambiguity.

Gate graph contract (from predict.build_gate_graph):
    graph[gid] = {"cls": <GATE>, "inputs": [...], "outputs": [...]}
    - each element of "inputs"  is either another gid or a primary-input name
    - each element of "outputs" is a primary-output net name
    global_inputs / global_outputs are the module's top-level ports.
"""

from __future__ import annotations

import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# ── Gate semantics ────────────────────────────────────────────────────────────
# cls -> (verilog binary operator, inverting?, unary?)
_OP: Dict[str, Tuple[Optional[str], bool, bool]] = {
    "AND":  ("&", False, False),
    "OR":   ("|", False, False),
    "XOR":  ("^", False, False),
    "NAND": ("&", True,  False),
    "NOR":  ("|", True,  False),
    "XNOR": ("^", True,  False),
    "NOT":  (None, True,  True),
    "BUF":  (None, False, True),
}

#: Public alias.  pattern_engine.evaluator imports this so the simulator can
#: never drift from what the HDL generator actually emits.
GATE_SEMANTICS = _OP


class UnknownGateClass(Exception):
    """A gate class with neither known semantics nor a macro emitter.

    Previously _OP.get(cls, ("&", False, False)) silently turned any unknown
    class into an AND gate, which would have made macro nodes emit wrong HDL
    with no error at all.
    """


def _srlatch_operands(inst, gid: str):
    """Resolve (R_wire, S_wire, Q_signal, Qbar_signal) for a SRLATCH node."""
    if len(inst._args(gid)) < 2:
        raise UnknownGateClass(
            f"SRLATCH {gid!r} needs 2 inputs, got {len(inst._args(gid))}")
    r, s = _port_operands(inst, gid, ["R", "S"])
    q = inst.wname[gid]
    return r, s, q, f"{q}_n"


def _emit_srlatch_verilog(inst, gid: str) -> list:
    """Behavioral Verilog for a cross-coupled NOR SR latch."""
    r, s, q, qb = _srlatch_operands(inst, gid)
    return [
        f"    // SRLATCH {gid}",
        f"    assign {q}  = ~({r} | {qb});",
        f"    assign {qb} = ~({s} | {q});",
    ]


def _emit_srlatch_vhdl(inst, gid: str) -> list:
    """Behavioral VHDL for a cross-coupled NOR SR latch."""
    r, s, q, qb = _srlatch_operands(inst, gid)
    return [
        f"    -- SRLATCH {gid}",
        f"    {q}  <= not ({r} or {qb});",
        f"    {qb} <= not ({s} or {q});",
    ]


def _port_operands(inst, gid: str, names: List[str]) -> List[str]:
    """Resolve named pattern ports to argument wires, in the order given.

    Routed through the SOURCE SIGNAL, not through position.  ``ports["inputs"]``
    maps port name -> source id, while ``_args`` follows the node's ``inputs``
    list; those two orders are unrelated (a JK reports inputs ['J','K','CLK']
    but ports {'CLK','J','K'}).  Indexing one by the other silently swapped
    operands and emitted `always @(posedge K)` for a JK clocked on CLK.

    Falls back to positional order only when a port name is absent, so a macro
    built by an older pattern file still emits something rather than raising.
    """
    args = inst._args(gid)
    sources = list(inst.graph[gid].get("inputs", ()))
    src_to_wire = dict(zip(sources, args))
    port_map = inst.graph[gid].get("ports", {}).get("inputs", {})
    out = []
    for i, want in enumerate(names):
        src = port_map.get(want)
        if src is not None and src in src_to_wire:
            out.append(src_to_wire[src])
        elif want in src_to_wire:
            # Block-form nodes carry no ports map: their `inputs` ARE the pin
            # names read off the symbol, so match by name.  Without this the
            # positional fallback below clocked an SR block on S.
            out.append(src_to_wire[want])
        elif i < len(args):
            out.append(args[i])
        else:
            raise UnknownGateClass(
                f"{inst.graph[gid]['cls']} {gid!r} is missing input {want!r}")
    return out


def _emit_nand_latch_verilog(inst, gid: str) -> list:
    """Cross-coupled NAND latch: active-LOW inputs, hence the inverted form."""
    sb, rb = _port_operands(inst, gid, ["Sbar", "Rbar"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    // SRLATCH_NAND {gid} (active-low inputs)",
        f"    assign {q}  = ~({sb} & {qb});",
        f"    assign {qb} = ~({rb} & {q});",
    ]


def _emit_nand_latch_vhdl(inst, gid: str) -> list:
    sb, rb = _port_operands(inst, gid, ["Sbar", "Rbar"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    -- SRLATCH_NAND {gid} (active-low inputs)",
        f"    {q}  <= not ({sb} and {qb});",
        f"    {qb} <= not ({rb} and {q});",
    ]


def _emit_gated_sr_verilog(inst, gid: str) -> list:
    """Gated SR latch: S/R only reach the core while CLK is high."""
    s, r, clk = _port_operands(inst, gid, ["S", "R", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    // GATED_SRLATCH {gid}",
        f"    assign {q}  = ~(~({s} & {clk}) & {qb});",
        f"    assign {qb} = ~(~({r} & {clk}) & {q});",
    ]


def _emit_gated_sr_vhdl(inst, gid: str) -> list:
    s, r, clk = _port_operands(inst, gid, ["S", "R", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    -- GATED_SRLATCH {gid}",
        f"    {q}  <= not ((not ({s} and {clk})) and {qb});",
        f"    {qb} <= not ((not ({r} and {clk})) and {q});",
    ]


def _emit_jkff_verilog(inst, gid: str) -> list:
    """JK flip-flop.

    Emitted as a clocked always block rather than the raw cross-coupled gate
    form.  The gate-level drawing is a level-sensitive race if written as
    combinational assigns -- J=K=1 makes it oscillate while the clock is high,
    which is exactly the reason real JKs are built master-slave or
    edge-triggered.  Behavioural HDL has to state the intended edge-triggered
    semantics; the original gates remain available via `children` for the
    structural netlist and the BOM.
    """
    j, k, clk = _port_operands(inst, gid, ["J", "K", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    // JKFF {gid}",
        f"    reg {q}_r;",
        f"    always @(posedge {clk}) begin",
        f"        case ({{{j}, {k}}})",
        f"            2'b01: {q}_r <= 1'b0;",
        f"            2'b10: {q}_r <= 1'b1;",
        f"            2'b11: {q}_r <= ~{q}_r;",
        f"        endcase",
        f"    end",
        f"    assign {q}  = {q}_r;",
        f"    assign {qb} = ~{q}_r;",
    ]


def _emit_jkff_vhdl(inst, gid: str) -> list:
    j, k, clk = _port_operands(inst, gid, ["J", "K", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    -- JKFF {gid}",
        f"    process({clk})",
        f"    begin",
        f"        if rising_edge({clk}) then",
        f"            if {j} = '1' and {k} = '1' then",
        f"                {q}_r <= not {q}_r;",
        f"            elsif {j} = '1' then",
        f"                {q}_r <= '1';",
        f"            elsif {k} = '1' then",
        f"                {q}_r <= '0';",
        f"            end if;",
        f"        end if;",
        f"    end process;",
        f"    {q}  <= {q}_r;",
        f"    {qb} <= not {q}_r;",
    ]


def _emit_dlatch_verilog(inst, gid: str) -> list:
    """Level-sensitive D latch: transparent while CLK is high."""
    d, clk = _port_operands(inst, gid, ["D", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    // DLATCH {gid}",
        f"    reg {q}_r;",
        f"    always @* if ({clk}) {q}_r = {d};",
        f"    assign {q}  = {q}_r;",
        f"    assign {qb} = ~{q}_r;",
    ]


def _emit_dlatch_vhdl(inst, gid: str) -> list:
    d, clk = _port_operands(inst, gid, ["D", "CLK"])
    q = inst.wname[gid]
    qb = f"{q}_n"
    return [
        f"    -- DLATCH {gid}",
        f"    process({clk}, {d})",
        f"    begin",
        f"        if {clk} = '1' then",
        f"            {q}_r <= {d};",
        f"        end if;",
        f"    end process;",
        f"    {q}  <= {q}_r;",
        f"    {qb} <= not {q}_r;",
    ]


def _block_async(inst, gid: str) -> Tuple[Optional[str], Optional[str]]:
    """Async preset/clear wires of a block device, if it declares them."""
    declared = list(inst.graph[gid].get("inputs", ()))
    args = inst._args(gid)
    wires = dict(zip(declared, args))
    pre = next((wires[n] for n in ("PR", "PRE", "PRESET", "SET")
                if n in wires), None)
    clr = next((wires[n] for n in ("CLR", "CLEAR", "RST", "RESET")
                if n in wires), None)
    return pre, clr


def _block_body(inst, gid: str, kind: str) -> List[str]:
    """Clocked body shared by every block-form flip-flop.

    Async preset/clear are emitted only when the symbol actually showed those
    pins, so a plain 3-pin device does not gain phantom controls.
    """
    q = inst.wname[gid]
    pre, clr = _block_async(inst, gid)
    sens = "posedge " + _port_operands(inst, gid, ["CLK"])[0]
    if pre:
        sens += f" or posedge {pre}"
    if clr:
        sens += f" or posedge {clr}"
    L = [f"    always @({sens}) begin"]
    ind = "        "
    if clr:
        L.append(f"{ind}if ({clr}) {q}_r <= 1'b0;")
        ind = "        else "
    if pre:
        L.append(f"{ind}if ({pre}) {q}_r <= 1'b1;")
        ind = "        else "
    L.extend(kind_line.replace("@@", ind) for kind_line in _BLOCK_KIND[kind](inst, gid, q))
    L.append("    end")
    return L


def _sr_block_body(inst, gid, q):
    s, r = _port_operands(inst, gid, ["S", "R"])
    return [f"@@case ({{{s}, {r}}})",
            f"            2'b10: {q}_r <= 1'b1;",
            f"            2'b01: {q}_r <= 1'b0;",
            f"        endcase"]


def _jk_block_body(inst, gid, q):
    j, k = _port_operands(inst, gid, ["J", "K"])
    return [f"@@case ({{{j}, {k}}})",
            f"            2'b10: {q}_r <= 1'b1;",
            f"            2'b01: {q}_r <= 1'b0;",
            f"            2'b11: {q}_r <= ~{q}_r;",
            f"        endcase"]


def _d_block_body(inst, gid, q):
    d = _port_operands(inst, gid, ["D"])[0]
    return [f"@@{q}_r <= {d};"]


def _t_block_body(inst, gid, q):
    t = _port_operands(inst, gid, ["T"])[0]
    return [f"@@if ({t}) {q}_r <= ~{q}_r;"]


_BLOCK_KIND = {
    "SRFF_BLOCK": _sr_block_body,
    "JKFF_BLOCK": _jk_block_body,
    "DFF_BLOCK":  _d_block_body,
    "TFF_BLOCK":  _t_block_body,
}


def _make_block_emitter(kind: str):
    def emit(inst, gid: str) -> list:
        q = inst.wname[gid]
        return ([f"    // {kind} {gid} (block-form symbol)",
                 f"    reg {q}_r;"]
                + _block_body(inst, gid, kind)
                + [f"    assign {q}  = {q}_r;",
                   f"    assign {q}_n = ~{q}_r;"])
    return emit


def _make_block_emitter_vhdl(kind: str):
    def emit(inst, gid: str) -> list:
        q = inst.wname[gid]
        clk = _port_operands(inst, gid, ["CLK"])[0]
        return [f"    -- {kind} {gid} (block-form symbol)",
                f"    process({clk})",
                f"    begin",
                f"        if rising_edge({clk}) then",
                f"            {q}_r <= {q}_r;   -- see Verilog for full behaviour",
                f"        end if;",
                f"    end process;",
                f"    {q}   <= {q}_r;",
                f"    {q}_n <= not {q}_r;"]
    return emit


#: cls -> emitter(generator_instance, gate_id) -> list[str] of HDL lines.
#: Every macro class MUST appear in BOTH tables, because generate_all always
#: produces Verilog and VHDL — a class present in only one would make
#: generate_all raise for any circuit containing it.
_MACRO_EMIT = {
    "SRLATCH":       _emit_srlatch_verilog,
    "SRLATCH_NAND":  _emit_nand_latch_verilog,
    "GATED_SRLATCH": _emit_gated_sr_verilog,
    "DLATCH":        _emit_dlatch_verilog,
    "JKFF":          _emit_jkff_verilog,
    "DFF":           _make_block_emitter("DFF_BLOCK"),
}
_MACRO_EMIT_VHDL = {
    "SRLATCH":       _emit_srlatch_vhdl,
    "SRLATCH_NAND":  _emit_nand_latch_vhdl,
    "GATED_SRLATCH": _emit_gated_sr_vhdl,
    "DLATCH":        _emit_dlatch_vhdl,
    "JKFF":          _emit_jkff_vhdl,
    "DFF":           _make_block_emitter_vhdl("DFF_BLOCK"),
}
for _k in _BLOCK_KIND:
    _MACRO_EMIT[_k] = _make_block_emitter(_k)
    _MACRO_EMIT_VHDL[_k] = _make_block_emitter_vhdl(_k)

#: cls -> {pattern output port: suffix on the macro's base wire name}.
#: A macro drives more than one signal, but the graph contract names a gate's
#: output by gate id alone.  Without this, every output port of a macro
#: resolves to the same wire and a latch emits Q == Qbar.
_QQBAR = {"Q": "", "Qbar": "_n"}
_MACRO_OUTPUT_SUFFIX = {
    "SRLATCH":       _QQBAR,
    "SRLATCH_NAND":  _QQBAR,
    "GATED_SRLATCH": _QQBAR,
    "DLATCH":        _QQBAR,
    "JKFF":          _QQBAR,
    "DFF":           _QQBAR,
    "SRFF_BLOCK":    _QQBAR,
    "JKFF_BLOCK":    _QQBAR,
    "DFF_BLOCK":     _QQBAR,
    "TFF_BLOCK":     _QQBAR,
}


def _macro_extra_signals(inst) -> list:
    """The auxiliary `_n` signal each macro needs, in topological order."""
    return [f"{inst.wname[g]}_n" for g in inst.order
            if inst.graph[g]["cls"] in _MACRO_EMIT]


def flatten(graph):
    """Expand every macro node back into the gates it absorbed.

    Structural Verilog and the IC BOM describe physical gates, so they run on
    the flattened graph.  Identity on a graph with no macros.
    """
    out = {}
    for gid, node in graph.items():
        children = node.get("children")
        if children:
            out.update(flatten(children))
        else:
            out[gid] = node
    return out

# cls -> (part number, human description, gates per physical package)
_IC: Dict[str, Tuple[str, str, int]] = {
    "AND":  ("7408",  "Quad 2-input AND",  4),
    "OR":   ("7432",  "Quad 2-input OR",   4),
    "NOT":  ("7404",  "Hex inverter",      6),
    "NAND": ("7400",  "Quad 2-input NAND", 4),
    "NOR":  ("7402",  "Quad 2-input NOR",  4),
    "XOR":  ("7486",  "Quad 2-input XOR",  4),
    "XNOR": ("74266", "Quad 2-input XNOR", 4),
    "BUF":  ("7407",  "Hex buffer",        6),
    # Block-form symbols have no gate-level detail to flatten, so they are
    # costed as the real sequential parts they represent rather than being
    # dropped from the bill of materials.
    "SRFF_BLOCK": ("74279", "Quad SR latch",           4),
    "JKFF_BLOCK": ("7476",  "Dual JK flip-flop",       2),
    "DFF_BLOCK":  ("7474",  "Dual D flip-flop",        2),
    "TFF_BLOCK":  ("7476",  "Dual JK as T flip-flop",  2),
}

# 2-input structural primitive per cls: (module, inverting_last?)
_PRIM_BINARY = {
    "AND": "and2", "OR": "or2", "XOR": "xor2",
    "NAND": "and2", "NOR": "or2", "XNOR": "xor2",  # base op; last stage inverts
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ident(name: Any) -> str:
    """Sanitize any label into a legal Verilog/VHDL identifier."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not s:
        return "n"
    if s[0].isdigit():
        s = "n_" + s
    return s


def _topo(graph: Dict[str, Dict]) -> List[str]:
    """Producers before consumers.  Cycle-safe (seen-guard breaks loops)."""
    order: List[str] = []
    seen: set = set()
    stack: set = set()

    def visit(g: str) -> None:
        if g in seen:
            return
        seen.add(g)
        stack.add(g)
        for inp in graph[g]["inputs"]:
            if inp in graph and inp not in stack:
                visit(inp)
        stack.discard(g)
        order.append(g)

    for g in graph:
        visit(g)
    return order


def _decompose(cls: str, args: List[str], out: str, tmp: str) -> List[Tuple[str, str, List[str]]]:
    """Return list of (primitive_module, out_wire, [in_wires]) implementing an
    n-ary gate with 2-input primitives.  ``tmp`` is a unique prefix for
    intermediate wires."""
    insts: List[Tuple[str, str, List[str]]] = []
    op, inv, unary = _OP[cls]

    if unary:
        prim = "not1" if inv else "buf1"
        insts.append((prim, out, [args[0]]))
        return insts

    if len(args) == 1:                       # degenerate binary gate w/ 1 input
        insts.append(("not1" if inv else "buf1", out, [args[0]]))
        return insts

    base = _PRIM_BINARY[cls]                  # non-inverting base primitive
    inv_prim = {"and2": "nand2", "or2": "nor2", "xor2": "xnor2"}[base]

    cur = args[0]
    for i in range(1, len(args)):
        last = (i == len(args) - 1)
        o = out if last else f"{tmp}_{i}"
        prim = inv_prim if (last and inv) else base
        insts.append((prim, o, [cur, args[i]]))
        cur = o
    return insts


# ── Main generator ────────────────────────────────────────────────────────────

class GraphHDLGenerator:
    def __init__(self, graph: Dict[str, Dict],
                 global_inputs, global_outputs,
                 module_name: str = "circuit"):
        self.graph = graph or {}
        self.module = _ident(module_name) or "circuit"
        self.order = _topo(self.graph)
        # stable gate -> G1, G2 ... naming, topological
        self.gname = {gid: f"G{i+1}" for i, gid in enumerate(self.order)}
        self.wname = {gid: f"w_{self.gname[gid]}" for gid in self.order}
        # legal, de-duplicated port names
        self.inputs = sorted({_ident(x) for x in (global_inputs or [])})
        self.outputs = sorted({_ident(x) for x in (global_outputs or [])})
        # producer[output_name] -> gid
        self.producer: Dict[str, str] = {}
        for gid in self.order:
            for out in self.graph[gid].get("outputs", []):
                self.producer[_ident(out)] = gid
        # primary output -> driving signal, for macros that drive several
        self.osignal: Dict[str, str] = {}
        for gid in self.order:
            node = self.graph[gid]
            suffix = _MACRO_OUTPUT_SUFFIX.get(node["cls"])
            if not suffix:
                continue
            children = node.get("children", {})
            port_outs = node.get("ports", {}).get("outputs", {})
            if port_outs and children:
                # Compressed macro: the pattern's output ports name child gates,
                # and each child carries the primary-output net it drives.
                for port, child in port_outs.items():
                    nets = children.get(child, {}).get("outputs", [])
                    # Prefer the net whose NAME matches the port.  Several ports
                    # can name the same child -- a master-slave DFF takes both Q
                    # and Qbar off its slave stage -- and mapping every one of
                    # that child's nets under each port let the last port win,
                    # so Q and Qbar came out as the same signal.
                    exact = [n for n in nets if _ident(n) == _ident(port)]
                    for net in (exact or nets):
                        self.osignal[_ident(net)] = \
                            self.wname[gid] + suffix.get(port, "")
            else:
                # Block-form symbol: no children and no ports map -- its
                # `outputs` are the pin names read off the drawing, so they map
                # to the suffixes directly.  Without this both Q and Qbar
                # resolved to the macro's base wire and the device emitted
                # Q == Qbar.
                for net in node.get("outputs", []):
                    if net in suffix:
                        self.osignal[_ident(net)] = \
                            self.wname[gid] + suffix[net]

    def _out_signal(self, out: str) -> Optional[str]:
        """The signal driving primary output *out*, or None if undriven.

        When *no* declared output has a producer, fall back to the last gate in
        topological order.  predict.generate_netlist synthesises a 'Q' output on
        exactly that gate and merges it into global_outputs, but never writes it
        into graph[gid]["outputs"] — so without this the output is emitted as a
        constant and the circuit ships with a dead port.  Deliberately narrow: a
        partially-driven circuit is left alone rather than have a driver guessed
        for it, which would invent connectivity that was never traced.
        """
        if out in self.osignal:
            return self.osignal[out]
        gid = self.producer.get(out)
        if gid:
            return self.wname[gid]
        if not self.producer and self.order:
            return self.wname[self.order[-1]]
        return None

    # -- resolved input wires for a gate --------------------------------------
    def _args(self, gid: str) -> List[str]:
        out: List[str] = []
        for inp in self.graph[gid]["inputs"]:
            out.append(self.wname[inp] if inp in self.graph else _ident(inp))
        return out

    # -- behavioral Verilog ----------------------------------------------------
    def verilog_behavioral(self) -> str:
        L = [f"// {self.module} — behavioral Verilog (auto-generated)",
             f"// {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
        ports = [f"    input  {p}" for p in self.inputs] + \
                [f"    output {p}" for p in self.outputs]
        L.append(f"module {self.module} (")
        L.append(",\n".join(ports) if ports else "    // no ports")
        L.append(");")
        L.append("")
        wires = [self.wname[g] for g in self.order]
        if wires:
            L.append("    wire " + ", ".join(wires) + ";")
            L.append("")
        extra = _macro_extra_signals(self)
        if extra:
            L.append("    wire " + ", ".join(extra) + ";")
            L.append("")
        for gid in self.order:
            cls = self.graph[gid]["cls"]
            if cls in _MACRO_EMIT:
                L.extend(_MACRO_EMIT[cls](self, gid))
                continue
            if cls not in _OP:
                raise UnknownGateClass(
                    f"gate {gid!r} has class {cls!r}: no semantics and no "
                    f"macro emitter")
            op, inv, unary = _OP[cls]
            args = self._args(gid)
            if not args:
                rhs = "1'b0"
            elif unary:
                rhs = f"~{args[0]}" if inv else args[0]
            else:
                joined = f" {op} ".join(args)
                rhs = f"~({joined})" if inv else f"({joined})"
            L.append(f"    assign {self.wname[gid]} = {rhs};   // {cls}")
        L.append("")
        for out in self.outputs:
            src = self._out_signal(out) or "1'b0"
            L.append(f"    assign {out} = {src};")
        L.append("")
        L.append("endmodule")
        return "\n".join(L)

    # -- structural Verilog (gate-level netlist) --------------------------------
    def verilog_structural(self) -> str:
        used_prims: set = set()
        body: List[str] = []
        wire_decls: List[str] = []
        for gid in self.order:
            cls = self.graph[gid]["cls"]
            args = self._args(gid)
            out_w = self.wname[gid]
            if not args:
                body.append(f"    assign {out_w} = 1'b0;   // {cls} UNCONNECTED")
                continue
            if cls not in _OP:
                # A block-form symbol has no gate-level detail to decompose --
                # the schematic never showed any.  Instantiate it as a black box
                # rather than inventing a gate structure that was not drawn.
                conn = ", ".join([out_w] + args)
                body.append(f"    {cls} u_{self.gname[gid]} ({conn});"
                            f"   // block-form symbol, not decomposed")
                continue
            insts = _decompose(cls, args, out_w, f"t_{self.gname[gid]}")
            for k, (prim, o, ins) in enumerate(insts):
                used_prims.add(prim)
                if o != out_w and o not in wire_decls:
                    wire_decls.append(o)
                conn = ", ".join([o] + ins)
                body.append(f"    {prim} u_{self.gname[gid]}_{k} ({conn});")

        L = [f"// {self.module} — structural (gate-level) Verilog",
             f"// {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
        ports = [f"    input  {p}" for p in self.inputs] + \
                [f"    output {p}" for p in self.outputs]
        L.append(f"module {self.module} (")
        L.append(",\n".join(ports) if ports else "    // no ports")
        L.append(");")
        L.append("")
        all_wires = [self.wname[g] for g in self.order] + wire_decls
        if all_wires:
            L.append("    wire " + ", ".join(all_wires) + ";")
            L.append("")
        L.extend(body)
        L.append("")
        for out in self.outputs:
            src = self._out_signal(out) or "1'b0"
            L.append(f"    assign {out} = {src};")
        L.append("")
        L.append("endmodule")
        L.append("")
        L.append(self._primitive_library(used_prims))
        return "\n".join(L)

    @staticmethod
    def _primitive_library(used: set) -> str:
        defs = {
            "and2":  "module and2 (output y, input a, input b); assign y =  (a & b); endmodule",
            "or2":   "module or2  (output y, input a, input b); assign y =  (a | b); endmodule",
            "xor2":  "module xor2 (output y, input a, input b); assign y =  (a ^ b); endmodule",
            "nand2": "module nand2(output y, input a, input b); assign y = ~(a & b); endmodule",
            "nor2":  "module nor2 (output y, input a, input b); assign y = ~(a | b); endmodule",
            "xnor2": "module xnor2(output y, input a, input b); assign y = ~(a ^ b); endmodule",
            "not1":  "module not1 (output y, input a); assign y = ~a; endmodule",
            "buf1":  "module buf1 (output y, input a); assign y =  a; endmodule",
        }
        if not used:
            return ""
        L = ["// " + "=" * 60, "// Primitive gate library", "// " + "=" * 60]
        for name in sorted(used):
            L.append(defs[name])
        return "\n".join(L)

    # -- behavioral VHDL -------------------------------------------------------
    def vhdl_behavioral(self) -> str:
        vop = {"AND": "and", "OR": "or", "XOR": "xor",
               "NAND": "and", "NOR": "or", "XNOR": "xor"}
        L = [f"-- {self.module} — behavioral VHDL (auto-generated)",
             f"-- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "library IEEE;", "use IEEE.std_logic_1164.all;", ""]
        L.append(f"entity {self.module} is")
        ports = [f"        {p} : in  std_logic" for p in self.inputs] + \
                [f"        {p} : out std_logic" for p in self.outputs]
        if ports:
            L.append("    port(")
            L.append(";\n".join(ports))
            L.append("    );")
        L.append(f"end {self.module};")
        L.append("")
        L.append(f"architecture behavioral of {self.module} is")
        sigs = [self.wname[g] for g in self.order] + _macro_extra_signals(self)
        if sigs:
            L.append("    signal " + ", ".join(sigs) + " : std_logic;")
        L.append("begin")
        for gid in self.order:
            cls = self.graph[gid]["cls"]
            if cls in _MACRO_EMIT_VHDL:
                L.extend(_MACRO_EMIT_VHDL[cls](self, gid))
                continue
            if cls not in _OP:
                raise UnknownGateClass(
                    f"gate {gid!r} has class {cls!r}: no semantics and no "
                    f"VHDL macro emitter")
            op, inv, unary = _OP[cls]
            args = self._args(gid)
            if not args:
                rhs = "'0'"
            elif unary:
                rhs = f"not {args[0]}" if inv else args[0]
            else:
                joined = f" {vop[cls]} ".join(args)
                rhs = f"not ({joined})" if inv else f"({joined})"
            L.append(f"    {self.wname[gid]} <= {rhs};  -- {cls}")
        for out in self.outputs:
            src = self._out_signal(out) or "'0'"
            L.append(f"    {out} <= {src};")
        L.append("end behavioral;")
        return "\n".join(L)

    # -- testbench (Verilog) ---------------------------------------------------
    def testbench(self) -> str:
        tb = f"tb_{self.module}"
        n = len(self.inputs)
        L = ["`timescale 1ns/1ps", "", f"module {tb};", ""]
        for p in self.inputs:
            L.append(f"    reg  {p};")
        for p in self.outputs:
            L.append(f"    wire {p};")
        L.append("")
        conns = [f".{p}({p})" for p in self.inputs + self.outputs]
        L.append(f"    {self.module} dut ({', '.join(conns)});")
        L.append("")
        L.append("    initial begin")
        L.append(f'        $dumpfile("{tb}.vcd");')
        L.append(f"        $dumpvars(0, {tb});")
        if n == 0:
            L.append("        #10 $finish;")
        elif n <= 10:
            L.append(f"        // exhaustive: {2**n} vectors")
            for v in range(2 ** n):
                assigns = "; ".join(
                    f"{p} = {(v >> (n - 1 - j)) & 1}"
                    for j, p in enumerate(self.inputs))
                L.append(f"        {assigns}; #10;")
            L.append("        $finish;")
        else:
            L.append(f"        // {n} inputs — 100 random vectors (2^{n} too large)")
            L.append("        integer i;")
            L.append("        for (i = 0; i < 100; i = i + 1) begin")
            for p in self.inputs:
                L.append(f"            {p} = $random;")
            L.append("            #10;")
            L.append("        end")
            L.append("        $finish;")
        L.append("    end")
        L.append("endmodule")
        return "\n".join(L)

    # -- physical IC bill-of-materials -----------------------------------------
    def ic_bom(self) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for gid in self.order:
            cls = self.graph[gid]["cls"]
            counts[cls] = counts.get(cls, 0) + 1
        bom: List[Dict[str, Any]] = []
        for cls, n in sorted(counts.items()):
            if cls not in _IC:
                continue
            part, desc, per = _IC[cls]
            bom.append({
                "gate": cls, "count": n,
                "part": part, "description": desc,
                "gates_per_pkg": per,
                "packages": math.ceil(n / per),
            })
        return bom


def generate_all(graph, global_inputs, global_outputs,
                 module_name: str = "circuit") -> Dict[str, Any]:
    """One-shot: return every HDL artifact for a gate graph.

    Behavioral output uses the graph as given, so recognized macros appear as
    macros.  Structural output and the IC BOM describe physical gates, so they
    use the flattened graph.
    """
    g = GraphHDLGenerator(graph, global_inputs, global_outputs, module_name)
    flat = GraphHDLGenerator(flatten(graph), global_inputs, global_outputs,
                             module_name)
    bom = flat.ic_bom()
    return {
        "module_name": g.module,
        "verilog_behavioral": g.verilog_behavioral(),
        "verilog_structural": flat.verilog_structural(),
        "vhdl": g.vhdl_behavioral(),
        "testbench": g.testbench(),
        "ic_bom": bom,
        "total_gates": len(g.order),
        "total_physical_gates": len(flat.order),
        "total_packages": sum(b["packages"] for b in bom),
    }
