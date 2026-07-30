"""Pattern library: JSON topology plus a named validator.

Topology is data.  Functional constraints are predicates and live in
validators.py — expressing them as data would mean inventing a DSL.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "cls", "level", "nodes", "edges", "ports", "validator"],
    "additionalProperties": False,
    "properties": {
        "name":  {"type": "string", "minLength": 1},
        "cls":   {"type": "string", "minLength": 1},
        "level": {"type": "integer", "minimum": 1},
        "nodes": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "string", "minLength": 1},
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "pins"],
                "additionalProperties": False,
                "properties": {
                    "from": {"type": "string"},
                    "to":   {"type": "string"},
                    "pins": {"type": "array", "items": {"type": "integer", "minimum": 0},
                             "minItems": 1},
                },
            },
        },
        "ports": {
            "type": "object",
            "required": ["inputs", "outputs"],
            "additionalProperties": False,
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "attach"],
                        "additionalProperties": False,
                        "properties": {
                            "name":   {"type": "string", "minLength": 1},
                            "attach": {
                                "type": "array", "minItems": 1,
                                "items": {
                                    "type": "array",
                                    "minItems": 2, "maxItems": 2,
                                    "prefixItems": [{"type": "string"},
                                                    {"type": "integer", "minimum": 0}],
                                },
                            },
                        },
                    },
                },
                "outputs": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["name", "node"],
                        "additionalProperties": False,
                        "properties": {"name": {"type": "string", "minLength": 1},
                                       "node": {"type": "string"}},
                    },
                },
            },
        },
        "validator":           {"type": "string", "minLength": 1},
        "expected":            {"type": "string"},
        "allow_shared_inputs": {"type": "boolean"},
    },
}


@dataclass(frozen=True)
class PatternEdge:
    src:  str
    dst:  str
    pins: Tuple[int, ...]


@dataclass(frozen=True)
class InputPort:
    name:   str
    attach: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class OutputPort:
    name: str
    node: str


@dataclass(frozen=True)
class Pattern:
    name:                str
    cls:                 str
    level:               int
    nodes:               Tuple[Tuple[str, str], ...]   # frozen; see .node_map
    edges:               Tuple[PatternEdge, ...]
    inputs:              Tuple[InputPort, ...]
    outputs:             Tuple[OutputPort, ...]
    validator:           str
    expected:            Optional[str] = None
    allow_shared_inputs: bool = False

    @property
    def node_map(self) -> Dict[str, str]:
        """local id -> gate class."""
        return dict(self.nodes)


def load_pattern(path: str) -> Pattern:
    """Load and validate one pattern file.  Raises ValueError on any problem."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    try:
        jsonschema.validate(raw, _SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"{path}: schema violation: {exc.message}") from exc

    nodes: Dict[str, str] = raw["nodes"]

    for e in raw["edges"]:
        for end in (e["from"], e["to"]):
            if end not in nodes:
                raise ValueError(f"{path}: edge references unknown node {end!r}")

    for p in raw["ports"]["inputs"]:
        for node, _pin in p["attach"]:
            if node not in nodes:
                raise ValueError(f"{path}: input port {p['name']!r} "
                                 f"references unknown node {node!r}")
    for p in raw["ports"]["outputs"]:
        if p["node"] not in nodes:
            raise ValueError(f"{path}: output port {p['name']!r} "
                             f"references unknown node {p['node']!r}")

    if raw["validator"] == "truth_table" and not raw.get("expected"):
        raise ValueError(f"{path}: validator 'truth_table' requires 'expected'")

    return Pattern(
        name=raw["name"],
        cls=raw["cls"],
        level=raw["level"],
        nodes=tuple(sorted(nodes.items())),
        edges=tuple(PatternEdge(e["from"], e["to"], tuple(e["pins"]))
                    for e in raw["edges"]),
        inputs=tuple(InputPort(p["name"],
                               tuple((n, int(pin)) for n, pin in p["attach"]))
                     for p in raw["ports"]["inputs"]),
        outputs=tuple(OutputPort(p["name"], p["node"])
                      for p in raw["ports"]["outputs"]),
        validator=raw["validator"],
        expected=raw.get("expected"),
        allow_shared_inputs=raw.get("allow_shared_inputs", False),
    )


def load_library(directory: Optional[str] = None) -> List[Pattern]:
    """Load every *.json in the patterns directory, sorted by (level, name)."""
    if directory is None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "patterns")
    patterns = [load_pattern(p)
                for p in sorted(glob.glob(os.path.join(directory, "*.json")))]
    return sorted(patterns, key=lambda p: (p.level, p.name))
