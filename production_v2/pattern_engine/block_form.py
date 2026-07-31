"""Recognition of BLOCK-FORM devices: a labelled box instead of gates.

Textbooks often draw a flip-flop as a rectangle with its name written inside
and its pins labelled around the edge.  There are no gate symbols at all, so
the YOLO detector finds nothing and the whole gate/wire pipeline produces an
empty graph -- the device is unreadable to Part 1 even though it is the
easiest case for a human, because the answer is written on it.

This module takes the other route: find the box, read what is written inside,
and emit the corresponding macro node directly.  It is deliberately separate
from the gate path -- nothing here touches predict.py's detection or tracing.

Detection uses the box's ENCLOSED INTERIOR rather than its outline.  An
outline contour merges with the pin stubs poking out of it (measured: one
contour, 15 vertices, 53% rectangular fill), whereas the interior is a clean
enclosed white region that does not touch the image border.  On a real block
symbol that region fills 94% of its bounding box; the wiring loops of a
gate-level schematic reach only ~72%, which separates the two cleanly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Minimum interior-fill ratio for a region to count as a drawn box.
#: Measured: block symbol 0.94, gate-schematic wiring loops <= 0.72.
BLOCK_FILL_MIN = 0.85

#: Interior must be at least this fraction of the image, so stray enclosed
#: specks (the hole in an 'o', a bubble on a gate) are never boxes.
BLOCK_AREA_MIN_FRAC = 0.02

#: How far outside the box edge a pin label may sit, as a fraction of box size.
PIN_LABEL_MARGIN = 0.18

#: Device name -> (macro class, canonical inputs, canonical outputs).
#: Canonical pins are used when OCR cannot recover every label -- small edge
#: text is the first thing to be missed, and a flip-flop's pin set is fixed by
#: its type, so guessing it from the type is safer than dropping pins.
DEVICE_TABLE: Dict[str, Tuple[str, List[str], List[str]]] = {
    "SR":  ("SRFF_BLOCK", ["S", "R", "CLK"],      ["Q", "Qbar"]),
    "JK":  ("JKFF_BLOCK", ["J", "K", "CLK"],      ["Q", "Qbar"]),
    "D":   ("DFF_BLOCK",  ["D", "CLK"],           ["Q", "Qbar"]),
    "T":   ("TFF_BLOCK",  ["T", "CLK"],           ["Q", "Qbar"]),
}

#: Level-sensitive counterparts, selected when the box says LATCH.
#: A latch is transparent while its enable is high; a flip-flop samples on an
#: edge.  Mapping both onto the flip-flop class emitted `always @(posedge CLK)`
#: for a box clearly labelled "D LATCH" -- different hardware, no warning.
#: The enable pin is named EN rather than CLK for the same reason.
LATCH_TABLE: Dict[str, Tuple[str, List[str], List[str]]] = {
    "SR": ("SRLATCH_BLOCK", ["S", "R", "EN"], ["Q", "Qbar"]),
    "D":  ("DLATCH_BLOCK",  ["D", "EN"],      ["Q", "Qbar"]),
}

#: Async control pins, recognised wherever they appear on the box.
ASYNC_PINS = {"PR", "PRE", "PRESET", "SET", "CLR", "CLEAR", "RST", "RESET"}

#: Tokens that name the device kind rather than a pin.
_KIND_WORDS = {"FLIP", "FLOP", "FLIPFLOP", "LATCH", "FF"}


@dataclass
class Block:
    """One recognised block-form device."""
    cls:     str
    bbox:    Tuple[int, int, int, int]          # x, y, w, h
    name:    str                                 # e.g. "SR FLIP FLOP"
    inputs:  List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def to_graph_node(self) -> Dict[str, Any]:
        return {"cls": self.cls, "inputs": list(self.inputs),
                "outputs": list(self.outputs), "block": self.name}


def find_box_interiors(gray, fill_min: float = BLOCK_FILL_MIN,
                       area_min_frac: float = BLOCK_AREA_MIN_FRAC
                       ) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes of enclosed, near-rectangular white regions.

    Takes a grayscale image; returns (x, y, w, h) of each box INTERIOR.
    """
    import cv2
    import numpy as np

    h_img, w_img = gray.shape[:2]
    ink = (gray < 128).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(1 - ink, 4)

    out: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min_frac * h_img * w_img:
            continue
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            continue                       # open region, not an enclosed box
        if w * h == 0 or area / float(w * h) < fill_min:
            continue                       # not rectangular enough
        out.append((int(x), int(y), int(w), int(h)))
    out.sort(key=lambda b: -b[2] * b[3])
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9']", "", text.upper())


def classify_device(tokens: List[str]) -> Optional[Tuple[str, str]]:
    """Map the words written inside a box to (macro class, device name).

    Requires a KIND word ("FLIP FLOP" / "LATCH") alongside the type letters:
    an "S" and an "R" floating in a box are pin labels, not a device name, and
    without this a box of unrelated text would be read as a flip-flop.
    """
    norm = [_norm(t) for t in tokens]
    joined = "".join(norm)
    if not any(k in joined for k in _KIND_WORDS):
        return None
    # "D-type flip-flop" normalises to DTYPE, so a bare letter key would never
    # be seen as a token.  Accept the "<letter>TYPE" spelling as well.
    variants = set(norm)
    for t in norm:
        if t.endswith("TYPE") and len(t) > 4:
            variants.add(t[:-4])
    is_latch = "LATCH" in joined
    for key in ("JK", "SR", "D", "T"):
        if key in variants or (len(key) == 2 and key in joined):
            # A latch is level-sensitive, so it gets its own class where one
            # exists.  JK and T have no meaningful level-sensitive form (both
            # depend on the previous state, which a transparent latch cannot
            # hold), so they stay edge-triggered even if the drawing says
            # "latch".
            table = LATCH_TABLE if (is_latch and key in LATCH_TABLE) else DEVICE_TABLE
            return table[key][0], f"{key} {'LATCH' if is_latch else 'FLIP FLOP'}"
    return None


def pins_for(cls: str) -> Optional[Tuple[List[str], List[str]]]:
    """Canonical (inputs, outputs) for a macro class, latch or flip-flop."""
    for table in (DEVICE_TABLE, LATCH_TABLE):
        for _key, (c, ins, outs) in table.items():
            if c == cls:
                return list(ins), list(outs)
    return None


#: Output pin spellings, including the ways OCR renders a bar over Q.
_Q_TOKENS = {"Q", "Q'", "QBAR", "QN", "QB", "NQ"}

#: Data-pin signatures that identify a device on their own, most specific
#: first: a box carrying J and K is a JK flip-flop whether or not anyone wrote
#: "flip-flop" inside it.
_PIN_SIGNATURES: List[Tuple[str, frozenset]] = [
    ("JK", frozenset({"J", "K"})),
    ("SR", frozenset({"S", "R"})),
    ("D",  frozenset({"D"})),
    ("T",  frozenset({"T"})),
]


def classify_by_pins(tokens: List[str]) -> Optional[Tuple[str, str]]:
    """Identify a device from its PIN LABELS when no device name is written.

    Textbook symbols often label only the pins and mark the clock with a
    triangle rather than the word CLK, so there is no name to read -- a T
    flip-flop came back as just T, Q and Q'.  The pin set is still decisive.

    Requires a Q-like output so that arbitrary boxed text cannot qualify, and
    is only ever consulted after the device-name reading fails.
    """
    norm = {_norm(t) for t in tokens}
    norm.discard("")
    if not (norm & _Q_TOKENS):
        return None
    for key, pins in _PIN_SIGNATURES:
        if pins <= norm:
            return DEVICE_TABLE[key][0], f"{key} FLIP FLOP"
    return None


def _side_of(cx: float, cy: float, box: Tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    dl, dr = abs(cx - x), abs(cx - (x + w))
    dt, db = abs(cy - y), abs(cy - (y + h))
    return ["left", "right", "top", "bottom"][
        [dl, dr, dt, db].index(min(dl, dr, dt, db))]


def blocks_from_ocr(gray, ocr_results, fill_min: float = BLOCK_FILL_MIN,
                    reocr=None) -> List[Block]:
    """Build Block records from a grayscale image plus OCR output.

    *ocr_results* is EasyOCR's shape: a list of (polygon, text, confidence).
    Kept as a parameter rather than run here so this module has no OCR or
    model dependency and stays unit-testable.

    *reocr* is an optional callable ``(x, y, w, h) -> ocr_results`` used to read
    the box region again at higher magnification.  Whole-image OCR routinely
    misses the small, low-contrast pin letters written on a block symbol -- a
    T flip-flop came back as just ['INPUT', 'CLK'], with T, Q and Q' all
    missed, which left nothing to classify.  Re-reading only the box, enlarged,
    recovers them.  Results from both passes are merged.
    """
    blocks: List[Block] = []
    for box in find_box_interiors(gray, fill_min=fill_min):
        x, y, w, h = box
        mx, my = PIN_LABEL_MARGIN * w, PIN_LABEL_MARGIN * h

        results = list(ocr_results)
        if reocr is not None:
            try:
                results = results + list(reocr(x, y, w, h))
            except Exception:
                pass

        inside_words: List[str] = []
        edge_labels: List[Tuple[str, str]] = []          # (side, text)
        for poly, text, _conf in results:
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            if not (x - mx <= cx <= x + w + mx and y - my <= cy <= y + h + my):
                continue
            token = _norm(text)
            if not token:
                continue
            near_edge = (cx < x + mx or cx > x + w - mx
                         or cy < y + my or cy > y + h - my)
            if near_edge:
                edge_labels.append((_side_of(cx, cy, box), token))
            else:
                inside_words.append(token)

        # A device word sitting near an edge still names the device.  Falling
        # back to the pin signature covers symbols that carry no name at all.
        all_tokens = inside_words + [t for _s, t in edge_labels]
        device = (classify_device(inside_words)
                  or classify_device(all_tokens)
                  or classify_by_pins(all_tokens))
        if device is None:
            continue
        cls, name = device
        key = name.split()[0]
        # Pins come from whichever table the class actually belongs to, so a
        # latch gets EN rather than CLK.
        canon = pins_for(cls) or (DEVICE_TABLE[key][1], DEVICE_TABLE[key][2])
        canon_in, canon_out = canon

        found_async = [t for _s, t in edge_labels if t in ASYNC_PINS]
        inputs = list(canon_in) + [t for t in dict.fromkeys(found_async)]
        blocks.append(Block(cls=cls, bbox=box, name=name,
                            inputs=inputs, outputs=list(canon_out)))
    return blocks


def graph_from_blocks(blocks: List[Block]) -> Dict[str, Dict[str, Any]]:
    """A gate-graph dict containing one node per recognised block."""
    graph: Dict[str, Dict[str, Any]] = {}
    for i, b in enumerate(blocks, 1):
        graph[f"B{i}"] = b.to_graph_node()
    return graph
