"""Block-form (labelled box) device recognition.

Uses synthetic OCR output so the tests need neither EasyOCR nor a real image.
"""
import numpy as np
import pytest

from pattern_engine.block_form import (DEVICE_TABLE, blocks_from_ocr,
                                       classify_device, find_box_interiors,
                                       graph_from_blocks)


def _canvas(w=400, h=300, box=(60, 40, 260, 200)):
    """White image with a black rectangle outline."""
    img = np.full((h, w), 255, np.uint8)
    x, y, bw, bh = box
    img[y:y + bh, x:x + 3] = 0
    img[y:y + bh, x + bw - 3:x + bw] = 0
    img[y:y + 3, x:x + bw] = 0
    img[y + bh - 3:y + bh, x:x + bw] = 0
    return img


def _ocr(items):
    """(text, cx, cy) -> EasyOCR-shaped results."""
    out = []
    for text, cx, cy in items:
        poly = [[cx - 10, cy - 6], [cx + 10, cy - 6],
                [cx + 10, cy + 6], [cx - 10, cy + 6]]
        out.append((poly, text, 0.99))
    return out


def test_finds_the_box_interior():
    boxes = find_box_interiors(_canvas())
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert 60 <= x <= 70 and 40 <= y <= 50


def test_ignores_regions_touching_the_image_border():
    """An open drawing has no enclosed interior and must not be a box."""
    img = np.full((200, 200), 255, np.uint8)
    img[:, 100:103] = 0                      # a bare line, nothing enclosed
    assert find_box_interiors(img) == []


def test_rejects_non_rectangular_enclosures():
    """Gate-schematic wiring loops enclose area but are not rectangles.

    Measured: a real block symbol fills 94% of its bounding box, wiring loops
    reach only ~72%, which is what BLOCK_FILL_MIN separates.
    """
    img = np.full((300, 300), 255, np.uint8)
    tri = np.array([[40, 260], [260, 260], [150, 40]])
    import cv2
    cv2.polylines(img, [tri], True, 0, 3)
    assert find_box_interiors(img) == []     # triangle fills ~50%


def test_classifies_sr_flip_flop():
    assert classify_device(["SR", "FLIP", "FLOP"])[0] == "SRFF_BLOCK"


def test_classifies_jk_and_d_and_t():
    assert classify_device(["JK", "FLIP", "FLOP"])[0] == "JKFF_BLOCK"
    assert classify_device(["D", "LATCH"])[0] == "DFF_BLOCK"
    assert classify_device(["T", "FLIP", "FLOP"])[0] == "TFF_BLOCK"


def test_latch_and_flipflop_are_named_differently():
    assert "LATCH" in classify_device(["D", "LATCH"])[1]
    assert "FLIP FLOP" in classify_device(["SR", "FLIP", "FLOP"])[1]


def test_bare_pin_letters_are_not_a_device():
    """An S and an R floating in a box are pin labels, not a device name."""
    assert classify_device(["S", "R", "Q"]) is None
    assert classify_device(["HELLO", "WORLD"]) is None


def test_end_to_end_block_with_async_pins():
    gray = _canvas()
    res = _ocr([("SR", 190, 110), ("FLIP", 190, 140), ("FLOP", 190, 170),
                ("PR", 190, 50), ("CLR", 190, 232), ("CLK", 75, 140)])
    blocks = blocks_from_ocr(gray, res)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.cls == "SRFF_BLOCK"
    assert b.name == "SR FLIP FLOP"
    assert {"S", "R", "CLK"} <= set(b.inputs)
    assert "PR" in b.inputs and "CLR" in b.inputs      # async pins recovered
    assert b.outputs == ["Q", "Qbar"]


def test_box_without_a_device_name_is_ignored():
    """A plain rectangle of unrelated text must not become a flip-flop."""
    gray = _canvas()
    assert blocks_from_ocr(gray, _ocr([("NOTES", 190, 140)])) == []


def test_graph_node_shape_matches_the_gate_graph_contract():
    gray = _canvas()
    res = _ocr([("JK", 190, 110), ("FLIP", 190, 140), ("FLOP", 190, 170)])
    g = graph_from_blocks(blocks_from_ocr(gray, res))
    assert list(g) == ["B1"]
    node = g["B1"]
    assert node["cls"] == "JKFF_BLOCK"
    assert set(node) >= {"cls", "inputs", "outputs"}
    assert node["inputs"] == DEVICE_TABLE["JK"][1]
    assert node["outputs"] == DEVICE_TABLE["JK"][2]
