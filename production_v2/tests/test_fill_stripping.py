"""Fill-stripped rendering: the helpers behind coloured-symbol detection.

A solid fill costs the detector two different things.  It hides the inversion
bubble, so a NOR reads as XNOR at 0.96 confidence; and it can hide a gate
outright -- an SR latch drawn with orange bodies gave up only two of its four
gates, both misclassified, while the stripped rendering found all four with the
right classes.

These cover the pure geometry and image helpers, which need no model.
"""
import numpy as np
import pytest

from predict import _fill_frac, _iou, _strip_fill, FILL_STROKE_MAX


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


class TestIou:
    def test_identical_boxes(self):
        assert _iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert _iou(_box(0, 0, 10, 10), _box(50, 50, 10, 10)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert _iou(_box(0, 0, 10, 10), _box(10, 0, 10, 10)) == 0.0

    def test_half_overlap(self):
        # 10x10 boxes sharing a 5x10 strip -> 50 / 150
        assert _iou(_box(0, 0, 10, 10), _box(5, 0, 10, 10)) == pytest.approx(1 / 3)

    def test_is_symmetric(self):
        a, b = _box(0, 0, 10, 10), _box(4, 4, 10, 10)
        assert _iou(a, b) == _iou(b, a)


class TestStripFill:
    def test_dark_strokes_survive(self):
        img = np.full((20, 20, 3), 255, np.uint8)
        img[5:15, 9:11] = 0                      # a black line
        out = _strip_fill(img)
        assert (out < 128).any(), "stroke was erased"

    def test_midtone_fill_is_discarded(self):
        """Otsu keeps a mid-tone fill as ink, which turns a filled gate into a
        solid blob; only dark strokes should survive."""
        img = np.full((20, 20, 3), 255, np.uint8)
        img[4:16, 4:16] = FILL_STROKE_MAX + 60   # a pale body, well above the cut
        out = _strip_fill(img)
        assert (out > 200).all(), "fill was kept as ink"

    def test_outline_survives_its_own_fill(self):
        img = np.full((30, 30, 3), 255, np.uint8)
        img[5:25, 5:25] = 200                    # pale body
        img[5:25, 5:7] = 20                      # dark outline down one side
        out = _strip_fill(img)
        assert (out[5:25, 5:7] < 128).all()      # outline kept
        assert (out[10:20, 12:22] > 200).all()   # body dropped


class TestFillFrac:
    def test_plain_white_is_not_filled(self):
        img = np.full((40, 40, 3), 255, np.uint8)
        assert _fill_frac(img, _box(0, 0, 40, 40)) == 0.0

    def test_greyscale_line_art_is_not_filled(self):
        """Anti-aliased hand-drawn strokes are grey, not coloured.  Counting a
        grey band as fill flipped 9 corpus images' gate classes."""
        img = np.full((40, 40, 3), 255, np.uint8)
        img[10:30, 10:30] = 128
        assert _fill_frac(img, _box(0, 0, 40, 40)) == 0.0

    def test_saturated_colour_is_filled(self):
        img = np.full((40, 40, 3), 255, np.uint8)
        img[0:40, 0:40] = (0, 140, 255)          # orange, BGR
        assert _fill_frac(img, _box(0, 0, 40, 40)) > 0.9

    def test_empty_box_is_safe(self):
        img = np.full((10, 10, 3), 255, np.uint8)
        assert _fill_frac(img, _box(50, 50, 5, 5)) == 0.0
