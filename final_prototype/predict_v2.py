"""
predict_v2.py — Redesigned schematic-to-netlist pipeline.

Design goals (vs predict.py):
  1. Pin-first wire tracing — pin positions are detected from where wires
     ACTUALLY touch the gate bbox, not from fixed fractional offsets.
  2. Robust crossing handling — junction dots are detected by circularity
     AND size; 4-way no-dot intersections are broken and reconnected
     direction-preservingly via Union-Find with widened scans.
  3. Multi-input gate support — gates can have N inputs (N detected from pins).
  4. Floating-wire rescue — wires that nearly touch a pin are connected
     by a small dilation pass.
  5. Heavy debug imagery — every intermediate stage written to disk.

Stages:  detect_gates → preprocess → trace_wires → detect_pins
       → assign_wires → build_graph → generate_netlist
"""
from __future__ import annotations

import cv2
import numpy as np
import sys
import os
import logging
import glob
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("predict_v2")

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = ["AND", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]

GATE_INPUT_COUNT_DEFAULT: Dict[str, int] = {
    "AND": 2, "NAND": 2, "NOR": 2, "OR": 2,
    "XOR": 2, "XNOR": 2, "NOT": 1, "BUF": 1,
}

YOLO_CONF              = 0.25
MIN_GATE_AREA          = 600
MIN_WIRE_AREA          = 12

# Crossing detection
ARM_MIN_REACH          = 7     # arm must extend ≥7 px from junction centre to count
BREAK_KERNEL_SIZE      = 9     # erosion kernel at crossing erase sites
CROSS_SCAN_DIST        = 40    # px to scan from crossing centre to find a wire label

# Geometric pin assignment
PIN_SEARCH_NEAR        = 24    # narrow first-pass X range (right at gate edge)
PIN_SEARCH_MID         = 55    # medium second-pass X range
PIN_SEARCH_DIST        = 90    # wide fallback X range
MIN_PIN_WIRE_PIXELS    = 15    # minimum wire pixels in patch to accept assignment

# Theoretical input-pin Y fractions per gate class (from gate top)
GATE_PIN_FRACS: Dict[str, Dict] = {
    "AND":  {"in": [0.30, 0.70], "out": 0.50},
    "NAND": {"in": [0.30, 0.70], "out": 0.50},
    "OR":   {"in": [0.30, 0.70], "out": 0.50},
    "NOR":  {"in": [0.30, 0.70], "out": 0.50},
    "XOR":  {"in": [0.30, 0.70], "out": 0.50},
    "XNOR": {"in": [0.30, 0.70], "out": 0.50},
    "NOT":  {"in": [0.50],       "out": 0.50},
    "BUF":  {"in": [0.50],       "out": 0.50},
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Pin:
    side: str           # "in" or "out"
    x: int
    y: int
    gate_id: str

@dataclass
class CircuitResult:
    netlist: str
    equations: str
    graph: Dict[str, Any]
    global_inputs: set
    global_outputs: set
    gates: List[Dict]
    warnings: List[str] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    debug_dir: Optional[str] = None


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj: return
        if self.rank[ri] < self.rank[rj]: ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]: self.rank[ri] += 1


# ── Model loading ────────────────────────────────────────────────────────────

def find_best_model(search_root: Optional[str] = None) -> Optional[str]:
    if search_root is None:
        search_root = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(search_root, "**/best.pt"), recursive=True)
    if not candidates: return None
    # Prefer roboflow_gates if present (best mAP on this task)
    rf = [c for c in candidates if "roboflow_gates" in c.replace("\\", "/")]
    if rf: return rf[0]
    return max(candidates, key=os.path.getctime)


# ── Stage 1: Gate detection ──────────────────────────────────────────────────

def detect_gates(image_path: str, model: YOLO) -> Tuple[List[Dict], np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = model(image_path, conf=YOLO_CONF, verbose=False)[0]

    boxes: List[Dict] = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"UNK_{cls_id}"
        boxes.append({"id": f"Gate_{i}", "cls": cls_name, "conf": conf,
                      "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

    boxes = [b for b in boxes if b['w'] * b['h'] >= MIN_GATE_AREA]

    # Suppress overlapping detections (same gate detected twice as different classes)
    boxes = _suppress_overlapping(boxes)

    if not boxes:
        raise ValueError("No logic gates detected in the image.")

    boxes.sort(key=lambda b: (b["x"], b["y"]))
    log.info("Detected %d gate(s): %s", len(boxes),
             ", ".join(f"{b['cls']}({b['conf']:.2f})" for b in boxes))
    return boxes, img


def _suppress_overlapping(boxes: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    if not boxes: return boxes
    boxes_sorted = sorted(boxes, key=lambda b: -b['conf'])
    keep: List[Dict] = []
    for b in boxes_sorted:
        ok = True
        for k in keep:
            if _iou(b, k) > iou_thresh:
                ok = False; break
        if ok: keep.append(b)
    return keep


def _iou(a: Dict, b: Dict) -> float:
    ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
    bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
    inter_x1, inter_y1 = max(a['x'], b['x']), max(a['y'], b['y'])
    inter_x2, inter_y2 = min(ax2, bx2),       min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter  = iw * ih
    union  = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / union if union > 0 else 0.0


# ── Stage 2: Preprocess (binarize, isolate wires) ────────────────────────────

def preprocess(img: np.ndarray, boxes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wire_binary, dot_mask)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    adapt   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    raw     = cv2.bitwise_or(otsu, dark)
    raw     = cv2.bitwise_or(raw, adapt)

    # Erase gate bodies on raw (so dots near gates aren't picked up)
    raw_for_dots = raw.copy()
    for b in boxes:
        pad = 6
        cv2.rectangle(raw_for_dots,
                      (max(0, b['x'] - pad),         max(0, b['y'] - pad)),
                      (min(w, b['x'] + b['w'] + pad), min(h, b['y'] + b['h'] + pad)),
                      0, -1)
    raw_for_dots = _erase_text(raw_for_dots)

    # Detect dots on the RAW threshold (before any morphology) — dots are
    # easiest to spot here because they're at full thickness.
    dot_mask = _detect_dots(raw_for_dots)

    # Now build the wire binary that subsequent stages will use.
    binary = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for b in boxes:
        pad = 6
        cv2.rectangle(binary,
                      (max(0, b['x'] - pad),         max(0, b['y'] - pad)),
                      (min(w, b['x'] + b['w'] + pad), min(h, b['y'] + b['h'] + pad)),
                      0, -1)
    binary = _erase_text(binary)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    return binary, dot_mask


def _erase_text(binary: np.ndarray) -> np.ndarray:
    """Remove small connected components that look like glyphs (low aspect, small)."""
    out = binary.copy()
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60 and max(w, h) < 18:
            out[lbl == i] = 0
    return out


def _detect_dots(binary: np.ndarray) -> np.ndarray:
    """Find junction dots — local thickenings in the wire image.

    Dots are usually drawn as filled circles roughly 1.5-2x the wire thickness.
    We detect them via distance transform: pixels whose distance to background
    exceeds the typical wire thickness are dot candidates.
    """
    if binary.sum() == 0:
        return np.zeros_like(binary)

    # Distance from each foreground pixel to nearest background pixel
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    # Typical wire half-thickness ≈ median of distance transform on skeleton
    skel = _skeletonize(binary)
    skel_dt = dt[skel > 0]
    if len(skel_dt) == 0:
        wire_thickness = 1.0
    else:
        wire_thickness = float(np.median(skel_dt))

    # Threshold: pixels at least 1.6× wire half-thickness ⇒ likely dot core
    thresh_val = max(2.0, wire_thickness * 1.6 + 0.5)
    dot_core = (dt >= thresh_val).astype(np.uint8)

    # Filter spurious gate-corner blobs by size (real dots are small & round)
    dot_mask_out = np.zeros_like(binary)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(dot_core)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 1 or area > 80:    continue
        if max(w, h) > 12:           continue
        cv2.circle(dot_mask_out, (int(cents[i][0]), int(cents[i][1])), 9, 255, -1)
    return dot_mask_out


# ── Stage 3: Wire tracing with crossing handling ────────────────────────────

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen-like thinning to 1px skeleton via OpenCV ximgproc fallback."""
    try:
        return cv2.ximgproc.thinning(binary,
                                      thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        # Iterative morphological skeleton
        skel = np.zeros_like(binary)
        img  = binary.copy()
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, elem)
            opened = cv2.dilate(eroded, elem)
            temp   = cv2.subtract(img, opened)
            skel   = cv2.bitwise_or(skel, temp)
            img    = eroded.copy()
            if cv2.countNonZero(img) == 0: break
        return skel


def _arm_reaches(skel: np.ndarray, cy: int, cx: int,
                 near: int = 2, far: int = 22) -> Tuple[int, int, int, int]:
    """Measure how far the skeleton extends outward in each cardinal direction.

    Returns (L_reach, R_reach, T_reach, B_reach) in pixels.
    0 means no skeleton pixel found in that direction within `far` pixels.
    Scanning backward from far→near gives the LONGEST reach first, so the
    first hit is the most distant skeleton pixel in that direction.
    """
    h, w = skel.shape

    def reach(dx: int, dy: int) -> int:
        for d in range(far, near - 1, -1):
            x, y = cx + dx * d, cy + dy * d
            if not (0 <= x < w and 0 <= y < h):
                continue
            if dx != 0:
                region = skel[max(0, y - 1):min(h, y + 2), x]
            else:
                region = skel[y, max(0, x - 1):min(w, x + 2)]
            if region.max() > 0:
                return d
        return 0

    return reach(-1, 0), reach(1, 0), reach(0, -1), reach(0, 1)


def _propagate_labels(skel_labels: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """BFS-flood skeleton wire labels into adjacent binary pixels.

    Each binary pixel is assigned the label of the nearest labeled skeleton
    pixel (nearest in the BFS sense — walking only over foreground binary
    pixels).  This heals small rendering gaps that would otherwise split one
    physical wire into multiple binary components.
    """
    h, w = binary.shape
    out = np.zeros((h, w), dtype=np.int32)

    queue: deque = deque()
    visited = np.zeros((h, w), dtype=bool)

    # Seed queue with every labeled skeleton pixel
    ys, xs = np.where(skel_labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lbl = int(skel_labels[y, x])
        out[y, x] = lbl
        visited[y, x] = True
        queue.append((y, x))

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        y, x = queue.popleft()
        lbl  = out[y, x]
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if visited[ny, nx] or binary[ny, nx] == 0:
                continue
            visited[ny, nx] = True
            out[ny, nx] = lbl
            queue.append((ny, nx))

    return out


def trace_wires(binary: np.ndarray, dot_mask: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Skeleton-CCL wire tracer with crossing detection and label propagation.

    Steps:
    1. Skeletonize → find junction pixels (≥3 neighbours)
    2. Group close junction pixels into "junction cores"
    3. Measure arm reaches in each cardinal direction per core
    4. Classify as 4-way crossing only if ALL 4 arms extend ≥ ARM_MIN_REACH px
       AND no junction dot is present nearby
    5. Erase confirmed crossing cores from the SKELETON (binary untouched)
    6. Run CCL on the erased skeleton → each segment is a wire component
    7. Re-link L↔R and T↔B across each crossing via Union-Find
    8. Propagate skeleton labels into the full binary (heals micro-gaps)
    9. Compact label range and return
    """
    h, w = binary.shape

    skel     = _skeletonize(binary)
    skel_bin = (skel > 0).astype(np.uint8)

    # Find junction pixels: skeleton pixels with ≥3 skeleton neighbours
    kernel      = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neigh_count = cv2.filter2D(skel_bin, -1, kernel)
    junction_pix = ((skel_bin == 1) & (neigh_count >= 3)).astype(np.uint8) * 255

    # Group spatially close junction pixels into one core (dilate + CCL)
    junction_dilated = cv2.dilate(junction_pix, np.ones((5, 5), np.uint8))
    n_cores, core_lbl = cv2.connectedComponents(junction_dilated)

    crossing_cores: List[Tuple[int, int, int, Tuple[int,int,int,int]]] = []
    crossing_mask = np.zeros_like(skel_bin, dtype=np.uint8)

    for c in range(1, n_cores):
        ys, xs = np.where(core_lbl == c)
        if len(xs) == 0:
            continue
        cx_c, cy_c = int(np.mean(xs)), int(np.mean(ys))

        # Determine how far the core extends so arm detection starts outside it
        core_radius = max(
            int(ys.max()) - int(ys.min()) + 1,
            int(xs.max()) - int(xs.min()) + 1,
        ) // 2
        arm_near = max(core_radius + 3, 4)   # start scan just outside the core
        arm_far  = max(arm_near + 14, 22)

        L_r, R_r, T_r, B_r = _arm_reaches(skel_bin, cy_c, cx_c,
                                            near=arm_near, far=arm_far)

        # Dot check: scan the bounding box of the junction core (±20 px margin)
        # to handle slight spatial offset between junction centroid and dot centroid.
        y0 = max(0, int(ys.min()) - 20);  y1 = min(h, int(ys.max()) + 21)
        x0 = max(0, int(xs.min()) - 20);  x1 = min(w, int(xs.max()) + 21)
        local_dot = (dot_mask[y0:y1, x0:x1].max() > 0)

        # Only treat as a wire crossing when all 4 arms are substantial AND no dot
        if (not local_dot
                and L_r >= ARM_MIN_REACH and R_r >= ARM_MIN_REACH
                and T_r >= ARM_MIN_REACH and B_r >= ARM_MIN_REACH):
            crossing_mask[ys, xs] = 1
            crossing_cores.append(
                (c, cx_c, cy_c,
                 (int(L_r > 0), int(R_r > 0), int(T_r > 0), int(B_r > 0))))

    log.info("Crossings detected: %d", len(crossing_cores))

    # Erase confirmed crossing regions from SKELETON only (binary stays intact)
    cross_dilate = cv2.dilate(crossing_mask,
                               np.ones((BREAK_KERNEL_SIZE, BREAK_KERNEL_SIZE),
                                       np.uint8))
    skel_split = skel_bin.copy()
    skel_split[cross_dilate > 0] = 0

    # CCL on the split skeleton → each segment becomes a candidate wire
    n, skel_labels, stats, _ = cv2.connectedComponentsWithStats(
        skel_split.astype(np.uint8))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 2:   # single stray pixels
            skel_labels[skel_labels == i] = 0

    # Re-link wire segments across crossings: L↔R and T↔B
    uf = UnionFind(n)
    for (cid, cx_c, cy_c, (L, R, T, B)) in crossing_cores:
        l_id = _scan_labels(cx_c, cy_c, -1,  0, skel_labels, h, w) if L else 0
        r_id = _scan_labels(cx_c, cy_c,  1,  0, skel_labels, h, w) if R else 0
        t_id = _scan_labels(cx_c, cy_c,  0, -1, skel_labels, h, w) if T else 0
        b_id = _scan_labels(cx_c, cy_c,  0,  1, skel_labels, h, w) if B else 0
        log.debug("  crossing @(%d,%d): L=%d R=%d T=%d B=%d",
                  cx_c, cy_c, l_id, r_id, t_id, b_id)
        if l_id and r_id: uf.union(l_id, r_id)
        if t_id and b_id: uf.union(t_id, b_id)

    # Apply Union-Find mapping to skeleton labels
    mapping     = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    skel_merged = mapping[skel_labels]

    # Propagate skeleton labels into the full binary (BFS over binary pixels)
    # This heals rendering micro-gaps that break binary connectivity but are
    # spanned by a continuous skeleton path.
    labels = _propagate_labels(skel_merged, binary)

    # Remove tiny noise components
    for lbl_id in np.unique(labels):
        if lbl_id == 0:
            continue
        if int(np.sum(labels == lbl_id)) < MIN_WIRE_AREA:
            labels[labels == lbl_id] = 0

    # Compact label range (remove Union-Find / propagation gaps)
    uniq = np.unique(labels[labels > 0])
    if uniq.size == 0:
        return labels, (cross_dilate * 255).astype(np.uint8)
    remap = np.zeros(int(labels.max()) + 2, dtype=np.int32)
    for new_id, old_id in enumerate(uniq, start=1):
        remap[int(old_id)] = new_id
    labels = remap[labels]

    return labels, (cross_dilate * 255).astype(np.uint8)


def _scan_labels(cx: int, cy: int, dx: int, dy: int,
                 labels: np.ndarray, h: int, w: int) -> int:
    """Scan from (cx,cy) in direction (dx,dy); return the first nonzero label found."""
    for d in range(2, CROSS_SCAN_DIST + 1):
        x, y = cx + dx * d, cy + dy * d
        if not (0 <= x < w and 0 <= y < h):
            break
        patch = labels[max(0, y - 3):min(h, y + 4),
                       max(0, x - 3):min(w, x + 4)]
        ids = patch[patch > 0]
        if ids.size:
            return int(np.bincount(ids).argmax())
    return 0


def _split_bus_wires(labels: np.ndarray) -> np.ndarray:
    """Break wires that are connected only through thin bridges (bus bars).

    A bus bar is a long narrow connector linking several horizontal stubs.
    After aggressive erosion its separate "thick cores" appear; we re-label
    each core and grow it back into the full wire region so each stub becomes
    its own net.
    """
    out      = labels.copy()
    next_lbl = int(out.max()) + 1

    for wid in np.unique(labels):
        if wid == 0:
            continue
        mask = (labels == wid).astype(np.uint8)
        area = int(mask.sum())
        if area < 150:
            continue   # too small to bother splitting

        # Try progressively less aggressive erosion until we find multiple cores
        split_found = False
        for n_iter in (3, 2):
            eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=n_iter)
            n_cores, core_lbl = cv2.connectedComponents(eroded)
            if n_cores > 2:          # background + ≥2 real cores
                split_found = True
                break

        if not split_found:
            continue

        # Grow each core back into unassigned mask pixels (competition dilation)
        assigned   = core_lbl.copy()          # 0=unassigned, 1..N=core labels
        remaining  = (mask > 0) & (assigned == 0)

        for _ in range(250):
            if not np.any(remaining):
                break
            prev_remaining = remaining.copy()
            for c in range(1, n_cores):
                c_mask  = (assigned == c).astype(np.uint8)
                dilated = cv2.dilate(c_mask, np.ones((3, 3), np.uint8))
                new_px  = dilated.astype(bool) & remaining
                assigned[new_px] = c
                remaining[new_px] = False
            if np.array_equal(remaining, prev_remaining):
                break   # no progress → stop

        # Assign any leftover bridge pixels to the nearest assigned neighbour
        leftover_ys, leftover_xs = np.where(remaining)
        for ly, lx in zip(leftover_ys, leftover_xs):
            patch = assigned[max(0, ly - 3):ly + 4, max(0, lx - 3):lx + 4]
            vals  = patch[patch > 0]
            if vals.size:
                assigned[ly, lx] = int(np.bincount(vals).argmax())

        # Rewrite out: core 1 keeps original label, cores 2..N get fresh labels
        out[labels == wid] = 0
        for c in range(1, n_cores):
            pixels_c = (assigned == c) & (mask > 0)
            lbl      = wid if c == 1 else next_lbl
            if c > 1:
                next_lbl += 1
            out[pixels_c] = lbl

    return out


# ── Stage 4+5: Geometric pin assignment ─────────────────────────────────────

def _geo_pin_coords(b: Dict) -> Tuple[List[Tuple[int,int]], Tuple[int,int]]:
    """Theoretical pin positions from gate bbox + class (no image scanning)."""
    gx, gy, gw, gh = b['x'], b['y'], b['w'], b['h']
    cfg     = GATE_PIN_FRACS.get(b['cls'], GATE_PIN_FRACS["AND"])
    in_pins = [(gx, int(gy + gh * f)) for f in cfg["in"]]
    out_pin = (gx + gw, int(gy + gh * cfg["out"]))
    return in_pins, out_pin


def assign_pins_geometric(
        boxes: List[Dict],
        labels: np.ndarray,
) -> Tuple[Dict[Tuple[str,str,int], int], Dict[str, Dict[str, List[Tuple[int,int]]]]]:
    """Assign wire labels to each gate pin using strictly directional windows.

    Input  pins → search LEFT  of gate.x only (never right)
    Output pin  → search RIGHT of gate.x+w only (never left)

    Three-tier X-range search (narrow → medium → wide) ensures the wire stub
    RIGHT AT THE GATE EDGE is preferred over a distant bus wire that may share
    the same Y band.  Y tolerance also widens progressively.
    """
    h, w = labels.shape
    assignment: Dict[Tuple[str,str,int], int] = {}
    pins_dict:  Dict[str, Dict[str, List[Tuple[int,int]]]] = {}

    for b in boxes:
        gid             = b['id']
        gx, gy, gw, gh = b['x'], b['y'], b['w'], b['h']
        in_pins, out_pin = _geo_pin_coords(b)
        pins_dict[gid]  = {"in": list(in_pins), "out": [out_pin]}

        # Y tolerances: tight → medium → loose
        y_tols = [max(6, gh // 7), max(10, gh // 4), max(16, gh // 3)]

        # X search ranges for INPUT pins (right-biased: gate edge first)
        # Gate region is erased ~6 px inward, so wire ends ≈ gx-6
        x_ranges_in = [
            (max(0, gx - PIN_SEARCH_NEAR), gx + 6),    # narrow — right at gate
            (max(0, gx - PIN_SEARCH_MID),  gx + 6),    # medium
            (max(0, gx - PIN_SEARCH_DIST), gx + 6),    # wide fallback
        ]

        for idx, (_, py) in enumerate(in_pins):
            wid = 0
            for x_lo, x_hi in x_ranges_in:
                for ytol in y_tols:
                    patch = labels[max(0, py - ytol):min(h, py + ytol),
                                   x_lo:min(w, x_hi)]
                    ids   = patch[patch > 0]
                    if ids.size >= MIN_PIN_WIRE_PIXELS:
                        wid = int(np.bincount(ids).argmax())
                        break
                if wid:
                    break
            assignment[(gid, 'in', idx)] = wid

        # X search for OUTPUT pin (right of gate)
        x_lo_out = gx + gw - 6
        x_ranges_out = [
            (x_lo_out, min(w, gx + gw + PIN_SEARCH_NEAR)),
            (x_lo_out, min(w, gx + gw + PIN_SEARCH_MID)),
            (x_lo_out, min(w, gx + gw + PIN_SEARCH_DIST)),
        ]
        py_out = out_pin[1]
        wid    = 0
        for x_lo, x_hi in x_ranges_out:
            for ytol in y_tols:
                patch = labels[max(0, py_out - ytol):min(h, py_out + ytol),
                               x_lo:x_hi]
                ids   = patch[patch > 0]
                if ids.size >= MIN_PIN_WIRE_PIXELS:
                    wid = int(np.bincount(ids).argmax())
                    break
            if wid:
                break
        assignment[(gid, 'out', 0)] = wid

    return assignment, pins_dict


# ── Stage 6: Build dependency graph ─────────────────────────────────────────

def build_graph(boxes: List[Dict],
                assignment: Dict[Tuple[str, str, int], int]
                ) -> Tuple[Dict[str, Dict], set, set, List[str]]:
    """Build a gate dependency graph from the wire-to-pin assignment.

    Each wire label is classified as:
      • primary input  — touched by input pin(s) of gate(s), no gate output drives it
      • primary output — driven by a gate output, not consumed by any gate input
      • internal net   — driven by a gate output AND consumed by other gate input(s)
    """
    gate_ids = {b['id'] for b in boxes}
    cls_of   = {b['id']: b['cls'] for b in boxes}

    # Map wire label → gates that drive it (output) / consume it (input)
    producers: Dict[int, List[str]] = defaultdict(list)   # wid → [gate_id, ...]
    consumers: Dict[int, List[Tuple[str,int]]] = defaultdict(list)  # wid → [(gate_id, pin_idx)]

    for (gid, side, idx), wid in assignment.items():
        if wid == 0: continue
        if side == "out": producers[wid].append(gid)
        else:             consumers[wid].append((gid, idx))

    graph: Dict[str, Dict[str, Any]] = {
        gid: {"cls": cls_of[gid], "inputs": [], "outputs": []} for gid in gate_ids
    }
    global_inputs:  Set[str] = set()
    global_outputs: Set[str] = set()
    warnings: List[str]      = []

    in_letter_idx = 0
    out_idx       = 1
    pin_inputs: Dict[str, Dict[int, Any]] = {gid: {} for gid in gate_ids}

    def _next_in_name() -> str:
        nonlocal in_letter_idx
        s, n = "", in_letter_idx
        while True:
            s = chr(ord('A') + n % 26) + s
            n = n // 26 - 1
            if n < 0: break
        in_letter_idx += 1
        return s

    all_wires = set(producers.keys()) | set(consumers.keys())
    for wid in all_wires:
        srcs = producers.get(wid, [])
        dsts = consumers.get(wid, [])

        # De-duplicate: same gate appearing as both producer AND consumer of the
        # same wire is a bbox-artefact — treat the wire as produced by the gate only.
        real_dsts = [(gid, idx) for (gid, idx) in dsts if gid not in srcs]

        if not srcs and dsts:
            name = _next_in_name()
            global_inputs.add(name)
            for (gid, idx) in dsts:
                pin_inputs[gid][idx] = name

        elif srcs and not real_dsts:
            for sgid in set(srcs):
                name = f"OUT_{out_idx}"; out_idx += 1
                graph[sgid]["outputs"].append(name)
                global_outputs.add(name)

        elif srcs and real_dsts:
            for sgid in set(srcs):
                for (dgid, didx) in real_dsts:
                    pin_inputs[dgid][didx] = sgid

    for gid in gate_ids:
        graph[gid]["inputs"] = [pin_inputs[gid][k]
                                 for k in sorted(pin_inputs[gid].keys())]

    # Validate and clamp
    for b in boxes:
        node  = graph[b['id']]
        n     = len(node["inputs"])
        cls   = b['cls']
        if cls in ("NOT", "BUF"):
            if n == 0: warnings.append(f"{b['id']} ({cls}): no input detected")
            elif n > 1: node["inputs"] = node["inputs"][:1]
        else:
            if n == 0:   warnings.append(f"{b['id']} ({cls}): no inputs detected")
            elif n == 1: warnings.append(f"{b['id']} ({cls}): only 1 input found (expected ≥2)")

    return graph, global_inputs, global_outputs, warnings


# ── Stage 7: Topological order + netlist generation ────────────────────────

def _topo_order(boxes: List[Dict], graph: Dict[str, Dict]) -> List[str]:
    gate_ids = {b['id'] for b in boxes}
    in_deg = {b['id']: 0 for b in boxes}
    children: Dict[str, List[str]] = defaultdict(list)
    for b in boxes:
        for inp in graph[b['id']]['inputs']:
            if inp in gate_ids:
                in_deg[b['id']] += 1
                children[inp].append(b['id'])
    q = deque(g for g in gate_ids if in_deg[g] == 0)
    order: List[str] = []
    while q:
        gid = q.popleft(); order.append(gid)
        for c in children[gid]:
            in_deg[c] -= 1
            if in_deg[c] == 0: q.append(c)
    leftover = [b['id'] for b in boxes if b['id'] not in set(order)]
    if leftover:
        log.warning("Cycle detected: appending %s", leftover)
        order.extend(leftover)
    return order


def generate_netlist(boxes: List[Dict], graph: Dict[str, Dict]
                     ) -> Tuple[str, str]:
    ordered = _topo_order(boxes, graph)
    name_map = {gid: f"G{i+1}" for i, gid in enumerate(ordered)}
    netlist_lines: List[str] = []
    final_eqs: Dict[str, str] = {}

    for gid in ordered:
        node = graph[gid]; gt = node['cls']; new_id = name_map[gid]
        resolved = [name_map.get(inp, inp) for inp in node['inputs']] or ["UNCONNECTED"]
        netlist_lines.append(f"{new_id} = {gt}({', '.join(resolved)});")

        eq_args = [final_eqs.get(inp, inp) for inp in node['inputs']]
        if not eq_args:                              eq_str = "UNCONNECTED"
        elif gt == "NOT":                            eq_str = f"(~{eq_args[0]})"
        elif gt == "BUF":                            eq_str = eq_args[0]
        elif gt == "NAND":                           eq_str = f"(~({' & '.join(eq_args)}))"
        elif gt == "NOR":                            eq_str = f"(~({' | '.join(eq_args)}))"
        elif gt == "XNOR":                           eq_str = f"(~({' ^ '.join(eq_args)}))"
        elif gt == "AND":                            eq_str = f"({' & '.join(eq_args)})"
        elif gt == "OR":                             eq_str = f"({' | '.join(eq_args)})"
        elif gt == "XOR":                            eq_str = f"({' ^ '.join(eq_args)})"
        else:                                        eq_str = f"{gt}({', '.join(eq_args)})"
        final_eqs[gid] = eq_str

        for out_name in node['outputs']:
            netlist_lines.append(f"assign {out_name} = {new_id};")

    out_eqs: List[str] = []
    for gid in ordered:
        for out_name in graph[gid]['outputs']:
            out_eqs.append(f"{out_name} = {final_eqs[gid]}")
    if not out_eqs:
        last = ordered[-1]
        out_eqs.append(f"Q = {final_eqs.get(last, '?')}")
        netlist_lines.append(f"assign Q = {name_map[last]};")
    return "\n".join(netlist_lines), "\n".join(out_eqs)


# ── Debug imagery ────────────────────────────────────────────────────────────

def _color_label_image(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    uniq = np.unique(labels)
    palette = {0: np.array([0, 0, 0], np.uint8)}
    for u in uniq:
        if u == 0: continue
        palette[int(u)] = rng.randint(40, 256, size=3, dtype=np.int32).astype(np.uint8)
    for u, c in palette.items():
        out[labels == u] = c
    return out


def _annotate(img: np.ndarray, boxes: List[Dict],
              pins_dict: Dict[str, Dict[str, List[Tuple[int,int]]]],
              graph: Dict[str, Dict]) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x, y, bw, bh = b['x'], b['y'], b['w'], b['h']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 220, 100), 2)
        # Label shows class + inputs found
        inp_str = ", ".join(str(v) for v in graph[b['id']]["inputs"]) or "?"
        label   = f"{b['cls']}({inp_str})"
        cv2.putText(out, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1, cv2.LINE_AA)
        for (px, py) in pins_dict[b['id']]['in']:
            cv2.circle(out, (px, py), 5, (0, 120, 255), -1)   # blue  = input
        for (px, py) in pins_dict[b['id']]['out']:
            cv2.circle(out, (px, py), 5, (0, 200, 255), -1)   # yellow = output
    return out


def save_debug(debug_dir: str, name: str, img: np.ndarray) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, name), img)


# ── End-to-end ──────────────────────────────────────────────────────────────

def predict_circuit(image_path: str,
                    model_path: Optional[str] = None,
                    model: Optional[YOLO] = None,
                    debug_root: Optional[str] = None) -> CircuitResult:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        if model_path is None: model_path = find_best_model()
        if not model_path:     raise FileNotFoundError("No best.pt found in runs/")
        model = YOLO(model_path)
        log.info("Loaded model: %s", model_path)

    if debug_root is None:
        debug_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "debug_intermediates")
    name = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = os.path.join(debug_root, name)
    os.makedirs(debug_dir, exist_ok=True)

    log.info("=== Stage 1: Gate Detection ===")
    boxes, img = detect_gates(image_path, model)

    log.info("=== Stage 2: Preprocess ===")
    binary, dot_mask = preprocess(img, boxes)
    save_debug(debug_dir, "01_original.png", img)
    save_debug(debug_dir, "02_binary.png",   binary)
    save_debug(debug_dir, "03_dots.png",     dot_mask)

    log.info("=== Stage 3: Wire Tracing ===")
    labels, broken = trace_wires(binary, dot_mask)
    save_debug(debug_dir, "04_broken.png",   broken)
    save_debug(debug_dir, "05_wires.png",    _color_label_image(labels))

    log.info("=== Stage 4+5: Geometric Pin Assignment ===")
    assignment, pins_dict = assign_pins_geometric(boxes, labels)

    log.info("=== Stage 6: Graph Build ===")
    graph, gi, go, warns = build_graph(boxes, assignment)

    log.info("=== Stage 7: Netlist ===")
    netlist, equations = generate_netlist(boxes, graph)

    annotated = _annotate(img, boxes, pins_dict, graph)
    save_debug(debug_dir, "06_annotated.png", annotated)

    # Wire+gate overlay for inspection
    overlay = _color_label_image(labels)
    for b in boxes:
        cv2.rectangle(overlay, (b['x'], b['y']),
                      (b['x'] + b['w'], b['y'] + b['h']), (255, 255, 255), 2)
    save_debug(debug_dir, "07_wires_with_gates.png", overlay)

    log.info("Inputs: %s | Outputs: %s", sorted(gi), sorted(go))
    log.info("NETLIST:\n%s", netlist)
    log.info("EQUATIONS:\n%s", equations)
    for w in warns: log.warning(w)

    return CircuitResult(
        netlist=netlist, equations=equations, graph=graph,
        global_inputs=gi, global_outputs=go, gates=boxes,
        warnings=warns, annotated_image=annotated, debug_dir=debug_dir,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_v2.py <image_path>")
        sys.exit(1)
    res = predict_circuit(sys.argv[1])
    print("\n=== NETLIST ===");   print(res.netlist)
    print("\n=== EQUATIONS ==="); print(res.equations)
    if res.warnings:
        print("\n=== WARNINGS ===")
        for w in res.warnings: print(" ! " + w)
    print(f"\nDebug images: {res.debug_dir}")
