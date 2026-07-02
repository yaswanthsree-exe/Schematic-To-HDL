"""
predict.py  —  Schematic-to-Netlist inference pipeline (v4).

Stages
------
1.   detect_gates       — YOLO detection + NMS duplicate suppression
1.5  reclassify_gates   — optional CNN gate-type refinement
2.   preprocess         — gate erasure · OCR text removal · morphological closing
3.   trace_wires_graph  — skeletonise → skeleton-graph walk → net labelling
3.5  detect_orange_pins — optional HSV orange-dot pin markers
4.   assign_pins_zones  — endpoint-proximity + pixel-fallback zone snapping
5.   build_graph        — wire label → gate dependency graph
6.   generate_netlist   — topological netlist + Boolean equations

Design notes
------------
* Wire tracing now builds an explicit skeleton graph (junctions as nodes,
  degree-2 walks as edges).  Each connected component of the graph is one net.
* Pin assignment matches skeleton endpoints / nearby wire pixels to theoretical
  pin-zone rectangles derived from gate bbox + class-specific Y-fractions.
  Two passes: endpoint proximity (most reliable), then pixel proximity fallback.
* OCR-based text removal (EasyOCR, graceful fallback to area filter) cleans
  label annotations (A, B, Sum …) that would otherwise fragment wire nets.
* Morphological closing fills small gaps in handwritten / scanned wires before
  skeletonisation.
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
from typing import Any, Dict, List, Optional, Set, Tuple
from ultralytics import YOLO

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("predict")

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES: List[str] = ["AND", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]

GATE_INPUT_COUNT: Dict[str, int] = {
    "AND": 2, "NAND": 2, "NOR": 2, "OR": 2,
    "XOR": 2, "XNOR": 2, "NOT": 1, "BUF": 1,
}

YOLO_CONF     = 0.25
MIN_GATE_AREA = 600      # px²
MIN_WIRE_AREA = 10       # px

# Preprocessing
GATE_PAD      = 6        # px erased around each gate bbox
CLOSE_KERN    = 5        # morphological closing kernel for gap-fill (handwritten)

# Skeleton graph
NODE_CLUSTER_K = 5       # dilation kernel for clustering nearby junction pixels
DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Pin-zone snapping (primary assignment method)
PIN_INSET        = 5    # px inside gate bbox where wires are expected to terminate
PIN_ZONE_PRIMARY = 20   # px — tight initial snap radius (high precision)
PIN_ZONE_SNAP    = 50   # px — wider fallback snap radius
DIRECTION_WEIGHT = 30   # px equivalent bonus for perfect directional alignment
Y_ALIGN_WEIGHT   = 15   # px equivalent bonus for matching expected pin Y position
DIR_SAMPLES      = 10   # skeleton pixels to sample when estimating wire direction
BOUNDARY_NEAR    = 30   # px — horizontal tolerance for vertical-position fallback
POST_CORRECT_R   = 70   # px — search radius for post-correction pass
HOUGH_COVERAGE_THRESH = 0.60  # fall back to skeleton-graph if fewer pins are hit

# Orange-dot pin detection (HSV, OpenCV 0-180 hue scale)
ORANGE_HSV_LOW  = np.array([4,  120, 120], dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255], dtype=np.uint8)
ORANGE_MIN_AREA = 10
ORANGE_MAX_AREA = 800
PIN_SNAP_DIST   = 30

# Legacy boundary-contact fallback (kept for reference)
WIRE_REACH_X  = 130
WIRE_REACH_Y  = 180
GATE_SLOP     = 10
MIN_PIN_PIX   = 2
MAX_CONN_DIST = 40

# Theoretical pin Y-fractions (fraction of gate height from top)
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

CNN_CONF_THRESHOLD = 0.85


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CircuitResult:
    netlist:         str
    equations:       str
    graph:           Dict[str, Any]
    global_inputs:   set
    global_outputs:  set
    gates:           List[Dict]
    warnings:        List[str]              = field(default_factory=list)
    annotated_image: Optional[np.ndarray]  = None
    debug_dir:       Optional[str]         = None


@dataclass
class SkelGraph:
    """Skeleton decomposed into nodes (junctions/endpoints) and edge paths."""
    node_xy:    List[Tuple[int,int]]           # (x, y) for each node
    node_type:  List[str]                      # 'endpoint' | 'junction' | 'isolated'
    edges:      List[Tuple[int, int, List]]    # (node_i, node_j, [(x,y),...])
    pixel_node: np.ndarray                     # (h,w) int32; -1 = not a node pixel


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1


# ── Model / classifier discovery ──────────────────────────────────────────────

def find_best_model(search_root: Optional[str] = None) -> Optional[str]:
    if search_root is None:
        search_root = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(search_root, "**/best.pt"), recursive=True)
    if not candidates:
        return None
    rf = [c for c in candidates if "roboflow" in c.replace("\\", "/")]
    return rf[0] if rf else max(candidates, key=os.path.getctime)


def find_gate_classifier(search_root: Optional[str] = None) -> Optional[str]:
    if not _TORCH_AVAILABLE:
        return None
    if search_root is None:
        search_root = os.path.dirname(os.path.abspath(__file__))
    directory = search_root
    for _ in range(4):
        candidate = os.path.join(directory, "newmodel.pth")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return None


# ── CNN gate reclassifier (optional Stage 1.5) ───────────────────────────────

if _TORCH_AVAILABLE:
    class _GateClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1      = nn.Conv2d(1,  32, 3)
            self.conv2      = nn.Conv2d(32, 64, 3)
            self.conv3      = nn.Conv2d(64, 128, 3)
            self.fc_shared  = nn.Linear(32768, 512)
            self.fc_class   = nn.Linear(512, 7)
            self.fc_fan_in  = nn.Linear(512, 1)
            self.fc_fan_out = nn.Linear(512, 1)

        def forward(self, x):
            x      = F.relu(self.conv1(x))
            x      = F.relu(self.conv2(x))
            x      = F.relu(self.conv3(x))
            x      = x.flatten(1)
            shared = F.relu(self.fc_shared(x))
            return self.fc_class(shared), self.fc_fan_in(shared), self.fc_fan_out(shared)


def load_gate_classifier(path: str):
    if not _TORCH_AVAILABLE:
        return None
    try:
        m = _GateClassifier()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        m.eval()
        log.info("Gate classifier loaded: %s", path)
        return m
    except Exception as exc:
        log.warning("Could not load gate classifier %s: %s", path, exc)
        return None


def reclassify_gates(boxes: List[Dict], img: np.ndarray, classifier) -> List[Dict]:
    if classifier is None or not _TORCH_AVAILABLE:
        return boxes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for b in boxes:
        x1, y1 = max(0, b['x']), max(0, b['y'])
        x2, y2 = min(img.shape[1], b['x'] + b['w']), min(img.shape[0], b['y'] + b['h'])
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_r = cv2.resize(crop, (22, 22)).astype(np.float32) / 255.0
        tensor = torch.tensor(crop_r).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cls_logits, fan_in_pred, _ = classifier(tensor)
        probs        = F.softmax(cls_logits, dim=1)
        max_prob     = float(probs.max().item())
        pred_cls     = CLASSES[int(cls_logits.argmax(1).item())]
        type_max     = GATE_INPUT_COUNT.get(pred_cls, 2)
        pred_fan_in  = max(1, min(round(float(fan_in_pred.item())), type_max))
        b['cls_yolo'] = b['cls']
        b['fan_in']   = pred_fan_in
        if max_prob >= CNN_CONF_THRESHOLD:
            b['cls'] = pred_cls
    log.info("CNN reclassification done (%d gates).", len(boxes))
    return boxes


# ── Stage 1: Gate detection ───────────────────────────────────────────────────

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
    boxes = _nms(boxes)
    if not boxes:
        raise ValueError("No logic gates detected in the image.")
    boxes.sort(key=lambda b: (b["x"], b["y"]))
    log.info("Detected %d gate(s): %s", len(boxes),
             ", ".join(f"{b['cls']}({b['conf']:.2f})" for b in boxes))
    return boxes, img


def _nms(boxes: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    boxes_s = sorted(boxes, key=lambda b: -b['conf'])
    keep: List[Dict] = []
    for b in boxes_s:
        if all(_iou(b, k) <= iou_thresh for k in keep):
            keep.append(b)
    return keep


def _iou(a: Dict, b: Dict) -> float:
    ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
    bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
    ix1, iy1 = max(a['x'], b['x']), max(a['y'], b['y'])
    ix2, iy2 = min(ax2, bx2),       min(ay2, by2)
    iw, ih   = max(0, ix2 - ix1),   max(0, iy2 - iy1)
    inter    = iw * ih
    union    = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / union if union > 0 else 0.0


# ── Stage 2: Preprocessing ────────────────────────────────────────────────────

# OCR reader (lazy-initialised, cached)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], verbose=False)
            log.info("EasyOCR reader initialised.")
        except Exception:
            _ocr_reader = False  # sentinel: not available
    return _ocr_reader if _ocr_reader is not False else None


def _remove_text_regions(binary: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Erase text label regions from the wire binary.

    Uses EasyOCR when available; falls back to the compact-blob area filter.
    Running OCR before gate erasure prevents text inside gate boxes from
    leaving wire-like artefacts.
    """
    reader = _get_ocr_reader()
    if reader is not None:
        try:
            results = reader.readtext(img)
            out = binary.copy()
            count = 0
            for (bbox, text, conf) in results:
                if conf < 0.2 or not text.strip():
                    continue
                pts = np.array(bbox, dtype=np.int32)
                # Expand the OCR box by a few pixels to cover serifs / anti-aliasing
                cx  = int(pts[:, 0].mean()); cy = int(pts[:, 1].mean())
                pts2 = pts.copy()
                for i in range(4):
                    pts2[i, 0] = int(cx + 1.15 * (pts[i, 0] - cx))
                    pts2[i, 1] = int(cy + 1.15 * (pts[i, 1] - cy))
                cv2.fillPoly(out, [pts2], 0)
                count += 1
            log.info("OCR text removal: erased %d region(s).", count)
            return out
        except Exception as exc:
            log.warning("OCR failed (%s) — using compact-blob filter.", exc)

    return _erase_glyphs(binary)


def _erase_glyphs(binary: np.ndarray) -> np.ndarray:
    """Remove isolated compact blobs that look like text characters."""
    out = binary.copy()
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, n):
        _, _, cw, ch, area = stats[i]
        if area < 40 and max(cw, ch) < 12:
            out[lbl == i] = 0
    return out


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    try:
        return cv2.ximgproc.thinning(binary,
                                      thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        skel = np.zeros_like(binary)
        img  = binary.copy()
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, elem)
            opened = cv2.dilate(eroded, elem)
            temp   = cv2.subtract(img, opened)
            skel   = cv2.bitwise_or(skel, temp)
            img    = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel


def _detect_dots(binary: np.ndarray) -> np.ndarray:
    """Detect junction-dot thickenings via distance transform."""
    if binary.sum() == 0:
        return np.zeros_like(binary)
    dt   = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    skel = _skeletonize(binary)
    skel_dt = dt[skel > 0]
    wire_thickness = float(np.median(skel_dt)) if len(skel_dt) > 0 else 1.0
    thresh_val = max(2.0, wire_thickness * 1.6 + 0.5)
    dot_core   = (dt >= thresh_val).astype(np.uint8)
    dot_out    = np.zeros_like(binary)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(dot_core)
    for i in range(1, n):
        _, _, cw, ch, area = stats[i]
        if 1 <= area <= 80 and max(cw, ch) <= 12:
            cv2.circle(dot_out, (int(cents[i][0]), int(cents[i][1])), 9, 255, -1)
    return dot_out


def preprocess(img: np.ndarray,
               boxes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wire_binary, dot_mask).

    Pipeline:
    1. Triple-threshold binarisation (Otsu + dark + adaptive).
    2. Morphological opening (noise removal).
    3. Text removal via OCR (easyocr) or compact-blob filter — runs BEFORE
       gate erasure so that wire stubs attached to the main wire survive the
       area-based filter step.
    4. Gate-body erasure (with GATE_PAD padding).
    5. Morphological closing — fills small gaps typical in handwritten wires.
    6. Dilation — heals micro-gaps before skeletonisation.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    adapt   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize=25, C=10)
    raw = cv2.bitwise_or(cv2.bitwise_or(otsu, dark), adapt)

    PAD = GATE_PAD

    # Dot detection on RAW (before gate erasure) to keep boundary dots
    raw_for_dots = raw.copy()
    for b in boxes:
        cv2.rectangle(raw_for_dots,
                      (max(0, b['x'] - PAD),          max(0, b['y'] - PAD)),
                      (min(w, b['x'] + b['w'] + PAD), min(h, b['y'] + b['h'] + PAD)),
                      0, -1)
    raw_for_dots = _erase_glyphs(raw_for_dots)
    dot_mask     = _detect_dots(raw_for_dots)

    # Wire binary pipeline
    binary = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Text removal BEFORE gate erasure: stubs are still attached to large wire
    # components so they pass the area filter unharmed.
    binary = _remove_text_regions(binary, img)

    # Erase gate bodies
    for b in boxes:
        cv2.rectangle(binary,
                      (max(0, b['x'] - PAD),          max(0, b['y'] - PAD)),
                      (min(w, b['x'] + b['w'] + PAD), min(h, b['y'] + b['h'] + PAD)),
                      0, -1)

    # Closing: fill small gaps (critical for handwritten / scanned schematics)
    ck     = CLOSE_KERN
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               np.ones((ck, ck), np.uint8))

    # Slight dilation before skeletonisation heals 1-px rendering breaks
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    return binary, dot_mask


# ── Stage 3: Wire tracing — skeleton graph walk ───────────────────────────────

def _build_skel_graph(skel_bin: np.ndarray) -> SkelGraph:
    """Decompose a skeleton into a node/edge graph.

    Nodes   — pixels with degree ≠ 2  (endpoint: 1,  junction: ≥3,  isolated: 0)
    Edges   — degree-2 walks between adjacent nodes
    """
    h, w = skel_bin.shape
    kern     = np.ones((3, 3), np.uint8); kern[1, 1] = 0
    neigh_ct = cv2.filter2D(skel_bin.astype(np.uint8), -1, kern)
    deg      = np.where(skel_bin > 0, neigh_ct.astype(np.int32), 0)

    is_endpoint = (skel_bin > 0) & (deg == 1)
    is_junction = (skel_bin > 0) & (deg >= 3)
    is_isolated = (skel_bin > 0) & (deg == 0)
    is_node     = is_endpoint | is_junction | is_isolated

    # Cluster nearby node pixels → one representative node per cluster
    node_dil   = cv2.dilate(is_node.astype(np.uint8),
                             np.ones((NODE_CLUSTER_K, NODE_CLUSTER_K), np.uint8))
    n_c, c_lbl = cv2.connectedComponents(node_dil)

    node_xy:    List[Tuple[int, int]] = []
    node_type:  List[str]             = []
    pixel_node = np.full((h, w), -1, dtype=np.int32)

    for ci in range(1, n_c):
        ys, xs = np.where((c_lbl == ci) & is_node)
        if len(xs) == 0:
            continue
        nidx = len(node_xy)
        cx   = int(round(float(np.mean(xs))))
        cy   = int(round(float(np.mean(ys))))
        node_xy.append((cx, cy))
        # Dominant type
        if int(np.sum((c_lbl == ci) & is_junction)) > 0:
            node_type.append('junction')
        elif int(np.sum((c_lbl == ci) & is_endpoint)) > 0:
            node_type.append('endpoint')
        else:
            node_type.append('isolated')
        # Mark ALL cluster pixels (not just node-type pixels) so the walker
        # treats the whole cluster as already-visited territory.
        ys_all, xs_all = np.where(c_lbl == ci)
        for y2, x2 in zip(ys_all.tolist(), xs_all.tolist()):
            pixel_node[y2, x2] = nidx

    # Walk degree-2 paths between nodes
    visited = (pixel_node >= 0).copy()   # node pixels are pre-visited

    edges: List[Tuple[int, int, List]] = []

    for nidx, (nx, ny) in enumerate(node_xy):
        for dy, dx in DIRS8:
            sx, sy = nx + dx, ny + dy
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            if skel_bin[sy, sx] == 0 or visited[sy, sx]:
                continue
            # Walk this arm
            path: List[Tuple[int, int]] = [(nx, ny), (sx, sy)]
            visited[sy, sx] = True
            cx, cy = sx, sy

            while True:
                found_end = False
                nxt       = None
                for dy2, dx2 in DIRS8:
                    nx2, ny2 = cx + dx2, cy + dy2
                    if not (0 <= nx2 < w and 0 <= ny2 < h):
                        continue
                    if skel_bin[ny2, nx2] == 0:
                        continue
                    if pixel_node[ny2, nx2] >= 0:
                        # Reached another (or the same) node
                        nj = int(pixel_node[ny2, nx2])
                        path.append(node_xy[nj])
                        if nj != nidx:          # skip pure self-loops
                            edges.append((nidx, nj, path))
                        found_end = True
                        break
                    if not visited[ny2, nx2] and nxt is None:
                        nxt = (nx2, ny2)
                if found_end:
                    break
                if nxt is None:
                    # Dangling path — treat last pixel as an implicit endpoint
                    # (degree changed after cluster dilation; rare edge case)
                    break
                px, py = nxt
                visited[py, px] = True
                path.append((px, py))
                cx, cy = px, py

    return SkelGraph(node_xy=node_xy, node_type=node_type,
                     edges=edges, pixel_node=pixel_node)


def _label_nets(graph: SkelGraph) -> np.ndarray:
    """Union-Find over edges → compact 1-based net label per node."""
    n = len(graph.node_xy)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    uf = UnionFind(n)
    for (ni, nj, _) in graph.edges:
        if ni >= 0 and nj >= 0:
            uf.union(ni, nj)
    root_map: Dict[int, int] = {}
    net_label = np.zeros(n, dtype=np.int32)
    next_lbl  = 1
    for i in range(n):
        r = uf.find(i)
        if r not in root_map:
            root_map[r] = next_lbl
            next_lbl   += 1
        net_label[i] = root_map[r]
    return net_label


def _propagate_labels(skel_labels: np.ndarray,
                      binary: np.ndarray) -> np.ndarray:
    """BFS flood of skeleton labels into adjacent binary-foreground pixels."""
    h, w    = binary.shape
    out     = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    ys, xs = np.where(skel_labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lbl = int(skel_labels[y, x])
        out[y, x]     = lbl
        visited[y, x] = True
        queue.append((y, x))

    dirs4 = ((-1,0),(1,0),(0,-1),(0,1))
    while queue:
        y, x = queue.popleft()
        lbl  = out[y, x]
        for dy, dx in dirs4:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if visited[ny, nx] or binary[ny, nx] == 0:
                continue
            visited[ny, nx] = True
            out[ny, nx]     = lbl
            queue.append((ny, nx))
    return out


def trace_wires_graph(binary: np.ndarray,
                      dot_mask: np.ndarray
                      ) -> Tuple[np.ndarray, SkelGraph]:
    """Skeleton-graph wire tracer.

    Returns:
        labels — (h,w) int32; 0 = background, 1..N = wire nets
        graph  — SkelGraph (used by assign_pins_zones for endpoint lookup)
    """
    h, w = binary.shape

    skel     = _skeletonize(binary)
    skel_bin = (skel > 0).astype(np.uint8)

    graph     = _build_skel_graph(skel_bin)
    net_label = _label_nets(graph)

    # Build pixel-level skeleton label image
    skel_lbl = np.zeros((h, w), dtype=np.int32)

    if len(net_label) > 0:
        pn = graph.pixel_node
        # Label node pixels
        nz_y, nz_x = np.where(pn >= 0)
        for y, x in zip(nz_y.tolist(), nz_x.tolist()):
            ni = int(pn[y, x])
            if ni < len(net_label):
                skel_lbl[y, x] = int(net_label[ni])

        # Label edge path pixels
        for (ni, nj, path) in graph.edges:
            lbl = int(net_label[ni]) if ni >= 0 and ni < len(net_label) else 0
            if lbl == 0 and nj >= 0 and nj < len(net_label):
                lbl = int(net_label[nj])
            for (px, py) in path[1:-1]:
                if 0 <= py < h and 0 <= px < w:
                    skel_lbl[py, px] = lbl

    # BFS-propagate skeleton labels into full binary
    labels = _propagate_labels(skel_lbl, binary)

    # Remove tiny noise fragments
    for uid in np.unique(labels):
        if uid == 0:
            continue
        if int(np.sum(labels == uid)) < MIN_WIRE_AREA:
            labels[labels == uid] = 0

    # Compact label range
    uniq = np.unique(labels[labels > 0])
    if uniq.size > 0:
        remap = np.zeros(int(labels.max()) + 2, dtype=np.int32)
        for new_id, old_id in enumerate(uniq, start=1):
            remap[int(old_id)] = new_id
        labels = remap[labels]

    return labels, graph


# ── Direction analysis helpers ────────────────────────────────────────────────

def _build_endpoint_directions(graph: SkelGraph) -> Dict[int, List[Tuple[float, float]]]:
    """Return, for each node index, a list of direction vectors (one per attached edge).

    Each vector points INTO the wire path FROM that node.  For example, if node N
    is at x=100 and the edge goes leftward toward x=50, the vector at N is (-1, 0).
    """
    node_dirs: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    n = DIR_SAMPLES

    for (ni, nj, path) in graph.edges:
        if len(path) < 2:
            continue

        # Direction INTO path from ni (start of path)
        pts = path[:min(n, len(path))]
        dx  = pts[-1][0] - pts[0][0]
        dy  = pts[-1][1] - pts[0][1]
        length = (dx*dx + dy*dy) ** 0.5
        if length > 0:
            node_dirs[ni].append((dx / length, dy / length))

        # Direction INTO path from nj (end of path, reversed)
        pts = path[max(0, len(path) - n):]
        dx  = pts[0][0] - pts[-1][0]
        dy  = pts[0][1] - pts[-1][1]
        length = (dx*dx + dy*dy) ** 0.5
        if length > 0 and ni != nj:
            node_dirs[nj].append((dx / length, dy / length))

    return node_dirs


def _direction_alignment(dirs: List[Tuple[float, float]],
                         pin_side: str) -> float:
    """Alignment score ∈ [-1, 1] between wire directions and expected pin side.

    pin_side='in'  → best direction is LEFTWARD  (wire leaves pin going left)
    pin_side='out' → best direction is RIGHTWARD (wire leaves pin going right)

    Returns the maximum dot product with the preferred axis across all attached edges.
    """
    if not dirs:
        return 0.0
    preferred_x = -1.0 if pin_side == 'in' else 1.0
    return max(dx * preferred_x for (dx, _) in dirs)


# ── Stage 3.5: Orange-dot pin detection ──────────────────────────────────────

def detect_orange_pins(img: np.ndarray,
                       boxes: List[Dict]
                       ) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
    dots: List[Tuple[int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if ORANGE_MIN_AREA <= area <= ORANGE_MAX_AREA:
            dots.append((int(cents[i][0]), int(cents[i][1])))

    if not dots:
        return {}

    pin_map: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        b['id']: {"in": [], "out": []} for b in boxes
    }
    for (dx, dy) in dots:
        best_gid, best_dist = None, float('inf')
        for b in boxes:
            margin = 15
            if (b['x'] - margin <= dx <= b['x'] + b['w'] + margin and
                    b['y'] - margin <= dy <= b['y'] + b['h'] + margin):
                cx   = b['x'] + b['w'] / 2
                dist = abs(dx - cx)
                if dist < best_dist:
                    best_dist, best_gid = dist, b['id']
        if best_gid is None:
            continue
        b       = next(bb for bb in boxes if bb['id'] == best_gid)
        gate_cx = b['x'] + b['w'] / 2
        if dx <= gate_cx:
            pin_map[best_gid]['in'].append((dx, dy))
        else:
            pin_map[best_gid]['out'].append((dx, dy))

    for gid in pin_map:
        pin_map[gid]['in'].sort(key=lambda p: p[1])

    total = sum(len(v['in']) + len(v['out']) for v in pin_map.values())
    log.info("Orange-dot detection: %d dots across %d gate(s).", total, len(boxes))
    return pin_map


def _find_nearest_wire(px: int, py: int,
                       labels: np.ndarray, max_dist: int) -> int:
    h, w = labels.shape
    x0 = max(0, px - max_dist);  x1 = min(w, px + max_dist + 1)
    y0 = max(0, py - max_dist);  y1 = min(h, py + max_dist + 1)
    region = labels[y0:y1, x0:x1]
    ys, xs = np.where(region > 0)
    if len(xs) == 0:
        return 0
    dists = np.sqrt((xs + x0 - px) ** 2 + (ys + y0 - py) ** 2)
    idx   = int(dists.argmin())
    return int(region[ys[idx], xs[idx]]) if dists[idx] <= max_dist else 0


def assign_pins_proximity(
        pin_map:   Dict[str, Dict[str, List[Tuple[int, int]]]],
        boxes:     List[Dict],
        labels:    np.ndarray,
        snap_dist: int = PIN_SNAP_DIST,
) -> Dict[Tuple[str, str, int], int]:
    assignment: Dict[Tuple[str, str, int], int] = {}
    for b in boxes:
        gid  = b['id']
        pins = pin_map.get(gid, {"in": [], "out": []})
        for idx, (px, py) in enumerate(pins['in']):
            wid = _find_nearest_wire(px, py, labels, snap_dist)
            if wid:
                assignment[(gid, 'in', idx)] = wid
        for (px, py) in pins['out']:
            wid = _find_nearest_wire(px, py, labels, snap_dist)
            if wid:
                assignment[(gid, 'out', 0)] = wid
    return assignment


# ── Hough-line wire tracer + direct pin assignment ────────────────────────────

def trace_and_assign_hough(
        binary:    np.ndarray,
        boxes:     List[Dict],
        snap_dist: int = 30,
        min_line:  int = 15,
        max_gap:   int = 20,
) -> Tuple[np.ndarray, Dict[Tuple[str, str, int], int], float]:
    """Hough-line based wire tracer and direct pin assignment.

    Faster than skeleton-graph for schematics with straight wires.
    Returns (labels, assignment, coverage) where coverage ∈ [0,1] is the
    fraction of expected gate pins successfully assigned.
    """
    h, w = binary.shape

    skel = cv2.ximgproc.thinning(binary)

    lines = cv2.HoughLinesP(
        skel, 1, np.pi / 180, threshold=20,
        minLineLength=min_line, maxLineGap=max_gap,
    )
    if lines is None:
        return np.zeros((h, w), dtype=np.int32), {}, 0.0

    segments: List[Tuple[int, int, int, int]] = [tuple(ln[0]) for ln in lines]

    # Grid-snap endpoints to cluster nearby line termini into shared nodes
    Q = 8
    node_ids:  Dict[Tuple[int, int], int] = {}
    node_count = 0

    def _get_node(x: int, y: int) -> int:
        nonlocal node_count
        key = (round(x / Q) * Q, round(y / Q) * Q)
        if key not in node_ids:
            node_ids[key] = node_count
            node_count += 1
        return node_ids[key]

    for x1, y1, x2, y2 in segments:
        _get_node(x1, y1)
        _get_node(x2, y2)

    if node_count == 0:
        return np.zeros((h, w), dtype=np.int32), {}, 0.0

    uf = UnionFind(node_count)
    for x1, y1, x2, y2 in segments:
        uf.union(_get_node(x1, y1), _get_node(x2, y2))

    # Paint segments onto a label image (label = Union-Find root + 1)
    labels = np.zeros((h, w), dtype=np.int32)
    for x1, y1, x2, y2 in segments:
        net = uf.find(_get_node(x1, y1)) + 1
        cv2.line(labels, (x1, y1), (x2, y2), int(net), 1)

    # BFS: flood-fill labels from painted skeleton pixels into the full binary
    visited = labels > 0
    queue: deque = deque()
    seed_ys, seed_xs = np.where(visited)
    for sy, sx in zip(seed_ys.tolist(), seed_xs.tolist()):
        queue.append((int(sy), int(sx)))

    while queue:
        cy, cx = queue.popleft()
        lbl = int(labels[cy, cx])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if binary[ny, nx] > 0:
                    visited[ny, nx] = True
                    labels[ny, nx] = lbl
                    queue.append((ny, nx))

    # Compact label IDs to 1..N
    uniq = sorted(u for u in np.unique(labels) if u > 0)
    if uniq:
        remap = np.zeros(int(labels.max()) + 2, dtype=np.int32)
        for new_id, old_id in enumerate(uniq, start=1):
            remap[old_id] = new_id
        labels = remap[labels]

    # Assign gate pins by proximity to labelled pixels
    assignment: Dict[Tuple[str, str, int], int] = {}
    total_pins = 0
    hit_pins   = 0

    for b in boxes:
        gid     = b['id']
        centers = _expected_pin_centers(b)
        n_in    = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))
        used:   Set[int] = set()

        for pin_idx, (px, py) in enumerate(centers['in'][:n_in]):
            total_pins += 1
            lbl = _find_nearest_wire(px, py, labels, snap_dist)
            if lbl:
                assignment[(gid, 'in', pin_idx)] = lbl
                used.add(lbl)
                hit_pins += 1

        ox, oy = centers['out'][0]
        total_pins += 1
        lbl = _find_nearest_wire(ox, oy, labels, snap_dist)
        if lbl and lbl not in used:
            assignment[(gid, 'out', 0)] = lbl
            hit_pins += 1
        elif lbl and lbl in used:
            for r in range(snap_dist + 15, snap_dist * 3, 15):
                lbl2 = _find_nearest_wire(ox, oy, labels, r)
                if lbl2 and lbl2 not in used:
                    assignment[(gid, 'out', 0)] = lbl2
                    hit_pins += 1
                    break

    coverage = hit_pins / total_pins if total_pins > 0 else 0.0
    log.info("Hough tracer: %d segment(s), %d/%d pins assigned (%.0f%% coverage)",
             len(segments), hit_pins, total_pins, coverage * 100)
    return labels, assignment, coverage


# ── Stage 4: Pin assignment — zone snapping ───────────────────────────────────

def _expected_pin_centers(b: Dict) -> Dict[str, List[Tuple[int, int]]]:
    """Theoretical pin positions (x, y) for a gate given its bbox + class.

    Pin centres are placed PIN_INSET pixels INSIDE the bounding box edges.
    After gate body erasure (PAD=6), wire stubs terminate ~6px outside the
    bbox, so a centre at bbox_edge + PIN_INSET=5 puts the search zone over
    both the erased gap and the wire stub with a tight 20px radius.
    """
    gx, gy, gw, gh = b['x'], b['y'], b['w'], b['h']
    cfg = GATE_PIN_FRACS.get(b['cls'], GATE_PIN_FRACS['AND'])
    return {
        'in':  [(gx + PIN_INSET,      int(gy + gh * f)) for f in cfg['in']],
        'out': [(gx + gw - PIN_INSET, int(gy + gh * cfg['out']))],
    }


def assign_pins_zones(
        boxes:     List[Dict],
        labels:    np.ndarray,
        graph:     SkelGraph,
        snap_dist: int = PIN_ZONE_SNAP,
) -> Dict[Tuple[str, str, int], int]:
    """Directional zone-snapping pin assignment.

    For each expected pin position three passes are tried, stopping at the first
    one that succeeds:

    Pass 1 — Directional endpoint snapping:
        Score each skeleton endpoint within snap_dist using:
            effective_dist = euclidean_dist - DIRECTION_WEIGHT * alignment
        where alignment ∈ [-1, 1] measures how well the wire's direction at
        that endpoint matches the expected pin side (left → input, right → output).
        Pin exclusivity is enforced: once a pin slot is filled it won't be
        overwritten by another wire.

    Pass 2 — Pixel proximity fallback:
        If no endpoint was close enough, search for the nearest labelled wire
        pixel within snap_dist.  This catches wires that approach a gate via a
        junction rather than terminating as a skeleton endpoint.

    Pass 3 — Vertical boundary fallback:
        If pass 1 & 2 both fail, scan for skeleton endpoints that are within
        BOUNDARY_NEAR pixels horizontally of the gate boundary (left for inputs,
        right for outputs) and pick the one whose Y coordinate is closest to the
        expected pin Y.  This handles short stubs that fall just outside the
        primary snap zone.
    """
    h, w = labels.shape
    assignment: Dict[Tuple[str, str, int], int] = {}

    # Pre-build: per-node direction lookup and label lookup
    ep_dirs = _build_endpoint_directions(graph)   # node_idx → [direction vectors]

    # Endpoint data: (node_idx, x, y, wire_label, directions)
    endpoints: List[Tuple[int, int, int, int, List[Tuple[float, float]]]] = []
    for nidx, (nx, ny) in enumerate(graph.node_xy):
        if graph.node_type[nidx] != 'endpoint':
            continue
        if not (0 <= ny < h and 0 <= nx < w):
            continue
        lbl = int(labels[ny, nx])
        if lbl == 0:
            continue
        endpoints.append((nidx, nx, ny, lbl, ep_dirs.get(nidx, [])))

    def _score_endpoint(ex: int, ey: int, dirs: List[Tuple[float, float]],
                        px: int, py: int, gate_h: int, pin_side: str) -> float:
        """Effective distance (lower = better).

        Bonuses (subtracted from raw distance):
        - DIRECTION_WEIGHT × directional alignment  (wire approaching from correct side)
        - Y_ALIGN_WEIGHT  × vertical alignment      (endpoint Y close to pin Y)

        Vertical alignment score = max(0, 1 - |ey-py| / (gate_h*0.25))
        so it is 1.0 when endpoint is exactly at pin Y and 0 when > 25% of gate_h away.
        """
        dist      = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
        dir_score = _direction_alignment(dirs, pin_side)
        y_norm    = abs(ey - py) / max(1, gate_h * 0.25)
        y_score   = max(0.0, 1.0 - y_norm)
        return dist - DIRECTION_WEIGHT * dir_score - Y_ALIGN_WEIGHT * y_score

    def _best_directional(px: int, py: int, gate_h: int, pin_side: str,
                          exclude_lbl: Set[int],
                          radius: int) -> Tuple[int, float]:
        """Return (wire_label, effective_dist) for the best scoring endpoint."""
        best_lbl, best_eff = 0, float('inf')
        for (_, ex, ey, lbl, dirs) in endpoints:
            if lbl in exclude_lbl:
                continue
            raw_d = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
            if raw_d > radius:
                continue
            eff = _score_endpoint(ex, ey, dirs, px, py, gate_h, pin_side)
            if eff < best_eff:
                best_eff, best_lbl = eff, lbl
        return best_lbl, best_eff

    def _vertical_boundary_fallback(gx_face: int, py: int,
                                    pin_side: str,
                                    exclude_lbl: Set[int]) -> int:
        """Find nearest endpoint that is close to the gate's x-face by Y position."""
        best_lbl, best_dy = 0, float('inf')
        for (_, ex, ey, lbl, _) in endpoints:
            if lbl in exclude_lbl:
                continue
            if abs(ex - gx_face) > BOUNDARY_NEAR:
                continue
            dy = abs(ey - py)
            if dy < best_dy:
                best_dy, best_lbl = dy, lbl
        return best_lbl

    for b in boxes:
        gid     = b['id']
        centers = _expected_pin_centers(b)
        n_in    = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))
        gx      = b['x']
        gx_out  = b['x'] + b['w']
        gh      = b['h']
        assigned_in: Set[int] = set()

        # ── Input pins ────────────────────────────────────────────────────────
        for pin_idx, (px, py) in enumerate(centers['in'][:n_in]):
            # Pass 1a: tight directional endpoint (high confidence)
            lbl, _ = _best_directional(px, py, gh, 'in', set(), PIN_ZONE_PRIMARY)
            # Pass 1b: wider directional endpoint (moderate confidence)
            if lbl == 0:
                lbl, _ = _best_directional(px, py, gh, 'in', set(), snap_dist)
            # Pass 2: pixel fallback
            if lbl == 0:
                lbl = _find_nearest_wire(px, py, labels, snap_dist)
            # Pass 3: vertical boundary fallback
            if lbl == 0:
                lbl = _vertical_boundary_fallback(gx, py, 'in', set())
            if lbl:
                assignment[(gid, 'in', pin_idx)] = lbl
                assigned_in.add(lbl)

        # ── Output pin ────────────────────────────────────────────────────────
        px, py = centers['out'][0]
        # Pass 1a: tight directional endpoint (exclude input wires)
        lbl, _ = _best_directional(px, py, gh, 'out', assigned_in, PIN_ZONE_PRIMARY)
        # Pass 1b: wider directional endpoint
        if lbl == 0:
            lbl, _ = _best_directional(px, py, gh, 'out', assigned_in, snap_dist)
        # Pass 2: pixel fallback (still excluding input wires)
        if lbl == 0:
            candidate = _find_nearest_wire(px, py, labels, snap_dist)
            if candidate and candidate not in assigned_in:
                lbl = candidate
            elif candidate in assigned_in:
                for r in range(snap_dist + 15, snap_dist * 3, 15):
                    candidate = _find_nearest_wire(px, py, labels, r)
                    if candidate and candidate not in assigned_in:
                        lbl = candidate
                        break
        # Pass 3: vertical boundary fallback
        if lbl == 0:
            lbl = _vertical_boundary_fallback(gx_out, py, 'out', assigned_in)
        if lbl:
            assignment[(gid, 'out', 0)] = lbl

    return assignment


# ── Stage 4 legacy: boundary-contact scan (kept as reference) ─────────────────

def assign_pins(boxes: List[Dict],
                labels: np.ndarray) -> Dict[Tuple[str, str, int], int]:
    """Legacy boundary-contact assignment (used as ultimate fallback)."""
    h, w = labels.shape

    wire_pixels: Dict[int, np.ndarray] = {}
    for uid in np.unique(labels):
        if uid == 0:
            continue
        ys, xs = np.where(labels == uid)
        if len(xs) >= MIN_PIN_PIX:
            wire_pixels[int(uid)] = np.column_stack(
                (xs.astype(np.float32), ys.astype(np.float32)))

    assignment: Dict[Tuple[str, str, int], int] = {}

    for b in boxes:
        gx, gy, gw, gh = b['x'], b['y'], b['w'], b['h']
        n_in = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))

        y0 = max(0, gy - WIRE_REACH_Y)
        y1 = min(h, gy + gh + WIRE_REACH_Y)
        x0_in = max(0, gx - WIRE_REACH_X)
        x1_in = min(w, gx + GATE_SLOP)

        in_cands: List[Tuple[float, float, int]] = []
        for wid, pts in wire_pixels.items():
            strip = pts[(pts[:,0] >= x0_in) & (pts[:,0] <= x1_in) &
                        (pts[:,1] >= y0)    & (pts[:,1] <= y1)]
            if len(strip) < MIN_PIN_PIX:
                continue
            rightmost_x = float(strip[:,0].max())
            dist        = float(gx) - rightmost_x
            if dist < 0:
                dist = 0.0
            rcluster = strip[strip[:,0] >= rightmost_x - 3.0]
            y_center = float(rcluster[:,1].mean())
            in_cands.append((dist, y_center, wid))

        in_cands.sort(key=lambda t: t[0])
        selected = in_cands[:n_in]
        selected.sort(key=lambda t: t[1])
        for idx, (dist, _, wid) in enumerate(selected):
            if dist <= MAX_CONN_DIST:
                assignment[(b['id'], 'in', idx)] = wid

        x0_out = max(0, gx + gw - GATE_SLOP)
        x1_out = min(w, gx + gw + WIRE_REACH_X)
        in_wids = {assignment.get((b['id'], 'in', i)) for i in range(n_in)} - {None}

        out_cands: List[Tuple[float, int, int, int]] = []
        for wid, pts in wire_pixels.items():
            strip = pts[(pts[:,0] >= x0_out) & (pts[:,0] <= x1_out) &
                        (pts[:,1] >= y0)      & (pts[:,1] <= y1)]
            if len(strip) < MIN_PIN_PIX:
                continue
            leftmost_x = float(strip[:,0].min())
            dist       = leftmost_x - float(gx + gw)
            if dist < 0:
                dist = 0.0
            conflict = 1 if wid in in_wids else 0
            total_px = len(wire_pixels[wid])
            out_cands.append((dist, conflict, total_px, wid))

        out_cands.sort(key=lambda t: (t[0], t[1], t[2]))
        if out_cands and out_cands[0][0] <= MAX_CONN_DIST:
            assignment[(b['id'], 'out', 0)] = out_cands[0][3]

    return assignment


# ── Post-correction: re-assign missing pins ───────────────────────────────────

def post_correct_assignment(boxes: List[Dict],
                            labels: np.ndarray,
                            assignment: Dict[Tuple[str, str, int], int],
                            radius: int = POST_CORRECT_R) -> Dict[Tuple[str, str, int], int]:
    """After zone-snapping, find gates with missing pins and try harder.

    Strategy:
    1. Collect all wire labels already used as gate OUTPUTS (produced wires).
       These should NOT be re-assigned as inputs to the gate that produces them.
    2. For each missing input pin, search a wider radius (POST_CORRECT_R) for
       any labelled wire pixel — preferring wires not already used by this gate.
    3. For each missing output pin, search similarly, but exclude wires already
       found as inputs to this gate.

    This pass intentionally uses a larger radius than the primary assignment to
    recover short or oddly-routed stubs that were outside the primary zones.
    """
    out = dict(assignment)   # work on a copy

    # Build set of wires produced by each gate (from the current assignment)
    produced: Dict[str, int] = {}
    for (gid, side, _), wid in out.items():
        if side == 'out':
            produced[gid] = wid

    for b in boxes:
        gid    = b['id']
        n_in   = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))
        centers = _expected_pin_centers(b)
        gate_out_wid = produced.get(gid, 0)

        # ── Missing inputs ────────────────────────────────────────────────────
        current_in_wids = {out.get((gid, 'in', i)) for i in range(n_in)} - {None}
        for pin_idx, (px, py) in enumerate(centers['in'][:n_in]):
            if (gid, 'in', pin_idx) in out:
                continue   # already assigned
            # Try closest wire not already used as this gate's output
            best_lbl, best_d = 0, float('inf')
            h, w = labels.shape
            x0, x1 = max(0, px - radius), min(w, px + radius + 1)
            y0, y1 = max(0, py - radius), min(h, py + radius + 1)
            region = labels[y0:y1, x0:x1]
            ys, xs = np.where(region > 0)
            for ry, rx in zip(ys, xs):
                lbl = int(region[ry, rx])
                if lbl == gate_out_wid:
                    continue   # don't assign our own output as input
                d = ((rx + x0 - px) ** 2 + (ry + y0 - py) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_lbl = d, lbl
            if best_lbl:
                out[(gid, 'in', pin_idx)] = best_lbl
                log.info("Post-correction: assigned wire %d to (%s, in, %d) dist=%.1f",
                         best_lbl, gid, pin_idx, best_d)

        # ── Missing output ────────────────────────────────────────────────────
        if (gid, 'out', 0) not in out:
            px, py = centers['out'][0]
            current_in_wids2 = {out.get((gid, 'in', i)) for i in range(n_in)} - {None}
            h, w = labels.shape
            x0, x1 = max(0, px - radius), min(w, px + radius + 1)
            y0, y1 = max(0, py - radius), min(h, py + radius + 1)
            region = labels[y0:y1, x0:x1]
            ys, xs = np.where(region > 0)
            best_lbl, best_d = 0, float('inf')
            for ry, rx in zip(ys, xs):
                lbl = int(region[ry, rx])
                if lbl in current_in_wids2:
                    continue
                d = ((rx + x0 - px) ** 2 + (ry + y0 - py) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_lbl = d, lbl
            if best_lbl:
                out[(gid, 'out', 0)] = best_lbl
                log.info("Post-correction: assigned wire %d to (%s, out, 0) dist=%.1f",
                         best_lbl, gid, best_d)

    return out


# ── Wire label healing ────────────────────────────────────────────────────────

# Wires smaller than this pixel count are never named as primary inputs.
# Genuine primary inputs run for dozens–hundreds of pixels; noise stubs are tiny.
MIN_PRIMARY_PIX = 40

def _reconnect_through_wires(labels: np.ndarray,
                              boxes: List[Dict],
                              scan_w: int = 18,
                              pin_y_margin: int = 18) -> np.ndarray:
    """Reconnect wire segments that were split by gate-body erasure.

    When a gate bounding box is slightly larger than the drawn gate symbol, the
    PAD-extended erasure zone cuts wires that run *beside* the gate (not through
    its input/output pins).  Those cuts split one physical net into two labelled
    fragments, both appearing as separate primary inputs.

    For each gate, we scan thin strips just outside the LEFT and RIGHT boundaries.
    At every Y row that is NOT near an expected pin Y (i.e. a "passing-by" wire),
    if we see different labels on the left vs right side we union them — they are
    the same physical wire that was bisected by the erasure.

    Only rows that are genuinely far from pin Y-positions are touched; rows near
    pins are left alone so input/output signals stay distinct.
    """
    h, w = labels.shape
    unique_lbls = [u for u in np.unique(labels) if u > 0]
    if len(unique_lbls) < 2:
        return labels

    lbl_to_idx = {lbl: i for i, lbl in enumerate(unique_lbls)}
    uf = UnionFind(len(unique_lbls))

    for b in boxes:
        gx, gy, gw, gh = b['x'], b['y'], b['w'], b['h']
        centers = _expected_pin_centers(b)
        n_in    = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))

        # Expected pin Y-positions (left and right face)
        pin_ys = set()
        for (_, py) in centers['in'][:n_in]:
            pin_ys.add(py)
        for (_, py) in centers['out']:
            pin_ys.add(py)

        # Strips just outside gate boundaries
        xl_outer = max(0, gx - scan_w)
        xl_inner = max(0, gx)
        xr_inner = min(w, gx + gw)
        xr_outer = min(w, gx + gw + scan_w)

        for y in range(max(0, gy - scan_w), min(h, gy + gh + scan_w)):
            # Skip rows near expected pins — those are real connections
            if any(abs(y - py) <= pin_y_margin for py in pin_ys):
                continue

            left_lbls  = set(int(v) for v in labels[y, xl_outer:xl_inner] if v > 0)
            right_lbls = set(int(v) for v in labels[y, xr_inner:xr_outer] if v > 0)

            # Merge labels that face each other across the gate body at this Y
            for ll in left_lbls:
                for rl in right_lbls:
                    if ll != rl and ll in lbl_to_idx and rl in lbl_to_idx:
                        uf.union(lbl_to_idx[ll], lbl_to_idx[rl])

    # Apply merging
    out = labels.copy()
    for i, lbl in enumerate(unique_lbls):
        root_lbl = unique_lbls[uf.find(i)]
        if lbl != root_lbl:
            out[out == lbl] = root_lbl

    # Compact label range
    uniq = [u for u in np.unique(out) if u > 0]
    if uniq:
        remap = np.zeros(int(out.max()) + 2, dtype=np.int32)
        for new_id, old_id in enumerate(uniq, start=1):
            remap[int(old_id)] = new_id
        out = remap[out]

    n_merged = len(unique_lbls) - len([u for u in np.unique(out) if u > 0])
    if n_merged > 0:
        log.info("Wire healing: merged %d fragment(s) across gate boundaries.", n_merged)
    return out


# ── Stage 5: Build dependency graph ───────────────────────────────────────────

def build_graph(boxes: List[Dict],
                assignment: Dict[Tuple[str, str, int], int],
                labels: Optional[np.ndarray] = None,
                ) -> Tuple[Dict[str, Dict], Set[str], Set[str], List[str]]:
    gate_ids = {b['id'] for b in boxes}
    cls_of   = {b['id']: b['cls'] for b in boxes}

    producers: Dict[int, List[str]]            = defaultdict(list)
    consumers: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for (gid, side, idx), wid in assignment.items():
        if wid == 0:
            continue
        if side == 'out':
            producers[wid].append(gid)
        else:
            consumers[wid].append((gid, idx))

    graph: Dict[str, Dict[str, Any]] = {
        gid: {"cls": cls_of[gid], "inputs": [], "outputs": []} for gid in gate_ids
    }
    global_inputs:  Set[str] = set()
    global_outputs: Set[str] = set()
    warnings:  List[str]     = []
    pin_inputs: Dict[str, Dict[int, Any]] = {gid: {} for gid in gate_ids}

    letter_idx = 0
    out_idx    = 1

    def _next_input_name() -> str:
        nonlocal letter_idx
        s, n = "", letter_idx
        while True:
            s = chr(ord('A') + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                break
        letter_idx += 1
        return s

    # Wire pixel counts — used to filter phantom primary inputs
    wire_px: Dict[int, int] = {}
    if labels is not None:
        for uid in np.unique(labels):
            if uid > 0:
                wire_px[int(uid)] = int(np.sum(labels == uid))

    all_wires = set(producers.keys()) | set(consumers.keys())
    for wid in all_wires:
        srcs      = list(dict.fromkeys(producers.get(wid, [])))
        dsts      = consumers.get(wid, [])
        real_dsts = [(gid, idx) for (gid, idx) in dsts if gid not in srcs]

        if not srcs and dsts:
            # Only name as a primary input if the wire is large enough to be real.
            # Tiny stubs (< MIN_PRIMARY_PIX px) are noise fragments; skip them so
            # they don't inflate the primary-input count with phantom letters.
            px_count = wire_px.get(wid, MIN_PRIMARY_PIX)
            if px_count < MIN_PRIMARY_PIX:
                log.debug("Skipping tiny wire %d (%d px) as primary input.", wid, px_count)
                continue
            name = _next_input_name()
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

    for b in boxes:
        node = graph[b['id']]
        cls  = b['cls']
        n    = len(node["inputs"])
        if cls in ("NOT", "BUF"):
            if n == 0:
                warnings.append(f"{b['id']} ({cls}): no input found")
            elif n > 1:
                node["inputs"] = node["inputs"][:1]
        else:
            if n == 0:
                warnings.append(f"{b['id']} ({cls}): no inputs found")
            elif n == 1:
                warnings.append(f"{b['id']} ({cls}): only 1 input found (expected ≥2)")

    return graph, global_inputs, global_outputs, warnings


# ── Stage 6: Topological sort ─────────────────────────────────────────────────

def _topo_order(boxes: List[Dict], graph: Dict[str, Dict]) -> List[str]:
    gate_ids = {b['id'] for b in boxes}
    in_deg:   Dict[str, int]       = {b['id']: 0 for b in boxes}
    children: Dict[str, List[str]] = defaultdict(list)

    for b in boxes:
        for inp in graph[b['id']]['inputs']:
            if inp in gate_ids:
                in_deg[b['id']] += 1
                children[inp].append(b['id'])

    queue = deque(gid for gid in gate_ids if in_deg[gid] == 0)
    order: List[str] = []
    while queue:
        gid = queue.popleft()
        order.append(gid)
        for c in children[gid]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)

    leftover = [b['id'] for b in boxes if b['id'] not in set(order)]
    if leftover:
        log.warning("Cycle detected — appending: %s", leftover)
    order.extend(leftover)
    return order


# ── Stage 7: Netlist generation ───────────────────────────────────────────────

def generate_netlist(boxes: List[Dict],
                     graph: Dict[str, Dict]) -> Tuple[str, str]:
    ordered  = _topo_order(boxes, graph)
    name_map = {gid: f"G{i+1}" for i, gid in enumerate(ordered)}

    netlist_lines: List[str] = []
    final_eqs:     Dict[str, str] = {}

    for gid in ordered:
        node   = graph[gid]
        gt     = node['cls']
        new_id = name_map[gid]

        resolved = [name_map.get(inp, inp) for inp in node['inputs']] or ["UNCONNECTED"]
        netlist_lines.append(f"{new_id} = {gt}({', '.join(resolved)});")

        eq_args = [final_eqs.get(inp, inp) for inp in node['inputs']]
        if not eq_args:        eq_str = "UNCONNECTED"
        elif gt == "NOT":      eq_str = f"(~{eq_args[0]})"
        elif gt == "BUF":      eq_str = eq_args[0]
        elif gt == "NAND":     eq_str = f"(~({' & '.join(eq_args)}))"
        elif gt == "NOR":      eq_str = f"(~({' | '.join(eq_args)}))"
        elif gt == "XNOR":     eq_str = f"(~({' ^ '.join(eq_args)}))"
        elif gt == "AND":      eq_str = f"({' & '.join(eq_args)})"
        elif gt == "OR":       eq_str = f"({' | '.join(eq_args)})"
        elif gt == "XOR":      eq_str = f"({' ^ '.join(eq_args)})"
        else:                  eq_str = f"{gt}({', '.join(eq_args)})"
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


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _draw_pin_map(img: np.ndarray,
                  pin_map: Dict[str, Dict[str, List[Tuple[int, int]]]]) -> np.ndarray:
    out = img.copy()
    for pins in pin_map.values():
        for (px, py) in pins['in']:
            cv2.circle(out, (px, py), 6, (0, 120, 255), -1)
        for (px, py) in pins['out']:
            cv2.circle(out, (px, py), 6, (255, 200, 0), -1)
    return out


def _draw_skel_graph(img: np.ndarray, graph: SkelGraph,
                     labels: np.ndarray) -> np.ndarray:
    """Debug overlay: skeleton endpoints (red), junctions (yellow), edges (green)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Draw edge paths
    for (_, _, path) in graph.edges:
        for i in range(len(path) - 1):
            cv2.line(out, path[i], path[i+1], (0, 200, 0), 1)
    # Draw nodes
    for nidx, (nx, ny) in enumerate(graph.node_xy):
        color = (0, 255, 255) if graph.node_type[nidx] == 'junction' else (0, 0, 255)
        cv2.circle(out, (nx, ny), 4, color, -1)
    return out


def _draw_pin_zones(img: np.ndarray,
                    boxes: List[Dict],
                    snap_dist: int = PIN_ZONE_SNAP) -> np.ndarray:
    """Debug overlay: expected pin zones (blue = input, cyan = output)."""
    out = img.copy()
    for b in boxes:
        centers = _expected_pin_centers(b)
        n_in    = b.get('fan_in', GATE_INPUT_COUNT.get(b['cls'], 2))
        for (px, py) in centers['in'][:n_in]:
            cv2.circle(out, (px, py), snap_dist, (255, 100, 0), 1)
            cv2.circle(out, (px, py), 4, (255, 100, 0), -1)
        for (px, py) in centers['out']:
            cv2.circle(out, (px, py), snap_dist, (0, 255, 255), 1)
            cv2.circle(out, (px, py), 4, (0, 255, 255), -1)
    return out


def _color_labels(labels: np.ndarray) -> np.ndarray:
    h, w  = labels.shape
    out   = np.zeros((h, w, 3), dtype=np.uint8)
    rng   = np.random.RandomState(42)
    for uid in np.unique(labels):
        if uid == 0:
            continue
        c = rng.randint(60, 256, size=3, dtype=np.int32).astype(np.uint8)
        out[labels == uid] = c
    return out


def _annotate(img: np.ndarray, boxes: List[Dict],
              graph: Dict[str, Dict]) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x, y, bw, bh = b['x'], b['y'], b['w'], b['h']
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 220, 100), 2)

        yolo_cls = b.get('cls_yolo', b['cls'])
        tag      = (f"{b['cls']}({yolo_cls})" if yolo_cls != b['cls'] else b['cls'])
        inputs   = ", ".join(str(v) for v in graph[b['id']]["inputs"]) or "?"
        label    = f"{tag}: {inputs}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 6, y), (0, 220, 100), -1)
        cv2.putText(out, label, (x + 3, max(th + 2, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _save_debug(debug_dir: str, name: str, img: np.ndarray) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, name), img)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def predict_circuit(image_path: str,
                    model_path:  Optional[str]  = None,
                    model:       Optional[YOLO] = None,
                    classifier                  = None,
                    debug_root:  Optional[str]  = None) -> CircuitResult:
    """End-to-end pipeline: image → gates → wires → graph → netlist."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        if model_path is None:
            model_path = find_best_model()
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("No trained model found. Run train_yolo.py first.")
        log.info("Loading YOLO model: %s", model_path)
        model = YOLO(model_path)

    if classifier is None and _TORCH_AVAILABLE:
        clf_path = find_gate_classifier()
        if clf_path:
            classifier = load_gate_classifier(clf_path)

    name = os.path.splitext(os.path.basename(image_path))[0]
    if debug_root is None:
        debug_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "debug_intermediates")
    debug_dir = os.path.join(debug_root, name)
    os.makedirs(debug_dir, exist_ok=True)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    log.info("=== Stage 1: Gate Detection ===")
    boxes, img = detect_gates(image_path, model)
    _save_debug(debug_dir, "01_original.png", img)

    # ── Stage 1.5 ─────────────────────────────────────────────────────────────
    if classifier is not None:
        log.info("=== Stage 1.5: CNN Reclassification ===")
        boxes = reclassify_gates(boxes, img, classifier)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    log.info("=== Stage 2: Preprocess ===")
    binary, dot_mask = preprocess(img, boxes)
    _save_debug(debug_dir, "02_binary.png", binary)
    _save_debug(debug_dir, "03_dots.png",   dot_mask)

    # ── Stage 3: Skeleton graph wire tracing ──────────────────────────────────
    log.info("=== Stage 3: Wire Tracing (skeleton graph) ===")
    labels, skel_graph = trace_wires_graph(binary, dot_mask)

    # ── Stage 3b: Wire healing — reconnect segments cut by gate-body erasure ──
    log.info("=== Stage 3b: Wire Healing ===")
    labels = _reconnect_through_wires(labels, boxes)
    _save_debug(debug_dir, "04_wires.png", _color_labels(labels))

    # Wire + gate overlay
    overlay = _color_labels(labels)
    for b in boxes:
        cv2.rectangle(overlay, (b['x'], b['y']),
                      (b['x']+b['w'], b['y']+b['h']), (255, 255, 255), 2)
    _save_debug(debug_dir, "05_wires_gates.png", overlay)

    # Skeleton graph debug image
    skel_dbg = _draw_skel_graph(img, skel_graph, labels)
    skel_dbg = _draw_pin_zones(skel_dbg, boxes)
    _save_debug(debug_dir, "05b_skel_graph.png", skel_dbg)

    # ── Stage 3.5: Orange-dot pin detection ───────────────────────────────────
    log.info("=== Stage 3.5: Pin Detection ===")
    orange_pins = detect_orange_pins(img, boxes)

    # ── Stage 4: Pin assignment ────────────────────────────────────────────────
    log.info("=== Stage 4: Pin Assignment ===")
    if orange_pins:
        log.info("Using orange-dot proximity assignment.")
        assignment = assign_pins_proximity(orange_pins, boxes, labels)
        _save_debug(debug_dir, "04b_orange_pins.png",
                    _draw_pin_map(img, orange_pins))
    else:
        # Try Hough-line tracer first — fast and reliable for straight wires
        log.info("Trying Hough-line wire tracing...")
        h_labels, h_assignment, h_coverage = trace_and_assign_hough(binary, boxes)
        if h_coverage >= HOUGH_COVERAGE_THRESH:
            log.info("Hough accepted (%.0f%% coverage) — updating wire labels.",
                     h_coverage * 100)
            labels     = _reconnect_through_wires(h_labels, boxes)
            assignment = h_assignment
            _save_debug(debug_dir, "04h_hough_wires.png", _color_labels(labels))
        else:
            log.info("Hough coverage %.0f%% below threshold — using skeleton-graph.",
                     h_coverage * 100)
            assignment = assign_pins_zones(boxes, labels, skel_graph)

            missing = {b['id'] for b in boxes
                       if not any(k[0] == b['id'] for k in assignment)}
            if missing:
                log.info("Zone-snap missed %d gate(s) — running boundary-contact fallback.",
                         len(missing))
                fallback_boxes = [b for b in boxes if b['id'] in missing]
                fallback_asgn  = assign_pins(fallback_boxes, labels)
                assignment.update(fallback_asgn)

    # ── Post-correction ───────────────────────────────────────────────────────
    log.info("=== Post-correction: filling missing pins ===")
    assignment = post_correct_assignment(boxes, labels, assignment)

    # ── Stage 5: Graph build ───────────────────────────────────────────────────
    log.info("=== Stage 5: Graph Build ===")
    graph, gi, go, warns = build_graph(boxes, assignment, labels=labels)
    log.info("Primary inputs: %s | Outputs: %s", sorted(gi), sorted(go))
    for w in warns:
        log.warning(w)

    # ── Stage 6: Netlist generation ───────────────────────────────────────────
    log.info("=== Stage 6: Netlist Generation ===")
    netlist, equations = generate_netlist(boxes, graph)
    log.info("NETLIST:\n%s", netlist)
    log.info("EQUATIONS:\n%s", equations)

    annotated = _annotate(img, boxes, graph)
    _save_debug(debug_dir, "06_annotated.png", annotated)

    return CircuitResult(
        netlist=netlist, equations=equations,
        graph=graph, global_inputs=gi, global_outputs=go,
        gates=boxes, warnings=warns,
        annotated_image=annotated, debug_dir=debug_dir,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)
    res = predict_circuit(sys.argv[1],
                          model_path=sys.argv[2] if len(sys.argv) > 2 else None)
    print("\n=== NETLIST ===");   print(res.netlist)
    print("\n=== EQUATIONS ==="); print(res.equations)
    if res.warnings:
        print("\n=== WARNINGS ===")
        for w in res.warnings: print(f"  ! {w}")
    print(f"\nDebug images written to: {res.debug_dir}")
