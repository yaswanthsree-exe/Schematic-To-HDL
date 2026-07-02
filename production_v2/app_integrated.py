"""
Unified Demo: Schematic → Netlist → Synthesizable HDL
Part 1: YOLOv8 + CV pipeline (predict.py) extracts gates, wires, Boolean equations
Part 2: ic_hdl_generator synthesizes equations into Verilog/VHDL with IC mapping

HDL synthesis is OPTIONAL — if ic_hdl_generator is not available the app works
exactly like the original app.py (netlist + equations always shown).
"""

import sys
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
import cv2

# ── Path setup ──────────────────────────────────────────────────────────────
PART1_DIR = Path(__file__).parent                        # production_v2/
PART2_DIR = Path(r"C:\Users\yaswa\Downloads\ic_hdl_generator-main-part2\ic_hdl_generator-main")

# Add PART2 to sys.path only if it actually exists — avoids polluting the path
# with a non-existent directory that could confuse import resolution.
if PART2_DIR.exists() and str(PART2_DIR) not in sys.path:
    sys.path.insert(0, str(PART2_DIR))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SchematicAI → HDL",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0E1117; color: #FAFAFA; }
  .block-box {
    background-color: #1A1E27;
    padding: 18px 20px;
    border-radius: 10px;
    border: 1px solid #2E3340;
    margin-bottom: 10px;
  }
  .pipeline-arrow {
    text-align: center;
    font-size: 2rem;
    color: #FF6B35;
    margin: 6px 0;
  }
  .stage-badge {
    display: inline-block;
    background: #FF6B35;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
  }
  .custom-box {
    background-color: #1E2127;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #333;
  }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    "## 🔌 Schematic → Netlist → Synthesizable HDL",
    help="End-to-end pipeline: image → Boolean equations → Verilog/VHDL"
)
st.caption(
    "**Part 1** — YOLOv8 gate detection + CV wire tracing  ·  "
    "**Part 2** — Boolean-to-HDL synthesis with IC mapping"
)
st.write("---")

# ── Load Part 1 resources ───────────────────────────────────────────────────
from predict import (CircuitResult, find_best_model, predict_circuit,
                     find_gate_classifier, load_gate_classifier)
from ultralytics import YOLO


@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    path = find_best_model()
    if not path:
        return None, None
    return YOLO(path), path


@st.cache_resource(show_spinner="Loading CNN classifier…")
def load_classifier():
    p = find_gate_classifier()
    return load_gate_classifier(p) if p else None


model, model_path = load_model()
if model is None:
    st.error("No trained model found in `runs/`. Run `train_yolo.py` first.")
    st.stop()

classifier = load_classifier()

# ── Load Part 2 resources (optional) ────────────────────────────────────────
# Never os.chdir() at the top level — it affects the entire Streamlit process
# across reruns.  Instead, the synthesize() helper chdir's locally with a
# try/finally guard so the working directory is always restored.
_hdl_gen     = None
_part2_ok    = False
_part2_err   = ""

if PART2_DIR.exists():
    try:
        from boolean_to_hdl import BooleanToHDLGenerator   # type: ignore
        # Instantiate with cwd temporarily set to PART2_DIR so that any
        # relative JSON/resource paths inside the constructor resolve correctly.
        _saved_cwd = os.getcwd()
        try:
            os.chdir(str(PART2_DIR))
            _hdl_gen = BooleanToHDLGenerator()
        finally:
            os.chdir(_saved_cwd)
        _part2_ok = True
    except Exception as _e:
        _part2_ok  = False
        _part2_err = str(_e)
else:
    _part2_err = f"HDL generator directory not found: {PART2_DIR}"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    if _part2_ok:
        hdl_language    = st.selectbox("HDL Output Language", ["verilog", "vhdl"], index=0)
        show_testbench  = st.checkbox("Include testbench in output", value=True)
        show_ic_map     = st.checkbox("Show IC mapping (7400 series)", value=True)
    st.divider()
    st.success(f"Model: `{os.path.relpath(model_path)}`")
    if classifier is not None:
        st.success("CNN classifier: active")
    else:
        st.info("CNN classifier: not found")
    if _part2_ok:
        st.success("HDL synthesizer: ready")
    else:
        st.warning("HDL synthesizer: unavailable")
        if _part2_err:
            st.caption(_part2_err)
    st.caption("v5 production pipeline: RDP polylines · junction Union-Find · OCR net naming")

# ── Equation conversion helpers ──────────────────────────────────────────────
def part1_to_part2_expr(eq_line: str) -> "tuple[str, str]":
    """Convert 'Y = ~(A & B)'  →  (output_name, part2_expression).
    Part 1 uses ~ for NOT; Part 2 parser uses ! for NOT.
    Returns ("", "") if the line doesn't look like an assignment.
    """
    m = re.match(r"^\s*(\w+)\s*=\s*(.+)$", eq_line.strip())
    if not m:
        return "", ""
    name = m.group(1).strip()
    expr = m.group(2).strip()
    expr = re.sub(r"~(?!=)", "!", expr)
    expr = re.sub(r"\s+", "", expr)
    expr = re.sub(r"([&|^])", r" \1 ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return name, expr


def parse_equations_block(equations_text: str) -> "list[dict]":
    results = []
    for line in equations_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        name, expr = part1_to_part2_expr(line)
        if name and expr:
            original_m = re.match(r"^\s*\w+\s*=\s*(.+)$", line)
            part1_expr = original_m.group(1).strip() if original_m else expr
            results.append({"name": name, "part1_expr": part1_expr, "part2_expr": expr})
    return results


def synthesize(expr: str, circuit_name: str, language: str) -> dict:
    """Run Part 2 HDL generator with cwd temporarily set to PART2_DIR."""
    saved = os.getcwd()
    try:
        os.chdir(str(PART2_DIR))
        return _hdl_gen.generate(expr, circuit_name, language)
    finally:
        os.chdir(saved)


# ════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Upload ──────────────────────────────────────────────────────────
st.markdown('<span class="stage-badge">STAGE 1</span> **Upload Schematic**', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a digital logic schematic image here",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("👆 Upload a schematic to start the pipeline.")
    st.stop()

# Save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = tmp.name

# ── Stage 2: Part 1 pipeline ─────────────────────────────────────────────────
st.write("")
st.markdown('<span class="stage-badge">STAGE 2</span> **Schematic → Netlist & Boolean Equations**', unsafe_allow_html=True)

col_orig, col_annot = st.columns(2)
with col_orig:
    st.caption("Original image")
    st.image(uploaded, use_container_width=True)

result: "CircuitResult | None" = None
with st.spinner("Running AI pipeline (YOLO + CV)…"):
    try:
        result = predict_circuit(tmp_path, model=model, classifier=classifier)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.stop()

with col_annot:
    st.caption("AI detections")
    st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

# Summary bar
inputs_str  = ", ".join(sorted(result.global_inputs))  or "none"
outputs_str = ", ".join(sorted(result.global_outputs)) or "none"
st.success(
    f"Extracted **{len(result.gates)}** gate(s) — "
    f"Inputs: `{inputs_str}` | Outputs: `{outputs_str}`"
)
for w in result.warnings:
    st.warning(f"⚠ {w}")

# ── Netlist + Equations (always shown) ────────────────────────────────────────
st.write("---")
st.subheader("Extracted Logic")

eq_col, nl_col = st.columns(2)
with eq_col:
    st.markdown('<div class="custom-box"><h4>Boolean Equations</h4></div>',
                unsafe_allow_html=True)
    st.code(result.equations or "— none extracted —", language="text")
with nl_col:
    st.markdown('<div class="custom-box"><h4>Netlist</h4></div>',
                unsafe_allow_html=True)
    st.code(result.netlist or "— none extracted —", language="verilog")

with st.expander("Internal Graph Representation"):
    st.json({
        "global_inputs":     sorted(result.global_inputs),
        "global_outputs":    sorted(result.global_outputs),
        "gate_dependencies": result.graph,
    })

if result.debug_dir:
    st.caption(f"Debug images written to: `{result.debug_dir}`")

# ── Stage 3: HDL synthesis (only when Part 2 is available) ───────────────────
st.write("")
st.markdown('<span class="stage-badge">STAGE 3</span> **Boolean Equations → Synthesizable HDL**', unsafe_allow_html=True)

if not _part2_ok:
    st.info(
        "HDL synthesis is not available — the ic_hdl_generator library was not found.  \n"
        "Install it at the configured path to enable Verilog/VHDL output."
    )
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

# ── HDL path continues only when _part2_ok == True ───────────────────────────

if not result.equations or result.equations.strip() == "":
    st.warning("No Boolean equations extracted — cannot synthesize HDL.")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

equations = parse_equations_block(result.equations)
if not equations:
    st.warning("Could not parse equations into synthesizable form.")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

# ── UNCONNECTED guard ─────────────────────────────────────────────────────────
def _has_unconnected(eq: dict) -> bool:
    return "UNCONNECTED" in eq["part1_expr"]

unconn_eqs = [e for e in equations if _has_unconnected(e)]
clean_eqs  = [e for e in equations if not _has_unconnected(e)]

if unconn_eqs:
    names = ", ".join(e["name"] for e in unconn_eqs)
    st.warning(
        f"⚠️ Output(s) **{names}** contain untraced wires (`UNCONNECTED`). "
        "These equations are excluded from HDL synthesis."
    )

if not clean_eqs:
    st.error("All extracted equations have unconnected wires — cannot synthesize HDL.")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

equations = clean_eqs

with st.expander("ℹ️ Operator translation (Part 1 → Part 2)"):
    rows = [f"  {e['name']} = {e['part1_expr']}   →   {e['name']} = {e['part2_expr']}"
            for e in equations]
    st.code("\n".join(rows), language="text")

if len(equations) == 1:
    selected_eq = equations[0]
else:
    options = {f"{e['name']} = {e['part1_expr']}": e for e in equations}
    choice = st.selectbox("Select output equation to synthesize:", list(options.keys()))
    selected_eq = options[choice]

circuit_name = f"circuit_{selected_eq['name'].lower()}"

st.write("")
with st.spinner(f"Synthesizing `{circuit_name}` in {hdl_language.upper()}…"):
    synth = synthesize(selected_eq["part2_expr"], circuit_name, hdl_language)

if "error" in synth:
    st.error(f"Synthesis error: {synth['error']}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

# ── HDL Results ───────────────────────────────────────────────────────────────
st.success(
    f"✅ Synthesized  `{synth['circuit_name']}`  ·  "
    f"Variables: `{', '.join(synth['variables'])}`  ·  "
    f"Simplified: `{synth['simplified_expression']}`"
)

if show_ic_map and synth.get("gate_mapping"):
    st.markdown("**📦 Physical IC Requirements (7400/CMOS series)**")
    ic_cols = st.columns([1, 1, 2, 1, 1])
    ic_cols[0].markdown("**Gate**");  ic_cols[1].markdown("**Count**")
    ic_cols[2].markdown("**IC**");    ic_cols[3].markdown("**# ICs**")
    ic_cols[4].markdown("**Gates/pkg**")
    seen: set = set()
    for m in synth["gate_mapping"]:
        key = (m["gate_type"], m["ic_number"])
        if key in seen:
            continue
        seen.add(key)
        gates_per = m["count"] // m["num_ics"] if m["num_ics"] else m["count"]
        c0, c1, c2, c3, c4 = st.columns([1, 1, 2, 1, 1])
        c0.write(m["gate_type"]);  c1.write(str(m["count"]))
        c2.write(f"{m['ic_number']} — {m['ic_name']}")
        c3.write(str(m["num_ics"])); c4.write(str(gates_per))

st.markdown("**Generated HDL**")
full_code = synth["hdl_code"]

tb_start = full_code.find("`timescale")
if tb_start != -1:
    module_code    = full_code[:tb_start].rstrip()
    testbench_code = full_code[tb_start:]
else:
    module_code    = full_code
    testbench_code = ""

hdl_tab, tb_tab = st.tabs(["Module", "Testbench"])

with hdl_tab:
    st.code(module_code, language="verilog" if hdl_language == "verilog" else "vhdl")
    st.download_button(
        "⬇️ Download module",
        data=module_code,
        file_name=f"{circuit_name}.{'v' if hdl_language == 'verilog' else 'vhd'}",
        mime="text/plain",
    )

with tb_tab:
    if show_testbench and testbench_code:
        st.code(testbench_code, language="verilog")
        st.download_button(
            "⬇️ Download testbench",
            data=testbench_code,
            file_name=f"tb_{circuit_name}.v",
            mime="text/plain",
        )
    elif not testbench_code:
        st.info("No testbench generated (too many inputs or VHDL mode).")
    else:
        st.info("Testbench disabled — toggle in sidebar.")

if show_testbench and testbench_code:
    bundle = f"{module_code}\n\n// {'='*72}\n// TESTBENCH\n// {'='*72}\n\n{testbench_code}"
else:
    bundle = module_code

st.download_button(
    "📦 Download full HDL bundle",
    data=bundle,
    file_name=f"{circuit_name}_bundle.v",
    mime="text/plain",
)

# ── Pipeline summary ──────────────────────────────────────────────────────────
st.write("")
st.write("---")
st.markdown("### 🔁 Pipeline Summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Gates detected",  str(len(result.gates)))
summary_cols[1].metric("Circuit inputs",  str(len(result.global_inputs)))
summary_cols[2].metric("Circuit outputs", str(len(result.global_outputs)))
summary_cols[3].metric("HDL lines",       str(module_code.count("\n") + 1))

# Cleanup
if os.path.exists(tmp_path):
    os.remove(tmp_path)
