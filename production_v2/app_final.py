"""
Integrated pipeline UI — Schematic image → Netlist → Synthesizable HDL.

Part 1  (predict.py)      : YOLOv8 + CV  →  gates, wires, netlist, equations
Part 2  (hdl_gen)         : gate graph   →  Verilog/VHDL + testbench + IC BOM
"""
import os
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

from predict import (CircuitResult, find_best_model, predict_circuit,
                     find_gate_classifier, load_gate_classifier)
from hdl_gen import generate_all
from pattern_engine import compress

st.set_page_config(page_title="Schematic → Netlist → HDL",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ── Access gate ────────────────────────────────────────────────────────────
# Private-by-link: the app only renders when the URL includes the correct
# ?key=... query parameter. No login, no public listing — works the same
# regardless of which host serves this (Streamlit Cloud / Render / etc.),
# so privacy isn't tied to a platform-specific setting.
#
# Set the real secret in Streamlit Cloud's "Secrets" panel as:
#   ACCESS_KEY = "your-long-random-string"
# Locally, set the ACCESS_KEY environment variable, or edit the fallback
# default below for quick local testing only — never commit a real secret.
try:
    _ACCESS_KEY = st.secrets.get("ACCESS_KEY", "") or os.environ.get("ACCESS_KEY", "")
except Exception:
    # No secrets.toml at all (e.g. fresh local checkout) — fall back to env var.
    _ACCESS_KEY = os.environ.get("ACCESS_KEY", "")
if _ACCESS_KEY:
    if st.query_params.get("key") != _ACCESS_KEY:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:80vh;font-family:sans-serif;">
          <div style="text-align:center;">
            <h2>🔒 Private link required</h2>
            <p style="opacity:.7">This app is only accessible with a valid access link.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

st.markdown("""
<style>
  .stApp { background-color: #0E1117; color: #FAFAFA; }
  .custom-box { background:#1E2127; padding:16px 20px; border-radius:10px; border:1px solid #333; }
  .main-title { font-size:2.6rem !important; font-weight:800 !important;
     background:-webkit-linear-gradient(45deg,#FF4B2B,#FF416C);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0; }
  .stage { display:inline-block; background:#FF4B2B; color:white; border-radius:12px;
     padding:2px 12px; font-size:.72rem; font-weight:700; margin-right:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Schematic → Netlist → HDL</p>', unsafe_allow_html=True)
st.caption("**Part 1** — YOLOv8 gate detection + CV wire tracing   ·   "
           "**Part 2** — gate graph → synthesizable Verilog / VHDL + IC bill-of-materials")
st.write("---")


@st.cache_resource(show_spinner="Loading YOLO model…")
def _load_model():
    p = find_best_model()
    return (YOLO(p), p) if p else (None, None)


@st.cache_resource(show_spinner="Loading CNN classifier…")
def _load_clf():
    p = find_gate_classifier()
    return load_gate_classifier(p) if p else None


model, model_path = _load_model()
if model is None:
    st.error("No trained YOLO model found (looked for **/best.pt).")
    st.stop()
classifier = _load_clf()

with st.sidebar:
    st.header("⚙ Settings")
    hdl_lang = st.radio("HDL style to feature", ["Verilog (behavioral)",
                        "Verilog (structural)", "VHDL"], index=0)
    show_tb = st.checkbox("Show testbench", value=True)
    show_bom = st.checkbox("Show IC bill-of-materials", value=True)
    st.divider()
    st.success(f"Model: `{os.path.relpath(model_path)}`")
    st.success("CNN classifier: active" if classifier is not None else "CNN classifier: none")
    st.caption("v5 pipeline · contact pins · raw-ink bridge · IC mapping")

# ── Stage 1: upload ───────────────────────────────────────────────────────────
st.markdown('<span class="stage">STAGE 1</span> **Upload schematic**', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a digital logic schematic",
                            type=["png", "jpg", "jpeg"], label_visibility="collapsed")
if uploaded is None:
    st.info("👆 Upload a schematic image to run the full pipeline.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = tmp.name

# ── Stage 2: detection + netlist ──────────────────────────────────────────────
st.write("")
st.markdown('<span class="stage">STAGE 2</span> **Schematic → Netlist & Equations**',
            unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.caption("Original")
    st.image(uploaded, use_container_width=True)

try:
    with st.spinner("Running AI pipeline (YOLO + CV)…"):
        result: CircuitResult = predict_circuit(tmp_path, model=model, classifier=classifier)
except Exception as e:
    st.error(f"Pipeline error: {e}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

with c2:
    st.caption("AI detections")
    st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

ins = ", ".join(sorted(result.global_inputs)) or "none"
outs = ", ".join(sorted(result.global_outputs)) or "none"
st.success(f"Extracted **{len(result.gates)}** gate(s) — Inputs: `{ins}` | Outputs: `{outs}`")
for w in result.warnings:
    st.warning(f"⚠ {w}")

e1, e2 = st.columns(2)
with e1:
    st.markdown('<div class="custom-box"><h4>Netlist</h4></div>', unsafe_allow_html=True)
    st.code(result.netlist or "— none —", language="verilog")
with e2:
    st.markdown('<div class="custom-box"><h4>Boolean Equations</h4></div>', unsafe_allow_html=True)
    st.code(result.equations or "— none —", language="text")

# ── Stage 3: HDL generation ───────────────────────────────────────────────────
st.write("")
st.markdown('<span class="stage">STAGE 3</span> **Netlist → Synthesizable HDL**',
            unsafe_allow_html=True)

if not result.graph:
    st.warning("No gate graph extracted — cannot generate HDL.")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    st.stop()

# ── Stage 2.5: functional pattern recognition ────────────────────────────────
with st.spinner("Recognizing functional blocks…"):
    compression = compress(result.graph)

if compression.matches:
    st.info(f"🧩 Recognized **{len(compression.matches)}** functional block(s)")
    st.table([
        {"Block": m.cls, "Pattern": m.pattern, "Level": m.level,
         "Absorbed gates": ", ".join(m.absorbed)}
        for m in compression.matches
    ])
else:
    st.caption("No higher-level functional blocks recognized — "
               "emitting gate-level HDL.")

mod_name = os.path.splitext(uploaded.name)[0]
hdl = generate_all(compression.graph, result.global_inputs,
                   result.global_outputs, mod_name)

st.success(f"Generated HDL for module **{hdl['module_name']}** — "
           f"{hdl['total_gates']} gate(s) → **{hdl['total_packages']}** physical IC(s)")

tab_labels, tab_payload = [], []
tab_labels.append("Verilog (behavioral)"); tab_payload.append(("verilog", hdl["verilog_behavioral"], f"{hdl['module_name']}.v"))
tab_labels.append("Verilog (structural)"); tab_payload.append(("verilog", hdl["verilog_structural"], f"{hdl['module_name']}_structural.v"))
tab_labels.append("VHDL");               tab_payload.append(("vhdl", hdl["vhdl"], f"{hdl['module_name']}.vhd"))
if show_tb:
    tab_labels.append("Testbench");      tab_payload.append(("verilog", hdl["testbench"], f"tb_{hdl['module_name']}.v"))

# reorder so the sidebar-selected style is first
order = {"Verilog (behavioral)": 0, "Verilog (structural)": 1, "VHDL": 2}
sel = order.get(hdl_lang, 0)
tabs = st.tabs(tab_labels)
for i, tab in enumerate(tabs):
    lang, code, fname = tab_payload[i]
    with tab:
        st.code(code, language=lang)
        st.download_button(f"⬇ Download {fname}", data=code, file_name=fname,
                           mime="text/plain", key=f"dl_{i}")

if show_bom and hdl["ic_bom"]:
    st.write("")
    st.markdown("**📦 Physical IC Requirements (7400 / 4000 series)**")
    st.table([
        {"Gate": b["gate"], "Count": b["count"], "IC Part": b["part"],
         "Description": b["description"], "Gates/pkg": b["gates_per_pkg"],
         "# ICs": b["packages"]}
        for b in hdl["ic_bom"]
    ])

# full bundle download
bundle = (f"// ==== {hdl['module_name']} — behavioral ====\n{hdl['verilog_behavioral']}\n\n"
          f"// ==== structural ====\n{hdl['verilog_structural']}\n\n"
          f"// ==== testbench ====\n{hdl['testbench']}\n")
st.download_button("📦 Download full Verilog bundle (module + structural + testbench)",
                   data=bundle, file_name=f"{hdl['module_name']}_bundle.v", mime="text/plain")

with st.expander("Internal graph representation"):
    st.json({"global_inputs": sorted(result.global_inputs),
             "global_outputs": sorted(result.global_outputs),
             "gate_dependencies": result.graph})

if result.debug_dir:
    st.caption(f"Debug images: `{result.debug_dir}`")

if os.path.exists(tmp_path):
    os.remove(tmp_path)
