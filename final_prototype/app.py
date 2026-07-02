import streamlit as st
import cv2
import os
import tempfile
from predict import (CircuitResult, find_best_model, predict_circuit,
                     find_gate_classifier, load_gate_classifier)
from ultralytics import YOLO

st.set_page_config(
    page_title="Schematic to Netlist AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .custom-box {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Schematic to Netlist AI</p>', unsafe_allow_html=True)
st.write("Upload a digital logic schematic to extract its Netlist and Boolean Equations "
         "using YOLOv8 & Computer Vision.")
st.write("---")


@st.cache_resource
def load_model():
    model_path = find_best_model()
    if not model_path:
        return None, None
    return YOLO(model_path), model_path


@st.cache_resource
def load_classifier():
    clf_path = find_gate_classifier()
    if not clf_path:
        return None
    return load_gate_classifier(clf_path)


model, model_path = load_model()
if model is None:
    st.error("No trained model found in `runs/`. Run `train_yolo.py` first.")
    st.stop()

classifier = load_classifier()

st.sidebar.success(f"Model loaded: `{os.path.relpath(model_path)}`")
if classifier is not None:
    st.sidebar.success("CNN gate classifier active (newmodel.pth)")
else:
    st.sidebar.info("No CNN classifier found — using YOLO labels only")
st.sidebar.caption(
    "Pin assignment: orange-dot proximity if dots are present in the image, "
    "otherwise boundary-contact scan."
)

uploaded_file = st.file_uploader(
    "Upload Schematic Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)

    with st.spinner("Processing circuit through AI pipeline..."):
        try:
            result: CircuitResult = predict_circuit(
                tmp_path, model=model, classifier=classifier)

            with col2:
                st.subheader("AI Detections")
                annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            inputs_str  = ", ".join(sorted(result.global_inputs))  or "none"
            outputs_str = ", ".join(sorted(result.global_outputs)) or "none"
            st.success(
                f"Extracted **{len(result.gates)}** gate(s) — "
                f"Inputs: `{inputs_str}` | Outputs: `{outputs_str}`"
            )

            for w in result.warnings:
                st.warning(f"⚠ {w}")

            st.write("---")
            st.subheader("Extracted Logic")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown('<div class="custom-box"><h4>Netlist</h4></div>',
                            unsafe_allow_html=True)
                st.code(result.netlist, language="verilog")

            with res_col2:
                st.markdown('<div class="custom-box"><h4>Boolean Equations</h4></div>',
                            unsafe_allow_html=True)
                st.code(result.equations, language="text")

            with st.expander("Internal Graph Representation"):
                st.json({
                    "global_inputs":     sorted(result.global_inputs),
                    "global_outputs":    sorted(result.global_outputs),
                    "gate_dependencies": result.graph,
                })

            if result.debug_dir:
                st.caption(f"Debug images written to: `{result.debug_dir}`")

        except ValueError as e:
            st.warning(str(e))
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            raise

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
