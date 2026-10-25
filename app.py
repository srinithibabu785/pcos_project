"""
Interactive Streamlit Application for PCOS AI Decision Support.
Implements multi-model prediction, risk scoring tiers, Grad-CAM visual explanations,
and responsible AI clinical disclaimers.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_loader import get_model_manager
from src.gradcam import generate_gradcam_overlay
from src.report import RESPONSIBLE_AI_DISCLAIMER

st.set_page_config(
    page_title="PCOS AI — Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom medical styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .score-card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_manager():
    manager = get_model_manager()
    manager.load_models()
    return manager

def main():
    st.markdown('<div class="main-title">PCOS AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predict • Explain • Support | Explainable Ensemble Transfer Learning</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ System Status")
        with st.spinner("Loading deep learning backbones..."):
            manager = load_cached_manager()
        st.success("✅ Models Loaded from Disk")
        st.write(f"**EfficientNetB0:** Ready")
        st.write(f"**DenseNet121:** Ready")
        st.write(f"**Ensemble Weights:** w1={manager.ensemble.w_eff:.2f}, w2={manager.ensemble.w_dense:.2f}")
        st.write(f"**Decision Threshold:** {manager.ensemble.threshold:.2f}")
        
        st.markdown("---")
        st.subheader("Select Sample Test Image")
        test_dir = BASE_DIR / "data_processed" / "clean_images" / "test"
        if not test_dir.exists():
            test_dir = BASE_DIR / "data_processed" / "images" / "test"

        samples = []
        if test_dir.exists():
            h_samples = list((test_dir / "healthy").glob("*.jpg"))[:4]
            p_samples = list((test_dir / "infected").glob("*.jpg"))[:4]
            samples = h_samples + p_samples

        sample_names = ["-- Or Upload Your Own Below --"] + [f"{s.parent.name.upper()}: {s.name}" for s in samples]
        selected_sample = st.selectbox("Sample Library:", sample_names)

    # Image upload or selection
    uploaded_file = st.file_uploader("Upload Transvaginal Ultrasound Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
    
    image_to_process = None
    display_name = ""

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        display_name = uploaded_file.name
    elif selected_sample != "-- Or Upload Your Own Below --":
        idx = sample_names.index(selected_sample) - 1
        chosen_path = samples[idx]
        image_to_process = Image.open(chosen_path).convert("RGB")
        display_name = chosen_path.name

    if image_to_process is not None:
        # Save temporary image for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_to_process.save(tmp.name)
            temp_img_path = tmp.name

        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.subheader("📷 Ultrasound Input")
            st.image(image_to_process, caption=display_name, use_container_width=True)

        # Inference
        with st.spinner("Analyzing follicular features & computing ensemble score..."):
            result = manager.predict_image(temp_img_path)

        eff_score = result["efficientnet_score"] * 100.0
        dense_score = result["densenet_score"] * 100.0
        ens_score = result["ensemble_score"] * 100.0
        pred_class = result["predicted_class"]
        threshold = result["decision_threshold"]
        risk_tier = result["risk_tier"]
        risk_cat = result["risk_category"]
        risk_col = result["risk_color"]
        risk_rng = result["risk_range"]

        with col_pred:
            st.subheader("📊 Model Prediction")
            
            # Main prediction card
            st.markdown(f"""
            <div class="score-card" style="border-left-color: {risk_col.lower()};">
                <h3 style="margin:0; color: #1E293B;">AI-Predicted Category: <b>{pred_class}</b></h3>
                <p style="font-size: 1.1rem; margin-top: 5px; margin-bottom: 5px;">
                    Ensemble Likelihood Score: <b>{ens_score:.2f}%</b> (Threshold: {threshold:.2f})
                </p>
                <span style="background-color: #E2E8F0; padding: 4px 10px; border-radius: 12px; font-weight: 600;">
                    Risk Tier {risk_tier}: {risk_cat} ({risk_rng})
                </span>
                <p style="font-size: 0.9rem; color: #64748B; margin-top: 8px;">{result['clinical_note']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("🔬 Model Comparison")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("EfficientNetB0", f"{eff_score:.2f}%")
            with c2:
                st.metric("DenseNet121", f"{dense_score:.2f}%")
            with c3:
                st.metric("Ensemble (0.5 / 0.5)", f"{ens_score:.2f}%")

            # Progress bar representation
            st.write("**Integrated Likelihood Scale:**")
            st.progress(float(result["ensemble_score"]))

        # Explainability Section
        st.markdown("---")
        st.subheader("🔍 Explainable AI (Grad-CAM Follicle Attribution)")
        st.write("Visualizes high-activation spatial regions influencing the deep neural network decision scores.")

        cam_tab1, cam_tab2 = st.tabs(["EfficientNetB0 Attribution", "DenseNet121 Attribution"])

        with cam_tab1:
            with st.spinner("Computing Gradient-weighted Class Activation Map for EfficientNetB0..."):
                _, eff_cam, eff_overlay = generate_gradcam_overlay(
                    image_path=temp_img_path,
                    model=manager.eff_model,
                    model_name="efficientnetb0"
                )
            g1, g2, g3 = st.columns(3)
            with g1:
                st.image(image_to_process, caption="Original Ultrasound", use_container_width=True)
            with g2:
                st.image(eff_cam, caption="Grad-CAM Heatmap (top_conv)", use_container_width=True)
            with g3:
                st.image(eff_overlay, caption="Feature Overlay Attribution", use_container_width=True)

        with cam_tab2:
            with st.spinner("Computing Gradient-weighted Class Activation Map for DenseNet121..."):
                _, dense_cam, dense_overlay = generate_gradcam_overlay(
                    image_path=temp_img_path,
                    model=manager.dense_model,
                    model_name="densenet121"
                )
            d1, d2, d3 = st.columns(3)
            with d1:
                st.image(image_to_process, caption="Original Ultrasound", use_container_width=True)
            with d2:
                st.image(dense_cam, caption="Grad-CAM Heatmap (conv5_block16_2_conv)", use_container_width=True)
            with d3:
                st.image(dense_overlay, caption="Feature Overlay Attribution", use_container_width=True)

    else:
        st.info("👆 Please select a test ultrasound image from the sidebar or upload your own ultrasound scan to begin.")

    # Responsible AI Note
    st.markdown(f"""
    <div class="disclaimer-box">
        <b>RESPONSIBLE AI & ACADEMIC GOVERNANCE NOTE:</b><br>
        {RESPONSIBLE_AI_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
