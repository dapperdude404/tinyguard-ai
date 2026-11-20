"""SmartPlant AI: research-grade transparency-first Streamlit classroom app."""
import io
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from smartplant.core import decode_image, leaf_observations, load_model, predict
from smartplant.cnn import load_cnn, predict_cnn
from smartplant.model_download import ensure_cnn_checkpoint

st.set_page_config(page_title="SmartPlant AI | Plant Health Research", page_icon="🌿", layout="wide")
st.markdown("""<style>
.block-container{max-width:1220px;padding-top:1.5rem}
h1{letter-spacing:-.04em}
div[data-testid="stMetric"]{background:#122d23;border:1px solid #235d45;border-radius:15px;padding:14px}
</style>""", unsafe_allow_html=True)

st.title("🌿 SmartPlant AI")
st.markdown("### Plant Leaf Disease Classification & Visual Health Explorer")
st.caption("M.Tech Artificial Intelligence · Explainable image features · Honest model evaluation")

with st.sidebar:
    st.header("Research workspace")
    st.success("Image analysis available")
    st.markdown("**How it works**\n\n1. Upload a leaf photo.\n2. Inspect visual color indicators.\n3. If a trained classifier is installed, view predicted categories.")
    st.info("The application never fabricates a disease diagnosis when no trained model is available.")
    st.markdown("**Models:** CNN when trained checkpoint installed; Random Forest fallback.")
    st.markdown("**Train locally:** `python train.py --data data/train`")
    st.caption("Research demonstration, not agricultural or medical advice.")

tab1, tab2, tab3 = st.tabs(["📷 Analyze a leaf", "🧠 How the AI works", "📚 Research & deployment"])

with tab1:
    upload = st.file_uploader("Upload a clear image of a single plant leaf", type=["png","jpg","jpeg","webp"])
    if upload:
        try:
            image = decode_image(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        left, right = st.columns([1,1], gap="large")
        with left:
            st.image(image, caption="Uploaded leaf image · normalized for analysis", use_container_width=True)
        with right:
            st.subheader("Visible color indicators")
            observations = leaf_observations(image)
            a,b,c = st.columns(3)
            a.metric("Green", f"{observations['green_pixels_pct']}%")
            b.metric("Yellow", f"{observations['yellow_pixels_pct']}%")
            c.metric("Brown", f"{observations['brown_pixels_pct']}%")
            st.caption("These are simple pixel-color heuristics. They are not plant segmentation, disease severity, or a clinical diagnosis.")
            cnn = None
            cnn_path = "models/smartplant_mobilenetv3.pt"
            checkpoint_ready = False
            try:
                checkpoint_ready = ensure_cnn_checkpoint(cnn_path) is not None
            except Exception as exc:
                st.warning(f"CNN checkpoint verification failed: {exc}")
            if checkpoint_ready:
                try:
                    cnn = load_cnn(cnn_path)
                except Exception as exc:
                    st.error(f"CNN checkpoint could not be loaded: {exc}")
            if cnn is not None:
                results = predict_cnn(image, cnn)
                st.success(f"CNN predicted category: **{results[0][0].replace('_', ' ')}**")
                st.metric("CNN model class probability", f"{results[0][1]:.1%}")
                st.caption("Class probability is not a verified diagnosis confidence.")
                chart = pd.DataFrame(results, columns=["Category", "Probability"])
                st.plotly_chart(px.bar(chart, x="Probability", y="Category", orientation="h", range_x=[0,1]), use_container_width=True)
                st.download_button("Download CNN predictions", chart.to_csv(index=False), "smartplant_cnn_predictions.csv", "text/csv")
            model = load_model("models/plant_model.joblib") if cnn is None else None
            if model is None and cnn is None:
                st.warning("Disease classifier not installed yet. Train on labeled leaf images and add the resulting trusted model artifact to models/plant_model.joblib.")
                st.markdown("**What you can demonstrate now:** leaf upload, image preprocessing, explainable color analysis and the end-to-end research workflow.")
            elif model is not None:
                results = predict(model, image)
                name, confidence = results[0]
                st.success(f"Top predicted category: **{name.replace('_',' ')}**")
                st.metric("Model class probability", f"{confidence*100:.1f}%")
                st.caption("A class probability is not a guarantee of diagnosis. Validate the model on independent, real field images.")
                chart = pd.DataFrame(results[:5], columns=["Category", "Probability"])
                fig = px.bar(chart, x="Probability", y="Category", orientation="h", range_x=[0,1],
                             color="Probability", color_continuous_scale="Greens")
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Random held-out image validation accuracy: {model.validation_accuracy:.1%} on a dataset of {model.dataset_size} images; may be optimistic due to image similarity.")
                st.download_button("Download prediction CSV", chart.to_csv(index=False), "smartplant_predictions.csv", "text/csv")
    else:
        st.info("Upload a leaf image to begin. You can use your own photograph of a plant leaf.")
        st.markdown("**Photo tips:** one leaf in focus, natural light, avoid heavy shadows, and keep the leaf visible.")

with tab2:
    st.subheader("Simple classroom explanation")
    st.markdown("""
**Step 1 — Image preprocessing.** Convert the uploaded image to RGB and resize it to 128 × 128 pixels.

**Step 2 — Feature extraction.** Compute HOG texture/edge features and color histograms.

**Step 3 — Classification.** A trained Random Forest maps these features to labeled disease categories.

**Step 4 — Interpret the result.** Show the predicted class and competing class probabilities, alongside simple color statistics.

**Step 5 — Scientific evaluation.** Report independent test-set accuracy, macro F1, per-class precision/recall, confusion matrix, and failure cases.

**Important:** Without a trained dataset and model, the app deliberately shows only descriptive color indicators. A green or yellow leaf is not enough to diagnose disease.
""")
    st.code("python train.py --data data/train --output models/plant_model.joblib", language="bash")
    st.markdown("Dataset folders should be named after the labels, e.g. `data/train/Healthy/`, `data/train/Tomato_Early_blight/`, and `data/train/Tomato_Late_blight/`.")

with tab3:
    st.subheader("Research design")
    st.markdown("""
**Research question:** Can lightweight interpretable visual features and a Random Forest distinguish labeled plant-leaf disease categories, and how well do they generalize beyond the training environment?

**Evaluation plan:** Use a labeled dataset, separate train/validation/test sets, group near-duplicate images by source plant, measure macro F1 and per-class recall, and test with new field photographs.

**Current limitation:** CNN inference is integrated but no trained CNN checkpoint is bundled. Do not claim CNN accuracy until a checkpoint is trained and independently evaluated.

**Deployment:** Connect this GitHub repository to Streamlit Community Cloud, use branch `main`, and set `app.py` as the entrypoint.

**Safety:** Results are educational and should not be the sole basis for pesticide or crop-management decisions.
""")
    st.markdown("See `docs/SMARTPLANT_RESEARCH.md` and `docs/SMARTPLANT_CLASSROOM.md` in the repository.")

st.divider()
st.caption("SmartPlant AI · M.Tech AI Research Edition · Transparent, reproducible prototype")
