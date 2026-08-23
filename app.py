import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO
import matplotlib.cm as cm
from datetime import datetime

st.set_page_config(
    page_title="Brain AI DSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #0A1628 0%, #0D1B2E 100%);
    }

    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        background: rgba(13, 27, 46, 0.9);
        border: 1px solid rgba(34, 211, 238, 0.2);
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .navbar-brand {
        font-size: 1.4rem;
        font-weight: 800;
        color: #22D3EE;
        letter-spacing: 1px;
    }
    .navbar-tabs {
        color: #22D3EE;
        font-size: 0.95rem;
        font-weight: 600;
        border-bottom: 2px solid #22D3EE;
        padding-bottom: 4px;
    }
    .navbar-meta {
        color: #94A3B8;
        font-size: 0.85rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 30, 50, 0.75);
        border: 1px solid rgba(34, 211, 238, 0.18) !important;
        border-radius: 16px !important;
    }

    .panel-title {
        color: #22D3EE;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        border-bottom: 1px solid rgba(34,211,238,0.15);
        padding-bottom: 0.5rem;
    }

    .metric-label {
        color: #64748B;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #E2E8F0;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .diag-box {
        background: rgba(34, 211, 238, 0.06);
        border-left: 3px solid #22D3EE;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .diag-label {
        color: #64748B;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .diag-value {
        color: #F1F5F9;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .diag-value.tumor { color: #FCA5A5; }
    .diag-value.notumor { color: #86EFAC; }

    .center-caption {
        text-align: center;
        color: #22D3EE;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 0.6rem;
        font-size: 1.05rem;
    }

    div[data-testid="stFileUploader"] {
        border: 1.5px dashed rgba(34, 211, 238, 0.4);
        border-radius: 12px;
        background: rgba(34, 211, 238, 0.03);
    }

    .footer-note {
        text-align:center;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 1.2rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS_TR = {
    "glioma": "GLIOMA", "meningioma": "MENINGIOMA",
    "notumor": "TÜMÖR YOK", "pituitary": "PİTUİTER TÜMÖR"
}
IMG_SIZE = (380, 380)
MODEL_METRICS = {"Sınıflandırma": "%90.1", "Segmentasyon mAP50": "%90.4"}

@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model("best_model_v2.keras")

@st.cache_resource
def load_segmentation_model():
    return YOLO("yolo_best_seg.pt")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    base_model = gap_layer = dense_layer = None
    for layer in model.layers:
        if "efficientnet" in layer.name:
            base_model = layer
        if "global_average_pooling" in layer.name:
            gap_layer = layer
        if "dense" in layer.name:
            dense_layer = layer

    grad_model = tf.keras.models.Model(
        base_model.input, [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    x = tf.keras.applications.efficientnet.preprocess_input(img_array)
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(x)
        gap_out = gap_layer(base_outputs)
        preds = dense_layer(gap_out)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index.numpy(), preds.numpy()[0]

today = datetime.now().strftime("%Y-%m-%d")
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">🧠 BRAIN AI DSS</div>
    <div class="navbar-tabs">DASHBOARD</div>
    <div class="navbar-meta">TÜBİTAK 2209-B Prototip &nbsp;|&nbsp; {today}</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("MR görüntüsü yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Analiz ediliyor..."):
        clf_model = load_classification_model()
        img_resized = image.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img_resized), axis=0).astype("float32")
        heatmap, pred_idx, probs = make_gradcam_heatmap(img_array, clf_model, "top_activation")
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

    heatmap_resized = np.array(Image.fromarray(heatmap).resize(IMG_SIZE))
    base_img = np.array(img_resized).astype("float32")
    jet = cm.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized)[..., :3] * 255
    gradcam_overlay = (base_img * 0.55 + jet_heatmap * 0.45).astype("uint8")

    if pred_class != "notumor":
        seg_model = load_segmentation_model()
        result = seg_model.predict(image, verbose=False)[0]
        center_display_img = result.plot()[..., ::-1]
    else:
        center_display_img = np.array(image)

left, center, right = st.columns([1.1, 1.6, 1.1], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">AI Model Performansı</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-label">Sınıflandırma</div><div class="metric-value">{MODEL_METRICS["Sınıflandırma"]}</div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-label">Segmentasyon</div><div class="metric-value">{MODEL_METRICS["Segmentasyon mAP50"]}</div>', unsafe_allow_html=True)
        st.caption("Test seti üzerinde ölçülen değerlerdir (EfficientNetB4 + YOLOv11).")

    with st.container(border=True):
        st.markdown('<div class="panel-title">Açıklanabilir Yapay Zeka (XAI)</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            st.image(gradcam_overlay, use_container_width=True, caption="Grad-CAM Isı Haritası")
        else:
            st.caption("Grad-CAM ısı haritası, görsel yüklendiğinde burada görünecek.")

with center:
    with st.container(border=True):
        if uploaded_file is not None:
            st.image(center_display_img, use_container_width=True)
            st.markdown(f'<div class="center-caption">TÜMÖR TAHMİNİ: {CLASS_LABELS_TR[pred_class]}</div>', unsafe_allow_html=True)
            st.progress(float(confidence / 100), text=f"Güven: %{confidence:.1f}")
        else:
            st.info("👆 Analiz başlatmak için yukarıdan bir MR görüntüsü yükleyin.")

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Karar Destek Özeti</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            diag_class = "notumor" if pred_class == "notumor" else "tumor"
            st.markdown(f"""
            <div class="diag-box">
                <div class="diag-label">Model Tahmini</div>
                <div class="diag-value {diag_class}">{CLASS_LABELS_TR[pred_class]}</div>
            </div>
            <div class="diag-box">
                <div class="diag-label">Güven Skoru</div>
                <div class="diag-value">%{confidence:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="diag-label" style="margin-top:0.6rem;">Tüm Sınıf Olasılıkları</div>', unsafe_allow_html=True)
            for cname in CLASS_NAMES:
                idx = CLASS_NAMES.index(cname)
                st.progress(float(probs[idx]), text=f"{CLASS_LABELS_TR[cname]}: %{probs[idx]*100:.1f}")
        else:
            st.caption("Görüntü yüklendiğinde model tahmini ve olasılık dağılımı burada gösterilecek.")

st.markdown(
    '<div class="footer-note">⚠️ Bu sistem bir araştırma prototipidir, klinik teşhis yerine geçmez. '
    'Kesin tanı için mutlaka bir uzman hekime danışın.</div>',
    unsafe_allow_html=True
)