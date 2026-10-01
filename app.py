import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm
from datetime import datetime
import sqlite3
import time
import uuid

ARSIV_KLASORU = "arsiv"
os.makedirs(ARSIV_KLASORU, exist_ok=True)

DB_PATH = "analiz_gecmisi.db"
MAX_GORSEL = 4          # tek seferde en fazla 4 görsel
HEDEF_SURE_SN = 10.0    # proje önerisindeki hedef: görsel başına < 10 sn


# ----------------------------------------------------------------------------
# VERİTABANI
# ----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analizler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dosya_adi TEXT,
            tahmin TEXT,
            guven REAL,
            tarih TEXT,
            islem_suresi_sn REAL,
            orijinal_gorsel TEXT,
            islenmis_gorsel TEXT
        )
    """)
    # Eski veritabanına batch_id sütununu güvenle ekle
    try:
        conn.execute("ALTER TABLE analizler ADD COLUMN batch_id TEXT")
    except sqlite3.OperationalError:
        pass  # sütun zaten var
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batch_performans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            gorsel_sayisi INTEGER,
            toplam_sure_sn REAL,
            ortalama_sure_sn REAL,
            tarih TEXT
        )
    """)
    conn.commit()
    conn.close()


def kaydet_analiz(dosya_adi, tahmin, guven, islem_suresi_sn,
                  orijinal_gorsel_yolu, islenmis_gorsel_yolu, batch_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO analizler
           (dosya_adi, tahmin, guven, tarih, islem_suresi_sn,
            orijinal_gorsel, islenmis_gorsel, batch_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (dosya_adi, tahmin, guven, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         islem_suresi_sn, orijinal_gorsel_yolu, islenmis_gorsel_yolu, batch_id)
    )
    conn.commit()
    conn.close()


def kaydet_batch(batch_id, gorsel_sayisi, toplam_sure):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO batch_performans
           (batch_id, gorsel_sayisi, toplam_sure_sn, ortalama_sure_sn, tarih)
           VALUES (?, ?, ?, ?, ?)""",
        (batch_id, gorsel_sayisi, toplam_sure, toplam_sure / gorsel_sayisi,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def batch_ozeti_getir():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT gorsel_sayisi, COUNT(*), AVG(toplam_sure_sn),
                  AVG(ortalama_sure_sn), MIN(toplam_sure_sn), MAX(toplam_sure_sn)
           FROM batch_performans GROUP BY gorsel_sayisi ORDER BY gorsel_sayisi"""
    ).fetchall()
    conn.close()
    return rows


def son_analizleri_getir(limit=5):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT dosya_adi, tahmin, guven, tarih, islem_suresi_sn, orijinal_gorsel, islenmis_gorsel
           FROM analizler ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


init_db()

# ----------------------------------------------------------------------------
# SAYFA AYARLARI VE CSS
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# MODELLER
# ----------------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS_TR = {
    "glioma": "GLIOMA", "meningioma": "MENINGIOMA",
    "notumor": "TÜMÖR YOK", "pituitary": "PİTUİTER TÜMÖR"
}
IMG_SIZE = (380, 380)
MODEL_METRICS = {"Sınıflandırma": "%90.1", "Segmentasyon mAP50": "%90.4"}


@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model("best_model_v3.keras")


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


@st.cache_resource
def modelleri_yukle_ve_isit():
    """Modelleri yükler ve bir kez sahte girdi ile çalıştırır (warm-up).
    Böylece ölçülen süreler model yükleme / ilk çalıştırma gecikmesini içermez."""
    clf = load_classification_model()
    seg = load_segmentation_model()
    make_gradcam_heatmap(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype="float32"),
                         clf, "top_activation")
    seg.predict(Image.new("RGB", (640, 640)), verbose=False)
    return clf, seg


# ----------------------------------------------------------------------------
# TEK GÖRSEL ANALİZİ (süre ölçümü burada)
# ----------------------------------------------------------------------------
def analiz_et(image, dosya_adi, clf_model, seg_model):
    t_basla = time.perf_counter()

    # 1) Sınıflandırma + Grad-CAM
    t0 = time.perf_counter()
    img_resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized), axis=0).astype("float32")
    heatmap, pred_idx, probs = make_gradcam_heatmap(img_array, clf_model, "top_activation")
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)

    heatmap_resized = np.array(Image.fromarray(heatmap).resize(IMG_SIZE))
    base_img = np.array(img_resized).astype("float32")
    jet_heatmap = cm.jet(heatmap_resized)[..., :3] * 255
    gradcam_overlay = (base_img * 0.55 + jet_heatmap * 0.45).astype("uint8")
    sinif_sure = time.perf_counter() - t0

    # 2) Segmentasyon (yalnızca tümör varsa)
    seg_sure = 0.0
    if pred_class != "notumor":
        t1 = time.perf_counter()
        result = seg_model.predict(image, verbose=False)[0]
        center_display_img = result.plot()[..., ::-1]
        seg_sure = time.perf_counter() - t1
    else:
        center_display_img = np.array(image)

    toplam_sure = time.perf_counter() - t_basla

    return {
        "dosya_adi": dosya_adi,
        "image": image,
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
        "gradcam_overlay": gradcam_overlay,
        "center_display_img": center_display_img,
        "sinif_sure": sinif_sure,
        "seg_sure": seg_sure,
        "toplam_sure": toplam_sure,
    }


def arsive_kaydet(r, batch_id):
    benzersiz_id = uuid.uuid4().hex[:8]
    orijinal_yol = os.path.join(ARSIV_KLASORU, f"{benzersiz_id}_orijinal.png")
    islenmis_yol = os.path.join(ARSIV_KLASORU, f"{benzersiz_id}_islenmis.png")
    r["image"].save(orijinal_yol)
    Image.fromarray(r["center_display_img"].astype("uint8")).save(islenmis_yol)
    kaydet_analiz(r["dosya_adi"], r["pred_class"], r["confidence"],
                  r["toplam_sure"], orijinal_yol, islenmis_yol, batch_id)


def benchmark_calistir(items, clf_model, seg_model, tekrar=3):
    """1, 2 ve 4 görsel için toplam/ortalama süreyi ölçer (arşive kaydetmez).
    Görsel sayısı 4'ten azsa görseller tekrar edilerek 4'e tamamlanır."""
    havuz = (items * MAX_GORSEL)[:MAX_GORSEL]
    satirlar = []
    for n in (1, 2, 4):
        sureler = []
        for _ in range(tekrar):
            t0 = time.perf_counter()
            for img, ad in havuz[:n]:
                analiz_et(img, ad, clf_model, seg_model)
            sureler.append(time.perf_counter() - t0)
        toplam = float(np.mean(sureler))
        satirlar.append({
            "Görsel sayısı": n,
            "Toplam süre (sn)": round(toplam, 2),
            "Görsel başına ortalama (sn)": round(toplam / n, 2),
            "Tekrar": tekrar,
        })
    return pd.DataFrame(satirlar)


# ----------------------------------------------------------------------------
# ARAYÜZ BİLEŞENLERİ
# ----------------------------------------------------------------------------
def render_sonuc(r):
    left, center, right = st.columns([1.1, 1.6, 1.1], gap="medium")
    pred_class, confidence, probs = r["pred_class"], r["confidence"], r["probs"]

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
            st.image(r["gradcam_overlay"], use_container_width=True, caption="Grad-CAM Isı Haritası")

    with center:
        with st.container(border=True):
            st.image(r["center_display_img"], use_container_width=True)
            st.markdown(f'<div class="center-caption">TÜMÖR TAHMİNİ: {CLASS_LABELS_TR[pred_class]}</div>', unsafe_allow_html=True)
            st.progress(float(confidence / 100), text=f"Güven: %{confidence:.1f}")

    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Karar Destek Özeti</div>', unsafe_allow_html=True)
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
            <div class="diag-box">
                <div class="diag-label">İşlem Süresi</div>
                <div class="diag-value">{r["toplam_sure"]:.2f} sn</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="diag-label" style="margin-top:0.6rem;">Tüm Sınıf Olasılıkları</div>', unsafe_allow_html=True)
            for idx, cname in enumerate(CLASS_NAMES):
                st.progress(float(probs[idx]), text=f"{CLASS_LABELS_TR[cname]}: %{probs[idx]*100:.1f}")


def render_bos_durum():
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
            st.caption("Grad-CAM ısı haritası, görsel yüklendiğinde burada görünecek.")
    with center:
        with st.container(border=True):
            st.info(f"👆 Analiz başlatmak için yukarıdan 1, 2 veya 4 adet (en fazla {MAX_GORSEL}) MR görüntüsü yükleyin.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Karar Destek Özeti</div>', unsafe_allow_html=True)
            st.caption("Görüntü yüklendiğinde model tahmini ve olasılık dağılımı burada gösterilecek.")


def render_performans(sonuclar, batch_toplam, clf_model, seg_model):
    n = len(sonuclar)
    st.markdown('<div class="panel-title">Bu Analizin Performansı</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Yüklenen görsel sayısı", n)
    c2.metric("Toplam işlem süresi", f"{batch_toplam:.2f} sn")
    c3.metric("Görsel başına ortalama", f"{batch_toplam / n:.2f} sn")

    df = pd.DataFrame([{
        "Görsel": r["dosya_adi"][:30],
        "Tahmin": CLASS_LABELS_TR[r["pred_class"]],
        "Güven (%)": round(r["confidence"], 1),
        "Sınıflandırma + Grad-CAM (sn)": round(r["sinif_sure"], 2),
        "Segmentasyon (sn)": round(r["seg_sure"], 2),
        "Toplam (sn)": round(r["toplam_sure"], 2),
        f"< {HEDEF_SURE_SN:.0f} sn hedefi": "✅" if r["toplam_sure"] < HEDEF_SURE_SN else "⚠️",
    } for r in sonuclar])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Görsel")[["Sınıflandırma + Grad-CAM (sn)", "Segmentasyon (sn)"]])
    st.caption("Süreler model yükleme ve ilk çalıştırma (warm-up) gecikmesi hariç, "
               "görselleri sırayla işleyen tek bir akış için ölçülmüştür.")

    st.divider()
    st.markdown('<div class="panel-title">1 / 2 / 4 Görsel Karşılaştırması</div>', unsafe_allow_html=True)

    st.caption("Butona basınca yüklediğiniz görseller ile 1, 2 ve 4 görsellik testler 3'er kez çalıştırılır "
               "ve ortalama alınır. Yüklenen görsel 4'ten azsa görseller tekrar edilerek 4'e tamamlanır. "
               "Bu test arşive kaydedilmez.")
    if st.button("▶️ 1 / 2 / 4 görsel testini çalıştır"):
        with st.spinner("Karşılaştırma testi çalışıyor..."):
            items = [(r["image"], r["dosya_adi"]) for r in sonuclar]
            st.session_state["benchmark"] = benchmark_calistir(items, clf_model, seg_model)
    if st.session_state.get("benchmark") is not None:
        bdf = st.session_state["benchmark"]
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        st.bar_chart(bdf.set_index("Görsel sayısı")[["Toplam süre (sn)"]])

    ozet = batch_ozeti_getir()
    if ozet:
        st.divider()
        st.markdown('<div class="panel-title">Geçmiş Yüklemelerin Ortalaması (Arşivden)</div>', unsafe_allow_html=True)
        gdf = pd.DataFrame(ozet, columns=[
            "Görsel sayısı", "Yükleme adedi", "Ort. toplam süre (sn)",
            "Ort. görsel başına (sn)", "En kısa toplam (sn)", "En uzun toplam (sn)"
        ]).round(2)
        st.dataframe(gdf, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# ÜST BAR VE YÜKLEME
# ----------------------------------------------------------------------------
today = datetime.now().strftime("%Y-%m-%d")
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">🧠 BRAIN AI DSS</div>
    <div class="navbar-tabs">DASHBOARD</div>
    <div class="navbar-meta">TÜBİTAK 2209-B Prototip &nbsp;|&nbsp; {today}</div>
</div>
""", unsafe_allow_html=True)

dosyalar = st.file_uploader(
    "MR görüntüsü yükle", type=["jpg", "jpeg", "png"],
    accept_multiple_files=True, label_visibility="collapsed"
)

if dosyalar:
    if len(dosyalar) > MAX_GORSEL:
        st.warning(f"En fazla {MAX_GORSEL} görsel analiz edilir; ilk {MAX_GORSEL} görsel kullanıldı.")
        dosyalar = dosyalar[:MAX_GORSEL]

    imza = tuple((f.name, f.size) for f in dosyalar)

    # Aynı dosyalar için sayfa yeniden çalışınca analizi ve arşiv kaydını tekrarlama
    if st.session_state.get("imza") != imza:
        with st.spinner("Modeller hazırlanıyor..."):
            clf_model, seg_model = modelleri_yukle_ve_isit()

        batch_id = uuid.uuid4().hex[:8]
        sonuclar = []
        with st.spinner(f"{len(dosyalar)} görsel analiz ediliyor..."):
            t_batch = time.perf_counter()
            for f in dosyalar:
                f.seek(0)
                image = Image.open(f).convert("RGB")
                sonuclar.append(analiz_et(image, f.name, clf_model, seg_model))
            batch_toplam = time.perf_counter() - t_batch

        for r in sonuclar:
            arsive_kaydet(r, batch_id)
        kaydet_batch(batch_id, len(sonuclar), batch_toplam)

        st.session_state["imza"] = imza
        st.session_state["sonuclar"] = sonuclar
        st.session_state["batch_toplam"] = batch_toplam
        st.session_state["benchmark"] = None

    sonuclar = st.session_state["sonuclar"]
    batch_toplam = st.session_state["batch_toplam"]
    clf_model, seg_model = modelleri_yukle_ve_isit()

    tab_adlari = [f"Görsel {i+1}" for i in range(len(sonuclar))] + ["⏱️ Performans Analizi"]
    sekmeler = st.tabs(tab_adlari)
    for i, r in enumerate(sonuclar):
        with sekmeler[i]:
            st.caption(f"📄 {r['dosya_adi']}")
            render_sonuc(r)
    with sekmeler[-1]:
        render_performans(sonuclar, batch_toplam, clf_model, seg_model)

    st.caption(f"⏱️ Toplam işlem süresi ({len(sonuclar)} görsel): {batch_toplam:.2f} saniye "
               f"— görsel başına ortalama {batch_toplam / len(sonuclar):.2f} sn")
else:
    st.session_state["imza"] = None
    render_bos_durum()

# ----------------------------------------------------------------------------
# ARŞİV
# ----------------------------------------------------------------------------
with st.expander("🗄️ Analiz Arşivi (Son 5 Kayıt)"):
    gecmis = son_analizleri_getir(5)
    if gecmis:
        for dosya, tahmin, guven, tarih, sure, orij_yol, islenmis_yol in gecmis:
            st.markdown(f"**{tarih}** — {dosya[:25]} → {CLASS_LABELS_TR.get(tahmin, tahmin)} (%{guven:.1f}) — {sure:.2f}s")
            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(orij_yol):
                    st.image(orij_yol, caption="Orijinal", use_container_width=True)
            with c2:
                if os.path.exists(islenmis_yol):
                    st.image(islenmis_yol, caption="İşlenmiş", use_container_width=True)
            st.divider()
    else:
        st.caption("Henüz analiz geçmişi yok.")

st.markdown(
    '<div class="footer-note">⚠️ Bu sistem bir araştırma prototipidir, klinik teşhis yerine geçmez. '
    'Kesin tanı için mutlaka bir uzman hekime danışın.</div>',
    unsafe_allow_html=True
)