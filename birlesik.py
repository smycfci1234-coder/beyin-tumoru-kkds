"""Birden fazla görselin (örn. aynı hastanın 2-4 kesiti) birleşik değerlendirmesi."""
import numpy as np
import pandas as pd


def birlesik_degerlendir(sonuclar, class_names, class_labels):
    n = len(sonuclar)
    notumor_idx = class_names.index("notumor")
    probs = np.array([r["probs"] for r in sonuclar], dtype="float64")   # (n, sınıf)
    tumor_olasilik = 1.0 - probs[:, notumor_idx]                        # görsel başına P(tümör)
    guvenler = np.array([r["confidence"] / 100.0 for r in sonuclar])    # görsel başına güven (0-1)
    tumor_oyu = np.array([r["pred_class"] != "notumor" for r in sonuclar])
    oy_sayisi = int(tumor_oyu.sum())

    # ANA SONUÇ: Güven-ağırlıklı ortalama tümör olasılığı.
    # Modelin daha güvenli olduğu görseller, sonuca daha fazla etki eder.
    agirlik_toplam = guvenler.sum()
    if agirlik_toplam > 0:
        agirlikli_tumor_p = float((tumor_olasilik * guvenler).sum() / agirlik_toplam)
    else:
        agirlikli_tumor_p = float(tumor_olasilik.mean())

    # Basit (ağırlıksız) ortalama — karşılaştırma için
    ort_tumor_p = float(tumor_olasilik.mean())

    ana_karar = "TÜMÖR VAR" if agirlikli_tumor_p >= 0.5 else "TÜMÖR YOK"

    if oy_sayisi * 2 > n:
        cogunluk = "TÜMÖR VAR"
    elif oy_sayisi * 2 < n:
        cogunluk = "TÜMÖR YOK"
    else:
        cogunluk = "BELİRSİZ (eşit oy)"

    en_az_bir = "TÜMÖR VAR" if oy_sayisi >= 1 else "TÜMÖR YOK"

    tip_idx = [i for i in range(len(class_names)) if i != notumor_idx]
    tip_ort = probs[:, tip_idx].mean(axis=0)
    tip_ad = class_labels[class_names[tip_idx[int(np.argmax(tip_ort))]]]
    tip_orani = float(tip_ort.max() / tip_ort.sum()) if tip_ort.sum() > 0 else 0.0

    yontemler = {ana_karar, cogunluk.split(" (")[0], en_az_bir}
    celiski = len(yontemler) > 1 or (0 < oy_sayisi < n)

    return {
        "n": n, "oy_sayisi": oy_sayisi, "tumor_olasilik": tumor_olasilik,
        "agirlikli_tumor_p": agirlikli_tumor_p, "ort_tumor_p": ort_tumor_p,
        "karar": ana_karar,
        "cogunluk": cogunluk, "en_az_bir": en_az_bir,
        "tip_ad": tip_ad, "tip_orani": tip_orani, "celiski": celiski,
    }


def render_birlesik(sonuclar, class_names, class_labels, dusuk_guven_esigi=70.0):
    import streamlit as st

    b = birlesik_degerlendir(sonuclar, class_names, class_labels)
    n = b["n"]
    p = b["agirlikli_tumor_p"]

    st.markdown('<div class="panel-title">Birleşik Değerlendirme</div>', unsafe_allow_html=True)
    st.caption(f"{n} görselin sonuçları, her görselin kendi güven skoruyla ağırlıklandırılarak "
               "tek bir olasılığa birleştirilir. Görsellerin aynı hastaya ait farklı kesitler "
               "olduğu varsayılır.")

    # --- ANA SONUÇ: büyük, net olasılık ---
    renk = "#FCA5A5" if b["karar"] == "TÜMÖR VAR" else "#86EFAC"
    arka = "rgba(239,68,68,0.08)" if b["karar"] == "TÜMÖR VAR" else "rgba(34,197,94,0.08)"
    ek = f" — olası tip: {b['tip_ad']} (%{b['tip_orani']*100:.0f})" if b["karar"] == "TÜMÖR VAR" else ""
    st.markdown(f"""
    <div style="background:{arka}; border:1px solid {renk}; border-radius:14px;
                padding:1.2rem 1.5rem; margin-bottom:1rem;">
        <div style="color:#94A3B8; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">
            {n} Görsele Göre Tümör Olasılığı
        </div>
        <div style="color:{renk}; font-size:2.6rem; font-weight:800; line-height:1.2;">
            %{p*100:.1f}
        </div>
        <div style="color:{renk}; font-size:1.2rem; font-weight:700;">
            {b['karar']}{ek}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tümör diyen görsel", f"{b['oy_sayisi']} / {n}")
    c2.metric("Güven-ağırlıklı olasılık", f"%{p*100:.1f}")
    c3.metric("Basit ortalama olasılık", f"%{b['ort_tumor_p']*100:.1f}")

    if b["celiski"]:
        st.warning("⚠️ Görseller veya yöntemler birbiriyle çelişiyor. "
                   "Sonuç belirsiz kabul edilmeli, uzman değerlendirmesi önerilir.")
    if any(r["confidence"] < dusuk_guven_esigi for r in sonuclar):
        st.warning(f"⚠️ En az bir görselin güven skoru %{dusuk_guven_esigi:.0f} altında — "
                   "bu görsel ağırlıklı sonuca daha az etki etti.")

    with st.expander("Diğer yöntemlerle karşılaştırma (detay)"):
        st.dataframe(pd.DataFrame([
            {"Yöntem": "Güven-ağırlıklı olasılık (ana sonuç)", "Karar": b["karar"],
             "Açıklama": "Her görselin P(tümör) değeri, kendi güven skoruyla ağırlıklandırılıp ortalanır"},
            {"Yöntem": "Basit ortalama olasılık", "Karar": "TÜMÖR VAR" if b["ort_tumor_p"] >= 0.5 else "TÜMÖR YOK",
             "Açıklama": "Görsellerin P(tümör) değerlerinin ağırlıksız ortalaması"},
            {"Yöntem": "Çoğunluk oyu", "Karar": b["cogunluk"],
             "Açıklama": "Görsellerin çoğu tümör dediyse tümör"},
            {"Yöntem": "En az bir görselde tümör", "Karar": b["en_az_bir"],
             "Açıklama": "Tek bir görsel bile tümör dediyse tümör (duyarlılık odaklı)"},
        ]), use_container_width=True, hide_index=True)

        st.markdown("**Görsel bazında katkı**")
        st.dataframe(pd.DataFrame([{
            "Görsel": r["dosya_adi"][:30],
            "Tahmin": class_labels[r["pred_class"]],
            "Güven (%)": round(r["confidence"], 1),
            "Tümör olasılığı (%)": round(float(tp) * 100, 1),
        } for r, tp in zip(sonuclar, b["tumor_olasilik"])]),
            use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({
            "Tümör olasılığı (%)": [float(tp) * 100 for tp in b["tumor_olasilik"]]},
            index=[r["dosya_adi"][:20] for r in sonuclar]))