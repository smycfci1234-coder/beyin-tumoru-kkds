# 🧠 Beyin Tümörü MR Klinik Karar Destek Sistemi (KKDS)

**TÜBİTAK 2209-B** — Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı

EfficientNetB4 (sınıflandırma) ve YOLOv11 (segmentasyon) modellerini birleştiren, Grad-CAM ile
açıklanabilirlik sağlayan, beyin tümörü MR görüntülerini analiz eden web tabanlı bir prototip.

🔗 **Canlı Demo:** [beyin-tumoru-kkds.streamlit.app](https://beyin-tumoru-kkds.streamlit.app)

---

## 📌 Proje Hakkında

Bu sistem, bir MR görüntüsü yüklendiğinde:
1. **Tümör tipini sınıflandırır** — glioma, meningioma, pituiter tümör veya tümör yok (EfficientNetB4)
2. **Tümörün sınırlarını işaretler** — piksel bazlı segmentasyon (YOLOv11)
3. **Kararını görselleştirerek açıklar** — modelin görüntünün hangi bölgesine odaklandığını gösteren ısı haritası (Grad-CAM)

⚠️ **Bu sistem bir araştırma prototipidir, klinik teşhis amaçlı kullanılamaz.**

## 🎯 Sonuçlar

| Model | Metrik | Değer |
|---|---|---|
| EfficientNetB4 (Sınıflandırma) | Test Doğruluğu | %90.1 |
| EfficientNetB4 (Sınıflandırma) | Glioma Recall | %76 |
| YOLOv11 (Segmentasyon) | mAP50 | %90.4 |
| YOLOv11 (Segmentasyon) | Precision / Recall | %89.6 / %85.1 |

## 🛠️ Kullanılan Teknolojiler

- **Model Eğitimi:** Python, TensorFlow/Keras, Ultralytics YOLOv11, Kaggle Notebooks (GPU)
- **Web Arayüzü:** Streamlit
- **Barındırma:** Streamlit Community Cloud
- **Versiyon Kontrolü:** Git + Git LFS (büyük model dosyaları için)

## 📊 Veri Setleri

- **Sınıflandırma:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7.023 görüntü, 4 sınıf
- **Segmentasyon:** [Brain Tumor Image Dataset: Semantic Segmentation](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation) — 2.146 görüntü, piksel bazlı etiketler

## 🚀 Yerel Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için:

```bash
# 1. Depoyu klonlayın
git clone https://github.com/smycfci1234-coder/beyin-tumoru-kkds.git
cd beyin-tumoru-kkds

# 2. Gerekli kütüphaneleri kurun
pip install -r requirements.txt

# 3. Uygulamayı başlatın
streamlit run app.py
```

Uygulama tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

**Not:** Model dosyaları (`best_model_v3.keras`, `yolo_best_seg.pt`) Git LFS ile saklanmaktadır.
Klonlama sırasında [Git LFS](https://git-lfs.com/) kurulu olmalıdır.

## 📁 Proje Yapısı

```
beyin-tumoru-kkds/
├── app.py                       # Streamlit web uygulaması
├── best_model_v3.keras          # Sınıflandırma modeli (EfficientNetB4)
├── yolo_best_seg.pt             # Segmentasyon modeli (YOLOv11)
├── requirements.txt             # Python kütüphane bağımlılıkları
├── packages.txt                 # Sistem seviyesi bağımlılıklar (OpenCV için)
├── runtime.txt                  # Python sürümü
└── .streamlit/
    └── config.toml              # Arayüz tema ayarları
```

## 🧪 Model Eğitim Süreci (Özet)

1. **Veri Ön İşleme:** MR görüntüleri 380×380 boyutuna getirildi, veri artırma (augmentation) uygulandı
2. **Sınıflandırma:** ImageNet ön eğitimli EfficientNetB4 üzerine transfer learning, ardından fine-tuning
3. **Sınıf Dengesizliği:** Glioma sınıfının recall değerini artırmak için manuel sınıf ağırlıklandırması uygulandı
4. **Segmentasyon:** COCO formatındaki etiketler YOLO formatına dönüştürülerek YOLOv11-nano üzerinde eğitildi
5. **Açıklanabilirlik:** Grad-CAM ile modelin karar verirken odaklandığı bölgeler görselleştirildi

## 👥 Proje Ekibi

- **Yürütücü:** Sümeyye ÇİFÇİ
- **Danışman:** Öğr. Gör. Dr. Emre ÖZGÜL
- **Kurum:** Zonguldak Bülent Ecevit Üniversitesi / Zonguldak Teknopark

## 📄 Lisans

Bu proje akademik/araştırma amaçlıdır.
