# 🩺 Analisis Faktor Risiko dan Prediksi Risiko Preeklamsia pada Ibu Hamil di Kabupaten Toba

**Proyek Terintegrasi Mata Kuliah Data Analitik & Data Science**  
**Institut Teknologi Del — Semester Ganjil 2026/2027**  
**Dosen Pengampu:** Oppir Hutapea, M.Kom  

---

## 📌 1. Gambaran Umum Proyek
Aplikasi web terintegrasi berbasis **Streamlit + Plotly + SQLite Data Warehouse + Scikit-Learn** ini menyajikan solusi *end-to-end* untuk membantu **Dinas Kesehatan Kabupaten Toba, 6 Puskesmas (Balige, Silaen, Lumban Julu, Laguboti, Porsea, Ajibata), dan RS Rujukan Toba** dalam:
1. **Data Analitik:** Memantau indikator kinerja utama (KPI), sebaran geospasial, tren waktu, alur rujukan longitudinal (Sankey), distribusi hemodinamik (Violin/Box), dan korelasi faktor risiko preeklamsia melalui **10 Visualisasi Interaktif (8 Jenis Visual Advance)**.
2. **Data Science:** Memprediksi risiko preeklamsia secara *real-time* pada kunjungan Antenatal Care (ANC) menggunakan **4 algoritma Machine Learning** (*Extra Trees, Random Forest, Gradient Boosting, dan Logistic Regression Baseline*) yang bebas dari kebocoran data (*zero target leakage*) sesuai aturan dikunci pada **Master 103 Variabel**.

---

## 🗂️ 2. Struktur Repositori
```text
Project-DASI_DALIS-Explainable-AI/
├── app.py                                      # Entry-point utama untuk Streamlit Community Cloud
├── requirements.txt                            # Daftar pustaka Python untuk deployment
├── README.md                                   # Dokumentasi utama repositori
├── data/
│   ├── dataset_sintetis_preeklamsia.csv        # Dataset mentah (1.000 baris ANC, 250 pasien, 103 kolom)
│   ├── cleaned_preeklamsia_toba.csv            # Dataset bersih hasil preprocessing & feature engineering
│   ├── warehouse_preeklamsia_toba.db           # Data Warehouse SQLite (Skema Bintang: 1 Fakta + 5 Dimensi)
│   └── data_quality_report.json                # Laporan metrik kualitas data sebelum & sesudah ETL
├── etl/
│   └── run_etl.py                              # Skrip otomatis Extract, Transform, Load (Star Schema)
├── models/
│   ├── train_models.py                         # Skrip pelatihan, tuning GroupKFold, & evaluasi 4 model
│   ├── preeclampsia_models_bundle.joblib       # Artefak model terbaik siap pakai di aplikasi
│   └── model_evaluation_metrics.json           # Hasil metrik uji, ROC Curve, Confusion Matrix, & Error Analysis
├── app/
│   └── streamlit_app.py                        # Kode utama aplikasi Dashboard Advance & Modul Prediksi ML
└── notebooks/
    └── EDA_dan_Pemodelan_Preeklamsia.ipynb     # Notebook eksplorasi data (5 temuan EDA) & eksperimen model
```

---

## ⚙️ 3. Cara Menjalankan Secara Lokal
1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/Viddsnrt/Project-DASI_DALIS-Explainable-AI.git
   cd Project-DASI_DALIS-Explainable-AI
   ```
2. **Instal seluruh dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Opsional) Jalankan ulang pipeline ETL dan pelatihan model:**
   ```bash
   python etl/run_etl.py
   python models/train_models.py
   ```
4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ 4. Cara Deploy Gratis ke Streamlit Community Cloud
1. Pastikan seluruh file di atas sudah di-*push* ke repositori GitHub: `https://github.com/Viddsnrt/Project-DASI_DALIS-Explainable-AI`.
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan login dengan akun GitHub.
3. Klik **Create app** -> Pilih repositori `Viddsnrt/Project-DASI_DALIS-Explainable-AI`.
4. Pilih branch `main` dan isi **Main file path** dengan `app.py` (atau `app/streamlit_app.py`).
5. Klik **Deploy!** — aplikasi akan langsung aktif dengan URL publik.
