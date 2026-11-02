"""
Aplikasi Web Terintegrasi: Dashboard Visualisasi Advance & Modul Prediksi Machine Learning
Topik: Analisis Faktor Risiko dan Prediksi Risiko Preeklamsia pada Ibu Hamil di Kabupaten Toba
Mata Kuliah: Data Analitik & Data Science - Institut Teknologi Del
"""

import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard & Prediksi Preeklamsia - Kab. Toba | IT Del",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path File
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "app" else os.path.dirname(os.path.abspath(__file__))
DW_DB_PATH = os.path.join(BASE_DIR, "data", "warehouse_preeklamsia_toba.db")
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_preeklamsia_toba.csv")
QUALITY_JSON_PATH = os.path.join(BASE_DIR, "data", "data_quality_report.json")
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "models", "preeclampsia_models_bundle.joblib")
MODEL_METRICS_PATH = os.path.join(BASE_DIR, "models", "model_evaluation_metrics.json")

# Pastikan ETL & Model telah dijalankan bila file belum ada di cloud
if not os.path.exists(CLEAN_CSV_PATH) or not os.path.exists(DW_DB_PATH):
    import subprocess
    subprocess.run(["python3", os.path.join(BASE_DIR, "etl", "run_etl.py")], check=True)
if not os.path.exists(MODEL_BUNDLE_PATH) or not os.path.exists(MODEL_METRICS_PATH):
    import subprocess
    subprocess.run(["python3", os.path.join(BASE_DIR, "models", "train_models.py")], check=True)


@st.cache_data
def load_analytic_data():
    if os.path.exists(DW_DB_PATH):
        conn = sqlite3.connect(DW_DB_PATH)
        df = pd.read_sql_query("SELECT * FROM vw_analytic_master", conn)
        conn.close()
    else:
        df = pd.read_csv(CLEAN_CSV_PATH)
    df["visit_date_dt"] = pd.to_datetime(df["visit_date"])
    return df


@st.cache_data
def load_json_metadata():
    with open(QUALITY_JSON_PATH, "r", encoding="utf-8") as f:
        q_rep = json.load(f)
    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        m_met = json.load(f)
    return q_rep, m_met


@st.cache_resource
def load_ml_bundle():
    return joblib.load(MODEL_BUNDLE_PATH)


df_master = load_analytic_data()
quality_report, model_metrics = load_json_metadata()
ml_bundle = load_ml_bundle()

# Styling CSS Kustom
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-left: 5px solid #0284c7;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 0.85rem; color: #475569; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 1.85rem; color: #0f172a; font-weight: 800; margin: 4px 0; }
    .kpi-sub { font-size: 0.80rem; color: #0369a1; font-weight: 500; }
    .story-box {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR: NAVIGASI & FILTER GLOBAL =================
st.sidebar.title("🩺 Preeklamsia Kab. Toba")
st.sidebar.caption("Proyek Terintegrasi Data Analitik & Data Science — Institut Teknologi Del")

page = st.sidebar.radio(
    "📌 Navigasi Modul Utama:",
    [
        "1. Ringkasan KPI & Geospasial",
        "2. Analisis Mendalam (Advance Viz)",
        "3. Data Storytelling & Simulasi Alur",
        "4. Modul Prediksi ML & Evaluasi",
        "5. Data Warehouse & Kualitas Data"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Global Interaktif")

all_districts = sorted(df_master["facility_district"].unique())
selected_districts = st.sidebar.multiselect(
    "Kecamatan Fasilitas (Kab. Toba):",
    options=all_districts,
    default=all_districts
)

all_trimesters = ["TRIMESTER_1", "TRIMESTER_2", "TRIMESTER_3"]
selected_trimesters = st.sidebar.multiselect(
    "Trimester Kehamilan:",
    options=all_trimesters,
    default=all_trimesters
)

all_risk_cats = ["RENDAH", "SEDANG", "TINGGI"]
selected_risk_cats = st.sidebar.multiselect(
    "Kategori Skrining Risiko (Buku KIA):",
    options=all_risk_cats,
    default=all_risk_cats
)

min_date = df_master["visit_date_dt"].min().date()
max_date = df_master["visit_date_dt"].max().date()
date_range = st.sidebar.date_input(
    "Rentang Tanggal Kunjungan ANC:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Terapkan Filter Global
df_filtered = df_master[
    (df_master["facility_district"].isin(selected_districts)) &
    (df_master["trimester"].isin(selected_trimesters)) &
    (df_master["pe_screening_risk_category"].isin(selected_risk_cats))
].copy()

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[(df_filtered["visit_date_dt"] >= start_d) & (df_filtered["visit_date_dt"] <= end_d)]

st.sidebar.info(
    f"**Data Terfilter:** {len(df_filtered):,} dari {len(df_master):,} kunjungan ANC "
    f"({df_filtered['patient_id'].nunique()} ibu hamil unik)."
)

# ================= HALAMAN 1: RINGKASAN KPI & GEOSPASIAL =================
if page == "1. Ringkasan KPI & Geospasial":
    st.title("📊 Ringkasan Eksekutif & Sebaran Spasial Preeklamsia di Kabupaten Toba")
    st.markdown(
        "Dashboard ini memantau **1.000 kunjungan Antenatal Care (ANC)** dari **250 episode kehamilan** di 6 kecamatan "
        "Kabupaten Toba untuk mendukung keputusan klinis Dinas Kesehatan, Puskesmas, dan RS Rujukan."
    )

    if len(df_filtered) == 0:
        st.warning("Tidak ada data yang sesuai dengan filter global yang dipilih.")
    else:
        # Hitung 4 KPI Utama dengan Pembanding
        total_visits = len(df_filtered)
        pe_visits = int((df_filtered["final_diagnosis"] == "PREECLAMPSIA").sum())
        pe_rate = (pe_visits / total_visits) * 100

        map_high_count = int((df_filtered["mean_arterial_pressure_mmHg"] >= 90).sum())
        map_high_rate = (map_high_count / total_visits) * 100

        prot_exam_count = int((df_filtered["proteinuria_examined_flag"] == "YA").sum())
        prot_exam_rate = (prot_exam_count / total_visits) * 100

        ref_count = int((df_filtered["referral_status"] == "DIRUJUK").sum())
        ref_rate = (ref_count / total_visits) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">1. Prevalensi Preeklamsia</div>
                <div class="kpi-val">{pe_rate:.1f}%</div>
                <div class="kpi-sub">{pe_visits} kunjungan ({df_filtered[df_filtered['final_diagnosis']=='PREECLAMPSIA']['patient_id'].nunique()} ibu) | Target Daerah: &lt;10,0%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">2. Rasio MAP Tinggi (≥90 mmHg)</div>
                <div class="kpi-val">{map_high_rate:.1f}%</div>
                <div class="kpi-sub">{map_high_count} kunjungan berisiko hemodinamik | Rata-rata MAP: {df_filtered['mean_arterial_pressure_mmHg'].mean():.1f} mmHg</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">3. Cakupan Uji Protein Urine</div>
                <div class="kpi-val">{prot_exam_rate:.1f}%</div>
                <div class="kpi-sub">{prot_exam_count} diperiksa vs {total_visits - prot_exam_count} belum diperiksa | Target ANC: 100%</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">4. Tingkat Rujukan RS & NICU</div>
                <div class="kpi-val">{ref_rate:.1f}%</div>
                <div class="kpi-sub">{ref_count} dirujuk ke RS Toba | Admisi NICU Bayi: {(df_filtered['nicu_admission_flag']=='YA').mean()*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.15, 1.0])

        with col_left:
            # VISUALISASI 1 (Advance 1): Peta Geospasial / Choropleth Bubble Kabupaten Toba
            dist_agg = df_filtered.groupby(["facility_district", "district_lat", "district_lon"], as_index=False).agg(
                total_kunjungan=("anc_visit_id", "count"),
                kasus_pe=("target_preeclampsia", "sum"),
                rerata_map=("mean_arterial_pressure_mmHg", "mean"),
                rujukan_rs=("referral_status", lambda s: (s == "DIRUJUK").sum())
            )
            dist_agg["prevalensi_pe_pct"] = (dist_agg["kasus_pe"] / dist_agg["total_kunjungan"] * 100).round(1)
            dist_agg["rerata_map"] = dist_agg["rerata_map"].round(1)

            fig_map = px.scatter_mapbox(
                dist_agg,
                lat="district_lat",
                lon="district_lon",
                size="total_kunjungan",
                color="prevalensi_pe_pct",
                hover_name="facility_district",
                hover_data={
                    "total_kunjungan": True,
                    "kasus_pe": True,
                    "prevalensi_pe_pct": True,
                    "rerata_map": True,
                    "rujukan_rs": True,
                    "district_lat": False,
                    "district_lon": False
                },
                color_continuous_scale="YlOrRd",
                size_max=32,
                zoom=9.3,
                center={"lat": 2.45, "lon": 99.08},
                mapbox_style="carto-positron",
                title="<b>Visual 1 (Geo-Map): Kecamatan Balige & Silaen Mencatat Volume ANC dan Kasus Preeklamsia Tertinggi</b>"
            )
            fig_map.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_map, use_container_width=True)

        with col_right:
            # VISUALISASI 2 (Advance 2): Sunburst Hierarki Drill-Down
            fig_sun = px.sunburst(
                df_filtered,
                path=["facility_district", "trimester", "pe_screening_risk_category", "final_diagnosis"],
                color="final_diagnosis",
                color_discrete_map={"PREECLAMPSIA": "#dc2626", "NON_PREECLAMPSIA": "#0284c7"},
                title="<b>Visual 2 (Sunburst Drill-Down): Konsentrasi Preeklamsia per Kecamatan, Trimester & Skrining</b>"
            )
            fig_sun.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_sun, use_container_width=True)

        # VISUALISASI 3 (Advance 3): Annotated Time Series
        ts_df = df_filtered.groupby(["visit_year_month", "final_diagnosis"], as_index=False).agg(
            jumlah_kunjungan=("anc_visit_id", "count"),
            rerata_sistolik=("systolic_bp_mmHg", "mean")
        ).sort_values("visit_year_month")

        fig_ts = px.line(
            ts_df,
            x="visit_year_month",
            y="jumlah_kunjungan",
            color="final_diagnosis",
            markers=True,
            color_discrete_map={"PREECLAMPSIA": "#dc2626", "NON_PREECLAMPSIA": "#0284c7"},
            labels={"visit_year_month": "Bulan Kunjungan ANC", "jumlah_kunjungan": "Jumlah Kunjungan ANC", "final_diagnosis": "Diagnosis Akhir"},
            title="<b>Visual 3 (Deret Waktu Beranotasi): Tren Bulanan Kunjungan ANC & Puncak Deteksi Preeklamsia (2025–2026)</b>"
        )
        # Tambahkan anotasi pada puncak kasus Preeklamsia
        pe_ts = ts_df[ts_df["final_diagnosis"] == "PREECLAMPSIA"]
        if not pe_ts.empty:
            peak_row = pe_ts.loc[pe_ts["jumlah_kunjungan"].idxmax()]
            fig_ts.add_annotation(
                x=peak_row["visit_year_month"],
                y=peak_row["jumlah_kunjungan"],
                text=f"Puncak Kasus PE: {int(peak_row['jumlah_kunjungan'])} kunjungan ({peak_row['visit_year_month']})",
                showarrow=True,
                arrowhead=2,
                bgcolor="#fee2e2",
                bordercolor="#dc2626"
            )
        fig_ts.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_ts, use_container_width=True)


# ================= HALAMAN 2: ANALISIS MENDALAM (ADVANCE VIZ) =================
elif page == "2. Analisis Mendalam (Advance Viz)":
    st.title("🔬 Analisis Klinis Mendalam & Eksplorasi Multidimensi (Visual 4 s.d. 8)")
    st.markdown(
        "Halaman ini menyajikan **5 visualisasi lanjutan (*advance visualizations*)** untuk membedah distribusi "
        "hemodinamik, alur perjalanan klinis pasien, profil laboratorium, serta korelasi antar-variabel."
    )

    # Baris 1: Violin/Box Plot (Visual 4) & Correlation Heatmap (Visual 5)
    c_v1, c_v2 = st.columns(2)

    with c_v1:
        metric_choice = st.selectbox(
            "Pilih Parameter Klinis untuk Distribusi Violin & Box Plot:",
            options=[
                ("mean_arterial_pressure_mmHg", "Mean Arterial Pressure / MAP (mmHg)"),
                ("systolic_bp_mmHg", "Tekanan Darah Sistolik (mmHg)"),
                ("diastolic_bp_mmHg", "Tekanan Darah Diastolik (mmHg)"),
                ("prepregnancy_bmi", "IMT Pra-Kehamilan (kg/m²)"),
                ("birth_weight_g", "Berat Lahir Bayi (gram)")
            ],
            format_func=lambda x: x[1]
        )
        fig_violin = px.violin(
            df_filtered,
            x="trimester",
            y=metric_choice[0],
            color="final_diagnosis",
            box=True,
            points="outliers",
            category_orders={"trimester": ["TRIMESTER_1", "TRIMESTER_2", "TRIMESTER_3"]},
            color_discrete_map={"PREECLAMPSIA": "#dc2626", "NON_PREECLAMPSIA": "#0284c7"},
            labels={"trimester": "Trimester Kehamilan", metric_choice[0]: metric_choice[1], "final_diagnosis": "Diagnosis"},
            title=f"<b>Visual 4 (Violin & Box Plot): Lonjakan {metric_choice[1]} pada Kasus Preeklamsia</b>"
        )
        if metric_choice[0] == "mean_arterial_pressure_mmHg":
            fig_violin.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Ambang Waspada MAP (90 mmHg)")
        fig_violin.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_violin, use_container_width=True)

    with c_v2:
        corr_cols = [
            "maternal_age_years", "prepregnancy_bmi", "gestational_age_weeks",
            "systolic_bp_mmHg", "diastolic_bp_mmHg", "mean_arterial_pressure_mmHg",
            "cumulative_risk_score", "clinical_symptom_score", "proteinuria_dipstick_score",
            "birth_weight_g", "target_preeclampsia"
        ]
        corr_names = [
            "Usia Ibu", "IMT Pra-Hamil", "Usia Kehamilan",
            "TD Sistolik", "TD Diastolik", "MAP",
            "Skor Risiko Ibu", "Skor Gejala ANC", "Skor Proteinuria",
            "Berat Lahir Bayi", "Diagnosis PE"
        ]
        corr_matrix = df_filtered[corr_cols].corr().round(2)
        corr_matrix.index = corr_names
        corr_matrix.columns = corr_names

        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="<b>Visual 5 (Heatmap Korelasi): MAP (r=0.54) & Skor Risiko Berhubungan Kuat dengan Preeklamsia</b>"
        )
        fig_heat.update_layout(height=450, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Baris 2: Sankey Diagram Alur Klinis (Visual 6)
    st.subheader("🔄 Visual 6 (Diagram Sankey): Alur Perjalanan Klinis Ibu Hamil dari Skrining Puskesmas hingga Persalinan")
    sankey_df = df_filtered.copy()
    s1 = "Skrining: " + sankey_df["pe_screening_risk_category"]
    s2 = "Rujukan: " + sankey_df["referral_status"]
    s3 = "Diagnosis: " + sankey_df["final_diagnosis"]
    s4 = "Persalinan: " + sankey_df["delivery_mode"]

    links_12 = pd.DataFrame({"source": s1, "target": s2}).value_counts().reset_index(name="value")
    links_23 = pd.DataFrame({"source": s2, "target": s3}).value_counts().reset_index(name="value")
    links_34 = pd.DataFrame({"source": s3, "target": s4}).value_counts().reset_index(name="value")
    all_links = pd.concat([links_12, links_23, links_34], ignore_index=True)

    unique_nodes = list(pd.unique(all_links[["source", "target"]].values.ravel("K")))
    node_map = {node: idx for idx, node in enumerate(unique_nodes)}

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=unique_nodes,
            color=["#38bdf8" if "RENDAH" in n or "NON_PREECLAMPSIA" in n or "TIDAK_DIRUJUK" in n else "#f87171" if "TINGGI" in n or "PREECLAMPSIA" in n or "DIRUJUK" in n else "#fbbf24" for n in unique_nodes]
        ),
        link=dict(
            source=all_links["source"].map(node_map),
            target=all_links["target"].map(node_map),
            value=all_links["value"]
        )
    )])
    fig_sankey.update_layout(
        title_text="<b>Alur Longitudinal: Kategori Skrining Risiko → Status Rujukan Puskesmas → Diagnosis Akhir RS → Metode Persalinan</b>",
        height=400,
        margin=dict(l=15, r=15, t=40, b=15)
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # Baris 3: Parallel Coordinates (Visual 7) & Small Multiples Faceted Scatter (Visual 8)
    c_v3, c_v4 = st.columns(2)

    with c_v3:
        fig_par = px.parallel_coordinates(
            df_filtered,
            color="target_preeclampsia",
            dimensions=[
                "maternal_age_years",
                "prepregnancy_bmi",
                "systolic_bp_mmHg",
                "diastolic_bp_mmHg",
                "mean_arterial_pressure_mmHg",
                "birth_weight_g"
            ],
            labels={
                "maternal_age_years": "Usia (Thn)",
                "prepregnancy_bmi": "IMT Pra-Hamil",
                "systolic_bp_mmHg": "Sistolik",
                "diastolic_bp_mmHg": "Diastolik",
                "mean_arterial_pressure_mmHg": "MAP (mmHg)",
                "birth_weight_g": "Berat Bayi (g)",
                "target_preeclampsia": "Diagnosis PE (1=Ya)"
            },
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="<b>Visual 7 (Parallel Coordinates): Profil Multidimensi Ibu Hamil Preeklamsia (Merah) vs Normal</b>"
        )
        fig_par.update_layout(height=420, margin=dict(l=25, r=25, t=45, b=15))
        st.plotly_chart(fig_par, use_container_width=True)

    with c_v4:
        fig_facet = px.scatter(
            df_filtered,
            x="gestational_age_weeks",
            y="mean_arterial_pressure_mmHg",
            color="final_diagnosis",
            facet_col="facility_district",
            facet_col_wrap=3,
            opacity=0.75,
            color_discrete_map={"PREECLAMPSIA": "#dc2626", "NON_PREECLAMPSIA": "#0284c7"},
            labels={
                "gestational_age_weeks": "Usia Kehamilan (Mgg)",
                "mean_arterial_pressure_mmHg": "MAP (mmHg)",
                "final_diagnosis": "Diagnosis"
            },
            title="<b>Visual 8 (Small Multiples): Tren Kenaikan MAP terhadap Usia Kehamilan di 6 Kecamatan</b>"
        )
        fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_facet.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_facet, use_container_width=True)


# ================= HALAMAN 3: STORYTELLING & SIMULASI ALUR =================
elif page == "3. Data Storytelling & Simulasi Alur":
    st.title("📖 Data Storytelling, Faktor Risiko Dominan & Simulasi Perjalanan Pasien")
    st.markdown(
        "Halaman ini merangkum **temuan kunci untuk pengambilan keputusan (Keputusan Klinis & Kebijakan Dinas Kesehatan)** "
        "serta menampilkan **Simulasi Alur Longitudinal Ibu Hamil (Tahap 1 s.d. Tahap 8)** sesuai *Master Variabel*."
    )

    # VISUALISASI 9: Ranked Risk Prevalence Bar Chart
    risk_cols_info = [
        ("antiphospholipid_syndrome_flag", "Sindrom Antifosfolipid (APS)"),
        ("chronic_hypertension_flag", "Hipertensi Kronik"),
        ("kidney_disease_flag", "Riwayat Penyakit Ginjal"),
        ("visual_disturbance_present", "Keluhan Gangguan Penglihatan"),
        ("previous_preeclampsia_flag", "Riwayat Preeklamsia Sebelumnya"),
        ("preexisting_diabetes_flag", "Diabetes Pra-Kehamilan"),
        ("edema_persistent_flag", "Edema Menetap (Persisten)"),
        ("headache_present", "Keluhan Sakit Kepala saat ANC"),
        ("assisted_reproduction_flag", "Teknologi Reproduksi Berbantu"),
        ("obesity_prepregnancy_flag", "Obesitas Pra-Kehamilan (IMT ≥30)"),
        ("family_history_preeclampsia_flag", "Riwayat Preeklamsia Keluarga"),
        ("edema_present", "Edema / Bengkak Saat ANC"),
        ("new_partner_pregnancy_flag", "Kehamilan dengan Pasangan Baru"),
        ("nulliparity_flag", "Nulipara (Kehamilan Pertama)")
    ]

    risk_rows = []
    for col, label in risk_cols_info:
        sub_ya = df_filtered[df_filtered[col] == "YA"]
        sub_tdk = df_filtered[df_filtered[col] == "TIDAK"]
        rate_ya = (sub_ya["target_preeclampsia"].mean() * 100) if len(sub_ya) > 0 else 0.0
        rate_tdk = (sub_tdk["target_preeclampsia"].mean() * 100) if len(sub_tdk) > 0 else 0.0
        risk_rows.append({
            "Faktor Risiko": label,
            "Prevalensi PE jika YA (%)": round(rate_ya, 1),
            "Prevalensi PE jika TIDAK (%)": round(rate_tdk, 1),
            "Jumlah Kunjungan (YA)": len(sub_ya),
            "Relative Risk (RR)": round(rate_ya / max(rate_tdk, 0.1), 2)
        })
    risk_df = pd.DataFrame(risk_rows).sort_values("Prevalensi PE jika YA (%)", ascending=True)

    c_s1, c_s2 = st.columns([1.35, 1.0])
    with c_s1:
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            y=risk_df["Faktor Risiko"],
            x=risk_df["Prevalensi PE jika YA (%)"],
            name="Memiliki Faktor Risiko (YA)",
            orientation="h",
            marker_color="#dc2626",
            text=risk_df["Prevalensi PE jika YA (%)"].astype(str) + "%",
            textposition="outside"
        ))
        fig_risk.add_trace(go.Bar(
            y=risk_df["Faktor Risiko"],
            x=risk_df["Prevalensi PE jika TIDAK (%)"],
            name="Tanpa Faktor Risiko (TIDAK)",
            orientation="h",
            marker_color="#94a3b8"
        ))
        fig_risk.update_layout(
            barmode="group",
            title="<b>Visual 9 (Risk Ranking): APS (100%), Hipertensi Kronik (66%), dan Penyakit Ginjal (62,5%) Memicu Risiko Tertinggi</b>",
            xaxis_title="Prevalensi Diagnosis Preeklamsia (%)",
            height=500,
            margin=dict(l=10, r=30, t=45, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with c_s2:
        st.markdown("""
        <div class="story-box">
            <h4>🎯 3 Insight Utama & Rekomendasi Keputusan (Actionable Insights)</h4>
            <ol>
                <li><b>Prioritaskan Skrining Komorbiditas Sejak Kunjungan Pertama (Trimester 1):</b><br>
                Ibu hamil dengan <i>Hipertensi Kronik</i> (risiko 66,0% vs 13,7%), <i>Penyakit Ginjal</i> (62,5%), dan <i>Riwayat Preeklamsia</i> (51,3%) memiliki risiko >3,5x lipat mengalami preeklamsia. Petugas ANC di 6 Puskesmas wajib melabeli buku KIA pasien ini sebagai <b>Risiko Tinggi</b> sejak K1.</li>
                <li><b>Tutup Kesenjangan Pemeriksaan Proteinuria Dipstick di Puskesmas:</b><br>
                Saat ini cakupan uji protein urine baru mencapai <b>26,9%</b> (731 dari 1.000 kunjungan berstatus <code>TIDAK_DIPERIKSA</code>). Padahal, 100% pasien dengan hasil dipstick <code>2+</code> dan <code>3+</code> terbukti positif preeklamsia. Dinas Kesehatan Kabupaten Toba perlu mendistribusikan strip reagen dipstick agar setiap ibu hamil dengan MAP ≥90 mmHg wajib diperiksa urine.</li>
                <li><b>Intervensi Dini di Kecamatan Balige, Silaen, dan Lumban Julu:</b><br>
                Ketiga kecamatan ini menyumbang 59% beban kunjungan ANC daerah. Penggunaan model prediksi ML pada Trimester 2 dapat mencegah keterlambatan rujukan ke RS Rujukan Toba dan menekan angka kelahiran prematur serta admisi NICU.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏥 Penelusuran Longitudinal Pasien (Simulasi 8 Tahap Sesuai Master Variabel)")
    st.caption(
        "Pilih kode pasien (`patient_id`) untuk melihat bagaimana 103 variabel dicatat secara kronologis dari ANC 1 di Puskesmas "
        "hingga rujukan, diagnosis RS, persalinan, dan luaran bayi."
    )

    pe_patients = sorted(df_master[df_master["final_diagnosis"] == "PREECLAMPSIA"]["patient_id"].unique())
    non_pe_patients = sorted(df_master[df_master["final_diagnosis"] == "NON_PREECLAMPSIA"]["patient_id"].unique())
    selected_ptn = st.selectbox(
        "Pilih Pasien (Didahulukan Kasus Preeklamsia untuk Studi Kasus):",
        options=pe_patients + non_pe_patients,
        index=0
    )

    ptn_visits = df_master[df_master["patient_id"] == selected_ptn].sort_values("visit_number")
    first_row = ptn_visits.iloc[0]

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"**Tahap 1 & 2 — Profil Ibu & Kehamilan (`{first_row['patient_id']}` / `{first_row['pregnancy_id']}`)**")
        st.write(f"- **Usia & Domisili:** {first_row['maternal_age_years']} thn | Kec. {first_row['maternal_domicile_district']}")
        st.write(f"- **BB / TB / IMT Awal:** {first_row['prepregnancy_weight_kg']} kg / {first_row['height_cm']} cm ({first_row['prepregnancy_bmi']} kg/m² - Obesitas: `{first_row['obesity_prepregnancy_flag']}`)")
        st.write(f"- **Obstetri:** G{first_row['gravida']}P{first_row['parity']}A{first_row['abortion_count']} | Nulipara: `{first_row['nulliparity_flag']}` | Jarak Hamil: `{first_row['interpregnancy_interval_years']}` thn")
        st.write(f"- **Faktor Risiko Komorbiditas:** Hipertensi Kronik=`{first_row['chronic_hypertension_flag']}`, Riwayat PE=`{first_row['previous_preeclampsia_flag']}`, Riwayat Keluarga=`{first_row['family_history_preeclampsia_flag']}`, Ginjal=`{first_row['kidney_disease_flag']}`")
    with col_p2:
        st.markdown("**Tahap 6 — Skrining, Rujukan & Diagnosis Akhir**")
        st.write(f"- **Fasilitas ANC:** {first_row['facility_name']} ({first_row['facility_district']})")
        st.write(f"- **Diagnosis Akhir (Target):** `{first_row['final_diagnosis']}` ({first_row['preeclampsia_severity']})")
        st.write(f"- **Tanggal Diagnosis PE:** `{first_row['preeclampsia_diagnosis_date']}`")
        st.write(f"- **Status Rujukan Tertinggi:** `{'DIRUJUK' if (ptn_visits['referral_status']=='DIRUJUK').any() else 'TIDAK_DIRUJUK'}`")
    with col_p3:
        st.markdown(f"**Tahap 7 & 8 — Persalinan & Outcome Bayi (`{first_row['baby_id']}`)**")
        st.write(f"- **Tgl & Usia Persalinan:** {first_row['delivery_date']} ({first_row['gestational_age_at_delivery_weeks']} minggu)")
        st.write(f"- **Metode Persalinan:** `{first_row['delivery_mode']}` | Prematur: `{first_row['preterm_delivery_flag']}`")
        st.write(f"- **Berat & Panjang Bayi:** {first_row['birth_weight_g']} gram (`BBLR: {first_row['low_birth_weight_flag']}`) / {first_row['birth_length_cm']} cm")
        st.write(f"- **APGAR (1m/5m) & NICU:** {first_row['apgar_1min']} / {first_row['apgar_5min']} | Masuk NICU: `{first_row['nicu_admission_flag']}`")

    st.markdown("**Tahap 3, 4 & 5 — Riwayat Kronologis Kunjungan ANC, Tekanan Darah, Gejala & Laboratorium:**")
    st.dataframe(
        ptn_visits[[
            "visit_number", "anc_visit_id", "visit_date", "trimester", "gestational_age_weeks",
            "weight_kg", "systolic_bp_mmHg", "diastolic_bp_mmHg", "mean_arterial_pressure_mmHg",
            "systolic_bp_change_mmHg", "edema_present", "headache_present",
            "proteinuria_examined_flag", "proteinuria_dipstick_result",
            "pe_screening_risk_category", "suspected_preeclampsia_flag", "referral_status"
        ]],
        use_container_width=True
    )


# ================= HALAMAN 4: MODUL PREDIKSI ML & EVALUASI =================
elif page == "4. Modul Prediksi ML & Evaluasi":
    st.title("🤖 Modul Prediksi Risiko Preeklamsia & Evaluasi Performa Model (Data Science)")
    st.markdown(
        "Modul ini mengintegrasikan **4 algoritma Machine Learning** yang dilatih pada fitur klinis ANC Puskesmas "
        "**tanpa kebocoran data (*zero target leakage*)** menggunakan skema validasi **`GroupKFold` (5-Fold) berbasis `patient_id`**."
    )

    tab_pred, tab_eval, tab_err, tab_card = st.tabs([
        "🩺 Simulasi Prediksi Pasien Baru",
        "📈 Perbandingan 4 Model & Visual 10 (ROC/CM)",
        "🔍 Analisis Kesalahan (Error Analysis)",
        "📋 Model Card"
    ])

    with tab_pred:
        st.subheader("Formulir Input Klinis Kunjungan ANC Ibu Hamil")
        st.caption("Masukkan data anamnesis dan pemeriksaan fisik ANC untuk memprediksi probabilitas risiko Preeklamsia secara real-time.")

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown("**1. Karakteristik Ibu & Obstetri**")
            in_age = st.number_input("Usia Ibu (Tahun)", min_value=15, max_value=50, value=34)
            in_height = st.number_input("Tinggi Badan (cm)", min_value=135.0, max_value=185.0, value=155.0, step=0.5)
            in_pre_wt = st.number_input("Berat Badan Pra-Hamil (kg)", min_value=35.0, max_value=120.0, value=68.0, step=0.5)
            in_ga = st.slider("Usia Kehamilan Saat ANC (Minggu)", min_value=6, max_value=40, value=28)
            in_gravida = st.number_input("Gravida (Kehamilan ke-)", min_value=1, max_value=10, value=2)
            in_parity = st.number_input("Paritas (Jumlah Kelahiran)", min_value=0, max_value=8, value=1)
            in_abort = st.number_input("Riwayat Abortus", min_value=0, max_value=5, value=0)
            in_interval = st.number_input("Jarak Kehamilan Sebelumnya (Thn, 0 jika Nulipara)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

        with f2:
            st.markdown("**2. Riwayat Penyakit Maternal**")
            in_prev_pe = st.checkbox("Riwayat Preeklamsia Sebelumnya", value=False)
            in_fam_pe = st.checkbox("Riwayat Preeklamsia Keluarga", value=True)
            in_ch_htn = st.checkbox("Hipertensi Kronik", value=False)
            in_pre_dm = st.checkbox("Diabetes Pra-Kehamilan", value=False)
            in_gest_dm = st.checkbox("Diabetes Gestasional", value=False)
            in_kidney = st.checkbox("Riwayat Penyakit Ginjal", value=False)
            in_aps = st.checkbox("Sindrom Antifosfolipid (APS)", value=False)
            in_new_part = st.checkbox("Kehamilan dengan Pasangan Baru", value=False)
            in_art = st.checkbox("Teknologi Reproduksi Berbantu (IVF/ART)", value=False)
            in_multi = st.checkbox("Kehamilan Ganda / Kembar", value=False)

        with f3:
            st.markdown("**3. Pemeriksaan Fisik ANC (10T)**")
            in_curr_wt = st.number_input("Berat Badan Kunjungan Sekarang (kg)", min_value=35.0, max_value=130.0, value=74.5, step=0.5)
            in_lila = st.number_input("Lingkar Lengan Atas / LILA (cm)", min_value=18.0, max_value=40.0, value=26.5, step=0.5)
            in_sbp = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=80, max_value=220, value=144)
            in_dbp = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=50, max_value=140, value=92)
            in_sbp_chg = st.number_input("Perubahan Sistolik dari ANC Lalu (mmHg)", min_value=-40, max_value=80, value=18)
            in_dbp_chg = st.number_input("Perubahan Diastolik dari ANC Lalu (mmHg)", min_value=-30, max_value=50, value=12)
            in_prior_vis = st.number_input("Jumlah Kunjungan ANC Sebelumnya", min_value=0, max_value=8, value=2)

        with f4:
            st.markdown("**4. Gejala Klinis & Lab Puskesmas**")
            in_edema = st.checkbox("Edema / Bengkak Saat Kunjungan", value=True)
            in_edema_pers = st.checkbox("Edema Menetap (Persisten)", value=False)
            in_headache = st.checkbox("Keluhan Sakit Kepala", value=True)
            in_visual = st.checkbox("Gangguan Penglihatan / Kabur", value=False)
            in_epigastric = st.checkbox("Nyeri Ulu Hati / Epigastrium", value=False)
            in_dyspnea = st.checkbox("Keluhan Sesak Napas", value=False)
            in_dipstick = st.selectbox(
                "Hasil Uji Dipstick Protein Urine:",
                options=["TIDAK_DIPERIKSA", "NEGATIF", "TRACE", "1+", "2+", "3+"],
                index=3
            )
            in_hb = st.number_input("Kadar Hemoglobin / Hb (g/dL)", min_value=6.0, max_value=17.0, value=11.4, step=0.1)

        # Hitung fitur turunan otomatis sesuai rumus di ETL
        calc_bmi = round(in_pre_wt / ((in_height / 100.0) ** 2), 1)
        calc_obese = 1 if calc_bmi >= 30.0 else 0
        calc_nullipara = 1 if in_parity == 0 else 0
        calc_fetus = 2 if in_multi else 1
        calc_wt_chg = round(in_curr_wt - in_pre_wt, 1)
        calc_wt_rate = round(calc_wt_chg / max(in_ga, 1), 3)
        calc_map = round((in_sbp + 2 * in_dbp) / 3.0, 1)
        calc_pp = in_sbp - in_dbp
        dip_map = {"TIDAK_DIPERIKSA": 0, "NEGATIF": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4}
        calc_dip_score = dip_map[in_dipstick]
        calc_prot_exam = 0 if in_dipstick == "TIDAK_DIPERIKSA" else 1

        calc_cum_risk = sum([
            calc_obese, calc_nullipara, int(in_new_part), int(in_art), int(in_multi),
            int(in_prev_pe), int(in_fam_pe), int(in_ch_htn), int(in_pre_dm),
            int(in_gest_dm), int(in_kidney), int(in_aps)
        ])
        calc_symp_score = sum([
            int(in_edema), int(in_edema_pers), int(in_headache),
            int(in_visual), int(in_epigastric), int(in_dyspnea)
        ])
        calc_high_age = 1 if (in_age < 20 or in_age >= 35) else 0
        calc_htn_vis = 1 if (in_sbp >= 140 or in_dbp >= 90) else 0

        input_vector = pd.DataFrame([{
            "maternal_age_years": in_age,
            "prepregnancy_bmi": calc_bmi,
            "obesity_prepregnancy_flag_bin": calc_obese,
            "gestational_age_weeks": in_ga,
            "gravida": in_gravida,
            "parity": in_parity,
            "abortion_count": in_abort,
            "nulliparity_flag_bin": calc_nullipara,
            "interpregnancy_interval_clean": 0.0 if calc_nullipara else in_interval,
            "new_partner_pregnancy_flag_bin": int(in_new_part),
            "assisted_reproduction_flag_bin": int(in_art),
            "fetus_count": calc_fetus,
            "multiple_pregnancy_flag_bin": int(in_multi),
            "previous_preeclampsia_flag_bin": int(in_prev_pe),
            "family_history_preeclampsia_flag_bin": int(in_fam_pe),
            "chronic_hypertension_flag_bin": int(in_ch_htn),
            "preexisting_diabetes_flag_bin": int(in_pre_dm),
            "gestational_diabetes_flag_bin": int(in_gest_dm),
            "kidney_disease_flag_bin": int(in_kidney),
            "antiphospholipid_syndrome_flag_bin": int(in_aps),
            "weight_change_clean": calc_wt_chg,
            "weight_gain_per_week": calc_wt_rate,
            "lila_cm": in_lila,
            "systolic_bp_mmHg": in_sbp,
            "diastolic_bp_mmHg": in_dbp,
            "mean_arterial_pressure_mmHg": calc_map,
            "pulse_pressure_mmHg": calc_pp,
            "systolic_bp_change_clean": in_sbp_chg,
            "diastolic_bp_change_clean": in_dbp_chg,
            "prior_visit_count": in_prior_vis,
            "edema_present_bin": int(in_edema),
            "edema_persistent_flag_bin": int(in_edema_pers),
            "headache_present_bin": int(in_headache),
            "visual_disturbance_present_bin": int(in_visual),
            "epigastric_or_ruq_pain_present_bin": int(in_epigastric),
            "dyspnea_present_bin": int(in_dyspnea),
            "proteinuria_examined_flag_bin": calc_prot_exam,
            "proteinuria_dipstick_score": calc_dip_score,
            "hemoglobin_clean": in_hb,
            "cumulative_risk_score": calc_cum_risk,
            "clinical_symptom_score": calc_symp_score,
            "high_risk_age_flag_bin": calc_high_age,
            "hypertension_visit_flag_bin": calc_htn_vis
        }])[ml_bundle["feature_columns"]]

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi Seluruh Model & Rekomendasi Klinis")
        st.write(
            f"**Ringkasan Parameter Turunan Otomatis:** IMT Pra-Hamil = `{calc_bmi} kg/m²` | "
            f"MAP = `{calc_map} mmHg` | Tekanan Nadi = `{calc_pp} mmHg` | "
            f"Skor Akumulasi Risiko Maternal = `{calc_cum_risk}` | Skor Gejala Bahaya = `{calc_symp_score}`"
        )

        pred_cols = st.columns(4)
        for idx, (m_name, estimator) in enumerate(ml_bundle["models"].items()):
            prob_pe = float(estimator.predict_proba(input_vector)[0, 1])
            opt_th = model_metrics["models"][m_name]["optimal_threshold"]
            is_pe = prob_pe >= opt_th
            status_label = "🔴 RISIKO PREEKLAMSIA" if is_pe else "🟢 NON-PREEKLAMSIA"
            with pred_cols[idx]:
                st.metric(
                    label=m_name,
                    value=f"{prob_pe * 100:.1f}%",
                    delta=f"{status_label} (Ambang: {opt_th*100:.0f}%)",
                    delta_color="inverse" if is_pe else "normal"
                )

    with tab_eval:
        st.subheader("📊 Tabel Perbandingan Metrik Evaluasi 4 Model pada Data Uji (249 Kunjungan)")
        eval_rows = []
        for m_name, m_stat in model_metrics["models"].items():
            eval_rows.append({
                "Algoritma Model": m_name,
                "Accuracy": m_stat["accuracy"],
                "Precision": m_stat["precision"],
                "Recall (Sensitivity)": m_stat["recall"],
                "F1-Score": m_stat["f1_score"],
                "ROC-AUC": m_stat["roc_auc"],
                "Waktu Latih (detik)": m_stat["train_time_sec"],
                "Hyperparameter Terbaik": json.dumps(m_stat["best_params"])
            })
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

        # VISUALISASI 10: ROC Curve Multi-Model & Confusion Matrix + Feature Importance
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            fig_roc = go.Figure()
            for m_name, m_stat in model_metrics["models"].items():
                fig_roc.add_trace(go.Scatter(
                    x=m_stat["roc_curve"]["fpr"],
                    y=m_stat["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"{m_name} (AUC = {m_stat['roc_auc']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random Guess (AUC = 0.500)"
            ))
            fig_roc.update_layout(
                title="<b>Visual 10A (Kurva ROC Multi-Model): Extra Trees (0,949) & Random Forest (0,939) Unggul</b>",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall / Sensitivity)",
                height=420,
                margin=dict(l=10, r=10, t=45, b=10)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_e2:
            top_fi = pd.DataFrame(model_metrics["feature_importance"][:12]).sort_values("importance", ascending=True)
            fig_fi = px.bar(
                top_fi,
                x="importance",
                y="label",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                title="<b>Visual 10B (Top 12 Feature Importance): Skor Risiko, MAP & Tekanan Darah Mendominasi</b>",
                labels={"importance": "Kepentingan Fitur (Gini Importance)", "label": "Fitur Klinis"}
            )
            fig_fi.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

    with tab_err:
        st.subheader("🔍 Analisis Kesalahan (Error Analysis pada Data Uji)")
        st.markdown(
            "Sesuai rubrik penilaian **S5 (Error Analysis)**, berikut adalah bedah kasus *False Negative* dan *False Positive* "
            "beserta penjelasan klinis mengapa kesalahan klasifikasi terjadi:"
        )
        col_fn, col_fp = st.columns(2)
        with col_fn:
            st.markdown("**Contoh Kasus False Negative (Preeklamsia Terlewat pada Kunjungan Awal):**")
            st.dataframe(pd.DataFrame(model_metrics["error_analysis"]["false_negatives"]), use_container_width=True)
            st.info(
                "**Penyebab Klinis (False Negative):** Terjadi pada kunjungan **Trimester 1 atau awal Trimester 2** "
                "ketika tekanan darah ibu masih normal (<120/80 mmHg) dan proteinuria belum diperiksa (`TIDAK_DIPERIKSA`), "
                "namun pasien mengalami *late-onset preeclampsia* pada akhir Trimester 3."
            )
        with col_fp:
            st.markdown("**Contoh Kasus False Positive (Non-Preeklamsia Terprediksi Berisiko Tinggi):**")
            st.dataframe(pd.DataFrame(model_metrics["error_analysis"]["false_positives"]), use_container_width=True)
            st.warning(
                "**Penyebab Klinis (False Positive):** Pasien memiliki akumulasi komorbiditas tinggi (mis. obesitas pra-hamil "
                "+ usia ≥35 tahun + kenaikan tekanan darah borderline), tetapi hingga persalinan tidak berkembang menjadi preeklamsia."
            )

    with tab_card:
        st.subheader("📋 Model Card (Dokumentasi Standar Model Machine Learning)")
        st.markdown("""
        | Komponen Model Card | Rincian Spesifikasi |
        | :--- | :--- |
        | **Nama & Versi Model** | `Preeclampsia-Toba-Ensemble v1.0` (Extra Trees, Random Forest, Gradient Boosting & Logistic Regression) |
        | **Tujuan Penggunaan** | Sistem pendukung keputusan klinis (*Clinical Decision Support System*) untuk deteksi dini risiko preeklamsia pada kunjungan ANC di Puskesmas Kabupaten Toba. |
        | **Data Latih & Validasi** | 1.000 baris kunjungan ANC (250 ibu hamil unik) dari `dataset_sintetis_preeklamsia.csv`. Dibagi menggunakan `GroupShuffleSplit` (751 latih : 249 uji) dan `GroupKFold` (5-Fold) berbasis `patient_id`. |
        | **Performa Data Uji** | **Extra Trees (Tuned):** ROC-AUC = `0,9492`, F1 = `0,7742`, Recall = `0,8571`, Accuracy = `0,9157`<br>**Random Forest (Tuned):** ROC-AUC = `0,9389`, Accuracy = `0,8635`<br>**Logistic Regression (Baseline):** ROC-AUC = `0,9281`, Recall = `0,7619` |
        | **Pencegahan Data Leakage** | Mengikuti aturan dikunci *Master Variabel*: seluruh kolom diagnosis/skrining (`pe_screening_risk_category`, `suspected_preeclampsia_flag`, `referral_*`) serta kolom outcome persalinan/bayi dikeluarkan sepenuhnya dari matriks fitur `X`. |
        | **Keterbatasan & Etika** | Dilatih pada data sintetis terkalibrasi Kabupaten Toba. Tidak menggantikan diagnosis medis dokter spesialis obstetri dan ginekologi (Sp.OG). |
        """)


# ================= HALAMAN 5: DATA WAREHOUSE & KUALITAS DATA =================
elif page == "5. Data Warehouse & Kualitas Data":
    st.title("🗄️ Arsitektur Data Warehouse (Star Schema) & Laporan Kualitas Data")
    st.markdown(
        "Halaman ini mendokumentasikan lapisan **Data Engineering & Data Analitik (Komponen A2 & A3)**: "
        "pembersihan nilai sentinel, skema bintang (*Star Schema*), dan *Data Dictionary* 103 variabel."
    )

    c_q1, c_q2 = st.columns(2)
    with c_q1:
        st.subheader("1. Ringkasan Audit Kualitas Data (Sebelum vs Sesudah ETL)")
        q_table = pd.DataFrame([
            {"Dimensi Kualitas": "Completeness (Sentinel Numerik -1)", "Sebelum ETL": f"{quality_report['sentinel_numeric_neg1_before']:,} sel bernilai -1", "Sesudah ETL": "0 sel -1 (Dipetakan ke NaN untuk EDA & Imputasi + Flag _is_examined untuk ML)"},
            {"Dimensi Kualitas": "Completeness (Sentinel Kategori)", "Sebelum ETL": f"{quality_report['sentinel_string_before']:,} sel TIDAK_DIPERIKSA / TIDAK_BERLAKU", "Sesudah ETL": "Dipisahkan secara eksplisit antara 'Tidak Diperiksa' dan 'Hasil Negatif'"},
            {"Dimensi Kualitas": "Uniqueness (Duplikasi Kunjungan)", "Sebelum ETL": f"{quality_report['duplicates_before']} baris duplikat", "Sesudah ETL": "1.000 anc_visit_id unik & 250 patient_id unik tervalidasi"},
            {"Dimensi Kualitas": "Validity (Kolom Konstan & Rumus)", "Sebelum ETL": f"{len(quality_report['constant_columns_detected'])} kolom konstan", "Sesudah ETL": f"Deviasi rumus MAP = {quality_report['map_max_formula_deviation']} mmHg; Deviasi IMT = {quality_report['bmi_max_formula_deviation']} kg/m²"},
            {"Dimensi Kualitas": "Consistency (Penamaan & Tipe Data)", "Sebelum ETL": "Tipe tanggal teks & flag YA/TIDAK", "Sesudah ETL": "103 variabel master terkunci + 83 fitur turunan bersih"}
        ])
        st.dataframe(q_table, use_container_width=True)

    with c_q2:
        st.subheader("2. Skema Bintang (Star Schema) di SQLite Database")
        st.markdown("""
        Database `warehouse_preeklamsia_toba.db` terdiri atas **1 Tabel Fakta** dan **5 Tabel Dimensi**:
        - **`fact_anc_visit` (Tabel Fakta - 1.000 baris):** Menyimpan metrik longitudinal setiap kunjungan ANC (`anc_visit_id`, tekanan darah, MAP, berat badan, LILA, TFU, keluhan klinis, hasil lab, USG, dan status rujukan).
        - **`dim_patient` (Dimensi Ibu - 250 baris):** Menyimpan identitas anonim `patient_id`, domisili kecamatan, tinggi badan, dan riwayat penyakit kronik.
        - **`dim_pregnancy` (Dimensi Kehamilan - 250 baris):** Menyimpan `pregnancy_id`, usia ibu, IMT pra-hamil, gravida, paritas, jarak kehamilan, dan jumlah janin.
        - **`dim_facility` (Dimensi Fasilitas - 7 baris):** Menyimpan 6 Puskesmas di Kabupaten Toba dan 1 RS Rujukan beserta koordinat spasial.
        - **`dim_time` (Dimensi Waktu - 470 baris):** Menyimpan kalender `visit_date`, tahun, kuartal, dan bulan kunjungan.
        - **`dim_outcome_delivery` (Dimensi Luaran - 250 baris):** Menyimpan diagnosis akhir `final_diagnosis`, keparahan PE, cara persalinan, berat lahir bayi, APGAR, dan status NICU.
        """)

    st.subheader("3. Pratinjau Tabel Data Warehouse Secara Langsung (Live SQL Query)")
    table_choice = st.selectbox(
        "Pilih Tabel pada Data Warehouse:",
        options=["fact_anc_visit", "dim_patient", "dim_pregnancy", "dim_facility", "dim_time", "dim_outcome_delivery"]
    )
    if os.path.exists(DW_DB_PATH):
        conn = sqlite3.connect(DW_DB_PATH)
        df_sql = pd.read_sql_query(f"SELECT * FROM {table_choice} LIMIT 50", conn)
        count_sql = pd.read_sql_query(f"SELECT COUNT(*) AS total_rows FROM {table_choice}", conn).iloc[0, 0]
        conn.close()
        st.caption(f"Menampilkan 50 baris pertama dari total **{count_sql:,} baris** pada tabel `{table_choice}`:")
        st.dataframe(df_sql, use_container_width=True)
