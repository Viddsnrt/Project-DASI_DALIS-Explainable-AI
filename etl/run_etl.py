"""
Pipeline ETL & Data Warehouse (Star Schema)
Proyek: Analisis Faktor Risiko dan Prediksi Risiko Preeklamsia pada Ibu Hamil di Kabupaten Toba
Mata Kuliah: Data Analitik & Data Science - Institut Teknologi Del
"""

import os
import sqlite3
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "dataset_sintetis_preeklamsia.csv")
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_preeklamsia_toba.csv")
DW_DB_PATH = os.path.join(BASE_DIR, "data", "warehouse_preeklamsia_toba.db")
QUALITY_JSON_PATH = os.path.join(BASE_DIR, "data", "data_quality_report.json")

# Koordinat geografis 6 Kecamatan di Kabupaten Toba untuk Peta Choropleth / Geo-Map
DISTRICT_COORDS = {
    "Balige": {"lat": 2.3333, "lon": 99.0667},
    "Laguboti": {"lat": 2.3514, "lon": 99.1258},
    "Silaen": {"lat": 2.4000, "lon": 99.1833},
    "Porsea": {"lat": 2.4436, "lon": 99.1553},
    "Lumban Julu": {"lat": 2.5667, "lon": 99.0500},
    "Ajibata": {"lat": 2.6583, "lon": 98.9361}
}

SENTINEL_NUMERIC_COLS = [
    "interpregnancy_interval_years", "weight_change_from_baseline_kg",
    "systolic_bp_change_mmHg", "diastolic_bp_change_mmHg", "hemoglobin_g_dl",
    "fasting_glucose_mg_dl", "glucose_2h_postprandial_mg_dl", "urine_protein_24h_mg",
    "urine_protein_creatinine_ratio", "platelet_count_per_uL", "serum_creatinine_mg_dl",
    "ast_u_l", "alt_u_l", "estimated_fetal_weight_g"
]

SENTINEL_STR_VALUES = ["TIDAK_BERLAKU", "TIDAK_DIPERIKSA", "NOT_APPLICABLE", "BELUM_DINILAI", "TIDAK_ADA"]

BINARY_FLAG_COLS = [
    "obesity_prepregnancy_flag", "nulliparity_flag", "new_partner_pregnancy_flag",
    "assisted_reproduction_flag", "multiple_pregnancy_flag", "previous_preeclampsia_flag",
    "family_history_preeclampsia_flag", "chronic_hypertension_flag", "preexisting_diabetes_flag",
    "gestational_diabetes_flag", "kidney_disease_flag", "systemic_lupus_erythematosus_flag",
    "antiphospholipid_syndrome_flag", "edema_present", "edema_persistent_flag",
    "headache_present", "visual_disturbance_present", "epigastric_or_ruq_pain_present",
    "dyspnea_present", "proteinuria_examined_flag", "suspected_preeclampsia_flag",
    "eclampsia_flag", "hellp_syndrome_flag", "pulmonary_edema_flag",
    "placental_abruption_flag", "maternal_icu_admission_flag", "preterm_delivery_flag",
    "low_birth_weight_flag", "nicu_admission_flag"
]


def extract_data(csv_path=RAW_CSV_PATH):
    """Tahap 1 (Extract): Membaca dataset mentah 103 variabel."""
    df = pd.read_csv(csv_path)
    return df


def transform_data(df_raw):
    """Tahap 2 (Transform): Audit kualitas data, pembersihan sentinel, dan feature engineering."""
    df = df_raw.copy()

    # Audit sebelum pembersihan
    total_cells = int(df.shape[0] * df.shape[1])
    neg1_count_before = int(sum((df[c] == -1).sum() for c in SENTINEL_NUMERIC_COLS))
    str_sentinel_before = int(sum(df[c].isin(SENTINEL_STR_VALUES).sum() for c in df.select_dtypes(include="object").columns))
    duplicates_before = int(df.duplicated(subset=["anc_visit_id"]).sum())
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

    # 1. Konversi biner _flag (YA -> 1, TIDAK -> 0)
    for col in BINARY_FLAG_COLS:
        if col in df.columns:
            df[col + "_bin"] = (df[col] == "YA").astype(int)

    # 2. Pembersihan nilai sentinel (-1) untuk EDA (_eda menjadi NaN) dan untuk ML (_clean menjadi nilai terimputasi)
    for col in SENTINEL_NUMERIC_COLS:
        df[col + "_is_examined"] = (df[col] != -1).astype(int)
        df[col + "_eda"] = df[col].replace(-1, np.nan)

    # Aturan khusus klinis sesuai Master Variabel:
    # - interpregnancy_interval_years: jika nulliparity_flag == YA (-1), set 0 untuk ML
    df["interpregnancy_interval_clean"] = np.where(df["interpregnancy_interval_years"] == -1, 0.0, df["interpregnancy_interval_years"])
    df["weight_change_clean"] = np.where(df["weight_change_from_baseline_kg"] == -1, 0.0, df["weight_change_from_baseline_kg"])
    df["systolic_bp_change_clean"] = np.where(df["systolic_bp_change_mmHg"] == -1, 0.0, df["systolic_bp_change_mmHg"])
    df["diastolic_bp_change_clean"] = np.where(df["diastolic_bp_change_mmHg"] == -1, 0.0, df["diastolic_bp_change_mmHg"])

    # Imputasi median pada lab rutin Puskesmas bila tidak diperiksa (-1) disertai flag _is_examined
    for lab_col in ["hemoglobin_g_dl", "fasting_glucose_mg_dl", "glucose_2h_postprandial_mg_dl"]:
        med_val = float(df[lab_col + "_eda"].median())
        df[lab_col.replace("_g_dl", "").replace("_mg_dl", "") + "_clean"] = df[lab_col + "_eda"].fillna(med_val)

    # 3. Membedakan 'Tidak Diperiksa' dan 'Hasil Negatif' pada Proteinuria Dipstick (Aturan Dikunci Bab 2 & Bab 7 Master Variabel)
    dipstick_severity_map = {
        "TIDAK_DIPERIKSA": 0,
        "NEGATIF": 0,
        "TRACE": 1,
        "1+": 2,
        "2+": 3,
        "3+": 4
    }
    df["proteinuria_dipstick_score"] = df["proteinuria_dipstick_result"].map(dipstick_severity_map).fillna(0).astype(int)
    df["proteinuria_positive_flag_bin"] = (df["proteinuria_dipstick_score"] >= 2).astype(int)

    # 4. Feature Engineering (Komponen S2 Data Science - Tanpa Data Leakage)
    maternal_risk_bins = [
        "obesity_prepregnancy_flag_bin", "nulliparity_flag_bin", "new_partner_pregnancy_flag_bin",
        "assisted_reproduction_flag_bin", "multiple_pregnancy_flag_bin", "previous_preeclampsia_flag_bin",
        "family_history_preeclampsia_flag_bin", "chronic_hypertension_flag_bin", "preexisting_diabetes_flag_bin",
        "gestational_diabetes_flag_bin", "kidney_disease_flag_bin", "antiphospholipid_syndrome_flag_bin"
    ]
    df["cumulative_risk_score"] = df[maternal_risk_bins].sum(axis=1)

    symptom_bins = [
        "edema_present_bin", "edema_persistent_flag_bin", "headache_present_bin",
        "visual_disturbance_present_bin", "epigastric_or_ruq_pain_present_bin", "dyspnea_present_bin"
    ]
    df["clinical_symptom_score"] = df[symptom_bins].sum(axis=1)

    df["pulse_pressure_mmHg"] = df["systolic_bp_mmHg"] - df["diastolic_bp_mmHg"]
    df["high_risk_age_flag_bin"] = ((df["maternal_age_years"] < 20) | (df["maternal_age_years"] >= 35)).astype(int)
    df["hypertension_visit_flag_bin"] = ((df["systolic_bp_mmHg"] >= 140) | (df["diastolic_bp_mmHg"] >= 90)).astype(int)
    df["map_elevated_flag_bin"] = (df["mean_arterial_pressure_mmHg"] >= 90.0).astype(int)
    df["weight_gain_per_week"] = (df["weight_change_clean"] / np.maximum(df["gestational_age_weeks"], 1)).round(3)

    # Fitur Longitudinal (Riwayat kumulatif per pregnancy_id hingga kunjungan saat ini - bebas leakage masa depan)
    df = df.sort_values(["pregnancy_id", "visit_number"]).reset_index(drop=True)
    df["cum_max_systolic_bp"] = df.groupby("pregnancy_id")["systolic_bp_mmHg"].cummax()
    df["cum_max_diastolic_bp"] = df.groupby("pregnancy_id")["diastolic_bp_mmHg"].cummax()
    df["cum_max_map"] = df.groupby("pregnancy_id")["mean_arterial_pressure_mmHg"].cummax()

    # 5. Geografi & Dimensi Waktu
    df["district_lat"] = df["facility_district"].map(lambda d: DISTRICT_COORDS.get(d, {}).get("lat", 2.35))
    df["district_lon"] = df["facility_district"].map(lambda d: DISTRICT_COORDS.get(d, {}).get("lon", 99.12))

    visit_dt = pd.to_datetime(df["visit_date"])
    df["visit_year"] = visit_dt.dt.year
    df["visit_month"] = visit_dt.dt.month
    df["visit_year_month"] = visit_dt.dt.strftime("%Y-%m")
    df["visit_quarter"] = "Q" + visit_dt.dt.quarter.astype(str)

    # Target biner
    df["target_preeclampsia"] = (df["final_diagnosis"] == "PREECLAMPSIA").astype(int)

    # Simpan laporan kualitas data
    quality_report = {
        "rows_total": int(df.shape[0]),
        "unique_patients": int(df["patient_id"].nunique()),
        "unique_pregnancies": int(df["pregnancy_id"].nunique()),
        "raw_columns_count": int(df_raw.shape[1]),
        "enriched_columns_count": int(df.shape[1]),
        "total_cells_raw": total_cells,
        "sentinel_numeric_neg1_before": neg1_count_before,
        "sentinel_string_before": str_sentinel_before,
        "duplicates_before": duplicates_before,
        "constant_columns_detected": constant_cols,
        "bmi_max_formula_deviation": float((df["prepregnancy_bmi"] - (df["prepregnancy_weight_kg"] / ((df["height_cm"] / 100) ** 2)).round(1)).abs().max()),
        "map_max_formula_deviation": float((df["mean_arterial_pressure_mmHg"] - ((df["systolic_bp_mmHg"] + 2 * df["diastolic_bp_mmHg"]) / 3).round(1)).abs().max())
    }
    with open(QUALITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    df.to_csv(CLEAN_CSV_PATH, index=False)
    return df, quality_report


def load_star_schema(df, db_path=DW_DB_PATH):
    """Tahap 3 (Load): Membangun Data Warehouse Skema Bintang (1 Tabel Fakta + 5 Tabel Dimensi) di SQLite."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # 1. dim_facility (Dimensi Fasilitas Kesehatan & Kecamatan)
    dim_facility = df[[
        "facility_id", "facility_name", "facility_type", "facility_district", "district_lat", "district_lon"
    ]].drop_duplicates(subset=["facility_id"]).reset_index(drop=True)
    dim_facility.to_sql("dim_facility", conn, index=False, if_exists="replace")

    # 2. dim_patient (Dimensi Ibu & Faktor Risiko Komorbiditas Level Ibu)
    dim_patient = df[[
        "patient_id", "date_of_birth", "height_cm", "maternal_domicile_district",
        "family_history_preeclampsia_flag", "chronic_hypertension_flag",
        "preexisting_diabetes_flag", "kidney_disease_flag",
        "systemic_lupus_erythematosus_flag", "antiphospholipid_syndrome_flag"
    ]].drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    dim_patient.to_sql("dim_patient", conn, index=False, if_exists="replace")

    # 3. dim_pregnancy (Dimensi Episode Kehamilan & Riwayat Obstetri)
    dim_pregnancy = df[[
        "pregnancy_id", "patient_id", "maternal_age_years", "prepregnancy_weight_kg",
        "prepregnancy_bmi", "obesity_prepregnancy_flag", "lmp_date", "estimated_due_date",
        "gravida", "parity", "abortion_count", "nulliparity_flag", "previous_delivery_date",
        "interpregnancy_interval_years", "new_partner_pregnancy_flag",
        "assisted_reproduction_flag", "fetus_count", "multiple_pregnancy_flag",
        "previous_preeclampsia_flag", "gestational_diabetes_flag", "cumulative_risk_score"
    ]].drop_duplicates(subset=["pregnancy_id"]).reset_index(drop=True)
    dim_pregnancy.to_sql("dim_pregnancy", conn, index=False, if_exists="replace")

    # 4. dim_time (Dimensi Waktu Kunjungan ANC)
    dim_time = df[[
        "visit_date", "visit_year", "visit_quarter", "visit_month", "visit_year_month"
    ]].drop_duplicates(subset=["visit_date"]).sort_values("visit_date").reset_index(drop=True)
    dim_time.to_sql("dim_time", conn, index=False, if_exists="replace")

    # 5. dim_outcome_delivery (Dimensi Diagnosis Akhir, Persalinan, dan Bayi)
    dim_outcome_delivery = df[[
        "pregnancy_id", "final_diagnosis", "target_preeclampsia", "preeclampsia_diagnosis_date",
        "preeclampsia_severity", "eclampsia_flag", "hellp_syndrome_flag", "pulmonary_edema_flag",
        "placental_abruption_flag", "maternal_icu_admission_flag", "delivery_date",
        "gestational_age_at_delivery_weeks", "delivery_mode", "preterm_delivery_flag",
        "maternal_discharge_status", "baby_id", "birth_order", "baby_sex",
        "birth_weight_g", "low_birth_weight_flag", "birth_length_cm",
        "apgar_1min", "apgar_5min", "birth_status", "nicu_admission_flag", "neonatal_outcome"
    ]].drop_duplicates(subset=["pregnancy_id"]).reset_index(drop=True)
    dim_outcome_delivery.to_sql("dim_outcome_delivery", conn, index=False, if_exists="replace")

    # 6. fact_anc_visit (Tabel Fakta Kunjungan ANC, Klinis, Lab, USG, dan Rujukan)
    fact_anc_visit = df[[
        "anc_visit_id", "pregnancy_id", "patient_id", "facility_id", "visit_date",
        "visit_number", "gestational_age_weeks", "trimester", "prior_visit_count",
        "days_since_previous_visit", "weight_kg", "weight_change_from_baseline_kg",
        "weight_gain_per_week", "lila_cm", "systolic_bp_mmHg", "diastolic_bp_mmHg",
        "mean_arterial_pressure_mmHg", "pulse_pressure_mmHg", "systolic_bp_change_mmHg",
        "diastolic_bp_change_mmHg", "edema_present", "edema_location",
        "edema_persistent_flag", "headache_present", "visual_disturbance_present",
        "epigastric_or_ruq_pain_present", "dyspnea_present", "clinical_symptom_score",
        "tfu_cm", "fetal_presentation", "fetal_heart_rate_bpm", "iron_tablets_given_count",
        "proteinuria_examined_flag", "proteinuria_dipstick_result", "proteinuria_dipstick_score",
        "hemoglobin_g_dl", "fasting_glucose_mg_dl", "glucose_2h_postprandial_mg_dl",
        "urine_protein_24h_mg", "urine_protein_creatinine_ratio", "platelet_count_per_uL",
        "serum_creatinine_mg_dl", "ast_u_l", "alt_u_l", "estimated_fetal_weight_g",
        "fetal_growth_restriction_flag", "umbilical_artery_doppler_abnormal_flag",
        "pe_screening_risk_category", "suspected_preeclampsia_flag", "referral_status",
        "referral_date", "referral_destination", "referral_reason_code", "target_preeclampsia"
    ]]
    fact_anc_visit.to_sql("fact_anc_visit", conn, index=False, if_exists="replace")

    # Buat Analytic View terintegrasi untuk memudahkan query Dashboard
    df.to_sql("vw_analytic_master", conn, index=False, if_exists="replace")

    conn.commit()
    conn.close()
    print(f"Data Warehouse Star Schema berhasil dibuat di: {db_path}")


if __name__ == "__main__":
    raw_df = extract_data()
    clean_df, report = transform_data(raw_df)
    load_star_schema(clean_df)
    print("Ringkasan Kualitas Data:", json.dumps(report, indent=2))
