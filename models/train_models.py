"""
Pelatihan, Tuning, Evaluasi, dan Penyimpanan Model Machine Learning
Proyek: Analisis Faktor Risiko dan Prediksi Risiko Preeklamsia pada Ibu Hamil di Kabupaten Toba
Sesuai Master Variabel Bab 3 & Bab 7 (Bebas Data Leakage, GroupSplit berbasis patient_id)
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_preeklamsia_toba.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Daftar Fitur Input ML yang Sah (Sesuai Master Variabel Bab 3 Bagian B, C, D, E)
# Tidak menyertakan variabel Tahap 6 (pe_screening_risk_category, suspected_preeclampsia_flag, referral_*),
# variabel RS lanjutan yang rawan leakage (urine_protein_24h_mg, dll.), maupun variabel Outcome Persalinan/Bayi.
FEATURE_COLUMNS = [
    # Karakteristik Ibu & Kehamilan (Bagian B)
    "maternal_age_years",
    "prepregnancy_bmi",
    "obesity_prepregnancy_flag_bin",
    "gestational_age_weeks",
    "gravida",
    "parity",
    "abortion_count",
    "nulliparity_flag_bin",
    "interpregnancy_interval_clean",
    "new_partner_pregnancy_flag_bin",
    "assisted_reproduction_flag_bin",
    "fetus_count",
    "multiple_pregnancy_flag_bin",
    # Riwayat Penyakit Maternal (Bagian C)
    "previous_preeclampsia_flag_bin",
    "family_history_preeclampsia_flag_bin",
    "chronic_hypertension_flag_bin",
    "preexisting_diabetes_flag_bin",
    "gestational_diabetes_flag_bin",
    "kidney_disease_flag_bin",
    "antiphospholipid_syndrome_flag_bin",
    # Pemeriksaan Fisik & Gejala Kunjungan ANC Puskesmas (Bagian D)
    "weight_change_clean",
    "weight_gain_per_week",
    "lila_cm",
    "systolic_bp_mmHg",
    "diastolic_bp_mmHg",
    "mean_arterial_pressure_mmHg",
    "pulse_pressure_mmHg",
    "systolic_bp_change_clean",
    "diastolic_bp_change_clean",
    "prior_visit_count",
    "edema_present_bin",
    "edema_persistent_flag_bin",
    "headache_present_bin",
    "visual_disturbance_present_bin",
    "epigastric_or_ruq_pain_present_bin",
    "dyspnea_present_bin",
    # Lab Dasar Puskesmas (Bagian E - memisahkan flag diperiksa dan hasil dipstick)
    "proteinuria_examined_flag_bin",
    "proteinuria_dipstick_score",
    "hemoglobin_clean",
    # Fitur Rekayasa (Feature Engineering S2)
    "cumulative_risk_score",
    "clinical_symptom_score",
    "high_risk_age_flag_bin",
    "hypertension_visit_flag_bin"
]

FEATURE_LABELS_ID = {
    "maternal_age_years": "Usia Ibu (Tahun)",
    "prepregnancy_bmi": "IMT Pra-Kehamilan (kg/m²)",
    "obesity_prepregnancy_flag_bin": "Obesitas Pra-Hamil",
    "gestational_age_weeks": "Usia Kehamilan (Minggu)",
    "gravida": "Gravida (Jumlah Kehamilan)",
    "parity": "Paritas",
    "abortion_count": "Riwayat Abortus",
    "nulliparity_flag_bin": "Nulipara (Belum Pernah Melahirkan)",
    "interpregnancy_interval_clean": "Jarak Kehamilan (Tahun)",
    "new_partner_pregnancy_flag_bin": "Kehamilan Pasangan Baru",
    "assisted_reproduction_flag_bin": "Teknologi Reproduksi Berbantu",
    "fetus_count": "Jumlah Janin",
    "multiple_pregnancy_flag_bin": "Kehamilan Ganda/Kembar",
    "previous_preeclampsia_flag_bin": "Riwayat Preeklamsia Sebelumnya",
    "family_history_preeclampsia_flag_bin": "Riwayat Preeklamsia Keluarga",
    "chronic_hypertension_flag_bin": "Hipertensi Kronik",
    "preexisting_diabetes_flag_bin": "Diabetes Pra-Kehamilan",
    "gestational_diabetes_flag_bin": "Diabetes Gestasional",
    "kidney_disease_flag_bin": "Penyakit Ginjal",
    "antiphospholipid_syndrome_flag_bin": "Sindrom Antifosfolipid (APS)",
    "weight_change_clean": "Kenaikan Berat Badan dari Baseline (kg)",
    "weight_gain_per_week": "Laju Kenaikan BB per Minggu (kg/mgg)",
    "lila_cm": "Lingkar Lengan Atas / LILA (cm)",
    "systolic_bp_mmHg": "Tekanan Darah Sistolik (mmHg)",
    "diastolic_bp_mmHg": "Tekanan Darah Diastolik (mmHg)",
    "mean_arterial_pressure_mmHg": "Mean Arterial Pressure / MAP (mmHg)",
    "pulse_pressure_mmHg": "Tekanan Nadi / Pulse Pressure (mmHg)",
    "systolic_bp_change_clean": "Perubahan Sistolik dari ANC Sebelumnya (mmHg)",
    "diastolic_bp_change_clean": "Perubahan Diastolik dari ANC Sebelumnya (mmHg)",
    "prior_visit_count": "Jumlah Kunjungan ANC Sebelumnya",
    "edema_present_bin": "Edema / Bengkak Saat Kunjungan",
    "edema_persistent_flag_bin": "Edema Menetap (Persisten)",
    "headache_present_bin": "Keluhan Sakit Kepala",
    "visual_disturbance_present_bin": "Gangguan Penglihatan / Pandangan Kabur",
    "epigastric_or_ruq_pain_present_bin": "Nyeri Ulu Hati / Epigastrium",
    "dyspnea_present_bin": "Keluhan Sesak Napas",
    "proteinuria_examined_flag_bin": "Status Pemeriksaan Protein Urine Dilakukan",
    "proteinuria_dipstick_score": "Skor Hasil Dipstick Protein Urine (0-4)",
    "hemoglobin_clean": "Kadar Hemoglobin (g/dL)",
    "cumulative_risk_score": "Skor Akumulasi Risiko Maternal",
    "clinical_symptom_score": "Skor Akumulasi Gejala Bahaya ANC",
    "high_risk_age_flag_bin": "Usia Ibu Risiko Tinggi (<20 atau >=35 Thn)",
    "hypertension_visit_flag_bin": "Hipertensi Saat Kunjungan (>=140/90 mmHg)"
}


def train_and_evaluate():
    df = pd.read_csv(CLEAN_CSV_PATH)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target_preeclampsia"].copy()
    groups = df["patient_id"].copy()

    # Pembagian Train/Test berbasis GroupShuffleSplit (patient_id) agar tidak ada pasien yang bocor antar-set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    df_test = df.iloc[test_idx].copy()

    gkf = GroupKFold(n_splits=5)

    # 1. Baseline: Logistic Regression
    t0 = time.time()
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
    ])
    lr_grid = {"clf__C": [0.1, 0.5, 1.0, 2.0]}
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=gkf, scoring="roc_auc", n_jobs=-1)
    lr_search.fit(X_train, y_train, groups=groups_train)
    lr_time = round(time.time() - t0, 3)

    # 2. Random Forest (Model Lanjutan 1)
    t0 = time.time()
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10],
        "min_samples_leaf": [2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    }
    rf_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=gkf, scoring="roc_auc", n_jobs=-1)
    rf_search.fit(X_train, y_train, groups=groups_train)
    rf_time = round(time.time() - t0, 3)

    # 3. Gradient Boosting (Model Lanjutan 2)
    t0 = time.time()
    gb_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0]
    }
    gb_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_grid, cv=gkf, scoring="roc_auc", n_jobs=-1)
    gb_search.fit(X_train, y_train, groups=groups_train)
    gb_time = round(time.time() - t0, 3)

    # 4. Extra Trees Ensemble (Model Lanjutan 3)
    t0 = time.time()
    et_grid = {
        "n_estimators": [100, 200],
        "max_depth": [6, 8, 10],
        "min_samples_leaf": [2, 4],
        "class_weight": ["balanced"]
    }
    et_search = GridSearchCV(ExtraTreesClassifier(random_state=42), et_grid, cv=gkf, scoring="roc_auc", n_jobs=-1)
    et_search.fit(X_train, y_train, groups=groups_train)
    et_time = round(time.time() - t0, 3)

    models_dict = {
        "Logistic Regression (Baseline)": (lr_search.best_estimator_, lr_search.best_params_, lr_time, 0.50),
        "Random Forest (Tuned)": (rf_search.best_estimator_, rf_search.best_params_, rf_time, 0.40),
        "Gradient Boosting (Tuned)": (gb_search.best_estimator_, gb_search.best_params_, gb_time, 0.25),
        "Extra Trees (Tuned)": (et_search.best_estimator_, et_search.best_params_, et_time, 0.45)
    }

    evaluation_summary = {}
    saved_estimators = {}

    for name, (estimator, best_params, train_time, opt_thresh) in models_dict.items():
        y_prob = estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= opt_thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        evaluation_summary[name] = {
            "best_params": best_params,
            "optimal_threshold": opt_thresh,
            "train_time_sec": train_time,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr]
            }
        }
        saved_estimators[name] = estimator

    # Feature Importance dari Random Forest Terbaik
    best_rf = rf_search.best_estimator_
    importances = best_rf.feature_importances_
    feat_imp_list = sorted(
        [
            {
                "feature": col,
                "label": FEATURE_LABELS_ID.get(col, col),
                "importance": round(float(imp), 4)
            }
            for col, imp in zip(FEATURE_COLUMNS, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Error Analysis (Contoh kasus False Negative & False Positive pada Data Uji)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.40).astype(int)
    df_test["pred_prob"] = rf_probs.round(3)
    df_test["pred_label"] = rf_preds

    fn_cases = df_test[(df_test["target_preeclampsia"] == 1) & (df_test["pred_label"] == 0)][[
        "anc_visit_id", "patient_id", "trimester", "gestational_age_weeks",
        "systolic_bp_mmHg", "diastolic_bp_mmHg", "mean_arterial_pressure_mmHg",
        "proteinuria_dipstick_result", "cumulative_risk_score", "pred_prob", "final_diagnosis"
    ]].head(5).to_dict(orient="records")

    fp_cases = df_test[(df_test["target_preeclampsia"] == 0) & (df_test["pred_label"] == 1)][[
        "anc_visit_id", "patient_id", "trimester", "gestational_age_weeks",
        "systolic_bp_mmHg", "diastolic_bp_mmHg", "mean_arterial_pressure_mmHg",
        "proteinuria_dipstick_result", "cumulative_risk_score", "pred_prob", "final_diagnosis"
    ]].head(5).to_dict(orient="records")

    # Simpan artefak model dan metrik
    bundle = {
        "models": saved_estimators,
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels": FEATURE_LABELS_ID,
        "default_model": "Random Forest (Tuned)"
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, "preeclampsia_models_bundle.joblib"))

    metrics_payload = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "feature_count": len(FEATURE_COLUMNS),
        "models": evaluation_summary,
        "feature_importance": feat_imp_list,
        "error_analysis": {
            "false_negatives": fn_cases,
            "false_positives": fp_cases
        }
    }
    with open(os.path.join(MODELS_DIR, "model_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Model berhasil dilatih dan disimpan!")
    for m_name, m_stat in evaluation_summary.items():
        print(f"  {m_name:32s} -> AUC: {m_stat['roc_auc']:.4f} | F1: {m_stat['f1_score']:.4f} | Acc: {m_stat['accuracy']:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
