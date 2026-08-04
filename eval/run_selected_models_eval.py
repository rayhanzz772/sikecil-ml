"""
Evaluasi khusus untuk model:
1. Linear Regression
2. Exponential Regression
3. Gaussian Process Regression + WHO Prior

Metodologi Evaluasi (dua metode, keduanya ditampilkan di report):
  A. Out-of-Sample (Ground Truth)
     - Model dilatih pada data history awal
     - Diuji pada ground_truth bulan-bulan berikutnya yang belum pernah dilihat
     - Paling representatif untuk kondisi real-world (prediksi masa depan)

  B. LOOCV (Leave-One-Out Cross-Validation)
     - Setiap titik history digilir sebagai test set
     - Sama dengan metode yang digunakan API /api/predict/v3 di runtime
     - Memungkinkan perbandingan langsung antara report dan output API

Output:
  - eval/selected_models_results.json
  - eval/selected_models_predictions.csv
  - eval/selected_models_report.txt
"""

import sys
import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.who_service import load_who_lms
from services.preprocessing_service import build_feature
from services.model_service import (
    train_gpr_who,
    train_linear,
    train_exponential,
    gpr_predict_with_who
)


def run_target_models(history, sex, who_lms_df):
    """Train GPR WHO Prior, Linear Regression, dan Exponential Regression."""
    X, y = build_feature(history)

    candidates = [
        lambda: train_gpr_who(X, y, sex, who_lms_df),
        lambda: train_linear(X, y),
        lambda: train_exponential(X, y),
    ]

    models = []
    for fn in candidates:
        try:
            m = fn()
            if m is not None:
                models.append(m)
        except Exception:
            continue
    return models


def evaluate_model(model_dict, last_age, horizon, ground_truth):
    """Run one model, return metrics dict and raw predictions."""
    try:
        if model_dict.get("type") == "gpr_who":
            preds = gpr_predict_with_who(model_dict, last_age, horizon)
            y_pred = [p["height"] for p in preds]
        else:
            predictor = model_dict["model"]
            future_ages = np.array([[last_age + i] for i in range(1, horizon + 1)], dtype=float)
            y_pred = list(predictor.predict(future_ages))
    except Exception:
        return None, []

    y_true = [p["height"] for p in ground_truth]

    if len(y_pred) != len(y_true):
        return None, []

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = None

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
    }

    formatted_preds = [
        {"age": gt["age"], "height": round(p, 2)}
        for gt, p in zip(ground_truth, y_pred)
    ]

    return metrics, formatted_preds


def evaluate_model_loocv(history, sex, who_lms_df):
    """
    Evaluasi LOOCV (Leave-One-Out Cross-Validation) pada data history.
    Metode ini identik dengan yang digunakan API /api/predict/v3 di runtime.

    Returns
    -------
    dict: { model_name: {"mae": float, "rmse": float, "r2": float} }
    """
    X, y = build_feature(history)
    n = len(X)

    if n < 3:
        return {}

    loo = LeaveOneOut()
    loocv_preds = {
        "GPR WHO Prior": [],
        "Linear Regression": [],
        "Exponential Regression": [],
    }
    y_true_list = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(X_tr) < 2:
            continue

        y_true_list.append(float(y_te[0]))

        # GPR WHO Prior
        try:
            g = train_gpr_who(X_tr, y_tr, sex, who_lms_df)
            loocv_preds["GPR WHO Prior"].append(
                float(g["model"].predict(X_te)[0]) if g else np.nan
            )
        except Exception:
            loocv_preds["GPR WHO Prior"].append(np.nan)

        # Linear Regression
        try:
            lm = train_linear(X_tr, y_tr)
            loocv_preds["Linear Regression"].append(
                float(lm["model"].predict(X_te)[0])
            )
        except Exception:
            loocv_preds["Linear Regression"].append(np.nan)

        # Exponential Regression
        try:
            em = train_exponential(X_tr, y_tr)
            loocv_preds["Exponential Regression"].append(
                float(em["model"].predict(X_te)[0]) if em else np.nan
            )
        except Exception:
            loocv_preds["Exponential Regression"].append(np.nan)

    results = {}
    for model_name, preds in loocv_preds.items():
        valid_idx = [i for i, p in enumerate(preds) if not np.isnan(p)]
        if len(valid_idx) < 2:
            continue
        yt = [y_true_list[i] for i in valid_idx]
        yp = [preds[i] for i in valid_idx]
        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = None
        results[model_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4) if r2 is not None else None,
        }

    return results


def run_selected_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, "simulation_cases.json")

    if not os.path.exists(cases_path):
        from generate_simulation import generate_cases
        generate_cases()

    with open(cases_path, "r") as f:
        cases = json.load(f)

    who_lms_df = load_who_lms(
        os.path.join(base_dir, "..", "data", "who_lms.csv")
    )

    results = []
    test_data_details = []
    loocv_all = []  # kumpulkan hasil LOOCV semua kasus

    for case in cases:
        history = case["history"]
        ground_truth = case["ground_truth"]
        sex = case["sex"]
        last_age = history[-1]["age"]
        horizon = len(ground_truth)

        # --- Out-of-Sample Evaluation (Ground Truth) ---
        models = run_target_models(history, sex, who_lms_df)

        for m in models:
            metrics, preds = evaluate_model(m, last_age, horizon, ground_truth)
            if metrics is None:
                continue

            results.append({
                "case_id": case["case_id"],
                "mode": case["mode"],
                "label": case["label"],
                "sex": sex,
                "model": m["name"],
                "history": history,
                "ground_truth": ground_truth,
                "predictions": preds,
                **metrics,
            })

            for gt, p_entry in zip(ground_truth, preds):
                test_data_details.append({
                    "case_id": case["case_id"],
                    "mode": case["mode"],
                    "label": case["label"],
                    "sex": sex,
                    "model": m["name"],
                    "target_age": gt["age"],
                    "ground_truth_height": round(gt["height"], 2),
                    "predicted_height": p_entry["height"],
                    "abs_error": round(abs(gt["height"] - p_entry["height"]), 4)
                })

        # --- LOOCV Evaluation (identik dengan metode API runtime) ---
        # Gabungkan history + ground_truth sebagai dataset penuh untuk LOOCV
        full_data = history + ground_truth
        loocv_metrics = evaluate_model_loocv(full_data, sex, who_lms_df)
        for model_name, m_metrics in loocv_metrics.items():
            loocv_all.append({
                "case_id": case["case_id"],
                "mode": case["mode"],
                "label": case["label"],
                "sex": sex,
                "model": model_name,
                **m_metrics,
            })

    out_json = os.path.join(base_dir, "selected_models_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df_details = pd.DataFrame(test_data_details)
    out_csv = os.path.join(base_dir, "selected_models_predictions.csv")
    df_details.to_csv(out_csv, index=False)

    generate_report(results, loocv_all, base_dir)
    print(f"Done. Evaluated {len(results)} model-case pairs across {len(cases)} cases.")
    print(f"Saved results to selected_models_results.json, selected_models_predictions.csv, and selected_models_report.txt.")


def generate_report(results, loocv_all, base_dir):
    df = pd.DataFrame(results)

    lines = []
    lines.append("=" * 75)
    lines.append("LAPORAN EVALUASI MODEL: LINEAR, POLYNOMIAL, & GAUSSIAN PROCESS REGRESSION")
    lines.append("=" * 75)
    lines.append("")

    # --- Tabel 1: Rata-rata metrik per model ---
    lines.append("-" * 75)
    lines.append("TABEL 1: RATA-RATA METRIK PER MODEL (SEMUA KASUS)")
    lines.append("-" * 75)
    summary = df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
    lines.append(f"{'Model':<28} {'MAE (cm)':>12} {'RMSE (cm)':>12} {'R2':>12}")
    lines.append("-" * 68)
    for model, row in summary.iterrows():
        r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
        lines.append(f"{model:<28} {row['mae']:>12.4f} {row['rmse']:>12.4f} {r2_str:>12}")
    lines.append("")

    # --- Tabel 2: Break down per mode ---
    for mode in ["early_months", "normal"]:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            continue
        mode_label = "EARLY MONTHS (riwayat 4 bulan)" if mode == "early_months" else "NORMAL MODE (riwayat 6 bulan)"
        lines.append("-" * 75)
        lines.append(f"TABEL 2: {mode_label}")
        lines.append("-" * 75)
        summary_mode = mode_df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
        lines.append(f"{'Model':<28} {'MAE (cm)':>12} {'RMSE (cm)':>12} {'R2':>12}")
        lines.append("-" * 68)
        for model, row in summary_mode.iterrows():
            r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
            lines.append(f"{model:<28} {row['mae']:>12.4f} {row['rmse']:>12.4f} {r2_str:>12}")
        lines.append("")

    # --- Tabel 3: Model Terbaik per Kasus ---
    lines.append("-" * 75)
    lines.append("TABEL 3: MODEL TERBAIK PER KASUS (RMSE TERENDAH)")
    lines.append("-" * 75)
    best = df.loc[df.groupby("case_id")["rmse"].idxmin()]
    lines.append(f"{'Case ID':<28} {'Best Model':<28} {'RMSE (cm)':>12}")
    lines.append("-" * 70)
    for _, row in best.sort_values("case_id").iterrows():
        lines.append(f"{row['case_id']:<28} {row['model']:<28} {row['rmse']:>12.4f}")
    lines.append("")

    # --- Tabel 4: Detail per Kasus ---
    lines.append("-" * 75)
    lines.append("TABEL 4: DETAIL HASIL PER KASUS & MODEL")
    lines.append("-" * 75)
    lines.append(f"{'Case ID':<26} {'Model':<24} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    lines.append("-" * 82)
    for _, row in df.sort_values(["case_id", "rmse"]).iterrows():
        r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
        lines.append(
            f"{row['case_id']:<26} {row['model']:<24} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}"
        )
    lines.append("")

    # --- Tabel 5: LOOCV (sama dengan metode yang digunakan API) ---
    lines.append("-" * 75)
    lines.append("TABEL 5: EVALUASI LOOCV — IDENTIK DENGAN METODE API /api/predict/v3")
    lines.append("-" * 75)
    lines.append("Catatan: LOOCV mengevaluasi interpolasi pada data yang tersedia (bukan")
    lines.append("         ekstrapolasi ke masa depan). Nilainya bisa berbeda dari Tabel 1")
    lines.append("         (out-of-sample) karena kondisi pengujian berbeda.")
    lines.append("")
    if loocv_all:
        df_loocv = pd.DataFrame(loocv_all)
        loocv_summary = df_loocv.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
        lines.append(f"{'Model':<28} {'MAE (cm)':>12} {'RMSE (cm)':>12} {'R2':>12}")
        lines.append("-" * 68)
        for model, row in loocv_summary.iterrows():
            r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
            lines.append(f"{model:<28} {row['mae']:>12.4f} {row['rmse']:>12.4f} {r2_str:>12}")
        lines.append("")

        # Break down per mode
        for mode in ["early_months", "normal"]:
            mode_df = df_loocv[df_loocv["mode"] == mode]
            if mode_df.empty:
                continue
            mode_label = "Early Months (riwayat 4 bulan)" if mode == "early_months" else "Normal Mode (riwayat 6 bulan)"
            lines.append(f"  [{mode_label}]")
            mode_summary = mode_df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
            lines.append(f"  {'Model':<26} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
            lines.append("  " + "-" * 60)
            for model, row in mode_summary.iterrows():
                r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
                lines.append(f"  {model:<26} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")
            lines.append("")

        # Break down per label (status stunting)
        lines.append("  [Per Label / Status Pertumbuhan]")
        label_order = ["Severely Stunted", "Stunted", "At Risk", "Normal", "Normal Atas"]
        for lbl in label_order:
            lbl_df = df_loocv[df_loocv["label"] == lbl]
            if lbl_df.empty:
                continue
            lines.append(f"  [{lbl}]")
            lbl_summary = lbl_df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
            lines.append(f"  {'Model':<26} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
            lines.append("  " + "-" * 60)
            for model, row in lbl_summary.iterrows():
                r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
                lines.append(f"  {model:<26} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")
            lines.append("")
    else:
        lines.append("  (Tidak ada data LOOCV yang berhasil dihitung.)")
        lines.append("")

    # --- Tabel 6: Contoh Data Input & Target Prediksi ---
    lines.append("-" * 75)
    lines.append("TABEL 6: CONTOH DATA INPUT, TARGET (GROUND TRUTH), DAN HASIL PREDIKSI")
    lines.append("-" * 75)

    sample_case_ids = ["EARLY-L-Normal", "NORM-L-Normal", "NORM-L-Stunted"]
    for case_id in sample_case_ids:
        case_rows = [r for r in results if r["case_id"] == case_id]
        if not case_rows:
            continue
        first_row = case_rows[0]
        lines.append(f"Kasus ID  : {first_row['case_id']} (Mode: {first_row['mode']}, Label: {first_row['label']}, Sex: {first_row['sex']})")

        hist_str = ", ".join([f"Bln {h['age']}: {h['height']}cm" for h in first_row['history']])
        lines.append(f"  Data Input (History)   : [{hist_str}]")

        gt_str = ", ".join([f"Bln {g['age']}: {g['height']}cm" for g in first_row['ground_truth']])
        lines.append(f"  Target Actual (GT)     : [{gt_str}]")

        lines.append("  Hasil Prediksi Model   :")
        for r in case_rows:
            pred_str = ", ".join([f"Bln {p['age']}: {p['height']}cm" for p in r['predictions']])
            lines.append(f"    - {r['model']:<24}: [{pred_str}] (MAE: {r['mae']} cm, RMSE: {r['rmse']} cm)")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(base_dir, "selected_models_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    run_selected_evaluation()