"""
Evaluasi khusus untuk model:
1. Linear Regression
2. Polynomial Regression (Degree 2 & 3)
3. Gaussian Process Regression (GPR WHO Prior)

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.who_service import load_who_lms
from services.preprocessing_service import build_feature
from services.model_service import (
    train_linear,
    train_polynomial,
    train_gpr_who,
)
from services.prediction_service import recursive_predict


def run_target_models(history, sex, who_lms_df):
    """Train ONLY Linear, Polynomial (Deg 2 & 3), and GPR WHO Prior models."""
    X, y = build_feature(history)

    candidates = [
        lambda: train_linear(X, y),
        lambda: train_polynomial(X, y, degree=2),
        lambda: train_polynomial(X, y, degree=3),
        lambda: train_gpr_who(X, y, sex, who_lms_df),
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
        preds = recursive_predict(model_dict, last_age, horizon)
        y_pred = [p["height"] for p in preds]
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

    for case in cases:
        history = case["history"]
        ground_truth = case["ground_truth"]
        sex = case["sex"]
        last_age = history[-1]["age"]
        horizon = len(ground_truth)

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

    out_json = os.path.join(base_dir, "selected_models_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    df_details = pd.DataFrame(test_data_details)
    out_csv = os.path.join(base_dir, "selected_models_predictions.csv")
    df_details.to_csv(out_csv, index=False)

    generate_report(results, base_dir)
    print(f"Done. Evaluated {len(results)} model-case pairs across {len(cases)} cases.")
    print(f"Saved results to selected_models_results.json, selected_models_predictions.csv, and selected_models_report.txt.")


def generate_report(results, base_dir):
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

    # --- Tabel 5: Contoh Data Input & Target Prediksi ---
    lines.append("-" * 75)
    lines.append("TABEL 5: CONTOH DATA INPUT, TARGET (GROUND TRUTH), DAN HASIL PREDIKSI")
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
