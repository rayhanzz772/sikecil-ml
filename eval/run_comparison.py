"""
Evaluasi head-to-head semua model untuk laporan.
Setiap model dijalankan di setiap kasus (tidak ada auto-select),
menghasilkan tabel perbandingan MAE, RMSE, R2.

Output:
  - eval/comparison_results.json  (data mentah)
  - eval/comparison_report.txt    (tabel siap laporan)
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
    train_bayesian_ridge,
    train_linear,
    train_polynomial,
    train_von_bertalanffy,
    train_gompertz,
    train_gpr_who,
)
from services.haz_predictor import train_haz_predictor
from services.prediction_service import recursive_predict


def run_all_models(history, sex, who_lms_df):
    """Train all models on history, return list of model dicts (skip failures)."""
    X, y = build_feature(history)
    last_age = history[-1]["age"]

    candidates = [
        lambda: train_linear(X, y),
        lambda: train_polynomial(X, y, degree=2),
        lambda: train_polynomial(X, y, degree=3),
        lambda: train_bayesian_ridge(X, y),
        lambda: train_von_bertalanffy(X, y),
        lambda: train_gompertz(X, y),
        lambda: train_gpr_who(X, y, sex, who_lms_df),
        lambda: train_haz_predictor(history, sex, who_lms_df),
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
    """Run one model, return metrics dict or None on failure."""
    try:
        preds = recursive_predict(model_dict, last_age, horizon)
        y_pred = [p["height"] for p in preds]
    except Exception:
        return None

    y_true = [p["height"] for p in ground_truth]

    if len(y_pred) != len(y_true):
        return None

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
    }


def run_comparison():
    cases_path = os.path.join(os.path.dirname(__file__), "simulation_cases.json")
    with open(cases_path, "r") as f:
        cases = json.load(f)

    who_lms_df = load_who_lms(
        os.path.join(os.path.dirname(__file__), "..", "data", "who_lms.csv")
    )

    results = []

    for case in cases:
        history = case["history"]
        ground_truth = case["ground_truth"]
        sex = case["sex"]
        last_age = history[-1]["age"]
        horizon = len(ground_truth)

        models = run_all_models(history, sex, who_lms_df)

        for m in models:
            metrics = evaluate_model(m, last_age, horizon, ground_truth)
            if metrics is None:
                continue

            try:
                preds = recursive_predict(m, last_age, horizon)
                formatted_preds = [
                    {"age": p["age"], "height": round(p["height"], 2)} for p in preds
                ]
            except Exception:
                formatted_preds = []

            results.append({
                "case_id": case["case_id"],
                "mode": case["mode"],
                "label": case["label"],
                "sex": sex,
                "model": m["name"],
                "history": history,
                "ground_truth": ground_truth,
                "predictions": formatted_preds,
                **metrics,
            })

    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    generate_report(results)
    print(f"Done. {len(results)} evaluations across {len(cases)} cases.")


def generate_report(results):
    df = pd.DataFrame(results)

    lines = []
    lines.append("=" * 70)
    lines.append("LAPORAN PERBANDINGAN MODEL — HEAD-TO-HEAD EVALUATION")
    lines.append("=" * 70)
    lines.append("")

    # --- Tabel 1: Rata-rata per model (semua kasus) ---
    lines.append("-" * 70)
    lines.append("TABEL 1: RATA-RATA METRIK PER MODEL (SEMUA KASUS)")
    lines.append("-" * 70)
    summary = df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
    lines.append(f"{'Model':<28} {'MAE (cm)':>10} {'RMSE (cm)':>10} {'R2':>10}")
    lines.append("-" * 60)
    for model, row in summary.iterrows():
        r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
        lines.append(f"{model:<28} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")
    lines.append("")

    # --- Tabel 2: Rata-rata per model, split by mode ---
    for mode in ["early_months", "normal"]:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            continue
        mode_label = "EARLY MONTHS (riwayat <= 4 bulan)" if mode == "early_months" else "NORMAL (riwayat >= 6 bulan)"
        lines.append("-" * 70)
        lines.append(f"TABEL 2: {mode_label}")
        lines.append("-" * 70)
        summary_mode = mode_df.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("rmse")
        lines.append(f"{'Model':<28} {'MAE (cm)':>10} {'RMSE (cm)':>10} {'R2':>10}")
        lines.append("-" * 60)
        for model, row in summary_mode.iterrows():
            r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
            lines.append(f"{model:<28} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")
        lines.append("")

    # --- Tabel 3: Detail per kasus ---
    lines.append("-" * 70)
    lines.append("TABEL 3: DETAIL PER KASUS")
    lines.append("-" * 70)
    lines.append(f"{'Case ID':<28} {'Model':<28} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    lines.append("-" * 82)
    for _, row in df.sort_values(["case_id", "rmse"]).iterrows():
        r2_str = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "N/A"
        lines.append(
            f"{row['case_id']:<28} {row['model']:<28} {row['mae']:>8.4f} {row['rmse']:>8.4f} {r2_str:>8}"
        )
    lines.append("")

    # --- Tabel 4: Model terbaik per kasus ---
    lines.append("-" * 70)
    lines.append("TABEL 4: MODEL TERBAIK PER KASUS (RMSE TERENDAH)")
    lines.append("-" * 70)
    best = df.loc[df.groupby("case_id")["rmse"].idxmin()]
    lines.append(f"{'Case ID':<28} {'Best Model':<28} {'RMSE':>8}")
    lines.append("-" * 66)
    for _, row in best.sort_values("case_id").iterrows():
        lines.append(f"{row['case_id']:<28} {row['model']:<28} {row['rmse']:>8.4f}")
    lines.append("")

    # --- Tabel 5: Jumlah "menang" per model ---
    lines.append("-" * 70)
    lines.append("TABEL 5: JUMLAH KASUS TERBAIK PER MODEL")
    lines.append("-" * 70)
    win_counts = best["model"].value_counts()
    for model, count in win_counts.items():
        lines.append(f"  {model:<28} {count} kasus")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    report_path = os.path.join(os.path.dirname(__file__), "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)


if __name__ == "__main__":
    run_comparison()
