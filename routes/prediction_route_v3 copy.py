"""
prediction_route_v3.py — Endpoint /api/predict/v3

Model yang digunakan:
    - Gaussian Process Regression (GPR) + WHO Median Prior  [primary]
    - Linear Regression                                      [comparison]
    - Polynomial Regression (degree 2 & 3)                  [comparison]

Keunggulan dibanding v1 dan v2:
    - Tidak pernah menghasilkan prediksi yang menurun (guaranteed)
    - Menggunakan kurva WHO sebagai "pengetahuan awal" (prior)
    - Menghasilkan uncertainty_band (interval kepercayaan 95%)
    - Semakin sedikit data -> lebih dekat ke WHO
    - Semakin banyak data -> lebih dipersonalisasi ke individu
    - [v3.1] Mendukung prediksi TIGA indikator sekaligus:
        1. Tinggi Badan (HAZ)
        2. Berat Badan (WAZ)   [opsional]
        3. Lingkar Kepala (HCAZ) [opsional]
    - [v3.2] Model perbandingan (Linear & Polynomial) disertakan di:
        - metrics            : MAE, RMSE, R² in-sample
        - model_comparisons  : prediksi masa depan per model
"""
import os
import math
import numpy as np
from flask import Blueprint, request, jsonify

from services.preprocessing_service import (
    build_feature,
    build_feature_weight,
    build_feature_hc,
)
from services.model_service import (
    train_gpr_who,
    gpr_predict_with_who,
    train_linear,
    train_polynomial,
)
from services.prediction_service import build_prediction
from services.who_service import (
    load_who_lms,
    load_who_waz,
    load_who_hcaz,
    compute_waz,
    compute_hcaz,
    classify_zscore_status,
    get_who_weight_median,
    get_who_hc_median,
)
from services.growth_validator import add_velocity_info

# ============================================================
# BLUEPRINT & DATA PATHS
# ============================================================

prediction_v3_bp = Blueprint("prediction_v3", __name__)

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

_WHO_LMS_PATH  = os.path.join(_DATA_DIR, "who_lms.csv")
_WHO_WAZ_PATH  = os.path.join(_DATA_DIR, "who_waz.csv")
_WHO_HCAZ_PATH = os.path.join(_DATA_DIR, "who_hcaz.csv")

# Lazy-load cache (loaded once, reused for all requests)
_who_lms_df_v3  = None
_who_waz_df_v3  = None
_who_hcaz_df_v3 = None


def _get_who_lms():
    global _who_lms_df_v3
    if _who_lms_df_v3 is None:
        _who_lms_df_v3 = load_who_lms(_WHO_LMS_PATH)
    return _who_lms_df_v3


def _get_who_waz():
    global _who_waz_df_v3
    if _who_waz_df_v3 is None:
        _who_waz_df_v3 = load_who_waz(_WHO_WAZ_PATH)
    return _who_waz_df_v3


def _get_who_hcaz():
    global _who_hcaz_df_v3
    if _who_hcaz_df_v3 is None:
        _who_hcaz_df_v3 = load_who_hcaz(_WHO_HCAZ_PATH)
    return _who_hcaz_df_v3


# ============================================================
# HELPER: Hitung metrik in-sample GPR
# ============================================================

def _compute_gpr_metrics(gpr_dict: dict, X, y) -> dict:
    """Menghitung MAE, RMSE, R² dari hasil fitting GPR pada data historis."""
    try:
        predictor = gpr_dict["model"]
        y_fitted  = predictor.predict(X)
        mae_val   = float(np.mean(np.abs(y - y_fitted)))
        rmse_val  = float(math.sqrt(np.mean((y - y_fitted) ** 2)))
        ss_res    = float(np.sum((y - y_fitted) ** 2))
        ss_tot    = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_val    = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        return {"mae": round(mae_val, 6), "rmse": round(rmse_val, 6), "r2": round(r2_val, 6)}
    except Exception:
        return {"mae": None, "rmse": None, "r2": None}


def _compute_sklearn_metrics(model_dict: dict, X, y) -> dict:
    """
    Menghitung MAE, RMSE, R² in-sample untuk model sklearn
    (LinearRegression, Pipeline Polynomial, dst).
    """
    try:
        model     = model_dict["model"]
        y_fitted  = model.predict(X)
        mae_val   = float(np.mean(np.abs(y - y_fitted)))
        rmse_val  = float(math.sqrt(np.mean((y - y_fitted) ** 2)))
        ss_res    = float(np.sum((y - y_fitted) ** 2))
        ss_tot    = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_val    = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        return {"mae": round(mae_val, 6), "rmse": round(rmse_val, 6), "r2": round(r2_val, 6)}
    except Exception:
        return {"mae": None, "rmse": None, "r2": None}


def _predict_sklearn_future(
    model_dict: dict,
    last_age: int,
    horizon: int,
) -> list[dict]:
    """
    Hasilkan prediksi masa depan (age = last_age+1 … last_age+horizon)
    menggunakan model sklearn biasa (Linear / Polynomial).

    Returns
    -------
    list of dict: [{"age": int, "value": float}, ...]
    """
    try:
        model = model_dict["model"]
        future_ages = np.array([[last_age + i] for i in range(1, horizon + 1)], dtype=float)
        preds = model.predict(future_ages)
        return [
            {"age": int(last_age + i + 1), "value": round(max(0.0, float(preds[i])), 3)}
            for i in range(horizon)
        ]
    except Exception:
        return []


# ============================================================
# HELPER: Prediksi + enrich untuk satu indikator GPR
# ============================================================

def _predict_indicator(
    gpr_dict: dict,
    last_age: int,
    horizon: int,
    sex: str,
    who_df,
    compute_z_fn,
    classify_fn,
    indicator_key: str,
    z_key: str,
    unit: str,
) -> list[dict]:
    """
    Generik: jalankan GPR predict + hitung z-score + classify untuk satu indikator.

    Returns list of dicts per bulan prediksi dengan format:
        {"age": int, "value": float, "z_score": float, "status": str, "uncertainty_band": float}
    """
    preds_with_band = gpr_predict_with_who(gpr_dict, last_age, horizon)
    results = []
    for p in preds_with_band:
        age    = p["age"]
        val    = p["height"]  # note: gpr_predict_with_who returns key "height" generically

        # Monotonicity guard — nilai tidak boleh turun dari WHO floor
        who_floor = (
            get_who_weight_median(age, sex, who_df)
            if indicator_key == "weight"
            else get_who_hc_median(age, sex, who_df)
            if indicator_key == "head_circ"
            else None
        )
        # Pastikan nilai tidak negatif
        val = max(0.0, val)

        z    = compute_z_fn(val, age, sex, who_df)
        stat = classify_fn(z, indicator_key)

        results.append({
            "age":              age,
            "value":            round(val, 3),
            "z_score":          round(z, 3) if not np.isnan(z) else None,
            "status":           stat,
            "uncertainty_band": round(p["uncertainty_band"], 3),
        })
    return results


# ============================================================
# POST /api/predict/v3
# ============================================================

@prediction_v3_bp.route("/api/predict/v3", methods=["POST"])
def predict_v3():
    """
    Endpoint prediksi multi-indikator menggunakan Gaussian Process Regression + WHO Prior.

    Request Body (JSON):
    --------------------
    {
        "sex":     "L" | "P",
        "horizon": int          (opsional, default=6, range 1-24)
        "history": [
            {
                "age":       int,    // wajib
                "height":    float,  // wajib (cm)
                "weight":    float,  // opsional (kg)
                "head_circ": float   // opsional (cm)
            },
            ...
        ]
    }

    Response (success):
    -------------------
    {
        "success": true,
        "version": "v3",
        "selected_model": "GPR WHO Prior",
        "n_history": 6,
        "metrics": { "GPR WHO Prior": {"mae": .., "rmse": .., "r2": ..} },
        "prediction": [
            {
                "age": 6,
                "height":    {"value": 67.85, "haz":  0.21,  "status": "Normal",    "uncertainty_band": 0.42},
                "weight":    {"value": 7.34,  "waz": -0.15,  "status": "Gizi Baik", "uncertainty_band": 0.08},
                "head_circ": {"value": 43.2,  "hcaz": -0.10, "status": "Normal",    "uncertainty_band": 0.30}
            },
            ...
        ]
    }
    """
    # 1. Parse JSON
    if not request.is_json:
        return _error("Request harus menggunakan Content-Type: application/json.", 400)

    data = request.get_json(silent=True)
    if data is None:
        return _error("Body request bukan JSON yang valid.", 400)

    # 2. Validasi sex
    sex = data.get("sex")
    if sex is None:
        return _error("Field 'sex' wajib diisi.", 400)
    if not isinstance(sex, str) or sex not in ("L", "P"):
        return _error("Field 'sex' harus bernilai 'L' atau 'P'.", 400)

    # 3. Validasi history
    history = data.get("history")
    if history is None:
        return _error("Field 'history' wajib diisi.", 400)
    if not isinstance(history, list):
        return _error("Field 'history' harus berupa array.", 400)

    # 4. Validasi horizon
    horizon = data.get("horizon", 6)
    if not isinstance(horizon, int) or not (1 <= horizon <= 24):
        return _error("Field 'horizon' harus berupa integer antara 1 dan 24.", 400)

    # 5. Preprocessing — Tinggi Badan (wajib)
    try:
        X_h, y_h = build_feature(history)
    except ValueError as e:
        return _error(str(e), 400)

    n_samples = len(X_h)
    last_age  = int(X_h[-1][0])

    # 6. Preprocessing — Berat Badan (opsional)
    has_weight = any("weight" in e for e in history)
    X_w, y_w = None, None
    if has_weight:
        try:
            history_w = [e for e in history if "weight" in e]
            X_w, y_w = build_feature_weight(history_w)
        except ValueError:
            has_weight = False  # Data tidak valid, skip saja

    # 7. Preprocessing — Lingkar Kepala (opsional)
    has_hc = any("head_circ" in e for e in history)
    X_hc, y_hc = None, None
    if has_hc:
        try:
            history_hc = [e for e in history if "head_circ" in e]
            X_hc, y_hc = build_feature_hc(history_hc)
        except ValueError:
            has_hc = False  # Data tidak valid, skip saja

    # 8. Load WHO tables
    try:
        who_lms_df  = _get_who_lms()
        who_waz_df  = _get_who_waz()  if has_weight else None
        who_hcaz_df = _get_who_hcaz() if has_hc     else None
    except Exception as e:
        return _error(f"Gagal memuat data WHO: {str(e)}", 500)

    # 9. Latih & prediksi GPR — Tinggi Badan
    gpr_h = train_gpr_who(X_h, y_h, sex, who_lms_df)
    if gpr_h is None:
        return _error("GPR fitting tinggi badan gagal. Periksa data historis.", 500)

    try:
        preds_h_raw = gpr_predict_with_who(gpr_h, last_age, horizon)
    except Exception as e:
        return _error(f"Prediksi GPR tinggi badan gagal: {str(e)}", 500)

    # Enrich dengan HAZ (pakai build_prediction yang sudah ada)
    preds_h_plain = [{"age": p["age"], "height": p["height"]} for p in preds_h_raw]
    try:
        preds_h_enriched = build_prediction(preds_h_plain, sex, who_lms_df)
    except Exception as e:
        return _error(f"Gagal enrich data HAZ: {str(e)}", 500)

    # Tambahkan uncertainty_band ke height
    for i, p in enumerate(preds_h_enriched):
        p["uncertainty_band"] = preds_h_raw[i]["uncertainty_band"]

    # Tambahkan velocity info (opsional, tidak fatal jika gagal)
    try:
        preds_h_enriched = add_velocity_info(preds_h_enriched, history, sex, who_lms_df)
    except Exception:
        pass

    # 10. Latih & prediksi GPR — Berat Badan (jika ada)
    preds_w_enriched = None
    gpr_w = None
    if has_weight and X_w is not None:
        last_age_w = int(X_w[-1][0])
        gpr_w = train_gpr_who(X_w, y_w, sex, who_waz_df)
        if gpr_w is not None:
            try:
                preds_w_raw = gpr_predict_with_who(gpr_w, last_age_w, horizon)
                preds_w_enriched = []
                for p in preds_w_raw:
                    age = p["age"]
                    val = max(0.0, p["height"])
                    waz  = compute_waz(val, age, sex, who_waz_df)
                    stat = classify_zscore_status(waz, "weight")
                    preds_w_enriched.append({
                        "age":              age,
                        "value":            round(val, 3),
                        "waz":              round(waz, 3) if not np.isnan(waz) else None,
                        "status":           stat,
                        "uncertainty_band": round(p["uncertainty_band"], 3),
                    })
            except Exception:
                preds_w_enriched = None

    # 11. Latih & prediksi GPR — Lingkar Kepala (jika ada)
    preds_hc_enriched = None
    gpr_hc = None
    if has_hc and X_hc is not None:
        last_age_hc = int(X_hc[-1][0])
        gpr_hc = train_gpr_who(X_hc, y_hc, sex, who_hcaz_df)
        if gpr_hc is not None:
            try:
                preds_hc_raw = gpr_predict_with_who(gpr_hc, last_age_hc, horizon)
                preds_hc_enriched = []
                for p in preds_hc_raw:
                    age = p["age"]
                    val = max(0.0, p["height"])
                    hcaz = compute_hcaz(val, age, sex, who_hcaz_df)
                    stat = classify_zscore_status(hcaz, "head_circ")
                    preds_hc_enriched.append({
                        "age":              age,
                        "value":            round(val, 3),
                        "hcaz":             round(hcaz, 3) if not np.isnan(hcaz) else None,
                        "status":           stat,
                        "uncertainty_band": round(p["uncertainty_band"], 3),
                    })
            except Exception:
                preds_hc_enriched = None

    # 12. Latih model perbandingan: Linear Regression & Polynomial Regression
    #     (hanya untuk tinggi badan — indikator primer)
    comparison_models = []
    linear_dict = None
    poly2_dict  = None
    poly3_dict  = None

    try:
        linear_dict = train_linear(X_h, y_h)
        comparison_models.append(linear_dict)
    except Exception:
        linear_dict = None

    try:
        poly2_dict = train_polynomial(X_h, y_h, degree=2)
        if poly2_dict is not None:
            comparison_models.append(poly2_dict)
    except Exception:
        poly2_dict = None

    try:
        poly3_dict = train_polynomial(X_h, y_h, degree=3)
        if poly3_dict is not None:
            comparison_models.append(poly3_dict)
    except Exception:
        poly3_dict = None

    # 12a. Hitung metrik in-sample — GPR + model perbandingan
    all_metrics = {"GPR WHO Prior": _compute_gpr_metrics(gpr_h, X_h, y_h)}
    for m in comparison_models:
        if m is not None:
            all_metrics[m["name"]] = _compute_sklearn_metrics(m, X_h, y_h)

    # 12b. Buat prediksi masa depan untuk model perbandingan
    comparison_predictions = {}
    for m in comparison_models:
        if m is None:
            continue
        future_preds = _predict_sklearn_future(m, last_age, horizon)
        comparison_predictions[m["name"]] = future_preds

    # (alias untuk kompatibilitas ke bawah)
    gpr_metrics = all_metrics

    # 13. Gabungkan prediksi menjadi satu list per bulan
    combined = []
    for i, h in enumerate(preds_h_enriched):
        entry = {
            "age": h["age"],
            "height": {
                "value":            h.get("height"),
                "haz":              h.get("haz"),
                "status":           h.get("status"),
                "uncertainty_band": h.get("uncertainty_band"),
            }
        }
        # Tambahkan weight jika ada
        if preds_w_enriched and i < len(preds_w_enriched):
            w = preds_w_enriched[i]
            entry["weight"] = {
                "value":            w["value"],
                "waz":              w.get("waz"),
                "status":           w["status"],
                "uncertainty_band": w["uncertainty_band"],
            }
        # Tambahkan head_circ jika ada
        if preds_hc_enriched and i < len(preds_hc_enriched):
            hc = preds_hc_enriched[i]
            entry["head_circ"] = {
                "value":            hc["value"],
                "hcaz":             hc.get("hcaz"),
                "status":           hc["status"],
                "uncertainty_band": hc["uncertainty_band"],
            }
        combined.append(entry)

    # 14. Bangun response
    response = {
        "success":        True,
        "version":        "v3",
        "description":    (
            "Gaussian Process Regression dengan WHO Median sebagai Prior Mean. "
            "Prediksi mencerminkan trajektori individu dianchored ke kurva populasi WHO. "
            f"Indikator aktif: tinggi badan"
            + (", berat badan" if preds_w_enriched else "")
            + (", lingkar kepala" if preds_hc_enriched else "")
            + "."
        ),
        "selected_model": "GPR WHO Prior",
        "n_history":      n_samples,
        "skipped_models": [],
        "metrics":        all_metrics,
        "prediction":     combined,
        # Model perbandingan — hanya prediksi tinggi badan untuk evaluasi
        "model_comparisons": comparison_predictions,
    }

    return jsonify(response), 200


def _error(message: str, status_code: int):
    return jsonify({"success": False, "error": message}), status_code
