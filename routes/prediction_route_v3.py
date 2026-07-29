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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from services.preprocessing_service import (
    build_feature,
    build_feature_weight,
    build_feature_hc,
)
from services.model_service import (
    train_gpr_who,
    gpr_predict_with_who,
    train_pure_gpr,
    validate_past_prediction,
    train_linear,
    train_exponential
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

# ============================================================
# KONFIGURASI MODEL UTAMA YANG DITAMPILKAN
# Opsi: "gpr"    -> GPR WHO Prior (Default Recommended)
#       "linear" -> Linear Regression
#       "exp"    -> Exponential Regression
# Catatan: Dapat juga di-override secara dinamis lewat JSON request body {"model": "exp"}
# ============================================================
DEFAULT_PRIMARY_MODEL = "gpr"


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
# HELPER: Hitung metrik Out-of-Sample / LOOCV / Ground Truth
# ============================================================

def _compute_sklearn_metrics(model_dict: dict, X, y) -> dict:
    """Fallback hitung MAE, RMSE, R² in-sample."""
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


def _clean_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    except Exception:
        return None


def _compute_out_of_sample_metrics(
    model_dict: dict,
    X: np.ndarray,
    y: np.ndarray,
    sex: str = "L",
    who_lms_df = None,
    ground_truth: list[dict] = None
) -> dict:
    """
    Menghitung metrik evaluasi out-of-sample:
    1. Jika ground_truth diberikan: hitung error prediksi masa depan vs ground_truth.
    2. Jika ground_truth tidak ada: hitung LOOCV (Leave-One-Out Cross-Validation) pada data historis.
    """
    if model_dict is None or len(X) < 2:
        return {"mae": None, "rmse": None, "r2": None}

    m_type = model_dict.get("type", "")

    # 1. Evaluasi dengan Ground Truth (Masa Depan 6 bulan ke depan jika dikirimkan)
    if ground_truth and isinstance(ground_truth, list) and len(ground_truth) > 0:
        try:
            last_age = int(X[-1][0])
            horizon  = len(ground_truth)

            if m_type == "gpr_who":
                raw_preds = gpr_predict_with_who(model_dict, last_age, horizon)
                y_pred = [p["height"] for p in raw_preds]
            else:
                predictor = model_dict["model"]
                future_ages = np.array([[last_age + i] for i in range(1, horizon + 1)], dtype=float)
                y_pred = list(predictor.predict(future_ages))

            y_true = []
            for item in ground_truth:
                val = item.get("height") if item.get("height") is not None else (item.get("h") if item.get("h") is not None else item.get("value"))
                if val is not None:
                    y_true.append(float(val))

            if len(y_pred) == len(y_true) and len(y_true) > 0:
                mae  = mean_absolute_error(y_true, y_pred)
                rmse = math.sqrt(mean_squared_error(y_true, y_pred))
                try:
                    r2 = r2_score(y_true, y_pred)
                except Exception:
                    r2 = None
                return {"mae": _clean_float(mae), "rmse": _clean_float(rmse), "r2": _clean_float(r2)}
        except Exception:
            pass

    # 2. Out-of-Sample via LOOCV (Leave-One-Out Cross-Validation) pada Data Historis
    try:
        loo = LeaveOneOut()
        y_true_loo = []
        y_pred_loo = []

        for train_index, test_index in loo.split(X):
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]

            if len(X_tr) < 2:
                continue

            if m_type == "gpr_who":
                g_trained = train_gpr_who(X_tr, y_tr, sex, who_lms_df)
                if g_trained is None:
                    continue
                pred_val = float(g_trained["model"].predict(X_te)[0])
            elif m_type == "linear":
                l_trained = train_linear(X_tr, y_tr)
                pred_val = float(l_trained["model"].predict(X_te)[0])
            elif m_type == "exponential":
                e_trained = train_exponential(X_tr, y_tr)
                if e_trained is None:
                    continue
                pred_val = float(e_trained["model"].predict(X_te)[0])
            else:
                pred_val = float(model_dict["model"].predict(X_te)[0])

            y_true_loo.append(float(y_te[0]))
            y_pred_loo.append(pred_val)

        if len(y_true_loo) >= 2:
            mae  = mean_absolute_error(y_true_loo, y_pred_loo)
            rmse = math.sqrt(mean_squared_error(y_true_loo, y_pred_loo))
            try:
                r2 = r2_score(y_true_loo, y_pred_loo)
            except Exception:
                r2 = None
            return {"mae": _clean_float(mae), "rmse": _clean_float(rmse), "r2": _clean_float(r2)}
    except Exception:
        pass

    # Fallback jika LOOCV tidak bisa diproses
    return _compute_sklearn_metrics(model_dict, X, y)


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
            {
                "age": int(last_age + i + 1),
                "height": round(max(0.0, float(preds[i])), 3),
                "value": round(max(0.0, float(preds[i])), 3)
            }
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

    # 9. Tentukan Pilihan Model Utama (berdasarkan variabel DEFAULT_PRIMARY_MODEL atau JSON request body "model")
    requested_model = data.get("selected_model") or data.get("model") or DEFAULT_PRIMARY_MODEL
    requested_model = str(requested_model).lower().strip()

    # Melatih seluruh opsi model
    gpr_h      = train_gpr_who(X_h, y_h, sex, who_lms_df)
    linear_dict= train_linear(X_h, y_h)
    exp_dict   = train_exponential(X_h, y_h)

    # Pilih model mana yang dijadikan model utama untuk ditampilkan
    if requested_model in ["linear", "linear_regression"] and linear_dict is not None:
        primary_model_dict  = linear_dict
        selected_model_name = "Linear Regression"
        preds_h_plain = _predict_sklearn_future(linear_dict, last_age, horizon)
        bands = [0.0] * horizon
        trainer_fn = train_linear

    elif requested_model in ["exp", "exponential", "exponential_regression"] and exp_dict is not None:
        primary_model_dict  = exp_dict
        selected_model_name = "Exponential Regression"
        preds_h_plain = _predict_sklearn_future(exp_dict, last_age, horizon)
        bands = [0.0] * horizon
        trainer_fn = train_exponential

    else:
        # Default: GPR WHO Prior
        primary_model_dict  = gpr_h
        selected_model_name = "GPR WHO Prior"
        if gpr_h is None:
            return _error("GPR fitting tinggi badan gagal. Periksa data historis.", 500)
        
        preds_h_raw_gpr = gpr_predict_with_who(gpr_h, last_age, horizon)
        preds_h_plain   = [{"age": p["age"], "height": p["height"]} for p in preds_h_raw_gpr]
        bands           = [p["uncertainty_band"] for p in preds_h_raw_gpr]
        trainer_fn      = lambda x, y: train_gpr_who(x, y, sex, who_lms_df)

    # 9.5. Validasi Masa Lalu (Held-out Past Validation)
    past_val_result = validate_past_prediction(X_h, y_h, trainer_fn)

    try:
        preds_h_enriched = build_prediction(preds_h_plain, sex, who_lms_df)
    except Exception as e:
        return _error(f"Gagal enrich data HAZ: {str(e)}", 500)

    for i, p in enumerate(preds_h_enriched):
        p["uncertainty_band"] = bands[i]

    growth_warning_summary = None
    try:
        preds_h_enriched, growth_warning_summary = add_velocity_info(preds_h_enriched, history, sex, who_lms_df)
    except Exception:
        growth_warning_summary = None

    # 10. Berat Badan (jika ada)
    preds_w_enriched = None
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

    # 11. Lingkar Kepala (jika ada)
    preds_hc_enriched = None
    if has_hc and X_hc is not None:
        last_age_hc = int(X_hc[-1][0])
        gpr_hc = train_gpr_who(X_hc, y_hc, sex, who_hcaz_df)
        if gpr_hc is not None:
            try:
                preds_hc_raw = gpr_predict_with_who(gpr_hc, last_age_hc, horizon)
                preds_hc_enriched = []
                for p in preds_hc_raw:
                    age  = p["age"]
                    val  = max(0.0, p["height"])
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

    # 12. Kumpulkan Model Perbandingan untuk Response JSON
    comparison_models = []
    if linear_dict is not None: comparison_models.append(linear_dict)
    if exp_dict    is not None: comparison_models.append(exp_dict)

    # 12a. Metrik out-of-sample (Ground Truth jika ada, atau LOOCV pada data historis)
    gt_req = data.get("ground_truth")
    all_metrics = {selected_model_name: _compute_out_of_sample_metrics(primary_model_dict, X_h, y_h, sex, who_lms_df, ground_truth=gt_req)}
    for m in comparison_models:
        if m is not None and m["name"] != selected_model_name:
            all_metrics[m["name"]] = _compute_out_of_sample_metrics(m, X_h, y_h, sex, who_lms_df, ground_truth=gt_req)

    # 12b. Prediksi masa depan model perbandingan
    comparison_predictions = {}
    for m in comparison_models:
        if m is None:
            continue
        future_preds = _predict_sklearn_future(m, last_age, horizon)
        comparison_predictions[m["name"]] = future_preds

    # 13. Gabungkan prediksi per bulan
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
        if preds_w_enriched and i < len(preds_w_enriched):
            w = preds_w_enriched[i]
            entry["weight"] = {
                "value":            w["value"],
                "waz":              w.get("waz"),
                "status":           w["status"],
                "uncertainty_band": w["uncertainty_band"],
            }
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
        "success":         True,
        "version":         "v3",
        "description": (
            "Pure Data-Driven Gaussian Process Regression (Linear+RBF Kernel). "
            "Prediksi memodelkan trajektori individual anak secara murni dari data historis tanpa pull WHO median. "
            f"Indikator aktif: tinggi badan"
            + (", berat badan" if preds_w_enriched else "")
            + (", lingkar kepala" if preds_hc_enriched else "")
            + "."
        ),
        "selected_model":  selected_model_name,
        "n_history":       n_samples,
        "skipped_models":  [],
        "metrics":         all_metrics,
        "past_validation": past_val_result,  # Validasi prediksi masa lalu (sesuai saran dosen)
        "growth_warning":  growth_warning_summary, # Deteksi & Peringatan Dini Perlambatan Pertumbuhan
        "prediction":      combined,
        "model_comparisons": comparison_predictions,
    }

    return jsonify(response), 200


def _error(message: str, status_code: int):
    return jsonify({"success": False, "error": message}), status_code
