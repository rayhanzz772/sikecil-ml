"""
prediction_route_v3.py — Endpoint /api/predict/v3

Model yang digunakan:
    - Gaussian Process Regression (GPR) + WHO Median Prior

Keunggulan dibanding v1 dan v2:
    - Tidak pernah menghasilkan prediksi yang menurun (guaranteed)
    - Menggunakan kurva WHO sebagai "pengetahuan awal" (prior)
    - Menghasilkan uncertainty_band (interval kepercayaan 95%)
    - Semakin sedikit data -> lebih dekat ke WHO
    - Semakin banyak data -> lebih dipersonalisasi ke individu
"""
import os
from flask import Blueprint, request, jsonify

from services.preprocessing_service import build_feature
from services.model_service import (
    train_gpr_who,
    gpr_predict_with_who
)
from services.prediction_service import build_prediction
from services.who_service import load_who_lms
from services.growth_validator import add_velocity_info

# ============================================================
# BLUEPRINT & WHO LMS
# ============================================================

prediction_v3_bp = Blueprint("prediction_v3", __name__)

_WHO_LMS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "who_lms.csv"
)

_who_lms_df_v3 = None


def _get_who_lms():
    global _who_lms_df_v3
    if _who_lms_df_v3 is None:
        _who_lms_df_v3 = load_who_lms(os.path.abspath(_WHO_LMS_PATH))
    return _who_lms_df_v3


# ============================================================
# POST /api/predict/v3
# ============================================================

@prediction_v3_bp.route("/api/predict/v3", methods=["POST"])
def predict_v3():
    """
    Endpoint prediksi menggunakan Gaussian Process Regression + WHO Prior.

    Tidak memerlukan banyak data — GPR akan menggunakan WHO sebagai
    'ekspektasi awal' dan belajar dari penyimpangan data individu.

    Request Body (JSON):
    --------------------
    {
        "sex":     "L" | "P",
        "history": [
            {"age": int, "height": float},
            ...
        ],
        "horizon": int  (opsional, default=6, range 1-24)
    }

    Response (success):
    -------------------
    {
        "success": true,
        "version": "v3",
        "selected_model": "GPR WHO Prior",
        "n_history": 6,
        "prediction": [
            {
                "age": 6,
                "height": 67.85,
                "haz": 0.21,
                "status": "Normal",
                "uncertainty_band": 0.42
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

    # 5. Preprocessing
    try:
        X, y = build_feature(history)
    except ValueError as e:
        return _error(str(e), 400)

    n_samples = len(X)
    last_age  = int(X[-1][0])

    try:
        who_lms_df = _get_who_lms()
    except Exception as e:
        return _error(f"Gagal memuat data WHO LMS: {str(e)}", 500)

    # 6. Latih GPR + WHO Prior
    gpr_dict = train_gpr_who(X, y, sex, who_lms_df)
    if gpr_dict is None:
        return _error("GPR fitting gagal. Periksa data historis yang diberikan.", 500)

    # 7. Prediksi (mendapatkan tinggi + uncertainty_band)
    try:
        preds_with_band = gpr_predict_with_who(gpr_dict, last_age, horizon)
    except Exception as e:
        return _error(f"Prediksi GPR gagal: {str(e)}", 500)

    # 8. Enrich dengan HAZ dan status (pisahkan uncertainty_band dulu)
    preds_plain = [{"age": p["age"], "height": p["height"]} for p in preds_with_band]

    try:
        preds_enriched = build_prediction(preds_plain, sex, who_lms_df)
    except Exception as e:
        return _error(f"Gagal memperkaya data prediksi: {str(e)}", 500)

    # 9. Gabungkan kembali uncertainty_band ke preds_enriched
    for i, p in enumerate(preds_enriched):
        p["uncertainty_band"] = preds_with_band[i]["uncertainty_band"]

    # 10. Tambahkan growth velocity info
    try:
        preds_enriched = add_velocity_info(preds_enriched, history, sex, who_lms_df)
    except Exception:
        pass  # velocity info bersifat opsional

    # 11. Hitung in-sample metrics (GPR pada data historis sendiri)
    #     Digunakan agar format response selaras dengan v1 dan v2.
    try:
        import math
        import numpy as _np
        predictor = gpr_dict["model"]
        y_fitted  = predictor.predict(X)
        mae_val   = float(_np.mean(_np.abs(y - y_fitted)))
        rmse_val  = float(math.sqrt(_np.mean((y - y_fitted) ** 2)))
        ss_res    = float(_np.sum((y - y_fitted) ** 2))
        ss_tot    = float(_np.sum((y - float(_np.mean(y))) ** 2))
        r2_val    = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        gpr_metrics = {
            "GPR WHO Prior": {
                "mae":  round(mae_val, 6),
                "rmse": round(rmse_val, 6),
                "r2":   round(r2_val, 6)
            }
        }
    except Exception:
        gpr_metrics = {
            "GPR WHO Prior": {"mae": None, "rmse": None, "r2": None}
        }

    # 12. Bangun response (format selaras dengan v1 & v2)
    response = {
        "success":        True,
        "version":        "v3",
        "description":    (
            "Gaussian Process Regression dengan WHO Median sebagai Prior Mean. "
            "Prediksi mencerminkan trajektori individu dianchored ke kurva populasi WHO."
        ),
        "selected_model": "GPR WHO Prior",
        "n_history":      n_samples,
        "skipped_models": [],
        "metrics":        gpr_metrics,
        "prediction":     preds_enriched
    }

    return jsonify(response), 200


def _error(message: str, status_code: int):
    return jsonify({"success": False, "error": message}), status_code
