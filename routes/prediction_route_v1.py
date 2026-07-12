"""
prediction_route_v1.py — Endpoint /api/predict/v1

Model yang digunakan (Klasik):
    - Linear Regression
    - Polynomial Regression Degree 2
    - Polynomial Regression Degree 3

Endpoint ini disediakan untuk keperluan PERBANDINGAN AKADEMIS.
Tidak menggunakan HAZ Predictor atau Growth Validator.
"""
import os
from flask import Blueprint, request, jsonify

from services.preprocessing_service import build_feature
from services.model_service import (
    train_linear,
    train_polynomial,
    evaluate_models,
    get_sorted_models
)
from services.prediction_service import recursive_predict, build_prediction
from services.who_service import load_who_lms

# ============================================================
# BLUEPRINT & WHO LMS
# ============================================================

prediction_v1_bp = Blueprint("prediction_v1", __name__)

_WHO_LMS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "who_lms.csv"
)

_who_lms_df_v1 = None


def _get_who_lms():
    global _who_lms_df_v1
    if _who_lms_df_v1 is None:
        _who_lms_df_v1 = load_who_lms(os.path.abspath(_WHO_LMS_PATH))
    return _who_lms_df_v1


# ============================================================
# POST /api/predict/v1
# ============================================================

@prediction_v1_bp.route("/api/predict/v1", methods=["POST"])
def predict_v1():
    """
    Endpoint prediksi menggunakan model KLASIK (Linear + Polynomial).

    Digunakan untuk perbandingan akademis dengan /api/predict/v2.

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
        "version": "v1",
        "models_used": ["Linear Regression", "Polynomial Degree 2", "Polynomial Degree 3"],
        "selected_model": "...",
        "metrics": { ... },
        "prediction": [ ... ],
        "warning": "..."
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
    last_age = int(X[-1][0])

    try:
        who_lms_df = _get_who_lms()
    except Exception as e:
        return _error(f"Gagal memuat data WHO LMS: {str(e)}", 500)

    # 6. Latih model klasik
    trained_models = [
        train_linear(X, y),
        train_polynomial(X, y, degree=2),   # None jika data tidak cukup
        train_polynomial(X, y, degree=3),   # None jika data tidak cukup
    ]

    try:
        metrics = evaluate_models(trained_models, X, y)
    except Exception as e:
        return _error(f"Evaluasi model gagal: {str(e)}", 500)

    try:
        sorted_models = get_sorted_models(trained_models, metrics)
    except ValueError as e:
        return _error(str(e), 500)

    # 7. Prediksi menggunakan model terbaik (RMSE terendah)
    best_model = sorted_models[0]

    try:
        preds_raw = recursive_predict(best_model, last_age, horizon)
        prediction = build_prediction(preds_raw, sex, who_lms_df)
    except Exception as e:
        return _error(f"Prediksi gagal: {str(e)}", 500)

    # 8. Models yang aktif (tidak None)
    active_models = [m["name"] for m in trained_models if m is not None]
    skipped_models = [m for m in ["Linear Regression", "Polynomial Degree 2", "Polynomial Degree 3"]
                      if m not in active_models]

    # 9. Bangun response
    response = {
        "success": True,
        "version": "v1",
        "description": "Model Klasik: Linear Regression & Polynomial Regression",
        "models_used": active_models,
        "selected_model": best_model["name"],
        "n_history": n_samples,
        "skipped_models": skipped_models,
        "metrics": metrics,
        "prediction": prediction,
        "warning": (
            "PERINGATAN AKADEMIS: Model Linear dan Polynomial tidak memiliki batas atas "
            "biologis (asymptote). Prediksi jangka panjang berisiko menghasilkan nilai "
            "yang tidak realistis atau kurva yang menurun."
        )
    }

    return jsonify(response), 200


def _error(message: str, status_code: int):
    return jsonify({"success": False, "error": message}), status_code
