import os
from flask import Blueprint, request, jsonify

from services.preprocessing_service import build_feature
from services.model_service import (
    train_bayesian_ridge,
    train_von_bertalanffy,
    train_gompertz,
    evaluate_models,
    get_sorted_models
)
from services.prediction_service import recursive_predict, build_prediction
from services.who_service import load_who_lms
from services.haz_predictor import train_haz_predictor
from services.growth_validator import is_growth_realistic, add_velocity_info


# ==========================================================
# BLUEPRINT & WHO LMS (lazy-loaded saat pertama request)
# ==========================================================

prediction_bp = Blueprint("prediction", __name__)

# Path ke file WHO LMS CSV
_WHO_LMS_PATH = os.path.join(
    os.path.dirname(__file__),   # .../routes/
    "..",                         # .../
    "data",
    "who_lms.csv"
)

# Cache agar tidak re-load setiap request
_who_lms_df = None


def _get_who_lms():
    """Lazy-load dan cache tabel WHO LMS dari CSV."""
    global _who_lms_df
    if _who_lms_df is None:
        _who_lms_df = load_who_lms(os.path.abspath(_WHO_LMS_PATH))
    return _who_lms_df


# ==========================================================
# POST /api/predict
# ==========================================================

@prediction_bp.route("/api/predict", methods=["POST"])
@prediction_bp.route("/api/predict/v2", methods=["POST"])
def predict():
    """
    Endpoint prediksi pertumbuhan tinggi badan balita — Smart Hybrid System (v2).

    Juga dapat diakses via /api/predict (backward-compatible).
    Untuk perbandingan dengan model klasik, gunakan /api/predict/v1.
    Request Body (JSON):
    --------------------
    {
        "sex":     "L" | "P",
        "history": [
            {"age": int, "height": float},
            ...
        ],
        "horizon": int  (opsional, default=6, range 1–24)
    }

    Response (success):
    -------------------
    {
        "success": true,
        "selected_model": "...",
        "metrics": {
            "Linear Regression": {"mae": ..., "rmse": ..., "r2": ...},
            ...
        },
        "prediction": [
            {"age": int, "height": float, "haz": float, "status": str},
            ...
        ]
    }

    Response (error):
    -----------------
    {
        "success": false,
        "error": "Pesan error"
    }
    """

    # ----------------------------------------------------------
    # 1. Parse JSON body
    # ----------------------------------------------------------
    if not request.is_json:
        return _error("Request harus menggunakan Content-Type: application/json.", 400)

    data = request.get_json(silent=True)
    if data is None:
        return _error("Body request bukan JSON yang valid.", 400)

    # ----------------------------------------------------------
    # 2. Validasi field wajib: sex
    # ----------------------------------------------------------
    sex = data.get("sex")
    if sex is None:
        return _error("Field 'sex' wajib diisi.", 400)
    if not isinstance(sex, str) or sex not in ("L", "P"):
        return _error("Field 'sex' harus bernilai 'L' (Laki-laki) atau 'P' (Perempuan).", 400)

    # ----------------------------------------------------------
    # 3. Validasi field wajib: history
    # ----------------------------------------------------------
    history = data.get("history")
    if history is None:
        return _error("Field 'history' wajib diisi.", 400)
    if not isinstance(history, list):
        return _error("Field 'history' harus berupa array.", 400)

    # ----------------------------------------------------------
    # 4. Validasi field opsional: horizon
    # ----------------------------------------------------------
    horizon = data.get("horizon", 6)
    if not isinstance(horizon, int) or not (1 <= horizon <= 24):
        return _error("Field 'horizon' harus berupa integer antara 1 dan 24.", 400)

    # ----------------------------------------------------------
    # 5. Preprocessing — bangun fitur X dan y
    # ----------------------------------------------------------
    try:
        X, y = build_feature(history)
    except ValueError as e:
        return _error(str(e), 400)

    n_samples = len(X)
    last_age = int(X[-1][0])   # usia terakhir dari data historis
    
    try:
        who_lms_df = _get_who_lms()
    except Exception as e:
        return _error(f"Gagal memuat data WHO LMS: {str(e)}", 500)

    # ----------------------------------------------------------
    # 6. Branching: Early Months vs Normal Mode
    # ----------------------------------------------------------
    best_model = None
    prediction = None
    prediction_mode = "normal"
    skipped_models = []
    metrics = {}
    
    if n_samples <= 5:
        # --- EARLY MONTHS MODE (HAZ-Space) ---
        prediction_mode = "early_months"
        haz_model_dict = train_haz_predictor(history, sex, who_lms_df)
        
        if haz_model_dict:
            try:
                preds_raw = recursive_predict(haz_model_dict, last_age, horizon)
                preds_enriched = build_prediction(preds_raw, sex, who_lms_df)
                
                if is_growth_realistic(preds_enriched, sex, who_lms_df):
                    best_model = haz_model_dict
                    prediction = preds_enriched
            except Exception as e:
                pass # fallback ke normal mode jika gagal
                
    if prediction is None:
        # --- NORMAL MODE (Model Based) ---
        prediction_mode = "normal"
        
        trained_models = [
            train_bayesian_ridge(X, y),
            # train_von_bertalanffy(X, y),        # None jika fitting gagal
            train_gompertz(X, y)                # None jika fitting gagal
        ]

        try:
            metrics = evaluate_models(trained_models, X, y)
        except Exception as e:
            return _error(f"Evaluasi model gagal: {str(e)}", 500)

        try:
            sorted_models = get_sorted_models(trained_models, metrics)
        except ValueError as e:
            return _error(str(e), 500)
            
        skipped_models = _get_skipped_models(trained_models)
        
        # Cek model dari yang terbaik ke terburuk
        for model in sorted_models:
            try:
                preds_raw = recursive_predict(model, last_age, horizon)
                preds_enriched = build_prediction(preds_raw, sex, who_lms_df)
                
                if is_growth_realistic(preds_enriched, sex, who_lms_df):
                    best_model = model
                    prediction = preds_enriched
                    break
            except Exception:
                continue

        # Jika semua model menghasilkan prediksi tidak realistis, fallback ke Bayesian Ridge 
        if best_model is None:
            best_model = next((m for m in sorted_models if m["type"] == "bayesian_ridge"), sorted_models[0])
            try:
                preds_raw = recursive_predict(best_model, last_age, horizon)
                prediction = build_prediction(preds_raw, sex, who_lms_df)
            except Exception as e:
                return _error(f"Prediksi fallback gagal: {str(e)}", 500)

    # ----------------------------------------------------------
    # 7. Tambahkan Growth Velocity Info
    # ----------------------------------------------------------
    if prediction:
        prediction = add_velocity_info(prediction, history, sex, who_lms_df)

    # ----------------------------------------------------------
    # 8. Bangun dan kembalikan response
    # ----------------------------------------------------------
    response = {
        "success":        True,
        "selected_model": best_model["name"] if best_model else "Unknown",
        "prediction_mode": prediction_mode,
        "n_history":      n_samples,
        "skipped_models": skipped_models,
        "metrics":        metrics,
        "prediction":     prediction
    }

    return jsonify(response), 200


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _error(message: str, status_code: int):
    """Membuat response error yang terstandarisasi."""
    return jsonify({"success": False, "error": message}), status_code


def _get_skipped_models(trained_models: list) -> list[str]:
    """
    Mengembalikan nama model yang dilewati (bernilai None).

    Digunakan untuk transparansi pada response.
    """
    all_names = ["Bayesian Ridge", "Von Bertalanffy", "Gompertz"]
    active_names = {m["name"] for m in trained_models if m is not None}
    return [name for name in all_names if name not in active_names]
