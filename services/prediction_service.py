import numpy as np
import pandas as pd

from services.who_service import compute_haz, classify_status


# ==========================================================
# RECURSIVE PREDICT
# ==========================================================

def recursive_predict(
    best_model_dict: dict,
    last_age: int,
    horizon: int
) -> list[dict]:
    """
    Memprediksi tinggi badan untuk horizon bulan ke depan
    menggunakan model terpilih.

    Setiap prediksi dilakukan secara sequential:
        age = last_age + 1, last_age + 2, ..., last_age + horizon

    Catatan: Disebut "recursive" karena prediksi berjalan berurutan.
    Model dilatih dengan fitur tunggal 'age', bukan truly autoregressive.

    Parameters
    ----------
    best_model_dict : dict
        Salah satu dari output train_linear / train_polynomial / train_gompertz.
        Harus memiliki key 'model' (objek dengan method .predict).
    last_age : int
        Usia terakhir dari data historis (dalam bulan).
    horizon : int
        Jumlah bulan ke depan yang ingin diprediksi.

    Returns
    -------
    list of dict
        [
            {"age": int, "height": float},
            ...
        ]
    """
    model = best_model_dict["model"]
    predictions = []

    for step in range(1, horizon + 1):
        future_age = last_age + step

        # Buat array fitur shape (1, 1)
        X_input = np.array([[future_age]], dtype=float)

        height_pred = float(model.predict(X_input)[0])

        # Pastikan tinggi tidak negatif (safeguard)
        height_pred = max(0.0, height_pred)

        predictions.append({
            "age":    future_age,
            "height": round(height_pred, 2)
        })

    return predictions


# ==========================================================
# BUILD PREDICTION (dengan HAZ & Status)
# ==========================================================

def build_prediction(
    predictions_raw: list[dict],
    sex: str,
    who_lms_df: pd.DataFrame
) -> list[dict]:
    """
    Memperkaya hasil prediksi tinggi badan dengan HAZ dan status stunting
    menggunakan tabel WHO LMS.

    Parameters
    ----------
    predictions_raw : list of dict
        Output dari recursive_predict():
            [{"age": int, "height": float}, ...]
    sex : str
        Jenis kelamin: 'L' (Laki-laki) atau 'P' (Perempuan).
    who_lms_df : pd.DataFrame
        Tabel WHO LMS yang sudah dimuat dari CSV.

    Returns
    -------
    list of dict
        [
            {
                "age":    int,
                "height": float,
                "haz":    float,
                "status": str
            },
            ...
        ]
    """
    enriched = []

    for entry in predictions_raw:
        age    = entry["age"]
        height = entry["height"]

        # Hitung HAZ menggunakan who_service
        haz = compute_haz(
            height_cm=height,
            age_months=age,
            sex=sex,
            who_lms_df=who_lms_df
        )

        # Klasifikasikan status stunting
        status = classify_status(haz)

        # Bulatkan HAZ ke 2 desimal; tangani NaN
        haz_rounded = round(float(haz), 2) if not np.isnan(haz) else None

        enriched.append({
            "age":    age,
            "height": height,
            "haz":    haz_rounded,
            "status": status
        })

    return enriched
