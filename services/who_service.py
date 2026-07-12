import pandas as pd
import numpy as np


# ==========================================================
# LOAD WHO LMS TABLE
# ==========================================================

def load_who_lms(csv_path):
    """
    Load WHO LMS table dari file CSV.

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    pandas.DataFrame
    """

    return pd.read_csv(csv_path)


# ==========================================================
# HAZ (Height-for-Age Z-score)
# ==========================================================

def compute_haz(
    height_cm,
    age_months,
    sex,
    who_lms_df
):
    """
    Menghitung Height-for-Age Z-score (HAZ)
    berdasarkan metode LMS WHO.

    Parameters
    ----------
    height_cm : float
    age_months : int
    sex : 'L' atau 'P'
    who_lms_df : DataFrame
    """

    if pd.isna(height_cm):
        return np.nan

    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_lms_df[
        (who_lms_df["age"] == age) &
        (who_lms_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    L = float(row["L"].iloc[0])
    M = float(row["M"].iloc[0])
    S = float(row["S"].iloc[0])

    if abs(L) < 1e-8:

        haz = np.log(height_cm / M) / S

    else:

        haz = (((height_cm / M) ** L) - 1) / (L * S)

    return float(haz)


# ==========================================================
# HAZ -> HEIGHT
# ==========================================================

def haz_to_height(
    haz,
    age_months,
    sex,
    who_lms_df
):
    """
    Mengubah HAZ menjadi tinggi badan.
    """

    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_lms_df[
        (who_lms_df["age"] == age) &
        (who_lms_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    L = float(row["L"].iloc[0])
    M = float(row["M"].iloc[0])
    S = float(row["S"].iloc[0])

    if abs(L) < 1e-8:

        height = M * np.exp(haz * S)

    else:

        height = M * ((1 + L * S * haz) ** (1 / L))

    return float(height)


# ==========================================================
# HAZ -> STATUS
# ==========================================================

def classify_status(haz):
    """
    Klasifikasi status stunting berdasarkan HAZ WHO.

    Return
    ------
    Severely Stunted : HAZ < -3
    Stunted          : -3 <= HAZ < -2
    At Risk          : -2 <= HAZ < -1
    Normal           : HAZ >= -1
    """

    if pd.isna(haz):
        return "Unknown"

    if haz < -3:
        return "Severely Stunted"

    elif haz < -2:
        return "Stunted"

    elif haz < -1:
        return "At Risk"

    else:
        return "Normal"


# ==========================================================
# HEIGHT -> STATUS
# ==========================================================

def get_growth_status(
    height_cm,
    age_months,
    sex,
    who_lms_df
):
    """
    Helper function.

    Input:
        Height

    Output:
        {
            "haz": ...,
            "status": ...
        }
    """

    haz = compute_haz(
        height_cm=height_cm,
        age_months=age_months,
        sex=sex,
        who_lms_df=who_lms_df
    )

    return {

        "haz": round(float(haz), 2),

        "status": classify_status(haz)

    }


# ==========================================================
# EXPECTED GROWTH VELOCITY
# ==========================================================

def get_expected_growth(
    age1_months,
    age2_months,
    sex,
    who_lms_df
):
    """
    Menghitung expected growth velocity (cm) antara dua usia
    berdasarkan nilai Median (M) dari tabel WHO LMS.
    """
    age1 = max(0, min(60, int(round(age1_months))))
    age2 = max(0, min(60, int(round(age2_months))))
    
    if age1 == age2:
        return 0.0
        
    row1 = who_lms_df[(who_lms_df["age"] == age1) & (who_lms_df["sex"] == sex)]
    row2 = who_lms_df[(who_lms_df["age"] == age2) & (who_lms_df["sex"] == sex)]
    
    if row1.empty or row2.empty:
        return 0.0
        
    m1 = float(row1["M"].iloc[0])
    m2 = float(row2["M"].iloc[0])
    
    return max(0.0, m2 - m1)


# ==========================================================
# HAZ SERIES CONVERSION
# ==========================================================

def get_haz_series(
    history,
    sex,
    who_lms_df
):
    """
    Mengubah history tinggi badan menjadi time-series HAZ.
    
    Parameters
    ----------
    history : list of dict
        [{"age": int, "height": float}, ...]
        
    Returns
    -------
    list of dict
        [{"age": int, "haz": float}, ...]
    """
    haz_series = []
    
    for entry in history:
        age = entry["age"]
        height = entry["height"]
        
        haz = compute_haz(
            height_cm=height,
            age_months=age,
            sex=sex,
            who_lms_df=who_lms_df
        )
        
        haz_series.append({
            "age": age,
            "haz": haz if not np.isnan(haz) else 0.0
        })
        
    return haz_series


# ==========================================================
# WHO MEDIAN HELPER (untuk GPR Prior)
# ==========================================================

def get_who_median(age_months: int, sex: str, who_lms_df) -> float:
    """
    Mengembalikan nilai median tinggi badan (M) dari tabel WHO LMS
    untuk usia dan jenis kelamin tertentu.

    Ini adalah fungsi prior mean untuk Gaussian Process Regression.
    Nilai M = tinggi badan rata-rata anak sehat di populasi WHO.

    Parameters
    ----------
    age_months : int   - usia dalam bulan (0-60)
    sex        : str   - "L" atau "P"
    who_lms_df : DataFrame

    Returns
    -------
    float — nilai median tinggi badan (cm)
    """
    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_lms_df[
        (who_lms_df["age"] == age) &
        (who_lms_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    return float(row["M"].iloc[0])