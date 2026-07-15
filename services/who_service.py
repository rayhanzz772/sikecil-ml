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


# ==========================================================
# LOAD WHO WAZ & HCAZ
# ==========================================================

def load_who_waz(csv_path):
    """Load WHO Weight-for-Age LMS table dari file CSV."""
    return pd.read_csv(csv_path)


def load_who_hcaz(csv_path):
    """Load WHO Head Circumference-for-Age LMS table dari file CSV."""
    return pd.read_csv(csv_path)


# ==========================================================
# WAZ (Weight-for-Age Z-score)
# ==========================================================

def compute_waz(weight_kg, age_months, sex, who_waz_df) -> float:
    """
    Menghitung Weight-for-Age Z-score (WAZ) menggunakan metode LMS WHO.

    Parameters
    ----------
    weight_kg   : float — berat badan dalam kg
    age_months  : int   — usia dalam bulan (0-60)
    sex         : str   — "L" atau "P"
    who_waz_df  : DataFrame — tabel WHO WAZ LMS

    Returns
    -------
    float — nilai WAZ (z-score berat badan)
    """
    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_waz_df[
        (who_waz_df["age"] == age) &
        (who_waz_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    L = float(row["L"].iloc[0])
    M = float(row["M"].iloc[0])
    S = float(row["S"].iloc[0])

    if abs(L) < 1e-8:
        waz = np.log(weight_kg / M) / S
    else:
        waz = (((weight_kg / M) ** L) - 1) / (L * S)

    return float(waz)


# ==========================================================
# HCAZ (Head Circumference-for-Age Z-score)
# ==========================================================

def compute_hcaz(hc_cm, age_months, sex, who_hcaz_df) -> float:
    """
    Menghitung Head Circumference-for-Age Z-score (HCAZ) menggunakan metode LMS WHO.

    Parameters
    ----------
    hc_cm       : float — lingkar kepala dalam cm
    age_months  : int   — usia dalam bulan (0-60)
    sex         : str   — "L" atau "P"
    who_hcaz_df : DataFrame — tabel WHO HCAZ LMS

    Returns
    -------
    float — nilai HCAZ (z-score lingkar kepala)
    """
    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_hcaz_df[
        (who_hcaz_df["age"] == age) &
        (who_hcaz_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    L = float(row["L"].iloc[0])
    M = float(row["M"].iloc[0])
    S = float(row["S"].iloc[0])

    if abs(L) < 1e-8:
        hcaz = np.log(hc_cm / M) / S
    else:
        hcaz = (((hc_cm / M) ** L) - 1) / (L * S)

    return float(hcaz)


# ==========================================================
# CLASSIFY STATUS — Z-SCORE UNIVERSAL
# ==========================================================

def classify_zscore_status(z_score: float, indicator: str = "height") -> str:
    """
    Mengklasifikasikan status pertumbuhan berdasarkan nilai Z-score.

    Parameters
    ----------
    z_score   : float  — nilai z-score (HAZ / WAZ / HCAZ)
    indicator : str    — "height" / "weight" / "head_circ"

    Returns
    -------
    str — label status pertumbuhan
    """
    if np.isnan(z_score):
        return "Tidak Diketahui"

    if indicator == "weight":
        if z_score < -3:
            return "Gizi Buruk"
        elif z_score < -2:
            return "Gizi Kurang"
        elif z_score <= 2:
            return "Gizi Baik"
        else:
            return "Gizi Lebih"

    elif indicator == "head_circ":
        if z_score < -3:
            return "Mikrosefali Berat"
        elif z_score < -2:
            return "Mikrosefali"
        elif z_score <= 2:
            return "Normal"
        else:
            return "Makrosefali"

    else:  # height / HAZ default
        if z_score < -3:
            return "Stunting Berat"
        elif z_score < -2:
            return "Stunting"
        elif z_score <= 2:
            return "Normal"
        else:
            return "Tinggi"


# ==========================================================
# WHO MEDIAN HELPERS — BERAT BADAN & LINGKAR KEPALA
# ==========================================================

def get_who_weight_median(age_months: int, sex: str, who_waz_df) -> float:
    """
    Mengembalikan nilai median berat badan (M) dari tabel WHO WAZ
    untuk usia dan jenis kelamin tertentu.
    Digunakan sebagai prior mean untuk GPR berat badan.
    """
    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_waz_df[
        (who_waz_df["age"] == age) &
        (who_waz_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    return float(row["M"].iloc[0])


def get_who_hc_median(age_months: int, sex: str, who_hcaz_df) -> float:
    """
    Mengembalikan nilai median lingkar kepala (M) dari tabel WHO HCAZ
    untuk usia dan jenis kelamin tertentu.
    Digunakan sebagai prior mean untuk GPR lingkar kepala.
    """
    age = int(round(age_months))
    age = max(0, min(60, age))

    row = who_hcaz_df[
        (who_hcaz_df["age"] == age) &
        (who_hcaz_df["sex"] == sex)
    ]

    if row.empty:
        return np.nan

    return float(row["M"].iloc[0])