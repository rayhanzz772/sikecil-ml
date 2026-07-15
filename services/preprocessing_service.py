import numpy as np


# ==========================================================
# BUILD FEATURE (Generic Internal Helper)
# ==========================================================

def _build_feature_generic(
    history: list[dict],
    value_key: str,
    value_label: str,
    min_val: float = 0.0,
    allow_zero: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper generik untuk mengekstrak (X_ages, y_values) dari history.

    Parameters
    ----------
    history     : list of dict — setiap entry harus punya 'age' dan value_key
    value_key   : str — nama key nilai target (misal: 'height', 'weight', 'head_circ')
    value_label : str — label untuk pesan error
    min_val     : float — batas minimum nilai yang valid
    allow_zero  : bool — jika True, nilai 0.0 diizinkan

    Returns
    -------
    X : np.ndarray shape (n, 1)
    y : np.ndarray shape (n,)
    """
    if not history:
        raise ValueError("History tidak boleh kosong.")
    if len(history) < 2:
        raise ValueError(
            f"Minimal 2 data poin diperlukan. Diterima: {len(history)} poin."
        )

    for i, entry in enumerate(history):
        if "age" not in entry:
            raise ValueError(f"Entry ke-{i} tidak memiliki key 'age'.")
        if value_key not in entry:
            raise ValueError(f"Entry ke-{i} tidak memiliki key '{value_key}'.")
        if not isinstance(entry["age"], (int, float)) or entry["age"] < 0:
            raise ValueError(
                f"Entry ke-{i}: 'age' harus berupa angka non-negatif. "
                f"Diterima: {entry['age']}"
            )
        val = entry[value_key]
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"Entry ke-{i}: '{value_key}' harus berupa angka. Diterima: {val}"
            )
        if allow_zero and val < min_val:
            raise ValueError(
                f"Entry ke-{i}: '{value_key}' harus >= {min_val}. Diterima: {val}"
            )
        elif not allow_zero and val <= min_val:
            raise ValueError(
                f"Entry ke-{i}: '{value_key}' ({value_label}) harus berupa "
                f"angka positif. Diterima: {val}"
            )

    sorted_history = sorted(history, key=lambda e: e["age"])
    ages = [e["age"] for e in sorted_history]
    if len(ages) != len(set(ages)):
        raise ValueError(f"Terdapat duplikat nilai 'age' di dalam history.")

    X = np.array([[e["age"]] for e in sorted_history], dtype=float)
    y = np.array([e[value_key] for e in sorted_history], dtype=float)
    return X, y


# ==========================================================
# BUILD FEATURE — TINGGI BADAN
# ==========================================================

def build_feature(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengubah riwayat pengukuran menjadi array fitur X dan target y
    untuk model prediksi tinggi badan.

    Parameters
    ----------
    history : list of dict
        Setiap dict berisi key 'age' (int, bulan) dan 'height' (float, cm).

    Returns
    -------
    X : np.ndarray, shape (n, 1)  — usia dalam bulan
    y : np.ndarray, shape (n,)    — tinggi badan dalam cm
    """
    return _build_feature_generic(history, "height", "tinggi badan")


# ==========================================================
# BUILD FEATURE — BERAT BADAN
# ==========================================================

def build_feature_weight(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengubah riwayat pengukuran menjadi array fitur X dan target y
    untuk model prediksi berat badan.

    Parameters
    ----------
    history : list of dict
        Setiap dict berisi key 'age' (int, bulan) dan 'weight' (float, kg).

    Returns
    -------
    X : np.ndarray, shape (n, 1)  — usia dalam bulan
    y : np.ndarray, shape (n,)    — berat badan dalam kg
    """
    return _build_feature_generic(history, "weight", "berat badan (kg)")


# ==========================================================
# BUILD FEATURE — LINGKAR KEPALA
# ==========================================================

def build_feature_hc(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Mengubah riwayat pengukuran menjadi array fitur X dan target y
    untuk model prediksi lingkar kepala.

    Parameters
    ----------
    history : list of dict
        Setiap dict berisi key 'age' (int, bulan) dan 'head_circ' (float, cm).

    Returns
    -------
    X : np.ndarray, shape (n, 1)  — usia dalam bulan
    y : np.ndarray, shape (n,)    — lingkar kepala dalam cm
    """
    return _build_feature_generic(history, "head_circ", "lingkar kepala (cm)")

    """
    Mengubah riwayat pengukuran menjadi array fitur X dan target y.

    Parameters
    ----------
    history : list of dict
        Setiap dict berisi key 'age' (int, bulan) dan 'height' (float, cm).
        Contoh:
            [
                {"age": 1, "height": 50.0},
                {"age": 2, "height": 52.0},
                ...
            ]

    Returns
    -------
    X : np.ndarray, shape (n, 1)
        Array usia dalam bulan sebagai fitur tunggal.
    y : np.ndarray, shape (n,)
        Array tinggi badan dalam cm sebagai target.

    Raises
    ------
    ValueError
        Jika history kosong, kurang dari 3 titik, atau ada key yang hilang.
    """

    if not history:
        raise ValueError("History tidak boleh kosong.")

    if len(history) < 2:
        raise ValueError(
            f"Minimal 2 data poin diperlukan untuk pelatihan model. "
            f"Diterima: {len(history)} poin."
        )

    # Validasi setiap entry
    for i, entry in enumerate(history):
        if "age" not in entry:
            raise ValueError(f"Entry ke-{i} tidak memiliki key 'age'.")
        if "height" not in entry:
            raise ValueError(f"Entry ke-{i} tidak memiliki key 'height'.")
        if not isinstance(entry["age"], (int, float)) or entry["age"] < 0:
            raise ValueError(
                f"Entry ke-{i}: 'age' harus berupa angka non-negatif. "
                f"Diterima: {entry['age']}"
            )
        if not isinstance(entry["height"], (int, float)) or entry["height"] <= 0:
            raise ValueError(
                f"Entry ke-{i}: 'height' harus berupa angka positif. "
                f"Diterima: {entry['height']}"
            )

    # Sort ascending berdasarkan usia
    sorted_history = sorted(history, key=lambda e: e["age"])

    # Cek duplikat usia
    ages = [e["age"] for e in sorted_history]
    if len(ages) != len(set(ages)):
        raise ValueError("Terdapat duplikat nilai 'age' di dalam history.")

    # Ekstrak fitur
    X = np.array([[e["age"]] for e in sorted_history], dtype=float)  # shape (n, 1)
    y = np.array([e["height"] for e in sorted_history], dtype=float)  # shape (n,)

    return X, y
