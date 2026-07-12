import numpy as np


# ==========================================================
# BUILD FEATURE
# ==========================================================

def build_feature(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
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
