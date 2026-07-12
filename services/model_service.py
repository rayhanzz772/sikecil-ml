import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import curve_fit
from services.who_service import get_who_median


# ==========================================================
# GOMPERTZ WRAPPER
# ==========================================================

def _gompertz_func(x, A, B, C):
    """
    Fungsi Gompertz: f(x) = A * exp(-B * exp(-C * x))

    Parameters
    ----------
    x  : array-like — usia dalam bulan
    A  : float      — asimptot atas (tinggi maksimal teoritis)
    B  : float      — parameter geser horizontal
    C  : float      — laju pertumbuhan
    """
    return A * np.exp(-B * np.exp(-C * x))


class GompertzPredictor:
    """
    Wrapper Gompertz Growth Model agar API-nya seragam dengan sklearn.

    Attributes
    ----------
    params_ : tuple (A, B, C)
        Parameter hasil fitting curve_fit.
    """

    def __init__(self, params: tuple):
        self.params_ = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi tinggi badan menggunakan model Gompertz.

        Parameters
        ----------
        X : np.ndarray, shape (n, 1)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        x_flat = X.flatten()
        A, B, C = self.params_
        return _gompertz_func(x_flat, A, B, C)


# ==========================================================
# VON BERTALANFFY WRAPPER
# ==========================================================

def _von_bertalanffy_func(x, A, K, B):
    """
    Fungsi Von Bertalanffy: f(x) = A - B * exp(-K * x)

    Parameters
    ----------
    x : array-like — usia dalam bulan
    A : float      — asimptot atas (tinggi maksimal)
    K : float      — laju pertumbuhan
    B : float      — selisih antara tinggi maksimal dan tinggi awal
    """
    return A - B * np.exp(-K * x)


class VonBertalanffyPredictor:
    """
    Wrapper Von Bertalanffy Growth Model agar API-nya seragam dengan sklearn.
    """
    def __init__(self, params: tuple):
        self.params_ = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_flat = X.flatten()
        A, K, B = self.params_
        return _von_bertalanffy_func(x_flat, A, K, B)


# ==========================================================
# TRAIN FUNCTIONS
# ==========================================================

def train_bayesian_ridge(X: np.ndarray, y: np.ndarray) -> dict:

    model = BayesianRidge()
    model.fit(X, y)

    return {
        "name": "Bayesian Ridge",
        "model": model,
        "type": "bayesian_ridge"
    }


def train_linear(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Melatih model Linear Regression.
    Disediakan untuk keperluan perbandingan akademis (/api/predict/v1).
    """
    model = LinearRegression()
    model.fit(X, y)
    return {
        "name": "Linear Regression",
        "model": model,
        "type": "linear"
    }


def train_polynomial(X: np.ndarray, y: np.ndarray, degree: int) -> dict | None:
    """
    Melatih model Polynomial Regression.
    Disediakan untuk keperluan perbandingan akademis (/api/predict/v1).
    Otomatis dilewati jika jumlah data tidak cukup.
    """
    n_samples = len(X)
    if n_samples <= degree + 1:
        return None

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear", LinearRegression())
    ])
    pipeline.fit(X, y)
    return {
        "name": f"Polynomial Degree {degree}",
        "model": pipeline,
        "type": "poly",
        "degree": degree
    }


# ==========================================================
# GAUSSIAN PROCESS REGRESSION + WHO PRIOR
# ==========================================================

class GPRWHOPredictor:
    """
    Wrapper untuk GPR + WHO Prior agar API-nya seragam dengan model lain.

    Menyimpan model GPR yang sudah dilatih pada ruang deviasi WHO,
    serta referensi sex dan who_lms_df untuk rekonstruksi prediksi akhir.
    """
    def __init__(self, gpr_model, sex: str, who_lms_df):
        self.gpr_model   = gpr_model
        self.sex         = sex
        self.who_lms_df  = who_lms_df

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi tinggi badan (bukan deviasi) untuk array usia X."""
        ages = X.flatten()
        who_medians = np.array([
            get_who_median(int(round(a)), self.sex, self.who_lms_df)
            for a in ages
        ])
        dev_pred = self.gpr_model.predict(X)
        return dev_pred + who_medians

    def predict_with_std(self, X: np.ndarray):
        """Prediksi tinggi + std (interval kepercayaan) untuk array usia X."""
        ages = X.flatten()
        who_medians = np.array([
            get_who_median(int(round(a)), self.sex, self.who_lms_df)
            for a in ages
        ])
        dev_pred, dev_std = self.gpr_model.predict(X, return_std=True)
        return dev_pred + who_medians, dev_std


def train_gpr_who(
    X: np.ndarray,
    y: np.ndarray,
    sex: str,
    who_lms_df
) -> dict | None:
    """
    Melatih Gaussian Process Regression dengan WHO Median sebagai prior mean.

    Strategi:
    - Hitung deviasi: y_dev = y_aktual - y_who_median
    - Latih GPR pada (X, y_dev)
    - Prediksi: gpr.predict(X_future) + who_median_future

    Kernel:
    - ConstantKernel * RBF : menangkap tren halus
    - WhiteKernel          : menangkap noise pengukuran

    Returns
    -------
    dict atau None jika fitting gagal
    """
    try:
        ages = X.flatten()

        # Ambil nilai median WHO untuk setiap usia latih
        who_medians = np.array([
            get_who_median(int(round(a)), sex, who_lms_df)
            for a in ages
        ])

        # Hitung deviasi individu terhadap WHO
        y_deviation = y - who_medians

        # Definisikan kernel: smooth trend + noise
        kernel = (
            ConstantKernel(1.0, (0.01, 10.0))
            * RBF(length_scale=4.0, length_scale_bounds=(1.0, 20.0))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 2.0))
        )

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        gpr.fit(X, y_deviation)

        predictor = GPRWHOPredictor(gpr_model=gpr, sex=sex, who_lms_df=who_lms_df)

        return {
            "name":  "GPR WHO Prior",
            "model": predictor,
            "type":  "gpr_who",
            "sex":   sex
        }

    except Exception:
        return None


def gpr_predict_with_who(
    gpr_who_dict: dict,
    last_age: int,
    horizon: int
) -> list[dict]:
    """
    Menghasilkan prediksi dari GPR + WHO Prior.
    Berbeda dari recursive_predict, fungsi ini juga mengembalikan
    uncertainty_band (interval kepercayaan 95%).

    Returns
    -------
    list of dict:
        [{"age": int, "height": float, "uncertainty_band": float}, ...]
    """
    predictor = gpr_who_dict["model"]
    results = []

    future_ages = np.array([[last_age + i] for i in range(1, horizon + 1)])
    heights, stds = predictor.predict_with_std(future_ages)

    for i, age_arr in enumerate(future_ages):
        age = int(age_arr[0])
        height = float(heights[i])
        height = max(height, 0.0)  # failsafe
        band = round(float(stds[i] * 1.96), 2)  # 95% confidence interval

        results.append({
            "age":              age,
            "height":           round(height, 2),
            "uncertainty_band": band
        })

    return results


def train_von_bertalanffy(X: np.ndarray, y: np.ndarray) -> dict | None:
    x_flat = X.flatten()

    # Estimasi parameter awal
    A_init = float(np.max(y)) * 1.5   
    K_init = 0.05
    B_init = A_init - float(np.min(y))

    p0 = [A_init, K_init, B_init]
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    try:
        params, _ = curve_fit(
            _von_bertalanffy_func,
            x_flat,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        von_obj = VonBertalanffyPredictor(params=tuple(params))

        return {
            "name": "Von Bertalanffy",
            "model": von_obj,
            "type": "von_bertalanffy"
        }

    except (RuntimeError, ValueError) as e:
        return None


def train_gompertz(X: np.ndarray, y: np.ndarray) -> dict | None:
    x_flat = X.flatten()

    # Estimasi parameter awal berdasarkan data
    A_init = float(np.max(y)) * 1.5   # asimptot ~ 1.5x tinggi maks
    B_init = 5.0                        # geser horizontal
    C_init = 0.1                        # laju pertumbuhan

    p0 = [A_init, B_init, C_init]

    # Bounds: A > 0, B > 0, C > 0
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    try:
        params, _ = curve_fit(
            _gompertz_func,
            x_flat,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        gompertz_obj = GompertzPredictor(params=tuple(params))

        return {
            "name": "Gompertz",
            "model": gompertz_obj,
            "type": "gompertz"
        }

    except (RuntimeError, ValueError) as e:
        # Konvergensi gagal — model dilewati
        return None


# ==========================================================
# EVALUATE MODELS
# ==========================================================

def evaluate_models(
    trained_models: list[dict],
    X: np.ndarray,
    y: np.ndarray
) -> dict:
    metrics = {}
    loo = LeaveOneOut()

    for m in trained_models:
        if m is None:
            continue

        name   = m["name"]
        m_type = m["type"]

        y_true = []
        y_pred = []
        loocv_failed = False

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                if m_type == "bayesian_ridge":
                    model = BayesianRidge()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)[0]

                elif m_type == "von_bertalanffy":
                    x_flat = X_train.flatten()
                    A_init = float(np.max(y_train)) * 1.5
                    K_init = 0.05
                    B_init = A_init - float(np.min(y_train))
                    p0 = [A_init, K_init, B_init]
                    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
                    
                    params, _ = curve_fit(
                        _von_bertalanffy_func,
                        x_flat,
                        y_train,
                        p0=p0,
                        bounds=bounds,
                        maxfev=10000
                    )
                    pred = _von_bertalanffy_func(X_test.flatten()[0], *params)

                elif m_type == "gompertz":
                    x_flat = X_train.flatten()
                    A_init = float(np.max(y_train)) * 1.5
                    p0 = [A_init, 5.0, 0.1]
                    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
                    
                    params, _ = curve_fit(
                        _gompertz_func,
                        x_flat,
                        y_train,
                        p0=p0,
                        bounds=bounds,
                        maxfev=10000
                    )
                    pred = _gompertz_func(X_test.flatten()[0], *params)

                elif m_type == "linear":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)[0]

                elif m_type == "poly":
                    degree = m["degree"]
                    if len(X_train) <= degree:
                        loocv_failed = True
                        break
                    pipeline = Pipeline([
                        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                        ("linear", LinearRegression())
                    ])
                    pipeline.fit(X_train, y_train)
                    pred = pipeline.predict(X_test)[0]

            except Exception:
                loocv_failed = True
                break

            y_true.append(y_test[0])
            y_pred.append(pred)

        if not loocv_failed and len(y_true) > 1:
            mae  = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2   = float(r2_score(y_true, y_pred))

            metrics[name] = {
                "mae":  round(mae,  6),
                "rmse": round(rmse, 6),
                "r2":   round(r2,   6)
            }

    return metrics


# ==========================================================
# GET SORTED MODELS
# ==========================================================

def get_sorted_models(
    trained_models: list[dict],
    metrics: dict
) -> list[dict]:
    # Hanya pertimbangkan model yang berhasil dilatih (ada di metrics)
    valid_models = [m for m in trained_models if m is not None and m["name"] in metrics]

    if not valid_models:
        raise ValueError(
            "Tidak ada model yang berhasil dilatih. "
            "Pastikan data historis cukup dan valid."
        )

    # Urutkan: RMSE ascending, lalu R² descending sebagai tie-breaker
    valid_models.sort(
        key=lambda m: (
            metrics[m["name"]]["rmse"],
            -metrics[m["name"]]["r2"]   # negatif agar min() memilih R² terbesar
        )
    )

    return valid_models
