import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from scipy.optimize import curve_fit
from services.who_service import get_who_median


# ==========================================================
# TRAIN FUNCTIONS
# ==========================================================

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


def train_exponential(X: np.ndarray, y: np.ndarray) -> dict | None:
    """
    Melatih model Exponential Regression: y = a * exp(b * x).
    Diimplementasikan dengan regresi linier pada (X, ln(y)).
    Disediakan untuk keperluan perbandingan akademis.
    """
    try:
        if np.any(y <= 0):
            return None

        y_log = np.log(y)
        model = LinearRegression()
        model.fit(X, y_log)

        class ExponentialPredictor:
            def __init__(self, lin_model):
                self.lin_model = lin_model

            def predict(self, X_input: np.ndarray) -> np.ndarray:
                log_pred = self.lin_model.predict(X_input)
                return np.exp(log_pred)

        predictor = ExponentialPredictor(model)
        return {
            "name": "Exponential Regression",
            "model": predictor,
            "type": "exponential"
        }
    except Exception:
        return None


# ==========================================================
# GAUSSIAN PROCESS REGRESSION + WHO PRIOR
# ==========================================================

class GPRWHOPredictor:
    """
    Wrapper untuk GPR + WHO Prior dengan Blending Fleksibel.

    Menyimpan model GPR yang dilatih pada deviasi WHO, serta model tren empiris
    anak untuk fleksibilitas yang lebih alami (tidak kaku/sensitif berlebihan ke WHO).
    """
    def __init__(self, gpr_model, sex: str, who_lms_df, emp_model=None, emp_weight: float = 0.20):
        self.gpr_model   = gpr_model
        self.sex         = sex
        self.who_lms_df  = who_lms_df
        self.emp_model   = emp_model
        self.emp_weight  = emp_weight

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([X, np.sqrt(X + 1.0)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        ages = X.flatten()
        who_medians = np.array([
            get_who_median(int(round(a)), self.sex, self.who_lms_df)
            for a in ages
        ])
        dev_pred = self.gpr_model.predict(X)
        who_pred = dev_pred + who_medians

        if self.emp_model is not None:
            X_feat   = self._transform_features(X)
            emp_pred = self.emp_model.predict(X_feat)
            w = self.emp_weight
            return (1.0 - w) * who_pred + w * emp_pred

        return who_pred

    def predict_with_std(self, X: np.ndarray):
        ages = X.flatten()
        who_medians = np.array([
            get_who_median(int(round(a)), self.sex, self.who_lms_df)
            for a in ages
        ])
        dev_pred, dev_std = self.gpr_model.predict(X, return_std=True)
        who_pred = dev_pred + who_medians

        if self.emp_model is not None:
            X_feat   = self._transform_features(X)
            emp_pred = self.emp_model.predict(X_feat)
            w = self.emp_weight
            final_pred = (1.0 - w) * who_pred + w * emp_pred
            return final_pred, dev_std

        return who_pred, dev_std


def train_gpr_who(
    X: np.ndarray,
    y: np.ndarray,
    sex: str,
    who_lms_df
) -> dict | None:
    """
    Melatih Gaussian Process Regression dengan WHO Median sebagai prior mean,
    dikombinasikan secara fleksibel dengan tren empiris anak.

    Keunggulan:
    - Tidak kaku/sensitif berlebihan ke garis WHO.
    - Menyesuaikan dengan laju pertumbuhan empiris anak.
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

        # Kernel offset + trend deviasi + noise
        kernel = (
            ConstantKernel(1.0, (0.1, 100.0))
            + DotProduct(sigma_0=1.0, sigma_0_bounds=(0.01, 10.0))
            + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
        )

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=False
        )
        gpr.fit(X, y_deviation)

        # Fit tren empiris pendukung
        X_feat = np.column_stack([X, np.sqrt(X + 1.0)])
        emp_model = LinearRegression()
        emp_model.fit(X_feat, y)

        predictor = GPRWHOPredictor(
            gpr_model=gpr,
            sex=sex,
            who_lms_df=who_lms_df,
            emp_model=emp_model,
            emp_weight=0.20
        )

        return {
            "name":  "GPR WHO Prior",
            "model": predictor,
            "type":  "gpr_who",
            "sex":   sex
        }

    except Exception:
        return None


# ==========================================================
# PURE DATA-DRIVEN GPR (TANPA WHO MEAN REVERSION PULL)
# ==========================================================

class PureGPRPredictor:
    """
    Wrapper untuk Pure Data-Driven GPR (Decelerated Baseline + GPR Residual).
    Memodelkan tren pertumbuhan anak secara murni dari data empiris tanpa WHO pull.
    """
    def __init__(self, linear_model, gpr_model):
        self.linear_model = linear_model
        self.gpr_model    = gpr_model

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([X, np.sqrt(X + 1.0)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._transform_features(X)
        trend  = self.linear_model.predict(X_feat)
        res    = self.gpr_model.predict(X)
        return trend + res

    def predict_with_std(self, X: np.ndarray):
        X_feat = self._transform_features(X)
        trend  = self.linear_model.predict(X_feat)
        res, res_std = self.gpr_model.predict(X, return_std=True)
        return trend + res, res_std


def train_pure_gpr(
    X: np.ndarray,
    y: np.ndarray
) -> dict | None:
    """
    Melatih Pure Data-Driven Gaussian Process Regression pada data historis anak.
    - Base trend: Linear + Sqrt Deceleration Trend dari data anak itu sendiri
    - Residual fit: GPR RBF Kernel untuk variasi halus
    - Bebas dari WHO Mean Reversion pull
    """
    try:
        X_feat  = np.column_stack([X, np.sqrt(X + 1.0)])
        linear  = LinearRegression()
        linear.fit(X_feat, y)
        y_trend = linear.predict(X_feat)
        y_res   = y - y_trend

        kernel = (
            ConstantKernel(1.0, (0.01, 10.0))
            * RBF(length_scale=4.0, length_scale_bounds=(1.0, 24.0))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 2.0))
        )

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        gpr.fit(X, y_res)

        predictor = PureGPRPredictor(linear_model=linear, gpr_model=gpr)

        return {
            "name":  "Pure Data-Driven GPR",
            "model": predictor,
            "type":  "pure_gpr"
        }
    except Exception:
        return None


# ==========================================================
# VALIDASI MASA LALU (PAST HELD-OUT VALIDATION / BACKTEST)
# ==========================================================

def validate_past_prediction(
    X: np.ndarray,
    y: np.ndarray,
    model_trainer_func
) -> dict:
    """
    Memvalidasi akurasi model dengan memprediksi data masa lalu (held-out past validation).

    Metodologi:
    - Sembunyikan 1 atau 2 titik historis terakhir.
    - Latih model hanya pada titik historis awal.
    - Prediksi titik masa lalu yang disembunyikan.
    - Hitung MAE dan tentukan kelayakan model sebelum memprediksi masa depan.
    """
    n = len(X)
    if n < 3:
        return {
            "validation_skipped": True,
            "reason": f"Data terlalu sedikit ({n} titik). Membutuhkan minimal 3 titik data."
        }

    k = 1
    X_train, X_held_out = X[:-k], X[-k:]
    y_train, y_held_out = y[:-k], y[-k:]

    try:
        model_dict = model_trainer_func(X_train, y_train)
        if model_dict is None:
            raise ValueError("Model fitting gagal pada subset validasi.")

        predictor = model_dict["model"]
        y_pred = predictor.predict(X_held_out)

        held_out_ages = [int(a[0]) for a in X_held_out]
        actual_vals = [round(float(v), 2) for v in y_held_out]
        pred_vals = [round(float(v), 2) for v in y_pred]

        errors = [round(abs(actual_vals[i] - pred_vals[i]), 2) for i in range(len(actual_vals))]
        mae = round(float(np.mean(errors)), 2)
        passed = mae <= 1.5

        return {
            "validation_skipped": False,
            "held_out_ages": held_out_ages,
            "actual_values": actual_vals,
            "predicted_values": pred_vals,
            "errors": errors,
            "mae": mae,
            "validation_passed": passed,
            "status": "Valid — Prediksi masa lalu terbukti akurat" if passed else "Perhatian — Margin error masa lalu cukup tinggi"
        }
    except Exception as e:
        return {
            "validation_skipped": True,
            "reason": f"Gagal memproses validasi masa lalu: {str(e)}"
        }


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


# ==========================================================
# TRAIN FUNCTIONS (SKLEARN & REGRESSION MODELS)
# ==========================================================

def train_bayesian_ridge(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Melatih model Bayesian Ridge Regression.
    Robust terhadap data sedikit karena regularisasi otomatis.
    """
    model = BayesianRidge()
    model.fit(X, y)
    return {
        "name": "Bayesian Ridge",
        "model": model,
        "type": "bayesian_ridge"
    }


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

                elif m_type == "linear":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)[0]

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
