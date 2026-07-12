import numpy as np
from sklearn.linear_model import BayesianRidge
from services.who_service import get_haz_series, haz_to_height

class HAZPredictorWrapper:
    """
    Wrapper model yang memprediksi dalam ruang HAZ lalu mengubahnya
    menjadi tinggi badan (cm) agar kompatibel dengan API model lain (memiliki method .predict).
    """
    def __init__(self, history, sex, who_lms_df, strategy="linear_trend"):
        self.history = sorted(history, key=lambda x: x["age"])
        self.sex = sex
        self.who_lms_df = who_lms_df
        self.strategy = strategy
        
        self.haz_series = get_haz_series(self.history, self.sex, self.who_lms_df)
        self.last_age = self.haz_series[-1]["age"]
        self.last_haz = self.haz_series[-1]["haz"]
        
        self._fit_strategy()

    def _fit_strategy(self):
        ages = np.array([e["age"] for e in self.haz_series]).reshape(-1, 1)
        hazs = np.array([e["haz"] for e in self.haz_series])
        n = len(self.haz_series)
        
        if self.strategy == "mean_reversion" or n < 3:
            # Dampened trend for n=2
            if n >= 2:
                self.trend = hazs[-1] - hazs[-2]
                age_diff = ages[-1][0] - ages[-2][0]
                if age_diff > 0:
                    self.trend /= age_diff
            else:
                self.trend = 0.0
                
        elif self.strategy == "linear_trend" or n == 3:
            self.model = BayesianRidge()
            self.model.fit(ages, hazs)
            self.trend_func = lambda x: self.model.predict(np.array([[x]]))[0]
            
        elif self.strategy == "weighted_trend":
            self.model = BayesianRidge()
            # Poin terbaru mendapat bobot lebih tinggi
            weights = np.array([1.5 ** i for i in range(n)])
            self.model.fit(ages, hazs, sample_weight=weights)
            self.trend_func = lambda x: self.model.predict(np.array([[x]]))[0]
            
    def _predict_haz(self, future_age):
        n = len(self.haz_series)
        if self.strategy == "mean_reversion" or n < 3:
            steps_ahead = future_age - self.last_age
            # Dampening the trend by halving it each step ahead
            dampened_trend = self.trend * (0.5 ** steps_ahead)
            # Add small mean reversion towards 0 (normal) if it's very extreme
            pred_haz = self.last_haz + dampened_trend * steps_ahead
        else:
            pred_haz = self.trend_func(future_age)
            
        # Clamp HAZ to biologically reasonable bounds
        pred_haz = max(-6.0, min(3.0, pred_haz))
        return pred_haz

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediksi tinggi badan (cm) untuk feature X (age).
        X = [[age1], [age2], ...]
        """
        heights = []
        for x_val in X:
            future_age = float(x_val[0])
            pred_haz = self._predict_haz(future_age)
            pred_height = haz_to_height(pred_haz, future_age, self.sex, self.who_lms_df)
            heights.append(pred_height)
            
        return np.array(heights)


def train_haz_predictor(history, sex, who_lms_df) -> dict | None:
    """
    Melatih model HAZ-Space Predictor yang otomatis memilih strategi berdasarkan n.
    
    Returns
    -------
    dict
        {
            "name": "HAZ Trend",
            "model": <HAZPredictorWrapper>,
            "type": "haz_trend"
        }
    """
    n = len(history)
    if n < 2:
        return None
        
    if n == 2:
        strategy = "mean_reversion"
        name = "HAZ Constant/Dampened Trend"
    elif n == 3:
        strategy = "linear_trend"
        name = "HAZ Linear Trend"
    else:
        strategy = "weighted_trend"
        name = "HAZ Weighted Trend"
        
    model = HAZPredictorWrapper(history, sex, who_lms_df, strategy=strategy)
    
    return {
        "name": name,
        "model": model,
        "type": "haz_trend",
        "strategy": strategy
    }
