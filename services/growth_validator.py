from services.who_service import get_expected_growth

def is_growth_realistic(
    pred_heights_with_ages,
    sex,
    who_lms_df,
    max_velocity_ratio=2.5
):
    """
    Memvalidasi apakah lintasan prediksi pertumbuhan realistis
    dibandingkan dengan ekspektasi WHO growth velocity.
    
    pred_heights_with_ages: list of dict 
        Harus mencakup titik terakhir dari data asli di awal list agar
        bisa mengecek delta pertama ke titik prediksi.
    """
    if len(pred_heights_with_ages) < 2:
        return True

    for i in range(1, len(pred_heights_with_ages)):
        prev = pred_heights_with_ages[i - 1]
        curr = pred_heights_with_ages[i]
        
        age1 = prev["age"]
        age2 = curr["age"]
        h1 = prev["height"]
        h2 = curr["height"]
        
        diff = h2 - h1
        
        # 1. Tinggi tidak boleh turun (biologis mustahil)
        # Margin 0.1cm untuk rounding errors.
        if diff < -0.1:
            return False
            
        # 2. Bandingkan dengan WHO velocity
        expected_diff = get_expected_growth(age1, age2, sex, who_lms_df)
        
        if expected_diff > 0:
            ratio = diff / expected_diff
            
            # Jika kenaikannya lebih dari 1cm per bulan DAN lebih dari max ratio
            # maka dianggap terlalu eksponensial/tidak wajar.
            if diff > 1.0 and ratio > max_velocity_ratio:
                return False
                
    return True


def add_velocity_info(
    predictions,
    history,
    sex,
    who_lms_df
):
    """
    Menambahkan field 'growth_velocity', 'expected_velocity', 'velocity_ratio' 
    ke hasil prediksi.
    """
    if not history or not predictions:
        return predictions
        
    sorted_history = sorted(history, key=lambda x: x["age"])
    last_hist = sorted_history[-1]
    
    all_points = [last_hist] + predictions
    enriched = []
    
    for i, pred in enumerate(predictions):
        prev = all_points[i]
        
        diff = pred["height"] - prev["height"]
        expected = get_expected_growth(prev["age"], pred["age"], sex, who_lms_df)
        
        ratio = diff / expected if expected > 0 else 0
        
        new_pred = dict(pred)
        new_pred["growth_velocity"] = round(float(diff), 2)
        new_pred["expected_velocity"] = round(float(expected), 2)
        new_pred["velocity_ratio"] = round(float(ratio), 2)
        
        enriched.append(new_pred)
        
    return enriched
