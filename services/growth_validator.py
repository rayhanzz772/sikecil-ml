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
    Menambahkan field 'growth_velocity', 'expected_velocity', 'velocity_ratio',
    serta indikator 'growth_warning' ke hasil prediksi.
    """
    if not history or not predictions:
        return predictions, {"has_warning": False, "message": "Data tidak cukup."}
        
    sorted_history = sorted(history, key=lambda x: x["age"])
    last_hist = sorted_history[-1]
    
    all_points = [last_hist] + predictions
    enriched = []
    
    slow_count = 0
    total_months = len(predictions)

    for i, pred in enumerate(predictions):
        prev = all_points[i]
        
        diff = pred["height"] - prev["height"]
        expected = get_expected_growth(prev["age"], pred["age"], sex, who_lms_df)
        
        ratio = round(float(diff / expected), 2) if expected > 0 else 1.0
        
        # Deteksi perlambatan per bulan (< 70% ekspektasi WHO)
        if ratio < 0.70:
            warning_status = "Pertumbuhan Melambat"
            slow_count += 1
        elif ratio < 0.40:
            warning_status = "Pertumbuhan Sangat Lambat (Faltering)"
            slow_count += 1
        else:
            warning_status = "Normal"

        new_pred = dict(pred)
        new_pred["growth_velocity"] = round(float(diff), 2)
        new_pred["expected_velocity"] = round(float(expected), 2)
        new_pred["velocity_ratio"] = ratio
        new_pred["growth_warning"] = warning_status
        
        enriched.append(new_pred)

    # Buat summary warning keseluruhan untuk sistem
    has_warning = (slow_count > 0)
    if slow_count >= total_months * 0.5:
        summary_type = "GROWTH_FALTERING"
        summary_msg  = (
            "PERINGATAN KRITIS: Pertumbuhan anak diprediksi melambat signifikan dari standar WHO "
            f"({slow_count} dari {total_months} bulan di bawah 70% ekspektasi). Berisiko Growth Faltering/Stunting."
        )
    elif slow_count > 0:
        summary_type = "SLOW_GROWTH_WARNING"
        summary_msg  = (
            f"PERINGATAN: Terdeteksi perlambatan pertumbuhan pada {slow_count} bulan prediksi. "
            "Disarankan pemantauan nutrisi dan konsultasi ke Posyandu/Faskes."
        )
    else:
        summary_type = "NORMAL_GROWTH"
        summary_msg  = "Pertumbuhan anak berjalan optimal sesuai trajektori biologis standar."

    warning_summary = {
        "has_warning": has_warning,
        "warning_type": summary_type,
        "slow_months_count": slow_count,
        "message": summary_msg
    }
        
    return enriched, warning_summary
