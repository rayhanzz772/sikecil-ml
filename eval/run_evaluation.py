import sys
import os
import json
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.who_service import load_who_lms
from services.preprocessing_service import build_feature
from services.model_service import (
    train_bayesian_ridge,
    train_von_bertalanffy,
    train_gompertz,
    evaluate_models,
    get_sorted_models
)
from services.haz_predictor import train_haz_predictor
from services.prediction_service import recursive_predict, build_prediction
from services.growth_validator import is_growth_realistic

def run_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, 'simulation_cases.json')
    who_csv_path = os.path.join(base_dir, '..', 'data', 'who_lms.csv')
    
    with open(cases_path, 'r') as f:
        cases = json.load(f)
        
    who_lms_df = load_who_lms(who_csv_path)
    
    results = []
    test_data_details = []
    
    for case in cases:
        case_id = case['case_id']
        mode = case['mode']
        sex = case['sex']
        history = case['history']
        ground_truth = case['ground_truth']
        horizon = len(ground_truth)
        last_age = history[-1]['age']
        
        preds_raw = []
        selected_model = ""
        
        if mode == "early_months":
            haz_model_dict = train_haz_predictor(history, sex, who_lms_df)
            preds = recursive_predict(haz_model_dict, last_age, horizon)
            preds_raw = [p['height'] for p in preds]
            selected_model = haz_model_dict["name"]
        else:
            X, y = build_feature(history)
            trained_models = [
                train_bayesian_ridge(X, y),
                train_von_bertalanffy(X, y),
                train_gompertz(X, y)
            ]
            metrics = evaluate_models(trained_models, X, y)
            sorted_models = get_sorted_models(trained_models, metrics)
            best_model = None
            
            for m in sorted_models:
                try:
                    preds = recursive_predict(m, last_age, horizon)
                    prediction = build_prediction(preds, sex, who_lms_df)
                    
                    # Cek dengan growth_validator
                    if is_growth_realistic(prediction, sex, who_lms_df):
                        best_model = m
                        preds_raw = [p['height'] for p in preds]
                        selected_model = m['name']
                        break
                except Exception:
                    continue
                    
            if best_model is None:
                best_model = next((m for m in sorted_models if m["type"] == "bayesian_ridge"), sorted_models[0])
                preds = recursive_predict(best_model, last_age, horizon)
                preds_raw = [p['height'] for p in preds]
                selected_model = best_model['name']
                
        # Calculate metrics
        y_true = [p['height'] for p in ground_truth]
        y_pred = preds_raw
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate R2 manually or with sklearn
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = 0.0
            
        formatted_preds = [
            {"age": gt["age"], "height": round(p, 2)}
            for gt, p in zip(ground_truth, y_pred)
        ]

        results.append({
            "case_id": case_id,
            "mode": mode,
            "label": case['label'],
            "sex": sex,
            "model": selected_model,
            "history": history,
            "ground_truth": ground_truth,
            "predictions": formatted_preds,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4)
        })

        for gt, p in zip(ground_truth, y_pred):
            test_data_details.append({
                "case_id": case_id,
                "mode": mode,
                "label": case['label'],
                "sex": sex,
                "model": selected_model,
                "target_age": gt['age'],
                "ground_truth_height": round(gt['height'], 2),
                "predicted_height": round(p, 2),
                "abs_error": round(abs(gt['height'] - p), 4)
            })
        
    res_path = os.path.join(base_dir, 'evaluation_results.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    df_details = pd.DataFrame(test_data_details)
    csv_path = os.path.join(base_dir, 'test_dataset_predictions.csv')
    df_details.to_csv(csv_path, index=False)

    print(f"Evaluated {len(results)} cases. Saved test data to evaluation_results.json and test_dataset_predictions.csv.")

if __name__ == '__main__':
    run_evaluation()
