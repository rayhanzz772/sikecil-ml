import os
import json
import pandas as pd

def generate_cases():
    # Load WHO LMS data
    who_df = pd.read_csv('../data/who_lms.csv')
    
    scenarios = [
        {"name": "Normal", "z_score": 0.0},
        {"name": "Normal Atas", "z_score": 1.0},
        {"name": "At Risk", "z_score": -1.0},
        {"name": "Stunted", "z_score": -2.0},
        {"name": "Severely Stunted", "z_score": -3.0},
    ]
    
    sexes = ["L", "P"]
    cases = []
    
    # Helper to calculate height from Z-score using LMS formula
    def calc_height(l, m, s, z):
        if l == 0:
            return m * (2.71828 ** (s * z))
        else:
            return m * ((1 + l * s * z) ** (1 / l))
            
    for sex in sexes:
        sex_df = who_df[who_df['sex'] == sex].set_index('age')
        
        for sc in scenarios:
            z = sc['z_score']
            
            # We create two variations for each scenario:
            # 1. Early Months (4 data points: ages 0-3) to test HAZ Predictor
            # 2. Normal Mode (6 data points: ages 0-5) to test Bio-Mathematical models
            
            history_4 = []
            ground_truth_4 = []
            
            history_6 = []
            ground_truth_6 = []
            
            # Generate all heights from age 0 to 11
            heights = {}
            for age in range(12):
                row = sex_df.loc[age]
                h = calc_height(row['L'], row['M'], row['S'], z)
                heights[age] = round(h, 2)
                
            # Populate Case 1: Early Months (4 history points)
            for age in range(4):
                history_4.append({"age": age, "height": heights[age]})
            for age in range(4, 10):
                ground_truth_4.append({"age": age, "height": heights[age]})
                
            # Populate Case 2: Normal Mode (6 history points)
            for age in range(6):
                history_6.append({"age": age, "height": heights[age]})
            for age in range(6, 12):
                ground_truth_6.append({"age": age, "height": heights[age]})
                
            case_early = {
                "case_id": f"EARLY-{sex}-{sc['name'].replace(' ', '')}",
                "sex": sex,
                "target_z": z,
                "label": sc['name'],
                "mode": "early_months",
                "history": history_4,
                "ground_truth": ground_truth_4
            }
            
            case_normal = {
                "case_id": f"NORM-{sex}-{sc['name'].replace(' ', '')}",
                "sex": sex,
                "target_z": z,
                "label": sc['name'],
                "mode": "normal",
                "history": history_6,
                "ground_truth": ground_truth_6
            }
            
            cases.append(case_early)
            cases.append(case_normal)
            
    # Save to JSON
    with open('simulation_cases.json', 'w') as f:
        json.dump(cases, f, indent=4)
        
    print(f"Generated {len(cases)} simulation cases.")

if __name__ == '__main__':
    generate_cases()
