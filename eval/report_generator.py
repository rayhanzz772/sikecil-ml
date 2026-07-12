import json
import pandas as pd

def generate_report():
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
        
    df = pd.DataFrame(results)
    
    # Split into early and normal modes
    df_early = df[df['mode'] == 'early_months'].copy()
    df_normal = df[df['mode'] == 'normal'].copy()
    
    report_lines = []
    report_lines.append("============================================================")
    report_lines.append("LAPORAN EVALUASI SISTEM PREDIKSI PERTUMBUHAN BALITA")
    report_lines.append("Metode: WHO-Simulation + LOOCV")
    report_lines.append("============================================================\n")
    
    def print_section(title, subset_df):
        report_lines.append(f"--- {title} ---")
        if len(subset_df) == 0:
            report_lines.append("Data tidak tersedia.\n")
            return
            
        header = f"{'Kasus ID':<20} | {'Kondisi':<20} | {'Model':<20} | {'MAE (cm)':<8} | {'RMSE':<8} | {'R2':<8}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        for _, row in subset_df.iterrows():
            line = f"{row['case_id']:<20} | {row['label']:<20} | {row['model']:<20} | {row['mae']:<8.3f} | {row['rmse']:<8.3f} | {row['r2']:<8.3f}"
            report_lines.append(line)
            
        report_lines.append("-" * len(header))
        avg_mae = subset_df['mae'].mean()
        avg_rmse = subset_df['rmse'].mean()
        avg_r2 = subset_df['r2'].mean()
        
        summary = f"{'RATA-RATA':<20} | {'Semua Kasus':<20} | {'':<20} | {avg_mae:<8.3f} | {avg_rmse:<8.3f} | {avg_r2:<8.3f}"
        report_lines.append(summary)
        report_lines.append("\n")

    print_section("SKENARIO 1: EARLY MONTHS (History: 4 bulan, Prediksi: 6 bulan)", df_early)
    print_section("SKENARIO 2: NORMAL MODE (History: 6 bulan, Prediksi: 6 bulan)", df_normal)
    
    # Generate model summary
    report_lines.append("--- RINGKASAN PERFORMA PER MODEL ---")
    model_groups = df.groupby('model').agg({
        'mae': 'mean',
        'rmse': 'mean',
        'r2': 'mean',
        'case_id': 'count'
    }).reset_index()
    
    m_header = f"{'Model':<20} | {'Jumlah Kasus':<12} | {'MAE (cm)':<8} | {'RMSE':<8} | {'R2':<8}"
    report_lines.append(m_header)
    report_lines.append("-" * len(m_header))
    for _, row in model_groups.iterrows():
        line = f"{row['model']:<20} | {int(row['case_id']):<12} | {row['mae']:<8.3f} | {row['rmse']:<8.3f} | {row['r2']:<8.3f}"
        report_lines.append(line)
        
    report_text = "\n".join(report_lines)
    
    with open('evaluation_report.txt', 'w') as f:
        f.write(report_text)
        
    print("Report generated at evaluation_report.txt")

if __name__ == '__main__':
    generate_report()
