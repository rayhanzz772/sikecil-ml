from services.who_service import (
    load_who_waz, load_who_hcaz,
    compute_waz, compute_hcaz,
    classify_zscore_status,
    get_who_weight_median, get_who_hc_median
)
from services.preprocessing_service import build_feature_weight, build_feature_hc

# Test WHO load
waz_df  = load_who_waz('data/who_waz.csv')
hcaz_df = load_who_hcaz('data/who_hcaz.csv')
print('WAZ rows:', len(waz_df), '| HCAZ rows:', len(hcaz_df))

# Test WAZ compute (bayi 6 bulan, 7.9 kg, Laki-laki = Normal)
waz  = compute_waz(7.9, 6, 'L', waz_df)
stat = classify_zscore_status(waz, 'weight')
print(f'WAZ L 6bln 7.9kg: {waz:.3f} -> {stat}')

# Test HCAZ compute (bayi 6 bulan, 43.3 cm, Laki-laki = Normal)
hcaz = compute_hcaz(43.3, 6, 'L', hcaz_df)
stat_hc = classify_zscore_status(hcaz, 'head_circ')
print(f'HCAZ L 6bln 43.3cm: {hcaz:.3f} -> {stat_hc}')

# Test WHO median
m_w  = get_who_weight_median(6, 'L', waz_df)
m_hc = get_who_hc_median(6, 'L', hcaz_df)
print(f'WHO Median Berat 6bln L: {m_w} kg | HC: {m_hc} cm')

# Test preprocessing
history = [
    {'age': 0, 'weight': 3.3},
    {'age': 1, 'weight': 4.5},
    {'age': 3, 'weight': 6.4},
]
X, y = build_feature_weight(history)
print('build_feature_weight OK:', X.flatten().tolist(), '->', y.tolist())

history_hc = [
    {'age': 0, 'head_circ': 34.5},
    {'age': 3, 'head_circ': 40.5},
    {'age': 6, 'head_circ': 43.3},
]
X2, y2 = build_feature_hc(history_hc)
print('build_feature_hc OK:', X2.flatten().tolist(), '->', y2.tolist())

print('ALL TESTS PASSED!')
