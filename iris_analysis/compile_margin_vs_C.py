import os
import sys
import math
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.isdir(RESULTS_DIR):
    print('Results folder not found:', RESULTS_DIR)
    sys.exit(1)

candidates = [
    os.path.join(RESULTS_DIR, 'iris_pairwise_C_study_summary.csv'),
    os.path.join(RESULTS_DIR, 'iris_pairwise_C_study.csv'),
    os.path.join(RESULTS_DIR, 'iris_pairwise_C_study.csv')
]
found = None
for p in candidates:
    if os.path.exists(p):
        found = p
        break
if found is None:
    files = glob.glob(os.path.join(RESULTS_DIR, '*.csv'))
    if not files:
        print('No CSV files found in', RESULTS_DIR)
        sys.exit(1)
    for f in files:
        try:
            df_try = pd.read_csv(f, nrows=5)
            if 'pair' in df_try.columns and 'C' in df_try.columns:
                found = f
                break
        except Exception:
            continue
    if found is None:
        parts = []
        for f in files:
            try:
                parts.append(pd.read_csv(f))
            except Exception:
                pass
        if not parts:
            print('No readable CSV files in', RESULTS_DIR)
            sys.exit(1)
        df = pd.concat(parts, ignore_index=True, sort=False)
else:
    df = pd.read_csv(found)

df.columns = [c.strip() for c in df.columns]
if 'C' not in df.columns:
    print('No C column found in data.')
    sys.exit(1)

df['C'] = pd.to_numeric(df['C'], errors='coerce')
if 'margin' in df.columns:
    df['margin'] = pd.to_numeric(df['margin'], errors='coerce')
else:
    df['margin'] = np.nan
if 'n_support' in df.columns:
    df['n_support'] = pd.to_numeric(df['n_support'], errors='coerce')
else:
    df['n_support'] = np.nan

trend_map = {}
for pair, g in df.groupby('pair'):
    sub = g.dropna(subset=['C','margin'])
    if len(sub) >= 2:
        xs = np.log10(sub['C'].values)
        ys = sub['margin'].values
        try:
            slope = np.polyfit(xs, ys, 1)[0]
        except Exception:
            slope = float('nan')
        trend_map[pair] = slope
    else:
        trend_map[pair] = float('nan')

explanations = []
for idx, row in df.iterrows():
    pair = row.get('pair', 'unknown')
    C = row.get('C', float('nan'))
    margin = row.get('margin', float('nan'))
    n_support = row.get('n_support', float('nan'))
    slope = trend_map.get(pair, float('nan'))

    if not math.isnan(slope):
        if slope < -1e-6:
            trend = 'margin decreases as C increases (expected)'
        elif slope > 1e-6:
            trend = 'margin increases as C increases (unexpected)'
        else:
            trend = 'margin roughly stable across C'
    else:
        trend = 'insufficient data for trend'

    expl = f'Pair {pair}: C={C}. margin={margin if not math.isnan(margin) else "NA"}; supports={int(n_support) if not math.isnan(n_support) else "NA"}. Trend: {trend}. Interpretation: As C (regularization strength) increases the model penalizes slack less and fits tighter; margins typically shrink and number of support vectors often decreases.'
    explanations.append(expl)

df['margin_trend_slope'] = df['pair'].map(trend_map)
df['explanation_auto'] = explanations

out_csv = os.path.join(RESULTS_DIR, 'iris_margin_C_compilation.csv')
df.to_csv(out_csv, index=False)
out_json = os.path.join(RESULTS_DIR, 'iris_margin_C_compilation.json')
df.to_json(out_json, orient='records', indent=2)
print('Wrote:', out_csv, out_json)
