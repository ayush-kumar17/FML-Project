"""
Aggregate Iris SVM results and produce explanation CSV
- Reads iris_analysis/results/iris_pairwise_C_study.csv (preferred). If not present,
  it will attempt to read any pairwise CSVs in the results folder and assemble a table.
- Computes per-row comparison between soft and hard margins/support vectors.
- Computes margin trend vs log(C) per pair and attaches it to each row.
- Writes iris_analysis/results/iris_explained_summary.csv
"""
import os
import sys
import math
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.isdir(RESULTS_DIR):
    print('Results directory not found:', RESULTS_DIR)
    sys.exit(1)

primary = os.path.join(RESULTS_DIR, 'iris_pairwise_C_study.csv')

if os.path.exists(primary):
    df = pd.read_csv(primary)
else:
    # try to assemble from any pairwise files
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    if not files:
        print('No CSV results found in', RESULTS_DIR)
        sys.exit(1)
    parts = []
    for f in files:
        p = os.path.join(RESULTS_DIR, f)
        try:
            part = pd.read_csv(p)
            parts.append(part)
        except Exception:
            continue
    if not parts:
        print('No readable CSV parts found in', RESULTS_DIR)
        sys.exit(1)
    df = pd.concat(parts, ignore_index=True, sort=False)

# normalize column names
cols = [c.strip() for c in df.columns]
df.columns = cols

# Ensure types
if 'C' in df.columns:
    df['C'] = pd.to_numeric(df['C'], errors='coerce')
else:
    df['C'] = np.nan

# margins and supports
for c in ('margin','n_support','hard_margin','hard_n_support'):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    else:
        df[c] = np.nan

# Group-level margin trend: slope of margin vs log10(C)
trend_map = {}
for pair, g in df.groupby('pair'):
    sub = g.dropna(subset=['C','margin'])
    if len(sub) >= 2:
        xs = np.log10(sub['C'].values)
        ys = sub['margin'].values
        # ignore flat margins
        try:
            slope = np.polyfit(xs, ys, 1)[0]
        except Exception:
            slope = 0.0
        if slope < -1e-6:
            trend = 'decreasing'
        elif slope > 1e-6:
            trend = 'increasing'
        else:
            trend = 'flat'
    else:
        trend = 'insufficient_data'
    trend_map[pair] = trend

# Build explanation column per row
explanations = []
for idx, row in df.iterrows():
    pair = row.get('pair', 'unknown')
    C = row.get('C', float('nan'))
    margin = row.get('margin', float('nan'))
    n_support = row.get('n_support', float('nan'))
    hard_feasible = bool(row.get('hard_feasible', False)) if 'hard_feasible' in row.index else False
    hard_margin = row.get('hard_margin', float('nan'))
    hard_n = row.get('hard_n_support', float('nan'))

    expl = []
    expl.append(f'Pair: {pair}. C={C}.')
    if not math.isnan(margin):
        expl.append(f'margin={margin:.4g} (2/||w||).')
    else:
        expl.append('margin=NA.')
    if not math.isnan(n_support):
        expl.append(f'support_vectors={int(n_support)}.')
    else:
        expl.append('support_vectors=NA.')

    if hard_feasible and not math.isnan(hard_margin):
        # compare
        if not math.isnan(margin):
            if margin > hard_margin:
                expl.append('Soft margin is wider than hard-margin reference.')
            elif margin < hard_margin:
                expl.append('Soft margin is narrower than hard-margin reference.')
            else:
                expl.append('Soft margin equals hard-margin reference.')
        if (not math.isnan(n_support)) and (not math.isnan(hard_n)):
            if n_support > hard_n:
                expl.append('Soft SVM uses more support vectors than hard reference.')
            elif n_support < hard_n:
                expl.append('Soft SVM uses fewer support vectors than hard reference.')
            else:
                expl.append('Support vector counts equal.')
    else:
        expl.append('Hard-margin reference not feasible on training split (data not linearly separable). Kernel or slack required.')

    # margin trend for the pair
    trend = trend_map.get(pair, 'insufficient_data')
    if trend == 'decreasing':
        expl.append('As C increases margin tends to decrease (models fit harder).')
    elif trend == 'increasing':
        expl.append('As C increases margin tends to increase (unusual - check data).')
    elif trend == 'flat':
        expl.append('Margin is roughly stable across C values.')
    else:
        expl.append('Insufficient data to infer margin trend across C.')

    # general interpretation
    expl.append('Interpretation: Soft-margin SVM typically uses more support vectors and yields a wider/smoother decision boundary compared to a strict hard-margin solution when hard-margin exists.')

    explanations.append(' '.join(expl))

df['explanation_auto'] = explanations

out_path = os.path.join(RESULTS_DIR, 'iris_explained_summary.csv')
df.to_csv(out_path, index=False)
print('Wrote explanation CSV:', out_path)
