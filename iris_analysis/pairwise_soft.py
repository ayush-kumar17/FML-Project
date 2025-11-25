"""
Pairwise Soft-Margin SVM analysis for Iris dataset
Creates per-pair decision-boundary plots and saves pairwise metrics (accuracy, precision, recall, f1), margins and support vector counts.
Outputs saved to iris_analysis/results/
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure parent project dir on path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from visualizations import SVMVisualizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_iris(csv_path, features=('PetalLengthCm', 'PetalWidthCm')):
    df = pd.read_csv(csv_path)
    df = df.copy()
    df['label'] = pd.Categorical(df['Species']).codes
    X = df[list(features)].values
    y = df['label'].values
    return X, y


def compute_margin_from_coef(coef):
    w = np.ravel(coef)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return None
    return 2.0 / w_norm


def run_pairwise_soft(csv_path, C=1.0, features=('PetalLengthCm', 'PetalWidthCm')):
    X, y = load_iris(csv_path, features=features)
    classes = np.unique(y)
    results = []

    for a, b in combinations(classes, 2):
        mask = np.logical_or(y == a, y == b)
        X_pair = X[mask]
        y_pair = y[mask]
        y_bin = np.where(y_pair == a, 0, 1)

        X_tr, X_te, y_tr, y_te = train_test_split(X_pair, y_bin, test_size=0.3, random_state=42, stratify=y_bin)

        model = SVC(kernel='linear', C=C)
        model.fit(X_tr, y_tr)
        margin = compute_margin_from_coef(model.coef_)
        sv = len(model.support_)

        y_pred = model.predict(X_te)
        metrics = {
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred, zero_division=0)),
            'recall': float(recall_score(y_te, y_pred, zero_division=0)),
            'f1': float(f1_score(y_te, y_pred, zero_division=0))
        }

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        SVMVisualizer.plot_decision_boundary(model, X_pair, y_bin, title=f'Soft Margin C={C} - {a} vs {b}', ax=ax)
        path = os.path.join(RESULTS_DIR, f'soft_pair_{a}_vs_{b}_C_{C}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        results.append({
            'pair': f'{a}_vs_{b}',
            'class_a': int(a),
            'class_b': int(b),
            'margin': float(margin) if margin is not None else None,
            'n_support': int(sv),
            'metrics': metrics,
            'plot': os.path.basename(path)
        })

    out_json = os.path.join(RESULTS_DIR, f'pairwise_soft_C_{C}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    out_csv = os.path.join(RESULTS_DIR, f'pairwise_soft_C_{C}.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print('Saved:', out_json, out_csv)


if __name__ == '__main__':
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Iris.csv'))
    print('Running pairwise soft-margin analysis on', csv_path)
    run_pairwise_soft(csv_path, C=1.0)
