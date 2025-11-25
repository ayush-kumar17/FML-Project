"""
Pairwise Hard-Margin SVM analysis for Breast Cancer dataset
Attempts strict hard-margin by accepting a linear SVM only if training split is perfectly separable.
If linear infeasible, tries kernel tricks (poly2, poly3, rbf) with very large C.
Outputs saved to drestcanceranalysis/results/
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure parent project dir on path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from visualizations import SVMVisualizer
except Exception:
    SVMVisualizer = None

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_margin_from_coef(coef):
    w = np.ravel(coef)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return None
    return 2.0 / w_norm


def run_pairwise_hard(feature_idx=(0, 1), random_state=42):
    data = load_breast_cancer()
    X_full = data['data']
    y = data['target']
    feature_names = data['feature_names']

    fi0, fi1 = feature_idx
    X = X_full[:, [fi0, fi1]]
    fname = f"{feature_names[fi0]}__{feature_names[fi1]}".replace(' ', '_')

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    svc_hard = SVC(kernel='linear', C=1e10)
    svc_hard.fit(X_tr, y_tr)
    y_tr_pred = svc_hard.predict(X_tr)

    if np.mean(y_tr_pred == y_tr) < 1.0:
        print('Linear hard-margin infeasible on training split -> trying kernels')
        svc_hard = None
        kernel_used = None
        kernel_params = None

        kernel_candidates = [
            {'kernel': 'poly', 'degree': 2},
            {'kernel': 'poly', 'degree': 3},
            {'kernel': 'rbf', 'gamma': 'scale'}
        ]

        for cand in kernel_candidates:
            if cand['kernel'] == 'poly':
                svc_k = SVC(kernel='poly', degree=cand['degree'], C=1e10, gamma='scale')
            else:
                svc_k = SVC(kernel=cand['kernel'], C=1e10, gamma=cand.get('gamma', 'scale'))
            svc_k.fit(X_tr, y_tr)
            y_tr_pred_k = svc_k.predict(X_tr)
            if np.mean(y_tr_pred_k == y_tr) == 1.0:
                svc_hard = svc_k
                kernel_used = cand['kernel']
                kernel_params = cand
                print(f'Separable with kernel {kernel_used} {kernel_params}')
                break

    if svc_hard is None:
        infeasible = True
        margin = None
        sv = None
        metrics = None
        print('No kernel found that perfectly separates training split')
    else:
        infeasible = False
        if getattr(svc_hard, 'kernel', None) == 'linear':
            margin = compute_margin_from_coef(svc_hard.coef_)
        else:
            margin = None
        sv = len(svc_hard.support_)
        y_pred = svc_hard.predict(X_te)
        metrics = {
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred)),
            'recall': float(recall_score(y_te, y_pred)),
            'f1': float(f1_score(y_te, y_pred))
        }

    out = {
        'dataset': 'breast_cancer',
        'features': [feature_names[fi0], feature_names[fi1]],
        'infeasible': infeasible,
        'kernel_used': kernel_used if not infeasible else None,
        'kernel_params': kernel_params if not infeasible else None,
        'margin': margin,
        'n_support': sv,
        'metrics': metrics
    }

    out_json = os.path.join(RESULTS_DIR, f'breast_hard_features_{fi0}_{fi1}.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    out_csv = os.path.join(RESULTS_DIR, f'breast_hard_features_{fi0}_{fi1}.csv')
    pd.DataFrame([out]).to_csv(out_csv, index=False)

    out_png = os.path.join(RESULTS_DIR, f'breast_hard_features_{fi0}_{fi1}.png')
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    if not infeasible and svc_hard is not None:
        SVMVisualizer.plot_decision_boundary(svc_hard, X_tr, y_tr, title=f'Hard ({kernel_used}) {fname}', ax=ax)
    else:
        ax.text(0.5, 0.5, 'Hard margin infeasible\n(no kernel found that perfectly separates)', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved:', out_json, out_csv, out_png)


if __name__ == '__main__':
    run_pairwise_hard(feature_idx=(0,1))
