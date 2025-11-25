"""
Pairwise Hard-Margin SVM analysis for Iris dataset
Attempts strict hard-margin by accepting a linear SVM only if training split is perfectly separable.
Creates per-pair plots and saves pairwise metrics (accuracy, precision, recall, f1), margins and support vector counts.
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


def run_pairwise_hard(csv_path, features=('PetalLengthCm', 'PetalWidthCm')):
    X, y = load_iris(csv_path, features=features)
    classes = np.unique(y)
    results = []

    for a, b in combinations(classes, 2):
        mask = np.logical_or(y == a, y == b)
        X_pair = X[mask]
        y_pair = y[mask]
        y_bin = np.where(y_pair == a, 0, 1)

        X_tr, X_te, y_tr, y_te = train_test_split(X_pair, y_bin, test_size=0.3, random_state=42, stratify=y_bin)

        # Try strict linear hard-margin first
        svc_hard = SVC(kernel='linear', C=1e10)
        svc_hard.fit(X_tr, y_tr)
        y_tr_pred = svc_hard.predict(X_tr)
        kernel_used = 'linear'
        kernel_params = {}

        if np.mean(y_tr_pred == y_tr) < 1.0:
            # Linear infeasible; try kernel tricks
            print(f'Pair {a}_vs_{b}: linear hard-margin infeasible on training split -> trying kernel tricks')
            svc_hard = None
            kernel_used = None
            kernel_params = None

            # Candidate kernels to try (poly degree 2, poly degree 3, rbf)
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
                    # Found a kernel that separates training data
                    svc_hard = svc_k
                    kernel_used = cand['kernel']
                    kernel_params = {k: v for k, v in cand.items()}
                    print(f'Pair {a}_vs_{b}: separable with kernel {kernel_used} params {kernel_params}')
                    break

            if svc_hard is None:
                # No kernel achieved perfect separability on training split
                infeasible = True
                margin = None
                sv = None
                metrics = None
                print(f'Pair {a}_vs_{b}: no kernel found that yields perfect separability on training split')
            else:
                infeasible = False
                # For kernel models (non-linear), margin in input space isn't defined via coef_, so set margin=None
                if getattr(svc_hard, 'kernel', None) == 'linear':
                    margin = compute_margin_from_coef(svc_hard.coef_)
                else:
                    margin = None
                sv = len(svc_hard.support_)
                y_pred = svc_hard.predict(X_te)
                metrics = {
                    'accuracy': float(accuracy_score(y_te, y_pred)),
                    'precision': float(precision_score(y_te, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_te, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_te, y_pred, zero_division=0))
                }
        else:
            # Linear hard-margin accepted
            infeasible = False
            margin = compute_margin_from_coef(svc_hard.coef_)
            sv = len(svc_hard.support_)
            y_pred = svc_hard.predict(X_te)
            metrics = {
                'accuracy': float(accuracy_score(y_te, y_pred)),
                'precision': float(precision_score(y_te, y_pred, zero_division=0)),
                'recall': float(recall_score(y_te, y_pred, zero_division=0)),
                'f1': float(f1_score(y_te, y_pred, zero_division=0))
            }
            kernel_used = 'linear'
            kernel_params = {}

        # For visualization, show only the accepted hard model decision boundary (linear or kernel)
        fig, ax = plt.subplots(1, 1, figsize=(6,5))
        if not infeasible and svc_hard is not None:
            # plot accepted hard model (could be kernel)
            SVMVisualizer.plot_decision_boundary(svc_hard, X_pair, y_bin, title=f'Hard ({kernel_used}) {a} vs {b}', ax=ax)
        else:
            ax.text(0.5, 0.5, 'Hard margin infeasible\n(no kernel found that perfectly separates)', ha='center', va='center', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
        path = os.path.join(RESULTS_DIR, f'hard_pair_{a}_vs_{b}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        results.append({
            'pair': f'{a}_vs_{b}',
            'class_a': int(a),
            'class_b': int(b),
            'infeasible': infeasible,
            'kernel_used': kernel_used,
            'kernel_params': kernel_params,
            'margin': float(margin) if margin is not None else None,
            'n_support': int(sv) if sv is not None else None,
            'metrics': metrics,
            'plot': os.path.basename(path)
        })

    out_json = os.path.join(RESULTS_DIR, 'pairwise_hard.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    out_csv = os.path.join(RESULTS_DIR, 'pairwise_hard.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print('Saved:', out_json, out_csv)


if __name__ == '__main__':
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Iris.csv'))
    print('Running pairwise hard-margin analysis on', csv_path)
    run_pairwise_hard(csv_path)
