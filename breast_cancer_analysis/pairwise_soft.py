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

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from visualizations import SVMVisualizer
except Exception:
    SVMVisualizer = None

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SVM_C = 10000


def compute_margin_from_coef(coef):
    w = np.ravel(coef)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return None
    return 2.0 / w_norm


def run_pairwise_soft(feature_idx=(0, 1), C_values=None, random_state=42):
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    if isinstance(C_values, (int, float)):
        C_values = [float(C_values)]

    data = load_breast_cancer()
    X_full = data['data']
    y = data['target']
    feature_names = data['feature_names']

    fi0, fi1 = feature_idx
    X = X_full[:, [fi0, fi1]]
    fname = f"{feature_names[fi0]}__{feature_names[fi1]}".replace(' ', '_')

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    summary = []
    for C in C_values:
        model = SVC(kernel='linear', C=float(C))
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        metrics = {
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred)),
            'recall': float(recall_score(y_te, y_pred)),
            'f1': float(f1_score(y_te, y_pred))
        }

        margin = None
        if hasattr(model, 'coef_'):
            margin = compute_margin_from_coef(model.coef_)

        n_support = int(len(model.support_))

        result = {
            'dataset': 'breast_cancer',
            'features': [feature_names[fi0], feature_names[fi1]],
            'C': float(C),
            'margin': margin,
            'n_support': n_support,
            'metrics': metrics
        }

        c_tag = str(C).replace('.', 'p')
        out_json = os.path.join(RESULTS_DIR, f'breast_soft_features_{fi0}_{fi1}_C_{c_tag}.json')
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)

        out_csv = os.path.join(RESULTS_DIR, f'breast_soft_features_{fi0}_{fi1}_C_{c_tag}.csv')
        pd.DataFrame([result]).to_csv(out_csv, index=False)

        out_png = os.path.join(RESULTS_DIR, f'breast_soft_features_{fi0}_{fi1}_C_{c_tag}.png')
        if SVMVisualizer is not None:
            try:
                SVMVisualizer.plot_decision_boundary(model, X_tr, y_tr, title=f"Soft SVM C={C} ({fname})")
                plt.savefig(out_png, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass
        else:
            xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                                 np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = model.decision_function(grid).reshape(xx.shape)
            plt.figure(figsize=(6,5))
            plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.2)
            plt.contour(xx, yy, Z, levels=[-1,0,1], linestyles=['--','-','--'], colors='k')
            plt.scatter(X_tr[:,0], X_tr[:,1], c=y_tr, edgecolor='k', cmap='bwr', s=40)
            plt.title(f"Soft SVM C={C} ({fname})")
            plt.xlabel(feature_names[fi0])
            plt.ylabel(feature_names[fi1])
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()

        print('Saved:', out_json, out_csv, out_png)
        summary.append({
            'C': float(C),
            'margin': margin,
            'n_support': n_support,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

    out_summary_json = os.path.join(RESULTS_DIR, f'breast_soft_features_{fi0}_{fi1}_C_study.json')
    with open(out_summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    out_summary_csv = os.path.join(RESULTS_DIR, f'breast_soft_features_{fi0}_{fi1}_C_study.csv')
    pd.DataFrame(summary).to_csv(out_summary_csv, index=False)

    print('Saved summary:', out_summary_json, out_summary_csv)


if __name__ == '__main__':
    ev = os.environ.get('SVM_C')
    if ev:
        try:
            c_val = float(ev)
        except Exception:
            c_val = SVM_C
    else:
        c_val = SVM_C
    run_pairwise_soft(feature_idx=(0,1), C_values=c_val)
