"""
Run C study for Iris pairwise soft-margin SVMs
- Trains linear SVM for each pair and each C value
- Saves per-pair per-C decision-boundary plots (2D only), and per-pair charts:
  C vs accuracy, C vs #support vectors, C vs margin
- Writes a summary CSV with an explanation column
Usage:
  SVM_C_VALUES="0.01,0.1,1,10" python pairwise_C_analysis.py
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
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

DEFAULT_CS = [0.01, 0.1, 1.0, 10.0, 100.0]


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


def plot_decision_boundary_fallback(model, X, y, title, out_path):
    # fallback contour using decision_function
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.decision_function(grid).reshape(xx.shape)
        plt.figure(figsize=(6,5))
        plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.2)
        plt.contour(xx, yy, Z, levels=[-1,0,1], linestyles=['--','-','--'], colors='k')
    except Exception:
        plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def run(csv_path, features=('PetalLengthCm', 'PetalWidthCm')):
    ev = os.environ.get('SVM_C_VALUES')
    if ev:
        try:
            C_values = [float(x) for x in ev.split(',') if x.strip()]
        except Exception:
            C_values = DEFAULT_CS
    else:
        C_values = DEFAULT_CS

    X, y = load_iris(csv_path, features=features)
    classes = np.unique(y)

    rows = []

    for a,b in combinations(classes,2):
        mask = np.logical_or(y==a, y==b)
        X_pair = X[mask]
        y_pair = y[mask]
        y_bin = np.where(y_pair==a, 0, 1)

        X_tr, X_te, y_tr, y_te = train_test_split(X_pair, y_bin, test_size=0.3, random_state=42, stratify=y_bin)

        accs = []
        supports = []
        margins = []

        for C in C_values:
            model = SVC(kernel='linear', C=float(C))
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            acc = float(accuracy_score(y_te, y_pred))
            precision = float(precision_score(y_te, y_pred, zero_division=0))
            recall = float(recall_score(y_te, y_pred, zero_division=0))
            f1 = float(f1_score(y_te, y_pred, zero_division=0))

            margin = compute_margin_from_coef(model.coef_) if hasattr(model,'coef_') else None
            n_support = int(len(model.support_)) if hasattr(model,'support_') else None

            accs.append(acc)
            supports.append(n_support if n_support is not None else math.nan)
            margins.append(margin if margin is not None else math.nan)

            # save per-pair per-C plot
            out_png = os.path.join(RESULTS_DIR, f'iris_pair_{a}_vs_{b}_C_{str(C).replace(".","p")}.png')
            title = f'Iris pair {a} vs {b} (C={C})'
            if SVMVisualizer is not None:
                try:
                    SVMVisualizer.plot_decision_boundary(model, X_pair, y_bin, title=title)
                    plt.savefig(out_png, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    plot_decision_boundary_fallback(model, X_pair, y_bin, title, out_png)
            else:
                plot_decision_boundary_fallback(model, X_pair, y_bin, title, out_png)

            explanation = (
                f'Pair {a}_vs_{b}, C={C}: test_acc={acc:.4f}, support={n_support}, margin={margin if margin is not None else "NA"}. '
                f'As C increases margin typically decreases and model fits harder; support vector count often drops.'
            )

            rows.append({
                'pair': f'{a}_vs_{b}',
                'class_a': int(a),
                'class_b': int(b),
                'C': float(C),
                'test_accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_support': n_support,
                'margin': margin,
                'plot': os.path.basename(out_png),
                'explanation': explanation
            })

        # per-pair charts: accuracy, support, margin vs C
        try:
            xs = C_values
            plt.figure(figsize=(6,4))
            plt.plot(xs, accs, marker='o')
            plt.xscale('log')
            plt.xlabel('C (log scale)')
            plt.ylabel('Test accuracy')
            plt.title(f'Accuracy vs C for pair {a} vs {b}')
            outA = os.path.join(RESULTS_DIR, f'accuracy_vs_C_pair_{a}_vs_{b}.png')
            plt.savefig(outA, dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(xs, supports, marker='o')
            plt.xscale('log')
            plt.xlabel('C (log scale)')
            plt.ylabel('Number of support vectors')
            plt.title(f'Support vectors vs C for pair {a} vs {b}')
            outS = os.path.join(RESULTS_DIR, f'supports_vs_C_pair_{a}_vs_{b}.png')
            plt.savefig(outS, dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(xs, margins, marker='o')
            plt.xscale('log')
            plt.xlabel('C (log scale)')
            plt.ylabel('Margin (2/||w||)')
            plt.title(f'Margin vs C for pair {a} vs {b}')
            outM = os.path.join(RESULTS_DIR, f'margin_vs_C_pair_{a}_vs_{b}.png')
            plt.savefig(outM, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    # write summary CSV
    out_csv = os.path.join(RESULTS_DIR, 'iris_pairwise_C_study_summary.csv')
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print('Saved summary:', out_csv)


if __name__ == '__main__':
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Iris.csv'))
    run(csv_path)
