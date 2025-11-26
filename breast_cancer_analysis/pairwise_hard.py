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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tr_check, X_te_check, y_tr_check, y_te_check = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    fast_linear = LinearSVC(C=1e6, max_iter=20000, tol=1e-4, dual=False, random_state=random_state)
    fast_linear.fit(X_tr_check, y_tr_check)
    train_acc_fast = np.mean(fast_linear.predict(X_tr_check) == y_tr_check)

    svc_hard = None
    kernel_used = None
    kernel_params = None

    if train_acc_fast == 1.0:
        svc_hard = fast_linear
        kernel_used = 'linear (LinearSVC)'
        kernel_params = {}
        infeasible = False
        margin = compute_margin_from_coef(fast_linear.coef_)
        sv = None
        y_pred = fast_linear.predict(X_te_check)
        metrics = {
            'accuracy': float(accuracy_score(y_te_check, y_pred)),
            'precision': float(precision_score(y_te_check, y_pred)),
            'recall': float(recall_score(y_te_check, y_pred)),
            'f1': float(f1_score(y_te_check, y_pred))
        }
        X_tr, X_te, y_tr, y_te = X_tr_check, X_te_check, y_tr_check, y_te_check
    else:
        X_tr, X_te, y_tr, y_te = X_tr_check, X_te_check, y_tr_check, y_te_check
        svc_hard = SVC(kernel='linear', C=1e6, tol=1e-3, max_iter=10000, cache_size=500)
        svc_hard.fit(X_tr, y_tr)
        y_tr_pred = svc_hard.predict(X_tr)
        if np.mean(y_tr_pred == y_tr) < 1.0:
            print('Linear hard-margin infeasible on training split -> trying bounded kernels')
            svc_hard = None

            kernel_candidates = [
                {'kernel': 'poly', 'degree': 2, 'C': 1e4},
                {'kernel': 'poly', 'degree': 3, 'C': 1e4},
                {'kernel': 'rbf', 'gamma': 'scale', 'C': 1e4}
            ]

            GPU_AVAILABLE = False
            USE_CUML = False
            USE_THUNDERSVM = False
            try:
                from cuml.svm import SVC as cuSVC
                import cupy as cp
                GPU_AVAILABLE = True
                USE_CUML = True
                print('cuML GPU SVM detected: will use GPU for kernel SVM attempts')
            except Exception:
                try:
                    from thundersvm import SVC as thSVC
                    GPU_AVAILABLE = True
                    USE_THUNDERSVM = True
                    print('ThunderSVM detected: will use GPU for kernel SVM attempts')
                except Exception:
                    GPU_AVAILABLE = False

            for cand in kernel_candidates:
                if GPU_AVAILABLE and USE_CUML:
                    try:
                        X_tr_dev = cp.asarray(X_tr.astype('float32'))
                        y_tr_dev = cp.asarray(y_tr.astype('int32'))
                        if cand['kernel'] == 'poly':
                            svc_k = cuSVC(kernel='poly', degree=cand['degree'], C=float(cand['C']), gamma='scale')
                        else:
                            svc_k = cuSVC(kernel=cand['kernel'], C=float(cand['C']), gamma=cand.get('gamma','scale'))
                        svc_k.fit(X_tr_dev, y_tr_dev)
                        y_tr_pred_k = cp.asnumpy(svc_k.predict(X_tr_dev)).astype(int)
                    except Exception as e:
                        svc_k = None
                        y_tr_pred_k = np.array([])
                elif GPU_AVAILABLE and USE_THUNDERSVM:
                    try:
                        svc_k = thSVC(kernel=cand['kernel'], C=cand.get('C', 1e4))
                        svc_k.fit(X_tr.astype('float32'), y_tr.astype('int32'))
                        y_tr_pred_k = svc_k.predict(X_tr).astype(int)
                    except Exception:
                        svc_k = None
                        y_tr_pred_k = np.array([])
                else:
                    if cand['kernel'] == 'poly':
                        svc_k = SVC(kernel='poly', degree=cand['degree'], C=cand['C'], gamma='scale', tol=1e-3, max_iter=10000, cache_size=500)
                    else:
                        svc_k = SVC(kernel=cand['kernel'], C=cand['C'], gamma=cand.get('gamma','scale'), tol=1e-3, max_iter=10000, cache_size=500)
                    svc_k.fit(X_tr, y_tr)
                    y_tr_pred_k = svc_k.predict(X_tr)

                if y_tr_pred_k.size and np.mean(y_tr_pred_k == y_tr) == 1.0:
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
            print('No kernel found that perfectly separates training split (bounded attempts)')
        else:
            infeasible = False
            if getattr(svc_hard, 'kernel', None) == 'linear' and hasattr(svc_hard, 'coef_'):
                margin = compute_margin_from_coef(svc_hard.coef_)
            elif hasattr(svc_hard, 'coef_'):
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
        try:
            SVMVisualizer.plot_decision_boundary(svc_hard, X_tr, y_tr, title=f'Hard ({kernel_used}) {fname}', ax=ax)
        except Exception:
            xx, yy = np.meshgrid(np.linspace(X_tr[:,0].min()-1, X_tr[:,0].max()+1, 300),
                                 np.linspace(X_tr[:,1].min()-1, X_tr[:,1].max()+1, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            if hasattr(svc_hard, 'decision_function'):
                Z = svc_hard.decision_function(grid).reshape(xx.shape)
                ax.contour(xx, yy, Z, levels=[0], colors='k')
            ax.scatter(X_tr[:,0], X_tr[:,1], c=y_tr, cmap='bwr', edgecolor='k')
            ax.set_title(f'Hard ({kernel_used}) {fname}')
    else:
        ax.text(0.5, 0.5, 'Hard margin infeasible\n(no kernel found that perfectly separates)', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved:', out_json, out_csv, out_png)


if __name__ == '__main__':
    run_pairwise_hard(feature_idx=(0,1))
