"""
pairwise_hard_2.py

Hard-margin SVM in higher dimensions (default: first 30 features of breast_cancer).
Behavior:
- Scales features with StandardScaler
- Uses a fast LinearSVC pre-check with very large C to detect perfect linear separability
- If separable, saves metrics, margin and a PCA projection plot
- By default avoids expensive kernel attempts in high-dim space. Set environment
  TRY_KERNELS=1 to allow bounded kernel attempts (with conservative params).
- Results (JSON/CSV/PNG) saved to results/ next to this file.
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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_margin_from_coef(coef):
    w = np.ravel(coef)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return None
    return 2.0 / w_norm


def run_pairwise_hard_30d(n_features=30, try_kernels=False, random_state=42):
    data = load_breast_cancer()
    X_full = data['data']
    y = data['target']
    feature_names = list(data['feature_names'])

    n_features = min(n_features, X_full.shape[1])
    X = X_full[:, :n_features]
    used_feature_names = feature_names[:n_features]
    tag = f'{n_features}d_first'

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    # fast linear separability check
    fast_linear = LinearSVC(C=1e6, max_iter=20000, tol=1e-4, dual=False, random_state=random_state)
    fast_linear.fit(X_tr, y_tr)
    train_acc = float(np.mean(fast_linear.predict(X_tr) == y_tr))

    svc_model = None
    infeasible = True
    kernel_used = None
    kernel_params = None
    margin = None
    n_support = None
    metrics = None

    if train_acc == 1.0:
        svc_model = fast_linear
        infeasible = False
        kernel_used = 'linear (LinearSVC)'
        kernel_params = {}
        margin = compute_margin_from_coef(fast_linear.coef_)
        n_support = None
        y_pred = fast_linear.predict(X_te)
        metrics = {
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'precision': float(precision_score(y_te, y_pred)),
            'recall': float(recall_score(y_te, y_pred)),
            'f1': float(f1_score(y_te, y_pred))
        }
        print('Linear separable in', n_features, 'D (fast LinearSVC)')
    else:
        print('Not linearly separable on training split (fast LinearSVC).')
        if try_kernels:
            print('try_kernels requested: performing conservative bounded kernel attempts (may be slow)')
            # conservative kernel candidates for high-dim
            kernel_candidates = [
                {'kernel': 'rbf', 'C': 100.0, 'gamma': 'scale'},
                {'kernel': 'poly', 'C': 100.0, 'degree': 2, 'gamma': 'scale'}
            ]
            for cand in kernel_candidates:
                try:
                    if cand['kernel'] == 'poly':
                        m = SVC(kernel='poly', degree=cand.get('degree', 2), C=cand.get('C', 100.0), gamma=cand.get('gamma','scale'), tol=1e-3, max_iter=2000)
                    else:
                        m = SVC(kernel=cand['kernel'], C=cand.get('C', 100.0), gamma=cand.get('gamma','scale'), tol=1e-3, max_iter=2000)
                    m.fit(X_tr, y_tr)
                    y_tr_pred = m.predict(X_tr)
                    if float(np.mean(y_tr_pred == y_tr)) == 1.0:
                        svc_model = m
                        infeasible = False
                        kernel_used = cand['kernel']
                        kernel_params = cand
                        margin = None
                        if hasattr(m, 'coef_'):
                            margin = compute_margin_from_coef(m.coef_)
                        n_support = int(len(m.support_)) if hasattr(m, 'support_') else None
                        y_pred = m.predict(X_te)
                        metrics = {
                            'accuracy': float(accuracy_score(y_te, y_pred)),
                            'precision': float(precision_score(y_te, y_pred)),
                            'recall': float(recall_score(y_te, y_pred)),
                            'f1': float(f1_score(y_te, y_pred))
                        }
                        print('Separable with kernel', kernel_used, kernel_params)
                        break
                except Exception as e:
                    print('Kernel attempt failed:', e)
        else:
            print('Skipping kernel attempts (set try_kernels=True to enable).')

    out = {
        'dataset': 'breast_cancer',
        'n_features': n_features,
        'features': used_feature_names,
        'infeasible': infeasible,
        'kernel_used': kernel_used if not infeasible else None,
        'kernel_params': kernel_params if not infeasible else None,
        'margin': margin,
        'n_support': n_support,
        'metrics': metrics
    }

    out_json = os.path.join(RESULTS_DIR, f'breast_hard_{tag}.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    out_csv = os.path.join(RESULTS_DIR, f'breast_hard_{tag}.csv')
    pd.DataFrame([out]).to_csv(out_csv, index=False)

    # plotting: for >2D produce PCA projection and hyperplane contour + weight bars + distance histogram
    # PCA projection + hyperplane contour (if model gives decision function or linear coef)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X_tr)

        # build grid in PCA space
        x_min, x_max = Xp[:,0].min() - 1, Xp[:,0].max() + 1
        y_min, y_max = Xp[:,1].min() - 1, Xp[:,1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_pca = np.c_[xx.ravel(), yy.ravel()]
        # map grid points back to original feature space
        grid_orig = pca.inverse_transform(grid_pca)

        # compute decision values on original space
        if svc_model is not None and hasattr(svc_model, 'coef_'):
            # linear model: use w and b for fast evaluation
            w = np.ravel(svc_model.coef_)
            b = float(svc_model.intercept_[0]) if hasattr(svc_model, 'intercept_') else 0.0
            decisions = grid_orig.dot(w) + b
        elif svc_model is not None and hasattr(svc_model, 'decision_function'):
            try:
                decisions = svc_model.decision_function(grid_orig)
            except Exception:
                decisions = np.zeros(grid_orig.shape[0])
        else:
            decisions = np.zeros(grid_orig.shape[0])

        Z = decisions.reshape(xx.shape)

        fig, ax = plt.subplots(1,1,figsize=(7,6))
        ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1.2)
        ax.scatter(Xp[:,0], Xp[:,1], c=y_tr, cmap='bwr', edgecolor='k', s=30)
        ax.set_title(f'PCA projection + hyperplane contour ({n_features}D)')
        out_pca_png = os.path.join(RESULTS_DIR, f'breast_hard_{tag}_pca_hyperplane.png')
        plt.savefig(out_pca_png, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        out_pca_png = None

    # weight magnitudes bar chart (only for linear models)
    out_weights_png = None
    try:
        if svc_model is not None and hasattr(svc_model, 'coef_'):
            w = np.ravel(svc_model.coef_)
            abs_w = np.abs(w)
            # show top 30 weights
            num_show = min(len(abs_w), 30)
            idx_sorted = np.argsort(-abs_w)[:num_show]
            fe_names = used_feature_names[:len(abs_w)] if 'used_feature_names' in locals() else [f'feat{i}' for i in range(len(abs_w))]
            labels = [fe_names[i] for i in idx_sorted]
            vals = abs_w[idx_sorted]
            fig, ax = plt.subplots(1,1,figsize=(9,4))
            ax.bar(range(len(vals)), vals, tick_label=labels)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title('Top |w_i| for linear hard SVM')
            out_weights_png = os.path.join(RESULTS_DIR, f'breast_hard_{tag}_weights.png')
            plt.tight_layout()
            plt.savefig(out_weights_png, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception:
        out_weights_png = None

    # signed distance histogram
    out_dist_png = None
    try:
        if svc_model is not None and hasattr(svc_model, 'coef_'):
            w = np.ravel(svc_model.coef_)
            b = float(svc_model.intercept_[0]) if hasattr(svc_model, 'intercept_') else 0.0
            w_norm = np.linalg.norm(w) if np.linalg.norm(w) != 0 else 1.0
            dists = (X_tr.dot(w) + b) / w_norm
            fig, ax = plt.subplots(1,1,figsize=(7,4))
            ax.hist([dists[y_tr==0], dists[y_tr==1]], bins=30, label=['class0','class1'], alpha=0.7)
            ax.legend()
            ax.set_title('Signed distances to hyperplane (train)')
            out_dist_png = os.path.join(RESULTS_DIR, f'breast_hard_{tag}_distances.png')
            plt.savefig(out_dist_png, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception:
        out_dist_png = None

    # summary main PNG (if PCA produced one use that, else fallback)
    out_main_png = out_pca_png or out_weights_png or out_dist_png
    if out_main_png is None:
        # minimal placeholder
        out_main_png = os.path.join(RESULTS_DIR, f'breast_hard_{tag}.png')
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        ax.text(0.5,0.5,'No visualization available',ha='center')
        plt.savefig(out_main_png, dpi=300, bbox_inches='tight')
        plt.close()

    print('Saved:', out_json, out_csv, out_pca_png, out_weights_png, out_dist_png)


if __name__ == '__main__':
    # control via environment variables if needed
    try_k = bool(int(os.environ.get('TRY_KERNELS','0')))
    run_pairwise_hard_30d(n_features=30, try_kernels=try_k)
