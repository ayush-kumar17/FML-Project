import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def decision_on_grid(model, pca, xx, yy):
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_orig = pca.inverse_transform(grid)
    if hasattr(model, 'coef_'):
        w = np.ravel(model.coef_)
        b = float(model.intercept_[0]) if hasattr(model, 'intercept_') else 0.0
        vals = X_orig.dot(w) + b
    else:
        vals = model.decision_function(X_orig)
    return vals.reshape(xx.shape)


def run(n_features=30, soft_C=1.0, hard_C=1e6, random_state=42):
    data = load_breast_cancer()
    X_full = data['data']
    y = data['target']

    n_features = min(int(n_features), X_full.shape[1])
    X = X_full[:, :n_features]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=random_state, stratify=y)

    hard_model = LinearSVC(C=hard_C, dual=False, max_iter=20000, tol=1e-4, random_state=random_state)
    hard_train_acc = None
    hard_feasible_30d = False
    try:
        hard_model.fit(X_tr, y_tr)
        hard_train_acc = float(accuracy_score(y_tr, hard_model.predict(X_tr)))
        hard_feasible_30d = (hard_train_acc == 1.0)
    except Exception:
        hard_model = None

    soft_model = SVC(kernel='linear', C=float(soft_C))
    soft_model.fit(X_tr, y_tr)
    soft_train_acc = float(accuracy_score(y_tr, soft_model.predict(X_tr)))

    pca2 = PCA(n_components=2, random_state=random_state)
    Xp2 = pca2.fit_transform(Xs)
    Xtr_p2 = pca2.transform(X_tr)

    lin2d = LinearSVC(C=1e6, dual=False, max_iter=20000, tol=1e-4, random_state=random_state)
    p2_train_acc = None
    p2_separable = False
    try:
        lin2d.fit(Xtr_p2, y_tr)
        p2_train_acc = float(accuracy_score(y_tr, lin2d.predict(Xtr_p2)))
        p2_separable = (p2_train_acc == 1.0)
    except Exception:
        p2_separable = False

    pad = 1.0
    x_min, x_max = Xp2[:, 0].min() - pad, Xp2[:, 0].max() + pad
    y_min, y_max = Xp2[:, 1].min() - pad, Xp2[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z_soft = decision_on_grid(soft_model, pca2, xx, yy)
    Z_hard = None
    if hard_model is not None:
        try:
            Z_hard = decision_on_grid(hard_model, pca2, xx, yy)
        except Exception:
            Z_hard = None

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('bwr')
    plt.scatter(Xtr_p2[:, 0], Xtr_p2[:, 1], c=y_tr, cmap=cmap, edgecolor='k', s=40, alpha=0.8)
    try:
        cs_soft = plt.contour(xx, yy, Z_soft, levels=[-1, 0, 1], colors=['gray','k','gray'], linestyles=['--','-','--'], linewidths=[1.0,2.0,1.0], alpha=0.9)
    except Exception:
        cs_soft = plt.contour(xx, yy, Z_soft, levels=[0], colors='k', linestyles='-', linewidths=2, alpha=0.9)
    try:
        cs_soft.collections[len(cs_soft.levels)//2].set_label(f'soft C={soft_C}')
    except Exception:
        pass

    try:
        if hasattr(soft_model, 'support_'):
            sv = X_tr[soft_model.support_]
            svp = pca2.transform(sv)
            plt.scatter(svp[:,0], svp[:,1], facecolors='none', edgecolors='k', s=120, linewidths=1.5, label='soft SVs')
    except Exception:
        pass

    if Z_hard is not None and hard_feasible_30d:
        try:
            cs_hard = plt.contour(xx, yy, Z_hard, levels=[-1, 0, 1], colors=['lightgreen','green','lightgreen'], linestyles=['--','-','--'], linewidths=[1.0,2.0,1.0], alpha=0.9)
            try:
                cs_hard.collections[len(cs_hard.levels)//2].set_label(f'hard approx C={int(hard_C)}')
            except Exception:
                pass
        except Exception:
            try:
                cs_hard = plt.contour(xx, yy, Z_hard, levels=[0], colors='green', linestyles='--', linewidths=2, alpha=0.9)
            except Exception:
                cs_hard = None
        try:
            if hard_model is not None and hasattr(hard_model, 'coef_'):
                w_h = np.ravel(hard_model.coef_)
                b_h = float(hard_model.intercept_[0]) if hasattr(hard_model, 'intercept_') else 0.0
                scores_h = X_tr.dot(w_h) + b_h
                tol = max(0.05, 0.02 * np.std(scores_h))
                mask_h = np.abs(np.abs(scores_h) - 1.0) <= tol
                if mask_h.any():
                    sv_approx = X_tr[mask_h]
                    svp_h = pca2.transform(sv_approx)
                    plt.scatter(svp_h[:,0], svp_h[:,1], facecolors='none', edgecolors='green', s=120, linewidths=1.5, label='hard approx SVs')
        except Exception:
            pass
    else:
        plt.text(0.02, 0.98, 'Hard-margin infeasible in 30D (training)', transform=plt.gca().transAxes, verticalalignment='top', fontsize=10, color='red')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Breast Cancer PCA-2D: soft C={soft_C}  hard feasible={hard_feasible_30d}')
    plt.legend(loc='lower right')
    out_png2d = os.path.join(RESULTS_DIR, f'breast_pca2d_softC_{soft_C}_hardC_{int(hard_C)}.png')
    plt.savefig(out_png2d, dpi=300, bbox_inches='tight')
    plt.close()

    pca3 = PCA(n_components=3, random_state=random_state)
    Xp3 = pca3.fit_transform(Xs)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xp3[:, 0], Xp3[:, 1], Xp3[:, 2], c=y, cmap=cmap, s=30, edgecolor='k', alpha=0.8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Breast Cancer PCA-3D (30D -> 3D)')
    out_png3d = os.path.join(RESULTS_DIR, 'breast_pca3d_30d.png')
    plt.savefig(out_png3d, dpi=300, bbox_inches='tight')
    plt.close()

    model_for_weights = hard_model if (hard_model is not None and hard_feasible_30d) else soft_model
    weights_plot_png = None
    if hasattr(model_for_weights, 'coef_'):
        w = np.ravel(model_for_weights.coef_)
        idx = np.arange(len(w))
        plt.figure(figsize=(10, 4))
        plt.bar(idx, np.abs(w))
        plt.xlabel('Feature index (0..')
        plt.ylabel('|w_i|')
        plt.title('Absolute weight magnitudes')
        weights_plot_png = os.path.join(RESULTS_DIR, f'breast_weight_magnitudes_{"hard" if (hard_model is not None and hard_feasible_30d) else "soft"}.png')
        plt.savefig(weights_plot_png, dpi=300, bbox_inches='tight')
        plt.close()

    report = {
        'n_features': int(n_features),
        'hard_train_acc_30d': hard_train_acc,
        'hard_feasible_30d': bool(hard_feasible_30d),
        'soft_train_acc_30d': float(soft_train_acc),
        'pca2_train_acc_linear_on_projection': p2_train_acc,
        'pca2_separable': bool(p2_separable),
        'plots': {
            'pca2_decision_png': os.path.basename(out_png2d),
            'pca3_scatter_png': os.path.basename(out_png3d),
            'weights_png': os.path.basename(weights_plot_png) if weights_plot_png else None
        }
    }
    out_json = os.path.join(RESULTS_DIR, f'breast_pca_visual_report_{n_features}d.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    print('Saved PCA visualization and report:', out_png2d, out_png3d, out_json)
    return report


if __name__ == '__main__':
    import os as _os
    soft_c = float(_os.environ.get('SOFT_C', '1.0'))
    hard_c = float(_os.environ.get('HARD_C', '1e6'))
    n_f = int(_os.environ.get('N_FEATURES', '30'))
    run(n_features=n_f, soft_C=soft_c, hard_C=hard_c)
