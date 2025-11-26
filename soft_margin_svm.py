import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


class SoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.model = SVC(kernel='linear', C=C)
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        y_converted = np.where(y == 0, -1, y)
        self.model.fit(X, y_converted)
        self.support_vectors_ = self.model.support_vectors_
        self.support_vector_indices_ = self.model.support_
        self.n_support_ = len(self.support_vectors_)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        print(f"Soft Margin SVM (C={self.C}) trained successfully!")
        print(f"Number of support vectors: {self.n_support_}")
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def decision_function(self, X):
        return self.model.decision_function(X)
    
    def get_margin_width(self):
        w_norm = np.linalg.norm(self.coef_)
        margin = 2.0 / w_norm
        return margin
    
    def get_hyperplane_params(self):
        return self.coef_[0], self.intercept_[0]
    
    def calculate_slack_variables(self, X, y):
        y_converted = np.where(y == 0, -1, y)
        decision_values = self.decision_function(X)
        slack = np.maximum(0, 1 - y_converted * decision_values)
        return slack
    
    def get_misclassified_count(self, X, y):
        predictions = self.predict(X)
        y_converted = np.where(y == 0, -1, y)
        return np.sum(predictions != y_converted)


def compare_different_C_values():
    print("=" * 60)
    print("SOFT MARGIN SVM: EFFECT OF C PARAMETER")
    print("=" * 60)
    X = np.array([
        [1, 2], [2, 3], [2, 2], [3, 3],
        [6, 5], [7, 6], [6, 6], [7, 7],
        [4, 4],  
        [5, 4]   
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    C_values = [0.01, 0.1, 1, 10, 100]
    
    print("\nTraining data shape:", X.shape)
    print("Class distribution:", np.bincount(y))
    print("\n" + "-" * 60)
    
    results = []
    
    for C in C_values:
        print(f"\nC = {C}")
        print("-" * 40)
        
        svm = SoftMarginSVM(C=C)
        svm.fit(X, y)
        
        margin = svm.get_margin_width()
        slack = svm.calculate_slack_variables(X, y)
        total_slack = np.sum(slack)
        misclassified = svm.get_misclassified_count(X, y)
        
        print(f"Margin width: {margin:.4f}")
        print(f"Total slack: {total_slack:.4f}")
        print(f"Misclassified samples: {misclassified}")
        print(f"Support vectors: {svm.n_support_}")
        
        results.append({
            'C': C,
            'margin': margin,
            'slack': total_slack,
            'misclassified': misclassified,
            'n_support': svm.n_support_
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'C':<10} {'Margin':<12} {'Total Slack':<15} {'Errors':<10} {'SV Count':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['C']:<10} {r['margin']:<12.4f} {r['slack']:<15.4f} {r['misclassified']:<10} {r['n_support']:<10}")
    
    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("- Small C: Larger margin, more errors allowed")
    print("- Large C: Smaller margin, fewer errors (closer to hard margin)")
    print("=" * 60)


def demonstrate_soft_margin():
    print("\n\n" + "=" * 60)
    print("BASIC SOFT MARGIN SVM DEMONSTRATION")
    print("=" * 60)
    X = np.array([
        [1, 2], [2, 3], [2, 2], [3, 3],
        [6, 5], [7, 6], [6, 6], [7, 7],
        [4, 4]  
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    
    print("\nTraining on non-separable data...")
    print(f"Data shape: {X.shape}")
    
    svm = SoftMarginSVM(C=1.0)
    svm.fit(X, y)
    
    w, b = svm.get_hyperplane_params()
    margin = svm.get_margin_width()
    
    print(f"\nHyperplane: w = {w}, b = {b:.4f}")
    print(f"Margin width: {margin:.4f}")
    print(f"Support vectors:\n{svm.support_vectors_}")
    
    slack = svm.calculate_slack_variables(X, y)
    print(f"\nSlack variables (ξ):")
    for i, s in enumerate(slack):
        if s > 1e-5: 
            print(f"  Sample {i}: ξ = {s:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_soft_margin()
    compare_different_C_values()