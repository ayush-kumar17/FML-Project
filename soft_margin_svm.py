"""
Soft Margin SVM Implementation
Works with both linearly separable and non-separable data
Uses regularization parameter C to control trade-off
"""

import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


class SoftMarginSVM:
    """
    Soft Margin SVM Classifier
    Allows misclassifications with penalty parameter C
    """
    
    def __init__(self, C=1.0):
        """
        Initialize Soft Margin SVM
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter. Larger C means less regularization
            - Large C: less tolerance for errors (similar to hard margin)
            - Small C: more tolerance for errors (larger margin, more errors)
        """
        self.C = C
        self.model = SVC(kernel='linear', C=C)
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        """
        Train the Soft Margin SVM
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (+1 or -1, or 0 and 1)
            
        Returns:
        --------
        self : object
        """
        # Convert labels to -1 and 1 if they're 0 and 1
        y_converted = np.where(y == 0, -1, y)
        
        # Fit the model
        self.model.fit(X, y_converted)
        
        # Store support vectors and parameters
        self.support_vectors_ = self.model.support_vectors_
        self.support_vector_indices_ = self.model.support_
        self.n_support_ = len(self.support_vectors_)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        print(f"Soft Margin SVM (C={self.C}) trained successfully!")
        print(f"Number of support vectors: {self.n_support_}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        return self.model.predict(X)
    
    def decision_function(self, X):
        """
        Calculate decision function values
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        decision : array, shape (n_samples,)
            Decision function values
        """
        return self.model.decision_function(X)
    
    def get_margin_width(self):
        """
        Calculate the margin width (2/||w||)
        
        Returns:
        --------
        margin : float
            Width of the margin
        """
        w_norm = np.linalg.norm(self.coef_)
        margin = 2.0 / w_norm
        return margin
    
    def get_hyperplane_params(self):
        """
        Get hyperplane parameters w and b
        
        Returns:
        --------
        w : array, shape (n_features,)
            Normal vector to hyperplane
        b : float
            Intercept term
        """
        return self.coef_[0], self.intercept_[0]
    
    def calculate_slack_variables(self, X, y):
        """
        Calculate slack variables (ξ) for training data
        ξ_i = max(0, 1 - y_i * f(x_i))
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Labels
            
        Returns:
        --------
        slack : array, shape (n_samples,)
            Slack variable for each sample
        """
        y_converted = np.where(y == 0, -1, y)
        decision_values = self.decision_function(X)
        slack = np.maximum(0, 1 - y_converted * decision_values)
        return slack
    
    def get_misclassified_count(self, X, y):
        """
        Get number of misclassified training samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        count : int
            Number of misclassified samples
        """
        predictions = self.predict(X)
        y_converted = np.where(y == 0, -1, y)
        return np.sum(predictions != y_converted)


def compare_different_C_values():
    """
    Demonstrate the effect of different C values
    """
    print("=" * 60)
    print("SOFT MARGIN SVM: EFFECT OF C PARAMETER")
    print("=" * 60)
    
    # Create non-separable data with outliers
    X = np.array([
        [1, 2], [2, 3], [2, 2], [3, 3],
        [6, 5], [7, 6], [6, 6], [7, 7],
        [4, 4],  # Outlier from class 0
        [5, 4]   # Outlier from class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    
    # Test different C values
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
        
        # Calculate metrics
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
    
    # Summary table
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
    """
    Basic demonstration of Soft Margin SVM
    """
    print("\n\n" + "=" * 60)
    print("BASIC SOFT MARGIN SVM DEMONSTRATION")
    print("=" * 60)
    
    # Example with non-separable data
    X = np.array([
        [1, 2], [2, 3], [2, 2], [3, 3],
        [6, 5], [7, 6], [6, 6], [7, 7],
        [4, 4]  # This point is in the "wrong" region
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
    
    # Show slack variables
    slack = svm.calculate_slack_variables(X, y)
    print(f"\nSlack variables (ξ):")
    for i, s in enumerate(slack):
        if s > 1e-5:  # Show only non-zero slack
            print(f"  Sample {i}: ξ = {s:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_soft_margin()
    compare_different_C_values()