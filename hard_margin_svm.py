"""
Hard Margin SVM Implementation
Works only with linearly separable data
"""

import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


class HardMarginSVM:
    """
    Hard Margin SVM Classifier
    Uses no slack variables - requires perfectly linearly separable data
    """
    
    def __init__(self):
        """Initialize Hard Margin SVM"""
        # Using sklearn's SVC with very large C approximates hard margin
        # C -> infinity means no tolerance for errors
        self.model = SVC(kernel='linear', C=1e10)  # Very large C for hard margin
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        """
        Train the Hard Margin SVM
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (+1 or -1)
            
        Returns:
        --------
        self : object
        """
        try:
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
            
            print(f"Hard Margin SVM trained successfully!")
            print(f"Number of support vectors: {self.n_support_}")
            
        except Exception as e:
            print(f"Hard Margin SVM failed to converge!")
            print(f"This data may not be linearly separable.")
            print(f"Error: {str(e)}")
            raise ValueError("Hard Margin SVM requires linearly separable data")
            
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
        where the hyperplane is defined as w·x + b = 0
        
        Returns:
        --------
        w : array, shape (n_features,)
            Normal vector to hyperplane
        b : float
            Intercept term
        """
        return self.coef_[0], self.intercept_[0]
    
    def is_data_separable(self, X, y):
        """
        Check if data is linearly separable
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to check
        y : array-like, shape (n_samples,)
            Labels
            
        Returns:
        --------
        separable : bool
            True if data is linearly separable
        """
        try:
            temp_model = SVC(kernel='linear', C=1e10)
            y_converted = np.where(y == 0, -1, y)
            temp_model.fit(X, y_converted)
            predictions = temp_model.predict(X)
            accuracy = np.mean(predictions == y_converted)
            return accuracy == 1.0
        except:
            return False


def demonstrate_hard_margin():
    """
    Demonstration of Hard Margin SVM
    """
    print("=" * 60)
    print("HARD MARGIN SVM DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Linearly separable data
    print("\n1. Testing on Linearly Separable Data:")
    print("-" * 60)
    
    # Create simple linearly separable data
    X_sep = np.array([
        [1, 2], [2, 3], [2, 2], [3, 3],
        [6, 5], [7, 6], [6, 6], [7, 7]
    ])
    y_sep = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    svm = HardMarginSVM()
    svm.fit(X_sep, y_sep)
    
    w, b = svm.get_hyperplane_params()
    margin = svm.get_margin_width()
    
    print(f"Hyperplane: w = {w}, b = {b:.4f}")
    print(f"Margin width: {margin:.4f}")
    print(f"Support vectors:\n{svm.support_vectors_}")
    
    # Example 2: Non-separable data (will fail)
    print("\n\n2. Testing on Non-Separable Data:")
    print("-" * 60)
    
    X_nonsep = np.array([
        [1, 2], [2, 3], [3, 3], [4, 5],
        [2, 2], [3, 4], [4, 4], [5, 6]
    ])
    y_nonsep = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    
    try:
        svm_fail = HardMarginSVM()
        svm_fail.fit(X_nonsep, y_nonsep)
    except ValueError as e:
        print(f"Expected failure: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_hard_margin()