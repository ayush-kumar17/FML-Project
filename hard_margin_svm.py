import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


class HardMarginSVM:
    def __init__(self):
        self.model = SVC(kernel='linear', C=1e10) 
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        try:
            y_converted = np.where(y == 0, -1, y)
            
            self.model.fit(X, y_converted)
            
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
        return self.model.predict(X)
    
    def decision_function(self, X):
        return self.model.decision_function(X)
    
    def get_margin_width(self):
        w_norm = np.linalg.norm(self.coef_)
        margin = 2.0 / w_norm
        return margin
    
    def get_hyperplane_params(self):
        return self.coef_[0], self.intercept_[0]
    
    def is_data_separable(self, X, y):
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
    print("=" * 60)
    print("HARD MARGIN SVM DEMONSTRATION")
    print("=" * 60)
    print("\n1. Testing on Linearly Separable Data:")
    print("-" * 60)
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