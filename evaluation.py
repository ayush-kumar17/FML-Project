"""
Evaluation Module for SVM Project
Calculates performance metrics and creates comparison reports
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import time
import pandas as pd


class SVMEvaluator:
    """Evaluation tools for SVM models"""
    
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        """
        Comprehensive evaluation of a single SVM model
        
        Parameters:
        -----------
        model : SVM model
            Trained SVM model
        X_train : array
            Training features
        y_train : array
            Training labels
        X_test : array
            Test features
        y_test : array
            Test labels
            
        Returns:
        --------
        metrics : dict
            Dictionary containing all evaluation metrics
        """
        # Convert labels if needed
        y_train_conv = np.where(y_train == 0, -1, y_train)
        y_test_conv = np.where(y_test == 0, -1, y_test)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train_conv, y_train_pred),
            'test_accuracy': accuracy_score(y_test_conv, y_test_pred),
            'train_precision': precision_score(y_train_conv, y_train_pred, average='binary'),
            'test_precision': precision_score(y_test_conv, y_test_pred, average='binary'),
            'train_recall': recall_score(y_train_conv, y_train_pred, average='binary'),
            'test_recall': recall_score(y_test_conv, y_test_pred, average='binary'),
            'train_f1': f1_score(y_train_conv, y_train_pred, average='binary'),
            'test_f1': f1_score(y_test_conv, y_test_pred, average='binary'),
            'n_support_vectors': model.n_support_,
            'margin_width': model.get_margin_width()
        }
        
        # Confusion matrix
        metrics['confusion_matrix_train'] = confusion_matrix(y_train_conv, y_train_pred)
        metrics['confusion_matrix_test'] = confusion_matrix(y_test_conv, y_test_pred)
        
        return metrics
    
    @staticmethod
    def compare_models(hard_model, soft_model, X_train, y_train, X_test, y_test):
        """
        Compare Hard Margin and Soft Margin SVM
        
        Parameters:
        -----------
        hard_model : HardMarginSVM or None
            Trained Hard Margin model (None if failed)
        soft_model : SoftMarginSVM
            Trained Soft Margin model
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
            
        Returns:
        --------
        comparison : dict
            Comparison results
        """
        results = {}
        
        # Evaluate Hard Margin (if available)
        if hard_model is not None:
            try:
                results['hard_margin'] = SVMEvaluator.evaluate_model(
                    hard_model, X_train, y_train, X_test, y_test
                )
                results['hard_margin']['status'] = 'Success'
            except Exception as e:
                results['hard_margin'] = {
                    'status': 'Failed',
                    'error': str(e)
                }
        else:
            results['hard_margin'] = {
                'status': 'Not Applicable (Data not separable)'
            }
        
        # Evaluate Soft Margin
        try:
            results['soft_margin'] = SVMEvaluator.evaluate_model(
                soft_model, X_train, y_train, X_test, y_test
            )
            results['soft_margin']['status'] = 'Success'
            results['soft_margin']['C_parameter'] = soft_model.C
        except Exception as e:
            results['soft_margin'] = {
                'status': 'Failed',
                'error': str(e)
            }
        
        return results
    
    @staticmethod
    def print_evaluation_report(metrics, model_name="SVM"):
        """
        Print detailed evaluation report
        
        Parameters:
        -----------
        metrics : dict
            Metrics dictionary from evaluate_model
        model_name : str
            Name of the model
        """
        print("=" * 70)
        print(f"{model_name} EVALUATION REPORT")
        print("=" * 70)
        
        if metrics.get('status') != 'Success' and 'status' in metrics:
            print(f"\nStatus: {metrics['status']}")
            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
            return
        
        print(f"\n{'Metric':<30} {'Training':<20} {'Testing':<20}")
        print("-" * 70)
        print(f"{'Accuracy':<30} {metrics['train_accuracy']:<20.4f} {metrics['test_accuracy']:<20.4f}")
        print(f"{'Precision':<30} {metrics['train_precision']:<20.4f} {metrics['test_precision']:<20.4f}")
        print(f"{'Recall':<30} {metrics['train_recall']:<20.4f} {metrics['test_recall']:<20.4f}")
        print(f"{'F1-Score':<30} {metrics['train_f1']:<20.4f} {metrics['test_f1']:<20.4f}")
        
        print(f"\n{'Model Characteristics':<50}")
        print("-" * 70)
        print(f"{'Number of Support Vectors':<50} {metrics['n_support_vectors']}")
        print(f"{'Margin Width':<50} {metrics['margin_width']:.6f}")
        
        if 'C_parameter' in metrics:
            print(f"{'Regularization Parameter (C)':<50} {metrics['C_parameter']}")
        
        print("\nConfusion Matrix (Test Set):")
        print(metrics['confusion_matrix_test'])
        print("=" * 70 + "\n")
    
    @staticmethod
    def print_comparison_report(comparison_results):
        """
        Print comparison between Hard and Soft Margin
        
        Parameters:
        -----------
        comparison_results : dict
            Results from compare_models
        """
        print("\n" + "=" * 80)
        print("HARD MARGIN vs SOFT MARGIN SVM - COMPARISON")
        print("=" * 80)
        
        # Extract metrics
        hard = comparison_results.get('hard_margin', {})
        soft = comparison_results.get('soft_margin', {})
        
        print(f"\n{'Metric':<35} {'Hard Margin':<22} {'Soft Margin':<22}")
        print("-" * 80)
        
        metrics_to_compare = [
            ('Test Accuracy', 'test_accuracy'),
            ('Test Precision', 'test_precision'),
            ('Test Recall', 'test_recall'),
            ('Test F1-Score', 'test_f1'),
            ('Support Vectors', 'n_support_vectors'),
            ('Margin Width', 'margin_width')
        ]
        
        for label, key in metrics_to_compare:
            hard_val = hard.get(key, 'N/A')
            soft_val = soft.get(key, 'N/A')
            
            if isinstance(hard_val, (int, float)) and isinstance(soft_val, (int, float)):
                if isinstance(hard_val, int):
                    print(f"{label:<35} {hard_val:<22} {soft_val:<22}")
                else:
                    print(f"{label:<35} {hard_val:<22.4f} {soft_val:<22.4f}")
            else:
                print(f"{label:<35} {str(hard_val):<22} {str(soft_val):<22}")
        
        print("\nStatus:")
        print(f"  Hard Margin: {hard.get('status', 'Unknown')}")
        print(f"  Soft Margin: {soft.get('status', 'Unknown')}")
        
        print("=" * 80 + "\n")
    
    @staticmethod
    def measure_training_time(model_class, X, y, **kwargs):
        """
        Measure training time for a model
        
        Parameters:
        -----------
        model_class : class
            SVM model class
        X : array
            Features
        y : array
            Labels
        **kwargs : dict
            Additional parameters for model initialization
            
        Returns:
        --------
        training_time : float
            Time taken to train in seconds
        model : object
            Trained model
        """
        model = model_class(**kwargs)
        
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        return training_time, model
    
    @staticmethod
    def create_comparison_dataframe(results_list):
        """
        Create pandas DataFrame for easy comparison
        
        Parameters:
        -----------
        results_list : list of dict
            List of result dictionaries
            
        Returns:
        --------
        df : pandas.DataFrame
            Comparison dataframe
        """
        data = []
        
        for result in results_list:
            row = {
                'Dataset': result.get('dataset_name', 'Unknown'),
                'Model': result.get('model_type', 'Unknown'),
                'C': result.get('C_parameter', 'N/A'),
                'Test Accuracy': result.get('test_accuracy', 'N/A'),
                'Test F1': result.get('test_f1', 'N/A'),
                'Support Vectors': result.get('n_support_vectors', 'N/A'),
                'Margin Width': result.get('margin_width', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'Status': result.get('status', 'Unknown')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def analyze_C_parameter_impact(X, y, C_values, test_size=0.3, random_state=42):
        """
        Analyze impact of different C values
        
        Parameters:
        -----------
        X : array
            Features
        y : array
            Labels
        C_values : list
            List of C values to test
        test_size : float
            Test set proportion
        random_state : int
            Random seed
            
        Returns:
        --------
        results : list
            Results for each C value
        """
        from soft_margin_svm import SoftMarginSVM
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = []
        
        print("\n" + "=" * 70)
        print("ANALYZING C PARAMETER IMPACT")
        print("=" * 70)
        
        for C in C_values:
            print(f"\nTesting C = {C}...")
            
            # Train model
            training_time, model = SVMEvaluator.measure_training_time(
                SoftMarginSVM, X_train, y_train, C=C
            )
            
            # Evaluate
            metrics = SVMEvaluator.evaluate_model(
                model, X_train, y_train, X_test, y_test
            )
            
            # Store results
            result = {
                'C': C,
                'training_time': training_time,
                **metrics
            }
            results.append(result)
            
            print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Support Vectors: {metrics['n_support_vectors']}")
            print(f"  Training Time: {training_time:.4f}s")
        
        print("\n" + "=" * 70)
        
        # Print summary table
        print("\nSUMMARY TABLE:")
        print("-" * 70)
        print(f"{'C':<10} {'Accuracy':<12} {'F1-Score':<12} {'SV Count':<12} {'Time (s)':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['C']:<10} {r['test_accuracy']:<12.4f} {r['test_f1']:<12.4f} "
                  f"{r['n_support_vectors']:<12} {r['training_time']:<12.4f}")
        print("=" * 70 + "\n")
        
        return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Import this module to use evaluation functions.")