"""
Main Execution Script for SVM Project
Runs complete comparison between Hard Margin and Soft Margin SVMs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import SVMDatasetGenerator
from hard_margin_svm import HardMarginSVM
from soft_margin_svm import SoftMarginSVM
from visualizations import SVMVisualizer
from evaluation import SVMEvaluator


def run_experiment_on_dataset(X, y, dataset_name, C_value=1.0):
    """
    Run complete experiment on a single dataset
    
    Parameters:
    -----------
    X : array
        Feature matrix
    y : array
        Labels
    dataset_name : str
        Name of the dataset
    C_value : float
        C parameter for Soft Margin SVM
        
    Returns:
    --------
    results : dict
        Results dictionary
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {dataset_name}")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nDataset split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Try Hard Margin SVM
    print("\n" + "-" * 80)
    print("1. Training Hard Margin SVM...")
    print("-" * 80)
    
    hard_model = None
    hard_training_time = 0
    
    try:
        hard_training_time, hard_model = SVMEvaluator.measure_training_time(
            HardMarginSVM, X_train, y_train
        )
        print(f"✓ Hard Margin SVM trained successfully in {hard_training_time:.4f}s")
    except Exception as e:
        print(f"✗ Hard Margin SVM failed: {str(e)}")
    
    # Train Soft Margin SVM
    print("\n" + "-" * 80)
    print(f"2. Training Soft Margin SVM (C={C_value})...")
    print("-" * 80)
    
    soft_training_time, soft_model = SVMEvaluator.measure_training_time(
        SoftMarginSVM, X_train, y_train, C=C_value
    )
    print(f"✓ Soft Margin SVM trained successfully in {soft_training_time:.4f}s")
    
    # Evaluate both models
    print("\n" + "-" * 80)
    print("3. Evaluation")
    print("-" * 80)
    
    comparison_results = SVMEvaluator.compare_models(
        hard_model, soft_model, X_train, y_train, X_test, y_test
    )
    
    # Print individual reports
    if hard_model is not None:
        SVMEvaluator.print_evaluation_report(
            comparison_results['hard_margin'], 
            f"Hard Margin SVM - {dataset_name}"
        )
    
    SVMEvaluator.print_evaluation_report(
        comparison_results['soft_margin'],
        f"Soft Margin SVM - {dataset_name}"
    )
    
    # Print comparison
    SVMEvaluator.print_comparison_report(comparison_results)
    
    # Visualize
    print("\n" + "-" * 80)
    print("4. Creating Visualizations...")
    print("-" * 80)
    
    SVMVisualizer.compare_hard_soft_margin(
        hard_model, soft_model, X, y, dataset_name
    )
    
    return {
        'dataset_name': dataset_name,
        'hard_model': hard_model,
        'soft_model': soft_model,
        'comparison_results': comparison_results,
        'hard_training_time': hard_training_time,
        'soft_training_time': soft_training_time
    }


def analyze_C_parameter(X, y, dataset_name):
    """
    Analyze effect of different C values
    
    Parameters:
    -----------
    X : array
        Feature matrix
    y : array
        Labels
    dataset_name : str
        Name of dataset
    """
    print("\n" + "=" * 80)
    print(f"C PARAMETER ANALYSIS: {dataset_name}")
    print("=" * 80)
    
    C_values = [0.01, 0.1, 1, 10, 100]
    
    # Visualize effect of C
    print("\nCreating C parameter effect visualization...")
    results = SVMVisualizer.plot_C_parameter_effect(X, y, C_values, dataset_name)
    
    # Analyze metrics
    print("\nAnalyzing metrics for different C values...")
    analysis_results = SVMEvaluator.analyze_C_parameter_impact(X, y, C_values)
    
    # Plot support vector analysis
    print("\nCreating support vector analysis plot...")
    SVMVisualizer.plot_support_vectors_analysis(X, y, C_values)
    
    return analysis_results


def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("SVM PROJECT: HARD MARGIN vs SOFT MARGIN COMPARISON")
    print("Support Vector Machines Implementation and Analysis")
    print("=" * 80)
    
    # Generate datasets
    print("\n[STEP 1] Generating Datasets...")
    print("-" * 80)
    
    generator = SVMDatasetGenerator()
    
    datasets = {
        'Linearly Separable': generator.generate_linearly_separable(n_samples=100),
        'Nearly Separable': generator.generate_nearly_separable(n_samples=100, n_outliers=5),
        'Non-Separable': generator.generate_non_separable(n_samples=150),
    }
    
    print("✓ Generated 3 datasets:")
    for name, (X, y) in datasets.items():
        print(f"  - {name}: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run experiments on each dataset
    print("\n\n[STEP 2] Running Experiments on Each Dataset...")
    print("=" * 80)
    
    all_results = []
    
    for dataset_name, (X, y) in datasets.items():
        result = run_experiment_on_dataset(X, y, dataset_name, C_value=1.0)
        all_results.append(result)
    
    # Detailed C parameter analysis on Non-Separable dataset
    print("\n\n[STEP 3] C Parameter Analysis...")
    print("=" * 80)
    
    X_nonsep, y_nonsep = datasets['Non-Separable']
    c_analysis_results = analyze_C_parameter(X_nonsep, y_nonsep, 'Non-Separable')
    
    # Create final summary
    print("\n\n[STEP 4] Creating Final Summary...")
    print("=" * 80)
    
    # Prepare summary data
    summary_data = {}
    
    for result in all_results:
        dataset_name = result['dataset_name']
        comp_results = result['comparison_results']
        
        summary_data[dataset_name] = []
        
        # Hard Margin
        if result['hard_model'] is not None:
            hard_metrics = comp_results['hard_margin']
            summary_data[dataset_name].append({
                'model_type': 'Hard Margin',
                'margin': hard_metrics.get('margin_width', 0),
                'n_support': hard_metrics.get('n_support_vectors', 0),
                'errors': 0,  # Hard margin doesn't allow errors
                'status': 'Success'
            })
        else:
            summary_data[dataset_name].append({
                'model_type': 'Hard Margin',
                'margin': 0,
                'n_support': 'N/A',
                'errors': 'N/A',
                'status': 'Failed'
            })
        
        # Soft Margin
        soft_metrics = comp_results['soft_margin']
        summary_data[dataset_name].append({
            'model_type': 'Soft Margin',
            'margin': soft_metrics.get('margin_width', 0),
            'n_support': soft_metrics.get('n_support_vectors', 0),
            'errors': 'Varies with C',
            'status': 'Success'
        })
    
    # Create comparison table visualization
    SVMVisualizer.create_summary_comparison_table(summary_data)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\n✓ All visualizations saved as PNG files")
    print("✓ Check the following files:")
    print("  - comparison_*.png (for each dataset)")
    print("  - C_effect_*.png (C parameter analysis)")
    print("  - support_vectors_analysis.png")
    print("  - comparison_summary_table.png")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("""
1. HARD MARGIN SVM:
   - Works only on linearly separable data
   - No tolerance for misclassification
   - Produces maximum margin when data is separable
   - Sensitive to outliers

2. SOFT MARGIN SVM:
   - Works on both separable and non-separable data
   - Controlled by parameter C
   - Small C: Large margin, more errors allowed
   - Large C: Small margin, fewer errors (approaches hard margin)
   - More robust to outliers and noise

3. C PARAMETER EFFECT:
   - C controls trade-off between margin width and errors
   - Optimal C depends on data and application
   - Use cross-validation to select best C in practice
    """)
    print("=" * 80)


if __name__ == "__main__":
    # Run complete analysis
    main()
    
    print("\n\nPress Enter to exit...")
    input()