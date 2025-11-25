"""
Visualization Module for SVM Project
Creates all plots and visualizations for comparing Hard and Soft Margin SVMs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


class SVMVisualizer:
    """Visualization tools for SVM analysis"""
    
    @staticmethod
    def plot_decision_boundary(model, X, y, title="SVM Decision Boundary", 
                               ax=None, show_margin=True):
        """
        Plot decision boundary with margin and support vectors
        
        Parameters:
        -----------
        model : SVM model object
            Trained SVM model
        X : array, shape (n_samples, 2)
            Feature matrix
        y : array, shape (n_samples,)
            Labels
        title : str
            Plot title
        ax : matplotlib axis
            Axis to plot on
        show_margin : bool
            Whether to show margin boundaries
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Get decision function values
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', 
                  linestyles='solid', label='Decision Boundary')
        
        if show_margin:
            ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, 
                      colors='black', linestyles='dashed', alpha=0.5)
        
        # Plot filled contours for regions
        ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], 
                   colors=['lightblue', 'lightcoral'], alpha=0.3)
        
        # Plot data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=50, 
                  edgecolors='k', label='Class 0', alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50, 
                  edgecolors='k', label='Class 1', alpha=0.7)
        
        # Highlight support vectors
        if hasattr(model, 'support_vectors_'):
            ax.scatter(model.support_vectors_[:, 0], 
                      model.support_vectors_[:, 1],
                      s=200, linewidths=2, facecolors='none', 
                      edgecolors='green', label='Support Vectors')
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def compare_hard_soft_margin(hard_model, soft_model, X, y, 
                                 dataset_name="Dataset"):
        """
        Side-by-side comparison of Hard and Soft Margin SVM
        
        Parameters:
        -----------
        hard_model : HardMarginSVM or None
            Trained Hard Margin SVM (None if failed)
        soft_model : SoftMarginSVM
            Trained Soft Margin SVM
        X : array
            Feature matrix
        y : array
            Labels
        dataset_name : str
            Name of dataset
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot Hard Margin (if available)
        if hard_model is not None:
            SVMVisualizer.plot_decision_boundary(
                hard_model, X, y, 
                title=f"Hard Margin SVM\n{dataset_name}",
                ax=axes[0]
            )
            margin_hard = hard_model.get_margin_width()
            axes[0].text(0.02, 0.98, 
                        f'Margin: {margin_hard:.3f}\nSV: {hard_model.n_support_}',
                        transform=axes[0].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=10)
        else:
            axes[0].text(0.5, 0.5, 'Hard Margin SVM\nFailed to Converge\n(Data not linearly separable)',
                        ha='center', va='center', fontsize=14, color='red',
                        transform=axes[0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[0].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=50, alpha=0.5)
            axes[0].scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=50, alpha=0.5)
            axes[0].set_title(f"Hard Margin SVM\n{dataset_name}", fontweight='bold')
        
        # Plot Soft Margin
        SVMVisualizer.plot_decision_boundary(
            soft_model, X, y,
            title=f"Soft Margin SVM (C={soft_model.C})\n{dataset_name}",
            ax=axes[1]
        )
        margin_soft = soft_model.get_margin_width()
        axes[1].text(0.02, 0.98,
                    f'Margin: {margin_soft:.3f}\nSV: {soft_model.n_support_}',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=10)
        
        plt.tight_layout()
        filename = f'comparison_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()
    
    @staticmethod
    def plot_C_parameter_effect(X, y, C_values, dataset_name="Dataset"):
        """
        Show effect of different C values on Soft Margin SVM
        
        Parameters:
        -----------
        X : array
            Feature matrix
        y : array
            Labels
        C_values : list
            List of C values to test
        dataset_name : str
            Name of dataset
        """
        from soft_margin_svm import SoftMarginSVM
        
        n_plots = len(C_values)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel() if n_plots > 1 else [axes]
        
        results = []
        
        for idx, C in enumerate(C_values):
            if idx < len(axes):
                ax = axes[idx]
                
                # Train model
                model = SoftMarginSVM(C=C)
                model.fit(X, y)
                
                # Plot
                SVMVisualizer.plot_decision_boundary(
                    model, X, y,
                    title=f"C = {C}",
                    ax=ax
                )
                
                # Add info
                margin = model.get_margin_width()
                misclass = model.get_misclassified_count(X, y)
                
                info_text = f'Margin: {margin:.3f}\nSV: {model.n_support_}\nErrors: {misclass}'
                ax.text(0.02, 0.98, info_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       fontsize=9)
                
                results.append({
                    'C': C,
                    'margin': margin,
                    'n_support': model.n_support_,
                    'errors': misclass
                })
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Effect of C Parameter - {dataset_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        filename = f'C_effect_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()
        
        return results
    
    @staticmethod
    def plot_support_vectors_analysis(X, y, C_values):
        """
        Analyze how number of support vectors changes with C
        
        Parameters:
        -----------
        X : array
            Feature matrix
        y : array
            Labels
        C_values : list
            List of C values to test
        """
        from soft_margin_svm import SoftMarginSVM
        
        sv_counts = []
        margins = []
        errors = []
        
        for C in C_values:
            model = SoftMarginSVM(C=C)
            model.fit(X, y)
            sv_counts.append(model.n_support_)
            margins.append(model.get_margin_width())
            errors.append(model.get_misclassified_count(X, y))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Support vectors vs C
        axes[0].plot(C_values, sv_counts, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('C (log scale)', fontsize=12)
        axes[0].set_ylabel('Number of Support Vectors', fontsize=12)
        axes[0].set_title('Support Vectors vs C', fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Margin width vs C
        axes[1].plot(C_values, margins, 'o-', linewidth=2, markersize=8, color='green')
        axes[1].set_xlabel('C (log scale)', fontsize=12)
        axes[1].set_ylabel('Margin Width', fontsize=12)
        axes[1].set_title('Margin Width vs C', fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # Errors vs C
        axes[2].plot(C_values, errors, 'o-', linewidth=2, markersize=8, color='red')
        axes[2].set_xlabel('C (log scale)', fontsize=12)
        axes[2].set_ylabel('Misclassified Samples', fontsize=12)
        axes[2].set_title('Training Errors vs C', fontweight='bold')
        axes[2].set_xscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('support_vectors_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: support_vectors_analysis.png")
        plt.show()
    
    @staticmethod
    def create_summary_comparison_table(results_dict):
        """
        Create visual comparison table
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with results for each dataset
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        headers = ['Dataset', 'Model', 'Margin Width', 'Support Vectors', 'Errors', 'Status']
        table_data = []
        
        for dataset_name, results in results_dict.items():
            for result in results:
                row = [
                    dataset_name,
                    result['model_type'],
                    f"{result.get('margin', 0):.4f}",
                    str(result.get('n_support', 'N/A')),
                    str(result.get('errors', 'N/A')),
                    result.get('status', 'OK')
                ]
                table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.1, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('SVM Comparison Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('comparison_summary_table.png', dpi=300, bbox_inches='tight')
        print("Saved: comparison_summary_table.png")
        plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Import this module to use visualization functions.")