import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs

class SVMDatasetGenerator:
    @staticmethod
    def generate_linearly_separable(n_samples=100, random_state=42):
        np.random.seed(random_state)
        
        X_0 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
        
        X_1 = np.random.randn(n_samples, 2) * 0.5 + np.array([5, 5])
        
        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def generate_nearly_separable(n_samples=100, n_outliers=5, random_state=42):
        np.random.seed(random_state)
    
        X_0 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
        X_1 = np.random.randn(n_samples, 2) * 0.5 + np.array([5, 5])
        
        outliers_0 = np.random.randn(n_outliers, 2) * 0.3 + np.array([5, 5])
        X_0 = np.vstack([X_0, outliers_0])
        
        outliers_1 = np.random.randn(n_outliers, 2) * 0.3 + np.array([2, 2])
        X_1 = np.vstack([X_1, outliers_1])
        
        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(n_samples + n_outliers), 
                       np.ones(n_samples + n_outliers)])
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def generate_non_separable(n_samples=150, random_state=42):
        np.random.seed(random_state)
        
        X_0 = np.random.randn(n_samples, 2) * 1.2 + np.array([3, 3])
        X_1 = np.random.randn(n_samples, 2) * 1.2 + np.array([4, 4])

        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def generate_xor_pattern(n_samples=100, random_state=42):
        np.random.seed(random_state)
        
        cluster_size = n_samples // 4
        
        X_0_tl = np.random.randn(cluster_size, 2) * 0.3 + np.array([1, 4])
        X_0_br = np.random.randn(cluster_size, 2) * 0.3 + np.array([4, 1])
        X_0 = np.vstack([X_0_tl, X_0_br])
        
        X_1_tr = np.random.randn(cluster_size, 2) * 0.3 + np.array([4, 4])
        X_1_bl = np.random.randn(cluster_size, 2) * 0.3 + np.array([1, 1])
        X_1 = np.vstack([X_1_tr, X_1_bl])
        
        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(len(X_0)), np.ones(len(X_1))])

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y


def visualize_datasets():
    generator = SVMDatasetGenerator()
    
    datasets = [
        ("Linearly Separable", generator.generate_linearly_separable()),
        ("Nearly Separable", generator.generate_nearly_separable()),
        ("Non-Separable", generator.generate_non_separable()),
        ("XOR Pattern", generator.generate_xor_pattern())
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (title, (X, y)) in enumerate(datasets):
        ax = axes[idx]
        
        scatter = ax.scatter(X[y == 0, 0], X[y == 0, 1], 
                           c='blue', label='Class 0', alpha=0.6, s=50)
        scatter = ax.scatter(X[y == 1, 0], X[y == 1, 1], 
                           c='red', label='Class 1', alpha=0.6, s=50)
        
        ax.set_title(f"{title}\n(n={len(X)} samples)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        class_0_count = np.sum(y == 0)
        class_1_count = np.sum(y == 1)
        ax.text(0.02, 0.98, f'Class 0: {class_0_count}\nClass 1: {class_1_count}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('datasets_overview.png', dpi=300, bbox_inches='tight')
    print("Dataset visualization saved as 'datasets_overview.png'")
    plt.show()


def save_datasets_to_file():
    generator = SVMDatasetGenerator()
    
    datasets = {
        'linearly_separable': generator.generate_linearly_separable(),
        'nearly_separable': generator.generate_nearly_separable(),
        'non_separable': generator.generate_non_separable(),
        'xor_pattern': generator.generate_xor_pattern()
    }
    
    for name, (X, y) in datasets.items():
        np.save(f'data_{name}_X.npy', X)
        np.save(f'data_{name}_y.npy', y)
        print(f"Saved {name}: X shape {X.shape}, y shape {y.shape}")
    
    print("\nAll datasets saved successfully!")


def print_dataset_statistics():
    """Print statistics for all datasets"""
    generator = SVMDatasetGenerator()
    
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    datasets = [
        ("Linearly Separable", generator.generate_linearly_separable()),
        ("Nearly Separable", generator.generate_nearly_separable()),
        ("Non-Separable", generator.generate_non_separable()),
        ("XOR Pattern", generator.generate_xor_pattern())
    ]
    
    for title, (X, y) in datasets:
        print(f"\n{title}:")
        print("-" * 70)
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Class 0 samples: {np.sum(y == 0)}")
        print(f"  Class 1 samples: {np.sum(y == 1)}")
        print(f"  Feature 1 range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
        print(f"  Feature 2 range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    
        class_0_mean = X[y == 0].mean(axis=0)
        class_1_mean = X[y == 1].mean(axis=0)
        distance = np.linalg.norm(class_0_mean - class_1_mean)
        print(f"  Distance between class centers: {distance:.2f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Generating datasets...\n")

    print_dataset_statistics()

    print("\nGenerating visualizations...")
    visualize_datasets()

    print("\nSaving datasets to files...")
    save_datasets_to_file()
    
    print("\n✓ Dataset generation complete!")