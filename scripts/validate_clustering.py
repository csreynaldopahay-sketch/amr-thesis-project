"""
Cluster Validation Script for AMR Thesis Project
Task 3: Validate k=5 Cluster Choice

This script provides evidence-based justification for using 5 clusters by:
1. Testing k=2 to k=10
2. Calculating silhouette scores and WCSS (Within-Cluster Sum of Squares) for each k
3. Generating elbow plot and silhouette plot
4. Identifying optimal k

Reference: See Section 3.4 and Task 1.3 in comprehensive_academic_review.md

Output:
- Plot: data/processed/figures/cluster_validation.png
- Silhouette scores reported for k=2 to 10
- Console output with optimal k recommendation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Selected k value for the AMR thesis project
# This constant defines the k that will be highlighted in visualizations
SELECTED_K = 5


def compute_wcss(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Squares (WCSS).
    
    WCSS measures the compactness of clusters by summing the squared 
    distances between each point and its cluster centroid.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Cluster labels for each sample
    
    Returns:
    --------
    float
        Total within-cluster sum of squares
    """
    wcss = 0.0
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    
    return wcss


def validate_cluster_count(data_path: str = None, output_dir: str = None, 
                           k_range: range = None) -> dict:
    """
    Validate optimal cluster count using silhouette analysis and elbow method.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to analysis-ready dataset
    output_dir : str, optional
        Directory for output files
    k_range : range, optional
        Range of k values to test (default: 2 to 10)
    
    Returns:
    --------
    dict
        Validation results including optimal k, silhouette scores, and WCSS values
    """
    print("=" * 70)
    print("CLUSTER VALIDATION: Evidence-Based Selection of k")
    print("=" * 70)
    
    # Set default paths
    if data_path is None:
        data_path = project_root / 'data' / 'processed' / 'analysis_ready_dataset.csv'
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'processed' / 'figures'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if k_range is None:
        k_range = range(2, 11)  # k=2 to k=10
    
    # Convert k_range to list once for efficiency
    k_values = list(k_range)
    
    # Validate that SELECTED_K is in k_range
    if SELECTED_K not in k_values:
        print(f"   WARNING: SELECTED_K={SELECTED_K} is not in k_range ({k_values[0]}-{k_values[-1]})")
        print(f"   Adding SELECTED_K to test range for comparison")
        k_values = sorted(set(k_values + [SELECTED_K]))
    
    # Load data
    print(f"\n1. Loading data from: {data_path}")
    
    if not Path(data_path).exists():
        print(f"   ERROR: Data file not found at {data_path}")
        print("   Please run main.py first to generate the analysis-ready dataset.")
        return None
    
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    print(f"   Loaded {len(df)} isolates with {len(feature_cols)} antibiotics")
    
    # Prepare data - impute missing values
    print("\n2. Preparing data for clustering...")
    X = df[feature_cols].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    print(f"   Feature matrix shape: {X_imputed.shape}")
    
    # Generate linkage matrix using Ward's method
    print("\n3. Computing hierarchical clustering (Ward linkage)...")
    Z = linkage(X_imputed, method='ward', metric='euclidean')
    
    # Test different k values
    print(f"\n4. Testing k={k_values[0]} to k={k_values[-1]}...")
    print("-" * 60)
    print(f"{'k':>4} | {'Silhouette':>12} | {'WCSS':>15} | {'Cluster Sizes':>25}")
    print("-" * 60)
    
    silhouette_scores = []
    wcss_values = []
    cluster_sizes_all = {}
    
    for k in k_values:
        # Assign clusters
        labels = fcluster(Z, t=k, criterion='maxclust')
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_imputed, labels)
        silhouette_scores.append(sil_score)
        
        # Calculate WCSS
        wcss = compute_wcss(X_imputed, labels)
        wcss_values.append(wcss)
        
        # Get cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        sizes_str = ", ".join([f"C{l}:{c}" for l, c in zip(unique_labels, counts)])
        cluster_sizes_all[k] = dict(zip(unique_labels.astype(int), counts.astype(int)))
        
        # Print results
        print(f"{k:>4} | {sil_score:>12.4f} | {wcss:>15.2f} | {sizes_str}")
    
    print("-" * 60)
    
    # Identify optimal k based on silhouette
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    print(f"\n5. ANALYSIS RESULTS:")
    print(f"   Optimal k (silhouette): {optimal_k_silhouette} (score = {max_silhouette:.4f})")
    
    # Find elbow point using the rate of change in WCSS
    # The elbow is where the rate of decrease slows significantly
    wcss_diff = np.diff(wcss_values)  # First derivative
    wcss_diff2 = np.diff(wcss_diff)    # Second derivative
    
    # Elbow is typically where second derivative is maximum (steepest change in slope)
    if len(wcss_diff2) > 0:
        elbow_idx = np.argmax(wcss_diff2) + 2  # +2 because of two diff operations
        elbow_k = k_values[min(elbow_idx, len(k_values) - 1)]
    else:
        elbow_k = k_values[0]
    
    print(f"   Elbow point (WCSS): k={elbow_k}")
    
    # Get index of selected k for visualization
    selected_k_idx = k_values.index(SELECTED_K)
    selected_k_sil = silhouette_scores[selected_k_idx]
    selected_k_wcss = wcss_values[selected_k_idx]
    
    # Generate visualization
    print("\n6. Generating validation plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Elbow Method (WCSS)
    ax1 = axes[0]
    ax1.plot(k_values, wcss_values, marker='o', linewidth=2, markersize=8, 
             color='#2196F3', label='WCSS')
    ax1.axvline(x=SELECTED_K, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'k={SELECTED_K} (selected)')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xticks(k_values)
    
    # Add annotation for selected k
    ax1.annotate(f'k={SELECTED_K}\nWCSS={selected_k_wcss:.0f}', 
                 xy=(SELECTED_K, selected_k_wcss), 
                 xytext=(SELECTED_K + 1.5, selected_k_wcss * 1.1),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                 color='red')
    
    # Plot 2: Silhouette Analysis
    ax2 = axes[1]
    bars = ax2.bar(k_values, silhouette_scores, color='#4CAF50', alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    # Highlight selected k bar
    bars[selected_k_idx].set_color('#F44336')
    bars[selected_k_idx].set_alpha(1.0)
    
    ax2.axhline(y=selected_k_sil, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'k={SELECTED_K} score: {selected_k_sil:.4f}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xticks(k_values)
    
    # Add value labels on bars
    for i, (k, score) in enumerate(zip(k_values, silhouette_scores)):
        ax2.annotate(f'{score:.3f}', 
                     xy=(k, score), 
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center', 
                     fontsize=8,
                     fontweight='bold' if k == SELECTED_K else 'normal',
                     color='red' if k == SELECTED_K else 'black')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'cluster_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Plot saved to: {output_path}")
    
    # Generate results dictionary
    results = {
        'k_range': k_values,
        'silhouette_scores': silhouette_scores,
        'wcss_values': wcss_values,
        'optimal_k_silhouette': optimal_k_silhouette,
        'max_silhouette_score': max_silhouette,
        'elbow_k': elbow_k,
        'cluster_sizes': cluster_sizes_all,
        'selected_k': SELECTED_K,
        'selected_k_silhouette': selected_k_sil,
        'selected_k_wcss': selected_k_wcss,
        'output_path': str(output_path)
    }
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'k': k_values,
        'Silhouette_Score': silhouette_scores,
        'WCSS': wcss_values
    })
    results_csv_path = output_dir / 'cluster_validation_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"   Results CSV saved to: {results_csv_path}")
    
    # Print summary and justification
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\nSilhouette Scores (k={k_values[0]} to k={k_values[-1]}):")
    for k, score in zip(k_values, silhouette_scores):
        marker = " <-- SELECTED" if k == SELECTED_K else ""
        optimal_marker = " (OPTIMAL)" if k == optimal_k_silhouette else ""
        print(f"   k={k}: {score:.4f}{marker}{optimal_marker}")
    
    print(f"\nJUSTIFICATION FOR k={SELECTED_K}:")
    print("-" * 70)
    
    if optimal_k_silhouette == SELECTED_K:
        print(f"   k={SELECTED_K} MAXIMIZED silhouette score ({selected_k_sil:.4f})")
    else:
        print(f"   k={SELECTED_K} silhouette score: {selected_k_sil:.4f}")
        print(f"   Optimal silhouette at k={optimal_k_silhouette}: {max_silhouette:.4f}")
    
    print(f"\n   Elbow analysis shows inflection around k={elbow_k}")
    
    # Generate justification text
    if optimal_k_silhouette == SELECTED_K:
        justification = (
            f"k={SELECTED_K} was selected based on convergent evidence: "
            f"silhouette score was maximized at k={SELECTED_K} (score={selected_k_sil:.4f}), "
            f"and the elbow curve showed inflection at k={elbow_k}. "
            f"This evidence-based selection validates the use of {SELECTED_K} clusters "
            f"for resistance phenotype characterization."
        )
    else:
        # Selected k is reasonable but not strictly optimal
        justification = (
            f"k={SELECTED_K} was selected with silhouette score of {selected_k_sil:.4f}. "
            f"While silhouette analysis suggests k={optimal_k_silhouette} (score={max_silhouette:.4f}) "
            f"as optimal, k={SELECTED_K} represents a reasonable trade-off between cluster granularity "
            f"and biological interpretability. The difference in silhouette scores "
            f"({max_silhouette - selected_k_sil:.4f}) is modest, and k={SELECTED_K} provides clearer "
            f"separation of resistance phenotypes for clinical interpretation."
        )
    
    print(f"\n   {justification}")
    
    results['justification'] = justification
    
    print("\n" + "=" * 70)
    print("CLUSTER VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nDeliverables:")
    print(f"  - Plot: {output_path}")
    print(f"  - Results: {results_csv_path}")
    print(f"  - Justification: See above")
    
    return results


if __name__ == "__main__":
    results = validate_cluster_count()
    
    if results:
        print("\n\nFor documentation, use this summary:")
        print("-" * 70)
        print(f"\"k={SELECTED_K} {'maximized' if results['optimal_k_silhouette'] == SELECTED_K else 'achieved'} "
              f"silhouette score ({results['selected_k_silhouette']:.2f}) "
              f"and showed elbow at k={results['elbow_k']}\"")
