"""
Sensitivity Analysis for Data Cleaning Thresholds - AMR Thesis Project
Task 7: Validate that threshold choices don't affect results

This script tests multiple threshold pairs for data cleaning and compares 
cluster assignments using Adjusted Rand Index (ARI) to assess robustness.

Reference: See Section 2 (Phase 2) in comprehensive_academic_review.md

Thresholds tested:
- min_antibiotic_coverage: 50%, 60%, 70%, 80%
- max_isolate_missing: 20%, 30%, 40%

Output:
- Comparison matrix of ARI scores between threshold configurations
- Summary statistics showing result stability
- Visualization of threshold impact on clustering
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
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# Import preprocessing modules
from src.preprocessing.data_cleaning import clean_dataset


def load_raw_data():
    """
    Load the unified raw dataset before cleaning.
    
    Returns:
    --------
    pd.DataFrame or None
        Raw unified dataset
    """
    raw_path = project_root / 'data' / 'processed' / 'unified_raw_dataset.csv'
    
    if not raw_path.exists():
        # Try to generate it
        print("Raw dataset not found. Attempting to regenerate...")
        try:
            from src.preprocessing.data_ingestion import unify_data_sources
            csv_files = list(project_root.glob('*.csv'))
            if csv_files:
                df_raw = unify_data_sources(csv_files, str(raw_path.parent))
                return df_raw
        except Exception as e:
            print(f"Could not generate raw dataset: {e}")
            return None
    
    return pd.read_csv(raw_path)


def apply_thresholds_and_cluster(df_raw, min_coverage, max_missing, n_clusters=5):
    """
    Apply cleaning thresholds and perform clustering.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw dataset
    min_coverage : float
        Minimum antibiotic coverage percentage (0-100)
    max_missing : float
        Maximum missing percentage per isolate (0-100)
    n_clusters : int
        Number of clusters to form
    
    Returns:
    --------
    dict
        Results including labels, sample size, silhouette score
    """
    try:
        # Clean the dataset with specified thresholds
        df_clean, _ = clean_dataset(
            df_raw.copy(), 
            min_antibiotic_coverage=min_coverage,
            max_isolate_missing=max_missing
        )
        
        if len(df_clean) < 50:  # Minimum sample size
            return {
                'success': False,
                'error': f'Insufficient samples: {len(df_clean)}',
                'n_samples': len(df_clean)
            }
        
        # Get encoded columns
        feature_cols = [col for col in df_clean.columns if col.endswith('_encoded')]
        
        if len(feature_cols) < 5:  # Minimum antibiotics
            return {
                'success': False,
                'error': f'Insufficient antibiotics: {len(feature_cols)}',
                'n_antibiotics': len(feature_cols)
            }
        
        # Prepare features
        X = df_clean[feature_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Perform clustering
        Z = linkage(X_imputed, method='ward', metric='euclidean')
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_imputed, labels)
        
        # Get cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))
        
        return {
            'success': True,
            'labels': labels,
            'n_samples': len(df_clean),
            'n_antibiotics': len(feature_cols),
            'silhouette': sil_score,
            'cluster_sizes': cluster_sizes,
            'sample_indices': df_clean.index.tolist()  # Track which samples were included
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_samples': 0
        }


def compute_ari_for_common_samples(result1, result2):
    """
    Compute Adjusted Rand Index for samples present in both configurations.
    
    Parameters:
    -----------
    result1, result2 : dict
        Results from apply_thresholds_and_cluster
    
    Returns:
    --------
    float or None
        ARI score, or None if comparison not possible
    """
    if not (result1['success'] and result2['success']):
        return None
    
    # Find common samples (by index)
    common_indices = set(result1['sample_indices']) & set(result2['sample_indices'])
    
    if len(common_indices) < 50:  # Minimum for meaningful ARI
        return None
    
    # Get labels for common samples
    idx1_map = {idx: i for i, idx in enumerate(result1['sample_indices'])}
    idx2_map = {idx: i for i, idx in enumerate(result2['sample_indices'])}
    
    labels1 = [result1['labels'][idx1_map[idx]] for idx in common_indices]
    labels2 = [result2['labels'][idx2_map[idx]] for idx in common_indices]
    
    return adjusted_rand_score(labels1, labels2)


def run_sensitivity_analysis(output_dir=None):
    """
    Run complete sensitivity analysis across threshold pairs.
    
    Parameters:
    -----------
    output_dir : str or Path, optional
        Directory for output files
    
    Returns:
    --------
    dict
        Sensitivity analysis results
    """
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: Data Cleaning Thresholds")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'processed' / 'figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print("\n1. Loading raw data...")
    df_raw = load_raw_data()
    
    if df_raw is None:
        print("   ERROR: Could not load raw data.")
        print("   Please run main.py first to generate the unified raw dataset.")
        return None
    
    print(f"   Loaded {len(df_raw)} raw isolates")
    
    # Define threshold pairs to test
    coverage_thresholds = [50.0, 60.0, 70.0, 80.0]
    missing_thresholds = [20.0, 30.0, 40.0]
    
    threshold_pairs = list(product(coverage_thresholds, missing_thresholds))
    
    print(f"\n2. Testing {len(threshold_pairs)} threshold configurations...")
    print("-" * 60)
    
    # Run clustering for each threshold pair
    results = {}
    for coverage, missing in threshold_pairs:
        config_name = f"cov{int(coverage)}_miss{int(missing)}"
        print(f"   {config_name}: ", end="")
        
        result = apply_thresholds_and_cluster(df_raw, coverage, missing)
        results[config_name] = result
        
        if result['success']:
            print(f"n={result['n_samples']}, {result['n_antibiotics']} ABs, "
                  f"silhouette={result['silhouette']:.3f}")
        else:
            print(f"FAILED - {result.get('error', 'Unknown error')}")
    
    print("-" * 60)
    
    # Compute ARI between all pairs
    print("\n3. Computing Adjusted Rand Index between configurations...")
    
    config_names = list(results.keys())
    n_configs = len(config_names)
    ari_matrix = np.zeros((n_configs, n_configs))
    ari_matrix[:] = np.nan
    
    for i, name1 in enumerate(config_names):
        for j, name2 in enumerate(config_names):
            if i == j:
                ari_matrix[i, j] = 1.0
            elif results[name1]['success'] and results[name2]['success']:
                ari = compute_ari_for_common_samples(results[name1], results[name2])
                if ari is not None:
                    ari_matrix[i, j] = ari
    
    # Create ARI DataFrame
    ari_df = pd.DataFrame(ari_matrix, index=config_names, columns=config_names)
    
    # Summary statistics
    print("\n4. SENSITIVITY ANALYSIS RESULTS:")
    print("-" * 60)
    
    # Get ARI values (excluding diagonal and NaN)
    ari_values = []
    for i in range(n_configs):
        for j in range(i+1, n_configs):
            if not np.isnan(ari_matrix[i, j]):
                ari_values.append(ari_matrix[i, j])
    
    if ari_values:
        mean_ari = np.mean(ari_values)
        min_ari = np.min(ari_values)
        max_ari = np.max(ari_values)
        std_ari = np.std(ari_values)
        
        print(f"\n   ARI Statistics (pairwise comparisons):")
        print(f"   Mean ARI: {mean_ari:.4f}")
        print(f"   Std ARI:  {std_ari:.4f}")
        print(f"   Min ARI:  {min_ari:.4f}")
        print(f"   Max ARI:  {max_ari:.4f}")
        
        # Interpretation
        print(f"\n   Interpretation:")
        if mean_ari >= 0.90:
            stability = "EXCELLENT"
            interpretation = "Clustering is highly robust to threshold choices"
        elif mean_ari >= 0.80:
            stability = "GOOD"
            interpretation = "Clustering is generally stable across thresholds"
        elif mean_ari >= 0.70:
            stability = "MODERATE"
            interpretation = "Some threshold sensitivity observed; results should be interpreted with caution"
        else:
            stability = "POOR"
            interpretation = "Clustering is sensitive to threshold choices; careful threshold selection required"
        
        print(f"   Stability: {stability}")
        print(f"   {interpretation}")
    else:
        print("   Could not compute ARI - insufficient valid configurations")
    
    # Compare to reference configuration (70/30)
    reference_config = "cov70_miss30"
    if reference_config in results and results[reference_config]['success']:
        print(f"\n   Comparison to reference configuration ({reference_config}):")
        ref_idx = config_names.index(reference_config)
        for i, name in enumerate(config_names):
            if name != reference_config and not np.isnan(ari_matrix[ref_idx, i]):
                print(f"   {name} vs {reference_config}: ARI = {ari_matrix[ref_idx, i]:.4f}")
    
    # Generate visualization
    print("\n5. Generating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: ARI Heatmap
    ax1 = axes[0]
    mask = np.isnan(ari_matrix)
    sns.heatmap(ari_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, mask=mask, ax=ax1,
                cbar_kws={'label': 'Adjusted Rand Index'})
    ax1.set_title('Cluster Assignment Stability\n(ARI between threshold configurations)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Configuration')
    
    # Rotate labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Plot 2: Sample size and silhouette by configuration
    ax2 = axes[1]
    
    # Prepare data for bar plot
    successful_configs = [name for name in config_names if results[name]['success']]
    n_samples = [results[name]['n_samples'] for name in successful_configs]
    silhouettes = [results[name]['silhouette'] for name in successful_configs]
    
    x = np.arange(len(successful_configs))
    width = 0.35
    
    # Primary y-axis: Sample size
    bars1 = ax2.bar(x - width/2, n_samples, width, label='Sample Size', color='#2196F3', alpha=0.8)
    ax2.set_ylabel('Sample Size', color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')
    ax2.set_ylim(0, max(n_samples) * 1.2)
    
    # Secondary y-axis: Silhouette
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, silhouettes, width, label='Silhouette', color='#4CAF50', alpha=0.8)
    ax2_twin.set_ylabel('Silhouette Score', color='#4CAF50')
    ax2_twin.tick_params(axis='y', labelcolor='#4CAF50')
    ax2_twin.set_ylim(0, 1.0)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(successful_configs, rotation=45, ha='right')
    ax2.set_xlabel('Threshold Configuration')
    ax2.set_title('Sample Size and Cluster Quality\nby Threshold Configuration', 
                  fontsize=12, fontweight='bold')
    
    # Highlight reference configuration
    if reference_config in successful_configs:
        ref_idx = successful_configs.index(reference_config)
        ax2.axvline(x=ref_idx, color='red', linestyle='--', alpha=0.5, label='Reference (70/30)')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'sensitivity_analysis_thresholds.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Figure saved to: {fig_path}")
    
    # Save results to CSV
    results_summary = []
    for name in config_names:
        result = results[name]
        coverage = int(name.split('_')[0].replace('cov', ''))
        missing = int(name.split('_')[1].replace('miss', ''))
        
        row = {
            'Configuration': name,
            'Min_Coverage_%': coverage,
            'Max_Missing_%': missing,
            'Success': result['success'],
            'N_Samples': result.get('n_samples', 0),
            'N_Antibiotics': result.get('n_antibiotics', 0),
            'Silhouette': result.get('silhouette', np.nan)
        }
        
        # Add ARI vs reference
        if reference_config in config_names and name != reference_config:
            ref_idx = config_names.index(reference_config)
            name_idx = config_names.index(name)
            row['ARI_vs_Reference'] = ari_matrix[ref_idx, name_idx]
        else:
            row['ARI_vs_Reference'] = 1.0 if name == reference_config else np.nan
        
        results_summary.append(row)
    
    results_df = pd.DataFrame(results_summary)
    csv_path = output_dir / 'sensitivity_analysis_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"   Results saved to: {csv_path}")
    
    # Save ARI matrix
    ari_path = output_dir / 'sensitivity_analysis_ari_matrix.csv'
    ari_df.to_csv(ari_path)
    print(f"   ARI matrix saved to: {ari_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    
    if ari_values:
        print(f"\nConclusion: Clustering stability is {stability} (Mean ARI = {mean_ari:.3f})")
        print(f"The reference thresholds (70% coverage, 30% missing) are {'VALIDATED' if mean_ari >= 0.80 else 'ACCEPTABLE but should be noted in limitations'}.")
    
    return {
        'results': results,
        'ari_matrix': ari_df,
        'summary': results_df,
        'mean_ari': mean_ari if ari_values else None,
        'stability': stability if ari_values else 'UNKNOWN'
    }


if __name__ == "__main__":
    run_sensitivity_analysis()
