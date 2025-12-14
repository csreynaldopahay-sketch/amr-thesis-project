"""
Regional and Environmental Analysis Module for AMR Thesis Project
Phase 5 - Cluster Distribution Analysis and PCA
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def cross_tabulate_clusters(df: pd.DataFrame,
                           cluster_col: str = 'CLUSTER',
                           group_col: str = 'REGION') -> pd.DataFrame:
    """
    Create cross-tabulation of clusters by grouping variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels
    cluster_col : str
        Column name for cluster labels
    group_col : str
        Column name for grouping variable
    
    Returns:
    --------
    pd.DataFrame
        Cross-tabulation table
    """
    if group_col not in df.columns or cluster_col not in df.columns:
        print(f"Required columns not found: {cluster_col}, {group_col}")
        return pd.DataFrame()
    
    crosstab = pd.crosstab(df[cluster_col], df[group_col], margins=True)
    
    return crosstab


def analyze_cluster_distribution(df: pd.DataFrame,
                                 cluster_col: str = 'CLUSTER') -> Dict:
    """
    Comprehensive cluster distribution analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels and metadata
    cluster_col : str
        Column name for cluster labels
    
    Returns:
    --------
    dict
        Distribution analysis results
    """
    print("=" * 50)
    print("PHASE 5.1: Cluster Distribution Analysis")
    print("=" * 50)
    
    analysis = {
        'by_region': None,
        'by_environment': None,
        'by_species': None,
        'chi_square_tests': {}
    }
    
    # Cross-tabulation by region
    if 'REGION' in df.columns:
        print("\n1. Cluster Distribution by Region:")
        region_crosstab = cross_tabulate_clusters(df, cluster_col, 'REGION')
        analysis['by_region'] = region_crosstab
        print(region_crosstab)
    
    # Cross-tabulation by environment (sample source)
    if 'SAMPLE_SOURCE' in df.columns:
        print("\n2. Cluster Distribution by Sample Source:")
        env_crosstab = cross_tabulate_clusters(df, cluster_col, 'SAMPLE_SOURCE')
        analysis['by_environment'] = env_crosstab
        print(env_crosstab)
    
    # Cross-tabulation by species
    if 'ISOLATE_ID' in df.columns:
        print("\n3. Cluster Distribution by Species:")
        species_crosstab = cross_tabulate_clusters(df, cluster_col, 'ISOLATE_ID')
        analysis['by_species'] = species_crosstab
        print(species_crosstab)
    
    # Chi-square tests for independence
    from scipy.stats import chi2_contingency
    
    for group_name, crosstab in [('region', analysis['by_region']),
                                  ('environment', analysis['by_environment']),
                                  ('species', analysis['by_species'])]:
        if crosstab is not None and not crosstab.empty:
            # Remove margins for chi-square test
            crosstab_clean = crosstab.iloc[:-1, :-1]
            if crosstab_clean.shape[0] > 1 and crosstab_clean.shape[1] > 1:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(crosstab_clean)
                    analysis['chi_square_tests'][group_name] = {
                        'chi_square': chi2,
                        'p_value': p_value,
                        'degrees_of_freedom': dof
                    }
                    print(f"\n   Chi-square test (clusters vs {group_name}):")
                    print(f"   χ² = {chi2:.4f}, p-value = {p_value:.4f}")
                except Exception as e:
                    print(f"   Chi-square test failed for {group_name}: {e}")
    
    return analysis


def perform_pca(df: pd.DataFrame,
                feature_cols: List[str],
                n_components: int = 2) -> Tuple[np.ndarray, PCA, Dict]:
    """
    Perform Principal Component Analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with encoded resistance values
    feature_cols : list
        List of feature column names
    n_components : int
        Number of components to extract
    
    Returns:
    --------
    tuple
        (Transformed data, PCA object, PCA info dict)
    """
    print("\n" + "=" * 50)
    print("PHASE 5.2: Principal Component Analysis")
    print("=" * 50)
    
    # Prepare data
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Analysis info
    pca_info = {
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'feature_names': [c.replace('_encoded', '') for c in existing_cols],
        'loadings': {}
    }
    
    # Component loadings
    for i in range(n_components):
        loadings = dict(zip(pca_info['feature_names'], pca.components_[i].tolist()))
        loadings = dict(sorted(loadings.items(), key=lambda x: abs(x[1]), reverse=True))
        pca_info['loadings'][f'PC{i+1}'] = loadings
    
    print(f"\n1. PCA with {n_components} components:")
    for i, (var_ratio, cum_var) in enumerate(zip(pca_info['explained_variance_ratio'],
                                                   pca_info['cumulative_variance'])):
        print(f"   PC{i+1}: {var_ratio*100:.2f}% variance (cumulative: {cum_var*100:.2f}%)")
    
    print("\n2. Top loadings per component:")
    for pc, loadings in pca_info['loadings'].items():
        top_loadings = list(loadings.items())[:5]
        print(f"   {pc}: {', '.join([f'{k}({v:.3f})' for k, v in top_loadings])}")
    
    return X_pca, pca, pca_info


def create_pca_plot(X_pca: np.ndarray,
                    df: pd.DataFrame,
                    color_col: str = 'CLUSTER',
                    pca_info: Dict = None,
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: str = None) -> plt.Figure:
    """
    Create PCA scatter plot.
    
    Parameters:
    -----------
    X_pca : np.ndarray
        PCA-transformed data
    df : pd.DataFrame
        Original dataframe for coloring
    color_col : str
        Column name for color coding
    pca_info : dict, optional
        PCA information for axis labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        PCA plot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    if color_col in df.columns:
        categories = df[color_col].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        
        for category in categories:
            mask = df[color_col] == category
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=[color_map[category]], label=str(category),
                      alpha=0.7, s=50)
        
        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
    
    # Axis labels
    if pca_info:
        var_ratio = pca_info['explained_variance_ratio']
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)')
    else:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    ax.set_title(f'PCA of Resistance Profiles (colored by {color_col})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved to: {save_path}")
    
    return fig


def create_pca_biplot(X_pca: np.ndarray,
                      pca: PCA,
                      feature_names: List[str],
                      df: pd.DataFrame = None,
                      color_col: str = 'CLUSTER',
                      figsize: Tuple[int, int] = (12, 10),
                      save_path: str = None) -> plt.Figure:
    """
    Create PCA biplot with loadings.
    
    Parameters:
    -----------
    X_pca : np.ndarray
        PCA-transformed data
    pca : PCA
        Fitted PCA object
    feature_names : list
        List of feature names
    df : pd.DataFrame, optional
        Original dataframe for coloring
    color_col : str
        Column name for color coding
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        Biplot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scale for visualization
    scale_factor = np.max(np.abs(X_pca)) / np.max(np.abs(pca.components_))
    
    # Plot data points
    if df is not None and color_col in df.columns:
        categories = df[color_col].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        
        for category in categories:
            mask = df[color_col] == category
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[color_map[category]], label=str(category),
                      alpha=0.5, s=30)
        
        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=30, c='gray')
    
    # Plot loadings
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0,
                pca.components_[0, i] * scale_factor * 0.8,
                pca.components_[1, i] * scale_factor * 0.8,
                head_width=0.05 * scale_factor, head_length=0.02 * scale_factor,
                fc='red', ec='red', alpha=0.8)
        ax.text(pca.components_[0, i] * scale_factor,
               pca.components_[1, i] * scale_factor,
               name, color='red', fontsize=8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Biplot: Resistance Profiles and Antibiotic Loadings')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Biplot saved to: {save_path}")
    
    return fig


def run_regional_environmental_analysis(df: pd.DataFrame,
                                        feature_cols: List[str],
                                        output_dir: str = None) -> Dict:
    """
    Run complete regional and environmental analysis pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clustered dataframe
    feature_cols : list
        List of feature column names
    output_dir : str, optional
        Directory to save outputs
    
    Returns:
    --------
    dict
        Complete analysis results
    """
    import os
    
    results = {
        'cluster_distribution': None,
        'pca': None,
        'figures': {}
    }
    
    # Cluster distribution analysis
    if 'CLUSTER' in df.columns:
        results['cluster_distribution'] = analyze_cluster_distribution(df)
    
    # PCA analysis
    X_pca, pca, pca_info = perform_pca(df, feature_cols)
    results['pca'] = {
        'transformed_data': X_pca,
        'pca_object': pca,
        'pca_info': pca_info
    }
    
    # Save figures if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # PCA plot by cluster
        if 'CLUSTER' in df.columns:
            pca_cluster_path = os.path.join(output_dir, 'pca_by_cluster.png')
            create_pca_plot(X_pca, df, 'CLUSTER', pca_info, save_path=pca_cluster_path)
            results['figures']['pca_cluster'] = pca_cluster_path
        
        # PCA plot by region
        if 'REGION' in df.columns:
            pca_region_path = os.path.join(output_dir, 'pca_by_region.png')
            create_pca_plot(X_pca, df, 'REGION', pca_info, save_path=pca_region_path)
            results['figures']['pca_region'] = pca_region_path
        
        # PCA plot by MDR status
        if 'MDR_CATEGORY' in df.columns:
            pca_mdr_path = os.path.join(output_dir, 'pca_by_mdr.png')
            create_pca_plot(X_pca, df, 'MDR_CATEGORY', pca_info, save_path=pca_mdr_path)
            results['figures']['pca_mdr'] = pca_mdr_path
        
        # Biplot
        feature_names = [c.replace('_encoded', '') for c in feature_cols if c in df.columns]
        biplot_path = os.path.join(output_dir, 'pca_biplot.png')
        create_pca_biplot(X_pca, pca, feature_names, df, 'CLUSTER', save_path=biplot_path)
        results['figures']['biplot'] = biplot_path
        
        plt.close('all')
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    clustered_path = project_root / "data" / "processed" / "clustered_dataset.csv"
    
    if clustered_path.exists():
        df = pd.read_csv(clustered_path)
        feature_cols = [c for c in df.columns if c.endswith('_encoded')]
        
        output_dir = project_root / "data" / "processed" / "figures"
        results = run_regional_environmental_analysis(df, feature_cols, str(output_dir))
        
        print("\nAnalysis complete!")
    else:
        print(f"Clustered dataset not found at {clustered_path}")
        print("Run hierarchical_clustering.py first.")
