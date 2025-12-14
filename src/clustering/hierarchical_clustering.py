"""
Clustering Module for AMR Thesis Project
Phase 3.1 - Hierarchical Agglomerative Clustering for Resistance Pattern Recognition
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings


def prepare_clustering_data(df: pd.DataFrame,
                           feature_cols: List[str],
                           impute_strategy: str = 'median') -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare data for clustering by handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with encoded resistance values
    feature_cols : list
        List of feature column names
    impute_strategy : str
        Strategy for imputing missing values ('mean', 'median', 'most_frequent')
    
    Returns:
    --------
    tuple
        (Imputed feature matrix as numpy array, Original dataframe with valid rows)
    """
    # Extract feature matrix
    existing_cols = [c for c in feature_cols if c in df.columns]
    feature_matrix = df[existing_cols].copy()
    
    # Track which rows have data
    valid_mask = feature_matrix.notna().any(axis=1)
    feature_matrix_valid = feature_matrix[valid_mask].copy()
    df_valid = df[valid_mask].copy()
    
    # Impute missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    imputed_data = imputer.fit_transform(feature_matrix_valid)
    
    print(f"Prepared {imputed_data.shape[0]} isolates with {imputed_data.shape[1]} features")
    
    return imputed_data, df_valid


def compute_distance_matrix(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix
    metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine')
    
    Returns:
    --------
    np.ndarray
        Condensed distance matrix
    """
    return pdist(data, metric=metric)


def perform_hierarchical_clustering(data: np.ndarray,
                                   method: str = 'ward',
                                   metric: str = 'euclidean') -> np.ndarray:
    """
    Perform hierarchical agglomerative clustering.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix (samples x features)
    method : str
        Linkage method ('ward', 'complete', 'average', 'single')
    metric : str
        Distance metric ('euclidean', 'manhattan')
    
    Returns:
    --------
    np.ndarray
        Linkage matrix for dendrogram
    """
    if method == 'ward':
        # Ward's method requires Euclidean distance
        metric = 'euclidean'
    
    # Compute linkage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = linkage(data, method=method, metric=metric)
    
    return Z


def assign_clusters(linkage_matrix: np.ndarray,
                   n_clusters: int = None,
                   distance_threshold: float = None) -> np.ndarray:
    """
    Assign cluster labels based on linkage matrix.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    n_clusters : int, optional
        Number of clusters to form
    distance_threshold : float, optional
        Distance threshold for forming clusters
    
    Returns:
    --------
    np.ndarray
        Cluster labels for each sample
    """
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    elif distance_threshold is not None:
        labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    else:
        # Default: use 5 clusters
        labels = fcluster(linkage_matrix, 5, criterion='maxclust')
    
    return labels


def determine_optimal_clusters(linkage_matrix: np.ndarray,
                              max_clusters: int = 10) -> Dict:
    """
    Analyze cluster quality for different numbers of clusters.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    max_clusters : int
        Maximum number of clusters to evaluate
    
    Returns:
    --------
    dict
        Dictionary with cluster quality metrics
    """
    from scipy.cluster.hierarchy import inconsistent
    
    # Compute inconsistency coefficients
    inconsist = inconsistent(linkage_matrix)
    
    # Analyze cluster sizes for different cuts
    cluster_analysis = {}
    
    for k in range(2, max_clusters + 1):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        cluster_analysis[k] = {
            'n_clusters': k,
            'cluster_sizes': dict(zip(unique_labels.astype(int), counts.astype(int))),
            'min_size': int(counts.min()),
            'max_size': int(counts.max()),
            'size_std': float(counts.std())
        }
    
    return cluster_analysis


def run_clustering_pipeline(df: pd.DataFrame,
                           feature_cols: List[str],
                           n_clusters: int = 5,
                           linkage_method: str = 'ward',
                           distance_metric: str = 'euclidean') -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Main clustering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with encoded resistance values
    feature_cols : list
        List of feature column names
    n_clusters : int
        Number of clusters to form
    linkage_method : str
        Linkage method for hierarchical clustering
    distance_metric : str
        Distance metric
    
    Returns:
    --------
    tuple
        (Dataframe with cluster labels, Linkage matrix, Clustering info dict)
    """
    print("=" * 50)
    print("PHASE 3.1: Resistance-Profile Clustering")
    print("=" * 50)
    
    # Prepare data
    print("\n1. Preparing data for clustering...")
    imputed_data, df_valid = prepare_clustering_data(df, feature_cols)
    
    # Perform clustering
    print(f"2. Performing hierarchical clustering (method: {linkage_method}, metric: {distance_metric})...")
    linkage_matrix = perform_hierarchical_clustering(
        imputed_data, method=linkage_method, metric=distance_metric
    )
    
    # Assign clusters
    print(f"3. Assigning {n_clusters} clusters...")
    cluster_labels = assign_clusters(linkage_matrix, n_clusters=n_clusters)
    
    # Add cluster labels to dataframe
    df_clustered = df_valid.copy()
    df_clustered['CLUSTER'] = cluster_labels
    
    # Analyze cluster distribution
    print("4. Analyzing cluster distribution...")
    cluster_dist = df_clustered['CLUSTER'].value_counts().sort_index()
    
    for cluster_id, count in cluster_dist.items():
        pct = (count / len(df_clustered)) * 100
        print(f"   Cluster {cluster_id}: {count} isolates ({pct:.1f}%)")
    
    # Determine optimal clusters analysis
    print("5. Analyzing optimal cluster numbers...")
    optimal_analysis = determine_optimal_clusters(linkage_matrix)
    
    # Clustering info
    clustering_info = {
        'method': linkage_method,
        'metric': distance_metric,
        'n_clusters': n_clusters,
        'total_isolates': len(df_clustered),
        'cluster_distribution': cluster_dist.to_dict(),
        'linkage_matrix_shape': linkage_matrix.shape,
        'optimal_cluster_analysis': optimal_analysis
    }
    
    print(f"\n6. Clustering complete: {len(df_clustered)} isolates in {n_clusters} clusters")
    
    return df_clustered, linkage_matrix, clustering_info


def get_cluster_profiles(df_clustered: pd.DataFrame,
                        feature_cols: List[str]) -> pd.DataFrame:
    """
    Calculate mean resistance profile for each cluster.
    
    Parameters:
    -----------
    df_clustered : pd.DataFrame
        Dataframe with cluster labels
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    pd.DataFrame
        Mean resistance profile per cluster
    """
    existing_cols = [c for c in feature_cols if c in df_clustered.columns]
    
    cluster_profiles = df_clustered.groupby('CLUSTER')[existing_cols].mean()
    
    return cluster_profiles


def get_cluster_summary(df_clustered: pd.DataFrame,
                       metadata_cols: List[str] = None) -> Dict:
    """
    Get summary statistics for each cluster.
    
    Parameters:
    -----------
    df_clustered : pd.DataFrame
        Dataframe with cluster labels
    metadata_cols : list, optional
        Metadata columns for cross-tabulation
    
    Returns:
    --------
    dict
        Summary statistics per cluster
    """
    summary = {}
    
    for cluster_id in sorted(df_clustered['CLUSTER'].unique()):
        cluster_df = df_clustered[df_clustered['CLUSTER'] == cluster_id]
        
        cluster_summary = {
            'n_isolates': len(cluster_df),
            'percentage': (len(cluster_df) / len(df_clustered)) * 100
        }
        
        # Species composition
        if 'ISOLATE_ID' in cluster_df.columns:
            species_dist = cluster_df['ISOLATE_ID'].value_counts().to_dict()
            cluster_summary['species_distribution'] = species_dist
        
        # MDR proportion
        if 'MDR_FLAG' in cluster_df.columns:
            mdr_count = cluster_df['MDR_FLAG'].sum()
            cluster_summary['mdr_count'] = int(mdr_count)
            cluster_summary['mdr_proportion'] = mdr_count / len(cluster_df)
        
        # Regional distribution
        if 'REGION' in cluster_df.columns:
            region_dist = cluster_df['REGION'].value_counts().to_dict()
            cluster_summary['regional_distribution'] = region_dist
        
        # Environmental distribution (sample source)
        if 'SAMPLE_SOURCE' in cluster_df.columns:
            env_dist = cluster_df['SAMPLE_SOURCE'].value_counts().to_dict()
            cluster_summary['environmental_distribution'] = env_dist
        
        # Mean MAR index
        if 'MAR_INDEX_COMPUTED' in cluster_df.columns:
            cluster_summary['mean_mar_index'] = float(cluster_df['MAR_INDEX_COMPUTED'].mean())
        
        summary[int(cluster_id)] = cluster_summary
    
    return summary


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    analysis_path = project_root / "data" / "processed" / "analysis_ready_dataset.csv"
    
    if analysis_path.exists():
        df = pd.read_csv(analysis_path)
        
        # Get encoded columns
        feature_cols = [c for c in df.columns if c.endswith('_encoded')]
        
        # Run clustering
        df_clustered, linkage_matrix, info = run_clustering_pipeline(
            df, feature_cols, n_clusters=5
        )
        
        # Get cluster summary
        summary = get_cluster_summary(df_clustered)
        
        print("\n" + "=" * 50)
        print("CLUSTER SUMMARY")
        print("=" * 50)
        
        for cluster_id, cluster_info in summary.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Isolates: {cluster_info['n_isolates']} ({cluster_info['percentage']:.1f}%)")
            if 'mdr_proportion' in cluster_info:
                print(f"  MDR proportion: {cluster_info['mdr_proportion']:.2%}")
            if 'mean_mar_index' in cluster_info:
                print(f"  Mean MAR index: {cluster_info['mean_mar_index']:.4f}")
        
        # Save clustered data
        clustered_path = project_root / "data" / "processed" / "clustered_dataset.csv"
        df_clustered.to_csv(clustered_path, index=False)
        print(f"\nClustered dataset saved to: {clustered_path}")
    else:
        print(f"Analysis-ready dataset not found at {analysis_path}")
        print("Run feature_engineering.py first.")
