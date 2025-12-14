"""
Visualization Module for AMR Thesis Project
Phase 3.2 - Heatmaps, Dendrograms, and Resistance Pattern Visualization
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram


# Color scheme for resistance values
RESISTANCE_COLORS = {
    0: '#4CAF50',  # Susceptible - Green
    1: '#FFC107',  # Intermediate - Yellow/Amber
    2: '#F44336',  # Resistant - Red
}

# Colormap for heatmap
RESISTANCE_CMAP = mcolors.ListedColormap(['#4CAF50', '#FFC107', '#F44336'])


def create_resistance_heatmap(df: pd.DataFrame,
                             feature_cols: List[str],
                             cluster_col: str = 'CLUSTER',
                             figsize: Tuple[int, int] = (14, 10),
                             save_path: str = None) -> plt.Figure:
    """
    Create a heatmap of resistance patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with encoded resistance values and cluster labels
    feature_cols : list
        List of feature column names
    cluster_col : str
        Column name for cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        Heatmap figure
    """
    # Prepare data
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    # Sort by cluster
    if cluster_col in df.columns:
        df_sorted = df.sort_values(cluster_col)
    else:
        df_sorted = df.copy()
    
    # Extract feature matrix
    data_matrix = df_sorted[existing_cols].values
    
    # Clean column names for display
    display_cols = [c.replace('_encoded', '') for c in existing_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data_matrix, aspect='auto', cmap=RESISTANCE_CMAP,
                   vmin=0, vmax=2)
    
    # Set labels
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, rotation=45, ha='right')
    ax.set_xlabel('Antibiotics')
    ax.set_ylabel('Isolates')
    ax.set_title('Resistance Profile Heatmap')
    
    # Add cluster boundaries
    if cluster_col in df.columns:
        cluster_labels = df_sorted[cluster_col].values
        cluster_boundaries = np.where(np.diff(cluster_labels) != 0)[0]
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary + 0.5, color='black', linewidth=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('Resistance Status')
    cbar.ax.set_yticklabels(['S (Susceptible)', 'I (Intermediate)', 'R (Resistant)'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    return fig


def create_dendrogram(linkage_matrix: np.ndarray,
                     labels: List[str] = None,
                     figsize: Tuple[int, int] = (12, 8),
                     color_threshold: float = None,
                     save_path: str = None) -> plt.Figure:
    """
    Create a dendrogram from hierarchical clustering.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    labels : list, optional
        Labels for leaf nodes
    figsize : tuple
        Figure size
    color_threshold : float, optional
        Threshold for coloring clusters
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        Dendrogram figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dendrogram
    dendro = dendrogram(
        linkage_matrix,
        ax=ax,
        labels=labels,
        color_threshold=color_threshold,
        leaf_rotation=90,
        leaf_font_size=8
    )
    
    ax.set_xlabel('Isolates')
    ax.set_ylabel('Distance')
    ax.set_title('Hierarchical Clustering Dendrogram')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved to: {save_path}")
    
    return fig


def create_clustered_heatmap_with_dendrogram(df: pd.DataFrame,
                                              feature_cols: List[str],
                                              linkage_matrix: np.ndarray,
                                              figsize: Tuple[int, int] = (16, 12),
                                              save_path: str = None) -> plt.Figure:
    """
    Create a heatmap with dendrogram overlay.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with encoded resistance values
    feature_cols : list
        List of feature column names
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        Combined heatmap and dendrogram figure
    """
    # Prepare data
    existing_cols = [c for c in feature_cols if c in df.columns]
    data_matrix = df[existing_cols].values
    display_cols = [c.replace('_encoded', '') for c in existing_cols]
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.02)
    
    # Dendrogram subplot
    ax_dendro = fig.add_subplot(gs[0])
    
    # Create dendrogram (rotated)
    dendro = dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        orientation='left',
        no_labels=True
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.invert_yaxis()
    
    # Reorder data according to dendrogram
    dendro_order = dendro['leaves']
    data_ordered = data_matrix[dendro_order, :]
    
    # Heatmap subplot
    ax_heatmap = fig.add_subplot(gs[1])
    
    im = ax_heatmap.imshow(data_ordered, aspect='auto', cmap=RESISTANCE_CMAP,
                           vmin=0, vmax=2)
    
    ax_heatmap.set_xticks(np.arange(len(display_cols)))
    ax_heatmap.set_xticklabels(display_cols, rotation=45, ha='right')
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlabel('Antibiotics')
    ax_heatmap.set_title('Clustered Resistance Profile Heatmap')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, ticks=[0, 1, 2], pad=0.02)
    cbar.ax.set_yticklabels(['S', 'I', 'R'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustered heatmap saved to: {save_path}")
    
    return fig


def create_cluster_profile_heatmap(cluster_profiles: pd.DataFrame,
                                   figsize: Tuple[int, int] = (12, 6),
                                   save_path: str = None) -> plt.Figure:
    """
    Create a heatmap showing mean resistance profile per cluster.
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        Mean resistance values per cluster
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        Cluster profile heatmap figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean column names
    display_cols = [c.replace('_encoded', '') for c in cluster_profiles.columns]
    cluster_labels = [f'Cluster {i}' for i in cluster_profiles.index]
    
    im = ax.imshow(cluster_profiles.values, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=2)
    
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)
    ax.set_xlabel('Antibiotics')
    ax.set_ylabel('Cluster')
    ax.set_title('Mean Resistance Profile by Cluster')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Resistance Level')
    
    # Add text annotations
    for i in range(len(cluster_labels)):
        for j in range(len(display_cols)):
            value = cluster_profiles.values[i, j]
            color = 'white' if value > 1.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster profile heatmap saved to: {save_path}")
    
    return fig


def create_mdr_distribution_plot(df: pd.DataFrame,
                                 group_col: str = 'CLUSTER',
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: str = None) -> plt.Figure:
    """
    Create a bar plot showing MDR distribution by group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with MDR_FLAG and group column
    group_col : str
        Column to group by
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        MDR distribution figure
    """
    if 'MDR_FLAG' not in df.columns:
        print("MDR_FLAG column not found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate MDR proportions
    mdr_by_group = df.groupby(group_col)['MDR_FLAG'].agg(['sum', 'count'])
    mdr_by_group['proportion'] = mdr_by_group['sum'] / mdr_by_group['count']
    
    groups = [f'{group_col} {i}' for i in mdr_by_group.index]
    proportions = mdr_by_group['proportion'].values * 100
    
    bars = ax.bar(groups, proportions, color='#F44336', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel(group_col)
    ax.set_ylabel('MDR Proportion (%)')
    ax.set_title(f'MDR Distribution by {group_col}')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, prop in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{prop:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MDR distribution plot saved to: {save_path}")
    
    return fig


def create_mar_distribution_plot(df: pd.DataFrame,
                                group_col: str = 'CLUSTER',
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: str = None) -> plt.Figure:
    """
    Create a box plot showing MAR index distribution by group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with MAR_INDEX_COMPUTED and group column
    group_col : str
        Column to group by
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.Figure
        MAR distribution figure
    """
    if 'MAR_INDEX_COMPUTED' not in df.columns:
        print("MAR_INDEX_COMPUTED column not found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot
    groups = sorted(df[group_col].unique())
    data_by_group = [df[df[group_col] == g]['MAR_INDEX_COMPUTED'].dropna().values 
                     for g in groups]
    
    bp = ax.boxplot(data_by_group, labels=[f'{group_col} {g}' for g in groups],
                    patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel(group_col)
    ax.set_ylabel('MAR Index')
    ax.set_title(f'MAR Index Distribution by {group_col}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MAR distribution plot saved to: {save_path}")
    
    return fig


def generate_all_visualizations(df: pd.DataFrame,
                               feature_cols: List[str],
                               linkage_matrix: np.ndarray,
                               output_dir: str) -> Dict[str, str]:
    """
    Generate all visualizations and save to output directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clustered dataframe
    feature_cols : list
        List of feature column names
    linkage_matrix : np.ndarray
        Linkage matrix
    output_dir : str
        Directory to save figures
    
    Returns:
    --------
    dict
        Dictionary with paths to saved figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_figures = {}
    
    print("=" * 50)
    print("PHASE 3.2: Generating Visualizations")
    print("=" * 50)
    
    # 1. Resistance heatmap
    print("\n1. Creating resistance heatmap...")
    heatmap_path = os.path.join(output_dir, 'resistance_heatmap.png')
    create_resistance_heatmap(df, feature_cols, save_path=heatmap_path)
    saved_figures['heatmap'] = heatmap_path
    
    # 2. Dendrogram
    print("2. Creating dendrogram...")
    dendro_path = os.path.join(output_dir, 'dendrogram.png')
    create_dendrogram(linkage_matrix, save_path=dendro_path)
    saved_figures['dendrogram'] = dendro_path
    
    # 3. Clustered heatmap with dendrogram
    print("3. Creating clustered heatmap...")
    clustered_path = os.path.join(output_dir, 'clustered_heatmap.png')
    create_clustered_heatmap_with_dendrogram(df, feature_cols, linkage_matrix, 
                                              save_path=clustered_path)
    saved_figures['clustered_heatmap'] = clustered_path
    
    # 4. Cluster profiles
    from ..clustering.hierarchical_clustering import get_cluster_profiles
    print("4. Creating cluster profile heatmap...")
    cluster_profiles = get_cluster_profiles(df, feature_cols)
    profile_path = os.path.join(output_dir, 'cluster_profiles.png')
    create_cluster_profile_heatmap(cluster_profiles, save_path=profile_path)
    saved_figures['cluster_profiles'] = profile_path
    
    # 5. MDR distribution
    print("5. Creating MDR distribution plot...")
    mdr_path = os.path.join(output_dir, 'mdr_distribution.png')
    create_mdr_distribution_plot(df, save_path=mdr_path)
    saved_figures['mdr_distribution'] = mdr_path
    
    # 6. MAR distribution
    print("6. Creating MAR distribution plot...")
    mar_path = os.path.join(output_dir, 'mar_distribution.png')
    create_mar_distribution_plot(df, save_path=mar_path)
    saved_figures['mar_distribution'] = mar_path
    
    plt.close('all')
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return saved_figures


if __name__ == "__main__":
    from pathlib import Path
    from ..clustering.hierarchical_clustering import run_clustering_pipeline
    
    project_root = Path(__file__).parent.parent.parent
    analysis_path = project_root / "data" / "processed" / "analysis_ready_dataset.csv"
    
    if analysis_path.exists():
        df = pd.read_csv(analysis_path)
        feature_cols = [c for c in df.columns if c.endswith('_encoded')]
        
        # Run clustering first
        df_clustered, linkage_matrix, _ = run_clustering_pipeline(df, feature_cols)
        
        # Generate visualizations
        output_dir = project_root / "data" / "processed" / "figures"
        generate_all_visualizations(df_clustered, feature_cols, linkage_matrix, str(output_dir))
    else:
        print(f"Analysis-ready dataset not found at {analysis_path}")
