"""
Integration and Synthesis Module for AMR Thesis Project
Phase 6 - Integration & Synthesis

OBJECTIVES:
    1. Compare unsupervised clusters with supervised discrimination results
    2. Identify dominant resistance archetypes
    3. Identify species–environment associations
    4. Identify MDR-enriched patterns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats


def compare_clusters_with_supervised(
    df: pd.DataFrame,
    cluster_col: str = 'CLUSTER',
    mdr_col: str = 'MDR_CATEGORY',
    species_col: str = 'ISOLATE_ID'
) -> Dict:
    """
    Compare unsupervised clusters with supervised discrimination results.
    
    Evaluates how well the unsupervised clusters align with known categories
    (MDR status and species) that supervised models can discriminate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels and category information
    cluster_col : str
        Column name for cluster labels
    mdr_col : str
        Column name for MDR category
    species_col : str
        Column name for species
    
    Returns:
    --------
    dict
        Comparison results including cluster-category alignment metrics
    """
    comparison = {
        'cluster_mdr_alignment': None,
        'cluster_species_alignment': None,
        'cluster_purity': {},
        'interpretation': []
    }
    
    if cluster_col not in df.columns:
        comparison['interpretation'].append("No cluster column found - run clustering first.")
        return comparison
    
    clusters = df[cluster_col].unique()
    
    # Compare clusters with MDR categories
    if mdr_col in df.columns:
        # Cross-tabulation of clusters vs MDR
        mdr_crosstab = pd.crosstab(df[cluster_col], df[mdr_col])
        comparison['cluster_mdr_alignment'] = mdr_crosstab.to_dict()
        
        # Calculate cluster purity with respect to MDR
        for cluster in clusters:
            cluster_df = df[df[cluster_col] == cluster]
            if len(cluster_df) > 0 and mdr_col in cluster_df.columns:
                mdr_counts = cluster_df[mdr_col].value_counts()
                dominant_mdr = mdr_counts.idxmax()
                purity = mdr_counts.max() / len(cluster_df)
                comparison['cluster_purity'][int(cluster)] = {
                    'mdr_dominant': dominant_mdr,
                    'mdr_purity': float(purity)
                }
        
        # Chi-square test for cluster-MDR association
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(mdr_crosstab)
            comparison['mdr_chi_square'] = {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            if p_value < 0.05:
                comparison['interpretation'].append(
                    f"Significant association between clusters and MDR status "
                    f"(χ²={chi2:.2f}, p={p_value:.4f}). "
                    "Unsupervised clustering captures MDR-related patterns."
                )
            else:
                comparison['interpretation'].append(
                    "No significant association between clusters and MDR status. "
                    "Clusters may capture other resistance patterns."
                )
        except Exception:
            pass
    
    # Compare clusters with species
    if species_col in df.columns:
        species_crosstab = pd.crosstab(df[cluster_col], df[species_col])
        comparison['cluster_species_alignment'] = species_crosstab.to_dict()
        
        # Calculate cluster purity with respect to species
        for cluster in clusters:
            cluster_df = df[df[cluster_col] == cluster]
            if len(cluster_df) > 0:
                species_counts = cluster_df[species_col].value_counts()
                dominant_species = species_counts.idxmax()
                purity = species_counts.max() / len(cluster_df)
                if int(cluster) in comparison['cluster_purity']:
                    comparison['cluster_purity'][int(cluster)]['species_dominant'] = dominant_species
                    comparison['cluster_purity'][int(cluster)]['species_purity'] = float(purity)
                else:
                    comparison['cluster_purity'][int(cluster)] = {
                        'species_dominant': dominant_species,
                        'species_purity': float(purity)
                    }
        
        # Chi-square test for cluster-species association
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(species_crosstab)
            comparison['species_chi_square'] = {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            if p_value < 0.05:
                comparison['interpretation'].append(
                    f"Significant association between clusters and species "
                    f"(χ²={chi2:.2f}, p={p_value:.4f}). "
                    "Resistance patterns relate to species identity."
                )
        except Exception:
            pass
    
    return comparison


def identify_resistance_archetypes(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str = 'CLUSTER',
    threshold_resistant: float = 1.5
) -> Dict:
    """
    Identify dominant resistance archetypes from clustering results.
    
    An archetype is a characteristic resistance profile that defines each cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels and resistance data
    feature_cols : list
        List of encoded resistance column names
    cluster_col : str
        Column name for cluster labels
    threshold_resistant : float
        Mean encoded value above which an antibiotic is considered
        predominantly resistant in a cluster (default 1.5 = between I and R)
    
    Returns:
    --------
    dict
        Archetype definitions for each cluster
    """
    archetypes = {
        'cluster_archetypes': {},
        'summary': []
    }
    
    if cluster_col not in df.columns:
        return archetypes
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    if not existing_cols:
        return archetypes
    
    # Calculate mean resistance profile for each cluster
    cluster_profiles = df.groupby(cluster_col)[existing_cols].mean()
    
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        cluster_size = len(df[df[cluster_col] == cluster])
        
        # Identify antibiotics with high resistance (mean > threshold)
        high_resistance = profile[profile > threshold_resistant].sort_values(ascending=False)
        high_resistance_abs = [ab.replace('_encoded', '') for ab in high_resistance.index.tolist()]
        
        # Identify antibiotics with low resistance (mean < 0.5, mostly susceptible)
        low_resistance = profile[profile < 0.5].sort_values()
        low_resistance_abs = [ab.replace('_encoded', '') for ab in low_resistance.index.tolist()]
        
        # Calculate overall resistance level
        mean_resistance = profile.mean()
        if mean_resistance > 1.5:
            resistance_level = "High resistance"
        elif mean_resistance > 1.0:
            resistance_level = "Moderate-high resistance"
        elif mean_resistance > 0.5:
            resistance_level = "Moderate resistance"
        else:
            resistance_level = "Low resistance"
        
        archetype = {
            'cluster_size': cluster_size,
            'mean_resistance_score': float(mean_resistance),
            'resistance_level': resistance_level,
            'resistant_to': high_resistance_abs[:10],  # Top 10
            'susceptible_to': low_resistance_abs[:10],  # Top 10
            'full_profile': {
                ab.replace('_encoded', ''): float(val)
                for ab, val in profile.items()
            }
        }
        
        archetypes['cluster_archetypes'][int(cluster)] = archetype
        
        # Generate summary description
        if high_resistance_abs:
            summary = (
                f"Cluster {cluster} ({cluster_size} isolates): {resistance_level}. "
                f"Resistant to: {', '.join(high_resistance_abs[:5])}"
            )
            if low_resistance_abs:
                summary += f". Susceptible to: {', '.join(low_resistance_abs[:3])}"
            archetypes['summary'].append(summary)
    
    return archetypes


def identify_species_environment_associations(
    df: pd.DataFrame,
    species_col: str = 'ISOLATE_ID',
    environment_col: str = 'SAMPLE_SOURCE',
    region_col: str = 'REGION'
) -> Dict:
    """
    Identify species–environment associations.
    
    Analyzes which species are associated with specific environmental
    sources and regions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with species and environment information
    species_col : str
        Column name for species
    environment_col : str
        Column name for environmental/sample source
    region_col : str
        Column name for region
    
    Returns:
    --------
    dict
        Species-environment association results
    """
    associations = {
        'species_environment': {},
        'species_region': {},
        'statistical_tests': {},
        'interpretation': []
    }
    
    # Species-Environment associations
    if species_col in df.columns and environment_col in df.columns:
        # Cross-tabulation
        env_crosstab = pd.crosstab(df[species_col], df[environment_col])
        associations['species_environment_crosstab'] = env_crosstab.to_dict()
        
        # For each species, find dominant environment
        for species in df[species_col].unique():
            species_df = df[df[species_col] == species]
            env_counts = species_df[environment_col].value_counts()
            if len(env_counts) > 0:
                associations['species_environment'][species] = {
                    'dominant_environment': env_counts.idxmax(),
                    'proportion': float(env_counts.max() / len(species_df)),
                    'all_environments': env_counts.to_dict()
                }
        
        # Chi-square test for species-environment association
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(env_crosstab)
            associations['statistical_tests']['species_environment'] = {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            if p_value < 0.05:
                associations['interpretation'].append(
                    f"Significant species-environment association detected "
                    f"(χ²={chi2:.2f}, p={p_value:.4f}). "
                    "Different species prefer different environmental sources."
                )
        except Exception:
            pass
    
    # Species-Region associations
    if species_col in df.columns and region_col in df.columns:
        region_crosstab = pd.crosstab(df[species_col], df[region_col])
        associations['species_region_crosstab'] = region_crosstab.to_dict()
        
        # For each species, find dominant region
        for species in df[species_col].unique():
            species_df = df[df[species_col] == species]
            region_counts = species_df[region_col].value_counts()
            if len(region_counts) > 0:
                associations['species_region'][species] = {
                    'dominant_region': region_counts.idxmax(),
                    'proportion': float(region_counts.max() / len(species_df)),
                    'all_regions': region_counts.to_dict()
                }
        
        # Chi-square test for species-region association
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(region_crosstab)
            associations['statistical_tests']['species_region'] = {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            
            if p_value < 0.05:
                associations['interpretation'].append(
                    f"Significant species-region association detected "
                    f"(χ²={chi2:.2f}, p={p_value:.4f}). "
                    "Species distribution varies by region."
                )
        except Exception:
            pass
    
    return associations


def identify_mdr_enriched_patterns(
    df: pd.DataFrame,
    feature_cols: List[str],
    cluster_col: str = 'CLUSTER',
    mdr_flag_col: str = 'MDR_FLAG',
    mdr_category_col: str = 'MDR_CATEGORY',
    region_col: str = 'REGION',
    environment_col: str = 'SAMPLE_SOURCE',
    species_col: str = 'ISOLATE_ID'
) -> Dict:
    """
    Identify MDR-enriched patterns in the data.
    
    Finds clusters, regions, environments, and species with higher
    than expected MDR rates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with MDR status and other metadata
    feature_cols : list
        List of encoded resistance column names
    cluster_col : str
        Column name for cluster labels
    mdr_flag_col : str
        Column name for MDR flag (binary)
    mdr_category_col : str
        Column name for MDR category
    region_col : str
        Column name for region
    environment_col : str
        Column name for environment/sample source
    species_col : str
        Column name for species
    
    Returns:
    --------
    dict
        MDR-enriched patterns analysis
    """
    mdr_patterns = {
        'overall_mdr_rate': None,
        'mdr_enriched_clusters': [],
        'mdr_enriched_regions': [],
        'mdr_enriched_environments': [],
        'mdr_enriched_species': [],
        'mdr_resistance_signature': {},
        'interpretation': []
    }
    
    # Overall MDR rate
    if mdr_flag_col in df.columns:
        overall_mdr = df[mdr_flag_col].mean()
        mdr_patterns['overall_mdr_rate'] = float(overall_mdr)
        mdr_patterns['interpretation'].append(
            f"Overall MDR prevalence: {overall_mdr*100:.1f}%"
        )
    else:
        return mdr_patterns
    
    # MDR-enriched clusters
    if cluster_col in df.columns:
        cluster_mdr = df.groupby(cluster_col)[mdr_flag_col].agg(['mean', 'count'])
        enriched_clusters = cluster_mdr[cluster_mdr['mean'] > overall_mdr].sort_values(
            'mean', ascending=False
        )
        
        for cluster, row in enriched_clusters.iterrows():
            enrichment = {
                'cluster': int(cluster),
                'mdr_rate': float(row['mean']),
                'sample_size': int(row['count']),
                'fold_enrichment': float(row['mean'] / overall_mdr) if overall_mdr > 0 else 0
            }
            mdr_patterns['mdr_enriched_clusters'].append(enrichment)
        
        if mdr_patterns['mdr_enriched_clusters']:
            top_cluster = mdr_patterns['mdr_enriched_clusters'][0]
            mdr_patterns['interpretation'].append(
                f"Cluster {top_cluster['cluster']} is most MDR-enriched "
                f"({top_cluster['mdr_rate']*100:.1f}% MDR, "
                f"{top_cluster['fold_enrichment']:.1f}x overall rate)"
            )
    
    # MDR-enriched regions
    if region_col in df.columns:
        region_mdr = df.groupby(region_col)[mdr_flag_col].agg(['mean', 'count'])
        enriched_regions = region_mdr[region_mdr['mean'] > overall_mdr].sort_values(
            'mean', ascending=False
        )
        
        for region, row in enriched_regions.iterrows():
            enrichment = {
                'region': region,
                'mdr_rate': float(row['mean']),
                'sample_size': int(row['count']),
                'fold_enrichment': float(row['mean'] / overall_mdr) if overall_mdr > 0 else 0
            }
            mdr_patterns['mdr_enriched_regions'].append(enrichment)
    
    # MDR-enriched environments
    if environment_col in df.columns:
        env_mdr = df.groupby(environment_col)[mdr_flag_col].agg(['mean', 'count'])
        enriched_envs = env_mdr[env_mdr['mean'] > overall_mdr].sort_values(
            'mean', ascending=False
        )
        
        for env, row in enriched_envs.iterrows():
            enrichment = {
                'environment': env,
                'mdr_rate': float(row['mean']),
                'sample_size': int(row['count']),
                'fold_enrichment': float(row['mean'] / overall_mdr) if overall_mdr > 0 else 0
            }
            mdr_patterns['mdr_enriched_environments'].append(enrichment)
    
    # MDR-enriched species
    if species_col in df.columns:
        species_mdr = df.groupby(species_col)[mdr_flag_col].agg(['mean', 'count'])
        enriched_species = species_mdr[species_mdr['mean'] > overall_mdr].sort_values(
            'mean', ascending=False
        )
        
        for species, row in enriched_species.iterrows():
            enrichment = {
                'species': species,
                'mdr_rate': float(row['mean']),
                'sample_size': int(row['count']),
                'fold_enrichment': float(row['mean'] / overall_mdr) if overall_mdr > 0 else 0
            }
            mdr_patterns['mdr_enriched_species'].append(enrichment)
    
    # MDR resistance signature (which antibiotics are most associated with MDR)
    existing_cols = [c for c in feature_cols if c in df.columns]
    if existing_cols and mdr_flag_col in df.columns:
        mdr_isolates = df[df[mdr_flag_col] == 1]
        non_mdr_isolates = df[df[mdr_flag_col] == 0]
        
        if len(mdr_isolates) > 0 and len(non_mdr_isolates) > 0:
            for col in existing_cols:
                ab_name = col.replace('_encoded', '')
                mdr_mean = mdr_isolates[col].mean()
                non_mdr_mean = non_mdr_isolates[col].mean()
                diff = mdr_mean - non_mdr_mean
                
                mdr_patterns['mdr_resistance_signature'][ab_name] = {
                    'mdr_mean': float(mdr_mean),
                    'non_mdr_mean': float(non_mdr_mean),
                    'difference': float(diff)
                }
            
            # Sort by difference to find most discriminating antibiotics
            sorted_sig = sorted(
                mdr_patterns['mdr_resistance_signature'].items(),
                key=lambda x: x[1]['difference'],
                reverse=True
            )
            
            top_ab = [ab for ab, _ in sorted_sig[:3]]
            if top_ab:
                mdr_patterns['interpretation'].append(
                    f"Antibiotics most associated with MDR: {', '.join(top_ab)}"
                )
    
    return mdr_patterns


def run_integration_synthesis(
    df: pd.DataFrame,
    feature_cols: List[str],
    supervised_results: Dict = None
) -> Dict:
    """
    Run complete Phase 6 Integration & Synthesis analysis.
    
    This function integrates results from:
    - Unsupervised clustering (Phase 3)
    - Supervised learning (Phase 4)
    - Regional/environmental analysis (Phase 5)
    
    And synthesizes findings about:
    - Dominant resistance archetypes
    - Species-environment associations
    - MDR-enriched patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clustered dataframe with all metadata
    feature_cols : list
        List of encoded resistance column names
    supervised_results : dict, optional
        Results from supervised learning pipeline
    
    Returns:
    --------
    dict
        Complete integration and synthesis results
    """
    print("\n" + "=" * 70)
    print("PHASE 6: INTEGRATION & SYNTHESIS")
    print("=" * 70)
    
    results = {
        'cluster_supervised_comparison': None,
        'resistance_archetypes': None,
        'species_environment_associations': None,
        'mdr_enriched_patterns': None,
        'synthesis_summary': []
    }
    
    # 1. Compare unsupervised clusters with supervised results
    print("\n1. Comparing unsupervised clusters with supervised discrimination...")
    results['cluster_supervised_comparison'] = compare_clusters_with_supervised(df)
    
    if results['cluster_supervised_comparison']['interpretation']:
        for interp in results['cluster_supervised_comparison']['interpretation']:
            print(f"   • {interp}")
    
    # 2. Identify dominant resistance archetypes
    print("\n2. Identifying dominant resistance archetypes...")
    results['resistance_archetypes'] = identify_resistance_archetypes(df, feature_cols)
    
    if results['resistance_archetypes']['summary']:
        for summary in results['resistance_archetypes']['summary']:
            print(f"   • {summary}")
    
    # 3. Identify species-environment associations
    print("\n3. Identifying species-environment associations...")
    results['species_environment_associations'] = identify_species_environment_associations(df)
    
    if results['species_environment_associations']['interpretation']:
        for interp in results['species_environment_associations']['interpretation']:
            print(f"   • {interp}")
    
    # 4. Identify MDR-enriched patterns
    print("\n4. Identifying MDR-enriched patterns...")
    results['mdr_enriched_patterns'] = identify_mdr_enriched_patterns(df, feature_cols)
    
    if results['mdr_enriched_patterns']['interpretation']:
        for interp in results['mdr_enriched_patterns']['interpretation']:
            print(f"   • {interp}")
    
    # 5. Generate synthesis summary
    print("\n5. Generating synthesis summary...")
    results['synthesis_summary'] = _generate_synthesis_summary(results, df, feature_cols)
    
    print("\n" + "-" * 50)
    print("SYNTHESIS SUMMARY:")
    print("-" * 50)
    for point in results['synthesis_summary']:
        print(f"   • {point}")
    
    return results


def _generate_synthesis_summary(results: Dict, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Generate a synthesis summary combining all analysis results.
    
    Parameters:
    -----------
    results : dict
        Results from all integration analyses
    df : pd.DataFrame
        Original dataframe
    feature_cols : list
        Feature column names
    
    Returns:
    --------
    list
        List of synthesis summary points
    """
    summary = []
    
    # Dataset overview
    n_isolates = len(df)
    n_antibiotics = len([c for c in feature_cols if c in df.columns])
    summary.append(f"Analyzed {n_isolates} isolates across {n_antibiotics} antibiotics.")
    
    # Clustering-supervised alignment
    comparison = results.get('cluster_supervised_comparison', {})
    if comparison.get('mdr_chi_square', {}).get('significant'):
        summary.append(
            "Unsupervised clusters significantly align with MDR status, "
            "suggesting clustering captures clinically relevant resistance patterns."
        )
    
    # Dominant archetypes
    archetypes = results.get('resistance_archetypes', {})
    if archetypes.get('cluster_archetypes'):
        n_archetypes = len(archetypes['cluster_archetypes'])
        high_res_clusters = [
            k for k, v in archetypes['cluster_archetypes'].items()
            if v.get('resistance_level') in ['High resistance', 'Moderate-high resistance']
        ]
        if high_res_clusters:
            summary.append(
                f"Identified {n_archetypes} distinct resistance archetypes, "
                f"with {len(high_res_clusters)} cluster(s) showing elevated resistance levels."
            )
    
    # Species-environment associations
    assoc = results.get('species_environment_associations', {})
    if assoc.get('statistical_tests', {}).get('species_environment', {}).get('significant'):
        summary.append(
            "Species distribution varies significantly by environmental source, "
            "indicating habitat-specific bacterial communities."
        )
    
    # MDR patterns
    mdr = results.get('mdr_enriched_patterns', {})
    if mdr.get('overall_mdr_rate') is not None:
        mdr_rate = mdr['overall_mdr_rate'] * 100
        n_enriched = len(mdr.get('mdr_enriched_clusters', []))
        summary.append(
            f"Overall MDR prevalence is {mdr_rate:.1f}%, "
            f"with {n_enriched} cluster(s) showing above-average MDR rates."
        )
    
    # Key antibiotics
    if mdr.get('mdr_resistance_signature'):
        sorted_abs = sorted(
            mdr['mdr_resistance_signature'].items(),
            key=lambda x: x[1]['difference'],
            reverse=True
        )
        top_abs = [ab for ab, _ in sorted_abs[:3]]
        if top_abs:
            summary.append(
                f"Key antibiotics discriminating MDR isolates: {', '.join(top_abs)}."
            )
    
    return summary


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    clustered_path = project_root / "data" / "processed" / "clustered_dataset.csv"
    
    if clustered_path.exists():
        df = pd.read_csv(clustered_path)
        feature_cols = [c for c in df.columns if c.endswith('_encoded')]
        
        results = run_integration_synthesis(df, feature_cols)
        
        print("\n" + "=" * 50)
        print("INTEGRATION & SYNTHESIS COMPLETE")
        print("=" * 50)
    else:
        print(f"Clustered dataset not found at {clustered_path}")
        print("Run the clustering pipeline first.")
