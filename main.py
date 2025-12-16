"""
Main Pipeline for AMR Thesis Project
This script runs all phases of the AMR pattern recognition analysis.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.data_ingestion import create_unified_dataset
from preprocessing.data_cleaning import clean_dataset, generate_cleaning_report
from preprocessing.resistance_encoding import create_encoded_dataset
from preprocessing.feature_engineering import prepare_analysis_ready_dataset
from clustering.hierarchical_clustering import run_clustering_pipeline, get_cluster_summary
from visualization.visualization import generate_all_visualizations
from supervised.supervised_learning import run_mdr_discrimination, save_model
from analysis.regional_environmental import run_regional_environmental_analysis
from analysis.integration_synthesis import run_integration_synthesis


def run_full_pipeline(data_dir: str = None, output_dir: str = None):
    """
    Run the complete AMR analysis pipeline.
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory containing raw CSV files
    output_dir : str, optional
        Directory to save processed data and outputs
    """
    # Set default paths
    if data_dir is None:
        data_dir = str(project_root)
    
    if output_dir is None:
        output_dir = str(project_root / 'data' / 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("AMR THESIS PROJECT - FULL ANALYSIS PIPELINE")
    print("=" * 70)
    
    # =============================================
    # PHASE 2: Data Preprocessing
    # =============================================
    
    # Phase 2.1: Data Ingestion
    print("\n" + "=" * 70)
    print("PHASE 2.1: DATA INGESTION AND CONSOLIDATION")
    print("=" * 70)
    
    unified_path = os.path.join(output_dir, 'unified_raw_dataset.csv')
    df_raw = create_unified_dataset(data_dir, unified_path)
    
    if df_raw.empty:
        print("ERROR: No data loaded. Check your CSV files.")
        return
    
    # Phase 2.2 & 2.3: Data Cleaning
    print("\n")
    df_clean, cleaning_report = clean_dataset(df_raw)
    
    clean_path = os.path.join(output_dir, 'cleaned_dataset.csv')
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to: {clean_path}")
    
    report_path = os.path.join(output_dir, 'cleaning_report.txt')
    generate_cleaning_report(cleaning_report, report_path)
    
    # Phase 2.4: Encoding
    print("\n")
    df_encoded, encoding_info = create_encoded_dataset(df_clean)
    
    encoded_path = os.path.join(output_dir, 'encoded_dataset.csv')
    df_encoded.to_csv(encoded_path, index=False)
    print(f"Encoded dataset saved to: {encoded_path}")
    
    # Phase 2.5: Feature Engineering
    print("\n")
    encoded_cols = encoding_info['encoded_columns']
    df_analysis, feature_matrix, metadata, feature_info = prepare_analysis_ready_dataset(
        df_encoded, encoded_cols
    )
    
    analysis_path = os.path.join(output_dir, 'analysis_ready_dataset.csv')
    df_analysis.to_csv(analysis_path, index=False)
    print(f"Analysis-ready dataset saved to: {analysis_path}")
    
    # =============================================
    # PHASE 3: Unsupervised Structure Identification
    # =============================================
    
    print("\n")
    feature_cols = [c for c in df_analysis.columns if c.endswith('_encoded')]
    
    # Create artifacts directory for clustering objects
    artifacts_dir = os.path.join(output_dir, 'clustering_artifacts')
    
    # Run clustering with explicit parameters and robustness check
    df_clustered, linkage_matrix, clustering_info = run_clustering_pipeline(
        df_analysis, 
        feature_cols, 
        n_clusters=5,
        perform_robustness=True,  # Enable robustness check with Manhattan distance
        output_dir=artifacts_dir  # Save clustering artifacts
    )
    
    clustered_path = os.path.join(output_dir, 'clustered_dataset.csv')
    df_clustered.to_csv(clustered_path, index=False)
    print(f"Clustered dataset saved to: {clustered_path}")
    
    # Phase 3.2: Visualization of resistance patterns
    print("\n")
    figures_dir = os.path.join(output_dir, 'figures')
    generate_all_visualizations(df_clustered, feature_cols, linkage_matrix, figures_dir, clustering_info)
    
    # Cluster interpretation using consistent C1, C2, ... labeling
    print("\n" + "=" * 70)
    print("PHASE 3.3: Cluster Interpretation (Resistance Phenotypes)")
    print("=" * 70)
    print("\nNOTE: Clusters represent RESISTANCE PHENOTYPES, not taxonomic groups.")
    print("      Metadata associations are correlational, not causal.\n")
    
    cluster_summary = get_cluster_summary(df_clustered, feature_cols)
    
    for cluster_id, info in cluster_summary.items():
        print(f"\nC{cluster_id} (Resistance Phenotype):")
        print(f"  Isolates: {info['n_isolates']} ({info['percentage']:.1f}%)")
        if 'mdr_proportion' in info:
            print(f"  MDR proportion: {info['mdr_proportion']:.1f}%")
        if 'mean_mar_index' in info:
            print(f"  Mean MAR index: {info['mean_mar_index']:.4f}")
        if 'dominant_species' in info:
            print(f"  Dominant species: {info['dominant_species']} ({info.get('dominant_species_pct', 0):.1f}%)")
        if 'top_resistant_antibiotics' in info:
            print(f"  Top resistant antibiotics: {', '.join(info['top_resistant_antibiotics'][:3])}")
    
    # ==============================================
    # PHASE 4: Supervised Learning
    # =============================================
    
    print("\n")
    
    # MDR Discrimination
    mdr_results = None
    if 'MDR_CATEGORY' in df_clustered.columns:
        try:
            mdr_results = run_mdr_discrimination(df_clustered, feature_cols)
            
            # Save model
            models_dir = os.path.join(output_dir, '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            save_model(
                mdr_results['best_model']['model_object'],
                mdr_results['scaler'],
                mdr_results['label_encoder'],
                os.path.join(models_dir, 'mdr_classifier.joblib')
            )
        except Exception as e:
            print(f"Warning: MDR discrimination failed: {e}")
    
    # =============================================
    # PHASE 5: Regional & Environmental Analysis
    # =============================================
    
    print("\n")
    figures_dir = os.path.join(output_dir, 'figures')
    regional_results = run_regional_environmental_analysis(
        df_clustered, feature_cols, figures_dir
    )
    
    # =============================================
    # PHASE 6: Integration & Synthesis
    # =============================================
    
    # Run comprehensive integration and synthesis analysis
    integration_results = run_integration_synthesis(
        df_clustered, feature_cols, supervised_results=mdr_results
    )
    
    # Additional summary statistics
    print("\n" + "-" * 50)
    print("ADDITIONAL STATISTICS:")
    print("-" * 50)
    
    print("\n1. Dataset Summary:")
    print(f"   Total isolates analyzed: {len(df_clustered)}")
    print(f"   Antibiotics tested: {len(feature_cols)}")
    if 'ISOLATE_ID' in df_clustered.columns:
        print(f"   Species identified: {df_clustered['ISOLATE_ID'].nunique()}")
    if 'REGION' in df_clustered.columns:
        print(f"   Regions: {df_clustered['REGION'].nunique()}")
    
    print("\n2. Resistance Patterns:")
    if 'MDR_FLAG' in df_clustered.columns:
        mdr_pct = df_clustered['MDR_FLAG'].mean() * 100
        print(f"   MDR prevalence: {mdr_pct:.1f}%")
    if 'MAR_INDEX_COMPUTED' in df_clustered.columns:
        print(f"   Mean MAR index: {df_clustered['MAR_INDEX_COMPUTED'].mean():.4f}")
    
    print("\n3. Clustering Results:")
    print(f"   Number of clusters: {clustering_info['n_clusters']}")
    for cluster_id, count in clustering_info['cluster_distribution'].items():
        pct = (count / len(df_clustered)) * 100
        print(f"   Cluster {cluster_id}: {count} isolates ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - unified_raw_dataset.csv")
    print("  - cleaned_dataset.csv")
    print("  - cleaning_report.txt")
    print("  - encoded_dataset.csv")
    print("  - analysis_ready_dataset.csv")
    print("  - clustered_dataset.csv")
    print("  - figures/ (visualization outputs)")
    
    print("\nTo run the interactive dashboard:")
    print("  streamlit run app/streamlit_app.py")
    
    return df_clustered, integration_results


if __name__ == "__main__":
    run_full_pipeline()
