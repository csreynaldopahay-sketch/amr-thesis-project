"""
AMR Pattern Recognition Dashboard
Phase 7 - Interactive Streamlit Application for AMR Surveillance

This tool is intended for exploratory pattern recognition and surveillance analysis only.
It should not be used for clinical decision support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="AMR Pattern Recognition Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme for resistance
RESISTANCE_CMAP = mcolors.ListedColormap(['#4CAF50', '#FFC107', '#F44336'])


def load_data(uploaded_file=None, default_path=None):
    """Load data from uploaded file or default path."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif default_path and os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


def get_antibiotic_cols(df):
    """Get antibiotic column names."""
    encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
    if encoded_cols:
        return encoded_cols
    # Try to identify antibiotic columns by common names
    possible_antibiotics = ['AM', 'AMC', 'CPT', 'CN', 'CF', 'CPD', 'CTX', 'CFO', 
                           'CFT', 'CZA', 'IPM', 'AN', 'GM', 'N', 'NAL', 'ENR',
                           'MRB', 'PRA', 'DO', 'TE', 'FT', 'C', 'SXT']
    return [c for c in df.columns if any(ab in c.upper() for ab in possible_antibiotics)]


def create_heatmap(df, feature_cols, cluster_col=None):
    """Create resistance heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    if cluster_col and cluster_col in df.columns:
        df_sorted = df.sort_values(cluster_col)
    else:
        df_sorted = df.copy()
    
    data = df_sorted[existing_cols].values
    display_cols = [c.replace('_encoded', '') for c in existing_cols]
    
    im = ax.imshow(data, aspect='auto', cmap=RESISTANCE_CMAP, vmin=0, vmax=2)
    
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, rotation=45, ha='right')
    ax.set_xlabel('Antibiotics')
    ax.set_ylabel('Isolates')
    ax.set_title('Resistance Profile Heatmap')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['S', 'I', 'R'])
    
    plt.tight_layout()
    return fig


def create_cluster_summary(df):
    """Create cluster summary statistics."""
    if 'CLUSTER' not in df.columns:
        return None
    
    summary = df.groupby('CLUSTER').agg({
        'CODE': 'count',
        'MAR_INDEX_COMPUTED': 'mean' if 'MAR_INDEX_COMPUTED' in df.columns else lambda x: None,
        'MDR_FLAG': 'mean' if 'MDR_FLAG' in df.columns else lambda x: None
    }).reset_index()
    
    summary.columns = ['Cluster', 'Count', 'Mean MAR Index', 'MDR Proportion']
    return summary


def perform_pca_analysis(df, feature_cols):
    """Perform PCA and return results."""
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols].copy()
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca


def create_pca_plot(X_pca, df, color_col, pca):
    """Create PCA scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_col in df.columns:
        categories = df[color_col].dropna().unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            mask = df[color_col] == category
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], label=str(category),
                      alpha=0.7, s=50)
        
        ax.legend(title=color_col, bbox_to_anchor=(1.05, 1))
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'PCA of Resistance Profiles')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 AMR Pattern Recognition Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Tool for Antimicrobial Resistance Pattern Analysis</p>',
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This tool is intended for exploratory pattern recognition 
        and surveillance analysis only. It should not be used for clinical decision support.
        No patient-level identifiers are processed.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload your AMR dataset in CSV format"
    )
    
    # Try to load default data
    default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 
                                'analysis_ready_dataset.csv')
    
    df = load_data(uploaded_file, default_path)
    
    if df is None:
        st.info("👆 Please upload a CSV dataset to begin analysis.")
        
        st.markdown("""
        ### Expected Data Format
        The dataset should contain:
        - **Isolate identifiers** (CODE, ISOLATE_ID)
        - **Antibiotic resistance data** (encoded as 0=S, 1=I, 2=R)
        - **Metadata** (Region, Site, Sample Source)
        - **Computed features** (MAR_INDEX_COMPUTED, MDR_FLAG, CLUSTER)
        """)
        return
    
    # Data loaded successfully
    st.sidebar.success(f"✅ Loaded {len(df)} isolates")
    
    # Identify columns
    antibiotic_cols = get_antibiotic_cols(df)
    
    # Display options
    st.sidebar.header("📊 Display Options")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["Overview", "Resistance Heatmap", "Cluster Analysis", "PCA Analysis", 
         "Regional Distribution", "Model Evaluation", "Integration & Synthesis"]
    )
    
    # Main content area
    if analysis_type == "Overview":
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Isolates", len(df))
        
        with col2:
            if 'ISOLATE_ID' in df.columns:
                st.metric("Species", df['ISOLATE_ID'].nunique())
        
        with col3:
            if 'MDR_FLAG' in df.columns:
                mdr_pct = df['MDR_FLAG'].mean() * 100
                st.metric("MDR Proportion", f"{mdr_pct:.1f}%")
        
        with col4:
            if 'CLUSTER' in df.columns:
                st.metric("Clusters", df['CLUSTER'].nunique())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Column information
        with st.expander("📚 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.notna().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
    
    elif analysis_type == "Resistance Heatmap":
        st.header("🔥 Resistance Profile Heatmap")
        
        if antibiotic_cols:
            cluster_col = None
            if 'CLUSTER' in df.columns:
                sort_by_cluster = st.checkbox("Sort by Cluster", value=True)
                if sort_by_cluster:
                    cluster_col = 'CLUSTER'
            
            fig = create_heatmap(df, antibiotic_cols, cluster_col)
            st.pyplot(fig)
            
            # Legend
            st.markdown("""
            **Legend:**
            - 🟢 Green (0): Susceptible (S)
            - 🟡 Yellow (1): Intermediate (I)
            - 🔴 Red (2): Resistant (R)
            """)
        else:
            st.warning("No antibiotic columns found in the dataset.")
    
    elif analysis_type == "Cluster Analysis":
        st.header("🎯 Cluster Analysis")
        
        if 'CLUSTER' not in df.columns:
            st.warning("No cluster information found. Please run clustering first.")
        else:
            # Cluster summary
            summary = create_cluster_summary(df)
            if summary is not None:
                st.subheader("Cluster Summary")
                st.dataframe(summary, use_container_width=True)
            
            # Cluster distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Size Distribution")
                cluster_counts = df['CLUSTER'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(8, 6))
                cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Count')
                ax.set_title('Isolates per Cluster')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                if 'MDR_FLAG' in df.columns:
                    st.subheader("MDR by Cluster")
                    mdr_by_cluster = df.groupby('CLUSTER')['MDR_FLAG'].mean() * 100
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mdr_by_cluster.plot(kind='bar', ax=ax, color='#F44336', edgecolor='black')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('MDR Proportion (%)')
                    ax.set_title('MDR Proportion by Cluster')
                    ax.set_ylim(0, 100)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    elif analysis_type == "PCA Analysis":
        st.header("📐 Principal Component Analysis")
        
        if antibiotic_cols:
            # Color by selection
            color_options = ['None']
            if 'CLUSTER' in df.columns:
                color_options.append('CLUSTER')
            if 'REGION' in df.columns:
                color_options.append('REGION')
            if 'MDR_CATEGORY' in df.columns:
                color_options.append('MDR_CATEGORY')
            if 'ISOLATE_ID' in df.columns:
                color_options.append('ISOLATE_ID')
            
            color_by = st.selectbox("Color by:", color_options)
            
            X_pca, pca = perform_pca_analysis(df, antibiotic_cols)
            
            # Variance explained
            st.subheader("Variance Explained")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PC1", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
            with col2:
                st.metric("PC2", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
            
            # PCA plot
            if color_by != 'None':
                fig = create_pca_plot(X_pca, df, color_by, pca)
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title('PCA of Resistance Profiles')
                plt.tight_layout()
            
            st.pyplot(fig)
            
            # Component loadings
            with st.expander("📊 Component Loadings"):
                feature_names = [c.replace('_encoded', '') for c in antibiotic_cols 
                               if c in df.columns]
                loadings_df = pd.DataFrame({
                    'Feature': feature_names,
                    'PC1': pca.components_[0],
                    'PC2': pca.components_[1]
                })
                loadings_df['PC1_abs'] = np.abs(loadings_df['PC1'])
                loadings_df = loadings_df.sort_values('PC1_abs', ascending=False)
                st.dataframe(loadings_df[['Feature', 'PC1', 'PC2']], use_container_width=True)
        else:
            st.warning("No antibiotic columns found for PCA.")
    
    elif analysis_type == "Regional Distribution":
        st.header("🗺️ Regional Distribution")
        
        if 'REGION' not in df.columns:
            st.warning("No region information found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Isolates by Region")
                region_counts = df['REGION'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                region_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel('Region')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                if 'MDR_FLAG' in df.columns:
                    st.subheader("MDR by Region")
                    mdr_by_region = df.groupby('REGION')['MDR_FLAG'].mean() * 100
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mdr_by_region.plot(kind='bar', ax=ax, color='#F44336', edgecolor='black')
                    ax.set_xlabel('Region')
                    ax.set_ylabel('MDR Proportion (%)')
                    ax.set_ylim(0, 100)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Cross-tabulation
            if 'CLUSTER' in df.columns:
                st.subheader("Cluster Distribution by Region")
                crosstab = pd.crosstab(df['CLUSTER'], df['REGION'])
                st.dataframe(crosstab, use_container_width=True)
    
    elif analysis_type == "Model Evaluation":
        st.header("🤖 Model Evaluation Summary")
        
        st.markdown("""
        This section shows the results of supervised learning models trained to 
        discriminate resistance patterns.
        
        **Interpretation Note:** Model metrics quantify how consistently resistance 
        patterns align with known categories (e.g., species, MDR status), 
        not predictive performance for future samples.
        """)
        
        # Check for saved model results
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
        
        if os.path.exists(models_dir):
            import joblib
            
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            if model_files:
                st.subheader("Trained Models")
                
                for model_file in model_files:
                    with st.expander(f"📁 {model_file}"):
                        try:
                            model_data = joblib.load(os.path.join(models_dir, model_file))
                            st.write(f"Model type: {type(model_data['model']).__name__}")
                            st.write("Classes:", model_data.get('label_encoder').classes_ 
                                   if model_data.get('label_encoder') else "N/A")
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
            else:
                st.info("No trained models found. Run the supervised learning pipeline first.")
        else:
            st.info("Models directory not found. Run the supervised learning pipeline first.")
    
    elif analysis_type == "Integration & Synthesis":
        st.header("🔗 Integration & Synthesis (Phase 6)")
        
        st.markdown("""
        This section integrates results from unsupervised clustering (Phase 3), 
        supervised learning (Phase 4), and regional/environmental analysis (Phase 5)
        to identify key patterns in AMR data.
        """)
        
        # Check for required columns
        has_clusters = 'CLUSTER' in df.columns
        has_mdr = 'MDR_FLAG' in df.columns or 'MDR_CATEGORY' in df.columns
        
        if not has_clusters:
            st.warning("⚠️ No cluster information found. Run the clustering pipeline first.")
        else:
            # Import integration module
            try:
                from analysis.integration_synthesis import (
                    compare_clusters_with_supervised,
                    identify_resistance_archetypes,
                    identify_species_environment_associations,
                    identify_mdr_enriched_patterns
                )
                
                # 1. Cluster vs Supervised Comparison
                st.subheader("1. Cluster-Supervised Comparison")
                comparison = compare_clusters_with_supervised(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if comparison.get('cluster_mdr_alignment'):
                        st.markdown("**Cluster vs MDR Status:**")
                        mdr_df = pd.DataFrame(comparison['cluster_mdr_alignment'])
                        st.dataframe(mdr_df, use_container_width=True)
                        
                        if comparison.get('mdr_chi_square'):
                            chi_test = comparison['mdr_chi_square']
                            if chi_test['significant']:
                                st.success(f"✅ Significant association (χ²={chi_test['statistic']:.2f}, p={chi_test['p_value']:.4f})")
                            else:
                                st.info(f"ℹ️ No significant association (p={chi_test['p_value']:.4f})")
                
                with col2:
                    if comparison.get('cluster_purity'):
                        st.markdown("**Cluster Purity:**")
                        purity_data = []
                        for cluster, info in comparison['cluster_purity'].items():
                            row = {'Cluster': cluster}
                            if 'mdr_dominant' in info:
                                row['Dominant MDR'] = info['mdr_dominant']
                                row['MDR Purity'] = f"{info['mdr_purity']*100:.1f}%"
                            if 'species_dominant' in info:
                                row['Dominant Species'] = info['species_dominant']
                            purity_data.append(row)
                        st.dataframe(pd.DataFrame(purity_data), use_container_width=True)
                
                # Interpretation
                if comparison.get('interpretation'):
                    with st.expander("📝 Interpretation"):
                        for interp in comparison['interpretation']:
                            st.write(f"• {interp}")
                
                # 2. Resistance Archetypes
                st.subheader("2. Dominant Resistance Archetypes")
                archetypes = identify_resistance_archetypes(df, antibiotic_cols)
                
                if archetypes.get('cluster_archetypes'):
                    for cluster_id, arch in archetypes['cluster_archetypes'].items():
                        with st.expander(f"📊 Cluster {cluster_id} - {arch['resistance_level']} ({arch['cluster_size']} isolates)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Resistance Score", f"{arch['mean_resistance_score']:.2f}")
                                if arch['resistant_to']:
                                    st.markdown("**Resistant to:**")
                                    st.write(", ".join(arch['resistant_to'][:7]))
                            with col2:
                                if arch['susceptible_to']:
                                    st.markdown("**Susceptible to:**")
                                    st.write(", ".join(arch['susceptible_to'][:7]))
                
                # 3. Species-Environment Associations
                st.subheader("3. Species-Environment Associations")
                associations = identify_species_environment_associations(df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if associations.get('species_environment'):
                        st.markdown("**Species by Environment:**")
                        env_data = []
                        for species, info in associations['species_environment'].items():
                            env_data.append({
                                'Species': species,
                                'Dominant Environment': info['dominant_environment'],
                                'Proportion': f"{info['proportion']*100:.1f}%"
                            })
                        st.dataframe(pd.DataFrame(env_data), use_container_width=True)
                
                with col2:
                    if associations.get('species_region'):
                        st.markdown("**Species by Region:**")
                        region_data = []
                        for species, info in associations['species_region'].items():
                            region_data.append({
                                'Species': species,
                                'Dominant Region': info['dominant_region'],
                                'Proportion': f"{info['proportion']*100:.1f}%"
                            })
                        st.dataframe(pd.DataFrame(region_data), use_container_width=True)
                
                # Statistical tests
                if associations.get('statistical_tests'):
                    with st.expander("📊 Statistical Tests"):
                        for test_name, test_result in associations['statistical_tests'].items():
                            if test_result['significant']:
                                st.success(f"✅ {test_name}: Significant (χ²={test_result['chi_square']:.2f}, p={test_result['p_value']:.4f})")
                            else:
                                st.info(f"ℹ️ {test_name}: Not significant (p={test_result['p_value']:.4f})")
                
                # 4. MDR-Enriched Patterns
                st.subheader("4. MDR-Enriched Patterns")
                
                if has_mdr:
                    mdr_patterns = identify_mdr_enriched_patterns(df, antibiotic_cols)
                    
                    # Overall MDR rate
                    if mdr_patterns.get('overall_mdr_rate') is not None:
                        st.metric("Overall MDR Prevalence", f"{mdr_patterns['overall_mdr_rate']*100:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if mdr_patterns.get('mdr_enriched_clusters'):
                            st.markdown("**MDR-Enriched Clusters:**")
                            cluster_data = []
                            for item in mdr_patterns['mdr_enriched_clusters']:
                                cluster_data.append({
                                    'Cluster': item['cluster'],
                                    'MDR Rate': f"{item['mdr_rate']*100:.1f}%",
                                    'Fold Enrichment': f"{item['fold_enrichment']:.1f}x",
                                    'Sample Size': item['sample_size']
                                })
                            st.dataframe(pd.DataFrame(cluster_data), use_container_width=True)
                    
                    with col2:
                        if mdr_patterns.get('mdr_enriched_regions'):
                            st.markdown("**MDR-Enriched Regions:**")
                            region_data = []
                            for item in mdr_patterns['mdr_enriched_regions']:
                                region_data.append({
                                    'Region': item['region'],
                                    'MDR Rate': f"{item['mdr_rate']*100:.1f}%",
                                    'Fold Enrichment': f"{item['fold_enrichment']:.1f}x"
                                })
                            st.dataframe(pd.DataFrame(region_data), use_container_width=True)
                    
                    # MDR Resistance Signature
                    if mdr_patterns.get('mdr_resistance_signature'):
                        with st.expander("🔬 MDR Resistance Signature"):
                            sig_data = []
                            for ab, stats in mdr_patterns['mdr_resistance_signature'].items():
                                sig_data.append({
                                    'Antibiotic': ab,
                                    'MDR Mean': f"{stats['mdr_mean']:.2f}",
                                    'Non-MDR Mean': f"{stats['non_mdr_mean']:.2f}",
                                    'Difference': f"{stats['difference']:.2f}"
                                })
                            sig_df = pd.DataFrame(sig_data)
                            sig_df = sig_df.sort_values('Difference', ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
                            st.dataframe(sig_df, use_container_width=True)
                    
                    # Interpretation
                    if mdr_patterns.get('interpretation'):
                        with st.expander("📝 Interpretation"):
                            for interp in mdr_patterns['interpretation']:
                                st.write(f"• {interp}")
                else:
                    st.info("ℹ️ MDR information not available in the dataset.")
                    
            except ImportError as e:
                st.error(f"Could not import integration module: {e}")
                st.info("Make sure the integration_synthesis.py module is in the src/analysis directory.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        AMR Pattern Recognition Dashboard | Built with Streamlit<br>
        For research and surveillance purposes only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
