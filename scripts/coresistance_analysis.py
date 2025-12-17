"""
Co-Resistance Network Analysis Module for AMR Thesis Project
Task 2: Replace circular MDR discrimination with scientifically rigorous co-resistance analysis

This module implements two components:
1. Network Construction: Build graph of statistically significant antibiotic co-resistance pairs 
   using chi-square/Fisher's exact test with Bonferroni correction
2. Predictive Modeling: Predict resistance to key antibiotics (TE, ENR, IPM, SXT, AM) from other 
   antibiotics using Random Forest

SCIENTIFIC RATIONALE:
====================
Unlike circular MDR discrimination (predicting MDR from the features that define it), 
co-resistance analysis reveals genuine biological relationships:

1. Genetic linkage: Resistance genes on the same plasmid or mobile genetic element
2. Shared mechanisms: Cross-resistance due to common efflux pumps or membrane changes
3. Epidemiological patterns: Antibiotics used together select for multi-resistance
4. Predictive power: Incomplete AST panels can predict missing resistances

This is NON-CIRCULAR because:
- Each antibiotic is an independent measurement
- Association between antibiotics is discovered empirically, not defined a priori
- Results reveal biological mechanisms (co-carriage, linkage disequilibrium)

References:
-----------
- Sánchez-Osuna M, et al. (2021). Co-resistance: An opportunity for the bacteria and 
  resistance genes. Front Microbiol. 12:655662.
- Cantón R, Ruiz-Garbajosa P. (2011). Co-resistance: an opportunity for the bacteria 
  and resistance genes. Curr Opin Pharmacol. 11(5):477-485.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# Antibiotic class mapping for biological interpretation
ANTIBIOTIC_CLASSES = {
    'AM': 'Penicillins', 'AMP': 'Penicillins',
    'AMC': 'β-lactam/BLI', 'PRA': 'β-lactam/BLI',
    'CN': 'Cephalosporins-1st', 'CF': 'Cephalosporins-1st',
    'CPD': 'Cephalosporins-3rd/4th', 'CTX': 'Cephalosporins-3rd/4th',
    'CFT': 'Cephalosporins-3rd/4th', 'CPT': 'Cephalosporins-3rd/4th',
    'CFO': 'Cephamycins',
    'CZA': 'Cephalosporin/BLI',
    'IPM': 'Carbapenems', 'MRB': 'Carbapenems',
    'AN': 'Aminoglycosides', 'GM': 'Aminoglycosides', 'N': 'Aminoglycosides',
    'NAL': 'Quinolones', 'ENR': 'Fluoroquinolones',
    'DO': 'Tetracyclines', 'TE': 'Tetracyclines',
    'FT': 'Nitrofurans',
    'C': 'Phenicols',
    'SXT': 'Folate pathway inhibitors'
}


def compute_phi_coefficient(contingency_table):
    """
    Compute phi coefficient (effect size) for 2x2 contingency table.
    
    Formula: φ = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    
    Parameters:
    -----------
    contingency_table : pd.DataFrame or np.ndarray
        2x2 contingency table
    
    Returns:
    --------
    float
        Phi coefficient (-1 to 1)
    """
    if isinstance(contingency_table, pd.DataFrame):
        contingency_table = contingency_table.values
    
    if contingency_table.shape != (2, 2):
        return 0.0
    
    a, b = contingency_table[0, 0], contingency_table[0, 1]
    c, d = contingency_table[1, 0], contingency_table[1, 1]
    
    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def build_coresistance_network(df, feature_cols, alpha=0.01, phi_threshold=0.2):
    """
    Build network where edges represent significant co-resistance associations.
    Uses chi-square/Fisher's exact test with Bonferroni correction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with encoded resistance values (S=0, I=1, R=2)
    feature_cols : list
        List of encoded antibiotic column names
    alpha : float
        Significance level before Bonferroni correction (default: 0.01)
    phi_threshold : float
        Minimum phi coefficient for edge inclusion (default: 0.2)
    
    Returns:
    --------
    tuple
        (NetworkX graph, co-resistance matrix DataFrame, edge statistics DataFrame)
    """
    print("\n" + "=" * 70)
    print("CO-RESISTANCE NETWORK CONSTRUCTION")
    print("=" * 70)
    
    # Extract antibiotic names
    antibiotics = [col.replace('_encoded', '') for col in feature_cols]
    n_antibiotics = len(antibiotics)
    n_tests = (n_antibiotics * (n_antibiotics - 1)) // 2
    
    # Bonferroni correction
    bonferroni_alpha = alpha / n_tests
    
    print(f"\n1. Network Construction Parameters:")
    print(f"   - Number of antibiotics: {n_antibiotics}")
    print(f"   - Total pairwise tests: {n_tests}")
    print(f"   - Bonferroni-corrected α: {bonferroni_alpha:.2e}")
    print(f"   - Phi coefficient threshold: {phi_threshold}")
    
    # Initialize graph and matrices
    G = nx.Graph()
    for ab in antibiotics:
        ab_class = ANTIBIOTIC_CLASSES.get(ab, 'Unknown')
        G.add_node(ab, antibiotic_class=ab_class)
    
    coresist_matrix = np.zeros((n_antibiotics, n_antibiotics))
    edge_stats = []
    
    print("\n2. Testing all pairwise associations...")
    
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols[i+1:], start=i+1):
            ab1 = antibiotics[i]
            ab2 = antibiotics[j]
            
            # Binarize: Resistant (2) vs. Non-resistant (0, 1)
            ab1_r = (df[col1] == 2).astype(int)
            ab2_r = (df[col2] == 2).astype(int)
            
            # Remove missing values pairwise
            mask = (df[col1].notna()) & (df[col2].notna())
            ab1_r = ab1_r[mask]
            ab2_r = ab2_r[mask]
            
            if len(ab1_r) < 10:  # Minimum sample size
                continue
            
            # Create contingency table
            contingency = pd.crosstab(ab1_r, ab2_r)
            
            # Ensure 2x2 table
            if contingency.shape != (2, 2):
                # Pad with zeros if necessary
                for idx in [0, 1]:
                    if idx not in contingency.index:
                        contingency.loc[idx] = 0
                    if idx not in contingency.columns:
                        contingency[idx] = 0
                contingency = contingency.loc[[0, 1], [0, 1]].fillna(0)
            
            # Statistical test: Fisher's exact for small samples, chi-square otherwise
            try:
                if contingency.min().min() < 5:
                    _, p_val = fisher_exact(contingency)
                else:
                    chi2, p_val, _, _ = chi2_contingency(contingency)
            except Exception:
                continue
            
            # Compute phi coefficient (effect size)
            phi = compute_phi_coefficient(contingency)
            
            # Store in matrix (symmetric)
            coresist_matrix[i, j] = phi
            coresist_matrix[j, i] = phi
            
            # Record statistics
            edge_stats.append({
                'Antibiotic_1': ab1,
                'Antibiotic_2': ab2,
                'Class_1': ANTIBIOTIC_CLASSES.get(ab1, 'Unknown'),
                'Class_2': ANTIBIOTIC_CLASSES.get(ab2, 'Unknown'),
                'Phi_Coefficient': phi,
                'P_Value': p_val,
                'Significant_Bonferroni': p_val < bonferroni_alpha,
                'Same_Class': ANTIBIOTIC_CLASSES.get(ab1, '') == ANTIBIOTIC_CLASSES.get(ab2, '')
            })
            
            # Add edge if significant and above phi threshold
            if p_val < bonferroni_alpha and phi > phi_threshold:
                G.add_edge(ab1, ab2, weight=phi, p_value=p_val)
    
    # Create DataFrame from matrix
    coresist_df = pd.DataFrame(
        coresist_matrix,
        columns=antibiotics,
        index=antibiotics
    )
    
    # Create edge statistics DataFrame
    edge_stats_df = pd.DataFrame(edge_stats)
    edge_stats_df = edge_stats_df.sort_values('Phi_Coefficient', ascending=False)
    
    # Network summary
    print(f"\n3. Network Summary:")
    print(f"   - Total nodes (antibiotics): {G.number_of_nodes()}")
    print(f"   - Total edges (significant associations): {G.number_of_edges()}")
    print(f"   - Network density: {nx.density(G):.4f}")
    
    if G.number_of_edges() > 0:
        # Find clusters
        connected_components = list(nx.connected_components(G))
        print(f"   - Connected components: {len(connected_components)}")
        
        # Top co-resistance pairs
        print(f"\n4. Top 10 Co-Resistance Pairs (by phi coefficient):")
        top_pairs = edge_stats_df[edge_stats_df['Significant_Bonferroni']].head(10)
        for _, row in top_pairs.iterrows():
            print(f"   - {row['Antibiotic_1']} ↔ {row['Antibiotic_2']}: "
                  f"φ={row['Phi_Coefficient']:.3f}, p={row['P_Value']:.2e}")
    
    return G, coresist_df, edge_stats_df


def visualize_coresistance_network(G, output_path, title="Co-Resistance Network"):
    """
    Visualize the co-resistance network with node colors by antibiotic class.
    
    Parameters:
    -----------
    G : nx.Graph
        Co-resistance network
    output_path : str
        Path to save the figure
    title : str
        Plot title
    """
    if G.number_of_edges() == 0:
        print("No edges to visualize in the network.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Node colors by antibiotic class
    class_colors = {
        'Penicillins': '#FF6B6B',
        'β-lactam/BLI': '#FF8E72',
        'Cephalosporins-1st': '#4ECDC4',
        'Cephalosporins-3rd/4th': '#45B7D1',
        'Cephamycins': '#96CEB4',
        'Cephalosporin/BLI': '#88D8B0',
        'Carbapenems': '#DDA0DD',
        'Aminoglycosides': '#F7DC6F',
        'Quinolones': '#BB8FCE',
        'Fluoroquinolones': '#A569BD',
        'Tetracyclines': '#85C1E9',
        'Nitrofurans': '#F8B500',
        'Phenicols': '#48C9B0',
        'Folate pathway inhibitors': '#F1948A',
        'Unknown': '#BDC3C7'
    }
    
    node_colors = []
    for node in G.nodes():
        ab_class = G.nodes[node].get('antibiotic_class', 'Unknown')
        node_colors.append(class_colors.get(ab_class, '#BDC3C7'))
    
    # Edge weights for thickness
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
    
    # Legend
    unique_classes = set(G.nodes[n].get('antibiotic_class', 'Unknown') for n in G.nodes())
    legend_elements = [plt.scatter([], [], c=class_colors.get(cls, '#BDC3C7'), 
                                   s=100, label=cls) for cls in sorted(unique_classes)]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Antibiotic Class')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   Network visualization saved to: {output_path}")


def predict_antibiotic_resistance(df, feature_cols, target_antibiotic, random_state=42):
    """
    Predict resistance to target antibiotic from other antibiotics using Random Forest.
    
    This is NON-CIRCULAR because:
    - Input: All antibiotics EXCEPT the target
    - Output: Resistance to the target antibiotic
    - Discovery: Which antibiotics predict the target's resistance pattern?
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with encoded resistance values
    feature_cols : list
        List of all encoded antibiotic column names
    target_antibiotic : str
        Name of target antibiotic (without _encoded suffix)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Prediction results including AUC, feature importance, and model metrics
    """
    target_col = f'{target_antibiotic}_encoded'
    
    if target_col not in feature_cols:
        print(f"   WARNING: {target_col} not found in feature columns")
        return None
    
    # Predictors: all except target
    predictor_cols = [col for col in feature_cols if col != target_col]
    
    # Prepare data
    X = df[predictor_cols].copy()
    y = (df[target_col] == 2).astype(int)  # Binary: Resistant vs. Not
    
    # Remove rows with missing target
    valid_mask = df[target_col].notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Check class balance
    resistance_rate = y.mean()
    n_resistant = y.sum()
    n_susceptible = len(y) - n_resistant
    
    # Skip if insufficient samples for either class
    if n_resistant < 3 or n_susceptible < 3:
        print(f"   SKIPPING: Insufficient samples for {target_antibiotic} "
              f"(resistant: {n_resistant}, susceptible: {n_susceptible})")
        return None
    
    if resistance_rate < 0.05 or resistance_rate > 0.95:
        print(f"   WARNING: Extreme class imbalance for {target_antibiotic} "
              f"(resistance rate: {resistance_rate:.1%}) - results may be unreliable")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Train-test split (leakage-safe: split before any further processing)
    # Use stratify only if both classes have sufficient samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=random_state, stratify=y
        )
    except ValueError as e:
        # Fall back to non-stratified split if stratification fails
        print(f"   WARNING: Stratified split failed for {target_antibiotic}, "
              f"using non-stratified split: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=random_state
        )
    
    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError as e:
        # Only one class present in test set - AUC undefined
        print(f"   WARNING: AUC calculation failed for {target_antibiotic}: "
              f"Only one class present in test set (classes: {np.unique(y_test)})")
        auc = np.nan
    
    # Feature importance
    predictor_names = [col.replace('_encoded', '') for col in predictor_cols]
    importance_df = pd.DataFrame({
        'Antibiotic': predictor_names,
        'Importance': model.feature_importances_,
        'Class': [ANTIBIOTIC_CLASSES.get(ab, 'Unknown') for ab in predictor_names]
    }).sort_values('Importance', ascending=False)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    results = {
        'target_antibiotic': target_antibiotic,
        'target_class': ANTIBIOTIC_CLASSES.get(target_antibiotic, 'Unknown'),
        'auc': auc,
        'resistance_rate': resistance_rate,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importance': importance_df,
        'top_5_predictors': importance_df.head(5).to_dict('records'),
        'classification_report': report,
        'model': model
    }
    
    return results


def run_predictive_analysis(df, feature_cols, target_antibiotics=None, random_state=42):
    """
    Run predictive modeling for multiple target antibiotics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with encoded resistance values
    feature_cols : list
        List of encoded antibiotic column names
    target_antibiotics : list, optional
        List of target antibiotic names. Default: ['TE', 'NAL', 'IPM']
    random_state : int
        Random seed
    
    Returns:
    --------
    dict
        Results for all target antibiotics
    """
    print("\n" + "=" * 70)
    print("CO-RESISTANCE PREDICTIVE MODELING")
    print("=" * 70)
    
    if target_antibiotics is None:
        # Default targets: clinically important antibiotics from different classes
        # NAL not in dataset, using ENR (fluoroquinolone) instead
        # Added SXT (folate pathway inhibitor) for diversity
        target_antibiotics = ['TE', 'ENR', 'IPM', 'SXT', 'AM']
    
    # Verify targets exist in data
    available_antibiotics = [col.replace('_encoded', '') for col in feature_cols]
    valid_targets = [ab for ab in target_antibiotics if ab in available_antibiotics]
    
    if not valid_targets:
        print("\nWARNING: None of the specified target antibiotics found in data.")
        print(f"Available antibiotics: {available_antibiotics}")
        # Fall back to available antibiotics
        valid_targets = available_antibiotics[:3]
        print(f"Using alternative targets: {valid_targets}")
    
    print(f"\n1. Target antibiotics for prediction: {valid_targets}")
    print(f"   - TE (Tetracycline): Aquaculture indicator")
    print(f"   - ENR (Enrofloxacin): Fluoroquinolone resistance indicator")
    print(f"   - IPM (Imipenem): Carbapenem (last-resort antibiotic)")
    print(f"   - SXT (Sulfamethoxazole/Trimethoprim): Folate pathway inhibitor")
    print(f"   - AM (Ampicillin): Penicillin class indicator")
    
    all_results = {}
    
    print("\n2. Running predictive models...")
    for target in valid_targets:
        print(f"\n   === {target} ({ANTIBIOTIC_CLASSES.get(target, 'Unknown')}) ===")
        results = predict_antibiotic_resistance(df, feature_cols, target, random_state)
        
        if results is not None:
            all_results[target] = results
            print(f"   AUC: {results['auc']:.3f}")
            print(f"   Resistance rate: {results['resistance_rate']:.1%}")
            print(f"   Top 3 predictors:")
            for pred in results['top_5_predictors'][:3]:
                print(f"     - {pred['Antibiotic']} ({pred['Class']}): {pred['Importance']:.4f}")
    
    # Summary
    print("\n3. PREDICTION SUMMARY:")
    print("-" * 50)
    print(f"{'Target':<10} {'AUC':<10} {'Res. Rate':<12} {'Top Predictor':<15}")
    print("-" * 50)
    for target, results in all_results.items():
        top_pred = results['top_5_predictors'][0]['Antibiotic'] if results['top_5_predictors'] else 'N/A'
        print(f"{target:<10} {results['auc']:.3f}      {results['resistance_rate']:.1%}         {top_pred:<15}")
    print("-" * 50)
    
    return all_results


def generate_biological_interpretation(network_results, prediction_results):
    """
    Generate biological interpretation of co-resistance findings.
    
    Parameters:
    -----------
    network_results : tuple
        (G, coresist_df, edge_stats_df) from build_coresistance_network
    prediction_results : dict
        Results from run_predictive_analysis
    
    Returns:
    --------
    str
        Markdown-formatted interpretation
    """
    G, coresist_df, edge_stats_df = network_results
    
    interpretation = []
    interpretation.append("\n## Biological Interpretation\n")
    
    # Network interpretation
    interpretation.append("### Co-Resistance Network Structure\n")
    interpretation.append(f"- **Network size**: {G.number_of_nodes()} antibiotics, "
                         f"{G.number_of_edges()} significant associations\n")
    interpretation.append(f"- **Network density**: {nx.density(G):.4f}\n")
    
    if G.number_of_edges() > 0:
        # Identify clusters
        connected_components = list(nx.connected_components(G))
        interpretation.append(f"- **Clusters identified**: {len(connected_components)}\n")
        
        for i, component in enumerate(connected_components, 1):
            if len(component) > 1:
                classes = set(ANTIBIOTIC_CLASSES.get(ab, 'Unknown') for ab in component)
                interpretation.append(f"  - Cluster {i}: {', '.join(sorted(component))} "
                                     f"({', '.join(sorted(classes))})\n")
        
        # Top pairs with biological reasoning
        interpretation.append("\n### Key Co-Resistance Pairs\n")
        top_pairs = edge_stats_df[edge_stats_df['Significant_Bonferroni']].head(5)
        for _, row in top_pairs.iterrows():
            ab1, ab2 = row['Antibiotic_1'], row['Antibiotic_2']
            phi = row['Phi_Coefficient']
            same_class = row['Same_Class']
            
            if same_class:
                mechanism = "likely shared mechanism (same class)"
            else:
                mechanism = "suggests plasmid-mediated co-carriage"
            
            interpretation.append(f"- **{ab1} ↔ {ab2}** (φ={phi:.3f}): {mechanism}\n")
    
    # Prediction interpretation
    interpretation.append("\n### Predictive Modeling Results\n")
    for target, results in prediction_results.items():
        auc = results['auc']
        top_pred = results['top_5_predictors'][0] if results['top_5_predictors'] else None
        
        # AUC interpretation
        if auc >= 0.8:
            auc_interp = "strong predictive capacity"
        elif auc >= 0.7:
            auc_interp = "moderate predictive capacity"
        elif auc >= 0.6:
            auc_interp = "limited predictive capacity"
        else:
            auc_interp = "weak predictive capacity (near random)"
        
        interpretation.append(f"\n#### {target} ({ANTIBIOTIC_CLASSES.get(target, 'Unknown')})\n")
        interpretation.append(f"- **AUC**: {auc:.3f} ({auc_interp})\n")
        
        if top_pred:
            pred_ab = top_pred['Antibiotic']
            pred_class = top_pred['Class']
            
            # Generate biological interpretation
            if ANTIBIOTIC_CLASSES.get(target, '') == pred_class:
                bio_interp = "shared resistance mechanism within class"
            else:
                bio_interp = "potential plasmid-mediated co-resistance"
            
            interpretation.append(f"- **Top predictor**: {pred_ab} ({pred_class}) - {bio_interp}\n")
    
    # Clinical implications
    interpretation.append("\n### Clinical Implications\n")
    interpretation.append("1. **Surveillance optimization**: Testing 'hub' antibiotics in the network "
                         "can help infer resistance to connected antibiotics\n")
    interpretation.append("2. **Treatment guidance**: Strong co-resistance associations suggest "
                         "alternative antibiotics may also be ineffective\n")
    interpretation.append("3. **Epidemiological insights**: Cross-class co-resistance patterns "
                         "suggest horizontal gene transfer via mobile genetic elements\n")
    
    return ''.join(interpretation)


def run_coresistance_analysis(data_path=None, output_dir=None):
    """
    Main function to run complete co-resistance analysis.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to analysis-ready dataset
    output_dir : str, optional
        Directory for output files
    
    Returns:
    --------
    tuple
        (network_results, prediction_results, interpretation)
    """
    print("=" * 70)
    print("AMR THESIS PROJECT - CO-RESISTANCE ANALYSIS")
    print("Replacing circular MDR discrimination with scientifically rigorous analysis")
    print("=" * 70)
    
    # Set default paths
    if data_path is None:
        data_path = project_root / 'data' / 'processed' / 'analysis_ready_dataset.csv'
    
    if output_dir is None:
        output_dir = project_root / 'data' / 'processed' / 'figures'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n1. Loading data from: {data_path}")
    
    if not Path(data_path).exists():
        # Try to run the pipeline first
        print("   Data file not found. Running main pipeline to generate data...")
        try:
            from main import run_full_pipeline
            run_full_pipeline()
        except Exception as e:
            print(f"   ERROR: Could not generate data: {e}")
            print("   Please run main.py first to generate the analysis-ready dataset.")
            return None, None, None
    
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    print(f"   Loaded {len(df)} isolates with {len(feature_cols)} antibiotics")
    
    # Phase 1: Build co-resistance network
    G, coresist_df, edge_stats_df = build_coresistance_network(
        df, feature_cols, alpha=0.01, phi_threshold=0.2
    )
    
    # Save network outputs
    network_path = output_dir / 'coresistance_network.graphml'
    nx.write_graphml(G, str(network_path))
    print(f"\n   Network saved to: {network_path}")
    
    matrix_path = output_dir / 'coresistance_matrix.csv'
    coresist_df.to_csv(matrix_path)
    print(f"   Matrix saved to: {matrix_path}")
    
    edge_stats_path = output_dir / 'coresistance_edge_stats.csv'
    edge_stats_df.to_csv(edge_stats_path, index=False)
    print(f"   Edge statistics saved to: {edge_stats_path}")
    
    # Visualize network
    viz_path = output_dir / 'coresistance_network.png'
    visualize_coresistance_network(G, str(viz_path), 
                                    title="Co-Resistance Network\n(Bonferroni-corrected, φ>0.2)")
    
    # Phase 2: Predictive modeling
    prediction_results = run_predictive_analysis(df, feature_cols)
    
    # Save prediction results
    pred_path = output_dir / 'coresistance_prediction_results.csv'
    pred_summary = []
    for target, results in prediction_results.items():
        pred_summary.append({
            'Target_Antibiotic': target,
            'Target_Class': results['target_class'],
            'AUC': results['auc'],
            'Resistance_Rate': results['resistance_rate'],
            'N_Train': results['n_train'],
            'N_Test': results['n_test'],
            'Top_Predictor_1': results['top_5_predictors'][0]['Antibiotic'] if results['top_5_predictors'] else 'N/A',
            'Top_Predictor_1_Importance': results['top_5_predictors'][0]['Importance'] if results['top_5_predictors'] else 0,
            'Top_Predictor_2': results['top_5_predictors'][1]['Antibiotic'] if len(results['top_5_predictors']) > 1 else 'N/A',
            'Top_Predictor_3': results['top_5_predictors'][2]['Antibiotic'] if len(results['top_5_predictors']) > 2 else 'N/A'
        })
    pd.DataFrame(pred_summary).to_csv(pred_path, index=False)
    print(f"\n   Prediction results saved to: {pred_path}")
    
    # Generate interpretation
    network_results = (G, coresist_df, edge_stats_df)
    interpretation = generate_biological_interpretation(network_results, prediction_results)
    
    interp_path = output_dir / 'coresistance_interpretation.md'
    with open(interp_path, 'w') as f:
        f.write(interpretation)
    print(f"   Interpretation saved to: {interp_path}")
    
    print("\n" + "=" * 70)
    print("CO-RESISTANCE ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {network_path}")
    print(f"  - {matrix_path}")
    print(f"  - {edge_stats_path}")
    print(f"  - {viz_path}")
    print(f"  - {pred_path}")
    print(f"  - {interp_path}")
    
    return network_results, prediction_results, interpretation


if __name__ == "__main__":
    run_coresistance_analysis()
