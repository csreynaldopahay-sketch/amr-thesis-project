"""
Feature Engineering Module for AMR Thesis Project
Phase 2.5 - Compute MAR index, MDR flag, and other derived features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


# MDR definitions by organism group (based on CDC/CLSI guidelines)
# For Enterobacteriaceae, MDR is typically defined as resistance to ≥3 antibiotic classes
ANTIBIOTIC_CLASSES = {
    # Penicillins
    'AM': 'Penicillins',
    'AMP': 'Penicillins',
    
    # Beta-lactam/Beta-lactamase inhibitor combinations
    'AMC': 'BL/BLI combinations',
    'AMO': 'BL/BLI combinations',
    'PRA': 'BL/BLI combinations',
    
    # Cephalosporins (1st gen)
    'CN': 'Cephalosporins-1st',
    'CF': 'Cephalosporins-1st',
    'CFX': 'Cephalosporins-1st',
    
    # Cephalosporins (3rd/4th gen)
    'CPD': 'Cephalosporins-3rd/4th',
    'CTX': 'Cephalosporins-3rd/4th',
    'CFT': 'Cephalosporins-3rd/4th',
    'CPT': 'Cephalosporins-3rd/4th',
    'CFA': 'Cephalosporins-3rd/4th',
    'CFV': 'Cephalosporins-3rd/4th',
    'CTF': 'Cephalosporins-3rd/4th',
    
    # Cephamycins
    'CFO': 'Cephamycins',
    
    # Cephalosporin/BLI combinations
    'CZA': 'Cephalosporin/BLI',
    'CFZ': 'Cephalosporin/BLI',
    
    # Carbapenems
    'IPM': 'Carbapenems',
    'MRB': 'Carbapenems',
    'IME': 'Carbapenems',
    'MAR': 'Carbapenems',
    
    # Aminoglycosides
    'AN': 'Aminoglycosides',
    'GM': 'Aminoglycosides',
    'N': 'Aminoglycosides',
    'AMI': 'Aminoglycosides',
    'GEN': 'Aminoglycosides',
    'NEO': 'Aminoglycosides',
    
    # Fluoroquinolones
    'NAL': 'Quinolones',
    'ENR': 'Fluoroquinolones',
    'NLA': 'Quinolones',
    
    # Tetracyclines
    'DO': 'Tetracyclines',
    'TE': 'Tetracyclines',
    'DOX': 'Tetracyclines',
    'TET': 'Tetracyclines',
    
    # Nitrofurans
    'FT': 'Nitrofurans',
    'NIT': 'Nitrofurans',
    
    # Phenicols
    'C': 'Phenicols',
    'CHL': 'Phenicols',
    
    # Folate pathway inhibitors
    'SXT': 'Folate pathway inhibitors',
}


def compute_mar_index(row: pd.Series, 
                      antibiotic_cols: List[str],
                      resistance_threshold: int = 2) -> Optional[float]:
    """
    Compute Multiple Antibiotic Resistance (MAR) Index.
    
    MAR Index = Number of antibiotics resistant / Total antibiotics tested
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values
    antibiotic_cols : list
        List of antibiotic column names
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    float or None
        MAR index value
    """
    tested = 0
    resistant = 0
    
    for col in antibiotic_cols:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                tested += 1
                try:
                    if int(value) >= resistance_threshold:
                        resistant += 1
                except (ValueError, TypeError):
                    # Handle non-numeric values
                    if str(value).strip().upper() == 'R':
                        resistant += 1
    
    if tested == 0:
        return None
    
    return round(resistant / tested, 4)


def compute_resistance_count(row: pd.Series,
                            antibiotic_cols: List[str],
                            resistance_threshold: int = 2) -> int:
    """
    Count number of resistant antibiotics.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values
    antibiotic_cols : list
        List of antibiotic column names
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    int
        Number of resistant antibiotics
    """
    resistant = 0
    
    for col in antibiotic_cols:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                try:
                    if int(value) >= resistance_threshold:
                        resistant += 1
                except (ValueError, TypeError):
                    if str(value).strip().upper() == 'R':
                        resistant += 1
    
    return resistant


def count_resistant_classes(row: pd.Series,
                           antibiotic_cols: List[str],
                           resistance_threshold: int = 2) -> int:
    """
    Count number of antibiotic classes with at least one resistant antibiotic.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values
    antibiotic_cols : list
        List of antibiotic column names
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    int
        Number of resistant antibiotic classes
    """
    resistant_classes = set()
    
    for col in antibiotic_cols:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                is_resistant = False
                try:
                    if int(value) >= resistance_threshold:
                        is_resistant = True
                except (ValueError, TypeError):
                    if str(value).strip().upper() == 'R':
                        is_resistant = True
                
                if is_resistant:
                    # Get the base antibiotic name (remove _encoded suffix)
                    ab_name = col.replace('_encoded', '').upper()
                    ab_class = ANTIBIOTIC_CLASSES.get(ab_name, 'Unknown')
                    if ab_class != 'Unknown':
                        resistant_classes.add(ab_class)
    
    return len(resistant_classes)


def determine_mdr_status(row: pd.Series,
                        antibiotic_cols: List[str],
                        min_classes: int = 3,
                        resistance_threshold: int = 2) -> bool:
    """
    Determine if an isolate is Multi-Drug Resistant (MDR).
    
    MDR is typically defined as resistance to at least 3 antibiotic classes.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values
    antibiotic_cols : list
        List of antibiotic column names
    min_classes : int
        Minimum number of resistant classes for MDR (default: 3)
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    bool
        True if MDR, False otherwise
    """
    resistant_classes_count = count_resistant_classes(row, antibiotic_cols, resistance_threshold)
    return resistant_classes_count >= min_classes


def add_derived_features(df: pd.DataFrame,
                        antibiotic_cols: List[str] = None) -> pd.DataFrame:
    """
    Add all derived features to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with resistance values
    antibiotic_cols : list, optional
        List of antibiotic column names (encoded)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added features
    """
    print("=" * 50)
    print("PHASE 2.5: Feature Engineering")
    print("=" * 50)
    
    df_features = df.copy()
    
    # Auto-detect encoded columns if not provided
    if antibiotic_cols is None:
        antibiotic_cols = [c for c in df.columns if c.endswith('_encoded')]
    
    print(f"\n1. Using {len(antibiotic_cols)} antibiotic columns for feature engineering")
    
    # Compute MAR index
    df_features['MAR_INDEX_COMPUTED'] = df_features.apply(
        lambda row: compute_mar_index(row, antibiotic_cols),
        axis=1
    )
    print("2. Computed MAR index")
    
    # Compute resistance count
    df_features['RESISTANCE_COUNT'] = df_features.apply(
        lambda row: compute_resistance_count(row, antibiotic_cols),
        axis=1
    )
    print("3. Computed resistance count")
    
    # Count resistant classes
    df_features['RESISTANT_CLASSES_COUNT'] = df_features.apply(
        lambda row: count_resistant_classes(row, antibiotic_cols),
        axis=1
    )
    print("4. Counted resistant antibiotic classes")
    
    # Determine MDR status
    df_features['MDR_FLAG'] = df_features.apply(
        lambda row: determine_mdr_status(row, antibiotic_cols),
        axis=1
    )
    df_features['MDR_CATEGORY'] = df_features['MDR_FLAG'].map({True: 'MDR', False: 'Non-MDR'})
    print("5. Determined MDR status")
    
    # Add binary resistance indicators
    for col in antibiotic_cols:
        ab_name = col.replace('_encoded', '')
        df_features[f'{ab_name}_RESISTANT'] = df_features[col].apply(
            lambda x: 1 if pd.notna(x) and int(x) >= 2 else 0
        )
    
    print("6. Created binary resistance indicators")
    
    # Summary statistics
    mdr_count = df_features['MDR_FLAG'].sum()
    mdr_pct = (mdr_count / len(df_features)) * 100
    
    print(f"\n7. Summary:")
    print(f"   Total isolates: {len(df_features)}")
    print(f"   MDR isolates: {mdr_count} ({mdr_pct:.1f}%)")
    print(f"   Mean MAR index: {df_features['MAR_INDEX_COMPUTED'].mean():.4f}")
    print(f"   Mean resistance count: {df_features['RESISTANCE_COUNT'].mean():.2f}")
    
    return df_features


def prepare_analysis_ready_dataset(df: pd.DataFrame,
                                   antibiotic_cols: List[str] = None,
                                   output_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create the final analysis-ready dataset with all features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Encoded dataframe
    antibiotic_cols : list, optional
        List of antibiotic columns (encoded)
    output_path : str, optional
        Path to save the dataset
    
    Returns:
    --------
    tuple
        (Full dataset, Feature matrix, Metadata dataframe, Feature info dict)
    """
    # Add derived features
    df_full = add_derived_features(df, antibiotic_cols)
    
    # Prepare feature matrix (encoded resistance values only)
    if antibiotic_cols is None:
        antibiotic_cols = [c for c in df_full.columns if c.endswith('_encoded')]
    
    feature_matrix = df_full[antibiotic_cols].copy()
    
    # Prepare metadata
    metadata_cols = ['CODE', 'ISOLATE_ID', 'REGION', 'SITE', 'NATIONAL_SITE',
                     'LOCAL_SITE', 'SAMPLE_SOURCE', 'REPLICATE', 'COLONY',
                     'ESBL', 'SOURCE_FILE', 'resistance_fingerprint',
                     'MAR_INDEX_COMPUTED', 'RESISTANCE_COUNT', 
                     'RESISTANT_CLASSES_COUNT', 'MDR_FLAG', 'MDR_CATEGORY']
    
    existing_metadata = [c for c in metadata_cols if c in df_full.columns]
    metadata = df_full[existing_metadata].copy()
    
    # Feature info
    feature_info = {
        'antibiotic_columns': antibiotic_cols,
        'total_antibiotics': len(antibiotic_cols),
        'total_isolates': len(df_full),
        'mdr_count': int(df_full['MDR_FLAG'].sum()),
        'mdr_percentage': float((df_full['MDR_FLAG'].sum() / len(df_full)) * 100),
        'mean_mar_index': float(df_full['MAR_INDEX_COMPUTED'].mean()),
        'mean_resistance_count': float(df_full['RESISTANCE_COUNT'].mean())
    }
    
    # Save if output path provided
    if output_path:
        df_full.to_csv(output_path, index=False)
        print(f"\nAnalysis-ready dataset saved to: {output_path}")
    
    return df_full, feature_matrix, metadata, feature_info


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    encoded_path = project_root / "data" / "processed" / "encoded_dataset.csv"
    
    if encoded_path.exists():
        df = pd.read_csv(encoded_path)
        
        output_path = project_root / "data" / "processed" / "analysis_ready_dataset.csv"
        df_full, feature_matrix, metadata, info = prepare_analysis_ready_dataset(
            df, output_path=str(output_path)
        )
        
        print("\nFeature Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print(f"Encoded dataset not found at {encoded_path}")
        print("Run resistance_encoding.py first.")
