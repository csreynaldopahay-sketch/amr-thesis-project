"""
Feature Engineering Module for AMR Thesis Project
Phase 2.5 - Compute MAR index, MDR flag, and other derived features

This module implements formalized feature engineering with explicit definitions:

MAR Index (Multiple Antibiotic Resistance Index):
    Formula: MAR = a / b
    Where:
        a = Number of antibiotics to which the isolate is resistant (R)
        b = Total number of antibiotics tested on the isolate
    Reference: Krumperman PH. (1983). Multiple antibiotic resistance indexing of 
               Escherichia coli to identify high-risk sources of fecal contamination 
               of foods. Applied and Environmental Microbiology, 46(1), 165-170.

MDR (Multi-Drug Resistant) Classification:
    Definition: An isolate is classified as MDR if it exhibits resistance to 
                at least one agent in ≥3 antimicrobial categories.
    Reference: Magiorakos AP, et al. (2012). Multidrug-resistant, extensively 
               drug-resistant and pandrug-resistant bacteria: an international 
               expert proposal for interim standard definitions for acquired 
               resistance. Clinical Microbiology and Infection, 18(3), 268-281.
               DOI: 10.1111/j.1469-0691.2011.03570.x
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# ANTIBIOTIC CLASS DEFINITIONS
# Reference: CDC/CLSI Guidelines for Antimicrobial Susceptibility Testing
# ============================================================================
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

# Encoding scheme for resistance values
RESISTANCE_ENCODING = {
    'S': 0,  # Susceptible
    'I': 1,  # Intermediate  
    'R': 2   # Resistant
}


# ============================================================================
# SPECIES-SPECIFIC ANTIBIOTIC CLASS DEFINITIONS FOR MDR CLASSIFICATION
# Reference: Magiorakos AP, et al. (2012). Clin Microbiol Infect, 18(3), 268-281.
# "MDR was defined for each organism separately based on resistance to at 
#  least one agent in three or more antimicrobial categories."
# ============================================================================
SPECIES_SPECIFIC_MDR_CLASSES = {
    # Escherichia coli MDR classes (Enterobacteriaceae)
    'Escherichia coli': {
        'AM': 'Penicillins',
        'AMP': 'Penicillins',
        'AMC': 'BL/BLI combinations',
        'PRA': 'BL/BLI combinations',
        'CN': 'Cephalosporins-1st/2nd',
        'CF': 'Cephalosporins-1st/2nd',
        'CPD': 'Cephalosporins-3rd/4th',
        'CTX': 'Cephalosporins-3rd/4th',
        'CFT': 'Cephalosporins-3rd/4th',
        'CPT': 'Cephalosporins-3rd/4th',
        'CFO': 'Cephamycins',
        'IPM': 'Carbapenems',
        'MRB': 'Carbapenems',
        'AN': 'Aminoglycosides',
        'GM': 'Aminoglycosides',
        'N': 'Aminoglycosides',
        'NAL': 'Quinolones',
        'ENR': 'Fluoroquinolones',
        'DO': 'Tetracyclines',
        'TE': 'Tetracyclines',
        'FT': 'Nitrofurans',
        'C': 'Phenicols',
        'SXT': 'Folate pathway inhibitors',
        # Note: CZA excluded for E. coli as not routinely tested
    },
    
    # Klebsiella pneumoniae MDR classes (similar to E. coli)
    'Klebsiella pneumoniae': {
        'AM': 'Penicillins',
        'AMP': 'Penicillins',
        'AMC': 'BL/BLI combinations',
        'PRA': 'BL/BLI combinations',
        'CN': 'Cephalosporins-1st/2nd',
        'CF': 'Cephalosporins-1st/2nd',
        'CPD': 'Cephalosporins-3rd/4th',
        'CTX': 'Cephalosporins-3rd/4th',
        'CFT': 'Cephalosporins-3rd/4th',
        'CPT': 'Cephalosporins-3rd/4th',
        'CZA': 'Cephalosporin/BLI',  # Important for Klebsiella
        'CFO': 'Cephamycins',
        'IPM': 'Carbapenems',
        'MRB': 'Carbapenems',
        'AN': 'Aminoglycosides',
        'GM': 'Aminoglycosides',
        'N': 'Aminoglycosides',
        'NAL': 'Quinolones',
        'ENR': 'Fluoroquinolones',
        'DO': 'Tetracyclines',
        'TE': 'Tetracyclines',
        'FT': 'Nitrofurans',
        'C': 'Phenicols',
        'SXT': 'Folate pathway inhibitors',
    },
    
    # Salmonella spp. MDR classes
    'Salmonella': {
        'AM': 'Penicillins',
        'AMP': 'Penicillins',
        'AMC': 'BL/BLI combinations',
        'CPD': 'Cephalosporins-3rd/4th',
        'CTX': 'Cephalosporins-3rd/4th',
        'CFT': 'Cephalosporins-3rd/4th',
        'CPT': 'Cephalosporins-3rd/4th',
        'IPM': 'Carbapenems',
        'MRB': 'Carbapenems',
        'AN': 'Aminoglycosides',
        'GM': 'Aminoglycosides',
        'NAL': 'Quinolones',
        'ENR': 'Fluoroquinolones',
        'DO': 'Tetracyclines',
        'TE': 'Tetracyclines',
        'C': 'Phenicols',
        'SXT': 'Folate pathway inhibitors',
        # Note: Salmonella has intrinsic resistance to some agents
    },
    
    # Pseudomonas aeruginosa MDR classes (different from Enterobacteriaceae)
    'Pseudomonas aeruginosa': {
        'PRA': 'Anti-pseudomonal penicillins',
        'CZA': 'Anti-pseudomonal cephalosporins',
        'CFT': 'Anti-pseudomonal cephalosporins',
        'IPM': 'Carbapenems',
        'MRB': 'Carbapenems',
        'AN': 'Aminoglycosides',
        'GM': 'Aminoglycosides',
        'ENR': 'Fluoroquinolones',
        'C': 'Phenicols',
        # Note: Different class structure for Pseudomonas
    },
    
    # Enterobacter spp. MDR classes
    'Enterobacter': {
        'AM': 'Penicillins',
        'AMP': 'Penicillins',
        'AMC': 'BL/BLI combinations',
        'PRA': 'BL/BLI combinations',
        'CPD': 'Cephalosporins-3rd/4th',
        'CTX': 'Cephalosporins-3rd/4th',
        'CFT': 'Cephalosporins-3rd/4th',
        'CPT': 'Cephalosporins-3rd/4th',
        'CZA': 'Cephalosporin/BLI',
        'IPM': 'Carbapenems',
        'MRB': 'Carbapenems',
        'AN': 'Aminoglycosides',
        'GM': 'Aminoglycosides',
        'NAL': 'Quinolones',
        'ENR': 'Fluoroquinolones',
        'DO': 'Tetracyclines',
        'TE': 'Tetracyclines',
        'C': 'Phenicols',
        'SXT': 'Folate pathway inhibitors',
        # Note: Enterobacter has intrinsic AmpC, so resistant to 1st/2nd gen cephalosporins
    },
}


def get_species_mdr_classes(species_name: str) -> Dict[str, str]:
    """
    Get species-specific antibiotic class mapping for MDR classification.
    
    Per Magiorakos et al. (2012), MDR should be defined for each organism 
    separately with appropriate antibiotic class definitions.
    
    Parameters:
    -----------
    species_name : str
        Species name (e.g., 'Escherichia coli', 'Klebsiella pneumoniae')
    
    Returns:
    --------
    dict
        Antibiotic to class mapping for the species
    """
    # Normalize species name
    if pd.isna(species_name):
        return ANTIBIOTIC_CLASSES  # Fall back to default
    
    species_lower = str(species_name).lower().strip()
    
    # Check for exact matches first
    for key in SPECIES_SPECIFIC_MDR_CLASSES:
        if key.lower() == species_lower:
            return SPECIES_SPECIFIC_MDR_CLASSES[key]
    
    # Check for partial matches (genus-level)
    for key in SPECIES_SPECIFIC_MDR_CLASSES:
        if key.lower().split()[0] in species_lower or species_lower in key.lower():
            return SPECIES_SPECIFIC_MDR_CLASSES[key]
    
    # Special handling for common species names in data
    if 'e. coli' in species_lower or 'e.coli' in species_lower:
        return SPECIES_SPECIFIC_MDR_CLASSES['Escherichia coli']
    if 'klebsiella' in species_lower:
        return SPECIES_SPECIFIC_MDR_CLASSES['Klebsiella pneumoniae']
    if 'salmonella' in species_lower:
        return SPECIES_SPECIFIC_MDR_CLASSES['Salmonella']
    if 'pseudomonas' in species_lower:
        return SPECIES_SPECIFIC_MDR_CLASSES['Pseudomonas aeruginosa']
    if 'enterobacter' in species_lower:
        return SPECIES_SPECIFIC_MDR_CLASSES['Enterobacter']
    
    # Fall back to default universal mapping
    return ANTIBIOTIC_CLASSES


def compute_mar_index(row: pd.Series, 
                      antibiotic_cols: List[str],
                      resistance_threshold: int = 2) -> Optional[float]:
    """
    Compute Multiple Antibiotic Resistance (MAR) Index.
    
    Formula: MAR = a / b
    Where:
        a = Number of antibiotics to which the isolate is resistant (encoded value >= threshold)
        b = Total number of antibiotics tested on the isolate (non-null values)
    
    Reference: Krumperman PH. (1983). Multiple antibiotic resistance indexing of 
               Escherichia coli. Applied and Environmental Microbiology, 46(1), 165-170.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values (encoded: S=0, I=1, R=2)
    antibiotic_cols : list
        List of antibiotic column names (typically ending in '_encoded')
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
        Note: Using 2 means only 'R' is counted as resistant
        Using 1 would include 'I' (intermediate) as resistant
    
    Returns:
    --------
    float or None
        MAR index value (0.0 to 1.0) or None if no antibiotics were tested
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
    
    This is used for MDR classification per Magiorakos et al. (2012) definition.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values (encoded)
    antibiotic_cols : list
        List of antibiotic column names
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    int
        Number of distinct antibiotic classes with at least one resistant agent
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
    
    Definition: An isolate is classified as MDR if it exhibits resistance to 
                at least one agent in ≥3 antimicrobial categories.
    
    Reference: Magiorakos AP, et al. (2012). Multidrug-resistant, extensively 
               drug-resistant and pandrug-resistant bacteria: an international 
               expert proposal for interim standard definitions for acquired 
               resistance. Clinical Microbiology and Infection, 18(3), 268-281.
               DOI: 10.1111/j.1469-0691.2011.03570.x
    
    Parameters:
    -----------
    row : pd.Series
        Row containing encoded resistance values (S=0, I=1, R=2)
    antibiotic_cols : list
        List of antibiotic column names (typically ending in '_encoded')
    min_classes : int
        Minimum number of resistant classes for MDR classification (default: 3)
        Per Magiorakos et al., MDR requires ≥3 antimicrobial categories
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R only)
    
    Returns:
    --------
    bool
        True if isolate meets MDR criteria, False otherwise
    """
    resistant_classes_count = count_resistant_classes(row, antibiotic_cols, resistance_threshold)
    return resistant_classes_count >= min_classes


def count_resistant_classes_species_specific(row: pd.Series,
                                             antibiotic_cols: List[str],
                                             species_col: str = 'ISOLATE_ID',
                                             resistance_threshold: int = 2) -> int:
    """
    Count number of antibiotic classes with at least one resistant antibiotic,
    using SPECIES-SPECIFIC class definitions per Magiorakos et al. (2012).
    
    This function correctly implements the Magiorakos MDR definition which
    specifies that MDR should be defined "for each organism separately."
    
    Parameters:
    -----------
    row : pd.Series
        Row containing resistance values (encoded) and species information
    antibiotic_cols : list
        List of antibiotic column names
    species_col : str
        Column name containing species identification (default: 'ISOLATE_ID')
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R)
    
    Returns:
    --------
    int
        Number of distinct antibiotic classes with at least one resistant agent
    """
    # Get species-specific class mapping
    species_name = row.get(species_col, None)
    class_mapping = get_species_mdr_classes(species_name)
    
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
                    ab_class = class_mapping.get(ab_name, None)
                    # Only count if antibiotic is relevant for this species
                    if ab_class is not None:
                        resistant_classes.add(ab_class)
    
    return len(resistant_classes)


def determine_mdr_status_species_specific(row: pd.Series,
                                          antibiotic_cols: List[str],
                                          species_col: str = 'ISOLATE_ID',
                                          min_classes: int = 3,
                                          resistance_threshold: int = 2) -> bool:
    """
    Determine if an isolate is Multi-Drug Resistant (MDR) using SPECIES-SPECIFIC
    antibiotic class definitions per Magiorakos et al. (2012).
    
    This is the CORRECT implementation of MDR classification per the original
    Magiorakos definition which states MDR should be defined "for each organism
    separately based on resistance to at least one agent in three or more 
    antimicrobial categories."
    
    Reference: Magiorakos AP, et al. (2012). Clin Microbiol Infect, 18(3), 268-281.
    
    Parameters:
    -----------
    row : pd.Series
        Row containing encoded resistance values and species information
    antibiotic_cols : list
        List of antibiotic column names (typically ending in '_encoded')
    species_col : str
        Column name containing species identification (default: 'ISOLATE_ID')
    min_classes : int
        Minimum number of resistant classes for MDR classification (default: 3)
    resistance_threshold : int
        Encoded value considered resistant (default: 2 for R only)
    
    Returns:
    --------
    bool
        True if isolate meets MDR criteria, False otherwise
    """
    resistant_classes_count = count_resistant_classes_species_specific(
        row, antibiotic_cols, species_col, resistance_threshold
    )
    return resistant_classes_count >= min_classes


def _safe_encode_binary_resistance(value, resistance_threshold: int = 2) -> Optional[int]:
    """
    Safely convert encoded resistance value to binary (R=1, non-R=0).
    
    Parameters:
    -----------
    value : any
        Encoded resistance value
    resistance_threshold : int
        Threshold for resistance (default: 2 for R)
    
    Returns:
    --------
    int or None
        1 if resistant, 0 if susceptible/intermediate, None if missing
    """
    if pd.isna(value):
        return None
    try:
        # Try float first to handle '2.0' strings, then convert to int
        return 1 if int(float(value)) >= resistance_threshold else 0
    except (ValueError, TypeError):
        # Handle non-numeric string values
        value_str = str(value).strip().upper()
        if value_str == 'R':
            return 1
        elif value_str in ('S', 'I'):
            return 0
        return None


def create_binary_resistance_features(df: pd.DataFrame,
                                     antibiotic_cols: List[str]) -> pd.DataFrame:
    """
    Create binary resistance features (R vs non-R) for each antibiotic.
    
    Binary features are useful for:
    - Simplified pattern analysis
    - Feature importance interpretation
    - Direct resistance prevalence calculations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with encoded resistance values
    antibiotic_cols : list
        List of encoded antibiotic column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional binary resistance columns
    """
    df_binary = df.copy()
    
    for col in antibiotic_cols:
        if col in df_binary.columns:
            ab_name = col.replace('_encoded', '')
            # R (encoded as 2) -> 1, S or I -> 0, missing -> NaN
            df_binary[f'{ab_name}_RESISTANT'] = df_binary[col].apply(_safe_encode_binary_resistance)
    
    return df_binary


def add_derived_features(df: pd.DataFrame,
                        antibiotic_cols: List[str] = None,
                        use_species_specific_mdr: bool = True) -> pd.DataFrame:
    """
    Add all derived features to the dataframe.
    
    Features added:
    - MAR_INDEX_COMPUTED: Multiple Antibiotic Resistance Index (0-1)
    - RESISTANCE_COUNT: Total number of resistant antibiotics
    - RESISTANT_CLASSES_COUNT: Number of antimicrobial categories with resistance
    - MDR_FLAG: Boolean Multi-Drug Resistant status (universal)
    - MDR_CATEGORY: Categorical "MDR" or "Non-MDR"
    - MDR_FLAG_SPECIES_SPECIFIC: Boolean MDR using species-specific definitions (if enabled)
    - MDR_CATEGORY_SPECIES_SPECIFIC: Categorical species-specific MDR
    - {AB}_RESISTANT: Binary resistance indicators for each antibiotic
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with encoded resistance values
    antibiotic_cols : list, optional
        List of antibiotic column names (encoded). If None, auto-detected.
    use_species_specific_mdr : bool
        If True, also compute species-specific MDR per Magiorakos (2012).
        Requires 'ISOLATE_ID' column. Default: True.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all derived features added
    """
    print("=" * 70)
    print("PHASE 2.5: Feature Engineering (Formalized Definitions)")
    print("=" * 70)
    
    # Print formal definitions
    print("\nFORMAL FEATURE DEFINITIONS:")
    print("-" * 50)
    print("MAR Index (Multiple Antibiotic Resistance Index):")
    print("  Formula: MAR = a / b")
    print("  Where: a = resistant antibiotics, b = total tested")
    print("  Reference: Krumperman PH. (1983). Appl Environ Microbiol.")
    print("")
    print("MDR (Multi-Drug Resistant) Classification:")
    print("  Definition: Resistance to ≥1 agent in ≥3 antimicrobial categories")
    print("  Reference: Magiorakos AP, et al. (2012). Clin Microbiol Infect.")
    if use_species_specific_mdr:
        print("  NOTE: Species-specific MDR classification will also be computed")
    print("-" * 50)
    
    df_features = df.copy()
    
    # Auto-detect encoded columns if not provided
    if antibiotic_cols is None:
        antibiotic_cols = [c for c in df.columns if c.endswith('_encoded')]
    
    print(f"\n1. Using {len(antibiotic_cols)} antibiotic columns for feature engineering")
    
    # Compute MAR index
    print("2. Computing MAR index (Krumperman, 1983)...")
    df_features['MAR_INDEX_COMPUTED'] = df_features.apply(
        lambda row: compute_mar_index(row, antibiotic_cols),
        axis=1
    )
    
    # Compute resistance count
    print("3. Computing resistance count...")
    df_features['RESISTANCE_COUNT'] = df_features.apply(
        lambda row: compute_resistance_count(row, antibiotic_cols),
        axis=1
    )
    
    # Count resistant classes (universal)
    print("4. Counting resistant antibiotic classes (universal)...")
    df_features['RESISTANT_CLASSES_COUNT'] = df_features.apply(
        lambda row: count_resistant_classes(row, antibiotic_cols),
        axis=1
    )
    
    # Determine MDR status (universal - Magiorakos et al., 2012)
    print("5. Determining MDR status (universal classification)...")
    df_features['MDR_FLAG'] = df_features.apply(
        lambda row: determine_mdr_status(row, antibiotic_cols),
        axis=1
    )
    df_features['MDR_CATEGORY'] = df_features['MDR_FLAG'].map({True: 'MDR', False: 'Non-MDR'})
    
    # Species-specific MDR classification (Task 8 implementation)
    if use_species_specific_mdr and 'ISOLATE_ID' in df_features.columns:
        print("6. Computing SPECIES-SPECIFIC MDR (Magiorakos et al., 2012)...")
        print("   Using species-appropriate antibiotic class definitions")
        
        # Count species-specific resistant classes
        df_features['RESISTANT_CLASSES_COUNT_SPECIES'] = df_features.apply(
            lambda row: count_resistant_classes_species_specific(
                row, antibiotic_cols, species_col='ISOLATE_ID'
            ),
            axis=1
        )
        
        # Determine species-specific MDR
        df_features['MDR_FLAG_SPECIES_SPECIFIC'] = df_features.apply(
            lambda row: determine_mdr_status_species_specific(
                row, antibiotic_cols, species_col='ISOLATE_ID'
            ),
            axis=1
        )
        df_features['MDR_CATEGORY_SPECIES_SPECIFIC'] = df_features['MDR_FLAG_SPECIES_SPECIFIC'].map(
            {True: 'MDR', False: 'Non-MDR'}
        )
        
        # Compare universal vs species-specific MDR
        mdr_universal = df_features['MDR_FLAG'].sum()
        mdr_species = df_features['MDR_FLAG_SPECIES_SPECIFIC'].sum()
        agreement = (df_features['MDR_FLAG'] == df_features['MDR_FLAG_SPECIES_SPECIFIC']).mean() * 100
        
        print(f"\n   MDR Classification Comparison:")
        print(f"   Universal MDR:        {mdr_universal} isolates ({mdr_universal/len(df_features)*100:.1f}%)")
        print(f"   Species-specific MDR: {mdr_species} isolates ({mdr_species/len(df_features)*100:.1f}%)")
        print(f"   Agreement rate:       {agreement:.1f}%")
        
        step_num = 7
    else:
        if use_species_specific_mdr and 'ISOLATE_ID' not in df_features.columns:
            print("6. SKIPPING species-specific MDR (ISOLATE_ID column not found)")
        step_num = 6
    
    # Add binary resistance indicators using the helper function
    print(f"{step_num}. Creating binary resistance indicators (R=1, S/I=0)...")
    df_features = create_binary_resistance_features(df_features, antibiotic_cols)
    
    # Summary statistics
    mdr_count = df_features['MDR_FLAG'].sum()
    mdr_pct = (mdr_count / len(df_features)) * 100
    
    print(f"\n{step_num + 1}. FEATURE ENGINEERING SUMMARY:")
    print(f"   Total isolates: {len(df_features)}")
    print(f"   MDR isolates (universal): {mdr_count} ({mdr_pct:.1f}%)")
    print(f"   Non-MDR isolates: {len(df_features) - mdr_count} ({100-mdr_pct:.1f}%)")
    print(f"   Mean MAR index: {df_features['MAR_INDEX_COMPUTED'].mean():.4f}")
    print(f"   Mean resistance count: {df_features['RESISTANCE_COUNT'].mean():.2f}")
    print(f"   Mean resistant classes: {df_features['RESISTANT_CLASSES_COUNT'].mean():.2f}")
    
    return df_features


def prepare_analysis_ready_dataset(df: pd.DataFrame,
                                   antibiotic_cols: List[str] = None,
                                   output_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create the final analysis-ready dataset with structural data separation.
    
    This function physically separates:
    - Feature matrix (X): Encoded resistance values for clustering/modeling
    - Metadata (meta): Sample identification and derived features
    
    This separation improves pipeline clarity and downstream modeling safety
    by preventing accidental use of metadata as model features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Encoded dataframe
    antibiotic_cols : list, optional
        List of antibiotic columns (encoded). If None, auto-detected.
    output_path : str, optional
        Path to save the full analysis-ready dataset
    
    Returns:
    --------
    tuple
        (Full dataset, Feature matrix X, Metadata dataframe, Feature info dict)
        
    Output Files (when output_path provided):
        - analysis_ready_dataset.csv: Full combined dataset
        - feature_matrix_X.csv: Feature matrix only (for clustering/ML)
        - metadata.csv: Metadata only (for interpretation)
    """
    # Add derived features
    df_full = add_derived_features(df, antibiotic_cols)
    
    # Prepare feature matrix (X) - encoded resistance values only
    if antibiotic_cols is None:
        antibiotic_cols = [c for c in df_full.columns if c.endswith('_encoded')]
    
    feature_matrix = df_full[antibiotic_cols].copy()
    
    # Prepare metadata (meta) - everything except raw antibiotic columns and encoded columns
    metadata_cols = ['CODE', 'ISOLATE_ID', 'REGION', 'SITE', 'ENVIRONMENT',
                     'SAMPLING_SOURCE', 'NATIONAL_SITE', 'LOCAL_SITE', 
                     'REPLICATE', 'COLONY', 'ESBL', 'SOURCE_FILE', 
                     'resistance_fingerprint', 'MAR_INDEX_COMPUTED', 
                     'RESISTANCE_COUNT', 'RESISTANT_CLASSES_COUNT', 
                     'MDR_FLAG', 'MDR_CATEGORY']
    
    existing_metadata = [c for c in metadata_cols if c in df_full.columns]
    
    # Also include binary resistance features in metadata for easier interpretation
    binary_cols = [c for c in df_full.columns if c.endswith('_RESISTANT')]
    existing_metadata.extend(binary_cols)
    
    metadata = df_full[existing_metadata].copy()
    
    # Feature info
    feature_info = {
        'antibiotic_columns': antibiotic_cols,
        'total_antibiotics': len(antibiotic_cols),
        'total_isolates': len(df_full),
        'feature_matrix_shape': feature_matrix.shape,
        'metadata_shape': metadata.shape,
        'mdr_count': int(df_full['MDR_FLAG'].sum()),
        'mdr_percentage': float((df_full['MDR_FLAG'].sum() / len(df_full)) * 100),
        'mean_mar_index': float(df_full['MAR_INDEX_COMPUTED'].mean()),
        'mean_resistance_count': float(df_full['RESISTANCE_COUNT'].mean()),
        'references': {
            'mar_index': 'Krumperman PH. (1983). Appl Environ Microbiol, 46(1):165-170.',
            'mdr_definition': 'Magiorakos AP, et al. (2012). Clin Microbiol Infect, 18(3):268-281.'
        }
    }
    
    # Save outputs if output_path provided
    if output_path:
        import os
        output_dir = os.path.dirname(output_path)
        
        # Save full dataset
        df_full.to_csv(output_path, index=False)
        print(f"\n8. STRUCTURAL DATA SEPARATION:")
        print(f"   Full dataset saved to: {output_path}")
        
        # Save separated feature matrix and metadata
        feature_matrix_path = os.path.join(output_dir, 'feature_matrix_X.csv')
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        
        feature_matrix.to_csv(feature_matrix_path, index=False)
        print(f"   Feature matrix (X) saved to: {feature_matrix_path}")
        print(f"     Shape: {feature_matrix.shape}")
        
        metadata.to_csv(metadata_path, index=False)
        print(f"   Metadata saved to: {metadata_path}")
        print(f"     Shape: {metadata.shape}")
    
    return df_full, feature_matrix, metadata, feature_info


if __name__ == "__main__":
    from pathlib import Path
    import os
    
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
            if key != 'references':
                print(f"  {key}: {value}")
        
        print("\nReferences:")
        for ref_key, ref_value in info['references'].items():
            print(f"  {ref_key}: {ref_value}")
    else:
        print(f"Encoded dataset not found at {encoded_path}")
        print("Run resistance_encoding.py first.")
