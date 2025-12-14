"""
Data Cleaning Module for AMR Thesis Project
Phase 2.2 and 2.3 - Data cleaning and handling missing data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


# Standardized species names mapping
SPECIES_STANDARDIZATION = {
    # Escherichia coli variants
    'escherichia coli': 'Escherichia coli',
    'e. coli': 'Escherichia coli',
    'e.coli': 'Escherichia coli',
    
    # Klebsiella pneumoniae variants
    'klebsiella pneumoniae ssp pneumoniae': 'Klebsiella pneumoniae',
    'klebsiella pneumoniae': 'Klebsiella pneumoniae',
    'k. pneumoniae': 'Klebsiella pneumoniae',
    
    # Enterobacter variants
    'enterobacter cloacae complex': 'Enterobacter cloacae',
    'enterobacter cloacae': 'Enterobacter cloacae',
    'enterobacter aerogenes': 'Enterobacter aerogenes',
    
    # Pseudomonas variants
    'pseudomonas aeruginosa': 'Pseudomonas aeruginosa',
    'p. aeruginosa': 'Pseudomonas aeruginosa',
    
    # Vibrio variants
    'vibrio fluvialis': 'Vibrio fluvialis',
    'v. fluvialis': 'Vibrio fluvialis',
}

# Standardized antibiotic names mapping
ANTIBIOTIC_STANDARDIZATION = {
    # Full names to abbreviations
    'ampicillin': 'AM',
    'amoxicillin-clavulanate': 'AMC',
    'cefepime': 'CPT',
    'cephalothin': 'CN',
    'cefazolin': 'CF',
    'cefpodoxime': 'CPD',
    'cefotaxime': 'CTX',
    'cefoxitin': 'CFO',
    'ceftriaxone': 'CFT',
    'ceftazidime-avibactam': 'CZA',
    'imipenem': 'IPM',
    'amikacin': 'AN',
    'gentamicin': 'GM',
    'neomycin': 'N',
    'nalidixic acid': 'NAL',
    'enrofloxacin': 'ENR',
    'meropenem': 'MRB',
    'piperacillin-tazobactam': 'PRA',
    'doxycycline': 'DO',
    'tetracycline': 'TE',
    'nitrofurantoin': 'FT',
    'chloramphenicol': 'C',
    'trimethoprim-sulfamethoxazole': 'SXT',
    
    # Alternate abbreviations
    'AMP': 'AM',
    'AMO': 'AMC',
    'CFX': 'CN',
    'CFP': 'CF',
    'CFA': 'CTX',
    'CFV': 'CFO',
    'CTF': 'CFT',
    'CFZ': 'CZA',
    'IME': 'IPM',
    'AMI': 'AN',
    'GEN': 'GM',
    'NEO': 'N',
    'NLA': 'NAL',
    'MAR': 'MRB',
    'DOX': 'DO',
    'TET': 'TE',
    'NIT': 'FT',
    'CHL': 'C',
}


def standardize_species_name(name: str) -> str:
    """
    Standardize bacterial species names.
    
    Parameters:
    -----------
    name : str
        Raw species name
    
    Returns:
    --------
    str
        Standardized species name
    """
    if pd.isna(name) or not isinstance(name, str):
        return np.nan
    
    name_lower = name.strip().lower()
    return SPECIES_STANDARDIZATION.get(name_lower, name.strip())


def standardize_antibiotic_name(name: str) -> str:
    """
    Standardize antibiotic names/abbreviations.
    
    Parameters:
    -----------
    name : str
        Raw antibiotic name
    
    Returns:
    --------
    str
        Standardized antibiotic abbreviation
    """
    if pd.isna(name) or not isinstance(name, str):
        return name
    
    name_clean = name.strip().upper()
    return ANTIBIOTIC_STANDARDIZATION.get(name_clean, name_clean)


def standardize_resistance_value(value) -> Optional[str]:
    """
    Standardize resistance interpretation values to S, I, or R.
    
    Parameters:
    -----------
    value : any
        Raw resistance value
    
    Returns:
    --------
    str or None
        Standardized value (S, I, R) or None for missing/invalid
    """
    if pd.isna(value):
        return None
    
    value_str = str(value).strip().upper()
    
    # Handle common variations
    if value_str in ['S', 'SUSCEPTIBLE']:
        return 'S'
    elif value_str in ['I', 'INTERMEDIATE']:
        return 'I'
    elif value_str in ['R', 'RESISTANT', '*R']:  # *R indicates borderline resistant
        return 'R'
    elif value_str in ['', 'NAN', 'NONE', '-']:
        return None
    else:
        return None


def clean_resistance_data(df: pd.DataFrame, antibiotic_cols: List[str]) -> pd.DataFrame:
    """
    Clean and standardize resistance data in antibiotic columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    antibiotic_cols : list
        List of antibiotic column names
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    for col in antibiotic_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(standardize_resistance_value)
    
    return df_clean


def remove_duplicate_isolates(df: pd.DataFrame, key_cols: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate isolates based on key columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    key_cols : list, optional
        Columns to use for identifying duplicates
    
    Returns:
    --------
    tuple
        (Cleaned dataframe, number of duplicates removed)
    """
    if key_cols is None:
        key_cols = ['CODE']
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=key_cols, keep='first')
    removed_count = initial_count - len(df_clean)
    
    return df_clean, removed_count


def check_inconsistent_values(df: pd.DataFrame, antibiotic_cols: List[str]) -> Dict[str, List]:
    """
    Check for impossible or inconsistent values in resistance data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    antibiotic_cols : list
        List of antibiotic column names
    
    Returns:
    --------
    dict
        Dictionary of issues found
    """
    issues = {
        'mixed_values': [],  # Cells with multiple S/I/R values
        'invalid_values': [],  # Values that aren't S, I, R, or NaN
    }
    
    valid_values = {'S', 'I', 'R', None, np.nan}
    
    for idx, row in df.iterrows():
        for col in antibiotic_cols:
            if col in df.columns:
                value = row[col]
                if pd.notna(value):
                    value_str = str(value).strip().upper()
                    
                    # Check for mixed values (e.g., "S/R")
                    if '/' in value_str or ',' in value_str:
                        issues['mixed_values'].append({
                            'index': idx,
                            'column': col,
                            'value': value
                        })
                    
                    # Check for invalid values after standardization
                    std_value = standardize_resistance_value(value)
                    if std_value not in {'S', 'I', 'R', None}:
                        issues['invalid_values'].append({
                            'index': idx,
                            'column': col,
                            'value': value
                        })
    
    return issues


def analyze_missing_data(df: pd.DataFrame, antibiotic_cols: List[str]) -> Dict[str, float]:
    """
    Analyze missing data patterns in antibiotic columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    antibiotic_cols : list
        List of antibiotic column names
    
    Returns:
    --------
    dict
        Dictionary with missing percentage for each antibiotic
    """
    missing_stats = {}
    total_isolates = len(df)
    
    for col in antibiotic_cols:
        if col in df.columns:
            # Count missing (None, NaN, empty string)
            missing_count = df[col].isna().sum()
            missing_count += (df[col] == '').sum() if df[col].dtype == 'object' else 0
            missing_stats[col] = (missing_count / total_isolates) * 100
    
    return missing_stats


def filter_antibiotics_by_coverage(df: pd.DataFrame, 
                                   antibiotic_cols: List[str],
                                   min_coverage: float = 50.0) -> List[str]:
    """
    Filter antibiotics to retain only those tested in majority of isolates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    antibiotic_cols : list
        List of antibiotic column names
    min_coverage : float
        Minimum percentage of isolates that must have data (default: 50%)
    
    Returns:
    --------
    list
        List of antibiotics meeting coverage threshold
    """
    missing_stats = analyze_missing_data(df, antibiotic_cols)
    retained_antibiotics = [
        col for col, missing_pct in missing_stats.items()
        if (100 - missing_pct) >= min_coverage
    ]
    
    return retained_antibiotics


def remove_isolates_with_excessive_missing(df: pd.DataFrame,
                                            antibiotic_cols: List[str],
                                            max_missing_pct: float = 50.0) -> Tuple[pd.DataFrame, int]:
    """
    Remove isolates with excessive missing AST values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    antibiotic_cols : list
        List of antibiotic column names
    max_missing_pct : float
        Maximum percentage of missing values allowed (default: 50%)
    
    Returns:
    --------
    tuple
        (Cleaned dataframe, number of isolates removed)
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Calculate missing percentage for each isolate
    existing_cols = [c for c in antibiotic_cols if c in df_clean.columns]
    
    if not existing_cols:
        return df_clean, 0
    
    missing_counts = df_clean[existing_cols].isna().sum(axis=1)
    missing_pcts = (missing_counts / len(existing_cols)) * 100
    
    # Filter out isolates with too much missing data
    df_clean = df_clean[missing_pcts <= max_missing_pct]
    removed_count = initial_count - len(df_clean)
    
    return df_clean, removed_count


def clean_dataset(df: pd.DataFrame, 
                  min_antibiotic_coverage: float = 50.0,
                  max_isolate_missing: float = 50.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Main data cleaning function.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw unified dataset
    min_antibiotic_coverage : float
        Minimum coverage percentage for antibiotics
    max_isolate_missing : float
        Maximum missing percentage allowed for isolates
    
    Returns:
    --------
    tuple
        (Cleaned dataframe, cleaning report dictionary)
    """
    print("=" * 50)
    print("PHASE 2.2 & 2.3: Data Cleaning and Missing Data Handling")
    print("=" * 50)
    
    report = {
        'initial_isolates': len(df),
        'initial_columns': len(df.columns),
        'duplicates_removed': 0,
        'isolates_removed_missing': 0,
        'antibiotics_retained': [],
        'antibiotics_excluded': [],
        'final_isolates': 0,
        'final_columns': 0,
        'missing_data_stats': {}
    }
    
    df_clean = df.copy()
    
    # Identify antibiotic columns (exclude metadata and summary columns)
    metadata_cols = ['CODE', 'ISOLATE_ID', 'REGION', 'SITE', 'NATIONAL_SITE',
                     'LOCAL_SITE', 'SAMPLE_SOURCE', 'REPLICATE', 'COLONY',
                     'ESBL', 'SOURCE_FILE', 'SCORED_RESISTANCE', 
                     'NUM_ANTIBIOTICS_TESTED', 'MAR_INDEX']
    
    antibiotic_cols = [c for c in df_clean.columns if c not in metadata_cols]
    
    print(f"\n1. Initial dataset: {len(df_clean)} isolates, {len(antibiotic_cols)} antibiotic columns")
    
    # Step 1: Standardize species names
    if 'ISOLATE_ID' in df_clean.columns:
        df_clean['ISOLATE_ID'] = df_clean['ISOLATE_ID'].apply(standardize_species_name)
        print("2. Standardized species names")
    
    # Step 2: Clean resistance values
    df_clean = clean_resistance_data(df_clean, antibiotic_cols)
    print("3. Standardized resistance values (S, I, R)")
    
    # Step 3: Remove duplicates
    df_clean, dup_removed = remove_duplicate_isolates(df_clean)
    report['duplicates_removed'] = dup_removed
    print(f"4. Removed {dup_removed} duplicate isolates")
    
    # Step 4: Analyze missing data
    missing_stats = analyze_missing_data(df_clean, antibiotic_cols)
    report['missing_data_stats'] = missing_stats
    
    # Step 5: Filter antibiotics by coverage
    retained_antibiotics = filter_antibiotics_by_coverage(
        df_clean, antibiotic_cols, min_antibiotic_coverage
    )
    excluded_antibiotics = [c for c in antibiotic_cols if c not in retained_antibiotics]
    
    report['antibiotics_retained'] = retained_antibiotics
    report['antibiotics_excluded'] = excluded_antibiotics
    
    print(f"5. Retained {len(retained_antibiotics)} antibiotics with ≥{min_antibiotic_coverage}% coverage")
    if excluded_antibiotics:
        print(f"   Excluded: {excluded_antibiotics}")
    
    # Step 6: Remove isolates with excessive missing data
    df_clean, iso_removed = remove_isolates_with_excessive_missing(
        df_clean, retained_antibiotics, max_isolate_missing
    )
    report['isolates_removed_missing'] = iso_removed
    print(f"6. Removed {iso_removed} isolates with >{max_isolate_missing}% missing values")
    
    # Final stats
    report['final_isolates'] = len(df_clean)
    report['final_columns'] = len(df_clean.columns)
    
    print(f"\n7. Final dataset: {len(df_clean)} isolates")
    
    return df_clean, report


def generate_cleaning_report(report: Dict, output_path: str = None) -> str:
    """
    Generate a text report of the cleaning process.
    
    Parameters:
    -----------
    report : dict
        Cleaning report dictionary
    output_path : str, optional
        Path to save the report
    
    Returns:
    --------
    str
        Report text
    """
    lines = [
        "=" * 60,
        "DATA CLEANING REPORT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 40,
        f"Initial isolates: {report['initial_isolates']}",
        f"Duplicates removed: {report['duplicates_removed']}",
        f"Isolates removed (missing data): {report['isolates_removed_missing']}",
        f"Final isolates: {report['final_isolates']}",
        "",
        "ANTIBIOTIC COLUMNS",
        "-" * 40,
        f"Retained ({len(report['antibiotics_retained'])}): {', '.join(report['antibiotics_retained'])}",
        f"Excluded ({len(report['antibiotics_excluded'])}): {', '.join(report['antibiotics_excluded'])}",
        "",
        "MISSING DATA BY ANTIBIOTIC",
        "-" * 40,
    ]
    
    for ab, missing_pct in sorted(report['missing_data_stats'].items(), key=lambda x: x[1]):
        lines.append(f"  {ab}: {missing_pct:.1f}% missing")
    
    report_text = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    import os
    from pathlib import Path
    from data_ingestion import create_unified_dataset
    
    project_root = Path(__file__).parent.parent.parent
    
    # Load unified dataset
    input_dir = project_root
    unified_path = project_root / "data" / "processed" / "unified_raw_dataset.csv"
    
    if not unified_path.exists():
        df = create_unified_dataset(str(input_dir), str(unified_path))
    else:
        df = pd.read_csv(unified_path)
    
    # Clean dataset
    df_clean, report = clean_dataset(df)
    
    # Save cleaned dataset
    clean_path = project_root / "data" / "processed" / "cleaned_dataset.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"\nCleaned dataset saved to: {clean_path}")
    
    # Generate and save report
    report_path = project_root / "data" / "processed" / "cleaning_report.txt"
    report_text = generate_cleaning_report(report, str(report_path))
    print(f"Cleaning report saved to: {report_path}")
