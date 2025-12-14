"""
Data Ingestion and Consolidation Module for AMR Thesis Project
Phase 2.1 - Load, merge, and add metadata to CSV files from different regions and sites
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional


# Standard antibiotic abbreviations we expect
STANDARD_ANTIBIOTICS = [
    'AM', 'AMC', 'CPT', 'CN', 'CF', 'CPD', 'CTX', 'CFO', 'CFT', 'CZA',
    'IPM', 'AN', 'GM', 'N', 'NAL', 'ENR', 'MRB', 'PRA', 'DO', 'TE',
    'FT', 'C', 'SXT',
    # Alternate names
    'AMP', 'AMO', 'CFX', 'CFP', 'CFA', 'CFV', 'CTF', 'CFZ', 'IME',
    'AMI', 'GEN', 'NEO', 'NLA', 'MAR', 'DOX', 'TET', 'NIT', 'CHL'
]


def parse_isolate_code(code: str) -> Dict[str, str]:
    """
    Parse isolate code to extract metadata based on naming convention.
    """
    metadata = {
        'national_site': None,
        'local_site': None,
        'sample_source': None,
        'replicate': None,
        'colony': None
    }
    
    if not code or not isinstance(code, str):
        return metadata
    
    # Remove prefix like EC_, VC_, SAL if present
    code_clean = re.sub(r'^[A-Z]+_', '', code.strip())
    
    # National site mapping
    national_site_map = {
        'O': 'Ormoc',
        'P': 'Pampanga',
        'M': 'Marawi'
    }
    
    # Local site mapping (second letter)
    local_site_map = {
        'A': 'Alegria',
        'L': 'Larrazabal',
        'G': 'Gabriel',
        'R': 'Roque',
        'D': 'Dayawan',
        'T': 'Tuca Kialdan',
        'P': 'APMC'
    }
    
    # Sample source mapping
    sample_source_map = {
        'DW': 'Drinking Water',
        'LW': 'Lake Water',
        'FB': 'Fish Banak',
        'FG': 'Fish Gusaw',
        'RW': 'River Water',
        'FT': 'Fish Tilapia',
        'EWU': 'Effluent Water Untreated',
        'EWT': 'Effluent Water Treated',
        'FK': 'Fish Kaolang'
    }
    
    # Parse national site (first character)
    if len(code_clean) > 0:
        national_char = code_clean[0].upper()
        metadata['national_site'] = national_site_map.get(national_char, national_char)
    
    # Parse local site (second character)
    if len(code_clean) > 1:
        local_char = code_clean[1].upper()
        metadata['local_site'] = local_site_map.get(local_char, local_char)
    
    # Parse sample source (two or three letters after local site)
    sample_match = re.search(r'[A-Z]{2}([A-Z]{2,3})', code_clean.upper())
    if sample_match:
        sample_code = sample_match.group(1)
        metadata['sample_source'] = sample_source_map.get(sample_code, sample_code)
    
    # Parse replicate number (R followed by digit)
    replicate_match = re.search(r'R(\d)', code_clean.upper())
    if replicate_match:
        metadata['replicate'] = int(replicate_match.group(1))
    
    # Parse colony number (C followed by digits)
    colony_match = re.search(r'C(\d+)', code_clean.upper())
    if colony_match:
        metadata['colony'] = int(colony_match.group(1))
    
    return metadata


def extract_region_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extract region and site information from filename.
    """
    region = None
    site = None
    
    # Extract region
    if 'BARMM' in filename:
        region = 'BARMM'
    elif 'Region VIII' in filename:
        region = 'Region VIII - Eastern Visayas'
    elif 'Region III' in filename:
        region = 'Region III - Central Luzon'
    else:
        region_match = re.search(r'Region\s+([IVX]+(?:-[A-Za-z\s]+)?)', filename)
        if region_match:
            region = f'Region {region_match.group(1)}'
    
    # Extract site (after LOR-)
    site_match = re.search(r'LOR-([A-Z\s]+)\.csv', filename, re.IGNORECASE)
    if site_match:
        site = site_match.group(1).strip()
    
    return region, site


def process_csv_file(filepath: str) -> pd.DataFrame:
    """
    Process a single CSV file and extract structured data.
    
    CSV Structure:
    - Row 3: CODE, ISOLATE ID headers + summary columns at end
    - Row 4: ESBL + antibiotic names (AM, AMC, etc.)
    - Row 5: MIC/INT labels
    - Row 6+: Data rows
    """
    filename = os.path.basename(filepath)
    region, site = extract_region_from_filename(filename)
    
    try:
        # Read CSV with all columns
        df = pd.read_csv(filepath, header=None)
        
        # Find the row with CODE header (usually row 3)
        code_row = None
        for idx in range(min(10, len(df))):
            row_str = df.iloc[idx].astype(str).str.upper()
            if 'CODE' in row_str.values:
                code_row = idx
                break
        
        if code_row is None:
            print(f"Warning: Could not find CODE header in {filename}")
            return pd.DataFrame()
        
        # The antibiotic row is the next row (row 4 in typical files)
        antibiotic_row = code_row + 1
        data_start = code_row + 3  # Skip header, antibiotic names, and MIC/INT labels
        
        # Get column indices from code row
        code_header = df.iloc[code_row]
        code_col_idx = None
        isolate_col_idx = None
        scored_res_idx = None
        num_ab_idx = None
        mar_idx = None
        
        for idx, val in enumerate(code_header):
            val_str = str(val).strip().upper()
            if val_str == 'CODE':
                code_col_idx = idx
            elif val_str == 'ISOLATE ID':
                isolate_col_idx = idx
            elif 'SCORED' in val_str and 'RESISTANCE' in val_str:
                scored_res_idx = idx
            elif 'NO.' in val_str and 'ANTIBIOTIC' in val_str:
                num_ab_idx = idx
            elif 'MAR' in val_str and 'INDEX' in val_str:
                mar_idx = idx
        
        # Get antibiotic names from antibiotic row
        ab_header = df.iloc[antibiotic_row]
        antibiotics = []  # List of (column_idx, antibiotic_name)
        esbl_col_idx = None
        
        for idx, val in enumerate(ab_header):
            val_str = str(val).strip().upper()
            if val_str == 'ESBL':
                esbl_col_idx = idx
            elif val_str in STANDARD_ANTIBIOTICS:
                antibiotics.append((idx, val_str))
        
        # Process data rows
        processed_rows = []
        
        for row_idx in range(data_start, len(df)):
            row = df.iloc[row_idx]
            row_data = {}
            
            # Get CODE
            if code_col_idx is not None:
                row_data['CODE'] = row.iloc[code_col_idx]
            
            # Skip rows without valid CODE
            if pd.isna(row_data.get('CODE')) or str(row_data.get('CODE')).strip() == '':
                continue
            
            # Get ISOLATE_ID (species)
            if isolate_col_idx is not None:
                row_data['ISOLATE_ID'] = row.iloc[isolate_col_idx]
            
            # Get ESBL
            if esbl_col_idx is not None:
                row_data['ESBL'] = row.iloc[esbl_col_idx]
            
            # Get antibiotic INT values (interpretation: S, I, R)
            # For each antibiotic at column idx, the INT value is at idx+1
            for ab_col_idx, ab_name in antibiotics:
                int_idx = ab_col_idx + 1  # INT column is right after the antibiotic name/MIC column
                if int_idx < len(row):
                    value = row.iloc[int_idx]
                    # Only store if it's a valid resistance value
                    if pd.notna(value):
                        val_str = str(value).strip().upper()
                        if val_str in ['S', 'I', 'R', '*R']:
                            row_data[ab_name] = val_str.replace('*', '')
            
            # Get summary columns
            if scored_res_idx is not None:
                row_data['SCORED_RESISTANCE'] = row.iloc[scored_res_idx]
            if num_ab_idx is not None:
                row_data['NUM_ANTIBIOTICS_TESTED'] = row.iloc[num_ab_idx]
            if mar_idx is not None:
                row_data['MAR_INDEX'] = row.iloc[mar_idx]
            
            # Add metadata from filename
            row_data['REGION'] = region
            row_data['SITE'] = site
            row_data['SOURCE_FILE'] = filename
            
            # Parse isolate code for additional metadata
            code_metadata = parse_isolate_code(str(row_data.get('CODE', '')))
            row_data.update({
                'NATIONAL_SITE': code_metadata['national_site'],
                'LOCAL_SITE': code_metadata['local_site'],
                'SAMPLE_SOURCE': code_metadata['sample_source'],
                'REPLICATE': code_metadata['replicate'],
                'COLONY': code_metadata['colony']
            })
            
            processed_rows.append(row_data)
        
        return pd.DataFrame(processed_rows)
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_all_csv_files(data_dir: str) -> pd.DataFrame:
    """
    Load and consolidate all CSV files from the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing CSV files
    
    Returns:
    --------
    pd.DataFrame
        Unified raw dataset with metadata columns
    """
    all_data = []
    csv_files = list(Path(data_dir).glob('*.csv'))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        df = process_csv_file(str(csv_file))
        if not df.empty:
            all_data.append(df)
            print(f"  -> Loaded {len(df)} isolates")
    
    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    master_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nTotal isolates loaded: {len(master_df)}")
    print(f"Columns: {list(master_df.columns)}")
    
    return master_df


def create_unified_dataset(input_dir: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main function to create unified raw dataset.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing raw CSV files
    output_path : str, optional
        Path to save the unified dataset
    
    Returns:
    --------
    pd.DataFrame
        Unified raw dataset
    """
    print("=" * 50)
    print("PHASE 2.1: Data Ingestion and Consolidation")
    print("=" * 50)
    
    # Load all CSV files
    master_df = load_all_csv_files(input_dir)
    
    if master_df.empty:
        return master_df
    
    # Reorder columns for clarity
    metadata_cols = ['CODE', 'ISOLATE_ID', 'REGION', 'SITE', 'NATIONAL_SITE', 
                     'LOCAL_SITE', 'SAMPLE_SOURCE', 'REPLICATE', 'COLONY', 
                     'ESBL', 'SOURCE_FILE']
    
    summary_cols = ['SCORED_RESISTANCE', 'NUM_ANTIBIOTICS_TESTED', 'MAR_INDEX']
    
    # Get antibiotic columns (everything else)
    all_cols = set(master_df.columns)
    existing_metadata = [c for c in metadata_cols if c in all_cols]
    existing_summary = [c for c in summary_cols if c in all_cols]
    antibiotic_cols = sorted([c for c in all_cols if c not in metadata_cols + summary_cols])
    
    # Reorder
    new_order = existing_metadata + antibiotic_cols + existing_summary
    master_df = master_df[[c for c in new_order if c in master_df.columns]]
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_df.to_csv(output_path, index=False)
        print(f"\nUnified dataset saved to: {output_path}")
    
    return master_df


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root
    output_path = project_root / "data" / "processed" / "unified_raw_dataset.csv"
    
    df = create_unified_dataset(str(input_dir), str(output_path))
    if not df.empty:
        print("\nDataset Summary:")
        print(f"Shape: {df.shape}")
        print(f"Regions: {df['REGION'].unique()}")
        print(f"Sites: {df['SITE'].unique()}")
