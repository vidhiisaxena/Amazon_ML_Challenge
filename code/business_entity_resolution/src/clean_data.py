"""
Data Cleaning Module — Amazon ML Challenge 2026 (Entity Resolution)

Provides functions for business name cleaning, address cleaning, dataframe transformations,
and chunked file processing for large-scale dataset cleaning.
"""

import re
import sys
import unicodedata
import pandas as pd

# Standardize legal suffixes mapping
SUFFIX_DICT = {
    'pvt': 'private',
    'ltd': 'limited',
    'corp': 'corporation',
    'inc': 'incorporated',
    'llc': 'llc',
    'co': 'company',
    '&': 'and',
    'dba': '',
    'the': '',
}
# Precompiled regex pattern for legal suffixes
SUFFIX_COMBINED = re.compile(r'\b(pvt|ltd|corp|inc|llc|co|&|dba|the)\b', re.IGNORECASE)

# Standardize address abbreviations mapping
ADDR_DICT = {
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'dr': 'drive',
    'ct': 'court',
    'ln': 'lane',
    'pkwy': 'parkway',
    'apt': 'apartment',
    'ste': 'suite',
    'fl': 'floor',
}
# Precompiled regex pattern for address abbreviations
ADDR_COMBINED = re.compile(r'\b(rd|st|ave|blvd|dr|ct|ln|pkwy|apt|ste|fl)\b', re.IGNORECASE)

# General regex for non-alphanumeric and extra whitespace
NON_ALPHANUM = re.compile(r'[^\w\s]')
EXTRA_WS = re.compile(r'\s+')


def clean_business_name(name: str) -> str:
    """
    Clean business name according to Entity Resolution specifications:
    - Lowercase & strip
    - Normalize unicode (NFKD) and ASCII conversion
    - Standardize legal suffixes (pvt -> private, ltd -> limited, corp -> corporation, etc.)
    - Strip punctuation while keeping alphanumeric characters & spaces
    - Collapse redundant whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    # Lowercase & strip
    name = name.lower().strip()
    
    # Normalize unicode (NFKD) and decode ascii
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('utf-8')
    
    # Standardize legal suffixes
    name = SUFFIX_COMBINED.sub(lambda m: SUFFIX_DICT[m.group().lower()], name)
    
    # Remove punctuation except alphanumeric and spaces
    name = NON_ALPHANUM.sub(' ', name)
    
    # Collapse whitespace
    return EXTRA_WS.sub(' ', name).strip()


def clean_address(address: str) -> str:
    """
    Clean business address according to Entity Resolution specifications:
    - Lowercase & strip
    - Normalize unicode (NFKD) and ASCII conversion
    - Standardize street/address abbreviations (rd -> road, st -> street, ave -> avenue, etc.)
    - Strip punctuation
    - Collapse redundant whitespace
    """
    if not isinstance(address, str) or not address.strip():
        return ''
    
    # Lowercase & strip
    address = address.lower().strip()
    
    # Normalize unicode (NFKD) and decode ascii
    address = unicodedata.normalize('NFKD', address)
    address = address.encode('ascii', 'ignore').decode('utf-8')
    
    # Standardize address abbreviations
    address = ADDR_COMBINED.sub(lambda m: ADDR_DICT[m.group().lower()], address)
    
    # Remove punctuation except alphanumeric and spaces
    address = NON_ALPHANUM.sub(' ', address)
    
    # Collapse whitespace
    return EXTRA_WS.sub(' ', address).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning to business_name and business_address columns,
    and standardizes country labels in a DataFrame.
    
    Adds 'name_clean' and 'addr_clean' columns.
    """
    out_df = df.copy()
    
    if 'business_name' in out_df.columns:
        out_df['name_clean'] = [clean_business_name(x) for x in out_df['business_name']]
    
    if 'business_address' in out_df.columns:
        out_df['addr_clean'] = [clean_address(x) for x in out_df['business_address']]
        
    if 'country' in out_df.columns:
        out_df['country'] = out_df['country'].fillna('').astype(str).str.strip().str.upper()
        
    return out_df


def process_file(input_path: str, output_path: str, chunk_size: int = 500000) -> None:
    """
    Process a TSV file in chunks and write the cleaned DataFrame to output TSV.
    Uses an atomic temporary file to ensure half-written files are never saved on interruption.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
        
    first_chunk = True
    chunk_num = 0
    total_rows = 0
    
    for chunk in pd.read_csv(input_path, sep='\t', dtype=str, chunksize=chunk_size):
        chunk_num += 1
        total_rows += len(chunk)
        cleaned_chunk = clean_dataframe(chunk)
        mode = 'w' if first_chunk else 'a'
        cleaned_chunk.to_csv(tmp_path, sep='\t', index=False, mode=mode, header=first_chunk)
        first_chunk = False
        print(f"    Chunk {chunk_num}: {total_rows:,} rows cleaned so far...", flush=True)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)

