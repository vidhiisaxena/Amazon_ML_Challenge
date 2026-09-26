"""
Master Dataset Cleaning & EDA Pipeline
Amazon ML Challenge 2026 (Entity Resolution)

Runs exploratory data analysis and performs text preprocessing & cleaning
on all training and test data sources based on the Data Processing Guide.
"""

import sys
import os
import time
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from clean_data import clean_dataframe, process_file

sys.stdout.reconfigure(encoding='utf-8')


def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def get_line_count(filepath: str) -> int:
    """Quickly count lines in a file using buffered binary chunks."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'rb') as f:
        return sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024 * 8), b''))


def run_dataset_cleaning(train_only: bool = False):
    pprint("=" * 70)
    pprint("      DATASET PREPROCESSING & CLEANING")
    pprint("=" * 70)
    
    tasks = [
        ("dataset/train/train_source1.tsv", "dataset/cleaned/train_source1.tsv"),
        ("dataset/train/train_source2.tsv", "dataset/cleaned/train_source2.tsv"),
        ("dataset/train/train_source3.tsv", "dataset/cleaned/train_source3.tsv"),
    ]
    if not train_only:
        tasks.extend([
            ("dataset/test/test_source1.tsv", "dataset/cleaned/test_source1.tsv"),
            ("dataset/test/test_source2.tsv", "dataset/cleaned/test_source2.tsv"),
            ("dataset/test/test_source3.tsv", "dataset/cleaned/test_source3.tsv"),
        ])

    t_start = time.time()
    for in_path, out_path in tasks:
        if not os.path.exists(in_path):
            pprint(f"Skipping {in_path} (file not found).")
            continue
            
        in_lines = get_line_count(in_path)
        out_lines = get_line_count(out_path)
        
        if out_lines == in_lines and in_lines > 0:
            pprint(f"\n{out_path} is already completely processed ({out_lines:,} lines). Skipping.")
            continue

        if out_lines > 0:
            pprint(f"\n{out_path} is incomplete ({out_lines:,} / {in_lines:,} lines). Re-cleaning from scratch...")
        else:
            pprint(f"\nProcessing {in_path} ({in_lines:,} lines) -> {out_path}...")

        t0 = time.time()
        process_file(in_path, out_path, chunk_size=500000)
        pprint(f"Completed {out_path} in {time.time() - t0:.2f}s.")

    # Also format and copy ground truth to cleaned/
    gt_in = "dataset/train/train_ground_truth.tsv"
    gt_out = "dataset/cleaned/train_ground_truth.tsv"
    if os.path.exists(gt_in):
        gt_in_lines = get_line_count(gt_in)
        gt_out_lines = get_line_count(gt_out)
        if gt_out_lines == gt_in_lines and gt_in_lines > 0:
            pprint(f"\n{gt_out} is already complete ({gt_out_lines:,} lines). Skipping.")
        else:
            pprint(f"\nFormatting ground truth: {gt_in} -> {gt_out}...")
            gt = pd.read_csv(gt_in, sep='\t', dtype=str)
            gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
            gt.to_csv(gt_out, sep='\t', index=False)
            pprint(f"Saved ground truth to {gt_out}.")

    pprint(f"\nDataset cleaning phase finished in {time.time() - t_start:.2f}s!")



def run_eda_summary():
    pprint("\n" + "=" * 70)
    pprint("      EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    pprint("=" * 70)
    
    cleaned_dir = "dataset/cleaned"
    for fname in sorted(os.listdir(cleaned_dir)):
        if fname.endswith(".tsv") and fname != "train_ground_truth.tsv":
            fpath = os.path.join(cleaned_dir, fname)
            pprint(f"\n--- {fname} ---")
            
            # Read in chunks to compute row count, nulls, and country distribution efficiently
            row_count = 0
            null_counts = {'name_clean': 0, 'addr_clean': 0, 'country': 0}
            country_counts = {}
            sample_rows = []
            
            for chunk in pd.read_csv(fpath, sep='\t', dtype=str, chunksize=500000):
                row_count += len(chunk)
                if len(sample_rows) < 3:
                    sample_rows.extend(chunk[['entity_id', 'business_name', 'name_clean', 'business_address', 'addr_clean', 'country']].head(3 - len(sample_rows)).to_dict('records'))
                
                for c in ['name_clean', 'addr_clean', 'country']:
                    if c in chunk.columns:
                        null_counts[c] += (chunk[c].isna() | (chunk[c] == '')).sum()
                
                if 'country' in chunk.columns:
                    vc = chunk['country'].value_counts()
                    for k, v in vc.items():
                        country_counts[k] = country_counts.get(k, 0) + v
                        
            pprint(f"  Total Rows: {row_count}")
            pprint(f"  Null / Empty Counts: {null_counts}")
            pprint(f"  Country Distribution: {country_counts}")
            pprint("  Samples:")
            for s in sample_rows:
                pprint(f"    {s['entity_id']} ({s['country']}): '{s['business_name']}' -> '{s['name_clean']}' | '{s['business_address']}' -> '{s['addr_clean']}'")

    # Ground Truth Statistics
    gt_path = "dataset/cleaned/train_ground_truth.tsv"
    if os.path.exists(gt_path):
        pprint("\n" + "=" * 70)
        pprint("      GROUND TRUTH STATISTICS")
        pprint("=" * 70)
        gt = pd.read_csv(gt_path, sep='\t', dtype=str)
        gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
        
        total_s1 = len(gt)
        singletons = (gt['matched_entity_ids'] == '').sum()
        non_singletons = total_s1 - singletons
        
        pprint(f"Total S1 entities: {total_s1}")
        pprint(f"Singletons (no match): {singletons} ({singletons / total_s1 * 100:.2f}%)")
        pprint(f"Non-singletons (with matches): {non_singletons} ({non_singletons / total_s1 * 100:.2f}%)")
        
        match_counts = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
        pprint("\nMatch count distribution (first 15 counts):")
        pprint(match_counts.value_counts().sort_index().head(15).to_string())


if __name__ == "__main__":
    train_only = "--all" not in sys.argv
    run_dataset_cleaning(train_only=train_only)
    run_eda_summary()

