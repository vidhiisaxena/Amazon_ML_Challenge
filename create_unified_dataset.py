"""
Merge cleaned training sources 1, 2, and 3 into a single unified dataset file.
Output: dataset/cleaned/train_all_sources.tsv
"""

import os
import sys
import time
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def build_unified_train_dataset():
    out_path = "dataset/cleaned/train_all_sources.tsv"
    tmp_path = out_path + ".tmp"
    
    sources = [
        ("dataset/cleaned/train_source1.tsv", "source1"),
        ("dataset/cleaned/train_source2.tsv", "source2"),
        ("dataset/cleaned/train_source3.tsv", "source3"),
    ]
    
    print("=" * 70, flush=True)
    print("   CREATING UNIFIED TRAINING DATASET: train_all_sources.tsv", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    first_chunk = True
    total_written = 0

    cols_order = ['entity_id', 'source', 'business_name', 'name_clean', 'business_address', 'addr_clean', 'country']

    for file_path, source_label in sources:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")
            
        print(f"\nAdding {file_path} (source: {source_label})...", flush=True)
        t0 = time.time()
        file_rows = 0
        
        for chunk in pd.read_csv(file_path, sep='\t', dtype=str, chunksize=500000):
            chunk['source'] = source_label
            # Reorder columns
            chunk = chunk[[c for c in cols_order if c in chunk.columns]]
            
            mode = 'w' if first_chunk else 'a'
            chunk.to_csv(tmp_path, sep='\t', index=False, mode=mode, header=first_chunk)
            first_chunk = False
            
            file_rows += len(chunk)
            total_written += len(chunk)
            print(f"  Processed chunk: {file_rows:,} rows from {source_label} ({total_written:,} total)...", flush=True)
            
        print(f"Finished {source_label} ({file_rows:,} rows) in {time.time() - t0:.2f}s.", flush=True)

    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(tmp_path, out_path)

    file_size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print("\n" + "=" * 70, flush=True)
    print(f"SUCCESS: Unified training dataset created at {out_path}", flush=True)
    print(f"Total Rows: {total_written:,}", flush=True)
    print(f"File Size: {file_size_gb:.2f} GB", flush=True)
    print(f"Total Time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    build_unified_train_dataset()
