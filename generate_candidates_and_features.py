"""
High-Performance Streaming Blocking + Pair Features Pipeline
Amazon ML Challenge 2026 (Entity Resolution)

Runs blocking.py and pair_features.py on dataset/cleaned/ data.

Features:
- Low-memory country streaming (RAM stays < 1 GB).
- RapidFuzz C++ string similarity for blazingly fast feature extraction.
- Automatic resume if interrupted.
- Streams results directly to dataset/cleaned/{split}_candidate_pairs.tsv and pair_features.tsv.

Usage:
    python generate_candidates_and_features.py --train-only
    python generate_candidates_and_features.py
"""

import argparse
import gc
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from blocking import _CountryBlock
from pair_features import pair_feature_dict

sys.stdout.reconfigure(encoding='utf-8')


def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def run_split(split: str, top_k: int = 50, chunk_size: int = 25000, max_s1: int = None):
    base = "dataset/cleaned"
    s1_path = f"{base}/{split}_source1.tsv"
    s2_path = f"{base}/{split}_source2.tsv"
    s3_path = f"{base}/{split}_source3.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not os.path.exists(p):
            pprint(f"Error: Missing {p}. Run clean_dataset.py first.")
            return

    cand_out = f"{base}/{split}_candidate_pairs.tsv"
    feat_out = f"{base}/{split}_pair_features.tsv"

    pprint("=" * 70)
    pprint(f"  BLOCKING & PAIR FEATURES PIPELINE: {split.upper()}")
    pprint("=" * 70)
    t_start = time.time()

    # Step 1: Discover countries present in S1
    s1_summary = pd.read_csv(s1_path, sep='\t', usecols=['country'], dtype=str)
    countries = sorted(s1_summary['country'].dropna().unique().tolist())
    pprint(f"Discovered countries in {split}: {countries}")
    del s1_summary
    gc.collect()

    # Check for resume
    processed_s1_ids = set()
    if os.path.exists(cand_out):
        pprint(f"Found existing candidate pairs at {cand_out}. Checking processed records...")
        for chunk in pd.read_csv(cand_out, sep='\t', usecols=['source1_entity_id'], chunksize=500000, dtype=str):
            processed_s1_ids.update(chunk['source1_entity_id'])
        pprint(f"Resuming: {len(processed_s1_ids):,} S1 entities already processed.")

    first_cand = not os.path.exists(cand_out) or len(processed_s1_ids) == 0
    first_feat = not os.path.exists(feat_out) or len(processed_s1_ids) == 0

    total_s1_processed = len(processed_s1_ids)
    total_pairs_written = 0

    for country in countries:
        pprint(f"\n{'=' * 50}")
        pprint(f"  PROCESSING COUNTRY: {country}")
        pprint(f"{'=' * 50}")

        # Step 2: Load only this country's pool from S2 and S3 in chunks
        t0 = time.time()
        pprint(f"Loading {country} candidate records from S2 and S3...")
        pool_records = []
        pool_lookup = {}
        
        for spath in [s2_path, s3_path]:
            for chunk in pd.read_csv(spath, sep='\t', usecols=['entity_id', 'country', 'name_clean', 'addr_clean'], chunksize=500000, dtype=str):
                mask = chunk['country'] == country
                sub = chunk[mask]
                if not sub.empty:
                    for row in sub.itertuples(index=False):
                        rec = {
                            'entity_id': row.entity_id,
                            'name_clean': row.name_clean or '',
                            'addr_clean': row.addr_clean or '',
                            'country': country
                        }
                        pool_records.append(rec)
                        pool_lookup[row.entity_id] = rec

        pool_size = len(pool_records)
        pprint(f"Loaded {pool_size:,} pool records for {country} in {time.time() - t0:.2f}s.")
        if pool_size == 0:
            pprint(f"No candidates for {country}. Skipping.")
            continue

        # Step 3: Build fast index
        t0 = time.time()
        pprint(f"Building fast inverted index for {country} ({pool_size:,} candidates)...")
        block = _CountryBlock(pool_records)
        pprint(f"Index built in {time.time() - t0:.2f}s.")

        # Step 4: Stream S1 records for this country
        pprint(f"Streaming S1 queries for {country} in chunks of {chunk_size:,}...")
        for s1_chunk in pd.read_csv(s1_path, sep='\t', usecols=['entity_id', 'country', 'name_clean', 'addr_clean'], chunksize=chunk_size, dtype=str):
            sub_s1 = s1_chunk[s1_chunk['country'] == country]
            if sub_s1.empty:
                continue

            # Skip already processed if resuming
            if processed_s1_ids:
                sub_s1 = sub_s1[~sub_s1['entity_id'].isin(processed_s1_ids)]
                if sub_s1.empty:
                    continue

            if max_s1 and total_s1_processed >= max_s1:
                pprint(f"Reached max_s1 limit ({max_s1:,}). Stopping.")
                return

            t_chunk = time.time()
            cand_rows = []
            feat_rows = []

            for row in sub_s1.itertuples(index=False):
                s1_id = row.entity_id
                name1 = row.name_clean or ''
                addr1 = row.addr_clean or ''

                cand_scores = block.query(name1, addr1, top_k)
                if not cand_scores:
                    cand_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
                    continue

                matched_cand_ids = list(cand_scores.keys())
                cand_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ','.join(matched_cand_ids)})

                for cand_id, score in cand_scores.items():
                    cand_rec = pool_lookup.get(cand_id)
                    if cand_rec is None:
                        continue
                    feats = pair_feature_dict(
                        name1, cand_rec['name_clean'],
                        addr1, cand_rec['addr_clean'],
                        country, country,
                    )
                    feats['source1_entity_id'] = s1_id
                    feats['candidate_entity_id'] = cand_id
                    feats['block_score'] = score
                    feat_rows.append(feats)

            # Append candidates
            cand_df = pd.DataFrame(cand_rows)
            mode_c = 'w' if first_cand else 'a'
            cand_df.to_csv(cand_out, sep='\t', index=False, mode=mode_c, header=first_cand)
            first_cand = False

            # Append features
            if feat_rows:
                feat_df = pd.DataFrame(feat_rows)
                id_cols = ['source1_entity_id', 'candidate_entity_id', 'block_score']
                other_cols = [c for c in feat_df.columns if c not in id_cols]
                feat_df = feat_df[id_cols + other_cols]

                mode_f = 'w' if first_feat else 'a'
                feat_df.to_csv(feat_out, sep='\t', index=False, mode=mode_f, header=first_feat)
                first_feat = False
                total_pairs_written += len(feat_rows)

            total_s1_processed += len(sub_s1)
            chunk_time = time.time() - t_chunk
            qps = len(sub_s1) / max(0.001, chunk_time)
            pprint(f"  [{country}] Processed {len(sub_s1):,} S1 ({total_s1_processed:,} total, {len(feat_rows):,} pairs featurized) in {chunk_time:.1f}s ({qps:.0f} S1/s)")

            if max_s1 and total_s1_processed >= max_s1:
                pprint(f"Reached max_s1 limit ({max_s1:,}). Stopping.")
                return

        del block, pool_records, pool_lookup
        gc.collect()

    pprint("\n" + "=" * 70)
    pprint(f"SUCCESS: Pipeline completed in {time.time() - t_start:.2f}s!")
    pprint(f"Candidate pairs: {cand_out} ({total_s1_processed:,} S1 entities)")
    pprint(f"Pair features:   {feat_out} ({total_pairs_written:,} candidate pairs)")
    pprint("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-only', action='store_true', default=True, help='Process only the training dataset')
    parser.add_argument('--top-k', type=int, default=50, help='Max candidates kept per S1 entity')
    parser.add_argument('--chunk-size', type=int, default=25000, help='Number of S1 entities per streaming chunk')
    parser.add_argument('--max-s1', type=int, default=None, help='Optional limit on number of S1 entities')
    args = parser.parse_args()

    run_split('train', top_k=args.top_k, chunk_size=args.chunk_size, max_s1=args.max_s1)
    if not args.train_only:
        run_split('test', top_k=args.top_k, chunk_size=args.chunk_size, max_s1=args.max_s1)


if __name__ == '__main__':
    main()
