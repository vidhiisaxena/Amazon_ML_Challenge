"""
Blocking + Pair-Feature stage — Amazon ML Challenge 2026 (Entity Resolution)

Reads the already-cleaned files produced by clean_dataset.py
(dataset/cleaned/{train,test}_source{1,2,3}.tsv), runs blocking, and writes:
    dataset/cleaned/{split}_candidate_pairs.tsv     -> submission-format candidate pairs
    dataset/cleaned/{split}_pair_features.tsv       -> features for the LightGBM stage

Run from the repo root (same place you run clean_dataset.py):
    python generate_candidates_and_features.py            # both train and test
    python generate_candidates_and_features.py --train-only
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from blocking import generate_candidates, to_candidate_pairs_tsv
from pair_features import build_feature_table


def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def run_split(split: str, top_k: int):
    base = "dataset/cleaned"
    s1_path = f"{base}/{split}_source1.tsv"
    s2_path = f"{base}/{split}_source2.tsv"
    s3_path = f"{base}/{split}_source3.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not os.path.exists(p):
            pprint(f"Skipping {split} (missing {p}). Run clean_dataset.py first.")
            return

    pprint(f"\n{'=' * 70}\n  BLOCKING + PAIR FEATURES: {split}\n{'=' * 70}")
    t0 = time.time()

    s1 = pd.read_csv(s1_path, sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(s2_path, sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(s3_path, sep='\t', dtype=str).fillna('')

    pprint(f"Loaded S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} rows.")

    cand_long = generate_candidates(s1, s2, s3, top_k=top_k)
    n_with_cands = cand_long.dropna(subset=['candidate_entity_id'])['source1_entity_id'].nunique()
    pprint(f"Blocking done in {time.time() - t0:.1f}s. "
           f"{n_with_cands:,}/{len(s1):,} S1 entities have >=1 candidate.")

    cand_pairs = to_candidate_pairs_tsv(cand_long, list(s1['entity_id']))
    cand_out = f"{base}/{split}_candidate_pairs.tsv"
    cand_pairs.to_csv(cand_out, sep='\t', index=False)
    avg_cands = cand_long.dropna(subset=['candidate_entity_id']).groupby('source1_entity_id').size().mean()
    pprint(f"Wrote {cand_out} (avg {avg_cands:.1f} candidates per matched S1 entity).")

    t1 = time.time()
    feats = build_feature_table(s1, s2, s3, cand_long)
    feat_out = f"{base}/{split}_pair_features.tsv"
    feats.to_csv(feat_out, sep='\t', index=False)
    pprint(f"Wrote {feat_out} ({len(feats):,} pairs, {time.time() - t1:.1f}s for feature computation).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--top-k', type=int, default=50,
                         help='max candidates kept per S1 entity after merging blocking strategies')
    args = parser.parse_args()

    run_split('train', args.top_k)
    if not args.train_only:
        run_split('test', args.top_k)


if __name__ == '__main__':
    main()
