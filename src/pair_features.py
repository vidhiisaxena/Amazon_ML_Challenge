"""
Pair Feature Engineering — Amazon ML Challenge 2026 (Entity Resolution)

Takes the candidate pairs produced by blocking.py (or candidate_pairs.tsv)
and computes similarity features for every (S1, candidate) pair. This
feature table is what the next stage (LightGBM) trains / scores on.

Usage:
    from pair_features import build_feature_table
    feats = build_feature_table(s1_df, s2_df, s3_df, candidates_long)
    feats.to_csv("dataset/cleaned/pair_features.tsv", sep="\t", index=False)
"""

try:
    from rapidfuzz.distance import Levenshtein as rf_lev
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher

import pandas as pd

from blocking import char_ngrams, extract_postal_code, _tokens  # reuse blocking helpers


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(a: set, b: set) -> float:
    """Fraction of the smaller token set contained in the larger one."""
    if not a or not b:
        return 0.0
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    return len(smaller & larger) / len(smaller)


def _seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(rf_lev.normalized_similarity(a, b))
    return SequenceMatcher(None, a, b).ratio()


def _token_sort_ratio(name1: str, name2: str, sorted_name1: str, sorted_name2: str) -> float:
    if _HAS_RAPIDFUZZ:
        return float(rf_fuzz.token_sort_ratio(name1, name2)) / 100.0
    return _seq_ratio(sorted_name1, sorted_name2)


def _common_prefix_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n / max(len(a), len(b))



def pair_feature_dict(name1: str, name2: str, addr1: str, addr2: str,
                       country1: str, country2: str) -> dict:
    name1 = '' if (name1 is None or pd.isna(name1)) else str(name1)
    name2 = '' if (name2 is None or pd.isna(name2)) else str(name2)
    addr1 = '' if (addr1 is None or pd.isna(addr1)) else str(addr1)
    addr2 = '' if (addr2 is None or pd.isna(addr2)) else str(addr2)
    country1 = '' if (country1 is None or pd.isna(country1)) else str(country1)
    country2 = '' if (country2 is None or pd.isna(country2)) else str(country2)


    name_tokens1, name_tokens2 = set(_tokens(name1)), set(_tokens(name2))
    addr_tokens1, addr_tokens2 = set(_tokens(addr1)), set(_tokens(addr2))
    name_ng1, name_ng2 = char_ngrams(name1), char_ngrams(name2)
    addr_ng1, addr_ng2 = char_ngrams(addr1), char_ngrams(addr2)

    sorted_name1 = ' '.join(sorted(name_tokens1))
    sorted_name2 = ' '.join(sorted(name_tokens2))

    postal1, postal2 = extract_postal_code(addr1), extract_postal_code(addr2)

    return {
        'name_exact_match': int(name1 == name2 and name1 != ''),
        'name_jaccard_token': _jaccard(name_tokens1, name_tokens2),
        'name_jaccard_char3gram': _jaccard(name_ng1, name_ng2),
        'name_containment': _containment(name_tokens1, name_tokens2),
        'name_levenshtein_ratio': _seq_ratio(name1, name2),
        'name_token_sort_ratio': _token_sort_ratio(name1, name2, sorted_name1, sorted_name2),
        'name_common_prefix_ratio': _common_prefix_ratio(name1, name2),
        'name_length_diff': abs(len(name_tokens1) - len(name_tokens2)),

        'addr_jaccard_token': _jaccard(addr_tokens1, addr_tokens2),
        'addr_jaccard_char3gram': _jaccard(addr_ng1, addr_ng2),
        'addr_containment': _containment(addr_tokens1, addr_tokens2),
        'addr_levenshtein_ratio': _seq_ratio(addr1, addr2),
        'addr_length_diff': abs(len(addr_tokens1) - len(addr_tokens2)),

        'postal_match': int(bool(postal1) and postal1 == postal2),
        'postal_both_present': int(bool(postal1) and bool(postal2)),
        'country_match': int((country1 or '') == (country2 or '') and bool(country1)),
    }


def build_feature_table(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                         candidates_long: pd.DataFrame) -> pd.DataFrame:
    """
    candidates_long: source1_entity_id, candidate_entity_id, block_score
        (rows with a null candidate_entity_id — i.e. S1 entities with zero
        candidates from blocking — are skipped here since there is no pair
        to featurize)

    Returns one row per (source1_entity_id, candidate_entity_id) pair with
    the block_score carried through plus all similarity features.
    """
    s1_lookup = s1_df.set_index('entity_id')[['name_clean', 'addr_clean', 'country']].to_dict('index')
    pool_lookup = pd.concat([s2_df, s3_df], ignore_index=True) \
        .set_index('entity_id')[['name_clean', 'addr_clean', 'country']].to_dict('index')

    rows = []
    valid_pairs = candidates_long.dropna(subset=['candidate_entity_id'])
    for r in valid_pairs.itertuples(index=False):
        s1_rec = s1_lookup.get(r.source1_entity_id)
        cand_rec = pool_lookup.get(r.candidate_entity_id)
        if s1_rec is None or cand_rec is None:
            continue

        feats = pair_feature_dict(
            s1_rec['name_clean'], cand_rec['name_clean'],
            s1_rec['addr_clean'], cand_rec['addr_clean'],
            s1_rec['country'], cand_rec['country'],
        )
        feats['source1_entity_id'] = r.source1_entity_id
        feats['candidate_entity_id'] = r.candidate_entity_id
        feats['block_score'] = r.block_score
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df

    id_cols = ['source1_entity_id', 'candidate_entity_id', 'block_score']
    other_cols = [c for c in feat_df.columns if c not in id_cols]
    return feat_df[id_cols + other_cols]
