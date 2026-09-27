"""
Blocking / Candidate Generation — Amazon ML Challenge 2026 (Entity Resolution)

Given cleaned Source-1 records and a pool of Source-2 + Source-3 records,
produce a small candidate set of plausible matches per Source-1 entity.

Assumes each dataframe already has the columns produced by src/clean_data.py:
    entity_id, business_name, business_address, country, name_clean, addr_clean

Strategies combined (union of all, then capped per S1 entity):
    1. Exact normalized-name match
    2. Rare-token inverted index on name tokens (skips common words like
       "the", "company", "store" that would blow up the candidate set)
    3. Character n-gram inverted index on names, ranked by shared n-gram count
       (robust to typos / transliteration where whole tokens don't line up)
    4. Postal code exact match (extracted from address text)
    5. Country is used as a hard partition: all of the above only ever compare
       records that share the same normalized country value. This is what
       keeps blocking scalable, and it works for any open-set country label
       (US, India, France, or anything else) since nothing is hard-coded.

Usage:
    from blocking import generate_candidates
    candidates_df = generate_candidates(s1_df, s2_df, s3_df, top_k=50)
    # candidates_df: source1_entity_id, candidate_entity_id, block_score
"""

import re
from collections import defaultdict, Counter

import pandas as pd

NGRAM_N = 3
RARE_TOKEN_DF_THRESHOLD = 3       # tokens appearing in <= this many records count as "rare"
TOP_K_DEFAULT = 50                # max candidates kept per S1 entity after merging

POSTAL_PATTERNS = [
    re.compile(r'\b(\d{6})\b'),          # India PIN code (6 digits)
    re.compile(r'\b(\d{5})(?:-\d{4})?\b'),  # US ZIP / ZIP+4
]


def extract_postal_code(address_clean: str):
    """Best-effort postal code extraction; returns None if nothing found."""
    if not isinstance(address_clean, str) or not address_clean:
        return None
    for pat in POSTAL_PATTERNS:
        m = pat.search(address_clean)
        if m:
            return m.group(1)
    return None


def char_ngrams(s: str, n: int = NGRAM_N):
    s = s.replace(' ', '')
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _tokens(s: str):
    if not isinstance(s, str) or not s:
        return []
    return s.split()


class _CountryBlock:
    """Per-country inverted indices built once over the S2+S3 pool."""

    def __init__(self, records):
        # records: list of dicts with entity_id, name_clean, addr_clean
        self.records = records
        self.exact_name = defaultdict(list)      # name_clean -> [entity_id]
        self.token_index = defaultdict(list)      # rare token -> [entity_id]
        self.ngram_index = defaultdict(list)       # char n-gram -> [entity_id]
        self.postal_index = defaultdict(list)      # postal code -> [entity_id]
        self.token_df = Counter()

        for rec in records:
            for tok in set(_tokens(rec['name_clean'])):
                self.token_df[tok] += 1

        for rec in records:
            eid = rec['entity_id']
            name = rec['name_clean']
            self.exact_name[name].append(eid)

            for tok in set(_tokens(name)):
                if self.token_df[tok] <= RARE_TOKEN_DF_THRESHOLD:
                    self.token_index[tok].append(eid)

            for ng in char_ngrams(name):
                self.ngram_index[ng].append(eid)

            postal = extract_postal_code(rec.get('addr_clean', ''))
            if postal:
                self.postal_index[postal].append(eid)

    def query(self, name_clean: str, addr_clean: str, top_k: int) -> dict:
        """Returns {entity_id: score} for one S1 record within this country."""
        scores = Counter()

        # 1. Exact name match — strong signal, fixed bonus
        for eid in self.exact_name.get(name_clean, []):
            scores[eid] += 100

        # 2. Rare token overlap
        for tok in set(_tokens(name_clean)):
            if self.token_df.get(tok, 0) <= RARE_TOKEN_DF_THRESHOLD:
                for eid in self.token_index.get(tok, []):
                    scores[eid] += 10

        # 3. Character n-gram overlap (accumulate shared n-gram counts)
        for ng in char_ngrams(name_clean):
            for eid in self.ngram_index.get(ng, []):
                scores[eid] += 1

        # 4. Postal code match — strong signal
        postal = extract_postal_code(addr_clean)
        if postal:
            for eid in self.postal_index.get(postal, []):
                scores[eid] += 50

        if not scores:
            return {}

        return dict(scores.most_common(top_k))


def _build_pool(df: pd.DataFrame):
    """Groups an S2/S3 dataframe's records by normalized country."""
    pool = defaultdict(list)
    for row in df.itertuples(index=False):
        pool[getattr(row, 'country', '') or ''].append({
            'entity_id': row.entity_id,
            'name_clean': getattr(row, 'name_clean', '') or '',
            'addr_clean': getattr(row, 'addr_clean', '') or '',
        })
    return pool


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                         top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """
    Returns a long-format dataframe:
        source1_entity_id, candidate_entity_id, block_score
    One row per (S1, candidate) pair kept after blocking. Every S1 entity_id
    appears at least once as a placeholder even if it gets zero candidates,
    with candidate_entity_id = None — the caller can drop/keep as needed when
    writing candidate_pairs.tsv (empty string for no-candidate rows).
    """
    combined = pd.concat([s2_df.assign(_src='S2'), s3_df.assign(_src='S3')], ignore_index=True)
    pools_by_country = _build_pool(combined)
    blocks_by_country = {c: _CountryBlock(recs) for c, recs in pools_by_country.items()}

    out_rows = []
    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        country = (getattr(row, 'country', '') or '')
        name_clean = getattr(row, 'name_clean', '') or ''
        addr_clean = getattr(row, 'addr_clean', '') or ''

        block = blocks_by_country.get(country)
        cand_scores = block.query(name_clean, addr_clean, top_k) if block else {}

        if not cand_scores:
            out_rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': None, 'block_score': None})
            continue

        for cand_id, score in cand_scores.items():
            out_rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'block_score': score})

    return pd.DataFrame(out_rows)


def to_candidate_pairs_tsv(candidates_long: pd.DataFrame, s1_ids: list) -> pd.DataFrame:
    """
    Collapses the long-format candidate table into the submission format:
        source1_entity_id, candidate_entity_ids (comma-joined)
    Guarantees exactly one row per S1 id in s1_ids (even with zero candidates).
    """
    grouped = (
        candidates_long.dropna(subset=['candidate_entity_id'])
        .groupby('source1_entity_id')['candidate_entity_id']
        .apply(lambda ids: ','.join(dict.fromkeys(ids)))  # de-dup, preserve order
    )
    result = pd.DataFrame({'source1_entity_id': s1_ids})
    result['candidate_entity_ids'] = result['source1_entity_id'].map(grouped).fillna('')
    return result
