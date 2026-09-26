"""
Blocking / candidate generation.

Memory-safe sharding strategy:
  - TF-IDF vectorizer is FIT once on full corpus (cheap — just vocabulary)
  - S1 is transformed once (small matrix, e.g. 300K × 30K)
  - S23 is transformed and scored in SHARDS of S23_SHARD_SIZE rows
    so peak sparse matrix RAM = shard_size × MAX_FEATURES (not full 5M × 80K)
  - Three passes: name TF-IDF + address TF-IDF + numeric exact match
  - Results merged across shards, then capped at MAX_TOTAL per S1
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K_NAME        = 20      # name TF-IDF top-k per S1 (per source)
TOP_K_ADDR        = 20      # address TF-IDF top-k per S1
MAX_NUMERIC       = 30      # max numeric-match candidates per S1
MAX_TOTAL         = 100     # hard cap on TOTAL candidates per S1 (across S2+S3)
MAX_FEATURES      = 30_000  # reduced from 80K → smaller sparse matrices
MIN_DF            = 3       # raised from 2 → smaller vocabulary
S23_SHARD_SIZE    = 500_000 # transform S23 in shards to limit peak RAM
BATCH_SIZE        = 2_000   # S1 rows per batch for dot product
MIN_TFIDF_SCORE   = 0.05
MIN_NUMERIC_LEN   = 3
MAX_NUMERIC_DOCFREQ = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_tfidf(corpus_texts, max_features=MAX_FEATURES):
    """Fit TF-IDF vectorizer on corpus_texts. Returns fitted vectorizer."""
    vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vect.fit(corpus_texts)
    return vect


def _sparse_topk_batch(s1_csr, s23_shard_csr, k, s23_offset, batch_size=BATCH_SIZE):
    """
    Dot-product top-k for one S23 shard.
    Returns list of (s1_i, s23_global_j, score).
    s23_offset: index offset to convert shard-local j to global j.
    """
    n_s1 = s1_csr.shape[0]
    results = []

    for start in range(0, n_s1, batch_size):
        end   = min(start + batch_size, n_s1)
        batch = s1_csr[start:end]
        scores = batch.dot(s23_shard_csr.T).tocsr()

        for local_i in range(scores.shape[0]):
            rs = scores.indptr[local_i]
            re = scores.indptr[local_i + 1]
            cols = scores.indices[rs:re]
            vals = scores.data[rs:re]

            mask = vals >= MIN_TFIDF_SCORE
            cols, vals = cols[mask], vals[mask]
            if len(cols) == 0:
                continue

            if len(cols) > k:
                top_idx = np.argpartition(-vals, k)[:k]
                cols, vals = cols[top_idx], vals[top_idx]

            s1_global = start + local_i
            for c, v in zip(cols, vals):
                results.append((s1_global, int(c) + s23_offset, float(v)))

    return results


def _tfidf_pass_sharded(s1_df, s23_df, field, k):
    """
    TF-IDF blocking pass, processing S23 in memory-safe shards.

    Fits vectorizer on full corpus (cheap), transforms S1 once,
    then transforms S23 in chunks of S23_SHARD_SIZE rows.

    Returns list of (s1_idx, s23_idx, score) — s23_idx is global row index.
    """
    s1_text  = s1_df[field].fillna("").astype(str).tolist()
    s23_text = s23_df[field].fillna("").astype(str).tolist()

    # Fit on full corpus — just builds vocabulary, not matrices
    vect = _fit_tfidf(s1_text + s23_text)

    # Transform S1 once (small)
    s1_mat = sk_normalize(vect.transform(s1_text), norm="l2").tocsr()

    # Per-(s1_i) accumulate best-k across all shards
    best_per_s1 = defaultdict(list)   # s1_i → [(s23_global_j, score)]

    n_s23 = len(s23_text)
    for shard_start in range(0, n_s23, S23_SHARD_SIZE):
        shard_end  = min(shard_start + S23_SHARD_SIZE, n_s23)
        shard_text = s23_text[shard_start:shard_end]

        shard_mat = sk_normalize(vect.transform(shard_text), norm="l2").tocsr()
        triples   = _sparse_topk_batch(s1_mat, shard_mat, k, shard_start)

        for s1_i, s23_j, sc in triples:
            best_per_s1[s1_i].append((s23_j, sc))

        del shard_mat  # free shard matrix immediately

    # Merge: keep global top-k per S1
    results = []
    for s1_i, pairs in best_per_s1.items():
        if len(pairs) > k:
            pairs.sort(key=lambda x: -x[1])
            pairs = pairs[:k]
        for s23_j, sc in pairs:
            results.append((s1_i, s23_j, sc))

    return results


def _numeric_pass(s1_df, s23_df, max_per_s1=MAX_NUMERIC,
                  min_len=MIN_NUMERIC_LEN, max_df=MAX_NUMERIC_DOCFREQ):
    """
    Exact numeric-token match with inverted index.
    Only uses tokens that are long enough and rare enough to be discriminative.
    Returns list of (s1_idx, s23_idx, token_count).
    """
    doc_freq = defaultdict(int)
    for tokens in s23_df["numeric_tokens"]:
        for tok in tokens:
            if len(tok) >= min_len:
                doc_freq[tok] += 1

    inv_index = defaultdict(list)
    for j, tokens in enumerate(s23_df["numeric_tokens"]):
        for tok in tokens:
            if len(tok) >= min_len and doc_freq[tok] <= max_df:
                inv_index[tok].append(j)

    results = []
    for i, tokens in enumerate(s1_df["numeric_tokens"]):
        hit_counts = defaultdict(int)
        for tok in tokens:
            if len(tok) >= min_len and doc_freq.get(tok, 0) <= max_df:
                for j in inv_index.get(tok, []):
                    hit_counts[j] += 1

        sorted_hits = sorted(hit_counts.items(), key=lambda x: -x[1])
        for j, cnt in sorted_hits[:max_per_s1]:
            results.append((i, j, cnt))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k_name: int = TOP_K_NAME,
    top_k_addr: int = TOP_K_ADDR,
    verbose: bool = True,
    gt_pairs: set = None,
) -> pd.DataFrame:
    """
    Produce candidate (S1, S2/S3) pairs for scoring.

    Memory-safe: S23 TF-IDF transform is sharded so peak matrix RAM
    = S23_SHARD_SIZE × MAX_FEATURES (not full corpus size).

    Parameters
    ----------
    s1_df, s2_df, s3_df : normalized DataFrames with entity_id, country,
        norm_name, norm_addr, numeric_tokens columns.
    gt_pairs : optional set of (s1_id, s23_id) for per-pass recall logging.

    Returns
    -------
    DataFrame: source1_entity_id, candidate_id, name_score, addr_score,
               numeric_match, source
    """
    all_pairs = []
    countries = s1_df["country"].unique()

    after_name_pairs = set()
    after_addr_pairs = set()
    after_num_pairs  = set()

    for country in countries:
        if verbose:
            print(f"  [blocking] country={country}", flush=True)

        s1_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s2_c = s2_df[s2_df["country"] == country].reset_index(drop=True)
        s3_c = s3_df[s3_df["country"] == country].reset_index(drop=True)

        if len(s1_c) == 0:
            continue

        source_results = {}

        for src_name, s23_c in [("S2", s2_c), ("S3", s3_c)]:
            if len(s23_c) == 0:
                continue
            if verbose:
                print(f"    {src_name}: {len(s1_c):,} S1 × {len(s23_c):,} {src_name}", flush=True)

            name_triples = _tfidf_pass_sharded(s1_c, s23_c, "norm_name", top_k_name)
            addr_triples = _tfidf_pass_sharded(s1_c, s23_c, "norm_addr", top_k_addr)
            num_triples  = _numeric_pass(s1_c, s23_c)

            name_map = {(i, j): sc  for i, j, sc  in name_triples}
            addr_map = {(i, j): sc  for i, j, sc  in addr_triples}
            num_map  = {(i, j): cnt for i, j, cnt in num_triples}

            # Per-pass recall tracking
            if gt_pairs is not None:
                for (i, j) in name_map:
                    after_name_pairs.add((s1_c.iloc[i]["entity_id"], s23_c.iloc[j]["entity_id"]))
                for (i, j) in set(name_map) | set(addr_map):
                    after_addr_pairs.add((s1_c.iloc[i]["entity_id"], s23_c.iloc[j]["entity_id"]))
                for (i, j) in set(name_map) | set(addr_map) | set(num_map):
                    after_num_pairs.add((s1_c.iloc[i]["entity_id"], s23_c.iloc[j]["entity_id"]))

            all_keys = set(name_map) | set(addr_map) | set(num_map)

            pair_info = {}
            for (i, j) in all_keys:
                n_sc     = name_map.get((i, j), 0.0)
                a_sc     = addr_map.get((i, j), 0.0)
                n_cnt    = num_map.get((i, j), 0)
                combined = n_sc + a_sc + min(n_cnt, 5) * 0.1
                s23_id   = s23_c.iloc[j]["entity_id"]
                pair_info[(i, s23_id)] = (n_sc, a_sc, int((i, j) in num_map), combined, src_name)

            source_results[src_name] = pair_info

        # Merge S2 + S3, apply global cap per S1
        s1_buckets = defaultdict(list)
        for src_name, pair_info in source_results.items():
            for (i, s23_id), (n_sc, a_sc, num, combined, src) in pair_info.items():
                s1_buckets[i].append((s23_id, n_sc, a_sc, num, combined, src))

        for i, cand_list in s1_buckets.items():
            cand_list.sort(key=lambda x: -x[4])
            cand_list = cand_list[:MAX_TOTAL]
            s1_id = s1_c.iloc[i]["entity_id"]
            for (s23_id, n_sc, a_sc, num, _combined, src) in cand_list:
                all_pairs.append({
                    "source1_entity_id": s1_id,
                    "candidate_id":      s23_id,
                    "name_score":        n_sc,
                    "addr_score":        a_sc,
                    "numeric_match":     num,
                    "source":            src,
                })

    # Per-pass recall summary
    if gt_pairs is not None and verbose and gt_pairs:
        total = len(gt_pairs)
        def _r(pairs):
            n = len(gt_pairs & pairs)
            return f"{n}/{total} = {n/total:.4f}"
        print(f"\n  === Blocking Recall by Pass ===")
        print(f"  After name  TF-IDF : {_r(after_name_pairs)}")
        print(f"  After + addr TF-IDF: {_r(after_addr_pairs)}")
        print(f"  After + numeric    : {_r(after_num_pairs)}")
        print(f"  MAX_TOTAL={MAX_TOTAL}  shard={S23_SHARD_SIZE:,}  features={MAX_FEATURES:,}")

    if not all_pairs:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_id",
            "name_score", "addr_score", "numeric_match", "source"
        ])

    df = pd.DataFrame(all_pairs)
    df["_combined"] = df["name_score"] + df["addr_score"] + df["numeric_match"].astype(float)
    df = (df.sort_values("_combined", ascending=False)
            .drop_duplicates(subset=["source1_entity_id", "candidate_id"])
            .drop(columns=["_combined"])
            .reset_index(drop=True))
    return df
