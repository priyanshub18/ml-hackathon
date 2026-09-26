"""
Blocking / candidate generation.

Three passes per country, applied to S2 and S3 combined:
  1. TF-IDF on normalized NAME  — catches Latin name matches
  2. TF-IDF on normalized ADDRESS — catches transliteration cases where
     the address is ASCII even when the name is in a different script
  3. Numeric-token exact match — house / plot numbers, ZIP, PIN; very precise

MAX_TOTAL caps the total number of candidates per S1 across all passes and
both sources.  IDF is fit on the union of the corpus supplied + optional extra
documents so that France test vocabulary is covered.
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
TOP_K_NAME   = 20    # name TF-IDF top-k per S1 (per source separately)
TOP_K_ADDR   = 20    # address TF-IDF top-k per S1
MAX_NUMERIC  = 30    # max numeric-match candidates per S1
MAX_TOTAL    = 100   # hard cap on TOTAL candidates per S1 (across S2+S3)
MAX_FEATURES = 80_000
MIN_DF       = 2
BATCH_SIZE   = 3_000       # S1 rows per batch for sparse matrix multiply
MIN_TFIDF_SCORE   = 0.05   # discard very low cosine pairs
MIN_NUMERIC_LEN   = 3      # ignore 1- and 2-digit numbers
MAX_NUMERIC_DOCFREQ = 500  # skip tokens that appear in > N S23 records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_tfidf(corpus_texts, extra_texts=None, max_features=MAX_FEATURES):
    """Fit TF-IDF on corpus_texts + optional extra_texts."""
    texts = list(corpus_texts)
    if extra_texts is not None and len(extra_texts) > 0:
        texts.extend(extra_texts)
    vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=MIN_DF,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vect.fit(texts)
    return vect


def _sparse_topk_with_scores(s1_mat, s23_mat, k, batch_size=BATCH_SIZE):
    """
    For each row in s1_mat, find top-k rows in s23_mat by cosine similarity
    (both matrices must be L2-normalised).

    Returns list of (s1_i, s23_i, score) triples.
    """
    n_s1 = s1_mat.shape[0]
    results = []

    s1_csr  = s1_mat.tocsr()
    s23_csr = s23_mat.tocsr()

    for start in range(0, n_s1, batch_size):
        end   = min(start + batch_size, n_s1)
        batch = s1_csr[start:end]           # (batch_sz, vocab)
        scores = batch.dot(s23_csr.T)       # (batch_sz, n_s23) — sparse
        scores_csr = scores.tocsr()

        for local_i in range(scores_csr.shape[0]):
            rs = scores_csr.indptr[local_i]
            re = scores_csr.indptr[local_i + 1]
            cols = scores_csr.indices[rs:re]
            vals = scores_csr.data[rs:re]

            # Filter by minimum score first
            mask = vals >= MIN_TFIDF_SCORE
            cols = cols[mask]
            vals = vals[mask]

            if len(cols) == 0:
                continue
            if len(cols) > k:
                top_local = np.argpartition(-vals, k)[:k]
                cols = cols[top_local]
                vals = vals[top_local]

            s1_global = start + local_i
            for c, v in zip(cols, vals):
                results.append((s1_global, int(c), float(v)))

    return results


def _tfidf_pass(s1_df, s23_df, field, k,
                extra_texts_s1=None, extra_texts_s23=None):
    """
    TF-IDF blocking pass on a single field column.
    Returns list of (s1_idx, s23_idx, score).
    """
    s1_text  = s1_df[field].fillna("").astype(str)
    s23_text = s23_df[field].fillna("").astype(str)

    extra = None
    if extra_texts_s1 is not None and extra_texts_s23 is not None:
        extra = list(extra_texts_s1) + list(extra_texts_s23)

    vect    = _fit_tfidf(pd.concat([s1_text, s23_text]), extra)
    s1_mat  = sk_normalize(vect.transform(s1_text),  norm="l2")
    s23_mat = sk_normalize(vect.transform(s23_text), norm="l2")

    return _sparse_topk_with_scores(s1_mat, s23_mat, k)


def _numeric_pass(s1_df, s23_df, max_per_s1=MAX_NUMERIC,
                  min_len=MIN_NUMERIC_LEN, max_df=MAX_NUMERIC_DOCFREQ):
    """
    Exact numeric-token match.  Only uses tokens that are:
      - long enough (≥ min_len digits) to be discriminative
      - rare enough (appears in ≤ max_df S23 records) to avoid noise

    Returns list of (s1_idx, s23_idx, token_count).
    """
    # Document frequency of each token in S23
    doc_freq = defaultdict(int)
    for tokens in s23_df["numeric_tokens"]:
        for tok in tokens:
            if len(tok) >= min_len:
                doc_freq[tok] += 1

    # Inverted index for discriminative tokens only
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

    Passes: name TF-IDF + address TF-IDF + numeric exact match, per country.
    MAX_TOTAL caps total candidates per S1 entity across all sources / passes.

    Parameters
    ----------
    s1_df, s2_df, s3_df : normalized DataFrames (entity_id, norm_name,
        norm_addr, numeric_tokens, country are required).
    gt_pairs : optional set of (s1_entity_id, s23_entity_id) ground-truth
        pairs — used to print per-pass recall for analysis.

    Returns
    -------
    DataFrame: source1_entity_id, candidate_id, name_score, addr_score,
               numeric_match, source (S2/S3)
    """
    all_pairs = []
    countries = s1_df["country"].unique()

    # Accumulators for global per-pass recall tracking
    after_name_pairs  = set()
    after_addr_pairs  = set()
    after_num_pairs   = set()

    for country in countries:
        if verbose:
            print(f"  [blocking] country={country}")

        s1_c  = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s2_c  = s2_df[s2_df["country"] == country].reset_index(drop=True)
        s3_c  = s3_df[s3_df["country"] == country].reset_index(drop=True)

        if len(s1_c) == 0:
            continue

        # Run passes for each source separately to keep matrices manageable
        source_results = {}   # source → dict[(s1_i, s23_i)] → (name_sc, addr_sc, num_match)

        for src_name, s23_c in [("S2", s2_c), ("S3", s3_c)]:
            if len(s23_c) == 0:
                continue
            if verbose:
                print(f"    {src_name}: {len(s1_c):,} S1 × {len(s23_c):,} {src_name}")

            name_triples = _tfidf_pass(s1_c, s23_c, "norm_name", top_k_name)
            addr_triples = _tfidf_pass(s1_c, s23_c, "norm_addr", top_k_addr)
            num_triples  = _numeric_pass(s1_c, s23_c)

            name_map = {(i, j): sc for i, j, sc in name_triples}
            addr_map = {(i, j): sc for i, j, sc in addr_triples}
            num_map  = {(i, j): cnt for i, j, cnt in num_triples}

            # Track per-pass entity-id pairs for recall logging
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

        # Merge S2 and S3 results, then cap per S1
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

    # Print per-pass recall summary
    if gt_pairs is not None and verbose:
        total_gt = len(gt_pairs)
        def _r(pairs):
            n = len(gt_pairs & pairs)
            return f"{n}/{total_gt} = {n/total_gt:.4f}" if total_gt else "N/A"
        print(f"\n  === Blocking Recall by Pass ===")
        print(f"  After name  TF-IDF : {_r(after_name_pairs)}")
        print(f"  After addr  TF-IDF : {_r(after_addr_pairs)}  (+addr contribution)")
        print(f"  After numeric match: {_r(after_num_pairs)}  (+numeric contribution)")
        print(f"  MAX_TOTAL cap={MAX_TOTAL}  (applied after union of all passes)")

    if not all_pairs:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_id",
            "name_score", "addr_score", "numeric_match", "source"
        ])

    df = pd.DataFrame(all_pairs)
    # Final dedup (shouldn't be needed but safeguard)
    df["_combined"] = df["name_score"] + df["addr_score"] + df["numeric_match"].astype(float)
    df = (df.sort_values("_combined", ascending=False)
            .drop_duplicates(subset=["source1_entity_id", "candidate_id"])
            .drop(columns=["_combined"])
            .reset_index(drop=True))
    return df
