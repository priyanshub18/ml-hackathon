"""
Vectorized pair-level feature engineering.

Strategy: merge all required columns from S1 and S23 into the candidates
DataFrame in two fast hash-joins, then compute all features on aligned
numpy arrays.  This avoids per-row Python overhead even at 85M+ pairs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio


# ---------------------------------------------------------------------------
# Low-level vectorized helpers (list comprehensions over numpy object arrays)
# ---------------------------------------------------------------------------

def _jw(a, b):
    return np.fromiter(
        (JaroWinkler.normalized_similarity(x, y) for x, y in zip(a, b)),
        dtype=np.float32, count=len(a))

def _lev(a, b):
    return np.fromiter(
        (1.0 - Levenshtein.normalized_distance(x, y) for x, y in zip(a, b)),
        dtype=np.float32, count=len(a))

def _tsr(a, b):
    return np.fromiter(
        (token_set_ratio(x, y) / 100.0 for x, y in zip(a, b)),
        dtype=np.float32, count=len(a))

def _tsort(a, b):
    return np.fromiter(
        (token_sort_ratio(x, y) / 100.0 for x, y in zip(a, b)),
        dtype=np.float32, count=len(a))

def _c3j(a, b):
    def _ng(s, n=3):
        return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else set()
    def _j(x, y):
        sa, sb = _ng(x), _ng(y)
        if not sa and not sb: return 1.0
        if not sa or not sb:  return 0.0
        return len(sa & sb) / len(sa | sb)
    return np.fromiter((_j(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))

def _c4j(a, b):
    def _ng(s, n=4):
        return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else set()
    def _j(x, y):
        sa, sb = _ng(x), _ng(y)
        if not sa and not sb: return 1.0
        if not sa or not sb:  return 0.0
        return len(sa & sb) / len(sa | sb)
    return np.fromiter((_j(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))

def _tokj(a, b):
    def _j(x, y):
        sa = set(x.split()); sb = set(y.split())
        if not sa and not sb: return 1.0
        if not sa or not sb:  return 0.0
        return len(sa & sb) / len(sa | sb)
    return np.fromiter((_j(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))

def _first_tok_eq(a, b):
    def _eq(x, y):
        tx = x.split(); ty = y.split()
        return int(bool(tx and ty and tx[0] == ty[0]))
    return np.fromiter((_eq(x, y) for x, y in zip(a, b)), dtype=np.int8, count=len(a))

def _numj(a, b):
    def _j(x, y):
        sx = set(x); sy = set(y)
        if not sx and not sy: return 1.0
        if not sx or not sy:  return 0.0
        return len(sx & sy) / len(sx | sy)
    return np.fromiter((_j(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))

def _numovl(a, b):
    return np.fromiter(
        (int(bool(set(x) & set(y))) for x, y in zip(a, b)),
        dtype=np.int8, count=len(a))

def _zipm(a, b):
    def _z(x, y):
        zx = {n for n in x if len(n) in (5, 6)}
        zy = {n for n in y if len(n) in (5, 6)}
        return int(bool(zx & zy))
    return np.fromiter((_z(x, y) for x, y in zip(a, b)), dtype=np.int8, count=len(a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(
    candidates: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pair features for all candidate pairs.

    Uses two pandas hash-joins to pull S1/S23 columns into the candidates
    DataFrame before computing features — O(n_pairs) with minimal Python
    overhead.

    Parameters
    ----------
    candidates : DataFrame with columns
        source1_entity_id, candidate_id, name_score, addr_score,
        numeric_match, source
    s1_df, s23_df : normalized DataFrames (entity_id column required)

    Returns
    -------
    feature_df : DataFrame with FEATURE_COLS columns (same index as candidates)
    """
    S1_COLS  = ["entity_id", "norm_name", "norm_core", "norm_suffix",
                "norm_addr", "numeric_tokens"]
    S23_COLS = ["entity_id", "norm_name", "norm_core", "norm_suffix",
                "norm_addr", "numeric_tokens", "is_non_latin"]

    # Two hash-joins to pull all columns
    df = candidates.copy().reset_index(drop=True)
    df = df.merge(s1_df[S1_COLS], left_on="source1_entity_id",
                  right_on="entity_id", how="left",
                  suffixes=("", "_s1")).drop(columns=["entity_id_s1"] if "entity_id_s1" in df.columns else [])
    df = df.rename(columns={"norm_name":     "s1_norm_name",
                              "norm_core":     "s1_norm_core",
                              "norm_suffix":   "s1_norm_suffix",
                              "norm_addr":     "s1_norm_addr",
                              "numeric_tokens":"s1_nums"})
    df = df.merge(s23_df[S23_COLS], left_on="candidate_id",
                  right_on="entity_id", how="left",
                  suffixes=("", "_cnd")).drop(columns=["entity_id"] if "entity_id" in df.columns else [])
    df = df.rename(columns={"norm_name":     "cnd_norm_name",
                              "norm_core":     "cnd_norm_core",
                              "norm_suffix":   "cnd_norm_suffix",
                              "norm_addr":     "cnd_norm_addr",
                              "numeric_tokens":"cnd_nums",
                              "is_non_latin":  "cnd_non_latin"})

    # Extract aligned arrays
    s1_name  = df["s1_norm_name"].fillna("").values
    cnd_name = df["cnd_norm_name"].fillna("").values
    s1_core  = df["s1_norm_core"].fillna("").values
    cnd_core = df["cnd_norm_core"].fillna("").values
    s1_suf   = df["s1_norm_suffix"].fillna("").values
    cnd_suf  = df["cnd_norm_suffix"].fillna("").values
    s1_addr  = df["s1_norm_addr"].fillna("").values
    cnd_addr = df["cnd_norm_addr"].fillna("").values
    s1_nums  = [x if isinstance(x, list) else [] for x in df["s1_nums"]]
    cnd_nums = [x if isinstance(x, list) else [] for x in df["cnd_nums"]]
    non_lat  = df["cnd_non_latin"].fillna(False).astype(int).values

    # Suffix features
    n = len(df)
    suf_conflict = np.array(
        [(1 if (bool(a) and bool(b) and a != b) else 0) for a, b in zip(s1_suf, cnd_suf)],
        dtype=np.int8)
    suf_match = np.array(
        [(1 if (bool(a) and a == b) else 0) for a, b in zip(s1_suf, cnd_suf)],
        dtype=np.int8)

    # Blocking scores
    name_score = df["name_score"].values.astype(np.float32)
    addr_score = df["addr_score"].values.astype(np.float32)
    blk_num    = df["numeric_match"].values.astype(np.int8)
    is_s2      = (df["source"] == "S2").astype(np.int8).values
    combined   = name_score + addr_score + blk_num.astype(np.float32) * 0.5

    feat = pd.DataFrame({
        # Name
        "jaro_winkler":     _jw(s1_name, cnd_name),
        "levenshtein":      _lev(s1_name, cnd_name),
        "token_set_ratio":  _tsr(s1_name, cnd_name),
        "token_sort_ratio": _tsort(s1_name, cnd_name),
        "char3_jaccard":    _c3j(s1_core, cnd_core),
        "char4_jaccard":    _c4j(s1_core, cnd_core),
        "token_jaccard":    _tokj(s1_core, cnd_core),
        "first_token_eq":   _first_tok_eq(s1_core, cnd_core),
        "suffix_conflict":  suf_conflict,
        "suffix_match":     suf_match,
        "is_non_latin":     non_lat,
        # Address
        "numeric_jaccard":    _numj(s1_nums, cnd_nums),
        "numeric_overlap":    _numovl(s1_nums, cnd_nums),
        "addr_token_jaccard": _tokj(s1_addr, cnd_addr),
        "addr_char3_jaccard": _c3j(s1_addr, cnd_addr),
        "zip_match":          _zipm(s1_nums, cnd_nums),
        "s1_addr_empty":  (s1_addr  == "").astype(np.int8),
        "cnd_addr_empty": (cnd_addr == "").astype(np.int8),
        # Blocking pass-through
        "blk_name_score": name_score,
        "blk_addr_score": addr_score,
        "blk_num_match":  blk_num,
        # Source
        "is_s2": is_s2,
        # Combined
        "combined_score": combined,
    }, index=candidates.index)

    # Context features
    feat["_s1"] = df["source1_entity_id"].values
    feat["cand_rank"] = (
        feat.groupby("_s1")["combined_score"]
             .rank(ascending=False, method="first"))
    max_sc = feat.groupby("_s1")["combined_score"].transform("max")
    feat["score_gap_to_rank1"] = max_sc - feat["combined_score"]
    feat["n_candidates"]        = feat.groupby("_s1")["combined_score"].transform("count")
    feat = feat.drop(columns=["_s1"])

    return feat


FEATURE_COLS = [
    "jaro_winkler", "levenshtein", "token_set_ratio", "token_sort_ratio",
    "char3_jaccard", "char4_jaccard", "token_jaccard", "first_token_eq",
    "suffix_conflict", "suffix_match", "is_non_latin",
    "numeric_jaccard", "numeric_overlap", "addr_token_jaccard",
    "addr_char3_jaccard", "zip_match", "s1_addr_empty", "cnd_addr_empty",
    "blk_name_score", "blk_addr_score", "blk_num_match",
    "is_s2",
    "combined_score", "cand_rank", "score_gap_to_rank1", "n_candidates",
]
