"""
Evaluation utilities.

f_beta_score   — per-entity F_beta score (handles singletons correctly)
macro_f_beta   — macro-average over all S1 entities
blocking_recall — fraction of true matches that appear in candidates
reduction_ratio — how much we reduced the search space
"""

import numpy as np
import pandas as pd


def f_beta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """
    Per-entity F_beta.
    y_true / y_pred : 0/1 arrays for a single S1 entity.
    Singleton (y_true all-zero): returns 1.0 if y_pred also all-zero, else 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_true = y_true.sum()
    n_pred = y_pred.sum()

    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    tp = (y_true & y_pred).sum()
    if n_pred == 0:
        return 0.0

    prec   = tp / n_pred
    rec    = tp / n_true
    b2     = beta ** 2
    denom  = b2 * prec + rec
    if denom == 0:
        return 0.0
    return (1 + b2) * prec * rec / denom


def macro_f_beta(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    beta: float = 0.5,
) -> float:
    """
    Macro-average F_beta over all S1 entities.

    pred_df : columns source1_entity_id, matched_entity_ids (comma-sep or empty)
    gt_df   : same format (ground truth)
    """
    def _parse(s):
        if pd.isna(s) or str(s).strip() == "":
            return set()
        return set(x.strip() for x in str(s).split(",") if x.strip())

    gt_map   = {row.source1_entity_id: _parse(row.matched_entity_ids)
                for row in gt_df.itertuples()}
    pred_map = {row.source1_entity_id: _parse(row.matched_entity_ids)
                for row in pred_df.itertuples()}

    all_s1 = set(gt_map) | set(pred_map)
    scores = []
    for s1_id in all_s1:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        all_cands = true_set | pred_set
        if not all_cands:
            scores.append(1.0)
            continue
        id_list = sorted(all_cands)
        y_true = np.array([1 if c in true_set else 0 for c in id_list])
        y_pred = np.array([1 if c in pred_set else 0 for c in id_list])
        scores.append(f_beta_score(y_true, y_pred, beta=beta))

    return float(np.mean(scores))


def blocking_recall(
    candidates: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> float:
    """
    What fraction of true positive pairs appear in the candidate set?
    candidates : columns source1_entity_id, candidate_id
    gt_df      : columns source1_entity_id, matched_entity_ids
    """
    def _parse(s):
        if pd.isna(s) or str(s).strip() == "":
            return []
        return [x.strip() for x in str(s).split(",") if x.strip()]

    cand_set = set(zip(candidates["source1_entity_id"], candidates["candidate_id"]))

    total = found = 0
    for row in gt_df.itertuples():
        for mid in _parse(row.matched_entity_ids):
            total += 1
            if (row.source1_entity_id, mid) in cand_set:
                found += 1

    return found / total if total > 0 else 1.0


def reduction_ratio(
    candidates: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
) -> float:
    """
    1 - (num_candidate_pairs / total_possible_pairs).
    Higher is better (we eliminated more pairs).
    """
    n_s1  = len(s1_df)
    n_s23 = len(s23_df)
    total_possible = n_s1 * n_s23
    if total_possible == 0:
        return 0.0
    return 1.0 - len(candidates) / total_possible
