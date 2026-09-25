"""
Decision layer: converts per-pair probabilities → final matching_results.tsv.

Steps:
1. Each S2/S3 record can be assigned to at most one S1 (one-to-one from the
   candidate side — confirmed by profiling). So we can just threshold
   per-pair probabilities; no global assignment solver is needed.
2. Singleton gate: if all probabilities for an S1 are below `t_singleton`,
   predict empty (which scores 1.0 for a true singleton).
3. Per-entity expected-F0.5 selection: for each S1, sort candidates by
   probability and keep the prefix that maximises expected F0.5.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expected F0.5 per entity
# ---------------------------------------------------------------------------

def _expected_f05(proba_sorted: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Given probabilities sorted descending, compute for each prefix length m
    the expected F_beta score if we predict exactly those m candidates.

    E[F_beta | predict top-m] = sum over subsets, but here we use the simple
    greedy approximation: treat probabilities as independent Bernoulli and
    compute E[TP], E[FP], E[FN] for each prefix.

    For each prefix of length m:
        E[TP]  = sum(p_i) for i in 0..m-1
        E[FP]  = m - E[TP]
        E[FN]  = sum(p_i) for i in m..n-1
        E[Prec] = E[TP] / m  (handle m=0 separately)
        E[Rec]  = E[TP] / (E[TP] + E[FN])
    Returns array of expected F_beta for prefix lengths 0..n.
    """
    n = len(proba_sorted)
    cum_tp = np.concatenate([[0.0], np.cumsum(proba_sorted)])
    total_tp = cum_tp[-1]

    scores = np.zeros(n + 1)
    # m=0: predict empty
    # F_beta(empty) = 1 if no true positives (singleton), else 0
    # We can't know that here, so score 0 for non-empty estimator
    # The caller handles the singleton gate separately.
    scores[0] = 0.0

    for m in range(1, n + 1):
        e_tp  = cum_tp[m]
        e_fp  = m - e_tp
        e_fn  = total_tp - e_tp
        denom_prec = m
        denom_rec  = e_tp + e_fn
        if denom_prec == 0 or denom_rec == 0:
            scores[m] = 0.0
            continue
        prec  = e_tp / denom_prec
        rec   = e_tp / denom_rec
        b2    = beta ** 2
        num   = (1 + b2) * prec * rec
        den   = b2 * prec + rec
        scores[m] = num / den if den > 0 else 0.0

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_predictions(
    candidates: pd.DataFrame,
    probas: np.ndarray,
    all_s1_ids,
    threshold: float = 0.5,
    t_singleton: float = 0.2,
    beta: float = 0.5,
    use_expected_f05: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    candidates   : DataFrame with source1_entity_id, candidate_id
    probas       : predicted probability per row (aligned with candidates)
    all_s1_ids   : iterable of ALL S1 entity IDs — every one must appear in output
    threshold    : base probability cutoff
    t_singleton  : if best_prob < t_singleton for an S1, predict empty
    use_expected_f05 : if True, use per-entity expected-F0.5 prefix selection
                       instead of flat threshold

    Returns
    -------
    DataFrame with columns source1_entity_id, matched_entity_ids
    (one row per S1 entity, matched_entity_ids may be empty string)
    """
    cands = candidates.copy()
    cands["proba"] = probas

    results = {}

    # Group by S1
    for s1_id, grp in cands.groupby("source1_entity_id"):
        grp = grp.sort_values("proba", ascending=False).reset_index(drop=True)
        best_prob = grp["proba"].iloc[0]

        # Singleton gate
        if best_prob < t_singleton:
            results[s1_id] = []
            continue

        if use_expected_f05:
            proba_arr = grp["proba"].values
            ef_scores = _expected_f05(proba_arr, beta=beta)
            best_m = int(np.argmax(ef_scores))
            if best_m == 0 or ef_scores[best_m] <= 0:
                results[s1_id] = []
            else:
                results[s1_id] = list(grp["candidate_id"].iloc[:best_m])
        else:
            # Flat threshold
            matched = grp[grp["proba"] >= threshold]["candidate_id"].tolist()
            results[s1_id] = matched

    # Build output DataFrame — every S1 must appear
    rows = []
    for s1_id in all_s1_ids:
        matched = results.get(s1_id, [])
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(str(m) for m in matched),
        })

    return pd.DataFrame(rows)
