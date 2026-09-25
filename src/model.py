"""
LightGBM classifier for entity matching.

Training strategy:
- Positives: pairs from ground truth
- Negatives: blocking candidates that are NOT in ground truth (hard negatives)
- GroupKFold by source1_entity_id so the same entity never straddles train/val
- Threshold tuned directly on out-of-fold macro F0.5 (not on log-loss)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import pickle

from src.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _macro_f05(y_true: np.ndarray, y_pred_bin: np.ndarray,
               groups: np.ndarray) -> float:
    """Macro F0.5 — computed per S1 entity then averaged.
    Singletons (no true matches in the group) score 1.0 when y_pred_bin is
    all-zero for that group, 0.0 otherwise."""
    from evaluate import f_beta_score
    group_ids = np.unique(groups)
    scores = []
    for gid in group_ids:
        mask = groups == gid
        yt = y_true[mask]
        yp = y_pred_bin[mask]
        scores.append(f_beta_score(yt, yp, beta=0.5))
    return float(np.mean(scores))


def _tune_threshold(probas: np.ndarray, y_true: np.ndarray,
                    groups: np.ndarray,
                    thresholds=None) -> float:
    """
    Grid-search threshold on OOF predictions.
    Returns threshold that maximises macro F0.5.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = _macro_f05(y_true, (probas >= t).astype(int), groups)
        if f > best_f:
            best_f = f
            best_t = t

    print(f"  Best threshold={best_t:.3f}  F0.5={best_f:.4f}")
    return float(best_t)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
    n_jobs=-1,
)


def train_model(
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    neg_sample_ratio: int = 10,
    verbose: bool = True,
):
    """
    Train LightGBM with GroupKFold CV.

    Parameters
    ----------
    feat_df  : feature DataFrame (rows = pairs)
    labels   : 1 = match, 0 = non-match
    groups   : source1_entity_id per row (for GroupKFold)
    neg_sample_ratio : max negatives per positive to use in training

    Returns
    -------
    model       : trained LGBMClassifier on full data
    oof_probas  : out-of-fold predicted probabilities
    threshold   : F0.5-optimal threshold calibrated on OOF
    """
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y = labels.astype(np.int32)

    oof_probas = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        if verbose:
            print(f"  Fold {fold+1}/{n_splits} — "
                  f"train={len(train_idx):,}  val={len(val_idx):,}")

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Downsample negatives in training fold only
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        max_neg = len(pos_idx) * neg_sample_ratio
        if len(neg_idx) > max_neg:
            rng = np.random.default_rng(42 + fold)
            neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
        keep = np.concatenate([pos_idx, neg_idx])
        keep.sort()
        X_tr, y_tr = X_tr[keep], y_tr[keep]

        pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
        model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=pos_weight)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        oof_probas[val_idx] = model.predict_proba(X_val)[:, 1]

    # Tune threshold on OOF predictions
    threshold = _tune_threshold(oof_probas, y, groups)

    # Retrain on full data with best n_estimators from last fold
    if verbose:
        print("  Retraining on full data…")
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    max_neg = len(pos_idx) * neg_sample_ratio
    if len(neg_idx) > max_neg:
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    keep.sort()
    X_full, y_full = X[keep], y[keep]

    pos_weight = len(y_full[y_full == 0]) / max(len(y_full[y_full == 1]), 1)
    final_model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=pos_weight)
    final_model.fit(X_full, y_full)

    return final_model, oof_probas, threshold


def predict_proba(model, feat_df: pd.DataFrame) -> np.ndarray:
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    return model.predict_proba(X)[:, 1]


def save_model(model, threshold: float, path: str):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)


def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["threshold"]
