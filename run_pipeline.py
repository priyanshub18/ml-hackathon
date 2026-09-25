"""
End-to-end entity resolution pipeline.

Timeline estimate (on a 4-core laptop without GPU):
  - Normalize S2/S3 (10M each):  ~4 min
  - Block 300K train S1:         ~45 min
  - Feature train (~15M pairs):  ~10 min
  - LightGBM train (5-fold):     ~10 min
  - Block 1.7M test S1:          ~3.5 h
  - Feature test (~85M pairs):   ~45 min
  - Decision + outputs:          ~5 min
  Total:                         ~5.5 hours

Run from ml-hackathon/ directory:
    python run_pipeline.py
"""

import os, sys, time, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.normalize import normalize_df
from src.blocking  import generate_candidates
from src.features  import compute_features, FEATURE_COLS
from src.model     import train_model, predict_proba, save_model, load_model
from src.decide    import make_predictions
from evaluate      import macro_f_beta, blocking_recall, reduction_ratio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Auto-detect Kaggle input path if running on Kaggle
def _auto_detect_paths():
    kaggle_input = "/kaggle/input"
    if not os.path.isdir(kaggle_input):
        return None, None
    # Look for train_source1.tsv anywhere under /kaggle/input
    for root, dirs, files in os.walk(kaggle_input):
        if "train_source1.tsv" in files:
            train_dir = root
            # Guess test dir: sibling folder named 'test'
            parent = os.path.dirname(root)
            test_dir = os.path.join(parent, "test")
            if not os.path.isdir(test_dir):
                test_dir = root  # fallback: same folder
            return train_dir, test_dir
    return None, None

_auto_train, _auto_test = _auto_detect_paths()

DATA_TRAIN  = os.environ.get("DATA_TRAIN",  _auto_train  or "student_resource/dataset/train")
DATA_TEST   = os.environ.get("DATA_TEST",   _auto_test   or "student_resource/dataset/test")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/kaggle/working" if os.path.isdir("/kaggle/working") else "output")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model.pkl")
CACHE_DIR   = os.path.join(OUTPUT_DIR, "cache")

TRAIN_SAMPLE_S1   = 300_000
TOP_K_NAME        = 20
TOP_K_ADDR        = 20
NEG_SAMPLE_RATIO  = 10
N_FOLDS           = 5
T_SINGLETON       = 0.15
RANDOM_SEED       = 42
CHUNK_SIZE        = 5_000_000   # rows per feature-computation chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def load_and_normalize(s1_path, s2_path, s3_path, tag=""):
    print(f"[{ts()}] [{tag}] Loading…")
    s1 = pd.read_csv(s1_path, sep="\t")
    s2 = pd.read_csv(s2_path, sep="\t")
    s3 = pd.read_csv(s3_path, sep="\t")
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    print(f"[{ts()}] [{tag}] Normalizing S2/S3…")
    t0 = time.time()
    s2 = normalize_df(s2)
    s3 = normalize_df(s3)
    print(f"  {time.time()-t0:.0f}s")
    print(f"[{ts()}] [{tag}] Normalizing S1…")
    t0 = time.time()
    s1 = normalize_df(s1)
    print(f"  {time.time()-t0:.0f}s")
    s23 = pd.concat([s2, s3], ignore_index=True)
    return s1, s2, s3, s23


def label_candidates(cands, gt_df):
    gt_set = set()
    for row in gt_df.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not str(row.matched_entity_ids).strip():
            continue
        for mid in str(row.matched_entity_ids).split(","):
            mid = mid.strip()
            if mid:
                gt_set.add((row.source1_entity_id, mid))
    cands = cands.copy()
    cands["label"] = cands.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_id"]) in gt_set), axis=1
    )
    return cands, gt_set


def compute_features_chunked(cands, s1_df, s23_df, chunk_size=CHUNK_SIZE):
    """Compute features in chunks to limit peak memory usage."""
    parts = []
    for start in range(0, len(cands), chunk_size):
        chunk = cands.iloc[start:start + chunk_size]
        parts.append(compute_features(chunk, s1_df, s23_df))
        if (start // chunk_size) % 5 == 0:
            print(f"    chunk {start//chunk_size + 1} ({start:,} / {len(cands):,})", flush=True)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    # ===== 1. Load & normalize train =====
    s1_tr, s2_tr, s3_tr, s23_tr = load_and_normalize(
        f"{DATA_TRAIN}/train_source1.tsv",
        f"{DATA_TRAIN}/train_source2.tsv",
        f"{DATA_TRAIN}/train_source3.tsv",
        tag="TRAIN",
    )
    gt_full = pd.read_csv(f"{DATA_TRAIN}/train_ground_truth.tsv", sep="\t")

    # ===== 2. Load & normalize test =====
    s1_te, s2_te, s3_te, s23_te = load_and_normalize(
        f"{DATA_TEST}/test_source1.tsv",
        f"{DATA_TEST}/test_source2.tsv",
        f"{DATA_TEST}/test_source3.tsv",
        tag="TEST",
    )

    # ===== 3. Sample train S1 =====
    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(TRAIN_SAMPLE_S1, len(s1_tr))
    sample_ids = rng.choice(s1_tr["entity_id"].values, size=n_sample, replace=False)
    s1_sample  = s1_tr[s1_tr["entity_id"].isin(set(sample_ids))].reset_index(drop=True)
    gt_sample  = gt_full[gt_full["source1_entity_id"].isin(set(sample_ids))].reset_index(drop=True)
    print(f"[{ts()}] Train sample: {len(s1_sample):,} S1  GT rows: {len(gt_sample):,}")

    # Validation split (hold-out within sample)
    val_frac   = 0.1
    val_n      = int(len(s1_sample) * val_frac)
    val_ids    = set(rng.choice(s1_sample["entity_id"].values, size=val_n, replace=False))
    s1_trn     = s1_sample[~s1_sample["entity_id"].isin(val_ids)].reset_index(drop=True)
    s1_val     = s1_sample[ s1_sample["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_val     = gt_sample[gt_sample["source1_entity_id"].isin(val_ids)]

    # ===== 4. Blocking (train sample) =====
    cands_cache = os.path.join(CACHE_DIR, "train_candidates.parquet")
    if os.path.exists(cands_cache):
        print(f"[{ts()}] [BLOCK-TRAIN] Loading from cache…")
        train_cands = pd.read_parquet(cands_cache)
    else:
        print(f"[{ts()}] [BLOCK-TRAIN] Generating candidates…")
        t0 = time.time()
        train_cands = generate_candidates(
            s1_sample, s2_tr, s3_tr,
            extra_s1=s1_te, extra_s2=s2_te, extra_s3=s3_te,
            top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR,
        )
        print(f"  {time.time()-t0:.0f}s  cands={len(train_cands):,}  avg/S1={len(train_cands)/len(s1_sample):.1f}")
        train_cands.to_parquet(cands_cache, index=False)

    train_cands, gt_set = label_candidates(train_cands, gt_sample)
    n_pos = train_cands["label"].sum()
    print(f"  positives={n_pos:,}  negatives={(len(train_cands)-n_pos):,}")

    # Blocking recall on sample
    blk_rec = blocking_recall(train_cands, gt_sample)
    print(f"  Blocking recall (train sample): {blk_rec:.4f}")

    # ===== 5. Feature engineering (train) =====
    feats_cache = os.path.join(CACHE_DIR, "train_features.parquet")
    if os.path.exists(feats_cache):
        print(f"[{ts()}] [FEAT-TRAIN] Loading from cache…")
        train_feats = pd.read_parquet(feats_cache)
    else:
        print(f"[{ts()}] [FEAT-TRAIN] Computing features…")
        t0 = time.time()
        train_feats = compute_features_chunked(train_cands, s1_sample, s23_tr)
        print(f"  {time.time()-t0:.0f}s  shape={train_feats.shape}")
        train_feats.to_parquet(feats_cache, index=False)

    # ===== 6. Model training =====
    print(f"\n[{ts()}] [MODEL] Training LightGBM ({N_FOLDS}-fold)…")
    t0 = time.time()
    labels = train_cands["label"].values
    groups = train_cands["source1_entity_id"].values
    model, oof_probas, threshold = train_model(
        train_feats, labels, groups,
        n_splits=N_FOLDS, neg_sample_ratio=NEG_SAMPLE_RATIO,
    )
    print(f"  {time.time()-t0:.0f}s  threshold={threshold:.3f}")
    save_model(model, threshold, MODEL_PATH)

    # ===== 7. Validate on held-out set =====
    print(f"\n[{ts()}] [VALIDATE] Evaluating on val split ({len(s1_val):,} S1)…")
    t0 = time.time()
    val_cands = generate_candidates(s1_val, s2_tr, s3_tr,
                                    extra_s1=s1_te, extra_s2=s2_te, extra_s3=s3_te,
                                    top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR)
    val_cands, _ = label_candidates(val_cands, gt_val)
    val_feats    = compute_features_chunked(val_cands, s1_val, s23_tr)
    val_probs    = predict_proba(model, val_feats)
    val_pred     = make_predictions(val_cands, val_probs, s1_val["entity_id"].values,
                                    threshold=threshold, t_singleton=T_SINGLETON)
    val_f05      = macro_f_beta(val_pred, gt_val)
    print(f"  {time.time()-t0:.0f}s  Val macro F0.5 = {val_f05:.4f}")

    # ===== 8. Blocking (test) =====
    test_cands_cache = os.path.join(CACHE_DIR, "test_candidates.parquet")
    if os.path.exists(test_cands_cache):
        print(f"[{ts()}] [BLOCK-TEST] Loading from cache…")
        test_cands = pd.read_parquet(test_cands_cache)
    else:
        print(f"[{ts()}] [BLOCK-TEST] Generating candidates…")
        t0 = time.time()
        test_cands = generate_candidates(
            s1_te, s2_te, s3_te,
            extra_s1=s1_tr, extra_s2=s2_tr, extra_s3=s3_tr,
            top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR,
        )
        print(f"  {time.time()-t0:.0f}s  cands={len(test_cands):,}  avg/S1={len(test_cands)/len(s1_te):.1f}")
        test_cands.to_parquet(test_cands_cache, index=False)

    # Free train data from memory
    del s2_tr, s3_tr, s23_tr, train_cands, train_feats
    gc.collect()

    # ===== 9. Feature engineering (test) =====
    print(f"\n[{ts()}] [FEAT-TEST] Computing features ({len(test_cands):,} pairs)…")
    t0 = time.time()
    test_feats = compute_features_chunked(test_cands, s1_te, s23_te)
    print(f"  {time.time()-t0:.0f}s")

    # ===== 10. Inference =====
    print(f"[{ts()}] [INFER] Predicting…")
    test_probs = predict_proba(model, test_feats)

    # ===== 11. Decision layer =====
    print(f"[{ts()}] [DECIDE] Applying decision layer…")
    matching = make_predictions(
        test_cands, test_probs, s1_te["entity_id"].values,
        threshold=threshold, t_singleton=T_SINGLETON,
    )

    # ===== 12. Write outputs =====
    matching_path  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    matching.to_csv(matching_path, sep="\t", index=False)
    print(f"[{ts()}] Wrote {matching_path}")

    cand_grouped = (
        test_cands.groupby("source1_entity_id")["candidate_id"]
                  .apply(lambda x: ",".join(x.astype(str)))
                  .reset_index()
                  .rename(columns={"candidate_id": "candidate_entity_ids"})
    )
    all_s1_df = pd.DataFrame({"source1_entity_id": s1_te["entity_id"]})
    cand_out  = all_s1_df.merge(cand_grouped, on="source1_entity_id", how="left")
    cand_out["candidate_entity_ids"] = cand_out["candidate_entity_ids"].fillna("")
    cand_out.to_csv(candidate_path, sep="\t", index=False)
    print(f"[{ts()}] Wrote {candidate_path}")

    # ===== Summary =====
    n_matched = (matching["matched_entity_ids"].str.strip() != "").sum()
    n_empty   = (matching["matched_entity_ids"].str.strip() == "").sum()
    total_min = (time.time() - t_total) / 60
    print(f"\n=== SUMMARY ===")
    print(f"  Test S1:           {len(matching):,}")
    print(f"  Predicted matched: {n_matched:,}  ({n_matched/len(matching)*100:.1f}%)")
    print(f"  Singletons pred:   {n_empty:,}  ({n_empty/len(matching)*100:.1f}%)")
    print(f"  Val macro F0.5:    {val_f05:.4f}")
    print(f"  Total runtime:     {total_min:.0f} min")

    # Validate format
    print(f"\n[{ts()}] Validating output format…")
    ret = os.system(
        "cd student_resource && python3 utils/validate_submission.py "
        f"--matching ../{matching_path} "
        f"--candidate ../{candidate_path} "
        "--test-dir dataset/test"
    )
    return ret


if __name__ == "__main__":
    sys.exit(main())
