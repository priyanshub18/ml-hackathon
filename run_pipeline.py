"""
End-to-end entity resolution pipeline.

Memory strategy:
  - Normalize each file ONE AT A TIME → save to parquet cache → free RAM
  - Blocking loads only the columns it needs (5 cols instead of 8)
  - Features loads all cols from parquet (not re-normalized from scratch)
  - Train data fully freed before test data is loaded

Timeline estimate (Kaggle 16GB RAM, 4 CPU):
  - Normalize + cache 6 files:    ~10 min
  - Block 300K train S1:          ~45 min
  - Feature train (~15M pairs):   ~10 min
  - LightGBM train (5-fold):      ~10 min
  - Block 1.7M test S1:           ~3.5 h
  - Feature test (~85M pairs):    ~45 min
  - Decision + outputs:           ~5 min
  Total:                          ~5.5 hours
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
def _auto_detect_paths():
    kaggle_input = "/kaggle/input"
    if not os.path.isdir(kaggle_input):
        return None, None
    for root, dirs, files in os.walk(kaggle_input):
        if "train_source1.tsv" in files:
            train_dir = root
            test_dir = None
            for r2, d2, f2 in os.walk(kaggle_input):
                if "test_source1.tsv" in f2:
                    test_dir = r2
                    break
            return train_dir, test_dir
    return None, None

_auto_train, _auto_test = _auto_detect_paths()

DATA_TRAIN = os.environ.get("DATA_TRAIN", _auto_train or "student_resource/dataset/train")
DATA_TEST  = os.environ.get("DATA_TEST",  _auto_test  or "student_resource/dataset/test")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pkl")
CACHE_DIR  = os.path.join(OUTPUT_DIR, "cache")

TRAIN_SAMPLE_S1  = 300_000
TOP_K_NAME       = 20
TOP_K_ADDR       = 20
NEG_SAMPLE_RATIO = 10
N_FOLDS          = 5
T_SINGLETON      = 0.15
RANDOM_SEED      = 42
CHUNK_SIZE       = 5_000_000

# Columns needed by each step
BLOCK_COLS = ["entity_id", "country", "norm_name", "norm_addr", "numeric_tokens"]
FEAT_COLS  = ["entity_id", "country", "norm_name", "norm_core",
              "norm_suffix", "norm_addr", "numeric_tokens", "is_non_latin"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_and_cache(raw_path: str, cache_path: str, tag: str) -> None:
    """Normalize one raw TSV file and save to parquet. Frees RAM immediately."""
    if os.path.exists(cache_path):
        print(f"  [{tag}] Already cached: {cache_path}")
        return
    print(f"[{ts()}] [{tag}] Normalizing {os.path.basename(raw_path)}…")
    t0 = time.time()
    df = pd.read_csv(raw_path, sep="\t")
    df = normalize_df(df)
    df = df.drop(columns=[c for c in ["business_name", "business_address"] if c in df.columns])
    df.to_parquet(cache_path, index=False)
    print(f"  {len(df):,} rows  {time.time()-t0:.0f}s  → {cache_path}")
    del df
    gc.collect()


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
    parts = []
    for start in range(0, len(cands), chunk_size):
        chunk = cands.iloc[start:start + chunk_size]
        parts.append(compute_features(chunk, s1_df, s23_df))
        if (start // chunk_size) % 5 == 0:
            print(f"    chunk {start//chunk_size + 1} ({start:,}/{len(cands):,})", flush=True)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    # ===== 1. Normalize & cache all 6 files (one at a time, free RAM each time) =====
    print(f"\n[{ts()}] === PHASE 1: Normalize & Cache ===")
    norm = {
        "s1_tr": os.path.join(CACHE_DIR, "norm_s1_train.parquet"),
        "s2_tr": os.path.join(CACHE_DIR, "norm_s2_train.parquet"),
        "s3_tr": os.path.join(CACHE_DIR, "norm_s3_train.parquet"),
        "s1_te": os.path.join(CACHE_DIR, "norm_s1_test.parquet"),
        "s2_te": os.path.join(CACHE_DIR, "norm_s2_test.parquet"),
        "s3_te": os.path.join(CACHE_DIR, "norm_s3_test.parquet"),
    }
    normalize_and_cache(f"{DATA_TRAIN}/train_source1.tsv", norm["s1_tr"], "S1-TR")
    normalize_and_cache(f"{DATA_TRAIN}/train_source2.tsv", norm["s2_tr"], "S2-TR")
    normalize_and_cache(f"{DATA_TRAIN}/train_source3.tsv", norm["s3_tr"], "S3-TR")
    normalize_and_cache(f"{DATA_TEST}/test_source1.tsv",   norm["s1_te"], "S1-TE")
    normalize_and_cache(f"{DATA_TEST}/test_source2.tsv",   norm["s2_te"], "S2-TE")
    normalize_and_cache(f"{DATA_TEST}/test_source3.tsv",   norm["s3_te"], "S3-TE")
    print(f"[{ts()}] All files normalized and cached.")

    # ===== 2. Sample train S1 =====
    print(f"\n[{ts()}] === PHASE 2: Train ===")
    gt_full   = pd.read_csv(f"{DATA_TRAIN}/train_ground_truth.tsv", sep="\t")
    s1_tr_all = pd.read_parquet(norm["s1_tr"], columns=["entity_id", "country"])

    rng       = np.random.default_rng(RANDOM_SEED)
    n_sample  = min(TRAIN_SAMPLE_S1, len(s1_tr_all))
    sample_ids = rng.choice(s1_tr_all["entity_id"].values, size=n_sample, replace=False)
    sample_set = set(sample_ids)
    gt_sample  = gt_full[gt_full["source1_entity_id"].isin(sample_set)].reset_index(drop=True)

    val_n   = int(n_sample * 0.1)
    val_ids = set(rng.choice(sample_ids, size=val_n, replace=False))
    del s1_tr_all, gt_full
    gc.collect()

    # Load full S1 train (only blocking cols) filtered to sample
    s1_sample = pd.read_parquet(norm["s1_tr"], columns=FEAT_COLS)
    s1_sample = s1_sample[s1_sample["entity_id"].isin(sample_set)].reset_index(drop=True)
    s1_val    = s1_sample[s1_sample["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_val    = gt_sample[gt_sample["source1_entity_id"].isin(val_ids)]
    print(f"  Train sample: {len(s1_sample):,} S1  Val: {len(s1_val):,}  GT rows: {len(gt_sample):,}")

    # ===== 3. Blocking train (load s2/s3 with blocking cols only) =====
    cands_cache     = os.path.join(CACHE_DIR, "train_candidates.parquet")
    val_cands_cache = os.path.join(CACHE_DIR, "val_candidates.parquet")

    if os.path.exists(cands_cache):
        print(f"[{ts()}] [BLOCK-TRAIN] Loading from cache…")
        train_cands = pd.read_parquet(cands_cache)
        val_cands   = pd.read_parquet(val_cands_cache)
    else:
        print(f"[{ts()}] [BLOCK-TRAIN] Loading S2/S3 for blocking (slim cols)…")
        s2_tr_blk = pd.read_parquet(norm["s2_tr"], columns=BLOCK_COLS)
        s3_tr_blk = pd.read_parquet(norm["s3_tr"], columns=BLOCK_COLS)

        print(f"[{ts()}] [BLOCK-TRAIN] Generating candidates…")
        t0 = time.time()
        train_cands = generate_candidates(
            s1_sample[BLOCK_COLS], s2_tr_blk, s3_tr_blk,
            top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR,
        )
        print(f"  {time.time()-t0:.0f}s  cands={len(train_cands):,}  avg/S1={len(train_cands)/len(s1_sample):.1f}")
        train_cands.to_parquet(cands_cache, index=False)

        print(f"[{ts()}] [BLOCK-VAL] Generating candidates…")
        t0 = time.time()
        val_cands = generate_candidates(
            s1_val[BLOCK_COLS], s2_tr_blk, s3_tr_blk,
            top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR,
        )
        print(f"  {time.time()-t0:.0f}s")
        val_cands.to_parquet(val_cands_cache, index=False)

        del s2_tr_blk, s3_tr_blk
        gc.collect()

    train_cands, _ = label_candidates(train_cands, gt_sample)
    n_pos = train_cands["label"].sum()
    print(f"  positives={n_pos:,}  negatives={(len(train_cands)-n_pos):,}")
    print(f"  Blocking recall: {blocking_recall(train_cands, gt_sample):.4f}")

    # ===== 4. Features train (load s2/s3 with full feat cols) =====
    feats_cache     = os.path.join(CACHE_DIR, "train_features.parquet")
    val_feats_cache = os.path.join(CACHE_DIR, "val_features.parquet")

    if os.path.exists(feats_cache):
        print(f"[{ts()}] [FEAT-TRAIN] Loading from cache…")
        train_feats = pd.read_parquet(feats_cache)
        val_feats   = pd.read_parquet(val_feats_cache)
    else:
        print(f"[{ts()}] [FEAT-TRAIN] Loading S2/S3 for features…")
        s2_tr_ft = pd.read_parquet(norm["s2_tr"], columns=FEAT_COLS)
        s3_tr_ft = pd.read_parquet(norm["s3_tr"], columns=FEAT_COLS)
        s23_tr   = pd.concat([s2_tr_ft, s3_tr_ft], ignore_index=True)
        del s2_tr_ft, s3_tr_ft
        gc.collect()

        print(f"[{ts()}] [FEAT-TRAIN] Computing features…")
        t0 = time.time()
        train_feats = compute_features_chunked(train_cands, s1_sample, s23_tr)
        print(f"  {time.time()-t0:.0f}s  shape={train_feats.shape}")
        train_feats.to_parquet(feats_cache, index=False)

        val_cands, _ = label_candidates(val_cands, gt_val)
        val_feats = compute_features_chunked(val_cands, s1_val, s23_tr)
        val_feats.to_parquet(val_feats_cache, index=False)

        del s23_tr
        gc.collect()

    # ===== 5. Model training =====
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

    # ===== 6. Validate =====
    val_cands, _ = label_candidates(pd.read_parquet(val_cands_cache), gt_val)
    val_probs = predict_proba(model, val_feats)
    val_pred  = make_predictions(val_cands, val_probs, s1_val["entity_id"].values,
                                 threshold=threshold, t_singleton=T_SINGLETON)
    val_f05   = macro_f_beta(val_pred, gt_val)
    print(f"  Val macro F0.5 = {val_f05:.4f}")

    # ===== Free ALL train data =====
    del train_cands, train_feats, val_cands, val_feats, val_probs, val_pred
    del s1_sample, s1_val
    gc.collect()
    print(f"[{ts()}] Train data freed.")

    # ===== 7. Blocking test (load with slim cols) =====
    print(f"\n[{ts()}] === PHASE 3: Test ===")
    test_cands_cache = os.path.join(CACHE_DIR, "test_candidates.parquet")

    if os.path.exists(test_cands_cache):
        print(f"[{ts()}] [BLOCK-TEST] Loading from cache…")
        test_cands = pd.read_parquet(test_cands_cache)
    else:
        print(f"[{ts()}] [BLOCK-TEST] Loading S1/S2/S3 for blocking…")
        s1_te_blk = pd.read_parquet(norm["s1_te"], columns=BLOCK_COLS)
        s2_te_blk = pd.read_parquet(norm["s2_te"], columns=BLOCK_COLS)
        s3_te_blk = pd.read_parquet(norm["s3_te"], columns=BLOCK_COLS)

        print(f"[{ts()}] [BLOCK-TEST] Generating candidates…")
        t0 = time.time()
        test_cands = generate_candidates(
            s1_te_blk, s2_te_blk, s3_te_blk,
            top_k_name=TOP_K_NAME, top_k_addr=TOP_K_ADDR,
        )
        print(f"  {time.time()-t0:.0f}s  cands={len(test_cands):,}  avg/S1={len(test_cands)/len(s1_te_blk):.1f}")
        test_cands.to_parquet(test_cands_cache, index=False)
        del s2_te_blk, s3_te_blk
        gc.collect()

    # ===== 8. Features test =====
    print(f"\n[{ts()}] [FEAT-TEST] Loading S1/S2/S3 for features…")
    s1_te_ft = pd.read_parquet(norm["s1_te"], columns=FEAT_COLS)
    s2_te_ft = pd.read_parquet(norm["s2_te"], columns=FEAT_COLS)
    s3_te_ft = pd.read_parquet(norm["s3_te"], columns=FEAT_COLS)
    s23_te   = pd.concat([s2_te_ft, s3_te_ft], ignore_index=True)
    del s2_te_ft, s3_te_ft
    gc.collect()

    print(f"[{ts()}] [FEAT-TEST] Computing features ({len(test_cands):,} pairs)…")
    t0 = time.time()
    test_feats = compute_features_chunked(test_cands, s1_te_ft, s23_te)
    print(f"  {time.time()-t0:.0f}s")
    del s23_te
    gc.collect()

    # ===== 9. Inference + decision =====
    print(f"[{ts()}] [INFER] Predicting…")
    test_probs = predict_proba(model, test_feats)

    print(f"[{ts()}] [DECIDE] Applying decision layer…")
    s1_te_ids = pd.read_parquet(norm["s1_te"], columns=["entity_id"])["entity_id"].values
    matching  = make_predictions(
        test_cands, test_probs, s1_te_ids,
        threshold=threshold, t_singleton=T_SINGLETON,
    )

    # ===== 10. Write outputs =====
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
    all_s1_df = pd.DataFrame({"source1_entity_id": s1_te_ids})
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
