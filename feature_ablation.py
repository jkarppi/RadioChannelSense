#!/usr/bin/env python3
"""
feature_ablation.py — Feature-masked fingerprint localization for ablation tests.

Loads the FULL fingerprint cache from ``<scene_dir>/fingerprint_rt_dataset.h5``
(generated once with all features enabled), applies a boolean column mask to
keep only the feature groups enabled in ``features_config.json``, and then
evaluates all enabled localization methods.

This avoids re-running the expensive Sionna PathSolver for every feature
combination.  The one-time base cache must be created first by running
``03_localization.py`` with all features enabled.

Usage
-----
    python3 feature_ablation.py <scene_name> <result_dir> [combo_name]

    scene_name  — scene folder (features_config.json controls which
                  features and methods are active for this run)
    result_dir  — directory where localization_summary.json is written
    combo_name  — optional human-readable label stored inside the JSON

Outputs
-------
<result_dir>/localization_summary.json
    {
      "combo_name": "...",
      "enabled_features": [...],
      "n_features": 117,
      "n_train": 840, "n_test": 360,
      "methods": {
        "WKNN (IDW)":    {"mae": 2.45, "rmse": 3.12, "median": ..., "p90": ..., "p95": ...},
        "NN Regression": {...},
        "CNN Regression": {...}
      }
    }
"""

import sys
import json
import time
from pathlib import Path

import numpy as np


def main(scene_name: str, result_dir: str, combo_name: str = "") -> dict:
    """Run feature-masked localization and save metrics to *result_dir*.

    Parameters
    ----------
    scene_name  : Name of the scene folder.
    result_dir  : Output directory for the summary JSON.
    combo_name  : Human-readable label for this combination (stored in JSON).

    Returns
    -------
    dict  — the summary dict also saved to JSON.
    """
    from config import get_scene_config, get_features_config
    from features import (
        load_fingerprint_dataset,
        get_feature_columns,
        compute_feature_mask,
        _SC_STEP,
    )

    cfg         = get_scene_config(scene_name)
    features_cfg = get_features_config(scene_name)
    feat_flags  = features_cfg.get("fingerprint_features", {})
    loc_cfg     = features_cfg.get("localization", {})

    # ── Load the FULL base cache ───────────────────────────────────────────────
    scene_path = Path(scene_name)
    print(f"\n{'='*60}")
    print(f"Combo: {combo_name or 'unnamed'}")
    print(f"{'='*60}")

    X_full, positions = load_fingerprint_dataset(scene_path)

    # Warn if the cache is narrower than expected (not the full-feature base).
    all_cols_info  = get_feature_columns(cfg, _SC_STEP)
    expected_total = sum(n for _, n in all_cols_info)
    if X_full.shape[1] != expected_total:
        print(
            f"  WARNING: Cache has {X_full.shape[1]} columns but the full-feature "
            f"base should have {expected_total}.  The mask will be clipped to the "
            f"cache width.  Run with all features enabled first to get a correct base."
        )

    # ── Apply column mask ──────────────────────────────────────────────────────
    mask = compute_feature_mask(cfg, features_cfg, _SC_STEP)
    # Safety clip: never index beyond the actual cache width.
    mask = mask[: X_full.shape[1]]
    X = X_full[:, mask]

    n_features      = int(X.shape[1])
    enabled_feats   = [name for name, _ in all_cols_info
                       if feat_flags.get(name, True)]
    disabled_feats  = [name for name, _ in all_cols_info
                       if not feat_flags.get(name, True)]

    print(f"  Enabled  features : {enabled_feats}")
    if disabled_feats:
        print(f"  Disabled features : {disabled_feats}")
    print(f"  Feature columns   : {n_features}  (base cache: {X_full.shape[1]})")

    # ── Train / test split + scaling ──────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from localization import make_split_indices

    eval_cfg      = features_cfg.get("evaluation", {}) or {}
    split_method  = eval_cfg.get("split_method",  "random")
    test_fraction = eval_cfg.get("test_fraction", 0.3)
    random_state  = eval_cfg.get("random_state",  42)
    pca_whiten    = eval_cfg.get("wknn_pca_whiten",   False)
    pca_variance  = eval_cfg.get("wknn_pca_variance", 0.95)

    tr_idx, te_idx = make_split_indices(
        positions, cfg,
        split_method=split_method,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    X_train, X_test     = X[tr_idx],         X[te_idx]
    pos_train, pos_test = positions[tr_idx], positions[te_idx]

    scaler       = StandardScaler()
    X_train_s    = scaler.fit_transform(X_train)
    X_test_s     = scaler.transform(X_test)
    print(f"  Split: {split_method}  |  Train: {X_train_s.shape[0]}  |  "
          f"Test: {X_test_s.shape[0]}")

    results: dict = {}
    preds:   dict[str, np.ndarray] = {}   # method_name → (N_test, 2) pos_pred

    # Per-group column counts for inverse-sqrt-dim weighting (only enabled groups).
    # Reused by both the plain wKNN and the coarse-to-fine variant below.
    group_sizes = [n for name, n in all_cols_info
                   if feat_flags.get(name, True)]

    # ── wKNN ──────────────────────────────────────────────────────────────────
    wknn_r = None
    if loc_cfg.get("wknn", True):
        from localization import run_wknn
        t0 = time.time()
        wknn_r = run_wknn(X_train_s, X_test_s, pos_train, pos_test,
                          group_sizes=group_sizes,
                          pca_whiten=pca_whiten, pca_variance=pca_variance)
        elapsed = time.time() - t0
        results["WKNN (IDW)"] = _metrics(wknn_r)
        preds["WKNN (IDW)"]   = wknn_r["pos_pred"]
        print(f"  wKNN    (k={wknn_r['best_k']:2d}, p={wknn_r['best_power']:g})  "
              f"MAE={wknn_r['mae']:.2f} m  RMSE={wknn_r['rmse']:.2f} m  ({elapsed:.0f}s)")

    # ── wKNN coarse-to-fine ───────────────────────────────────────────────────
    if loc_cfg.get("wknn_c2f", False):
        from localization import run_wknn_coarse_to_fine
        t0 = time.time()
        c2f_r = run_wknn_coarse_to_fine(
            X_train_s, X_test_s, pos_train, pos_test,
            coarse_result=wknn_r,
            group_sizes=group_sizes,
            pca_whiten=pca_whiten, pca_variance=pca_variance,
        )
        elapsed = time.time() - t0
        results["WKNN C2F"] = _metrics(c2f_r)
        preds["WKNN C2F"]   = c2f_r["pos_pred"]
        print(f"  wKNN C2F (k={c2f_r['best_k']:2d}, p={c2f_r['best_power']:g}, "
              f"R={c2f_r['best_radius']:g} m, fb={c2f_r['fallback_count']})  "
              f"MAE={c2f_r['mae']:.2f} m  RMSE={c2f_r['rmse']:.2f} m  ({elapsed:.0f}s)")

    # ── NN Regression ─────────────────────────────────────────────────────────
    if loc_cfg.get("nn_regression", True):
        from localization import run_nn_regression
        t0 = time.time()
        r = run_nn_regression(X_train_s, X_test_s, pos_train, pos_test)
        elapsed = time.time() - t0
        results["NN Regression"] = _metrics(r)
        preds["NN Regression"]   = r["pos_pred"]
        print(f"  NN Reg          MAE={r['mae']:.2f} m  "
              f"RMSE={r['rmse']:.2f} m  ({elapsed:.0f}s)")

    # ── NN Classification (opt-in — slow for large grids) ────────────────────
    if loc_cfg.get("nn_classification", False):
        from localization import run_nn_classification
        t0 = time.time()
        r = run_nn_classification(X_train_s, X_test_s, pos_train, pos_test, cfg)
        elapsed = time.time() - t0
        results["NN Classification"] = _metrics(r)
        preds["NN Classification"]   = r["pos_pred"]
        print(f"  NN Clf          MAE={r['mae']:.2f} m  "
              f"RMSE={r['rmse']:.2f} m  acc={r['accuracy']:.1%}  ({elapsed:.0f}s)")

    # ── CNN Regression ────────────────────────────────────────────────────────
    if loc_cfg.get("cnn_regression", True):
        from localization import run_cnn_regression
        n_tx = len(cfg["tx_positions"])
        t0 = time.time()
        r = run_cnn_regression(X_train_s, X_test_s, pos_train, pos_test, n_tx=n_tx)
        elapsed = time.time() - t0
        results["CNN Regression"] = _metrics(r)
        preds["CNN Regression"]   = r["pos_pred"]
        print(f"  CNN Reg         MAE={r['mae']:.2f} m  "
              f"RMSE={r['rmse']:.2f} m  ({elapsed:.0f}s)")

    # ── Ensemble — inverse-MAE-weighted mean of available members ────────────
    if loc_cfg.get("ensemble", True) and len(preds) >= 2:
        from localization import run_ensemble
        members  = [(nm, preds[nm]) for nm in preds]
        inv_mae  = [1.0 / max(results[nm]["mae"], 1e-6) for nm in preds]
        t0 = time.time()
        ens_r = run_ensemble(members, pos_test, member_weights=inv_mae)
        elapsed = time.time() - t0
        results["Ensemble"] = _metrics(ens_r)
        print(f"  Ensemble (m={len(members)})  "
              f"MAE={ens_r['mae']:.2f} m  RMSE={ens_r['rmse']:.2f} m  "
              f"({elapsed:.0f}s)")

    if not results:
        print("  No localization methods enabled — nothing to evaluate.")
        return {}

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "combo_name":       combo_name,
        "enabled_features": enabled_feats,
        "disabled_features": disabled_feats,
        "n_features":       n_features,
        "split_method":     split_method,
        "n_train":          int(X_train_s.shape[0]),
        "n_test":           int(X_test_s.shape[0]),
        "methods":          results,
    }
    out_path = out_dir / "localization_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved → {out_path}")
    return summary


def _metrics(r: dict) -> dict:
    """Extract the standard scalar metrics from a localization result dict."""
    return {
        "mae":    float(r["mae"]),
        "rmse":   float(r["rmse"]),
        "median": float(r["median"]),
        "p90":    float(r["p90"]),
        "p95":    float(r["p95"]),
    }


if __name__ == "__main__":
    _scene  = sys.argv[1] if len(sys.argv) > 1 else "Otaniemi_small"
    _outdir = sys.argv[2] if len(sys.argv) > 2 else "ablation_results/unnamed"
    _name   = sys.argv[3] if len(sys.argv) > 3 else ""
    main(_scene, _outdir, _name)
