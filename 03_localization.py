"""
03_localization.py — Fingerprint-based positioning evaluation.

Loads (or generates) the CSI fingerprint dataset and evaluates up to four
localization methods, each independently enabled/disabled via the scene's
``features_config.json``:

  1. wKNN            — Weighted k-Nearest Neighbours (IDW)
  2. NN Regression   — 4-layer DNN → (x, y) coordinates
  3. NN Classification — DNN → grid cell → centroid
  4. CNN Regression  — 2-D CNN treating features as [feat_per_tx × n_tx] image

Which fingerprint feature groups to include in the CSI vector is also
controlled by ``features_config.json`` (section ``fingerprint_features``).

Requires: running 01_generate_dataset first (for Sionna scene files),
or having a valid fingerprint_rt_dataset.h5 cache in <SCENE_DIR>/.

Usage
-----
    python 03_localization.py [scene_name] [results_dir]

    scene_name   — name of the scene folder (default: "Otaniemi_small")
    results_dir  — optional override for the output directory

Change the scene_name argument in main() to switch between scenes.
"""

import os
import sys
import json
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

def setup_config(scene_name="Otaniemi_small"):
    """Load scene config and resolve the scene output directory.

    Parameters
    ----------
    scene_name : str
        Name of the scene folder (must contain ``scene_config.json``).

    Returns
    -------
    cfg : dict
        Scene configuration dict from :func:`config.get_scene_config`.
    SCENE_DIR : Path
        Absolute path to the scene output directory (created if absent).
    """
    from config import get_scene_config, scene_dir

    cfg = get_scene_config(scene_name)
    SCENE_DIR = scene_dir(cfg)
    print(f"Scene : {cfg['SCENE_XML_FILE_NAME']}")
    print(f"Data  : {SCENE_DIR}")
    return cfg, SCENE_DIR


def setup_pictures_dir(SCENE_DIR):
    """Create and return the picture output directory for this script.

    Parameters
    ----------
    SCENE_DIR : Path
        Root output directory for the scene.

    Returns
    -------
    Path
        ``<SCENE_DIR>/pictures/03_localization/``, created if absent.
    """
    PICTURES_DIR = SCENE_DIR / "pictures" / "03_localization"
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    return PICTURES_DIR


# ── Load or generate fingerprint dataset ─────────────────────────────────────

def load_or_generate_fingerprints(cfg, SCENE_DIR, features_cfg=None):
    """Return the CSI fingerprint matrix and UE positions.

    If ``<SCENE_DIR>/fingerprint_rt_dataset.h5`` already exists and its
    cache tag matches the current feature selection, it is loaded directly.
    Otherwise Sionna ray tracing is run and the result is cached to disk.

    Parameters
    ----------
    cfg : dict
        Scene configuration dict from :func:`config.get_scene_config`.
    SCENE_DIR : Path
        Directory that contains (or will receive) the HDF5 cache.
    features_cfg : dict or None
        Feature enable/disable dict from :func:`config.get_features_config`.
        Passed through to :func:`features.generate_fingerprint_grid` so that
        the correct feature groups are computed and cached.

    Returns
    -------
    csi_fingerprints : (N, D) ndarray
        Fingerprint feature matrix (only enabled feature groups).
    fingerprint_positions : (N, 2) ndarray
        Corresponding [x, y] UE positions in scene coordinates.
    """
    import tensorflow as tf
    from features import load_fingerprint_dataset

    fp_h5 = SCENE_DIR / 'fingerprint_rt_dataset.h5'
    if fp_h5.exists():
        csi_fingerprints, fingerprint_positions = load_fingerprint_dataset(SCENE_DIR)
        print(f"Loaded fingerprint cache: {csi_fingerprints.shape}")
    else:
        print("No fingerprint cache found. Running generator…")
        import sionna
        from sionna.rt import (
            load_scene, PlanarArray, Transmitter, Receiver, PathSolver
        )
        try:
            from sionna.channel import cir_to_ofdm_channel
        except ModuleNotFoundError:
            from sionna.phy.channel import cir_to_ofdm_channel

        SCENE_XML_FILE_NAME = cfg['SCENE_XML_FILE_NAME']
        tx_positions = cfg['tx_positions']
        fft_size = cfg['fft_size']
        subcarrier_spacing = cfg['subcarrier_spacing']
        fc = cfg['fc']
        GRID_X_MIN = cfg['GRID_X_MIN']
        GRID_X_MAX = cfg['GRID_X_MAX']
        GRID_Y_MIN = cfg['GRID_Y_MIN']
        GRID_Y_MAX = cfg['GRID_Y_MAX']
        GRID_SPACING = cfg['GRID_SPACING']
        UE_H = cfg['UE_H']

        scene = load_scene(SCENE_XML_FILE_NAME)
        scene.tx_array = PlanarArray(num_rows=1, num_cols=4,
            vertical_spacing=0.5, horizontal_spacing=0.5, pattern='iso', polarization='V')
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5, pattern='iso', polarization='V')
        for i, pos in enumerate(tx_positions):
            scene.add(Transmitter(name=f'tx{i}', position=pos))

        x_locs = np.arange(GRID_X_MIN, GRID_X_MAX + GRID_SPACING, GRID_SPACING)
        y_locs = np.arange(GRID_Y_MIN, GRID_Y_MAX + GRID_SPACING, GRID_SPACING)
        xx, yy = np.meshgrid(x_locs, y_locs)
        train_positions = np.column_stack([
            xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, UE_H)
        ])

        frequencies = (
            tf.cast(tf.range(fft_size), tf.float32) - fft_size / 2.0
        ) * subcarrier_spacing + fc
        path_solver = PathSolver()

        from features import generate_fingerprint_grid
        csi_fingerprints, fingerprint_positions = generate_fingerprint_grid(
            scene, path_solver, train_positions, cfg,
            SCENE_DIR, cir_to_ofdm_channel, frequencies,
            features_cfg=features_cfg,
        )
        print(f"Generated fingerprints: {csi_fingerprints.shape}")

    return csi_fingerprints, fingerprint_positions


# ── Train / test split + scaling ─────────────────────────────────────────────

def split_and_scale(csi_fingerprints, fingerprint_positions, cfg,
                     eval_cfg=None):
    """Split the fingerprint dataset and apply standard scaling.

    The split strategy is controlled by ``eval_cfg`` (see
    :func:`localization.make_split_indices`) — options are ``"random"``,
    ``"checkerboard"`` (interpolation test on alternating grid cells), or
    ``"block"`` (contiguous extrapolation hold-out).  The scaler is fitted
    on the training set only and applied to both sets (no leakage).

    Parameters
    ----------
    csi_fingerprints : (N, D) ndarray
        Full fingerprint feature matrix.
    fingerprint_positions : (N, 2) ndarray
        Corresponding [x, y] UE positions.
    cfg : dict
        Scene config dict (needed for grid-aware splits).
    eval_cfg : dict or None
        Evaluation section of ``features_config.json``.  Missing keys use
        defaults.  When ``None``, a uniform 70/30 random split is used.

    Returns
    -------
    X_train_scaled, X_test_scaled : (N_train, D) / (N_test, D) ndarray
        Standardised feature matrices.
    pos_train, pos_test : (N_train, 2) / (N_test, 2) ndarray
        Corresponding position arrays.
    scaler : StandardScaler
        Fitted scaler (can be used to transform new observations).
    """
    from sklearn.preprocessing import StandardScaler
    from localization import make_split_indices

    eval_cfg      = eval_cfg or {}
    split_method  = eval_cfg.get("split_method",  "random")
    test_fraction = eval_cfg.get("test_fraction", 0.3)
    random_state  = eval_cfg.get("random_state",  42)

    tr_idx, te_idx = make_split_indices(
        fingerprint_positions, cfg,
        split_method=split_method,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    X_train, X_test   = csi_fingerprints[tr_idx],      csi_fingerprints[te_idx]
    pos_train, pos_test = fingerprint_positions[tr_idx], fingerprint_positions[te_idx]

    print(f"Split: {split_method}  |  Train: {X_train.shape[0]}  |  "
          f"Test: {X_test.shape[0]}  |  Features: {X_train.shape[1]}")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, pos_train, pos_test, scaler


# ── wKNN localization ─────────────────────────────────────────────────────────

def run_wknn_localization(X_train_scaled, X_test_scaled, pos_train, pos_test,
                           group_sizes=None, pca_whiten=False, pca_variance=0.95):
    """Run wKNN localization and print the result summary.

    Cross-validates *(k, IDW power)* jointly via 5-fold CV on the training set,
    then evaluates the best pair on the test set.  When *group_sizes* is given,
    each feature group's columns are pre-scaled by ``1 / sqrt(N_group)`` so a
    high-dimensional group cannot drown out a small one in the Euclidean
    distance.  When *pca_whiten* is True, the feature space is PCA-whitened
    first (Mahalanobis in the retained subspace); group weighting is then
    ignored because columns no longer map to groups.

    Returns
    -------
    dict
        Result dict from :func:`localization.run_wknn`.
    """
    from localization import run_wknn

    wknn_res = run_wknn(
        X_train_scaled, X_test_scaled, pos_train, pos_test,
        group_sizes=group_sizes,
        pca_whiten=pca_whiten, pca_variance=pca_variance,
    )
    pca_tag = (f", PCA={wknn_res['pca_n_components']}"
               if wknn_res.get("pca_n_components") else "")
    print(f"wKNN  (k={wknn_res['best_k']}, p={wknn_res['best_power']:g}{pca_tag})  "
          f"MAE={wknn_res['mae']:.2f} m  RMSE={wknn_res['rmse']:.2f} m  "
          f"Median={wknn_res['median']:.2f} m  "
          f"P90={wknn_res['p90']:.2f} m  P95={wknn_res['p95']:.2f} m")
    return wknn_res


# ── NN Regression ─────────────────────────────────────────────────────────────

def run_nn_regression_localization(X_train_scaled, X_test_scaled,
                                    pos_train, pos_test, PICTURES_DIR):
    """Train and evaluate the NN regression model, save the training-loss plot.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : (N, D) ndarray
        Standardised feature matrices.
    pos_train, pos_test : (N, 2) ndarray
        [x, y] position arrays.
    PICTURES_DIR : Path
        Directory where ``nn_regression_training.png`` is saved.

    Returns
    -------
    dict
        Result dict with keys: ``errors``, ``pos_pred``, ``history``,
        ``model``, ``rmse``, ``mae``, ``median``, ``p90``, ``p95``.
    """
    from localization import run_nn_regression

    reg_res = run_nn_regression(X_train_scaled, X_test_scaled, pos_train, pos_test)
    print(f"NN Reg   MAE={reg_res['mae']:.2f} m  RMSE={reg_res['rmse']:.2f} m  "
          f"Median={reg_res['median']:.2f} m  P90={reg_res['p90']:.2f} m  P95={reg_res['p95']:.2f} m")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reg_res['history'].history['loss'],     label='Train (Huber)')
    ax.plot(reg_res['history'].history['val_loss'], label='Val (Huber)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('NN Regression — Training')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'nn_regression_training.png', dpi=150)
    # plt.show()
    return reg_res


# ── NN Classification ─────────────────────────────────────────────────────────

def run_nn_classification_localization(X_train_scaled, X_test_scaled,
                                        pos_train, pos_test, cfg):
    """Train and evaluate the NN classification model.

    Discretises the UE grid into cells, trains a softmax classifier, and maps
    predicted class indices back to cell centroid coordinates.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : (N, D) ndarray
        Standardised feature matrices.
    pos_train, pos_test : (N, 2) ndarray
        [x, y] position arrays.
    cfg : dict
        Scene configuration dict — requires ``GRID_X_MIN/MAX``,
        ``GRID_Y_MIN/MAX``, and ``GRID_SPACING``.

    Returns
    -------
    dict
        Result dict with keys: ``errors``, ``pos_pred``, ``history``,
        ``model``, ``accuracy``, ``rmse``, ``mae``, ``median``, ``p90``,
        ``p95``.
    """
    from localization import run_nn_classification

    clf_res = run_nn_classification(X_train_scaled, X_test_scaled,
                                     pos_train, pos_test, cfg)
    print(f"NN Clf   MAE={clf_res['mae']:.2f} m  RMSE={clf_res['rmse']:.2f} m  "
          f"Accuracy={clf_res['accuracy']:.1%}")
    return clf_res


# ── CNN Regression ────────────────────────────────────────────────────────────

def run_cnn_regression_localization(X_train_scaled, X_test_scaled,
                                     pos_train, pos_test, n_tx, PICTURES_DIR):
    """Train and evaluate the CNN regression model, save the training-loss plot.

    The flat feature vector is reshaped to a ``[feat_per_tx × n_tx]`` 2-D
    image so that the CNN can exploit cross-TX spatial correlations.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : (N, D) ndarray
        Standardised feature matrices.
    pos_train, pos_test : (N, 2) ndarray
        [x, y] position arrays.
    n_tx : int
        Number of transmitter nodes (image width).
    PICTURES_DIR : Path
        Directory where ``cnn_regression_training.png`` is saved.

    Returns
    -------
    dict
        Result dict with keys: ``errors``, ``pos_pred``, ``history``,
        ``model``, ``n_feat_per_tx``, ``rmse``, ``mae``, ``median``,
        ``p90``, ``p95``.
    """
    from localization import run_cnn_regression

    cnn_res = run_cnn_regression(X_train_scaled, X_test_scaled,
                                  pos_train, pos_test, n_tx=n_tx)
    print(f"CNN Reg  MAE={cnn_res['mae']:.2f} m  RMSE={cnn_res['rmse']:.2f} m  "
          f"P90={cnn_res['p90']:.2f} m  P95={cnn_res['p95']:.2f} m  "
          f"[{cnn_res['n_feat_per_tx']} feat/TX × {n_tx} TX]")

    # Training curve
    hist = cnn_res['history'].history
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist['loss'],     label='Train loss')
    ax.plot(hist['val_loss'], label='Val loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Huber loss')
    ax.set_title('CNN Regression — Training Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'cnn_regression_training.png', dpi=150)
    plt.close()

    return cnn_res


# ── Summary + CDF plot ────────────────────────────────────────────────────────

def print_summary_and_plot_cdf(results, PICTURES_DIR):
    """Print a localization summary table and save the error CDF plot.

    Parameters
    ----------
    results : list of (str, dict)
        Sequence of ``(method_name, result_dict)`` pairs for each enabled and
        evaluated localization method.  Result dicts must contain at least
        ``mae``, ``rmse``, ``median``, ``p90``, ``p95``, and ``errors``.
    PICTURES_DIR : Path
        Directory where ``localization_error_cdf.png`` is saved.
    """
    methods    = [r[0] for r in results]
    res_dicts  = [r[1] for r in results]
    errors_all = [r['errors'] for r in res_dicts]

    print("\n" + "=" * 70)
    print("LOCALIZATION SUMMARY")
    print("=" * 70)
    print(f"{'Method':<22} {'MAE':>6}  {'RMSE':>6}  {'Median':>7}  {'P90':>6}  {'P95':>6}")
    print("-" * 63)
    for nm, r in zip(methods, res_dicts):
        print(f"  {nm:<20} {r['mae']:6.2f}  {r['rmse']:6.2f}  {r['median']:7.2f}  {r['p90']:6.2f}  {r['p95']:6.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (nm, errs) in enumerate(zip(methods, errors_all)):
        se  = np.sort(errs)
        cdf = np.arange(1, len(se) + 1) / len(se)
        ax.plot(se, cdf * 100, lw=2, label=nm, color=colors[i % len(colors)])
        for pct_val, pct in [(np.percentile(errs, 50), 50),
                              (np.percentile(errs, 90), 90),
                              (np.percentile(errs, 95), 95)]:
            ax.plot(pct_val, pct, 'o', color=colors[i % len(colors)], ms=5, zorder=5)
    for pct, ls in [(50, '--'), (90, ':'), (95, '-.')]:
        ax.axhline(pct, color='grey', lw=0.8, linestyle=ls, alpha=0.6,
                   label=f'P{pct} reference')
    ax.set_xlabel('Localisation Error (m)'); ax.set_ylabel('CDF (%)')
    ax.set_title('Fingerprint Localisation Error CDF')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'localization_error_cdf.png', dpi=150)
    # plt.show()


# ── Metrics comparison bar charts ────────────────────────────────────────────

def plot_metrics_comparison(results, PICTURES_DIR):
    """Save a side-by-side bar chart comparing RMSE/MAE and percentile errors.

    Parameters
    ----------
    results : list of (str, dict)
        Same format as for :func:`print_summary_and_plot_cdf`.
    PICTURES_DIR : Path
        Directory where ``localization_metrics_comparison.png`` is saved.
    """
    methods       = [r[0] for r in results]
    rmse_values   = [r[1]['rmse']   for r in results]
    mae_values    = [r[1]['mae']    for r in results]
    median_values = [r[1]['median'] for r in results]
    p90_values    = [r[1]['p90']    for r in results]
    p95_values    = [r[1]['p95']    for r in results]

    x_pos = np.arange(len(methods))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x_pos - width, rmse_values, width, label='RMSE', alpha=0.8)
    ax1.bar(x_pos,         mae_values,  width, label='MAE',  alpha=0.8)
    ax1.set_ylabel('Error (m)', fontsize=11)
    ax1.set_title('RMSE vs MAE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x_pos - width, median_values, width, label='Median',    alpha=0.8)
    ax2.bar(x_pos,         p90_values,    width, label='90th %ile', alpha=0.8)
    ax2.bar(x_pos + width, p95_values,    width, label='95th %ile', alpha=0.8)
    ax2.set_ylabel('Error (m)', fontsize=11)
    ax2.set_title('Median / P90 / P95 Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'localization_metrics_comparison.png', dpi=150)
    # plt.show()


# ── Spatial error heatmaps ────────────────────────────────────────────────────

def plot_error_heatmaps(pos_test, results, cfg, PICTURES_DIR):
    """Save per-method spatial error heatmaps overlaid on the scene grid.

    Parameters
    ----------
    pos_test : (N_test, 2) ndarray
        True [x, y] positions of the test samples.
    results : list of (str, dict)
        Same format as for :func:`print_summary_and_plot_cdf`.
    cfg : dict
        Scene configuration dict — requires ``GRID_*`` and ``tx_positions``.
    PICTURES_DIR : Path
        Directory where ``localization_error_heatmaps.png`` is saved.
    """
    methods = [r[0] for r in results]
    preds   = [r[1]['pos_pred'] for r in results]
    errors  = [r[1]['errors']   for r in results]

    grid_size = cfg['GRID_SPACING']
    x_bins = np.arange(cfg['GRID_X_MIN'] - grid_size / 2,
                        cfg['GRID_X_MAX'] + grid_size, grid_size)
    y_bins = np.arange(cfg['GRID_Y_MIN'] - grid_size / 2,
                        cfg['GRID_Y_MAX'] + grid_size, grid_size)
    tx_pos_arr = np.array(cfg['tx_positions'])

    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, method, error in zip(axes, methods, errors):
        heatmap = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
        counts  = np.zeros_like(heatmap)

        for pos, err in zip(pos_test, error):
            xi = int(np.clip(np.searchsorted(x_bins, pos[0]) - 1, 0, len(x_bins) - 2))
            yi = int(np.clip(np.searchsorted(y_bins, pos[1]) - 1, 0, len(y_bins) - 2))
            heatmap[yi, xi] += err
            counts[yi, xi]  += 1

        heatmap = np.divide(heatmap, counts,
                            where=counts > 0,
                            out=np.full_like(heatmap, np.nan))

        valid = heatmap[~np.isnan(heatmap)]
        vmin = valid.min() if len(valid) else 0.0
        vmax = valid.max() if len(valid) else 1.0

        im = ax.imshow(heatmap, cmap='RdYlGn_r', origin='lower',
                       extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                       vmin=vmin, vmax=vmax)
        ax.scatter(pos_test[:, 0], pos_test[:, 1],
                   c='blue', s=15, alpha=0.5, label='Test positions')
        ax.scatter(tx_pos_arr[:, 0], tx_pos_arr[:, 1],
                   c='red', s=100, marker='*', label='BS locations')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'{method}\nLocalisation Error', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Error (m)', fontsize=10)

    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'localization_error_heatmaps.png', dpi=150)
    # plt.show()


# ── Save results CSV + JSON ───────────────────────────────────────────────────

def save_results(pos_test, results, SCENE_DIR):
    """Save per-sample predictions to CSV and summary metrics to JSON.

    Parameters
    ----------
    pos_test : (N_test, 2) ndarray
        True [x, y] test positions.
    results : list of (str, dict)
        Same format as for :func:`print_summary_and_plot_cdf`.
    SCENE_DIR : Path
        Directory where the output files are written.

    Output files
    ------------
    fingerprint_localization_results.csv
        One row per test sample with true position and per-method predictions
        and errors.
    fingerprint_localization_summary.json
        Per-method MAE / RMSE / Median / P90 / P95 summary.
    """
    data = {'Position_X': pos_test[:, 0], 'Position_Y': pos_test[:, 1]}
    summary = {}

    for method, res in results:
        col_prefix = method.replace(' ', '_').replace('(', '').replace(')', '')
        data[f'{col_prefix}_Pred_X'] = res['pos_pred'][:, 0]
        data[f'{col_prefix}_Pred_Y'] = res['pos_pred'][:, 1]
        data[f'{col_prefix}_Error']  = res['errors']
        summary[method] = {
            'mae':    float(res['mae']),
            'rmse':   float(res['rmse']),
            'median': float(res['median']),
            'p90':    float(res['p90']),
            'p95':    float(res['p95']),
        }

    results_df = pd.DataFrame(data)
    csv_out = SCENE_DIR / 'fingerprint_localization_results.csv'
    results_df.to_csv(csv_out, index=False)

    json_out = SCENE_DIR / 'fingerprint_localization_summary.json'
    with open(json_out, 'w') as jf:
        json.dump(summary, jf, indent=2)

    print(f"Saved → {csv_out}")
    print(f"Saved → {json_out}")


# ── Save Markdown report ──────────────────────────────────────────────────────

def save_report(SCENE_DIR, PICTURES_DIR):
    """Convert the accumulated Markdown report to a .md file.

    Parameters
    ----------
    SCENE_DIR : Path
        Scene output directory.
    PICTURES_DIR : Path
        Directory containing the generated plot images.
    """
    import report_utils
    importlib.reload(report_utils)
    from report_utils import save_notebook_report

    save_notebook_report(
        Path.cwd() / "03_localization.ipynb",
        SCENE_DIR,
        pics_dir=PICTURES_DIR,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(scene_name="Otaniemi_small", results_dir=None):
    """Run the full fingerprint localization pipeline for *scene_name*.

    Steps:
    1. Load scene config and feature config.
    2. Load (or generate) the CSI fingerprint dataset.
    3. Split into train/test sets and apply standard scaling.
    4. Run each enabled localization method.
    5. Print a summary table and save CDF, bar-chart, and heatmap figures.
    6. Save per-sample results (CSV) and summary metrics (JSON).
    7. Save a Markdown report.

    Which methods run and which fingerprint feature groups are included are
    controlled by ``<scene_name>/features_config.json``.

    Parameters
    ----------
    scene_name : str
        Scene folder name (must contain ``scene_config.json`` and
        ``features_config.json``).
    results_dir : str or None
        Override for the output directory.  When ``None``, outputs are
        written to ``<SCENE_DIR>/``.
    """
    from config import get_features_config

    cfg, SCENE_DIR = setup_config(scene_name)
    feat_cfg = get_features_config(scene_name)
    loc_cfg  = feat_cfg["localization"]
    eval_cfg = feat_cfg.get("evaluation", {})
    pca_whiten   = eval_cfg.get("wknn_pca_whiten",   False)
    pca_variance = eval_cfg.get("wknn_pca_variance", 0.95)

    OUTPUT_DIR = Path(results_dir) if results_dir is not None else SCENE_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if results_dir is not None:
        PICTURES_DIR = OUTPUT_DIR / "pictures" / "03_localization"
        PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    else:
        PICTURES_DIR = setup_pictures_dir(SCENE_DIR)

    from report_utils import MarkdownReport
    report = MarkdownReport()
    report.add(f"""\
# 03 — Fingerprint Localization

Loads (or generates) the CSI fingerprint dataset and evaluates the enabled
localisation methods as configured in ``features_config.json``.

**Scene:** `{cfg['SCENE_XML_FILE_NAME']}`
**Data:** `{OUTPUT_DIR}`

**Requires:** `fingerprint_rt_dataset.h5` in the data directory
(generated by `01_generate_dataset.py`).""")

    with report.capture():
        # Cache always lives in SCENE_DIR (it is a scene artifact, not a run artifact).
        # Reports and pictures go to OUTPUT_DIR separately.
        csi_fingerprints, fingerprint_positions = load_or_generate_fingerprints(
            cfg, SCENE_DIR, feat_cfg
        )

    with report.capture():
        X_train_scaled, X_test_scaled, pos_train, pos_test, _ = split_and_scale(
            csi_fingerprints, fingerprint_positions, cfg,
            eval_cfg=feat_cfg.get("evaluation"),
        )

    # ── Run enabled localization methods ──────────────────────────────────────
    results = []   # [(method_name, result_dict), ...]

    # Per-group sizes for inverse-sqrt-dim weighting (only enabled groups).
    # Reused by both the plain wKNN and the coarse-to-fine variant below.
    from features import get_feature_columns
    feat_flags = feat_cfg.get("fingerprint_features", {})
    group_sizes = [n for name, n in get_feature_columns(cfg)
                   if feat_flags.get(name, True)]

    wknn_res = None
    if loc_cfg.get("wknn", True):
        with report.capture():
            wknn_res = run_wknn_localization(
                X_train_scaled, X_test_scaled, pos_train, pos_test,
                group_sizes=group_sizes,
                pca_whiten=pca_whiten, pca_variance=pca_variance,
            )
        results.append(("WKNN (IDW)", wknn_res))
    else:
        print("wKNN disabled — skipping.")

    if loc_cfg.get("wknn_c2f", False):
        from localization import run_wknn_coarse_to_fine
        with report.capture():
            c2f_res = run_wknn_coarse_to_fine(
                X_train_scaled, X_test_scaled, pos_train, pos_test,
                coarse_result=wknn_res,    # reuse coarse pass when available
                group_sizes=group_sizes,
                pca_whiten=pca_whiten, pca_variance=pca_variance,
            )
            print(f"wKNN C2F  (k={c2f_res['best_k']}, "
                  f"p={c2f_res['best_power']:g}, "
                  f"R={c2f_res['best_radius']:g} m, "
                  f"fallback={c2f_res['fallback_count']}/{len(pos_test)})  "
                  f"MAE={c2f_res['mae']:.2f} m  RMSE={c2f_res['rmse']:.2f} m  "
                  f"P90={c2f_res['p90']:.2f} m")
        results.append(("WKNN C2F", c2f_res))
    else:
        print("wKNN C2F disabled — skipping.")

    if loc_cfg.get("nn_regression", True):
        report.add("### NN Regression — Training Curve")
        with report.capture():
            reg_res = run_nn_regression_localization(
                X_train_scaled, X_test_scaled, pos_train, pos_test, PICTURES_DIR
            )
        results.append(("NN Regression", reg_res))
        report.figure(PICTURES_DIR / 'nn_regression_training.png', OUTPUT_DIR)
    else:
        print("NN Regression disabled — skipping.")

    if loc_cfg.get("nn_classification", True):
        with report.capture():
            clf_res = run_nn_classification_localization(
                X_train_scaled, X_test_scaled, pos_train, pos_test, cfg
            )
        results.append(("NN Classification", clf_res))
    else:
        print("NN Classification disabled — skipping.")

    if loc_cfg.get("cnn_regression", True):
        report.add("### CNN Regression — Training Curve")
        with report.capture():
            cnn_res = run_cnn_regression_localization(
                X_train_scaled, X_test_scaled, pos_train, pos_test,
                n_tx=len(cfg['tx_positions']), PICTURES_DIR=PICTURES_DIR
            )
        results.append(("CNN Regression", cnn_res))
        report.figure(PICTURES_DIR / 'cnn_regression_training.png', OUTPUT_DIR)
    else:
        print("CNN Regression disabled — skipping.")

    if not results:
        print("No localization methods are enabled. Edit features_config.json to enable at least one.")
        return

    # ── Ensemble — inverse-MAE-weighted mean of member predictions ────────────
    if loc_cfg.get("ensemble", True) and len(results) >= 2:
        from localization import run_ensemble
        members = [(nm, r["pos_pred"]) for nm, r in results]
        inv_mae_weights = [1.0 / max(r["mae"], 1e-6) for _, r in results]
        with report.capture():
            ens_res = run_ensemble(members, pos_test,
                                    member_weights=inv_mae_weights)
            print(f"Ensemble (members={len(members)})  "
                  f"MAE={ens_res['mae']:.2f} m  RMSE={ens_res['rmse']:.2f} m  "
                  f"Median={ens_res['median']:.2f} m  "
                  f"P90={ens_res['p90']:.2f} m  P95={ens_res['p95']:.2f} m")
        results.append(("Ensemble", ens_res))
    elif loc_cfg.get("ensemble", True) and len(results) < 2:
        print("Ensemble needs at least 2 enabled members — skipping.")

    # ── Summary plots and outputs ─────────────────────────────────────────────
    report.add("### Localisation Error CDF")
    with report.capture():
        print_summary_and_plot_cdf(results, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'localization_error_cdf.png', OUTPUT_DIR)

    report.add("### Metrics Comparison")
    with report.capture():
        plot_metrics_comparison(results, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'localization_metrics_comparison.png', OUTPUT_DIR)

    report.add("### Spatial Error Heatmaps")
    with report.capture():
        plot_error_heatmaps(pos_test, results, cfg, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'localization_error_heatmaps.png', OUTPUT_DIR)

    with report.capture():
        save_results(pos_test, results, OUTPUT_DIR)

    # ── Markdown analysis section ─────────────────────────────────────────────
    table_rows = "\n".join(
        f"| {nm} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['median']:.2f} "
        f"| {r['p90']:.2f} | {r['p95']:.2f} |"
        for nm, r in results
    )

    report.add(f"""\
## Analysis

{len(results)} localisation method(s) were evaluated on a {len(pos_test)}-sample
test set held out from the CSI fingerprint grid.

| Method | MAE (m) | RMSE (m) | Median (m) | P90 (m) | P95 (m) |
|--------|---------|----------|------------|---------|--------|
{table_rows}

**wKNN** is a non-parametric baseline: each test point is estimated as the
IDW-weighted centroid of its *k* nearest neighbours in the fingerprint library.
It is robust with small datasets but does not generalise beyond the grid.

**NN Regression** treats localisation as a direct coordinate regression problem.
With enough training data and a well-conditioned loss (Huber), it can outperform
wKNN in MAE while maintaining lower variance.

**NN Classification** maps each fingerprint to a discrete grid cell.  Accuracy
can be low when the grid is fine or the dataset is small — the network may fail
to separate adjacent-cell fingerprints from limited observations.

**CNN Regression** treats the feature matrix as a 2-D image where each column
is one BS.  Convolutional filters capture cross-BS spatial correlations that
fully-connected layers may miss.

The CDF plot above shows the empirical distribution of per-test-point errors.
A curve shifted leftmost indicates better performance for a given error budget.""")

    report.save(OUTPUT_DIR / '03_localization_report.md')


if __name__ == "__main__":
    _scene = sys.argv[1] if len(sys.argv) > 1 else "Otaniemi_small"
    _results_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_scene, _results_dir)
