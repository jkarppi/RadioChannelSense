"""
localization.py — Fingerprint-based localization methods.

Provides four independent estimators:
  1. wKNN            — Weighted k-Nearest Neighbours with IDW
  2. NN Regression   — 4-layer DNN mapping CSI features → (x, y)
  3. NN Classification — DNN mapping CSI features → grid-cell index → (x, y)
  4. CNN Regression  — 2D-CNN treating features as [feat_per_tx × n_tx] image

All functions are stateless with respect to global variables and receive their
data as explicit arguments.
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def make_split_indices(
    positions: np.ndarray,
    cfg: dict,
    split_method: str = "random",
    test_fraction: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, test_idx)`` per the chosen split strategy.

    Parameters
    ----------
    positions     : ``(N, 2)`` array of UE ``(x, y)`` coordinates.
    cfg           : scene config dict (needs ``GRID_X_MIN``, ``GRID_Y_MIN``,
                    ``GRID_SPACING`` for grid-aware splits).
    split_method  : ``"random"`` | ``"checkerboard"`` | ``"block"``.

                    * ``random``: uniform random split — the legacy behaviour.
                      On a dense grid this is interpolation-biased because
                      every test point has a near-duplicate training neighbour.
                    * ``checkerboard``: alternate cells by
                      ``(x_idx + y_idx) % 2`` — a genuine interpolation test
                      (~50/50 split; ``test_fraction`` is ignored).
                    * ``block``: contiguous top-right hold-out by ``x + y``
                      percentile — extrapolation test (typically much harder).
    test_fraction : fraction held out (``random`` and ``block`` only).
    random_state  : seed for ``random`` split.

    Returns
    -------
    train_idx, test_idx : 1-D integer arrays indexing into *positions*.
    """
    n = positions.shape[0]
    method = (split_method or "random").lower()

    if method == "random":
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(
            np.arange(n), test_size=test_fraction, random_state=random_state
        )
        return tr, te

    if method == "checkerboard":
        grid  = cfg["GRID_SPACING"]
        x_min = cfg["GRID_X_MIN"]
        y_min = cfg["GRID_Y_MIN"]
        x_idx = np.round((positions[:, 0] - x_min) / grid).astype(int)
        y_idx = np.round((positions[:, 1] - y_min) / grid).astype(int)
        is_test = ((x_idx + y_idx) % 2) == 1
        return np.where(~is_test)[0], np.where(is_test)[0]

    if method == "block":
        # Hold out a geometrically contiguous region in the top-right corner.
        diag = positions[:, 0] + positions[:, 1]
        threshold = np.percentile(diag, (1.0 - test_fraction) * 100.0)
        is_test = diag >= threshold
        return np.where(~is_test)[0], np.where(is_test)[0]

    raise ValueError(
        f"Unknown split_method: {split_method!r}. "
        f"Expected 'random', 'checkerboard', or 'block'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# wKNN
# ─────────────────────────────────────────────────────────────────────────────

def wknn_predict(
    distances: np.ndarray,
    indices: np.ndarray,
    train_positions: np.ndarray,
    power: float = 1.0,
) -> np.ndarray:
    """Inverse-distance-weighted position estimate.

    Parameters
    ----------
    distances      : (N_test, k) distances from kneighbors()
    indices        : (N_test, k) neighbour indices into *train_positions*
    train_positions: (N_train, 2) [x, y] training positions
    power          : IDW exponent p in weight = 1 / (d**p + eps).
                     p=1 → linear inverse, p=2 → squared inverse, etc.

    Returns
    -------
    (N_test, 2) predicted positions
    """
    weights = 1.0 / (distances ** power + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights[:, :, np.newaxis] * train_positions[indices]).sum(axis=1)


def _apply_pca_whiten(
    X_train: np.ndarray,
    X_test: np.ndarray,
    variance: float,
    random_state: int,
):
    """Fit PCA on *X_train* and whiten both sets.

    PCA-whitening in the retained subspace is algebraically equivalent to
    Mahalanobis distance in the original space restricted to that subspace:
    each component is divided by its singular value, so every component has
    unit variance and is uncorrelated with the others.  Euclidean distance
    in whitened-PCA space therefore accounts for feature correlations and
    down-weights redundant directions.

    Parameters
    ----------
    X_train, X_test : scaled feature matrices.
    variance        : fraction of total variance to retain (0 < v < 1), or an
                      integer number of components (v > 1).
    random_state    : passed to sklearn for deterministic SVD tie-breaks.

    Returns
    -------
    X_tr_white, X_te_white : whitened feature matrices.
    pca                    : the fitted ``sklearn.decomposition.PCA`` object.
    """
    from sklearn.decomposition import PCA

    n_components = variance if variance > 1 else float(variance)
    pca = PCA(n_components=n_components, whiten=True,
              svd_solver="auto", random_state=random_state)
    X_tr_white = pca.fit_transform(X_train)
    X_te_white = pca.transform(X_test)
    return X_tr_white, X_te_white, pca


def _group_weights(group_sizes: list[int] | None, n_features: int) -> np.ndarray | None:
    """Build a per-column weight vector that equalises each group's contribution.

    Each column of group g is multiplied by ``1 / sqrt(N_g)`` so that, after
    StandardScaler, every group contributes ~1 unit of variance to the squared
    Euclidean distance — regardless of how many columns it has.  This prevents
    high-dimensional groups (e.g. 3555-col OFDM) from drowning out small but
    informative groups (e.g. 9-col RSS).

    Returns ``None`` if *group_sizes* is None.  Sizes are clipped or padded to
    match *n_features* (the post-mask cache width); missing tail columns are
    treated as a single residual group of weight 1.0.
    """
    if group_sizes is None:
        return None
    w = np.ones(n_features, dtype=np.float64)
    col = 0
    for sz in group_sizes:
        if sz <= 0 or col >= n_features:
            continue
        end = min(col + sz, n_features)
        w[col:end] = 1.0 / np.sqrt(sz)
        col = end
    return w


def run_wknn(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    k_candidates: range | list[int] | None = None,
    power_candidates: tuple[float, ...] = (1.0, 2.0, 3.0),
    group_sizes: list[int] | None = None,
    pca_whiten: bool = False,
    pca_variance: float = 0.95,
    metric: str = "euclidean",
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Cross-validate (k, IDW power) jointly and evaluate wKNN on the test set.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : scaled feature matrices
    pos_train, pos_test           : (N, 2) position arrays
    k_candidates                  : candidate k values (default 1–25)
    power_candidates              : candidate IDW exponents (default 1, 2, 3)
    group_sizes                   : list of column counts per feature group, in
                                    the same column order as the feature matrix.
                                    When provided, columns of group g are
                                    pre-scaled by ``1 / sqrt(N_g)`` before the
                                    NN search.  Ignored when *pca_whiten* is
                                    True (PCA mixes columns so group structure
                                    no longer exists).
    pca_whiten                    : apply PCA whitening before the NN search.
                                    Euclidean distance in the whitened subspace
                                    is equivalent to Mahalanobis distance in the
                                    original feature space, restricted to the
                                    retained components.
    pca_variance                  : when *pca_whiten* is True, the fraction of
                                    total variance retained (0 < v ≤ 1), or the
                                    integer number of components (v > 1).
    metric                        : distance metric for NearestNeighbors
    cv_splits                     : number of CV folds
    random_state                  : random seed

    Returns
    -------
    dict with keys: errors, pos_pred, best_k, best_power, cv_errors,
                    pca_n_components, rmse, mae, median, p90, p95
    """
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors

    if k_candidates is None:
        k_candidates = list(range(1, 26))

    pca_n_components: int | None = None
    if pca_whiten:
        X_train_scaled, X_test_scaled, _pca = _apply_pca_whiten(
            X_train_scaled, X_test_scaled, pca_variance, random_state
        )
        pca_n_components = int(X_train_scaled.shape[1])
        # After PCA, per-column indices no longer map to feature groups.
        w = None
    else:
        # Optional per-group dimensionality re-weighting.
        w = _group_weights(group_sizes, X_train_scaled.shape[1])
    if w is not None:
        X_train_scaled = X_train_scaled * w
        X_test_scaled  = X_test_scaled  * w

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Cache (distances, indices) per (fold, k) so we can sweep power for free.
    cv_grid: dict[tuple[int, float], float] = {}
    for k in k_candidates:
        fold_data = []
        for tr_idx, val_idx in kf.split(X_train_scaled):
            nn = NearestNeighbors(n_neighbors=k, metric=metric)
            nn.fit(X_train_scaled[tr_idx])
            d, idx = nn.kneighbors(X_train_scaled[val_idx])
            fold_data.append((d, idx, pos_train[tr_idx], pos_train[val_idx]))
        for power in power_candidates:
            errs = []
            for d, idx, ptr, pval in fold_data:
                pred = wknn_predict(d, idx, ptr, power=power)
                errs.append(np.mean(np.linalg.norm(pred - pval, axis=1)))
            cv_grid[(k, float(power))] = float(np.mean(errs))

    best_k, best_power = min(cv_grid, key=cv_grid.get)

    # Backward-compat: cv_errors[k] = best mean MAE over power sweep for that k.
    cv_errors: dict[int, float] = {}
    for (k, _p), e in cv_grid.items():
        if k not in cv_errors or e < cv_errors[k]:
            cv_errors[k] = e

    nn_model = NearestNeighbors(n_neighbors=best_k, metric=metric)
    nn_model.fit(X_train_scaled)
    distances, indices = nn_model.kneighbors(X_test_scaled)
    pos_pred = wknn_predict(distances, indices, pos_train, power=best_power)
    errors   = np.linalg.norm(pos_pred - pos_test, axis=1)

    return dict(
        errors           = errors,
        pos_pred         = pos_pred,
        best_k           = best_k,
        best_power       = best_power,
        cv_errors        = cv_errors,
        cv_grid          = cv_grid,
        pca_n_components = pca_n_components,
        rmse             = float(np.sqrt(np.mean(errors ** 2))),
        mae              = float(np.mean(errors)),
        median           = float(np.median(errors)),
        p90              = float(np.percentile(errors, 90)),
        p95              = float(np.percentile(errors, 95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Coarse-to-fine wKNN
# ─────────────────────────────────────────────────────────────────────────────

def run_wknn_coarse_to_fine(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    coarse_result: dict | None = None,
    radius_candidates: tuple[float, ...] = (3.0, 5.0, 7.5, 10.0, 15.0),
    k_candidates: range | list[int] | None = None,
    power_candidates: tuple[float, ...] = (1.0, 2.0, 3.0),
    group_sizes: list[int] | None = None,
    pca_whiten: bool = False,
    pca_variance: float = 0.95,
    metric: str = "euclidean",
    cv_splits: int = 5,
    random_state: int = 42,
    min_candidates: int = 3,
) -> dict:
    """Two-stage wKNN: coarse feature-space pass, then physical-radius refinement.

    Stage 1 (coarse): a standard wKNN (or the supplied *coarse_result*) gives a
    rough position for every test point.

    Stage 2 (fine): for each test point, restrict the candidate pool to training
    fingerprints whose **physical** position lies within ``radius`` metres of
    the coarse estimate.  A new k-NN search runs in feature space among that
    restricted pool, and the IDW-weighted estimate becomes the refined position.

    The radius, k, and IDW power are jointly chosen by 5-fold CV using the
    coarse predictions on the training set.  When too few training neighbours
    fall inside the radius (``< min_candidates``), the coarse prediction is
    kept unchanged for that test point.

    Parameters
    ----------
    X_train_scaled, X_test_scaled : scaled feature matrices
    pos_train, pos_test           : (N, 2) position arrays
    coarse_result                 : optional pre-computed result dict from
                                    :func:`run_wknn`.  When ``None``, a fresh
                                    coarse pass is run with the same hyper-
                                    parameter grid.
    radius_candidates             : metres; CV picks the best.
    k_candidates                  : default 1–15 (kept tighter than coarse pass
                                    because the candidate pool is already small).
    power_candidates              : IDW exponents to try.
    group_sizes                   : per-group column counts for inverse-sqrt-dim
                                    weighting (forwarded to the coarse pass and
                                    applied to the fine pass too).
    min_candidates                : if fewer than this many training points are
                                    inside the radius, keep the coarse estimate.

    Returns
    -------
    dict with keys: errors, pos_pred, best_k, best_power, best_radius,
                    coarse_mae, fallback_count, rmse, mae, median, p90, p95
    """
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors

    if k_candidates is None:
        k_candidates = list(range(1, 16))

    # ── Optional PCA whitening (shared with stage 2 too). ────────────────────
    pca_n_components: int | None = None
    if pca_whiten:
        X_train_scaled, X_test_scaled, _pca = _apply_pca_whiten(
            X_train_scaled, X_test_scaled, pca_variance, random_state
        )
        pca_n_components = int(X_train_scaled.shape[1])
        w = None
    else:
        w = _group_weights(group_sizes, X_train_scaled.shape[1])
    if w is not None:
        X_train_scaled = X_train_scaled * w
        X_test_scaled  = X_test_scaled  * w

    # ── Stage 1: coarse predictions (test set + per-fold for CV). ─────────────
    if coarse_result is None:
        # Group weighting / PCA already applied above → don't re-apply inside.
        coarse = run_wknn(
            X_train_scaled, X_test_scaled, pos_train, pos_test,
            k_candidates=k_candidates, power_candidates=power_candidates,
            group_sizes=None, pca_whiten=False, metric=metric,
            cv_splits=cv_splits, random_state=random_state,
        )
    else:
        coarse = coarse_result
    coarse_pos_test = coarse["pos_pred"]
    coarse_mae      = float(coarse["mae"])

    # Coarse predictions for the *training* set, leaked-free, via CV.
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    coarse_pos_train = np.empty_like(pos_train, dtype=np.float64)
    coarse_k = int(coarse["best_k"])
    coarse_p = float(coarse["best_power"])
    for tr_idx, val_idx in kf.split(X_train_scaled):
        nn = NearestNeighbors(n_neighbors=coarse_k, metric=metric)
        nn.fit(X_train_scaled[tr_idx])
        d, idx = nn.kneighbors(X_train_scaled[val_idx])
        coarse_pos_train[val_idx] = wknn_predict(d, idx, pos_train[tr_idx],
                                                  power=coarse_p)

    # ── Stage-2 refinement helper. ────────────────────────────────────────────
    def _refine(query_X, query_coarse_pos, candidate_X, candidate_pos,
                radius, k, power):
        out = np.empty_like(query_coarse_pos, dtype=np.float64)
        for i in range(query_X.shape[0]):
            dist_phys = np.linalg.norm(candidate_pos - query_coarse_pos[i], axis=1)
            mask = dist_phys <= radius
            if mask.sum() < max(min_candidates, k):
                out[i] = query_coarse_pos[i]
                continue
            X_pool = candidate_X[mask]
            P_pool = candidate_pos[mask]
            kk = min(k, X_pool.shape[0])
            d_feat = np.linalg.norm(X_pool - query_X[i], axis=1)
            order  = np.argsort(d_feat)[:kk]
            d_sel  = d_feat[order][np.newaxis, :]
            ix_sel = order[np.newaxis, :]
            out[i] = wknn_predict(d_sel, ix_sel, P_pool, power=power)[0]
        return out

    # ── CV over (radius, k, power) using leak-free coarse-train positions. ────
    cv_grid: dict[tuple[float, int, float], float] = {}
    for radius in radius_candidates:
        for k in k_candidates:
            for power in power_candidates:
                fold_errs = []
                for tr_idx, val_idx in kf.split(X_train_scaled):
                    pred = _refine(
                        X_train_scaled[val_idx], coarse_pos_train[val_idx],
                        X_train_scaled[tr_idx],  pos_train[tr_idx],
                        radius=radius, k=k, power=power,
                    )
                    fold_errs.append(
                        np.mean(np.linalg.norm(pred - pos_train[val_idx], axis=1))
                    )
                cv_grid[(float(radius), k, float(power))] = float(np.mean(fold_errs))

    best_radius, best_k, best_power = min(cv_grid, key=cv_grid.get)

    # ── Final test-set prediction. ────────────────────────────────────────────
    pos_pred = _refine(
        X_test_scaled, coarse_pos_test,
        X_train_scaled, pos_train,
        radius=best_radius, k=best_k, power=best_power,
    )
    errors = np.linalg.norm(pos_pred - pos_test, axis=1)

    # Diagnostic: how often did we fall back to the coarse estimate?
    fallback_count = int(np.sum(np.all(pos_pred == coarse_pos_test, axis=1)))

    return dict(
        errors           = errors,
        pos_pred         = pos_pred,
        best_k           = best_k,
        best_power       = best_power,
        best_radius      = best_radius,
        coarse_mae       = coarse_mae,
        fallback_count   = fallback_count,
        cv_grid          = cv_grid,
        pca_n_components = pca_n_components,
        rmse             = float(np.sqrt(np.mean(errors ** 2))),
        mae              = float(np.mean(errors)),
        median           = float(np.median(errors)),
        p90              = float(np.percentile(errors, 90)),
        p95              = float(np.percentile(errors, 95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(
    member_preds: list[tuple[str, np.ndarray]],
    pos_test: np.ndarray,
    member_weights: list[float] | np.ndarray | None = None,
    trim: float = 0.0,
) -> dict:
    """Combine per-test-point predictions from several methods into one estimate.

    Each member contributes an ``(N_test, 2)`` prediction array.  The ensemble
    output is the weighted mean (per axis) of those arrays.  When ``trim > 0``
    the most extreme fraction of members per axis is discarded before averaging
    (robust mean), which damps the impact of one methods' outlier predictions.

    Parameters
    ----------
    member_preds   : list of ``(method_name, pos_pred)`` pairs.
                     ``pos_pred`` must be ``(N_test, 2)``.
    pos_test       : ``(N_test, 2)`` ground-truth positions for metrics.
    member_weights : optional weights per member (in the same order).
                     Normalised internally.  ``None`` → uniform weights.
                     Ignored when ``trim > 0`` (trimmed mean is unweighted).
    trim           : fraction in ``[0, 0.5)`` of extreme values to trim
                     per axis before averaging.  ``0`` disables trimming.

    Returns
    -------
    dict with keys: errors, pos_pred, members, member_weights,
                    rmse, mae, median, p90, p95
    """
    if not member_preds:
        raise ValueError("Ensemble requires at least one member prediction.")

    names = [nm for nm, _ in member_preds]
    stack = np.stack([p for _, p in member_preds], axis=0)   # (M, N, 2)

    if trim and 0.0 < trim < 0.5:
        from scipy.stats import trim_mean
        pos_pred = trim_mean(stack, proportiontocut=trim, axis=0)
        weights = None
    else:
        if member_weights is None:
            w = np.ones(stack.shape[0], dtype=np.float64)
        else:
            w = np.asarray(member_weights, dtype=np.float64)
            if w.shape[0] != stack.shape[0]:
                raise ValueError(
                    f"member_weights length {w.shape[0]} != members {stack.shape[0]}"
                )
            if np.any(w < 0):
                raise ValueError("Ensemble weights must be non-negative.")
        w = w / w.sum()
        pos_pred = np.einsum("m,mnd->nd", w, stack)
        weights  = w

    errors = np.linalg.norm(pos_pred - pos_test, axis=1)
    return dict(
        errors         = errors,
        pos_pred       = pos_pred,
        members        = names,
        member_weights = weights,
        rmse           = float(np.sqrt(np.mean(errors ** 2))),
        mae            = float(np.mean(errors)),
        median         = float(np.median(errors)),
        p90            = float(np.percentile(errors, 90)),
        p95            = float(np.percentile(errors, 95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# NN Regression
# ─────────────────────────────────────────────────────────────────────────────

def build_regressor(input_dim: int):
    """4-layer DNN (256 → 128 → 64 → 32 → 2) with Huber loss δ=20 m."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(2, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.Huber(delta=20.0),
        metrics=["mae"],
    )
    return model


def run_nn_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    epochs: int = 500,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate the regression DNN.

    Returns
    -------
    dict with keys: errors, pos_pred, history, model, rmse, mae, median, p90
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Normalise target positions → zero-mean / unit-std (per axis).
    # Mirrors the MATLAB DNN/CNN approach; improves convergence on metre-scale targets.
    pos_mean = pos_train.mean(axis=0)
    pos_std  = pos_train.std(axis=0) + 1e-8
    pos_train_norm = (pos_train - pos_mean) / pos_std

    model = build_regressor(X_train.shape[1])
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X_train, pos_train_norm,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    pos_pred_norm = model.predict(X_test, verbose=0)
    pos_pred = pos_pred_norm * pos_std + pos_mean    # denormalise
    errors   = np.linalg.norm(pos_pred - pos_test, axis=1)

    return dict(
        errors   = errors,
        pos_pred = pos_pred,
        history  = history,
        model    = model,
        rmse     = float(np.sqrt(np.mean(errors ** 2))),
        mae      = float(np.mean(errors)),
        median   = float(np.median(errors)),
        p90      = float(np.percentile(errors, 90)),
        p95      = float(np.percentile(errors, 95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# NN Classification
# ─────────────────────────────────────────────────────────────────────────────

def discretize_positions(
    positions: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    """Map continuous (x, y) positions to 1-D grid-cell class indices."""
    x_idx = np.clip(np.digitize(positions[:, 0], x_bins) - 1, 0, len(x_bins) - 2)
    y_idx = np.clip(np.digitize(positions[:, 1], y_bins) - 1, 0, len(y_bins) - 2)
    return x_idx + y_idx * (len(x_bins) - 1)


def class_to_position(
    class_indices: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    grid_size: float,
) -> np.ndarray:
    """Convert grid-cell class indices back to cell-centroid positions."""
    nx = len(x_bins) - 1
    positions = []
    for c in class_indices:
        xi = c % nx
        yi = c // nx
        positions.append([
            x_bins[0] + xi * grid_size + grid_size / 2,
            y_bins[0] + yi * grid_size + grid_size / 2,
        ])
    return np.array(positions)


def build_classifier(input_dim: int, n_classes: int):
    """4-layer DNN (128 → 64 → 32 → n_classes) with sparse-categorical cross-entropy."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_nn_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    cfg: dict,
    epochs: int = 100,
    batch_size: int = 8,
    val_split: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and evaluate the classification DNN.

    Parameters
    ----------
    cfg : scene config dict; requires ``GRID_X_MIN/MAX``, ``GRID_Y_MIN/MAX``,
          ``GRID_SPACING`` keys.

    Returns
    -------
    dict with keys: errors, pos_pred, history, model, accuracy,
                    rmse, mae, median, p90
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    grid_size = cfg["GRID_SPACING"]
    x_bins = np.arange(cfg["GRID_X_MIN"] - grid_size / 2,
                        cfg["GRID_X_MAX"] + grid_size, grid_size)
    y_bins = np.arange(cfg["GRID_Y_MIN"] - grid_size / 2,
                        cfg["GRID_Y_MAX"] + grid_size, grid_size)

    pos_train_class = discretize_positions(pos_train, x_bins, y_bins)
    pos_test_class  = discretize_positions(pos_test,  x_bins, y_bins)
    n_classes = (len(x_bins) - 1) * (len(y_bins) - 1)

    model = build_classifier(X_train.shape[1], n_classes)
    history = model.fit(
        X_train, pos_train_class,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            )
        ],
    )

    class_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    pos_pred   = class_to_position(class_pred, x_bins, y_bins, grid_size)
    errors     = np.linalg.norm(pos_pred - pos_test, axis=1)
    accuracy   = float(np.mean(class_pred == pos_test_class))

    return dict(
        errors   = errors,
        pos_pred = pos_pred,
        history  = history,
        model    = model,
        accuracy = accuracy,
        rmse     = float(np.sqrt(np.mean(errors ** 2))),
        mae      = float(np.mean(errors)),
        median   = float(np.median(errors)),
        p90      = float(np.percentile(errors, 90)),
        p95      = float(np.percentile(errors, 95)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CNN Regression (2-D feature map: features_per_tx × n_tx)
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_regressor(n_feat_per_tx: int, n_tx: int):
    """2-D CNN that treats the fingerprint as a [n_feat_per_tx × n_tx] image.

    Mirrors the MATLAB cnn1d_localization.m architecture:
        3 conv blocks (32→64→128 filters, 3×3, same padding)
        Global average pooling → FC(128) → Dropout(0.3) → FC(64) → FC(2)

    Parameters
    ----------
    n_feat_per_tx : number of features per TX (height of the image)
    n_tx          : number of transmitters (width of the image)

    Returns
    -------
    Compiled Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_feat_per_tx, n_tx, 1))

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Global average pooling → vector
    x = layers.GlobalAveragePooling2D()(x)

    # FC head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(2, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.Huber(delta=20.0),
        metrics=["mae"],
    )
    return model


def run_cnn_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    n_tx: int,
    epochs: int = 500,
    batch_size: int = 64,
    val_split: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Reshape features as a 2-D image per sample and train the CNN regressor.

    The flat feature vector of length D is reshaped as
    ``[n_feat_per_tx, n_tx]`` where ``n_feat_per_tx = D // n_tx``.
    Features assumed to be arranged TX-major (all features for TX0 first,
    then TX1, …), matching the output of ``features.generate_fingerprint_grid``.

    Only the leading ``n_feat_per_tx * n_tx`` features are used; any trailing
    features that don't divide evenly (e.g. reached flags if they are appended
    separately) are silently dropped — pass a pre-sliced ``X_train`` if finer
    control is needed.

    Parameters
    ----------
    X_train, X_test : (N, D) scaled feature matrices
    pos_train, pos_test : (N, 2) position arrays
    n_tx            : number of TX nodes
    epochs, batch_size, val_split, random_state : training hyper-parameters

    Returns
    -------
    dict with keys: errors, pos_pred, history, model, rmse, mae, median, p90
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    D = X_train.shape[1]
    n_feat_per_tx = D // n_tx
    usable = n_feat_per_tx * n_tx

    # Reshape to [N, n_feat_per_tx, n_tx, 1]
    def _reshape(X):
        X_use = X[:, :usable]
        return X_use.reshape(-1, n_feat_per_tx, n_tx, 1).astype(np.float32)

    X_tr_4d = _reshape(X_train)
    X_te_4d = _reshape(X_test)

    # Normalise target positions → zero-mean / unit-std (per axis).
    pos_mean = pos_train.mean(axis=0)
    pos_std  = pos_train.std(axis=0) + 1e-8
    pos_train_norm = (pos_train - pos_mean) / pos_std

    model = build_cnn_regressor(n_feat_per_tx, n_tx)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X_tr_4d, pos_train_norm,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    pos_pred_norm = model.predict(X_te_4d, verbose=0)
    pos_pred = pos_pred_norm * pos_std + pos_mean    # denormalise
    errors   = np.linalg.norm(pos_pred - pos_test, axis=1)

    return dict(
        errors        = errors,
        pos_pred      = pos_pred,
        history       = history,
        model         = model,
        n_feat_per_tx = n_feat_per_tx,
        rmse          = float(np.sqrt(np.mean(errors ** 2))),
        mae           = float(np.mean(errors)),
        median        = float(np.median(errors)),
        p90           = float(np.percentile(errors, 90)),
        p95           = float(np.percentile(errors, 95)),
    )
