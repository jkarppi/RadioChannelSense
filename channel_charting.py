"""
channel_charting.py — Dimensionality reduction for channel charting.

Provides PCA, t-SNE, Autoencoder, and UMAP runners, each returning the 2-D
embedding together with Trustworthiness (TW), Continuity (CT), and Kruskal
Stress (KS) quality metrics.

Also exposes the lower-level metric helpers so they can be reused by the
comprehensive analysis notebook cell.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_tw_ct(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10,
) -> tuple[float, float]:
    """Compute Trustworthiness (TW) and Continuity (CT).

    Parameters
    ----------
    X_high : high-dimensional input space
    X_low  : 2-D embedding
    k      : neighbourhood size

    Returns
    -------
    (tw, ct) — both in [0, 1]; higher is better
    """
    n = X_high.shape[0]
    k = min(k, n - 1)

    nh = NearestNeighbors(n_neighbors=n - 1).fit(X_high)
    nl = NearestNeighbors(n_neighbors=n - 1).fit(X_low)
    _, ih = nh.kneighbors(X_high)
    _, il = nl.kneighbors(X_low)

    # Vectorised rank matrices
    rh = np.zeros((n, n), dtype=np.int32)
    rl = np.zeros((n, n), dtype=np.int32)
    rows = np.repeat(np.arange(n), n - 1)
    rnks = np.tile(np.arange(1, n), n)
    rh[rows, ih.ravel()] = rnks
    rl[rows, il.ravel()] = rnks

    denom = n * k * (2 * n - 3 * k - 1) / 2 or 1.0

    # Vectorised: for each point i, find neighbours in low but NOT in high
    # (false neighbours → TW penalty) and vice versa (tears → CT penalty).
    ih_k = ih[:, :k]   # [n, k] top-k in high space
    il_k = il[:, :k]   # [n, k] top-k in low  space

    # Build boolean membership matrices [n, n]
    in_high = np.zeros((n, n), dtype=bool)
    in_low  = np.zeros((n, n), dtype=bool)
    rows_idx = np.repeat(np.arange(n), k)
    in_high[rows_idx, ih_k.ravel()] = True
    in_low [rows_idx, il_k.ravel()] = True

    # False neighbours: in low k-NN but not in high k-NN
    fn_mask = in_low & ~in_high          # [n, n]
    # Tears:  in high k-NN but not in low k-NN
    tear_mask = in_high & ~in_low        # [n, n]

    # Penalty = rank in the OTHER space - k  (only for off-set neighbours)
    # rh[i,j] is the rank of j in high space from i
    tw_s = float(np.sum(rh * fn_mask   - k * fn_mask))
    ct_s = float(np.sum(rl * tear_mask - k * tear_mask))

    return 1.0 - 2 * tw_s / denom, 1.0 - 2 * ct_s / denom


def compute_kruskal_stress(X_high: np.ndarray, X_low: np.ndarray) -> float:
    """Kruskal Stress — lower is better (0 = perfect distance preservation)."""
    dh = cdist(X_high, X_high).ravel()
    dl = cdist(X_low,  X_low ).ravel()
    m  = dh > 0
    return float(np.sqrt(np.sum((dh[m] - dl[m]) ** 2) / np.sum(dh[m] ** 2)))


def compute_nnle(Z_low: np.ndarray, pos_true: np.ndarray) -> float:
    """Nearest-Neighbour Localisation Error (NNLE).

    For each point, find its nearest neighbour in the 2-D chart (excluding
    itself) and report the mean physical distance between the query and its
    chart-nearest neighbour.  Lower is better; 0 = chart-neighbours are
    co-located in real space.

    Parameters
    ----------
    Z_low    : (N, 2) 2-D embedding
    pos_true : (N, 2) ground-truth (x, y) positions in metres

    Returns
    -------
    Mean NNLE in metres
    """
    nn = NearestNeighbors(n_neighbors=2).fit(Z_low)  # 2 so we can skip self
    _, idx = nn.kneighbors(Z_low)
    neighbour_idx = idx[:, 1]                        # skip self (col 0)
    errors = np.linalg.norm(pos_true - pos_true[neighbour_idx], axis=1)
    return float(errors.mean())


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────

# Number of PCA components used as pre-processing for t-SNE / UMAP.
# Captures the bulk of variance while drastically reducing dimensionality so
# manifold methods work in a well-conditioned space.
_PCA_PREPROC_COMPONENTS: int = 50


def run_pca(
    X_scaled: np.ndarray,
    k: int = 10,
) -> dict:
    """PCA channel chart.

    Returns dict with keys: Z, X_pca50, explained_var, tw, ct, ks
    ``X_pca50`` is the 50-component projection used as pre-processing input
    for t-SNE and UMAP.
    """
    from sklearn.decomposition import PCA

    pca2 = PCA(n_components=2, random_state=42)
    Z = pca2.fit_transform(X_scaled)

    # 50-component projection for downstream manifold methods
    n_comp50 = min(_PCA_PREPROC_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca50 = PCA(n_components=n_comp50, random_state=42)
    X_pca50 = pca50.fit_transform(X_scaled)

    pca_full = PCA(random_state=42).fit(X_scaled)
    explained_var = np.cumsum(pca_full.explained_variance_ratio_)
    tw, ct = compute_tw_ct(X_scaled, Z, k=k)
    ks     = compute_kruskal_stress(X_scaled, Z)
    return dict(Z=Z, X_pca50=X_pca50, explained_var=explained_var,
                tw=tw, ct=ct, ks=ks)


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(
    X_scaled: np.ndarray,
    X_pca50: np.ndarray | None = None,
    perplexities: list[int] | None = None,
    pos_true: np.ndarray | None = None,
    k: int = 10,
) -> dict:
    """t-SNE channel chart with perplexity sweep.

    PCA pre-processing to ``_PCA_PREPROC_COMPONENTS`` dimensions is applied
    before t-SNE to avoid the curse of dimensionality.  Pass ``X_pca50``
    (output of ``run_pca``) to reuse the already-fitted projection.

    Best perplexity is selected by **Trustworthiness** (local faithfulness)
    rather than Kruskal Stress, since t-SNE optimises local topology.

    Returns dict with keys: Z (best), best_perp, tw, ct, ks, nnle, all_results
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if perplexities is None:
        perplexities = [5, 15, 30, 50]

    n = X_scaled.shape[0]

    # PCA pre-processing: reduces dimensionality so t-SNE works in well-
    # conditioned space while preserving the bulk of the variance.
    if X_pca50 is not None:
        X_input = X_pca50
    else:
        n_comp = min(_PCA_PREPROC_COMPONENTS, X_scaled.shape[1], n - 1)
        X_input = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
    print(f"  t-SNE input: {X_input.shape[1]}-D PCA projection")

    all_results: dict[int, dict] = {}
    for perp in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=min(perp, n // 2 - 1),
            random_state=42,
            max_iter=1000,
            n_jobs=1,
        )
        Z = tsne.fit_transform(X_input)
        tw_p, _ = compute_tw_ct(X_scaled, Z, k=k)
        ks_p    = compute_kruskal_stress(X_scaled, Z)
        all_results[perp] = dict(Z=Z, tw=tw_p, ks=ks_p)
        print(f"  t-SNE perplexity={perp}: TW={tw_p:.4f}  KS={ks_p:.4f}")

    # Select by TW (local faithfulness), not KS
    best_perp = max(all_results, key=lambda p: all_results[p]["tw"])
    Z_best    = all_results[best_perp]["Z"]
    tw, ct    = compute_tw_ct(X_scaled, Z_best, k=k)
    ks        = all_results[best_perp]["ks"]
    nnle      = compute_nnle(Z_best, pos_true) if pos_true is not None else float('nan')

    return dict(Z=Z_best, best_perp=best_perp, tw=tw, ct=ct, ks=ks, nnle=nnle,
                all_results=all_results)


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def run_autoencoder(
    X_scaled: np.ndarray,
    latent_dim: int = 2,
    epochs: int = 300,
    batch_size: int = 32,
    val_split: float = 0.15,
    k: int = 10,
    random_state: int = 42,
    pos_true: np.ndarray | None = None,
) -> dict:
    """Autoencoder channel chart.

    Uses a deeper encoder (input → 512 → 256 → 128 → 64 → 32 → 2) with
    dropout to prevent early collapse to a PCA-like solution on large feature
    vectors.  The symmetric decoder mirrors this architecture.

    Returns dict with keys: Z, encoder, autoencoder, history, tw, ct, ks, nnle
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    input_dim = X_scaled.shape[1]

    # ── Encoder: wide → narrow → 2-D latent ─────────────────────────────────
    enc_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(512, activation="relu")(enc_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear", name="latent")(x)
    encoder = keras.Model(enc_input, z, name="encoder")

    # ── Decoder: symmetric ──────────────────────────────────────────────────
    dec_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(dec_input)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    dec_out = layers.Dense(input_dim, activation="linear")(x)
    decoder = keras.Model(dec_input, dec_out, name="decoder")

    ae_input  = keras.Input(shape=(input_dim,))
    ae_output = decoder(encoder(ae_input))
    autoencoder = keras.Model(ae_input, ae_output, name="autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
    )

    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=30, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
            ),
        ],
    )

    Z      = encoder.predict(X_scaled, verbose=0)
    tw, ct = compute_tw_ct(X_scaled, Z, k=k)
    ks     = compute_kruskal_stress(X_scaled, Z)
    nnle   = compute_nnle(Z, pos_true) if pos_true is not None else float('nan')

    return dict(Z=Z, encoder=encoder, autoencoder=autoencoder,
                history=history, tw=tw, ct=ct, ks=ks, nnle=nnle)


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(
    X_scaled: np.ndarray,
    X_pca50: np.ndarray | None = None,
    n_neighbors_list: list[int] | None = None,
    min_dist: float = 0.1,
    k: int = 10,
    pos_true: np.ndarray | None = None,
) -> dict:
    """UMAP channel chart with n_neighbors sweep.

    PCA pre-processing to ``_PCA_PREPROC_COMPONENTS`` dimensions is applied
    before UMAP.  Pass ``X_pca50`` (output of ``run_pca``) to reuse the
    already-fitted projection.

    Best n_neighbors is selected by **Trustworthiness** (local faithfulness).

    Returns dict with keys: Z (best), best_n, tw, ct, ks, nnle, all_results
    Raises ImportError if umap-learn is not installed.
    """
    import umap  # umap-learn
    from sklearn.decomposition import PCA

    if n_neighbors_list is None:
        n_neighbors_list = [5, 10, 20, 30]

    n = X_scaled.shape[0]

    # PCA pre-processing
    if X_pca50 is not None:
        X_input = X_pca50
    else:
        n_comp = min(_PCA_PREPROC_COMPONENTS, X_scaled.shape[1], n - 1)
        X_input = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
    print(f"  UMAP input: {X_input.shape[1]}-D PCA projection")

    all_results: dict[int, dict] = {}
    for nn in n_neighbors_list:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(nn, n - 1),
            min_dist=min_dist,
            random_state=42,
        )
        Z    = reducer.fit_transform(X_input)
        tw_n, _ = compute_tw_ct(X_scaled, Z, k=k)
        ks_n    = compute_kruskal_stress(X_scaled, Z)
        all_results[nn] = dict(Z=Z, tw=tw_n, ks=ks_n)
        print(f"  UMAP n_neighbors={nn}: TW={tw_n:.4f}  KS={ks_n:.4f}")

    # Select by TW
    best_n = max(all_results, key=lambda nn: all_results[nn]["tw"])
    Z_best = all_results[best_n]["Z"]
    tw, ct = compute_tw_ct(X_scaled, Z_best, k=k)
    ks     = all_results[best_n]["ks"]
    nnle   = compute_nnle(Z_best, pos_true) if pos_true is not None else float('nan')

    return dict(Z=Z_best, best_n=best_n, tw=tw, ct=ct, ks=ks, nnle=nnle,
                all_results=all_results)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_cc_cache(
    path: str,
    results: dict,
    pos_cc: np.ndarray,
    extra_attrs: dict | None = None,
) -> None:
    """Save channel charting embeddings + metrics to an HDF5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        for key, r in results.items():
            g = f.create_group(key)
            g.create_dataset("Z", data=r["Z"].astype(np.float32), compression="gzip")
            for attr in ("tw", "ct", "ks"):
                if attr in r:
                    g.attrs[attr] = r[attr]
            if "nnle" in r:
                g.attrs["nnle"] = r["nnle"]
        f.create_dataset("pos_cc", data=pos_cc.astype(np.float32), compression="gzip")
        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v


def load_cc_cache(path: str) -> dict:
    """Load channel charting embeddings + metrics from an HDF5 file.

    Returns a dict keyed by method name; each value is a dict with
    ``Z``, ``tw``, ``ct``, ``ks``, and optionally ``nnle``.
    Also includes ``"pos_cc"`` at the top level.
    """
    import h5py

    results: dict = {}
    with h5py.File(path, "r") as f:
        for key in f:
            if key == "pos_cc":
                continue
            g = f[key]
            results[key] = dict(Z=g["Z"][:])
            for attr in ("tw", "ct", "ks", "nnle"):
                if attr in g.attrs:
                    results[key][attr] = float(g.attrs[attr])
        if "pos_cc" in f:
            results["pos_cc"] = f["pos_cc"][:]
    return results
