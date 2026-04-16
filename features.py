"""
features.py — CSI fingerprint feature extraction from Sionna RT channel matrices.

The main entry point is :func:`generate_fingerprint_grid`, which either loads a
cached HDF5 file or runs the Sionna PathSolver over a 2-D grid and extracts up
to eight feature groups (each independently enable/disable-able via the
``features_cfg`` argument supplied by ``config.get_features_config()``):

  • ofdm_mag_gd    — dB-magnitude + group-delay per sub-carrier per BS
  • tdoa           — TDoA between every BS pair (ns)
  • aoa            — Amplitude-weighted circular mean AoA per BS
                     (sin/cos of azimuth and elevation)
  • rss            — RSS per BS (dB relative, sum of all-path power)
  • path_loss      — Path loss of the dominant (earliest) path per BS (dB)
  • delay          — Absolute delay of the dominant path per BS (ns)
  • cov_eigenvalues— Spatial covariance eigenvalues (top-K per BS, ULA model)
  • reached_flags  — Binary "reached" flag per BS

Domain-aware imputation is always applied for unreached TX-RX pairs so that
disabled-TX entries do not corrupt the enabled features.

The result is cached to ``<out_dir>/fingerprint_rt_dataset.h5``.  The cache
tag encodes the enabled feature set so that changing the feature selection
automatically triggers a cache miss and recomputation.
"""

from __future__ import annotations
import os
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Low-level feature extraction
# ─────────────────────────────────────────────────────────────────────────────

_SC_STEP: int = 16      # keep every 16th sub-carrier → 198 out of 3168
_GD_CLIP: float = 500.0  # clip group-delay to ±500 samples


def extract_fp(H_link: np.ndarray, sc_step: int = _SC_STEP, gd_clip: float = _GD_CLIP) -> np.ndarray:
    """Extract a 1-D real feature vector from a single BS–UE channel matrix.

    Parameters
    ----------
    H_link : ndarray, shape ``[n_tx_ant, n_time, fft_size]``
    sc_step : sub-carrier decimation factor
    gd_clip : group-delay absolute clip value (samples)

    Returns
    -------
    1-D ndarray of length ``fft_size // sc_step + fft_size // sc_step - 1``
    """
    H_avg = np.mean(H_link, axis=(0, 1))[::sc_step]
    mag_feat = 20.0 * np.log10(np.abs(H_avg) + 1e-6)
    pwr_mask = np.abs(H_avg) > 1e-5
    phase = np.unwrap(np.angle(H_avg))
    gd_raw = np.diff(phase)
    gd_feat = np.where(
        pwr_mask[:-1] & pwr_mask[1:],
        np.clip(gd_raw, -gd_clip, gd_clip),
        0.0,
    )
    return np.concatenate([mag_feat, gd_feat])


# ─────────────────────────────────────────────────────────────────────────────
# ULA spatial covariance helper
# ─────────────────────────────────────────────────────────────────────────────

def _ula_steering_matrix(az_rad: np.ndarray, el_rad: np.ndarray,
                         n_ant: int, fc: float) -> np.ndarray:
    """Compute ULA steering vectors for a batch of (az, el) directions.

    Uses a λ/2-spaced ULA along the y-axis, matching the MATLAB
    ``phased.ULA`` convention in otavectorizedv3.m.

    Parameters
    ----------
    az_rad, el_rad : 1-D arrays of length n_paths (radians)
    n_ant          : number of antenna elements (default 16)
    fc             : carrier frequency (Hz)

    Returns
    -------
    A : complex ndarray, shape ``[n_ant, n_paths]``
    """
    c = 299_792_458.0
    lam = c / fc
    d = 0.5 * lam                                # λ/2 spacing
    # Spatial frequency: u = d/λ · sin(el) · sin(az)  (ULA along y-axis)
    u = (d / lam) * np.sin(el_rad) * np.sin(az_rad)
    n = np.arange(n_ant)[:, np.newaxis]          # [n_ant, 1]
    return np.exp(1j * 2 * np.pi * n * u[np.newaxis, :])  # [n_ant, n_paths]


def _cov_eigenvalues(az_rad: np.ndarray, el_rad: np.ndarray,
                     gains: np.ndarray, n_ant: int, fc: float,
                     n_eig: int) -> np.ndarray:
    """Return the top-*n_eig* eigenvalues of the spatial covariance matrix.

    Replicates the MATLAB covariance construction in otavectorizedv3.m:
        A = steering_matrix(az, el)
        B = A * abs(gain)          (amplitude-weighted)
        R = B @ B.H                (Hermitian PSD)
        R /= trace(R)              (normalise)

    Parameters
    ----------
    az_rad, el_rad : ray arrival angles (radians), shape (n_paths,)
    gains          : complex path gains, shape (n_paths,)
    n_ant, fc      : ULA parameters
    n_eig          : number of eigenvalues to return

    Returns
    -------
    eigenvalues : real ndarray of length *n_eig* (descending order)
    """
    if len(az_rad) == 0:
        return np.zeros(n_eig)
    A = _ula_steering_matrix(az_rad, el_rad, n_ant, fc)   # [n_ant, n_paths]
    B = A * np.abs(gains)[np.newaxis, :]                   # amplitude-weighted
    R = B @ B.conj().T                                     # [n_ant, n_ant] Hermitian PSD
    tr = np.real(np.trace(R))
    if tr > 0:
        R /= tr
    ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]    # descending
    return ev[:n_eig]


# ─────────────────────────────────────────────────────────────────────────────
# Grid generation + caching
# ─────────────────────────────────────────────────────────────────────────────

def generate_fingerprint_grid(
    scene,
    path_solver,
    train_positions: np.ndarray,
    cfg: dict,
    out_dir: str | Path,
    cir_to_ofdm_fn,
    frequencies,
    sc_step: int = _SC_STEP,
    gd_clip: float = _GD_CLIP,
    features_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PathSolver over *train_positions* and extract CSI features.

    Results are cached in ``<out_dir>/fingerprint_rt_dataset.h5``.  A cache
    hit is recognised when n_samples, n_tx, fft_size, and the feature tag all
    match.  Changing the enabled feature set invalidates the cache automatically
    because the tag encodes which feature groups are active.

    Parameters
    ----------
    scene           : Sionna scene object (receivers are temporarily swapped)
    path_solver     : Sionna PathSolver callable
    train_positions : (N, 3) array of UE positions in scene coordinates
    cfg             : scene config dict (from :func:`config.get_scene_config`)
    out_dir         : directory where the HDF5 cache is stored
    cir_to_ofdm_fn  : ``sionna.channel.cir_to_ofdm_channel`` function reference
    frequencies     : frequency axis passed to *cir_to_ofdm_fn*
    sc_step         : sub-carrier decimation factor (default: every 16th sub-carrier)
    gd_clip         : group-delay absolute clip value in samples (default: 500)
    features_cfg    : feature enable/disable dict from
                      :func:`config.get_features_config`.  If ``None``, all
                      feature groups are included (backward-compatible default).

    Returns
    -------
    csi_fingerprints : (N, D) float64 feature matrix — only enabled groups
    fingerprint_positions : (N, 2) [x, y] positions
    """
    import inspect
    import tensorflow as tf
    from sionna.rt import Receiver

    n_tx = len(cfg["tx_positions"])
    fft_size = cfg["fft_size"]
    grid_spacing = cfg["GRID_SPACING"]
    n_all = len(train_positions)

    # ── Feature flags ─────────────────────────────────────────────────────────
    _feat_flags = (features_cfg or {}).get("fingerprint_features", {})

    def _enabled(key: str) -> bool:
        """Return True when a feature group is enabled (default: True)."""
        return _feat_flags.get(key, True)

    # Build the cache tag.  When all features are enabled (the default), the
    # tag is identical to the legacy format so existing caches remain valid.
    _all_feat_keys = {
        "ofdm_mag_gd", "tdoa", "aoa", "rss",
        "path_loss", "delay", "cov_eigenvalues", "reached_flags",
    }
    _disabled = {k for k in _all_feat_keys if not _enabled(k)}
    _feat_suffix = (
        "_feat-" + "-".join(sorted(_all_feat_keys - _disabled))
        if _disabled else ""
    )
    cache_tag = (
        f"dB_mag_gd_tdoa_aoa_rss_pl_delay_cov_reached"
        f"_gs{grid_spacing}_ntx{n_tx}_fft{fft_size}{_feat_suffix}"
    )
    fp_h5 = os.path.join(str(out_dir), "fingerprint_rt_dataset.h5")

    # ── Try cache ─────────────────────────────────────────────────────────────
    if os.path.exists(fp_h5):
        try:
            with h5py.File(fp_h5, "r") as _f:
                n_cached   = int(_f.attrs.get("n_samples",  -1))
                ntx_cached = int(_f.attrs.get("n_tx",        0))
                fft_cached = int(_f.attrs.get("fft_size",    0))
                feat_cached = _f.attrs.get("feature_type", "")
                if (n_cached == n_all and ntx_cached == n_tx
                        and fft_cached == fft_size and cache_tag in feat_cached):
                    csi_fingerprints      = _f["csi_fingerprints"][:]
                    fingerprint_positions = _f["fingerprint_positions"][:]
                    print(f"Loaded RT fingerprint cache  ({fp_h5})")
                    print(f"  Feature shape: {csi_fingerprints.shape}")
                    return csi_fingerprints, fingerprint_positions
                else:
                    print("Cache mismatch; recomputing…")
        except Exception as exc:
            print(f"Cache load failed ({exc}); recomputing…")

    # ── Swap scene receivers for fingerprint grid ──────────────────────────────
    saved_rxs = {name: list(np.array(rx.position))
                 for name, rx in scene.receivers.items()}
    for n in list(saved_rxs):
        scene.remove(n)
    for i, pos in enumerate(train_positions):
        scene.add(Receiver(name=f"fp_{i}", position=pos.tolist()))
    print(f"Added {n_all} fingerprint receivers to scene")

    # ── Run PathSolver ─────────────────────────────────────────────────────────
    max_refl = cfg.get("MAX_REFLECTION_DEPTH", 4)
    fp_kw = dict(max_depth=max_refl, los=True, reflection=True,
                 diffraction=True, scattering=False)
    sup = set(inspect.signature(path_solver.__call__).parameters)
    fp_kw = {k: v for k, v in fp_kw.items() if k in sup}
    print("Running PathSolver on fingerprint grid …")
    fp_paths = path_solver(scene=scene, **fp_kw)
    print("PathSolver complete.")

    # ── CIR → OFDM channel matrix (only if ofdm_mag_gd feature is enabled) ───
    a_fp, tau_fp = fp_paths.cir(out_type="numpy")
    a_fp   = a_fp.astype(np.complex64, copy=False)
    tau_fp = tau_fp.astype(np.float32,  copy=False)
    if a_fp.ndim >= 6 and a_fp.shape[3] > 1:
        a_fp = np.mean(a_fp, axis=3, keepdims=True)

    if _enabled("ofdm_mag_gd"):
        def _chunked_cir_to_ofdm(freqs, a, tau, normalize=True, rx_chunk=64):
            n_rx = a.shape[1]
            chunks = []
            for s in range(0, n_rx, rx_chunk):
                e = min(s + rx_chunk, n_rx)
                chunks.append(
                    cir_to_ofdm_fn(freqs, a[:, s:e], tau[:, s:e], normalize=normalize)
                )
            return tf.concat(chunks, axis=1)

        H_fp = np.array(_chunked_cir_to_ofdm(
            frequencies, a_fp[np.newaxis], tau_fp[np.newaxis], normalize=False
        ))
        print(f"H_fp shape: {H_fp.shape}")

        # ── OFDM features: dB-mag + group-delay ───────────────────────────────
        ofdm_feats = np.stack([
            np.concatenate([
                extract_fp(H_fp[0, rx_i, 0, tx_i, :, :, :], sc_step, gd_clip)
                for tx_i in range(n_tx)
            ])
            for rx_i in range(n_all)
        ])
    else:
        print("ofdm_mag_gd disabled — skipping OFDM channel matrix computation.")
        ofdm_feats = np.empty((n_all, 0), dtype=np.float32)

    # ── RSS per TX-RX: total received power across all paths (dB relative) ────
    # Squeeze a_fp to [n_all, n_tx, n_paths] mean amplitude
    _a = a_fp
    if _a.ndim == 6:
        # [n_rx, n_time, n_tx, n_rx_ant, n_paths, n_tx_ant] → avg over antenna dims
        a_abs = np.mean(np.abs(_a[:, 0, :, 0, :, :]), axis=-1)   # [n_all, n_tx, n_paths]
    elif _a.ndim == 5:
        a_abs = np.abs(_a[:, 0, :, 0, :])                         # [n_all, n_tx, n_paths]
    elif _a.ndim == 4:
        a_abs = np.mean(np.abs(_a), axis=-1)                      # [n_all, n_tx, n_paths]
    else:
        a_abs = np.abs(_a).reshape(n_all, n_tx, -1)
    rss_features = 10.0 * np.log10(np.sum(a_abs ** 2, axis=-1) + 1e-20)  # [n_all, n_tx]

    # ── Reached mask: True when at least one valid path exists ────────────────
    # A TX-RX pair is "reached" if the max amplitude across paths is above noise
    reached = a_abs.max(axis=-1) > 1e-15   # [n_all, n_tx] bool

    # ── TDoA features ─────────────────────────────────────────────────────────
    tau_np = np.array(fp_paths.tau)
    if tau_np.ndim == 5:
        tau_np = tau_np[:, 0, :, 0, :]
    elif tau_np.ndim == 4:
        tau_np = tau_np[:, 0, :, :]
    elif tau_np.ndim == 2:
        tau_np = tau_np[:, :, np.newaxis]
    tau_min = tau_np.min(axis=-1)
    tdoa_pairs = list(combinations(range(n_tx), 2))
    tdoa_features = np.stack(
        [(tau_min[:, i] - tau_min[:, j]) / 1e-9 for i, j in tdoa_pairs],
        axis=1,
    )

    # ── TDoA imputation for unreached TX pairs ────────────────────────────────
    # If either TX in a pair has no valid path to this RX, the raw difference
    # uses tau_min=0 (or noise floor), producing a misleading feature.
    # Replace with the mean of the reached RX for that pair.
    for pair_idx, (tx_i, tx_j) in enumerate(tdoa_pairs):
        missing_pair = ~reached[:, tx_i] | ~reached[:, tx_j]
        if missing_pair.any():
            valid_pair = ~missing_pair
            imputed = tdoa_features[valid_pair, pair_idx].mean() if valid_pair.any() else 0.0
            tdoa_features[missing_pair, pair_idx] = imputed

    # ── AoA features ──────────────────────────────────────────────────────────
    def _squeeze_angle(arr):
        if arr.ndim == 5:
            return arr[:, 0, :, 0, :]
        elif arr.ndim == 4:
            return arr[:, 0, :, :]
        elif arr.ndim == 2:
            return arr[:, :, np.newaxis]
        return arr

    phi_r_np   = _squeeze_angle(np.array(fp_paths.phi_r))
    theta_r_np = _squeeze_angle(np.array(fp_paths.theta_r))
    dom_idx = tau_np.argmin(axis=-1)
    rx_idx = np.arange(n_all)[:, np.newaxis]
    tx_idx = np.arange(n_tx)[np.newaxis, :]
    # Amplitude-weighted circular mean AoA across all paths.
    # Avoids dominant-path NLOS bias: strong reflections contribute proportionally.
    weights     = a_abs / (a_abs.sum(axis=-1, keepdims=True) + 1e-20)  # [n_all, n_tx, n_paths]
    mean_sin_az = np.sum(np.sin(phi_r_np)   * weights, axis=-1)        # [n_all, n_tx]
    mean_cos_az = np.sum(np.cos(phi_r_np)   * weights, axis=-1)
    mean_sin_el = np.sum(np.sin(theta_r_np) * weights, axis=-1)
    mean_cos_el = np.sum(np.cos(theta_r_np) * weights, axis=-1)
    aoa_features = np.concatenate([
        mean_sin_az, mean_cos_az,
        mean_sin_el, mean_cos_el,
    ], axis=1)

    # ── Path loss + absolute delay of dominant path ────────────────────────────
    a_dom = a_abs[rx_idx, tx_idx, dom_idx]                        # [n_all, n_tx]
    path_loss_features = -20.0 * np.log10(a_dom + 1e-20)         # [n_all, n_tx] dB
    delay_features = tau_min / 1e-9                               # [n_all, n_tx] ns

    # ── Spatial covariance eigenvalues (ULA, top-K per TX) ────────────────────
    n_ant  = cfg.get("num_ant", 16)
    n_eig  = cfg.get("num_eig_keep", 3)
    fc_hz  = cfg["fc"]

    if _enabled("cov_eigenvalues"):
        # Build complex gains array: a_abs is [n_all, n_tx, n_paths] (real amplitudes)
        # Pair with AoA angles to form the spatial covariance per (rx, tx)
        # phi_r_np / theta_r_np are already [n_all, n_tx, n_paths]
        cov_eig_features = np.zeros((n_all, n_tx * n_eig), dtype=np.float32)
        for tx_i in range(n_tx):
            col_start = tx_i * n_eig
            for rx_i in range(n_all):
                az  = phi_r_np[rx_i, tx_i, :]       # [n_paths]
                el  = theta_r_np[rx_i, tx_i, :]     # [n_paths]
                amp = a_abs[rx_i, tx_i, :]           # [n_paths] real amplitudes
                valid = np.isfinite(az) & np.isfinite(el) & (amp > 0)
                cov_eig_features[rx_i, col_start:col_start + n_eig] = \
                    _cov_eigenvalues(az[valid], el[valid], amp[valid], n_ant, fc_hz, n_eig)
        print(f"Covariance eigenvalue features computed: {cov_eig_features.shape}")
    else:
        cov_eig_features = np.empty((n_all, 0), dtype=np.float32)

    # ── Domain-aware imputation for unreached TX-RX pairs ─────────────────────
    # Mirrors build_fingerprint_db_v3.m: RSS floor, path-loss ceiling, delay
    # mean; AoA sin/cos → 0 (uniform prior); cov-eig already 0 from np.zeros.
    n_missing_total = 0
    for tx_i in range(n_tx):
        missing = ~reached[:, tx_i]           # [n_all] bool
        n_miss = int(missing.sum())
        if n_miss == 0:
            continue
        valid = reached[:, tx_i]

        # RSS: 10 dB below the minimum observed for this TX
        rss_valid = rss_features[valid, tx_i]
        rss_features[missing, tx_i] = (rss_valid.min() - 10.0) if rss_valid.size else -200.0

        # Path loss: 10 dB above the maximum observed for this TX
        pl_valid = path_loss_features[valid, tx_i]
        path_loss_features[missing, tx_i] = (pl_valid.max() + 10.0) if pl_valid.size else 300.0

        # Delay: mean of reached RX for this TX
        dl_valid = delay_features[valid, tx_i]
        delay_features[missing, tx_i] = dl_valid.mean() if dl_valid.size else 0.0

        # AoA sin/cos: 0.0 (no directional information available)
        for offset in [0, n_tx, 2 * n_tx, 3 * n_tx]:   # sin_az, cos_az, sin_el, cos_el
            aoa_features[missing, offset + tx_i] = 0.0

        # cov_eig: already 0.0 from np.zeros initialisation

        n_missing_total += n_miss

    if n_missing_total:
        print(f"Imputed {n_missing_total} unreached (rx, tx) pairs across {n_tx} TXs")

    # ── Binary reached flags ──────────────────────────────────────────────────
    reached_flags = reached.astype(np.float32)   # [n_all, n_tx]

    # ── Concatenate enabled feature groups ────────────────────────────────────
    # Each group is only included when its flag is True (or absent → True).
    _parts: list[np.ndarray] = []
    _desc:  list[str]        = []

    if _enabled("ofdm_mag_gd"):
        _parts.append(ofdm_feats);            _desc.append(f"{ofdm_feats.shape[1]} OFDM")
    if _enabled("tdoa"):
        _parts.append(tdoa_features);         _desc.append(f"{tdoa_features.shape[1]} TDoA")
    if _enabled("aoa"):
        _parts.append(aoa_features);          _desc.append(f"{aoa_features.shape[1]} AoA")
    if _enabled("rss"):
        _parts.append(rss_features);          _desc.append(f"{rss_features.shape[1]} RSS")
    if _enabled("path_loss"):
        _parts.append(path_loss_features);    _desc.append(f"{path_loss_features.shape[1]} path-loss")
    if _enabled("delay"):
        _parts.append(delay_features);        _desc.append(f"{delay_features.shape[1]} delay")
    if _enabled("cov_eigenvalues"):
        _parts.append(cov_eig_features);      _desc.append(f"{cov_eig_features.shape[1]} cov-eig")
    if _enabled("reached_flags"):
        _parts.append(reached_flags);         _desc.append(f"{reached_flags.shape[1]} reached-flags")

    if not _parts:
        raise ValueError(
            "All fingerprint feature groups are disabled in features_config.json. "
            "Enable at least one group before running."
        )

    csi_fingerprints      = np.concatenate(_parts, axis=1)
    fingerprint_positions = train_positions[:, :2].copy()
    print(f"Feature shape: {csi_fingerprints.shape}  ({' + '.join(_desc)})")

    # ── Save cache ─────────────────────────────────────────────────────────────
    with h5py.File(fp_h5, "w") as _f:
        _f.create_dataset("csi_fingerprints",      data=csi_fingerprints,     compression="gzip")
        _f.create_dataset("fingerprint_positions", data=fingerprint_positions, compression="gzip")
        # H_fp is only available when ofdm_mag_gd was enabled.
        if _enabled("ofdm_mag_gd"):
            _f.create_dataset("H_fp_real", data=H_fp.real, compression="gzip")
            _f.create_dataset("H_fp_imag", data=H_fp.imag, compression="gzip")
        _f.attrs["n_tx"]         = n_tx
        _f.attrs["fft_size"]     = fft_size
        _f.attrs["grid_spacing"] = grid_spacing
        _f.attrs["n_samples"]    = n_all
        _f.attrs["feature_type"] = f"{cache_tag}: {' + '.join(_desc)}, grid={grid_spacing}m"
    print(f"Saved fingerprint cache → {fp_h5}")

    # ── Restore scene receivers ────────────────────────────────────────────────
    for i in range(n_all):
        scene.remove(f"fp_{i}")
    for n, pos in saved_rxs.items():
        scene.add(Receiver(name=n, position=pos))
    print("Scene receivers restored.")

    return csi_fingerprints, fingerprint_positions


# ─────────────────────────────────────────────────────────────────────────────
# Cache loading helper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Feature column layout helpers  (used by the ablation test framework)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(cfg: dict, sc_step: int = _SC_STEP) -> list[tuple[str, int]]:
    """Return an ordered list of ``(feature_group_name, n_columns)`` pairs.

    The order matches the concatenation order inside
    :func:`generate_fingerprint_grid` so that slicing the output matrix
    by these widths recovers each group.

    Parameters
    ----------
    cfg     : scene config dict (requires ``tx_positions``, ``fft_size``,
              and optionally ``num_eig_keep``).
    sc_step : sub-carrier decimation factor (must match the value used when
              generating the cache; default: ``_SC_STEP = 16``).

    Returns
    -------
    list of (str, int)
        E.g. ``[("ofdm_mag_gd", 3555), ("tdoa", 36), …]`` for Otaniemi_small.
    """
    from itertools import combinations as _combos
    n_tx   = len(cfg["tx_positions"])
    fft_sz = cfg["fft_size"]
    n_eig  = cfg.get("num_eig_keep", 3)
    n_sc   = fft_sz // sc_step                            # decimated sub-carrier count
    n_tdoa = len(list(_combos(range(n_tx), 2)))           # C(n_tx, 2)
    return [
        ("ofdm_mag_gd",     n_tx * (2 * n_sc - 1)),      # mag + group-delay per TX
        ("tdoa",            n_tdoa),                       # TDoA pairs (ns)
        ("aoa",             4 * n_tx),                    # sin/cos az+el per TX
        ("rss",             n_tx),                        # RSS per TX
        ("path_loss",       n_tx),                        # dominant-path PL per TX
        ("delay",           n_tx),                        # dominant-path delay per TX
        ("cov_eigenvalues", n_tx * n_eig),                # top-K eigenvalues per TX
        ("reached_flags",   n_tx),                        # binary reachability per TX
    ]


def compute_feature_mask(
    cfg: dict,
    features_cfg: dict | None,
    sc_step: int = _SC_STEP,
) -> np.ndarray:
    """Return a boolean column mask that selects the enabled feature groups.

    Intended for the ablation test framework: load the **full** fingerprint
    cache (all features enabled) and then ``X[:, mask]`` to keep only the
    features specified in *features_cfg*.

    Parameters
    ----------
    cfg          : scene config dict.
    features_cfg : feature enable/disable dict from
                   :func:`config.get_features_config`.  If ``None``, all
                   features are selected (mask is all-True).
    sc_step      : sub-carrier decimation factor (must match the cache).

    Returns
    -------
    mask : bool ndarray of shape ``(D_total,)``
        ``True`` for columns belonging to an enabled feature group.

    Examples
    --------
    Exclude TDoA and AoA::

        features_cfg = {"fingerprint_features": {"tdoa": False, "aoa": False}}
        mask = compute_feature_mask(cfg, features_cfg)
        X_reduced = X_full[:, mask]   # drop TDoA + AoA columns
    """
    feat_flags = (features_cfg or {}).get("fingerprint_features", {})
    parts: list[np.ndarray] = []
    for name, n_cols in get_feature_columns(cfg, sc_step):
        enabled = feat_flags.get(name, True)
        parts.append(np.ones(n_cols, dtype=bool) if enabled
                     else np.zeros(n_cols, dtype=bool))
    return np.concatenate(parts)


def load_fingerprint_dataset(scene_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a pre-generated fingerprint dataset from ``<scene_dir>/fingerprint_rt_dataset.h5``.

    Returns
    -------
    csi_fingerprints : (N, D) float64
    fingerprint_positions : (N, 2) [x, y]
    """
    fp_h5 = os.path.join(str(scene_dir), "fingerprint_rt_dataset.h5")
    if not os.path.exists(fp_h5):
        raise FileNotFoundError(f"Fingerprint dataset not found: {fp_h5}")
    with h5py.File(fp_h5, "r") as _f:
        X   = _f["csi_fingerprints"][:]
        pos = _f["fingerprint_positions"][:]
    return X, pos
