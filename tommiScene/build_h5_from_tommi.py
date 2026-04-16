"""
build_h5_from_tommi.py — Convert Tommi's MATLAB/Capon JSON outputs into the
``fingerprint_rt_dataset.h5`` format expected by ``03_localization.py``.

Inputs (under ``tomminfilet/``):
  * ``small_otaniemi_matlab_locs.json`` — UE side info: ``[x, y, bts_count, los_present]``
  * ``aod_otaniemi_16b16.json``         — Per-BS dominant AoA: ``[az_deg, el_deg, amplitude]``
  * ``timing_otaniemi.json``            — Per-BS timing: ``[mean_delay_spread_ns,
                                          delay_spread_var, strong_path_idx]``

Output (in this directory):
  * ``fingerprint_rt_dataset.h5`` — same shape contract as ``features.py``:
      - ``csi_fingerprints``      : (N, D) float64
      - ``fingerprint_positions`` : (N, 2) float64

Feature layout matches ``features.get_feature_columns`` for the four enabled
groups in ``tommiScene/features_config.json``:

    aoa            (4 * n_tx)  : sin_az | cos_az | sin_el | cos_el
    rss            (n_tx)      : 20*log10(amplitude)
    delay          (n_tx)      : mean_delay_spread (ns)
    reached_flags  (n_tx)      : 1.0 where BS produced a path, 0.0 otherwise

Domain-aware imputation (mirrors ``features.py``) is applied for unreached
TX-RX pairs so disabled BSs do not corrupt the enabled features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TOMMI_DIR = PROJECT_ROOT / "tomminfilet"


def _load_json(name: str) -> np.ndarray:
    p = TOMMI_DIR / name
    with p.open() as fh:
        return np.asarray(json.load(fh))


def main() -> None:
    locs   = _load_json("small_otaniemi_matlab_locs.json")  # (N, 4)
    aoa    = _load_json("aod_otaniemi_16b16.json")           # (N, n_tx, 3)
    timing = _load_json("timing_otaniemi.json")              # (N, n_tx, 3)

    n_ue, n_tx, _ = aoa.shape
    assert timing.shape == (n_ue, n_tx, 3), f"timing shape {timing.shape}"
    assert locs.shape[0] == n_ue, f"locs rows {locs.shape[0]} != ue {n_ue}"

    print(f"Loaded {n_ue} UEs × {n_tx} BSs")

    # ── Reached mask: a (UE, BS) pair is reached when amplitude > 0 ───────────
    amp     = aoa[:, :, 2].astype(np.float64)            # (N, n_tx)
    az_deg  = aoa[:, :, 0].astype(np.float64)
    el_deg  = aoa[:, :, 1].astype(np.float64)
    mds_ns  = timing[:, :, 0].astype(np.float64)         # mean delay spread (ns)

    reached = amp > 0.0                                    # (N, n_tx) bool

    # ── AoA features (sin/cos of az and el, world frame) ──────────────────────
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    sin_az = np.sin(az_rad);  cos_az = np.cos(az_rad)
    sin_el = np.sin(el_rad);  cos_el = np.cos(el_rad)

    # ── RSS (dB from linear amplitude) ────────────────────────────────────────
    rss_db = 20.0 * np.log10(amp + 1e-20)

    # ── Delay (mean delay spread in ns, used as a per-BS delay feature) ───────
    delay_ns = mds_ns.copy()

    # ── Domain-aware imputation for unreached pairs (mirrors features.py) ─────
    n_missing = 0
    for tx_i in range(n_tx):
        miss  = ~reached[:, tx_i]
        valid = reached[:, tx_i]
        n_miss = int(miss.sum())
        if n_miss == 0:
            continue

        # RSS: 10 dB below the minimum observed for this TX
        rss_valid = rss_db[valid, tx_i]
        rss_db[miss, tx_i] = (rss_valid.min() - 10.0) if rss_valid.size else -200.0

        # Delay: mean of reached UEs for this TX
        d_valid = delay_ns[valid, tx_i]
        delay_ns[miss, tx_i] = d_valid.mean() if d_valid.size else 0.0

        # AoA sin/cos: 0 (no directional information available)
        sin_az[miss, tx_i] = 0.0
        cos_az[miss, tx_i] = 0.0
        sin_el[miss, tx_i] = 0.0
        cos_el[miss, tx_i] = 0.0

        n_missing += n_miss

    if n_missing:
        print(f"Imputed {n_missing} unreached (UE, TX) pairs across {n_tx} TXs")

    aoa_feats = np.concatenate(
        [sin_az, cos_az, sin_el, cos_el], axis=1
    )                                                       # (N, 4 * n_tx)
    reached_flags = reached.astype(np.float32)              # (N, n_tx)

    # ── Concatenate in the same order as features.get_feature_columns ─────────
    # (aoa, rss, delay, reached_flags) — the disabled groups are simply omitted.
    csi_fingerprints = np.concatenate(
        [aoa_feats, rss_db, delay_ns, reached_flags], axis=1
    )

    desc = (
        f"{aoa_feats.shape[1]} AoA + {rss_db.shape[1]} RSS + "
        f"{delay_ns.shape[1]} delay + {reached_flags.shape[1]} reached-flags"
    )
    print(f"Feature shape: {csi_fingerprints.shape}  ({desc})")

    fingerprint_positions = locs[:, :2].astype(np.float64)

    # ── Cache tag matching features.py format for the enabled subset ──────────
    grid_spacing = 2.5
    fft_size     = 3168
    enabled_keys = ["aoa", "delay", "reached_flags", "rss"]   # alphabetical
    cache_tag = (
        "dB_mag_gd_tdoa_aoa_rss_pl_delay_cov_reached"
        f"_gs{grid_spacing}_ntx{n_tx}_fft{fft_size}"
        f"_feat-" + "-".join(enabled_keys)
    )

    out = HERE / "fingerprint_rt_dataset.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("csi_fingerprints",      data=csi_fingerprints,      compression="gzip")
        f.create_dataset("fingerprint_positions", data=fingerprint_positions, compression="gzip")
        f.attrs["n_tx"]         = n_tx
        f.attrs["fft_size"]     = fft_size
        f.attrs["grid_spacing"] = grid_spacing
        f.attrs["n_samples"]    = n_ue
        f.attrs["feature_type"] = f"{cache_tag}: {desc}, grid={grid_spacing}m"
    print(f"Saved → {out}")


if __name__ == "__main__":
    sys.exit(main())
