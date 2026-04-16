"""
rt_utils.py — Mitsuba ray-tracer helpers for CIR / AoA / AoD computation.

All functions are pure (no global state) and accept NumPy arrays so they can
be called from any notebook or script after importing.
"""

from __future__ import annotations
import numpy as np


SPEED_OF_LIGHT: float = 299_792_458.0


# ─────────────────────────────────────────────────────────────────────────────
# Scene loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mitsuba_scene(xml_path: str):
    """Load a Mitsuba scene, changing to the XML directory so relative PLY
    paths resolve correctly.  Returns the mitsuba.Scene object."""
    import mitsuba as mi
    import os
    from pathlib import Path

    xml_path = Path(xml_path)
    xml_dir  = str(xml_path.parent)
    xml_name = str(xml_path.name)
    old_cwd  = os.getcwd()
    try:
        os.chdir(xml_dir)
        scene_mi = mi.load_file(xml_name)
    finally:
        os.chdir(old_cwd)
    return scene_mi


# ─────────────────────────────────────────────────────────────────────────────
# CIR tracer
# ─────────────────────────────────────────────────────────────────────────────

def trace_paths(
    scene_mi,
    origin_np: np.ndarray,
    target_np: np.ndarray,
    max_depth: int = 4,
    n_probe: int = 4096,
    rng_seed: int = 42,
) -> list[tuple[float, float]]:
    """Return (delay_ns, amplitude) tuples for LOS + single-bounce paths.

    Parameters
    ----------
    scene_mi  : mitsuba.Scene
    origin_np : (3,) transmitter position in scene coordinates
    target_np : (3,) receiver position in scene coordinates
    max_depth : maximum number of reflections to consider (0 = LOS only)
    n_probe   : number of random directions for single-bounce probing
    rng_seed  : NumPy random seed for reproducibility
    """
    import mitsuba as mi
    import drjit as dr

    results: list[tuple[float, float]] = []
    origin = mi.Point3f(float(origin_np[0]), float(origin_np[1]), float(origin_np[2]))

    # ── LOS ──────────────────────────────────────────────────────────────────
    d_vec = target_np - origin_np
    dist = float(np.linalg.norm(d_vec))
    d_hat = d_vec / dist
    ray_d = mi.Ray3f(
        origin,
        mi.Vector3f(float(d_hat[0]), float(d_hat[1]), float(d_hat[2])),
    )
    ray_d.maxt = float(dist * 0.9999)
    si_los = scene_mi.ray_intersect(ray_d)
    if dr.none(si_los.is_valid()):
        results.append((dist / SPEED_OF_LIGHT / 1e-9, 1.0 / dist))

    # ── Single-bounce reflections ─────────────────────────────────────────────
    if max_depth >= 1:
        rng = np.random.default_rng(rng_seed)
        dirs = rng.standard_normal((n_probe, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        rays = mi.Ray3f(
            o=mi.Point3f(
                np.tile(origin_np.astype(np.float32), (n_probe, 1)).T.tolist()
            ),
            d=mi.Vector3f(dirs.T.tolist()),
        )
        si = scene_mi.ray_intersect(rays)
        valid = np.array(si.is_valid(), dtype=bool)

        hit_p = np.array(si.p).T[valid, :]
        hit_n = np.array(si.n).T[valid, :]
        dirs_valid = dirs[valid]

        for k in range(hit_p.shape[0]):
            p = hit_p[k]
            n = hit_n[k]
            to_rx = target_np - p
            d_rx = float(np.linalg.norm(to_rx))
            ray_rx = mi.Ray3f(
                mi.Point3f(float(p[0]), float(p[1]), float(p[2])),
                mi.Vector3f(
                    float(to_rx[0] / d_rx),
                    float(to_rx[1] / d_rx),
                    float(to_rx[2] / d_rx),
                ),
            )
            ray_rx.maxt = float(d_rx * 0.9999)
            if dr.any(scene_mi.ray_intersect(ray_rx).is_valid()):
                continue
            d_tx = float(np.linalg.norm(p - origin_np))
            cos_i = abs(float(np.dot(n, -dirs_valid[k])))
            cos_r = abs(float(np.dot(n, to_rx / d_rx)))
            amp = cos_i * cos_r / (d_tx * d_rx)
            results.append(((d_tx + d_rx) / SPEED_OF_LIGHT / 1e-9, amp))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AoA tracer
# ─────────────────────────────────────────────────────────────────────────────

def compute_aoa_mitsuba(
    scene_mi,
    origin_np: np.ndarray,
    target_np: np.ndarray,
    max_depth: int = 4,
    n_probe: int = 4096,
    rng_seed: int = 42,
) -> list[tuple[float, float, float, float]]:
    """Return (delay_ns, amplitude, az_aoa_deg, el_aoa_deg) tuples.

    AoA is the direction of the incoming ray at the receiver.
    """
    import mitsuba as mi
    import drjit as dr

    results: list[tuple[float, float, float, float]] = []
    origin = mi.Point3f(float(origin_np[0]), float(origin_np[1]), float(origin_np[2]))

    # ── LOS ──────────────────────────────────────────────────────────────────
    d_vec = target_np - origin_np
    dist = float(np.linalg.norm(d_vec))
    d_hat = d_vec / dist
    ray_d = mi.Ray3f(
        origin,
        mi.Vector3f(float(d_hat[0]), float(d_hat[1]), float(d_hat[2])),
    )
    ray_d.maxt = float(dist * 0.9999)
    si_los = scene_mi.ray_intersect(ray_d)
    if dr.none(si_los.is_valid()):
        inc = -d_hat
        az = np.degrees(np.arctan2(inc[1], inc[0]))
        el = np.degrees(np.arcsin(np.clip(inc[2], -1.0, 1.0)))
        results.append((dist / SPEED_OF_LIGHT / 1e-9, 1.0 / dist, az, el))

    # ── Single-bounce reflections ─────────────────────────────────────────────
    if max_depth >= 1:
        rng = np.random.default_rng(rng_seed)
        dirs = rng.standard_normal((n_probe, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        rays = mi.Ray3f(
            o=mi.Point3f(
                np.tile(origin_np.astype(np.float32), (n_probe, 1)).T.tolist()
            ),
            d=mi.Vector3f(dirs.T.tolist()),
        )
        si = scene_mi.ray_intersect(rays)
        valid = np.array(si.is_valid(), dtype=bool)

        hit_p = np.array(si.p).T[valid, :]
        hit_n = np.array(si.n).T[valid, :]
        dirs_valid = dirs[valid]

        for k in range(hit_p.shape[0]):
            p = hit_p[k]
            n = hit_n[k]
            to_rx = target_np - p
            dist_rx = float(np.linalg.norm(to_rx))
            to_rx_hat = to_rx / dist_rx

            ray_rx = mi.Ray3f(
                mi.Point3f(float(p[0]), float(p[1]), float(p[2])),
                mi.Vector3f(
                    float(to_rx_hat[0]),
                    float(to_rx_hat[1]),
                    float(to_rx_hat[2]),
                ),
            )
            ray_rx.maxt = float(dist_rx * 0.9999)
            if dr.any(scene_mi.ray_intersect(ray_rx).is_valid()):
                continue

            dist_tx = float(np.linalg.norm(p - origin_np))
            delay = (dist_tx + dist_rx) / SPEED_OF_LIGHT / 1e-9
            cos_i = abs(float(np.dot(n, -dirs_valid[k])))
            cos_r = abs(float(np.dot(n, to_rx_hat)))
            amp = cos_i * cos_r / (dist_tx * dist_rx)

            inc = -to_rx_hat
            az = np.degrees(np.arctan2(inc[1], inc[0]))
            el = np.degrees(np.arcsin(np.clip(inc[2], -1.0, 1.0)))
            results.append((delay, amp, az, el))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AoD helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_aod_mitsuba(
    origin_np: np.ndarray,
    target_np: np.ndarray,
    paths_aoa_list: list[tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (az_aod_deg, el_aod_deg) arrays for each path in paths_aoa_list.

    For reflected paths the AoD is approximated as the reverse of the AoA
    direction at the scatterer — a first-order approximation suitable for
    far-field single-bounce scenarios.
    """
    az_aod: list[float] = []
    el_aod: list[float] = []

    # LOS: direction TX → RX
    d = (target_np - origin_np).astype(float)
    d /= np.linalg.norm(d)
    az_aod.append(float(np.degrees(np.arctan2(d[1], d[0]))))
    el_aod.append(float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0)))))

    for path in paths_aoa_list[1:]:
        az_r, el_r = path[2], path[3]
        inc = np.array([
            np.cos(np.radians(el_r)) * np.cos(np.radians(az_r)),
            np.cos(np.radians(el_r)) * np.sin(np.radians(az_r)),
            np.sin(np.radians(el_r)),
        ])
        out = -inc
        az_aod.append(float(np.degrees(np.arctan2(out[1], out[0]))))
        el_aod.append(float(np.degrees(np.arcsin(np.clip(out[2], -1.0, 1.0)))))

    return np.array(az_aod), np.array(el_aod)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_path_statistics(
    name: str,
    delays: np.ndarray,
    amplitudes: np.ndarray,
    gains_dB: np.ndarray,
) -> dict:
    """Return a dict of summary statistics suitable for a DataFrame row."""
    return {
        "Tool": name,
        "Num Paths": len(delays),
        "Min Delay (ns)": float(delays.min()),
        "Max Delay (ns)": float(delays.max()),
        "Mean Delay (ns)": float(delays.mean()),
        "Std Delay (ns)": float(delays.std()),
        "Min Amplitude": float(amplitudes.min()),
        "Max Amplitude": float(amplitudes.max()),
        "Mean Amplitude": float(amplitudes.mean()),
        "Min Gain (dB)": float(gains_dB.min()),
        "Max Gain (dB)": float(gains_dB.max()),
        "Mean Gain (dB)": float(gains_dB.mean()),
        "Total Gain (dB)": float(
            10 * np.log10(np.sum(10 ** (gains_dB / 10)) + 1e-30)
        ),
    }


def rms_delay_spread(delays: np.ndarray, gains_dB: np.ndarray) -> float:
    """RMS delay spread [ns] from path delays and dB gains."""
    gains_linear = 10 ** (gains_dB / 10)
    mean_delay = np.sum(delays * gains_linear) / np.sum(gains_linear)
    return float(
        np.sqrt(
            np.sum((delays - mean_delay) ** 2 * gains_linear) / np.sum(gains_linear)
        )
    )
