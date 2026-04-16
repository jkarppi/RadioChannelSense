"""
01_generate_dataset.py — Ray-tracing dataset generation.

Runs Sionna RT and Mitsuba ray tracing over the selected scene and exports
HDF5 datasets and CSV path logs.  This is the first step of the pipeline;
the generated datasets are consumed by the downstream scripts
(02_rt_comparison.py, 03_localization.py, 04_channel_charting.py).

Outputs written to <SCENE_DIR>/:
  sionna_dataset.h5              complex channel coefficients (CIR + OFDM matrix)
  mitsuba_dataset.h5             Mitsuba path delays and amplitudes
  sionna_paths.csv               per-path delay / amplitude / gain (Sionna)
  mitsuba_paths.csv              per-path delay / amplitude / gain (Mitsuba)
  pictures/01_generate_dataset/  scene renders and CIR plots

Usage
-----
    python 01_generate_dataset.py [scene_name] [results_dir]

    scene_name   — name of the scene folder (default: "Otaniemi_small")
    results_dir  — optional override for the output directory

Change the scene_name argument in main() to switch between scenes.
"""

import os
import sys
import inspect
import importlib

import numpy as np
import pandas as pd
import h5py
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
    print(f"Output: {SCENE_DIR}")
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
        ``<SCENE_DIR>/pictures/01_generate_dataset/``, created if absent.
    """
    PICTURES_DIR = SCENE_DIR / "pictures" / "01_generate_dataset"
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    return PICTURES_DIR


# ── GPU / framework setup ─────────────────────────────────────────────────────

def setup_gpu_and_mitsuba():
    """Configure TensorFlow GPU memory growth and select the Mitsuba variant.

    Sets ``TF_CPP_MIN_LOG_LEVEL=3`` to suppress TensorFlow C++ log noise, then
    enables memory growth on all visible GPUs.  Mitsuba is initialised with the
    CUDA polarised variant when a GPU is available, falling back to the LLVM
    variant on CPU-only machines.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"GPUs: {[g.name for g in gpus]}")
    else:
        print("No GPU — running on CPU")

    import mitsuba as mi
    mi.set_variant('cuda_ad_mono_polarized', 'llvm_ad_mono_polarized')
    print(f"Mitsuba variant: {mi.variant()}")


# ── Load Sionna scene ─────────────────────────────────────────────────────────

def load_sionna_scene(cfg):
    """Load the Sionna scene and place transmitters and a reference UE.

    Reads the scene XML file specified in *cfg*, sets up 4-element TX arrays
    and a single-element RX array, adds all configured transmitters, and
    places the first reference UE position as receiver ``rx0``.

    Parameters
    ----------
    cfg : dict
        Scene configuration dict from :func:`config.get_scene_config`.
        Requires ``SCENE_XML_FILE_NAME``, ``tx_positions``, and
        ``ue_positions``.

    Returns
    -------
    scene : sionna.rt.Scene
        Loaded scene object with TX and RX nodes attached.
    """
    import tensorflow as tf
    from sionna.rt import (
        load_scene, PlanarArray, Transmitter, Receiver,
    )

    SCENE_XML_FILE_NAME = cfg['SCENE_XML_FILE_NAME']
    tx_positions = cfg['tx_positions']
    ue_positions = cfg['ue_positions']

    scene = load_scene(SCENE_XML_FILE_NAME)
    print(f"Scene loaded: {SCENE_XML_FILE_NAME}")

    scene.tx_array = PlanarArray(
        num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern='iso', polarization='V'
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern='iso', polarization='V'
    )

    for i, pos in enumerate(tx_positions):
        scene.add(Transmitter(name=f'tx{i}', position=pos))
    print(f"Added {len(tx_positions)} transmitters.")

    scene.add(Receiver(name='rx0', position=ue_positions[0]))
    print(f"UE at {ue_positions[0]}")

    return scene


# ── Render scene figure ───────────────────────────────────────────────────────

def render_scene_figure(scene, cfg, PICTURES_DIR):
    """Render the Sionna scene from the configured camera and save to a file.

    Uses the CAM_POSITION and CAM_LOOK_AT values from the scene config.
    Suppresses any interactive plt.show() call that Sionna may trigger
    internally so the function works in non-notebook/headless environments.
    """
    from sionna.rt import Camera

    cam_pos  = cfg.get('CAM_POSITION', [0, -700, 350])
    cam_look = cfg.get('CAM_LOOK_AT',  [0, 0, 35])
    resolution = [650, 500]

    cam = Camera(position=cam_pos, look_at=cam_look)

    figs_before = set(plt.get_fignums())

    # Prevent any plt.show() inside scene.render() from opening a GUI window.
    _orig_show = plt.show
    plt.show = lambda *a, **kw: None
    try:
        scene.render(camera=cam, num_samples=512, resolution=resolution,
                     show_devices=False)
    finally:
        plt.show = _orig_show

    new_figs = set(plt.get_fignums()) - figs_before
    out_path = PICTURES_DIR / 'scene_render.png'

    for fig_num in new_figs:
        fig = plt.figure(fig_num)
        # Sionna's renderer may leave numpy.float64 dpi values which cause
        # matplotlib's savefig to fail when restoring _original_dpi.
        fig.dpi = float(fig.dpi)
        if hasattr(fig, '_original_dpi'):
            fig._original_dpi = float(fig._original_dpi)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Scene render saved → {out_path}")
        return out_path

    print("scene.render() did not produce a new matplotlib figure.")
    return None


def render_scene_with_devices(scene, cfg, PICTURES_DIR):
    """Render the scene showing placed TX/RX devices and save to a file.

    Equivalent to the notebook cell that renders after adding transmitters
    and receivers (num_samples=1024, show_devices=True).
    """
    from sionna.rt import Camera

    cam_pos  = cfg.get('CAM_POSITION', [0, -700, 350])
    cam_look = cfg.get('CAM_LOOK_AT',  [0, 0, 35])
    resolution = [650, 500]

    cam = Camera(position=cam_pos, look_at=cam_look)

    figs_before = set(plt.get_fignums())

    _orig_show = plt.show
    plt.show = lambda *a, **kw: None
    try:
        scene.render(camera=cam, num_samples=1024, resolution=resolution,
                     show_devices=True)
    finally:
        plt.show = _orig_show

    new_figs = set(plt.get_fignums()) - figs_before
    out_path = PICTURES_DIR / 'scene_with_devices.png'

    for fig_num in new_figs:
        fig = plt.figure(fig_num)
        fig.dpi = float(fig.dpi)
        if hasattr(fig, '_original_dpi'):
            fig._original_dpi = float(fig._original_dpi)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Scene with devices saved → {out_path}")
        return out_path

    print("scene.render() did not produce a new matplotlib figure.")
    return None


def render_scene_with_paths(scene, paths, cfg, PICTURES_DIR):
    """Render the scene with propagation paths overlaid and save to a file.

    Equivalent to the notebook cell that renders after running PathSolver
    (num_samples=8192, show_devices=True, paths=paths).
    """
    from sionna.rt import Camera

    cam_pos  = cfg.get('CAM_POSITION', [0, -700, 350])
    cam_look = cfg.get('CAM_LOOK_AT',  [0, 0, 35])
    resolution = [650, 500]

    cam = Camera(position=cam_pos, look_at=cam_look)

    figs_before = set(plt.get_fignums())

    _orig_show = plt.show
    plt.show = lambda *a, **kw: None
    try:
        scene.render(camera=cam, paths=paths, num_samples=8192,
                     resolution=resolution, show_devices=True)
    finally:
        plt.show = _orig_show

    new_figs = set(plt.get_fignums()) - figs_before
    out_path = PICTURES_DIR / 'scene_with_paths.png'

    for fig_num in new_figs:
        fig = plt.figure(fig_num)
        fig.dpi = float(fig.dpi)
        if hasattr(fig, '_original_dpi'):
            fig._original_dpi = float(fig._original_dpi)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Scene with paths saved → {out_path}")
        return out_path

    print("scene.render() did not produce a new matplotlib figure.")
    return None


# ── OFDM frequency axis ───────────────────────────────────────────────────────

def setup_ofdm_frequencies(cfg):
    """Build the OFDM sub-carrier frequency axis.

    Constructs a centred frequency grid of length ``fft_size`` with spacing
    ``subcarrier_spacing`` Hz around the carrier frequency ``fc``.

    Parameters
    ----------
    cfg : dict
        Scene configuration dict — requires ``fft_size``,
        ``subcarrier_spacing``, and ``fc``.

    Returns
    -------
    frequencies : tf.Tensor of shape (fft_size,)
        Per-sub-carrier frequency values in Hz.
    """
    import tensorflow as tf

    fft_size = cfg['fft_size']
    subcarrier_spacing = cfg['subcarrier_spacing']
    fc = cfg['fc']

    frequencies = (
        tf.cast(tf.range(fft_size), tf.float32) - fft_size / 2.0
    ) * subcarrier_spacing + fc
    print(f"OFDM: {fft_size} sub-carriers, Δf={subcarrier_spacing/1e3:.0f} kHz, "
          f"fc={fc/1e9:.3f} GHz")
    return frequencies


# ── Run Sionna PathSolver ─────────────────────────────────────────────────────

def run_sionna_pathsolver(scene, cfg):
    """Run the Sionna PathSolver on the current scene.

    Traces LOS, reflection, and diffraction paths up to
    ``MAX_REFLECTION_DEPTH`` bounces.  Only keywords supported by the
    installed Sionna version are forwarded to the solver.

    Parameters
    ----------
    scene : sionna.rt.Scene
        Scene object returned by :func:`load_sionna_scene`.
    cfg : dict
        Scene configuration dict — requires ``MAX_REFLECTION_DEPTH``.

    Returns
    -------
    paths : sionna.rt.Paths
        Ray-traced path object containing delays, angles, and gains.
    """
    from sionna.rt import PathSolver

    MAX_REFLECTION_DEPTH = cfg['MAX_REFLECTION_DEPTH']

    path_solver = PathSolver()
    ps_kw = dict(
        max_depth=MAX_REFLECTION_DEPTH,
        los=True, reflection=True, diffraction=True, scattering=False
    )
    sup = set(inspect.signature(path_solver.__call__).parameters)
    ps_kw = {k: v for k, v in ps_kw.items() if k in sup}

    paths = path_solver(scene=scene, **ps_kw)
    print("PathSolver complete.")
    print(f"  Paths tau shape: {np.array(paths.tau).shape}")
    return paths


# ── Compute CIR and OFDM channel matrix ──────────────────────────────────────

def compute_cir_and_ofdm(paths, frequencies):
    """Convert Sionna ray-traced paths to a CIR and OFDM channel matrix.

    Calls ``paths.cir()`` to obtain complex path gains *a* and delays *tau*,
    then maps them to the frequency domain via ``cir_to_ofdm_channel``.

    Parameters
    ----------
    paths : sionna.rt.Paths
        Ray-traced paths from :func:`run_sionna_pathsolver`.
    frequencies : tf.Tensor
        Sub-carrier frequency axis from :func:`setup_ofdm_frequencies`.

    Returns
    -------
    a : ndarray
        Complex path gain coefficients.
    tau : ndarray
        Path delays in seconds.
    H : ndarray
        OFDM channel matrix ``H[batch, rx, tx, subcarrier]``.
    """
    try:
        from sionna.channel import cir_to_ofdm_channel
    except ModuleNotFoundError:
        from sionna.phy.channel import cir_to_ofdm_channel

    a, tau = paths.cir(out_type='numpy')
    print(f"a shape: {a.shape}")
    print(f"tau shape: {tau.shape}")

    H = np.array(cir_to_ofdm_channel(frequencies, a[np.newaxis], tau[np.newaxis], normalize=False))
    print(f"H shape: {H.shape}")
    return a, tau, H


# ── Best-link CIR plot ────────────────────────────────────────────────────────

def plot_sionna_cir(a, tau, PICTURES_DIR):
    if tau.ndim == 5 and a.ndim == 6:
        n_rx = tau.shape[0]
        n_tx = tau.shape[2]
        best = None
        best_score = -np.inf
        for rx_i in range(n_rx):
            for tx_i in range(n_tx):
                t_cand = tau[rx_i, 0, tx_i, 0, :] / 1e-9
                a_cand = np.abs(a[rx_i, 0, tx_i, 0, :, 0])
                valid  = np.isfinite(t_cand) & (t_cand > 0) & np.isfinite(a_cand) & (a_cand > 0)
                if np.any(valid):
                    score = float(np.max(a_cand[valid]))
                    if score > best_score:
                        best_score = score
                        best = (rx_i, tx_i, t_cand[valid], a_cand[valid])
        if best is None:
            print("No valid CIR paths found.")
            t_plot, a_plot, best_tx = np.array([0., 1.]), np.array([0., 0.]), -1
        else:
            rx_sel, best_tx, t_sel, a_sel = best
            order  = np.argsort(t_sel)
            t_plot = t_sel[order]
            a_plot = a_sel[order]
            print(f"Plotting CIR: RX{rx_sel}, TX{best_tx} ({len(t_plot)} valid paths).")
    else:
        tau_sq = np.squeeze(tau)
        a_sq   = np.squeeze(a)

        if tau_sq.ndim == 1:
            tau_sq = tau_sq[np.newaxis, :]
        if a_sq.ndim == 1:
            a_sq = a_sq[np.newaxis, :]

        n_tx    = tau_sq.shape[0]
        best_tx = 0
        best_score = -np.inf
        for tx_i in range(n_tx):
            t_cand = tau_sq[tx_i].ravel() / 1e-9
            a_cand = np.abs(a_sq[tx_i]).ravel()[:t_cand.size]
            valid  = np.isfinite(t_cand) & (t_cand > 0) & np.isfinite(a_cand) & (a_cand > 0)
            score  = float(np.sum(a_cand[valid] ** 2))
            if score > best_score:
                best_score = score
                best_tx    = tx_i

        t_raw  = tau_sq[best_tx].ravel() / 1e-9
        a_raw  = np.abs(a_sq[best_tx]).ravel()
        if a_sq.ndim == 3:
            a_raw = np.mean(np.abs(a_sq[best_tx]), axis=0).ravel()
        n = min(len(t_raw), len(a_raw))
        valid  = np.isfinite(t_raw[:n]) & (t_raw[:n] > 0) & np.isfinite(a_raw[:n]) & (a_raw[:n] > 0)
        t_sel  = t_raw[:n][valid]
        a_sel  = a_raw[:n][valid]
        order  = np.argsort(t_sel)
        t_plot = t_sel[order]
        a_plot = a_sel[order]
        if len(t_plot) == 0:
            print("No valid CIR paths found after filtering.")
            t_plot, a_plot = np.array([0., 1.]), np.array([0., 0.])
        else:
            print(f"Plotting CIR: TX{best_tx} ({len(t_plot)} valid paths).")

    a_max = float(np.max(a_plot)) if len(a_plot) else 0.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(t_plot, a_plot, markerfmt='C0o', linefmt='C0-', basefmt='k-')
    ax.set_xlabel(r'$\tau$ [ns]')
    ax.set_ylabel(r'$|a|$')
    ax.set_title(f'Sionna RT CIR — best link (TX{best_tx})')
    ax.set_xlim([0, max(1.0, float(np.max(t_plot)) * 1.1)])
    ax.set_ylim([-2e-6, max(1e-6, a_max * 1.1)])
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'sionna_cir.png', dpi=150)
    # plt.show()

    return best_tx


# ── Export Sionna dataset to HDF5 ─────────────────────────────────────────────

def export_sionna_dataset(a, tau, H, cfg, SCENE_DIR):
    """Save the Sionna CIR and OFDM channel matrix to HDF5 and CSV.

    Writes the complex channel coefficients to a compressed HDF5 file and
    exports a flat CSV of per-path delay / amplitude / gain for quick
    inspection.

    Parameters
    ----------
    a : ndarray
        Complex path gains from :func:`compute_cir_and_ofdm`.
    tau : ndarray
        Path delays in seconds.
    H : ndarray
        OFDM channel matrix.
    cfg : dict
        Scene configuration dict — requires ``fc``, ``fft_size``,
        ``subcarrier_spacing``, and ``MAX_REFLECTION_DEPTH``.
    SCENE_DIR : Path
        Output directory for ``sionna_dataset.h5`` and ``sionna_paths.csv``.
    """
    fc = cfg['fc']
    fft_size = cfg['fft_size']
    subcarrier_spacing = cfg['subcarrier_spacing']
    MAX_REFLECTION_DEPTH = cfg['MAX_REFLECTION_DEPTH']

    sionna_h5 = SCENE_DIR / 'sionna_dataset.h5'
    with h5py.File(sionna_h5, 'w') as f:
        f.create_dataset('CIR/a_real', data=a.real, compression='gzip')
        f.create_dataset('CIR/a_imag', data=a.imag, compression='gzip')
        f.create_dataset('CIR/tau_s',  data=tau,     compression='gzip')
        f.create_dataset('CIR/H_real', data=H.real,  compression='gzip')
        f.create_dataset('CIR/H_imag', data=H.imag,  compression='gzip')
        f.attrs['fc']                 = fc
        f.attrs['fft_size']           = fft_size
        f.attrs['subcarrier_spacing'] = subcarrier_spacing
        f.attrs['max_depth']          = MAX_REFLECTION_DEPTH
    print(f"Saved → {sionna_h5}  ({sionna_h5.stat().st_size / 1e6:.2f} MB)")

    a_csv   = np.squeeze(np.abs(a))
    tau_csv = np.squeeze(tau)

    while tau_csv.ndim < a_csv.ndim:
        tau_csv = np.expand_dims(tau_csv, axis=-2)

    tau_flat = np.broadcast_to(tau_csv, a_csv.shape).ravel()
    a_flat   = a_csv.ravel()
    valid    = (tau_flat >= 0) & (a_flat > 1e-20)

    df_s = pd.DataFrame({
        'delay_ns': tau_flat[valid] / 1e-9,
        'amplitude': a_flat[valid],
        'gain_dB':   20 * np.log10(a_flat[valid] + 1e-20),
    })
    csv_s = SCENE_DIR / 'sionna_paths.csv'
    df_s.to_csv(csv_s, index=False)
    print(f"Saved → {csv_s}  ({len(df_s)} paths, {csv_s.stat().st_size / 1e3:.1f} KB)")


# ── Mitsuba scene and CIR ─────────────────────────────────────────────────────

def run_mitsuba_raytracing(cfg, SCENE_DIR, PICTURES_DIR):
    """Run Mitsuba geometric ray tracing for TX0 → UE and plot the CIR.

    Traces paths from the first transmitter to the first reference UE using
    the Mitsuba backend, plots the resulting delay-amplitude profile, and
    returns the raw path data for downstream export.

    Parameters
    ----------
    cfg : dict
        Scene configuration dict — requires ``SCENE_XML_FILE_NAME``,
        ``tx_positions``, ``ue_positions``, and ``MAX_DEPTH``.
    SCENE_DIR : Path
        Scene directory (used only for its path, not written to here).
    PICTURES_DIR : Path
        Output directory for ``mitsuba_cir.png``.

    Returns
    -------
    mi_paths : list or None
        Raw Mitsuba path records (empty list if none found).
    delays_mi : ndarray or None
        Per-path delays (ns).
    amps_mi : ndarray or None
        Per-path amplitudes.
    gains_mi : ndarray or None
        Per-path gains in dB.
    origin_np : ndarray
        TX0 position as a (3,) float64 array.
    target_np : ndarray
        Reference UE position as a (3,) float64 array.
    """
    import rt_utils
    importlib.reload(rt_utils)
    from rt_utils import load_mitsuba_scene, trace_paths

    SCENE_XML_FILE_NAME = cfg['SCENE_XML_FILE_NAME']
    tx_positions = cfg['tx_positions']
    ue_positions = cfg['ue_positions']
    MAX_DEPTH = cfg['MAX_DEPTH']

    scene_mi = load_mitsuba_scene(SCENE_XML_FILE_NAME)
    print(f"Mitsuba scene loaded: {SCENE_XML_FILE_NAME}")

    origin_np = np.array(tx_positions[0], dtype=float)
    target_np = np.array(ue_positions[0], dtype=float)

    mi_paths = trace_paths(scene_mi, origin_np, target_np, max_depth=MAX_DEPTH)
    print(f"Mitsuba traced {len(mi_paths)} paths (TX0 → UE).")

    delays_mi, amps_mi, gains_mi = None, None, None

    if mi_paths:
        delays_mi = np.array([p[0] for p in mi_paths])
        amps_mi   = np.array([p[1] for p in mi_paths])
        gains_mi  = 20 * np.log10(amps_mi + 1e-20)

        valid = np.isfinite(delays_mi) & (delays_mi > 0) & np.isfinite(amps_mi) & (amps_mi > 0)
        t_plot = delays_mi[valid]
        a_plot = amps_mi[valid]
        order  = np.argsort(t_plot)
        t_plot = t_plot[order]
        a_plot = a_plot[order]

        a_max = float(np.max(a_plot)) if len(a_plot) else 0.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stem(t_plot, a_plot, markerfmt='C1o', linefmt='C1-', basefmt='k-')
        ax.set_xlabel(r'$\tau$ [ns]')
        ax.set_ylabel(r'$|a|$')
        ax.set_title('Mitsuba CIR (TX0 → UE)')
        ax.set_xlim([0, max(1.0, float(np.max(t_plot)) * 1.1)])
        ax.set_ylim([-2e-6, max(1e-6, a_max * 1.1)])
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(PICTURES_DIR / 'mitsuba_cir.png', dpi=150)
        # plt.show()
        print(f"Plotted {len(t_plot)} valid paths.")
    else:
        print("No Mitsuba paths found.")

    return mi_paths, delays_mi, amps_mi, gains_mi, origin_np, target_np


# ── Export Mitsuba dataset to HDF5 ───────────────────────────────────────────

def export_mitsuba_dataset(mi_paths, delays_mi, amps_mi, gains_mi,
                            origin_np, target_np, SCENE_DIR):
    """Save Mitsuba path data to HDF5 and CSV.

    Parameters
    ----------
    mi_paths : list
        Raw Mitsuba path records from :func:`run_mitsuba_raytracing`.
    delays_mi : ndarray or None
        Per-path delays (ns), or ``None`` if no paths were found.
    amps_mi : ndarray or None
        Per-path amplitude values.
    gains_mi : ndarray or None
        Per-path gains in dB.
    origin_np : ndarray
        TX0 position (3,).
    target_np : ndarray
        Reference UE position (3,).
    SCENE_DIR : Path
        Output directory for ``mitsuba_dataset.h5`` and ``mitsuba_paths.csv``.
    """
    if mi_paths:
        mi_h5 = SCENE_DIR / 'mitsuba_dataset.h5'
        with h5py.File(mi_h5, 'w') as f:
            f.create_dataset('paths/delay_ns',      data=delays_mi, compression='gzip')
            f.create_dataset('paths/amplitude_abs', data=amps_mi,   compression='gzip')
            f.create_dataset('CIR/tau_ns',          data=delays_mi, compression='gzip')
            f.attrs['tx_position'] = list(origin_np)
            f.attrs['rx_position'] = list(target_np)
            f.attrs['n_paths']     = len(mi_paths)
        print(f"Saved → {mi_h5}  ({mi_h5.stat().st_size / 1e6:.2f} MB)")

        df_m = pd.DataFrame({'delay_ns': delays_mi, 'amplitude': amps_mi,
                             'gain_dB': gains_mi})
        csv_m = SCENE_DIR / 'mitsuba_paths.csv'
        df_m.to_csv(csv_m, index=False)
        print(f"Saved → {csv_m}  ({len(df_m)} paths, {csv_m.stat().st_size / 1e3:.1f} KB)")
    else:
        print("No Mitsuba paths found — skipping HDF5 export.")


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
        Path.cwd() / "01_generate_dataset.ipynb",
        SCENE_DIR,
        pics_dir=PICTURES_DIR,
    )


def report_scene_config(cfg) -> str:
    """Return a Markdown table summarising the scene configuration."""
    import json

    # ── Scalar parameters ─────────────────────────────────────────────────────
    skip = {'SCENE_NAME', 'SCENE_XML_FILE_NAME', 'SPEED_OF_LIGHT',
            'tx_positions', 'ue_positions', 'CAM_POSITION', 'CAM_LOOK_AT'}
    rows = [
        ('Scene name',          cfg.get('SCENE_NAME', '—')),
        ('Scene XML',           cfg.get('SCENE_XML_FILE_NAME', '—')),
        ('Carrier frequency',   f"{cfg.get('fc', 0) / 1e9:.3f} GHz"),
        ('TX power',            f"{cfg.get('TX_POWER_DBM', '—')} dBm"),
        ('FFT size',            cfg.get('fft_size', '—')),
        ('Subcarrier spacing',  f"{cfg.get('subcarrier_spacing', 0) / 1e3:.0f} kHz"),
        ('BS antenna height',   f"{cfg.get('BS_H', '—')} m"),
        ('UE antenna height',   f"{cfg.get('UE_H', '—')} m"),
        ('Grid X range',        f"{cfg.get('GRID_X_MIN', '—')} … {cfg.get('GRID_X_MAX', '—')} m"),
        ('Grid Y range',        f"{cfg.get('GRID_Y_MIN', '—')} … {cfg.get('GRID_Y_MAX', '—')} m"),
        ('Grid spacing',        f"{cfg.get('GRID_SPACING', '—')} m"),
        ('Max reflection depth',cfg.get('MAX_REFLECTION_DEPTH', '—')),
        ('Max ray depth',       cfg.get('MAX_DEPTH', '—')),
        ('Transmitters',        len(cfg.get('tx_positions', []))),
        ('Reference UE pos',    ', '.join(
            f"({p[0]}, {p[1]}, {p[2]})" for p in cfg.get('ue_positions', []))),
    ]

    lines = [
        '### Scene Configuration',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
    ]
    for name, val in rows:
        lines.append(f'| {name} | `{val}` |')

    # TX positions sub-table
    tx_positions = cfg.get('tx_positions', [])
    if tx_positions:
        lines += [
            '',
            '**Transmitter positions (x, y, z) [m]:**',
            '',
            '| TX | x | y | z |',
            '|----|---|---|---|',
        ]
        for i, pos in enumerate(tx_positions):
            lines.append(f'| TX{i} | {pos[0]} | {pos[1]} | {pos[2]} |')

    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(scene_name="Otaniemi_small", results_dir=None):
    """Run the full dataset generation pipeline for *scene_name*.

    Steps:
    1. Load scene config and initialise GPU / Mitsuba.
    2. Load the Sionna scene and place TX/RX nodes.
    3. Render scene visualisations (empty, with devices, with ray paths).
    4. Run the Sionna PathSolver and compute the OFDM channel matrix.
    5. Export Sionna CIR and channel matrix to HDF5 + CSV.
    6. Run Mitsuba geometric ray tracing (TX0 → UE0) and export results.
    7. Save a Markdown report.

    Parameters
    ----------
    scene_name : str
        Scene folder name (must contain ``scene_config.json``).
    results_dir : str or None
        Override for the output directory.  When ``None``, outputs are
        written to ``<SCENE_DIR>/``.
    """
    cfg, SCENE_DIR = setup_config(scene_name)
    OUTPUT_DIR = Path(results_dir) if results_dir is not None else SCENE_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if results_dir is not None:
        PICTURES_DIR = OUTPUT_DIR / "pictures" / "01_generate_dataset"
        PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    else:
        PICTURES_DIR = setup_pictures_dir(SCENE_DIR)

    from report_utils import MarkdownReport
    report = MarkdownReport()
    report.add(f"""\
# 01 — Generate Dataset

Runs Sionna RT and Mitsuba ray tracing over the selected scene and exports HDF5 datasets.

**Scene:** `{cfg['SCENE_XML_FILE_NAME']}`  
**Output:** `{OUTPUT_DIR}`

**Outputs written to `OUTPUT_DIR/`:**
- `sionna_dataset.h5` — complex channel coefficients and OFDM channel matrix
- `mitsuba_dataset.h5` — Mitsuba path delays and amplitudes
- `sionna_paths.csv` — per-path delay / amplitude / gain (Sionna)
- `mitsuba_paths.csv` — per-path delay / amplitude / gain (Mitsuba)""")

    report.add(report_scene_config(cfg))

    with report.capture():
        setup_gpu_and_mitsuba()

    with report.capture():
        scene = load_sionna_scene(cfg)

    report.add("### Scene Render")
    with report.capture():
        render_scene_figure(scene, cfg, PICTURES_DIR)
    if (PICTURES_DIR / 'scene_render.png').exists():
        report.figure(PICTURES_DIR / 'scene_render.png', OUTPUT_DIR)

    report.add("### Scene with Devices")
    with report.capture():
        render_scene_with_devices(scene, cfg, PICTURES_DIR)
    if (PICTURES_DIR / 'scene_with_devices.png').exists():
        report.figure(PICTURES_DIR / 'scene_with_devices.png', OUTPUT_DIR)

    with report.capture():
        frequencies = setup_ofdm_frequencies(cfg)
        paths = run_sionna_pathsolver(scene, cfg)

    report.add("### Scene with Paths")
    with report.capture():
        render_scene_with_paths(scene, paths, cfg, PICTURES_DIR)
    if (PICTURES_DIR / 'scene_with_paths.png').exists():
        report.figure(PICTURES_DIR / 'scene_with_paths.png', OUTPUT_DIR)

    with report.capture():
        a, tau, H = compute_cir_and_ofdm(paths, frequencies)

    report.add("### Sionna RT — Channel Impulse Response")
    with report.capture():
        plot_sionna_cir(a, tau, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'sionna_cir.png', OUTPUT_DIR)

    with report.capture():
        export_sionna_dataset(a, tau, H, cfg, OUTPUT_DIR)

    report.add("### Mitsuba — Channel Impulse Response")
    with report.capture():
        mi_paths, delays_mi, amps_mi, gains_mi, origin_np, target_np = \
            run_mitsuba_raytracing(cfg, SCENE_DIR, PICTURES_DIR)
    if (PICTURES_DIR / 'mitsuba_cir.png').exists():
        report.figure(PICTURES_DIR / 'mitsuba_cir.png', OUTPUT_DIR)

    with report.capture():
        export_mitsuba_dataset(mi_paths, delays_mi, amps_mi, gains_mi,
                               origin_np, target_np, OUTPUT_DIR)

    report.add("""\
## Analysis

Both Sionna RT and Mitsuba independently traced multipath propagation through the scene.

**Sionna RT** solves the full electromagnetic problem including diffraction and produces
complex-valued per-path channel coefficients (`a`), delays (`tau`), and the OFDM
channel matrix `H`.  The data are saved as compressed HDF5 for downstream use.

**Mitsuba** performs geometric ray tracing, yielding path amplitudes from power-law
attenuation without phase.  It typically finds far more candidate paths than Sionna
because it does not enforce electromagnetic validity.

The two CIR plots above show the delay–amplitude profile for the strongest link
as seen by each tracer.  Large differences in path count or delay spread indicate
that one tracer found reflections or diffractions the other missed.  These datasets
are the input to `02_rt_comparison.py` and `03_localization.py`.""")

    report.save(OUTPUT_DIR / '01_generate_dataset_report.md')


if __name__ == "__main__":
    _scene = sys.argv[1] if len(sys.argv) > 1 else "Otaniemi_small"
    _results_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_scene, _results_dir)
