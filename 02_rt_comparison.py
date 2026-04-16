"""
02_rt_comparison.py

Loads the pre-generated HDF5 datasets produced by 01_generate_dataset and
produces side-by-side comparisons of Sionna RT and Mitsuba ray-tracing results.

Requires:
  <SCENE_DIR>/sionna_dataset.h5
  <SCENE_DIR>/mitsuba_dataset.h5

Change the scene_name argument in main() to switch between scenes.
"""

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
    from config import get_scene_config, scene_dir

    cfg = get_scene_config(scene_name)
    SCENE_DIR = scene_dir(cfg)
    print(f"Scene : {cfg['SCENE_XML_FILE_NAME']}")
    print(f"Data  : {SCENE_DIR}")
    return cfg, SCENE_DIR


def setup_pictures_dir(SCENE_DIR):
    PICTURES_DIR = SCENE_DIR / "pictures" / "02_rt_comparison"
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    return PICTURES_DIR


# ── Load datasets ─────────────────────────────────────────────────────────────

def load_sionna_dataset(cfg, SCENE_DIR):
    sionna_h5 = SCENE_DIR / 'sionna_dataset.h5'
    with h5py.File(sionna_h5, 'r') as f:
        a_real = f['CIR/a_real'][:]
        a_imag = f['CIR/a_imag'][:]
        tau_s  = f['CIR/tau_s'][:]
        fc_saved  = float(f.attrs.get('fc',       cfg['fc']))
        fft_saved = int(  f.attrs.get('fft_size', cfg['fft_size']))

    a_sionna   = a_real + 1j * a_imag
    tau_sionna = tau_s
    print(f"Sionna  a shape  : {a_sionna.shape}")
    print(f"Sionna  tau shape: {tau_sionna.shape}")
    return a_sionna, tau_sionna


def load_mitsuba_dataset(SCENE_DIR):
    mi_h5 = SCENE_DIR / 'mitsuba_dataset.h5'
    with h5py.File(mi_h5, 'r') as f:
        delays_mi = f['paths/delay_ns'][:]
        amps_mi   = f['paths/amplitude_abs'][:]

    gains_mi = 20 * np.log10(amps_mi + 1e-20)
    print(f"Mitsuba paths: {len(delays_mi)}")
    return delays_mi, amps_mi, gains_mi


# ── Extract best Sionna link ──────────────────────────────────────────────────

def extract_best_sionna_link(a_sionna, tau_sionna):
    a_sq   = np.squeeze(a_sionna)
    tau_sq = np.squeeze(tau_sionna)

    if a_sq.ndim == 1:
        a_sq   = a_sq[np.newaxis, :]
    if tau_sq.ndim == 1:
        tau_sq = tau_sq[np.newaxis, :]

    if a_sq.ndim == 3:
        a_2d = np.mean(np.abs(a_sq), axis=1)
    else:
        a_2d = np.abs(a_sq)

    tau_2d = tau_sq

    best_tx = int(np.argmax(np.sum(a_2d ** 2, axis=-1)))
    a_link  = a_2d[best_tx]
    t_link  = tau_2d[best_tx]

    valid = np.isfinite(t_link) & (t_link > 0) & np.isfinite(a_link) & (a_link > 1e-20)
    a_link, t_link = a_link[valid], t_link[valid]
    order  = np.argsort(t_link)
    a_link, t_link = a_link[order], t_link[order]
    gains_s = 20 * np.log10(a_link + 1e-20)
    print(f"Sionna best TX: {best_tx}, {valid.sum()} valid paths")
    return a_link, t_link, gains_s, best_tx, valid


# ── Side-by-side CIR comparison ───────────────────────────────────────────────

def plot_cir_comparison(t_link, gains_s, a_link, delays_mi, gains_mi, best_tx, PICTURES_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].stem(t_link / 1e-9, gains_s,
                 markerfmt='C0o', linefmt='C0-', basefmt='k-')
    axes[0].set_xlabel('Delay (ns)'); axes[0].set_ylabel('Gain (dB)')
    axes[0].set_title(f'Sionna RT — TX{best_tx} → UE  ({len(a_link)} paths)')
    axes[0].grid(True)

    axes[1].stem(delays_mi, gains_mi,
                 markerfmt='C1o', linefmt='C1-', basefmt='k-')
    axes[1].set_xlabel('Delay (ns)'); axes[1].set_ylabel('Gain (dB)')
    axes[1].set_title(f'Mitsuba — TX0 → UE  ({len(delays_mi)} paths)')
    axes[1].grid(True)

    plt.suptitle('Channel Impulse Response Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'cir_comparison.png', dpi=150)
    # plt.show()


# ── Path statistics comparison ────────────────────────────────────────────────

def compute_path_stats(t_link, a_link, gains_s, delays_mi, amps_mi, gains_mi):
    from rt_utils import compute_path_statistics, rms_delay_spread

    stats_s = compute_path_statistics(
        'Sionna RT', t_link / 1e-9, np.abs(a_link), gains_s
    )
    stats_m = compute_path_statistics(
        'Mitsuba', delays_mi, amps_mi, gains_mi
    )

    df_stats = pd.DataFrame([stats_s, stats_m])
    print(df_stats.to_string(index=False))

    rds_s = rms_delay_spread(t_link / 1e-9, gains_s)
    rds_m = rms_delay_spread(delays_mi, gains_mi)
    print(f"\nRMS Delay Spread — Sionna: {rds_s:.2f} ns  |  Mitsuba: {rds_m:.2f} ns")
    return rds_s, rds_m


# ── AoA comparison ────────────────────────────────────────────────────────────

def run_aoa_comparison(cfg, best_tx, PICTURES_DIR):
    try:
        import tensorflow as tf
        import mitsuba as mi
        import sionna
        from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
        try:
            from sionna.channel import cir_to_ofdm_channel
        except ModuleNotFoundError:
            from sionna.phy.channel import cir_to_ofdm_channel

        SCENE_XML_FILE_NAME = cfg['SCENE_XML_FILE_NAME']
        tx_positions = cfg['tx_positions']
        ue_positions = cfg['ue_positions']
        MAX_REFLECTION_DEPTH = cfg['MAX_REFLECTION_DEPTH']
        MAX_DEPTH = cfg['MAX_DEPTH']

        scene = load_scene(SCENE_XML_FILE_NAME)
        scene.tx_array = PlanarArray(num_rows=1, num_cols=4,
            vertical_spacing=0.5, horizontal_spacing=0.5, pattern='iso', polarization='V')
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5, pattern='iso', polarization='V')
        for i, pos in enumerate(tx_positions):
            scene.add(Transmitter(name=f'tx{i}', position=pos))
        scene.add(Receiver(name='rx0', position=ue_positions[0]))

        path_solver = PathSolver()
        sup = set(inspect.signature(path_solver.__call__).parameters)
        ps_kw = {k: v for k, v in dict(max_depth=MAX_REFLECTION_DEPTH,
            los=True, reflection=True, diffraction=True, scattering=False).items()
            if k in sup}
        paths = path_solver(scene=scene, **ps_kw)

        phi_r   = np.array(paths.phi_r)
        theta_r = np.array(paths.theta_r)
        tau_rt  = np.array(paths.tau)
        a_rt, _ = paths.cir(out_type='numpy')

        def _flat(arr, tx_idx=best_tx):
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                return arr[tx_idx]
            return arr

        phi_r_1d   = np.degrees(_flat(phi_r))
        theta_r_1d = np.degrees(_flat(theta_r))
        valid_rt   = np.abs(_flat(np.squeeze(a_rt))) > 1e-20

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sc0 = axes[0].scatter(phi_r_1d[valid_rt], theta_r_1d[valid_rt],
                               c=_flat(tau_rt / 1e-9)[valid_rt], cmap='viridis', s=60)
        plt.colorbar(sc0, ax=axes[0], label='Delay (ns)')
        axes[0].set_xlabel('Azimuth AoA (°)'); axes[0].set_ylabel('Elevation AoA (°)')
        axes[0].set_title(f'Sionna RT AoA — TX{best_tx}')
        axes[0].grid(True)

        from rt_utils import compute_aoa_mitsuba, load_mitsuba_scene
        mi.set_variant('llvm_ad_mono_polarized', 'llvm_ad_rgb')
        scene_mi = load_mitsuba_scene(SCENE_XML_FILE_NAME)
        origin_np = np.array(tx_positions[0], dtype=float)
        target_np = np.array(ue_positions[0], dtype=float)
        mi_aoa = compute_aoa_mitsuba(scene_mi, origin_np, target_np, max_depth=MAX_DEPTH)
        if mi_aoa:
            az_mi = np.array([p[2] for p in mi_aoa])
            el_mi = np.array([p[3] for p in mi_aoa])
            dl_mi = np.array([p[0] for p in mi_aoa])
            sc1 = axes[1].scatter(az_mi, el_mi, c=dl_mi, cmap='viridis', s=60)
            plt.colorbar(sc1, ax=axes[1], label='Delay (ns)')
        axes[1].set_xlabel('Azimuth AoA (°)'); axes[1].set_ylabel('Elevation AoA (°)')
        axes[1].set_title('Mitsuba AoA (TX0 → UE)')
        axes[1].grid(True)

        plt.suptitle('Angle-of-Arrival Comparison', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PICTURES_DIR / 'aoa_comparison.png', dpi=150)
        # plt.show()
    except Exception as e:
        print(f"AoA comparison skipped ({e})")


# ── Delay spread comparison bar chart ─────────────────────────────────────────

def plot_delay_spread_comparison(valid, delays_mi, rds_s, rds_m, PICTURES_DIR):
    methods  = ['Sionna RT', 'Mitsuba']
    n_paths  = [int(valid.sum()), len(delays_mi)]
    rds_vals = [rds_s, rds_m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(methods, n_paths, color=['C0', 'C1'], alpha=0.85, edgecolor='k')
    ax1.set_ylabel('Number of Paths'); ax1.set_title('Path Count')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(methods, rds_vals, color=['C0', 'C1'], alpha=0.85, edgecolor='k')
    ax2.set_ylabel('Delay Spread (ns)'); ax2.set_title('RMS Delay Spread')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('RT Tool Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PICTURES_DIR / 'rt_comparison_stats.png', dpi=150)
    # plt.show()


# ── Save Markdown report ──────────────────────────────────────────────────────

def save_report(SCENE_DIR, PICTURES_DIR):
    import report_utils
    importlib.reload(report_utils)
    from report_utils import save_notebook_report

    save_notebook_report(
        Path.cwd() / "02_rt_comparison.ipynb",
        SCENE_DIR,
        pics_dir=PICTURES_DIR,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(scene_name="Otaniemi_small", results_dir=None):
    cfg, SCENE_DIR = setup_config(scene_name)
    OUTPUT_DIR = Path(results_dir) if results_dir is not None else SCENE_DIR
    if results_dir is not None:
        PICTURES_DIR = OUTPUT_DIR / "pictures" / "02_rt_comparison"
        PICTURES_DIR.mkdir(parents=True, exist_ok=True)
    else:
        PICTURES_DIR = setup_pictures_dir(SCENE_DIR)

    from report_utils import MarkdownReport
    report = MarkdownReport()
    report.add(f"""\
# 02 — RT Comparison

Loads the pre-generated HDF5 datasets produced by `01_generate_dataset.py`
and produces side-by-side comparisons of Sionna RT and Mitsuba ray-tracing results.

**Scene:** `{cfg['SCENE_XML_FILE_NAME']}`  
**Data:** `{OUTPUT_DIR}`

**Requires:** `sionna_dataset.h5` and `mitsuba_dataset.h5` in the data directory.""")

    with report.capture():
        a_sionna, tau_sionna = load_sionna_dataset(cfg, OUTPUT_DIR)
        delays_mi, amps_mi, gains_mi = load_mitsuba_dataset(OUTPUT_DIR)

    with report.capture():
        a_link, t_link, gains_s, best_tx, valid = extract_best_sionna_link(
            a_sionna, tau_sionna
        )

    report.add("### CIR Comparison — Sionna RT vs Mitsuba")
    with report.capture():
        plot_cir_comparison(t_link, gains_s, a_link, delays_mi, gains_mi,
                            best_tx, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'cir_comparison.png', OUTPUT_DIR)

    with report.capture():
        rds_s, rds_m = compute_path_stats(t_link, a_link, gains_s,
                                          delays_mi, amps_mi, gains_mi)

    report.add("### Angle of Arrival Comparison")
    with report.capture():
        run_aoa_comparison(cfg, best_tx, PICTURES_DIR)
    if (PICTURES_DIR / 'aoa_comparison.png').exists():
        report.figure(PICTURES_DIR / 'aoa_comparison.png', OUTPUT_DIR)

    report.add("### Delay Spread Comparison")
    with report.capture():
        plot_delay_spread_comparison(valid, delays_mi, rds_s, rds_m, PICTURES_DIR)
    report.figure(PICTURES_DIR / 'rt_comparison_stats.png', OUTPUT_DIR)

    report.add(f"""\
## Analysis

Sionna RT models the full propagation physics including diffraction and polarisation,
while Mitsuba performs purely geometric ray tracing.

**RMS delay spread** — Sionna: `{rds_s:.2f} ns` | Mitsuba: `{rds_m:.2f} ns`

Key observations:
- **Path counts**: Sionna typically finds fewer but physically consistent paths;
  Mitsuba may trace many more purely geometric reflections.
- **Delay spread**: Large differences indicate the two tracers disagree on the
  multipath structure of the channel. Sionna's diffraction model tends to produce
  longer delays; Mitsuba's power-law model may concentrate energy at shorter delays.
- **AoA**: When available, angle-of-arrival spread reflects scatterer geometry.
  Disagreements between tools highlight model limitations in diffuse scattering.

The path statistics table above quantifies the spread numerically.  These outputs
feed directly into `03_localization.py` as the ground-truth channel dataset.""")

    report.save(OUTPUT_DIR / '02_rt_comparison_report.md')


if __name__ == "__main__":
    _scene = sys.argv[1] if len(sys.argv) > 1 else "Otaniemi_small"
    _results_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_scene, _results_dir)
