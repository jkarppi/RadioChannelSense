"""
config.py — Single source of truth for all scene-specific parameters.

Overview
--------
Each scene has its parameters stored in a JSON file located at:

    <scene_name>/scene_config.json

A second per-scene JSON file controls which features and algorithms are active:

    <scene_name>/features_config.json

All four analysis scripts (01_generate_dataset.py … 04_channel_charting.py)
call ``get_scene_config(scene_name)`` at startup to receive a single ``cfg``
dict that drives every scene-dependent value: frequency, transmitter positions,
simulation grid, ray-tracing depth, and camera placement.

Scripts 03_localization.py and 04_channel_charting.py additionally call
``get_features_config(scene_name)`` to obtain the enable/disable toggles for
fingerprint feature groups, localization methods, and channel-charting methods.

To customise a scene, edit its JSON files directly — no Python changes needed.

Typical usage
-------------
    from config import get_scene_config, get_features_config, scene_dir

    cfg      = get_scene_config("Otaniemi_small")
    feat_cfg = get_features_config("Otaniemi_small")

    # Scene parameters
    fc        = cfg["fc"]               # carrier frequency (Hz)
    tx_pos    = cfg["tx_positions"]     # list of [x, y, z] lists (metres)
    grid_step = cfg["GRID_SPACING"]     # UE grid spacing (metres)

    # Feature toggles
    use_tdoa = feat_cfg["fingerprint_features"]["tdoa"]
    use_wknn = feat_cfg["localization"]["wknn"]

    # Derived helpers
    output_path = scene_dir(cfg)        # creates <scene_name>/ if needed

scene_config.json keys
----------------------
Derived (always computed, never stored in JSON):
    SCENE_NAME          str     scene folder name, e.g. "Otaniemi_small"
    SCENE_XML_FILE_NAME str     relative path to the Mitsuba/Sionna XML file
    SPEED_OF_LIGHT      float   299 792 458 m/s

Radio / channel:
    fc                  float   carrier frequency in Hz  (default 3.6e9)
    TX_POWER_DBM        float   transmit power in dBm
    fft_size            int     OFDM FFT size
    subcarrier_spacing  float   OFDM subcarrier spacing in Hz

Scene geometry:
    SCENE_CX / SCENE_CY float   centre of the scene in scene coordinates (m)
    BS_H                float   base-station antenna height (m)
    UE_H                float   UE antenna height (m)
    tx_positions        list    list of [x, y, z] positions for each TX (m)
    ue_positions        list    list of [x, y, z] reference UE positions (m)

UE sampling grid (used by 01_generate_dataset.py):
    GRID_X_MIN / GRID_X_MAX   float   west/east boundary (m)
    GRID_Y_MIN / GRID_Y_MAX   float   south/north boundary (m)
    GRID_SPACING              float   grid step size (m)

Ray-tracing (Sionna / Mitsuba):
    MAX_DEPTH               int   maximum total ray interactions
    MAX_REFLECTION_DEPTH    int   maximum reflection bounces

Visualisation:
    CAM_POSITION    list    [x, y, z] camera position for scene renders
    CAM_LOOK_AT     list    [x, y, z] camera look-at target

features_config.json keys
-------------------------
fingerprint_features : dict[str, bool]
    ofdm_mag_gd     — OFDM dB-magnitude + group-delay per sub-carrier per BS
    tdoa            — Time Difference of Arrival between every BS pair (ns)
    aoa             — Amplitude-weighted mean AoA per BS (sin/cos az & el)
    rss             — Received Signal Strength per BS (dB, all-path power)
    path_loss       — Path loss of the dominant (earliest) path per BS (dB)
    delay           — Absolute delay of the dominant path per BS (ns)
    cov_eigenvalues — Top-K spatial covariance eigenvalues per BS (ULA)
    reached_flags   — Binary flag: BS-UE link has at least one valid path

localization : dict[str, bool]
    wknn              — Weighted k-Nearest Neighbours (IDW)
    nn_regression     — 4-layer DNN regressing (x, y) coordinates
    nn_classification — DNN classifying grid cells → centroid positions
    cnn_regression    — 2-D CNN treating the feature matrix as an image

channel_charting : dict[str, bool]
    pca         — Principal Component Analysis
    tsne        — t-distributed Stochastic Neighbour Embedding
    autoencoder — Unsupervised autoencoder with 2-D latent space
    umap        — Uniform Manifold Approximation and Projection
"""

from __future__ import annotations
import copy
from pathlib import Path
import json
import os

# Physical constant used across all scripts when converting between
# frequency and wavelength (lambda = SPEED_OF_LIGHT / fc).
SPEED_OF_LIGHT: float = 299_792_458.0

# Fallback carrier frequency used when "fc" is absent from the JSON file.
FC_DEFAULT: float = 3.6e9


def _json_path(scene_name: str) -> Path:
    """Return the expected path of the per-scene JSON config file.

    The file lives beside the scene's XML file so that all scene assets
    are self-contained in one folder:

        <scene_name>/scene_config.json
    """
    return Path(scene_name) / "scene_config.json"


def get_scene_config(scene_name: str | None = None) -> dict:
    """Load and return the config dict for *scene_name*.

    Parameters
    ----------
    scene_name:
        Name of the scene folder, e.g. ``"Otaniemi_small"``.
        If ``None``, the environment variable ``SCENE_NAME`` is read;
        if that is also unset, ``"Otaniemi_small"`` is used as the default.

    Returns
    -------
    dict
        A flat dictionary containing all keys described in the module
        docstring above.  It is safe to unpack with ``cfg["key"]`` or
        pass directly to ``apply_config_to_globals()``.

    Raises
    ------
    FileNotFoundError
        If ``<scene_name>/scene_config.json`` does not exist.
        Create the file by copying an existing one and adjusting values,
        or call ``save_scene_config_json()`` after defining defaults.
    """
    if scene_name is None:
        scene_name = os.getenv("SCENE_NAME", "Otaniemi_small")

    jp = _json_path(scene_name)
    if not jp.exists():
        raise FileNotFoundError(
            f"No config found for scene '{scene_name}'. "
            f"Expected: {jp.resolve()}"
        )

    with jp.open() as fh:
        data = json.load(fh)

    # "fc" is pulled out explicitly so it can be cast to float and placed
    # alongside the other derived keys rather than at a random dict position.
    fc = float(data.pop("fc", FC_DEFAULT))

    return dict(
        SCENE_NAME=scene_name,
        SCENE_XML_FILE_NAME=f"{scene_name}/{scene_name}.xml",
        SPEED_OF_LIGHT=SPEED_OF_LIGHT,
        fc=fc,
        **data,           # all remaining JSON keys (TX_POWER_DBM, tx_positions, …)
    )


def apply_config_to_globals(cfg: dict, globs: dict) -> None:
    """Inject all config keys into a ``globals()`` dict.

    Intended for Jupyter notebooks where it is convenient to have every
    config key available as a bare variable name rather than via ``cfg[...]``.

    Parameters
    ----------
    cfg:
        Dict returned by ``get_scene_config()``.
    globs:
        The notebook's global namespace — pass ``globals()`` from the
        calling cell.

    Example
    -------
        cfg = get_scene_config("OtaniemiScene_100m")
        apply_config_to_globals(cfg, globals())
        print(fc)          # now available as a bare variable
        print(tx_positions)
    """
    globs.update(cfg)


def scene_dir(cfg: dict, base_dir: str | Path | None = None) -> Path:
    """Return (and create if needed) the scene output directory.

    Parameters
    ----------
    cfg:
        Dict returned by ``get_scene_config()``.
    base_dir:
        Root directory under which the scene folder is located.
        Defaults to the current working directory.

    Returns
    -------
    Path
        Absolute path to ``<base_dir>/<scene_name>/``, created if absent.
    """
    root = Path(base_dir) if base_dir else Path.cwd()
    d = root / Path(cfg["SCENE_XML_FILE_NAME"]).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


# Default feature flags — every feature and method is enabled by default.
# Any key absent from features_config.json falls back to these values.
_FEATURES_DEFAULTS: dict = {
    "fingerprint_features": {
        "ofdm_mag_gd":     True,
        "tdoa":            True,
        "aoa":             True,
        "rss":             True,
        "path_loss":       True,
        "delay":           True,
        "cov_eigenvalues": True,
        "reached_flags":   True,
    },
    "localization": {
        "wknn":              True,
        "wknn_c2f":          True,
        "nn_regression":     True,
        "nn_classification": True,
        "cnn_regression":    True,
        "ensemble":          True,
    },
    "channel_charting": {
        "pca":         True,
        "tsne":        True,
        "autoencoder": True,
        "umap":        True,
    },
    "evaluation": {
        # How to split the fingerprint grid into train / test.
        #   "random"       : uniform random split (default; interpolation-biased
        #                    on dense grids — every test point has near-duplicate
        #                    training neighbours so numbers can be optimistic).
        #   "checkerboard" : alternate grid cells by (x_idx + y_idx) % 2;
        #                    genuine interpolation test, ~50/50 split.
        #   "block"        : contiguous hold-out region (top-right corner by
        #                    x + y); extrapolation test — typically much harder.
        "split_method":  "random",
        "test_fraction": 0.3,
        "random_state":  42,
        # wKNN-only: PCA-whiten the feature space before the NN search.
        # Equivalent to Mahalanobis distance in the retained subspace — useful
        # when many columns are correlated (OFDM subcarriers).  Mutually
        # exclusive with the per-group inverse-sqrt-dim weighting; when enabled
        # group_sizes is ignored.
        "wknn_pca_whiten":   False,
        "wknn_pca_variance": 0.95,
    },
}


def get_features_config(scene_name: str | None = None) -> dict:
    """Load feature enable/disable flags for *scene_name*.

    Reads ``<scene_name>/features_config.json`` if the file exists and merges
    it on top of the built-in defaults.  Any key absent from the file is left
    at its default value (``True`` — enabled).  Unknown keys in the file are
    silently ignored.

    Parameters
    ----------
    scene_name:
        Name of the scene folder, e.g. ``"Otaniemi_small"``.
        If ``None``, the environment variable ``SCENE_NAME`` is consulted;
        if that is also unset, ``"Otaniemi_small"`` is used.

    Returns
    -------
    dict
        Nested dict with three top-level sections:

        * ``fingerprint_features`` — ``{str: bool}`` controlling which CSI
          feature groups are included in the fingerprint vector.
        * ``localization`` — ``{str: bool}`` enabling/disabling each
          localization algorithm in ``03_localization.py``.
        * ``channel_charting`` — ``{str: bool}`` enabling/disabling each
          dimensionality-reduction method in ``04_channel_charting.py``.

    Examples
    --------
    Disable TDoA and CNN regression for a quick test run::

        # features_config.json
        {
          "fingerprint_features": {"tdoa": false},
          "localization":         {"cnn_regression": false}
        }

        feat_cfg = get_features_config("Otaniemi_small")
        # All other flags remain True (enabled).
    """
    if scene_name is None:
        scene_name = os.getenv("SCENE_NAME", "Otaniemi_small")

    cfg = copy.deepcopy(_FEATURES_DEFAULTS)
    jp = Path(scene_name) / "features_config.json"

    if jp.exists():
        with jp.open() as fh:
            data = json.load(fh)
        # Deep merge: update only recognised sections and keys whose value
        # type matches the default (or is compatible — e.g. an int for a
        # float default).  This allows the ``evaluation`` section to carry
        # strings/ints alongside the boolean feature/method toggles.
        for section in _FEATURES_DEFAULTS:
            if section in data and isinstance(data[section], dict):
                for key, val in data[section].items():
                    if key not in cfg[section]:
                        continue
                    default = cfg[section][key]
                    if isinstance(default, bool):
                        if isinstance(val, bool):
                            cfg[section][key] = val
                    elif isinstance(default, (int, float)):
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            cfg[section][key] = val
                    elif isinstance(default, str):
                        if isinstance(val, str):
                            cfg[section][key] = val

    return cfg


def save_scene_config_json(scene_name: str | None = None) -> Path:
    """Write (or overwrite) the JSON config file for *scene_name*.

    Reads the current in-memory config via ``get_scene_config()`` and
    serialises it back to ``<scene_name>/scene_config.json``.  The derived
    keys (``SCENE_NAME``, ``SCENE_XML_FILE_NAME``, ``SPEED_OF_LIGHT``) are
    intentionally omitted from the file because they are always computed at
    load time.

    Parameters
    ----------
    scene_name:
        Name of the scene folder.  Follows the same ``None`` → env-var →
        default resolution as ``get_scene_config()``.

    Returns
    -------
    Path
        Path of the written JSON file.
    """
    cfg = get_scene_config(scene_name)
    jp = _json_path(cfg["SCENE_NAME"])
    # Omit keys that are derived at runtime and should not be hardcoded in JSON.
    _skip = {"SCENE_NAME", "SCENE_XML_FILE_NAME", "SPEED_OF_LIGHT"}
    payload = {k: v for k, v in cfg.items() if k not in _skip}
    jp.parent.mkdir(parents=True, exist_ok=True)
    with jp.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return jp
