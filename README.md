# Project work

This project explores **ray-tracing-based wireless channel simulation** in an outdoor/urban environment (Aalto University Otaniemi campus) using two complementary tools:

- **Sionna RT** — NVIDIA's differentiable ray-tracing module for communication-focused channel modelling (path coefficients, OFDM channel matrices, antenna patterns).
- **Mitsuba 3** — a physically-based renderer used as an independent geometric ray-tracer to cross-validate propagation paths.

The current notebook (`Otaniemi.ipynb`) builds the environment by converting OpenStreetMap data (`otaniemi.osm`) into a Mitsuba scene (`OtaniemiScene.xml`) via `osm_to_mitsuba.py`. It configures a rooftop base station, two pedestrian UEs, and computes multipath components, delays, angles, and channel statistics. The workflow includes dataset export as HDF5/CSV, quantitative Sionna-vs-Mitsuba comparison, and visualizations for CIR/AoA/AoD.

Results and methodology are documented in the IEEE-format paper (`Project.tex` + `sections/`), with evaluation focused on the Otaniemi campus scenario.

---

## Files and Folders

| Path | Description |
|------|-------------|
| `Otaniemi.ipynb` | Monolithic notebook — full pipeline from scene loading → ray tracing → dataset generation → localization → channel charting (kept as reference) |
| `01_generate_dataset.py` | Run Sionna RT + Mitsuba ray tracing and export HDF5/CSV datasets |
| `02_rt_comparison.py` | Load pre-generated HDF5 datasets and produce side-by-side Sionna vs Mitsuba comparisons |
| `03_localization.py` | Load fingerprint dataset and evaluate wKNN, NN Regression, and NN Classification localization methods |
| `04_channel_charting.py` | Load fingerprint dataset and run PCA, t-SNE, Autoencoder, and UMAP channel charting |
| `run-analysis.sh` | Shell script to run the full analysis pipeline; accepts a scene name as argument |
| `config.py` | Scene configuration — `get_scene_config(scene_name)` returns all constants for a scene; change one string to switch scenes |
| `rt_utils.py` | Mitsuba ray-tracer helpers: `load_mitsuba_scene`, `trace_paths`, `compute_aoa_mitsuba`, `compute_aod_mitsuba`, path statistics |
| `features.py` | CSI fingerprint extraction: OFDM dB-magnitude + group-delay, TDoA, and AoA features with HDF5 caching |
| `localization.py` | Localization methods: `run_wknn` (IDW), `run_nn_regression`, `run_nn_classification` |
| `channel_charting.py` | Dimensionality reduction: `run_pca`, `run_tsne`, `run_autoencoder`, `run_umap` with TW/CT/KS metrics |
| `generate_report.py` | Generates a PDF summary report from analysis results |
| `report_utils.py` | Helper utilities for report generation |
| `osm_to_mitsuba.py` | Python converter: `otaniemi.osm` → `OtaniemiScene.xml` for Sionna/Mitsuba workflows |
| `glb_to_mitsuba.py` | Python converter: `.glb` → Mitsuba XML + one `.ply` per mesh primitive |
| `sionna_env.yml` | Conda environment specification (Python + Sionna + Mitsuba) |
| `OtaniemiScene/` | Scene folder: `otaniemi.osm`, `OtaniemiScene.xml`, `otaniemimat.glb`, `ply/`, `scene_config.json` |
| `Otaniemi_small/` | Scene folder: `Otaniemi_small.xml`, `small_otaniemi.glb`, `ply/`, `scene_config.json` |
| `OtaniemiScene_100m/` | Scene folder: `OtaniemiScene_100m.xml`, `ply/`, `scene_config.json` |
| `GarageScene/` | Scene folder: `GarageScene.xml`, `ply/`, `scene_config.json` |
| `doc/` | LaTeX report: `Project.tex`, `sections/`, `pictures/`, `references.bib`, `Makefile` |


## How to reproduce

# Environment 

## WSL2 - install
Any Linux system. I used Windows Subsystem Linux 2 (WSL2) on my Windows10 laptop. 
Instructionsto install WSL2: 
 https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/

## Conda - install

- Download the Installer: Download the latest Miniconda installer for Linux.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
```
- Run the Installer: Run the shell script to start the installation.
```bash
bash ~/miniconda.sh -b -p $HOME/miniconda
```

- Initialize Conda: Run the following to initialize conda for your shell.
```bash
HOME/miniconda/bin/conda init bash
```

- Refresh Terminal: Close and open your terminal or run:
```bash
source ~/.bashrc
```

## Set the SSH-Key
https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Clone the code
https://docs.github.com/en/get-started/git-basics/about-remote-repositories
```bash
git clone git@github.com:jkarppi/Machine-Learning-for-Wireless-Comunications-E7340.git
git switch APP
```
## Conda - env
https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
```bash
conda env create -f sionna_env.yml
```
## Install VS Code in Windows
https://code.visualstudio.com/docs/remote/wsl

## Create a project to your VS code
- In WSL2 CLI go to your project folder. 

```bash
code .
```

- Opens the VS Code in your project.  
- Install your favorite plugins to VS Code, Python, Jupyter, Matlab, Draw-io, Gnuplot, LaTex, Markdown, PPT viewer, H5Web, GitHub, etc. 
  

## Run the analysis

1. Install dependencies and activate the environment:
   ```bash
   conda env create -f sionna_env.yml
   conda activate sionna_env2
   ```
2. Run the full analysis pipeline with `run-analysis.sh`:
   ```bash
   ./run-analysis.sh <scene_name>
   ```
   Available scenes: `Otaniemi_small`, `OtaniemiScene`, `OtaniemiScene_100m`

   The script runs the following steps in order:
   - `01_generate_dataset.py` — Sionna RT + Mitsuba ray tracing, exports HDF5/CSV datasets
   - `02_rt_comparison.py` — Sionna vs Mitsuba side-by-side comparison plots
   - `03_localization.py` — wKNN, NN Regression, and NN Classification localization evaluation
   - `04_channel_charting.py` — PCA, t-SNE, Autoencoder, and UMAP channel charting

3. Key outputs (saved to `<scene_name>-results-<date>-<time>/`):
   - `sionna_dataset.h5`, `mitsuba_dataset.h5`
   - `sionna_paths.csv`, `mitsuba_paths.csv`, `sionna_frequency_domain.csv`
   - Figures and plots
   - `logs.txt` — timestamped log of each step
   - A PDF report generated by `generate_report.py`
   - A `.tgz` tarball of all results archived one level up

   For usage details:
   ```bash
   ./run-analysis.sh --help
   ```

