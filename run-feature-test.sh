#!/bin/bash
# =============================================================================
# run-feature-test.sh — Feature ablation study for fingerprint localization
#
# Tests 23 feature combinations to find which CSI feature groups contribute
# most to localization accuracy.
#
# Strategy
# --------
#   Phase 1 – Ablation   : start from ALL, remove ONE feature group at a time
#   Phase 2 – Solo       : enable ONE feature group at a time
#   Phase 3 – Groups     : logical combinations (geometry, radio, etc.)
#
# Key design: the base cache (all features ON) is generated ONCE.  Every
# subsequent combination loads that cache and masks columns — no extra
# ray-tracing is needed per combination.
#
# Usage
# -----
#   conda activate sionna_env2
#   ./run-feature-test.sh [scene_name]
#
#   scene_name  – default: Otaniemi_small
#
# Outputs
# -------
#   <scene>-ablation-<date>/          root result directory
#     ALL/localization_summary.json   one JSON per combination
#     No_OFDM/localization_summary.json
#     ...
#     ablation_results.csv            flat CSV of all results
#     ablation_results_table.txt      human-readable comparison table
#     logs.txt                        timestamped stdout log
#
# After the run, the original features_config.json is restored.
# =============================================================================

set -e   # exit on first error

SCENE_NAME="${1:-Otaniemi_small}"

DATENOW="$(date +%F)"
TIMENOW="$(date +%H%M%S)"
START_SEC="$(date +%s)"
RESULTDIR="${SCENE_NAME}-ablation-${DATENOW}-${TIMENOW}"

mkdir -p "$RESULTDIR"
LOGFILE="$RESULTDIR/logs.txt"

echo "==========================================================" | tee -a "$LOGFILE"
echo "  Feature ablation study" | tee -a "$LOGFILE"
echo "  Scene     : $SCENE_NAME" | tee -a "$LOGFILE"
echo "  Started   : $(date +%H:%M:%S)" | tee -a "$LOGFILE"
echo "  Result dir: $RESULTDIR" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"

# ── Helpers ───────────────────────────────────────────────────────────────────

# Backup the original features_config.json so we can restore it at the end.
ORIG_CONFIG="${SCENE_NAME}/features_config.json"
BACKUP_CONFIG="${SCENE_NAME}/features_config.json.bak"
cp "$ORIG_CONFIG" "$BACKUP_CONFIG"
echo "Backed up $ORIG_CONFIG → $BACKUP_CONFIG" | tee -a "$LOGFILE"

# restore_config: called on EXIT (normal or error) to put the original back.
restore_config() {
    if [ -f "$BACKUP_CONFIG" ]; then
        cp "$BACKUP_CONFIG" "$ORIG_CONFIG"
        rm -f "$BACKUP_CONFIG"
        echo "Restored original features_config.json." | tee -a "$LOGFILE"
    fi
}
trap restore_config EXIT

# run_combo <combo_name> <feature_list>
#   Sets features_config.json, runs feature_ablation.py, logs timing.
run_combo() {
    local NAME="$1"
    local FEATS="$2"           # comma-separated list, or "ALL"
    local OUTDIR="$RESULTDIR/$NAME"

    echo "" | tee -a "$LOGFILE"
    echo "── $NAME ───────────────────────────────────────────────" | tee -a "$LOGFILE"
    echo "   Features: $FEATS" | tee -a "$LOGFILE"
    echo "   Started : $(date +%H:%M:%S)" | tee -a "$LOGFILE"

    # Write features_config.json for this combination.
    # nn_classification is disabled by default (slow for large grids).
    python3 write_features_config.py "$SCENE_NAME" "$FEATS" \
        --methods "wknn,nn_regression,cnn_regression" 2>&1 | tee -a "$LOGFILE"

    # Run the masked localization.
    local t0="$(date +%s)"
    python3 feature_ablation.py "$SCENE_NAME" "$OUTDIR" "$NAME" 2>&1 | tee -a "$LOGFILE"
    local t1="$(date +%s)"
    local elapsed=$((t1 - t0))
    echo "   Finished: $(date +%H:%M:%S)  (${elapsed}s)" | tee -a "$LOGFILE"
}

# =============================================================================
# STEP 0 — Generate (or verify) the full-feature base cache
#
# This is the only time the Sionna PathSolver is invoked.  All subsequent
# combinations load this cache and apply a column mask.
#
# If fingerprint_rt_dataset.h5 already exists and has the expected number of
# columns, this step is skipped automatically by 03_localization.py.
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "STEP 0: Generating full-feature base cache …" | tee -a "$LOGFILE"
python3 write_features_config.py "$SCENE_NAME" "ALL" \
    --methods "wknn,nn_regression,cnn_regression" 2>&1 | tee -a "$LOGFILE"

python3 03_localization.py "$SCENE_NAME" "$RESULTDIR/ALL" 2>&1 | tee -a "$LOGFILE"
echo "STEP 0 complete." | tee -a "$LOGFILE"

# =============================================================================
# PHASE 1 — ABLATION  (all features ON, remove one at a time)
#
# Purpose: identify which feature group hurts the most when removed.
#          A large MAE increase → that group is critical.
#          A small change or improvement → it may be safe to drop.
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "PHASE 1 — Ablation (one feature removed at a time)" | tee -a "$LOGFILE"

NO_OFDM="tdoa,aoa,rss,path_loss,delay,cov_eigenvalues,reached_flags"
NO_TDOA="ofdm_mag_gd,aoa,rss,path_loss,delay,cov_eigenvalues,reached_flags"
NO_AOA="ofdm_mag_gd,tdoa,rss,path_loss,delay,cov_eigenvalues,reached_flags"
NO_RSS="ofdm_mag_gd,tdoa,aoa,path_loss,delay,cov_eigenvalues,reached_flags"
NO_PL="ofdm_mag_gd,tdoa,aoa,rss,delay,cov_eigenvalues,reached_flags"
NO_DELAY="ofdm_mag_gd,tdoa,aoa,rss,path_loss,cov_eigenvalues,reached_flags"
NO_COV="ofdm_mag_gd,tdoa,aoa,rss,path_loss,delay,reached_flags"
NO_REACHED="ofdm_mag_gd,tdoa,aoa,rss,path_loss,delay,cov_eigenvalues"

run_combo "No_OFDM"    "$NO_OFDM"
run_combo "No_TDoA"    "$NO_TDOA"
run_combo "No_AoA"     "$NO_AOA"
run_combo "No_RSS"     "$NO_RSS"
run_combo "No_PathLoss" "$NO_PL"
run_combo "No_Delay"   "$NO_DELAY"
run_combo "No_CovEig"  "$NO_COV"
run_combo "No_Reached" "$NO_REACHED"

# =============================================================================
# PHASE 2 — SOLO  (only one feature group at a time)
#
# Purpose: understand the standalone contribution of each feature group.
#          High solo accuracy → that group encodes strong positional info.
#          Low solo accuracy  → the group is only useful in combination.
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "PHASE 2 — Solo (one feature group at a time)" | tee -a "$LOGFILE"

run_combo "Solo_OFDM"    "ofdm_mag_gd"
run_combo "Solo_TDoA"    "tdoa"
run_combo "Solo_AoA"     "aoa"
run_combo "Solo_RSS"     "rss"
run_combo "Solo_PathLoss" "path_loss"
run_combo "Solo_Delay"   "delay"
run_combo "Solo_CovEig"  "cov_eigenvalues"
run_combo "Solo_Reached" "reached_flags"

# =============================================================================
# PHASE 3 — GROUPS  (logical feature combinations)
#
# Purpose: test practical subsets that make physical sense.
#
#   Geometry   — range + angle only (TDoA, AoA, delay)
#   Radio      — signal strength + quality (RSS, PL, reached)
#   Spectral   — full OFDM channel only
#   Light      — small fast set: TDoA + AoA + RSS + reached (no heavy cov/OFDM)
#   No_OFDM_Light — everything except OFDM and cov_eigenvalues
#   OFDM_Radio — OFDM + signal quality (OFDM + RSS + PL)
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "PHASE 3 — Logical groups" | tee -a "$LOGFILE"

run_combo "Geometry"     "tdoa,aoa,delay"
run_combo "Radio"        "rss,path_loss,reached_flags"
run_combo "Spectral"     "ofdm_mag_gd"
run_combo "Light"        "tdoa,aoa,rss,reached_flags"
run_combo "No_OFDM_Light" "tdoa,aoa,rss,path_loss,delay,reached_flags"
run_combo "OFDM_Radio"   "ofdm_mag_gd,rss,path_loss"

# =============================================================================
# RESULTS TABLE
# =============================================================================
echo "" | tee -a "$LOGFILE"
echo "Generating comparison table …" | tee -a "$LOGFILE"

# Sort by wKNN MAE and print to console + save to text file.
python3 print_ablation_results.py "$RESULTDIR" \
    --sort-by wknn_mae \
    --csv "$RESULTDIR/ablation_results.csv" \
    2>&1 | tee "$RESULTDIR/ablation_results_table.txt" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "Also printing sorted by best_mae …" | tee -a "$LOGFILE"
python3 print_ablation_results.py "$RESULTDIR" --sort-by best_mae 2>&1 | tee -a "$LOGFILE"

# =============================================================================
# DONE
# =============================================================================
END_SEC="$(date +%s)"
ELAPSED=$((END_SEC - START_SEC))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
echo "  All done  $(date +%H:%M:%S)" | tee -a "$LOGFILE"
echo "  Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s" | tee -a "$LOGFILE"
echo "  Results   : $RESULTDIR/" | tee -a "$LOGFILE"
echo "  CSV table : $RESULTDIR/ablation_results.csv" | tee -a "$LOGFILE"
echo "  Text table: $RESULTDIR/ablation_results_table.txt" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
