#!/bin/sh
## Simple application to run: 
## 1. 01_generate_dataset.py
## 2. 02_rt_comparison.py
## 3. 03_localization.py
## 4. 04_channel_charting.py

FILENAME01=01_generate_dataset.py
FILENAME02=02_rt_comparison.py
FILENAME03=03_localization.py
FILENAME04=04_channel_charting.py
SCENE_PATH=Otaniemi_small

print_help() {
  echo ""
  echo "Usage:"
  echo "  ./run-analysis.sh [OPTIONS] <scene_name>"
  echo ""
  echo "Arguments:"
  echo "  <scene_name>   Name of the scene to analyse."
  echo "                 Available scenes: Otaniemi_small, OtaniemiScene, OtaniemiScene_100m"
  echo "                 Default: Otaniemi_small"
  echo ""
  echo "Options:"
  echo "  -h, --help     Show this help message and exit."
  echo ""
  echo "Conda environment:"
  echo "  The analysis requires the 'sionna_env2' conda environment."
  echo "  To create it (first time only):"
  echo "    conda env create -f sionna_env.yml"
  echo ""
  echo "  To activate it before running the script:"
  echo "    conda activate sionna_env2"
  echo ""
  echo "Example:"
  echo "  conda activate sionna_env2"
  echo "  ./run-analysis.sh Otaniemi_small"
  echo ""
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]
then
  print_help
  exit 0
elif [ $# -eq 0 ]
then
  echo "./run-analysis.sh <scene_name>"
  echo "Run './run-analysis.sh --help' for more information."
  exit 1
elif [ $# -eq 1 ]
then
  SCENE_PATH=$1
fi

DATENOW="`date +%F`"
TIMENOW="`date +%H%M%S`"
START_SECONDS=$(date +%s)
RESULTDIR=$SCENE_PATH-results-$DATENOW-$TIMENOW

mkdir $RESULTDIR
echo "$SCENE_PATH Test started: `date +%H:%M:%S`"
echo "Saving logs and results to the folder $RESULTDIR"


LOGFILENAME=$RESULTDIR/logs.txt

echo "$DATENOW $TIMENOW" > $LOGFILENAME

# 1. Run 01_generate_dataset.py
echo "`date +%H%M%S`: $FILENAME01  started" > $LOGFILENAME
python3 $FILENAME01 $SCENE_PATH $RESULTDIR >> $LOGFILENAME 2>&1
echo "`date +%H%M%S`: $FILENAME01  ended" >> $LOGFILENAME

# 2. Run 02_rt_comparison.py
echo "`date +%H%M%S`: $FILENAME02  started" >> $LOGFILENAME
python3 $FILENAME02 $SCENE_PATH $RESULTDIR >> $LOGFILENAME 2>&1
echo "`date +%H%M%S`: $FILENAME02  ended" >> $LOGFILENAME

# 3. Run 03_localization.py
echo "`date +%H%M%S`: $FILENAME03  started" >> $LOGFILENAME
python3 $FILENAME03 $SCENE_PATH $RESULTDIR >> $LOGFILENAME 2>&1
echo "`date +%H%M%S`: $FILENAME03  ended" >> $LOGFILENAME

# 4. Run 04_channel_charting.py
echo "`date +%H%M%S`: $FILENAME04  started" >> $LOGFILENAME
python3 $FILENAME04 $SCENE_PATH $RESULTDIR >> $LOGFILENAME 2>&1
echo "`date +%H%M%S`: $FILENAME04  ended" >> $LOGFILENAME



END_SECONDS=$(date +%s)
ELAPSED=$((END_SECONDS - START_SECONDS))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))
echo "$SCENE_PATH all done `date +%H:%M:%S` see results from $RESULTDIR/"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s)"

echo "generating report as an pdf file..."
## Generate a PDF report from the results
python3 generate_report.py $SCENE_PATH $RESULTDIR >> $LOGFILENAME 2>&1

echo "generating ablation summary as markdown and pdf..."
## Generate ablation results summary as Markdown and PDF
if [ -f "$RESULTDIR"/*/localization_summary.json ]; then
    ABLATION_MD="$RESULTDIR/ablation_summary.md"
    ABLATION_PDF="$RESULTDIR/ablation_summary.pdf"
    python3 print_ablation_results.py "$RESULTDIR" --markdown "$ABLATION_MD" --pdf "$ABLATION_PDF" >> $LOGFILENAME 2>&1
else
    echo "Note: No ablation results found, skipping ablation summary generation." >> $LOGFILENAME
fi

echo "generating tarball of the results..."
## Create a tarball of the results
tar -czRf ../$SCENE_PATH-analysis-$DATENOW-$TIMENOW.tgz --dereference $RESULTDIR/* 
