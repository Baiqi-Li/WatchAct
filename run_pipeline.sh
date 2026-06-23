#!/usr/bin/env bash
# WatchAct minimal evaluation pipeline (gpt_schema / gpt-5.1).
#
# Four stages, run in order for each task:
#   evaluation -> translator -> simulation -> scoring
#
# Data is read from $DATA_ROOT (download WatchAct/data into it first, see
# README.md). All outputs go under $OUTPUT_ROOT.
#
# Usage:
#   bash vlm_planning_github/run_pipeline.sh                 # all tasks in data/
#   bash vlm_planning_github/run_pipeline.sh Ordinal
#   bash vlm_planning_github/run_pipeline.sh Ordinal Temporal_Sort
#   FORCE=1 bash vlm_planning_github/run_pipeline.sh         # ignore check_complete
#
# A task name is the jsonl filename stem under $DATA_ROOT (e.g. the file
# data/Ordinal.jsonl -> task "Ordinal").
#
# Override defaults via env, e.g.:
#   MODEL_ID=gpt-5.4 NUM_WORKERS=8 bash vlm_planning_github/run_pipeline.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-gpt_schema}"
MODEL_ID="${MODEL_ID:-gpt-5.1}"
DATASET_ROOT="${DATA_ROOT:-$HERE/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HERE/outputs}"
CONFIG_ROOT="${CONFIG_ROOT:-$HERE/config}"
SCRIPTS="$HERE/scripts"
CHECK="$SCRIPTS/check_complete.py"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: dataset dir not found: $DATASET_ROOT" >&2
  echo "Download it first:  huggingface-cli download WatchAct/data --repo-type dataset --local-dir $DATASET_ROOT" >&2
  exit 1
fi

# WatchAct layout: jsonl rows live under $DATASET_ROOT/data, with meta_data/
# and videos/ as siblings. Fall back to a flat layout if there is no data/
# subdir, so both arrangements work.
if ls "$DATASET_ROOT"/data/*.jsonl >/dev/null 2>&1; then
  JSONL_DIR="$DATASET_ROOT/data"
else
  JSONL_DIR="$DATASET_ROOT"
fi
META_DIR="$DATASET_ROOT/meta_data"
VIDEO_ROOT="$DATASET_ROOT/videos"

# Task list: positional args, else every *.jsonl stem under JSONL_DIR.
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  mapfile -t TASKS < <(cd "$JSONL_DIR" && ls *.jsonl 2>/dev/null | sed 's/\.jsonl$//' | sort)
fi
if [ ${#TASKS[@]} -eq 0 ]; then
  echo "ERROR: no *.jsonl datasets found under $JSONL_DIR" >&2
  exit 1
fi

echo "############################################################"
echo "# model      : $MODEL / $MODEL_ID"
echo "# jsonl dir  : $JSONL_DIR"
echo "# meta dir   : $META_DIR"
echo "# video root : $VIDEO_ROOT"
echo "# output root: $OUTPUT_ROOT"
echo "# tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "############################################################"

# ---------------------------------------------------------------------------
# [1/4] Evaluation: VLM watches each video -> action_plan
# ---------------------------------------------------------------------------
echo; echo "===== [1/4] evaluation ====="
python "$SCRIPTS/run_vlm_evaluation.py" \
  --skip-existing \
  --reuse-ignore-fields language_instruction \
  --model "$MODEL" \
  --model-id "$MODEL_ID" \
  --num-workers "$NUM_WORKERS" \
  --save-every "$SAVE_EVERY" \
  --parallel-group-by original_id \
  --data-dir "$JSONL_DIR" \
  --video-root "$VIDEO_ROOT" \
  --output-dir "$OUTPUT_ROOT/action_plan" \
  --datasets "${TASKS[@]}"

# ---------------------------------------------------------------------------
# [2/4] Translator: align VLM action_plan to canonical objects/regions
# ---------------------------------------------------------------------------
echo; echo "===== [2/4] translator ====="
for task in "${TASKS[@]}"; do
  mkdir -p "$OUTPUT_ROOT/translation/$task"
  for f in "$OUTPUT_ROOT/action_plan/$task"/*.json; do
    [ -f "$f" ] || continue
    fname="$(basename "$f")"
    out="$OUTPUT_ROOT/translation/$task/$fname"
    if [ -z "${FORCE:-}" ] && python "$CHECK" --stage translator --input "$f" --output "$out"; then
      echo "skip (complete): $task/$fname"; continue
    fi
    echo "translate: $task/$fname"
    python "$SCRIPTS/run_translator_alignment.py" \
      --skip-existing \
      --model "$MODEL" \
      --model-id "$MODEL_ID" \
      --data-path "$META_DIR/$task.json" \
      --results-path "$f" \
      --output-path "$out" \
      --region-descriptions-path "$CONFIG_ROOT/ALL_REGION_DESCRIPTONS.json" \
      --object-descriptions-path "$CONFIG_ROOT/OBJECT_DESCRIPTIONS.json" \
      --human-camera-reference-path "$CONFIG_ROOT/human2camera_reference.json" \
      --num-workers "$NUM_WORKERS" \
      --save-every "$SAVE_EVERY"
  done
done

# ---------------------------------------------------------------------------
# [3/4] Simulation: symbolically execute the aligned plan
# ---------------------------------------------------------------------------
echo; echo "===== [3/4] simulation ====="
for task in "${TASKS[@]}"; do
  outdir="$OUTPUT_ROOT/action_execution/$task"
  mkdir -p "$outdir"
  for f in "$OUTPUT_ROOT/translation/$task"/*.json; do
    [ -f "$f" ] || continue
    fname="$(basename "$f")"
    out="$outdir/simulation_results_$fname"
    if [ -z "${FORCE:-}" ] && python "$CHECK" --stage simulation --input "$f" --output "$out"; then
      echo "skip (complete): $task/$fname"; continue
    fi
    echo "simulate: $task/$fname"
    python "$SCRIPTS/run_action_plan_simulation.py" \
      --translator-results-path "$f" \
      --output-path "$out"
  done
done

# ---------------------------------------------------------------------------
# [4/4] Scoring: compare simulated final state vs goal
# ---------------------------------------------------------------------------
echo; echo "===== [4/4] scoring ====="
SCORE_DIR="$OUTPUT_ROOT/scores"
mkdir -p "$SCORE_DIR"
for task in "${TASKS[@]}"; do
  for f in "$OUTPUT_ROOT/action_execution/$task"/*.json; do
    [ -f "$f" ] || continue
    fname="$(basename "$f")"
    if [ -z "${FORCE:-}" ] && python "$CHECK" --stage scoring --input "$f" --output-dir "$SCORE_DIR"; then
      echo "skip (complete): $task/$fname"; continue
    fi
    echo "score: $task/$fname"
    python "$SCRIPTS/run_action_plan_scoring.py" \
      --input-path "$f" \
      --output-dir "$SCORE_DIR"
  done
done

# Aggregate across every scored task into one count-weighted summary
# (overall / high-level category / per-task subcategory). This directory pass
# scans all simulation results under action_execution, so the summary always
# reflects every task present, not just the last one scored.
echo; echo "===== summary ====="
SUMMARY_PATH="$OUTPUT_ROOT/summary.json"
python "$SCRIPTS/run_action_plan_scoring.py" \
  --input-path "$OUTPUT_ROOT/action_execution" \
  --output-dir "$SCORE_DIR" \
  --summary-path "$SUMMARY_PATH"

echo; echo "############################################################"
echo "# Pipeline complete."
echo "#   per-task scores -> $SCORE_DIR"
echo "#   summary         -> $SUMMARY_PATH"
echo "############################################################"
