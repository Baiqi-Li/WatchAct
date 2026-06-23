# WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation

🌐 [Project Page](https://baiqi-li.github.io/watchact_page/) &nbsp;|&nbsp; 🤗 [Dataset](https://huggingface.co/datasets/BaiqiL/WatchAct) &nbsp;|&nbsp; 💻 [Code](https://github.com/Baiqi-Li/WatchAct)

WatchAct is a behavior-grounded benchmark for robotic manipulation, where the robot reasons over observed human behavior and a language instruction to perform the corresponding task.

## 1. Download the data

```bash
huggingface-cli download WatchAct/data --repo-type dataset \
  --local-dir vlm_planning_github/data
```

This produces:

```
vlm_planning_github/data/
├── data/<task>.jsonl        # one file per task (the evaluation rows)
├── meta_data/<task>.json    # ground-truth scene / goal definitions
├── videos/<task>/{front,left,left_back}/*.MP4
└── bddl_files/<task>/       # LIBERO task files (not needed for scoring)
```

A `<task>` is the jsonl filename stem. The 14 tasks, grouped by capability:

| Category | Tasks |
|---|---|
| Procedural Reasoning | `Imitation`, `Reversal`, `Temporal_Sort` |
| Event Grounding | `Fine-Grained_Action`, `Count`, `Ordinal`, `State_Change`, `Moment` |
| Implicit Intent Inference | `Nonverbal_Cue`, `Reference_Disambiguation` |
| Episodic Reasoning | `Restore_Previous_State`, `Task_Continuation`, `Error_Correction`, `Conditional_Execution` |

## 2. Run the pipeline

Single task (the argument is the jsonl filename without `.jsonl`):

```bash
bash vlm_planning_github/run_pipeline.sh Ordinal
```

All tasks found under `data/`:

```bash
bash vlm_planning_github/run_pipeline.sh
```

Useful overrides (environment variables):

```bash
MODEL_ID=gpt-5.4 bash vlm_planning_github/run_pipeline.sh Ordinal
FORCE=1          bash vlm_planning_github/run_pipeline.sh   # re-run completed files
NUM_WORKERS=12   bash vlm_planning_github/run_pipeline.sh
```

## 3. Outputs

Everything is written under `vlm_planning_github/outputs/`:

```
outputs/
├── action_plan/<task>/<model_id>_<task>.json          # stage 1
├── translation/<task>/...                              # stage 2
├── action_execution/<task>/simulation_results_*.json  # stage 3
├── scores/scores_<task>.json                           # stage 4: per-task scores
└── summary.json                                        # stage 4: final aggregated metrics
```

**Where the final scores are:**

- **`outputs/summary.json`** — the headline result file. Count-weighted
  `Plan Success Rate` and `Progress Rate`, ordered as `overall` → per
  high-level `category` → per-task `subcategory`. Tasks not scored in the
  current run appear with `null` rates and are listed under
  `missing_subcategories`, so partial runs are still valid.
- **`outputs/scores/scores_<task>.json`** — per-task scores: the same two
  metrics overall and broken down `by_combo` (camera_perspective ×
  spatial_reference), plus per-row results.

Override the summary location with `--summary-path` on
`scripts/run_action_plan_scoring.py` (default `outputs/summary.json`).

## Layout

```
vlm_planning_github/
├── run_pipeline.sh              # 4-stage orchestrator
├── scripts/                     # stage entry points + check_complete gating
├── vlm_models/                  # self-contained gpt_schema backend
├── config/                      # region / object / reference descriptions
├── data/                        # downloaded from WatchAct/data
└── outputs/                     # generated results
```

## Notes

- Only the `gpt_schema` backend (OpenAI structured output) is shipped. To add
  another model, drop a `BaseVLM` subclass under `vlm_models/` and import it in
  `vlm_models/__init__.py`.
- Stages 1–2 call the model and incur API cost; stages 3–4 are pure-Python and
  free. Re-running reuses completed work unless `FORCE=1` is set.

## Contact

Corresponding author: Baiqi Li (baiqili@cs.unc.edu)
