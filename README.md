# WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation

🌐 [Project Page](https://baiqi-li.github.io/watchact_page/) &nbsp;|&nbsp; 🤗 [Dataset](https://huggingface.co/datasets/BaiqiL/WatchAct) &nbsp;|&nbsp; 💻 [Code](https://github.com/Baiqi-Li/WatchAct)

WatchAct is a behavior-grounded benchmark for robotic manipulation: the robot
must reason over an observed human video **and** a language instruction to
perform the intended task.

The repo has two independent parts:

1. **VLM Planning Evaluation** (`video_planning/`) — score a VLM planner. The
   VLM watches each video and produces an action plan; the pipeline then aligns
   the plan, simulates it symbolically, and compares the result against the
   ground-truth goal, writing one concise result file.
2. **Execution on LIBERO** (`action_execution/`) — take generated plans and roll them out
   in the LIBERO simulator on WatchAct's custom BDDL tasks.

## Setup

Download the dataset into `data/`:

```bash
hf download BaiqiL/WatchAct --repo-type dataset --local-dir data
```

```
data/
├── data/<task>.jsonl        # evaluation rows (one file per task)
├── meta_data/<task>.json    # ground-truth scenes / goals
├── videos/<task>/{front,side,oblique}/*.MP4
└── bddl_files/<task>/       # LIBERO task files (used by execution only)
```

The 14 tasks (`<task>` = jsonl filename stem), grouped by capability:

| Category | Tasks |
|---|---|
| Procedural Reasoning | `Imitation`, `Reversal`, `Temporal_Sort` |
| Event Grounding | `Fine-Grained_Action`, `Count`, `Ordinal`, `State_Change`, `Moment` |
| Implicit Intent Inference | `Nonverbal_Cue`, `Reference_Disambiguation` |
| Episodic Reasoning | `Restore_Previous_State`, `Task_Continuation`, `Error_Correction`, `Conditional_Execution` |

Dependencies: `pip install -r video_planning/requirements.txt` for evaluation, and
see [action_execution/README.md](action_execution/README.md) for execution.

The evaluation pipeline calls a VLM API, so set the matching key (as an
environment variable or in a `.env` file). For example, with Qwen
(get a key at [qwen.ai/apiplatform](https://qwen.ai/apiplatform)):

```bash
export DASHSCOPE_API_KEY=sk-...   # Qwen backend (also: pip install dashscope)
export OPENAI_API_KEY=sk-...      # GPT backend (default)
```

## 1. Evaluation

Run the four-stage pipeline over one, several, or all tasks:

```bash
bash video_planning/run_pipeline.sh                    # all tasks
bash video_planning/run_pipeline.sh Ordinal Temporal_Sort
MODEL_ID=gpt-5.4 bash video_planning/run_pipeline.sh   # override the planner model
MODEL=qwen MODEL_ID=qwen3-vl-235b-a22b-instruct bash video_planning/run_pipeline.sh
```

Stages (each task flows through them in order):

| # | Stage | What it does |
|---|---|---|
| 1 | `run_vlm_evaluation.py` | VLM watches the video → action plan |
| 2 | `run_translator_alignment.py` | align the plan to canonical objects/regions |
| 3 | `run_action_plan_simulation.py` | symbolically execute the aligned plan |
| 4 | `run_action_plan_scoring.py` | compare the final state against the goal |

**Result:** `outputs/summary.json` — count-weighted `Plan Success Rate` and
`Progress Rate` (overall → capability category → per-task). Every per-stage
artifact lives under `outputs/intermediate/`. Completed work is reused on re-run
unless `FORCE=1` is set.

Two backends ship: `gpt_schema` (OpenAI structured output, the default) and
`qwen_api` (optional; needs `pip install dashscope` and `DASHSCOPE_API_KEY`).
Add a model by dropping a `BaseVLM` subclass in `video_planning/vlm_models/`.

## 2. Execution (LIBERO)

Fill your plan steps into `action_execution/action_steps.json`, then roll them
out in LIBERO on WatchAct's custom BDDL tasks (with the custom objects and the
`Right`/`Horizontal` goal predicates registered at runtime).

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a git submodule;
fetch it and apply the one-line patch that removes an upstream debug breakpoint:

```bash
git submodule update --init third_party/LIBERO
git -C third_party/LIBERO apply ../libero_remove_pdb.patch
cd action_execution
python build_execution_instructions.py            # action_steps.json -> config
python run_libero_execution.py --config execution_instructions.yaml \
    --exp-id sequential --host <policy_host> --port <policy_port>
```

Full details, the policy-server protocol, and custom-object notes are in
[action_execution/README.md](action_execution/README.md).

## Layout

```
WatchAct/
├── video_planning/      # evaluation: pipeline, stage scripts, VLM backends, config
├── action_execution/    # execution: run plans in LIBERO (self-contained)
├── custom_assets/       # custom object models for execution
├── third_party/LIBERO/  # simulator, git submodule (execution only)
├── assets/              # teaser figure
├── data/                # dataset (downloaded; git-ignored)
└── outputs/             # summary.json (result) + intermediate/ (git-ignored)
```

## Citation

```bibtex
@article{li2026watchact,
  title={WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation},
  author={Li, Baiqi and Zhang, Ce and Fang, Yu and Yang, Yue and Li, Shangzhe and Ding, Mingyu and Bertasius, Gedas},
  journal={arXiv preprint arXiv:2606.26443},
  year={2026}
}
```

## Contact

Corresponding author: Baiqi Li (baiqili@cs.unc.edu)
