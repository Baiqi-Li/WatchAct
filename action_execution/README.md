# WatchAct — Custom LIBERO Task Execution

Run WatchAct action plans in the LIBERO simulator. You provide a plan (a list of
step-by-step instructions, or a single instruction), a policy server that maps
observations to actions, and this executor drives the simulation and reports
success rate + goal-completion ratio with replay videos.

This directory is **self-contained**: it uses the bundled LIBERO checkout under
`../third_party/LIBERO`, registers WatchAct's custom objects from
`../custom_assets/`, and does not depend on `openpi`.

## Contents

| File | Purpose |
|---|---|
| `action_steps.json` | Where you paste your generated plan (the `sequential` steps). |
| `build_execution_instructions.py` | Turns `action_steps.json` into an execution config, pulling `bddl` + `interest_objects_final_states` from `../data`. |
| `run_libero_execution.py` | The executor: builds a LIBERO env per task, rolls out a policy, scores, saves videos. |
| `custom_objects.py` | Registers custom objects (apple, banana, mango, pear, tennis_ball, cheese, …) into LIBERO at runtime. |
| `custom_predicates.py` | Registers the custom goal predicates `Right` / `Horizontal` into LIBERO at runtime. |
| `policy_client.py` | Minimal WebSocket policy client + the observation/action protocol. |
| `generate_init_states.py` | (Optional) pre-sample reproducible initial states. |
| `_libero_utils.py` | Vendored image/quat helpers + BDDL path resolution. |

## Setup

```bash
# 1) LIBERO is a git submodule at ../third_party/LIBERO (used via sys.path
#    automatically). Fetch it and apply the one-line patch that removes an
#    upstream pdb.set_trace() debug breakpoint (it would hang headless runs):
git submodule update --init third_party/LIBERO      # from the repo root
git -C third_party/LIBERO apply ../libero_remove_pdb.patch
# 2) Install LIBERO's dependencies + a working robosuite/mujoco once:
pip install -r ../third_party/LIBERO/requirements.txt
# 3) This directory's extra deps:
pip install -r requirements.txt
```

Headless rendering uses EGL; the scripts set `MUJOCO_GL=egl` by default. Tested
with the `libero` conda env.

## Workflow

```bash
# 1) Paste your plan into action_steps.json (mainly the "sequential" prompts).
#    Format: see action_steps.json / execution_instruction_example.yaml.

# 2) Build the execution config (looks up bddl + goal states from ../data):
python build_execution_instructions.py            # -> execution_instructions.yaml

# 3) Start your policy server (see "Policy interface" below), then run:
python run_libero_execution.py \
    --config execution_instructions.yaml \
    --exp-id sequential \                          # or: single_instruction
    --host 127.0.0.1 --port 8001 \
    --num-trials 10
```

Results and videos are written under `../outputs/libero_execution/`
(`results.json` + `<task_id>/<exp_id>/ep*.mp4`).

`--exp-id single_instruction` sends one instruction for the whole episode;
`--exp-id sequential` sends the plan step-by-step, switching prompts every
`steps_per_prompt` steps.

## Policy interface

The executor connects to a **policy server** over WebSocket — it never imports
your model, so any framework works as long as the server speaks the protocol in
[`policy_client.py`](policy_client.py):

- The executor sends an observation dict each step:
  `observation/image` (uint8 HxWx3), `observation/wrist_image` (uint8 HxWx3),
  `observation/state` (float `[eef_pos(3), axis_angle(3), gripper(2)]`),
  and `prompt` (str).
- The server replies `{"actions": <float array (T, 7)>}` with `T >=
  --replan-steps`.

This is the same wire protocol as openpi's `WebsocketPolicyServer`, so an
existing openpi LIBERO server works unchanged. To use a different model, serve
it behind a WebSocket endpoint that follows the same protocol.

## Custom objects

WatchAct BDDLs use object categories not in stock LIBERO. `custom_objects.py`
registers them at runtime from `../custom_assets/<folder>/model_tier1.xml` via
LIBERO's public `OBJECTS_DICT` — **without editing the LIBERO source**. The
executor calls this automatically and then preflights every BDDL's object
categories, failing early with a clear message if any category has neither a
LIBERO class nor a `custom_assets` folder.

Currently mapped (see `CATEGORY_TO_FOLDER`): `apple, banana, mango, pear,
tennis_ball, cheese (→ cheese_can), cheese_can, band_aid, coke, football,
lemon`. To add another object, drop a LIBERO-format `model_tier1.xml` (+ meshes)
under `../custom_assets/<name>/` and add a `CATEGORY_TO_FOLDER` entry (or, if the
BDDL category matches the folder name, it auto-registers during preflight).

## Init states (optional)

By default each episode starts from `env.reset()` (objects sampled within the
BDDL regions). For reproducible starts, pre-generate them:

```bash
python generate_init_states.py --config execution_instructions.yaml \
    --out-dir ../outputs/init_files --num-states 20
# then pass them to the executor:
python run_libero_execution.py --config execution_instructions.yaml \
    --exp-id sequential --init-states-dir ../outputs/init_files ...
```

## Custom goal predicates (Right / Horizontal)

Some BDDLs (mainly `Temporal_Sort`) use two goal predicates not in stock
LIBERO. `custom_predicates.py` registers them at runtime (no LIBERO source
edit); the executor enables them automatically:

- `(Right A B)` — A is to the right of B. On the kitchen table the world y-axis
  runs left(+) → right(−), so this is `A.y < B.y`.
- `(Horizontal A B [C ...])` — the objects lie on one left-right line, i.e. their
  front/back position (world x) is nearly equal. True when the spread of their
  x-coordinates is `<= 0.05 m`. Accepts 2+ objects.

The `0.05 m` tolerance fits the table grid: rows (back/middle/front) are ~0.15 m
apart in x, so 0.05 absorbs within-row jitter without merging adjacent rows.
Adjust via `HORIZONTAL_X_TOL` in `custom_predicates.py`.

LIBERO's built-in predicate dispatch only handles unary/binary predicates, so
`custom_predicates.py` also patches the kitchen problem's `_eval_predicate` with
an equivalent n-ary version (needed for 3-object `Horizontal`).
