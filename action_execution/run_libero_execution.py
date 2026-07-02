"""Execute WatchAct action plans in LIBERO.

Reads an execution config (produced by ``build_execution_instructions.py``),
drives a LIBERO simulation for each task via a WebSocket policy server, and
reports success rate + goal-completion ratio, saving replay videos.

The config is YAML with the shape::

    tasks:
    - task_id: imitation_data_0
      bddl: KITCHEN_0_FOLLOWING_SEQUENCES_FOLLOWING_SEQUENCES_0
      interest_objects_final_states:
      - [In, bbq_sauce_1, wooden_tray_1_contain_region]
      experiments:
      - exp_id: single_instruction
        prompts: ["..."]
      - exp_id: sequential
        prompts: ["step 1", "step 2", ...]
        steps_per_prompt: 300

Pick which experiment to run with ``--exp-id single_instruction`` or
``--exp-id sequential``.

Example::

    python run_libero_execution.py \
        --config execution_instructions.yaml \
        --exp-id sequential \
        --host 127.0.0.1 --port 8001 \
        --num-trials 10

WatchAct uses the LIBERO checkout under ``third_party/LIBERO`` (added to
sys.path automatically). Custom objects (apple, mango, ...) are registered at
import time from ``custom_assets/``.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import re
import sys
from typing import Any

import numpy as np

# --- Make WatchAct self-contained: use the bundled LIBERO and headless GL -----
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_WATCHACT_ROOT = _THIS_DIR.parent
_LIBERO_DIR = _WATCHACT_ROOT / "third_party" / "LIBERO"
if _LIBERO_DIR.is_dir() and str(_LIBERO_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBERO_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio  # noqa: E402
import yaml  # noqa: E402

import custom_objects  # noqa: E402  (WatchAct custom object registration)
import custom_predicates  # noqa: E402  (WatchAct Right/Horizontal goal predicates)
from _libero_utils import (  # noqa: E402
    convert_to_uint8,
    quat2axisangle,
    resize_with_pad,
    resolve_bddl_path,
)
from policy_client import WebsocketPolicy  # noqa: E402

logger = logging.getLogger("watchact.libero")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # matches training-data render resolution


# ---------------------------------------------------------------------------
# Interest-predicate evaluation (ported from examples/libero/main.py)
# ---------------------------------------------------------------------------
def _normalize_predicate_arg(arg: Any) -> Any:
    """Table regions are stored bare in the config but registered as
    ``kitchen_table_<region>`` in the BDDL. Object-attached regions
    (``*_contain_region``, ``*_top_region``) are already fully qualified."""
    if isinstance(arg, str) and (arg.startswith("main_") or arg.startswith("fixture_")):
        return f"kitchen_table_{arg}"
    return arg


_PREDICATE_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$")


def _predicate_tokens(pred: Any) -> list:
    """Accept a predicate as a list ``[name, *args]`` or a string ``Name(a, b)``.

    Relational goals (Temporal_Sort) store ``interest_objects_final_states`` as
    strings like ``"Horizontal(cheese_1, milk_1)"``; On/In goals store lists.
    """
    if isinstance(pred, str):
        m = _PREDICATE_RE.match(pred)
        if m:
            return [m.group(1)] + [a.strip() for a in m.group(2).split(",") if a.strip()]
    return list(pred)


def _normalize_predicate(pred: Any) -> list:
    tokens = _predicate_tokens(pred)
    return [str(tokens[0]).lower()] + [_normalize_predicate_arg(a) for a in tokens[1:]]


def _compute_completion_ratio(env, predicates, success):
    """Fraction of interest predicates satisfied; falls back to float(success)."""
    if not predicates:
        return float(success), []
    satisfied = 0
    details = []
    for pred in predicates:
        normalized = _normalize_predicate(pred)
        entry = {"target": pred, "evaluated": normalized, "satisfied": False, "error": None}
        try:
            result = bool(env.env._eval_predicate(normalized))
            entry["satisfied"] = result
            satisfied += int(result)
        except Exception as e:  # one bad predicate must not sink the whole ratio
            entry["error"] = f"{type(e).__name__}: {e}"
            logger.warning("predicate %s eval failed, counting unmatched: %s", pred, e)
        details.append(entry)
    return satisfied / len(predicates), details


# ---------------------------------------------------------------------------
# Prompt scheduling for long-horizon (sequential) tasks
# ---------------------------------------------------------------------------
def _get_current_prompt(prompts, switch_steps, effective_step):
    if len(prompts) == 1:
        return 0, prompts[0]
    idx = 0
    for i, s in enumerate(switch_steps):
        if effective_step >= s:
            idx = i + 1
    return idx, prompts[idx]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _get_libero_env(bddl_file, resolution, seed):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file), camera_heights=resolution, camera_widths=resolution
    )
    env.seed(seed)
    return env


def _run_episode(env, policy, initial_state, prompts, switch_steps, max_steps, args, interest_predicates):
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    replay_images, replay_wrist = [], []
    prev_prompt_idx = 0
    t = 0

    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # Rotate 180 deg to match training-data preprocessing.
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            replay_images.append(img)
            replay_wrist.append(wrist_img)

            img_in = convert_to_uint8(resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_in = convert_to_uint8(resize_with_pad(wrist_img, args.resize_size, args.resize_size))

            effective_step = t - args.num_steps_wait
            prompt_idx, current_prompt = _get_current_prompt(prompts, switch_steps, effective_step)
            if prompt_idx != prev_prompt_idx:
                action_plan.clear()
                logger.info("  [step %d] prompt -> [%d]: %s", effective_step, prompt_idx, current_prompt)
                prev_prompt_idx = prompt_idx

            if not action_plan:
                element = {
                    "observation/image": img_in,
                    "observation/wrist_image": wrist_in,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": current_prompt,
                }
                action_chunk = policy.infer(element)["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise RuntimeError(
                        f"policy returned {len(action_chunk)} actions < replan_steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            if done:
                completion, details = _compute_completion_ratio(env, interest_predicates, True)
                return True, completion, details, replay_images, replay_wrist
            t += 1
        except Exception as e:
            logger.error("episode aborted: %s", e)
            completion, details = _compute_completion_ratio(env, interest_predicates, False)
            return False, completion, details, replay_images, replay_wrist

    completion, details = _compute_completion_ratio(env, interest_predicates, False)
    return False, completion, details, replay_images, replay_wrist


# ---------------------------------------------------------------------------
# Init states
# ---------------------------------------------------------------------------
def _load_init_states(init_states_dir, bddl_stem):
    if init_states_dir is None:
        return None
    path = pathlib.Path(init_states_dir) / f"{bddl_stem}.pruned_init"
    if not path.is_file():
        return None
    import torch

    return torch.load(path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "tasks" not in config:
        raise ValueError("config must be a mapping with a top-level 'tasks' key")
    for t in config["tasks"]:
        for key in ("task_id", "bddl", "experiments"):
            if key not in t:
                raise ValueError(f"task missing '{key}': {list(t.keys())}")
    return config


# The arena table is provided by the scene, not loaded via get_object_fn; the
# kitchen problem builder skips it explicitly.
_ARENA_FIXTURES = {"kitchen_table"}


def _experiment_categories(bddl_path):
    """Object/fixture categories a BDDL needs loaded via get_object_fn.

    Covers both ``(:objects X - cat)`` and ``(:fixtures X - cat)`` declarations
    (both resolve through OBJECTS_DICT), minus arena-level fixtures.
    """
    text = pathlib.Path(bddl_path).read_text()
    cats = {m.group(1) for m in re.finditer(r"\b[a-z0-9_]+ - ([a-z0-9_]+)", text)}
    return cats - _ARENA_FIXTURES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Execution config YAML (from build_execution_instructions.py).")
    p.add_argument("--exp-id", default="sequential", help="Which experiment to run per task.")
    p.add_argument("--task-id", default=None, help="Run only this task_id (exact or prefix match).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    p.add_argument("--num-trials", type=int, default=10, help="Rollouts per task.")
    p.add_argument("--max-steps", type=int, default=400, help="Max env steps (single-prompt / no steps_per_prompt).")
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bddl-root", default=str(_WATCHACT_ROOT / "data" / "bddl_files"))
    p.add_argument("--init-states-dir", default=None, help="Optional dir with <bddl_stem>.pruned_init files.")
    p.add_argument("--video-out", default=str(_WATCHACT_ROOT / "outputs" / "libero_execution"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)

    config = _load_config(args.config)
    tasks = config["tasks"]
    if args.task_id is not None:
        exact = [t for t in tasks if t["task_id"] == args.task_id]
        tasks = exact or [t for t in tasks if str(t["task_id"]).startswith(args.task_id)]
        if not tasks:
            raise SystemExit(f"task_id {args.task_id!r} not found in config.")

    bddl_root = pathlib.Path(args.bddl_root)

    # Register custom objects + custom goal predicates (Right/Horizontal), then
    # preflight every BDDL's object categories.
    custom_objects.register_watchact_objects()
    custom_predicates.register_watchact_predicates()
    from libero.libero.envs.objects import get_object_dict

    resolved_bddls = {}
    all_categories: set[str] = set()
    for t in tasks:
        # Prefer the full "<Category>/<stem>.bddl" path when present (unambiguous);
        # fall back to the bare stem for older configs.
        bddl_path = resolve_bddl_path(t.get("bddl_file") or t["bddl"], bddl_root)
        resolved_bddls[t["task_id"]] = bddl_path
        all_categories |= _experiment_categories(bddl_path)
    missing = custom_objects.ensure_categories_registered(all_categories)
    unresolved = [c for c in missing if c not in get_object_dict()]
    if unresolved:
        raise SystemExit(
            "Object categories not registered (no LIBERO class and no custom_assets folder): "
            + ", ".join(sorted(unresolved))
        )

    policy = WebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)

    video_root = pathlib.Path(args.video_out)
    video_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}

    for t in tasks:
        tid = t["task_id"]
        bddl_path = resolved_bddls[tid]
        bddl_stem = bddl_path.stem
        interest = t.get("interest_objects_final_states") or []

        experiments = [e for e in t["experiments"] if e["exp_id"] == args.exp_id]
        if not experiments:
            avail = [e["exp_id"] for e in t["experiments"]]
            logger.info("skip %s: exp_id %r not in %s", tid, args.exp_id, avail)
            continue

        logger.info("\n%s\nTask %s (bddl=%s)\n%s", "=" * 60, tid, bddl_stem, "=" * 60)
        try:
            env = _get_libero_env(bddl_path, LIBERO_ENV_RESOLUTION, args.seed)
        except Exception as e:
            logger.error("env init failed for %s: %s", tid, e)
            all_results[tid] = {"error": f"{type(e).__name__}: {e}"}
            continue

        init_states = _load_init_states(args.init_states_dir, bddl_stem)
        num_trials = args.num_trials
        if init_states is not None and len(init_states) < num_trials:
            num_trials = len(init_states)

        task_results = {}
        for exp in experiments:
            eid = exp["exp_id"]
            prompts = exp["prompts"]
            steps_per_prompt = exp.get("steps_per_prompt")
            switch_steps = exp.get("switch_steps", [])
            if steps_per_prompt is not None and len(prompts) > 1:
                switch_steps = [steps_per_prompt * (i + 1) for i in range(len(prompts) - 1)]
                max_steps = steps_per_prompt * len(prompts)
            else:
                max_steps = args.max_steps

            # Key by task_id (unique, matches results.json) so tasks sharing a
            # BDDL don't overwrite each other's videos.
            out_dir = video_root / str(tid) / eid
            out_dir.mkdir(parents=True, exist_ok=True)

            successes, completion_sum, episodes = 0, 0.0, []
            for ep in range(num_trials):
                init_state = init_states[ep] if init_states is not None else None
                success, completion, details, imgs, wrist = _run_episode(
                    env, policy, init_state, prompts, switch_steps, max_steps, args, interest
                )
                successes += int(success)
                completion_sum += completion
                episodes.append(
                    {"ep_idx": ep, "success": bool(success), "completion_ratio": completion,
                     "predicate_evaluations": details}
                )
                tag = "success" if success else "failure"
                if imgs:
                    imageio.mimwrite(out_dir / f"ep{ep:02d}_{tag}.mp4", [np.asarray(x) for x in imgs], fps=10)
                logger.info(
                    "  ep %d/%d: %s completion=%.2f (%d/%d ok)",
                    ep + 1, num_trials, tag, completion, successes, ep + 1,
                )

            stats = {
                "success_rate": successes / num_trials if num_trials else 0.0,
                "mean_completion_ratio": completion_sum / num_trials if num_trials else 0.0,
                "successes": successes,
                "total_episodes": num_trials,
                "prompts": prompts,
                "episodes": episodes,
            }
            task_results[eid] = stats
            logger.info("  >>> %s / %s: SR=%.1f%% completion=%.2f",
                        bddl_stem, eid, 100 * stats["success_rate"], stats["mean_completion_ratio"])

        all_results[tid] = {"bddl": bddl_stem, "interest_objects_final_states": interest, **task_results}
        env.close()

    results_path = video_root / "results.json"
    if results_path.exists():
        existing = json.loads(results_path.read_text())
        existing.update(all_results)
        all_results = existing
    results_path.write_text(json.dumps(all_results, indent=2))
    logger.info("\nResults saved to %s", results_path)


if __name__ == "__main__":
    main()
