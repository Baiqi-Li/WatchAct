#!/usr/bin/env python3
"""Build an execution-instruction YAML (same shape as
``execution_instruction_example.yaml``) from ``action_steps.json``.

For every entry in ``action_steps.json`` this script looks up the matching
sample in ``WatchAct/data`` (by ``data_id`` == the jsonl ``id`` field) and
fills in the fields that live in the dataset rather than in action_steps.json:

  * ``bddl``                          - stem of the sample's ``bddl_file``
  * ``interest_objects_final_states`` - taken from the category meta_data
                                        (``simulation_task`` block, already in
                                        the bare-region form the YAML expects)

The ``experiments`` block is carried over verbatim from action_steps.json. If a
``single_instruction`` experiment has an empty ``prompts`` list it is filled
with the sample's ``language_instruction`` so the output is self-contained even
when action_steps.json has not been pre-populated.

Usage:
    python build_execution_instructions.py
    python build_execution_instructions.py --output /path/to/out.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import yaml

# action_execution/ -> WatchAct/
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_WATCHACT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _WATCHACT_ROOT / "data" / "data"
_META_DIR = _WATCHACT_ROOT / "data" / "meta_data"


def build_sample_index() -> dict[str, dict[str, Any]]:
    """Map each jsonl ``id`` to {original_id, bddl_file, category}.

    ``category`` is the jsonl filename stem (e.g. ``Imitation``), which is also
    the meta_data filename stem.
    """
    index: dict[str, dict[str, Any]] = {}
    for jsonl_path in sorted(_DATA_DIR.glob("*.jsonl")):
        category = jsonl_path.stem
        for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            index[row["id"]] = {
                "original_id": row.get("original_id"),
                "bddl_file": row.get("bddl_file"),
                "language_instruction": row.get("language_instruction"),
                "category": category,
            }
    return index


_META_CACHE: dict[str, dict[str, Any]] = {}


def load_interest_objects_final_states(category: str, original_id: str) -> list[Any]:
    """Return ``interest_objects_final_states`` from <category>.json for original_id."""
    if category not in _META_CACHE:
        meta_path = _META_DIR / f"{category}.json"
        _META_CACHE[category] = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = _META_CACHE[category]
    for example in meta.get("examples", []):
        if example.get("id") == original_id:
            sim = example.get("simulation_task", {}) or {}
            return list(sim.get("interest_objects_final_states", []) or [])
    return []


def build_experiments(
    raw_experiments: list[dict[str, Any]],
    language_instruction: str | None,
) -> list[dict[str, Any]]:
    """Carry experiments over verbatim, ordering keys like the example YAML.

    A ``single_instruction`` experiment with no prompts is back-filled from the
    sample's ``language_instruction``.
    """
    experiments: list[dict[str, Any]] = []
    for exp in raw_experiments:
        exp_id = exp.get("exp_id")
        prompts = list(exp.get("prompts", []) or [])
        if exp_id == "single_instruction" and not prompts and language_instruction:
            prompts = [language_instruction]

        ordered: dict[str, Any] = {"exp_id": exp_id, "prompts": prompts}
        if "steps_per_prompt" in exp:
            ordered["steps_per_prompt"] = exp["steps_per_prompt"]
        experiments.append(ordered)
    return experiments


def build_config(action_steps: list[dict[str, Any]]) -> dict[str, Any]:
    index = build_sample_index()
    tasks: list[dict[str, Any]] = []

    for item in action_steps:
        data_id = item["data_id"]
        sample = index.get(data_id)
        if sample is None:
            raise KeyError(
                f"data_id {data_id!r} not found in {_DATA_DIR} jsonl files"
            )

        bddl_file = sample["bddl_file"]
        if not bddl_file:
            raise ValueError(f"sample {data_id!r} has no bddl_file")

        interest = load_interest_objects_final_states(
            sample["category"], sample["original_id"]
        )

        task_entry: dict[str, Any] = {
            "task_id": data_id,
            "bddl": pathlib.PurePosixPath(bddl_file).stem,
            # Full "<Category>/<stem>.bddl" path so the runner resolves it
            # unambiguously even when a stem is not globally unique.
            "bddl_file": bddl_file,
        }
        if interest:
            task_entry["interest_objects_final_states"] = interest
        task_entry["experiments"] = build_experiments(
            item.get("experiments", []), sample["language_instruction"]
        )
        tasks.append(task_entry)

    return {"tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action-steps",
        type=pathlib.Path,
        default=_SCRIPT_DIR / "action_steps.json",
        help="Path to action_steps.json (default: alongside this script).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=_SCRIPT_DIR / "execution_instructions.yaml",
        help="Output YAML path (default: action_execution/execution_instructions.yaml).",
    )
    args = parser.parse_args()

    action_steps = json.loads(args.action_steps.read_text(encoding="utf-8"))
    config = build_config(action_steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Wrote {len(config['tasks'])} tasks to {args.output}")


if __name__ == "__main__":
    main()
