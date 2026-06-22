#!/usr/bin/env python3

#consider to add state-level edit distance metric

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_SUCCESS_RATE = "Plan Success Rate"
METRIC_COMPLETION_RATIO = "Progress Rate"
METRIC_SEQUENCE_SR = "Sequence SR"

# Datasets where each translation result carries a `Sequence SR` field
# (predicted vs reference-frame-transformed GT primitive_action_sequence).
# Only these get the extra metric copied into score outputs.
SEQUENCE_SR_DATASETS = {
    "Imitation",
    "Reversal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score simulated final states against goal states from action_execution outputs."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/action_execution",
        help=(
            "Simulation results file or directory. If a directory is given, all "
            "*simulation_results_*.json files (including in subdirectories) are scored."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/scores",
        help="Directory to save score JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--translation-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/translation",
        help=(
            "Root directory containing translation outputs. Used to source "
            "the per-result `Sequence SR` field for the 4 datasets where it "
            "is computed. Default: %(default)s."
        ),
    )
    return parser.parse_args()


def _norm_token(x: Any) -> Any:
    return x.strip().replace(".", "_") if isinstance(x, str) else x


def _norm_relation(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    s = x.strip().capitalize()
    if s in ("On", "In"):
        return s
    return None


def _norm_open_close(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    if s == "open":
        return "Open"
    if s == "close":
        return "Close"
    return None


def parse_state_entries(state_entries: list[Any]) -> dict[str, dict[str, Any]]:
    """
    Parse state list to a comparable dictionary.

    Supported entries:
      - [relation, object, region] where relation in {On, In}
      - [object, region]
      - [Open/Close, region]
    """
    parsed: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(state_entries):
        if not isinstance(item, list):
            raise ValueError(f"state[{i}] is not list: {item}")

        if len(item) == 3:
            rel, obj, region = item
            obj_name = _norm_token(obj)
            region_name = _norm_token(region)
            if not isinstance(obj_name, str) or not isinstance(region_name, str):
                raise ValueError(f"invalid object state at index {i}: {item}")
            parsed[obj_name] = {
                "kind": "object",
                "relation": _norm_relation(rel),
                "region": region_name,
            }
            continue

        if len(item) == 2:
            a, b = item
            open_close = _norm_open_close(a)
            rhs = _norm_token(b)
            if not isinstance(rhs, str):
                raise ValueError(f"invalid 2-tuple state at index {i}: {item}")
            if open_close is not None:
                parsed[f"status::{rhs}"] = {
                    "kind": "status",
                    "state": open_close,
                }
            else:
                lhs = _norm_token(a)
                if not isinstance(lhs, str):
                    raise ValueError(f"invalid 2-tuple object state at index {i}: {item}")
                parsed[lhs] = {
                    "kind": "object",
                    "relation": None,
                    "region": rhs,
                }
            continue

        raise ValueError(f"unsupported state length at index {i}: {item}")
    return parsed


def _values_equal(v1: dict[str, Any] | None, v2: dict[str, Any] | None) -> bool:
    if v1 is None or v2 is None:
        return False
    if v1.get("kind") != v2.get("kind"):
        return False

    if v1["kind"] == "status":
        return v1.get("state") == v2.get("state")

    if v1["kind"] == "object":
        if v1.get("region") != v2.get("region"):
            return False
        r1 = v1.get("relation")
        r2 = v2.get("relation")
        if r1 is None or r2 is None:
            return True
        return r1 == r2

    return False


def _pretty_key(key: str) -> str:
    if key.startswith("status::"):
        return key.split("status::", 1)[1]
    return key


def get_camera_perspective(row: dict[str, Any]) -> Any:
    value = row.get("camera_perspective")
    if value is None:
        value = row.get("camera_perspective")
    return value


def evaluate_final_state(final_state: list[Any], example: dict[str, Any]) -> dict[str, Any]:
    """
    Return:
    {
        "all_success": bool,
        "Plan Success Rate": float,
        "Progress Rate": float,
        "final_state_goal_match_ratio": float,
        "interest_mismatch_objects": list[str],
        "non_interest_mismatch_ratio": float,
        "non_interest_mismatch_objects": list[str],
    }
    """
    initial_state = example.get("initial_state")
    goal_state = example.get("goal_state")
    if not isinstance(initial_state, list):
        raise TypeError("example.initial_state must be a list")
    if not isinstance(goal_state, list):
        raise TypeError("example.goal_state must be a list")
    if not isinstance(final_state, list):
        raise TypeError("final_state must be a list")

    init_dict = parse_state_entries(initial_state)
    goal_dict = parse_state_entries(goal_state)
    final_dict = parse_state_entries(final_state)

    if len(final_state) != len(goal_state):
        raise ValueError(
            "final_state and final_goal must have the same number of entries, "
            f"got {len(final_state)} vs {len(goal_state)}"
        )

    base_keys = sorted(set(init_dict.keys()) | set(goal_dict.keys()))
    interest_keys = [
        key for key in base_keys if not _values_equal(init_dict.get(key), goal_dict.get(key))
    ]
    non_interest_keys = [
        key for key in base_keys if _values_equal(init_dict.get(key), goal_dict.get(key))
    ]

    interest_mismatch_keys: list[str] = []
    correct_count = 0
    for key in interest_keys:
        if _values_equal(final_dict.get(key), goal_dict.get(key)):
            correct_count += 1
        else:
            interest_mismatch_keys.append(key)

    if interest_keys:
        completion_ratio = correct_count / len(interest_keys)
    else:
        completion_ratio = 1.0

    # Success Rate: goal-relevant changes are all correct (ignores side-effects).
    success_rate = 1.0 if completion_ratio == 1.0 else 0.0

    non_interest_mismatch_keys: list[str] = []
    for key in non_interest_keys:
        if key in final_dict and not _values_equal(final_dict.get(key), goal_dict.get(key)):
            non_interest_mismatch_keys.append(key)

    if non_interest_keys:
        non_interest_mismatch_ratio = len(non_interest_mismatch_keys) / len(non_interest_keys)
    else:
        non_interest_mismatch_ratio = 0.0

    different_state_count = 0
    for key in final_dict:
        if not _values_equal(final_dict.get(key), goal_dict.get(key)):
            different_state_count += 1
    final_state_goal_match_ratio = (
        1.0 - (different_state_count / len(final_state)) if final_state else 1.0
    )

    # all_success is stricter: full final-state exact match against goal_state.
    all_success = (
        len(final_dict) == len(goal_dict)
        and set(final_dict.keys()) == set(goal_dict.keys())
        and all(_values_equal(final_dict.get(k), goal_dict.get(k)) for k in goal_dict.keys())
    )

    return {
        "all_success": all_success,
        METRIC_SUCCESS_RATE: success_rate,
        METRIC_COMPLETION_RATIO: completion_ratio,
        "final_state_goal_match_ratio": final_state_goal_match_ratio,
        "interest_mismatch_objects": [_pretty_key(k) for k in interest_mismatch_keys],
        "non_interest_mismatch_ratio": non_interest_mismatch_ratio,
        "non_interest_mismatch_objects": [_pretty_key(k) for k in non_interest_mismatch_keys],
    }


def find_translation_file(
    input_file: Path, translation_root: Path
) -> tuple[str | None, Path | None]:
    """Return (dataset, translation_file) if the input file maps to one of
    the SEQUENCE_SR_DATASETS and the corresponding translation file exists.
    Otherwise return (dataset_or_None, None)."""
    dataset = input_file.parent.name
    if dataset not in SEQUENCE_SR_DATASETS:
        return (dataset if dataset else None, None)
    stem = input_file.stem
    if not stem.startswith("simulation_results_"):
        return (dataset, None)
    model_dataset = stem[len("simulation_results_"):]
    candidate = translation_root / dataset / f"{model_dataset}.json"
    return (dataset, candidate if candidate.is_file() else None)


def build_sequence_sr_index(
    translation_file: Path,
) -> dict[tuple[Any, Any, Any], int]:
    """Returns {(original_id, camera_perspective, spatial_reference) -> 0|1}.
    Rows whose `Sequence SR` is null or missing are skipped."""
    payload = json.load(open(translation_file, "r", encoding="utf-8"))
    index: dict[tuple[Any, Any, Any], int] = {}
    for r in payload.get("results", []) or []:
        sr = r.get(METRIC_SEQUENCE_SR)
        if sr is None:
            continue
        key = (
            r.get("original_id"),
            r.get("camera_perspective"),
            r.get("spatial_reference"),
        )
        index[key] = int(sr)
    return index


def infer_dataset_name(input_file: Path) -> str:
    stem = input_file.stem
    # Try known prefixes
    for prefix in ("gpt_simulation_results_", "simulation_results_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    # Try generic pattern: <model>_simulation_results_<dataset>
    import re
    m = re.match(r"^\w+_simulation_results_(.+)$", stem)
    if m:
        return m.group(1)
    # Fallback to parent directory name if it looks like a dataset
    parent = input_file.parent.name
    if re.match(r"^[0-9]+_.+$", parent):
        return parent
    return stem


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def score_one_file(
    input_file: Path,
    output_dir: Path,
    translation_root: Path | None = None,
) -> Path:
    payload = json.load(open(input_file, "r", encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input must be JSON object: {input_file}")

    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Input missing list field 'results': {input_file}")

    sequence_sr_index: dict[tuple[Any, Any, Any], int] | None = None
    sequence_sr_source: Path | None = None
    if translation_root is not None:
        dataset, trans_path = find_translation_file(input_file, translation_root)
        if dataset in SEQUENCE_SR_DATASETS:
            if trans_path is not None:
                sequence_sr_index = build_sequence_sr_index(trans_path)
                sequence_sr_source = trans_path
            else:
                print(
                    f"  [WARN] dataset={dataset} is in SEQUENCE_SR_DATASETS but "
                    f"no translation file found for {input_file.name}; "
                    f"skipping Sequence SR injection.",
                    flush=True,
                )

    scored_rows: list[dict[str, Any]] = []
    eval_count = 0
    success_rate_sum = 0.0
    completion_ratio_sum = 0.0
    final_state_goal_match_ratio_sum = 0.0
    non_interest_mismatch_ratio_sum = 0.0
    sequence_sr_sum = 0
    sequence_sr_count = 0
    combo_stats: dict[tuple[Any, Any], dict[str, float]] = {}

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            scored_rows.append(
                {
                    "status": "error",
                    "error": f"Row {idx} is not a dict.",
                }
            )
            continue

        row_id = row.get("id")
        original_id = row.get("original_id")
        camera_perspective = get_camera_perspective(row)
        spatial_reference = row.get("spatial_reference")
        language_instruction = row.get("language_instruction")
        video_path = row.get("video_path")

        try:
            final_state = row.get("final_state")
            initial_state = row.get("initial_state")
            goal_state = row.get("final_goal")

            metric = evaluate_final_state(
                final_state=final_state,
                example={
                    "initial_state": initial_state,
                    "goal_state": goal_state,
                },
            )

            sequence_sr_value: int | None = None
            if sequence_sr_index is not None:
                sequence_sr_value = sequence_sr_index.get(
                    (original_id, camera_perspective, spatial_reference)
                )
                if sequence_sr_value is not None:
                    metric[METRIC_SEQUENCE_SR] = sequence_sr_value
                    metric[METRIC_SUCCESS_RATE] = sequence_sr_value

            eval_count += 1
            success_rate_sum += metric[METRIC_SUCCESS_RATE]
            completion_ratio_sum += metric[METRIC_COMPLETION_RATIO]
            final_state_goal_match_ratio_sum += metric["final_state_goal_match_ratio"]
            non_interest_mismatch_ratio_sum += metric["non_interest_mismatch_ratio"]
            if sequence_sr_value is not None:
                sequence_sr_sum += sequence_sr_value
                sequence_sr_count += 1

            combo_key = (camera_perspective, spatial_reference)
            if combo_key not in combo_stats:
                combo_stats[combo_key] = {
                    "evaluated_count": 0.0,
                    "success_rate_sum": 0.0,
                    "completion_ratio_sum": 0.0,
                    "final_state_goal_match_ratio_sum": 0.0,
                    "non_interest_mismatch_ratio_sum": 0.0,
                    "sequence_sr_sum": 0.0,
                    "sequence_sr_count": 0.0,
                }
            combo = combo_stats[combo_key]
            combo["evaluated_count"] += 1.0
            combo["success_rate_sum"] += metric[METRIC_SUCCESS_RATE]
            combo["completion_ratio_sum"] += metric[METRIC_COMPLETION_RATIO]
            combo["final_state_goal_match_ratio_sum"] += metric["final_state_goal_match_ratio"]
            combo["non_interest_mismatch_ratio_sum"] += metric["non_interest_mismatch_ratio"]
            if sequence_sr_value is not None:
                combo["sequence_sr_sum"] += sequence_sr_value
                combo["sequence_sr_count"] += 1.0

            scored_rows.append(
                {
                    "id": row_id,
                    "original_id": original_id,
                    "camera_perspective": camera_perspective,
                    "spatial_reference": spatial_reference,
                    "language_instruction": language_instruction,
                    "video_path": video_path,
                    "simulation_status": row.get("status"),
                    "metric": metric,
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            scored_rows.append(
                {
                    "id": row_id,
                    "original_id": original_id,
                    "camera_perspective": camera_perspective,
                    "spatial_reference": spatial_reference,
                    "language_instruction": language_instruction,
                    "video_path": video_path,
                    "simulation_status": row.get("status"),
                    "status": "error",
                    "error": str(exc),
                }
            )

    combo_metrics: list[dict[str, Any]] = []
    for (camera, spatial), combo in sorted(
        combo_stats.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        combo_count = int(combo["evaluated_count"])
        combo_entry = {
            "camera_perspective": camera,
            "spatial_reference": spatial,
            "evaluated_count": combo_count,
            METRIC_SUCCESS_RATE: (
                combo["success_rate_sum"] / combo_count if combo_count else 0.0
            ),
            METRIC_COMPLETION_RATIO: (
                combo["completion_ratio_sum"] / combo_count if combo_count else 0.0
            ),
            "avg_final_state_goal_match_ratio": (
                combo["final_state_goal_match_ratio_sum"] / combo_count
                if combo_count
                else 0.0
            ),
            "avg_non_interest_mismatch_ratio": (
                combo["non_interest_mismatch_ratio_sum"] / combo_count
                if combo_count
                else 0.0
            ),
        }
        seq_count = int(combo["sequence_sr_count"])
        if seq_count > 0:
            combo_entry[METRIC_SEQUENCE_SR] = combo["sequence_sr_sum"] / seq_count
            combo_entry["sequence_sr_evaluated_count"] = seq_count
        combo_metrics.append(combo_entry)

    summary = {
        "evaluated_count": eval_count,
        METRIC_SUCCESS_RATE: (success_rate_sum / eval_count) if eval_count else 0.0,
        METRIC_COMPLETION_RATIO: (completion_ratio_sum / eval_count) if eval_count else 0.0,
        "avg_final_state_goal_match_ratio": (
            final_state_goal_match_ratio_sum / eval_count
        )
        if eval_count
        else 0.0,
        "avg_non_interest_mismatch_ratio": (
            non_interest_mismatch_ratio_sum / eval_count
        )
        if eval_count
        else 0.0,
        "by_camera_perspective_spatial_reference": combo_metrics,
    }
    if sequence_sr_count > 0:
        summary[METRIC_SEQUENCE_SR] = sequence_sr_sum / sequence_sr_count
        summary["sequence_sr_evaluated_count"] = sequence_sr_count
        summary["sequence_sr_source"] = (
            str(sequence_sr_source) if sequence_sr_source is not None else None
        )

    dataset_name = infer_dataset_name(input_file)
    output_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_action_execution_path": str(input_file),
        "total": len(scored_rows),
        "ok": sum(1 for r in scored_rows if r.get("status") == "ok"),
        "error": sum(1 for r in scored_rows if r.get("status") == "error"),
        "summary": summary,
        "results": scored_rows,
    }

    output_file = output_dir / f"scores_{dataset_name}.json"
    save_json(output_file, output_payload)
    return output_file


def list_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # Search both flat and subdirectory layouts
        files = sorted(input_path.glob("**/simulation_results_*.json"))
        if not files:
            files = sorted(input_path.glob("**/*_simulation_results_*.json"))
        if files:
            return files
        # fallback: score all json files if naming pattern is absent
        return sorted(input_path.glob("*.json"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    args = parse_args()
    input_files = list_input_files(args.input_path)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found under: {args.input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_file in input_files:
        output_file = score_one_file(
            input_file=input_file,
            output_dir=args.output_dir,
            translation_root=args.translation_dir,
        )
        print(f"Saved: {output_file}", flush=True)


if __name__ == "__main__":
    main()
