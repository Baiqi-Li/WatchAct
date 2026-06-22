#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ACTION_COMMANDS = {"pick", "place", "open", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate aligned action plans from translator outputs and save final states."
        )
    )
    parser.add_argument(
        "--translator-results-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/translation/0_following_sequences/gpt-5.1_0_following_sequences_random20_seed12345.json",
        help="Path to translator results JSON (expected: <model_id>_<dataset>_<suffix>.json; default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save simulation results JSON. If omitted, defaults to "
            "outputs/action_execution/<dataset>/simulation_results_<model_id>_<dataset>_<suffix>.json."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    return parser.parse_args()


def infer_dataset_name_from_path(path: Path) -> str | None:
    stem = path.stem
    patterns = (
        r"^([0-9]+_.+)$",
        r"^_([0-9]+_.+)$",
        r"^[A-Za-z0-9._-]+_results_([0-9]+_.+)$",
        r"^[A-Za-z0-9._-]+_translator_results_([0-9]+_.+)$",
        r"^[A-Za-z0-9._-]+_simulation_results_([0-9]+_.+)$",
    )
    for pattern in patterns:
        matched = re.match(pattern, stem)
        if matched:
            return matched.group(1)

    parent = path.parent.name
    if re.match(r"^[0-9]+_.+$", parent):
        return parent
    return None


def infer_translator_name_parts(
    *, translator_results_path: Path, dataset_name: str | None
) -> tuple[str | None, str | None]:
    """Infer (model_id, suffix) from translator results filename.

    Supported stems include:
    - <model_id>_<task>_<suffix>
    - <model_id>_translator_results_<task>
    - <model_id>_translator_results_<task>_<suffix>
    """
    stem = translator_results_path.stem

    if dataset_name:
        marker = f"_{dataset_name}_"
        if marker in stem:
            prefix, suffix = stem.split(marker, 1)
            model_id = re.sub(r"_translator_results$", "", prefix).strip()
            suffix = suffix.strip()
            if model_id:
                return model_id, (suffix or None)

    matched = re.match(r"^(.+?)_translator_results_([0-9]+_.+)$", stem)
    if matched is not None:
        return matched.group(1).strip(), None

    return None, None


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path

    dataset_name = infer_dataset_name_from_path(args.translator_results_path)
    model_id, suffix = infer_translator_name_parts(
        translator_results_path=args.translator_results_path,
        dataset_name=dataset_name,
    )

    if dataset_name and model_id and suffix:
        return (
            PROJECT_ROOT
            / "outputs/action_execution"
            / dataset_name
            / f"simulation_results_{model_id}_{dataset_name}_{suffix}.json"
        )

    if dataset_name and model_id:
        return (
            PROJECT_ROOT
            / "outputs/action_execution"
            / dataset_name
            / f"simulation_results_{model_id}_{dataset_name}.json"
        )

    if dataset_name is None:
        return PROJECT_ROOT / "outputs/action_execution/simulation_results.json"

    return (
        PROJECT_ROOT
        / "outputs/action_execution"
        / dataset_name
        / f"simulation_results_{dataset_name}.json"
    )


def _norm_token(x: Any) -> Any:
    return x.strip().replace(".", "_") if isinstance(x, str) else x


def _is_unmatched_object_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip().lower()
    return "unmatch" in token


def _is_cabinet_drawer_region(region: str) -> bool:
    """Check if region is a cabinet drawer (requires open/close)."""
    return bool(re.search(r"wooden_cabinet_\d+_(top|middle|bottom)_region", region))


def get_camera_perspective(row: dict[str, Any]) -> Any:
    value = row.get("camera_perspective")
    if value is None:
        value = row.get("camera_perspective")
    return value


def _relation_from_region(region: Any, fallback: str | None = None) -> str:
    r = _norm_token(region)
    if isinstance(r, str) and "_contain_region" in r:
        return "In"
    if isinstance(r, str) and re.search(r"wooden_cabinet_\d+_(top|middle|bottom)_region", r):
        return "In"
    if isinstance(r, str) and (r.startswith("main_") or r.startswith("fixture_")):
        return "On"
    if isinstance(fallback, str) and fallback.capitalize() in ("In", "On"):
        return fallback.capitalize()
    return "On"


def normalize_initial_states_dots(initial_states: list[Any]) -> list[list[Any]]:
    """
    Normalize object/region tokens by replacing '.' with '_'.
    Supports:
      [relation, object, region]
      [object, region]
      [Open/Close, region]
    """
    out: list[list[Any]] = []
    for i, item in enumerate(initial_states):
        if not isinstance(item, list):
            raise ValueError(f"initial_states[{i}] is not list: {item}")

        if len(item) == 3:
            rel, obj, region = item
            out.append(
                [str(rel).strip().capitalize(), _norm_token(obj), _norm_token(region)]
            )
        elif len(item) == 2:
            a, b = item
            out.append([_norm_token(a), _norm_token(b)])
        else:
            raise ValueError(f"initial_states[{i}] length must be 2 or 3: {item}")
    return out


def _normalize_action_plan(action_plan: list[Any]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for i, action in enumerate(action_plan):
        if not isinstance(action, list) or len(action) < 2:
            raise ValueError(f"action_plan[{i}] is invalid: {action}")
        cmd = str(action[0]).strip().lower()
        rest = [_norm_token(x) for x in action[1:]]
        out.append([cmd, *rest])
    return out


def simulate_actions(
    initial_state: list[Any], action_plan: list[Any]
) -> tuple[bool, list[list[Any]], list[dict[str, Any]]]:
    """
    Supports state formats:
      1) [relation, object, region]
      2) [object, region]
      3) [Open/Close, region]

    Supports actions:
      pick/place in pairs
      open/close as single actions
    """
    initial_state = normalize_initial_states_dots(initial_state)
    action_plan = _normalize_action_plan(action_plan)

    object_state: dict[str, dict[str, str]] = {}
    open_state: dict[str, str] = {}
    execution_errors: list[dict[str, Any]] = []
    picked_not_placed: set[str] = set()

    for i, item in enumerate(initial_state):
        if len(item) == 3:
            rel, obj, region = item
            if not isinstance(obj, str):
                raise ValueError(f"initial_state[{i}] object is not string: {item}")
            if not isinstance(region, str):
                raise ValueError(f"initial_state[{i}] region is not string: {item}")
            object_state[obj] = {
                "relation": _relation_from_region(region, fallback=str(rel)),
                "region": region,
            }
        else:
            a, b = item
            tag = str(a).strip().lower()
            if tag in ("open", "close"):
                if not isinstance(b, str):
                    raise ValueError(f"initial_state[{i}] region is not string: {item}")
                open_state[b] = "Open" if tag == "open" else "Close"
            else:
                obj, region = a, b
                if not isinstance(obj, str):
                    raise ValueError(f"initial_state[{i}] object is not string: {item}")
                if not isinstance(region, str):
                    raise ValueError(f"initial_state[{i}] region is not string: {item}")
                object_state[obj] = {
                    "relation": _relation_from_region(region),
                    "region": region,
                }

    all_success = True
    i = 0
    while i < len(action_plan):
        action = action_plan[i]
        cmd = action[0]

        if cmd == "pick":
            if len(action) != 3:
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": "invalid_pick_action",
                        "message": f"invalid pick action: {action}",
                    }
                )
                i += 1
                continue

            pick_obj_for_check = action[1]
            if isinstance(pick_obj_for_check, str) and not _is_unmatched_object_name(
                pick_obj_for_check
            ):
                if pick_obj_for_check in picked_not_placed:
                    execution_errors.append(
                        {
                            "action_index": i,
                            "type": "duplicate_pick_without_place",
                            "message": "same object is picked again before being placed",
                            "object_unified_name": pick_obj_for_check,
                            "command": action,
                        }
                    )
                    all_success = False
                    i += 1
                    continue
                picked_not_placed.add(pick_obj_for_check)

            if i + 1 >= len(action_plan):
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": "pick_without_place",
                        "message": "pick has no matching place",
                    }
                )
                i += 1
                continue

            place_action = action_plan[i + 1]
            if place_action[0] != "place" or len(place_action) != 3:
                next_is_same_object_pick = (
                    place_action[0] == "pick"
                    and len(place_action) >= 2
                    and place_action[1] == pick_obj_for_check
                )
                if next_is_same_object_pick:
                    execution_errors.append(
                        {
                            "action_index": i + 1,
                            "type": "duplicate_pick_without_place",
                            "message": "same object is picked again before being placed",
                            "object_unified_name": pick_obj_for_check,
                            "command": place_action,
                        }
                    )
                    all_success = False
                    i += 2
                    continue
                execution_errors.append(
                    {
                        "action_index": i + 1,
                        "type": "invalid_place_after_pick",
                        "message": f"invalid place action after pick: {place_action}",
                    }
                )
                i += 1
                continue

            _, pick_obj, pick_region = action
            _, place_obj, place_region = place_action
            if pick_obj != place_obj:
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": "pick_place_object_mismatch",
                        "message": f"pick/place object mismatch: {pick_obj} vs {place_obj}",
                    }
                )
                i += 2
                continue
            if _is_unmatched_object_name(pick_obj):
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": "unmatched_object",
                        "message": (
                            "skip action pair because object_unified_name is unmatched"
                        ),
                        "object_unified_name": pick_obj,
                    }
                )
                i += 2
                continue

            cur = object_state.get(str(pick_obj))
            if cur is None or cur["region"] != pick_region:
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": "pick_source_location_mismatch",
                        "message": "pick source location does not match current object state",
                        "command": action,
                        "paired_place_command": place_action,
                        "object_unified_name": pick_obj,
                        "pick_source_region": pick_region,
                        "current_object_region": cur["region"] if cur is not None else None,
                    }
                )
                all_success = False
                i += 2
                continue

            if not isinstance(place_region, str):
                execution_errors.append(
                    {
                        "action_index": i + 1,
                        "type": "invalid_place_region",
                        "message": f"place region is not string: {place_action}",
                    }
                )
                i += 2
                continue
            if isinstance(pick_region, str) and _is_cabinet_drawer_region(pick_region):
                if open_state.get(pick_region) != "Open":
                    execution_errors.append(
                        {
                            "action_index": i,
                            "type": "pick_from_closed_drawer",
                            "message": "cannot pick from closed drawer",
                            "command": action,
                            "drawer_region": pick_region,
                            "drawer_state": open_state.get(pick_region, "Close"),
                        }
                    )
                    all_success = False
                    if isinstance(pick_obj, str):
                        picked_not_placed.discard(pick_obj)
                    i += 2
                    continue
            if _is_cabinet_drawer_region(place_region):
                if open_state.get(place_region) != "Open":
                    execution_errors.append(
                        {
                            "action_index": i + 1,
                            "type": "place_into_closed_drawer",
                            "message": "cannot place into closed drawer",
                            "command": place_action,
                            "drawer_region": place_region,
                            "drawer_state": open_state.get(place_region, "Close"),
                        }
                    )
                    all_success = False
                    if isinstance(pick_obj, str):
                        picked_not_placed.discard(pick_obj)
                    i += 2
                    continue
            object_state[str(pick_obj)] = {
                "relation": _relation_from_region(place_region),
                "region": place_region,
            }
            if isinstance(pick_obj, str):
                picked_not_placed.discard(pick_obj)
            i += 2
            continue

        if cmd in ("open", "close"):
            if len(action) == 2:
                target_region = action[1]
            elif len(action) == 3:
                target_region = action[2]
            else:
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": f"invalid_{cmd}_action",
                        "message": f"invalid {cmd} action: {action}",
                    }
                )
                i += 1
                continue

            if not isinstance(target_region, str):
                execution_errors.append(
                    {
                        "action_index": i,
                        "type": f"invalid_{cmd}_target_region",
                        "message": f"{cmd} target region is not string: {action}",
                    }
                )
                i += 1
                continue
            open_state[target_region] = "Open" if cmd == "open" else "Close"
            i += 1
            continue

        execution_errors.append(
            {
                "action_index": i,
                "type": "unsupported_action_type",
                "message": f"unsupported action type: {action}",
            }
        )
        i += 1
        continue

    final_state: list[list[Any]] = []
    for obj, st in object_state.items():
        final_state.append([st["relation"], obj, st["region"]])
    for region, st in open_state.items():
        final_state.append([st, region])

    return all_success, final_state, execution_errors


def translator_reformat(translator_output_json: dict[str, Any]) -> dict[str, Any]:
    objects = translator_output_json.get("objects")
    actions = translator_output_json.get("actions")
    if objects is None:
        objects = translator_output_json.get("Objects", [])
    if actions is None:
        actions = translator_output_json.get("Actions", [])
    if not isinstance(objects, list):
        raise TypeError("translator_output.objects must be a list")
    if not isinstance(actions, list):
        raise TypeError("translator_output.actions must be a list")

    translator_reformat_json: dict[str, Any] = {
        "object_matched": True,
        "action_matched": True,
        "initial_state": [],
        "action_plan": [],
        "execution_errors": [],
    }

    for obj_idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise TypeError(f"object item must be dict: {obj}")
        if not obj.get("matched", False):
            translator_reformat_json["object_matched"] = False
            translator_reformat_json["execution_errors"].append(
                {
                    "object_index": obj_idx,
                    "type": "unmatched_object_in_objects",
                    "source_name": obj.get("source_name"),
                    "unified_name": obj.get("unified_name"),
                    "message": "object matched flag is false in translator output",
                }
            )
            continue
        translator_reformat_json["initial_state"].append(
            [obj.get("unified_name"), obj.get("unified_region")]
        )

    for action_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            translator_reformat_json["execution_errors"].append(
                {
                    "action_index": action_idx,
                    "type": "invalid_action_item",
                    "message": f"action item must be dict: {action}",
                }
            )
            continue
        command = str(action.get("command", "")).strip().lower()
        object_unified_name = action.get("object_unified_name")

        if command not in SUPPORTED_ACTION_COMMANDS:
            translator_reformat_json["execution_errors"].append(
                {
                    "action_index": action_idx,
                    "type": "unsupported_command",
                    "command": command,
                    "message": f"unsupported command in translator output: {command}",
                }
            )
            continue

        if _is_unmatched_object_name(object_unified_name):
            translator_reformat_json["execution_errors"].append(
                {
                    "action_index": action_idx,
                    "type": "unmatched_object",
                    "command": command,
                    "object_unified_name": object_unified_name,
                    "message": (
                        "skip action because object_unified_name is unmatched"
                    ),
                }
            )
            continue

        if command == "pick":
            translator_reformat_json["action_plan"].append(
                [
                    command,
                    action.get("object_unified_name"),
                    action.get("source_region_id"),
                ]
            )
        elif command == "place":
            translator_reformat_json["action_plan"].append(
                [
                    command,
                    action.get("object_unified_name"),
                    action.get("target_region_id"),
                ]
            )
        elif command in ("open", "close"):
            translator_reformat_json["action_plan"].append(
                [command, action.get("target_region_id")]
            )

    return translator_reformat_json


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args)
    payload = json.load(open(args.translator_results_path, "r", encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("translator results file must be a JSON object")

    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("translator results file must contain a list field: results")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            output_rows.append(
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
        task_information = row.get("task_information", {})
        translator_output = row.get("translator_output", {})

        print(f"[{idx}/{len(rows)}] id={row_id} original_id={original_id}", flush=True)

        try:
            if not isinstance(task_information, dict):
                raise TypeError("task_information must be dict")
            initial_states = task_information.get("initial_states", [])
            if not isinstance(initial_states, list):
                raise TypeError("task_information.initial_states must be a list")
            final_goal = task_information.get("final_goal")
            if final_goal is not None and not isinstance(final_goal, list):
                raise TypeError("task_information.final_goal must be a list when provided")

            if not isinstance(translator_output, dict):
                raise TypeError("translator_output must be dict")
            reformatted = translator_reformat(translator_output)

            all_success, final_state, simulation_errors = simulate_actions(
                initial_states, reformatted["action_plan"]
            )
            execution_errors = list(reformatted.get("execution_errors", []))
            execution_errors.extend(simulation_errors)
            execution_status = "error" if execution_errors else "ok"

            output_rows.append(
                {
                    "id": row_id,
                    "original_id": original_id,
                    "camera_perspective": camera_perspective,
                    "spatial_reference": spatial_reference,
                    "language_instruction": language_instruction,
                    "video_path": video_path,
                    "translator_status": row.get("status"),
                    "object_matched": reformatted["object_matched"],
                    "action_matched": reformatted["action_matched"],
                    "all_success": all_success,
                    "execution_status": execution_status,
                    "execution_errors": execution_errors,
                    "initial_state": normalize_initial_states_dots(initial_states),
                    "action_plan": _normalize_action_plan(reformatted["action_plan"]),
                    "final_state": final_state,
                    "final_goal": final_goal,
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            output_rows.append(
                {
                    "id": row_id,
                    "original_id": original_id,
                    "camera_perspective": camera_perspective,
                    "spatial_reference": spatial_reference,
                    "language_instruction": language_instruction,
                    "video_path": video_path,
                    "translator_status": row.get("status"),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(
                f"Failed id={row_id} original_id={original_id}: {exc}",
                flush=True,
            )

    output_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_translator_results_path": str(args.translator_results_path),
        "total": len(output_rows),
        "ok": sum(1 for row in output_rows if row.get("status") == "ok"),
        "error": sum(1 for row in output_rows if row.get("status") == "error"),
        "results": output_rows,
    }
    save_json(output_path, output_payload)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
