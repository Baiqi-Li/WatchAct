#!/usr/bin/env python3

# TODO: consider adding a state-level edit-distance metric.

from __future__ import annotations

import argparse
import json
import re
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

# High-level category -> subcategories. Embedded so this benchmark is
# self-contained. Each subcategory equals a dataset file name with spaces in
# place of underscores (e.g. "Temporal Sort" <-> Temporal_Sort).
TASK_TAXONOMY: dict[str, list[str]] = {
    "Event Grounding": ["Fine-Grained Action", "Count", "Ordinal", "State Change", "Moment"],
    "Procedural Reasoning": ["Imitation", "Reversal", "Temporal Sort"],
    "Implicit Intent Inference": ["Nonverbal Cue", "Reference Disambiguation"],
    "Episodic Reasoning": [
        "Restore Previous State",
        "Task Continuation",
        "Error Correction",
        "Conditional Execution",
    ],
}

# Valid dataset file names (subcategory with spaces -> underscores). Used to
# recover the clean task name from filenames like
# simulation_results_<model_id>_<task>[_<suffix>].json.
KNOWN_DATASETS: set[str] = {
    sub.replace(" ", "_") for subs in TASK_TAXONOMY.values() for sub in subs
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
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/summary.json",
        help="Path to save the count-weighted overall/high-level/subcategory summary (default: %(default)s).",
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


# ---------------------------------------------------------------------------
# Relational goal evaluation (e.g. Temporal_Sort: Horizontal / Right)
#
# final_goal predicates are expressed in the canonical camera-front frame,
# while final_state region ids are in the row's target frame (already remapped
# by the translator). We therefore resolve how Horizontal/Right map onto region
# ids based on (spatial_reference, camera_perspective).
# ---------------------------------------------------------------------------

_MAIN_REGION_RE = re.compile(r"main_(back|middle|front)_(left|center|right)_region")
_PREDICATE_RE = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$")

# Each rule: which (row|col) component must be equal for Horizontal, and the
# rank table for Right such that Right(A, B) holds iff rank(A) > rank(B).
FRAME_RULES: dict[str, dict[str, Any]] = {
    # camera + front (canonical): same row = horizontal; right by column left<center<right.
    "camera_front": {
        "horiz": "row",
        "right_component": "col",
        "right_order": {"left": 0, "center": 1, "right": 2},
    },
    # camera + side (the "side" view, formerly "left"; 90 deg rotation): same column = horizontal; right by row front>middle>back.
    "camera_left": {
        "horiz": "col",
        "right_component": "row",
        "right_order": {"back": 0, "middle": 1, "front": 2},
    },
    # human (perspective-agnostic, left-right mirrored): same row = horizontal;
    # right reversed left>center>right.
    "human": {
        "horiz": "row",
        "right_component": "col",
        "right_order": {"right": 0, "center": 1, "left": 2},
    },
}


def _parse_main_region(region: Any) -> tuple[str, str] | None:
    """Return (row, col) for a main_<row>_<col>_region id, else None."""
    if not isinstance(region, str):
        return None
    m = _MAIN_REGION_RE.match(region.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def _resolve_frame_rule(spatial_reference: Any, camera_perspective: Any) -> dict[str, Any]:
    spatial = str(spatial_reference or "").strip().lower()
    camera = str(camera_perspective or "").strip().lower()
    if spatial == "human":
        # human frame ignores camera_perspective (front / side / oblique).
        return FRAME_RULES["human"]
    if spatial == "camera" and camera == "side":
        return FRAME_RULES["camera_left"]
    if spatial == "camera":
        return FRAME_RULES["camera_front"]
    raise ValueError(
        "Unsupported (spatial_reference, camera_perspective): "
        f"({spatial_reference!r}, {camera_perspective!r})"
    )


def _is_relational_goal(goal_state: Any) -> bool:
    """Relational goals are lists of predicate strings like 'Horizontal(a, b)'."""
    return isinstance(goal_state, list) and any(isinstance(g, str) for g in goal_state)


def _parse_predicate(pred: Any) -> tuple[str | None, list[str]]:
    if not isinstance(pred, str):
        return None, []
    m = _PREDICATE_RE.match(pred)
    if not m:
        return None, []
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    return m.group(1), args


def _horiz_value(region: Any, rule: dict[str, Any]) -> str | None:
    parsed = _parse_main_region(region)
    if parsed is None:
        return None
    row, col = parsed
    return row if rule["horiz"] == "row" else col


def _right_rank(region: Any, rule: dict[str, Any]) -> int | None:
    parsed = _parse_main_region(region)
    if parsed is None:
        return None
    row, col = parsed
    component = row if rule["right_component"] == "row" else col
    return rule["right_order"].get(component)


def evaluate_relational_goal(
    final_state: list[Any],
    final_goal: list[Any],
    spatial_reference: Any,
    camera_perspective: Any,
) -> dict[str, Any]:
    """Score a relational final_goal (Horizontal / Right) against final_state.

    final_goal predicates are in the canonical camera-front frame; final_state
    region ids are in the row's target frame, so Horizontal/Right are resolved
    onto region ids via (spatial_reference, camera_perspective). A predicate
    whose objects are missing or sit outside the main grid counts as not met.
    """
    if not isinstance(final_state, list):
        raise TypeError("final_state must be a list")
    if not isinstance(final_goal, list):
        raise TypeError("final_goal must be a list")

    rule = _resolve_frame_rule(spatial_reference, camera_perspective)
    final_dict = parse_state_entries(final_state)

    def region_of(obj: Any) -> str | None:
        entry = final_dict.get(_norm_token(obj))
        if entry and entry.get("kind") == "object":
            return entry.get("region")
        return None

    total = 0
    satisfied = 0
    failed_objects: list[str] = []

    for pred in final_goal:
        name, args = _parse_predicate(pred)
        if name is None:
            continue
        total += 1
        regions = [region_of(a) for a in args]
        if name == "Horizontal" and len(args) >= 2:
            values = [_horiz_value(r, rule) for r in regions]
            ok = all(v is not None for v in values) and len(set(values)) == 1
        elif name == "Right" and len(args) == 2:
            ra = _right_rank(regions[0], rule)
            rb = _right_rank(regions[1], rule)
            ok = ra is not None and rb is not None and ra > rb
        else:
            ok = False
        if ok:
            satisfied += 1
        else:
            for a in args:
                if a not in failed_objects:
                    failed_objects.append(a)

    if total:
        progress = satisfied / total
        all_success = satisfied == total
    else:
        progress = 1.0
        all_success = True
    success_rate = 1.0 if all_success else 0.0

    return {
        "all_success": all_success,
        METRIC_SUCCESS_RATE: success_rate,
        METRIC_COMPLETION_RATIO: progress,
        "final_state_goal_match_ratio": progress,
        "interest_mismatch_objects": failed_objects,
        "non_interest_mismatch_ratio": 0.0,
        "non_interest_mismatch_objects": [],
    }


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

    if _is_relational_goal(goal_state):
        return evaluate_relational_goal(
            final_state=final_state,
            final_goal=goal_state,
            spatial_reference=example.get("spatial_reference"),
            camera_perspective=example.get("camera_perspective"),
        )

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
    # Canonical pipeline layout: action_execution/<task>/simulation_results_*.json
    if input_file.parent.name in KNOWN_DATASETS:
        return input_file.parent.name
    stem = input_file.stem
    # Recover a known task embedded in the filename (handles
    # simulation_results_<model_id>_<task>[_<suffix>].json). Longest match first
    # so e.g. "State_Change" wins over any shorter accidental match.
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if re.search(rf"(?:^|_){re.escape(ds)}(?:_|$)", stem):
            return ds
    # Legacy fallbacks (numbered datasets like "0_following_sequences").
    for prefix in ("gpt_simulation_results_", "simulation_results_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    m = re.match(r"^\w+_simulation_results_(.+)$", stem)
    if m:
        return m.group(1)
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
                    "spatial_reference": spatial_reference,
                    "camera_perspective": camera_perspective,
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
                    METRIC_SUCCESS_RATE: metric[METRIC_SUCCESS_RATE],
                    METRIC_COMPLETION_RATIO: metric[METRIC_COMPLETION_RATIO],
                }
            )
        except Exception:
            scored_rows.append(
                {
                    "id": row_id,
                    "original_id": original_id,
                    "status": "error",
                }
            )

    combo_metrics: list[dict[str, Any]] = []
    for (camera, spatial), combo in sorted(
        combo_stats.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        combo_count = int(combo["evaluated_count"])
        combo_metrics.append(
            {
                "camera_perspective": camera,
                "spatial_reference": spatial,
                "count": combo_count,
                METRIC_SUCCESS_RATE: (
                    combo["success_rate_sum"] / combo_count if combo_count else 0.0
                ),
                METRIC_COMPLETION_RATIO: (
                    combo["completion_ratio_sum"] / combo_count if combo_count else 0.0
                ),
            }
        )

    dataset_name = infer_dataset_name(input_file)
    output_payload = {
        "dataset": dataset_name,
        "evaluated_count": eval_count,
        "error_count": sum(1 for r in scored_rows if r.get("status") == "error"),
        METRIC_SUCCESS_RATE: (success_rate_sum / eval_count) if eval_count else 0.0,
        METRIC_COMPLETION_RATIO: (completion_ratio_sum / eval_count) if eval_count else 0.0,
        "by_combo": combo_metrics,
        "results": scored_rows,
    }

    # Name the score file after the input file (not just the dataset) so multiple
    # simulation files for one task (different model/suffix) never collide.
    output_file = output_dir / f"scores_{input_file.stem}.json"
    save_json(output_file, output_payload)
    return output_file, output_payload


def _subcategory_to_dataset(subcategory: str) -> str:
    """Taxonomy subcategory name -> dataset file name (underscores for spaces)."""
    return subcategory.replace(" ", "_")


def build_summary(
    payloads: list[dict[str, Any]],
    task_taxonomy: dict[str, list[str]],
) -> dict[str, Any]:
    """Count-weighted summary: overall, then high-level, then subcategory.

    Weighted rate = sum(rate_i * count_i) / sum(count_i), i.e. total correct
    over total evaluated. Every taxonomy category/subcategory is always listed;
    those without results this run show null rates (and are flagged under
    `missing_subcategories`) so partial runs are explicit rather than fatal.
    """
    def acc() -> dict[str, float]:
        return {"evaluated_count": 0, "error_count": 0, "success_sum": 0.0, "progress_sum": 0.0}

    def add(target: dict[str, float], cnt: int, err: int, sr: float, pr: float) -> None:
        target["evaluated_count"] += cnt
        target["error_count"] += err
        target["success_sum"] += sr * cnt
        target["progress_sum"] += pr * cnt

    def rates(a: dict[str, float]) -> dict[str, Any]:
        cnt = a["evaluated_count"]
        return {
            "evaluated_count": cnt,
            "error_count": a["error_count"],
            METRIC_SUCCESS_RATE: (a["success_sum"] / cnt) if cnt else None,
            METRIC_COMPLETION_RATIO: (a["progress_sum"] / cnt) if cnt else None,
        }

    # Index scored payloads by their subcategory (dataset name with underscores).
    by_sub: dict[str, dict[str, Any]] = {}
    for p in payloads:
        ds = p.get("dataset")
        if ds is not None:
            by_sub[str(ds).replace("_", " ")] = p

    overall = acc()
    high_level_entries: list[dict[str, Any]] = []
    subcategory_entries: list[dict[str, Any]] = []
    missing_subcategories: list[str] = []
    known_subs: set[str] = set()

    for high, subs in task_taxonomy.items():
        high_acc = acc()
        for sub in subs:
            known_subs.add(sub)
            p = by_sub.get(sub)
            if p is None:
                # Not scored in this run: list it explicitly with null rates.
                missing_subcategories.append(_subcategory_to_dataset(sub))
                subcategory_entries.append(
                    {
                        "dataset": _subcategory_to_dataset(sub),
                        "evaluated_count": 0,
                        "error_count": 0,
                        METRIC_SUCCESS_RATE: None,
                        METRIC_COMPLETION_RATIO: None,
                    }
                )
                continue
            cnt = int(p.get("evaluated_count", 0) or 0)
            err = int(p.get("error_count", 0) or 0)
            sr = float(p.get(METRIC_SUCCESS_RATE, 0.0) or 0.0)
            pr = float(p.get(METRIC_COMPLETION_RATIO, 0.0) or 0.0)
            subcategory_entries.append(
                {
                    "dataset": p.get("dataset"),
                    "evaluated_count": cnt,
                    "error_count": err,
                    METRIC_SUCCESS_RATE: sr,
                    METRIC_COMPLETION_RATIO: pr,
                }
            )
            add(high_acc, cnt, err, sr, pr)
            add(overall, cnt, err, sr, pr)
        high_level_entries.append({"category": high, **rates(high_acc)})

    # Scored datasets that are not part of the taxonomy: keep them visible too.
    for sub, p in by_sub.items():
        if sub in known_subs:
            continue
        cnt = int(p.get("evaluated_count", 0) or 0)
        err = int(p.get("error_count", 0) or 0)
        sr = float(p.get(METRIC_SUCCESS_RATE, 0.0) or 0.0)
        pr = float(p.get(METRIC_COMPLETION_RATIO, 0.0) or 0.0)
        subcategory_entries.append(
            {
                "dataset": p.get("dataset"),
                "evaluated_count": cnt,
                "error_count": err,
                METRIC_SUCCESS_RATE: sr,
                METRIC_COMPLETION_RATIO: pr,
            }
        )
        add(overall, cnt, err, sr, pr)
        print(
            f"[summary] dataset '{p.get('dataset')}' is not in the taxonomy; "
            f"counted in overall only.",
            flush=True,
        )

    summary = {
        "overall": rates(overall),
        "high_level": high_level_entries,
        "subcategory": subcategory_entries,
    }
    if missing_subcategories:
        summary["missing_subcategories"] = missing_subcategories
    return summary


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
    payloads: list[dict[str, Any]] = []
    for input_file in input_files:
        output_file, payload = score_one_file(
            input_file=input_file,
            output_dir=args.output_dir,
            translation_root=args.translation_dir,
        )
        payloads.append(payload)
        print(f"Saved: {output_file}", flush=True)

    summary = build_summary(payloads, TASK_TAXONOMY)
    save_json(args.summary_path, summary)
    print(f"Saved summary: {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
