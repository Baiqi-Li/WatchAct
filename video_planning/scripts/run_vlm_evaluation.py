#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import threading
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_models import get_model, list_models
from vlm_models.base import BaseVLM

# Camera perspective prompt template
VLM_evaluation_prompt_camera = """You are a robot with only one arm.
You can execute the following commands:
 1. pick(object, '[source location: a container, a table region, or a position relative to another objects]')
 2. place(object, '[target location: a container, a table region, or a position relative to another objects]')
 3. open(object, '[target part: description of the target drawer or sub-part of the object]')
 4. close(object, '[target part: description of the target drawer or sub-part of the object]')
Task Instruction:
 Watch the provided video. Then follow this instruction:
  "{language_instruction}"
 Your goal is to reason about the video and output the sequence of actions you would take to complete this task.

Spatial Reference Rules:
 When describing positions and directions, always use the viewer's (or camera's/photographer's) perspective as the reference:
 1. The left side corresponds to the camera's left.
 2. The right side corresponds to the camera's right.
 3. The front side refers to the area closer to the camera.
 4. The back side refers to the area farther from the camera.

Output Format:
 Your response should contain:
 1. A list of all the relevant objects involved in your upcoming actions.
 2. A sequence of robot actions in the proper order.

Example:
{
  "Objects": [
    {
      "name": "object_1",
      "description": "[description: containing appearance descriptions and spatial location descriptions.]"
    },
    {
      "name": "object_2",
      "description": "[description: containing appearance descriptions and spatial location descriptions.]"
    }
  ],
  "Actions": [
    {
      "command": "pick",
      "object": "object_1",
      "source_location": "[spatial location description: such as a source container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "place",
      "object": "object_1",
      "target_location": "[spatial location description: such as a target container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "open",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    },
    {
      "command": "close",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    }
  ]
}

Table Region Structure:
(1) Fixture Region: A vertical strip along the left edge of the table for fixed objects (such as a cabinet, tray, or basket). This region is divided into:
[fixture_front_region, fixture_back_region].
(2) Main Movable Objects Region:
The remaining tabletop area to the right of the fixture region, organized as a 3x3 grid for movable objects:
[main_back_left_region, main_back_center_region, main_back_right_region,
 main_middle_left_region, main_middle_center_region, main_middle_right_region,
 main_front_left_region, main_front_center_region, main_front_right_region]

Notes:
 1. When describing spatial positions, please strictly follow the 'Spatial Reference Rules'.
 2. Avoid referring to objects solely by abstract descriptions of past events or actions, such as “where it was previously located” or “the nth time it was picked up.”
 3. Ensure that your output is clear and unambiguous.
 4. Even objects with identical appearances should be treated as separate individual items.

Please output your answer in JSON format."""

# Human perspective prompt template
VLM_evaluation_prompt_human = """You are a robot with only one arm.
You can execute the following commands:
 1. pick(object, '[source location: a container, a table region, or a position relative to another objects]')
 2. place(object, '[target location: a container, a table region, or a position relative to another objects]')
 3. open(object, '[target part: description of the target drawer or sub-part of the object]')
 4. close(object, '[target part: description of the target drawer or sub-part of the object]')
Task Instruction:
 Watch the provided video. Then follow this instruction:
  "{language_instruction}"
 Your goal is to reason about the video and output the sequence of actions you would take to complete this task.
Spatial Reference Rules:
When describing object positions, always use the person in the video as the reference point:
1. The front side of the table is the side closest to the person.
2. The back side of the table is the side farthest from the person.
3. The right side of the table corresponds to the right-hand side of the person.
4. The left side of the table corresponds to the left-hand side of the person.
Output Format:
 Your response should contain:
 1. A list of all the relevant objects involved in your upcoming actions.
 2. A sequence of robot actions in the proper order.
Example:
{
  "Objects": [
    {
      "name": "object_1",
      "description": "[description: containing appearance descriptions or location descriptions.]"
    },
    {
      "name": "object_2",
      "description": "[description: containing appearance descriptions or location descriptions.]"
    }
  ],
  "Actions": [
    {
      "command": "pick",
      "object": "object_1",
      "source_location": "[spatial location description: such as a source container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "place",
      "object": "object_1",
      "target_location": "[spatial location description: such as a target container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "open",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    },
    {
      "command": "close",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    }
  ]
}

Table Region Structure:
(1) Fixture Region: From the perspective of the person in the video, a vertical strip along the right edge of the table for fixed objects (such as a cabinet, tray, or basket). This region is divided into:
[fixture_front_region, fixture_back_region].
(2) Main Movable Objects Region:
From the perspective of the person in the video, the remaining tabletop area to the left of the fixture region, organized as a 3x3 grid for movable objects:
[main_back_left_region, main_back_center_region, main_back_right_region,
main_middle_left_region, main_middle_center_region, main_middle_right_region,
main_front_left_region, main_front_center_region, main_front_right_region]

Notes:
 1. When describing spatial positions, please strictly follow the 'Spatial Reference Rules'.
 2. Avoid referring to objects solely by abstract descriptions of past events or actions, such as “where it was previously located” or “the nth time it was picked up.”
 3. Ensure that your output is clear and unambiguous.
 4. Even objects with identical appearances should be treated as separate individual items.
Please output your answer in JSON format."""


# Side-camera perspective prompt template (the "side" view, formerly "left";
# selected when camera_perspective == "side").
VLM_evaluation_prompt_left_camera = """You are a robot with only one arm.
You can execute the following commands:
 1. pick(object, '[source location: a container, a table region, or a position relative to another objects]')
 2. place(object, '[target location: a container, a table region, or a position relative to another objects]')
 3. open(object, '[target part: description of the target drawer or sub-part of the object]')
 4. close(object, '[target part: description of the target drawer or sub-part of the object]')
Task Instruction:
 Watch the provided video. Then follow this instruction:
  "{language_instruction}"
 Your goal is to reason about the video and output the sequence of actions you would take to complete this task.

Spatial Reference Rules:
 When describing positions and directions, always use the viewer's (or camera's/photographer's) perspective as the reference:
 1. The left side corresponds to the camera's left.
 2. The right side corresponds to the camera's right.
 3. The front side refers to the area closer to the camera.
 4. The back side refers to the area farther from the camera.

Output Format:
 Your response should contain:
 1. A list of all the relevant objects involved in your upcoming actions.
 2. A sequence of robot actions in the proper order.

Example:
{
  "Objects": [
    {
      "name": "object_1",
      "description": "[description: containing appearance descriptions and spatial location descriptions.]"
    },
    {
      "name": "object_2",
      "description": "[description: containing appearance descriptions and spatial location descriptions.]"
    }
  ],
  "Actions": [
    {
      "command": "pick",
      "object": "object_1",
      "source_location": "[spatial location description: such as a source container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "place",
      "object": "object_1",
      "target_location": "[spatial location description: such as a target container (basket, wooden cabinet, or cabinet), a table region, or a position relative to other objects]"
    },
    {
      "command": "open",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    },
    {
      "command": "close",
      "object": "object_2",
      "target_location": "[object_2's top drawer/middle drawer/bottom drawer]"
    }
  ]
}

Table Region Structure:
(1) Fixture Region: A horizontal strip along the back edge of the table for fixed objects, such as a cabinet, tray, or basket. This region is divided into:
[fixture_back_left_region, fixture_back_right_region].
(2) Main Movable Objects Region:
The remaining tabletop area in front of the fixture region, organized as a 3x3 grid for movable objects:
[main_back_left_region, main_back_center_region, main_back_right_region,
main_middle_left_region, main_middle_center_region, main_middle_right_region,
main_front_left_region, main_front_center_region, main_front_right_region]

Notes:
 1. When describing spatial positions, please strictly follow the 'Spatial Reference Rules'.
 2. Avoid referring to objects solely by abstract descriptions of past events or actions, such as “where it was previously located” or “the nth time it was picked up.”
 3. Ensure that your output is clear and unambiguous.
 4. Even objects with identical appearances should be treated as separate individual items.

Please output your answer in JSON format."""


# The WatchAct dataset (downloaded from HuggingFace ``WatchAct/data``) is laid
# out as:
#   data/
#   ├── data/*.jsonl          <- evaluation rows (one file per task)
#   ├── meta_data/*.json
#   ├── videos/<task>/...
#   └── bddl_files/<task>/...
# A "task" name is a jsonl file stem (e.g. ``Ordinal``, ``Fine-Grained_Action``).
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATA_DIR = DEFAULT_DATASET_ROOT / "data"      # the jsonl directory
DEFAULT_VIDEO_ROOT = DEFAULT_DATASET_ROOT / "videos"


def discover_datasets(data_dir: Path) -> list[str]:
    """Return sorted dataset names (jsonl file stems) found under *data_dir*."""
    if not data_dir.is_dir():
        return []
    return sorted(p.stem for p in data_dir.glob("*.jsonl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VLM evaluation on JSONL datasets. Prompt template is selected by "
            "spatial_reference (human/camera)."
        )
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Root directory for relative video paths in JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/action_plan",
        help="Directory to save evaluation outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of samples per dataset (takes first N rows).",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=None,
        help=(
            "Randomly sample N original_id groups per dataset "
            "(keep all rows in selected groups). "
            "If both --max-samples and --random-samples are set, "
            "--random-samples takes priority."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random-samples group selection (default: %(default)s).",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save extracted frames while evaluating. Default is False.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="Fixed max frames sampled per video (default: %(default)s).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional dataset names (jsonl file stems) to run. "
            "Default: every *.jsonl under --data-dir. "
            "Example: --datasets Ordinal"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory holding the dataset jsonl files. Each dataset is read "
            "from {data_dir}/{dataset}.jsonl (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads for parallel evaluation within each dataset "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--parallel-group-by",
        choices=["original_id", "row"],
        default="original_id",
        help=(
            "Parallelization unit when --num-workers > 1. "
            "'original_id' runs rows with the same original_id serially in one worker; "
            "'row' runs every row independently."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help=(
            "Save checkpoint to output file every N processed rows "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If set and the target output file exists, reuse rows whose "
            "(original_id, camera_perspective, spatial_reference) triple already "
            "appears as status=='ok'. Only newly added / previously failed rows "
            "are (re)evaluated. Reused entries keep their action_plan but get "
            "their `id` field refreshed to the current sampled-row id."
        ),
    )
    parser.add_argument(
        "--reuse-ignore-fields",
        nargs="+",
        default=[],
        choices=list(FINGERPRINT_FIELDS),
        help=(
            "Fields to DROP from the reuse fingerprint when matching cached "
            "rows under --skip-existing. Pick any subset of: "
            f"{list(FINGERPRINT_FIELDS)}. Example: "
            "`--reuse-ignore-fields language_instruction` lets you reuse rows "
            "whose prompt text was rewritten while everything else matches."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt_schema",
        help=(
            "Model name registered in vlm_models/ "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the default model ID (e.g. gpt-4o, gpt-5.2, gemini-3-pro-preview).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Video sampling FPS. For FILE_PATH models (gemini, qwen) this is "
            "forwarded to the model/server. For FRAMES models (gpt) it drives "
            "client-side time_based frame sampling (rounded down to an integer "
            ">= 1, then capped by --max-frames). Default None means each model's "
            "own default (Gemini server-side ~1 FPS, Qwen 1, gpt sampling 1 FPS)."
        ),
    )
    parser.add_argument(
        "--frame-short-side",
        type=int,
        default=None,
        help=(
            "For FRAMES models (gpt): downscale each sampled frame so its "
            "shorter side is this many pixels (keeping aspect ratio, capped at "
            "max_long_side=2000) before base64 encoding. Source clips are 1440p, "
            "so e.g. 768 cuts upload bandwidth/memory ~3-4x. NOTE: with "
            "detail='auto' GPT renormalizes to a 768px short side internally, so "
            "this mainly reduces upload/latency, not token billing. Default None "
            "keeps the original full resolution (no resize)."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def get_original_id_group_key(row: dict[str, Any], fallback_idx: int) -> str:
    original_id = row.get("original_id")
    if original_id is None:
        return f"__row_{fallback_idx}"
    key = str(original_id).strip()
    return key or f"__row_{fallback_idx}"


def resolve_video_path(
    video_value: str,
    video_root: Path,
    *,
    dataset_name: str | None = None,
) -> Path:
    candidate = Path(video_value)
    if candidate.is_absolute():
        return candidate

    default_path = video_root / candidate
    if default_path.exists() or dataset_name is None:
        return default_path

    dataset_scoped_path = video_root / dataset_name / candidate
    if dataset_scoped_path.exists():
        return dataset_scoped_path

    return default_path


def choose_prompt_template(spatial_reference: str, camera_perspective: Any = None) -> str:
    spatial = str(spatial_reference or "").strip().lower()
    camera = str(camera_perspective or "").strip().lower()

    if spatial == "camera" and camera == "side":
        return VLM_evaluation_prompt_left_camera
    if spatial == "camera":
        return VLM_evaluation_prompt_camera
    if spatial == "human":
        return VLM_evaluation_prompt_human
    raise ValueError(f"Unsupported spatial_reference: {spatial_reference}")


def render_prompt(template: str, language_instruction: str) -> str:
    marker = "{language_instruction}"
    if marker not in template:
        raise ValueError(f"Prompt template missing placeholder: {marker}")
    # Use direct replacement to avoid conflicts with literal JSON braces in prompt examples.
    return template.replace(marker, language_instruction)


def get_camera_perspective(row: dict[str, Any]) -> Any:
    value = row.get("camera_perspective")
    if value is None:
        value = row.get("camera_perspective")
    return value


def evaluate_one_row(
    row: dict[str, Any],
    *,
    dataset_name: str,
    model: BaseVLM,
    video_root: Path,
    max_frames: int,
    save_frames: bool,
    sampling_fps: int = 1,
    frame_short_side: int | None = None,
) -> dict[str, Any]:
    language_instruction = str(row.get("language_instruction", "")).strip()
    spatial_reference = str(row.get("spatial_reference", "")).strip()
    camera_perspective = get_camera_perspective(row)
    video = str(row.get("video", "")).strip()
    if not language_instruction:
        raise ValueError("Missing language_instruction.")
    if not video:
        raise ValueError("Missing video.")

    template = choose_prompt_template(spatial_reference, camera_perspective)
    question = render_prompt(template, language_instruction)
    resolved_video_path = resolve_video_path(
        video,
        video_root,
        dataset_name=dataset_name,
    )
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_video_path}")

    max_frames = max(1, int(max_frames))
    sampling_fps = max(1, int(sampling_fps))
    print(
        f"Sampling config: mode=time_based, sampling_value={sampling_fps}, "
        f"max_frames={max_frames}, frame_short_side={frame_short_side}",
        flush=True,
    )
    # Pass resize args only for FRAMES-type models; FILE_PATH models (gemini)
    # ignore **extra in prepare_video, so this is harmless when unused.
    resize_kwargs: dict[str, Any] = {}
    if frame_short_side is not None and frame_short_side > 0:
        resize_kwargs = {"resize_mode": "short_side", "short_side": int(frame_short_side)}
    video_input = model.prepare_video(
        str(resolved_video_path),
        max_frames=max_frames,
        mode="time_based",
        sampling_value=sampling_fps,
        save_frames=save_frames,
        **resize_kwargs,
    )
    action_plan_json = model.query(
        question=question,
        video_input=video_input,
        json_schema_name="action_plan",
    )

    return {
        "id": row.get("id"),
        "original_id": row.get("original_id"),
        "language_instruction": language_instruction,
        "camera_perspective": camera_perspective,
        "spatial_reference": spatial_reference,
        "video": video,
        "video_path": str(resolved_video_path),
        "question": question,
        "action_plan_json": action_plan_json,
    }


FINGERPRINT_FIELDS: tuple[str, ...] = (
    "original_id",
    "camera_perspective",
    "spatial_reference",
    "video",
    "language_instruction",
)


def _row_fingerprint(
    row: dict[str, Any],
    ignore_fields: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    def _field(name: str) -> Any:
        if name in ignore_fields:
            return None
        if name == "camera_perspective":
            return get_camera_perspective(row)
        return row.get(name)

    return tuple(_field(name) for name in FINGERPRINT_FIELDS)


def _load_reusable_results(
    output_path: Path,
    ignore_fields: frozenset[str] = frozenset(),
) -> dict[tuple[Any, ...], dict[str, Any]]:
    # Reads not only output_path itself, but also every sibling file in the same
    # directory whose name starts with output_path.stem. This lets a full-data
    # run reuse rows from prior random-sampled runs (e.g. a new
    # `<model>_<dataset>.json` picks up rows from
    # `<model>_<dataset>_random20_seed12345.json`).
    sources: list[Path] = []
    if output_path.is_file():
        sources.append(output_path)
    parent = output_path.parent
    if parent.is_dir():
        try:
            output_resolved = output_path.resolve()
        except Exception:
            output_resolved = output_path
        for sibling in sorted(parent.glob(f"{output_path.stem}*.json")):
            try:
                if sibling.resolve() == output_resolved:
                    continue
            except Exception:
                if sibling == output_path:
                    continue
            sources.append(sibling)

    reusable: dict[tuple[Any, ...], dict[str, Any]] = {}
    collisions = 0
    for source in sources:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"[skip-existing] Failed to load {source}: {exc}; "
                "treating as empty.",
                flush=True,
            )
            continue
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "ok":
                continue
            fp = _row_fingerprint(item, ignore_fields)
            if fp in reusable:
                collisions += 1
                continue
            reusable[fp] = item
    if collisions and ignore_fields:
        print(
            f"[skip-existing] WARNING: {collisions} cached row(s) collided on the "
            f"reduced fingerprint (ignored={sorted(ignore_fields)}); kept the first.",
            flush=True,
        )
    return reusable


def evaluate_dataset(
    dataset_name: str,
    jsonl_path: Path,
    *,
    model: BaseVLM,
    video_root: Path,
    max_frames: int,
    sampling_fps: int = 1,
    frame_short_side: int | None = None,
    max_samples: int | None,
    random_samples: int | None = None,
    seed: int = 42,
    save_frames: bool,
    num_workers: int,
    parallel_group_by: str,
    output_path: Path,
    save_every: int,
    skip_existing: bool = False,
    reuse_ignore_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    rows = read_jsonl(jsonl_path)
    if random_samples is not None:
        rng = random.Random(seed)
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for idx, row in enumerate(rows):
            group_key = get_original_id_group_key(row, idx)
            grouped_rows.setdefault(group_key, []).append(row)

        total_groups = len(grouped_rows)
        n = min(random_samples, total_groups)
        selected_group_keys = set(rng.sample(list(grouped_rows.keys()), n))
        rows = [
            row
            for idx, row in enumerate(rows)
            if get_original_id_group_key(row, idx) in selected_group_keys
        ]
        print(
            f"[{dataset_name}] Randomly sampled {n}/{total_groups} original_id groups, "
            f"{len(rows)} rows (seed={seed})",
            flush=True,
        )
    elif max_samples is not None:
        rows = rows[:max_samples]

    total_rows = len(rows)
    ordered_results: list[dict[str, Any] | None] = [None] * total_rows
    processed_count = 0
    next_checkpoint_at = save_every

    if skip_existing and total_rows > 0:
        reusable = _load_reusable_results(output_path, reuse_ignore_fields)
        if reuse_ignore_fields:
            print(
                f"[{dataset_name}] skip-existing: ignoring fields "
                f"{sorted(reuse_ignore_fields)} when matching cached rows",
                flush=True,
            )
        if reusable:
            reused_count = 0
            for idx, row in enumerate(rows):
                cached = reusable.get(_row_fingerprint(row, reuse_ignore_fields))
                if cached is None:
                    continue
                refreshed = dict(cached)
                refreshed["id"] = row.get("id")
                ordered_results[idx] = refreshed
                reused_count += 1
            processed_count = reused_count
            while next_checkpoint_at <= processed_count:
                next_checkpoint_at += save_every
            print(
                f"[{dataset_name}] skip-existing: reused {reused_count}/{total_rows} rows "
                f"from {output_path}",
                flush=True,
            )

    def collect_completed_results() -> list[dict[str, Any]]:
        return [item for item in ordered_results if item is not None]

    def build_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "dataset": dataset_name,
            "source_jsonl": str(jsonl_path),
            "total": total_rows,
            "processed": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "error": sum(1 for r in results if r["status"] == "error"),
            "results": results,
        }

    def maybe_save_checkpoint() -> None:
        nonlocal next_checkpoint_at
        if save_every <= 0:
            return
        if processed_count < next_checkpoint_at:
            return
        payload = build_payload(collect_completed_results())
        save_json(output_path, payload)
        print(
            f"[{dataset_name}] Checkpoint saved at {processed_count}/{total_rows}: {output_path}",
            flush=True,
        )
        while processed_count >= next_checkpoint_at:
            next_checkpoint_at += save_every

    def build_error_item(row: dict[str, Any], error_message: str) -> dict[str, Any]:
        return {
            "id": row.get("id"),
            "original_id": row.get("original_id"),
            "camera_perspective": get_camera_perspective(row),
            "spatial_reference": row.get("spatial_reference"),
            "video": row.get("video"),
            "status": "error",
            "error": error_message,
            "action_plan_json": {},
        }

    _thread_local = threading.local()

    def _get_thread_model() -> BaseVLM:
        """Return a per-thread model clone to avoid sharing one API client."""
        if not hasattr(_thread_local, "model"):
            _thread_local.model = model.clone()
        return _thread_local.model

    def run_single(row_index: int, row: dict[str, Any]) -> dict[str, Any]:
        sample_id = row.get("id")
        print(f"[{dataset_name}] {row_index+1}/{total_rows} id={sample_id}", flush=True)
        thread_model = _get_thread_model() if num_workers > 1 else model
        try:
            item = evaluate_one_row(
                row,
                dataset_name=dataset_name,
                model=thread_model,
                video_root=video_root,
                max_frames=max_frames,
                save_frames=save_frames,
                sampling_fps=sampling_fps,
                frame_short_side=frame_short_side,
            )
            item["status"] = "ok"
            item["error"] = None
        except Exception as exc:
            item = build_error_item(row, str(exc))
            print(f"[{dataset_name}] id={sample_id} failed: {exc}", flush=True)
        return item

    if total_rows == 0:
        results: list[dict[str, Any]] = []
    elif num_workers <= 1:
        for idx, row in enumerate(rows):
            if ordered_results[idx] is not None:
                continue
            ordered_results[idx] = run_single(idx, row)
            processed_count += 1
            maybe_save_checkpoint()
        results = collect_completed_results()
    else:
        max_workers = max(1, int(num_workers))
        pending = [(idx, row) for idx, row in enumerate(rows) if ordered_results[idx] is None]
        if parallel_group_by == "original_id":
            grouped_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for idx, row in pending:
                group_key = get_original_id_group_key(row, idx)
                grouped_rows.setdefault(group_key, []).append((idx, row))
            work_units = list(grouped_rows.values())
            print(
                f"[{dataset_name}] Parallel mode: workers={max_workers}, "
                f"group_by=original_id, groups={len(work_units)}, rows={total_rows} "
                f"(pending={len(pending)})",
                flush=True,
            )
        else:
            work_units = [[(idx, row)] for idx, row in pending]
            print(
                f"[{dataset_name}] Parallel mode: workers={max_workers}, "
                f"group_by=row, groups={len(work_units)}, rows={total_rows} "
                f"(pending={len(pending)})",
                flush=True,
            )

        def run_work_unit(
            unit: list[tuple[int, dict[str, Any]]],
        ) -> list[tuple[int, dict[str, Any]]]:
            return [(idx, run_single(idx, row)) for idx, row in unit]

        base_processed = processed_count
        delta_completed = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for unit in work_units:
                futures[executor.submit(run_work_unit, unit)] = unit

            for future in as_completed(futures):
                unit = futures[future]
                try:
                    unit_results = future.result()
                except Exception as exc:
                    unit_results = [
                        (
                            idx,
                            build_error_item(
                                row,
                                f"Parallel worker crashed: {exc}",
                            ),
                        )
                        for idx, row in unit
                    ]
                    print(f"[{dataset_name}] Worker crashed: {exc}", flush=True)

                for idx, item in unit_results:
                    ordered_results[idx] = item
                delta_completed += len(unit_results)
                processed_count = base_processed + delta_completed
                print(
                    f"[{dataset_name}] Progress: {processed_count}/{total_rows}",
                    flush=True,
                )
                maybe_save_checkpoint()

        unresolved_indices = [idx for idx, item in enumerate(ordered_results) if item is None]
        for idx in unresolved_indices:
            ordered_results[idx] = build_error_item(
                rows[idx],
                "Internal error: missing parallel result.",
            )
            processed_count += 1
        results = collect_completed_results()
        maybe_save_checkpoint()
    return build_payload(results)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_filename_token(value: str) -> str:
    token = str(value).strip()
    if not token:
        return "unknown_model"
    return token.replace("/", "_").replace("\\", "_")


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    # Pick the right API key for the chosen model.
    _API_KEY_ENV: dict[str, str] = {
        "gpt": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "qwen_thinking": "DASHSCOPE_API_KEY",
        "internvl": "INTERNVL_API_KEY",
    }
    api_key_env = _API_KEY_ENV.get(args.model)
    api_key = os.getenv(api_key_env) if api_key_env else None

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.save_every <= 0:
        raise ValueError("--save-every must be >= 1")

    model_kwargs: dict[str, Any] = {}
    if api_key is not None:
        model_kwargs["api_key"] = api_key
    if args.model_id is not None:
        model_kwargs["model_id"] = args.model_id
    if args.fps is not None:
        model_kwargs["fps"] = args.fps
    model = get_model(args.model, **model_kwargs)
    effective_model_id = args.model_id or getattr(model, "model_id", None) or args.model
    model_file_tag = sanitize_filename_token(str(effective_model_id))
    print(
        f"Using model: {args.model} (model_id={effective_model_id}, available: {list_models()})",
        flush=True,
    )

    effective_save_frames = args.save_frames
    if args.num_workers > 1 and args.save_frames:
        print(
            "num_workers > 1 detected; forcing save_frames=False to avoid "
            "concurrent frame-write conflicts.",
            flush=True,
        )
        effective_save_frames = False

    selected_datasets = (
        args.datasets if args.datasets is not None else discover_datasets(args.data_dir)
    )
    if not selected_datasets:
        raise SystemExit(
            f"No datasets to run. No *.jsonl found under {args.data_dir} "
            "(did you download WatchAct/data into it?)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_id": str(effective_model_id),
        "video_root": str(args.video_root),
        "num_workers": args.num_workers,
        "parallel_group_by": args.parallel_group_by,
        "save_every": args.save_every,
        "save_frames_requested": args.save_frames,
        "save_frames_effective": effective_save_frames,
        "datasets": {},
    }

    # Build filename suffix: include random/seed info when applicable.
    if args.random_samples is not None:
        sampling_tag = f"_random{args.random_samples}_seed{args.seed}"
    else:
        sampling_tag = ""

    # For FRAMES-type models (e.g. gpt), --fps drives client-side time_based
    # frame sampling (sampling_value == FPS). For FILE_PATH models (qwen/gemini)
    # --fps is forwarded via model_kwargs instead and this value is unused.
    sampling_fps = int(args.fps) if args.fps is not None else 1

    for dataset_name in selected_datasets:
        jsonl_path = args.data_dir / f"{dataset_name}.jsonl"
        task_dir = args.output_dir / dataset_name
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{model_file_tag}_{dataset_name}{sampling_tag}.json"
        dataset_result = evaluate_dataset(
            dataset_name,
            jsonl_path,
            model=model,
            video_root=args.video_root,
            max_frames=args.max_frames,
            sampling_fps=sampling_fps,
            frame_short_side=args.frame_short_side,
            max_samples=args.max_samples,
            random_samples=args.random_samples,
            seed=args.seed,
            save_frames=effective_save_frames,
            num_workers=args.num_workers,
            parallel_group_by=args.parallel_group_by,
            output_path=output_path,
            save_every=args.save_every,
            skip_existing=args.skip_existing,
            reuse_ignore_fields=frozenset(args.reuse_ignore_fields),
        )
        summary["datasets"][dataset_name] = {
            "source_jsonl": dataset_result["source_jsonl"],
            "total": dataset_result["total"],
            "processed": dataset_result["processed"],
            "ok": dataset_result["ok"],
            "error": dataset_result["error"],
        }
        save_json(output_path, dataset_result)
        print(f"Saved: {output_path}")

    summary_path = args.output_dir / f"{model_file_tag}_results_summary{sampling_tag}.json"
    save_json(summary_path, summary)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
