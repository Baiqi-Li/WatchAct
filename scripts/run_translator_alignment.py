#!/usr/bin/env python3
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import threading
import os
import re
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


def read_json_file(file_path):
    """Read a JSON file and return parsed data, or None if it does not exist."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
You will receive:
(1) Reference Information:
   - Region descriptions
   - Object appearance descriptions
(2) Task Information:
   - A canonical object list
   - An init_state specifying each object and its current location region
(3) A Description of Object Manipulations:
   - An object list with descriptions
   - An action plan describing operations on those objects and regions
Your task is to use (1) and (2) as references to align the objects and regions mentioned in (3) with the corresponding objects in the init_state of the Task Information.
Your job is to: \n
- Match each described object based to at most one task object in init_state of TASK_INFORMATION (and vice versa).\n
- Match each described object to the object in the init_state with the most similar appearance and region description.\n
- During the matching process, allow a fuzzy matching. If an object's appearance description and region description both roughly correspond to an object and its region in the init_state of TASK_INFORMATION, match the object to that entry in the init_state. \n
- During the matching process, if multiple described objects could match the same entry in the init_state based on their appearance and region descriptions, select the one that is most similar for the match.
- After an object is matched to an entry in the init_state, use the object's canonical name and position from the init_state to represent its initial state. \n
- Container Location Override Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (For example, output: wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.
- After establishing a correspondence, every reference to that object and location in the entire object-manipulation action plan description must use the same unified name.\n
Region Descriptions:\n
{ALL_REGION_DESCRIPTIONS}\n
Some object appearance descriptions for reference:\n
{OBJECT_DESCRIPTIONS}\n
Task Information:\n
{TASK_INFORMATION}\n
Description of Object Manipulations:\n
{DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n
Please perform the matching and produce the updated description of object manipulations. Always respond ONLY with valid JSON that fits the schema.\n"""

# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.

# I have:
# - A structured definition of a real-world scene.
# - A VLM model's description of the objects in that scene and its predicted action sequence.

# Your role is to align the VLM output with the structured scene definition.  
# This alignment is used to standardize (format) the VLM output and to evaluate and compare the performance of different VLM models.

# If the VLM output significantly deviates from the real scene, the alignment should be considered a failure.  
# However, allow a small degree of fuzzy matching when the descriptions are reasonably close and logically consistent.

# You will receive:

# 1) Candidate Reference Set (for matching guidance):
#    - Region descriptions (candidate region identifiers)
#    - Object appearance descriptions (candidate object identities)

# 2) Task Information (real-world ground truth):
#    - A canonical object list
#    - An init_state specifying each object and its corresponding region

# 3) A Description of Object Manipulations (VLM output):
#    - An object list with descriptions
#    - An action plan describing operations on those objects and regions

# Your task is to align the objects and region phrases in (3) to the canonical objects and region identifiers defined in (1), using (2) as structural ground-truth reference and consistency constraint.

# Rules:
# - Match each described object to at most one task object (and vice versa).
# - Match each natural language location / region phrase to at most one canonical region id.
# - When an object or location cannot be matched, mark it as "unmatched".
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.
# - Region Priority Rule:
#   When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.
# - Initial-State Anchor Rule (Limited Override):
#   During the matching process, allow a small degree of fuzzy matching.
#   If an object's appearance description and region description both roughly correspond to an object and its region in the init_state of TASK_INFORMATION, match the object to that entry in the init_state.
#   Important: If two regions contain opposite directional terms (such as left vs. right, or front vs. back), they must never be matched to each other. This applies both between a region description and a region_id, and between two region_ids:
#   - any pair among [main_back_right_region, main_back_left_region, main_front_left_region, main_front_right_region]
#   - any pair among [main_back_center_region, main_front_center_region]
#   - any pair among [main_middle_left_region, main_middle_right_region]
# - Consistency Rule of Action Alignment:
#   For each object, when executing a pick action, the source_region_id should match the object's current state.
#   In particular, the first pick action's source region should be consistent with the object's unified_region.

# Region Descriptions:
# {ALL_REGION_DESCRIPTIONS}
# Some object appearance descriptions for reference:
# {OBJECT_DESCRIPTIONS}
# Task Information:
# {TASK_INFORMATION}
# Description of Object Manipulations:
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}
# Please perform the matching and produce the updated description of object manipulations.
# Always answer ONLY with valid JSON that fits the required schema."""


# You are an object-and-region alignment engine.

# You will receive:

# 1) Reference Information:
#    - Region descriptions
#    - Object appearance descriptions

# 2) Task Information:
#    - A canonical object list
#    - An init_state specifying each object and its corresponding region

# 3) A Description of Object Manipulations:
#    - An object list with descriptions
#    - An action plan describing operations on those objects and regions

# Your task is to use (1) and (2) as references to align the objects and regions mentioned in (3) to the canonical identifiers and initial states defined in (2).


# 28% ACCURACY PROMPT
# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:

# 1) Reference Information:
#    - Region descriptions
#    - Object appearance descriptions

# 2) Task Information:
#    - A canonical object list
#    - An init_state specifying each object and its corresponding region

# 3) A Description of Object Manipulations:
#    - An object list with descriptions
#    - An action plan describing operations on those objects and regions

# Your task is to use (1) and (2) as references to align the objects and regions mentioned in (3) to the canonical identifiers and initial states defined in (1).

# Your job is to:
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Initial-State Anchor Rule (Limited Override):
#   During the matching process, allow a small degree of fuzzy matching. If an object's appearance description and region description both roughly correspond to an object and its region in the init_state of TASK_INFORMATION, match the object to that entry in the init_state. For example, a region description such as "main_front_center_region" may, when necessary and based on the init_state information, be matched to the corresponding main_front_left_region.
#   IMPORTANT: Never matching in the following cases (including between a region description and a region_id, or between two region_ids):
#   - any pair among [main_back_right_region, main_back_left_region, main_front_left_region, main_front_right_region]
#   - any pair among [main_back_center_region, main_front_center_region]
#   - any pair among [main_middle_left_region, main_middle_right_region]
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region.
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""

# 20% ACCURACY PROMPT
# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:\n
# 1) Task Information: canonical object identifiers and region identifiers.\n
# 2) A description of Object Manipulations: A free-form description of objects and actions, consisting of two parts:(1) an object list with descriptions, and (2) an action plan involving those objects.\n
# \nYour job is to:\n
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - Initial-State Anchor Rule (Limited Override):
#   During the matching process, allow a small degree of fuzzy matching. If an object's appearance description and region description both roughly correspond to an object and its region in the init_state of TASK_INFORMATION, match the object to that entry in the init_state. For example, a region description such as "main_front_center_region" may, when necessary and based on the init_state information, be matched to the corresponding main_front_left_region.
#   IMPORTANT: Never override in the following cases (including between a region description and a region_id, or between two region_ids):
#   - any pair among [main_back_right_region, main_back_left_region, main_front_left_region, main_front_right_region]
#   - any pair among [main_back_center_region, main_front_center_region]
#   - any pair among [main_middle_left_region, main_middle_right_region]
#   If the regions are not directly adjacent, keep the text-mentioned aligned region.
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region.
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""



# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:\n
# 1) A task definition: canonical object identifiers and region identifiers.\n
# 2) A description of Object Manipulations: A free-form description of objects and actions, consisting of two parts:(1) an object list with descriptions, and (2) an action plan involving those objects.\n
# \nYour job is to:\n
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - Initial-State Anchor Rule:
#   During the matching process, allow a certain degree of fuzzy matching. You may override a object's region using init_state of TASK_INFORMATION ONLY when the text-mentioned table-grid region(or the textual description of the object's region) and the init_state region are the same region or very close.
#   Never override in the following cases (including between a region description and a region_id, or between two region_ids):
#   - any pair among [main_back_right_region, main_back_left_region, main_front_left_region, main_front_right_region]
#   - any pair among [main_back_center_region, main_front_center_region]
#   - any pair among [main_middle_left_region, main_middle_right_region]
#   If the regions are not directly adjacent, keep the text-mentioned aligned region.
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region.
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""



# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:\n
# 1) A task definition: canonical object identifiers and region identifiers.\n
# 2) A description of Object Manipulations: A free-form description of objects and actions, consisting of two parts:(1) an object list with descriptions, and (2) an action plan involving those objects.\n
# \nYour job is to:\n
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - Initial-State Anchor Rule (Limited Override):
#   During the matching process, allow a certain degree of fuzzy matching. You may override a text-mentioned table-grid region using init_state of TASK_INFORMATION only if the two regions are directly adjacent.
#   Directly adjacent means: center ↔ left, center ↔ right
#   Never override: left ↔ right, front ↔ back
#   - any regions that differ in row or column (e.g., front_right ↔ front_left, back_right ↔ front_right).
#   If the regions are not directly adjacent, keep the text-mentioned aligned region.
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region.
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""

# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:\n
# 1) A task definition: canonical object identifiers and region identifiers.\n
# 2) A description of Object Manipulations: A free-form description of objects and actions, consisting of two parts:(1) an object list with descriptions, and (2) an action plan involving those objects.\n
# \nYour job is to:\n
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - Initial-State Anchor Rule: During the matching process, allow a certain degree of fuzzy matching. If an object's appearance description and region description roughly correspond to an object and its region in the init_state of TASK_INFORMATION, match the object to that entry in the init_state. For example, a region description such as "main_front_center_region, slightly to the left" may, when necessary and based on the init_state information, be matched to the corresponding main_front_left_region. Never allow fuzzy matching between opposite directions such as "front" and "back," or "left" and "right."
# - Strictly prohibit fuzzy matching between regions with opposite directions, such as "front" vs. "back" or "left" vs. "right." For example, a description like "near the back-right area of the table (main_back_right_region)." must never match any region ID containing "front" or "left, such as main_front_right_region, and main_back_left_region."
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region.
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""

# TRANSLATOR_PROMPT_TEMPLATE = """You are an object-and-region alignment engine.\n
# You will receive:\n
# 1) A task definition: canonical object identifiers and region identifiers.\n
# 2) A description of Object Manipulations: A free-form description of objects and actions, consisting of two parts:(1) an object list with descriptions, and (2) an action plan involving those objects.\n
# \nYour job is to:\n
# - Match each described object to at most one task object (and vice versa).\n
# - Match each natural language location / region phrase to at most one canonical region id.\n
# - When an object or location cannot be matched, mark it as 'unmatched'.\n
# - After establishing a correspondence, every reference to that object and location in the entire object-manipulation description must use the same unified name.\n
# - Region Priority Rule: When a phrase specifies that an object is inside or on a container (e.g., tray, basket, cabinet drawer), and also mentions the container's position on the table, output ONLY the container region id (e.g., wooden_tray_<n>_contain_region, basket_<n>_contain_region, wooden_cabinet_<n>_top_region, wooden_cabinet_<n>_middle_region, wooden_cabinet_<n>_bottom_region, wooden_cabinet_<n>_top_side) when determining the object's location.\n
# - unified_region Definition: The `unified_region` of each matched object MUST be set to the region from the object's source description (i.e., the described location in the Description of Object Manipulations), NOT the init_state region. The init_state is only used to help disambiguate which task object a described object corresponds to.
# - Initial-State Anchor Rule (HARD CONSTRAINT): Object matching requires BOTH appearance AND spatial location to be consistent. Specifically:
#   (a) Fuzzy spatial matching is allowed for adjacent or nearby regions (e.g., "main_front_center_region, slightly to the left" may match main_front_left_region).
#   (b) NEVER match an object whose described region has an opposite direction to the init_state region. "front" vs "back" or "left" vs "right" are OPPOSITE and must NEVER be fuzzy-matched. For example, if the description says an object is at main_back_right_region but the init_state says the candidate task object is at main_front_left_region, this match is FORBIDDEN even if the appearance matches perfectly.
#   (c) When appearance matches but spatial location is contradictory (opposite direction), mark the object as unmatched rather than forcing an incorrect spatial alignment.
# - Consistency Rule of Action Alignment: For each object, when executing a pick action, the source_region_id should match the object's current state. In particular, the first pick action's source region should be consistent with the object's unified_region (which comes from the source description).
# Region Descriptions:\n
# {ALL_REGION_DESCRIPTIONS}\n\n
# Some object appearance descriptions for reference:\n
# {OBJECT_DESCRIPTIONS}\n\n
# Task Information:\n
# {TASK_INFORMATION}\n\n
# Description of Object Manipulations:\n
# {DESCRIPTION_OF_OBJECT_MANIPULATIONS}\n\n
# Please perform the matching and produce the updated description of object manipulations, always answer ONLY with valid JSON that fits the schema.\n"""



SUPPORTED_ACTION_COMMANDS = {"pick", "place", "open", "close"}


def _is_unmatched_object_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip().lower()
    return "unmatch" in token


def collect_translator_action_anomalies(
    translator_output: dict[str, Any],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    actions = translator_output.get("actions")
    if actions is None:
        actions = translator_output.get("Actions", [])
    if not isinstance(actions, list):
        anomalies.append(
            {
                "type": "invalid_actions_field",
                "message": "translator_output.actions must be a list",
            }
        )
        return anomalies

    for action_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            anomalies.append(
                {
                    "action_index": action_idx,
                    "type": "invalid_action_item",
                    "message": "action item is not a dict",
                }
            )
            continue

        command = str(action.get("command", "")).strip().lower()
        object_unified_name = action.get("object_unified_name")

        if command not in SUPPORTED_ACTION_COMMANDS:
            anomalies.append(
                {
                    "action_index": action_idx,
                    "type": "unsupported_command",
                    "command": command,
                    "message": f"Unsupported action command: {command}",
                }
            )

        if _is_unmatched_object_name(object_unified_name):
            anomalies.append(
                {
                    "action_index": action_idx,
                    "type": "unmatched_object",
                    "command": command,
                    "object_unified_name": object_unified_name,
                    "message": (
                        "Action references unmatched object in object_unified_name"
                    ),
                }
            )

    return anomalies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align VLM action-plan objects/regions to canonical task objects/regions "
            "using the translator schema."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data/meta_data/Imitation.json",
        help="Path to source generated-data JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/action_plan/Imitation/gpt-5.1_Imitation.json",
        help="Path to run_vlm_evaluation output JSON (expected: <model_id>_<dataset>_<suffix>.json; default: %(default)s).",
    )
    parser.add_argument(
        "--region-descriptions-path",
        type=Path,
        default=PROJECT_ROOT / "config/ALL_REGION_DESCRIPTONS.json",
        help="Path to ALL_REGION_DESCRIPTONS.json (default: %(default)s).",
    )
    parser.add_argument(
        "--object-descriptions-path",
        type=Path,
        default=PROJECT_ROOT / "config/OBJECT_DESCRIPTIONS.json",
        help="Path to OBJECT_DESCRIPTIONS.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output JSON path. If omitted, defaults to "
            "outputs/translation/<dataset>/<model_id>_<dataset>_<suffix>.json."
        ),
    )
    parser.add_argument(
        "--human-camera-reference-path",
        type=Path,
        default=PROJECT_ROOT / "config/human2camera_reference.json",
        help="Path to human/camera region-reference mapping JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Number of rows to process. If set, it overrides --max-samples "
            "(for backward compatibility)."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save partial outputs every N processed rows (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If set and the target output file exists, reuse rows whose "
            "(original_id, camera_perspective, spatial_reference) triple already "
            "appears as status=='ok'. Only newly added / previously failed rows "
            "are (re)translated. Reused entries keep their translator output but "
            "get their `id` field refreshed to the current input-row id."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads for parallel translation (default: %(default)s).",
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
        help=(
            "Underlying model id forwarded to the registered model "
            "(e.g. 'gpt-5.4' for gpt_schema). If omitted, the model's "
            "own default is used."
        ),
    )
    parser.add_argument(
        "--fix-source-region-mode",
        type=str,
        choices=["strict", "lenient", "none"],
        default="lenient",
        help=(
            "Post-process translator output to fix pick source_region_id inconsistencies. "
            "'strict': only fix first pick per object; "
            "'lenient': track state through place actions, fix all picks; "
            "'none': no fix. Default: %(default)s."
        ),
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


def infer_action_plan_name_parts(
    *, results_path: Path, dataset_name: str | None
) -> tuple[str | None, str | None]:
    """Infer (source_model_id, suffix) from an action-plan results filename.

    Supported stems include:
    - <model_id>_results_<task>_<suffix>
    - <model_id>_<task>_<suffix>
    """
    if not dataset_name:
        return None, None

    stem = results_path.stem
    marker = f"_{dataset_name}_"
    if marker not in stem:
        return None, None

    prefix, suffix = stem.split(marker, 1)
    if not prefix or not suffix:
        return None, None

    source_model_id = re.sub(r"_results$", "", prefix)
    source_model_id = source_model_id.strip()
    suffix = suffix.strip()

    if not source_model_id or not suffix:
        return None, None
    return source_model_id, suffix


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path

    dataset_name = infer_dataset_name_from_path(args.data_path) or infer_dataset_name_from_path(
        args.results_path
    )

    source_model_id, action_plan_suffix = infer_action_plan_name_parts(
        results_path=args.results_path,
        dataset_name=dataset_name,
    )

    if dataset_name and source_model_id and action_plan_suffix:
        return (
            PROJECT_ROOT
            / "outputs/translation"
            / dataset_name
            / f"{source_model_id}_{dataset_name}_{action_plan_suffix}.json"
        )

    # Fallback: keep old behavior when filename cannot be parsed.
    model_name = getattr(args, "model", "gpt")
    if dataset_name is None:
        return PROJECT_ROOT / f"outputs/translation/{model_name}_translator_results.json"
    return (
        PROJECT_ROOT
        / "outputs/translation"
        / dataset_name
        / f"{model_name}_translator_results_{dataset_name}.json"
    )


def infer_source_model_from_results_path(path: Path) -> str | None:
    stem = path.stem

    matched = re.match(r"^(.+?)_results_", stem)
    if matched is not None:
        return matched.group(1).lower()

    dataset_name = infer_dataset_name_from_path(path)
    source_model_id, _ = infer_action_plan_name_parts(
        results_path=path,
        dataset_name=dataset_name,
    )
    if source_model_id is None:
        return None
    return source_model_id.lower()

def ensure_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected dict for {field_name}, got {type(value)}")


def get_camera_perspective(row: dict[str, Any]) -> Any:
    value = row.get("camera_perspective")
    if value is None:
        value = row.get("camera_perspective")
    return value


def is_left_camera_reference(spatial_reference: Any, camera_perspective: Any) -> bool:
    return (
        str(spatial_reference or "").strip().lower() == "camera"
        and str(camera_perspective or "").strip().lower() == "left"
    )


def _row_fingerprint(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("original_id"),
        get_camera_perspective(row),
        row.get("spatial_reference"),
    )


def _load_reusable_results(output_path: Path) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    if not output_path.is_file():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[skip-existing] Failed to load {output_path}: {exc}; treating as empty.",
            flush=True,
        )
        return {}
    reusable: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        fp = (
            item.get("original_id"),
            item.get("camera_perspective"),
            item.get("spatial_reference"),
        )
        reusable[fp] = item
    return reusable


def load_examples_by_id(data_payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(data_payload, list):
        rows = data_payload
    elif isinstance(data_payload, dict) and isinstance(data_payload.get("examples"), list):
        rows = data_payload["examples"]
    else:
        raise ValueError("Unsupported data file structure. Expected list or {'examples': [...]}.")

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if row_id is None:
            continue
        indexed[str(row_id)] = row
    return indexed


def choose_region_descriptions(
    spatial_reference: str,
    camera_perspective: Any,
    all_regions: dict[str, Any],
) -> str:
    spatial = str(spatial_reference or "").strip().lower()

    if is_left_camera_reference(spatial_reference, camera_perspective):
        key = "ALL_REGION_DESCRIPTIONS_LEFT_CAMERA"
    elif spatial == "human":
        key = "ALL_REGION_DESCRIPTIONS_HUMAN"
    else:
        key = "ALL_REGION_DESCRIPTIONS_CAMERA"

    value = all_regions.get(key)
    if not isinstance(value, str):
        raise KeyError(f"Missing string key in region descriptions: {key}")
    return value


def fix_source_region_consistency(
    translator_output: dict[str, Any],
    *,
    initial_states: list[Any] | None = None,
    mode: str = "lenient",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fix pick actions whose source_region_id is inconsistent with the object's location.

    Args:
        translator_output: The translator output dict containing 'objects' and 'actions'.
        initial_states: Optional task initial states. Used as canonical fallback
            when object unified_region is missing or inconsistent.
        mode: 'strict' — only fix first pick per object (use current known region).
              'lenient' — track object location through place actions, fix all picks.
              'none' — do nothing, return as-is.

    Returns:
        (modified translator_output copy, list of fix records)
    """
    if mode == "none":
        return translator_output, []

    output = copy.deepcopy(translator_output)
    objects = output.get("objects") or output.get("Objects", [])
    actions = output.get("actions") or output.get("Actions", [])
    fixes: list[dict[str, Any]] = []

    def _norm_region(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        return value.strip().replace(".", "_")

    def _norm_object_name(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        return value.strip().replace(".", "_")

    # Build object -> region from task initial_states as canonical fallback.
    obj_init_region: dict[str, str] = {}
    if isinstance(initial_states, list):
        for item in initial_states:
            if not isinstance(item, list):
                continue
            if len(item) == 3:
                _, obj_name_raw, region_raw = item
            elif len(item) == 2:
                a, b = item
                if isinstance(a, str) and a.strip().lower() in ("open", "close"):
                    continue
                obj_name_raw, region_raw = a, b
            else:
                continue
            obj_name = _norm_object_name(obj_name_raw)
            if not obj_name:
                continue
            region = _norm_region(region_raw)
            if region:
                obj_init_region[obj_name] = region

    # In lenient mode, state tracking starts from initial states.
    obj_current_region: dict[str, str] = dict(obj_init_region)

    # Backfill missing unified_region for matched objects from initial states.
    # Do NOT override tracked canonical state from translator unified_region.
    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("matched", False):
            continue
        name = _norm_object_name(obj.get("unified_name"))
        if not name:
            continue

        region_from_obj = _norm_region(obj.get("unified_region"))
        region_from_init = obj_init_region.get(name)

        if not region_from_obj and region_from_init:
            obj["unified_region"] = region_from_init
            fixes.append({
                "type": "object_unified_region_backfill_from_initial_state",
                "object_unified_name": name,
                "filled_unified_region": region_from_init,
                "mode": mode,
            })

    seen_pick: set[str] = set()

    for action_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        command = str(action.get("command", "")).strip().lower()
        obj_name = action.get("object_unified_name")

        if command == "pick" and isinstance(obj_name, str):
            obj_name = _norm_object_name(obj_name)
            if not obj_name:
                continue
            src = _norm_region(action.get("source_region_id"))
            is_first_pick = obj_name not in seen_pick
            seen_pick.add(obj_name)
            # first pick is anchored to initial_states canonical region
            if is_first_pick:
                expected = obj_init_region.get(obj_name)
            else:
                expected = obj_current_region.get(obj_name)

            should_fix = False
            if mode == "strict" and is_first_pick:
                should_fix = True
            elif mode == "lenient":
                should_fix = True

            if should_fix and expected and src != expected:
                fixes.append({
                    "type": "pick_source_region_fix",
                    "action_index": action_idx,
                    "object_unified_name": obj_name,
                    "original_source_region_id": src,
                    "fixed_source_region_id": expected,
                    "is_first_pick": is_first_pick,
                    "mode": mode,
                })
                action["source_region_id"] = expected

        elif command == "place" and mode == "lenient" and isinstance(obj_name, str):
            obj_name = _norm_object_name(obj_name)
            if not obj_name:
                continue
            target = _norm_region(action.get("target_region_id"))
            if target:
                obj_current_region[obj_name] = target

    # Write back (handle both casing conventions)
    if "actions" in output:
        output["actions"] = actions
    elif "Actions" in output:
        output["Actions"] = actions

    return output, fixes


def parsed_translator_to_dict(parsed: Any) -> dict[str, Any]:
    if hasattr(parsed, "model_dump"):
        dumped = parsed.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(parsed, dict):
        return parsed
    raise TypeError(f"Unexpected translator output type: {type(parsed)}")


def remap_states_for_reference(
    states: list[Any],
    *,
    spatial_reference: str,
    camera_perspective: Any,
    human_camera_reference_payload: dict[str, Any],
) -> list[Any]:
    """
    Return a copied list of states with region ids remapped to the target frame.

    Mapping selection:
      - (camera_perspective='left', spatial_reference='camera') -> camera2leftCamera_reference
      - spatial_reference='human'                               -> camera2human_reference
      - otherwise                                               -> no remap (canonical frame)
    """
    remapped = copy.deepcopy(states)

    if is_left_camera_reference(spatial_reference, camera_perspective):
        mapping_key = "camera2leftCamera_reference"
    elif str(spatial_reference or "").strip().lower() == "human":
        mapping_key = "camera2human_reference"
    else:
        return remapped

    mapping = human_camera_reference_payload.get(mapping_key)
    if not isinstance(mapping, dict):
        raise ValueError(
            f"human-camera reference file must contain dict key: {mapping_key}"
        )

    for state in remapped:
        if not isinstance(state, list) or not state:
            continue
        last_item = state[-1]
        if not isinstance(last_item, str):
            continue
        state[-1] = mapping.get(last_item, last_item)

    return remapped


def render_translator_prompt(
    *,
    all_region_descriptions: str,
    object_descriptions: Any,
    task_information: dict[str, Any],
    description_of_object_manipulations: dict[str, Any] | str,
) -> str:
    if isinstance(description_of_object_manipulations, str):
        manipulations_text = description_of_object_manipulations
    else:
        manipulations_text = json.dumps(
            description_of_object_manipulations, ensure_ascii=False, indent=2
        )

    return TRANSLATOR_PROMPT_TEMPLATE.format(
        ALL_REGION_DESCRIPTIONS=all_region_descriptions,
        OBJECT_DESCRIPTIONS=json.dumps(object_descriptions, ensure_ascii=False, indent=2),
        TASK_INFORMATION=json.dumps(task_information, ensure_ascii=False, indent=2),
        DESCRIPTION_OF_OBJECT_MANIPULATIONS=manipulations_text,
    )


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_output_payload(
    *,
    created_at: str,
    args: argparse.Namespace,
    output_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": created_at,
        "source_data_path": str(args.data_path),
        "source_results_path": str(args.results_path),
        "region_descriptions_path": str(args.region_descriptions_path),
        "object_descriptions_path": str(args.object_descriptions_path),
        "total": len(output_rows),
        "ok": sum(1 for row in output_rows if row.get("status") == "ok"),
        "error": sum(1 for row in output_rows if row.get("status") == "error"),
        "results": output_rows,
    }


def process_result_row(
    *,
    idx: int,
    total_rows: int,
    row: Any,
    examples_by_id: dict[str, dict[str, Any]],
    regions_payload: dict[str, Any],
    object_descriptions_payload: dict[str, Any],
    human_camera_reference_payload: dict[str, Any],
    source_results_model: str | None,
    model: BaseVLM,
    fix_source_region_mode: str = "lenient",
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "status": "error",
            "error": f"Row {idx} is not a dict.",
        }

    row_id = row.get("id")
    original_id = str(row.get("original_id", "")).strip()
    language_instruction = row.get("language_instruction")
    camera_perspective = get_camera_perspective(row)
    spatial_reference = str(row.get("spatial_reference", "")).strip()
    video = row.get("video")
    video_path = row.get("video_path")
    task_information_for_output: dict[str, Any] | None = None
    translator_prompt: str | None = None

    print(f"[{idx}/{total_rows}] id={row_id} original_id={original_id}", flush=True)

    try:
        source_row = examples_by_id.get(original_id)
        if source_row is None:
            raise KeyError(f"original_id not found in source data: {original_id}")

        simulation_task = ensure_dict(
            source_row.get("simulation_task"), field_name="simulation_task"
        )
        objects = simulation_task.get("objects", [])
        initial_states = simulation_task.get("initial_states", [])
        final_goal = simulation_task.get("final_goal", [])
        if not isinstance(objects, list):
            raise TypeError("simulation_task.objects must be a list")
        if not isinstance(initial_states, list):
            raise TypeError("simulation_task.initial_states must be a list")
        if not isinstance(final_goal, list):
            raise TypeError("simulation_task.final_goal must be a list")
        remapped_initial_states = remap_states_for_reference(
            initial_states,
            spatial_reference=spatial_reference,
            camera_perspective=camera_perspective,
            human_camera_reference_payload=human_camera_reference_payload,
        )
        remapped_final_goal = remap_states_for_reference(
            final_goal,
            spatial_reference=spatial_reference,
            camera_perspective=camera_perspective,
            human_camera_reference_payload=human_camera_reference_payload,
        )

        task_information_for_prompt = {
            "objects": objects,
            "initial_states": remapped_initial_states,
        }
        task_information_for_output = {
            "objects": objects,
            "initial_states": remapped_initial_states,
            "final_goal": remapped_final_goal,
        }

        description_of_object_manipulations_raw = row.get("action_plan_json", {})
        source_model_name = str(source_results_model or "").strip().lower()
        if source_model_name == "gpt":
            description_of_object_manipulations: dict[str, Any] | str = ensure_dict(
                description_of_object_manipulations_raw,
                field_name="action_plan_json",
            )
        else:
            if isinstance(description_of_object_manipulations_raw, (dict, str)):
                description_of_object_manipulations = description_of_object_manipulations_raw
            else:
                raise TypeError(
                    "Expected dict or str for action_plan_json, "
                    f"got {type(description_of_object_manipulations_raw)}"
                )

        all_region_descriptions = choose_region_descriptions(
            spatial_reference=spatial_reference,
            camera_perspective=camera_perspective,
            all_regions=regions_payload,
        )
        translator_prompt = render_translator_prompt(
            all_region_descriptions=all_region_descriptions,
            object_descriptions=object_descriptions_payload,
            task_information=task_information_for_prompt,
            description_of_object_manipulations=description_of_object_manipulations,
        )

        translator_result = model.query(
            question=translator_prompt,
            video_input=None,
            json_schema_name="translator",
        )
        translator_json = parsed_translator_to_dict(translator_result)
        translator_json, source_region_fixes = fix_source_region_consistency(
            translator_json,
            initial_states=remapped_initial_states,
            mode=fix_source_region_mode,
        )
        translator_action_anomalies = collect_translator_action_anomalies(
            translator_json
        )

        return {
            "id": row_id,
            "original_id": original_id,
            "language_instruction": language_instruction,
            "camera_perspective": camera_perspective,
            "spatial_reference": spatial_reference,
            "video": video,
            "video_path": video_path,
            "translator_prompt": translator_prompt,
            "task_information": task_information_for_output,
            "description_of_object_manipulations": description_of_object_manipulations,
            "translator_output": translator_json,
            "source_region_fixes": source_region_fixes,
            "translator_action_has_anomaly": bool(translator_action_anomalies),
            "translator_action_anomalies": translator_action_anomalies,
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        error_row: dict[str, Any] = {
            "id": row_id,
            "original_id": original_id,
            "language_instruction": language_instruction,
            "camera_perspective": camera_perspective,
            "spatial_reference": spatial_reference,
            "video": video,
            "video_path": video_path,
            "translator_prompt": translator_prompt,
            "status": "error",
            "error": str(exc),
        }
        if task_information_for_output is not None:
            error_row["task_information"] = task_information_for_output
        print(
            f"Failed id={row_id} original_id={original_id}: {exc}",
            flush=True,
        )
        return error_row


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args)
    source_results_model = infer_source_model_from_results_path(args.results_path)
    created_at = datetime.now(timezone.utc).isoformat()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.model_id is not None:
        model_kwargs["model_id"] = args.model_id
    model = get_model(args.model, **model_kwargs)
    print(
        "Using model: "
        f"{args.model} (model_id={args.model_id or 'default'}; "
        f"available: {list_models()}); "
        f"source results model: {source_results_model or 'unknown'}",
        flush=True,
    )

    data_payload = read_json_file(str(args.data_path))
    results_payload = read_json_file(str(args.results_path))
    regions_payload = read_json_file(str(args.region_descriptions_path))
    object_descriptions_payload = read_json_file(str(args.object_descriptions_path))
    human_camera_reference_payload = read_json_file(str(args.human_camera_reference_path))

    if data_payload is None:
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if results_payload is None:
        raise FileNotFoundError(f"Results file not found: {args.results_path}")
    if regions_payload is None:
        raise FileNotFoundError(
            f"Region descriptions file not found: {args.region_descriptions_path}"
        )
    if object_descriptions_payload is None:
        raise FileNotFoundError(
            "Object descriptions file not found: "
            f"{args.object_descriptions_path}"
        )
    if human_camera_reference_payload is None:
        raise FileNotFoundError(
            "Human/camera reference file not found: "
            f"{args.human_camera_reference_path}"
        )

    examples_by_id = load_examples_by_id(data_payload)
    results_payload = ensure_dict(results_payload, field_name="results_payload")
    regions_payload = ensure_dict(regions_payload, field_name="regions_payload")
    object_descriptions_payload = ensure_dict(
        object_descriptions_payload, field_name="object_descriptions_payload"
    )
    human_camera_reference_payload = ensure_dict(
        human_camera_reference_payload, field_name="human_camera_reference_payload"
    )

    result_rows = results_payload.get("results")
    if not isinstance(result_rows, list):
        raise ValueError("results file must include a list field: 'results'.")

    sample_limit = args.num_samples if args.num_samples is not None else args.max_samples
    if sample_limit is not None:
        if sample_limit < 0:
            raise ValueError("--num-samples/--max-samples must be >= 0.")
        result_rows = result_rows[:sample_limit]

    if args.save_every <= 0:
        raise ValueError("--save-every must be >= 1.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1.")

    total_rows = len(result_rows)
    output_rows: list[dict[str, Any] | None] = [None] * total_rows

    if args.skip_existing and total_rows > 0:
        reusable = _load_reusable_results(output_path)
        if reusable:
            reused_count = 0
            for i, row in enumerate(result_rows):
                if not isinstance(row, dict):
                    continue
                cached = reusable.get(_row_fingerprint(row))
                if cached is None:
                    continue
                refreshed = dict(cached)
                refreshed["id"] = row.get("id")
                output_rows[i] = refreshed
                reused_count += 1
            print(
                f"skip-existing: reused {reused_count}/{total_rows} rows from {output_path}",
                flush=True,
            )

    if args.num_workers == 1:
        for idx, row in enumerate(result_rows, start=1):
            if output_rows[idx - 1] is not None:
                continue
            output_rows[idx - 1] = process_result_row(
                idx=idx,
                total_rows=total_rows,
                row=row,
                examples_by_id=examples_by_id,
                regions_payload=regions_payload,
                object_descriptions_payload=object_descriptions_payload,
                human_camera_reference_payload=human_camera_reference_payload,
                source_results_model=source_results_model,
                model=model,
                fix_source_region_mode=args.fix_source_region_mode,
            )

            if idx % args.save_every == 0:
                partial_rows = [item for item in output_rows if item is not None]
                partial_payload = build_output_payload(
                    created_at=created_at,
                    args=args,
                    output_rows=partial_rows,
                )
                save_json(output_path, partial_payload)
                print(
                    f"Checkpoint saved at {idx}/{total_rows}: {output_path}",
                    flush=True,
                )
    else:
        _thread_local = threading.local()

        def _get_thread_model() -> BaseVLM:
            """Return a per-thread model clone to avoid sharing one API client."""
            if not hasattr(_thread_local, "model"):
                _thread_local.model = model.clone()
            return _thread_local.model

        def _worker(*, idx: int, row: Any) -> dict[str, Any]:
            return process_result_row(
                idx=idx,
                total_rows=total_rows,
                row=row,
                examples_by_id=examples_by_id,
                regions_payload=regions_payload,
                object_descriptions_payload=object_descriptions_payload,
                human_camera_reference_payload=human_camera_reference_payload,
                source_results_model=source_results_model,
                model=_get_thread_model(),
                fix_source_region_mode=args.fix_source_region_mode,
            )

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_idx = {
                executor.submit(_worker, idx=idx, row=row): idx
                for idx, row in enumerate(result_rows, start=1)
                if output_rows[idx - 1] is None
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row = result_rows[idx - 1]
                try:
                    output_rows[idx - 1] = future.result()
                except Exception as exc:
                    if isinstance(row, dict):
                        output_rows[idx - 1] = {
                            "id": row.get("id"),
                            "original_id": str(row.get("original_id", "")).strip(),
                            "language_instruction": row.get("language_instruction"),
                            "camera_perspective": get_camera_perspective(row),
                            "spatial_reference": str(
                                row.get("spatial_reference", "")
                            ).strip(),
                            "video": row.get("video"),
                            "video_path": row.get("video_path"),
                            "translator_prompt": None,
                            "status": "error",
                            "error": f"Unhandled worker exception: {exc}",
                        }
                    else:
                        output_rows[idx - 1] = {
                            "status": "error",
                            "error": (
                                f"Unhandled worker exception on row {idx}: {exc}"
                            ),
                        }

                completed += 1
                if completed % args.save_every == 0:
                    partial_rows = [item for item in output_rows if item is not None]
                    partial_payload = build_output_payload(
                        created_at=created_at,
                        args=args,
                        output_rows=partial_rows,
                    )
                    save_json(output_path, partial_payload)
                    print(
                        f"Checkpoint saved at {completed}/{total_rows}: {output_path}",
                        flush=True,
                    )

    final_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(output_rows, start=1):
        if row is None:
            final_rows.append(
                {
                    "status": "error",
                    "error": f"Row {idx} was not processed.",
                }
            )
        else:
            final_rows.append(row)

    output_payload = build_output_payload(
        created_at=created_at,
        args=args,
        output_rows=final_rows,
    )

    save_json(output_path, output_payload)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
