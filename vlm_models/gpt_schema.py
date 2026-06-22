"""OpenAI GPT model wrapper with structured-output (JSON schema) support.
For plain-text output (no schema), see ``gpt.py`` (registered as ``"gpt_text"``).
"""
#gpt-5.4
#gpt-5.4-mini
#gpt-5.4-pro
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from vlm_models import register
from vlm_models.base import BaseVLM, VideoInputType

# =========================================================================
# Pydantic schemas for GPT structured output
# (migrated from models.py – only GPT uses these)
# =========================================================================

# --- Action Plan ---

class ObjectInfo(BaseModel):
    name: str = Field(..., description="e.g., object_1, object_2, etc.")
    description: str = Field(..., description="Clear description of the object's appearance, category, and position")


class ActionPick(BaseModel):
    command: Literal["pick"]
    object: str
    source_location: str


class ActionPlace(BaseModel):
    command: Literal["place"]
    object: str
    target_region: str


class ActionOpen(BaseModel):
    command: Literal["open"]
    object: str
    target_region: str


class ActionClose(BaseModel):
    command: Literal["close"]
    object: str
    target_region: str


class RobotPlan(BaseModel):
    Objects: List[ObjectInfo]
    Actions: List[ActionPick | ActionPlace | ActionOpen | ActionClose]


# --- Translator ---

class NormalizedObject(BaseModel):
    source_name: str
    source_description: str
    unified_name: str
    task_object_id: Optional[str]
    unified_region: Optional[str]
    matched: bool


class NormalizedAction(BaseModel):
    command: str
    object_source_name: str
    object_unified_name: str
    source_location_text: Optional[str] = None
    target_location_text: Optional[str] = None
    source_region_id: Optional[str] = None
    target_region_id: Optional[str] = None
    source_location_matched: Optional[bool] = None
    target_location_matched: Optional[bool] = None


class TranslatorSchema(BaseModel):
    objects: List[NormalizedObject]
    actions: List[NormalizedAction]


# --- Data Generation ---

class HumanVideoInstruction(BaseModel):
    setup: List[str]
    steps: List[str]


class SubGoal(BaseModel):
    after_action: str
    states: List[List[str]]


class SimulationTask(BaseModel):
    language_instruction: str
    objects: List[str]
    initial_states: List[List[str]]
    primitive_action_sequence: List[str]
    sub_goals: List[SubGoal]
    final_goal: List[List[str]]


class DataGenerationExample(BaseModel):
    id: str
    human_video_instruction: HumanVideoInstruction
    simulation_task: SimulationTask


class DataGenerationSchema(BaseModel):
    examples: List[DataGenerationExample]


class HumanVideoInstructionGeneral(BaseModel):
    setup: List[str]
    steps: Optional[List[str]] = None


class SimulationTaskGeneral(BaseModel):
    language_instruction: str
    objects: List[str]
    initial_states: List[Union[List[str], str]]
    final_goal: List[Union[List[str], str]]
    primitive_action_sequence: Optional[List[str]] = None
    sub_goals: Optional[List[SubGoal]] = None


class DataGenerationGeneralExample(BaseModel):
    id: str
    human_video_instruction: Optional[HumanVideoInstructionGeneral] = None
    simulation_task: SimulationTaskGeneral


class DataGenerationGeneralSchema(BaseModel):
    examples: List[DataGenerationGeneralExample]


# --- Oracle Plan ---

class OraclePlanSchema(BaseModel):
    """A single example's oracle plan: an ordered list of primitive action
    strings (e.g. ``"PICK(butter_2, main_middle_center_region)"``)."""

    oracle_plan: List[str]


# =========================================================================
# Schema registry (maps json_schema_name -> Pydantic model)
# =========================================================================

SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "action_plan": RobotPlan,
    "translator": TranslatorSchema,
    "data_generation": DataGenerationSchema,
    "data_generation_general": DataGenerationGeneralSchema,
    "oracle_plan": OraclePlanSchema,
}


# =========================================================================
# GPT Model
# =========================================================================

@register
class GPTModel(BaseVLM):
    """OpenAI GPT with structured JSON output via ``responses.parse``."""

    name = "gpt_schema"
    video_input_type = VideoInputType.FRAMES

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from openai import OpenAI

        api_key = kwargs.get("api_key")
        self.model_id: str = kwargs.get("model_id", "gpt-5.1")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def query(
        self,
        *,
        question: str,
        video_input: dict[str, Any] | None = None,
        json_schema_name: str = "action_plan",
        **kwargs: Any,
    ) -> dict[str, Any]:
        frames: list[str] = []
        if video_input is not None:
            frames = video_input.get("frames", [])

        text_format = SCHEMA_MAP.get(json_schema_name)
        if text_format is None:
            raise ValueError(
                f"Unsupported json_schema_name: {json_schema_name!r}. "
                f"Available: {list(SCHEMA_MAP.keys())}"
            )

        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": question},
            *[
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img}",
                    "detail": "auto",
                }
                for img in frames
            ],
        ]

        response = self.client.responses.parse(
            model=self.model_id,
            input=[{"role": "user", "content": content}],
            max_output_tokens=6000,
            text_format=text_format,
        )

        parsed = response.output_parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        raise TypeError(f"Unexpected parsed type: {type(parsed)}")
