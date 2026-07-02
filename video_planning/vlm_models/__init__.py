"""Minimal VLM model registry (self-contained gpt_schema-only build).

Usage::

    from vlm_models import get_model, list_models

    model = get_model("gpt_schema", model_id="gpt-5.1")
    video_input = model.prepare_video("demo.mp4", max_frames=256)
    result = model.query(question="...", video_input=video_input,
                         json_schema_name="action_plan")

This trimmed package ships the ``gpt_schema`` and ``qwen`` backends. To add
another model, drop a new module under ``vlm_models/`` that defines a
``BaseVLM`` subclass decorated with ``@register`` and import it below.
"""

from __future__ import annotations

from typing import Any

from vlm_models.base import BaseVLM, VideoInputType  # noqa: F401 – re-export

_REGISTRY: dict[str, type[BaseVLM]] = {}


def register(cls: type[BaseVLM]) -> type[BaseVLM]:
    """Class decorator – register a VLM implementation by its ``name``."""
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"{cls.__qualname__} must define a non-empty 'name' class attribute")
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str, **kwargs: Any) -> BaseVLM:
    """Instantiate and return a registered model by *name*."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list_models()}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


# Import the shipped backends so their ``@register`` decorators run.
from vlm_models import gpt_schema  # noqa: E402,F401
from vlm_models import qwen_api  # noqa: E402,F401
