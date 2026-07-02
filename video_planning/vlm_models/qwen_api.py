"""Qwen VL model wrapper via DashScope API."""
# qwen3-vl-235b-a22b-thinking
from __future__ import annotations

import os
from typing import Any

from vlm_models import register
from vlm_models.base import BaseVLM, VideoInputType


@register
class QwenModel(BaseVLM):
    """Qwen VL – consumes video via DashScope MultiModalConversation API."""

    name = "qwen"
    video_input_type = VideoInputType.FILE_PATH

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import dashscope  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "dashscope package is required for Qwen. "
                "Install with: pip install dashscope"
            ) from exc

        self.model_id: str = kwargs.get("model_id", "qwen3-vl-235b-a22b-instruct")
        self.api_key: str | None = kwargs.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        self.fps: float = kwargs.get("fps", 1)
        dashscope.base_http_api_url = kwargs.get(
            "base_url", "https://dashscope-intl.aliyuncs.com/api/v1"
        )

    def query(
        self,
        *,
        question: str,
        video_input: dict[str, Any] | None = None,
        json_schema_name: str = "action_plan",
        **kwargs: Any,
    ) -> dict[str, Any]:
        video_path = video_input.get("video_path") if video_input else None

        content: list[dict[str, Any]] = []
        if video_path:
            content.append({"video": f"file://{video_path}", "fps": self.fps})
        content.append({"text": question})

        messages = [{"role": "user", "content": content}]

        from dashscope import MultiModalConversation

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_id,
            messages=messages,
        )

        return response.output.choices[0].message.content[0]["text"]
