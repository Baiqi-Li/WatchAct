"""Base class and types for all VLM model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class VideoInputType(Enum):
    """How a model consumes video data."""

    FILE_PATH = "file_path"  # Model accepts a raw file path (e.g. Gemini upload)
    FRAMES = "frames"        # Model accepts base64-encoded frame images (e.g. GPT)


class BaseVLM(ABC):
    """Abstract base for every evaluable VLM.

    Subclasses must set the ``name`` class attribute (used as the registry
    key) and implement :meth:`query`.

    Video pre-processing is handled by :meth:`prepare_video`, which
    dispatches to :class:`~vlm_models.preprocess.frame_sampler.FrameSampler`
    when the model requires extracted frames.
    """

    name: str
    video_input_type: VideoInputType = VideoInputType.FRAMES

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def clone(self) -> "BaseVLM":
        """Create a new instance of this model with the same configuration.

        Useful for multi-threaded execution where each thread should own
        its own API client to avoid shared-state issues.
        """
        return self.__class__(**self.config)

    # ------------------------------------------------------------------
    # Video pre-processing
    # ------------------------------------------------------------------

    def prepare_video(
        self,
        video_path: str,
        *,
        max_frames: int = 32,
        mode: str = "time_based",
        sampling_value: int | None = 1,
        save_frames: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Prepare video input according to :attr:`video_input_type`.

        Returns a dict that is passed directly to :meth:`query` as
        ``video_input``.
        """
        if self.video_input_type == VideoInputType.FILE_PATH:
            return {"video_path": video_path}

        from vlm_models.preprocess.frame_sampler import FrameSampler

        sampler = FrameSampler()
        encoded = sampler.sample_and_encode(
            video_path,
            max_frames=max_frames,
            mode=mode,
            sampling_value=sampling_value,
            save_frames=save_frames,
            **extra,
        )
        return {"frames": encoded}

    # ------------------------------------------------------------------
    # Model query (must be implemented by each model)
    # ------------------------------------------------------------------

    @abstractmethod
    def query(
        self,
        *,
        question: str,
        video_input: dict[str, Any] | None = None,
        json_schema_name: str = "action_plan",
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """Send *question* (and optional video) to the model.

        Parameters
        ----------
        question:
            The fully-rendered prompt text.
        video_input:
            The dict returned by :meth:`prepare_video`, or ``None`` for
            text-only queries (e.g. translator alignment).
        json_schema_name:
            Schema identifier used by models that support structured
            output (e.g. ``"action_plan"``, ``"translator"``).

        Returns
        -------
        dict or str
            Parsed JSON dict (e.g. GPT structured output) or raw text string.
        """
        ...
