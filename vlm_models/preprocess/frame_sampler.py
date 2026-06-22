"""Video frame sampling and base64-JPEG encoding (self-contained).

Provides :class:`FrameSampler`, the single entry-point used by
:meth:`vlm_models.base.BaseVLM.prepare_video`. The low-level frame
extraction and encoding logic is inlined here so the package has no
dependency on a repo-level ``utils.py``.

Requires ``opencv-python`` (cv2), ``Pillow`` and ``numpy``.
"""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Any

import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers (inlined from the original utils.py)
# ---------------------------------------------------------------------------

def _uniform_pick_positions(total_count: int, pick_count: int) -> list[int]:
    """Pick ``pick_count`` positions uniformly from ``[0, total_count)``."""
    if total_count <= 0:
        return []
    pick_count = max(1, int(pick_count))
    if pick_count >= total_count:
        return list(range(total_count))
    step = total_count / float(pick_count)
    return [int(i * step) for i in range(pick_count)]


def _dedup_keep_order(values: list[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def extract_frames(
    video_path,
    max_frames=64,
    mode="uniform",
    sampling_value: int | None = None,
    save_frames=True,
    resize_mode=None,
    resize_to=None,
    short_side=720,
    max_long_side=2000,
    save_root=None,
):
    """Extract a list of BGR frames (numpy arrays) from ``video_path``."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = (total_frames / fps) if fps > 0 else 0.0
    print(
        f"Video information: total frames={total_frames}, FPS={fps:.2f}, duration={duration:.2f} seconds",
        flush=True,
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No readable frames in the video.")

    # save directory
    frames_dir = None
    if save_frames:
        if save_root is None:
            video_dir = os.path.dirname(os.path.abspath(video_path))
            save_root = os.path.join(video_dir, "extracted_frames")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(save_root, video_name)
        os.makedirs(frames_dir, exist_ok=True)

    max_frames = max(1, int(max_frames))
    mode = str(mode or "uniform").strip().lower()
    if mode not in {"uniform", "time_based"}:
        raise ValueError(f"Unsupported mode: {mode}. Use 'uniform' or 'time_based'.")

    # calculate frame indices
    frame_indices: list[int] = []

    if mode == "uniform":
        target_count = sampling_value if sampling_value is not None else max_frames
        target_count = max(1, int(target_count))
        target_count = min(target_count, max_frames)
        target_count = min(target_count, total_frames)
        frame_indices = _uniform_pick_positions(total_frames, target_count)

    elif mode == "time_based":
        sampling_fps = max(1, int(sampling_value)) if sampling_value is not None else 1

        if duration <= 0 or fps <= 0:
            target_count = min(max_frames, total_frames)
            frame_indices = _uniform_pick_positions(total_frames, target_count)
        else:
            interval_s = 1.0 / float(sampling_fps)
            num_points = max(1, int(math.ceil(duration * float(sampling_fps))))
            for i in range(num_points):
                t = i * interval_s
                idx = int(round(t * fps))
                if 0 <= idx < total_frames:
                    frame_indices.append(idx)

            if not frame_indices:
                frame_indices = [0]

            frame_indices = _dedup_keep_order(frame_indices)
            if len(frame_indices) > max_frames:
                keep_pos = _uniform_pick_positions(len(frame_indices), max_frames)
                frame_indices = [frame_indices[p] for p in keep_pos]

    # Final normalize
    frame_indices = _dedup_keep_order(
        [min(max(0, int(i)), total_frames - 1) for i in frame_indices]
    ) or [0]

    # actually extract frames
    frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if resize_mode == "fixed" and resize_to is not None:
            frame = cv2.resize(frame, resize_to)
        elif resize_mode == "short_side" and short_side is not None:
            h, w = frame.shape[:2]
            short = min(h, w)
            long_ = max(h, w)
            scale = short_side / float(short) if short > 0 else 1.0
            if max_long_side is not None and long_ * scale > max_long_side:
                scale = max_long_side / float(long_)
            if abs(scale - 1.0) > 1e-3:
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h))
        elif resize_mode == "none":
            pass

        frames.append(frame)

        if save_frames and frames_dir:
            fname = os.path.join(frames_dir, f"frame_{i+1:03d}_idx_{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame)

    cap.release()

    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        print(
            f"Final frame resolution: {w}x{h} (widthxheight), resize_mode={resize_mode}",
            flush=True,
        )

    print(
        f"Using {mode} sampling mode (sampling_value={sampling_value}), "
        f"extracted {len(frames)} frames"
        + (f". Saved to: {frames_dir}" if save_frames and frames_dir else ""),
        flush=True,
    )

    if len(frames) == 0:
        raise RuntimeError("Extracted frames are empty, please check video validity or sampling mode.")
    return frames


def encode_image(image_bgr) -> str:
    """Encode a BGR image (numpy array) to a base64 JPEG string."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FrameSampler:
    """Extract frames from a video and encode them as base64 JPEG strings."""

    def sample_and_encode(
        self,
        video_path: str,
        *,
        max_frames: int = 32,
        mode: str = "time_based",
        sampling_value: int | None = 1,
        save_frames: bool = False,
        **extra: Any,
    ) -> list[str]:
        """Return a list of base64-encoded JPEG strings for *video_path*."""
        frames = extract_frames(
            video_path,
            max_frames=max_frames,
            mode=mode,
            sampling_value=sampling_value,
            save_frames=save_frames,
            **extra,
        )
        return [encode_image(f) for f in frames]
