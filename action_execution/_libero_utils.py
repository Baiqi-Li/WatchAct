"""Self-contained helpers for the WatchAct LIBERO executor.

Vendors the few small utilities the executor needs so WatchAct does not depend
on the ``openpi_client`` package:

* ``convert_to_uint8`` / ``resize_with_pad`` -- image preprocessing, copied from
  ``openpi_client.image_tools`` (pure numpy + PIL).
* ``quat2axisangle`` -- quaternion -> axis-angle, from robosuite.
* ``resolve_bddl_path`` -- locate a ``.bddl`` file by stem under a WatchAct
  ``bddl_files`` tree.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image preprocessing (vendored from openpi_client.image_tools)
# ---------------------------------------------------------------------------
def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8. No-op for integer images."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resize images to (height, width) without distortion, padding with zeros.

    Replicates ``tf.image.resize_with_pad`` for a batch of images in
    ``[..., H, W, C]`` format using PIL.
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


# ---------------------------------------------------------------------------
# Robot state helper (from robosuite transform_utils)
# ---------------------------------------------------------------------------
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a (x, y, z, w) quaternion to a 3-vector axis-angle."""
    quat = np.asarray(quat, dtype=float).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
# BDDL path resolution
# ---------------------------------------------------------------------------
def resolve_bddl_path(bddl: str, bddl_root: pathlib.Path) -> pathlib.Path:
    """Resolve a BDDL reference to an absolute file path under ``bddl_root``.

    ``bddl`` may be a bare stem (``KITCHEN_..._0``), a name with ``.bddl``, or a
    ``<category>/<stem>`` relative path. Searches recursively and requires a
    unique match.
    """
    bddl_root = pathlib.Path(bddl_root)
    candidate = pathlib.Path(bddl)

    # Direct path (possibly relative to bddl_root) that already exists.
    for direct in (candidate, bddl_root / candidate):
        if direct.suffix == ".bddl" and direct.is_file():
            return direct.resolve()
    direct_stem = bddl_root / f"{candidate}.bddl"
    if direct_stem.is_file():
        return direct_stem.resolve()

    stem = candidate.name
    if stem.endswith(".bddl"):
        stem = stem[: -len(".bddl")]
    # Ignore stray copies inside image folders (e.g. <category>/images_custom/).
    matches = sorted(
        m for m in bddl_root.glob(f"**/{stem}.bddl")
        if not any(part.startswith("images") for part in m.relative_to(bddl_root).parts[:-1])
    )
    if not matches:
        raise FileNotFoundError(f"No BDDL named {stem!r} under {bddl_root}")
    if len(matches) > 1:
        rels = ", ".join(str(m.relative_to(bddl_root)) for m in matches)
        raise ValueError(
            f"BDDL stem {stem!r} is ambiguous under {bddl_root} ({rels}); "
            f"specify '<category>/{stem}'."
        )
    return matches[0].resolve()
