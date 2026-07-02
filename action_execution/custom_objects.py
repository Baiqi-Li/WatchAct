"""Register WatchAct custom objects into LIBERO's object registry.

WatchAct BDDL files reference object categories that are *not* shipped with
vanilla LIBERO (e.g. ``apple``, ``banana``, ``mango``, ``pear``,
``tennis_ball``, ``cheese``). The MuJoCo models for these live under
``WatchAct/custom_assets/<folder>/model_tier1.xml`` and are already authored in
LIBERO's object format (a ``<body name="object">`` with ``bottom_site`` /
``top_site`` / ``horizontal_radius_site``).

Instead of editing the LIBERO source tree, we register the objects at runtime
through LIBERO's public extension point: the global ``OBJECTS_DICT`` that
``get_object_fn(category)`` reads. Import this module and call
``register_watchact_objects()`` (or ``ensure_categories_registered(...)``)
*before* building any environment. Assets stay inside WatchAct; the LIBERO
checkout under ``third_party/LIBERO`` is left untouched.

A BDDL line ``(:objects apple_1 - apple)`` resolves as
``OBJECTS_DICT["apple"](name="apple_1")`` -> a ``MujocoXMLObject`` pointing at
``custom_assets/apple/model_tier1.xml``.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Iterable

from libero.libero.envs.base_object import OBJECTS_DICT
from robosuite.models.objects import MujocoXMLObject

logger = logging.getLogger(__name__)

# action_execution/ -> WatchAct/ -> custom_assets/
_WATCHACT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CUSTOM_ASSETS_DIR = _WATCHACT_ROOT / "custom_assets"
_MJCF_NAME = "model_tier1.xml"

# BDDL object category (the ``- <type>`` string) -> asset folder under
# custom_assets/. Most map 1:1; ``cheese`` is an alias for the cheese_can asset
# (WatchAct BDDLs write ``cheese_1 - cheese``).
CATEGORY_TO_FOLDER: dict[str, str] = {
    "apple": "apple",
    "banana": "banana",
    "mango": "mango",
    "pear": "pear",
    "tennis_ball": "tennis_ball",
    "cheese": "cheese_can",
    "cheese_can": "cheese_can",
    "band_aid": "band_aid",
    "coke": "coke",
    "football": "football",
    "lemon": "lemon",
}


class WatchActObject(MujocoXMLObject):
    """A rigid object loaded from ``custom_assets/<obj_name>/model_tier1.xml``.

    Mirrors the shape of LIBERO's built-in object classes (e.g.
    ``TurbosquidObjects``) so the kitchen problem builder and the placement
    sampler treat it identically to a native object.
    """

    def __init__(self, name, obj_name, category=None, joints=None):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        xml_path = _CUSTOM_ASSETS_DIR / obj_name / _MJCF_NAME
        if not xml_path.is_file():
            raise FileNotFoundError(
                f"WatchAct custom asset not found: {xml_path}. "
                f"Expected custom_assets/{obj_name}/{_MJCF_NAME}."
            )
        super().__init__(
            str(xml_path),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = category or obj_name
        self.rotation = (0, 0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


def _make_object_class(category: str, folder: str) -> type:
    """Build a WatchActObject subclass whose defaults bind to (category, folder).

    ``get_object_fn(category)`` returns this class and calls it as
    ``cls(name=<instance_name>)``, so ``obj_name`` must default to the folder.
    """

    def __init__(self, name=category, obj_name=folder, joints=None):
        WatchActObject.__init__(
            self, name=name, obj_name=folder, category=category, joints=joints
        )

    class_name = "".join(part.capitalize() for part in category.split("_")) or "WatchActObject"
    return type(class_name, (WatchActObject,), {"__init__": __init__})


def _register_one(category: str, folder: str, overwrite: bool) -> bool:
    """Insert one category into OBJECTS_DICT. Returns True if newly registered."""
    key = category.lower()
    if key in OBJECTS_DICT and not overwrite:
        # Already provided by vanilla LIBERO (or a previous call). Do not clobber.
        return False
    xml_path = _CUSTOM_ASSETS_DIR / folder / _MJCF_NAME
    if not xml_path.is_file():
        logger.warning(
            "Skipping custom object %r: asset missing (%s)", category, xml_path
        )
        return False
    OBJECTS_DICT[key] = _make_object_class(key, folder)
    return True


def register_watchact_objects(
    categories: Iterable[str] | None = None, overwrite: bool = False
) -> list[str]:
    """Register WatchAct custom objects into LIBERO's OBJECTS_DICT.

    Args:
        categories: which BDDL categories to register. Defaults to every entry
            in CATEGORY_TO_FOLDER.
        overwrite: if True, replace an existing OBJECTS_DICT entry (e.g. to
            override a vanilla object). Defaults to False so native LIBERO
            objects (basket, wooden_tray, ...) are never clobbered.

    Returns:
        The list of category names that were newly registered.
    """
    cats = list(categories) if categories is not None else list(CATEGORY_TO_FOLDER)
    registered: list[str] = []
    for cat in cats:
        folder = CATEGORY_TO_FOLDER.get(cat.lower(), cat.lower())
        if _register_one(cat, folder, overwrite):
            registered.append(cat.lower())
    if registered:
        logger.info("Registered WatchAct custom objects: %s", ", ".join(sorted(registered)))
    return registered


def ensure_categories_registered(categories: Iterable[str]) -> list[str]:
    """Make sure every given BDDL category resolves; auto-register from assets.

    For each category not already in OBJECTS_DICT, register it if a matching
    custom_assets folder exists (via CATEGORY_TO_FOLDER or a same-name folder).
    Returns the categories that still cannot be resolved (caller should error).
    """
    missing: list[str] = []
    for cat in categories:
        key = cat.lower()
        if key in OBJECTS_DICT:
            continue
        folder = CATEGORY_TO_FOLDER.get(key, key)
        if not _register_one(key, folder, overwrite=False):
            missing.append(cat)
    return missing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    added = register_watchact_objects()
    print(f"Newly registered: {sorted(added)}")
    print(f"OBJECTS_DICT now has {len(OBJECTS_DICT)} entries.")
    for c in sorted(CATEGORY_TO_FOLDER):
        print(f"  {c:12s} -> {'OK' if c in OBJECTS_DICT else 'MISSING'}")
