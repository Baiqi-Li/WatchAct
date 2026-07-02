"""Pre-sample deterministic LIBERO initial states for WatchAct tasks (optional).

For each BDDL referenced by an execution config, this resets the environment
``--num-states`` times (letting objects settle) and records the simulator
state, saving ``<out-dir>/<bddl_stem>.pruned_init``. ``run_libero_execution.py``
loads these automatically when ``--init-states-dir`` points here; without them
it just calls ``env.reset()`` each episode (still valid, just not identical
across runs).

Usage::

    python generate_init_states.py --config execution_instructions.yaml \
        --out-dir ../outputs/init_files --num-states 20
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_WATCHACT_ROOT = _THIS_DIR.parent
_LIBERO_DIR = _WATCHACT_ROOT / "third_party" / "LIBERO"
if _LIBERO_DIR.is_dir() and str(_LIBERO_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBERO_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import tqdm  # noqa: E402
import yaml  # noqa: E402

import custom_objects  # noqa: E402
import custom_predicates  # noqa: E402
from _libero_utils import resolve_bddl_path  # noqa: E402

logger = logging.getLogger("watchact.init_states")

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Execution config YAML (its 'bddl' fields are used).")
    p.add_argument("--bddl-root", default=str(_WATCHACT_ROOT / "data" / "bddl_files"))
    p.add_argument("--out-dir", default=str(_WATCHACT_ROOT / "outputs" / "init_files"))
    p.add_argument("--num-states", type=int, default=20)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(args.seed)

    import torch
    from libero.libero.envs import OffScreenRenderEnv

    custom_objects.register_watchact_objects()
    custom_predicates.register_watchact_predicates()  # env.step() checks the BDDL goal

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    bddl_root = pathlib.Path(args.bddl_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unique bddl stems across the config.
    stems: dict[str, pathlib.Path] = {}
    for t in config["tasks"]:
        path = resolve_bddl_path(t.get("bddl_file") or t["bddl"], bddl_root)
        stems[path.stem] = path

    for i, (stem, bddl_path) in enumerate(sorted(stems.items())):
        logger.info("[%d/%d] %s", i + 1, len(stems), stem)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        states = []
        for _ in tqdm.tqdm(range(args.num_states), desc=f"  {stem}"):
            env.reset()
            for _ in range(args.num_steps_wait):
                env.step(LIBERO_DUMMY_ACTION)
            states.append(env.get_sim_state())
        torch.save(np.array(states), str(out_dir / f"{stem}.pruned_init"))
        env.close()

    logger.info("Done. Init states in %s", out_dir)


if __name__ == "__main__":
    main()
