"""Register WatchAct's custom BDDL goal predicates into LIBERO at runtime.

Some WatchAct BDDLs (notably ``Temporal_Sort``) use two goal predicates that
vanilla LIBERO does not ship:

* ``(Right A B)``      -- A is to the right of B. On the kitchen table the world
  y-axis points left(+)->right(-), so "A right of B" means ``A.y < B.y``.
* ``(Horizontal A B [C ...])`` -- the objects lie on one left-right line, i.e.
  their front/back position (world x) is nearly equal. True when the spread of
  x-coordinates is within ``HORIZONTAL_X_TOL``. Accepts 2+ objects.

Why the axis mapping / tolerance are right: across all WatchAct BDDLs the table
grid regions are fixed -- back rows sit at x in [-0.23, -0.21], middle at
[-0.08, -0.06], front at [0.07, 0.09] (rows ~0.15 apart), and left/center/right
columns at y ~= 0.10 / -0.05 / -0.20. So x is front/back, y is left/right, and a
0.05 tolerance comfortably absorbs within-row jitter while staying well under
the ~0.15 gap between rows.

Like ``custom_objects``, this uses LIBERO's public registry and a small runtime
monkeypatch instead of editing the LIBERO source, so the bundled
``third_party/LIBERO`` stays pristine. Call ``register_watchact_predicates()``
before building an environment (the executor does this automatically).

Note on arity: LIBERO's ``_eval_predicate`` only dispatches unary (2-token) and
binary (3-token) states, so a 3-object ``(Horizontal A B C)`` would silently
evaluate to None. We therefore also patch ``_eval_predicate`` on the kitchen
problem with a general n-ary version (equivalent to the original for unary /
binary predicates).
"""

from __future__ import annotations

import logging

from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.predicates.base_predicates import BinaryAtomic, MultiarayAtomic

logger = logging.getLogger(__name__)

# Max spread (meters) of object x-coordinates for them to count as one
# horizontal (left-right) line. See module docstring for the justification.
HORIZONTAL_X_TOL = 0.05


class Right(BinaryAtomic):
    """(Right A B): A is to the right of B, i.e. A has the smaller world y."""

    def __call__(self, arg1, arg2):
        return arg1.get_geom_state()["pos"][1] < arg2.get_geom_state()["pos"][1]


class Horizontal(MultiarayAtomic):
    """(Horizontal A B [C ...]): all objects share one front/back row (world x)."""

    def __init__(self, tol=HORIZONTAL_X_TOL):
        super().__init__()
        self.tol = tol

    def __call__(self, *args):
        if len(args) < 2:
            return True
        xs = [a.get_geom_state()["pos"][0] for a in args]
        return (max(xs) - min(xs)) <= self.tol


def _general_eval_predicate(self, state):
    """N-ary drop-in for the kitchen problem's ``_eval_predicate``.

    ``state`` is ``[predicate_name, obj1, obj2, ...]``. Every predicate function
    is ``fn(*object_states)``, so this works for unary/binary/n-ary uniformly
    and matches the original behaviour for the 2- and 3-token cases.
    """
    if not state:
        return False
    name = state[0]
    objs = [self.object_states_dict[n] for n in state[1:]]
    from libero.libero.envs.predicates import eval_predicate_fn

    return eval_predicate_fn(name, *objs)


def _patch_eval_predicate() -> None:
    """Replace the kitchen problem's _eval_predicate with the n-ary version.

    The problem class is fetched from LIBERO's ``TASK_MAPPING`` registry:
    ``@register_problem`` returns None, so the module-level class name is bound
    to None and cannot be imported directly.
    """
    # Ensure the kitchen problem module is imported so it registers itself.
    import libero.libero.envs.problems.libero_kitchen_tabletop_manipulation  # noqa: F401
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING

    cls = TASK_MAPPING["libero_kitchen_tabletop_manipulation"]
    if getattr(cls, "_watchact_nary_patched", False):
        return
    cls._eval_predicate = _general_eval_predicate
    cls._watchact_nary_patched = True


def register_watchact_predicates(tol: float = HORIZONTAL_X_TOL) -> list[str]:
    """Register Right/Horizontal and enable n-ary predicate dispatch.

    Idempotent; safe to call multiple times. Returns the predicate keys added.
    """
    VALIDATE_PREDICATE_FN_DICT["right"] = Right()
    VALIDATE_PREDICATE_FN_DICT["horizontal"] = Horizontal(tol)
    _patch_eval_predicate()
    logger.info("Registered WatchAct predicates: right, horizontal (horizontal x-tol=%.3f)", tol)
    return ["right", "horizontal"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_watchact_predicates()
    print("predicate dict keys:", sorted(VALIDATE_PREDICATE_FN_DICT))
