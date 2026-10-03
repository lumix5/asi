"""One canonical, documented ceiling for every per-module scan sequence bound.

The 10,000-step scan ceiling was previously duplicated as private magic
numbers across more than a dozen modules (``core/horde.py``,
``core/horde_actor_critic.py``, ``core/sarsa.py``, ``steps/step9.py``, ...),
so no reader could tell whether the numbers agreed by design or by accident,
and a reviewer of any new scan-length guard had to re-litigate the value
each time (PR #3035 review: "A framework-wide limit should be canonical and
publicly documented").

``alberta_framework._scan_resources`` already owns the host-side scan
resource contracts (``ScanBudget``), so it publishes the canonical
``SCAN_SEQUENCE_MAX_STEPS`` that every per-module ceiling derives from.

The delegation pins below assert integer *identity*, not merely value
equality: each per-module constant must be the same object as the canonical
constant, so deleting a derivation while keeping a local ``10_000`` literal
fails here (a fresh literal is a distinct int object), and widening one
module locally also fails. Changing the program-wide value can only happen
at the canonical site, where the documented rationale lives.
"""

from __future__ import annotations

import importlib

import pytest

import alberta_framework
from alberta_framework._scan_resources import SCAN_SEQUENCE_MAX_STEPS

# (module, attribute) pairs for every ceiling derived from the canonical
# constant. ``ScanBudget`` sites are pinned through ``maximum_steps``.
_DERIVED_CEILINGS: list[tuple[str, str]] = [
    ("alberta_framework.core.compositional_features", "_COMPOSITIONAL_LOOP_MAX_STEPS"),
    ("alberta_framework.core.dreaming", "_DREAM_ROLLOUT_BUDGET.maximum_steps"),
    ("alberta_framework.core.feature_discovery", "_FEATURE_DISCOVERY_LOOP_MAX_STEPS"),
    ("alberta_framework.core.horde", "_HORDE_SEQUENCE_MAX_STEPS"),
    ("alberta_framework.core.horde_actor_critic", "_HORDE_AC_SEQUENCE_MAX_STEPS"),
    ("alberta_framework.core.learners", "_LEARNING_LOOP_MAX_STEPS"),
    ("alberta_framework.core.option_value_duration", "_OPTION_DURATION_SCAN_MAX_STEPS"),
    ("alberta_framework.core.sarsa", "_SARSA_SEQUENCE_MAX_STEPS"),
    ("alberta_framework.core.temporal_context", "_TEMPORAL_CONTEXT_LOOP_MAX_STEPS"),
    ("alberta_framework.core.upgd", "_UPGD_LOOP_MAX_STEPS"),
    ("alberta_framework.pipeline", "_PIPELINE_SCAN_BUDGET.maximum_steps"),
    ("alberta_framework.steps.step2", "_STEP2_LOOP_BUDGET.maximum_steps"),
    ("alberta_framework.steps.step8", "_STEP8_SMOKE_BUDGET.maximum_steps"),
    ("alberta_framework.steps.step9", "_STEP9_SEQUENCE_MAX_STEPS"),
]


def _resolve(module_name: str, attribute_path: str) -> object:
    module = importlib.import_module(module_name)
    value: object = module
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def test_canonical_ceiling_is_the_documented_program_value() -> None:
    assert type(SCAN_SEQUENCE_MAX_STEPS) is int
    assert SCAN_SEQUENCE_MAX_STEPS == 10_000


def test_canonical_ceiling_is_public_api() -> None:
    assert alberta_framework.SCAN_SEQUENCE_MAX_STEPS is SCAN_SEQUENCE_MAX_STEPS


@pytest.mark.parametrize(("module_name", "attribute_path"), _DERIVED_CEILINGS)
def test_module_ceiling_shares_the_canonical_object(
    module_name: str, attribute_path: str
) -> None:
    value = _resolve(module_name, attribute_path)
    assert type(value) is int
    # Identity, not equality: the module must derive its ceiling from the
    # canonical constant rather than restate the literal.
    assert value is SCAN_SEQUENCE_MAX_STEPS, (
        f"{module_name}.{attribute_path} must derive from "
        "alberta_framework._scan_resources.SCAN_SEQUENCE_MAX_STEPS "
        f"(canonical value {SCAN_SEQUENCE_MAX_STEPS}); got {value!r}"
    )
