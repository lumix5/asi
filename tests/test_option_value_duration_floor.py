"""The option-value duration floor must stay in the float32 normal range.

``OptionValueDurationConfig.duration_floor`` is the denominator floor of the
reported ``reward_rates`` division (``reward_values / maximum(durations, floor)``
in ``OptionValueDurationLearner.predict``).  Validation accepted any value that
narrows to a positive float32, including subnormals such as ``1e-45``.  A
subnormal floor defeats the division guard: with the legal
``duration_step_size=0.0`` the duration head stays exactly zero forever, so
``predict`` divides ordinary reward magnitudes by ``1.4e-45`` and returns
``inf`` (and ``nan`` for a zero reward head) through the public API.

The floor is now pinned to the float32 minimum normal — the same convention
already enforced for ``prototype_memory``'s ``bandwidth`` and ``upgd_memory``'s
``memory_bandwidth`` — so the worst legal configuration keeps ordinary reward
magnitudes finite in the reported rates.  These are unit mechanism tests, not
held-out evidence.
"""

import numpy as np
import pytest

from alberta_framework.core.option_value_duration import (
    OptionValueDurationConfig,
    OptionValueDurationLearner,
)

pytestmark = pytest.mark.unit

_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")


def test_subnormal_duration_floor_is_rejected() -> None:
    """A floor below the float32 minimum normal is refused at the boundary."""
    with pytest.raises(ValueError, match="duration_floor"):
        OptionValueDurationConfig(duration_floor=1e-45)
    payload = OptionValueDurationConfig().to_config()
    payload["duration_floor"] = 1e-45
    with pytest.raises(ValueError, match="duration_floor"):
        OptionValueDurationConfig.from_config(payload)


def test_smallest_normal_duration_floor_is_accepted() -> None:
    """The float32 minimum normal itself remains a legal floor."""
    config = OptionValueDurationConfig(duration_floor=_FLOAT32_MIN_NORMAL)
    assert config.duration_floor == _FLOAT32_MIN_NORMAL


def _zero_duration_state() -> tuple[OptionValueDurationLearner, object]:
    """One applied reward update with the legal zero duration step size."""
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.0,
            duration_floor=_FLOAT32_MIN_NORMAL,
        ),
    )
    state = learner.init(1)
    result = learner.update(
        state,
        observation=np.array([1.0], dtype=np.float32),
        option_index=np.int32(0),
        reward=np.float32(1.0),
        next_observation=np.array([1.0], dtype=np.float32),
        continuation_discount=np.float32(0.0),
    )
    return learner, result.state


def test_worst_case_floor_keeps_reported_rates_finite() -> None:
    """With the floor at the minimum normal, reported rates stay finite."""
    learner, state = _zero_duration_state()
    prediction = learner.predict(state, np.array([1.0], dtype=np.float32))
    assert bool(np.isfinite(np.asarray(prediction.reward_rates)).all())
