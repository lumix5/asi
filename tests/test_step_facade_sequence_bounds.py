"""Step-facade config sequence ceilings reject before any element walk.

Every other step facade already bounds serialized config sequences before
iterating them (``step9._require_sequence_length``), and every core learner
bounds the same shapes (``_MAX_HIDDEN_SIZES``/``_MAX_HORDE_DEMONS`` families).
Steps 2, 3, 4, and 8 still walk caller-supplied ``hidden_sizes`` and demon
lists unbounded, so a hostile serialized config (a JSON list is the entry
point via ``from_dict``) pays an unbounded element walk and reaches downstream
width/demon amplification before any ceiling fires.

The ceiling mirrors the already-frozen constants: 4096 entries, the same limit
as ``step9._MAX_CONFIG_SEQUENCE_LENGTH``, ``core.types._MAX_HORDE_DEMONS``, and
``core.horde_actor_critic._MAX_HIDDEN_SIZES``. Every previously valid config
stays valid: only sequences beyond the existing downstream ceilings are
rejected, and the rejection moves earlier (config construction) instead of
changing which configs are legal at the boundary of 4096.

Development test infrastructure; no performance or scientific claim.
"""

import pytest

from alberta_framework.steps.step2 import (
    Step2KernelConfig,
    Step2StrictDigitReadoutConfig,
)
from alberta_framework.steps.step3 import Step3HordeConfig
from alberta_framework.steps.step4 import Step4SARSAConfig
from alberta_framework.steps.step8 import Step8WorldModelConfig

_LIMIT = 4096
_OVER_LENGTH_MESSAGE = "must contain at most 4096 values"


class TestStep2KernelHiddenSizes:
    def test_rejects_oversized_hidden_sizes_before_element_walk(self) -> None:
        # Element value 0 is individually invalid: if the walk ran before the
        # length gate, the error would be the per-element message instead.
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step2KernelConfig(
                feature_dim=1,
                stream="frequency",
                hidden_sizes=(0,) * (_LIMIT + 1),
            )

    def test_accepts_boundary_hidden_sizes(self) -> None:
        config = Step2KernelConfig(stream="frequency", hidden_sizes=(1,) * _LIMIT)
        assert len(config.hidden_sizes) == _LIMIT


class TestStep2StrictDigitHiddenSizes:
    def test_rejects_oversized_hidden_sizes_before_element_walk(self) -> None:
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step2StrictDigitReadoutConfig(n_heads=1, hidden_sizes=(0,) * (_LIMIT + 1))

    def test_accepts_boundary_hidden_sizes(self) -> None:
        config = Step2StrictDigitReadoutConfig(hidden_sizes=(1,) * _LIMIT)
        assert len(config.hidden_sizes) == _LIMIT


class TestStep3HordeSequences:
    def test_rejects_oversized_demon_lists(self) -> None:
        # Valid probability elements: on an unguarded facade the config would
        # construct successfully and defer the failure to horde construction.
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step3HordeConfig(gammas=(0.5,) * (_LIMIT + 1), lamdas=(0.5,) * (_LIMIT + 1))

    def test_rejects_oversized_lamdas_alone(self) -> None:
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step3HordeConfig(gammas=(0.5,), lamdas=(0.5,) * (_LIMIT + 1))

    def test_rejects_oversized_hidden_sizes_before_element_walk(self) -> None:
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step3HordeConfig(hidden_sizes=(0,) * (_LIMIT + 1))

    def test_accepts_boundary_demon_lists_and_hidden_sizes(self) -> None:
        config = Step3HordeConfig(
            gammas=(0.5,) * _LIMIT,
            lamdas=(0.5,) * _LIMIT,
            hidden_sizes=(1,) * _LIMIT,
        )
        assert len(config.gammas) == _LIMIT
        assert len(config.hidden_sizes) == _LIMIT


class TestStep4SarsaHiddenSizes:
    def test_rejects_oversized_hidden_sizes_before_element_walk(self) -> None:
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step4SARSAConfig(hidden_sizes=(0,) * (_LIMIT + 1))

    def test_rejects_oversized_serialized_hidden_sizes(self) -> None:
        payload = Step4SARSAConfig().to_dict()
        payload["hidden_sizes"] = [0] * (_LIMIT + 1)
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step4SARSAConfig.from_dict(payload)

    def test_accepts_boundary_hidden_sizes(self) -> None:
        config = Step4SARSAConfig(hidden_sizes=(1,) * _LIMIT)
        assert len(config.hidden_sizes) == _LIMIT


class TestStep8WorldModelHiddenSizes:
    def test_rejects_oversized_hidden_sizes_before_element_walk(self) -> None:
        with pytest.raises(ValueError, match=_OVER_LENGTH_MESSAGE):
            Step8WorldModelConfig(hidden_sizes=(0,) * (_LIMIT + 1))

    def test_accepts_boundary_hidden_sizes(self) -> None:
        config = Step8WorldModelConfig(hidden_sizes=(1,) * _LIMIT)
        assert len(config.hidden_sizes) == _LIMIT
