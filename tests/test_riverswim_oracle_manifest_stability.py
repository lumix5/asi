"""Host-stability regressions for the RiverSwim oracle embedded in manifests.

``RiverSwimMDP.optimal_average_reward`` solves the stationary distribution
through ``numpy.linalg.lstsq``, and LAPACK's last unit in the last place
depends on the host BLAS kernel.  The reference-life adapter embeds that
scalar as ``oracle_average_reward`` in the environment manifest, where it
flows into ``config_sha256``, ``manifest_id``, ``life_config_sha256``, and
every ``resolved`` block the development scorecard aggregates.  Two runners
that disagree by one ulp therefore produce different identity bytes and the
scorecard's aggregate validation fails with "does not match the canonical
resolved components" — measured on run 34950746882, where 26 of 144 RiverSwim
shards were rejected this way.

The two oracle values below are the actual cross-host pair from that run:
``0.8571501418872561`` is what the shard runners committed, ``0.857150141887256``
is what an independent recomputation of the identical source returned.  The
adapter must bind one canonical scalar so both hosts emit identical manifest
bytes, and the runtime metrics oracle must stay bitwise equal to the embedded
value so the checkpoint's exact ``oracle_reward_sum`` recomputation holds.
"""

import struct

import pytest

from alberta_framework.reference_agent import SpaceSpec
from alberta_framework.reference_life import RiverSwimReferenceEnvironment
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

_SHARD_HOST_ORACLE = 0.8571501418872561
_RECOMPUTED_HOST_ORACLE = 0.857150141887256

_N_STATES = 5

_OBSERVATION_SPEC = SpaceSpec(
    kind="box",
    shape=(_N_STATES,),
    dtype="float32",
    semantic_id="tests.riverswim-oracle-stability.observation.v1",
    low=(0.0,) * _N_STATES,
    high=(1.0,) * _N_STATES,
)
_ACTION_SPEC = SpaceSpec(
    kind="discrete",
    shape=(),
    dtype="int32",
    semantic_id="tests.riverswim-oracle-stability.action.v1",
    cardinality=2,
)

_RIVER_CONFIG = RiverSwimConfig(  # type: ignore[call-arg]
    n_states=_N_STATES,
    p_right_up=0.3,
    p_right_down=0.6,
    reward_left=0.0,
    reward_right=1.0,
    initial_state=0,
)


def _environment_with_oracle(oracle: float, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        RiverSwimMDP, "optimal_average_reward", lambda self: oracle
    )
    return RiverSwimReferenceEnvironment(
        _RIVER_CONFIG,
        observation_spec=_OBSERVATION_SPEC,
        action_spec=_ACTION_SPEC,
        executor_id="asi.riverswim.executor",
        executor_epoch="asi.riverswim.executor_epoch.1",
    )


def test_manifest_identity_is_invariant_under_last_ulp_oracle_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-host one-ulp oracle noise must not change any identity byte."""
    shard_host = _environment_with_oracle(_SHARD_HOST_ORACLE, monkeypatch)
    recomputed_host = _environment_with_oracle(_RECOMPUTED_HOST_ORACLE, monkeypatch)

    assert shard_host.manifest.descriptor() == recomputed_host.manifest.descriptor()
    assert shard_host.manifest.manifest_id == recomputed_host.manifest.manifest_id
    assert (
        shard_host.manifest.config_sha256 == recomputed_host.manifest.config_sha256
    )


def test_runtime_oracle_stays_bitwise_equal_to_the_embedded_manifest_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint metrics recompute ``oracle_reward_sum`` from the manifest.

    The bitwise gate in ``validate_checkpoint_state`` only holds when the
    runtime oracle feeding the metrics is the exact scalar embedded in the
    manifest, so the canonicalization has to apply to both or neither.
    """
    environment = _environment_with_oracle(_SHARD_HOST_ORACLE, monkeypatch)
    runtime_oracle = environment._oracle_reward
    embedded = environment.manifest.config["oracle_average_reward"]
    assert struct.pack(">d", runtime_oracle) == struct.pack(">d", embedded)


def test_canonical_oracle_preserves_twelve_significant_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound scalar keeps the oracle to 1e-12 relative precision."""
    environment = _environment_with_oracle(_SHARD_HOST_ORACLE, monkeypatch)
    embedded = environment.manifest.config["oracle_average_reward"]
    assert abs(embedded - _SHARD_HOST_ORACLE) <= 1e-12 * abs(_SHARD_HOST_ORACLE)
