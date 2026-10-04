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

import math
import struct

import pytest

from alberta_framework.reference_agent import SpaceSpec
from alberta_framework.reference_life import RiverSwimReferenceEnvironment
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

_SHARD_HOST_ORACLE = 0.8571501418872561
_RECOMPUTED_HOST_ORACLE = 0.857150141887256

_DIGITS = 12  # must equal _CANONICAL_ORACLE_SIGNIFICANT_DIGITS in reference_life

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
    """The bound scalar keeps the oracle to 5e-12 relative precision.

    At 12 significant digits the worst-case rounding error of a value in
    ``[10**e, 10**(e+1))`` is ``0.5 * 10**(e-11)``, i.e. ``5e-12`` relative
    at the bottom of the decade.  Exercised on the historical host pair and
    on values from two other decimal decades so the bound is a property of
    the helper, not of one lucky magnitude.
    """
    for oracle in (
        _SHARD_HOST_ORACLE,
        _RECOMPUTED_HOST_ORACLE,
        12.345678901234567,  # one decade up, exact 17-digit tail
        0.012345678901234567,  # two decades down
    ):
        environment = _environment_with_oracle(oracle, monkeypatch)
        embedded = environment.manifest.config["oracle_average_reward"]
        assert abs(embedded - oracle) <= 5e-12 * abs(oracle)
        # Canonicalization is a projection: re-binding the bound scalar is
        # the identity, so repeated manifest builds cannot drift.
        again = _environment_with_oracle(embedded, monkeypatch)
        assert struct.pack(">d", again.manifest.config["oracle_average_reward"]) == struct.pack(
            ">d", embedded
        )


def _canonical(oracle: float) -> float:
    with pytest.MonkeyPatch.context() as monkeypatch:
        environment = _environment_with_oracle(oracle, monkeypatch)
    return environment.manifest.config["oracle_average_reward"]


def test_adjacent_float64_values_straddling_a_lattice_boundary_split() -> None:
    """The lattice genuinely resolves: a boundary-straddling ulp pair splits.

    Decimal formatting has rounding boundaries; adjacent IEEE-754 values on
    opposite sides of one boundary produce different canonical scalars and
    therefore different manifest hashes.  This is the failure mode the
    canonicalization exists to contain, so the test pins both halves: the
    straddling pair splits (the lattice has teeth), while the historical
    cross-host pair — which differs by one ulp but stays inside one
    quantum — collapses to a single scalar.
    """
    value = _SHARD_HOST_ORACLE
    straddling = None
    candidate = value
    for _ in range(20000):  # a 12-digit quantum is ~9000 ulps near 0.857
        bits = struct.unpack(">q", struct.pack(">d", candidate))[0] + 1
        candidate = struct.unpack(">d", struct.pack(">q", bits))[0]
        if f"{candidate:.12g}" != f"{value:.12g}":
            straddling = candidate
            break
    assert straddling is not None, "no lattice boundary near the test oracle"

    assert _canonical(value) != _canonical(straddling)
    # And the actual cross-host disagreement stays inside one quantum:
    assert _canonical(_SHARD_HOST_ORACLE) == _canonical(_RECOMPUTED_HOST_ORACLE)


def test_configured_scorecard_oracle_clears_the_nearest_lattice_boundary() -> None:
    """Host lstsq noise cannot push the configured oracle across a boundary.

    Boundary-straddling hosts only break identity if the true oracle sits
    within host-variation distance of a lattice boundary.  For the exact
    configuration the development scorecard runs (``RIVERSWIM_*`` in
    ``reference_life_scorecard``), assert the distance from the oracle to
    the nearest 12-significant-digit rounding boundary exceeds 100 float64
    ulps of the oracle itself.  The measured margin is over 2000 ulps,
    while the observed cross-host lstsq disagreement is 1 ulp; a finer 13-
    digit lattice would shrink the margin to ~54 ulps and is rejected for
    exactly that reason.
    """
    from alberta_framework.benchmarks.reference_life_scorecard import (
        RIVERSWIM_N_STATES,
        RIVERSWIM_P_RIGHT_DOWN,
        RIVERSWIM_P_RIGHT_UP,
        RIVERSWIM_REWARD_LEFT,
        RIVERSWIM_REWARD_RIGHT,
    )

    config = RiverSwimConfig(  # type: ignore[call-arg]
        n_states=RIVERSWIM_N_STATES,
        p_right_up=RIVERSWIM_P_RIGHT_UP,
        p_right_down=RIVERSWIM_P_RIGHT_DOWN,
        reward_left=RIVERSWIM_REWARD_LEFT,
        reward_right=RIVERSWIM_REWARD_RIGHT,
        initial_state=0,
    )
    oracle = float(RiverSwimMDP(config).optimal_average_reward())

    quantum = 10.0 ** (math.floor(math.log10(abs(oracle))) - (_DIGITS - 1))
    position = oracle / quantum
    distance_to_boundary = abs(position - math.floor(position) - 0.5)
    margin_in_ulps = distance_to_boundary * quantum / math.ulp(oracle)
    assert margin_in_ulps > 100, (
        f"configured oracle sits only {margin_in_ulps:.0f} ulps from the "
        "nearest lattice boundary; host BLAS variation could straddle it"
    )
