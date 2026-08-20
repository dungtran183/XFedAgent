from __future__ import annotations

import pytest

from xfedagent.ablation import DEFAULT_ARMS, _arm_config, parse_arm, parse_seeds
from xfedagent.config import AblationConfig


def test_full_arm_leaves_every_component_enabled():
    assert parse_arm("full") == AblationConfig()
    assert parse_arm("full").is_full


@pytest.mark.parametrize(
    "arm,field",
    [
        ("no-pov", "pov_enabled"),
        ("no-copy", "copy_detector_enabled"),
        ("no-rep", "reputation_enabled"),
        ("no-rot", "rotation_enabled"),
        ("no-xchain", "cross_chain_enabled"),
    ],
)
def test_single_component_arms_disable_exactly_one_field(arm, field):
    ablation = parse_arm(arm)
    disabled = [k for k, v in ablation.__dict__.items() if v is False]
    assert disabled == [field]


def test_compound_arm_disables_both_components():
    ablation = parse_arm("no-pov+rep")
    assert ablation.pov_enabled is False
    assert ablation.reputation_enabled is False
    assert ablation.rotation_enabled is True


def test_arm_label_round_trips():
    for arm in DEFAULT_ARMS:
        assert parse_arm(arm).label == arm


@pytest.mark.parametrize("bad", ["", "pov", "no-", "no-bogus", "no-pov+bogus"])
def test_malformed_arms_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_arm(bad)


def test_parse_seeds_supports_ranges_and_lists():
    assert parse_seeds("0-3") == (0, 1, 2, 3)
    assert parse_seeds("0,4,9") == (0, 4, 9)
    assert parse_seeds("0-1,5") == (0, 1, 5)


@pytest.mark.parametrize("bad", ["", "3-1"])
def test_malformed_seed_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_seeds(bad)


def test_cross_chain_arm_actually_disables_the_relay(synthetic_config):
    """A single-chain arm must stop relaying, not merely carry a label."""
    derived = _arm_config(synthetic_config, parse_arm("no-xchain"), seed=7)
    assert derived.relay.enabled is False
    assert derived.federation.seed == 7


def test_other_arms_leave_the_relay_running(synthetic_config):
    derived = _arm_config(synthetic_config, parse_arm("no-pov"), seed=2)
    assert derived.relay.enabled is synthetic_config.relay.enabled
    assert derived.federation.seed == 2


def test_arm_config_does_not_mutate_the_base(synthetic_config):
    before = synthetic_config.to_dict()
    _arm_config(synthetic_config, parse_arm("no-xchain"), seed=5)
    assert synthetic_config.to_dict() == before
