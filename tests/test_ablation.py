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


# --------------------------------------------------------------------------
# Two-factor analysis
# --------------------------------------------------------------------------
from xfedagent.ablation import factorial_effects  # noqa: E402

#: The four MIMIC-III cells reported in the manuscript, as fractions.
_PAPER_CELLS = [
    {"arm": "no-pov+rep", "accuracy_mean": 0.487},
    {"arm": "no-pov", "accuracy_mean": 0.684},
    {"arm": "no-rep", "accuracy_mean": 0.762},
    {"arm": "full", "accuracy_mean": 0.831},
]


def test_factorial_reproduces_the_reported_effects():
    e = factorial_effects(_PAPER_CELLS, metric="accuracy_mean")
    assert round(e["main_effect_pov"] * 100, 1) == 21.1
    assert round(e["main_effect_reputation"] * 100, 1) == 13.3
    assert round(e["interaction"] * 100, 1) == -12.8


def test_interaction_is_negative_so_effects_are_not_additive():
    """Guards the claim the manuscript makes: the mechanisms are partially redundant."""
    e = factorial_effects(_PAPER_CELLS, metric="accuracy_mean")
    assert e["interaction"] < 0
    assert e["additive_prediction"] > e["observed_both"]


def test_interaction_is_symmetric_in_the_two_factors():
    e = factorial_effects(_PAPER_CELLS, metric="accuracy_mean")
    pov_gain_from_rep = e["simple_effect_pov_with_reputation"] - e["simple_effect_pov_without_reputation"]
    rep_gain_from_pov = e["simple_effect_reputation_with_pov"] - e["simple_effect_reputation_without_pov"]
    assert abs(pov_gain_from_rep - rep_gain_from_pov) < 1e-9
    assert abs(pov_gain_from_rep - e["interaction"]) < 1e-9


def test_factorial_requires_all_four_cells():
    with pytest.raises(ValueError, match="missing arm"):
        factorial_effects(_PAPER_CELLS[:3], metric="accuracy_mean")


def test_factorial_rejects_an_absent_metric():
    with pytest.raises(ValueError, match="absent from arm"):
        factorial_effects(_PAPER_CELLS, metric="auc_roc_mean")


#: The same four cells as measured on the surrogate cohort, where a majority-class
#: attack is nearly invisible to raw accuracy but not to the class-wise rates.
_SURROGATE_CELLS = [
    {"arm": "full", "accuracy_mean": 0.8886, "balanced_accuracy_mean": 0.8701},
    {"arm": "no-rep", "accuracy_mean": 0.9022, "balanced_accuracy_mean": 0.8605},
    {"arm": "no-pov", "accuracy_mean": 0.8994, "balanced_accuracy_mean": 0.6824},
    {"arm": "no-pov+rep", "accuracy_mean": 0.8856, "balanced_accuracy_mean": 0.6091},
]


def test_the_outcome_metric_decides_whether_the_gate_has_an_effect():
    """The choice of outcome is not cosmetic on a skewed cohort.

    Measured on raw accuracy the gate appears to do nothing; measured on balanced
    accuracy the same four runs give it a main effect of roughly twenty points.
    This is the paper's own argument about ``P_acc`` applied to its own ablation,
    so the metric has to be selected deliberately rather than inherited.
    """
    on_accuracy = factorial_effects(_SURROGATE_CELLS, metric="accuracy_mean")
    on_balanced = factorial_effects(_SURROGATE_CELLS, metric="balanced_accuracy_mean")
    assert abs(on_accuracy["main_effect_pov"] * 100) < 1.0
    assert on_balanced["main_effect_pov"] * 100 > 20.0
    # Both agree on the sign of the interaction: the mechanisms overlap.
    assert on_accuracy["interaction"] < 0 and on_balanced["interaction"] < 0


def test_the_reported_metric_is_carried_in_the_result():
    """A factorial without its outcome named cannot be checked against a table."""
    assert factorial_effects(_SURROGATE_CELLS)["metric"] == "balanced_accuracy_mean"
    e = factorial_effects(_SURROGATE_CELLS, metric="balanced_accuracy_mean")
    assert e["metric"] == "balanced_accuracy_mean"
