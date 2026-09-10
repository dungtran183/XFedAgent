"""The paired-comparison guarantee, enforced before any compute is spent.

The two predicate arms share the seed, the data split, the model initialisation,
the client partition and the attack schedule, and only the gate may differ. These
tests pin that on the configurations the artifact ships, so a later edit to one arm
cannot silently unpair the comparison.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from xfedagent.config import (
    PAIRED_ALLOWED_DIFFERENCES,
    flatten_config,
    paired_differences,
    validate_config_pair,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHIPPED_PAIRS = (
    ("mimic-pacc.json", "mimic-pbal.json"),
    ("predicate-raw-accuracy.json", "predicate-class-aware.json"),
)


def _load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("left", "right"), SHIPPED_PAIRS)
def test_shipped_pairs_are_paired(left: str, right: str) -> None:
    differences = validate_config_pair(_load(left), _load(right))
    assert set(differences) <= PAIRED_ALLOWED_DIFFERENCES
    assert differences["pov.predicate"] == ("raw_accuracy", "class_aware")


def test_mimic_pair_differs_only_in_name_and_the_two_gate_switches() -> None:
    # The MIMIC arms are tighter than the principle requires: the thresholds are
    # identical too, so the only moving parts are the predicate and the sampler.
    assert set(paired_differences(_load("mimic-pacc.json"), _load("mimic-pbal.json"))) == {
        "name",
        "pov.predicate",
        "pov.balanced_validation",
    }


def test_mimic_arms_read_the_credentialed_cohort_not_the_surrogate() -> None:
    for name in ("mimic-pacc.json", "mimic-pbal.json"):
        config = _load(name)
        assert config["data"]["source"] == "csv_timeseries"
        assert config["data"]["csv_path"] == "data/processed/mimic_mortality_24h.csv"
        assert config["federation"]["rounds"] == 60


def test_a_differing_seed_unpairs_the_comparison() -> None:
    left = _load("mimic-pacc.json")
    right = deepcopy(_load("mimic-pbal.json"))
    right["federation"]["seed"] = 1
    with pytest.raises(ValueError, match="federation.seed"):
        validate_config_pair(left, right)


def test_a_differing_partition_unpairs_the_comparison() -> None:
    left = _load("mimic-pacc.json")
    right = deepcopy(_load("mimic-pbal.json"))
    right["data"]["dirichlet_alpha"] = 0.9
    with pytest.raises(ValueError, match="data.dirichlet_alpha"):
        validate_config_pair(left, right)


def test_two_arms_with_the_same_predicate_are_not_a_comparison() -> None:
    left = _load("mimic-pbal.json")
    right = deepcopy(left)
    right["name"] = "other"
    with pytest.raises(ValueError, match="same pov.predicate"):
        validate_config_pair(left, right)


def test_a_field_absent_from_one_arm_is_still_caught() -> None:
    # The comparison walks the raw mappings, so a key nobody has modelled in the
    # dataclasses cannot drift between the arms unnoticed.
    left = _load("mimic-pacc.json")
    right = deepcopy(_load("mimic-pbal.json"))
    right["federation"]["experimental_knob"] = True
    with pytest.raises(ValueError, match="federation.experimental_knob"):
        validate_config_pair(left, right)


def test_flatten_config_uses_dotted_paths() -> None:
    assert flatten_config({"a": 1, "b": {"c": {"d": 2}}, "e": [1, 2]}) == {
        "a": 1,
        "b.c.d": 2,
        "e": [1, 2],
    }
