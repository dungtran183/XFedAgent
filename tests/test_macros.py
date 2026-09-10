"""Tests for the single-source-of-numbers generator.

Two failure modes matter here: a number can be defined twice and drift, and a
*modelled* figure can quietly become a *measured* one on its way into the
manuscript. The tests below pin the guards against each, plus the formatting the
LaTeX has to receive.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from xfedagent.macros import (
    BUILD_MEASURED,
    COMPUTED,
    LOCAL_EVM,
    MODELLED,
    SURROGATE,
    Macro,
    build_macros,
    check_unique,
    collect_ablation_macros,
    collect_calibration_macros,
    collect_circuit_macros,
    collect_bound_macros,
    collect_freerider_macros,
    collect_paired_macros,
    collect_pdet_macros,
    collect_scale_macros,
    collect_scale_trend_macros,
    collect_theory_macros,
    find_hardcoded_numbers,
    latex_float,
    latex_int,
    latex_percent,
    latex_signed,
    render_macros,
    write_macros,
)


# ---------------------------------------------------------------------------
# Fixtures: the smallest build directory the collectors will read
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def build_dir(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    write_csv(
        build / "circuit_metrics.csv",
        ["metric", "value", "unit", "provenance"],
        [
            ["r1cs_constraints", "268676", "count", BUILD_MEASURED],
            ["r1cs_wires", "270314", "count", BUILD_MEASURED],
            ["private_inputs", "1838", "count", BUILD_MEASURED],
            ["public_inputs", "11", "count", BUILD_MEASURED],
            ["proof_json_bytes", "806", "bytes", BUILD_MEASURED],
            ["verification_key_bytes", "4758", "bytes", BUILD_MEASURED],
            ["proving_key_mb", "121.583", "MB", BUILD_MEASURED],
            ["proving_seconds_mean", "36.37", "s", BUILD_MEASURED],
            ["proving_seconds_sd", "2.577", "s", BUILD_MEASURED],
            ["proving_seconds_max", "39.277", "s", BUILD_MEASURED],
            ["proving_max_rss_kb", "3864544", "KB", BUILD_MEASURED],
            ["proving_runs", "10", "count", BUILD_MEASURED],
            ["build_machine_cpu", "Apple M4", "-", "environment"],
        ],
    )
    write_csv(
        build / "gas_report.csv",
        ["measurement", "case", "gas_used", "provenance"],
        [
            ["deploy Groth16Verifier", "-", "653571", LOCAL_EVM],
            ["deploy PoVVerifierAdapter", "-", "301969", LOCAL_EVM],
            ["verifyProof (view)", "T0", "291388", LOCAL_EVM],
            ["verifyProof (view)", "T3", "291400", LOCAL_EVM],
            ["submitUpdate", "T0#1", "344886", LOCAL_EVM],
            ["submitUpdate", "T0#2", "310328", LOCAL_EVM],
        ],
    )
    write_csv(
        build / "test_report.csv",
        ["test_id", "expected", "actual", "exit_code", "status", "scenario"],
        [
            ["T0", "pass", "pass", "0", "OK", "honest witness"],
            ["T1", "fail", "fail", "1", "OK", "counts do not sum to n"],
            ["T2", "fail", "pass", "0", "MISMATCH", "below threshold"],
        ],
    )
    return build


def by_name(macros):
    return {m.name: m for m in macros}


# The 2x2 cells are chosen so the factorial arithmetic is checkable by hand:
# accuracy is 0.880 with both mechanisms, 0.845 with the gate alone, 0.850 with
# reputation alone and 0.810 with neither.
def _arm(arm, accuracy, **extra):
    row = {"arm": arm, "runs": 2}
    for field, mean in [
        ("accuracy", accuracy),
        ("balanced_accuracy", extra.get("balacc", accuracy)),
        ("auc_roc", extra.get("auc", 0.960)),
        ("sensitivity", extra.get("sens", 0.850)),
        ("specificity", extra.get("spec", 0.900)),
    ]:
        row[f"{field}_mean"] = mean
        row[f"{field}_std"] = 0.010
    row.update(extra.get("rates", {}))
    return row


ABLATION_MANIFEST = {
    "base_config": "warm-ca70maj",
    "arms": ["full", "no-pov", "no-rep", "no-pov+rep", "experimental-arm"],
    "seeds": [0, 1],
    "aggregate": [
        _arm("full", 0.880, rates={
            "honest_false_reject_rate_mean": 0.4627,
            "malicious_rejection_rate_mean": 1.0,
        }),
        _arm("no-pov", 0.850, sens=0.700, rates={
            "honest_false_reject_rate_mean": 0.0,
            "malicious_rejection_rate_mean": 0.31,
        }),
        _arm("no-rep", 0.845),
        _arm("no-pov+rep", 0.810),
        _arm("experimental-arm", 0.500),
    ],
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_thousands_separator_is_the_latex_one():
    # A bare comma in math mode sets the wrong spacing, so the digits are grouped
    # with {,} instead.
    assert latex_int(268676) == "268{,}676"
    assert latex_int("2341902") == "2{,}341{,}902"
    assert latex_int(806) == "806"
    assert latex_int(11.0) == "11"


def test_float_percent_and_signed_forms():
    assert latex_float(0.96714, 3) == "0.967"
    assert latex_percent(0.28095) == "28.1"
    assert latex_percent(1.0) == "100.0"
    assert latex_signed(-1.234, 2) == "-1.23"
    assert latex_signed(1.234, 2) == "+1.23"      # the sign must be explicit
    assert latex_signed(0.0, 1) == "+0.0"


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------
def test_a_macro_name_that_would_not_compile_is_refused():
    for bad in ("Circuit_Constraints", "Verify2Gas", "", "Tau-Se"):
        with pytest.raises(ValueError):
            Macro(bad, "1", COMPUTED, "src")


def test_render_is_a_newcommand():
    macro = Macro("CircuitConstraints", "268{,}676", BUILD_MEASURED, "build/x.csv")
    assert macro.render() == r"\newcommand{\CircuitConstraints}{268{,}676}"


def test_a_name_defined_twice_is_an_error_naming_both_sources():
    macros = [
        Macro("AccFull", "83.4", SURROGATE, "results/a/manifest.json"),
        Macro("AccFull", "87.8", SURROGATE, "results/b/manifest.json"),
    ]
    with pytest.raises(ValueError) as excinfo:
        check_unique(macros)
    message = str(excinfo.value)
    assert "AccFull" in message
    assert "results/a/manifest.json" in message
    assert "results/b/manifest.json" in message


# ---------------------------------------------------------------------------
# Circuit collector
# ---------------------------------------------------------------------------
def test_circuit_macros_carry_the_measured_values(build_dir):
    macros = by_name(collect_circuit_macros(build_dir))
    assert macros["CircuitConstraints"].value == "268{,}676"
    assert macros["CircuitPublicInputs"].value == "11"
    assert macros["CircuitProvingKeyMB"].value == "121.6"
    assert macros["ProvingSecondsBuild"].value == "36.4"
    assert macros["ProvingSecondsBuildSD"].value == "2.6"
    assert macros["ProvingPeakRSSMB"].value == "3{,}774"      # 3864544 KB -> MB
    assert macros["BuildMachineCPU"].value == "Apple M4"
    assert macros["CircuitConstraints"].provenance == BUILD_MEASURED


def test_gas_macros_summarise_the_distribution_not_one_call(build_dir):
    macros = by_name(collect_circuit_macros(build_dir))
    assert macros["VerifyGas"].value == "291{,}394"           # mean of the two
    assert macros["VerifyGasMin"].value == "291{,}388"
    assert macros["VerifyGasMax"].value == "291{,}400"
    assert macros["VerifyGasSamples"].value == "2"
    assert macros["SubmitUpdateGas"].value == "327{,}607"
    assert macros["VerifierDeployGas"].value == "653{,}571"
    assert macros["VerifyGas"].provenance == LOCAL_EVM


def test_test_report_counts_matches_not_passes(build_dir):
    # T2 is a MISMATCH: the suite has 3 cases and 2 behaved as specified.
    macros = by_name(collect_circuit_macros(build_dir))
    assert macros["CircuitTestCases"].value == "3"
    assert macros["CircuitTestsMatching"].value == "2"
    assert macros["CircuitTestCases"].provenance == COMPUTED


def test_a_modelled_row_stays_modelled(build_dir):
    # A modelled figure may not be promoted to a measured one: the collector must
    # pass the label through, never default it.
    write_csv(
        build_dir / "circuit_metrics.csv",
        ["metric", "value", "unit", "provenance"],
        [["r1cs_constraints", "268676", "count", MODELLED]],
    )
    macros = by_name(collect_circuit_macros(build_dir))
    assert macros["CircuitConstraints"].provenance == MODELLED


def test_a_missing_artefact_contributes_nothing(tmp_path):
    assert collect_circuit_macros(tmp_path / "absent") == []
    assert build_macros() == []


# ---------------------------------------------------------------------------
# Experiment collectors
# ---------------------------------------------------------------------------
def test_pdet_macros_keep_the_wilson_bounds_beside_the_estimate(tmp_path):
    path = tmp_path / "pdet_manifest.json"
    path.write_text(
        json.dumps(
            {
                "pooled": [
                    {
                        "attack": "label_flip",
                        "submitted": 540,
                        "admitted": 3,
                        "p_det": 0.994444,
                        "ci_low": 0.983,
                        "ci_high": 0.998,
                    },
                    {"attack": "not_an_attack", "submitted": 1, "admitted": 0,
                     "p_det": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                ]
            }
        )
    )
    macros = by_name(collect_pdet_macros(path))
    assert macros["PdetLabelFlip"].value == "99.4"
    assert macros["PdetLabelFlipCILow"].value == "98.3"
    assert macros["PdetLabelFlipCIHigh"].value == "99.8"
    assert macros["PdetLabelFlipSubmissions"].value == "540"
    assert macros["PdetLabelFlip"].provenance == SURROGATE
    # An attack the paper does not name is skipped rather than guessed at.
    assert not [name for name in macros if "NotAn" in name]


def test_paired_macros_name_the_family_members(tmp_path):
    path = tmp_path / "paired_manifest.json"
    path.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "metric": "accuracy",
                        "arm_a": "full",
                        "arm_b": "no-pov",
                        "mean_difference": 0.0421,
                        "ci_low": 0.0180,
                        "ci_high": 0.0662,
                        "p_value_holm": 0.00391,
                        "effect_size_dz": 1.42,
                    }
                ]
            }
        )
    )
    macros = by_name(collect_paired_macros(path))
    assert macros["AccuracyFullVsNoPovDiff"].value == "+4.2"
    assert macros["AccuracyFullVsNoPovCILow"].value == "+1.8"
    assert macros["AccuracyFullVsNoPovPHolm"].value == "0.0039"
    assert macros["AccuracyFullVsNoPovDz"].value == "+1.42"
    assert macros["PairedFamilySize"].value == "1"
    assert macros["PairedFamilySize"].provenance == COMPUTED


def test_calibration_macros_report_the_chosen_pair_and_its_rates(tmp_path):
    path = tmp_path / "calibration_manifest.json"
    path.write_text(
        json.dumps(
            {
                "chosen_pair": [0.65, 0.65],
                "seeds": [0, 1, 2],
                "cells": [
                    {"tau_sensitivity": 0.65, "tau_specificity": 0.65,
                     "honest_false_reject_rate": 0.1023, "malicious_admission_rate": 0.0},
                    {"tau_sensitivity": 0.70, "tau_specificity": 0.70,
                     "honest_false_reject_rate": 0.2810, "malicious_admission_rate": 0.0},
                ],
            }
        )
    )
    macros = by_name(collect_calibration_macros(path))
    assert macros["TauSensitivity"].value == "0.65"
    assert macros["TauSpecificity"].value == "0.65"
    assert macros["CalibrationSeeds"].value == "3"
    # The rates quoted are the chosen row's, not the first row's.
    assert macros["HonestFalseRejectAtTau"].value == "10.2"
    assert macros["MaliciousAdmissionAtTau"].value == "0.0"


def test_ablation_macros_name_each_arm_and_keep_the_spread(tmp_path):
    path = tmp_path / "ablation_manifest.json"
    path.write_text(json.dumps(ABLATION_MANIFEST))
    macros = by_name(collect_ablation_macros(path))
    assert macros["AccFull"].value == "88.0"
    assert macros["AccFullSD"].value == "1.0"
    assert macros["BalAccFull"].value == "88.0"
    assert macros["AUCFull"].value == "0.960"
    assert macros["SensNoPoV"].value == "70.0"
    assert macros["AblationSeeds"].value == "2"
    assert macros["AblationSeeds"].provenance == COMPUTED
    assert macros["AccFull"].provenance == SURROGATE
    # An arm the paper does not tabulate is skipped, not transliterated.
    assert not [name for name in macros if "Experimental" in name]


def test_ablation_macros_take_the_gate_rates_from_the_full_arm_only(tmp_path):
    # The no-pov arm has no gate, so its rejection rates say nothing about the
    # framework; quoting them as the framework's would be wrong.
    path = tmp_path / "ablation_manifest.json"
    path.write_text(json.dumps(ABLATION_MANIFEST))
    macros = by_name(collect_ablation_macros(path))
    assert macros["HonestFalseReject"].value == "46.3"
    assert macros["MaliciousRejection"].value == "100.0"


def test_factorial_effects_are_signed_but_accuracy_levels_are_not(tmp_path):
    # A difference needs its sign shown; an accuracy level rendered as "+87.4"
    # reads as a change of 87.4 points, which is not what the number means.
    path = tmp_path / "ablation_manifest.json"
    path.write_text(json.dumps(ABLATION_MANIFEST))
    macros = by_name(collect_ablation_macros(path))
    assert macros["MainEffectPoV"].value.startswith(("+", "-"))
    assert macros["FactorialInteraction"].value.startswith(("+", "-"))
    assert macros["AdditivePrediction"].value == "88.5"
    assert macros["ObservedBoth"].value == "88.0"
    # y[1][1] - y[1][0] - y[0][1] + y[0][0] = 88.0 - 84.5 - 85.0 + 81.0
    assert macros["FactorialInteraction"].value == "-0.5"
    assert macros["MainEffectPoVSensitivity"].value.startswith(("+", "-"))


def test_ablation_without_all_four_cells_omits_the_factorial(tmp_path):
    manifest = dict(ABLATION_MANIFEST)
    manifest["aggregate"] = [
        row for row in ABLATION_MANIFEST["aggregate"] if row["arm"] != "no-pov+rep"
    ]
    path = tmp_path / "ablation_manifest.json"
    path.write_text(json.dumps(manifest))
    macros = by_name(collect_ablation_macros(path))
    assert "AccFull" in macros
    assert "MainEffectPoV" not in macros


# ---------------------------------------------------------------------------
# Closed-form quantities the prose states
# ---------------------------------------------------------------------------
def write_config(path, **pov):
    base = json.loads(
        (CONFIG_DIR / "predicate-class-aware.json").read_text(encoding="utf-8")
    )
    base["pov"].update(pov)
    path.write_text(json.dumps(base))
    return path


def test_theory_macros_recompute_qstar_from_the_thresholds(tmp_path):
    # q* = a*Dmax / (b + a*Dmax) with a=0.1, b=0.5. At 0.65/0.65 Dmax is 0.35,
    # so q* = 0.035 / 0.535 = 6.5%; at 0.70/0.70 it is 0.030 / 0.530 = 5.7%.
    at65 = by_name(collect_theory_macros(write_config(
        tmp_path / "a.json", threshold_sensitivity=0.65, threshold_specificity=0.65)))
    assert at65["DeltaMax"].value == "0.35"
    assert at65["QStar"].value == "6.5"
    assert at65["QStar"].provenance == COMPUTED

    at70 = by_name(collect_theory_macros(write_config(
        tmp_path / "b.json", threshold_sensitivity=0.70, threshold_specificity=0.70)))
    assert at70["QStar"].value == "5.7"


def test_theory_macros_recompute_the_random_admission_bound(tmp_path):
    # The retained configs have tolerance 0.03, so nominal 0.70/0.65 become
    # effective 0.67/0.62. The uniform Bernoulli bound uses one class tail.
    at70 = by_name(collect_theory_macros(write_config(
        tmp_path / "b.json", threshold_sensitivity=0.70, threshold_specificity=0.70)))
    assert at70["RandomAdmitBound"].value == "0.056"
    assert "RandomAdmitRounds" not in at70

    at65 = by_name(collect_theory_macros(write_config(
        tmp_path / "a.json", threshold_sensitivity=0.65, threshold_specificity=0.65)))
    assert at65["RandomAdmitBound"].value == "0.237"
    assert "RandomAdmitRounds" not in at65
    assert "Bernoulli" in at65["RandomAdmitBound"].note
    assert "% Conditional on 50 positive and 50 negative rows" in render_macros(list(at65.values()))
    exact = by_name(collect_theory_macros(write_config(
        tmp_path / "exact.json", threshold_sensitivity=0.65,
        threshold_specificity=0.65, tolerance=0.0)))
    assert exact["RandomAdmitBound"].value == "0.105"
    assert exact["QStar"].value == at65["QStar"].value


def test_an_unbalanced_draw_is_governed_by_the_smaller_class(tmp_path):
    # The scenario rounds the prevalence to 14 positives. Actual unstratified
    # counts fluctuate, so the metadata must make the conditioning explicit.
    macros = by_name(collect_theory_macros(write_config(
        tmp_path / "c.json", threshold_sensitivity=0.60, threshold_specificity=0.60,
        balanced_validation=False)))
    assert macros["RandomAdmitBound"].value == "0.872"
    assert macros["ValidationPositives"].value == "14"
    assert "not realised counts" in macros["RandomAdmitBound"].note


@pytest.mark.parametrize("se,sp,expected", [(0.65, 0.75, "0.056"), (0.45, 0.45, "1.000")])
def test_theory_bound_handles_asymmetric_and_nonseparating_thresholds(tmp_path, se, sp, expected):
    macros = by_name(collect_theory_macros(write_config(
        tmp_path / "asymmetric.json", threshold_sensitivity=se, threshold_specificity=sp)))
    assert macros["RandomAdmitBound"].value == expected


def test_the_raw_accuracy_gate_has_no_class_wise_bound(tmp_path):
    macros = by_name(collect_theory_macros(write_config(
        tmp_path / "d.json", predicate="raw_accuracy", threshold=0.75)))
    assert macros["DeltaMax"].value == "0.25"
    assert macros["QStar"].value == "4.8"
    # The bound of Eq. (9) is stated for the class-aware gate only.
    assert "RandomAdmitBound" not in macros


def test_theory_macros_do_not_redefine_the_calibration_thresholds(tmp_path):
    # collect_calibration_macros already owns TauSensitivity/TauSpecificity; two
    # collectors defining the same name is the failure this module exists to stop.
    names = {m.name for m in collect_theory_macros(write_config(tmp_path / "e.json"))}
    assert not (names & {"TauSensitivity", "TauSpecificity", "CalibrationSeeds"})


# ---------------------------------------------------------------------------
# Rendering and writing
# ---------------------------------------------------------------------------
def test_render_groups_by_provenance_and_keeps_the_label_visible():
    macros = [
        Macro("VerifyGas", "291{,}396", LOCAL_EVM, "gas.csv"),
        Macro("CircuitConstraints", "268{,}676", BUILD_MEASURED, "metrics.csv"),
        Macro("EdgeEnergyJoules", "12.4", MODELLED, "cost_model.py"),
        Macro("CircuitTestCases", "10", COMPUTED, "report.csv"),
    ]
    text = render_macros(macros)
    assert f"% ---- {MODELLED} ----" in text
    assert "do not relabel" in text
    # Groups appear in the declared order: computed, build, local EVM, ..., modelled.
    assert text.index(f"% ---- {COMPUTED} ----") < text.index(f"% ---- {BUILD_MEASURED} ----")
    assert text.index(f"% ---- {BUILD_MEASURED} ----") < text.index(f"% ---- {MODELLED} ----")
    assert text.startswith("% results_macros.tex")
    assert "do not edit by hand" in text


def test_render_refuses_a_duplicate_before_writing_anything():
    with pytest.raises(ValueError):
        render_macros([Macro("A", "1", COMPUTED, "x"), Macro("A", "2", COMPUTED, "y")])


def test_write_macros_emits_the_tex_and_the_provenance_table(tmp_path, build_dir):
    macros = collect_circuit_macros(build_dir)
    result = write_macros(macros, tmp_path / "out" / "results_macros.tex")

    tex = (tmp_path / "out" / "results_macros.tex").read_text()
    assert r"\newcommand{\CircuitConstraints}{268{,}676}" in tex

    rows = list(csv.DictReader((tmp_path / "out" / "results_provenance.csv").open()))
    assert {"macro", "value", "provenance", "source"} == set(rows[0])
    entry = next(r for r in rows if r["macro"] == r"\CircuitConstraints")
    assert entry["provenance"] == BUILD_MEASURED
    assert entry["source"].endswith("circuit_metrics.csv")
    assert result["macros"] == len(macros)
    assert result["by_provenance"][BUILD_MEASURED] > 0


def test_every_generated_macro_is_defined_exactly_once(build_dir):
    text = render_macros(collect_circuit_macros(build_dir))
    names = [line.split("}")[0] for line in text.splitlines() if line.startswith(r"\newcommand")]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Lint: numbers written by hand that a macro already defines
# ---------------------------------------------------------------------------
def test_lint_finds_a_number_the_manuscript_repeats_by_hand(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\n".join(
            [
                r"The circuit compiles to 268{,}676 constraints.",
                r"% a comment mentioning 268{,}676 is not a claim",
                r"\newcommand{\CircuitConstraints}{268{,}676}",
                r"Verification costs \VerifyGas{} gas.",
            ]
        )
    )
    macros = [
        Macro("CircuitConstraints", "268{,}676", BUILD_MEASURED, "metrics.csv"),
        Macro("VerifyGas", "291{,}396", LOCAL_EVM, "gas.csv"),
    ]
    hits = find_hardcoded_numbers([tex], macros)
    assert set(hits) == {"CircuitConstraints"}          # the macro call is not a hit
    places = hits["CircuitConstraints"]
    assert [lineno for _, lineno, _ in places] == [1]   # comment and definition skipped


def test_lint_ignores_values_too_short_to_be_distinctive(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text("Section 11 discusses 10 clients over 60 rounds.")
    macros = [
        Macro("CircuitPublicInputs", "11", BUILD_MEASURED, "metrics.csv"),
        Macro("ProvingRuns", "10", BUILD_MEASURED, "metrics.csv"),
    ]
    assert find_hardcoded_numbers([tex], macros) == {}


def test_lint_does_not_match_a_number_inside_a_longer_one(tmp_path):
    # "0.0" occurs inside "0.037" and "0.020"; neither is an occurrence of the
    # value, and a lint that says otherwise is unusable on a real manuscript.
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\n".join(
            [
                r"Per-round energy is $0.037$\,Wh against $0.020$ for vanilla FL.",
                r"Proving takes 136.4 seconds in the worst case.",
                r"The build machine proves in 36.4 seconds.",
            ]
        )
    )
    macros = [
        Macro("MaliciousAdmissionAtTau", "0.0", SURROGATE, "calibration.json"),
        Macro("ProvingSecondsBuild", "36.4", BUILD_MEASURED, "metrics.csv"),
    ]
    hits = find_hardcoded_numbers([tex], macros)
    # An all-zero value cannot be searched for at all, so it is skipped outright.
    assert "MaliciousAdmissionAtTau" not in hits
    # 136.4 is not 36.4; only the standalone occurrence is reported.
    assert [lineno for _, lineno, _ in hits["ProvingSecondsBuild"]] == [3]


def test_lint_counts_significant_digits_not_characters(tmp_path):
    # "0.5" is four characters and one digit. A manuscript says 0.5 for a dozen
    # unrelated reasons, and a lint that reports every one of them is one nobody
    # runs; the previous character-length rule let it through.
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\n".join(
            [
                r"With $\alpha=0.1$, $\beta=0.5$ the threshold follows.",
                r"Accuracy lies within $0.5$--$1.1$\,pp of unprotected FedAvg.",
                r"The chosen $|\mathcal{D}_{\text{val}}|=100$ is the knee.",
                r"Proving takes 121.6 seconds on the build machine.",
            ]
        )
    )
    macros = [
        Macro("ReputationBeta", "0.5", COMPUTED, "config.json"),
        Macro("ValidationSubsetSize", "100", COMPUTED, "config.json"),
        Macro("CircuitProvingKeyMB", "121.6", BUILD_MEASURED, "metrics.csv"),
    ]
    hits = find_hardcoded_numbers([tex], macros)
    assert set(hits) == {"CircuitProvingKeyMB"}


def test_theory_macros_omit_the_configuration_inputs(tmp_path):
    # alpha, beta and |D_val| are inputs q* is computed from, not results, and
    # emitting them made the lint report 40 lines of noise against 1 real hit.
    names = {m.name for m in collect_theory_macros(write_config(tmp_path / "f.json"))}
    assert not (names & {"ReputationAlpha", "ReputationBeta", "ValidationSubsetSize"})
    assert {"QStar", "DeltaMax", "RandomAdmitBound"} <= names


def test_lint_matches_a_thousands_separated_value_on_its_own(tmp_path):
    tex = tmp_path / "paper.tex"
    tex.write_text(
        "\n".join(
            [
                r"The circuit has 268{,}676 constraints.",
                r"A larger system has 1{,}268{,}676 constraints.",
            ]
        )
    )
    macros = [Macro("CircuitConstraints", "268{,}676", BUILD_MEASURED, "metrics.csv")]
    hits = find_hardcoded_numbers([tex], macros)
    assert [lineno for _, lineno, _ in hits["CircuitConstraints"]] == [1]


# ---------------------------------------------------------------------------
# The free-rider, exclusion-bound and scale collectors
# ---------------------------------------------------------------------------
def _freerider_manifest(tmp_path: Path, pass_rate_no_copy: float = 0.9833) -> Path:
    path = tmp_path / "ablation_manifest.json"
    path.write_text(
        json.dumps(
            {
                "seeds": list(range(10)),
                "aggregate": [
                    {
                        "arm": "full",
                        "malicious_rejection_rate_mean": 1.0,
                        "honest_false_reject_rate_mean": 0.4617,
                        "accuracy_mean": 0.8886,
                        "balanced_accuracy_mean": 0.8701,
                    },
                    {
                        "arm": "no-copy",
                        "malicious_rejection_rate_mean": 1.0 - pass_rate_no_copy,
                        "honest_false_reject_rate_mean": 0.4686,
                        "accuracy_mean": 0.9011,
                        "balanced_accuracy_mean": 0.8749,
                    },
                ],
            }
        )
    )
    return path


def test_freerider_macros_report_the_measured_rate_not_a_round_number(tmp_path):
    """The pass rate without the gadget is 98.3%, and the macro must say so.

    The manuscript's previous claim was 100%. A collector that rounded to the
    nearest integer would reproduce the very error these macros exist to fix.
    """
    macros = by_name(collect_freerider_macros(_freerider_manifest(tmp_path)))
    assert macros["FreeRiderPassNoCopy"].value == "98.3"
    assert macros["FreeRiderPassFull"].value == "0.0"
    assert macros["FreeRiderSeeds"].value == "10"
    assert macros["FreeRiderSeeds"].provenance == COMPUTED


def test_freerider_macros_need_both_arms(tmp_path):
    path = tmp_path / "one_arm.json"
    path.write_text(json.dumps({"seeds": [0], "aggregate": [{"arm": "full"}]}))
    assert collect_freerider_macros(path) == []


def test_freerider_counts_come_from_the_runs_csv(tmp_path):
    """Denominators are summed over seeds, not averaged into a rate."""
    manifest = _freerider_manifest(tmp_path)
    runs = tmp_path / "ablation_runs.csv"
    write_csv(
        runs,
        ["arm", "seed", "malicious_submitted", "malicious_admitted"],
        [
            ["full", "0", "180", "0"],
            ["full", "1", "180", "0"],
            ["no-copy", "0", "180", "177"],
            ["no-copy", "1", "180", "177"],
        ],
    )
    macros = by_name(collect_freerider_macros(manifest, runs))
    assert macros["FreeRiderSubmissions"].value == "360"
    assert macros["FreeRiderAdmittedNoCopy"].value == "354"
    assert macros["FreeRiderAdmittedFull"].value == "0"
    assert macros["FreeRiderRefusedNoCopy"].value == "6"
    # The honest-rejection difference is signed, because its sign is the claim.
    assert macros["FreeRiderHonestFalseRejectDiff"].value == "-0.7"


def test_freerider_shielding_horizon_is_the_first_divergent_root(tmp_path):
    """The bit-identical-root horizon is where the two arms' commitments part.

    The claim in the manuscript is that reputation alone reproduces the fully
    protected system for the first several rounds, witnessed by equal Merkle
    roots. That horizon is a round index, so an off-by-one here would misstate
    the paper; the fixture makes the intended index unambiguous.
    """
    manifest = _freerider_manifest(tmp_path)
    results = tmp_path / "results"
    runs = results / "sweep" / "ablation_runs.csv"
    runs.parent.mkdir(parents=True)
    write_csv(
        runs,
        ["arm", "seed", "malicious_submitted", "malicious_admitted", "run_name"],
        [
            ["full", "0", "180", "0", "run-full-0"],
            ["no-copy", "0", "180", "177", "run-nocopy-0"],
        ],
    )
    for name, roots in (("run-full-0", "aaa"), ("run-nocopy-0", "aab")):
        d = results / name
        d.mkdir()
        write_csv(d / "rounds.csv", ["round", "global_model_root"],
                  [[str(i), r] for i, r in enumerate(roots)])
    macros = by_name(collect_freerider_macros(manifest, runs))
    assert macros["FreeRiderShieldRoundsMedian"].value == "2"
    assert macros["FreeRiderShieldRoundsMin"].value == "2"


def test_bound_macros_expose_both_the_failing_and_the_holding_form(tmp_path):
    """The reader has to be able to see which bound failed and which held.

    A collector that emitted only the passing form would leave the response
    letter asserting the published paraphrase is wrong with no number attached.
    """
    path = tmp_path / "bound_manifest.json"
    path.write_text(
        json.dumps(
            {
                "q_star": 0.0654205607476635,
                "bisected_boundary": 0.0654227,
                "boundary_relative_error": 0.0000335,
                "banking_target": 0.7,
                "bounds": {
                    "rates_excluded": 2500,
                    "published_violations": 2074,
                    "slack_violations": 0,
                    "slack_mean_looseness": 2.49,
                    "slack_max_looseness": 4.63,
                },
                "order_independence": {
                    "orderings_above_qstar": 18759,
                    "survivors_above_qstar": 0,
                    "orderings_below_qstar": 1241,
                    "survivors_below_qstar": 311,
                },
            }
        )
    )
    macros = by_name(collect_bound_macros(path))
    assert macros["BoundQStar"].value == "0.065421"
    assert macros["BoundPublishedViolations"].value == "2{,}074"
    assert macros["BoundSlackViolations"].value == "0"
    assert macros["BoundSurvivorsAbove"].value == "0"
    assert macros["BoundBankingTarget"].value == "0.70"
    # Every one of these follows from alpha, beta, r_0, r_min and the predicate,
    # so none of them may be labelled as measured on a cohort.
    assert {m.provenance for m in collect_bound_macros(path)} == {COMPUTED}


def _scale_manifest(tmp_path: Path, clients: int, accuracy: float) -> Path:
    path = tmp_path / f"n{clients}.json"
    path.write_text(
        json.dumps(
            {
                "base_config": f"scale-n{clients}-scaled",
                "seeds": [0, 1, 2, 3, 4],
                "aggregate": [
                    {
                        "arm": "full",
                        "accuracy_mean": accuracy,
                        "accuracy_std": 0.01,
                        "balanced_accuracy_mean": accuracy - 0.01,
                        "balanced_accuracy_std": 0.005,
                        "auc_roc_mean": 0.9516,
                        "sensitivity_mean": 0.856,
                        "specificity_mean": 0.8916,
                        "malicious_rejection_rate_mean": 0.9556,
                        "honest_false_reject_rate_mean": 0.5348,
                    }
                ],
            }
        )
    )
    return path


def test_scale_macro_names_spell_the_agent_count(tmp_path):
    """A LaTeX control word is letters only, so \\ScaleN100Acc cannot exist.

    The agent count is read from each manifest rather than passed alongside it,
    so a manifest filed under the wrong population produces a wrong macro name
    rather than a silently transposed row.
    """
    paths = [_scale_manifest(tmp_path, n, 0.88) for n in (10, 25, 100)]
    macros = by_name(collect_scale_macros(paths))
    assert macros["ScaleTenAcc"].value == "88.0"
    assert macros["ScaleTwentyFiveAcc"].value == "88.0"
    assert macros["ScaleHundredAcc"].value == "88.0"
    assert macros["ScaleMaxAgents"].value == "100"
    assert "ScaleN100Acc" not in macros


def test_scale_trend_macros_keep_the_sign_of_the_trend(tmp_path):
    """A falling malicious-rejection trend must not print as a rising one.

    The sign is the finding: the manuscript previously claimed detection improves
    with population, and the measured trend is the other way. latex_signed is
    what carries that, so an unsigned formatter here would erase the correction.
    """
    path = tmp_path / "trend_manifest.json"
    path.write_text(
        json.dumps(
            {
                "agent_counts": [10, 25, 50, 100],
                "seeds_per_cell": {"10": 5, "25": 5, "50": 5, "100": 5},
                "metrics": {
                    "accuracy": {"z": 0.27, "p_value": 0.7878},
                    "balanced_accuracy": {"z": 2.36, "p_value": 0.0185},
                    "auc_roc": {"z": 2.69, "p_value": 0.0071},
                    "malicious_rejection_rate": {"z": -2.79, "p_value": 0.0052},
                },
            }
        )
    )
    macros = by_name(collect_scale_trend_macros(path))
    assert macros["ScaleTrendMalRejZ"].value == "-2.79"
    assert macros["ScaleTrendAccuracyZ"].value == "+0.27"
    assert macros["ScaleTrendAccuracyP"].value == "0.7878"
    assert macros["ScaleTrendSeedsPerCell"].value == "5"
