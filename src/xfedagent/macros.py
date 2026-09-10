"""One source for every number that appears more than once in the manuscript.

Numbers that live in several places — the abstract, a table, the body, the
conclusion, the supplement — drift apart when they are edited by hand. This module
reads the measured artefacts and emits ``results_macros.tex``, so each number is
written once and every occurrence in the manuscript is a macro call.

Provenance travels with the value. Each macro is emitted under a comment naming
how it was obtained, and the companion CSV records the same label per macro, so
the distinction between *computed*, *measured (build machine)*,
*emulator-measured* and *modelled* stays attached to the number. Nothing here promotes a modelled figure to a
measured one; the collectors only pass through the label their source recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import re

#: Provenance labels, in the fixed vocabulary the manuscript uses.
COMPUTED = "computed"
BUILD_MEASURED = "measured (build machine)"
EMULATOR_MEASURED = "emulator-measured"
MODELLED = "modelled"
LOCAL_EVM = "measured (local EVM)"
SURROGATE = "measured (surrogate cohort)"
ENVIRONMENT = "environment"

_NAME_RE = re.compile(r"^[A-Za-z]+$")


@dataclass(frozen=True)
class Macro:
    """One ``\\newcommand`` with the provenance of its value."""

    name: str
    value: str
    provenance: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        # LaTeX command names admit only letters; a digit or underscore in one would
        # make the emitted file fail to compile.
        if not _NAME_RE.match(self.name):
            raise ValueError(f"macro name must be letters only, got {self.name!r}")

    def render(self) -> str:
        return f"\\newcommand{{\\{self.name}}}{{{self.value}}}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def latex_int(value: float | int | str) -> str:
    """``2341902`` -> ``2{,}341{,}902``, the form that keeps LaTeX spacing right."""
    n = int(float(value))
    return f"{n:,}".replace(",", "{,}")


def latex_float(value: float | int | str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def latex_percent(value: float | int | str, digits: int = 1) -> str:
    """A rate in [0, 1] rendered as a percentage *without* the sign."""
    return f"{float(value) * 100:.{digits}f}"


def latex_signed(value: float | int | str, digits: int = 1) -> str:
    return f"{float(value):+.{digits}f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _camel(label: str) -> str:
    """``no-pov+rep`` -> ``NoPovRep``; a LaTeX-safe, letters-only rendering."""
    return "".join(part.title() for part in re.split(r"[^A-Za-z]+", label) if part)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------
def collect_circuit_macros(build_dir: str | Path) -> list[Macro]:
    """Circuit size, key sizes, proving distribution, gas, and the test outcome.

    Every value here is measured on the build machine. The edge-device figures the
    paper reports stay modelled and are not touched by this collector.
    """
    build = Path(build_dir)
    macros: list[Macro] = []

    metrics_path = build / "circuit_metrics.csv"
    if metrics_path.exists():
        by_metric = {r["metric"]: r for r in _read_csv(metrics_path)}

        def metric(key: str) -> str | None:
            row = by_metric.get(key)
            return row["value"] if row else None

        def provenance(key: str, default: str = BUILD_MEASURED) -> str:
            row = by_metric.get(key)
            return row["provenance"] if row and row.get("provenance") else default

        integers = [
            ("CircuitConstraints", "r1cs_constraints"),
            ("CircuitWires", "r1cs_wires"),
            ("CircuitPrivateInputs", "private_inputs"),
            ("CircuitPublicInputs", "public_inputs"),
            ("CircuitProofBytes", "proof_json_bytes"),
            ("CircuitVerificationKeyBytes", "verification_key_bytes"),
        ]
        for name, key in integers:
            raw = metric(key)
            if raw is not None:
                macros.append(
                    Macro(name, latex_int(raw), provenance(key), str(metrics_path))
                )

        pk_mb = metric("proving_key_mb")
        if pk_mb is not None:
            macros.append(
                Macro(
                    "CircuitProvingKeyMB",
                    latex_float(pk_mb, 1),
                    provenance("proving_key_mb"),
                    str(metrics_path),
                )
            )

        for name, key, digits in [
            ("ProvingSecondsBuild", "proving_seconds_mean", 1),
            ("ProvingSecondsBuildSD", "proving_seconds_sd", 1),
            ("ProvingSecondsBuildMax", "proving_seconds_max", 1),
        ]:
            raw = metric(key)
            if raw is not None:
                macros.append(
                    Macro(name, latex_float(raw, digits), provenance(key), str(metrics_path))
                )

        rss = metric("proving_max_rss_kb")
        if rss is not None:
            macros.append(
                Macro(
                    "ProvingPeakRSSMB",
                    latex_int(round(float(rss) / 1024)),
                    provenance("proving_max_rss_kb"),
                    str(metrics_path),
                )
            )

        runs = metric("proving_runs")
        if runs is not None:
            macros.append(Macro("ProvingRuns", latex_int(runs), BUILD_MEASURED, str(metrics_path)))

        cpu = metric("build_machine_cpu")
        if cpu:
            macros.append(Macro("BuildMachineCPU", cpu, ENVIRONMENT, str(metrics_path)))

    gas_path = build / "gas_report.csv"
    if gas_path.exists():
        rows = _read_csv(gas_path)
        # TrustAnchor calls the adapter, so the adapter path is the cost the paper
        # quotes. The generated verifier is measured beside it and kept as its own
        # macro rather than pooled, since the two are different code paths.
        verify = [r for r in rows if r["measurement"].startswith("verifyProof")]
        if verify:
            values = [int(r["gas_used"]) for r in verify]
            macros.append(
                Macro("VerifyGas", latex_int(round(sum(values) / len(values))), LOCAL_EVM, str(gas_path))
            )
            macros.append(Macro("VerifyGasMin", latex_int(min(values)), LOCAL_EVM, str(gas_path)))
            macros.append(Macro("VerifyGasMax", latex_int(max(values)), LOCAL_EVM, str(gas_path)))
            macros.append(
                Macro("VerifyGasSamples", latex_int(len(values)), LOCAL_EVM, str(gas_path))
            )
        raw_verify = [r for r in rows if r["measurement"].startswith("groth16 verifyProof")]
        if raw_verify:
            values = [int(r["gas_used"]) for r in raw_verify]
            macros.append(
                Macro(
                    "VerifyGasGenerated",
                    latex_int(round(sum(values) / len(values))),
                    LOCAL_EVM,
                    str(gas_path),
                )
            )
        submit = [r for r in rows if r["measurement"].startswith("submitUpdate")]
        if submit:
            values = [int(r["gas_used"]) for r in submit]
            macros.append(
                Macro(
                    "SubmitUpdateGas",
                    latex_int(round(sum(values) / len(values))),
                    LOCAL_EVM,
                    str(gas_path),
                )
            )
        deploy = next((r for r in rows if r["measurement"].startswith("deploy Groth16")), None)
        if deploy:
            macros.append(
                Macro("VerifierDeployGas", latex_int(deploy["gas_used"]), LOCAL_EVM, str(gas_path))
            )

    report_path = build / "test_report.csv"
    if report_path.exists():
        rows = _read_csv(report_path)
        macros.append(Macro("CircuitTestCases", latex_int(len(rows)), COMPUTED, str(report_path)))
        macros.append(
            Macro(
                "CircuitTestsMatching",
                latex_int(sum(1 for r in rows if r["status"] == "OK")),
                COMPUTED,
                str(report_path),
            )
        )
    return macros


def collect_ablation_macros(manifest_path: str | Path, provenance: str = SURROGATE) -> list[Macro]:
    """Per-arm accuracy and the 2x2 effects, from an ablation manifest."""
    from .ablation import factorial_effects  # local import keeps the module import-light

    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    macros: list[Macro] = []
    by_arm = {row["arm"]: row for row in manifest["aggregate"]}

    macros.append(
        Macro("AblationSeeds", latex_int(len(manifest["seeds"])), COMPUTED, source)
    )

    # Arm labels carry a hyphen and a plus, neither of which a LaTeX command name
    # admits, so they are mapped to letter-only suffixes.
    suffix = {
        "full": "Full",
        "no-pov": "NoPoV",
        "no-rep": "NoRep",
        "no-pov+rep": "NoPoVRep",
        "no-rot": "NoRot",
        "no-copy": "NoCopy",
        "no-xchain": "NoXChain",
    }
    for arm, row in by_arm.items():
        tag = suffix.get(arm)
        if tag is None:
            continue
        for field, macro_stem, fmt in [
            ("accuracy", "Acc", latex_percent),
            ("balanced_accuracy", "BalAcc", latex_percent),
            ("auc_roc", "AUC", lambda v: latex_float(v, 3)),
            ("sensitivity", "Sens", latex_percent),
            ("specificity", "Spec", latex_percent),
        ]:
            mean = row.get(f"{field}_mean")
            if mean is None:
                continue
            macros.append(Macro(f"{macro_stem}{tag}", fmt(mean), provenance, source))
            std = row.get(f"{field}_std")
            if std is not None:
                macros.append(Macro(f"{macro_stem}{tag}SD", fmt(std), provenance, source))

    full = by_arm.get("full")
    if full is not None:
        for field, name in [
            ("honest_false_reject_rate_mean", "HonestFalseReject"),
            ("malicious_rejection_rate_mean", "MaliciousRejection"),
        ]:
            if field in full:
                macros.append(Macro(name, latex_percent(full[field]), provenance, source))

    if {"full", "no-pov", "no-rep", "no-pov+rep"} <= set(by_arm):
        # The primary outcome is balanced accuracy: raw accuracy is dominated by
        # the majority class on this skewed cohort, which is the failure mode the
        # class-aware predicate exists to fix.
        effects = factorial_effects(
            manifest["aggregate"], metric="balanced_accuracy_mean"
        )
        sensitivity_effects = factorial_effects(
            manifest["aggregate"], metric="sensitivity_mean"
        )
        raw_accuracy_effects = factorial_effects(
            manifest["aggregate"], metric="accuracy_mean"
        )
        # The first three are differences in percentage points and need their sign
        # shown; the last two are accuracy levels, and "+88.5" would read as a
        # change of 88.5 points rather than as the level it is.
        signed = lambda v: latex_signed(v * 100.0)  # noqa: E731 - a formatter, inline
        for key, name, fmt in [
            ("main_effect_pov", "MainEffectPoV", signed),
            ("main_effect_reputation", "MainEffectRep", signed),
            ("interaction", "FactorialInteraction", signed),
            ("additive_prediction", "AdditivePrediction", latex_percent),
            ("observed_both", "ObservedBoth", latex_percent),
        ]:
            if key in effects:
                macros.append(Macro(name, fmt(effects[key]), provenance, source))
        for key, name in [
            ("main_effect_pov", "MainEffectPoVSensitivity"),
            ("main_effect_reputation", "MainEffectRepSensitivity"),
            ("interaction", "FactorialInteractionSensitivity"),
        ]:
            macros.append(
                Macro(name, signed(sensitivity_effects[key]), provenance, source)
            )
        for key, name in [
            ("main_effect_pov", "MainEffectPoVRawAccuracy"),
            ("main_effect_reputation", "MainEffectRepRawAccuracy"),
            ("interaction", "FactorialInteractionRawAccuracy"),
        ]:
            macros.append(
                Macro(name, signed(raw_accuracy_effects[key]), provenance, source)
            )
    return macros


def collect_pdet_macros(manifest_path: str | Path, provenance: str = SURROGATE) -> list[Macro]:
    """Pooled p_det per attack, with the Wilson bounds that go beside it."""
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    tag = {
        "majority_class": "MajorityClass",
        "label_flip": "LabelFlip",
        "random_gradient": "RandomGradient",
        "alie": "ALIE",
        "minmax": "MinMax",
        "backdoor": "Backdoor",
    }
    macros: list[Macro] = []
    for row in manifest["pooled"]:
        name = tag.get(row["attack"])
        if name is None:
            continue
        macros.append(Macro(f"Pdet{name}", latex_percent(row["p_det"]), provenance, source))
        macros.append(Macro(f"Pdet{name}CILow", latex_percent(row["ci_low"]), provenance, source))
        macros.append(Macro(f"Pdet{name}CIHigh", latex_percent(row["ci_high"]), provenance, source))
        macros.append(
            Macro(f"Pdet{name}Submissions", latex_int(row["submitted"]), provenance, source)
        )
    return macros


def collect_freerider_macros(
    manifest_path: str | Path,
    runs_csv: str | Path | None = None,
    stats_manifest: str | Path | None = None,
    provenance: str = SURROGATE,
) -> list[Macro]:
    """Copy-detector effect against the replay free-rider, from the on/off ablation.

    The manuscript previously asserted that every replayed global model passes PoV
    without the gadget. It does not: the round-0 replay carries the *untrained*
    initialisation, which the utility predicate rejects on its own merits, so the
    measured pass rate is short of 100% by exactly one round of adversaries per
    seed. The denominator is emitted too, because a bare rate invites the reader
    to reconstruct the wrong experiment.
    """
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    by_arm = {row["arm"]: row for row in manifest["aggregate"]}
    if not {"full", "no-copy"} <= set(by_arm):
        return []
    seeds = len(manifest["seeds"])
    macros = [
        Macro(
            "FreeRiderPassNoCopy",
            latex_percent(1.0 - by_arm["no-copy"]["malicious_rejection_rate_mean"], 1),
            provenance,
            source,
        ),
        Macro(
            "FreeRiderPassFull",
            latex_percent(1.0 - by_arm["full"]["malicious_rejection_rate_mean"], 1),
            provenance,
            source,
        ),
        Macro("FreeRiderSeeds", latex_int(seeds), COMPUTED, source),
    ]

    if runs_csv is None:
        return macros
    rows = [r for r in _read_csv(Path(runs_csv)) if r["arm"] in {"full", "no-copy"}]
    if not rows:
        return macros
    submitted = {arm: 0 for arm in ("full", "no-copy")}
    admitted = dict(submitted)
    for row in rows:
        submitted[row["arm"]] += int(row["malicious_submitted"])
        admitted[row["arm"]] += int(row["malicious_admitted"])
    csv_source = str(runs_csv)
    macros += [
        Macro("FreeRiderSubmissions", latex_int(submitted["no-copy"]), provenance, csv_source),
        Macro("FreeRiderAdmittedNoCopy", latex_int(admitted["no-copy"]), provenance, csv_source),
        Macro("FreeRiderAdmittedFull", latex_int(admitted["full"]), provenance, csv_source),
        Macro(
            "FreeRiderRefusedNoCopy",
            latex_int(submitted["no-copy"] - admitted["no-copy"]),
            provenance,
            csv_source,
        ),
    ]
    for metric, stem in (
        ("accuracy", "FreeRiderAccuracy"),
        ("balanced_accuracy", "FreeRiderBalancedAccuracy"),
    ):
        macros.append(
            Macro(
                f"{stem}Diff",
                latex_signed(
                    (by_arm["full"][f"{metric}_mean"] - by_arm["no-copy"][f"{metric}_mean"]) * 100.0
                ),
                provenance,
                source,
            )
        )
    # The honest-rejection side of the claim: the gadget must not cost false rejections.
    delta = (
        by_arm["full"]["honest_false_reject_rate_mean"]
        - by_arm["no-copy"]["honest_false_reject_rate_mean"]
    )
    macros.append(
        Macro("FreeRiderHonestFalseRejectDiff", latex_signed(delta * 100.0, 1), provenance, source)
    )

    # With the detector off, replays pass PoV from round 1, but the round-0 refusal
    # costs beta = r_0 and zeroes the free-rider, which cannot re-enter the average
    # until roughly beta / (alpha * Delta) further rounds are banked. Until then the
    # two arms commit bit-identical global models.
    if not all("run_name" in row and "seed" in row for row in rows):
        return macros
    results_root = Path(runs_csv).parent.parent
    by_seed: dict[str, dict[str, str]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row["run_name"]
    horizons: list[int] = []
    for arms in by_seed.values():
        if {"full", "no-copy"} - set(arms):
            continue
        roots = {}
        for arm in ("full", "no-copy"):
            path = results_root / arms[arm] / "rounds.csv"
            if not path.exists():
                return macros
            roots[arm] = [r["global_model_root"] for r in _read_csv(path)]
        pairs = zip(roots["full"], roots["no-copy"])
        first = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
        if first is not None:
            horizons.append(first)
    if horizons:
        horizons.sort()
        mid = len(horizons) // 2
        median = (
            float(horizons[mid])
            if len(horizons) % 2
            else (horizons[mid - 1] + horizons[mid]) / 2.0
        )
        macros += [
            Macro("FreeRiderShieldRoundsMedian", latex_int(int(round(median))), provenance, csv_source),
            Macro("FreeRiderShieldRoundsMin", latex_int(horizons[0]), provenance, csv_source),
            Macro("FreeRiderShieldRoundsMax", latex_int(horizons[-1]), provenance, csv_source),
        ]

    if stats_manifest is None:
        return macros
    # Paired seed-level tests for the same sweep. Only the three metrics the prose
    # quotes are lifted: the gadget's soundness gain is deterministic and needs no
    # test, but its *cost* and the honest-rejection null both do. ``effect_size_dz``
    # is not exported: for a difference with zero variance across seeds it diverges.
    wanted = {
        "accuracy": "FreeRiderAccuracy",
        "balanced_accuracy": "FreeRiderBalancedAccuracy",
        "honest_false_reject_rate": "FreeRiderHonestFalseReject",
    }
    stats_source = str(stats_manifest)
    for row in _read_json(stats_manifest)["comparisons"]:
        stem = wanted.get(row["metric"])
        if stem is None or {row["arm_a"], row["arm_b"]} != {"full", "no-copy"}:
            continue
        macros += [
            Macro(f"{stem}CILow", latex_signed(row["ci_low"] * 100.0, 1), provenance, stats_source),
            Macro(f"{stem}CIHigh", latex_signed(row["ci_high"] * 100.0, 1), provenance, stats_source),
            Macro(f"{stem}P", latex_float(row["p_value"], 4), provenance, stats_source),
            Macro(f"{stem}PHolm", latex_float(row["p_value_holm"], 4), provenance, stats_source),
        ]
    return macros


_UNITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
_TENS = ("", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def _spell_int(value: int) -> str:
    """Spell a small non-negative integer, so it can sit inside a macro name.

    A LaTeX control word admits letters only, so an agent count cannot be spliced
    into the name numerically. Only the values the sweep uses need to work; a
    count outside the supported range raises rather than silently colliding with
    another macro.
    """
    if value < 10:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (_UNITS[unit] if unit else "")
    if value == 100:
        return "Hundred"
    raise ValueError(f"no spelling for agent count {value}")


def collect_scale_macros(
    manifest_paths: list[str | Path], provenance: str = SURROGATE
) -> list[Macro]:
    """The measured agent-count sweep, one manifest per population size.

    The previous version of the scalability table carried accuracy at 25, 50 and
    100 agents that no configuration in the repository produced: the values were
    a smooth interpolation from the ten-agent measurement. These macros come from
    runs that exist, one per agent count, so the table can be regenerated and the
    interpolation cannot creep back in. The agent count is read from each
    manifest's own config rather than passed in, so a mislabelled column is a
    build error rather than a silent transposition.
    """
    macros: list[Macro] = []
    counts: list[int] = []
    for path in manifest_paths:
        manifest = _read_json(path)
        source = str(path)
        rows = [row for row in manifest["aggregate"] if row["arm"] == "full"]
        if not rows:
            continue
        row = rows[0]
        clients = int(str(manifest["base_config"]).split("-n")[1].split("-")[0])
        counts.append(clients)
        # LaTeX control words are letters only, so the agent count is spelled out:
        # \ScaleHundredAcc, not \ScaleN100Acc, which would not parse.
        stem = f"Scale{_spell_int(clients)}"
        for name, key, scale, digits in (
            ("Acc", "accuracy_mean", 100.0, 1),
            ("AccSD", "accuracy_std", 100.0, 1),
            ("BalAcc", "balanced_accuracy_mean", 100.0, 1),
            ("BalAccSD", "balanced_accuracy_std", 100.0, 1),
            ("Auc", "auc_roc_mean", 1.0, 3),
            ("Sens", "sensitivity_mean", 100.0, 1),
            ("Spec", "specificity_mean", 100.0, 1),
            ("MalRej", "malicious_rejection_rate_mean", 100.0, 1),
            ("HonestFalseReject", "honest_false_reject_rate_mean", 100.0, 1),
        ):
            macros.append(Macro(f"{stem}{name}", latex_float(row[key] * scale, digits), provenance, source))
        macros.append(Macro(f"{stem}Seeds", latex_int(len(manifest["seeds"])), COMPUTED, source))
    if counts:
        macros.append(
            Macro("ScaleMaxAgents", latex_int(max(counts)), COMPUTED, "config agent counts")
        )
    return macros


def collect_scale_trend_macros(manifest_path: str | Path) -> list[Macro]:
    """Ordered-alternative trend tests over the agent-count sweep.

    The manuscript asserted that accuracy improves with population size because
    reputation gains statistical power to identify malicious agents. Both halves
    are now tested rather than asserted, and only one survives, so the p-values
    for the three metrics the revised text quotes are emitted here. The test is
    a property of the sweep's per-seed values, so its provenance follows the runs.
    """
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    wanted = {
        "accuracy": "ScaleTrendAccuracy",
        "balanced_accuracy": "ScaleTrendBalancedAccuracy",
        "auc_roc": "ScaleTrendAuc",
        "malicious_rejection_rate": "ScaleTrendMalRej",
    }
    macros: list[Macro] = []
    for metric, stem in wanted.items():
        row = manifest["metrics"].get(metric)
        if row is None:
            continue
        macros.append(Macro(f"{stem}P", latex_float(row["p_value"], 4), SURROGATE, source))
        macros.append(Macro(f"{stem}Z", latex_signed(row["z"], 2), SURROGATE, source))
    macros.append(
        Macro("ScaleTrendSeedsPerCell", latex_int(min(manifest["seeds_per_cell"].values())), COMPUTED, source)
    )
    return macros


def collect_bound_macros(manifest_path: str | Path) -> list[Macro]:
    """Theorem 2's boundary and hitting-time bound, checked against the recursion.

    Everything here is a property of the reputation update rather than of a run,
    so the provenance is ``computed``: the same numbers follow from the config's
    alpha, beta, r_0, r_min and the predicate's largest awardable margin with no
    federated execution involved. They are macros rather than literals because
    the response letter quotes them and the reviewer is entitled to regenerate
    them with one command.
    """
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    bounds = manifest["bounds"]
    orders = manifest["order_independence"]
    return [
        Macro("BoundQStar", latex_float(manifest["q_star"], 6), COMPUTED, source),
        Macro("BoundBoundary", latex_float(manifest["bisected_boundary"], 6), COMPUTED, source),
        Macro("BoundBoundaryErrorPct", latex_float(manifest["boundary_relative_error"] * 100.0, 3), COMPUTED, source),
        Macro("BoundBankingTarget", latex_float(manifest["banking_target"], 2), COMPUTED, source),
        Macro("BoundRatesTested", latex_int(bounds["rates_excluded"]), COMPUTED, source),
        Macro("BoundPublishedViolations", latex_int(bounds["published_violations"]), COMPUTED, source),
        Macro("BoundSlackViolations", latex_int(bounds["slack_violations"]), COMPUTED, source),
        Macro("BoundSlackLooseness", latex_float(bounds["slack_mean_looseness"], 1), COMPUTED, source),
        Macro("BoundOrderingsAbove", latex_int(orders["orderings_above_qstar"]), COMPUTED, source),
        Macro("BoundSurvivorsAbove", latex_int(orders["survivors_above_qstar"]), COMPUTED, source),
        Macro("BoundOrderingsBelow", latex_int(orders["orderings_below_qstar"]), COMPUTED, source),
        Macro("BoundSurvivorsBelow", latex_int(orders["survivors_below_qstar"]), COMPUTED, source),
    ]


def collect_paired_macros(manifest_path: str | Path, provenance: str = SURROGATE) -> list[Macro]:
    """Paired differences, Holm-adjusted p-values and paired effect sizes."""
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    macros: list[Macro] = []
    seen: set[str] = set()
    for row in manifest["comparisons"]:
        # Arm labels carry hyphens and pluses ("no-pov+rep") and metric names carry
        # underscores ("auc_roc"); both are camel-cased per token so the joined name
        # reads as one word. Each arm is cased on its own, or "Vs" would swallow the
        # first letter of the arm behind it ("FullVsnoPov").
        stem = _camel(row["metric"])
        base = f"{stem}{_camel(row['arm_a'])}Vs{_camel(row['arm_b'])}"
        if base in seen:          # two identical comparisons would collide silently
            continue
        seen.add(base)
        macros.append(Macro(f"{base}Diff", latex_signed(row["mean_difference"] * 100), provenance, source))
        macros.append(Macro(f"{base}CILow", latex_signed(row["ci_low"] * 100), provenance, source))
        macros.append(Macro(f"{base}CIHigh", latex_signed(row["ci_high"] * 100), provenance, source))
        macros.append(Macro(f"{base}PHolm", latex_float(row["p_value_holm"], 4), provenance, source))
        macros.append(Macro(f"{base}Dz", latex_signed(row["effect_size_dz"], 2), provenance, source))
        # Per-arm terminal means and spreads, so a table comparing the two arms
        # can be typeset from the same manifest the paired test came from. Without
        # these, one column of every such table would be a hand-copied literal.
        scale = 1.0 if row["metric"] == "auc_roc" else 100.0
        digits = 3 if row["metric"] == "auc_roc" else 1
        for side, arm in (("a", row["arm_a"]), ("b", row["arm_b"])):
            if f"mean_{side}" not in row:   # a comparison-only manifest carries no arm means
                continue
            label = f"{stem}{_camel(arm)}"
            macros.append(Macro(f"{label}Mean", latex_float(row[f"mean_{side}"] * scale, digits), provenance, source))
            macros.append(Macro(f"{label}SD", latex_float(row[f"sd_{side}"] * scale, digits), provenance, source))
    macros.append(
        Macro("PairedFamilySize", latex_int(len(manifest["comparisons"])), COMPUTED, source)
    )
    return macros


def collect_calibration_macros(manifest_path: str | Path, provenance: str = SURROGATE) -> list[Macro]:
    """The chosen threshold pair and the rates that justified it."""
    manifest = _read_json(manifest_path)
    source = str(manifest_path)
    chosen = manifest["chosen_pair"]
    macros = [
        Macro("TauSensitivity", latex_float(chosen[0], 2), COMPUTED, source),
        Macro("TauSpecificity", latex_float(chosen[1], 2), COMPUTED, source),
        Macro(
            "CalibrationSeeds", latex_int(len(manifest["seeds"])), COMPUTED, source
        ),
    ]
    for row in manifest["cells"]:
        if [row["tau_sensitivity"], row["tau_specificity"]] != chosen:
            continue
        macros.append(
            Macro(
                "HonestFalseRejectAtTau",
                latex_percent(row["honest_false_reject_rate"]),
                provenance,
                source,
            )
        )
        macros.append(
            Macro(
                "MaliciousAdmissionAtTau",
                latex_percent(row["malicious_admission_rate"]),
                provenance,
                source,
            )
        )
    return macros


def collect_theory_macros(config_path: str | Path) -> list[Macro]:
    """Nominal-reward quantities and a conditional Bernoulli admission bound.

    ``q*`` uses nominal reward thresholds. The one-round admission calculation
    uses effective thresholds after tolerance and assumes independent,
    input-independent Bernoulli predictions conditional on the stated class
    counts. A configured prevalence gives only a candidate count scenario for an
    unstratified draw, not a bound on its realised counts. Arbitrary untrained
    networks and the software rate rule need not satisfy the Bernoulli model.

    No automatic multi-round horizon is emitted. Exponentiating a one-round bound
    requires that the same upper bound hold conditional on every prior history;
    rotating a reused validation pool does not establish that requirement, and
    consecutive admissions do not equal reputation survival.

    ``TauSensitivity`` and ``TauSpecificity`` are deliberately *not* emitted:
    :func:`collect_calibration_macros` owns them, because the calibration
    manifest records which pair the pre-registered rule actually selected.
    """
    from .config import parse_config  # local import keeps the module import-light

    source = str(config_path)
    cfg = parse_config(_read_json(config_path))
    pov, fed, data = cfg.pov, cfg.federation, cfg.data
    alpha, beta = fed.reputation_reward_alpha, fed.reputation_penalty_beta

    if pov.predicate == "class_aware":
        delta_max = 1.0 - max(pov.threshold_sensitivity, pov.threshold_specificity)
    else:
        delta_max = 1.0 - pov.threshold
    q_star = alpha * delta_max / (beta + alpha * delta_max)

    macros = [
        Macro("DeltaMax", latex_float(delta_max, 2), COMPUTED, source),
        Macro("QStar", latex_percent(q_star), COMPUTED, source),
    ]
    # alpha, beta and |D_val| are configuration *inputs*, not results. They are
    # read above because q* depends on them, but they are not emitted: the paper
    # states them once in the setup, and as macros their values ("0.5", "100") would
    # match dozens of unrelated numbers and make the lint unusable.
    if pov.predicate != "class_aware":
        return macros

    # At least one of sensitivity p and specificity 1-p is below its effective
    # cutoff by gamma. One Hoeffding tail on that class suffices; no union factor
    # is needed. This is a conditional calculation for the stated class counts.
    n = pov.validation_size
    if pov.balanced_validation:
        n_pos = n // 2
    else:
        n_pos = round(n * data.positive_rate)
    n_neg = n - n_pos
    effective_se = pov.threshold_sensitivity - pov.tolerance
    effective_sp = pov.threshold_specificity - pov.tolerance
    gamma = max(0.0, (effective_se + effective_sp - 1.0) / 2.0)
    bound = math.exp(-2 * min(n_pos, n_neg) * gamma**2)
    assumptions = (
        f"Conditional on {n_pos} positive and {n_neg} negative rows and independent, "
        "input-independent Bernoulli predictions; effective thresholds include "
        "tolerance. Configured class counts require sufficient pool support and "
        "are a scenario, not realised counts, for unstratified sampling."
    )

    macros += [
        Macro("ValidationPositives", latex_int(n_pos), COMPUTED, source, assumptions),
        Macro("RandomAdmitBound", latex_float(bound, 3), COMPUTED, source, assumptions),
    ]
    return macros


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
#: Provenance groups in the order they are written, each with the sentence that
#: explains what the label licenses a reader to conclude.
_GROUP_NOTES = {
    COMPUTED: "Derived from the protocol or the constraint system itself.",
    BUILD_MEASURED: "Measured on the build machine named below, not on an edge device.",
    LOCAL_EVM: "Measured against a local EVM, not a public network.",
    EMULATOR_MEASURED: "Measured inside the QEMU emulator, not on physical hardware.",
    SURROGATE: "Measured on the surrogate cohort, not on MIMIC-III.",
    MODELLED: "Derived from a cost model. Not a measurement; do not relabel.",
    ENVIRONMENT: "Describes where the measurements were taken.",
}


def check_unique(macros: list[Macro]) -> None:
    """A name defined twice is the failure mode this file exists to prevent."""
    seen: dict[str, Macro] = {}
    for macro in macros:
        if macro.name in seen:
            first = seen[macro.name]
            raise ValueError(
                f"macro \\{macro.name} defined twice: {first.value!r} from {first.source} "
                f"and {macro.value!r} from {macro.source}"
            )
        seen[macro.name] = macro


def render_macros(macros: list[Macro]) -> str:
    """Emit the ``.tex`` body, grouped by provenance with the label kept visible."""
    check_unique(macros)
    lines = [
        "% results_macros.tex — generated; do not edit by hand.",
        "% Regenerate with: python3 -m xfedagent macros ...",
        "%",
        "% Every number the manuscript repeats is defined here once. Provenance is",
        "% recorded per group below and per macro in results_provenance.csv.",
        "",
    ]
    ordered = [g for g in _GROUP_NOTES if any(m.provenance == g for m in macros)]
    ordered += sorted({m.provenance for m in macros} - set(_GROUP_NOTES))
    for group in ordered:
        members = [m for m in macros if m.provenance == group]
        if not members:
            continue
        note = _GROUP_NOTES.get(group, "")
        lines.append(f"% ---- {group} ----")
        if note:
            lines.append(f"% {note}")
        for macro in sorted(members, key=lambda m: m.name):
            if macro.note:
                lines.extend(f"% {line}" for line in macro.note.splitlines())
            lines.append(macro.render())
        lines.append("")
    return "\n".join(lines)


def write_macros(
    macros: list[Macro],
    output_tex: str | Path,
    provenance_csv: str | Path | None = None,
) -> dict:
    """Write ``results_macros.tex`` and, beside it, the per-macro provenance table."""
    tex_path = Path(output_tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(render_macros(macros), encoding="utf-8")

    csv_path = Path(provenance_csv) if provenance_csv else tex_path.with_name(
        "results_provenance.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["macro", "value", "provenance", "source"])
        for macro in sorted(macros, key=lambda m: m.name):
            writer.writerow([f"\\{macro.name}", macro.value, macro.provenance, macro.source])

    counts: dict[str, int] = {}
    for macro in macros:
        counts[macro.provenance] = counts.get(macro.provenance, 0) + 1
    return {
        "tex": str(tex_path),
        "provenance_csv": str(csv_path),
        "macros": len(macros),
        "by_provenance": counts,
    }


def build_macros(
    build_dir: str | Path | None = None,
    ablation_manifest: str | Path | None = None,
    freerider_manifest: str | Path | None = None,
    freerider_runs: str | Path | None = None,
    freerider_stats: str | Path | None = None,
    pdet_manifest: str | Path | None = None,
    paired_manifest: str | Path | None = None,
    scale_manifests: list[str | Path] | None = None,
    scale_trend_manifest: str | Path | None = None,
    bound_manifest: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    config_path: str | Path | None = None,
    experiment_provenance: str = SURROGATE,
) -> list[Macro]:
    """Collect whatever artefacts are present. Absent inputs contribute nothing."""
    macros: list[Macro] = []
    if build_dir is not None:
        macros += collect_circuit_macros(build_dir)
    if ablation_manifest is not None:
        macros += collect_ablation_macros(ablation_manifest, experiment_provenance)
    if freerider_manifest is not None:
        macros += collect_freerider_macros(
            freerider_manifest, freerider_runs, freerider_stats, experiment_provenance
        )
    if pdet_manifest is not None:
        macros += collect_pdet_macros(pdet_manifest, experiment_provenance)
    if paired_manifest is not None:
        macros += collect_paired_macros(paired_manifest, experiment_provenance)
    if scale_manifests:
        macros += collect_scale_macros(list(scale_manifests), experiment_provenance)
    if scale_trend_manifest is not None:
        macros += collect_scale_trend_macros(scale_trend_manifest)
    if bound_manifest is not None:
        macros += collect_bound_macros(bound_manifest)
    if calibration_manifest is not None:
        macros += collect_calibration_macros(calibration_manifest, experiment_provenance)
    if config_path is not None:
        macros += collect_theory_macros(config_path)
    check_unique(macros)
    return macros


def find_hardcoded_numbers(
    tex_files: list[str | Path], macros: list[Macro], minimum_digits: int = 3
) -> dict[str, list[tuple[str, int, str]]]:
    """Find places in the manuscript that write a number a macro already defines.

    The checklist asks that every repeated number come from ``results_macros.tex``.
    A literal ``83.4`` sitting in the body while ``\\AccFull`` expands to ``83.4``
    is a number maintained in two places, which is how the previous round's figures
    drifted. Only values that some macro defines are searched for, so the check
    stays specific rather than flagging every digit in the paper.

    A match must stand alone: ``0.0`` inside ``0.037`` is not an occurrence of the
    value ``0.0``, and reporting it as one would bury the real hits. Values whose
    digits are all zeros are skipped entirely, because they carry no identifying
    information — every ``0.0`` in a table would match.

    Distinctiveness is counted in *significant* digits, not characters. ``0.5`` is
    four characters wide but carries one digit, and a manuscript says ``0.5`` for
    a dozen unrelated reasons; a lint that reports all of them is one nobody runs.
    Leading and trailing zeros therefore do not count: ``100`` is as likely to be
    an agent count or a sample size as the validation subset it was defined for,
    and a round number cannot be distinguished from an unrelated round number.
    The cost is that a genuinely repeated round value goes unreported, which is
    the right way round for a check whose output a human has to read.

    Returns ``{macro name: [(file, line number, line)]}``.
    """
    patterns: list[tuple[Macro, re.Pattern[str]]] = []
    for macro in macros:
        value = macro.value
        if value.startswith("%"):
            continue
        digits = [ch for ch in value if ch.isdigit()]
        if not digits or set(digits) == {"0"}:
            continue
        significant = "".join(digits).strip("0")
        if len(significant) < minimum_digits:
            continue
        # Neither side may continue the number: no digit, decimal point, or the
        # {,} thousands separator LaTeX uses.
        patterns.append(
            (
                macro,
                re.compile(
                    r"(?<![\d.])(?<!\{,\})" + re.escape(value) + r"(?![\d.])(?!\{,\})"
                ),
            )
        )

    hits: dict[str, list[tuple[str, int, str]]] = {}
    for path in tex_files:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith(r"\newcommand"):
                continue
            for macro, pattern in patterns:
                if pattern.search(line):
                    hits.setdefault(macro.name, []).append((str(path), lineno, line.strip()))
    return hits
