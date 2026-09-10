from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

from .ablation import (
    DEFAULT_ARMS,
    factorial_effects,
    format_factorial_table,
    format_latex_table,
    parse_seeds,
    run_ablation,
)
from .calibration import (
    DEFAULT_PAIRS,
    calibrate,
    format_calibration_table,
    needs_calibration,
    parse_pairs,
)
from .config import load_config, validate_config, validate_config_pair
from .detection import DEFAULT_ATTACKS, format_pdet_table, measure_p_det
from .data import audit_csv_dataset, audit_csv_splits
from .macros import build_macros, find_hardcoded_numbers, write_macros
from .reachability import DEFAULT_CHALLENGES, format_reachability_table, measure_reachability
from .runner import ExperimentRunner
from .statistics import compare_family, format_paired_table


# One point of prevalence is the widest drift a rebuild of the same cohort should
# show; more than that is a different cohort.
PREVALENCE_TOLERANCE = 0.01


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="xfedagent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", default=None)

    ablation_parser = subparsers.add_parser(
        "ablation", help="sweep component ablation arms across seeds"
    )
    ablation_parser.add_argument("--config", required=True)
    ablation_parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="comma-separated arm labels, e.g. full,no-pov,no-rep,no-pov+rep",
    )
    ablation_parser.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    ablation_parser.add_argument("--output", default="results/ablation")
    ablation_parser.add_argument(
        "--latex", action="store_true", help="also print a LaTeX booktabs table body"
    )
    ablation_parser.add_argument(
        "--factorial",
        action="store_true",
        help="report 2x2 main effects and the interaction for the PoV/reputation design",
    )
    ablation_parser.add_argument(
        "--factorial-metric",
        default="balanced_accuracy_mean",
        help=(
            "outcome the 2x2 effects are computed on (default: "
            "balanced_accuracy_mean). On an imbalanced cohort a majority-class "
            "attack barely moves accuracy_mean"
        ),
    )

    pdet_parser = subparsers.add_parser(
        "pdet", help="measure per-attack detection rate with Wilson intervals"
    )
    pdet_parser.add_argument("--config", required=True)
    pdet_parser.add_argument(
        "--attacks",
        default=",".join(DEFAULT_ATTACKS),
        help="comma-separated attack names, measured one at a time",
    )
    pdet_parser.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    pdet_parser.add_argument("--output", default="results/pdet")
    pdet_parser.add_argument(
        "--latex", action="store_true", help="also print a LaTeX booktabs table body"
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="threshold sweep for the class-aware gate, under a pre-registered rule",
    )
    calibrate_parser.add_argument("--config", required=True)
    calibrate_parser.add_argument(
        "--pairs",
        default=",".join(f"{a:.2f}" for a, _ in DEFAULT_PAIRS),
        help="candidate thresholds, e.g. 0.65,0.70,0.75 or 0.65:0.70,0.75:0.75",
    )
    calibrate_parser.add_argument(
        "--seeds", default="0-2", help="pilot seeds, kept small on purpose"
    )
    calibrate_parser.add_argument("--output", default="results/calibration")
    calibrate_parser.add_argument(
        "--latex", action="store_true", help="also print a LaTeX booktabs table body"
    )
    calibrate_parser.add_argument(
        "--observed-false-reject",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "honest false-reject rate already measured at the operating point; the "
            "sweep is refused unless it exceeds the 15%% trigger"
        ),
    )

    stats_parser = subparsers.add_parser(
        "stats", help="paired seed-level comparisons over an ablation_runs.csv"
    )
    stats_parser.add_argument("--runs", required=True, help="path to ablation_runs.csv")
    stats_parser.add_argument(
        "--compare",
        action="append",
        required=True,
        metavar="METRIC:ARM_A:ARM_B",
        help="a family member, e.g. accuracy:full:no-pov; repeat for the whole family",
    )
    stats_parser.add_argument("--output", default="results/stats")
    stats_parser.add_argument("--family-label", default="unnamed")
    stats_parser.add_argument(
        "--latex", action="store_true", help="also print a LaTeX booktabs table body"
    )

    macros_parser = subparsers.add_parser(
        "macros",
        help="emit results_macros.tex, the single definition of every repeated number",
    )
    macros_parser.add_argument("--build", default="build", help="directory of circuit artefacts")
    macros_parser.add_argument(
        "--config",
        default=None,
        help="config whose thresholds and reputation constants fix q* and the "
        "random-admission bound the prose states",
    )
    macros_parser.add_argument("--ablation", default=None, help="ablation_manifest.json")
    macros_parser.add_argument(
        "--freerider",
        default=None,
        help="ablation_manifest.json of the replay free-rider arms (full vs no-copy)",
    )
    macros_parser.add_argument(
        "--freerider-runs",
        default=None,
        help="ablation_runs.csv of the same sweep; supplies the submission counts",
    )
    macros_parser.add_argument(
        "--freerider-stats",
        default=None,
        help="paired_manifest.json of the same sweep; supplies the paired p-values",
    )
    macros_parser.add_argument("--pdet", default=None, help="pdet_manifest.json")
    macros_parser.add_argument("--paired", default=None, help="paired_manifest.json")
    macros_parser.add_argument(
        "--scale",
        action="append",
        default=None,
        metavar="MANIFEST",
        help="ablation_manifest.json of one agent-count arm of the scale sweep; repeatable",
    )
    macros_parser.add_argument(
        "--scale-trend",
        default=None,
        help="trend_manifest.json from scripts/scale_trend.py",
    )
    macros_parser.add_argument(
        "--bound",
        default=None,
        help="bound_manifest.json from scripts/verify_exclusion_bound.py",
    )
    macros_parser.add_argument("--calibration", default=None, help="calibration_manifest.json")
    macros_parser.add_argument("--output", default="results_macros.tex")
    macros_parser.add_argument(
        "--cohort",
        default="surrogate",
        choices=("surrogate", "mimic"),
        help="which cohort the experiment numbers came from; sets their provenance label",
    )
    macros_parser.add_argument(
        "--lint",
        action="append",
        default=None,
        metavar="TEX",
        help="also report hard-coded copies of these numbers in a .tex file; repeatable",
    )

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument(
        "--pair-with",
        default=None,
        help="second arm of a paired comparison; only the gate may differ",
    )

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)

    reach_parser = subparsers.add_parser(
        "reachability",
        help="can the configured gate be satisfied on this cohort? one fit, not a sweep",
    )
    reach_parser.add_argument("--config", required=True)
    reach_parser.add_argument(
        "--challenges",
        type=int,
        default=DEFAULT_CHALLENGES,
        help="rotated validation subsets the verdict is averaged over",
    )
    reach_parser.add_argument("--output", default=None)
    reach_parser.add_argument(
        "--latex", action="store_true", help="also print a LaTeX booktabs table body"
    )

    subparsers.add_parser("env")

    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(args.config)
        if args.output is not None:
            object.__setattr__(cfg.output, "directory", args.output)
        summary = ExperimentRunner(cfg).run()
        print(f"run_name={summary['run_name']}")
        print(f"summary={Path(cfg.output.directory) / summary['run_name'] / 'summary.json'}")
        print(f"final_accuracy={summary['final_metrics']['accuracy']:.6f}")
        print(f"final_auc_roc={summary['final_metrics']['auc_roc']:.6f}")
        return
    if args.command == "ablation":
        cfg = load_config(args.config)
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
        seeds = parse_seeds(args.seeds)
        manifest = run_ablation(cfg, arms, seeds, args.output)
        print(f"runs_csv={manifest['runs_csv']}")
        print(f"summary_csv={manifest['summary_csv']}")
        if args.latex:
            print("--- LaTeX table body ---")
            print(format_latex_table(manifest))
        if args.factorial:
            effects = factorial_effects(manifest["aggregate"], metric=args.factorial_metric)
            print(f"--- 2x2 factorial effects on {args.factorial_metric} (percentage points) ---")
            for key in (
                "main_effect_pov",
                "main_effect_reputation",
                "interaction",
                "additive_prediction",
                "observed_both",
            ):
                print(f"{key}={effects[key] * 100:+.2f}")
            print(format_factorial_table(effects))
        return
    if args.command == "pdet":
        cfg = load_config(args.config)
        attacks = tuple(a.strip() for a in args.attacks.split(",") if a.strip())
        seeds = parse_seeds(args.seeds)
        manifest = measure_p_det(cfg, attacks, seeds, args.output)
        print(f"per_seed_csv={manifest['per_seed_csv']}")
        print(f"pooled_csv={manifest['pooled_csv']}")
        if args.latex:
            print("--- LaTeX table body ---")
            print(format_pdet_table(manifest))
        return
    if args.command == "calibrate":
        cfg = load_config(args.config)
        # The sweep runs only once the operating point has been measured and found
        # above the trigger, so a threshold is not tuned just because the command
        # exists.
        if args.observed_false_reject is not None and not needs_calibration(
            args.observed_false_reject
        ):
            print(
                f"honest_false_reject={args.observed_false_reject:.4f} is at or below the "
                f"0.15 trigger; operating point retained, no sweep run"
            )
            return
        pairs = parse_pairs(args.pairs)
        seeds = parse_seeds(args.seeds)
        manifest = calibrate(cfg, pairs, seeds, args.output)
        print(f"csv={manifest['csv']}")
        if args.latex:
            print("--- LaTeX table body ---")
            print(format_calibration_table(manifest))
        return
    if args.command == "stats":
        comparisons = []
        for spec in args.compare:
            parts = spec.split(":")
            if len(parts) != 3:
                raise SystemExit(f"--compare expects METRIC:ARM_A:ARM_B, got {spec!r}")
            comparisons.append((parts[0], parts[1], parts[2]))
        manifest = compare_family(args.runs, comparisons, args.output, args.family_label)
        print(f"csv={manifest['csv']}")
        if args.latex:
            print("--- LaTeX table body ---")
            print(format_paired_table(manifest))
        return
    if args.command == "macros":
        cohort = (
            "measured (surrogate cohort)"
            if args.cohort == "surrogate"
            else "measured (MIMIC-III cohort)"
        )
        collected = build_macros(
            build_dir=args.build,
            ablation_manifest=args.ablation,
            freerider_manifest=args.freerider,
            freerider_runs=args.freerider_runs,
            freerider_stats=args.freerider_stats,
            config_path=args.config,
            pdet_manifest=args.pdet,
            paired_manifest=args.paired,
            scale_manifests=args.scale,
            scale_trend_manifest=args.scale_trend,
            bound_manifest=args.bound,
            calibration_manifest=args.calibration,
            experiment_provenance=cohort,
        )
        result = write_macros(collected, args.output)
        print(f"tex={result['tex']}")
        print(f"provenance_csv={result['provenance_csv']}")
        print(f"macros={result['macros']}")
        for label, count in sorted(result["by_provenance"].items()):
            print(f"  {label}: {count}")
        if args.lint:
            hits = find_hardcoded_numbers(args.lint, collected)
            if not hits:
                print("lint: no hard-coded copies of a generated number")
            else:
                print("lint: numbers written by hand that a macro already defines")
                for name, places in sorted(hits.items()):
                    for path, lineno, line in places:
                        print(f"  \\{name} -> {path}:{lineno}: {line[:100]}")
        return
    if args.command == "validate-config":
        cfg = load_config(args.config)
        validate_config(cfg)
        print("config_valid=true")
        if args.pair_with:
            other = load_config(args.pair_with)
            validate_config(other)
            print("pair_config_valid=true")
            differences = validate_config_pair(
                json.loads(Path(args.config).read_text(encoding="utf-8")),
                json.loads(Path(args.pair_with).read_text(encoding="utf-8")),
            )
            print("paired=true")
            for path, (left, right) in sorted(differences.items()):
                print(f"  differs {path}: {left!r} -> {right!r}")
        return
    if args.command == "preflight":
        cfg = load_config(args.config)
        validate_config(cfg)
        checks, cohort = preflight_checks(cfg)
        failed = [name for name, ok in checks.items() if not ok]
        for name, ok in checks.items():
            print(f"{name}={str(ok).lower()}")
        # Aggregate cohort figures only: counts and rates, never a record.
        for name, value in cohort.items():
            print(f"cohort_{name}={value}")
        if failed:
            raise SystemExit(1)
        return
    if args.command == "reachability":
        cfg = load_config(args.config)
        validate_config(cfg)
        manifest = measure_reachability(cfg, args.challenges, args.output)
        print(f"predicate={manifest['predicate']}")
        print(f"decision_rule={manifest['decision_rule']}")
        if manifest["decision_rule"] == "rate":
            print(f"predicted_positive_rate={manifest['predicted_positive_rate']:.4f}")
        else:
            print(f"decision_threshold={manifest['decision_threshold']}")
        print(
            f"required_rates={manifest['effective_threshold']:.4f} "
            f"(tau {manifest['threshold_sensitivity']:.2f}/"
            f"{manifest['threshold_specificity']:.2f} less tolerance "
            f"{manifest['tolerance']:.2f})"
        )
        print(f"binormal_auc_threshold={manifest['auc_floor']:.4f} (equal-variance assumption)")
        for row in manifest["bounds"]:
            print(
                f"{row['bound']}: rows={row['train_rows']} trials={row['trials']} "
                f"auc={row['auc_roc']:.3f} "
                f"se={row['sensitivity']:.3f} sp={row['specificity']:.3f} "
                f"reachable_min={row['reachable_min_se_sp']:.3f}"
                f"(pred {row['predicted_min_se_sp']:.3f})@t="
                f"{row['reachable_threshold']:.3f} pass_rate={row['pass_rate']:.2f} "
                f"mean_score={row['mean_score']:.3f}"
            )
        print(
            f"suggested_decision_threshold={manifest['suggested_decision_threshold']}"
            f" (fitted reference: {manifest['binding_bound']}; requires calibration)"
        )
        # The reading that decides the verdict: not whether round 0 ever admits, but
        # how often a whole round of clients admits nothing.
        print(
            f"observed_round0_empty_probability={manifest['round0_stall_probability']:.4f}"
            f" over {manifest['clients_per_round']} drawn"
            f" (clients never admitted: {manifest['clients_never_admitted']})"
        )
        # A run can pass the stall test and still be unusable: §3.3's ladder is
        # triggered by the honest false-reject rate, not by whether round 0 starts.
        if manifest["honest_false_reject_above_trigger"]:
            print(
                f"implied_honest_false_reject={manifest['implied_honest_false_reject']:.4f}"
                f" exceeds the pre-registered trigger"
                f" {manifest['trigger_honest_false_reject']:.2f}: the run may start and"
                " still reject most honest updates, and the reputation penalty"
                " compounds that. Assess the pre-specified calibration procedure"
                " before changing the operating point."
            )
        if manifest["thresholds_disagree"]:
            print(
                "Candidate thresholds differ by "
                f"{manifest['reachable_threshold_gap']:.3f}; this heuristic does not "
                "establish that no shared threshold passes. A predicted-positive "
                f"rate of {manifest['suggested_predicted_positive_rate']:.4f} is an "
                "alternative policy requiring its own evaluation."
            )
        print(f"verdict={manifest['verdict']}")
        if "csv" in manifest:
            print(f"csv={manifest['csv']}")
        if args.latex:
            print("--- LaTeX table body ---")
            print(format_reachability_table(manifest))
        # A finite diagnostic is not an impossibility test and must not silently
        # veto a configured experiment on that interpretation.
        return
    if args.command == "env":
        for binary in ("python3", "circom", "snarkjs", "qemu-system-aarch64", "ipfs", "node", "npm"):
            path = shutil.which(binary)
            status = path if path else "missing"
            print(f"{binary}={status}")
        return
    raise SystemExit(f"unknown command: {args.command}")


def preflight_checks(cfg) -> tuple[dict[str, bool], dict[str, float | int]]:
    """Return the boolean gate checks and the aggregate cohort figures to log."""

    checks = {
        "python": shutil.which("python3") is not None,
        "node": shutil.which("node") is not None,
        "npm": shutil.which("npm") is not None,
    }
    if cfg.pov.backend == "snarkjs":
        checks["snarkjs"] = shutil.which(cfg.pov.snarkjs_bin) is not None
        checks["circuit_wasm"] = cfg.pov.circuit_wasm is not None and Path(cfg.pov.circuit_wasm).exists()
        checks["proving_key"] = cfg.pov.proving_key is not None and Path(cfg.pov.proving_key).exists()
    if cfg.toolchain.require_snarkjs:
        checks["snarkjs"] = shutil.which(cfg.pov.snarkjs_bin) is not None
    if cfg.toolchain.require_qemu:
        checks["qemu_system_aarch64"] = shutil.which("qemu-system-aarch64") is not None
    if cfg.toolchain.require_ipfs:
        checks["ipfs"] = shutil.which("ipfs") is not None
    if cfg.toolchain.require_hardhat:
        checks["hardhat"] = shutil.which("hardhat") is not None or Path("node_modules/.bin/hardhat").exists()
    if cfg.relay.enabled:
        checks["relay_quorum"] = cfg.relay.relayers > 3 * cfg.relay.faulty_relayers
    cohort: dict[str, float | int] = {}
    if cfg.data.source == "csv_timeseries":
        checks["dataset"] = cfg.data.csv_path is not None and Path(cfg.data.csv_path).exists()
        if checks["dataset"]:
            audit = audit_csv_dataset(cfg.data)
            for name in (
                "schema_valid",
                "patient_ids_complete",
                "binary_labels",
                "enough_patients",
            ):
                checks[f"dataset_{name}"] = bool(audit[name])
            for name in ("rows", "patients", "positives", "prevalence"):
                cohort[name] = audit[name]
            cohort["configured_positive_rate"] = cfg.data.positive_rate
            # A cohort whose prevalence has drifted is a different cohort, so the
            # preflight reports rather than runs on it.
            checks["dataset_prevalence_matches_config"] = (
                abs(float(audit["prevalence"]) - cfg.data.positive_rate) <= PREVALENCE_TOLERANCE
            )
            if checks["dataset_schema_valid"] and checks["dataset_enough_patients"]:
                splits = audit_csv_splits(cfg.data, cfg.federation.seed)
                for name in ("patient_disjoint", "validation_pool_satisfied", "rows_assigned_once"):
                    checks[f"dataset_{name}"] = bool(splits[name])
                for name in (
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                    "train_patients",
                    "validation_patients",
                    "test_patients",
                    "shared_patients",
                ):
                    cohort[name] = splits[name]
    return checks, cohort


if __name__ == "__main__":
    main(sys.argv[1:])
