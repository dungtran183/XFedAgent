from __future__ import annotations

from pathlib import Path
import argparse
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
from .config import load_config, validate_config
from .runner import ExperimentRunner


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

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", required=True)

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
            effects = factorial_effects(manifest["aggregate"])
            print("--- 2x2 factorial effects (percentage points) ---")
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
    if args.command == "validate-config":
        cfg = load_config(args.config)
        validate_config(cfg)
        print("config_valid=true")
        return
    if args.command == "preflight":
        cfg = load_config(args.config)
        validate_config(cfg)
        checks = preflight_checks(cfg)
        failed = [name for name, ok in checks.items() if not ok]
        for name, ok in checks.items():
            print(f"{name}={str(ok).lower()}")
        if failed:
            raise SystemExit(1)
        return
    if args.command == "env":
        for binary in ("python3", "circom", "snarkjs", "qemu-system-aarch64", "ipfs", "node", "npm"):
            path = shutil.which(binary)
            status = path if path else "missing"
            print(f"{binary}={status}")
        return
    raise SystemExit(f"unknown command: {args.command}")


def preflight_checks(cfg) -> dict[str, bool]:
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
    if cfg.data.source == "csv_timeseries":
        checks["dataset"] = cfg.data.csv_path is not None and Path(cfg.data.csv_path).exists()
    return checks


if __name__ == "__main__":
    main(sys.argv[1:])
