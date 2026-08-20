import json

from xfedagent.runner import ExperimentRunner


def test_runner_produces_artifacts(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "rounds.csv").exists()
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["config_digest"] == summary["config_digest"]


def test_summary_reports_full_framework(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()

    # Tier 3: global-model integrity commitment is published every round.
    assert isinstance(summary["final_model_root"], str) and summary["final_model_root"]

    # Tier 2: cross-chain relay accounting is surfaced, with a Byzantine quorum
    # and a positive gas budget per relayed message.
    relay = summary["relay"]
    assert relay["enabled"] is True
    assert relay["quorum"] > 2 * relay["faulty_relayers"]
    assert relay["total_gas"] > 0
    assert relay["messages_relayed"] > 0

    # Energy: the four-component model yields a non-negative per-update mean.
    assert summary["energy_wh"]["mean"] >= 0.0


def test_rounds_csv_carries_commitments(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    header = (run_dir / "rounds.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "global_model_root" in header
    assert "input_binding_hash" in header


def test_save_round_models_flag(tmp_path, synthetic_config) -> None:
    cfg = synthetic_config
    object.__setattr__(cfg.output, "directory", str(tmp_path))
    object.__setattr__(cfg.output, "save_round_models", True)
    summary = ExperimentRunner(cfg).run()
    run_dir = tmp_path / summary["run_name"]
    saved = sorted((run_dir / "models").glob("round_*.pt"))
    assert len(saved) == cfg.federation.rounds
