from __future__ import annotations

from pathlib import Path
import argparse
import ast
import importlib.util
import json
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kaggle_mimic.py"
DEMO_SLUG = "xfedagent-mimiciii-demo-cohort"
MIRROR_SLUG = "xfedagent-hfmimic-cohort"
#: What the refusal says. Asserted as a substring rather than by exact wording so the
#: message can be reworded, but it must keep naming the release it refuses -- a guard
#: whose message stops saying which cohort is barred is a guard nobody can audit.
REFUSAL = "credentialed MIMIC-III v1.4 release"


def _module():
    spec = importlib.util.spec_from_file_location("kaggle_mimic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _embedded_payload(script: Path) -> str:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EMBEDDED_ASSETS_B64"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("embedded payload assignment not found")


def test_prepare_builds_private_self_contained_paired_shards(tmp_path) -> None:
    module = _module()
    args = argparse.Namespace(
        dataset=f"researcher/{DEMO_SLUG}",
        owner="researcher",
        bundle_root=str(tmp_path),
        base_config=str(ROOT / "configs" / "full.json"),
        seeds="0",
    )
    module.prepare(args)

    configs = {}
    for arm in ("pacc", "pbal"):
        directory = tmp_path / arm / "seed-00"
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        assert metadata["is_private"] == "true"
        assert metadata["enable_internet"] == "false"
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["dataset_sources"] == [f"researcher/{DEMO_SLUG}"]
        configs[arm] = json.loads((directory / "config.json").read_text())

        payload = _embedded_payload(directory / "run.py")
        assert payload != "__XFEDAGENT_EMBEDDED_ASSETS__"
        assert "kaggle.json" not in (directory / "run.py").read_text(encoding="utf-8")
        compile((directory / "run.py").read_text(encoding="utf-8"), "run.py", "exec")
        with zipfile.ZipFile(directory / "source.zip") as archive:
            assert "src/xfedagent/runner.py" in archive.namelist()

    module.validate_pair(configs["pacc"], configs["pbal"])
    assert configs["pacc"]["pov"]["predicate"] == "raw_accuracy"
    assert configs["pbal"]["pov"]["predicate"] == "class_aware"
    assert configs["pacc"]["pov"]["balanced_validation"] is False
    assert configs["pbal"]["pov"]["balanced_validation"] is True


def test_submit_requires_explicit_authorized_data_confirmation(tmp_path) -> None:
    module = _module()
    args = argparse.Namespace(
        confirm_authorized_data=False,
        bundle_root=str(tmp_path),
        arm="pbal",
        seeds="0",
    )

    with pytest.raises(RuntimeError, match="submission refused"):
        module.submit(args)


def test_load_manifest_rejects_missing_patient_split_evidence(tmp_path) -> None:
    module = _module()
    manifest = {
        "patient_level_output": False,
        "observed_rounds": 100,
        "expected_rounds": 100,
        "final_metrics": {name: 0.5 for name in module.METRICS},
    }
    (tmp_path / "shard_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="patient-disjoint"):
        module.load_manifest(tmp_path)


def test_only_authorised_cohorts_may_be_put_on_kaggle(tmp_path) -> None:
    """The DUA forbids sharing the credentialed release with a third party.

    Private Kaggle storage is still a third party, and credentialing does not change
    that, so there is deliberately no argument spelling that admits the full cohort.
    Two cohorts are admitted, each under its own slug: the open-access Demo (ODbL) and
    the public Hugging Face mirror subset. The credentialed v1.4 release is admitted by
    neither, and -- because each slug also fixes the cohort filename it carries --
    renaming it into an admitted slug does not get it through either.
    """
    module = _module()
    module.refuse_non_demo_cohort(f"researcher/{DEMO_SLUG}")
    module.refuse_non_demo_cohort(f"researcher/{MIRROR_SLUG}")

    for dataset in (
        "researcher/private-mimic",
        "researcher/mimic-iii-clinical-database",
        f"researcher/{DEMO_SLUG}-full",
        f"researcher/{MIRROR_SLUG}-full",
        # The name the credentialed cohort actually arrives under.
        "researcher/xfedagent-mimic-cohort",
    ):
        with pytest.raises(RuntimeError, match=REFUSAL):
            module.refuse_non_demo_cohort(dataset)

    # Each admitted slug carries exactly one cohort filename, so a bundle cannot
    # point an admitted slug at a different cohort.
    assert set(module.ALLOWED_COHORT_FILES) == {DEMO_SLUG, MIRROR_SLUG}
    assert module.ALLOWED_COHORT_FILES[DEMO_SLUG] == "mimic_mortality_24h.csv"
    assert module.ALLOWED_COHORT_FILES[MIRROR_SLUG] == "hfmimic_mortality_24h.csv"

    args = argparse.Namespace(
        dataset="researcher/mimic-iii-full-cohort",
        owner="researcher",
        bundle_root=str(tmp_path),
        base_config=str(ROOT / "configs" / "full.json"),
        seeds="0",
    )
    with pytest.raises(RuntimeError, match=REFUSAL):
        module.prepare(args)
    assert not list(tmp_path.iterdir())


def test_submit_rechecks_the_dataset_each_shard_mounts(tmp_path) -> None:
    """A bundle prepared before the guard existed must still be refused."""
    module = _module()
    directory = tmp_path / "pbal" / "seed-00"
    directory.mkdir(parents=True)
    (directory / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "researcher/stale-bundle",
                "is_private": "true",
                "enable_internet": "false",
                "dataset_sources": ["researcher/private-mimic"],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        confirm_authorized_data=True, bundle_root=str(tmp_path), arm="pbal", seeds="0"
    )

    with pytest.raises(RuntimeError, match=REFUSAL):
        module.submit(args)


def test_the_mirror_bundle_carries_the_shipped_gate_and_the_mirror_cohort(tmp_path) -> None:
    """The prediction is only worth anything if it predicts the run that is blocked.

    So the mirror bundle must take its gate from ``configs/mimic-pbal.json`` verbatim --
    including ``decision_threshold`` being absent, which means 0.5, the value that
    actually ships -- and change only the cohort. A bundle that quietly carried the
    0.167 threshold added later, or the rate rule, would predict a different run.
    """
    module = _module()
    shipped = json.loads((ROOT / "configs" / "mimic-pbal.json").read_text(encoding="utf-8"))
    mirror_data = json.loads(
        (ROOT / "configs" / "hfmimic-pbal.json").read_text(encoding="utf-8")
    )["data"]

    cohort = tmp_path / "hfmimic_mortality_24h.csv"
    cohort.write_text("subject_id,mortality_48h\n1,0\n", encoding="utf-8")
    module.mirror(
        argparse.Namespace(
            cohort=str(cohort),
            owner="researcher",
            bundle_root=str(tmp_path / "bundles"),
            seeds="0",
            upload=False,
            submit=False,
        )
    )
    kernels = tmp_path / "bundles" / "mirror-kernels"
    config = json.loads((kernels / "pbal" / "seed-00" / "config.json").read_text(encoding="utf-8"))

    # The gate is the shipped one, field for field.
    for field in ("validation_size", "tolerance", "threshold", "rotation", "backend"):
        assert config["pov"][field] == shipped["pov"][field], field
    # Absent in the shipped config, so absent here: the default 0.5 is what ships.
    assert "decision_threshold" not in config["pov"]
    assert "decision_rule" not in config["pov"]
    assert config["federation"]["rounds"] == shipped["federation"]["rounds"]
    assert config["model"]["local_epochs"] == shipped["model"]["local_epochs"]

    # And the cohort is the mirror's, mounted from the slug that admits it.
    assert config["data"]["samples"] == mirror_data["samples"]
    assert config["data"]["positive_rate"] == mirror_data["positive_rate"]
    assert config["data"]["csv_path"] == (
        f"/kaggle/input/{MIRROR_SLUG}/hfmimic_mortality_24h.csv"
    )

    metadata = json.loads(
        (kernels / "pbal" / "seed-00" / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["is_private"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["dataset_sources"] == [f"researcher/{MIRROR_SLUG}"]


def test_the_mirror_path_refuses_a_cohort_arriving_under_another_name(tmp_path) -> None:
    """Renaming is the only way a barred cohort could reach an admitted slug.

    The slug check alone would pass a file called ``mimic_mortality_24h.csv`` pushed
    under the mirror's slug. The filename check is what closes that, so it gets its own
    test: the credentialed cohort's own filename is refused on the mirror path.
    """
    module = _module()
    for name in ("mimic_mortality_24h.csv", "cohort.csv", "hfmimic.csv"):
        cohort = tmp_path / name
        cohort.write_text("subject_id,mortality_48h\n1,0\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="is not this cohort"):
            module.mirror(
                argparse.Namespace(
                    cohort=str(cohort),
                    owner="researcher",
                    bundle_root=str(tmp_path / "bundles"),
                    seeds="0",
                    upload=False,
                    submit=False,
                )
            )
    assert not (tmp_path / "bundles").exists()


def test_the_kernel_resolves_the_cohort_wherever_kaggle_mounts_it(tmp_path, monkeypatch) -> None:
    """The failure that cost the first two shards, pinned so it cannot recur.

    The host builds the cohort path from the dataset slug, but Kaggle mounted this
    dataset at ``/kaggle/input/datasets/<owner>/<slug>/`` rather than
    ``/kaggle/input/<slug>/``, and the shard died on a path it had computed instead of
    observed. The runner now searches the mount root by filename -- while keeping the
    two properties that make it safe: below the mount root only, and exactly the
    configured basename, with an ambiguous match refused rather than guessed.
    """
    import importlib.util

    mount = tmp_path / "input"
    actual = mount / "datasets" / "owner" / MIRROR_SLUG
    actual.mkdir(parents=True)
    (actual / "hfmimic_mortality_24h.csv").write_text("subject_id\n1\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_INPUT_DIR", str(mount))
    monkeypatch.setenv("KAGGLE_WORKING_DIR", str(tmp_path / "working"))

    spec = importlib.util.spec_from_file_location(
        "kaggle_runner", ROOT / "kaggle" / "xfedagent-mimic" / "run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    configured = f"{mount}/{MIRROR_SLUG}/hfmimic_mortality_24h.csv"
    assert runner._resolve_cohort(configured) == actual / "hfmimic_mortality_24h.csv"
    # Already-correct paths are returned untouched, so the Demo path is unaffected.
    assert runner._resolve_cohort(str(actual / "hfmimic_mortality_24h.csv")) == (
        actual / "hfmimic_mortality_24h.csv"
    )
    # A path outside the mount root is still refused before any search happens.
    with pytest.raises(RuntimeError, match="must be mounted below"):
        runner._resolve_cohort(str(tmp_path / "elsewhere" / "hfmimic_mortality_24h.csv"))
    # A missing cohort still fails, and says what was mounted instead of guessing.
    with pytest.raises(FileNotFoundError, match="Mounted files:"):
        runner._resolve_cohort(f"{mount}/{MIRROR_SLUG}/absent.csv")
    # Two files of the same name are ambiguous, so the shard refuses to pick one.
    second = mount / "datasets" / "owner" / "another"
    second.mkdir(parents=True)
    (second / "hfmimic_mortality_24h.csv").write_text("subject_id\n2\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="2 file"):
        runner._resolve_cohort(configured)


def test_a_completed_shard_is_not_rejected_over_a_metric_name(tmp_path) -> None:
    """``BinaryMetrics`` calls sensitivity ``recall``, and only one path renamed it.

    The local ablation path renames ``recall`` to ``sensitivity`` on the way into its
    results table; the Kaggle path reads ``summary.json`` straight through and so never
    saw the rename. ``load_manifest`` asked for ``sensitivity``, the shard carried
    ``recall``, and a finished 60-round shard was thrown away with a ``KeyError``.
    Pinned with the field names ``BinaryMetrics`` actually emits.
    """
    module = _module()
    from xfedagent.metrics import BinaryMetrics

    emitted = BinaryMetrics(
        accuracy=0.1458, auc_roc=0.6407, precision=0.1458, recall=1.0, f1=0.2545,
        samples=4102, specificity=0.0, balanced_accuracy=0.5,
    ).to_dict()
    # The shard reports exactly these names, so the mapping has to start from them.
    assert "sensitivity" not in emitted
    assert "recall" in emitted

    directory = tmp_path / "pbal" / "seed-00" / "output"
    directory.mkdir(parents=True)
    (directory / "shard_manifest.json").write_text(
        json.dumps(
            {
                "arm": "pbal",
                "seed": 0,
                "expected_rounds": 60,
                "observed_rounds": 60,
                "patient_level_output": False,
                "patient_disjoint_split": True,
                "final_metrics": emitted,
            }
        ),
        encoding="utf-8",
    )
    manifest = module.load_manifest(tmp_path / "pbal" / "seed-00")
    metrics = module.shard_metrics(manifest)
    assert set(metrics) == set(module.METRICS)
    assert metrics["sensitivity"] == emitted["recall"]
    assert metrics["accuracy"] == emitted["accuracy"]


def test_a_refused_kernel_push_is_not_reported_as_submitted(tmp_path, monkeypatch) -> None:
    """``kaggle kernels push`` prints its error and exits zero, so exit status lies.

    A batch that hit the account's concurrent-GPU-session cap printed
    ``Kernel push error: Maximum batch GPU session count of 2 reached`` for sixteen of
    eighteen shards while the script reported all eighteen as submitted. The failure
    only surfaced later, as 404s from ``kernels status`` -- by which point the
    submission was believed done. So the output is inspected, not the exit code.
    """
    module = _module()
    directory = tmp_path / "pbal" / "seed-01"
    directory.mkdir(parents=True)

    class _Result:
        def __init__(self, stdout: str) -> None:
            self.stdout, self.stderr = stdout, ""

    monkeypatch.setattr(module, "kaggle_binary", lambda: "/usr/bin/true")

    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *a, **k: _Result("Kernel push error: Maximum batch GPU session count of 2 reached."),
    )
    with pytest.raises(RuntimeError, match="push refused"):
        module.push_kernel(directory)

    # Silence is refused too: a push that says nothing did not confirm anything.
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: _Result(""))
    with pytest.raises(RuntimeError, match="push refused"):
        module.push_kernel(directory)

    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *a, **k: _Result("Kernel version 1 successfully pushed.  Please check progress at ..."),
    )
    module.push_kernel(directory)


def test_the_merge_names_the_cohort_the_shards_actually_read(tmp_path) -> None:
    """A prediction must not be labelled as the credentialed measurement.

    A hard-coded ``measured (MIMIC-III cohort)`` would put a run over the Hugging
    Face mirror under the same provenance label as the credentialed measurement. The
    label is read from the ``csv_path`` in each run's own saved config, so a merge
    cannot be told which cohort it processed, and a set spanning two cohorts is
    refused rather than labelled.
    """
    module = _module()

    def shard(directory, cohort_file):
        run = directory / "output" / "run-x"
        run.mkdir(parents=True)
        (run / "config.json").write_text(
            json.dumps({"data": {"csv_path": f"/kaggle/input/x/{cohort_file}"}}), encoding="utf-8"
        )
        return directory

    mirror = shard(tmp_path / "a", "hfmimic_mortality_24h.csv")
    credentialed = shard(tmp_path / "b", "mimic_mortality_24h.csv")

    assert module.cohort_provenance([mirror]) == "measured (Hugging Face mirror cohort)"
    assert module.cohort_provenance([credentialed]) == "measured (MIMIC-III cohort)"
    # The two labels are distinct, which is the whole point of reading them.
    assert len(set(module.COHORT_PROVENANCE.values())) == 2
    # Mixed cohorts are not a paired comparison, so they are refused.
    with pytest.raises(RuntimeError, match="did not read one known cohort"):
        module.cohort_provenance([mirror, credentialed])
    # An unrecognised cohort is refused too, rather than silently labelled.
    unknown = shard(tmp_path / "c", "something_else.csv")
    with pytest.raises(RuntimeError, match="did not read one known cohort"):
        module.cohort_provenance([unknown])
