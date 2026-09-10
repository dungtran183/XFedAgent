#!/usr/bin/env python3
"""Prepare and manage private, aggregate-only Kaggle MIMIC shards."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import base64
import csv
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "kaggle" / "xfedagent-mimic"
DEFAULT_BUNDLE_ROOT = ROOT / ".kaggle" / "xfedagent-mimic"
DEFAULT_DOWNLOAD_ROOT = ROOT / "results" / "mimic-kaggle" / "shards"
ARMS = ("pacc", "pbal")
# Only the official open-access Demo (ODbL, DOI 10.13026/C2HM2Q) may be copied to
# Kaggle. The credentialed full release is covered by the PhysioNet DUA, which
# forbids sharing it with a third party, and a private Kaggle dataset is still a
# third party. The full cohort runs locally through scripts/run_mimic_arms.sh.
DEMO_DATASET_SLUG_SUFFIX = "xfedagent-mimiciii-demo-cohort"
#: A second cohort the owner has authorised onto their own private Kaggle: the
#: subset built from the public Hugging Face mirror, used to predict how the gate
#: will behave before the credentialed release is downloaded. It is admitted under
#: its own slug rather than by relaxing the check, so the credentialed cohort --
#: which arrives as ``mimic_mortality_24h.csv`` down the PhysioNet path and is
#: bound by the DUA -- still has no slug that accepts it. Results from this cohort
#: are prediction, never a submitted number.
MIRROR_DATASET_SLUG_SUFFIX = "xfedagent-hfmimic-cohort"
#: The cohort file each admitted slug must carry. The filename is the enforcement
#: point, not the slug alone: pushing the credentialed cohort under the mirror's
#: slug would still have to rename it, and the rename is what this refuses.
ALLOWED_COHORT_FILES = {
    DEMO_DATASET_SLUG_SUFFIX: "mimic_mortality_24h.csv",
    MIRROR_DATASET_SLUG_SUFFIX: "hfmimic_mortality_24h.csv",
}
EXPECTED_SEEDS = tuple(range(10))
METRICS = (
    "accuracy",
    "auc_roc",
    "f1",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_seeds(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    seeds = tuple(dict.fromkeys(values))
    if not seeds or any(seed not in EXPECTED_SEEDS for seed in seeds):
        raise ValueError("seeds must be selected from 0-9")
    return seeds


def selected_shards(arm: str, seeds: str) -> list[tuple[str, int]]:
    arms = ARMS if arm == "all" else (arm,)
    return [(label, seed) for label in arms for seed in parse_seeds(seeds)]


def kaggle_owner() -> str:
    credential = Path.home() / ".kaggle" / "kaggle.json"
    if not credential.is_file():
        raise RuntimeError("Kaggle credential file is unavailable")
    owner = str(read_json(credential).get("username", "")).strip()
    if not owner:
        raise RuntimeError("Kaggle username is absent from the credential file")
    return owner


def kaggle_binary() -> str:
    binary = shutil.which("kaggle")
    fallback = Path.home() / ".local" / "bin" / "kaggle"
    if binary:
        return binary
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("Kaggle CLI is unavailable")


def kernel_slug(owner: str, prefix: str, arm: str, seed: int) -> str:
    return f"{owner}/xfedagent-{prefix}-{arm}-s{seed:02d}"


def bundle_dir(root: Path, arm: str, seed: int) -> Path:
    return root / arm / f"seed-{seed:02d}"


def source_archive(destination: Path) -> None:
    files = sorted(
        path
        for path in (ROOT / "src" / "xfedagent").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT)
            info = zipfile.ZipInfo(str(relative), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def render_kernel_script(destination: Path, config: dict, shard: dict, source_zip: Path) -> None:
    assets = io.BytesIO()
    with zipfile.ZipFile(assets, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (
            ("config.json", json.dumps(config, indent=2, sort_keys=True).encode("utf-8") + b"\n"),
            ("shard.json", json.dumps(shard, indent=2, sort_keys=True).encode("utf-8") + b"\n"),
            ("source.zip", source_zip.read_bytes()),
        ):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    encoded = base64.b64encode(assets.getvalue()).decode("ascii")
    template = (TEMPLATE_DIR / "run.py").read_text(encoding="utf-8")
    sentinel = "__XFEDAGENT_EMBEDDED_ASSETS__"
    if template.count(sentinel) != 2:
        raise RuntimeError("Kaggle runner embedding sentinel changed unexpectedly")
    # Replace only the assigned payload; retain the comparison sentinel used to
    # make the checked-in template runnable with adjacent assets.
    generated = template.replace(
        f'EMBEDDED_ASSETS_B64 = "{sentinel}"',
        f'EMBEDDED_ASSETS_B64 = "{encoded}"',
        1,
    )
    destination.write_text(generated, encoding="utf-8")


def resolved_config(base: dict, arm: str, seed: int, dataset: str, prefix: str = "mimic") -> dict:
    config = deepcopy(base)
    config["name"] = f"{prefix}-{arm}-s{seed:02d}"
    # The filename follows from the slug rather than being assumed, so a bundle
    # cannot quietly point at a cohort the slug was not admitted for.
    suffix = dataset.split("/", 1)[1]
    config["data"]["csv_path"] = f"/kaggle/input/{suffix}/{ALLOWED_COHORT_FILES[suffix]}"
    config["federation"]["seed"] = seed
    config["pov"]["threshold_sensitivity"] = 0.65
    config["pov"]["threshold_specificity"] = 0.65
    config["pov"]["predicate"] = "raw_accuracy" if arm == "pacc" else "class_aware"
    config["pov"]["balanced_validation"] = arm == "pbal"
    config["output"]["directory"] = "/kaggle/working/output"
    config["output"]["save_round_models"] = False
    config["toolchain"] = {
        "require_snarkjs": False,
        "require_qemu": False,
        "require_ipfs": False,
        "require_hardhat": False,
    }
    return config


def validate_pair(pacc: dict, pbal: dict) -> None:
    left, right = deepcopy(pacc), deepcopy(pbal)
    left.pop("name")
    right.pop("name")
    for config in (left, right):
        config["pov"].pop("predicate")
        config["pov"].pop("balanced_validation")
    if left != right:
        raise RuntimeError("paired configs differ outside predicate and validation sampling")


def refuse_non_demo_cohort(dataset: str) -> None:
    """Allow the two cohorts the owner has authorised onto Kaggle, and nothing else.

    The open-access Demo (ODbL) and the public Hugging Face mirror subset. What is
    still refused is the credentialed MIMIC-III v1.4 release: it has no admitted
    slug, and its cohort filename is not among the admitted ones, so there is no
    argument spelling that pushes it. That one runs locally.
    """

    suffix = dataset.split("/", 1)[-1]
    if suffix not in ALLOWED_COHORT_FILES:
        raise RuntimeError(
            f"refusing Kaggle dataset {dataset!r}: only {DEMO_DATASET_SLUG_SUFFIX} "
            f"(open-access Demo, ODbL) or {MIRROR_DATASET_SLUG_SUFFIX} (public "
            "Hugging Face mirror subset) may leave this machine. The credentialed "
            "MIMIC-III v1.4 release is bound by the PhysioNet DUA and has no slug "
            "here; run it locally with scripts/run_mimic_arms.sh"
        )


def prepare(args: argparse.Namespace) -> None:
    if args.dataset.count("/") != 1:
        raise ValueError("--dataset must be OWNER/PRIVATE_DATASET")
    refuse_non_demo_cohort(args.dataset)
    owner = args.owner or kaggle_owner()
    root = Path(args.bundle_root).resolve()
    base = read_json(Path(args.base_config))
    prefix = getattr(args, "prefix", "mimic")
    seeds = parse_seeds(args.seeds)
    prepared = 0
    for seed in seeds:
        pair = {
            arm: resolved_config(base, arm, seed, args.dataset, prefix=prefix)
            for arm in ARMS
        }
        validate_pair(pair["pacc"], pair["pbal"])
        for arm in ARMS:
            destination = bundle_dir(root, arm, seed)
            destination.mkdir(parents=True, exist_ok=True)
            source_archive(destination / "source.zip")
            write_json(destination / "config.json", pair[arm])
            shard = {
                "schema_version": 1,
                "arm": arm,
                "seed": seed,
                "expected_rounds": int(pair[arm]["federation"]["rounds"]),
            }
            write_json(destination / "shard.json", shard)
            render_kernel_script(destination / "run.py", pair[arm], shard, destination / "source.zip")
            write_json(
                destination / "kernel-metadata.json",
                {
                    "id": kernel_slug(owner, prefix, arm, seed),
                    # Kaggle derives the URL slug from the title and may ignore a
                    # mismatched id. Keeping them byte-aligned prevents an update
                    # from accidentally creating a second kernel.
                    "title": kernel_slug(owner, prefix, arm, seed).split("/", 1)[1],
                    "code_file": "run.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_tpu": "false",
                    "enable_internet": "false",
                    "machine_shape": "NvidiaTeslaT4",
                    "dataset_sources": [args.dataset],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                },
            )
            prepared += 1
    print(f"prepared={prepared}")
    print(f"bundle_root={root}")
    print("privacy=private kernels, internet disabled, aggregate outputs only")


def demo(args: argparse.Namespace) -> None:
    """Upload the official open demo cohort privately and run two preview arms."""

    owner = args.owner or kaggle_owner()
    cohort = Path(args.cohort).resolve()
    if not cohort.is_file():
        raise FileNotFoundError(
            f"processed official demo cohort not found: {cohort}; run xfedagent.mimic first"
        )
    runtime_root = Path(args.bundle_root).resolve()
    dataset_slug = f"{owner}/xfedagent-mimiciii-demo-cohort"
    dataset_dir = runtime_root / "demo-dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cohort, dataset_dir / "mimic_mortality_24h.csv")
    write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "title": "XFedAgent MIMIC-III official demo cohort",
            "id": dataset_slug,
            "licenses": [{"name": "ODbL-1.0"}],
            "description": (
                "Private working copy derived from the official open-access "
                "MIMIC-III Clinical Database Demo v1.4 (PhysioNet DOI "
                "10.13026/C2HM2Q). Contains a 24-hour, 17-channel aggregate "
                "experiment cohort; not the credentialed full MIMIC-III database."
            ),
        },
    )
    if args.upload:
        print(f"creating_private_dataset={dataset_slug}")
        run_kaggle(["datasets", "create", "-p", str(dataset_dir), "-r", "zip"])

    base = read_json(ROOT / "configs" / "full.json")
    base["data"].update(
        {
            "samples": 69,
            "clients": 4,
            "validation_pool_size": 12,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "positive_rate": 24 / 69,
        }
    )
    base["model"].update({"local_epochs": 2, "batch_size": 8})
    base["pov"]["validation_size"] = 10
    base["federation"].update({"rounds": 10, "clients_per_round": 4, "seed": 0})
    base_path = runtime_root / "demo-base-config.json"
    write_json(base_path, base)
    kernel_root = runtime_root / "demo-kernels"
    prepare(
        argparse.Namespace(
            dataset=dataset_slug,
            owner=owner,
            bundle_root=str(kernel_root),
            base_config=str(base_path),
            seeds="0",
            prefix="mimic-demo",
        )
    )
    if args.submit:
        if not args.upload:
            print("note=using an already-created private demo dataset")
        for arm in ARMS:
            directory = bundle_dir(kernel_root, arm, 0)
            metadata = read_json(directory / "kernel-metadata.json")
            print(f"submitting={metadata['id']}")
            run_kaggle(
                [
                    "kernels",
                    "push",
                    "-p",
                    str(directory),
                    "--accelerator",
                    "NvidiaTeslaT4",
                ]
            )


def mirror(args: argparse.Namespace) -> None:
    """Upload the Hugging Face mirror cohort privately and run the paired arms on it.

    The point of this cohort is prediction: it stands in for the credentialed release
    to answer what the gate will do before 2.7 h is spent finding out, so the gate
    block is taken from the shipped MIMIC configs verbatim and only the data block is
    swapped. Nothing measured here is a submitted number.
    """

    owner = args.owner or kaggle_owner()
    cohort = Path(args.cohort).resolve()
    if not cohort.is_file():
        raise FileNotFoundError(f"mirror cohort not found: {cohort}")
    if cohort.name != ALLOWED_COHORT_FILES[MIRROR_DATASET_SLUG_SUFFIX]:
        raise RuntimeError(
            f"refusing {cohort.name!r}: the mirror slug carries "
            f"{ALLOWED_COHORT_FILES[MIRROR_DATASET_SLUG_SUFFIX]!r}. A cohort arriving "
            "under another name is not this cohort, and renaming it here is exactly "
            "what the check exists to prevent."
        )
    runtime_root = Path(args.bundle_root).resolve()
    dataset_slug = f"{owner}/{MIRROR_DATASET_SLUG_SUFFIX}"
    dataset_dir = runtime_root / "mirror-dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cohort, dataset_dir / cohort.name)
    write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "title": "XFedAgent MIMIC-III mirror cohort",
            "id": dataset_slug,
            "licenses": [{"name": "other"}],
            "description": (
                "Private working copy of a 24-hour, 17-channel experiment cohort "
                "derived from a public Hugging Face mirror of MIMIC-III. Used to "
                "predict gate behaviour before the credentialed release is "
                "processed; not the credentialed PhysioNet distribution, and no "
                "number measured from it is reported as a result."
            ),
        },
    )
    if args.upload:
        print(f"creating_private_dataset={dataset_slug}")
        run_kaggle(["datasets", "create", "-p", str(dataset_dir), "-r", "zip"])

    # The gate under test is the one that ships, so it is read from the shipped
    # config rather than rebuilt here; only the cohort moves.
    base = read_json(ROOT / "configs" / "mimic-pbal.json")
    mirror_data = read_json(ROOT / "configs" / "hfmimic-pbal.json")["data"]
    base["data"] = dict(mirror_data)
    base["output"]["directory"] = "/kaggle/working/output"
    base_path = runtime_root / "mirror-base-config.json"
    write_json(base_path, base)
    kernel_root = runtime_root / "mirror-kernels"
    prepare(
        argparse.Namespace(
            dataset=dataset_slug,
            owner=owner,
            bundle_root=str(kernel_root),
            base_config=str(base_path),
            seeds=args.seeds,
            prefix="mimic-mirror",
        )
    )
    if args.submit:
        if not args.upload:
            print("note=using an already-created private mirror dataset")
        for arm, seed in selected_shards("all", args.seeds):
            directory = bundle_dir(kernel_root, arm, seed)
            metadata = read_json(directory / "kernel-metadata.json")
            for dataset in metadata.get("dataset_sources", []):
                refuse_non_demo_cohort(dataset)
            print(f"submitting={metadata['id']}")
            push_kernel(directory)


def smoke(args: argparse.Namespace) -> None:
    owner = args.owner or kaggle_owner()
    destination = Path(args.bundle_root).resolve() / "smoke"
    destination.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / "configs" / "predicate-class-aware.json")
    config["name"] = "kaggle-private-smoke"
    config["data"].update(
        {
            "source": "synthetic_mimic",
            "samples": 240,
            "clients": 4,
            "validation_pool_size": 60,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "csv_path": None,
        }
    )
    config["model"]["local_epochs"] = 1
    config["federation"].update(
        {"rounds": 2, "clients_per_round": 4, "seed": 0, "attacks": ["majority_class"]}
    )
    config["output"].update(
        {"directory": "/kaggle/working/output", "save_round_models": False}
    )
    config["toolchain"] = {
        "require_snarkjs": False,
        "require_qemu": False,
        "require_ipfs": False,
        "require_hardhat": False,
    }
    shard = {
        "schema_version": 1,
        "arm": "smoke",
        "seed": 0,
        "expected_rounds": 2,
        "smoke": True,
    }
    source_archive(destination / "source.zip")
    write_json(destination / "config.json", config)
    write_json(destination / "shard.json", shard)
    render_kernel_script(destination / "run.py", config, shard, destination / "source.zip")
    metadata = {
        "id": f"{owner}/xfedagent-private-smoke",
        "title": "XFedAgent private smoke",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(destination / "kernel-metadata.json", metadata)
    print(f"smoke_bundle={destination}")
    if args.submit:
        print(f"submitting={metadata['id']}")
        push_kernel(destination)


def run_kaggle(command: list[str]) -> None:
    subprocess.run([kaggle_binary(), *command], check=True)


def push_kernel(directory: Path, accelerator: str = "NvidiaTeslaT4") -> None:
    """Push one kernel, refusing to treat a rejected push as a submission.

    ``kaggle kernels push`` prints ``Kernel push error: ...`` and still exits zero, so
    a batch that hits the account's concurrent-GPU-session cap reports every shard as
    submitted while creating none of them. Sixteen of eighteen were "pushed" that way,
    and the failure only surfaced later as 404s from ``kernels status``. The output is
    inspected because the exit code cannot be trusted here.
    """

    result = subprocess.run(
        [kaggle_binary(), "kernels", "push", "-p", str(directory), "--accelerator", accelerator],
        check=True,
        capture_output=True,
        text=True,
    )
    combined = f"{result.stdout}{result.stderr}".strip()
    print(combined)
    if "error" in combined.lower() or "successfully pushed" not in combined.lower():
        raise RuntimeError(f"kernel push refused for {directory}: {combined}")


def submit(args: argparse.Namespace) -> None:
    # The flag attests that the shards about to be pushed read the open-access
    # Demo cohort. It is not a credentialing declaration: no declaration makes the
    # credentialed release shareable, which is why the dataset each shard mounts
    # is re-checked below rather than trusted from prepare time.
    if not args.confirm_authorized_data:
        raise RuntimeError(
            "submission refused: pass --confirm-authorized-data to confirm these shards "
            "read the open-access MIMIC-III Demo cohort. The credentialed full cohort is "
            "never submitted here; run it locally with scripts/run_mimic_arms.sh"
        )
    root = Path(args.bundle_root).resolve()
    for arm, seed in selected_shards(args.arm, args.seeds):
        directory = bundle_dir(root, arm, seed)
        metadata = read_json(directory / "kernel-metadata.json")
        if metadata.get("is_private") != "true" or metadata.get("enable_internet") != "false":
            raise RuntimeError(f"unsafe Kaggle metadata in {directory}")
        for dataset in metadata.get("dataset_sources", []):
            refuse_non_demo_cohort(dataset)
        print(f"submitting={metadata['id']}")
        push_kernel(directory)


def status(args: argparse.Namespace) -> None:
    root = Path(args.bundle_root).resolve()
    for arm, seed in selected_shards(args.arm, args.seeds):
        metadata = read_json(bundle_dir(root, arm, seed) / "kernel-metadata.json")
        print(f"kernel={metadata['id']}")
        run_kaggle(["kernels", "status", metadata["id"]])


def download(args: argparse.Namespace) -> None:
    bundle_root = Path(args.bundle_root).resolve()
    download_root = Path(args.download_root).resolve()
    for arm, seed in selected_shards(args.arm, args.seeds):
        metadata = read_json(bundle_dir(bundle_root, arm, seed) / "kernel-metadata.json")
        destination = download_root / arm / f"seed-{seed:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        print(f"downloading={metadata['id']}")
        run_kaggle(["kernels", "output", metadata["id"], "-p", str(destination), "--force"])


def load_manifest(directory: Path) -> dict:
    matches = list(directory.rglob("shard_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one shard_manifest.json below {directory}, found {len(matches)}")
    manifest = read_json(matches[0])
    if manifest.get("patient_level_output") is not False:
        raise RuntimeError(f"patient-level output declaration failed in {matches[0]}")
    if manifest.get("patient_disjoint_split") is not True:
        raise RuntimeError(f"patient-disjoint split evidence absent in {matches[0]}")
    if manifest.get("observed_rounds") != manifest.get("expected_rounds"):
        raise RuntimeError(f"incomplete round count in {matches[0]}")
    metrics = shard_metrics(manifest)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite final metric in {matches[0]}")
    return manifest


def shard_metrics(manifest: dict) -> dict[str, float]:
    """The reported metric names, read out of ``BinaryMetrics``' own field names.

    ``BinaryMetrics`` calls sensitivity ``recall``, and the local ablation path renames
    it on the way into its results table so the tables use the clinical term the
    predicate is stated in. The Kaggle path reads ``summary.json`` directly and so never
    saw that rename: ``load_manifest`` asked for ``sensitivity``, the shard carried
    ``recall``, and a completed 60-round shard was rejected with a ``KeyError``. The
    rename happens here so both paths produce the same columns from the same run.
    """

    reported = manifest["final_metrics"]
    aliases = {"sensitivity": "recall"}
    return {
        name: float(reported[aliases.get(name, name)])
        for name in METRICS
    }


#: Provenance label per cohort file, in the manuscript's fixed vocabulary. Keyed by
#: filename because that is the one thing about a shard's cohort that cannot be
#: misreported: it is the file the kernel opened.
COHORT_PROVENANCE = {
    "hfmimic_mortality_24h.csv": "measured (Hugging Face mirror cohort)",
    "mimic_mortality_24h.csv": "measured (MIMIC-III cohort)",
}


def cohort_provenance(directories: list[Path]) -> str:
    """Which cohort these shards read, read from the config each shard saved.

    Taken from the resolved ``csv_path`` in each run's own ``config.json`` rather than
    passed in, so a merge cannot be told which cohort it processed. A mixed set is
    refused rather than labelled: a paired comparison spanning two cohorts is not a
    paired comparison, and a merge over mirror shards claiming the credentialed label
    would be a relabelling of provenance.
    """

    names: set[str] = set()
    for directory in directories:
        for config_path in directory.rglob("config.json"):
            data = read_json(config_path).get("data", {})
            path = str(data.get("csv_path", ""))
            if path:
                names.add(Path(path).name)
    known = {COHORT_PROVENANCE[name] for name in names if name in COHORT_PROVENANCE}
    if len(known) > 1 or (names and not known):
        raise RuntimeError(f"shards did not read one known cohort: {sorted(names)}")
    return known.pop() if known else "measured (cohort not identified by the shards)"


def merge(args: argparse.Namespace) -> None:
    download_root = Path(args.download_root).resolve()
    output = Path(args.output).resolve()
    manifests: dict[tuple[str, int], dict] = {}
    shard_directories: list[Path] = []
    for arm in ARMS:
        for seed in EXPECTED_SEEDS:
            directory = download_root / arm / f"seed-{seed:02d}"
            shard_directories.append(directory)
            manifest = load_manifest(directory)
            key = (str(manifest["arm"]), int(manifest["seed"]))
            if key != (arm, seed) or key in manifests:
                raise RuntimeError(f"mismatched or duplicate shard: expected {(arm, seed)}, got {key}")
            manifests[key] = manifest

    for seed in EXPECTED_SEEDS:
        if manifests[("pacc", seed)].get("pairing_digest") != manifests[("pbal", seed)].get(
            "pairing_digest"
        ):
            raise RuntimeError(f"arm configs are not paired for seed {seed}")

    rows = []
    for (arm, seed), manifest in sorted(manifests.items()):
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                **shard_metrics(manifest),
                "malicious_rejection_rate": manifest["malicious_rejection_rate"],
                "honest_false_reject_rate": manifest["honest_false_reject_rate"],
                "malicious_submitted": manifest["malicious_submitted"],
                "malicious_admitted": manifest["malicious_admitted"],
                "honest_submitted": manifest["honest_submitted"],
                "honest_rejected": manifest["honest_rejected"],
                "accepted_updates": manifest["accepted_updates"],
                "rejected_updates": manifest["rejected_updates"],
                "run_name": manifest["run_name"],
                "config_digest": manifest["config_digest"],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    runs_csv = output / "predicate_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sys.path.insert(0, str(ROOT / "src"))
    from xfedagent.statistics import compare_family

    stats = compare_family(
        runs_csv,
        [(name, "pbal", "pacc") for name in (*METRICS, "honest_false_reject_rate")],
        output / "stats",
        "mimic-predicate",
    )
    write_json(
        output / "merge_manifest.json",
        {
            "schema_version": 1,
            "arms": list(ARMS),
            "seeds": list(EXPECTED_SEEDS),
            "runs": len(rows),
            "runs_csv": str(runs_csv),
            "paired_manifest": str(output / "stats" / "paired_manifest.json"),
            # The label names the cohort that was actually read, not the one the
            # pipeline was written for: a merge over mirror shards claiming
            # "measured (MIMIC-III cohort)" would put a prediction under the same
            # label as the credentialed measurement. Read from the shards.
            "cohort_provenance": cohort_provenance(shard_directories),
            "patient_disjoint_required": True,
            "patient_level_output": False,
            "family": stats["family"],
        },
    )
    print(f"merged_runs={len(rows)}")
    print(f"runs_csv={runs_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", required=True)
    prepare_parser.add_argument("--owner", default=None)
    prepare_parser.add_argument("--base-config", default=str(ROOT / "configs" / "full.json"))
    prepare_parser.add_argument("--seeds", default="0-9")
    prepare_parser.set_defaults(func=prepare)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--owner", default=None)
    smoke_parser.add_argument("--submit", action="store_true")
    smoke_parser.set_defaults(func=smoke)

    mirror_parser = subparsers.add_parser("mirror")
    mirror_parser.add_argument(
        "--cohort",
        default=str(ROOT / "data" / "processed" / "hfmimic_mortality_24h.csv"),
    )
    mirror_parser.add_argument("--owner", default=None)
    mirror_parser.add_argument("--seeds", default="0-9")
    mirror_parser.add_argument("--upload", action="store_true")
    mirror_parser.add_argument("--submit", action="store_true")
    mirror_parser.set_defaults(func=mirror)

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument(
        "--cohort", default=str(ROOT / "data" / "processed" / "mimic_mortality_24h_demo.csv")
    )
    demo_parser.add_argument("--owner", default=None)
    demo_parser.add_argument("--upload", action="store_true")
    demo_parser.add_argument("--submit", action="store_true")
    demo_parser.set_defaults(func=demo)

    for name, function in (("submit", submit), ("status", status), ("download", download)):
        command = subparsers.add_parser(name)
        command.add_argument("--arm", choices=("all", *ARMS), default="all")
        command.add_argument("--seeds", default="0-9")
        if name == "submit":
            command.add_argument("--confirm-authorized-data", action="store_true")
        if name == "download":
            command.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
        command.set_defaults(func=function)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    merge_parser.add_argument("--output", default=str(ROOT / "results" / "mimic-kaggle" / "merged"))
    merge_parser.set_defaults(func=merge)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
