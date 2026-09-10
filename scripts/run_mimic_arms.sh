#!/usr/bin/env bash
# The paired predicate comparison over MIMIC-III, executed entirely on this
# machine.
#
# The credentialed data never leaves the local disk: PhysioNet restricts access to
# individually credentialed people and forbids sharing, which rules out any
# third-party or hosted service. Kaggle is used for one thing only, the official
# open-access Demo (ODbL), which is not credentialed data.
#
# Nothing here copies, uploads or prints a patient record. The only artefacts are
# aggregate per-round and per-run tables plus the paired statistics.
#
#   DEMO=1 ./scripts/run_mimic_arms.sh     rehearse on the official open Demo
#   ./scripts/run_mimic_arms.sh            the real run, once the DUA is signed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-src}"

DEMO="${DEMO:-0}"
SEEDS="${SEEDS:-0-9}"

if [ "$DEMO" = "1" ]; then
  # The rehearsal proves the pipeline end to end on open-access data. It is not
  # evidence: 69 ICU stays cannot support a claim, and the numbers it produces
  # are never reported as the cohort.
  OUT="${OUT:-results/mimic-demo-local}"
  RUNTIME="${RUNTIME:-.kaggle/local-demo-configs}"
  SEEDS="${SEEDS_DEMO:-0}"
  mkdir -p "$RUNTIME"
  python3 - "$RUNTIME" <<'PY'
import json, sys
from copy import deepcopy
from pathlib import Path

runtime = Path(sys.argv[1])
scaled = {
    "data": {
        "samples": 69,
        "clients": 4,
        "validation_pool_size": 12,
        "validation_fraction": 0.20,
        "test_fraction": 0.20,
        "positive_rate": 24 / 69,
        "csv_path": "data/processed/mimic_mortality_24h_demo.csv",
    },
    "model": {"local_epochs": 2, "batch_size": 8},
    "pov": {"validation_size": 10},
    "federation": {"rounds": 10, "clients_per_round": 4},
}
for arm in ("pacc", "pbal"):
    config = json.loads(Path(f"configs/mimic-{arm}.json").read_text(encoding="utf-8"))
    for section, patch in scaled.items():
        config[section].update(deepcopy(patch))
    config["name"] = f"mimic-demo-{arm}"
    (runtime / f"mimic-demo-{arm}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {runtime / f'mimic-demo-{arm}.json'}")
PY
  PACC_CFG="$RUNTIME/mimic-demo-pacc.json"
  PBAL_CFG="$RUNTIME/mimic-demo-pbal.json"
  COHORT="data/processed/mimic_mortality_24h_demo.csv"
  LABEL="official MIMIC-III Demo v1.4 (ODbL, open access) -- pipeline rehearsal"
else
  OUT="${OUT:-results/mimic-predicate}"
  PACC_CFG="${PACC_CFG:-configs/mimic-pacc.json}"
  PBAL_CFG="${PBAL_CFG:-configs/mimic-pbal.json}"
  COHORT="data/processed/mimic_mortality_24h.csv"
  LABEL="MIMIC-III Clinical Database v1.4 (PhysioNet credentialed, signed DUA)"
fi

if [ ! -f "$COHORT" ]; then
  cat >&2 <<MSG
missing cohort: $COHORT

  Sign the Data Use Agreement on the MIMIC-III v1.4 project page, then:
    ./scripts/fetch_mimic.sh
    PYTHONPATH=src python3 -m xfedagent.mimic \\
        data/raw/mimic-iii-clinical-database-1.4 $COHORT

  To rehearse the pipeline on the open-access Demo instead:
    DEMO=1 ./scripts/run_mimic_arms.sh
MSG
  exit 2
fi

mkdir -p "$OUT"

echo "=== cohort: $LABEL ==="
echo "=== preflight: $PBAL_CFG ==="
python3 -m xfedagent preflight --config "$PBAL_CFG"
echo "=== preflight: $PACC_CFG ==="
python3 -m xfedagent preflight --config "$PACC_CFG"

# The arm runner re-checks the pairing, merges the two per-arm tables and runs
# the paired statistics. Reusing it keeps one implementation of that path.
SEEDS="$SEEDS" OUT="$OUT" PACC_CFG="$PACC_CFG" PBAL_CFG="$PBAL_CFG" \
  ./scripts/run_predicate_arms.sh

python3 - "$OUT" "$COHORT" "$LABEL" "$PACC_CFG" "$PBAL_CFG" "$SEEDS" <<'PY'
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out, cohort, label, pacc, pbal, seeds = (Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# The cohort digest identifies the input without disclosing it, which is what a
# reproduction needs: another credentialed reader rebuilds the cohort and
# compares the hash. The file itself stays on this disk.
manifest = {
    "cohort_label": label,
    "cohort_path": str(cohort),
    "cohort_sha256": digest(cohort),
    "cohort_bytes": cohort.stat().st_size,
    "arms": {"pacc": pacc, "pbal": pbal},
    "seeds": seeds,
    "provenance": "measured (local machine)",
    "data_left_this_machine": False,
    "uploaded_anywhere": False,
    "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": commit(),
    "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    "python": platform.python_version(),
}
path = out / "cohort_manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"cohort_manifest={path}")
print(f"cohort_sha256={manifest['cohort_sha256']}")
PY

cat <<NOTE

=== done ===
Aggregate tables only: $OUT/ablation_runs.csv, $OUT/stats, $OUT/cohort_manifest.json
The cohort CSV stays in data/, which .gitignore excludes. Do not commit it, copy
it into the handover package, upload it to Kaggle or any other service, or paste
it into a model prompt.
NOTE
