#!/usr/bin/env bash
# Paired predicate comparison: P_acc against P_bal, same seeds.
#
# The two arms differ only in the utility predicate the gate enforces, so the
# comparison is paired on the seed: every seed is run under both predicates and
# the difference is taken within the seed. Each arm is executed through the same
# ablation path as the component sweep (arm "full", nothing disabled), then the
# two per-run tables are merged into one with the arm column relabelled, which is
# the shape `xfedagent stats` reads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-0-9}"
OUT="${OUT:-results/surrogate-predicate}"
PACC_CFG="${PACC_CFG:-configs/predicate-raw-accuracy.json}"
PBAL_CFG="${PBAL_CFG:-configs/predicate-class-aware.json}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "$OUT"

# Refuse to run two arms that are not a paired comparison, before any compute is
# spent.
echo "=== pairing check ==="
python3 -m xfedagent validate-config --config "$PACC_CFG" --pair-with "$PBAL_CFG"

# Refuse to spend the compute when the class-aware arm cannot clear its own
# thresholds on this cohort: two models, ~40 s, validation pool only. A verdict
# other than "reachable" exits non-zero and stops the sweep rather than producing
# an empty table 2.7 h later. Set SKIP_REACHABILITY=1 to run anyway.
if [ "${SKIP_REACHABILITY:-0}" = "1" ]; then
  echo "=== reachability pre-flight: skipped by SKIP_REACHABILITY=1 ==="
else
  echo "=== reachability pre-flight on the class-aware arm: $PBAL_CFG ==="
  python3 -m xfedagent reachability --config "$PBAL_CFG" --output "$OUT/reachability"
fi

echo "=== arm pacc: $PACC_CFG (seeds $SEEDS) ==="
python3 -m xfedagent ablation --config "$PACC_CFG" --arms full --seeds "$SEEDS" \
  --output "$OUT/pacc"

echo "=== arm pbal: $PBAL_CFG (seeds $SEEDS) ==="
python3 -m xfedagent ablation --config "$PBAL_CFG" --arms full --seeds "$SEEDS" \
  --output "$OUT/pbal"

echo "=== merging into $OUT/ablation_runs.csv ==="
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path

out = Path(sys.argv[1])
rows, header = [], None
# The arm label in each per-arm table is "full", because each was run as the
# unablated framework. The predicate is what differs, so that is what the merged
# arm column has to name.
for label, source in (("pacc", out / "pacc"), ("pbal", out / "pbal")):
    with (source / "ablation_runs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        for row in reader:
            row["arm"] = label
            rows.append(row)

with (out / "ablation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} runs -> {out / 'ablation_runs.csv'}")
PY

echo "=== paired statistics over the predicate family ==="
python3 -m xfedagent stats --runs "$OUT/ablation_runs.csv" \
  --compare accuracy:pbal:pacc \
  --compare auc_roc:pbal:pacc \
  --compare sensitivity:pbal:pacc \
  --compare specificity:pbal:pacc \
  --compare honest_false_reject_rate:pbal:pacc \
  --family-label predicate --output "$OUT/stats" --latex
