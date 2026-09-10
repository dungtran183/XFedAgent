#!/usr/bin/env bash
# Agent-count sweep: measures the accuracy column of the scalability table.
#
# Accuracy at 25, 50 and 100 agents is measured here rather than extrapolated from
# the n=10 run. Extrapolation is defensible for gas, where verification is one call
# per agent, and not for accuracy.
#
# "Scalability" has two readings and they answer different questions, so the two
# config families are kept apart rather than averaged:
#
#   scaled  keeps the per-agent shard fixed at 1,200 samples, so adding agents adds
#           data. This is the federation growing, and it is what the table claims.
#   fixed   keeps the cohort at 12,000 samples, so adding agents divides the same
#           data further. This measures shard starvation, which is a property of
#           the partition rather than of the protocol.
#
# Only the client count and the sample budget move. The predicate, thresholds,
# rounds, model and attack schedule are those of configs/predicate-class-aware.json,
# so a row of this sweep is comparable with the P_bal results elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-0-4}"
FAMILY="${FAMILY:-scaled}"
SIZES="${SIZES:-10 25 50 100}"
OUT="${OUT:-results/scale-$FAMILY}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "$OUT"

# The pre-flight costs ~40 s per config against hours for a stalled sweep, and it
# has called every stall we have seen. A class-aware gate on a starved shard is
# exactly the configuration that deadlocks at round 0, so it is checked first.
for n in $SIZES; do
  cfg="configs/scale-n$n-$FAMILY.json"
  echo "=== pre-flight n=$n ($cfg) ==="
  python3 -m xfedagent validate-config --config "$cfg"
  python3 -m xfedagent reachability --config "$cfg" --output "$OUT/reachability-n$n"
done

for n in $SIZES; do
  cfg="configs/scale-n$n-$FAMILY.json"
  echo "=== n=$n seeds $SEEDS ($cfg) ==="
  python3 -m xfedagent ablation --config "$cfg" --arms full --seeds "$SEEDS" \
    --output "$OUT/n$n"
done

echo "=== ordered-alternative trend test across agent counts ==="
# The table alone cannot say whether a column trends; four means with five seeds
# each look like whatever the reader expects. This is the step whose absence let
# the previous version assert a rising accuracy trend that the data do not show.
python3 scripts/scale_trend.py \
  $(for n in $SIZES; do printf ' --runs %s:%s/n%s/ablation_runs.csv' "$n" "$OUT" "$n"; done) \
  --output "$OUT/trend_manifest.json"

echo "=== merging into $OUT/ablation_runs.csv ==="
python3 - "$OUT" "$SIZES" <<'PY'
import csv, sys
from pathlib import Path

out, sizes = Path(sys.argv[1]), sys.argv[2].split()
rows, header = [], None
for n in sizes:
    path = out / f"n{n}" / "ablation_runs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        for row in reader:
            row["arm"] = f"n{n}"       # the arm column names the agent count
            rows.append(row)
with (out / "ablation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} runs -> {out / 'ablation_runs.csv'}")
PY
