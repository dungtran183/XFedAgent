#!/usr/bin/env bash
# Copy-detector on/off against the replay free-rider.
#
# Not every replayed global model clears PoV without the copy detector: the model
# replayed in round 0 is the untrained initialisation, which the utility predicate
# rejects on its own merits. Measuring that requires the two arms to be paired
# within seed, since the claim is a difference between them.
#
# Only the copy detector is toggled. Partition, initialisation, model, attack
# schedule, predicate and thresholds come from the config, so the difference
# between the arms is the gadget alone.
#
# The sweep also produces the evidence that the two arms' committed global-model
# roots agree bit for bit for the first several rounds: a round-0 refusal costs
# beta = r_0 and zeroes the free-rider, so reputation alone reproduces the fully
# protected system until the attacker has re-banked credit. That is read from the
# per-run rounds.csv files by the macro generator, so the runs must be kept.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/predicate-class-aware-replay.json}"
SEEDS="${SEEDS:-0-9}"
OUT="${OUT:-results/ablation-freerider}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "=== pre-flight ($CONFIG) ==="
python3 -m xfedagent validate-config --config "$CONFIG"

echo "=== full vs no-copy, seeds $SEEDS ==="
python3 -m xfedagent ablation --config "$CONFIG" \
  --arms full,no-copy --seeds "$SEEDS" --output "$OUT"

# Malicious rejection is deterministic in this design and needs no test; accuracy
# and honest false rejection do, and the honest-rejection null is the half of the
# claim a reader is entitled to see an interval for.
echo "=== paired tests within seed ==="
python3 -m xfedagent stats \
  --runs "$OUT/ablation_runs.csv" \
  --compare malicious_rejection_rate:full:no-copy \
  --compare accuracy:full:no-copy \
  --compare balanced_accuracy:full:no-copy \
  --compare specificity:full:no-copy \
  --compare honest_false_reject_rate:full:no-copy \
  --family-label freerider-copy-detector \
  --output "$OUT/stats"

echo "=== done: $OUT/ablation_manifest.json and $OUT/stats/paired_manifest.json ==="
