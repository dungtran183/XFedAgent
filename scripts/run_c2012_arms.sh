#!/usr/bin/env bash
# The paired predicate comparison on PhysioNet/CinC Challenge 2012, set A.
#
# INTERNAL CROSS-CHECK, NOT A REPORTED RESULT. It exercises the whole path --
# cohort build, pairing check, both gates, Holm-corrected paired statistics -- on
# real ICU data. Nothing it produces goes into the manuscript, response.tex or
# results_macros.tex. Set A is open access, so no DUA or credentialing applies, but
# it is not the cohort the paper describes either: the Challenge publishes 13 of
# the 17 channels, so capillary refill and the three GCS sub-scores sit at their
# normal values throughout. Every number below is a pipeline check.
#
# The reported cohort is MIMIC-III v1.4 via scripts/run_mimic_full.sh.
#
# The reachability pre-flight refuses this cohort: at the thresholds in the arm
# configs a round-0 client cannot clear the gate, so the sweep would spend twenty
# minutes producing an empty table. Reproducing that stall on purpose is the only
# reason to run it now:
#
#   ./scripts/run_c2012_arms.sh                              refuses, with the verdict
#   SKIP_REACHABILITY=1 ./scripts/run_c2012_arms.sh          10 seeds, both arms
#   SEEDS=0-2 SKIP_REACHABILITY=1 ./scripts/run_c2012_arms.sh  a shorter pass
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-src}"

SEEDS="${SEEDS:-0-9}"
OUT="${OUT:-results/c2012-predicate}"
PACC_CFG="${PACC_CFG:-configs/c2012-pacc.json}"
PBAL_CFG="${PBAL_CFG:-configs/c2012-pbal.json}"
COHORT="data/processed/challenge2012_mortality_24h.csv"
RAW="data/raw/challenge-2012"

if [ ! -f "$COHORT" ]; then
  if [ ! -d "$RAW/set-a" ]; then
    cat >&2 <<MSG
missing Challenge 2012 set A: $RAW/set-a

  Open access, no credentialing required:
    mkdir -p $RAW && cd $RAW
    curl -fLO https://physionet.org/files/challenge-2012/1.0.0/set-a.zip
    curl -fLO https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt
    unzip -q set-a.zip
MSG
    exit 2
  fi
  echo "=== building cohort: $COHORT ==="
  python3 -m xfedagent.challenge2012 "$RAW" "$COHORT"
fi

echo "=== cohort: PhysioNet/CinC Challenge 2012 set A (open access) -- INTERNAL CHECK ==="
echo "cohort_sha256=$(shasum -a 256 "$COHORT" | cut -d' ' -f1)"
echo "=== preflight: $PBAL_CFG ==="
python3 -m xfedagent preflight --config "$PBAL_CFG"
echo "=== preflight: $PACC_CFG ==="
python3 -m xfedagent preflight --config "$PACC_CFG"

# One implementation of the paired path: the same script the surrogate and MIMIC
# comparisons use, so a difference between cohorts cannot come from the harness.
SEEDS="$SEEDS" OUT="$OUT" PACC_CFG="$PACC_CFG" PBAL_CFG="$PBAL_CFG" \
  ./scripts/run_predicate_arms.sh

cat <<NOTE

=== done ===
Tables: $OUT/ablation_runs.csv, $OUT/stats
Internal check only. Do not copy these numbers into paper.tex, appendix.tex,
response.tex, response_r2.tex, or results_macros.tex.
NOTE
