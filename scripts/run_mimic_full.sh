#!/usr/bin/env bash
# Everything from a signed DUA to the paired statistics, in one unattended run.
#
#   1. download the five tables PhysioNet serves (resumable, checksummed)
#   2. build the 24-hour / 17-channel cohort straight from the .csv.gz
#   3. preflight both arms, then run 2 arms x 10 seeds x 60 rounds locally
#   4. merge and run the paired statistics
#
# Requires a physionet.org entry in ~/.netrc first, so nothing here prompts:
#   ./scripts/physionet_login.sh
# Steps already finished are skipped, so this is safe to re-run after a stop.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW="${RAW:-data/raw/mimic-iii-clinical-database-1.4}"
COHORT="${COHORT:-data/processed/mimic_mortality_24h.csv}"
export PYTHONPATH="${PYTHONPATH:-src}"

JAR="${PHYSIONET_COOKIE_JAR:-$HOME/.physionet-session}"
if [ ! -s "$JAR" ]; then
  if [ -f "$HOME/.netrc" ] && grep -q '^[[:space:]]*machine[[:space:]]\+physionet\.org' "$HOME/.netrc"; then
    # The credential is stored; the session just needs opening, and that needs no
    # further input, so do it here rather than stopping to ask.
    echo "opening a PhysioNet session"
    ./scripts/physionet_session.sh
  else
    cat >&2 <<'MSG'
No physionet.org entry in ~/.netrc, so the download would stop at a password
prompt. Store the credential once, open a session, then re-run this script:

    ./scripts/physionet_login.sh     # asks for username and password, mode 600
    ./scripts/run_mimic_full.sh      # opens the session itself from here on

MSG
    exit 2
  fi
fi

echo "########## 1/4 download ##########"
missing=0
for table in PATIENTS ADMISSIONS ICUSTAYS CHARTEVENTS LABEVENTS; do
  [ -s "$RAW/$table.csv.gz" ] || missing=1
done
if [ "$missing" = 0 ]; then
  echo "all five tables already present in $RAW; skipping"
else
  DEST="$RAW" ./scripts/fetch_mimic.sh
fi

echo "########## 2/4 cohort ##########"
if [ -s "$COHORT" ]; then
  echo "cohort already built: $COHORT"
else
  python3 -m xfedagent.mimic "$RAW" "$COHORT"
fi
# The cohort is the only artefact the rest of the pipeline reads, so its hash is
# recorded here: it is what makes a rebuild on another machine checkable without
# either machine sending a single record anywhere.
shasum -a 256 "$COHORT"

echo "########## 3/4 + 4/4 twenty shards, then paired statistics ##########"
SEEDS="${SEEDS:-0-9}" ./scripts/run_mimic_arms.sh
