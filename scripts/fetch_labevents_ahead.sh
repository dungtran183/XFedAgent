#!/usr/bin/env bash
# Fetch LABEVENTS.csv.gz out of alphabetical order, into the tree wget_mimic.sh
# mirrors into.
#
# The cohort builder needs five of the release's 26 tables and wget -r walks the
# index alphabetically, so LABEVENTS, the last of the five, lands only after about
# 640 MB the cohort does not read. A second connection costs little because
# PhysioNet's per-connection rate is the binding limit (98 KB/s on one connection
# against 200 KB/s aggregate on four), and the cohort build then starts as soon as
# CHARTEVENTS finishes.
#
# Safe to run alongside the mirror: when the mirror reaches a complete LABEVENTS it
# gets a 416 and reports it already retrieved, and the two never write the same
# file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/physionet.org/files/mimiciii/1.4"
JAR="${PHYSIONET_COOKIE_JAR:-$HOME/.physionet-session}"
NAME=LABEVENTS.csv.gz
EXPECT=335843735

mkdir -p "$DEST"
if [ -s "$JAR" ]; then
  AUTH=(-b "$JAR" -c "$JAR")
else
  # Basic works only with a wget User-Agent on this deployment; see README.
  AUTH=(--netrc -A 'Wget/1.25.0')
fi

code=0
curl "${AUTH[@]}" -f -L -C - --retry 20 --retry-delay 15 --retry-connrefused \
     -o "$DEST/$NAME" "https://physionet.org/files/mimiciii/1.4/$NAME" || code=$?
if [ "$code" -ne 0 ] && [ "$code" -ne 33 ] && [ "$code" -ne 36 ]; then
  echo "curl exit $code on $NAME" >&2
  exit "$code"
fi

got=$(stat -f%z "$DEST/$NAME")
if [ "$got" -ne "$EXPECT" ]; then
  echo "$NAME is $got B, expected $EXPECT B -- incomplete" >&2
  exit 1
fi
echo "$NAME complete: $got B"
