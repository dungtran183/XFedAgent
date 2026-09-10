#!/usr/bin/env bash
# Fetch MIMIC-III v1.4 from PhysioNet under the signed DUA.
#
# The data are credentialed-access, so this does three things and no more: takes
# the PhysioNet username interactively, downloads the gzipped CSV distribution
# into a directory .gitignore excludes, and verifies the checksums PhysioNet
# publishes alongside the files. Nothing is uploaded, mirrored or reshared.
#
# The password is never placed on a command line, in the environment, or in a log.
# It reaches curl either from the terminal for the life of this process or from
# ~/.netrc (mode 600), written by scripts/physionet_login.sh.
#
# Only the five tables the cohort builder reads are fetched by default: roughly
# 4.1 GB as served and 37.2 GB uncompressed, against 46.6 GB for the whole
# release. Pass TABLES= to override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="https://physionet.org/files/mimiciii/1.4"
DEST="${DEST:-data/raw/mimic-iii-clinical-database-1.4}"
TABLES="${TABLES:-PATIENTS ADMISSIONS ICUSTAYS CHARTEVENTS LABEVENTS}"

# PhysioNet decides whether to offer the HTTP Basic challenge on /files/ from the
# User-Agent, so a correct Basic credential sent by curl's default UA returns 403,
# the same status an unsigned DUA produces and with no WWW-Authenticate header to
# tell them apart. Two routes work: send "Wget/..." as the UA (see
# scripts/wget_mimic.sh), or use the Django session cookie that
# scripts/physionet_session.sh leaves in a mode-600 jar. The cookie is preferred
# because it does not rest on a User-Agent heuristic; the Basic fallback sets the
# UA explicitly.
JAR="${PHYSIONET_COOKIE_JAR:-$HOME/.physionet-session}"
if [ -s "$JAR" ]; then
  echo "auth=session cookie ($JAR)"
  AUTH=(-b "$JAR" -c "$JAR")
elif [ -f "$HOME/.netrc" ] && grep -q '^[[:space:]]*machine[[:space:]]\+physionet\.org' "$HOME/.netrc"; then
  echo "auth=~/.netrc (HTTP Basic with a wget User-Agent, which is the only UA"
  echo "     PhysioNet offers the Basic challenge to; if this 403s the credential"
  echo "     itself was refused -- run scripts/physionet_session.sh)"
  AUTH=(--netrc -A 'Wget/1.25.0')
else
  echo "no session jar and no ~/.netrc entry; run these first:" >&2
  echo "  ./scripts/physionet_login.sh    # stores the credential, mode 600" >&2
  echo "  ./scripts/physionet_session.sh  # opens the session the files need" >&2
  exit 2
fi

mkdir -p "$DEST"

# One curl invocation per file, resumable, so an interrupted 33 GB CHARTEVENTS
# transfer continues rather than restarting.
for name in $TABLES SHA256SUMS; do
  case "$name" in
    SHA256SUMS) remote="SHA256SUMS.txt"; local_name="SHA256SUMS.txt" ;;
    *)          remote="$name.csv.gz";   local_name="$name.csv.gz" ;;
  esac
  echo "=== $remote ==="
  # -C - resumes a partial file, so an interrupted 4 GB CHARTEVENTS transfer
  # continues instead of restarting. A completed file re-requests as a 416, which
  # curl reports as an error, so that one status is treated as "already done".
  code=0
  curl "${AUTH[@]}" -f -L -C - \
       --retry 5 --retry-delay 10 --retry-connrefused \
       -o "$DEST/$local_name" "$BASE/$remote" || code=$?
  if [ "$code" -ne 0 ]; then
    if [ "$code" -eq 33 ] || [ "$code" -eq 36 ]; then
      echo "  (already complete, server refused the resume range)"
    else
      echo "  curl exit $code on $remote" >&2
      exit "$code"
    fi
  fi
done

echo "=== verifying checksums ==="
# PhysioNet's SHA256SUMS.txt covers the whole release; check only what we pulled.
( cd "$DEST" && for name in $TABLES; do
    line=$(grep -E "[[:space:]]$name\.csv\.gz$" SHA256SUMS.txt || true)
    if [ -z "$line" ]; then
      echo "  $name.csv.gz: NO CHECKSUM PUBLISHED" >&2; continue
    fi
    printf '%s\n' "$line" | shasum -a 256 -c -
  done )

echo
echo "=== sizes ==="
du -h "$DEST"/*.csv.gz
cat <<'NOTE'

Downloaded under the PhysioNet credentialed DUA. These files must not be
committed, copied into the handover package, uploaded to any third-party
service, or sent to any hosted model API. data/ is gitignored; keep it that way.
NOTE
