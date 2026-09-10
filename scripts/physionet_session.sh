#!/usr/bin/env bash
# Open an authenticated PhysioNet session and leave the cookie jar for fetch_mimic.sh.
#
# PhysioNet serves the HTTP Basic challenge on /files/ only to clients whose
# User-Agent identifies as Wget; the same request from curl or a browser UA gets a
# flat 403 with no challenge, which is indistinguishable from an unsigned DUA. Two
# routes work: a wget User-Agent with Basic auth (see scripts/wget_mimic.sh), or
# the Django session cookie plus CSRF token this script obtains, which does not
# depend on a User-Agent heuristic.
#
# The password is read from ~/.netrc by Python's netrc parser and written into a
# mode-600 POST body that is deleted immediately after use, so it never appears on
# a command line, in `ps`, or in shell history.
#
# The cookie jar is a bearer credential for the session, so it lives in the same
# mode-600 regime and should be removed when the download is done:
#   ./scripts/physionet_session.sh --forget
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JAR="${PHYSIONET_COOKIE_JAR:-$HOME/.physionet-session}"
PROBE="https://physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz"

if [ "${1:-}" = "--forget" ]; then
  rm -f "$JAR" && echo "removed $JAR"
  exit 0
fi

if [ ! -f "$HOME/.netrc" ] || ! grep -q '^[[:space:]]*machine[[:space:]]\+physionet\.org' "$HOME/.netrc"; then
  echo "no physionet.org entry in ~/.netrc; run ./scripts/physionet_login.sh first" >&2
  exit 2
fi

umask 177
BODY="$(mktemp -t physionet-post)"
trap 'rm -f "$BODY"' EXIT INT TERM

token=$(curl -s -c "$JAR" -b "$JAR" https://physionet.org/login/ \
        | grep -o 'name="csrfmiddlewaretoken" value="[^"]*"' | head -1 \
        | sed 's/.*value="//; s/"$//')
[ -n "$token" ] || { echo "could not obtain a CSRF token from the login page" >&2; exit 3; }

python3 - "$BODY" "$token" <<'PY'
import netrc, sys, urllib.parse
body_path, token = sys.argv[1], sys.argv[2]
user, _, password = netrc.netrc().authenticators("physionet.org")
with open(body_path, "w", encoding="utf-8") as handle:
    handle.write(urllib.parse.urlencode(
        {"csrfmiddlewaretoken": token, "username": user, "password": password, "next": "/"}
    ))
print(f"authenticating as {user}")
PY

curl -s -o /dev/null -c "$JAR" -b "$JAR" -e https://physionet.org/login/ \
     -d @"$BODY" https://physionet.org/login/
rm -f "$BODY"
chmod 600 "$JAR"

code=$(curl -s -o /dev/null -w '%{http_code}' -L -b "$JAR" -c "$JAR" \
       --max-time 30 --range 0-1023 "$PROBE")
case "$code" in
  200|206) echo "session=active (HTTP $code); jar=$JAR -- ready for ./scripts/fetch_mimic.sh" ;;
  403)     echo "session opened but the v1.4 DUA is not active on this account (HTTP 403)" >&2; exit 3 ;;
  *)       echo "unexpected HTTP $code; credentials may be wrong (re-run physionet_login.sh --forget)" >&2; exit 3 ;;
esac
