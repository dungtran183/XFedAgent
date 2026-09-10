#!/usr/bin/env bash
# Store the PhysioNet credential in ~/.netrc so the download can run unattended.
#
# fetch_mimic.sh passes --netrc-optional, so once this file exists curl
# authenticates without a prompt and the transfer can run in the background.
#
# The password is read with `read -s`, so it is never echoed, never placed on a
# command line, and never written to shell history. The only copy on disk is
# ~/.netrc, mode 600.
#
#   ./scripts/physionet_login.sh --forget   remove it when the download is done
#   ./scripts/physionet_login.sh --check     re-test a stored credential
set -euo pipefail

NETRC="$HOME/.netrc"

# The access probe must send a wget User-Agent. PhysioNet decides whether to issue
# the HTTP Basic challenge on /files/ from the User-Agent: a request identifying
# itself as Wget/... receives 401 + WWW-Authenticate: Basic realm="PhysioNet" and
# then 200/206 once the credential is supplied, while the byte-identical request
# from curl's default UA, from a browser UA, or with no UA at all receives a flat
# 403 carrying no challenge at all. Probing without that header therefore reports
# 403 for a valid credential on a signed DUA, and an earlier version of this
# script did exactly that and blamed the DUA for it. Do not drop the -A flag.
probe_access() {
  echo "checking access to the restricted files..."
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -L --netrc --max-time 30 \
         -A 'Wget/1.25.0' \
         --range 0-1023 https://physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz)
  case "$code" in
    200|206)
      echo "access=granted (HTTP $code) -- ready for ./scripts/fetch_mimic.sh"
      return 0 ;;
    403)
      echo "access=denied (HTTP 403) on a request that did carry the Basic challenge." >&2
      echo "  The credential reached the server and the server refused it, so this is" >&2
      echo "  an authorisation result and not a transport one. The likely cause is that" >&2
      echo "  the MIMIC-III v1.4 DUA is not active on this account -- confirm at" >&2
      echo "  https://physionet.org/settings/credentialing/ before concluding it." >&2
      echo "  ./scripts/physionet_session.sh reaches the same files over a Django" >&2
      echo "  session cookie if the Basic route is ever withdrawn." >&2
      return 3 ;;
    401)
      echo "access=denied (HTTP 401): username or password rejected;" >&2
      echo "  re-run after ./scripts/physionet_login.sh --forget" >&2
      return 3 ;;
    *)
      echo "unexpected HTTP $code" >&2
      return 3 ;;
  esac
}

if [ "${1:-}" = "--forget" ]; then
  [ -f "$NETRC" ] || { echo "no $NETRC"; exit 0; }
  # Drop only the physionet.org machine block, leave any other host alone.
  python3 - "$NETRC" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
tokens = path.read_text().split()
kept, skip = [], False
i = 0
while i < len(tokens):
    if tokens[i] == "machine":
        skip = tokens[i + 1] == "physionet.org"
    if not skip:
        kept.append(tokens[i])
    i += 1
path.write_text((" ".join(kept) + "\n") if kept else "")
print("physionet.org entry removed")
PY
  exit 0
fi

# --check runs only the probe. Without it the early exit below means the one
# command that reports whether access works cannot be run a second time.
if [ "${1:-}" = "--check" ]; then
  if [ -f "$NETRC" ] && grep -q '^[[:space:]]*machine[[:space:]]\+physionet\.org' "$NETRC"; then
    probe_access
    exit $?
  fi
  echo "no physionet.org entry in $NETRC; run $0 with no arguments first" >&2
  exit 2
fi

if [ -f "$NETRC" ] && grep -q '^[[:space:]]*machine[[:space:]]\+physionet\.org' "$NETRC"; then
  echo "physionet.org already present in $NETRC; nothing to do"
  echo "  (re-test access with: $0 --check)"
  exit 0
fi

read -r -p "PhysioNet username: " PN_USER
[ -n "$PN_USER" ] || { echo "username required" >&2; exit 2; }
read -r -s -p "PhysioNet password (not echoed): " PN_PASS
echo
[ -n "$PN_PASS" ] || { echo "password required" >&2; exit 2; }

umask 177
printf 'machine physionet.org\n  login %s\n  password %s\n' "$PN_USER" "$PN_PASS" >> "$NETRC"
chmod 600 "$NETRC"
unset PN_PASS

echo "stored in $NETRC (mode $(stat -f '%Lp' "$NETRC"))"
probe_access
