#!/usr/bin/env bash
# Recursive mirror of MIMIC-III v1.4:
#
#   wget -r -N -c -np https://physionet.org/files/mimiciii/1.4/
#
# Two constraints shape the flags. PhysioNet gates the HTTP Basic challenge on the
# User-Agent: a request for /files/mimiciii/1.4/ carrying "Wget/..." gets 401 +
# WWW-Authenticate: Basic realm="PhysioNet" and then 200 once the credential is
# supplied, while the same request from curl or a browser UA gets a flat 403 with
# no challenge. And --ask-password reads the terminal, so it cannot run
# unattended; the credential is supplied in a mode-600 wgetrc built from ~/.netrc
# instead, and never appears on a command line, in the environment, or in the log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$ROOT/data/raw}"        # gitignored; wget appends physionet.org/files/...
LOG="${LOG:-$DEST/wget-mimic.log}"

RC="$(umask 177; mktemp "$HOME/.wgetrc-physionet.XXXXXX")"
trap 'rm -f "$RC"' EXIT INT TERM
python3 - "$RC" <<'PY'
import netrc, sys
user, _, password = netrc.netrc().authenticators("physionet.org")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(f"user = {user}\npassword = {password}\n")
PY
chmod 600 "$RC"

mkdir -p "$DEST"
cd "$DEST"
umask 022                              # dirs need +x or wget cannot descend into them

echo "=== $(date '+%F %T') wget -r -N -c -np https://physionet.org/files/mimiciii/1.4/ ==="
WGETRC="$RC" wget -r -N -c -np \
  https://physionet.org/files/mimiciii/1.4/ 2>&1 | tee -a "$LOG"
