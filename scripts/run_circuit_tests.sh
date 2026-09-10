#!/usr/bin/env bash
# PoV circuit test suite T0-T8.
#
# Each case is driven end to end: witness generation, then Groth16 prove and
# verify for the cases expected to pass. A negative case is satisfied when the
# constraint system rejects it, which surfaces as a non-zero exit from witness
# generation; the run is recorded either way.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD_DIR:-build}"
CASES="tests/circuit/cases"
LOGS="$BUILD/test_logs"
WASM="$BUILD/binary_linear_pov_js/binary_linear_pov.wasm"
GENW="$BUILD/binary_linear_pov_js/generate_witness.js"
ZKEY="$BUILD/pov_final.zkey"
VKEY="$BUILD/verification_key.json"
SNARKJS="${SNARKJS_BIN:-./node_modules/.bin/snarkjs}"
REPORT="$BUILD/test_report.csv"

for f in "$WASM" "$GENW" "$ZKEY" "$VKEY"; do
  if [ ! -f "$f" ]; then
    echo "missing $f — run 'npm run compile:linear-pov' and the Groth16 setup first" >&2
    exit 2
  fi
done

mkdir -p "$LOGS"
echo "test_id,expected,actual,exit_code,status,scenario" > "$REPORT"

overall=0
for dir in "$CASES"/T*; do
  id="$(basename "$dir")"
  expected="$(node -e "
    const m=require('./$CASES/manifest.json');
    const c=m.find(c=>c.id==='$id');
    process.stdout.write(c?c.expected:'');
  ")"
  scenario="$(node -e "
    const m=require('./$CASES/manifest.json');
    const c=m.find(c=>c.id==='$id');
    process.stdout.write(c?c.scenario:'');
  ")"
  [ -n "$expected" ] || { echo "$id not in manifest" >&2; continue; }

  log="$LOGS/$id.log"
  : > "$log"
  {
    echo "### $id — expected: $expected — $scenario"
    echo "\$ node $GENW $WASM $dir/input.json $BUILD/proofs/$id/witness.wtns"
  } >> "$log"

  mkdir -p "$BUILD/proofs/$id"
  node "$GENW" "$WASM" "$dir/input.json" "$BUILD/proofs/$id/witness.wtns" >> "$log" 2>&1
  wcode=$?
  echo "witness exit=$wcode" >> "$log"

  if [ $wcode -ne 0 ]; then
    actual="fail"; code=$wcode
  else
    {
      echo "\$ $SNARKJS groth16 prove $ZKEY $BUILD/proofs/$id/witness.wtns ..."
    } >> "$log"
    "$SNARKJS" groth16 prove "$ZKEY" "$BUILD/proofs/$id/witness.wtns" \
      "$BUILD/proofs/$id/proof.json" "$BUILD/proofs/$id/public.json" >> "$log" 2>&1
    pcode=$?
    echo "prove exit=$pcode" >> "$log"
    if [ $pcode -ne 0 ]; then
      actual="fail"; code=$pcode
    else
      "$SNARKJS" groth16 verify "$VKEY" "$BUILD/proofs/$id/public.json" \
        "$BUILD/proofs/$id/proof.json" >> "$log" 2>&1
      vcode=$?
      echo "verify exit=$vcode" >> "$log"
      if [ $vcode -eq 0 ]; then actual="pass"; else actual="fail"; fi
      code=$vcode
    fi
  fi

  if [ "$actual" = "$expected" ]; then status="OK"; else status="MISMATCH"; overall=1; fi
  printf '%s,%s,%s,%s,%s,"%s"\n' "$id" "$expected" "$actual" "$code" "$status" "$scenario" >> "$REPORT"
  printf '%-5s expected=%-5s actual=%-5s exit=%-3s %s\n' "$id" "$expected" "$actual" "$code" "$status"
done

echo
echo "report: $REPORT   logs: $LOGS/"
if [ $overall -ne 0 ]; then
  echo "SUITE FAILED: at least one case did not match its expectation" >&2
else
  echo "SUITE PASSED: every case matched its expectation"
fi
exit $overall
