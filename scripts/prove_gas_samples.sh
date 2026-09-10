#!/usr/bin/env bash
# Proofs for the extra admissible witnesses the gas measurement needs.
#
# The T0-T8 suite yields three proofs; the gas requirement asks for at least five
# verify transactions. This script proves the samples in tests/circuit/gas_cases
# into build/proofs/<id>/ so scripts/measure_gas.js can submit them alongside the
# suite's own proofs. Every sample must verify; a failure means the extra
# witnesses were built wrong.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD_DIR:-build}"
CASES="tests/circuit/gas_cases"
LOGS="$BUILD/test_logs"
WASM="$BUILD/binary_linear_pov_js/binary_linear_pov.wasm"
GENW="$BUILD/binary_linear_pov_js/generate_witness.js"
ZKEY="$BUILD/pov_final.zkey"
VKEY="$BUILD/verification_key.json"
SNARKJS="${SNARKJS_BIN:-./node_modules/.bin/snarkjs}"

for f in "$WASM" "$GENW" "$ZKEY" "$VKEY"; do
  if [ ! -f "$f" ]; then
    echo "missing $f — run 'npm run compile:linear-pov' and the Groth16 setup first" >&2
    exit 2
  fi
done
if [ ! -d "$CASES" ]; then
  echo "missing $CASES — run 'node tests/circuit/gen_gas_samples.mjs' first" >&2
  exit 2
fi

mkdir -p "$LOGS"
overall=0
for dir in "$CASES"/G*; do
  id="$(basename "$dir")"
  log="$LOGS/$id.log"
  out="$BUILD/proofs/$id"
  mkdir -p "$out"
  : > "$log"
  echo "### $id — extra admissible witness for gas measurement" >> "$log"

  node "$GENW" "$WASM" "$dir/input.json" "$out/witness.wtns" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "$id: witness generation failed — the sample is not admissible" >&2
    overall=1
    continue
  fi
  "$SNARKJS" groth16 prove "$ZKEY" "$out/witness.wtns" "$out/proof.json" "$out/public.json" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "$id: proving failed" >&2
    overall=1
    continue
  fi
  "$SNARKJS" groth16 verify "$VKEY" "$out/public.json" "$out/proof.json" >> "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "$id: off-chain verification failed" >&2
    overall=1
    continue
  fi
  echo "$id proved and verified -> $out/proof.json"
done

if [ $overall -ne 0 ]; then
  echo "at least one gas sample did not produce a verifying proof" >&2
fi
exit $overall
