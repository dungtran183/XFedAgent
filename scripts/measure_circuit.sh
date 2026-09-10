#!/usr/bin/env bash
# Circuit cost measurement on the build machine.
#
# Records R1CS size, key and proof sizes, and the proving-time distribution over
# repeated runs of the honest witness (T0). Every number written here is measured
# on the build machine, not on an edge device; the edge figures the paper reports
# are modelled and stay labelled as such.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD_DIR:-build}"
SNARKJS="${SNARKJS_BIN:-./node_modules/.bin/snarkjs}"
REPS="${REPS:-10}"
CASE="${CASE:-T0}"
OUT="$BUILD/circuit_metrics.csv"
TIMES="$BUILD/proving_times.csv"

WTNS="$BUILD/proofs/$CASE/witness.wtns"
ZKEY="$BUILD/pov_final.zkey"
VKEY="$BUILD/verification_key.json"

for f in "$WTNS" "$ZKEY" "$VKEY"; do
  [ -f "$f" ] || { echo "missing $f — run scripts/run_circuit_tests.sh first" >&2; exit 2; }
done

"$SNARKJS" r1cs info "$BUILD/binary_linear_pov.r1cs" > "$BUILD/r1cs_info.txt" 2>&1
field() { grep -i "$1" "$BUILD/r1cs_info.txt" | tail -1 | sed 's/.*: *//' | tr -d '\r'; }

constraints=$(field '# of Constraints')
wires=$(field '# of Wires')
privates=$(field '# of Private Inputs')
publics=$(field '# of Public Inputs')
labels=$(field '# of Labels')

bytes() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
zkey_bytes=$(bytes "$ZKEY")
vkey_bytes=$(bytes "$VKEY")
proof_bytes=$(bytes "$BUILD/proofs/$CASE/proof.json")
wasm_bytes=$(bytes "$BUILD/binary_linear_pov_js/binary_linear_pov.wasm")

echo "run,seconds,max_rss_kb" > "$TIMES"
echo "measuring proving time over $REPS runs of $CASE ..."
for i in $(seq 1 "$REPS"); do
  start=$(python3 -c 'import time; print(time.perf_counter())')
  /usr/bin/time -l "$SNARKJS" groth16 prove "$ZKEY" "$WTNS" \
      "$BUILD/proofs/$CASE/proof.json" "$BUILD/proofs/$CASE/public.json" \
      > /dev/null 2> "$BUILD/.time_$i.txt"
  end=$(python3 -c 'import time; print(time.perf_counter())')
  secs=$(python3 -c "print(f'{$end - $start:.4f}')")
  rss=$(grep -o '[0-9]*  *maximum resident set size' "$BUILD/.time_$i.txt" | awk '{print int($1/1024)}')
  [ -n "$rss" ] || rss=$(grep -o 'Maximum resident set size[^0-9]*[0-9]*' "$BUILD/.time_$i.txt" | grep -o '[0-9]*$')
  [ -n "$rss" ] || rss=0
  echo "$i,$secs,$rss" >> "$TIMES"
  printf '  run %2d: %ss  rss=%s KB\n' "$i" "$secs" "$rss"
  rm -f "$BUILD/.time_$i.txt"
done

python3 - "$TIMES" "$OUT" "$constraints" "$wires" "$privates" "$publics" "$labels" \
        "$zkey_bytes" "$vkey_bytes" "$proof_bytes" "$wasm_bytes" "$REPS" "$CASE" <<'PY'
import csv, statistics, sys, platform, subprocess

times_path, out_path = sys.argv[1], sys.argv[2]
constraints, wires, privates, publics, labels = sys.argv[3:8]
zkey_b, vkey_b, proof_b, wasm_b = (int(v) for v in sys.argv[8:12])
reps, case = sys.argv[12], sys.argv[13]

with open(times_path) as fh:
    rows = list(csv.DictReader(fh))
secs = [float(r["seconds"]) for r in rows]
rss = [int(r["max_rss_kb"]) for r in rows]

def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "unknown"

mb = lambda b: round(b / 1048576, 3)
rows_out = [
    ("r1cs_constraints", constraints, "count", "measured (build machine)"),
    ("r1cs_wires", wires, "count", "measured (build machine)"),
    ("private_inputs", privates, "count", "measured (build machine)"),
    ("public_inputs", publics, "count", "measured (build machine)"),
    ("labels", labels, "count", "measured (build machine)"),
    ("proving_key_bytes", zkey_b, "bytes", "measured (build machine)"),
    ("proving_key_mb", mb(zkey_b), "MB", "measured (build machine)"),
    ("verification_key_bytes", vkey_b, "bytes", "measured (build machine)"),
    ("proof_json_bytes", proof_b, "bytes", "measured (build machine)"),
    ("witness_wasm_bytes", wasm_b, "bytes", "measured (build machine)"),
    ("proving_seconds_mean", round(statistics.fmean(secs), 4), "s", "measured (build machine)"),
    ("proving_seconds_sd", round(statistics.stdev(secs), 4) if len(secs) > 1 else 0.0, "s", "measured (build machine)"),
    ("proving_seconds_min", round(min(secs), 4), "s", "measured (build machine)"),
    ("proving_seconds_max", round(max(secs), 4), "s", "measured (build machine)"),
    ("proving_max_rss_kb", max(rss), "KB", "measured (build machine)"),
    ("proving_runs", reps, "count", "measured (build machine)"),
    ("witness_case", case, "id", "measured (build machine)"),
    ("build_machine_cpu", cpu_name(), "text", "environment"),
    ("build_machine_platform", f"{platform.system()} {platform.release()} {platform.machine()}", "text", "environment"),
]
with open(out_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["metric", "value", "unit", "provenance"])
    w.writerows(rows_out)

print()
print(f"constraints={constraints}  zkey={mb(zkey_b)} MB  vkey={vkey_b} B  proof={proof_b} B")
print(f"proving: mean={statistics.fmean(secs):.3f}s  sd={statistics.stdev(secs) if len(secs)>1 else 0:.3f}s  "
      f"max={max(secs):.3f}s  maxRSS={max(rss)/1024:.0f} MB  over {reps} runs")
print(f"written: {out_path}")
PY
