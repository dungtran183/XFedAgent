"""Cross-checks between the software predicate and the compiled circuit.

The Python backend and the Circom circuit must agree on the admission decision,
including at the threshold boundary. These tests re-derive the circuit's exact
integer form — cross-multiplication over a public denominator — and compare it to
``metrics.passes_class_aware`` on the same counts, so a drift in either
implementation shows up here rather than in a manuscript number.

The generated circuit test vectors (``tests/circuit/cases``) are checked against
the same predicate when they are present, which ties the two suites together
without requiring circom to be installed to run the Python tests.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from xfedagent.metrics import ConfusionCounts, passes_class_aware, predicate_margin

CASES_DIR = Path(__file__).parent / "circuit" / "cases"
DENOM = 100          # public denominator d used by the generator
TAU_SE_NUM = 70      # tau_se = 0.70
TAU_SP_NUM = 70      # tau_sp = 0.70


def circuit_predicate(counts: ConfusionCounts, tau_se_num: int, tau_sp_num: int, denom: int) -> bool:
    """The predicate exactly as the circuit enforces it: integers, no division."""
    if denom <= 0 or not 0 < tau_se_num <= denom or not 0 < tau_sp_num <= denom:
        return False
    if counts.positives <= 0 or counts.negatives <= 0:
        return False
    sens = counts.tp * denom >= tau_se_num * (counts.tp + counts.fn)
    spec = counts.tn * denom >= tau_sp_num * (counts.tn + counts.fp)
    return sens and spec


@pytest.mark.parametrize(
    "tp,tn",
    [(40, 40), (35, 40), (40, 35), (34, 40), (40, 34), (50, 50), (0, 50), (50, 0)],
)
def test_circuit_form_agrees_with_the_software_predicate(tp, tn):
    counts = ConfusionCounts(tp=tp, tn=tn, fp=50 - tn, fn=50 - tp)
    assert circuit_predicate(counts, TAU_SE_NUM, TAU_SP_NUM, DENOM) == passes_class_aware(
        counts, TAU_SE_NUM / DENOM, TAU_SP_NUM / DENOM, tolerance=0.0
    )


def test_threshold_boundary_is_inclusive_in_both_implementations():
    # 35/50 = 0.70 exactly: admitted by Definition 2's non-strict inequality.
    at_bound = ConfusionCounts(tp=35, tn=40, fp=10, fn=15)
    assert circuit_predicate(at_bound, TAU_SE_NUM, TAU_SP_NUM, DENOM) is True
    assert passes_class_aware(at_bound, 0.70, 0.70, tolerance=0.0) is True

    below = ConfusionCounts(tp=34, tn=40, fp=10, fn=16)
    assert circuit_predicate(below, TAU_SE_NUM, TAU_SP_NUM, DENOM) is False
    assert passes_class_aware(below, 0.70, 0.70, tolerance=0.0) is False


@pytest.mark.parametrize("correct,admitted,positive_margin", [(30, False, False), (31, True, False), (32, True, False), (33, True, True)])
def test_nominal_65_tolerance_03_matches_integer_62_admission(correct, admitted, positive_margin):
    counts = ConfusionCounts(tp=correct, tn=correct, fp=50 - correct, fn=50 - correct)
    assert passes_class_aware(counts, 0.65, 0.65, tolerance=0.03) is admitted
    assert circuit_predicate(counts, 62, 62, 100) is admitted
    # Tolerance affects admission only: the nominal reward does not start until
    # both counts reach 33/50, and exact 65/100 admission has that same boundary.
    assert circuit_predicate(counts, 65, 65, 100) is positive_margin
    margin = predicate_margin("class_aware", counts.accuracy, counts.sensitivity, counts.specificity, 0.75, 0.65, 0.65)
    assert (margin > 0) is positive_margin


def test_integer_form_avoids_the_floating_point_boundary_error():
    """A rational comparison in floating point can misjudge an exact boundary.

    At tau = 0.56 with 50 positives the exact bound is 28, but ``0.56 * 50``
    evaluates to 28.000000000000004, so ``tp >= tau * positives`` rejects tp = 28.
    Cross-multiplication over a public denominator is exact at every threshold,
    which is why the circuit enforces the predicate that way. (tau = 0.70 happens
    to be representable well enough that 0.7 * 50 == 35.0, so the hazard is not
    visible at the paper's operating point — it is visible at neighbouring ones.)
    """
    tau, n, exact = 0.56, 50, 28
    assert tau * n > exact                      # the floating-point hazard itself
    assert not (exact >= tau * n)               # a rational gate rejects the boundary
    assert exact * 100 >= 56 * n                # the integer form admits it

    counts = ConfusionCounts(tp=exact, tn=40, fp=10, fn=n - exact)
    assert circuit_predicate(counts, 56, 56, 100) is True


def test_constant_classifier_fails_the_circuit_form_too():
    majority = ConfusionCounts(tp=0, tn=50, fp=0, fn=50)
    assert circuit_predicate(majority, TAU_SE_NUM, TAU_SP_NUM, DENOM) is False


@pytest.mark.skipif(not (CASES_DIR / "manifest.json").exists(), reason="circuit vectors not generated")
def test_generated_vectors_match_their_declared_expectation():
    manifest = json.loads((CASES_DIR / "manifest.json").read_text())
    checked = 0
    for entry in manifest:
        case_dir = CASES_DIR / entry["id"]
        payload = json.loads((case_dir / "input.json").read_text())
        counts = ConfusionCounts(
            tp=int(payload["tp"]), tn=int(payload["tn"]),
            fp=int(payload["fp"]), fn=int(payload["fn"]),
        )
        total = counts.tp + counts.tn + counts.fp + counts.fn
        predicate_ok = circuit_predicate(
            counts, int(payload["tauSensNum"]), int(payload["tauSpecNum"]), int(payload["denom"])
        )
        # A case is expected to pass only if the counts sum to n and the predicate
        # holds. The remaining failure modes (replay, stale or edited commitments)
        # are invisible to the counts, so they are exercised by the circuit suite.
        if entry["expected"] == "pass":
            assert total == 100, f"{entry['id']}: counts must sum to n"
            assert predicate_ok, f"{entry['id']}: expected to clear the predicate"
            checked += 1
        elif "count" in entry["scenario"] or "sensitivity" in entry["scenario"] or "specificity" in entry["scenario"]:
            assert not (total == 100 and predicate_ok), f"{entry['id']}: expected to fail on counts"
            checked += 1
    assert checked >= 6


@pytest.mark.skipif(not (CASES_DIR / "manifest.json").exists(), reason="circuit vectors not generated")
def test_boundary_cases_are_present_in_the_generated_suite():
    manifest = json.loads((CASES_DIR / "manifest.json").read_text())
    ids = {entry["id"] for entry in manifest}
    # T3 and T5 are the inclusive-boundary cases; without them a circuit that
    # rejects at the threshold would pass the suite.
    assert {"T0", "T3", "T5"} <= ids
    by_id = {e["id"]: e for e in manifest}
    assert by_id["T3"]["expected"] == "pass"
    assert by_id["T5"]["expected"] == "pass"


@pytest.mark.skipif(not shutil.which("circom") or not shutil.which("node"), reason="Circom/Node unavailable")
def test_compiled_arithmetic_domains_and_nonempty_classes(tmp_path):
    """Check actual witness satisfiability; no setup, proof or retained-file writes."""
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "node_modules/circomlib").exists():
        pytest.skip("circomlib unavailable")
    built = subprocess.run(
        ["circom", str(repo / "circuits/binary_linear_pov.circom"), "--wasm", "-o", str(tmp_path)],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    assert built.returncode == 0, built.stderr + built.stdout
    script = r"""
      const {init, arithmeticCases} = await import(process.argv[1]);
      const {createRequire} = await import('node:module');
      const require = createRequire(import.meta.url);
      const fs = require('node:fs');
      const builder = require(process.argv[2] + '/binary_linear_pov_js/witness_calculator.js');
      await init();
      const witness = await builder(fs.readFileSync(process.argv[2] + '/binary_linear_pov_js/binary_linear_pov.wasm'));
      const results = [];
      for (const c of arithmeticCases()) {
        let accepted = false;
        try { await witness.calculateWitness(c.input, true); accepted = true; } catch (_) {}
        results.push({name: c.name, expected: c.accepted, accepted});
      }
      console.log(JSON.stringify(results));
    """
    checked = subprocess.run(
        ["node", "--input-type=module", "-e", script,
         (repo / "tests/circuit/gen_inputs.mjs").as_uri(), str(tmp_path)],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    assert checked.returncode == 0, checked.stderr + checked.stdout
    results = json.loads(checked.stdout.strip().splitlines()[-1])
    assert len(results) == 13
    assert all(case["accepted"] == case["expected"] for case in results), results
