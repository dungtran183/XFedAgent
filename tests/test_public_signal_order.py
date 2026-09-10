"""The public instance must have one order, and the same one everywhere.

The public-signal order has to be identical in four places: the Circom ``main`` component, the emitted ``public.json``, the
Solidity verifier, and Algorithm 1 of Appendix A. Groth16 verification is
positional — the verifier multiplies signal *i* by the key element ``IC[i+1]`` and
never learns a name — so a permutation between any two of those places produces a
verifier that accepts a proof about a different statement than the one the paper
describes, with no error anywhere to notice.

These checks are cheap and mechanical, which is the point: the ordering is easy to
get right once and easy to break later by inserting a signal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CIRCUIT = REPO / "circuits" / "binary_linear_pov.circom"
VERIFIER = REPO / "contracts" / "PoVVerifier.sol"
ADAPTER = REPO / "contracts" / "PoVVerifierAdapter.sol"
CASE_T0 = REPO / "tests" / "circuit" / "cases" / "T0" / "input.json"
PUBLIC_T0 = REPO / "build" / "proofs" / "T0" / "public.json"
MANUSCRIPT = REPO.parent          # paper.tex / appendix.tex live beside the repo

# x = (C_D, H_theta, H_glob, c, pp_P, eps_copy^2), the grouped public instance.
EXPECTED_ORDER = [
    "validationRoot",
    "modelRoot",
    "globalRoot",
    "tp",
    "tn",
    "fp",
    "fn",
    "tauSensNum",
    "tauSpecNum",
    "denom",
    "copyEpsilonSq",
]


def _circuit_source() -> str:
    return CIRCUIT.read_text(encoding="utf-8")


def declared_public_list() -> list[str]:
    """The names inside ``component main {public [...]}``, in written order."""
    body = re.search(r"component\s+main\s*\{\s*public\s*\[(.*?)\]\s*\}", _circuit_source(), re.S)
    assert body, "main component must declare its public signals explicitly"
    return [name.strip() for name in body.group(1).replace("\n", " ").split(",") if name.strip()]


def declared_input_order() -> list[str]:
    """Every ``signal input`` of the main template, in declaration order.

    Circom orders the public signals of ``public.json`` by declaration order in the
    template, not by the order of the ``public [...]`` list. The two agree here
    only because the declarations are written in the order of Eq. (12) — which is
    exactly the invariant worth pinning.
    """
    source = _circuit_source()
    start = source.index("template BinaryLinearPoV")
    end = source.index("component main")
    return re.findall(r"signal\s+input\s+([A-Za-z_][A-Za-z0-9_]*)", source[start:end])


def test_main_declares_the_public_instance_in_definition_2_order():
    assert declared_public_list() == EXPECTED_ORDER


def test_declaration_order_matches_the_public_list_order():
    declared = declared_input_order()
    assert declared[: len(EXPECTED_ORDER)] == EXPECTED_ORDER
    # The witness signals follow the public instance, so no private input can slip
    # in between and shift a public position.
    assert set(declared[len(EXPECTED_ORDER):]).isdisjoint(EXPECTED_ORDER)


def test_public_instance_is_exactly_eleven_signals():
    assert len(EXPECTED_ORDER) == 11
    assert len(declared_public_list()) == 11


@pytest.mark.skipif(not PUBLIC_T0.exists(), reason="T0 proof not generated")
def test_emitted_public_json_is_in_the_declared_order():
    """The strongest form of the check: the file the verifier consumes."""
    signals = json.loads(PUBLIC_T0.read_text())
    payload = json.loads(CASE_T0.read_text())
    assert len(signals) == len(EXPECTED_ORDER)
    for position, name in enumerate(EXPECTED_ORDER):
        assert signals[position] == str(payload[name]), (
            f"public.json position {position} is {signals[position]}, "
            f"but {name} in the witness is {payload[name]}"
        )


@pytest.mark.skipif(not VERIFIER.exists(), reason="Solidity verifier not exported")
def test_solidity_verifier_fixes_the_same_arity():
    source = VERIFIER.read_text(encoding="utf-8")
    signature = re.search(r"function\s+verifyProof\((.*?)\)\s*public\s+view", source, re.S)
    assert signature, "generated verifier must expose verifyProof"
    assert f"uint[{len(EXPECTED_ORDER)}] calldata _pubSignals" in signature.group(1)
    # One IC element per public signal, plus IC0 for the constant term.
    ic_indices = {int(m) for m in re.findall(r"IC(\d+)x", source)}
    assert ic_indices == set(range(len(EXPECTED_ORDER) + 1))


@pytest.mark.skipif(not ADAPTER.exists(), reason="adapter not present")
def test_adapter_agrees_on_the_signal_count():
    source = ADAPTER.read_text(encoding="utf-8")
    assert f"PUBLIC_SIGNALS = {len(EXPECTED_ORDER)}" in source
    assert f"uint256[{len(EXPECTED_ORDER)}] calldata pubSignals" in source


@pytest.mark.skipif(
    not (MANUSCRIPT / "appendix.tex").exists(), reason="manuscript not beside the repo"
)
def test_algorithm_1_lists_the_public_instance_in_the_same_order():
    text = (MANUSCRIPT / "appendix.tex").read_text(encoding="utf-8")
    # Single-letter TeX arguments may be braced or separated by whitespace.
    # The statement's meaning must not depend on an optional algorithm comment.
    text = re.sub(r"\\(mathcal|mathbf)\s+([A-Za-z])", r"\\\1{\2}", text)
    tuple_line = re.search(
        r"\\State\s*\$\s*x\s*\\gets\s*\((.*?)\)\s*\$", text, re.S
    )
    assert tuple_line, "Algorithm 1 must assemble the public instance explicitly"
    components = [c.strip() for c in tuple_line.group(1).split(",")]
    # Grouped form: the four counts travel as c and the three predicate parameters
    # as pp_P, so the tuple has six slots rather than eleven signals.
    assert len(components) == 6
    for position, needle in enumerate(
        [r"C_{\mathcal{D}}", r"H_\theta", r"H_{\mathrm{glob}}", r"\mathbf{c}",
         r"\mathsf{pp}_P", r"\epsilon_{\mathrm{copy}}^2"]
    ):
        assert needle in components[position], (
            f"Algorithm 1 slot {position} is {components[position]!r}, expected {needle}"
        )
