# XFedAgent — reference implementation and reproducibility manifest

Reference implementation accompanying:

> T.-D. Tran, P.-D. Bui, and V.-H. Pham, "XFedAgent: Efficient and Verifiable
> Cross-Chain Federated Learning for Trustworthy IoT Edge Intelligence,"
> *IEEE Internet of Things Journal*, Special Issue on Effective, Efficient, and
> Trustworthy AI Agents for IoT (under review).

XFedAgent gates federated model updates on a **Proof-of-Validation (PoV)**: an
agent proves that the model it committed to reaches an accuracy threshold on a
validation subset chosen *after* the commitment, rather than proving that it
executed training honestly. Admitted updates are weighted by an on-chain
reputation score, relayed across two ledgers, and folded into a global model
whose Merkle root is published for public recomputation.

This repository holds the experiment framework, the PoV circuit, the on-chain
contracts, and the exact configuration needed to re-run the study. It is a
**reference implementation and reproducibility manifest**, not a production
deployment.

---

## 1. What this code measures, and what it models

The paper's evaluation is **simulation-only**; no physical edge device is used
at any point. This repository reflects that scope precisely, and the
distinction matters when reading any number it prints:

| Quantity | Status | Where it comes from |
|---|---|---|
| Accuracy, AUC-ROC, F1, acceptance/rejection rates | **Computed** | Real PyTorch training and evaluation in this repository |
| Reputation trajectories, exclusion behaviour | **Computed** | `runner.py`, executed per round |
| Commitment binding, replay/nonce checks, Merkle roots | **Computed** | `commitments.py`, `relay.py` |
| Proof-generation latency | **Modelled** | `proof_seconds_per_sample x validation_size`; *not* a timed SNARK run |
| Per-round energy (Wh) | **Modelled** | Phase duration x a fixed power envelope (`energy.py`) |
| Gas cost | **Modelled** | Fixed per-operation constants from `relay.enabled` accounting |
| Thermal throttling, I/O and memory-bandwidth effects | **Not represented** | Out of scope for a simulation-only study |

Two substitutions are deliberate and are stated in the paper:

- The software PoV backend commits with **SHA3-256** as a stand-in for the
  **Poseidon-2** sponge used by the Circom circuit. Both are collision-resistant
  256-bit commitments; the substitution removes a native field-arithmetic
  dependency without changing the commitment's binding semantics. It does
  **not** produce a SNARK proof.
- `modeled_proof_seconds` is an analytic cost model. Any statement about edge
  feasibility derived from it is a *modelled* envelope, not a measurement.

Set `pov.backend: "snarkjs"` with a compiled `circuit_wasm` and `proving_key` to
run the real prover instead.

---

## 2. Installation

```bash
python3 -m pip install -e .
python3 -m xfedagent env          # report which external toolchain binaries are present
```

Python 3.11+ is required. Core dependencies are `numpy`, `scipy`,
`scikit-learn`, and `torch`.

---

## 3. Running without restricted-access data

The study uses MIMIC-III, which is **credentialed-access**: it cannot be
redistributed, and no patient data of any kind is contained in this repository.
Reviewers without a PhysioNet credential can still exercise the entire pipeline
against a synthetic surrogate that matches the real cohort's tensor shape
(24 hourly steps x 17 features) and non-IID partitioning:

```bash
python3 -m xfedagent validate-config --config configs/synthetic-smoke.json
python3 -m xfedagent run --config configs/synthetic.json --output results
```

**Synthetic-surrogate numbers are not MIMIC-III numbers** and must not be
compared against the tables in the paper. The surrogate exists to demonstrate
that the protocol runs, that the ablation switches take effect, and that the
integrity checks fire — not to reproduce reported accuracies.

### Reproducing the reported results

Obtain MIMIC-III v1.4 from PhysioNet, apply the preprocessing pipeline of
Harutyunyan *et al.* (48-hour in-hospital mortality, adult ICU stays > 48 h,
17 vitals/labs, hourly resampling, forward-fill imputation, per-feature
z-scoring), and write the flattened cohort to
`data/processed/mimic_mortality_24h.csv` with one row per sample,
`24 x 17 = 408` feature columns, a `mortality_48h` label column, and a
`subject_id` column used to keep a patient inside a single split. Then:

```bash
python3 -m xfedagent preflight --config configs/full.json
python3 -m xfedagent run --config configs/full.json --output results
```

---

## 4. Component ablation

`configs/full.json` carries an `ablation` block; every switch defaults to `true`,
so an omitted block reproduces the complete framework. The sweep re-runs one
base configuration once per (arm, seed) pair, holding the data partition, model,
attack schedule, and round count fixed, so the gap between two arms isolates the
components that separate them:

```bash
python3 -m xfedagent ablation \
  --config configs/full.json \
  --arms full,no-pov,no-rep,no-pov+rep,no-rot,no-copy,no-xchain \
  --seeds 0-9 \
  --output results/ablation \
  --latex
```

| Arm | Component removed | Effect |
|---|---|---|
| `full` | — | Complete framework |
| `no-pov` | PoV utility gate | Every update is admitted regardless of validation accuracy |
| `no-copy` | In-circuit copy detector | A replayed global model is no longer rejected |
| `no-rep` | Reputation weighting | Plain unweighted mean; no agent is ever excluded |
| `no-rot` | Dynamic validation rotation | The round-0 validation subset is frozen for all rounds |
| `no-xchain` | Cross-chain relay | Single-chain operation; relaying is actually disabled, not just relabelled |

Arms compose: `no-pov+rep` removes both. The sweep writes
`ablation_runs.csv` (one row per run), `ablation_summary.csv` (mean ± sd per arm
with the accuracy delta against `full`), and `ablation_manifest.json`. `--latex`
prints a `booktabs` table body.

### Two-factor analysis

`full`, `no-pov`, `no-rep` and `no-pov+rep` are the four cells of a 2x2 design
over the utility gate and reputation weighting. Pass `--factorial` to report its
main effects and interaction:

```bash
python3 -m xfedagent ablation --config configs/full.json \
  --arms full,no-pov,no-rep,no-pov+rep --seeds 0-9 \
  --output results/ablation --factorial
```

The **main effect** of a factor is its average simple effect across the levels of
the other factor; the **interaction** is the difference between the two simple
effects. A negative interaction means the mechanisms are partially redundant, so
adding the second recovers less than it would alone. Report the main effects and
the interaction rather than summing the two simple effects, which overstates the
combined benefit. On the reported MIMIC-III runs the effects are `+21.1` pp for
the gate, `+13.3` pp for reputation, and `-12.8` pp for the interaction: the
additive prediction of `95.9%` exceeds the `86.2%` that the same quantised model
reaches on clean data, so the design saturates against that ceiling.

> **Configure the sweep so honest agents can clear the gate.** The PoV threshold
> (`pov.threshold`, default 0.75) rejects *every* update — honest and malicious
> alike — until local training is good enough to exceed it. A deliberately short
> run (few rounds, one local epoch) therefore reports `full` as *worse* than
> `no-pov`, which is an artefact of the truncated schedule rather than a finding.
> Use the full 100-round, `E=5` schedule for any number that is reported.

---

## 5. Configuration reference

Everything needed to reproduce a run lives in one JSON file; `config.json` is
copied verbatim into each run directory, and `summary.json` records a
`config_digest` over it.

| Section | Reported setting |
|---|---|
| `data` | 26 424 samples, 10 clients, 24 timesteps, 17 features, validation pool 1 000, Dirichlet α = 0.5, 70/15/15 split |
| `model` | 1D-CNN, channels 8/16/32, Adam, lr 1e-3, weight decay 1e-4, batch 32, `E = 5` local epochs, 8-bit quantisation, copy ε = 1e-4 |
| `pov` | threshold τ = 0.75, `|D_val|` = 100, tolerance ε = 0.03, per-round rotation, 0.184 s/sample modelled proving |
| `federation` | 100 rounds, 10 clients/round, 30 % Byzantine, attacks {label_flip, random_gradient, alie, minmax, backdoor}, r₀ = 0.5, r_min = 0.2, α = 0.1, β = 0.5 |
| `relay` | 2 chains, 7 relayers, 2 faulty, 6-block source finality at 12.8 s, 2.0 s destination finality, 207 000 + 78 000 gas |
| `energy` | 5.2 W training, 5.8 W proving, 3.8 W communication, 2.7 W idle |

Seeds `{0,…,9}` are reported in the paper; every stochastic source is derived
from `federation.seed`.

---

## 6. Outputs

| File | Contents |
|---|---|
| `config.json` | Exact configuration used |
| `rounds.csv` | Per-round accuracy, AUC, F1, accepted/rejected counts, reputation state, global-model Merkle root, input-binding hash |
| `summary.json` | Final metrics, integrity counters, relay accounting, modelled proof and energy statistics, ablation label |
| `models/round_XXXX.pt` | Per-round global model, only when `output.save_round_models` is enabled |

## 7. Repository layout

```
src/xfedagent/    Federation runner, PoV backend, commitments, relay, energy, ablation sweep
circuits/         Circom PoV circuit (binary_linear_pov.circom)
contracts/        TrustAnchor and RelayBridge Solidity contracts
configs/          full.json (reported study), synthetic.json, synthetic-smoke.json
scripts/          Convenience drivers
tests/            Unit and pipeline tests
```

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

## 9. Data and privacy

No patient-level data, no credentialed-access extract, and no derived record
that could identify an individual is contained in this repository. MIMIC-III
must be obtained directly from PhysioNet under its own data-use agreement. The
synthetic surrogate in `data/synthetic` is generated from a parametric model and
contains no real observations.

## 10. Licence

Released under the MIT Licence; see `LICENSE`.
