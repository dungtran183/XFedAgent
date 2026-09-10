# XFedAgent — reference implementation and reproducibility manifest

Reference implementation accompanying:

> T.-D. Tran, P.-D. Bui, and V.-H. Pham, "XFedAgent: Efficient and Verifiable
> Cross-Chain Federated Learning for Trustworthy IoT Edge Intelligence,"
> *IEEE Internet of Things Journal*, Special Issue on Effective, Efficient, and
> Trustworthy AI Agents for IoT (under review).

XFedAgent gates federated model updates on a **Proof-of-Validation (PoV)**: an
agent proves that the model it committed to clears a class-aware utility
predicate on a validation subset chosen *after* the commitment, rather than
proving that it executed training honestly. The predicate enforces sensitivity
and specificity separately, because a raw-accuracy threshold is satisfiable by a
constant classifier on an imbalanced cohort (see §3b). Admitted updates are weighted by an on-chain
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
| R1CS size, key sizes, proof size | **Measured (build machine)** | `circom` + `snarkjs` on the committed circuit; `build/r1cs_info.txt`, `build/circuit_metrics.csv` |
| Proving time, peak prover memory | **Measured (build machine)** | 10 timed Groth16 runs; `build/proving_times.csv` |
| Proof-generation latency **on an edge device** | **Modelled** | `proof_seconds_per_sample x validation_size`; *not* a timed SNARK run, and not the build-machine measurement above |
| Per-round energy (Wh) | **Modelled** | Phase duration x a fixed power envelope (`energy.py`) |
| Verify / submit gas, deployed verifier | **Measured (local EVM)** | Hardhat against the snarkjs-generated verifier; `build/gas_report.csv` |
| Gas in the round accounting | **Modelled** | Fixed per-operation constants from `relay.enabled` accounting |
| Confusion counts, sensitivity, specificity, admission decisions | **Computed** | `metrics.py`, evaluated per submission |
| Thermal throttling, I/O and memory-bandwidth effects | **Not represented** | Out of scope for a simulation-only study |

The software and circuit execution paths have distinct representations:

- The software PoV backend commits with **SHA3-256** to the exact named tensors,
  dtypes, shapes and buffers that PyTorch evaluates. Its transcript also binds
  the global model, validation data, confusion counts, decision rule and policy.
  It does **not** produce a SNARK proof. The separate binary-linear circuit uses
  the circomlib Poseidon sponge over integer witnesses. A quantised commitment
  alone cannot bind an evaluation of different, unquantised tensors.
- `modeled_proof_seconds` is an analytic cost model. Any statement about edge
  feasibility derived from it is a *modelled* envelope, not a measurement.

The federation runner uses `pov.backend: "software"`. Real Groth16 execution is
provided by the standalone circuit scripts; selecting `snarkjs` in a federation
config does not connect the 1D-CNN runner to an integer CNN proving backend.

**Which circuit the measured figures describe.** The preceding reference build of
`circuits/binary_linear_pov.circom` compiled, proved and verified end to end, and every
figure labelled *measured (build machine)* or *measured (local EVM)* in
`results_provenance.csv` was taken from it. It is a binary-linear classifier, not
the 1D-CNN of the paper's evaluation. The paper's circuit-cost figures for the
1D-CNN (2 103 482 constraints, an 85 MB proving key, a modelled 18.4 s prove) are
*modelled* and remain so: they describe a larger circuit that has not been
compiled here, and the two sets of numbers are not interchangeable in either
direction. Read `build/r1cs_info.txt` for what was actually built.

The source corrections of 6 September 2026 preserve retained experiment tables
and reference-build measurements. They bind aggregation inputs to the actual
ordered `(client, model commitment, weight)` sequence and preceding global model,
reject duplicate/stale client rounds before allocating relay nonces, and bind
TrustAnchor acceptance to a registered owner and a sealed round challenge.
The contract API now requires `registerClient`, `beginRound`, `commitUpdate`,
`sealChallenge`, then `submitUpdate(clientId, roundId, proof, publicSignals)`.
When admission includes tolerance, the coordinator calls `setRewardPolicy`
between `beginRound` and the first commitment to fix the higher nominal reward
thresholds. Without that call, nominal and effective thresholds coincide.
The registered coordinator remains the validation-pool custodian. Its sampling
quality and randomness are assumptions; the wrapper enforces commitment order
and instance consistency. The generic relay authenticates a relayer quorum and
binds signatures to the destination chain and bridge address.

The circuit source now checks integer representation ranges, positive predicate
parameters and the presence of both classes. These checks keep the eleven public
signals unchanged but change the constraint system. The retained proving and
verification keys, generated verifier, constraint totals, timings and gas reports
describe the preceding reference build. A remote cryptographic rerun must rebuild
the circuit and its matching keys/verifier before reporting costs for this source.
The contract regression mode below uses retained valid proofs to test the wrapper;
it does not certify that those keys correspond to the revised circuit. Those
proofs use exact sensitivity/specificity thresholds of `70/100`. Separate
in-memory verifier fixtures test the `62/65` admission/reward policy arithmetic
without claiming a Groth16 proof for those instances.

The retained class-aware software configurations use nominal thresholds of
`0.65/0.65` and tolerance `0.03`, hence effective admission thresholds `0.62/0.62`.
On a balanced hundred-row challenge this requires at least `31/50` correct in
each class. Counts `31/50` and `32/50` earn no reward; `33/50` gives a nominal
weaker-class margin of `0.01`. The raw-accuracy configuration similarly uses
nominal `0.75` and effective `0.72`, with rewards clipped at zero below `0.75`.
The software's configurable decision threshold and predicted-positive-rate rule
are not implemented by the fixed zero-logit binary-linear circuit.

`attacks.minmax_update` now solves the distance-constrained MinMax objective
along the negative coordinate-wise standard-deviation direction from
[Shejwalkar and Houmansadr, NDSS 2021, Eq. (6)](https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf).
The previous Python branch used `-2 * mean_update`, a fixed sign/scale surrogate.
Retained results are not rewritten or reassigned to the corrected attack; their
original execution records determine which implementation they used. New remote
MinMax runs must use a separate output directory. ALIE remains a fixed-strength
`mean - 1.5 * std` attack. When several update attacks are named together, the
implemented priority is ALIE, then MinMax, then random-gradient; input/label
poisoning is applied first. Use one attack per configuration to measure each
threat separately.

Preprocessing now fits per-feature mean and standard deviation exclusively on
the training rows selected by each seed. CSV and synthetic loaders apply that
same transform to validation and test features. MIMIC and Challenge2012 cohort
builders retain imputed clinical-unit values and fit no cohort-wide statistics.
Existing cohort and result files are untouched; their historical preprocessing
is not retrospectively relabelled. For schema compatibility, MIMIC's existing
`mortality_48h` column still contains `hospital_expire_flag`, an **in-hospital
mortality** outcome from a 24-hour input window, not a death-within-48-hours label.

The reachability command reports fitted-model diagnostics. Its pooled model is
a reference rather than an attainable-performance ceiling. The empty-round
estimate uses the observed client-by-shared-challenge pass matrix and uniform
sampling without replacement; a failed fit cannot prove impossibility or
permanent training deadlock. The binormal AUC screen requires equal-variance
normal scores. For general monotone ROCs, simultaneous sensitivity/specificity
requirements imply only the necessary AUC floor equal to their product; the
stronger symmetric floor equal to the required rate needs concavity.

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

The official open-access **MIMIC-III Clinical Database Demo v1.4** may also be
used as an end-to-end pipeline check. It is still only a demo (69 eligible ICU
stays in the current extract), not a substitute for the credentialed full
cohort. The helper keeps the Kaggle dataset and kernels private, disables kernel
internet access, and exports aggregate metrics only:

```bash
python3 scripts/kaggle_mimic.py demo \
  --cohort data/processed/mimic_mortality_24h_demo.csv \
  --owner YOUR_KAGGLE_USERNAME --upload --submit
```

Do not use unofficial mirrors of the full database. A 46.6 GB copy of the full
release circulates on a public model hub under a self-applied MIT tag; it is the
credentialed database, mislabelled, and it is not used here.

**The credentialed full cohort is never uploaded to Kaggle, not even privately.**
PhysioNet restricts access to individually credentialed people, forbids sharing,
and its responsible-use guidance rules out sending the data to third-party or
hosted services. Kaggle therefore covers exactly one case in this artifact: the
official open-access Demo above, which is ODbL and not credentialed data.
`scripts/kaggle_mimic.py` enforces that rather than assuming it: both `prepare`
and `submit` refuse any dataset other than the Demo cohort, so no flag exists
that would push the credentialed cohort there. The full run executes on the
machine that holds the download:

```bash
./scripts/physionet_login.sh   # once: stores the credential in ~/.netrc, mode 600
./scripts/run_mimic_full.sh    # download -> cohort -> 20 shards -> paired stats
```

#### User-Agent gating on the download

PhysioNet decides whether to issue the HTTP Basic challenge on `/files/` from the
**User-Agent**. A request identifying itself as `Wget/...` receives
`401 Unauthorized` with `WWW-Authenticate: Basic realm="PhysioNet"`, and then
`200`/`206` once the credential is supplied. The byte-identical request from
curl's default UA, from a browser UA, or with no UA at all receives a flat `403`
with **no** `WWW-Authenticate` header:

```
UA=curl/8.7.1   -> 403      UA=Wget/1.25.0  -> 401, then 200 with the credential
UA=Mozilla/5.0  -> 403      UA=<none>       -> 403
```

That `403` is the same status an unsigned data-use agreement produces, so a client
that does not send a wget User-Agent cannot distinguish "your DUA is not signed"
from "this deployment did not offer you a challenge". Two routes work, and the
artifact ships
both:

```bash
./scripts/wget_mimic.sh          # -r -N -c -np over Basic auth with a wget UA
./scripts/physionet_session.sh   # Django session cookie + CSRF, as a browser does
```

Prefer the cookie route for anything long-lived: it does not rest on a
User-Agent heuristic that the site is free to change. Both are resumable, so an
interrupted `CHARTEVENTS.csv.gz` (4.0 GB gzipped, the largest of the 26 tables and
about two thirds of the 6.17 GiB release) continues rather than restarting, and
the two are interchangeable mid-transfer because both write a plain file that the
other can resume by byte offset.

### Reproducing the reported results

Obtain MIMIC-III v1.4 from PhysioNet and apply the channel mapping and imputation
conventions of Harutyunyan *et al.* to adult ICU stays longer than 48 hours.
The current builder uses the first 24 hours of 17 vitals/labs, hourly resampling
and forward-fill imputation for in-hospital mortality. Write the clinical-unit
flattened cohort to
`data/processed/mimic_mortality_24h.csv` with one row per sample,
`24 x 17 = 408` feature columns, a `mortality_48h` label column, and a
`subject_id` column used to keep a patient inside a single split. The experiment
loader fits z-scoring on the training split only. Use the original execution
records when reproducing retained results; these source corrections do not
establish the preprocessing provenance of those runs.

Two properties of those 17 channels matter when comparing cohorts. CHARTEVENTS is
written in blocks by charting era -- the first ~34 M rows are 100% MetaVision, then
CareVue -- so a coverage check over a prefix of the file under-reports channels
rather than finding a bug; profile itemids per chunk before concluding anything
from a partial download. And capillary refill rate is charted under three CareVue
itemids: 3348 is the neonatal one (440,467 readings in v1.4, and the 3,301 stays
holding the first 257,288 are all NICU, aged 0 or 1, so the adult filter removes
every one), while 115 and 8377 are the adult ones and put the channel on 795
cohort stays across CCU, CSRU, MICU, TSICU and SICU -- about 2.9% of stays, thin
but not empty.

`itemid_to_variable_map.csv` is the selection published with Harutyunyan *et al.*,
itemid for itemid: 114 itemids over the 17 channels, their `STATUS = ready` rows,
no departure in either charting era. MetaVision's 223951/224308 are excluded on
purpose -- they exist in `D_ITEMS` but the benchmark records `COUNT = 0` for both
and v1.4 agrees, so mapping them would widen the map by two itemids and no
readings. `test_itemid_map_matches_the_published_benchmark_selection` freezes all
114, so rebuilding the cohort from the benchmark's own resource file reproduces
this one.

Then run the preflight:

```bash
python3 -m xfedagent preflight --config configs/mimic-pbal.json
```

Preflight prints the aggregate cohort figures and refuses to continue unless the
prevalence is within one point of the configured 14.0% and every `subject_id`
falls inside a single split. A prevalence that has drifted further means a
different cohort, and the run should not start. Then either the single unablated
run, or the paired predicate comparison the paper reports:

```bash
python3 -m xfedagent run --config configs/full.json --output results
./scripts/run_mimic_arms.sh
```

`scripts/run_mimic_arms.sh` runs `configs/mimic-pacc.json` against
`configs/mimic-pbal.json` over seeds 0-9, re-checks the pairing before spending
any compute, merges the two per-arm tables, runs the paired statistics, and
records a `cohort_manifest.json` holding the cohort's SHA-256 rather than the
cohort. Another credentialed reader rebuilds the cohort and compares that digest.

### Rehearsing the pipeline without credentialed access

Two open-access rehearsals exercise the same path end to end while a PhysioNet
Data Use Agreement is pending. Neither is evidence, and no number either one
produces appears in the paper:

```bash
DEMO=1 ./scripts/run_mimic_arms.sh   # official MIMIC-III Demo (ODbL), 69 stays
./scripts/run_c2012_arms.sh          # PhysioNet/CinC Challenge 2012 set A, 3883 stays
```

The Demo is the same database at 100 patients, which is too small to support any
claim. Challenge 2012 is large enough to run the seed-paired comparison, and its
records cover the first 48 h of an adult ICU stay, so the 24-hour window and
24-hour gap fit unchanged and prevalence lands at 13.75% against the MIMIC cohort's
14.0%; but the Challenge publishes 13 of the 17 channels, so capillary refill and
the three Glasgow Coma Scale sub-scores are imputed at their normal values
throughout. `src/xfedagent/challenge2012.py` builds that cohort into the same
schema, reusing the MIMIC windowing, range filter and imputation conventions.
The experiment loader applies the same training-only normalization procedure to
both cohorts. Missing-channel patterns and cohort selection still differ.

---

## 3b. The admission predicate

A raw-accuracy threshold is unsound on an imbalanced validation set. At the
MIMIC-III prevalence of 14.0% positives, a classifier that always predicts the
majority class scores 86.0% and clears `tau=0.75` by 11.0 points while learning
nothing. `pov.predicate` selects the gate:

| Value | Predicate |
|---|---|
| `class_aware` (default) | sensitivity and specificity thresholds enforced separately over integer TP/TN/FP/FN, by cross-multiplication so no division is needed |
| `raw_accuracy` | the original gate, retained to reproduce the earlier configuration |

### Reputation follows the predicate, not raw accuracy

The reputation reward is the nonnegative *predicate margin* above the nominal
thresholds (`metrics.predicate_margin`). Admission tolerance does not lower the
reward thresholds. Under `raw_accuracy` it is `max(0, acc - tau)`. Under
`class_aware` it is `max(0, min(Se - tau_se, Sp - tau_sp))`, so an agent
cannot bank reputation by excelling on the majority class alone. The largest
attainable margin `Delta_max` (`metrics.max_predicate_margin`) is `1 - tau` and
`1 - max(tau_se, tau_sp)` respectively, which is what instantiates the
exclusion threshold `q* = alpha*Delta_max / (beta + alpha*Delta_max)`:
4.8% for the raw gate at 0.75 and 6.5% for the class-aware gate at the calibrated
0.65/0.65. Both are recomputed from the config by
`xfedagent macros --config`, so a threshold change does not leave a stale
percentage behind in the prose.

`q*` and the hitting-time bound that goes with it are properties of this recursion
rather than of any run, so they are checkable without a federated experiment:

```bash
python3 scripts/verify_exclusion_bound.py --config configs/predicate-class-aware.json
```

The script reads `alpha`, `beta`, `r_0`, `r_min` and the predicate's largest
awardable margin from the config, bisects the largest sustainable detected rate,
and replays random reorderings of a fixed detection count on either side of `q*`.
Two results bear on how the bound is quoted. The bisected boundary agrees
with the closed form to 0.004%. And the smooth form `(r_0 - r_min)/drift(s)` is
*violated* at 2,074 of 2,500 rates above `q*`, because a lattice schedule of
rational rate undershoots its own average between lattice points; the form that
holds unconditionally carries two extra constants,
`ceil((r_max - r_min + beta + alpha*Delta_max) / drift(s))`. Do not quote the
smooth form without its `for all T >= T_0` hypothesis.

### Measuring the two predicates against each other

`configs/predicate-raw-accuracy.json` and `configs/predicate-class-aware.json`
are identical in seed, partition, model, attack schedule and round count, and
differ only in the predicate. Both run 60 rounds at prevalence 0.137 against a
`majority_class` adversary:

```bash
python3 -m xfedagent run --config configs/predicate-raw-accuracy.json --output results/pacc
python3 -m xfedagent run --config configs/predicate-class-aware.json  --output results/pbal
```

The table below reports the **ten paired surrogate seeds** after calibration
selected 0.65/0.65 for the class-aware arm. Pairing holds the partition,
initialisation, model and attack schedule fixed within each seed. Use
`scripts/run_predicate_arms.sh` to reproduce the comparison; it reports
per-metric differences with Holm-corrected Wilcoxon tests and paired effect sizes.

| | raw accuracy, tau=0.75 | class-aware 0.65/0.65, balanced |
|---|---|---|
| Honest submissions rejected | 0.3% | 46.2% |
| Global accuracy | 88.8% | 88.9% |
| Global sensitivity | 24.2% | **84.4%** |
| Global AUC-ROC | 0.940 | **0.951** |

`rounds.csv` carries `honest_false_reject_rate` and `malicious_admission_rate`
per round, which is what these aggregates are computed from. There is no warm-up
relaxation in the code: thresholds are fixed from round 0. The 46.2% honest
false-rejection cost is reported as a limitation rather than hidden.

`pov.balanced_validation` draws the per-round subset with equal class counts.
Class-wise concentration depends on each class's sample count; at prevalence
0.137, an unstratified hundred-row draw contains only about 14 positives on
average. Any admission bound must use the effective thresholds after tolerance
and explicitly state assumptions about the committed model and validation pool.
Fresh challenges do not by themselves make adaptive submissions independent.
A multi-round bound needs a conditional admission bound at every round, and
consecutive successful admissions are distinct from reputation survival.

`data.positive_rate` sets the surrogate's prevalence, so the imbalance can be reproduced
without the restricted dataset:

```bash
python3 -m xfedagent run --config configs/synthetic.json --output results
```

## 3c. Choosing the thresholds

The pair `(tau_se, tau_sp)` is selected on **calibration data** — the gate's own
decisions over the rotating validation pool — and never on the test set. The rule
is fixed before the sweep runs: among pairs holding malicious admission at or
below 5%, take the one with the lowest honest false-rejection rate. Calibration is
triggered only when the operating point exceeds a 15% honest false-rejection rate.

```bash
python3 -m xfedagent calibrate --config configs/predicate-class-aware.json \
  --pairs 0.65,0.70,0.75 --seeds 0-2 --observed-false-reject 0.2810 \
  --output results/surrogate-calibration --latex
```

A bare value in `--pairs` is a symmetric pair; `0.65:0.70` sets the two thresholds
independently. `--observed-false-reject` is the rate already seen at the operating
point, and the sweep refuses to run unless it exceeds the 15% trigger, so the
condition is enforced by the tool rather than remembered.

`calibration_manifest.json` records the rule, the grid, the chosen pair, and
`test_set_consulted: false`. On the surrogate cohort the operating point 0.70/0.70
rejected 50.5% of honest submissions, above the trigger, and the rule selected
**0.65/0.65** at 46.3%. Every pair on the grid held malicious admission at 0.0%.

Two consequences are worth stating rather than burying, because the rule
optimises honest false-rejection alone and does not see them:

- The chosen pair sits at the **lower boundary of the pre-registered grid**. The
  grid is not widened after seeing the result; that would make the selection
  data-dependent in exactly the way pre-registration exists to prevent.
- Lowering the effective thresholds enlarges the admission region. Admission
  bounds must therefore incorporate the declared tolerance and their sampling
  assumptions. The retained `RandomAdmitBound` and `RandomAdmitRounds` macros
  were derived from exact nominal thresholds and independent repeated events;
  they do not certify the current software policy. The corrected generator uses
  the effective thresholds and a single-tail bound conditional on the stated
  class counts and independent, input-independent Bernoulli predictions. It no
  longer emits an automatic multi-round horizon. The nominal reward margin
  still determines `q*`, which increases from 5.7% to 6.5% when nominal targets
  decrease from 0.70 to 0.65.

A 4.2-point reduction in honest false-rejection for a 5.8x weaker per-round bound
is a poor exchange on its face. The pre-registered rule selected it, so it stands
and is reported; whether the rule should have carried the soundness bound as a
second criterion is a design question, not something to settle by re-running the
selection until it agrees.

## 4. Component ablation

The reported surrogate sweep uses `configs/predicate-class-aware.json`;
`configs/full.json` points at the credentialed MIMIC-III cohort and cannot run
without it. Every config carries an `ablation` block whose switches default to `true`,
so an omitted block reproduces the complete framework. The sweep re-runs one
base configuration once per (arm, seed) pair, holding the data partition, model,
attack schedule, and round count fixed, so the gap between two arms isolates the
components that separate them:

```bash
python3 -m xfedagent ablation \
  --config configs/predicate-class-aware.json \
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
python3 -m xfedagent ablation --config configs/predicate-class-aware.json \
  --arms full,no-pov,no-rep,no-pov+rep --seeds 0-9 \
  --output results/ablation --factorial \
  --factorial-metric balanced_accuracy_mean
```

The **main effect** of a factor is its average simple effect across the levels of
the other factor; the **interaction** is the difference between the two simple
effects. A negative interaction means the mechanisms are partially redundant, so
adding the second recovers less than it would alone. Report the main effects and
the interaction rather than summing the two simple effects, which overstates the
combined benefit. Balanced accuracy is the default and primary outcome. On the
ten-seed surrogate ablation, the effects are `+22.0` pp for the gate, `+4.1` pp
for reputation, and `-6.4` pp for the interaction. Sensitivity is secondary
(`+52.1`, `+9.9`, `-11.5` pp); raw accuracy is retained as a diagnostic and gives
only `+0.3`, `+0.0`, `-2.7` pp because the majority class dominates it. These are
surrogate effects and are not attributed to MIMIC-III.

> **Configure the sweep so honest agents can clear the gate.** The PoV threshold
> (`pov.threshold`, default 0.75) rejects *every* update — honest and malicious
> alike — until local training is good enough to exceed it. A deliberately short
> run (few rounds, one local epoch) therefore reports `full` as *worse* than
> `no-pov`, which is an artefact of the truncated schedule rather than a finding.
> Use the full 100-round, `E=5` schedule for any number that is reported.

### Gate-pass is not aggregation, and an on/off table hides the difference

`ablation_summary.csv` reports `malicious_rejection_rate` per arm, which counts
gate decisions. An update enters the average only if it clears the gate **and**
its agent still holds `reputation >= reputation_min`, so the two are not the same
number and the gap is where the mechanisms compose. `scripts/run_freerider_arms.sh`
measures it for the replay free-rider:

```bash
bash scripts/run_freerider_arms.sh          # full vs no-copy, 10 paired seeds
```

Without the copy detector, 1,770 of 1,800 replayed models clear the gate (98.3%,
not 100% — the 30 refusals are the round-0 replays of the *untrained*
initialisation, which the utility predicate rejects unaided). With it, none do.
But in the shipped configuration `beta` equals `reputation_initial`, so that
round-0 refusal zeroes the free-rider and it cannot re-enter the average for
roughly `beta / (alpha * Delta_max)` further rounds. The consequence is checkable
without trusting any summary statistic: the two arms' `global_model_root` values
in `rounds.csv` are **bit-identical for the first 9-13 rounds** depending on seed.
For that horizon, reputation alone reproduces the fully protected system, so a
factorial table crediting the copy detector with the whole free-rider defence
overstates it early and understates it later.

The gadget's own cost is accuracy, not false rejections: excluding replays moves
raw accuracy by -1.2 pp (Holm p = 0.0098), because a replayed `theta^(t)` inside a
weighted mean is a shrinkage term toward the previous iterate. Balanced accuracy
does not move. A soundness mechanism that improved every metric would be the
suspicious result.

### Does a metric trend with agent count?

`scripts/run_scale_arms.sh` sweeps 10/25/50/100 agents and then runs an
ordered-alternative (Jonckheere-Terpstra) test, because four means with five seeds
each look like whatever the reader expects:

```bash
SEEDS=0-4 bash scripts/run_scale_arms.sh     # sweep + trend test + merged runs csv
python3 scripts/scale_trend.py --runs 10:results/scale-scaled/n10/ablation_runs.csv ...
```

On the `scaled` family, balanced accuracy and AUC-ROC rise with population
(p = 0.019 and p = 0.007) while raw accuracy shows no trend (p = 0.79) and
malicious rejection *falls* (p = 0.005). The last one is the direction to expect
on reflection: the gate is a per-update predicate on a rotating validation subset,
so its power does not grow with the number of submissions, while the opportunities
to land inside `pov.tolerance` do.

---

## 5. Configuration reference

Everything needed to reproduce a run lives in one JSON file; `config.json` is
copied verbatim into each run directory, and `summary.json` records a
`config_digest` over it.

| Section | Reported setting |
|---|---|
| `data` | 26 424 samples, 10 clients, 24 timesteps, 17 features, validation pool 1 000, Dirichlet α = 0.5, 70/15/15 split |
| `model` | 1D-CNN, channels 8/16/32, Adam, lr 1e-3, weight decay 1e-4, batch 32, `E = 5` local epochs, 8-bit quantisation, copy ε = 1e-4 |
| `pov` | class-aware predicate, τ_sens = τ_spec = 0.65 (calibrated, §3c), `|D_val|` = 100 drawn class-balanced, tolerance ε = 0.03, per-round rotation, 0.184 s/sample modelled proving |
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
| `results_macros.tex` | Every number the manuscript states more than once, as `\newcommand` definitions grouped by provenance |
| `results_provenance.csv` | One row per macro: value, provenance label, and the artefact it was read from |

## 7. Repository layout

```
src/xfedagent/    Federation runner, PoV backend, commitments, relay, energy, ablation,
                  calibration, paired statistics, results_macros generator,
                  mimic.py (reported cohort) and challenge2012.py (rehearsal cohort)
circuits/         Circom PoV circuit (binary_linear_pov.circom)
contracts/        TrustAnchor, RelayBridge and PoVVerifierAdapter Solidity contracts
configs/          full.json (reported study), synthetic.json, synthetic-smoke.json,
                  predicate-raw-accuracy.json / predicate-class-aware.json (the two gates)
                  predicate-class-aware-replay.json (the same gate under replay attack)
                  mimic-pacc.json / mimic-pbal.json (the same two gates, MIMIC cohort)
                  c2012-pacc.json / c2012-pbal.json (open-access rehearsal, not reported)
                  scale-n{10,25,50,100}-{scaled,fixed}.json (agent-count sweep)
scripts/          compile_circuit.sh, run_circuit_tests.sh, prove_gas_samples.sh,
                  measure_gas.js, run_predicate_arms.sh, run_mimic_arms.sh,
                  run_c2012_arms.sh, run_freerider_arms.sh, run_scale_arms.sh,
                  verify_exclusion_bound.py, scale_trend.py
                  physionet_login.sh (--check re-tests access), physionet_session.sh,
                  fetch_mimic.sh, wget_mimic.sh (recursive mirror), run_mimic_full.sh
tests/            Unit and pipeline tests
tests/circuit/    Witness generators and the T0-T8 expected-outcome suite
build/            Compiled circuit, keys, per-test logs, measured metrics (git-ignored
                  except verification_key.json and r1cs_info.txt)
```

## 8. Tests

```bash
PYTHONPATH=src python3 -m pytest -q          # Python: runner, gate, statistics, macros
bash scripts/run_circuit_tests.sh            # Circom: T0-T8 expected-outcome suite
POV_CONTRACT_TEST_ONLY=1 npx hardhat run --no-compile scripts/measure_gas.js
```

The contract-only command compiles Solidity in memory and checks ownership,
commit/challenge ordering, public-instance substitution, class-aware reward and
relay replay rejection without writing measurement files. It uses retained real
proofs for the reference policy and a verifier fixture for isolated tolerance
and nominal-reward arithmetic. The Python circuit
regression builds WASM in a temporary test directory and checks thirteen witness
outcomes without running setup or proof generation. Neither command is a full
experiment.

The circuit suite asserts *outcomes*, not just exit codes: T0, T3 and T5 must
produce a witness and T1, T2, T4, T6, T7, T8 must fail, each for the stated
reason. `build/test_report.csv` records expected against actual for every case,
and a case that passes when it should fail is reported as `MISMATCH` rather than
being silently counted as a pass.

## 9. Data and privacy

No patient-level data, no credentialed-access extract, and no derived record
that could identify an individual is contained in this repository. MIMIC-III
must be obtained directly from PhysioNet under its own data-use agreement. The
synthetic surrogate in `data/synthetic` is generated from a parametric model and
contains no real observations.

## 10. Licence

Released under the MIT Licence; see `LICENSE`.
