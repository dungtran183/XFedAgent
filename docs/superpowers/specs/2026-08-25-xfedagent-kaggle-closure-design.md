# XFedAgent Scientific Closure and Kaggle Execution Design

**Date:** 2026-08-25  
**Scope:** Close every actionable item in `TODO.tex` while keeping the full
MIMIC-III experiment off the local Mac.  The local machine prepares and tests
code; Kaggle performs training once PhysioNet access and the MIMIC-III DUA are
active.

## 1. Scientific decisions

The implementation will use the conservative, provenance-preserving choices
approved as Option A.

1. Keep the calibrated class-aware operating point at
   `tau_sensitivity = tau_specificity = 0.65`.  This is the output of the
   predefined calibration rule.  The paper will disclose that 0.65 is the
   lower edge of the tested grid and that the random-admission bound weakens
   relative to 0.70.
2. Use balanced accuracy as the primary two-factor ablation outcome because
   the cohort is imbalanced and the predicate explicitly acts on sensitivity
   and specificity.  Report sensitivity as a secondary outcome.  Raw accuracy
   remains in the table but no longer supports the main-effect claim.
3. Do not replace the modelled 1D-CNN circuit estimates with measurements from
   a different circuit.  Present two explicitly separate rows: modelled
   1D-CNN and measured binary-linear artifact.  The text must state that they
   are not interchangeable.
4. Incorporate the completed six-attack `p_det` results, including pooled
   Wilson intervals and the negative result for norm-matched random-gradient
   and MinMax attacks.

## 2. Current external blocker

PhysioNet currently reports:

- credentialing: awaiting review;
- CITI Data or Specimens Only Research: under review;
- DUA: not yet available/signed;
- local data: MIMIC-III demo only, not the full v1.4 database.

Therefore the full MIMIC run is a gated phase.  No text will describe the full
MIMIC experiment as completed until the account shows active credentialing and
training, the DUA is signed, and the full data are available to the job.

## 3. Execution boundary

### Local Mac

Allowed work:

- edit code, configs, manuscript, appendix, response, and `TODO.tex`;
- run unit/regression tests and small synthetic/demo smoke tests;
- generate Kaggle job bundles and inspect their metadata;
- submit, monitor, and download aggregate Kaggle outputs;
- compile LaTeX.

Disallowed work:

- full MIMIC-III training or the 20-run paired experiment;
- committing or packaging raw/processed patient-level data;
- printing Kaggle credentials, PhysioNet credentials, or patient identifiers.

### Kaggle

Kaggle performs preprocessing/training and emits aggregate experiment
artifacts.  Every kernel must be private.  Internet access is disabled for the
training kernel unless a dependency cannot be bundled; any exception requires
an explicit review.  GPU acceleration is enabled when available, although the
pipeline must remain device-agnostic.

## 4. Data path and security

The preferred path minimizes copies of credentialed data:

1. After PhysioNet approval and DUA activation, obtain MIMIC-III v1.4 through
   the user's authorized account.
2. Create a **private** Kaggle input dataset owned only by the user.  Prefer a
   processed cohort containing only the 17 required variables, labels, and a
   split key over uploading all raw MIMIC tables.  The processed cohort remains
   restricted data and must never be made public or shared.
3. Store the Kaggle dataset slug in an ignored local runtime file or command
   argument, not in a public manuscript artifact.
4. Keep the kernel private and disable internet.  Do not include the Kaggle API
   token in kernel files, environment dumps, notebook cells, logs, or Git.
5. Use `subject_id` only to enforce patient-disjoint splits.  Remove it before
   constructing model tensors.  Kaggle outputs may contain seed, arm, round,
   aggregate metrics, counts, and provenance, but no patient-level rows or
   identifiers.
6. Download only aggregate CSV/JSON outputs into
   `results/mimic-kaggle/`.  Raw data and Kaggle caches remain ignored.

The PhysioNet DUA requires reasonable electronic security and prohibits
sharing access.  A private Kaggle dataset does not remove those obligations.
Before the first full upload, the user must confirm that this storage choice is
acceptable under their institution's data-governance rules.  If not, use the
official PhysioNet-linked GCP/AWS route instead of Kaggle.

## 5. Kaggle job layout

The implementation will add a small, reviewable Kaggle surface:

- `kaggle/xfedagent-mimic/kernel-metadata.json`: private script kernel;
- `kaggle/xfedagent-mimic/run.py`: validates inputs and invokes the existing
  XFedAgent CLI without duplicating experiment logic;
- `kaggle/xfedagent-mimic/README.md`: setup, private-data, submission, status,
  and output instructions;
- `scripts/kaggle_mimic.py`: local orchestrator for bundle generation,
  submission, status, logs, and output download.

The full experiment is split by arm and seed: `pacc` and `pbal`, seeds 0--9.
Each shard has one resolved config and one output directory.  This prevents a
single Kaggle runtime limit from invalidating all 20 runs and allows failed
seeds to be rerun without overwriting completed seeds.

The orchestrator will support these explicit phases:

1. `prepare`: create deterministic per-arm/per-seed bundles;
2. `smoke`: run one demo/synthetic seed;
3. `submit`: push only requested shards;
4. `status`: report queued/running/complete/error without exposing secrets;
5. `download`: retrieve completed aggregate outputs;
6. `merge`: verify that all expected arm/seed shards exist, then build the
   paired statistics and manuscript macros.

No incomplete or failed shard enters the merged result.

## 6. Manuscript and artifact updates before full MIMIC approval

The first implementation phase can close all non-MIMIC work:

- regenerate the complete macro/provenance set from circuit, gas, calibration,
  surrogate paired comparisons, ablation, and `p_det` manifests;
- replace repeated hard-coded values with macros;
- update the ablation claim to balanced accuracy with sensitivity secondary;
- add the measured binary-linear circuit row beside the modelled 1D-CNN row;
- report all six `p_det` estimates and limitations;
- preserve the MIMIC scoping paragraph and label surrogate results as
  surrogate;
- update `TODO.tex` so `p_det` is complete and PhysioNet is accurately marked
  as awaiting approval;
- do not tag/release the artifact until the manuscript and tests are clean.

After the full Kaggle outputs are available, regenerate the same macros with
`--cohort mimic`, replace only the explicitly scoped surrogate claims, and then
re-evaluate whether the MIMIC scoping paragraph can be removed.

## 7. Validation and failure handling

Validation proceeds in increasing cost order:

1. unit tests for metrics, MIMIC preprocessing, public-signal order,
   calibration, statistics, macros, and Kaggle bundle validation;
2. local synthetic/demo smoke test only;
3. private Kaggle smoke kernel;
4. full Kaggle shards after access/data approval;
5. merge-time checks for exactly two arms, seeds 0--9, paired configs, expected
   round count, finite metrics, and patient-disjoint split evidence;
6. LaTeX compilation and scans for undefined references/citations, overfull
   boxes, stale hard-coded values, and provenance mismatches.

Failures are isolated per shard.  Logs may contain stack traces and aggregate
counts but never credentials or patient rows.  A non-finite model output is
scored consistently by the existing metrics/confusion-count path and recorded
as a gate failure rather than crashing the entire measurement.

## 8. Deliverables for the supervisor

The handoff will include:

- updated source files and generated aggregate result tables;
- a concise Vietnamese status report with: completed work, final numbers,
  scientific decisions and rationale, current PhysioNet blocker, remaining
  Kaggle step, and risks/limitations;
- exact reproduction commands for local tests and Kaggle execution;
- a checklist mapping every original TODO item to evidence or an external
  blocker.

The report will not claim that a pending PhysioNet review or an unrun Kaggle
experiment is complete.
