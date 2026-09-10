// Witness generators for the PoV circuit test suite (T0-T8 of the revision plan).
//
// Every case is built from one deterministic reference model over a balanced
// 100-sample validation subset (n_+ = n_- = 50), so the cases differ only in the
// property under test. Field elements are reduced mod p here, because the circuit
// absorbs the same reduced values into its Poseidon sponges.
import { buildPoseidon } from "circomlibjs";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const SAMPLES = 100;
const FEATURES = 17;
const RATE = 15;
const DENOM = 100n;      // public denominator d
const TAU_SE = 70n;      // tau_se = 0.70 -> numerator over d
const TAU_SP = 70n;      // tau_sp = 0.70
const EPS_SQ = 1n;       // eps_copy^2 in quantised units: one unit of difference

const OUT = path.join(import.meta.dirname, "cases");
const mod = (v) => ((v % P) + P) % P;

let poseidon;
let F;

// Rate-15 Merkle-Damgard sponge, identical to PoseidonSponge in the circuit.
function sponge(iv, inputs) {
  let state = BigInt(iv);
  const chunks = Math.ceil(inputs.length / RATE);
  for (let k = 0; k < chunks; k++) {
    const block = [state];
    for (let j = 0; j < RATE; j++) {
      const idx = k * RATE + j;
      block.push(idx < inputs.length ? mod(inputs[idx]) : 0n);
    }
    state = F.toObject(poseidon(block));
  }
  return state;
}

// Deterministic LCG, so the suite is reproducible without a seeded RNG dependency.
function lcg(seed) {
  let s = BigInt(seed);
  return () => {
    s = (s * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return Number((s >> 33n) & 0x7fffffffn);
  };
}

// Reference quantised linear model. Feature 0 carries a large weight so the sign
// of the logit — and hence the prediction — is set by construction, letting each
// case place an exact confusion quadruple.
function buildModel() {
  const rand = lcg(20260823);
  const w = [100n];
  for (let j = 1; j < FEATURES; j++) w.push(BigInt((rand() % 7) - 3));
  return { w, bias: 0n };
}

// Features for a subset whose first `positives` samples predict 1, rest predict 0.
function buildFeatures(positives) {
  const rand = lcg(777);
  const x = [];
  for (let i = 0; i < SAMPLES; i++) {
    const row = [i < positives ? 5n : -5n];
    for (let j = 1; j < FEATURES; j++) row.push(BigInt((rand() % 9) - 4));
    x.push(row);
  }
  return x;
}

function logitOf(x, w, bias) {
  let dot = bias;
  for (let j = 0; j < FEATURES; j++) dot += x[j] * w[j];
  return dot;
}

// Labels realising an exact (TP, TN) on a balanced subset: FN = 50 - TP,
// FP = 50 - TN, and the predicted-positive count is TP + FP.
function buildCase({ tp, tn }) {
  const fn = 50 - tp;
  const fp = 50 - tn;
  const predPositives = tp + fp;
  const model = buildModel();
  const x = buildFeatures(predPositives);

  const y = new Array(SAMPLES).fill(0n);
  for (let i = 0; i < predPositives; i++) y[i] = i < tp ? 1n : 0n;          // TP then FP
  for (let i = predPositives; i < SAMPLES; i++) {
    y[i] = i - predPositives < fn ? 1n : 0n;                                // FN then TN
  }

  // The construction is only valid if the intended prediction split holds.
  const counts = { tp: 0, tn: 0, fp: 0, fn: 0 };
  for (let i = 0; i < SAMPLES; i++) {
    const pred = logitOf(x[i], model.w, model.bias) >= 0n ? 1n : 0n;
    if (pred === 1n && y[i] === 1n) counts.tp++;
    else if (pred === 0n && y[i] === 0n) counts.tn++;
    else if (pred === 1n && y[i] === 0n) counts.fp++;
    else counts.fn++;
  }
  if (counts.tp !== tp || counts.tn !== tn || counts.fp !== fp || counts.fn !== fn) {
    throw new Error(`construction mismatch: ${JSON.stringify(counts)} != ${JSON.stringify({ tp, tn, fp, fn })}`);
  }
  return { model, x, y, counts: { tp, tn, fp, fn } };
}

export { P, SAMPLES, FEATURES, DENOM, TAU_SE, TAU_SP, EPS_SQ, OUT, mod, sponge, buildCase, buildModel };

// The Poseidon instance is module state, because `sponge` is shared with the
// gas-sample generator. A caller that imports this module rather than running it
// must call `init` first, which is what `main` does below.
async function init() {
  if (!poseidon) {
    poseidon = await buildPoseidon();
    F = poseidon.F;
  }
}

// Assemble one input.json. `tamper` mutates the public instance after the honest
// witness is built, which is how T7 and T8 model a post-hoc edit of a public value.
function assemble({ tp, tn, replay = false, tamper = null }) {
  const { model, x, y, counts } = buildCase({ tp, tn });
  const salt = 12345n;
  const globalSalt = 67890n;

  // theta^(t): the current global model. Under replay the submitted model is
  // bit-identical to it, so the copy detector must reject.
  const globalW = replay ? [...model.w] : model.w.map((v, j) => (j === 0 ? v - 7n : v + 1n));
  const globalBias = replay ? model.bias : model.bias + 3n;

  const modelRoot = sponge(1n, [...model.w, model.bias, salt]);
  const globalRoot = sponge(2n, [...globalW, globalBias, globalSalt]);

  const flat = [];
  for (let i = 0; i < SAMPLES; i++) {
    for (let j = 0; j < FEATURES; j++) flat.push(x[i][j]);
    flat.push(y[i]);
  }
  const validationRoot = sponge(3n, flat);

  const input = {
    validationRoot: validationRoot.toString(),
    modelRoot: modelRoot.toString(),
    globalRoot: globalRoot.toString(),
    tp: String(counts.tp),
    tn: String(counts.tn),
    fp: String(counts.fp),
    fn: String(counts.fn),
    tauSensNum: TAU_SE.toString(),
    tauSpecNum: TAU_SP.toString(),
    denom: DENOM.toString(),
    copyEpsilonSq: EPS_SQ.toString(),
    x: x.map((row) => row.map((v) => mod(v).toString())),
    y: y.map((v) => v.toString()),
    w: model.w.map((v) => mod(v).toString()),
    bias: mod(model.bias).toString(),
    salt: salt.toString(),
    globalW: globalW.map((v) => mod(v).toString()),
    globalBias: mod(globalBias).toString(),
    globalSalt: globalSalt.toString(),
  };
  return tamper ? tamper(input) : input;
}

export { assemble, init };

// Additional arithmetic checks are built in memory by the regression test. They
// do not overwrite the retained T0-T8 inputs or their historical proof artifacts.
export function arithmeticCases() {
  const commit = (input) => {
    input.modelRoot = sponge(1n, [...input.w, input.bias, input.salt].map(BigInt)).toString();
    input.globalRoot = sponge(2n, [...input.globalW, input.globalBias, input.globalSalt].map(BigInt)).toString();
    input.validationRoot = sponge(3n, input.x.flatMap((row, i) => [...row, input.y[i]]).map(BigInt)).toString();
    return input;
  };
  const edit = (change) => { const input = assemble({ tp: 40, tn: 40 }); change(input); return commit(input); };
  return [
    { name: "valid", accepted: true, input: assemble({ tp: 40, tn: 40 }) },
    { name: "inclusive-boundary", accepted: true, input: assemble({ tp: 35, tn: 40 }) },
    { name: "zero-denominator", accepted: false, input: edit((i) => { i.denom = "0"; i.tauSensNum = "0"; i.tauSpecNum = "0"; }) },
    { name: "oversized-denominator", accepted: false, input: edit((i) => { i.denom = (1n << 20n).toString(); }) },
    { name: "zero-threshold", accepted: false, input: edit((i) => { i.tauSensNum = "0"; }) },
    { name: "threshold-above-one", accepted: false, input: edit((i) => { i.tauSensNum = "101"; }) },
    { name: "zero-copy-bound", accepted: false, input: edit((i) => { i.copyEpsilonSq = "0"; }) },
    { name: "missing-positive-class", accepted: false, input: edit((i) => {
      i.x.forEach((row) => { row[0] = mod(-5n).toString(); }); i.y.fill("0");
      i.tp = "0"; i.tn = "100"; i.fp = "0"; i.fn = "0";
    }) },
    { name: "missing-negative-class", accepted: false, input: edit((i) => {
      i.x.forEach((row) => { row[0] = "5"; }); i.y.fill("1");
      i.tp = "100"; i.tn = "0"; i.fp = "0"; i.fn = "0";
    }) },
    { name: "global-weight-range", accepted: false, input: edit((i) => { i.globalW[1] = "128"; }) },
    { name: "submitted-weight-range", accepted: false, input: edit((i) => {
      i.x.forEach((row) => { row[1] = "0"; }); i.w[1] = "128";
    }) },
    { name: "feature-range", accepted: false, input: edit((i) => { i.w[1] = "0"; i.x[0][1] = "32768"; }) },
    { name: "global-bias-range", accepted: false, input: edit((i) => { i.globalBias = (1n << 31n).toString(); }) },
  ];
}

// T0-T8 of the revision plan. `expect` is the required outcome of witness
// generation plus proof verification: a violated constraint makes the witness
// unsatisfiable, which is the circuit rejecting the submission.
const CASES = [
  { id: "T0", expect: "pass", why: "honest witness, all four conditions hold",
    build: () => assemble({ tp: 40, tn: 40 }) },

  { id: "T1", expect: "fail", why: "counts do not sum to n",
    build: () => assemble({ tp: 40, tn: 40, tamper: (i) => ({ ...i, fp: String(Number(i.fp) + 1) }) }) },

  { id: "T2", expect: "fail", why: "sensitivity one sample below threshold",
    build: () => assemble({ tp: 34, tn: 40 }) },

  { id: "T3", expect: "pass", why: "sensitivity exactly at threshold (inclusive bound)",
    build: () => assemble({ tp: 35, tn: 40 }) },

  { id: "T4", expect: "fail", why: "specificity one sample below threshold",
    build: () => assemble({ tp: 40, tn: 34 }) },

  { id: "T5", expect: "pass", why: "specificity exactly at threshold (inclusive bound)",
    build: () => assemble({ tp: 40, tn: 35 }) },

  { id: "T6", expect: "fail", why: "replay: theta_hat equals theta^(t)",
    build: () => assemble({ tp: 40, tn: 40, replay: true }) },

  { id: "T7", expect: "fail", why: "H_glob taken from a stale round",
    build: () => assemble({ tp: 40, tn: 40, tamper: (i) => ({ ...i, globalRoot: mod(BigInt(i.globalRoot) + 1n).toString() }) }) },

  { id: "T8", expect: "fail", why: "C_D edited after the proof was formed",
    build: () => assemble({ tp: 40, tn: 40, tamper: (i) => ({ ...i, validationRoot: mod(BigInt(i.validationRoot) + 1n).toString() }) }) },

  { id: "T8b", expect: "fail", why: "H_theta edited after the proof was formed",
    build: () => assemble({ tp: 40, tn: 40, tamper: (i) => ({ ...i, modelRoot: mod(BigInt(i.modelRoot) + 1n).toString() }) }) },
];

async function main() {
  await init();
  fs.mkdirSync(OUT, { recursive: true });
  const manifest = [];
  for (const c of CASES) {
    const dir = path.join(OUT, c.id);
    fs.mkdirSync(dir, { recursive: true });
    const input = c.build();
    fs.writeFileSync(path.join(dir, "input.json"), JSON.stringify(input, null, 1));
    manifest.push({ id: c.id, expected: c.expect, scenario: c.why });
    console.log(`${c.id.padEnd(4)} ${c.expect.padEnd(5)} TP=${input.tp} TN=${input.tn} FP=${input.fp} FN=${input.fn}  ${c.why}`);
  }
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\n${CASES.length} cases written to ${OUT}`);
}

// Regenerating the suite is a side effect, so it only happens when this file is
// the process entry point. Importing it (the gas-sample generator does) must not
// rewrite the cases the test report was produced from.
const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === url.fileURLToPath(import.meta.url);

if (invokedDirectly) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
