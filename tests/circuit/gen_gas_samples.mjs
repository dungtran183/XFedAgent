// Extra admissible witnesses, used only to measure on-chain verification cost.
//
// Task 1.5 asks for gasUsed over at least five verify transactions. The T0-T8
// suite contains exactly three admissible cases, because a rejected witness has
// no proof to submit, so three more passing witnesses are built here. They are
// kept out of tests/circuit/cases/ on purpose: the test matrix is T0-T8 and its
// report must not grow rows that test nothing.
//
// Each sample satisfies the class-aware predicate with a different confusion
// quadruple, so the measured spread reflects distinct public inputs rather than
// the same proof submitted repeatedly.
import fs from "node:fs";
import path from "node:path";

import { assemble, init, DENOM, TAU_SE, TAU_SP } from "./gen_inputs.mjs";

const OUT = path.join(import.meta.dirname, "gas_cases");

// Sensitivity = tp/50, specificity = tn/50 on the balanced subset; both must
// clear 0.70, i.e. tp >= 35 and tn >= 35.
const SAMPLES = [
  { id: "G1", tp: 50, tn: 50, why: "perfect classifier: every public count at its maximum" },
  { id: "G2", tp: 36, tn: 48, why: "sensitivity just above threshold, high specificity" },
  { id: "G3", tp: 48, tn: 36, why: "specificity just above threshold, high sensitivity" },
];

async function main() {
  await init();
  fs.mkdirSync(OUT, { recursive: true });
  const manifest = [];
  for (const s of SAMPLES) {
    const sens = (s.tp * Number(DENOM)) / 50;
    const spec = (s.tn * Number(DENOM)) / 50;
    if (sens < Number(TAU_SE) || spec < Number(TAU_SP)) {
      throw new Error(`${s.id} would not be admitted: sens=${sens} spec=${spec}`);
    }
    const dir = path.join(OUT, s.id);
    fs.mkdirSync(dir, { recursive: true });
    const input = assemble({ tp: s.tp, tn: s.tn });
    fs.writeFileSync(path.join(dir, "input.json"), JSON.stringify(input, null, 1));
    manifest.push({ id: s.id, expected: "pass", scenario: s.why });
    console.log(`${s.id} TP=${input.tp} TN=${input.tn} FP=${input.fp} FN=${input.fn}  ${s.why}`);
  }
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`\n${SAMPLES.length} gas samples written to ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
