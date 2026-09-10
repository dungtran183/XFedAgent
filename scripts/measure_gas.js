// Task 1.5: measure gasUsed for on-chain PoV verification.
//
// Deploys the exported Groth16 verifier, the adapter, and TrustAnchor on a local
// Hardhat EVM instance, then submits the proofs produced by the circuit test
// suite. Gas is reported per transaction so the paper can quote a measured
// distribution rather than a single modelled constant.
const fs = require("node:fs");
const path = require("node:path");

const BUILD = process.env.BUILD_DIR || "build";
// The admissible cases: a rejected witness has no proof, so only these can be
// submitted. T0/T3/T5 come from the T0-T8 suite; G1-G3 are the extra admissible
// witnesses of tests/circuit/gas_cases, present so the verify sample is at least
// the five transactions Task 1.5 asks for.
const CASES = ["T0", "T3", "T5", "G1", "G2", "G3"];
const MIN_VERIFY_SAMPLES = 5;

async function contractChecks() {
  // Compile in memory: these regression checks neither rewrite gas reports nor
  // regenerate Groth16 keys/proofs. The retained T0/T3 proofs exercise the wrapper.
  const assert = require("node:assert/strict");
  const solc = require("solc");
  const { ethers } = require("hardhat");
  const sources = Object.fromEntries(fs.readdirSync("contracts").filter((f) => f.endsWith(".sol"))
    .map((f) => [f, { content: fs.readFileSync(path.join("contracts", f), "utf8") }]));
  // The retained proofs use exact 70/100 thresholds. A verifier fixture isolates
  // round-policy arithmetic for 62/65 without fabricating a new Groth16 proof.
  sources["PolicyVerifierFixture.sol"] = { content: `
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.24;
    contract PolicyVerifierFixture {
      function verifyProof(bytes calldata, uint256[] calldata) external pure returns (bool) {
        return true;
      }
    }` };
  const compiled = JSON.parse(solc.compile(JSON.stringify({
    language: "Solidity", sources,
    settings: { optimizer: { enabled: true, runs: 200 }, viaIR: true,
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } } },
  })));
  const errors = (compiled.errors || []).filter((e) => e.severity === "error");
  assert.equal(errors.length, 0, errors.map((e) => e.formattedMessage).join("\n"));
  const [owner, stranger, third] = await ethers.getSigners();
  const deploy = async (file, name, args = []) => {
    const c = compiled.contracts[file][name];
    const result = await new ethers.ContractFactory(c.abi, c.evm.bytecode.object, owner).deploy(...args);
    await result.waitForDeployment();
    return result;
  };
  const groth16 = await deploy("PoVVerifier.sol", "Groth16Verifier");
  const adapter = await deploy("PoVVerifierAdapter.sol", "PoVVerifierAdapter", [await groth16.getAddress()]);
  const anchor = await deploy("TrustAnchor.sol", "TrustAnchor", [await adapter.getAddress(), 1000, 5000, 2000]);
  const coder = ethers.AbiCoder.defaultAbiCoder();
  const root = (n) => ethers.zeroPadValue(ethers.toBeHex(n), 32);
  let checks = 0;
  const reject = async (action, label) => {
    await assert.rejects(action, undefined, label);
    checks += 1;
  };
  await anchor.registerClient(1, owner.address);
  await anchor.registerClient(2, owner.address);
  assert.equal(await anchor.isActive(99), false); checks += 1;
  const { flat, publicSignals: s } = loadProof("T0");
  const proof = coder.encode(["uint256[8]"], [flat]);
  await reject(() => anchor.connect(stranger).beginRound(1, root(s[2]), s[7], s[8], s[9], s[10]), "round owner");
  await reject(() => anchor.beginRound(1, root(s[2]), s[7], s[8], 0, s[10]), "zero denominator");
  await anchor.beginRound(1, root(s[2]), s[7], s[8], s[9], s[10]);
  await reject(() => anchor.connect(stranger).commitUpdate(1, 1, root(s[1])), "client impersonation");
  await anchor.commitUpdate(1, 1, root(s[1]));
  await reject(() => anchor.commitUpdate(1, 1, root(s[1] + 1n)), "changed commitment");
  await reject(() => anchor.submitUpdate(1, 1, proof, s), "unsealed challenge");
  await anchor.sealChallenge(1, root(s[0]));
  await reject(() => anchor.commitUpdate(2, 1, root(s[1])), "post-challenge commitment");
  await reject(() => anchor.connect(stranger).submitUpdate(1, 1, proof, s), "third-party reputation drain");
  for (const index of [0, 1, 2, 7, 8, 9, 10]) {
    const altered = [...s]; altered[index] += 1n;
    await reject(() => anchor.submitUpdate(1, 1, proof, altered), `public signal ${index} binding`);
  }
  assert.equal(await anchor.nonceOf(1), 0n); checks += 1;
  await anchor.submitUpdate(1, 1, proof, s);
  assert.equal(await anchor.reputationBps(1), 5100n); checks += 1;
  await reject(() => anchor.submitUpdate(1, 1, proof, s), "duplicate client round");
  const boundary = loadProof("T3");
  const b = boundary.publicSignals;
  await anchor.beginRound(2, root(b[2]), b[7], b[8], b[9], b[10]);
  await anchor.commitUpdate(1, 2, root(b[1]));
  await anchor.sealChallenge(2, root(b[0]));
  await reject(() => anchor.submitUpdate(1, 1, proof, s), "stale round");
  await anchor.submitUpdate(1, 2, coder.encode(["uint256[8]"], [boundary.flat]), b);
  assert.equal(await anchor.reputationBps(1), 5100n, "zero weaker-class margin earns no reward"); checks += 1;

  const policyVerifier = await deploy("PolicyVerifierFixture.sol", "PolicyVerifierFixture");
  const policyAnchor = await deploy("TrustAnchor.sol", "TrustAnchor", [await policyVerifier.getAddress(), 1000, 5000, 2000]);
  await policyAnchor.registerClient(1, owner.address);
  await reject(() => policyAnchor.setRewardPolicy(0, 65, 65), "no active reward round");
  for (const [index, correct] of [31, 32, 33].entries()) {
    const round = index + 1;
    const p = [101n, 201n, 301n, BigInt(correct), BigInt(correct), BigInt(50 - correct), BigInt(50 - correct), 62n, 62n, 100n, 1n];
    await policyAnchor.beginRound(round, root(p[2]), p[7], p[8], p[9], p[10]);
    assert.equal(await policyAnchor.rewardSensNum(), 62n, "each round resets the default reward policy"); checks += 1;
    if (round === 1) {
      await reject(() => policyAnchor.connect(stranger).setRewardPolicy(round, 65, 65), "reward policy owner");
      await reject(() => policyAnchor.setRewardPolicy(round, 61, 65), "reward sensitivity below admission");
      await reject(() => policyAnchor.setRewardPolicy(round, 65, 61), "reward specificity below admission");
      await reject(() => policyAnchor.setRewardPolicy(round, 101, 65), "reward sensitivity above one");
      await reject(() => policyAnchor.setRewardPolicy(round, 65, 101), "reward specificity above one");
    }
    await policyAnchor.setRewardPolicy(round, 65, 65);
    await policyAnchor.commitUpdate(1, round, root(p[1]));
    await reject(() => policyAnchor.setRewardPolicy(round, 66, 66), "reward policy frozen at first commitment");
    await policyAnchor.sealChallenge(round, root(p[0]));
    assert.equal(await policyAnchor.submitUpdate.staticCall(1, round, "0x", p), true, `${correct}/50 satisfies 62/100 admission`); checks += 1;
    await policyAnchor.submitUpdate(1, round, "0x", p);
    assert.equal(await policyAnchor.reputationBps(1), correct < 33 ? 5000n : 5010n,
      `${correct}/50 reward is clipped relative to nominal 65/100`); checks += 1;
  }

  const relayArgs = [[owner.address, stranger.address, third.address], 2];
  await reject(() => deploy("RelayBridge.sol", "RelayBridge", [[owner.address, owner.address], 2]), "duplicate relayers");
  const bridge = await deploy("RelayBridge.sol", "RelayBridge", relayArgs);
  const otherBridge = await deploy("RelayBridge.sol", "RelayBridge", relayArgs);
  const message = ethers.toUtf8Bytes("source-chain:anchor:client1:round2:nonce1");
  const chainId = (await ethers.provider.getNetwork()).chainId;
  const digest = ethers.keccak256(coder.encode(["uint256", "address", "bytes32"], [chainId, await bridge.getAddress(), ethers.keccak256(message)]));
  const signers = [owner, stranger].sort((a, b) => BigInt(a.address) < BigInt(b.address) ? -1 : 1);
  const signatures = await Promise.all(signers.map((signer) => signer.signMessage(ethers.getBytes(digest))));
  await reject(() => bridge.accept(message, [signatures[0], signatures[0]]), "duplicate signer");
  await reject(() => otherBridge.accept(message, signatures), "cross-deployment replay");
  await bridge.accept(message, signatures);
  await reject(() => bridge.accept(message, signatures), "message replay");
  console.log(`${checks} contract regression checks passed (solc ${solc.version()}); retained Groth16 proofs plus isolated 62/65 policy fixture; no measurement files written`);
}

function availableCases() {
  const present = CASES.filter((id) =>
    fs.existsSync(path.join(BUILD, "proofs", id, "proof.json"))
  );
  const missing = CASES.filter((id) => !present.includes(id));
  if (missing.length) {
    console.warn(`no proof for ${missing.join(", ")} — run scripts/run_circuit_tests.sh and scripts/prove_gas_samples.sh`);
  }
  if (present.length < MIN_VERIFY_SAMPLES) {
    throw new Error(
      `only ${present.length} proofs available; Task 1.5 requires gasUsed over at least ` +
      `${MIN_VERIFY_SAMPLES} verify transactions`
    );
  }
  return present;
}

function loadProof(caseId) {
  const dir = path.join(BUILD, "proofs", caseId);
  const proof = JSON.parse(fs.readFileSync(path.join(dir, "proof.json"), "utf8"));
  const publicSignals = JSON.parse(fs.readFileSync(path.join(dir, "public.json"), "utf8"));
  // snarkjs order: A (2), B (2x2, inner pair reversed for the pairing), C (2).
  const flat = [
    proof.pi_a[0], proof.pi_a[1],
    proof.pi_b[0][1], proof.pi_b[0][0],
    proof.pi_b[1][1], proof.pi_b[1][0],
    proof.pi_c[0], proof.pi_c[1],
  ].map((v) => BigInt(v));
  return { flat, publicSignals: publicSignals.map((v) => BigInt(v)) };
}

async function main() {
  const { ethers } = require("hardhat");
  const coder = ethers.AbiCoder.defaultAbiCoder();

  const Groth16 = await ethers.getContractFactory("Groth16Verifier");
  const groth16 = await Groth16.deploy();
  await groth16.waitForDeployment();

  const Adapter = await ethers.getContractFactory("PoVVerifierAdapter");
  const adapter = await Adapter.deploy(await groth16.getAddress());
  await adapter.waitForDeployment();

  // alpha = 0.1, beta = 0.5, r_min = 0.2 of Section III-C, in basis points.
  const Anchor = await ethers.getContractFactory("TrustAnchor");
  const anchor = await Anchor.deploy(await adapter.getAddress(), 1000, 5000, 2000);
  await anchor.waitForDeployment();

  const deployGas = {
    Groth16Verifier: (await ethers.provider.getTransactionReceipt(groth16.deploymentTransaction().hash)).gasUsed,
    PoVVerifierAdapter: (await ethers.provider.getTransactionReceipt(adapter.deploymentTransaction().hash)).gasUsed,
    TrustAnchor: (await ethers.provider.getTransactionReceipt(anchor.deploymentTransaction().hash)).gasUsed,
  };

  const rows = [];
  const cases = availableCases();

  // Pure verification cost, isolated from any state write. Both paths are
  // reported: TrustAnchor calls the adapter, so that is the number the paper
  // quotes, and the generated verifier is measured beside it to show what the
  // adapter's decoding adds.
  for (const id of cases) {
    const { flat, publicSignals } = loadProof(id);
    const encoded = coder.encode(["uint256[8]"], [flat]);
    const ok = await adapter.verifyProof(encoded, publicSignals);
    if (!ok) throw new Error(`${id}: on-chain verification rejected a proof the CLI accepted`);
    const gas = await adapter.verifyProof.estimateGas(encoded, publicSignals);
    rows.push({ kind: "verifyProof (view)", case: id, gas });

    const pA = [flat[0], flat[1]];
    const pB = [[flat[2], flat[3]], [flat[4], flat[5]]];
    const pC = [flat[6], flat[7]];
    const rawOk = await groth16.verifyProof(pA, pB, pC, publicSignals);
    if (!rawOk) throw new Error(`${id}: generated verifier rejected a proof the adapter accepted`);
    const rawGas = await groth16.verifyProof.estimateGas(pA, pB, pC, publicSignals);
    rows.push({ kind: "groth16 verifyProof (view)", case: id, gas: rawGas });
  }

  // End-to-end submissions: verification plus reputation update plus events.
  let clientId = 1;
  let roundId = 1;
  const [account] = await ethers.getSigners();
  for (const id of cases) {
    const { flat, publicSignals } = loadProof(id);
    const encoded = coder.encode(["uint256[8]"], [flat]);
    await anchor.registerClient(clientId, account.address);
    for (let rep = 0; rep < 2; rep++) {
      await anchor.beginRound(roundId,
        ethers.zeroPadValue(ethers.toBeHex(publicSignals[2]), 32),
        publicSignals[7], publicSignals[8], publicSignals[9], publicSignals[10]);
      await anchor.commitUpdate(clientId, roundId, ethers.zeroPadValue(ethers.toBeHex(publicSignals[1]), 32));
      await anchor.sealChallenge(roundId, ethers.zeroPadValue(ethers.toBeHex(publicSignals[0]), 32));
      const tx = await anchor.submitUpdate(
        clientId, roundId, encoded, publicSignals,
      );
      const receipt = await tx.wait();
      rows.push({ kind: "submitUpdate", case: `${id}#${rep + 1}`, gas: receipt.gasUsed });
      roundId += 1;
    }
    clientId += 1;
  }

  const lines = ["measurement,case,gas_used,provenance"];
  for (const [name, gas] of Object.entries(deployGas)) {
    lines.push(`deploy ${name},-,${gas},measured (local EVM)`);
  }
  for (const r of rows) lines.push(`${r.kind},${r.case},${r.gas},measured (local EVM)`);

  const out = path.join(BUILD, "gas_report.csv");
  fs.writeFileSync(out, lines.join("\n") + "\n");

  const verifies = rows.filter((r) => r.kind === "verifyProof (view)").map((r) => Number(r.gas));
  const raws = rows.filter((r) => r.kind === "groth16 verifyProof (view)").map((r) => Number(r.gas));
  const submits = rows.filter((r) => r.kind === "submitUpdate").map((r) => Number(r.gas));
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

  if (verifies.length < MIN_VERIFY_SAMPLES) {
    throw new Error(
      `measured ${verifies.length} verify transactions; Task 1.5 requires at least ${MIN_VERIFY_SAMPLES}`
    );
  }

  console.log("deploy gas:", Object.entries(deployGas).map(([k, v]) => `${k}=${v}`).join("  "));
  console.log(`verifyProof (adapter): mean=${mean(verifies).toFixed(0)}  min=${Math.min(...verifies)}  max=${Math.max(...verifies)}  n=${verifies.length}`);
  console.log(`verifyProof (generated): mean=${mean(raws).toFixed(0)}  min=${Math.min(...raws)}  max=${Math.max(...raws)}  n=${raws.length}`);
  console.log(`submitUpdate: mean=${mean(submits).toFixed(0)}  min=${Math.min(...submits)}  max=${Math.max(...submits)}  n=${submits.length}`);
  console.log(`written: ${out}`);
}

(process.env.POV_CONTRACT_TEST_ONLY === "1" ? contractChecks() : main())
  .catch((e) => { console.error(e); process.exit(1); });
