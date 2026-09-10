pragma circom 2.1.6;

// Proof-of-Validation circuit, class-aware predicate P_bal.
//
// Relation R_PoV[P] of Definition 2 (main paper, Eq. 12-15), instantiated for the
// reference binary-linear model over the committed validation subset.
//
//   public   x = (C_D, H_theta, H_glob, c, pp_P, eps_copy)
//   witness  w = (theta_hat, D_val, theta_global)
//
// with c = (TP, TN, FP, FN) and pp_P = (tau_se_num, tau_sp_num, denom). The four
// conditions enforced below, in the order of Definition 2:
//
//   (1) three commitment openings   Poseidon(theta_hat)   = H_theta
//                                   Poseidon(D_val)       = C_D
//                                   Poseidon(theta_glob)  = H_glob
//   (2) count consistency           c = Confusion(M_theta, D_val),  sum(c) = n
//   (3) utility predicate           TP*d >= tau_se_num*(TP+FN)
//                                   TN*d >= tau_sp_num*(TN+FP)
//   (4) copy detector               ||theta_hat - theta_glob||_2^2 >= eps_copy^2
//
// Every comparison is an integer cross-multiplication; no division occurs.

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/bitify.circom";

// Streaming Poseidon sponge over an arbitrary-length input.
//
// circomlib's Poseidon admits at most 16 field elements (t <= 17), so a vector
// longer than that is absorbed in a Merkle-Damgard chain of rate-15 blocks:
// state_{k+1} = Poseidon([state_k, in_{15k}, ..., in_{15k+14}]), zero-padded on
// the final block. `iv` is a compile-time domain separator, so the three
// openings of condition (1) live in separate domains and a digest for one can
// never be reinterpreted as a digest for another.
template PoseidonSponge(nInputs, iv) {
    signal input in[nInputs];
    signal output out;

    var RATE = 15;
    var nChunks = (nInputs + RATE - 1) \ RATE;

    component h[nChunks];
    signal state[nChunks + 1];
    state[0] <== iv;

    for (var k = 0; k < nChunks; k++) {
        h[k] = Poseidon(RATE + 1);
        h[k].inputs[0] <== state[k];
        for (var j = 0; j < RATE; j++) {
            var idx = k * RATE + j;
            if (idx < nInputs) {
                h[k].inputs[j + 1] <== in[idx];
            } else {
                h[k].inputs[j + 1] <== 0;
            }
        }
        state[k + 1] <== h[k].out;
    }

    out <== state[nChunks];
}

// Signed comparison against zero for a value known to lie in
// (-2^(bits-1), 2^(bits-1)). Quantised logits are signed, so the sign test is
// performed by offsetting into the non-negative range before a range-checked
// unsigned comparison. out = 1 iff in >= 0.
template SignedRange(bits) {
    signal input in;
    component limbs = Num2Bits(bits);
    limbs.in <== in + (1 << (bits - 1));
}

template SignGEZero(bits) {
    signal input in;
    signal output out;

    component range = SignedRange(bits);
    range.in <== in;

    component cmp = GreaterEqThan(bits + 1);
    cmp.in[0] <== in + (1 << bits);
    cmp.in[1] <== (1 << bits);
    out <== cmp.out;
}

template BinaryLinearPoV(samples, features, logitBits, countBits) {
    // INT8 weights, INT16 features and INT32 bias keep every dot product and
    // squared parameter distance below the field modulus. These are arithmetic
    // representation bounds, not the optional update-norm bound B_tau.
    assert(features < (1 << 16));
    assert(logitBits >= 40);
    assert(countBits >= 64);
    assert(samples < (1 << 20));
    // ---- public instance x, in the order of Eq. (12) ----
    signal input validationRoot;      // C_D
    signal input modelRoot;           // H_theta
    signal input globalRoot;          // H_glob
    signal input tp;                  // c = (TP, TN, FP, FN)
    signal input tn;
    signal input fp;
    signal input fn;
    signal input tauSensNum;          // pp_P: numerator of tau_se over denom
    signal input tauSpecNum;          // pp_P: numerator of tau_sp over denom
    signal input denom;               // pp_P: shared public denominator d
    signal input copyEpsilonSq;       // eps_copy^2

    // ---- witness w ----
    signal input x[samples][features];   // D_val features
    signal input y[samples];             // D_val labels
    signal input w[features];            // theta_hat weights
    signal input bias;                   // theta_hat bias
    signal input salt;                   // commitment randomiser for theta_hat
    signal input globalW[features];      // theta^(t) weights
    signal input globalBias;             // theta^(t) bias
    signal input globalSalt;             // randomiser for theta^(t)

    component weightRange[features];
    component globalWeightRange[features];
    component featureRange[samples][features];
    for (var j = 0; j < features; j++) {
        weightRange[j] = SignedRange(8);
        weightRange[j].in <== w[j];
        globalWeightRange[j] = SignedRange(8);
        globalWeightRange[j].in <== globalW[j];
    }
    component biasRange = SignedRange(32);
    biasRange.in <== bias;
    component globalBiasRange = SignedRange(32);
    globalBiasRange.in <== globalBias;
    for (var i = 0; i < samples; i++) {
        for (var j = 0; j < features; j++) {
            featureRange[i][j] = SignedRange(16);
            featureRange[i][j].in <== x[i][j];
        }
    }

    component denomRange = Num2Bits(20);
    denomRange.in <== denom;
    component sensRange = Num2Bits(20);
    sensRange.in <== tauSensNum;
    component specRange = Num2Bits(20);
    specRange.in <== tauSpecNum;
    component denomZero = IsZero();
    denomZero.in <== denom;
    denomZero.out === 0;
    component sensZero = IsZero();
    sensZero.in <== tauSensNum;
    sensZero.out === 0;
    component specZero = IsZero();
    specZero.in <== tauSpecNum;
    specZero.out === 0;
    component sensBound = LessEqThan(20);
    sensBound.in[0] <== tauSensNum;
    sensBound.in[1] <== denom;
    sensBound.out === 1;
    component specBound = LessEqThan(20);
    specBound.in[0] <== tauSpecNum;
    specBound.in[1] <== denom;
    specBound.out === 1;
    component epsilonRange = Num2Bits(countBits);
    epsilonRange.in <== copyEpsilonSq;
    component epsilonZero = IsZero();
    epsilonZero.in <== copyEpsilonSq;
    epsilonZero.out === 0;

    // =====================================================================
    // (1) Three commitment openings.
    // =====================================================================
    component modelHash = PoseidonSponge(features + 2, 1);
    for (var j = 0; j < features; j++) {
        modelHash.in[j] <== w[j];
    }
    modelHash.in[features] <== bias;
    modelHash.in[features + 1] <== salt;
    modelHash.out === modelRoot;

    component globalHash = PoseidonSponge(features + 2, 2);
    for (var j = 0; j < features; j++) {
        globalHash.in[j] <== globalW[j];
    }
    globalHash.in[features] <== globalBias;
    globalHash.in[features + 1] <== globalSalt;
    globalHash.out === globalRoot;

    // The validation subset is absorbed as (features + label) per sample, in the
    // sample order the aggregator committed to.
    component dataHash = PoseidonSponge(samples * (features + 1), 3);
    for (var i = 0; i < samples; i++) {
        for (var j = 0; j < features; j++) {
            dataHash.in[i * (features + 1) + j] <== x[i][j];
        }
        dataHash.in[i * (features + 1) + features] <== y[i];
    }
    dataHash.out === validationRoot;

    // =====================================================================
    // (2) Count consistency: the four accumulators are derived from the
    //     forward pass over the committed set, and their sum is pinned to n.
    // =====================================================================
    signal logit[samples];
    signal pred[samples];
    component sign[samples];

    // R1CS admits one multiplication per constraint, so the inner product is
    // accumulated term by term rather than folded into a single expression.
    signal prod[samples][features];
    signal dotAcc[samples][features + 1];

    signal accTP[samples + 1];
    signal accTN[samples + 1];
    signal accFP[samples + 1];
    signal accFN[samples + 1];
    accTP[0] <== 0;
    accTN[0] <== 0;
    accFP[0] <== 0;
    accFN[0] <== 0;

    signal isTP[samples];
    signal isTN[samples];
    signal isFP[samples];
    signal isFN[samples];

    for (var i = 0; i < samples; i++) {
        // Labels are boolean.
        y[i] * (y[i] - 1) === 0;

        dotAcc[i][0] <== bias;
        for (var j = 0; j < features; j++) {
            prod[i][j] <== x[i][j] * w[j];
            dotAcc[i][j + 1] <== dotAcc[i][j] + prod[i][j];
        }
        logit[i] <== dotAcc[i][features];

        sign[i] = SignGEZero(logitBits);
        sign[i].in <== logit[i];
        pred[i] <== sign[i].out;

        // Route each sample into exactly one of the four accumulators using the
        // prediction and label bits already computed.
        isTP[i] <== pred[i] * y[i];
        isFP[i] <== pred[i] - isTP[i];             // pred * (1 - y)
        isFN[i] <== y[i] - isTP[i];                // (1 - pred) * y
        isTN[i] <== 1 - pred[i] - isFN[i];         // (1 - pred) * (1 - y)

        accTP[i + 1] <== accTP[i] + isTP[i];
        accTN[i + 1] <== accTN[i] + isTN[i];
        accFP[i + 1] <== accFP[i] + isFP[i];
        accFN[i + 1] <== accFN[i] + isFN[i];
    }

    // The published counts must equal the in-circuit accumulators. This is what
    // stops a prover from announcing counts unrelated to the committed set.
    tp === accTP[samples];
    tn === accTN[samples];
    fp === accFP[samples];
    fn === accFN[samples];
    tp + tn + fp + fn === samples;
    // Cross-multiplication alone treats a missing class as 0 >= 0.
    component positiveClassEmpty = IsZero();
    positiveClassEmpty.in <== tp + fn;
    positiveClassEmpty.out === 0;
    component negativeClassEmpty = IsZero();
    negativeClassEmpty.in <== tn + fp;
    negativeClassEmpty.out === 0;

    // =====================================================================
    // (3) Utility predicate P_bal, by cross-multiplication. No division.
    //     TP * d >= tau_se_num * (TP + FN)   and   TN * d >= tau_sp_num * (TN + FP)
    // =====================================================================
    signal sensLhs;
    signal sensRhs;
    signal specLhs;
    signal specRhs;
    sensLhs <== tp * denom;
    sensRhs <== tauSensNum * (tp + fn);
    specLhs <== tn * denom;
    specRhs <== tauSpecNum * (tn + fp);

    component sensOk = GreaterEqThan(countBits);
    sensOk.in[0] <== sensLhs;
    sensOk.in[1] <== sensRhs;

    component specOk = GreaterEqThan(countBits);
    specOk.in[0] <== specLhs;
    specOk.in[1] <== specRhs;

    // =====================================================================
    // (4) Copy detector: ||theta_hat - theta^(t)||_2^2 >= eps_copy^2.
    // =====================================================================
    signal diff[features + 1];
    signal sq[features + 1];
    signal distAcc[features + 2];
    distAcc[0] <== 0;
    for (var j = 0; j < features; j++) {
        diff[j] <== w[j] - globalW[j];
        sq[j] <== diff[j] * diff[j];
        distAcc[j + 1] <== distAcc[j] + sq[j];
    }
    diff[features] <== bias - globalBias;
    sq[features] <== diff[features] * diff[features];
    distAcc[features + 1] <== distAcc[features] + sq[features];
    component distanceRange = Num2Bits(countBits);
    distanceRange.in <== distAcc[features + 1];

    component notCopy = GreaterEqThan(countBits);
    notCopy.in[0] <== distAcc[features + 1];
    notCopy.in[1] <== copyEpsilonSq;

    // =====================================================================
    // Admission. Conditions (1) and (2) are hard equalities above; the
    // conjunction of (3) and (4) is pinned to 1, so a satisfying witness — and
    // therefore a proof — exists only for an admitted submission.
    // =====================================================================
    signal predicateOk;
    signal gate;
    predicateOk <== sensOk.out * specOk.out;
    gate <== predicateOk * notCopy.out;
    gate === 1;
}

// Reference PoV circuit over the 100-sample committed validation subset
// (|D_val| = 100, 17 features), matching Section IV-B and Appendix A. Public
// signal order follows Eq. (12); `main` declares it explicitly so that
// public.json, the Solidity verifier, and Algorithm 1 agree.
component main {public [
    validationRoot,
    modelRoot,
    globalRoot,
    tp, tn, fp, fn,
    tauSensNum, tauSpecNum, denom,
    copyEpsilonSq
]} = BinaryLinearPoV(100, 17, 40, 64);
