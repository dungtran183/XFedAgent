pragma circom 2.1.6;

include "../node_modules/circomlib/circuits/poseidon.circom";

template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc = 0;
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc += out[i] * (1 << i);
    }
    lc === in;
}

template LessThan(n) {
    signal input in[2];
    signal output out;
    component bits = Num2Bits(n + 1);
    bits.in <== in[0] + (1 << n) - in[1];
    out <== 1 - bits.out[n];
}

template BinaryLinearPoV(samples, features, bitWidth) {
    signal input x[samples][features];
    signal input y[samples];
    signal input w[features];
    signal input bias;
    signal input salt;
    signal input validationRoot;
    signal input modelRoot;
    signal input thresholdCorrect;
    signal output accepted;

    component modelHash = Poseidon(features + 2);
    for (var j = 0; j < features; j++) {
        modelHash.inputs[j] <== w[j];
    }
    modelHash.inputs[features] <== bias;
    modelHash.inputs[features + 1] <== salt;
    modelHash.out === modelRoot;

    signal rolling[samples + 1];
    rolling[0] <== salt;
    component sampleHash[samples];
    component geqZero[samples];
    signal logit[samples];
    signal correct[samples];
    signal correctCount[samples + 1];
    correctCount[0] <== 0;

    for (var i = 0; i < samples; i++) {
        sampleHash[i] = Poseidon(features + 2);
        sampleHash[i].inputs[0] <== rolling[i];
        sampleHash[i].inputs[1] <== y[i];
        var dot = bias;
        for (var j = 0; j < features; j++) {
            sampleHash[i].inputs[j + 2] <== x[i][j];
            dot += x[i][j] * w[j];
        }
        logit[i] <== dot;
        rolling[i + 1] <== sampleHash[i].out;
        geqZero[i] = LessThan(bitWidth);
        geqZero[i].in[0] <== logit[i] + (1 << (bitWidth - 1));
        geqZero[i].in[1] <== (1 << (bitWidth - 1));
        correct[i] <== 1 - (geqZero[i].out + y[i] - 2 * geqZero[i].out * y[i]);
        correctCount[i + 1] <== correctCount[i] + correct[i];
    }
    rolling[samples] === validationRoot;

    component threshold = LessThan(bitWidth);
    threshold.in[0] <== correctCount[samples];
    threshold.in[1] <== thresholdCorrect;
    accepted <== 1 - threshold.out;
}

// Reference PoV circuit over the 100-sample committed validation subset
// (|D_val| = 100, 17 features). The compact edge model is committed in full via
// a single Poseidon sponge over its quantised weight vector (see modelHash),
// matching the description in Section IV-B.
component main = BinaryLinearPoV(100, 17, 32);
