// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IPoVVerifier {
    function verifyProof(bytes calldata proof, uint256[] calldata publicSignals) external view returns (bool);
}

contract TrustAnchor {
    IPoVVerifier public immutable verifier;
    uint256 public immutable rewardAlphaBps;
    uint256 public immutable penaltyBps;
    uint256 public immutable minReputationBps;

    mapping(uint256 => uint256) public reputationBps;
    mapping(uint256 => uint256) public nonceOf;

    event UpdateAccepted(
        uint256 indexed clientId,
        uint256 indexed roundId,
        bytes32 modelRoot,
        bytes32 validationRoot,
        uint256 accuracyBps,
        uint256 reputationBps
    );
    event UpdateRejected(uint256 indexed clientId, uint256 indexed roundId, uint256 reputationBps);

    constructor(address verifier_, uint256 rewardAlphaBps_, uint256 penaltyBps_, uint256 minReputationBps_) {
        require(verifier_ != address(0), "verifier required");
        require(rewardAlphaBps_ <= 10000, "reward out of range");
        require(penaltyBps_ <= 10000, "penalty out of range");
        require(minReputationBps_ <= 10000, "minimum out of range");
        verifier = IPoVVerifier(verifier_);
        rewardAlphaBps = rewardAlphaBps_;
        penaltyBps = penaltyBps_;
        minReputationBps = minReputationBps_;
    }

    function submitUpdate(
        uint256 clientId,
        uint256 roundId,
        bytes32 modelRoot,
        bytes32 validationRoot,
        uint256 accuracyBps,
        uint256 thresholdBps,
        bytes calldata proof,
        uint256[] calldata publicSignals
    ) external returns (bool accepted) {
        require(accuracyBps <= 10000, "accuracy out of range");
        require(thresholdBps <= 10000, "threshold out of range");
        if (reputationBps[clientId] == 0 && nonceOf[clientId] == 0) {
            reputationBps[clientId] = 5000;
        }
        nonceOf[clientId] += 1;
        accepted = verifier.verifyProof(proof, publicSignals) && accuracyBps >= thresholdBps;
        if (accepted) {
            uint256 margin = accuracyBps - thresholdBps;
            uint256 reward = (rewardAlphaBps * margin) / 10000;
            uint256 updated = reputationBps[clientId] + reward;
            reputationBps[clientId] = updated > 10000 ? 10000 : updated;
            emit UpdateAccepted(clientId, roundId, modelRoot, validationRoot, accuracyBps, reputationBps[clientId]);
            return true;
        }
        reputationBps[clientId] = reputationBps[clientId] > penaltyBps ? reputationBps[clientId] - penaltyBps : 0;
        emit UpdateRejected(clientId, roundId, reputationBps[clientId]);
        return false;
    }

    function isActive(uint256 clientId) external view returns (bool) {
        return reputationBps[clientId] >= minReputationBps;
    }
}

