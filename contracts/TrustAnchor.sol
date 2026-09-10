// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IPoVVerifier {
    function verifyProof(bytes calldata proof, uint256[] calldata publicSignals) external view returns (bool);
}

contract TrustAnchor {
    address public immutable coordinator;
    IPoVVerifier public immutable verifier;
    uint256 public immutable rewardAlphaBps;
    uint256 public immutable penaltyBps;
    uint256 public immutable minReputationBps;

    mapping(uint256 => uint256) public reputationBps;
    mapping(uint256 => uint256) public nonceOf;
    mapping(uint256 => address) public clientOwner;
    mapping(uint256 => bytes32) public committedModel;
    mapping(uint256 => uint256) public committedRound;
    mapping(uint256 => uint256) public consumedRound;

    uint256 public activeRound;
    bytes32 public globalRoot;
    bytes32 public validationRoot;
    uint256 public tauSensNum;
    uint256 public tauSpecNum;
    uint256 public rewardSensNum;
    uint256 public rewardSpecNum;
    uint256 public denom;
    uint256 public copyEpsilonSq;
    uint256 public commitmentCount;

    event RoundOpened(uint256 indexed roundId, bytes32 globalRoot);
    event RewardPolicySet(uint256 indexed roundId, uint256 sensitivityNum, uint256 specificityNum);
    event ModelCommitted(uint256 indexed clientId, uint256 indexed roundId, bytes32 modelRoot);
    event ChallengeSealed(uint256 indexed roundId, bytes32 validationRoot);

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
        coordinator = msg.sender;
        rewardAlphaBps = rewardAlphaBps_;
        penaltyBps = penaltyBps_;
        minReputationBps = minReputationBps_;
    }

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "coordinator required");
        _;
    }

    function registerClient(uint256 clientId, address account) external onlyCoordinator {
        require(account != address(0) && clientOwner[clientId] == address(0), "invalid client registration");
        clientOwner[clientId] = account;
        reputationBps[clientId] = 5000;
    }

    // Policy and global model are fixed before commitments. The coordinator is
    // the validation-pool custodian; this contract does not implement a VRF or a
    // proof that its private pool was sampled representatively.
    function beginRound(
        uint256 roundId, bytes32 globalRoot_, uint256 tauSensNum_,
        uint256 tauSpecNum_, uint256 denom_, uint256 copyEpsilonSq_
    ) external onlyCoordinator {
        require(roundId > activeRound && globalRoot_ != bytes32(0), "invalid round");
        require(denom_ > 0 && denom_ <= 1000000, "invalid denominator");
        require(tauSensNum_ > 0 && tauSensNum_ <= denom_, "invalid sensitivity threshold");
        require(tauSpecNum_ > 0 && tauSpecNum_ <= denom_, "invalid specificity threshold");
        require(copyEpsilonSq_ > 0 && copyEpsilonSq_ < 2 ** 64, "invalid copy bound");
        activeRound = roundId;
        globalRoot = globalRoot_;
        validationRoot = bytes32(0);
        tauSensNum = tauSensNum_;
        tauSpecNum = tauSpecNum_;
        rewardSensNum = tauSensNum_;
        rewardSpecNum = tauSpecNum_;
        denom = denom_;
        copyEpsilonSq = copyEpsilonSq_;
        commitmentCount = 0;
        emit RoundOpened(roundId, globalRoot_);
    }

    // Admission may include a public tolerance, while rewards remain relative
    // to nominal thresholds. Both policies must be fixed before any commitment.
    // Omission retains the exact-threshold policy used by the reference proofs.
    function setRewardPolicy(
        uint256 roundId, uint256 nominalSensNum, uint256 nominalSpecNum
    ) external onlyCoordinator {
        require(roundId == activeRound && roundId > 0, "inactive round");
        require(commitmentCount == 0 && validationRoot == bytes32(0), "commitments already open");
        require(nominalSensNum >= tauSensNum && nominalSensNum <= denom, "invalid reward sensitivity");
        require(nominalSpecNum >= tauSpecNum && nominalSpecNum <= denom, "invalid reward specificity");
        rewardSensNum = nominalSensNum;
        rewardSpecNum = nominalSpecNum;
        emit RewardPolicySet(roundId, nominalSensNum, nominalSpecNum);
    }

    function commitUpdate(uint256 clientId, uint256 roundId, bytes32 modelRoot) external {
        require(msg.sender == clientOwner[clientId], "client owner required");
        require(roundId == activeRound && roundId > 0, "inactive round");
        require(validationRoot == bytes32(0), "challenge already sealed");
        require(committedRound[clientId] != roundId && modelRoot != bytes32(0), "model already committed or empty");
        committedModel[clientId] = modelRoot;
        committedRound[clientId] = roundId;
        commitmentCount += 1;
        emit ModelCommitted(clientId, roundId, modelRoot);
    }

    function sealChallenge(uint256 roundId, bytes32 validationRoot_) external onlyCoordinator {
        require(roundId == activeRound && commitmentCount > 0, "no committed round");
        require(validationRoot == bytes32(0) && validationRoot_ != bytes32(0), "invalid challenge");
        validationRoot = validationRoot_;
        emit ChallengeSealed(roundId, validationRoot_);
    }

    function submitUpdate(
        uint256 clientId,
        uint256 roundId,
        bytes calldata proof,
        uint256[] calldata publicSignals
    ) external returns (bool accepted) {
        require(msg.sender == clientOwner[clientId], "client owner required");
        require(roundId == activeRound && roundId > 0, "inactive round");
        require(validationRoot != bytes32(0), "challenge not sealed");
        require(committedRound[clientId] == roundId, "model not committed");
        require(consumedRound[clientId] != roundId, "client round consumed");
        require(publicSignals.length == 11, "invalid public instance");
        require(publicSignals[0] == uint256(validationRoot), "validation root mismatch");
        require(publicSignals[1] == uint256(committedModel[clientId]), "model root mismatch");
        require(publicSignals[2] == uint256(globalRoot), "global root mismatch");
        require(publicSignals[7] == tauSensNum && publicSignals[8] == tauSpecNum && publicSignals[9] == denom, "predicate policy mismatch");
        require(publicSignals[10] == copyEpsilonSq, "copy policy mismatch");
        uint256 tp = publicSignals[3];
        uint256 tn = publicSignals[4];
        uint256 fp = publicSignals[5];
        uint256 fn = publicSignals[6];
        require(tp <= 100 && tn <= 100 && fp <= 100 && fn <= 100, "count out of range");
        require(tp + tn + fp + fn == 100 && tp + fn > 0 && tn + fp > 0, "invalid class counts");
        consumedRound[clientId] = roundId;
        nonceOf[clientId] += 1;
        accepted = verifier.verifyProof(proof, publicSignals)
            && tp * denom >= tauSensNum * (tp + fn)
            && tn * denom >= tauSpecNum * (tn + fp);
        if (accepted) {
            // Reward the weaker verified class margin, not caller-supplied raw
            // accuracy. Integer division rounds the reward down in basis points.
            uint256 seRequired = rewardSensNum * (tp + fn);
            uint256 spRequired = rewardSpecNum * (tn + fp);
            uint256 seMargin = tp * denom > seRequired
                ? ((tp * denom - seRequired) * 10000) / (denom * (tp + fn)) : 0;
            uint256 spMargin = tn * denom > spRequired
                ? ((tn * denom - spRequired) * 10000) / (denom * (tn + fp)) : 0;
            uint256 margin = seMargin < spMargin ? seMargin : spMargin;
            uint256 reward = (rewardAlphaBps * margin) / 10000;
            uint256 updated = reputationBps[clientId] + reward;
            reputationBps[clientId] = updated > 10000 ? 10000 : updated;
            uint256 accuracyBps = ((tp + tn) * 10000) / 100;
            emit UpdateAccepted(clientId, roundId, committedModel[clientId], validationRoot, accuracyBps, reputationBps[clientId]);
            return true;
        }
        reputationBps[clientId] = reputationBps[clientId] > penaltyBps ? reputationBps[clientId] - penaltyBps : 0;
        emit UpdateRejected(clientId, roundId, reputationBps[clientId]);
        return false;
    }

    function isActive(uint256 clientId) external view returns (bool) {
        return clientOwner[clientId] != address(0) && reputationBps[clientId] >= minReputationBps;
    }
}
