// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract RelayBridge {
    mapping(address => bool) public isRelayer;
    mapping(bytes32 => bool) public consumed;
    uint256 public immutable threshold;

    event MessageAccepted(bytes32 indexed messageHash, uint256 signatureCount);

    constructor(address[] memory relayers, uint256 threshold_) {
        require(relayers.length > 0, "relayers required");
        require(threshold_ > 0 && threshold_ <= relayers.length, "invalid threshold");
        threshold = threshold_;
        for (uint256 i = 0; i < relayers.length; i++) {
            require(relayers[i] != address(0), "zero relayer");
            require(!isRelayer[relayers[i]], "duplicate relayer");
            isRelayer[relayers[i]] = true;
        }
    }

    function accept(bytes calldata message, bytes[] calldata signatures) external returns (bytes32 messageHash) {
        // Signatures are bound to this destination chain and bridge deployment.
        // Relayers attest the source event/finality in `message`; this contract
        // verifies their quorum, not the source consensus itself.
        messageHash = keccak256(abi.encode(block.chainid, address(this), keccak256(message)));
        require(!consumed[messageHash], "message consumed");
        bytes32 signedHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", messageHash));
        address previous = address(0);
        uint256 valid = 0;
        for (uint256 i = 0; i < signatures.length; i++) {
            address signer = recover(signedHash, signatures[i]);
            require(signer > previous, "signers not sorted");
            previous = signer;
            if (isRelayer[signer]) {
                valid += 1;
            }
        }
        require(valid >= threshold, "insufficient quorum");
        consumed[messageHash] = true;
        emit MessageAccepted(messageHash, valid);
    }

    function recover(bytes32 digest, bytes calldata signature) public pure returns (address) {
        require(signature.length == 65, "bad signature length");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(signature.offset)
            s := calldataload(add(signature.offset, 32))
            v := byte(0, calldataload(add(signature.offset, 64)))
        }
        if (v < 27) {
            v += 27;
        }
        require(v == 27 || v == 28, "bad signature v");
        require(uint256(s) <= 0x7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0, "noncanonical signature s");
        address signer = ecrecover(digest, v, r, s);
        require(signer != address(0), "invalid signature");
        return signer;
    }
}
