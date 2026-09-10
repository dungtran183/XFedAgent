require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.24",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      },
      // TrustAnchor.submitUpdate carries the full public instance plus the proof,
      // which exceeds what the legacy codegen can hold on the EVM stack. The IR
      // pipeline compiles it without altering the interface.
      viaIR: true
    }
  },
  paths: {
    sources: "./contracts",
    artifacts: "./artifacts/contracts",
    cache: "./artifacts/cache"
  }
};
