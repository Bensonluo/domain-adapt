# Week 20 completion plan

1. Inventory code, logs, and experiment artifacts; reconstruct the contract.
2. Run baseline syntax and loss-invariant checks.
3. Add a standard-library artifact validator covering data cardinality,
   uniqueness, score validity, result presence, and summary reconciliation.
4. Harden both orchestration scripts for portable paths and fail-fast behavior;
   use the validator as their completion gate.
5. Add focused unit tests for parsing, KD loss behavior, and validation helpers.
6. Run the full fast QA suite and validate the existing six-arm artifacts.
7. Review functional completeness, security, and code quality; clean transient
   autopilot state files.
