# Week 21 specification

## Objective

Build and execute an offline Self-Instruct + Evol-Instruct pipeline for Chinese medical multiple-choice data, quantify synthetic-data quality, and measure whether replacing 50% of a 2,000-example real SFT set preserves CMExam performance.

## Controlled experiment

- Control: Week 19 `real` arm, 2,000 human-authored CMExam examples.
- Treatment: 1,000 deterministic real examples plus 1,000 accepted synthetic examples.
- Fixed: base checkpoint, SFT hyperparameters, seed, training-set size, and 500-example CMExam holdout evaluator.
- Primary metric: CMExam holdout accuracy and paired treatment-minus-control delta.
- Operational rule: the point estimate passes when loss is at most 2 percentage points.
- Inferential rule: noninferiority is established only when the paired 95% interval lower bound exceeds -2 percentage points.

## Generation contract

- Derive exactly 100 seed records from CMExam training data with exact holdout exclusion.
- Support a local MLX model and deterministic mock backend for tests.
- Persist raw generations separately from accepted normalized records and bind material artifacts with SHA-256 lineage.
- Resume only when producer inputs and parameters match; archive stale outputs recoverably.
- Require one medical single-choice question, 4-5 non-empty options, one answer in A-E, a non-empty explanation, and novelty below the configured generation threshold.
- Evolve accepted questions for depth 1-3 while preserving a parseable single correct answer.

## Quality contract

- Report schema/answer validity, exact-duplicate rate, lexical diversity, approximate Self-BLEU, novelty against real data, length shift, label-distribution shift, and evolution-depth statistics.
- Screen accepted synthetic questions against the complete holdout using exact match and documented character 3-gram similarity; do not claim semantic-leakage guarantees.
- Label the deterministic 30-item review as manual AI review, not human or clinician evidence.
- Retain all 500 paired predictions and report paired bootstrap uncertainty plus exact McNemar statistics.

## Acceptance

- Pipelines, orchestration, tests, and deep validator complete.
- 1,000 Self-Instruct and 250 Evol-Instruct records accepted from the local teacher.
- Treatment checkpoint, fused model, both-arm full evaluation, statistics, and reports complete.
- Content lineage verifies all stages, and conclusions distinguish the operational point threshold from statistical noninferiority.
