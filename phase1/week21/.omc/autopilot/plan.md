# Week 21 execution plan

1. Build 100 seed instructions from CMExam train with exact holdout exclusion.
2. Generate 1,000 accepted MCQs with the local Qwen3 MLX teacher; retain raw responses and rejection statistics.
3. Evolve 250 accepted questions through deterministic depth 1-3 schedules; retain every stage.
4. Measure structural validity, lexical diversity, novelty, distribution shift, and a 30-item manual AI review explicitly labeled as non-human/non-clinical.
5. Construct exactly 1,000 real + 1,000 synthetic SFT examples with exact collision filtering; separately measure character 3-gram overlap against the complete holdout.
6. Train with the Week 19 real control's base and hyperparameters, merge, and evaluate the same 500-item CMExam holdout.
7. Re-evaluate both arms while retaining all 500 paired predictions; report paired bootstrap uncertainty and exact McNemar statistics against the -2 percentage-point margin.
8. Bind each stage with content hashes and deeply validate raw-to-accepted lineage, source reconciliation, checkpoints, full predictions, statistics, and reports.
