# Week 21 50% synthetic replacement experiment

## Controlled design

The Week 19 arm uses 2,000 real CMExam SFT examples. The Week 21 treatment keeps the same 2,000-example size, base checkpoint, seed, LoRA/SFT hyperparameters, and fixed 500-question holdout, replacing exactly 1,000 training examples with accepted synthetic examples. The pre-specified operational tolerance is −2 percentage points.

Both fused checkpoints were re-evaluated in one matched pass using CPU float32 greedy decoding, batch size 8, and 16 generated tokens; this paired rerun is the basis of the statistics below.

| Arm | Real | Synthetic | Holdout n | CMExam accuracy |
|---|---:|---:|---:|---:|
| Week 19 real control | 2,000 | 0 | 500 | 53.80% |
| Week 21 50% replacement | 1000 | 1000 | 500 | 53.00% |

Treatment minus control: **-0.80 percentage points**. Paired bootstrap 95% interval: **[-3.40, +1.80] percentage points**. Exact McNemar two-sided p-value: **0.6440**.

Decision: **operational point estimate passed, but statistical noninferiority was not established**. The point estimate (-0.80 points) is within the operational −2-point tolerance. The paired interval crosses that margin, so statistical noninferiority is not established.

## Answer to the anchor question

For this one model and dataset, 50% replacement is promising by point estimate, but the evidence is statistically inconclusive. A larger holdout or repeated training seeds are needed before claiming that synthetic data can replace half of the real SFT data.

## Integrity and limitations

- Full paired predictions for all 500 questions are retained and the aggregate accuracies are recomputed from them.
- Content hashes bind generation inputs/outputs, treatment data, adapter, fused model, evaluation, statistics, and reports.
- Synthetic questions passed exact-match and 0.78 character 3-gram holdout screening. Twelve high-overlap records in the treatment are real-source items; this inherited source-corpus overlap remains a limitation.
- The control checkpoint was reused from Week 19 and re-evaluated with the same full-prediction evaluator; training stochasticity across repeated seeds was not measured.
