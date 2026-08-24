# Week 21 synthetic-data quality

## Scope and provenance

- Self-Instruct accepted: 1000 (`mlx`, model `/Users/luopeng/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`)
- Evol-Instruct accepted: 250 (`mlx`, model `/Users/luopeng/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`)
- Real comparison: 5000 sampled from 42223 records
- Manual AI review: 30/30 reviewed; apparent accuracy 83.33%

The recorded reviewer kind is `ai_nonclinician`. This is an AI qualitative sanity check, not clinician or human validation.

## Measured quality

| Metric | Result |
|---|---:|
| Schema and answer validity | 100.00% |
| Exact duplicate rate | 0.00% |
| Distinct-1 / Distinct-2 | 0.021984 / 0.131606 |
| Approximate Self-BLEU | 0.471808 |
| Mean / P95 max 3-gram similarity to real sample | 0.087475 / 0.14876 |
| Exact matches to real sample | 0 |
| Synthetic/real mean question-length ratio | 2.033794 |
| Answer-label JS divergence | 0.09738 |
| Answer-text support in explanation (proxy) | 77.20% |
| Self-Instruct items ≥0.78 similarity to full holdout | 0 / 1000 |
| Evol-Instruct items ≥0.78 similarity to full holdout | 0 / 250 |

Approximate Self-BLEU is nearest-neighbour character-bigram Jaccard, not package BERTScore. The full-holdout screen is character 3-gram Jaccard and is not a semantic-embedding guarantee. All accepted synthetic items passed the 0.78 threshold; the mixed replacement set contains 12 high-overlap real-source items inherited from the source corpus.

## Limitations

The completed generation predates the MLX RNG-seeding fix now present in the scripts, so raw responses freeze and authenticate this run but cannot be regenerated bit-for-bit from its seed alone. Future fresh runs seed MLX per request. Medical correctness still requires independent clinician review before clinical use.
