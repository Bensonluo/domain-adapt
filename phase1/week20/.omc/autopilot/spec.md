# Week 20 completion specification

## Objective

Finish and verify the Week 20 distillation study without rerunning completed,
multi-hour experiments unnecessarily.

## Required deliverables

- Part A: three logit-KD arms (`kd_t2`, `kd_t5`, `kd_pure`).
- Part B: three on-policy SFT arms (`rs_mcq`, `rs_teacher`, `rs_both`).
- Reproducible data preparation and resumable orchestration.
- CMExam and CMMLU results for all six arms.
- A combined `week20_summary.json` consistent with the raw evaluation files.
- Fast tests and artifact validation suitable for checking completion locally.

## Acceptance criteria

1. Python sources compile and focused unit tests pass.
2. Shell orchestrators stop on failed stages and do not depend on a personal
   absolute checkout path.
3. Resume/completion decisions use validated records rather than trusting only
   line counts.
4. The artifact validator confirms 2,000 unique training questions, 16,000
   unique judge scores, 2,000 rows per selected SFT arm, and 500 CMExam
   predictions per evaluated arm.
5. The combined summary contains every arm and agrees with domain-gain,
   forgetting, run-config, and CMExam source artifacts.

## Constraints

- Preserve the completed experiment artifacts and reported scientific results.
- Do not modify unrelated files outside `phase1/week20`.
- Use only the Python standard library for artifact validation so it works in a
  lightweight environment.
