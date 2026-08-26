# AdaptStack architecture

AdaptStack separates research decisions from execution engines. A YAML file
describes the experiment; the planner turns it into a stable stage sequence;
registered stage implementations consume and produce artifact references.

## Four layers

1. **L1 Data** — ingestion, cleaning, deduplication, SFT/preference/synthetic
   dataset construction, and data provenance.
2. **L2 Training** — CPT, SFT, DPO, GRPO, and distillation as independently
   enabled stages.
3. **L3 Inference** — RAG, vLLM-compatible serving, and quantization boundaries.
4. **L4 Evaluation** — objective benchmarks, LLM judges, blind human evaluation,
   and bad-case feedback.

## Execution flow

```text
YAML config
    │
    ▼
strict loader ──► deterministic planner ──► stage registry
                                            │
                                            ▼
                                    immutable run context
                                            │
                                            ▼
                                 atomic artifact manifests
```

The runner passes only artifact references between stages. This keeps storage,
training frameworks, and model runtimes replaceable. A stage may mutate its own
output directory but not another stage's artifacts.

## Reproducibility boundary

Every dry-run manifest contains the run ID, domain, base model, configuration
SHA-256 digest, upstream artifacts, and stage-specific settings. Real backends
must extend this with code revision, dataset hashes, model hashes, random seeds,
hardware, runtime versions, duration, and cost.

## Failure model

- Invalid or unknown public configuration keys fail before planning. Integration-specific
  settings must be nested explicitly under `backend_options`.
- Duplicate YAML keys are rejected instead of silently using the last value.
- Config-owned output paths must remain relative to the project and cannot cross
  symbolic links; an explicit CLI `--output` is treated as deliberate user authority.
- Duplicate stage registrations fail immediately.
- A stage result must match the planned stage identity and return `planned` in
  dry-run or `completed` in execution mode; any other result fails the run.
- Real execution fails until a concrete backend explicitly declares support.
- Manifests are written through a temporary file and atomic replace.
- External tracking is opt-in; the scaffold never contacts WandB.

## Extension points

- `PipelineStage`: data/training/evaluation unit.
- `InferenceBackend`: generation or serving implementation.
- `Evaluator`: benchmark, judge, or human-evaluation adapter.
- `RunTracker`: local, WandB, or other tracking destination.

These interfaces are intentionally smaller than an underlying trainer API.
AdaptStack owns experiment methodology and provenance, while integrations own
framework-specific execution.
