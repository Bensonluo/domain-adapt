# Module interfaces

## PipelineStage

```python
class PipelineStage(Protocol):
    name: str
    kind: str

    def run(self, context: RunContext) -> StageResult: ...
```

`RunContext` is immutable from the stage's perspective and exposes domain,
base-model identity, run/output directories, configuration digest, dry-run
state, and upstream artifact references.

Known methodology settings are schema-validated. Framework-specific settings
belong under `backend_options` so extensions remain explicit without weakening
validation of the shared contract.

## StageResult

A result contains the stage name, status, output artifact references, metrics,
and metadata. Scaffold stages produce `planned` results; real stages should use
`completed` only after their outputs pass integrity validation.

## InferenceBackend

Inference backends declare a stable name and implement `generate(request)`. A
backend must document decoding defaults, model identity, and whether it sends
data to an external service.

## Evaluator

Evaluators consume a model/inference backend plus a dataset reference and return
structured metrics. Statistical uncertainty belongs in the evaluator result,
not only in a Markdown report.

## RunTracker

Trackers receive sanitized configuration, stage results, and final summaries.
The default `LocalRunTracker` writes JSON only. Networked trackers must be enabled
explicitly and must never receive secrets or private examples by default.
