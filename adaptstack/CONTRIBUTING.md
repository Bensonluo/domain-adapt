# Contributing to AdaptStack

Thank you for helping build a reproducible domain-adaptation toolchain.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

## Design rules

1. Keep orchestration domain-agnostic. Domain logic belongs in configuration or
   a registered plugin.
2. Preserve the stage contract: declared inputs, deterministic plan position,
   explicit outputs, and a structured result.
3. Fail closed. A missing real backend must raise a useful error rather than
   silently produce mock research evidence.
4. Keep dry-run offline and dependency-light.
5. Record provenance: configuration digest, stage name, upstream artifacts,
   implementation identity, and run ID.
6. Never label an AI review as human or expert review.

## Pull requests

- Keep one logical change per commit.
- Add tests for configuration or orchestration changes.
- Run unit tests, compilation, and both example dry-runs.
- Document new configuration keys and backward-incompatible changes.
- Do not commit model weights, credentials, private datasets, or WandB state.
- Update `constraints/demo.txt`, document the audit date, and review upstream
  advisories whenever changing the optional Web dependency.

## Adding a stage

Implement `PipelineStage`, register it under a stable name, document its config,
and test both enabled and disabled behavior. Real implementations should live in
integration packages when they require heavyweight training dependencies.
