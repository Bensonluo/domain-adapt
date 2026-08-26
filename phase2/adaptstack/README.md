# AdaptStack

AdaptStack is a domain-adaptation methodology and toolchain for small language
models. It organizes the full lifecycle—data preparation, continued
pretraining (CPT), supervised fine-tuning (SFT), preference alignment,
inference, and evaluation—behind explicit, replaceable interfaces.

The project is intentionally **domain-agnostic**. Medical adaptation is the
first research track; a legal configuration demonstrates that domain-specific
behavior belongs in configuration and plugins rather than orchestration code.

> Status: Week 22 architecture scaffold. The CLI, configuration validation,
> planning, registry, and deterministic dry-run manifests work today. Real
> model training backends are intentionally not claimed yet.

## Why AdaptStack?

General training frameworks answer “can this model be trained?” AdaptStack
focuses on “how should a domain-adaptation experiment be designed, evaluated,
and reproduced?” It can integrate tools such as Transformers, TRL,
LLaMA-Factory, Unsloth, vLLM, or bespoke trainers behind stable stage contracts.

## Architecture

```text
src/adaptstack/
├── data/          # L1: raw → clean → task datasets
├── training/      # L2: CPT → SFT → DPO/GRPO/Distill
├── inference/     # L3: RAG, serving, quantization interfaces
├── eval/          # L4: benchmark, judge, and human-eval interfaces
├── core/          # stage contracts, registry, context, manifests
├── tracking/      # experiment tracking boundary
├── config.py      # strict YAML configuration
├── pipeline.py    # deterministic orchestration
└── cli.py         # doctor, plan, and run commands
```

See [Architecture](docs/architecture.md) and
[Module interfaces](docs/module-interfaces.md) for the contracts.

## Quick start

```bash
cd adaptstack
python -m venv .venv
. .venv/bin/activate
pip install -e .

adaptstack doctor
adaptstack plan --config configs/medical.yaml
adaptstack run --config configs/medical.yaml --dry-run
```

Without installation, the repository script works directly:

```bash
python scripts/train.py --config configs/medical.yaml --show-plan
python scripts/train.py --config configs/medical.yaml --dry-run
python scripts/train.py --config configs/legal.yaml --dry-run \
  --output artifacts/legal-smoke
```

Dry-run writes one atomic JSON manifest per stage plus a run summary. It never
loads a model, contacts a tracker, or downloads data. A command without
`--dry-run` fails closed until a concrete execution backend is registered.

## Configuration

Every experiment defines:

- project/domain identity and output location;
- base model and data preparation policy;
- individually enabled CPT, SFT, DPO, GRPO, and distillation stages;
- benchmark evaluation and experiment tracking policy.

Shared methodology fields are strictly validated. Backend-specific escape-hatch
settings must be placed under an explicit `backend_options` mapping.

The same pipeline planner is used for
[`medical.yaml`](configs/medical.yaml) and [`legal.yaml`](configs/legal.yaml).

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts demo tests
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before adding a backend. New components
must preserve offline dry-run behavior and add provenance fields to artifacts.

The optional demo is installed with an audited direct constraint:

```bash
pip install -e '.[demo]' -c constraints/demo.txt
python demo/app.py
```

Review [SECURITY.md](SECURITY.md) before exposing the demo or adding a networked
integration.

## Safety and limitations

AdaptStack is research software. The scaffold does not provide medical or legal
advice, and generated outputs must not be used for clinical or legal decisions.
Real evaluations must disclose judge identity, dataset provenance, uncertainty,
and known leakage risks.

## License

MIT
