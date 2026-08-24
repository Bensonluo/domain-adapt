# Security policy

## Supported code

The current development version receives security fixes. This Week 22 scaffold
does not run training backends, download models or data, or contact trackers in
dry-run mode.

## Dependency policy

- Runtime dependencies stay minimal and use safe APIs.
- The optional Gradio demo uses an exact, audited direct dependency in both
  `pyproject.toml` and `constraints/demo.txt`.
- Re-audit the demo dependency and regenerate deployment-specific transitive
  locks before exposing the app to a network. Review at least monthly while a
  public demo is deployed.
- Do not enable public sharing, arbitrary file serving, or untrusted upload
  paths without a separate threat review.

## Reporting

Do not open a public issue for a suspected vulnerability. Contact the repository
maintainer privately with affected versions, reproduction steps, impact, and a
suggested mitigation when available.
