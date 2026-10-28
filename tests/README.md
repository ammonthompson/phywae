# Normalization refactor baseline

Run the focused regression benchmark from the repository root, using the existing
development environment:

```bash
.venv/bin/python -m pytest -q tests/test_normalization_contract.py tests/test_pipeline_regression.py
```

Run the complete suite with `.venv/bin/python -m pytest -q`.

`test_normalization_contract.py` pins explicit numerical expectations for tree,
character, and auxiliary scaling. It covers sample versus population variance,
padding masks, observed zeros and negative values, categorical preservation,
Fortran-order layout, constants, sparse observations, NaNs, training-only fitting,
column ordering, and inverse transforms.

`fixtures/pipeline_regression.json` contains numerical outputs captured from the
scikit-learn implementation before the refactor. Its metadata records the source
commit, dependency versions, seeds, CPU settings, and tolerances. It is a fixed
reference, and ordinary test runs never update it. Keep it unchanged during the
refactor; do not regenerate expected results from the replacement implementation.

`test_pipeline_regression.py` creates small deterministic HDF5 datasets for trees
without characters, continuous characters, and categorical characters. It checks:

- Exact split membership/order, batch shapes/dtypes, and reordered auxiliary
  columns, including a `num_taxa` column outside position zero.
- Fitted statistics and normalized/inverse-transformed inputs.
- Two CPU training epochs with reconstruction and MMD losses, all resulting model
  weights/buffers, and training/validation component histories.
- Encodings, reconstructions, and decoded fixed latent vectors.
- Freshly saved/reloaded models, inference from fresh trainer checkpoints, and
  resumed training matching uninterrupted training exactly.
- Numeric CSV/HDF5 outputs from the installed `encode`, `decode`, `predict`, and
  seeded `generate` commands, including categorical one-hot output and padding.

Floating outputs use `rtol=2e-5`, `atol=2e-6` to accommodate small float32 platform
differences. Indices, shapes, dtypes, and names must match exactly. The CPU benchmark
uses one thread with MKLDNN disabled and restores test process settings afterward.
It does not establish GPU parity, performance, or historical file compatibility.
Model/checkpoint files are created during each run and are not retained as fixtures.

Phyddle is an upstream data-generation tool, not a runtime requirement of
`src/phywae`. The analysis scripts under `scripts/` are excluded from both the
wheel and source distribution and are outside this benchmark. Dependency removal
does not change the fixed pre-refactor numerical reference. The core environment
does not require Phyddle or scikit-learn; historical names in the fixture record
the implementation used to capture that reference.
