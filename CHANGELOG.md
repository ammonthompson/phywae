# Changelog

## 0.3.0

### Breaking changes

- Models, checkpoints, and normalizers saved before 0.3 with the old
  scikit-learn-based normalization format are incompatible. Regenerate them or
  explicitly convert them to the new format before loading.

### Changed

- Replaced scikit-learn normalization with NumPy utilities while preserving
  scaling, masking, padding, and seeded data-splitting behavior.
- Save fitted normalization as versioned numeric state in models and checkpoints.
- Select dependencies by platform: Intel Macs use PyTorch 2.2.2 and NumPy
  `>=1.26.4,<2`; other supported platforms, including Apple Silicon Macs, use
  PyTorch `>=2.6,<3` and NumPy `>=2.2.2,<3`.
- Use `pyproject.toml` and pip for installation and development dependencies.

### Fixed

- Resumed training reuses checkpoint normalization statistics without refitting
  them on the supplied data.

### Added

- Fixed numerical regression baselines covering normalization, training,
  model persistence, checkpoint resume, and CLI inference workflows.

### Removed

- Scikit-learn and Phyddle runtime dependencies. Phyddle remains a separate
  upstream data-preparation application.
- The Phyddle-specific `setuptools<82` runtime constraint.
- The uv lockfile and uv-specific installation instructions.
