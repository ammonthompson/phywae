# PhyWAE 0.2.0
![status](https://img.shields.io/badge/status-active--development-orange)

<p align="center">
    <img src="docs/images/latent.png" width="300" style="vertical-align: middle;" />
    <img src="docs/images/tree_morph.gif" width="400" style="vertical-align: middle;" />
</p>

**PhyWAE** is a research tool for exploring the use of autoencoders to characterize the probability distribution of **phylogenetic data**, including phylogenies, tip-associated data, and other tree metadata.

It implements a Wasserstein autoencoder with maximum mean discrepancy regularization (**MMD-WAE**) in **PyTorch**.
Tree, character, and auxiliary normalization use NumPy utilities owned by PhyWAE.
Model files include fitted normalization state; new models and checkpoints must be
created after upgrading from the previous scikit-learn-based format.

## Installation

Requires Python 3.12 or newer. From the cloned repo's root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development, use `python -m pip install -e . --group dev` and run
`python -m pytest -q`.

Optional development setup with locked dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --dev
uv run pytest -q
```

## Training

Training reads HDF5 data in [Phyddle's format](https://phyddle.org/pipeline.html#format).
Phyddle is an upstream data-generation tool and is not installed with PhyWAE.
Install and run it in a separate environment to generate example BiSSE data:

```bash
cd example/phyddle_sim_data
phyddle -c phyddle_config.py
cd ../..
```

Return to the PhyWAE environment before training. Research scripts in `scripts/`
are outside the distributed package and manage their own dependencies.

Edit [`phytrain_config.py`](phytrain_config.py) for your dataset and model:

```bash
phywae train --trn_data path/to/training.hdf5 --config phytrain_config.py
```

Add `--optimize_data` for faster shuffled reads. This creates or reuses an uncompressed `training.phywae.hdf5` copy, preserving the source and requiring additional disk space.

## Encoding, Reconstruction, and Sampling

- `phywae encode`: map input data to latent coordinates.
- `phywae decode`: decode latent coordinates into data.
- `phywae predict`: encode and reconstruct data in one step.
- `phywae generate`: generate data by decoding samples from a standard normal prior.

Use `phywae --help` to list commands and `phywae <command> --help` for options.
