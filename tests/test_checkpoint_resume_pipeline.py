import random

import h5py
import numpy as np
import pytest
import torch
from torch.optim import AdamW

from phywae import cli, utils
from phywae.DataProcessors import AEData
from phywae.PhyloAEModel import AECNN
from phywae.PhyloAutoencoder import PhyloAutoencoder, _LossMetricTracker
from phywae.PhyLoss import PhyLoss


SEED = 731
NUM_ROWS = 24
MAX_TIPS = 8
NUM_CHANNELS = 2
BATCH_SIZE = 6
TOTAL_EPOCH_ARG = 4


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _write_small_dataset(path):
    rng = np.random.default_rng(2026)
    num_taxa = rng.integers(4, MAX_TIPS + 1, size=NUM_ROWS)
    phy = np.zeros((NUM_ROWS, NUM_CHANNELS, MAX_TIPS), dtype=np.float32)
    for row, tips in enumerate(num_taxa):
        phy[row, :, :tips] = rng.uniform(
            0.05, 0.95, size=(NUM_CHANNELS, tips)
        )

    aux = np.column_stack(
        (num_taxa.astype(np.float32), np.arange(NUM_ROWS, dtype=np.float32))
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("phy_data", data=phy.reshape(NUM_ROWS, -1, order="F"))
        h5.create_dataset("aux_data", data=aux)
        h5.create_dataset(
            "aux_data_names",
            data=np.asarray([[b"num_taxa", b"row_id"]], dtype="S32"),
        )


def _make_data_and_loaders(data_file):
    data = AEData(
        hdf5_file=str(data_file),
        prop_train=0.75,
        num_channels=NUM_CHANNELS,
        char_data_type="continuous",
        num_chars=0,
        seed=SEED,
        max_tips=MAX_TIPS,
    )
    loaders = data.get_dataloaders(
        BATCH_SIZE, shuffle=True, num_workers=0
    )
    return data, loaders


def _make_trainer(data, loaders, checkpoint_prefix=None):
    phy_normalizer, aux_normalizer = data.get_normalizers()
    model = AECNN(
        num_structured_input_channel=NUM_CHANNELS,
        structured_input_width=MAX_TIPS,
        unstructured_input_width=data.aux_width,
        aux_inner_dim=4,
        aux_numtips_idx=data.ntax_cidx,
        aux_data_names=data.aux_colnames,
        stride=[1, 1],
        kernel=[3, 3],
        out_channels=[4, 4],
        latent_output_dim=4,
        num_chars=0,
        char_type="continuous",
        device="cpu",
        phy_normalizer=phy_normalizer,
        aux_normalizer=aux_normalizer,
    )
    optimizer = AdamW(utils.split_params_by_wd(model, 1e-3), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=TOTAL_EPOCH_ARG,
        steps_per_epoch=len(loaders[0]),
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 0.0,
        "aux_loss_weight": 0.1,
        "mmd_loss_weight": 0.0,
    }
    loss = PhyLoss(
        weights,
        data.ntax_cidx,
        model.char_type,
    )
    trainer = PhyloAutoencoder(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        loss=loss,
        seed=SEED,
        device="cpu",
        checkpoints=[1] if checkpoint_prefix is not None else None,
        checkpt_file_prefix=(
            str(checkpoint_prefix) if checkpoint_prefix is not None else "unused"
        ),
    )
    trainer.set_data_loaders(*loaders)
    return trainer


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    elif torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


def _loader_order(loader):
    return torch.cat([batch[1][:, 1] for batch in loader]).tolist()


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    data_file = tmp_path / "training.hdf5"
    checkpoint_prefix = tmp_path / "resume"
    checkpoint_file = tmp_path / "resume_epoch_1.ckpt.pt"
    _write_small_dataset(data_file)

    _seed_everything()
    full_data, full_loaders = _make_data_and_loaders(data_file)
    uninterrupted = _make_trainer(full_data, full_loaders)
    uninterrupted.train(TOTAL_EPOCH_ARG, seed=SEED)

    _seed_everything()
    first_data, first_loaders = _make_data_and_loaders(data_file)
    first_segment = _make_trainer(
        first_data, first_loaders, checkpoint_prefix
    )
    first_segment.train(2, seed=SEED)
    assert checkpoint_file.exists()

    checkpoint_model = AECNN.load_pretrained_from_file(
        checkpoint_file, map_location="cpu"
    )
    _assert_nested_equal(
        checkpoint_model.state_dict(), first_segment.model.state_dict()
    )
    assert not checkpoint_model.training
    assert "latent_layer_type" not in checkpoint_model.get_config_dict()
    assert "out_prefix" not in checkpoint_model.get_config_dict()
    assert not hasattr(checkpoint_model, "out_prefix")

    resumed_data, resumed_loaders = _make_data_and_loaders(data_file)
    resumed = PhyloAutoencoder.load_checkpoint(
        checkpoint_file, map_location="cpu"
    )
    assert {id(parameter) for parameter in resumed.model.parameters()} == {
        id(parameter) for group in resumed.optimizer.param_groups for parameter in group["params"]
    }
    assert resumed.lr_sched.optimizer is resumed.optimizer
    assert resumed.model.phy_two_sided_ReLU.min_val.device.type == "cpu"
    resumed.set_data_loaders(*resumed_loaders)
    resumed.train(TOTAL_EPOCH_ARG)

    assert uninterrupted.epoch == resumed.epoch == 3
    _assert_nested_equal(
        resumed.model.state_dict(), uninterrupted.model.state_dict()
    )
    _assert_nested_equal(
        resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict()
    )
    _assert_nested_equal(
        resumed.lr_sched.state_dict(), uninterrupted.lr_sched.state_dict()
    )
    assert resumed.train_metrics.epoch_history == (
        uninterrupted.train_metrics.epoch_history
    )
    assert resumed.val_metrics.epoch_history == (
        uninterrupted.val_metrics.epoch_history
    )

    assert resumed.checkpoints == [1]
    assert resumed.checkpt_file_prefix == str(checkpoint_prefix)
    assert resumed._pending_data_loader_rng_state is None
    assert _loader_order(resumed.train_loader) == _loader_order(
        uninterrupted.train_loader
    )


def test_save_checkpoint_refuses_to_overwrite_existing_file(tmp_path):
    checkpoint_file = tmp_path / "existing.ckpt.pt"
    checkpoint_file.write_bytes(b"existing checkpoint")
    trainer = object.__new__(PhyloAutoencoder)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.save_checkpoint(checkpoint_file)

    assert checkpoint_file.read_bytes() == b"existing checkpoint"


def test_cli_resume_reuses_saved_normalization_on_shifted_data(tmp_path, monkeypatch):
    data_file = tmp_path / "training.hdf5"
    _write_small_dataset(data_file)
    _seed_everything()
    data, loaders = _make_data_and_loaders(data_file)
    trainer = _make_trainer(data, loaders)
    trainer.train(2, seed=SEED)
    checkpoint = tmp_path / "training.ckpt.pt"
    trainer.save_checkpoint(checkpoint)
    expected_tree_stats = trainer.model.phy_normalizer.tree_statistics()
    expected_aux_mean = trainer.model.aux_normalizer.mean_.copy()
    data.train_dataset.close()
    data.val_dataset.close()
    with h5py.File(data_file, "r+") as h5:
        h5["phy_data"][...] *= 0.5
        aux = h5["aux_data"][...]
        aux[:, 1] += 10
        h5["aux_data"][...] = aux

    def reject_refitting(*args, **kwargs):
        pytest.fail("Checkpoint resume must reuse saved normalizers")

    monkeypatch.setattr(AEData, "_fit_normalizers", reject_refitting)
    config = tmp_path / "resume_config.py"
    prefix = tmp_path / "resumed"
    config.write_text("settings = " + repr({
        "max_tips": MAX_TIPS, "num_channels": NUM_CHANNELS,
        "char_type": "continuous", "num_chars": 0,
        "proportion_train": 0.75, "batch_size": BATCH_SIZE,
        "num_epochs": 3, "device": "cpu", "out_prefix": str(prefix),
        "resume_from_checkpoint": str(checkpoint), "checkpoints": None,
    }))
    cli.main(["train", "-d", str(data_file), "--config", str(config)])
    loaded = AECNN.load_pretrained_from_file(str(prefix) + ".ae_trained.pt")
    np.testing.assert_array_equal(loaded.aux_normalizer.mean_, expected_aux_mean)
    for actual, expected in zip(loaded.phy_normalizer.tree_statistics(), expected_tree_stats):
        np.testing.assert_array_equal(actual, expected)


def test_checkpoint_rejects_unknown_format(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"checkpoint_version": -1}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        PhyloAutoencoder.load_checkpoint(path)


def test_loss_metric_tracker_restores_pending_and_completed_metrics():
    metrics = _LossMetricTracker()
    batch = {name: torch.tensor(2.0, requires_grad=True)
             for name in ("total", "phy", "char", "aux", "mmd")}
    metrics.record_batch(batch)
    metrics.finalize_epoch()
    metrics.record_batch(batch)
    restored = _LossMetricTracker()
    restored.load_state_dict(metrics.state_dict())
    restored.finalize_epoch()
    assert restored.epoch_history == {name: [2.0, 2.0] for name in batch}


@pytest.mark.parametrize("num_kernels", [0, 2])
def test_mmd_kernel_count_must_be_positive_and_odd(num_kernels):
    weights = {
        "phy_loss_weight": 1.0,
        "char_loss_weight": 0.0,
        "aux_loss_weight": 0.1,
        "mmd_loss_weight": 1.0,
    }
    assert PhyLoss(weights, ntax_cidx=0, mmd_num_kernels=5).mmd.num_kernels == 5
    with pytest.raises(ValueError, match="positive odd integer"):
        PhyLoss(weights, ntax_cidx=0, mmd_num_kernels=num_kernels)
