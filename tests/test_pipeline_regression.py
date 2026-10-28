"""Numerical baseline captured before replacing scikit-learn normalization."""

from contextlib import contextmanager
import json
from pathlib import Path
import random

import h5py
import numpy as np
import pytest
import torch
from torch.optim import AdamW

from phywae import cli, utils
from phywae.DataProcessors import AEData
from phywae.PhyloAEModel import AECNN
from phywae.PhyloAutoencoder import AETrainer
from phywae.PhyLoss import PhyLoss


BASELINE = Path(__file__).parent / "fixtures" / "pipeline_regression.json"
CASES = {"tree_only": ("continuous", 0), "continuous": ("continuous", 2),
         "categorical": ("categorical", 3)}
SEED = 731
WIDTH = 4
RTOL = 2e-5
ATOL = 2e-6


@contextmanager
def _reference_cpu():
    """Keep the reference run small and restore process settings afterward."""
    threads = torch.get_num_threads()
    mkldnn = torch.backends.mkldnn.enabled
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            torch.set_num_threads(1)
            torch.backends.mkldnn.enabled = False
            yield
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_num_threads(threads)
            torch.backends.mkldnn.enabled = mkldnn
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


@pytest.fixture(scope="module", autouse=True)
def reference_cpu():
    with _reference_cpu():
        yield


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "values": value.tolist()}


def _assert_snapshot(actual, expected, path="baseline"):
    assert actual.keys() == expected.keys(), path
    if "values" in expected:
        assert actual["dtype"] == expected["dtype"], path
        assert actual["shape"] == expected["shape"], path
        actual_values = np.asarray(actual["values"])
        expected_values = np.asarray(expected["values"])
        if np.issubdtype(np.dtype(expected["dtype"]), np.integer):
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=path)
        else:
            np.testing.assert_allclose(actual_values, expected_values,
                                       rtol=RTOL, atol=ATOL, err_msg=path)
        return
    for key in expected:
        if isinstance(expected[key], dict):
            _assert_snapshot(actual[key], expected[key], f"{path}.{key}")
        else:
            assert actual[key] == expected[key], f"{path}.{key}"


def _write_input(path, phy, aux):
    with h5py.File(path, "w") as h5:
        h5["phy_data"] = phy.reshape(len(phy), -1, order="F")
        h5["aux_data"] = aux
        h5["aux_data_names"] = np.asarray(
            [[b"num_taxa", b"constant", b"feature", b"unused"]], dtype="S32"
        )


def _input_arrays(char_type, num_chars):
    tips = np.array([2, 3, 4, 2, 4, 3, 4, 3, 2, 4, 2, 3])
    phy = np.zeros((len(tips), 2 + num_chars, WIDTH), dtype=np.float32)
    for row, count in enumerate(tips):
        positions = np.arange(count)
        phy[row, 0, :count] = (row + 1) / 20 + positions / 40
        phy[row, 1, :count] = (row + 1) / 25 + positions / 50
        if num_chars and char_type == "categorical":
            phy[row, 2 + (row + positions) % num_chars, positions] = 1
        elif num_chars:
            phy[row, 2:, :count] = (
                row - 5 + positions[None, :] / 2
                + np.arange(num_chars)[:, None] * 10
            )
    # An observed zero must be distinguished from padding in continuous mode.
    phy[0, 0, 0] = 0
    aux = np.column_stack((tips, np.full(len(tips), 7),
                           np.arange(len(tips)) / 4 - 1, np.arange(len(tips)) * 100))
    return phy, aux.astype(np.float32)


def _make_data(path, char_type, num_chars):
    data = AEData(str(path), prop_train=0.75, num_channels=2 + num_chars,
                  char_data_type=char_type, num_chars=num_chars,
                  seed=SEED, max_tips=WIDTH,
                  which_aux=["feature", "num_taxa", "constant"])
    return data, data.get_dataloaders(batch_size=3, shuffle=True, num_workers=0)


def _make_trainer(data, loaders):
    model = AECNN(
        num_structured_input_channel=data.num_channels,
        structured_input_width=WIDTH, unstructured_input_width=data.aux_width,
        unstructured_latent_width=2, aux_inner_dim=3,
        aux_numtips_idx=data.ntax_cidx, aux_data_names=data.aux_colnames,
        num_chars=data.num_chars, char_type=data.char_data_type,
        stride=[1, 1], kernel=[3, 3], out_channels=[2, 2], latent_output_dim=2,
        device="cpu", seed=SEED,
        phy_normalizer=data.phy_normalizer, aux_normalizer=data.aux_normalizer,
    )
    optimizer = AdamW(utils.split_params_by_wd(model, 1e-3), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=3, steps_per_epoch=len(loaders[0]),
        pct_start=0.1, anneal_strategy="cos", cycle_momentum=False,
    )
    loss = PhyLoss({"phy_loss_weight": 1.0,
                    "char_loss_weight": 0.75 if data.num_chars else 0.0,
                    "aux_loss_weight": 0.25, "mmd_loss_weight": 0.1},
                   data.ntax_cidx, data.char_data_type)
    trainer = AETrainer(model, optimizer, loss=loss, lr_scheduler=scheduler,
                        seed=SEED, device="cpu")
    trainer.set_data_loaders(*loaders)
    return trainer


def _inference(model, phy, aux):
    flat = phy.reshape(len(phy), -1, order="F")
    norm_phy = model.phy_normalizer.transform(flat)
    norm_aux = model.aux_normalizer.transform(aux)
    latent = model.norm_and_encode(phy, aux)
    pred_phy, pred_aux = model.norm_predict_denorm(phy, aux)
    gen_phy, gen_aux = model.decode_and_denorm(
        np.array([[-1.0, 0.5], [0.0, 0.0], [1.5, -0.5]], dtype=np.float32)
    )
    return {key: _array(value) for key, value in {
        "normalized_phy": norm_phy, "normalized_aux": norm_aux,
        "restored_phy": model.phy_normalizer.inverse_transform(norm_phy),
        "restored_aux": model.aux_normalizer.inverse_transform(norm_aux),
        "latent": latent, "prediction_phy": pred_phy, "prediction_aux": pred_aux,
        "fixed_latent_phy": gen_phy, "fixed_latent_aux": gen_aux,
    }.items()}


def _cli_outputs(model_file, query_file, output_dir):
    common = ["-m", str(model_file), "-o", str(output_dir / "query")]
    cli.main(["encode", *common, "-t", str(query_file)])
    encoded = output_dir / "query.ae_encoded.csv"
    cli.main(["decode", *common, "-e", str(encoded)])
    cli.main(["predict", *common, "-d", str(query_file), "-ofmt", "hdf5"])
    cli.main(["generate", "-m", str(model_file), "-o", str(output_dir / "sample"),
              "-n", "3", "--seed", "99"])
    outputs = {name: _array(np.loadtxt(output_dir / filename, delimiter=",", ndmin=2))
               for name, filename in {
                   "encoded": "query.ae_encoded.csv",
                   "decoded_phy": "query.ae_decoded.cblv.csv",
                   "decoded_aux": "query.ae_decoded.aux.csv",
                   "generated_phy": "sample.cblv.csv",
                   "generated_aux": "sample.aux.csv",
               }.items()}
    with h5py.File(output_dir / "query.hdf5", "r") as h5:
        outputs["prediction_phy"] = _array(h5["prediction/pred_phy"][...])
        outputs["prediction_aux"] = _array(h5["prediction/pred_aux"][...])
        outputs["aux_names"] = h5["prediction/aux_data_names"][...].astype(str).tolist()
    return outputs


def _capture_pipeline(directory, char_type, num_chars):
    directory.mkdir(parents=True, exist_ok=True)
    phy, aux = _input_arrays(char_type, num_chars)
    data_file = directory / "training.hdf5"
    _write_input(data_file, phy, aux)
    query_phy, query_aux = phy[[0, 5, 9]], aux[[0, 5, 9]]
    selected_aux = query_aux[:, [2, 0, 1]]
    query_file = directory / "input.hdf5"
    _write_input(query_file, query_phy, query_aux)

    data, loaders = _make_data(data_file, char_type, num_chars)
    samples = data.train_dataset.__getitems__([2, 0, 2, 1])
    batch = {name: _array(torch.stack(values)) for name, values in
             zip(("phy", "aux", "num_tips"), zip(*samples))}
    trainer = _make_trainer(data, loaders)
    trainer.train(3, seed=SEED)  # Two epochs under the current trainer API.
    inference = _inference(trainer.model, query_phy, selected_aux)

    model_file = directory / "model.pt"
    trainer.model.save_model(model_file)
    loaded = AECNN.load_pretrained_from_file(str(model_file), map_location="cpu")
    _assert_snapshot(_inference(loaded, query_phy, selected_aux), inference, "model_reload")

    # Create and consume checkpoints during this run; no historical pickle is retained.
    first_data, first_loaders = _make_data(data_file, char_type, num_chars)
    first = _make_trainer(first_data, first_loaders)
    first.train(2, seed=SEED)
    checkpoint = directory / "training.ckpt.pt"
    first.save_checkpoint(checkpoint)
    checkpoint_model = AECNN.load_pretrained_from_file(str(checkpoint), map_location="cpu")
    _assert_snapshot(_inference(checkpoint_model, query_phy, selected_aux),
                     _inference(first.model, query_phy, selected_aux), "checkpoint_inference")
    resumed_data, resumed_loaders = _make_data(data_file, char_type, num_chars)
    resumed = AETrainer.load_checkpoint(str(checkpoint), map_location="cpu")
    resumed.set_data_loaders(*resumed_loaders)
    resumed.train(3)
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    assert resumed.train_metrics.epoch_history == trainer.train_metrics.epoch_history
    assert resumed.val_metrics.epoch_history == trainer.val_metrics.epoch_history
    _assert_snapshot(_inference(resumed.model, query_phy, selected_aux), inference, "resume")

    tree_mean, tree_scale = data.phy_normalizer.tree_statistics()
    snapshot = {
        "train_indices": _array(data.train_dataset.indices),
        "validation_indices": _array(data.val_dataset.indices),
        "batch": batch,
        "statistics": {key: _array(value) for key, value in {
            "tree_mean": tree_mean, "tree_scale": tree_scale,
            "aux_mean": data.aux_normalizer.mean_, "aux_scale": data.aux_normalizer.scale_,
        }.items()},
        "trained_state": {name: _array(value) for name, value in trainer.model.state_dict().items()},
        "train_losses": {name: _array(value) for name, value in trainer.train_metrics.epoch_history.items()},
        "validation_losses": {name: _array(value) for name, value in trainer.val_metrics.epoch_history.items()},
        "inference": inference,
        "cli": _cli_outputs(model_file, query_file, directory),
    }
    for owner in (data, first_data, resumed_data):
        owner.train_dataset.close()
        owner.val_dataset.close()
    return snapshot


@pytest.mark.parametrize("case", CASES)
def test_pipeline_matches_pre_refactor_baseline(tmp_path, case):
    expected = json.loads(BASELINE.read_text())["cases"][case]
    actual = _capture_pipeline(tmp_path / case, *CASES[case])
    _assert_snapshot(actual, expected, case)
