import numpy as np
import pytest
import torch

from phywae import cli, utils
from phywae.PhyloAEModel import AECNN


def _make_model():
    phy = np.arange(64, dtype=np.float32).reshape(4, 16) / 80 + 0.1
    aux = np.array([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    return AECNN(
        num_structured_input_channel=2, structured_input_width=8,
        unstructured_input_width=1, aux_inner_dim=2, aux_numtips_idx=0,
        aux_data_names=np.array([b"num_taxa"]),
        stride=[1, 1], kernel=[3, 3], out_channels=[4, 4],
        latent_output_dim=4, device="cpu", seed=731,
        phy_normalizer=utils.StandardScalerPhyCategorical(0, 2, 8).fit(phy),
        aux_normalizer=utils.StandardScaler().fit(aux),
    ), phy, aux


def test_model_artifact_stores_numeric_normalizer_state(tmp_path, monkeypatch, capsys):
    model, phy, aux = _make_model()
    expected = model.norm_and_encode(phy.reshape(4, 2, 8, order="F"), aux)

    def reject_pickling_scaler(self, protocol):
        raise AssertionError("A fitted scaler object must not be pickled")

    for scaler_class in (utils.StandardScaler, utils.StandardScalerPhyCategorical):
        monkeypatch.setattr(scaler_class, "__reduce_ex__", reject_pickling_scaler)
    path = tmp_path / "model.pt"
    model.save_model(path)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    config = artifact["model_config"]
    assert config["aux_data_names"] == ["num_taxa"]
    for name in ("phy_normalizer", "aux_normalizer"):
        assert isinstance(config[name], dict)
        assert config[name]["version"] == 1
    loaded = AECNN.load_pretrained_from_file(path, map_location="cpu")
    np.testing.assert_array_equal(
        loaded.norm_and_encode(phy.reshape(4, 2, 8, order="F"), aux), expected
    )
    assert not loaded.training
    assert capsys.readouterr().out == ""


def test_reloaded_model_encodes_csv_inputs(tmp_path):
    model, phy, aux = _make_model()
    model_file, phy_file, aux_file = (tmp_path / name for name in ("model.pt", "phy.csv", "aux.csv"))
    model.save_model(model_file)
    np.savetxt(phy_file, phy, delimiter=",")
    np.savetxt(aux_file, aux, delimiter=",")
    prefix = tmp_path / "encoded"
    cli.main(["encode", "-m", str(model_file), "-t", str(phy_file),
              "-s", str(aux_file), "-nc", "2", "-mt", "8", "-o", str(prefix)])
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / "encoded.ae_encoded.csv", delimiter=","),
        model.norm_and_encode(phy.reshape(4, 2, 8, order="F"), aux),
        rtol=1e-6, atol=1e-7,
    )


def test_validate_num_tips_rejects_trees_wider_than_model():
    model, _, _ = _make_model()
    with pytest.raises(ValueError, match="supports at most 8"):
        model.validate_num_tips(np.array([[9.0]]))


@pytest.mark.parametrize("key", ["phy_two_sided_ReLU.min_val", "parameter"])
def test_model_load_rejects_missing_state(tmp_path, key):
    model, _, _ = _make_model()
    path = tmp_path / "model.pt"
    model.save_model(path)
    artifact = torch.load(path, weights_only=True)
    if key == "parameter":
        key = next(iter(dict(model.named_parameters())))
    artifact["model_state_dict"].pop(key)
    torch.save(artifact, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        AECNN.load_pretrained_from_file(path)


def test_model_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model_format_version": -1}, path)
    with pytest.raises(ValueError, match="Unsupported model format"):
        AECNN.load_pretrained_from_file(path)
