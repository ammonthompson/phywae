"""Data preparation must reuse checkpoint statistics instead of fitting again."""

import h5py
import numpy as np
import pytest

from phywae import DataProcessors
from phywae.DataProcessors import AEData


@pytest.fixture
def training_files(tmp_path):
    paths = [tmp_path / "original.hdf5", tmp_path / "shifted.hdf5"]
    tips = np.array([2, 3] * 4)
    original = np.zeros((8, 3, 3), dtype=np.float32)
    for row, count in enumerate(tips):
        original[row, :2, :count] = (row + np.arange(count)) / 20
        original[row, 2, :count] = row + np.arange(count) - 4
    shifted = original.copy()
    for row, count in enumerate(tips):
        shifted[row, :2, :count] += 0.2
        shifted[row, 2, :count] += 20
    for path, phy, offset in zip(paths, (original, shifted), (0, 100)):
        with h5py.File(path, "w") as h5:
            h5["phy_data"] = phy.reshape(8, -1, order="F")
            h5["aux_data"] = np.column_stack((np.arange(8) + offset, tips)).astype(np.float32)
            h5["aux_data_names"] = np.array([[b"feature", b"num_taxa"]])
    settings = dict(prop_train=0.75, num_channels=3, num_chars=1,
                    char_data_type="continuous", max_tips=3, seed=17)
    source = AEData(str(paths[0]), **settings)
    yield paths, settings, source, shifted, tips
    source.train_dataset.close()
    source.val_dataset.close()


def test_supplied_normalizers_skip_training_reads_and_fitting(training_files, monkeypatch):
    paths, settings, source, shifted, tips = training_files
    phy_normalizer, aux_normalizer = source.get_normalizers()
    original_mean = aux_normalizer.mean_.copy()

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Resumed data must not read training arrays or refit normalizers")

    with monkeypatch.context() as patch:
        patch.setattr(AEData, "_fit_normalizers", unexpected_fit)
        patch.setattr(DataProcessors, "_read_phy", unexpected_fit)
        patch.setattr(DataProcessors, "_read_aux", unexpected_fit)
        resumed = AEData(str(paths[1]), **settings,
                         phy_normalizer=phy_normalizer, aux_normalizer=aux_normalizer)

    assert resumed.get_normalizers() == (phy_normalizer, aux_normalizer)
    row = resumed.train_dataset.indices[0]
    phy, aux, num_tips = resumed.train_dataset[0]
    expected_phy = phy_normalizer.transform(shifted[row].reshape(1, -1, order="F"))
    np.testing.assert_allclose(phy.numpy(), expected_phy.reshape(3, 3, order="F"))
    np.testing.assert_allclose(aux.numpy(),
                               (np.array([row + 100, tips[row]]) - original_mean) / aux_normalizer.scale_)
    np.testing.assert_array_equal(aux_normalizer.mean_, original_mean)
    assert num_tips.item() == tips[row]
    resumed.train_dataset.close()
