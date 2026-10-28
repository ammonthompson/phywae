"""Numerical contracts captured before replacing scikit-learn normalization.

Expected values are independent of the fitted implementation. Keep them fixed
during the refactor; these tests intentionally require no scikit-learn imports.
"""

import h5py
import numpy as np
import pytest

from phywae import utils
from phywae.DataProcessors import AEData


def test_categorical_scaling_preserves_characters_and_padding():
    structured = np.array(
        [
            [[0, 2, 0], [0, 5, 0], [1, 0, 0], [0, 1, 0]],
            [[2, 4, 7], [2, 5, 0], [0, 1, 1], [1, 0, 0]],
            [[4, 0, 0], [4, 5, 0], [1, 0, 0], [0, 1, 0]],
        ],
        dtype=np.float32,
    )
    flat = structured.reshape(3, -1, order="F")
    scaler = utils.StandardScalerPhyCategorical(2, 4, 3).fit(flat)
    mean, scale = scaler.tree_statistics()
    np.testing.assert_allclose(mean, [[2, 3, 0], [2, 5, 0]])
    np.testing.assert_allclose(scale, [[2, np.sqrt(2), 1], [2, 1, 1]])

    expected = structured.copy()
    expected[:, :2] = [
        [[-1, -1 / np.sqrt(2), 0], [-1, 0, 0]],
        [[0, 1 / np.sqrt(2), 7], [0, 0, 0]],
        [[1, -3 / np.sqrt(2), 0], [1, 0, 0]],
    ]
    transformed = scaler.transform(flat)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(
        transformed, expected.reshape(3, -1, order="F"), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(
        transformed.reshape(structured.shape, order="F")[:, 2:], structured[:, 2:]
    )
    np.testing.assert_allclose(scaler.inverse_transform(transformed), flat, atol=1e-6)


def _continuous_input():
    structured = np.array(
        [
            [[0, 2, 0], [-2, 4, 0], [-2, 0, 0], [5, 5, 0]],
            [[2, 4, 6], [0, 6, 8], [2, 4, 6], [5, 5, 5]],
            [[4, 0, 0], [2, 0, 0], [8, 0, 0], [5, 0, 0]],
        ],
        dtype=np.float32,
    )
    mask = np.broadcast_to(
        np.arange(3)[None, None, :] < np.array([2, 3, 1])[:, None, None],
        structured.shape,
    ).copy()
    return structured.reshape(3, -1, order="F"), mask.reshape(3, -1, order="F")


def test_continuous_scaling_preserves_masked_sample_and_population_statistics():
    flat, mask = _continuous_input()
    scaler = utils.StandardScalerPhyContinuous(2, 4, 3).fit(flat, mask=mask)
    mean, scale = scaler.tree_statistics()
    np.testing.assert_allclose(mean, [[2, 3, 0], [0, 5, 0]])
    np.testing.assert_allclose(scale, [[2, np.sqrt(2), 1], [2, np.sqrt(2), 1]])

    # Six observed values [-2, 0, 2, 4, 6, 8] share one population scale.
    char_scale = np.sqrt(35 / 3)
    expected = np.array(
        [
            [
                [-1, -1 / np.sqrt(2), 0],
                [-1, -1 / np.sqrt(2), 0],
                [-5 / char_scale, -3 / char_scale, -3 / char_scale],
                [0, 0, -5],
            ],
            [
                [0, 1 / np.sqrt(2), 6],
                [0, 1 / np.sqrt(2), 8],
                [-1 / char_scale, 1 / char_scale, 3 / char_scale],
                [0, 0, 0],
            ],
            [
                [1, -3 / np.sqrt(2), 0],
                [1, -5 / np.sqrt(2), 0],
                [5 / char_scale, -3 / char_scale, -3 / char_scale],
                [0, -5, -5],
            ],
        ],
        dtype=np.float32,
    )
    transformed = scaler.transform(flat)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(
        transformed, expected.reshape(3, -1, order="F"), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(scaler.inverse_transform(transformed), flat, atol=1e-6)


def test_continuous_scaling_unobserved_character_is_identity():
    flat, mask = _continuous_input()
    mask[:, 3::4] = False
    scaler = utils.StandardScalerPhyContinuous(2, 4, 3).fit(flat, mask=mask)

    np.testing.assert_array_equal(scaler.transform(flat)[:, 3::4], flat[:, 3::4])


@pytest.mark.parametrize("mask", [None, np.ones((3, 1), dtype=bool)])
def test_continuous_scaling_requires_matching_mask(mask):
    flat, _ = _continuous_input()
    with pytest.raises(ValueError, match="mask must have the same shape as X"):
        utils.StandardScalerPhyContinuous(2, 4, 3).fit(flat, mask=mask)


@pytest.fixture
def auxiliary_data(tmp_path):
    path = tmp_path / "normalization.hdf5"
    with h5py.File(path, "w") as h5:
        h5["phy_data"] = np.ones((6, 4), dtype=np.float32)
        h5["aux_data"] = np.column_stack(
            ([2] * 6, [-2, 0, 2, 4, 6, 8], [7] * 6, [np.nan, 0, 2, 4, np.nan, 8])
        ).astype(np.float32)
        h5["aux_data_names"] = np.array(
            [[b"num_taxa", b"varying", b"constant", b"mixed"]]
        )
    data = AEData(
        str(path),
        prop_train=2 / 3,
        num_channels=2,
        char_data_type="continuous",
        seed=17,
        max_tips=2,
        which_aux=["mixed", "num_taxa", "constant", "varying"],
    )
    yield data
    data.train_dataset.close()
    data.val_dataset.close()


def test_auxiliary_scaling_uses_training_rows_and_preserves_nan(auxiliary_data):
    data = auxiliary_data
    np.testing.assert_array_equal(data.train_dataset.indices, [4, 0, 5, 1])
    np.testing.assert_array_equal(data.val_dataset.indices, [2, 3])
    np.testing.assert_array_equal(
        data.aux_colnames, ["mixed", "num_taxa", "constant", "varying"]
    )
    scaler = data.get_normalizers()[1]
    np.testing.assert_allclose(scaler.mean_, [4, 2, 7, 3])
    np.testing.assert_allclose(scaler.scale_, [4, 1, 1, np.sqrt(17)])
    raw = np.array([[np.nan, 2, 7, 6], [8, 2, 7, 8]], dtype=np.float32)
    expected = [[np.nan, 0, 0, 3 / np.sqrt(17)], [1, 0, 0, 5 / np.sqrt(17)]]
    transformed = scaler.transform(raw)
    assert transformed.dtype == np.float32
    np.testing.assert_allclose(transformed, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), raw, atol=1e-6)

    samples = data.val_dataset.__getitems__([0, 1])
    np.testing.assert_allclose(
        np.stack([aux.numpy() for _, aux, _ in samples]),
        [[-0.5, 0, 0, -1 / np.sqrt(17)], [0, 0, 0, 1 / np.sqrt(17)]],
        rtol=1e-6,
        atol=1e-7,
    )


def test_auxiliary_scaling_nearly_constant_float64_features(auxiliary_data):
    # This variance is nonzero but below the current numerical constant bound.
    values = np.array([[1e12 - 2**-12], [1e12 + 2**-12]] * 2)
    scaler = auxiliary_data.get_normalizers()[1].fit(values)
    np.testing.assert_array_equal(scaler.mean_, [1e12])
    np.testing.assert_array_equal(scaler.scale_, [1])
    transformed = scaler.transform(values)
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(transformed[:, 0], [-2**-12, 2**-12] * 2)
    np.testing.assert_array_equal(scaler.inverse_transform(transformed), values)
