"""Native scaler numerics and persistence, independent of scikit-learn."""

import numpy as np
import pytest
import torch

from phywae import utils


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_standard_scaler_population_statistics_nan_dtype_and_copy(dtype):
    values = np.array(
        [[1, 7, np.nan, np.nan], [3, 7, 0, np.nan],
         [5, 7, 4, np.nan], [7, 7, np.nan, np.nan]], dtype=dtype
    )
    original = values.copy()
    scaler = utils.StandardScaler()
    transformed = scaler.fit_transform(values)

    np.testing.assert_allclose(scaler.mean_, [4, 7, 2, np.nan])
    np.testing.assert_allclose(scaler.var_, [5, 0, 4, np.nan])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(5), 1, 2, np.nan])
    for statistic in (scaler.mean_, scaler.var_, scaler.scale_):
        assert statistic.dtype == np.float64
    assert transformed.dtype == dtype
    np.testing.assert_allclose(
        transformed,
        [[-3 / np.sqrt(5), 0, np.nan, np.nan], [-1 / np.sqrt(5), 0, -1, np.nan],
         [1 / np.sqrt(5), 0, 1, np.nan], [3 / np.sqrt(5), 0, np.nan, np.nan]],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), original, atol=1e-6)


def test_standard_scaler_integer_input_and_corrected_variance():
    transformed = utils.StandardScaler().fit_transform([[0, 7], [2, 7]])
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(transformed, [[-1, 0], [1, 0]])

    # The rounded mean is 1e16, so uncorrected residual squares give 4/3.
    scaler = utils.StandardScaler().fit([[1e16], [1e16], [1e16 + 2]])
    np.testing.assert_allclose(scaler.var_, [8 / 9])
    np.testing.assert_array_equal(scaler.scale_, [1])


@pytest.mark.parametrize("kind,num_chars", [
    ("standard", 0), ("positive", 0), ("categorical", 0),
    ("categorical", 2), ("continuous", 0), ("continuous", 2),
])
def test_normalizer_state_roundtrip_uses_tensors_and_restores_statistics(tmp_path, kind, num_chars):
    values = np.arange(36, dtype=np.float32).reshape(3, 12) / 40
    if kind == "standard":
        scaler = utils.StandardScaler().fit(values)
    elif kind == "positive":
        scaler = utils.PositiveStandardScaler().fit(values)
    elif kind == "categorical":
        scaler = utils.StandardScalerPhyCategorical(num_chars, 4, 3).fit(values)
    else:
        scaler = utils.StandardScalerPhyContinuous(num_chars, 4, 3).fit(
            values, mask=np.ones_like(values, dtype=bool)
        )

    path = tmp_path / "normalizer.pt"
    torch.save(scaler.state_dict(), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored = utils.normalizer_from_state_dict(state)
    np.testing.assert_array_equal(restored.transform(values), scaler.transform(values))
    np.testing.assert_allclose(
        restored.inverse_transform(restored.transform(values)), values, atol=1e-6
    )
    if kind in {"categorical", "continuous"}:
        np.testing.assert_array_equal(restored.tree_statistics(), scaler.tree_statistics())
        assert restored.mean_ is restored.cblv_scaler.mean_
        assert restored.scale_ is restored.std_ is restored.cblv_scaler.std_
        if kind == "continuous" and num_chars == 0:
            assert restored.char_scaler.mean_ is None


def test_normalizer_loader_rejects_unsupported_version():
    scaler = utils.StandardScaler().fit([[1, 2], [3, 4]])
    state = scaler.state_dict()
    with pytest.raises(ValueError, match="Unsupported normalizer state version"):
        utils.normalizer_from_state_dict({**state, "version": 2})
