import numpy as np

from moransi_sourcemasking import block_average_robust, noise_correlation_ratio, inverse_correlation_ratio


def test_block_average_robust_basic():
    img = np.ones((8, 8))
    err = np.full((8, 8), 0.1)
    mask = np.ones((8, 8), dtype=bool)

    img_b, err_b, mask_b = block_average_robust(img, err, mask, block_size=2)

    assert img_b.shape == (4, 4)
    assert np.allclose(img_b[mask_b], 1.0)
    assert mask_b.all()


def test_block_average_robust_rejects_low_good_count():
    img = np.ones((4, 4))
    err = np.full((4, 4), 0.1)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # only 1 good pixel in the single 4x4 block

    img_b, err_b, mask_b = block_average_robust(img, err, mask, block_size=4, min_good=2)

    assert mask_b[0, 0] == False  # noqa: E712


def test_noise_correlation_ratio_and_inverse_are_reciprocal():
    R = noise_correlation_ratio(p=0.8, s=1.0, block_size=1)
    invR = inverse_correlation_ratio(p=0.8, s=1.0, block_size=1)
    assert np.isclose(R * invR, 1.0)


def test_noise_correlation_ratio_rejects_nonpositive_inputs():
    import pytest

    with pytest.raises(ValueError):
        noise_correlation_ratio(p=0, s=1.0, block_size=1)
