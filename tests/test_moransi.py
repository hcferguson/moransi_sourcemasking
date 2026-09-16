import numpy as np

from moransi_sourcemasking import SlidingMoranSourceFilter


def test_flag_sources_runs_on_flat_image():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 1, size=(64, 64))

    filt = SlidingMoranSourceFilter(corr_half=1, bg_half=10, exclude_half=3)
    mask, istat = filt.flag_sources(image)

    assert mask.shape == image.shape
    assert mask.dtype == bool
    assert istat.shape == image.shape


def test_flag_sources_detects_injected_point_source():
    rng = np.random.default_rng(1)
    image = rng.normal(0, 1, size=(64, 64))
    image[32, 32] += 50  # bright source, should get flagged (mask False there)

    filt = SlidingMoranSourceFilter(corr_half=1, bg_half=10, exclude_half=3)
    mask, _ = filt.flag_sources(image)

    assert mask[32, 32] == False  # noqa: E712 (explicit bool check reads clearer here)
