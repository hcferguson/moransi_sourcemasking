import numpy as np
from box import Box

from moransi_sourcemasking import make_sourcemask, make_individual_sourcemasks


def _config():
    # Two tiers with different kernel scales, mirroring what would normally
    # come from a YAML file via `read_config`.
    return Box({
        "filters": {
            "small": {"corr_half": 1, "bg_half": 8, "exclude_half": 3},
            "large": {"corr_half": 2, "bg_half": 12, "exclude_half": 5},
        }
    })


def test_make_sourcemask_shape():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 1, size=(48, 48))
    config = _config()

    mask = make_sourcemask(image, config)

    assert mask.shape == image.shape
    assert mask.dtype == bool


def test_make_individual_sourcemasks_returns_per_tier_results():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 1, size=(48, 48))
    config = _config()

    source_mask, masks, istats = make_individual_sourcemasks(image, config)

    assert set(masks.keys()) == {"small", "large"}
    assert set(istats.keys()) == {"small", "large"}
    assert source_mask.shape == image.shape
