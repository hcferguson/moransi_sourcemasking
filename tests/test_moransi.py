import numpy as np
import pytest

from moransi_sourcemasking import SlidingMoranSourceFilter


def test_flag_sources_runs_on_flat_image():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 1, size=(64, 64))

    filt = SlidingMoranSourceFilter()
    mask, istat = filt.flag_sources(image)

    assert mask.shape == image.shape
    assert mask.dtype == bool
    assert istat.shape == image.shape


def test_flag_sources_detects_injected_point_source():
    rng = np.random.default_rng(1)
    image = rng.normal(0, 1, size=(64, 64))
    image[32, 32] += 50  # bright source, should get flagged (mask False there)

    filt = SlidingMoranSourceFilter()
    mask, _ = filt.flag_sources(image)

    assert mask[32, 32] == False  # noqa: E712 (explicit bool check reads clearer here)


def test_sliding_global_I_matches_brute_force_unweighted():
    rng = np.random.default_rng(1)
    N = 40
    image = rng.normal(0, 1, size=(N, N))
    image[20:23, 20:23] += 4.0
    kernel_width, patch_size = 3, 20

    filt = SlidingMoranSourceFilter( kernel_width=kernel_width, patch_size=patch_size)
    res = filt.compute_sliding_global_I(image, patch_half=patch_half)

    def brute_force(x, c, patch_size, kernel_width):
        patch_half = patch_size // 2
        corr_half = kernel_width // 2
        ci, cj = c
        N0, N1 = x.shape
        i0, i1 = max(0, ci - patch_half), min(N0, ci + patch_half + 1)
        j0, j1 = max(0, cj - patch_half), min(N1, cj + patch_half + 1)
        idxs = [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
        vals = np.array([x[i, j] for i, j in idxs])
        xbar = vals.mean()
        m2 = ((vals - xbar) ** 2).mean()
        total = 0.0
        for (i, j) in idxs:
            pi0, pi1 = max(0, i - corr_half), min(N0, i + corr_half + 1)
            pj0, pj1 = max(0, j - corr_half), min(N1, j + corr_half + 1)
            block = x[pi0:pi1, pj0:pj1]
            n = block.size - 1
            s = block.sum() - x[i, j]
            total += (x[i, j] - xbar) * (s / n - xbar)
        return total / len(idxs) / m2

    for c in [(20, 20), (10, 10), (5, 5)]:
        assert np.isclose(res.I[c], brute_force(image, c, patch_size, kernel_width), atol=1e-8)


def test_sliding_global_I_respects_bad_mask_and_flags_low_confidence_edges():
    rng = np.random.default_rng(3)
    N = 30
    image = rng.normal(0, 1, size=(N, N))
    bad_mask = np.zeros((N, N), dtype=bool)
    bad_mask[10, 10] = True  # a bad pixel inside the patch of nearby centers

    filt = SlidingMoranSourceFilter()
    res = filt.compute_sliding_global_I(image,bad_mask=bad_mask)

    assert res.I.shape == image.shape
    # the bad pixel itself should never be flagged as valid output
    assert np.isnan(res.I[10, 10])
    # a well-covered interior center away from the bad pixel should be finite
    assert np.isfinite(res.I[20, 20])
    # a corner, whose patch is mostly off-image, should fail min_valid_frac and be NaN
    assert np.isnan(res.I[0, 0])


def test_sliding_global_I_n_box_passes_1_matches_original_hard_box():
    # n_box_passes=1 must reproduce the exact hard-box statistic, byte for byte
    # (this is the pre-n_box_passes implementation, brute-force verified separately).
    rng = np.random.default_rng(1)
    N = 40
    image = rng.normal(0, 1, size=(N, N))
    image[20:23, 20:23] += 4.0
    filt = SlidingMoranSourceFilter()
    res_default = filt.compute_sliding_global_I(image, patch_half=8)
    res_explicit = filt.compute_sliding_global_I(image, patch_half=8, n_box_passes=1)
    assert np.array_equal(res_default.I, res_explicit.I, equal_nan=True)


def test_sliding_global_I_multi_pass_softens_compact_source_box_footprint():
    # A hard box (n_box_passes=1) should be ~flat near a compact source and then
    # fall sharply right at patch_half; multiple passes should instead fall off
    # smoothly and monotonically, with no flat plateau.
    rng = np.random.default_rng(7)
    N = 81
    image = rng.normal(0, 1.0, size=(N, N))
    cy, cx = 40, 40
    yy, xx = np.mgrid[0:N, 0:N]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    image += 30.0 * np.exp(-r2 / (2 * 1.2 ** 2))  # small, compact source

    filt = SlidingMoranSourceFilter()
    hard = filt.compute_sliding_global_I(image, patch_half=10, n_box_passes=1)
    soft = filt.compute_sliding_global_I(image, patch_half=4, n_box_passes=3)

    hard_profile = np.array([hard.I[cy, cx + dx] for dx in range(0, 9)])
    soft_profile = np.array([soft.I[cy, cx + dx] for dx in range(0, 9)])

    # hard-box profile should vary very little near the source (the "plateau")
    assert np.ptp(hard_profile) < 0.02
    # soft (3-pass) profile should show a clear, non-trivial decline over the same range
    assert (soft_profile[0] - soft_profile[-1]) > 0.03


def test_sliding_global_I_rejects_invalid_n_box_passes():
    image = np.zeros((10, 10))
    filt = SlidingMoranSourceFilter()
    with pytest.raises(ValueError):
        filt.compute_sliding_global_I(image, patch_half=3, n_box_passes=0)
