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


def test_source_mask_excludes_pixels_from_background_stats():
    rng = np.random.default_rng(2)
    image = rng.normal(0, 1, size=(40, 40))
    filt = SlidingMoranSourceFilter(
        corr_half=1, bg_half=8, exclude_half=3, sigma_clip=None, clip_iters=0
    )

    # An all-True source_mask (nothing excluded) should reproduce the
    # no-source_mask result exactly.
    res_none = filt.compute(image)
    res_all_bg = filt.compute(image, source_mask=np.ones(image.shape, dtype=bool))
    assert np.allclose(res_none.bg_mean, res_all_bg.bg_mean, equal_nan=True)

    # Excluding a patch should shift nearby background stats away from the
    # no-mask case (proves the patch was actually dropped from the annulus,
    # not silently kept via an AND/OR mix-up).
    source_mask = np.ones(image.shape, dtype=bool)
    source_mask[15:20, 15:20] = False
    res_excl = filt.compute(image, source_mask=source_mask)
    assert not np.allclose(res_excl.bg_mean, res_none.bg_mean, equal_nan=True)


def test_sliding_global_I_matches_brute_force_unweighted():
    rng = np.random.default_rng(1)
    N = 40
    image = rng.normal(0, 1, size=(N, N))
    image[20:23, 20:23] += 4.0
    corr_half, patch_half = 1, 8

    filt = SlidingMoranSourceFilter(
        corr_half=corr_half, bg_half=patch_half + 1, exclude_half=corr_half,
        sigma_clip=None, clip_iters=0, min_valid_frac=0.0,
    )
    res = filt.compute_sliding_global_I(image, patch_half=patch_half)

    def brute_force(x, c, patch_half, corr_half):
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
        assert np.isclose(res.I[c], brute_force(image, c, patch_half, corr_half), atol=1e-8)


def test_sliding_global_I_respects_bad_mask_and_flags_low_confidence_edges():
    rng = np.random.default_rng(3)
    N = 30
    image = rng.normal(0, 1, size=(N, N))
    bad_mask = np.zeros((N, N), dtype=bool)
    bad_mask[10, 10] = True  # a bad pixel inside the patch of nearby centers

    filt = SlidingMoranSourceFilter(corr_half=1, bg_half=9, exclude_half=1, sigma_clip=None, clip_iters=0)
    res = filt.compute_sliding_global_I(image, patch_half=8, bad_mask=bad_mask)

    assert res.I.shape == image.shape
    # the bad pixel itself should never be flagged as valid output
    assert np.isnan(res.I[10, 10])
    # a well-covered interior center away from the bad pixel should be finite
    assert np.isfinite(res.I[20, 20])
    # a corner, whose patch is mostly off-image, should fail min_valid_frac and be NaN
    assert np.isnan(res.I[0, 0])
