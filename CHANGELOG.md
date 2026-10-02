# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### [0.4.0] Refactored
- Removed options to smooth the image before and after computing the statistic.
  Removed the option to dilate the mask.
  Removed computation of statistics in an annulus.

### Refactored
- Removed the computation of the local Moran's I statistic and the tiered
  source masking. The global_I algorithm works at least as well without
  basically only one tuning parameter.

### Added
- `compute_sliding_global_I()` gained an `n_box_passes` option (default `1`,
  fully backward compatible). With `n_box_passes > 1`, each of the six
  underlying per-pixel maps is box-summed that many times in a row before
  being combined into the ratio -- the classic "repeated box blur
  approximates a Gaussian blur" construction -- which softens the flat,
  sharp-edged "box" footprint a hard patch gives compact sources into a
  smoothly tapered one, while staying entirely on the fast integral-image
  path (no FFT/astropy.convolution). Verified that `n_box_passes=1`
  reproduces the original hard-box statistic exactly, and that
  `n_box_passes=3` measurably removes the flat plateau near a compact
  test source, in `tests/test_moransi.py`.

- `SlidingMoranSourceFilter.compute_sliding_global_I()` in `moransi_sourcemasking.moransi`:
  a new statistic computing the textbook *Global* Moran's I (one shared
  mean/variance reference per patch) over a sliding patch centered on
  every pixel, as an alternative to `compute()`'s per-pixel annulus-based
  local reference. Reduces algebraically to a patch-local covariance
  between the image and its own neighbor-mean map divided by the
  patch-local variance; verified against a brute-force reference
  implementation (both unweighted and inverse-variance-weighted) in
  `tests/test_moransi.py`. Returns a new `SlidingGlobalIResult` dataclass.
- `compute()`, `compute_block_averaged()`, and `flag_sources()` in `moransi_sourcemasking.moransi`
  now accept an optional `source_mask` argument (True = background/sky), so a
  smaller-kernel tier's result can be chained into a larger tier's background
  estimation and excluded from it. `sourcemask.make_sourcemask()` and
  `make_individual_sourcemasks()` now chain tiers this way automatically
  (config tiers should be listed smallest-kernel first).

### Fixed
- `compute()` no longer crashes when called with the default `bad_mask=None`
  (a debug-logging line unconditionally did `~bad_mask`).


## [0.1.0] - 2026-09-15
### Added
- Initial package structure, assembled from standalone scripts:
  `sliding_moransi_blk.py` → `moransi_sourcemask.moransi`,
  `tiered_sourcemask.py` → `moransi_sourcemask.tiered_sourcemask`,
  `block_average_robust.py` → `moransi_sourcemask.block_average`.
