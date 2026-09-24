# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Added
- `SlidingMoranSourceFilter.compute_sliding_global_I()` in `fieldstats.moransi`:
  a new statistic computing the textbook *Global* Moran's I (one shared
  mean/variance reference per patch) over a sliding patch centered on
  every pixel, as an alternative to `compute()`'s per-pixel annulus-based
  local reference. Reduces algebraically to a patch-local covariance
  between the image and its own neighbor-mean map divided by the
  patch-local variance; verified against a brute-force reference
  implementation (both unweighted and inverse-variance-weighted) in
  `tests/test_moransi.py`. Returns a new `SlidingGlobalIResult` dataclass.
- `compute()`, `compute_block_averaged()`, and `flag_sources()` in `fieldstats.moransi`
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
