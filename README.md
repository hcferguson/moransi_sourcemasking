# moransi_sourcemasking

Masking and block-statistics tools for astronomical images:

- **`moransi`** — local (sliding-window) Moran's I spatial-autocorrelation
  statistic, with optional inverse-variance pixel weighting, used by
  `SlidingMoranSourceFilter` to flag source pixels vs. background.
- **`sourcemask`** — driver that builds a tiered source mask by combining
  several `SlidingMoranSourceFilter` configurations (radii, thresholds)
  read from a YAML config file or from a dictionary.
- **`block_average`** — robust (sigma-clipped) block-averaging of an image
  with propagated error and pixel masking, plus the drizzle
  noise-correlation ratio (Casertano et al. 2000 / Fruchter & Hook 2002).

## Installation

```bash
pip install -e .
```

(from PyPI once published: `pip install fieldstats`)

## Quick start

```python
from moransi_sourcemasking import SlidingMoranSourceFilter

filt = SlidingMoranSourceFilter(corr_half=1, bg_half=10, exclude_half=3)
source_mask, istat = filt.flag_sources(image, bad_mask=dq_mask, weight=weight_map)
```

```python
from moransi_sourcemasking import read_config, make_sourcemask

config = read_config("tiers.yaml")
mask = make_sourcemask(image, config, bad_mask=dq_mask, weight=weight_map)
```

```python
from moransi_sourcemasking import block_average_robust

image_b, err_b, mask_b = block_average_robust(image, err, mask, block_size=4)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

See `CHANGELOG.md` for release history.
