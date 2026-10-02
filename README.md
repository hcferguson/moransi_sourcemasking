# moransi_sourcemasking

Masking and block-statistics tools for astronomical images:

- **`moransi`** — Sliding-window computation Moran's I spatial-autocorrelation
  statistic, with optional inverse-variance pixel weighting, used by
  `SlidingMoranSourceFilter` to flag source pixels vs. background.
- **`block_average`** — For assessing the results of background subtraction.
  This includes a robust (sigma-clipped) block-averaging of an image
  with propagated error and pixel masking, plus the drizzle
  noise-correlation ratio (Casertano et al. 2000 / Fruchter & Hook 2002).

## Installation

```bash
pip install -e .
```

(from PyPI once published: `pip install moransi_sourcemasking`)

## Quick start

If you just want to mask the sources, using the default parameters, passing
in a bad-pixel mask (True => bad) and a weight map (E.g. the inverse variance
expected for each pixel.)

```python
from moransi_sourcemasking import make_sourcemask
source_mask = make_sourcemask(image, bad_mask=dq_mask, weight=weight_map)
```

For more control, you can pass the parameters in a yaml file or as a dictionary

```python
from moransi_sourcemasking import make_sourcemask, read_config

config = read_config("my_parameters.yaml")
config['threshold_type'] = 'percentage'
config['threshold_value'] = 40.
mask = make_sourcemask(image, config, bad_mask=dq_mask, weight=weight_map)
```

To look at the statistics as a function of scale after background subtraction
```python
from moransi_sourcemasking import block_average_robust

# On a scale of 10 pixels
block_size = 10
image_b, err_b, mask_b = block_average_robust(image, err, mask, block_size=block_size)

# Ratio of the measured sky RMS to that predicted from the error array
rms_ratio = mad_std(image_b[mask_b]) / err_b[mask_b].mean()

# Correct for the covariance introduce by resampling (if relevant)
suppression = inverse_correlation_ratio(pixfrac,scale_ratio,block_size)
corrected_ratio = rms_ratio / suppression
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Demo
There is a demo jupyter notebook in the notebook directory.

See `CHANGELOG.md` for release history.
