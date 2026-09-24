"""fieldstats: masking and block-statistics tools for astronomical images."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("moransi_sourcemasking")
except PackageNotFoundError:
    # package isn't installed (e.g. running from a source checkout without
    # `pip install -e .` yet)
    __version__ = "0.2.0.dev0"

__author__ = "Henry C. ferguson"   

from .moransi import SlidingMoranSourceFilter, LocalMoranResult, SlidingGlobalIResult
from .tiered_sourcemask import make_sourcemask, make_individual_sourcemasks, read_config
from .block_average import (
    block_average_robust,
    noise_correlation_ratio,
    inverse_correlation_ratio
)
from .gradient_mask_growth import gradient_grow_mask, gradient_significance_map

__all__ = [
    "SlidingMoranSourceFilter",
    "LocalMoranResult",
    "SlidingGlobalIResult",
    "make_sourcemask",
    "make_individual_sourcemasks",
    "read_config",
    "block_average_robust",
    "noise_correlation_ratio",
    "inverse_correlation_ratio",
    "gradient_grow_mask",
    "gradient_significance_map"
]
