# Driver script to build a source mask with a tiered set of kernels and smoothing
# when computing and thresholding the I statistic.
#
# Configuration is in a yaml file
#

import yaml
from box import Box
import numpy as np

from .moransi import SlidingMoranSourceFilter

# For logging 
import logging
logger = logging.getLogger(__name__)

def read_config(configfile):
    ''' Read yaml configuration file '''
    with open(configfile) as f:
        config = Box(yaml.safe_load(f))
    return config

def make_sourcemask(image,config,bad_mask=None,weight=None):
    ''' Make a source mask, applying all the tiers

        Parameters
        ----------
        image : 2D array
        config : dictionary of control parameters
        bad_mask : 2D bool array, optional
            True where the pixel is *unusable* (e.g. `dq != 0`). NaN/inf
            pixels in `image` are treated as bad automatically.
        weight : 2D array, optional
            Per-pixel weight, treated as proportional to inverse variance
            (e.g. a coadd exposure/read-noise weight map). Non-finite or
            non-positive weights are treated as bad automatically. If
            omitted, every valid pixel gets weight 1 (unweighted case).

        Returns
        -------
        source mask (OR of all the tiers)
    '''

    # Assume all pixels are background to start
    mask = np.ones(image.shape,dtype='bool') 

    # Loop through the tiers
    for f in config.filters.keys():
        filt = SlidingMoranSourceFilter(**config.filters[f])
        thismask, istat = filt.flag_sources(image,bad_mask=bad_mask,weight=weight)
        mask = mask & thismask
    return mask

def make_individual_sourcemasks(image,config,bad_mask=None,weight=None):
    ''' Make source masks for all the tiers, as well as a merged source mask

        Parameters
        ----------
        image : 2D array
        config : dictionary of control parameters
        bad_mask : 2D bool array, optional
            True where the pixel is *unusable* (e.g. `dq != 0`). NaN/inf
            pixels in `image` are treated as bad automatically.
        weight : 2D array, optional
            Per-pixel weight, treated as proportional to inverse variance
            (e.g. a coadd exposure/read-noise weight map). Non-finite or
            non-positive weights are treated as bad automatically. If
            omitted, every valid pixel gets weight 1 (unweighted case).

        Returns
        -------
        source_mask -- ORed for all the tiers
        masks -- dictionary of masks for each tier, indexed by the tier name
        istats -- dictionary of Moran's I statistics values for each tier, indixed by tier name

    '''
    # Set up an instance of the sliding filter for each tier
    filts = {}
    for k in config.filters.keys():
        filts[k] = SlidingMoranSourceFilter(**config.filters[k])

    # Bookkeeping to receive the results
    source_mask = np.ones(image.shape,dtype='bool')
    masks = {}
    istats = {}
    # Loop through the masks
    for k in config.filters.keys():
        logger.info(k)
        masks[k], istats[k] = filts[k].flag_sources(image,bad_mask=bad_mask,weight=weight)
        source_mask = source_mask & masks[k]
    return source_mask, masks, istats
