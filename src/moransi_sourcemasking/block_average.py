import warnings
import numpy as np
from astropy.nddata import block_reduce, reshape_as_blocks
from astropy.stats import sigma_clipped_stats
from functools import partial


def block_average_robust(img, err, mask, block_size, min_good=1, clip_sigma=5., clip_maxiters=2, clip_cenfunc='mean'):
    """
    Robust (sigma clipped) block-average of an image, with propagated error
    and pixel masking. Bad/padded pixels are excluded pixel-by-pixel
    from each block's computation; a block's output is kept (unmasked)
    if it has at least `min_good` good pixels, otherwise it is rejected
    (NaN, mask_block=False). If the image shape isn't evenly divisible
    by block_size, it is padded with masked (bad) pixels.

    Parameters
    ----------
    img, err, mask : 2D ndarray, same shape
        mask : good/bad, any dtype (bool, int, etc.) — nonzero/True = good
    block_size : int or (by, bx)
    min_good : int
        Minimum number of good pixels required within a block for that
        block to be kept in the output.
    clip_sigma : float
        Number of standard deviations for sigma clipping before taking mean of img array
    clip_cenfunc : float
        The statistic or callable function/object used to compute the center value for the clipping
    clip_maxiters : int
        The maximum number of sigma-clipping iterations to perform or None to clip until convergence 
        is achieved (i.e., iterate until the last iteration clips nothing).

    Returns
    -------
    img_block, err_block : 2D ndarray (NaN where block rejected)
    mask_block : 2D bool ndarray (True = kept, False = rejected)
    """
    if np.isscalar(block_size):
        block_size = (block_size, block_size)
    by, bx = block_size
    n_per_block = by * bx

    if not (1 <= min_good <= n_per_block):
        raise ValueError(f"min_good={min_good} must be between 1 and block size {n_per_block}")

    ny, nx = img.shape

    # --- pad to a multiple of block_size, padding as "bad" pixels ---
    pad_y = (-ny) % by
    pad_x = (-nx) % bx
    if pad_y or pad_x:
        pad_width = ((0, pad_y), (0, pad_x))
        img = np.pad(img, pad_width, mode='constant', constant_values=0.0)
        err = np.pad(err, pad_width, mode='constant', constant_values=1.0)  # dummy, finite & >0
        mask = np.pad(mask, pad_width, mode='constant', constant_values=0)

    # good-pixel definition: mask says good AND error is finite/positive
    good = (mask != 0) & np.isfinite(err) & (err > 0) & np.isfinite(img)

    # --- count good pixels per block; keep block if >= min_good ---
    mask_sum = block_reduce(good.astype(float), block_size, func=np.sum)
    mask_block = (mask_sum >= min_good)

    def _sigma_clipped_mean(arr, axis, mask=None, sigma=5, maxiters=2, cenfunc='mean'):
        mean, median, stddev = sigma_clipped_stats(arr,axis=axis, mask=mask, sigma=sigma,
                                                   maxiters=maxiters,cenfunc=cenfunc)
        # sigma_clipped_stats returns a MaskedArray when axis+mask are both given.
        # block_reduce forwards this straight through, and the later
        # np.where(mask_block, img_block, np.nan) call silently converts masked
        # entries (blocks fully sigma-clipped away) into leftover finite junk
        # (often 0.0) instead of NaN -- which then sails past the
        # np.isfinite(img_block) safety check below. Fill explicitly so a
        # fully-clipped block is unambiguously NaN before it leaves this function.
        if np.ma.isMaskedArray(mean):
            mean = np.ma.filled(mean, np.nan)
        return mean

    # Fill the arguments for sigma clipping
    # The mask should be True when a pixel is rejected before doing the statistics.
    # IMPORTANT: block_reduce internally reshapes `img` via reshape_as_blocks
    # before calling this func -- and depending on the installed astropy
    # version, that reshape may involve a transpose (grouping block-position
    # axes before within-block axes), not just a plain .reshape(). A flat,
    # full-image-shaped mask passed in here would only coincidentally line up
    # pixel-for-pixel under a pure-reshape layout, and silently scrambles under
    # a transpose-based layout (numpy.ma falls back to reshaping same-size
    # masks without checking that the *ordering* matches). To be correct
    # regardless of that internal detail, apply the identical
    # reshape_as_blocks transform to the mask ourselves.
    bad_reshaped = reshape_as_blocks(~good, block_size)
    sigma_clipped_mean = partial(_sigma_clipped_mean, mask = bad_reshaped, sigma=clip_sigma,
                                 maxiters=clip_maxiters,cenfunc=clip_cenfunc)

    with warnings.catch_warnings():
        # blocks with too few (or zero) good pixels produce NaN/warnings here;
        # they get overwritten by the mask_block check below regardless
        warnings.simplefilter("ignore", category=RuntimeWarning)
        img_block = block_reduce(img, block_size, func=sigma_clipped_mean)

    # --- inverse-variance quadrature error propagation (already excludes bad pixels) ---
    weight = np.zeros_like(err, dtype=float)
    weight[good] = 1.0 / err[good] ** 2
    weight_sum = block_reduce(weight, block_size, func=np.sum)
    with np.errstate(divide='ignore', invalid='ignore'):
        err_block = 1.0 / np.sqrt(weight_sum)

    # --- apply block-level rejection ---
    img_block = np.where(mask_block, img_block, np.nan)
    err_block = np.where(mask_block, err_block, np.nan)
    mask_block = mask_block & np.isfinite(img_block) & np.isfinite(err_block)

    return img_block, err_block, mask_block


def noise_correlation_ratio(p: float, s: float, block_size: float = 1.0) -> float:
    """ Compute R, the drizzle noise-correlation correction factor, following
    Casertano et al. (2000) / Fruchter & Hook (2002), as presented in the
    DrizzlePac Handbook Sec. 3.3.

    R = sigma_c / sigma_p is the "noise correlation ratio": the ratio of the
    naive (uncorrelated, pixfrac=0 equivalent) noise to the true, correlation-
    suppressed noise. 1/R = sigma_p / sigma_c is therefore the factor you'd
    multiply a naive/uncorrelated predicted error by to get the true expected
    rms -- i.e. measured_rms / predicted_from_err, in the many-dither,
    uniformly-filled-plane approximation.

    Definitions:
        p           = pixfrac
        s           = scale = output_pixel_size / input_pixel_size
        block_size  = N, the linear size (in output pixels) of an N x N
                      block average/sum. block_size=1 means no block averaging.

    The block-summed/averaged image is equivalent to having drizzled directly
    onto a pixel of size N*s, so we substitute s -> N*s before applying the
    r = p/s formula.

    Parameters
    ----------
    p : float
        pixfrac (drop size relative to the pixel size)
    s : float
        scale = output_pixel_size / input_pixel_size
    block_size : int or numpy.array of ints
        the linear size (in output pixels) of an N x N
        block average/sum. block_size=1 means no block averaging.

    Returns
    -------
    R : float
        The ratio of uncorrelated error to correlation-suppressed noise
    """
    if p <= 0 or s <= 0 or block_size <= 0:
        raise ValueError("p, s, and block_size must all be positive.")

    s_eff = s * block_size  # equivalent output pixel scale after block-averaging
    r = p / s_eff

    if r >= 1:
        R = r / (1 - 1 / (3 * r))
    else:
        # Fruchter & Hook (2002), PASP 114, 144, Eq. 10.
        # For large block_size (r << 1), this reduces exactly to
        # 1/R = 1 - p / (3 * s * block_size).
        R = 1 / (1 - r / 3)

    return R


def inverse_correlation_ratio(p: float, s: float, block_size: float = 1.0) -> float:
    """Return 1/R = sigma_p / sigma_c (measured_rms / predicted_from_err)."""
    return 1.0 / noise_correlation_ratio(p, s, block_size)
