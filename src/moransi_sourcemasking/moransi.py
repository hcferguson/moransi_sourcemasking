"""
Local (sliding-window) Moran's I spatial-autocorrelation statistic, with
optional inverse-variance pixel weighting for coadded images.

For every pixel i, computes:

    z_center = (x_i - mu_i) / sigma0_i
    z_nbr    = (weighted neighbor mean - mu_i) / sigma0_i
    I_i      = z_center * z_nbr

This is a local spatial-autocorrelation statistic, not a background
subtraction or detection/SNR statistic: mu_i and sigma0_i (a single shared
reference mean and standard deviation, estimated from the outer annulus)
are used only to standardize deviations onto a common scale -- "what would
this kernel's pixels look like if they had roughly the same standard
deviation as the pixels in the outer annulus." Both the center pixel and
the neighbor-mean term are divided by that *same* sigma0_i; neither term is
further rescaled by its own sampling precision. That symmetry is what keeps
I_i a correlation-type quantity (large when a pixel's deviation and its
neighbors' typical deviation move together) rather than a significance test
of whether the neighbor mean differs from the background.

Weighting enters in two places only:
  - mu_i and sigma0_i are a weighted mean and weighted (reduced-chi-square)
    variance over the annulus, so noisier/less-exposed annulus pixels
    contribute less and the reference scale self-calibrates if the weight
    map's absolute normalization is only "roughly" proportional to inverse
    variance (e.g. it omits sky Poisson noise, or resampling in a drizzled
    coadd correlates/inflates the true variance).
  - the neighbor mean itself is a weighted mean of the kernel's neighbor
    pixels, so a low-weight neighbor counts for less in forming the
    neighbor consensus value.

All windowed sums are computed via summed-area tables (integral images),
fully vectorized over the whole image -- no per-pixel Python loop, and cost
is independent of window size.

Convention: `bad_mask` is True where a pixel is unusable (e.g. DQ != 0).
Any NaN/inf pixel in the data array, or (if a weight array is given) any
non-finite or non-positive weight, is automatically treated as bad too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from astropy.convolution import convolve, convolve_fft, Tophat2DKernel, Gaussian2DKernel


##################################################################################
# For logging and profiling
import time
import resource
import logging
logger = logging.getLogger(__name__)

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes -> GB
##################################################################################

@dataclass
class LocalMoranResult:
    I: np.ndarray                  # local Moran's I map
    z_center: np.ndarray           # standardized center-pixel deviation
    z_neighbor_mean: np.ndarray    # standardized neighbor-mean deviation
    bg_mean: np.ndarray            # local (weighted) background mean, mu_i
    bg_scale: np.ndarray           # local reference noise scale, sigma0_i (sqrt of the reduced-chi-square factor)
    bg_valid_count: np.ndarray     # number of valid pixels used in the bg annulus
    bg_weight_sum: np.ndarray      # sum of weights used in the bg annulus
    neighbor_valid_count: np.ndarray   # number of valid neighbor pixels
    neighbor_weight_sum: np.ndarray    # sum of neighbor weights


def _pad_to_multiple(arr: np.ndarray, block: int, pad_value: float):
    """Pad a 2D array with a constant value so both dims are multiples of `block`."""
    H, W = arr.shape
    pad_h = (-H) % block
    pad_w = (-W) % block
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_value)
    return padded


def block_average(
    image: np.ndarray,
    block: int,
    bad_mask: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    min_valid_frac: float = 0.5,
):
    """
    Block-average `image` over block x block cells.

    Uses inverse-variance weighting if `weight` is given: the block value is
    the weighted mean of its valid native pixels, and the block's own
    effective weight is the *sum* of the native weights it contains -- the
    correct combined precision when averaging independent noisy samples,
    and directly consistent with how SlidingMoranSourceFilter already
    interprets a weight array.

    NaN/inf pixels and bad_mask/weight-derived bad pixels are excluded from
    the block average (contribute zero weight, not a poisoned value). The
    image is padded (with zero-weight, so it never biases a block mean)
    before reshaping, so it works for shapes that aren't an exact multiple
    of `block`.

    Returns
    -------
    block_image : 2D array, shape (ceil(H/block), ceil(W/block))
    block_weight : 2D array, same shape (sum of native weights per block)
    block_bad : 2D bool array, same shape
        True where a block has too few valid native pixels
        (< min_valid_frac * block**2) or zero total weight.
    orig_shape : (H, W) of the input, needed by block_replicate to crop
        the padding back off.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape
    finite = np.isfinite(image)

    if bad_mask is None:
        valid = finite
    else:
        valid = finite & ~np.asarray(bad_mask, dtype=bool)

    if weight is None:
        w = np.ones_like(image)
    else:
        weight = np.asarray(weight, dtype=np.float64)
        valid = valid & np.isfinite(weight) & (weight > 0)
        w = weight

    w_eff = np.where(valid, w, 0.0)
    vals = np.where(valid, image, 0.0)
    cnt = valid.astype(np.float64)

    w_eff_p = _pad_to_multiple(w_eff, block, 0.0)
    wx_p = _pad_to_multiple(w_eff * vals, block, 0.0)
    cnt_p = _pad_to_multiple(cnt, block, 0.0)

    Hp, Wp = w_eff_p.shape
    nby, nbx = Hp // block, Wp // block

    def _block_sum(a):
        return a.reshape(nby, block, nbx, block).sum(axis=(1, 3))

    block_Wsum = _block_sum(w_eff_p)
    block_WXsum = _block_sum(wx_p)
    block_cnt = _block_sum(cnt_p)

    block_Wsum_safe = np.maximum(block_Wsum, 1e-12)
    block_image = block_WXsum / block_Wsum_safe

    max_cnt = block * block
    block_bad = (block_cnt < min_valid_frac * max_cnt) | (block_Wsum <= 0)

    return block_image, block_Wsum, block_bad, (H, W)


def block_replicate(coarse: np.ndarray, block: int, orig_shape) -> np.ndarray:
    """Nearest-neighbor upsample a block-grid array back to native resolution."""
    H, W = orig_shape
    rep = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
    return rep[:H, :W]


class SlidingMoranSourceFilter:
    """
    Parameters
    ----------
    pre_tophat, post_tophat, dilation_tophat : int
        Native pixel scale; radii of a tophat filter to apply 
          (pre) before computing Moran's I,
          (post) to the array of Moran's I values, resampled back to the native pixel scale
          (dilation) to the dilate mask (1= source, 0 = background)
    dilation_threshold : float
        Threshold to apply to the dilated mask to convert back to a boolean
    lower_percentile : pixels with I below this threshold are masked as "sources"
    upper_percentile : pixels with I above this threshold are masked as "sources"
    corr_half, bg_half, exclude_half : int
        Always specified at *native pixel* scale, whether you call
        compute() directly or compute_block_averaged(): the latter
        converts them to block-grid units for you.
    corr_half : int
        Half-width of the correlation (neighbor) kernel. corr_half=1 gives
        the classic 3x3-with-hole kernel (8 neighbors).
    bg_half : int
        Half-width of the outer background-normalization box.
    exclude_half : int
        Half-width of the inner box excised from the background window.
        Should be >= corr_half.
    sigma_clip : float or None
        If set, iteratively excludes background-window pixels more than
        `sigma_clip` sigma (using each pixel's own weight) from the local
        mean before the final mu_i / sigma0_i estimate. Set to None to
        disable (single-pass estimate, cheaper).
    clip_iters : int
        Number of sigma-clipping iterations, ignored if sigma_clip is None.
    std_floor : float
        Minimum sigma0_i used in the denominator, to avoid blow-ups in
        near-flat / heavily masked regions.
    min_valid_frac : float
        If the fraction of valid pixels in a pixel's background annulus
        falls below this, that pixel's I is set to NaN.
    """

    def __init__(
        self,
        pre_tophat: int = 0,
        post_tophat: int = 0,
        dilation_tophat: int = 0,
        dilation_threshold: int = 0.05,
        i_lower_nsigma: float = 100,
        i_upper_nsigma: float = 10,
        block_size: int = 0,
        corr_half: int = 1,
        bg_half: int = 10,
        exclude_half: int = 3,
        sigma_clip: Optional[float] = 3.0,
        clip_iters: int = 2,
        std_floor: float = 1e-6,
        min_valid_frac: float = 0.3,
    ):
        if exclude_half < corr_half:
            raise ValueError("exclude_half should be >= corr_half")
        if bg_half <= exclude_half:
            raise ValueError("bg_half must be larger than exclude_half")
        self.corr_half = corr_half
        self.bg_half = bg_half
        self.exclude_half = exclude_half
        self.sigma_clip = sigma_clip
        self.clip_iters = clip_iters
        self.std_floor = std_floor
        self.min_valid_frac = min_valid_frac
        self.pre_tophat = pre_tophat
        self.post_tophat = post_tophat
        self.dilation_tophat = dilation_tophat
        self.dilation_threshold = dilation_threshold
        self.i_lower_nsigma = i_lower_nsigma
        self.i_upper_nsigma = i_upper_nsigma
        self.block_size = block_size

    # ------------------------------------------------------------------
    # Integral-image machinery
    # ------------------------------------------------------------------

    @staticmethod
    def _integral_image(arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        S = np.zeros((H + 1, W + 1), dtype=np.float64)
        S[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
        return S

    @staticmethod
    def _box_sum_from_integral(integral: np.ndarray, half: int, shape) -> np.ndarray:
        H, W = shape
        rows = np.arange(H)
        cols = np.arange(W)
        r0 = np.clip(rows - half, 0, H)
        r1 = np.clip(rows + half + 1, 0, H)
        c0 = np.clip(cols - half, 0, W)
        c1 = np.clip(cols + half + 1, 0, W)
        A = integral[np.ix_(r1, c1)]
        B = integral[np.ix_(r0, c1)]
        C = integral[np.ix_(r1, c0)]
        D = integral[np.ix_(r0, c0)]
        return A - B - C + D

    def _build_integrals(self, image: np.ndarray, w_eff: np.ndarray, valid_mask: np.ndarray):
        """Precompute the four integral images (count, W, WX, WX^2) once per call."""
        cnt = valid_mask.astype(np.float64)
        safe_image = np.where(valid_mask, image, 0.0)
        S_cnt = self._integral_image(cnt)
        S_w = self._integral_image(w_eff)
        S_wx = self._integral_image(w_eff * safe_image)
        S_wx2 = self._integral_image(w_eff * safe_image * safe_image)
        logger.debug(f"TD: NaN in image: {np.isnan(image).sum()}, NaN in w_eff*image: {np.isnan(w_eff*image).sum()}")
        return S_cnt, S_w, S_wx, S_wx2

    def _query_box(self, integrals, half: int, shape):
        S_cnt, S_w, S_wx, S_wx2 = integrals
        cnt = self._box_sum_from_integral(S_cnt, half, shape)
        Wsum = self._box_sum_from_integral(S_w, half, shape)
        WXsum = self._box_sum_from_integral(S_wx, half, shape)
        WX2sum = self._box_sum_from_integral(S_wx2, half, shape)
        return cnt, Wsum, WXsum, WX2sum

    def _query_annulus(self, integrals, half_outer: int, half_inner: int, shape):
        cnt_o, W_o, WX_o, WX2_o = self._query_box(integrals, half_outer, shape)
        cnt_i, W_i, WX_i, WX2_i = self._query_box(integrals, half_inner, shape)
        return cnt_o - cnt_i, W_o - W_i, WX_o - WX_i, WX2_o - WX2_i

    # ------------------------------------------------------------------
    # Background (mu_i, sigma0_i) estimation, with optional sigma clipping
    # ------------------------------------------------------------------

    def _compute_bg_stats(self, image: np.ndarray, weight: np.ndarray, valid_mask: np.ndarray):
        mask = valid_mask.copy()
        shape = image.shape

        for _ in range(self.clip_iters if self.sigma_clip is not None else 0):
            w_eff = np.where(mask, weight, 0.0)
            integrals = self._build_integrals(image, w_eff, mask)
            cnt, Wsum, WXsum, WX2sum = self._query_annulus(
                integrals, self.bg_half, self.exclude_half, shape
            )
            safe_W = np.maximum(Wsum, 1e-12)
            mu = WXsum / safe_W
            dof = np.maximum(cnt - 1.0, 1.0)
            sigma0_sq = np.maximum((WX2sum - mu ** 2 * Wsum) / dof, 0.0)
            sigma0_sq = np.maximum(sigma0_sq, self.std_floor ** 2)

            safe_w_pix = np.maximum(weight, 1e-12)
            z = (image - mu) / np.sqrt(sigma0_sq / safe_w_pix)
            mask = valid_mask & (np.abs(z) < self.sigma_clip)

        w_eff = np.where(mask, weight, 0.0)
        integrals = self._build_integrals(image, w_eff, mask)
        cnt, Wsum, WXsum, WX2sum = self._query_annulus(
            integrals, self.bg_half, self.exclude_half, shape
        )
        safe_W = np.maximum(Wsum, 1e-12)
        mu = WXsum / safe_W
        dof = np.maximum(cnt - 1.0, 1.0)
        sigma0_sq = np.maximum((WX2sum - mu ** 2 * Wsum) / dof, 0.0)
        sigma0_sq = np.maximum(sigma0_sq, self.std_floor ** 2)

        return mu, sigma0_sq, cnt, Wsum, mask

    # ------------------------------------------------------------------
    # Neighbor (correlation kernel) weighted mean
    # ------------------------------------------------------------------

    def _neighbor_stats(self, image: np.ndarray, weight: np.ndarray, valid_mask: np.ndarray):
        w_eff = np.where(valid_mask, weight, 0.0)
        integrals = self._build_integrals(image, w_eff, valid_mask)
        shape = image.shape
        cnt_box, W_box, WX_box, _ = self._query_box(integrals, self.corr_half, shape)

        center_w = w_eff
        center_wx = w_eff * image
        center_cnt = valid_mask.astype(np.float64)

        nbr_cnt = cnt_box - center_cnt
        nbr_W = W_box - center_w
        nbr_WX = WX_box - center_wx

        safe_W = np.maximum(nbr_W, 1e-12)
        nbr_mean = nbr_WX / safe_W
        return nbr_mean, nbr_cnt, nbr_W

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        image: np.ndarray,
        bad_mask: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
    ) -> LocalMoranResult:
        """
        Compute the local Moran's I map for `image`.

        Parameters
        ----------
        image : 2D array
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
        LocalMoranResult
        """
        image = np.asarray(image, dtype=np.float64)
        finite = np.isfinite(image)

        if bad_mask is None:
            valid_mask = finite
        else:
            bad_mask = np.asarray(bad_mask, dtype=bool)
            valid_mask = finite & ~bad_mask

        if weight is None:
            weight = np.ones_like(image)
        else:
            weight = np.asarray(weight, dtype=np.float64)
            valid_mask = valid_mask & np.isfinite(weight) & (weight > 0)

        logger.debug(f"    compute: {np.count_nonzero(~bad_mask) = }")
        logger.debug(f"    compute: {np.count_nonzero(valid_mask) = }")
        mu_bg, sigma0_sq_bg, bg_count, bg_Wsum, clipped_mask = self._compute_bg_stats(
            image, weight, valid_mask
        )
        nbr_mean, nbr_cnt, nbr_W = self._neighbor_stats(image, weight, valid_mask)

        sigma0_bg = np.sqrt(sigma0_sq_bg)
        sigma0_safe = np.maximum(sigma0_bg, self.std_floor)

        # Both terms are standardized by the SAME shared annulus scale --
        # neither is rescaled by its own sampling precision, so I_i remains
        # a correlation-type statistic rather than a significance test.
        z_center = (image - mu_bg) / sigma0_safe
        z_nbr = (nbr_mean - mu_bg) / sigma0_safe
        I = z_center * z_nbr

        logger.debug(f"TD: image shape: {image.shape}")
        logger.debug(f"TD: valid_mask frac: {valid_mask.mean():.3f}")
        outer_side = 2 * self.bg_half + 1
        inner_side = 2 * self.exclude_half + 1
        max_bg_count = outer_side ** 2 - inner_side ** 2
        logger.debug(f"TD: max_bg_count={max_bg_count}, threshold={self.min_valid_frac*max_bg_count:.0f}")
        logger.debug(f"TD: bg_count stats: min={bg_count.min():.0f} median={np.median(bg_count):.0f} max={bg_count.max():.0f}")
        logger.debug(f"TD: frac pixels below low_conf threshold: {(bg_count < self.min_valid_frac*max_bg_count).mean():.3f}")
        logger.debug(f"TD: frac nbr_W<=0: {(nbr_W<=0).mean():.3f}")
        low_conf = bg_count < (self.min_valid_frac * max_bg_count)

        bad = low_conf | (nbr_W <= 0) | ~valid_mask
        I = np.where(bad, np.nan, I)
        z_center = np.where(bad, np.nan, z_center)
        z_nbr = np.where(bad, np.nan, z_nbr)

        return LocalMoranResult(
            I=I,
            z_center=z_center,
            z_neighbor_mean=z_nbr,
            bg_mean=mu_bg,
            bg_scale=np.sqrt(sigma0_sq_bg),
            bg_valid_count=bg_count,
            bg_weight_sum=bg_Wsum,
            neighbor_valid_count=nbr_cnt,
            neighbor_weight_sum=nbr_W,
        )

    def _scale_params_to_block(self, block: int):
        """
        Convert the instance's native-pixel corr_half/bg_half/exclude_half
        to block-grid units (nearest integer, minimum 1), then re-enforce
        the same ordering constraints the constructor requires
        (exclude_half >= corr_half, bg_half > exclude_half) in case
        rounding collapsed the gap between two of them.
        """
        corr_half_b = max(1, round(self.corr_half / block))
        exclude_half_b = max(corr_half_b, round(self.exclude_half / block))
        bg_half_b = max(exclude_half_b + 1, round(self.bg_half / block))
        return corr_half_b, bg_half_b, exclude_half_b

    def compute_block_averaged(
        self,
        image: np.ndarray,
        block: int,
        bad_mask: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
        min_valid_frac_block: float = 0.5,
        replicate: bool = True,
        verbose: bool = True,
    ) -> LocalMoranResult:
        """
        Block-average the image by `block`x`block`, run the filter on the
        coarse grid, and (by default) replicate the result back to the
        native pixel grid.

        corr_half, bg_half, and exclude_half on this instance are always
        interpreted at *native pixel* scale (the same scale you'd use with
        plain compute()) -- this method converts them to block-grid units
        itself (nearest integer, minimum 1, with the usual ordering
        constraints re-applied after rounding) so you don't have to
        pre-divide by `block` yourself. Set verbose=False to silence the
        one-line report of the resolved block-grid window sizes.

        Block averaging by `block` reduces the pixel count -- and so the
        cost of every step, including the filter itself -- by block**2,
        making a large effective footprint (needed to clear the wings of
        big, extended sources) far cheaper than reaching it by growing
        corr_half/bg_half at native resolution.

        The coarse-grid I map is blocky (piecewise constant over each
        block) after replication -- fine for masking, but note it if you
        need a smooth map.
        """
        corr_half_b, bg_half_b, exclude_half_b = self._scale_params_to_block(block)
        logger.debug(
            "compute_block_averaged: block=%d -> corr_half=%d, bg_half=%d, exclude_half=%d "
            "(block-grid units; effective native footprint corr=%d, bg=%d, exclude=%d px)",
            block, corr_half_b, bg_half_b, exclude_half_b,
            corr_half_b * block, bg_half_b * block, exclude_half_b * block,
        )

        coarse_filt = SlidingMoranSourceFilter(
            corr_half=corr_half_b,
            bg_half=bg_half_b,
            exclude_half=exclude_half_b,
            sigma_clip=self.sigma_clip,
            clip_iters=self.clip_iters,
            std_floor=self.std_floor,
            min_valid_frac=self.min_valid_frac,
        )

        block_image, block_weight, block_bad, orig_shape = block_average(
            image, block, bad_mask=bad_mask, weight=weight, min_valid_frac=min_valid_frac_block
        )
        coarse = coarse_filt.compute(block_image, bad_mask=block_bad, weight=block_weight)

        if not replicate:
            return coarse

        rep = lambda a: block_replicate(a, block, orig_shape)
        return LocalMoranResult(
            I=rep(coarse.I),
            z_center=rep(coarse.z_center),
            z_neighbor_mean=rep(coarse.z_neighbor_mean),
            bg_mean=rep(coarse.bg_mean),
            bg_scale=rep(coarse.bg_scale),
            bg_valid_count=rep(coarse.bg_valid_count),
            bg_weight_sum=rep(coarse.bg_weight_sum),
            neighbor_valid_count=rep(coarse.neighbor_valid_count),
            neighbor_weight_sum=rep(coarse.neighbor_weight_sum),
        )

    def flag_sources(
            self,
            image: np.ndarray,
            bad_mask: Optional[np.ndarray] = None,
            weight: Optional[np.ndarray] = None,

    ) -> np.ndarray:
        """
         Carries out the following steps:
         - Convolves the image with a tophat of radius pre_tophat if pre_tophat > 0
         - Compputes Moran's I, either block resampled or at the native scale depending on block size
         - Convolves the Moran's I array with a tophat if post_tophat > 0
         - Thresholds to identify high & low values of Moran's I as sources
         - Convolves this mask with a tophat if dilation_tophat > 0
         - Thresholds this dilated mask to convert back to a boolean
         - Returns a mask with True for background and False for source

        Parameters
        ----------
        image : 2D array
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
        Mask : 2D array
           True for background pixels, False for source pixels

        """
        # Smooth the image if desired
        if bad_mask is None:
            bad_mask = np.zeros(image.shape,'bool')
        else:
            bad_mask = bad_mask.astype('bool')
        valid = np.isfinite(image) & ~bad_mask
        logger.debug(f"start")
        start = time.time()
        logger.debug(f"    before pre_tophat: {rss_gb()} GB")
        if self.pre_tophat > 0:
            #cimg = convolve_fft(image,Tophat2DKernel(self.pre_tophat),allow_huge=True)
            cimg = convolve_fft(image, Tophat2DKernel(self.pre_tophat),
                                mask=bad_mask,     # True = bad/invalid pixel, excluded from the convolution
                                boundary='fill',
                                fill_value=np.nan,       # To deal with borders
                                nan_treatment='interpolate',
                                normalize_kernel=True,
                                preserve_nan=True,
                                min_wt=0.5,
                                fft_pad=True,
                                allow_huge=True)
        else:
            cimg = image
        logger.debug(f"    {np.median(cimg[valid & ~np.isnan(cimg)]) = }")
        logger.debug(f"    {np.count_nonzero(np.isfinite(cimg[valid])) = }")
        logger.debug(f"    pre_convolved: time, resources: {time.time()-start}, {rss_gb()} GB")

        # Compute the I statistic
        if self.block_size > 0:
            moransi = self.compute_block_averaged(cimg, block=self.block_size, bad_mask=bad_mask, weight=weight)
        else:
            moransi = self.compute(cimg, bad_mask=bad_mask, weight=weight)
        logger.debug(f"    {np.count_nonzero(np.isfinite(moransi.I[valid])) = }")
        logger.debug(f"    {moransi.I.min() = }, {moransi.I.max() = } {np.median(moransi.I[valid]) = }") 
        logger.debug(f"    computed I: time, resources {time.time()-start}, {rss_gb()} GB")

        # Smooth the I statistic if desired
        if self.post_tophat > 0:
            #istat = convolve_fft(moransi.I,Tophat2DKernel(self.post_tophat),allow_huge=True)
            istat = convolve_fft(moransi.I, Tophat2DKernel(self.post_tophat),
                                 mask=bad_mask,     # True = bad/invalid pixel, excluded from the convolution
                                 boundary='fill',
                                 fill_value=np.nan,       # To deal with borders
                                 nan_treatment='interpolate',
                                 normalize_kernel=True,
                                 preserve_nan=True,
                                 min_wt=0.5,
                                 fft_pad=True,
                                 allow_huge=True)
            logger.debug(f"    {np.median(istat[valid & ~np.isnan(istat)]) = }")
        else:
            istat = moransi.I
        logger.debug(f"    post_convolved: time, resources: {time.time()-start}, {rss_gb()} GB")

        # Make a source mask
        if bad_mask is not None:
            good_istat  = np.isfinite(istat) & ~bad_mask
        else:
            good_istat  = np.isfinite(istat)
        logger.debug(f"    {good_istat.dtype = } {np.count_nonzero(good_istat) = }")
        logger.debug(f"    Identified valid pixels: time, resources: {time.time()-start}, {rss_gb()} GB")
        valid_istat = istat[good_istat]
        logger.debug(f"    Selected valid pixels: time, resources: {time.time()-start}, {rss_gb()} GB")
        p50 = np.percentile(valid_istat,50.)
        p16 = np.percentile(valid_istat,16.)
        p84 = np.percentile(valid_istat,84.)
        x = (istat-p50)/((p84-p16)/2.)
        logger.debug(f"  p16, p50, p84: {p16:8.5f} {p50:8.5f} {p84:8.5f}")
        flag_as_source = valid & ((x < -self.i_lower_nsigma) | (x > self.i_upper_nsigma))
        frac_tot_masked = np.count_nonzero(flag_as_source) /  len(image.flat)
        frac_valid_masked = np.count_nonzero(flag_as_source) /  np.count_nonzero(valid)
        logger.info(f"  percent of total masked, pre-dilation: {100*frac_tot_masked:.3f}")
        logger.info(f"  percent of valid masked, pre-dilation: {100*frac_valid_masked:.3f}")
        logger.debug(f"    Thresholded: time, resources: {time.time()-start}, {rss_gb()} GB")

        # Dilate the mask, if desired
        if self.dilation_tophat > 0:
            arr = flag_as_source.astype(np.float64)
            cmask = convolve_fft(arr, Tophat2DKernel(self.dilation_tophat),
                         mask=bad_mask,     # True = bad/invalid pixel, excluded from the convolution
                         boundary='fill',
                         fill_value=np.nan,       # <-- the key fix, see below
                         nan_treatment='interpolate',
                         normalize_kernel=True,
                         preserve_nan=True,
                         min_wt=0.5,
                         fft_pad=True,
                         allow_huge=True)
            logger.debug(f"{np.median(cmask[valid & ~np.isnan(cmask)]) = }")
            flag_as_source = np.where(cmask > self.dilation_threshold,True,False)
        logger.debug(f"    Dilated: time, resources: {time.time()-start}, {rss_gb()} GB")
        frac_tot_masked = np.count_nonzero(flag_as_source) /  len(image.flat)
        frac_valid_masked = np.count_nonzero(flag_as_source) /  np.count_nonzero(valid)
        logger.info(f"  percent of total masked, post-dilation: {100*frac_tot_masked:.3f}")
        logger.info(f"  percent of valid masked, post-dilation: {100*frac_valid_masked:.3f}")

        return ~flag_as_source,istat


# ----------------------------------------------------------------------
# Example usage (not executed on import):
#
#   from local_moran import SlidingMoranSourceFilter
#
#   filt = SlidingMoranSourceFilter(
#       corr_half=1, bg_half=10, exclude_half=3,
#       sigma_clip=3.0, clip_iters=2,
#   )
#
#   bad_mask = (dq_array != 0)
#   result = filt.compute(sci_array, bad_mask=bad_mask, weight=weight_array)
#   source_mask = filt.flag_sources(result, i_thresh=2.0)
#
#   # For a larger effective footprint (e.g. to mask the wings of bright,
#   # extended sources) cheaply: block-average by `block`. corr_half /
#   # bg_half / exclude_half above are native-pixel scale either way --
#   # compute_block_averaged converts them to block-grid units itself.
#   result_ba = filt.compute_block_averaged(
#       sci_array, block=5, bad_mask=bad_mask, weight=weight_array,
#   )
#   source_mask_ba = filt.flag_sources(result_ba, i_thresh=2.0)
# ----------------------------------------------------------------------
