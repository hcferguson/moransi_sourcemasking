"""
Moran's I spatial-autocorrelation statistic, with optional inverse-
variance pixel weighting for coadded images.

The global Moran's I statistic is described in a Wikipedia article.
https://en.wikipedia.org/wiki/Moran%27s_I.

This routine slides a patch of width patch_size across the image, computing
the Global Moran's I statistic for each patch. The kernel has a width
kernel_width (must be odd), that gives the dimensions of a fully connected
square kernel with a 0 in the central pixel. This is used together the
input weight array (if provided) to weight the fluxes in each pixel when
computing the I statistic.

There is the option to pass in a mask. Masked pixels are ignored in calculating
the I statistic for the patch. Patches are rejected (NaNs) if too few pixels
remain. 

Convention: `bad_mask` is True where a pixel is unusable (e.g. DQ != 0).
Any NaN/inf pixel in the data array, or (if a weight array is given) any
non-finite or non-positive weight, is automatically treated as bad too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# For reading configuration file
import yaml
from box import Box

import numpy as np

# For variants that convolve the image or the array of I statistics
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
class SlidingGlobalIResult:
    I: np.ndarray                  # sliding-patch Global Moran's I, see compute_sliding_global_I
    patch_mean: np.ndarray         # weighted patch mean, mu_P(c)
    patch_var: np.ndarray          # weighted patch variance, m2_P(c)
    patch_valid_count: np.ndarray  # number of valid pixels used in the patch
    patch_weight_sum: np.ndarray   # sum of weights used in the patch


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
    The important parameters: 

    threshold_type: string
        'percentage' -- set the value of I so that a fixed percentage of pixels are
           designated as sky. Values of around 30-40% work well for high-latitude scenes.
        'I value' -- set a threshold in the value of the global I statistic for each patch.
          I ranges from -1 to 1. Pure Gaussian noise has a narrow peak at I ~ 0. Fixed 
          pattern noise like 1/f striping or residual amplifier bias tends to induce a 
          positive skew. Limited testing suggests that thresholds of I > 0.5 (for sources)
          are safe (not falsely masking sky patches), but leave the wings of fluffy galaxies.
          Thresholds of I > 0.3 are preferable, but might start to mask real sky pixels. 
    threshold_value: float
         A percentage of pixels to designate as sky (typically ~30-40%) or a threshold
         in the value of I (typically I ~ 0.3-0.5), depending on threshold_type.
    opening_iterations: int
         The initial global-I statistic array may be eating into the sky noise enough that there
         are small patches in the resulting source mask that don't correspond to real sources.
         One way to mitigate this while still keeping the thresholds set to address the wings
         of fluffy galaxies is to weed out the small disconnected patches from the source mask
         using a binary opening morphology operation. It is best to keep the number of iterations
         low to avoid affecting the real sources (1-3 iterations).
    n_box_passes: int
         The patches used to compute the global I statistic are square (for performance). To
         smooth out the patches so that they aren't quite so blocky, we box filter the I statistic
         array in a few passes. One pass seems better than none, but more than a few makes it 
         harder to mask out fluffy sources. 
    
    The rest of the parameters are best left at their defaults.

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
        threshold_type: 'I value',
        threshold_value: float = 0.35,
        opening_iterations: int = 2,
        n_box_passes: int = 1,
        pre_tophat: int = 0,
        post_tophat: int = 0,
        dilation_tophat: int = 0,
        dilation_threshold: int = 0.05,
        block_size: int = 0,
        kernel_width: int = 3,
        patch_size: int = 10,
        sigma_clip: Optional[float] = 3.0,
        clip_iters: int = 2,
        std_floor: float = 1e-6,
        min_valid_frac: float = 0.3,
    ):
        if patch_size <= kernel_width:
            raise ValueError("patch_size must be larger than kernel_widthf")
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.opening_iterations = opening_iterations
        self.n_box_passes = n_box_passes
        self.corr_half = kernel_width // 2
        self.bg_half = patch_size // 2
        self.sigma_clip = sigma_clip
        self.clip_iters = clip_iters
        self.std_floor = std_floor
        self.min_valid_frac = min_valid_frac
        self.pre_tophat = pre_tophat
        self.post_tophat = post_tophat
        self.dilation_tophat = dilation_tophat
        self.dilation_threshold = dilation_threshold
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

    @classmethod
    def _repeated_box_sum(cls, arr: np.ndarray, half: int, n_passes: int) -> np.ndarray:
        """
        Apply the box-sum (integral image + 4-corner lookup) `n_passes`
        times in a row, each pass using the same `half`, each pass's
        output feeding the next pass's integral image.

        This is the classic "N passes of box blur approximate a Gaussian
        blur" construction, applied here as a *sum* (not a per-pass
        average) so it stays exactly linear: running it identically on
        every numerator/denominator map in compute_sliding_global_I and
        dividing only once at the end is mathematically equivalent to
        having used one smoother, tapered, near-Gaussian *weight kernel*
        for the whole patch calculation, rather than n_passes separate
        renormalizations. n_passes=1 reduces to a single ordinary box sum
        (byte-for-byte the same as calling _box_sum_from_integral once).

        Note the effective footprint grows with n_passes at fixed `half`
        (each pass adds reach), so this isn't a drop-in same-footprint
        replacement for a single wider box -- it trades some extra reach
        for a soft-edged, tapered profile instead of a hard cutoff.
        """
        out = arr
        shape = arr.shape
        for _ in range(n_passes):
            S = cls._integral_image(out)
            out = cls._box_sum_from_integral(S, half, shape)
        return out

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

    def compute_sliding_global_I(
        self,
        image: np.ndarray,
        patch_half: Optional[int] = None,
        n_box_passes: int = 1,
        bad_mask: Optional[np.ndarray] = None,
        weight: Optional[np.ndarray] = None,
    ) -> SlidingGlobalIResult:
        """
        Sliding-window ("moving-patch") Global Moran's I.

        At every output pixel c, treats a patch centered on c as its own
        standalone "image" and computes the textbook Global Moran's I
        (Cliff & Ord; the Wikipedia "Moran's I" definition) for that
        patch, using row-standardized weights for the correlation kernel
        (corr_half). With n_box_passes=1 (the default), that patch is a
        hard-edged (2*patch_half+1) x (2*patch_half+1) box.

        This is a genuinely different statistic from compute()'s I map --
        it is *not* the same as box-averaging compute()'s per-pixel I,
        because that would mix each pixel's own separately-centered
        reference stats rather than sharing one reference across the
        whole patch. Algebraically, the row-standardized Global I of a
        patch P centered at c reduces (derivation: expand the product,
        the two mu_P^2 cross-terms cancel exactly) to:

            I(c) = Cov_P(x, nbr_mean) / Var_P(x)
                 = ( <x*nbr_mean>_P - mu_P * <nbr_mean>_P ) / m2_P

        i.e. the patch-local (weighted) covariance between the image and
        its own local neighbor-mean map, divided by the patch-local
        (weighted) variance of the image -- a moving-window correlation
        between the image and its spatial lag. Every piece here is a
        weighted box-average of some per-pixel map over the sliding
        patch, so it's built entirely from the same integral-image
        machinery used elsewhere in this class: O(1) per output pixel,
        independent of patch_half.

        Both the patch reference (mu_P, m2_P) and the neighbor-kernel
        mean (nbr_mean) are inverse-variance weighted if `weight` is
        given, consistent with compute()'s weighting convention.

        Note this uses each pixel's *ordinary* (full-image) neighbor
        mean, not one whose neighbor kernel is truncated at the patch
        boundary -- the latter would be the literal "patch as a
        completely separate image" statistic, but isn't expressible as a
        fixed set of box-sums (the truncation itself would need to slide
        with c). The difference is a small, controlled edge effect
        confined to pixels within corr_half of the patch boundary, and
        is negligible once patch_half >> corr_half.

        A hard box patch gives compact sources a flat-topped, sharp-edged
        "box" footprint in I (constant while the source stays anywhere
        inside the patch, then a sudden drop once it falls outside --
        see project notes for the derivation). n_box_passes > 1 softens
        this: instead of a single box sum, each of the six underlying
        per-pixel maps (weight, weight*x, weight*x^2, weight*nbr_mean,
        weight*x*nbr_mean, and the validity count) is box-summed
        n_box_passes times in a row -- the classic "repeated box blur
        approximates a Gaussian blur" construction -- and only combined
        into the final ratio once, at the end. Because every map gets the
        identical repeated-sum treatment before dividing, this is exactly
        equivalent to having used one smoother, tapered, near-Gaussian
        weight kernel for the whole patch from the start, not merely a
        cosmetic post-hoc smoothing of the resulting I map. It stays on
        the fast integral-image path throughout -- no FFT, no
        astropy.convolution -- costing roughly n_box_passes times a
        single pass. Note the effective footprint grows somewhat with
        n_box_passes at fixed patch_half (each pass adds some reach), so
        it trades a bit of extra reach for the softer edge.

        Parameters
        ----------
        image : 2D array
        patch_half : int, optional
            Half-width of the box used for the sliding Global I
            calculation. With n_box_passes=1 this is the patch's outer
            half-width directly; with n_box_passes>1 it's the half-width
            used for *each* of the repeated passes. Defaults to
            self.bg_half.
        n_box_passes : int, optional
            Number of times to repeat the box-sum step (see above).
            1 (default) reproduces the original hard-box statistic
            exactly. 3 is a common choice for a good box-blur-to-Gaussian
            approximation.
        bad_mask : 2D bool array, optional
            True where the pixel is *unusable* (e.g. `dq != 0`). NaN/inf
            pixels in `image` are treated as bad automatically.
        weight : 2D array, optional
            Per-pixel weight, treated as proportional to inverse variance.
            Non-finite or non-positive weights are treated as bad
            automatically. If omitted, every valid pixel gets weight 1.

        Returns
        -------
        SlidingGlobalIResult
        """
        if n_box_passes < 1:
            raise ValueError(f"n_box_passes must be >= 1, got {n_box_passes}")

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

        if patch_half is None:
            patch_half = self.bg_half

        # Neighbor-kernel mean at every pixel -- same row-standardized,
        # weighted quantity compute()'s I map uses as its correlation term.
        nbr_mean, _, nbr_W = self._neighbor_stats(image, weight, valid_mask)
        nbr_ok = np.isfinite(nbr_mean) & (nbr_W > 0)

        # A pixel only contributes to the patch sums if it's valid AND has
        # a usable neighbor mean, so every box-sum below shares the same
        # effective-weight map -- numerator and denominator stay consistent.
        combined_valid = valid_mask & nbr_ok
        w_eff = np.where(combined_valid, weight, 0.0)
        safe_image = np.where(combined_valid, image, 0.0)
        safe_nbr = np.where(combined_valid, nbr_mean, 0.0)

        cnt = self._repeated_box_sum(combined_valid.astype(np.float64), patch_half, n_box_passes)
        Wsum = self._repeated_box_sum(w_eff, patch_half, n_box_passes)
        WXsum = self._repeated_box_sum(w_eff * safe_image, patch_half, n_box_passes)
        WX2sum = self._repeated_box_sum(w_eff * safe_image * safe_image, patch_half, n_box_passes)
        WNsum = self._repeated_box_sum(w_eff * safe_nbr, patch_half, n_box_passes)
        WXNsum = self._repeated_box_sum(w_eff * safe_image * safe_nbr, patch_half, n_box_passes)

        safe_W = np.maximum(Wsum, 1e-12)
        mu_P = WXsum / safe_W
        m2_P = np.maximum(WX2sum / safe_W - mu_P ** 2, self.std_floor ** 2)
        nbrbar_P = WNsum / safe_W
        xnbrbar_P = WXNsum / safe_W

        I = (xnbrbar_P - mu_P * nbrbar_P) / m2_P

        # "Fully valid" reference count for the same repeated-box-sum
        # kernel, read off an interior point far from any edge, so the
        # min_valid_frac check generalizes correctly to n_box_passes>1
        # (whose effective footprint isn't simply (2*patch_half+1)**2).
        ref_side = 2 * (patch_half * n_box_passes) + 5
        ref_ones = np.ones((ref_side, ref_side))
        ref_smoothed = self._repeated_box_sum(ref_ones, patch_half, n_box_passes)
        max_cnt = ref_smoothed[ref_side // 2, ref_side // 2]

        low_conf = cnt < (self.min_valid_frac * max_cnt)
        bad = low_conf | (Wsum <= 0) | ~valid_mask
        I = np.where(bad, np.nan, I)

        return SlidingGlobalIResult(
            I=I,
            patch_mean=mu_P,
            patch_var=m2_P,
            patch_valid_count=cnt,
            patch_weight_sum=Wsum,
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

    def flag_sources(
            self,
            image: np.ndarray,
            bad_mask: Optional[np.ndarray] = None,
            weight: Optional[np.ndarray] = None,

    ) -> np.ndarray:
        """
         Carries out the following steps:
         - Convolves the image with a tophat of radius pre_tophat if pre_tophat > 0
         - Convolves the Moran's I array with a tophat if post_tophat > 0
         - Thresholds to identify high values of Moran's I as sources
         - Fills in holes in this mask
         - Optionally removes small disconnected regions in this mask (which are probably not real sources)
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
        moransi = self.compute_sliding_global_I(cimg, bad_mask=bad_mask, weight=weight, n_box_passes = self.n_box_passes)
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

        # Set the I threshold
        # Either a fixed percentage of the pixels are designated as sky
        if self.threshold_type == 'percentile':  
             condition = istat > np.percentile(valid_istat,100.-self.threshold_value)
        # Or a fixed threshold in the I value is used
        else:
             condition = (istat > self.threshold_value)

        # Create the mask (True = Source)
        mask = np.where(condition,True,False)

        # Fill holes
        mask  = binary_fill_holes(mask)

        # Erode small segments
        if self.opening_iterations > 0:
             mask = binary_opening(mask,iterations=self.opening_iterations)

        # Report some statistics
        flag_as_source = mask
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

# Convenience functions to read parameters from a yaml file and use that to drive
# the sourcemasking

def read_config(configfile):
    ''' Read yaml configuration file '''
    with open(configfile) as f:
        config = Box(yaml.safe_load(f))
    return config

def make_sourcemask(image,configfile,bad_mask=None,weight=None):
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
    config = read_config(configfile)
    filt = SlidingMoranSourceFilter(**config)
    mask, istat = filt.flag_sources(image,bad_mask=bad_mask,weight=weight)
    return mask

