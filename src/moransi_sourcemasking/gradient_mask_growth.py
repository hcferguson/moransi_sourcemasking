"""
Gradient-significance based mask growth.

This fits a local weighted plane (intercept + gx + gy) to the 
image in a window around every pixel, using only currently-unmasked 
pixels. It then projects the fitted gradient onto the direction toward the
nearest masked pixel, and tests whether that "inward" gradient is
significant relative to its own analytically-propagated uncertainty.

Development assisted by claud.ai
"""

import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 2. Vectorized weighted local plane fit -> gradient + its covariance
# ----------------------------------------------------------------------
def _sep(arr, k0, ax0, k1, ax1):
    """
    Apply two 1D correlations in sequence, one per axis -- the
    separable-filter building block used throughout this module to
    compute local weighted moment sums over the whole image at once
    (no explicit per-pixel window loop).

    Parameters
    ----------
    arr : 2D array
        Array to filter.
    k0 : 1D array
        Kernel applied first, along axis `ax0`.
    ax0 : int
        Axis (0 or 1) for the first correlation.
    k1 : 1D array
        Kernel applied second, along axis `ax1`.
    ax1 : int
        Axis (0 or 1) for the second correlation.

    Returns
    -------
    2D array, same shape as `arr`.
    """
    a = ndi.correlate1d(arr, k0, axis=ax0, mode='constant', cval=0.0)
    return ndi.correlate1d(a, k1, axis=ax1, mode='constant', cval=0.0)


def _moment_sums(weight, half_size):
    """
    The 6 distinct entries of the symmetric 3x3 design-matrix
    sum(weight * [1,drow,dcol] outer [1,drow,dcol]) over a
    (2*half_size+1)^2 window, at every pixel, via separable 1D
    correlations. Reused for both the plain design matrix (weight =
    0/1 mask) and the variance "sandwich" matrix (weight = mask *
    sigma_map^2) needed for a spatially-varying noise map.

    Parameters
    ----------
    weight : 2D array
        Per-pixel weight to sum over each window. For the plain
        design matrix this is the 0/1 fit mask; for the variance
        sandwich matrix (see local_gradient_variance) it is the fit
        mask times a per-pixel noise-variance map.
    half_size : int
        Window half-size; the window is (2*half_size+1) pixels on a
        side, centered on each pixel.

    Returns
    -------
    (S0, Srow, Scol, Srr, Scc, Src) : tuple of 2D arrays, each the
        same shape as `weight`. S0 = sum(weight); Srow, Scol = first
        moments (sum of weight*drow, weight*dcol); Srr, Scc, Src =
        second moments/cross-moment, where (drow, dcol) are pixel
        offsets from the window center.
    """
    r = half_size
    d = np.arange(-r, r + 1, dtype=float)
    box, ramp, ramp2 = np.ones_like(d), d.copy(), d**2
    S0   = _sep(weight, box,   0, box,   1)
    Srow = _sep(weight, ramp,  0, box,   1)
    Scol = _sep(weight, box,   0, ramp,  1)
    Srr  = _sep(weight, ramp2, 0, box,   1)
    Scc  = _sep(weight, box,   0, ramp2, 1)
    Src  = _sep(weight, ramp,  0, ramp,  1)
    return S0, Srow, Scol, Srr, Scc, Src


def _assemble_3x3(sums):
    """
    Assemble the 6 distinct moment sums from _moment_sums into a full
    per-pixel symmetric 3x3 matrix.

    Parameters
    ----------
    sums : tuple of 2D arrays
        (S0, Srow, Scol, Srr, Scc, Src), as returned by _moment_sums.

    Returns
    -------
    M : (ny, nx, 3, 3) array
        Per-pixel 3x3 matrix, parameter order (intercept, g_row, g_col).
    """
    S0, Srow, Scol, Srr, Scc, Src = sums
    ny, nx = S0.shape
    M = np.zeros((ny, nx, 3, 3))
    M[..., 0, 0] = S0;   M[..., 0, 1] = Srow; M[..., 0, 2] = Scol
    M[..., 1, 0] = Srow; M[..., 1, 1] = Srr;  M[..., 1, 2] = Src
    M[..., 2, 0] = Scol; M[..., 2, 1] = Src;  M[..., 2, 2] = Scc
    return M


def local_weighted_gradient(image, weight, half_size):
    """
    At every pixel, fit intensity ~ a0 + g_row*drow + g_col*dcol over
    a (2*half_size+1)^2 window, weighted by `weight` (0/1, excludes
    masked pixels).

    Axis convention: axis 0 = rows, axis 1 = columns, matching the
    image array's own indexing -- no separate x/y labeling to avoid
    transposition bugs.

    Parameters
    ----------
    image : 2D array
        The image to fit local planes to.
    weight : 2D array, same shape as `image`
        0/1 (or otherwise non-negative) per-pixel weight for the fit;
        0 excludes a pixel entirely (e.g. currently masked or bad).
    half_size : int
        Fit-window half-size in pixels; the window is
        (2*half_size+1)^2 pixels, centered on each pixel.

    Returns
    -------
    g_row : 2D array
        Fitted d(intensity)/d(row) (axis-0 gradient component) at
        every pixel.
    g_col : 2D array
        Fitted d(intensity)/d(col) (axis-1 gradient component) at
        every pixel.
    Minv : (ny, nx, 3, 3) array
        Per-pixel inverse design matrix. Its [1:,1:] 2x2 block times
        sigma_pix^2 gives Cov(g_row, g_col) for *homogeneous* per-
        pixel noise -- see local_gradient_variance for the general
        (spatially-varying noise) case.
    valid : 2D bool array
        True where the local fit was well-determined (enough unmasked
        pixels in the window, and a non-singular design matrix).
        False (with g_row=g_col=0, Minv=0) elsewhere, e.g. deep inside
        a large mask or right at the image edge.
    """
    W = weight.astype(float)
    WI = W * image
    r = half_size
    d = np.arange(-r, r + 1, dtype=float)
    box, ramp = np.ones_like(d), d.copy()

    sums = _moment_sums(W, half_size)
    M = _assemble_3x3(sums)
    SI    = _sep(WI, box,  0, box,  1)
    SrowI = _sep(WI, ramp, 0, box,  1)
    ScolI = _sep(WI, box,  0, ramp, 1)
    b = np.stack([SI, SrowI, ScolI], axis=-1)

    S0 = sums[0]
    ny, nx = image.shape

    # Guard against singular / underdetermined fits (too few unmasked
    # pixels in the window, e.g. deep inside a big mask or at the edge).
    # IMPORTANT: check determinant per-pixel rather than calling
    # np.linalg.inv on the whole batch and catching LinAlgError --
    # a batched inv() raises (and previously nuked validity for the
    # WHOLE image) if even a single matrix in the stack is singular,
    # which becomes likely with many sources / masked edges.
    min_pix = 6  # need > 3 for a plane fit; pad a bit for stability
    valid = S0 >= min_pix

    det = np.zeros((ny, nx))
    det[valid] = np.linalg.det(M[valid])
    det_scale = np.maximum(np.abs(det), 1.0)
    valid &= np.abs(det) > 1e-8 * det_scale

    coeff = np.zeros((ny, nx, 3))
    Minv = np.zeros((ny, nx, 3, 3))
    if valid.any():
        Minv_v = np.linalg.inv(M[valid])  # safe now: none of these are singular
        coeff[valid] = np.einsum('nij,nj->ni', Minv_v, b[valid])
        Minv[valid] = Minv_v

    g_row = coeff[..., 1]
    g_col = coeff[..., 2]

    return g_row, g_col, Minv, valid


def local_gradient_variance(weight, half_size, Minv, sigma_pix):
    """
    Cov(g_row, g_col), as the [1:,1:] 2x2 block, per pixel.

    Parameters
    ----------
    weight : 2D array
        The same 0/1 fit-weight array passed to local_weighted_gradient.
    half_size : int
        The same fit-window half-size passed to local_weighted_gradient.
    Minv : (ny, nx, 3, 3) array
        The inverse design matrix returned by local_weighted_gradient.
    sigma_pix : float or 2D array
        Per-pixel sky noise (1-sigma), either:
          - scalar -> cheap path, Cov = sigma_pix^2 * Minv[1:,1:].
            Valid only for homogeneous per-pixel noise.
          - 2D array, same shape as the image -> full sandwich
            estimator Cov = Minv @ T @ Minv, with T built from the
            same moment-sum machinery but weight -> weight*sigma_map^2.
            Needed whenever noise varies across the fit window (e.g. a
            real inverse-variance weight map), since a single window
            can straddle very different exposure depths.

    Returns
    -------
    C : (ny, nx, 2, 2) array
        Per-pixel covariance of (g_row, g_col).
    """
    if np.isscalar(sigma_pix):
        return (sigma_pix**2) * Minv[..., 1:, 1:]

    sigma_map = np.asarray(sigma_pix, dtype=float)
    w2 = weight.astype(float) * sigma_map**2
    T = _assemble_3x3(_moment_sums(w2, half_size))
    sandwich = np.matmul(np.matmul(Minv, T), Minv)  # batched 3x3 matmul
    return sandwich[..., 1:, 1:]


def inward_unit_vector(mask):
    """
    For every pixel, unit vector (row, col components) pointing from
    that pixel toward the nearest currently-masked pixel (the
    "inward" direction), via a single Euclidean distance transform.
    Also returns the distance itself and the (row, col) indices of
    that nearest masked pixel -- reused by gradient_grow_mask both to
    define genuinely circular (not square/diamond) growth rings, and
    to assign each ring pixel to its nearest source label (a proper
    Voronoi-style assignment for crowded/adjacent sources).

    Parameters
    ----------
    mask : 2D bool array
        True where a pixel is currently masked (source).

    Returns
    -------
    u_row : 2D array
        Row component of the inward-pointing unit vector at every
        pixel (0 at masked pixels themselves, or wherever distance is
        0).
    u_col : 2D array
        Column component of the inward-pointing unit vector.
    dist : 2D array
        Euclidean distance from each pixel to the nearest masked
        pixel (0 at masked pixels).
    irow : 2D int array
        Row index of the nearest masked pixel, for every pixel.
    icol : 2D int array
        Column index of the nearest masked pixel, for every pixel.
    """
    dist, (irow, icol) = ndi.distance_transform_edt(~mask, return_indices=True)
    rr, cc = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    drow = irow - rr
    dcol = icol - cc
    d = np.sqrt(drow**2 + dcol**2)
    d_safe = np.where(d > 0, d, 1.0)
    u_row = drow / d_safe
    u_col = dcol / d_safe
    return u_row, u_col, dist, irow, icol


def gradient_significance_map(image, mask, half_size, sigma_pix, good_pixels=None):
    """
    Returns a per-pixel z-score: the component of the locally-fitted
    gradient pointing *toward* the nearest masked pixel, divided by
    its analytic standard error. Positive & significant => intensity
    is still rising toward the mask => likely still on a real source.

    Parameters
    ----------
    image : 2D array
        The image to test.
    mask : 2D bool array
        True where a pixel is currently masked (source); defines both
        the local fit's excluded region and the "inward" direction.
    half_size : int
        Fit-window half-size in pixels, passed to
        local_weighted_gradient.
    sigma_pix : float or 2D array
        Per-pixel sky noise (1-sigma); scalar for homogeneous noise,
        or a 2D array (e.g. sqrt(1/weight) from a real inverse-
        variance weight map) for spatially-varying noise -- see
        local_gradient_variance.
    good_pixels : 2D bool array, optional, same shape as `image`
        True where a pixel is usable data (finite, not flagged bad).
        Excluded from the local fit's weight the same way masked/
        source pixels are, but -- unlike `mask` -- NOT labeled as its
        own growable region, so a bad pixel doesn't spuriously seed a
        "source" that then grows. Defaults to all-True (no separate
        bad-pixel concept).

    Returns
    -------
    z : 2D array
        Per-pixel significance (z-score) of the inward gradient. 0
        wherever the local fit was invalid (see `valid`) or the
        standard error was 0.
    valid : 2D bool array
        True where the local plane fit was well-determined (as
        returned by local_weighted_gradient).
    dist : 2D array
        Euclidean distance to the nearest masked pixel (from
        inward_unit_vector).
    irow : 2D int array
        Row index of the nearest masked pixel, for every pixel.
    icol : 2D int array
        Column index of the nearest masked pixel, for every pixel.
    """
    if good_pixels is None:
        good_pixels = np.ones(image.shape, dtype=bool)
    weight = (~mask & good_pixels).astype(float)
    g_row, g_col, Minv, valid = local_weighted_gradient(image, weight, half_size)
    u_row, u_col, dist, irow, icol = inward_unit_vector(mask)
    C = local_gradient_variance(weight, half_size, Minv, sigma_pix)

    g_in = g_row * u_row + g_col * u_col  # gradient component toward mask

    var_g_in = (
        u_row**2 * C[..., 0, 0] + u_col**2 * C[..., 1, 1] +
        2 * u_row * u_col * C[..., 0, 1]
    )
    se = np.sqrt(np.clip(var_g_in, 0, None))
    z = np.where((se > 0) & valid, g_in / np.where(se > 0, se, 1.0), 0.0)
    return z, valid, dist, irow, icol


# ----------------------------------------------------------------------
# 3. Growth loop using the gradient z-score instead of ring means
# ----------------------------------------------------------------------
def gradient_grow_mask(image, mask, half_size=8, k=3.0, max_iter=40,
                        radial_step=1.0, label_structure=None,
                        sigma_pix=None, agg='median', good_pixels=None):
    """
    Adaptively grow a boolean source mask outward, using local
    gradient significance (rather than a fixed dilation radius or a
    ring-mean test) to decide, per source and per growth ring,
    whether the mask should keep expanding.

    Growth rings are defined by true Euclidean distance from the
    *original* seed mask (computed once, up front -- see the
    `dist0`/`nearest_label0` comment in the implementation for why),
    not by iterating a fixed structuring element: iterating a square
    (8-connected) or cross (4-connected) structuring element grows
    the mask in the Chebyshev or Manhattan metric respectively, whose
    iso-distance contours are axis-aligned squares/diamonds, no
    matter how small a step is taken. Euclidean rings give a genuinely
    isotropic (circular) growth front regardless of iteration count
    or image orientation. Each ring pixel is assigned to its nearest
    source label via the distance transform's own nearest-point
    indices -- a proper Voronoi assignment, which also handles
    adjacent/crowded sources cleanly.

    At each iteration, every source still marked "active" is offered
    the next ring of pixels at Euclidean distance <= (iteration+1) *
    radial_step from its seed. A source keeps that ring (and stays
    active for the next iteration) if the ring's aggregate gradient
    z-score (see gradient_significance_map) exceeds `k`; otherwise
    that source is frozen at its current extent for the rest of the
    run. Sources are grown independently and in parallel (one pass
    over the whole image per iteration, not one pass per source).

    Parameters
    ----------
    image : 2D array
        The image to grow the mask on.
    mask : 2D bool array
        The initial (seed) source mask. True = source. Each separate
        connected region (per `label_structure`) is grown
        independently.
    half_size : int, optional (default 8)
        Half-size (pixels) of the local plane-fit window used by the
        significance test at each ring -- see
        gradient_significance_map / local_weighted_gradient. Larger
        values average over more pixels, giving more sensitivity to
        shallow, extended wings, but do not by themselves change how
        far a single iteration grows the mask (see `radial_step`).
    k : float, optional (default 3.0)
        Significance threshold, in units of the ring's aggregate
        z-score (see `agg`). A ring is kept (source keeps growing)
        while its aggregate z-score exceeds `k`; growth for that
        source stops the first time it doesn't.
    max_iter : int, optional (default 40)
        Maximum number of growth iterations. Each iteration advances
        the growth front by `radial_step` (Euclidean pixels) from the
        original seed, so the maximum radius any source can reach is
        roughly `max_iter * radial_step`. Independent of `half_size`.
    radial_step : float, optional (default 1.0)
        Euclidean-distance thickness (in pixels; need not be an
        integer) of each growth ring/iteration. Smaller gives
        finer-grained stopping resolution (the final boundary sits
        closer to the true significance cutoff) at the cost of more
        iterations to reach a given radius; larger reaches a given
        radius in fewer iterations at coarser stopping resolution.
    label_structure : structuring element, optional
        Connectivity used only to identify separate source blobs in
        the *input* `mask` (passed to `scipy.ndimage.label`) --
        unrelated to the shape of growth itself, which is always
        Euclidean regardless of this. Defaults to full (8-connected)
        connectivity.
    sigma_pix : float, 2D array, or None, optional
        Per-pixel sky noise (1-sigma) used by the significance test;
        see gradient_significance_map / local_gradient_variance for
        the scalar-vs-array distinction. If None (default), a single
        scalar sigma is estimated robustly (1.4826 * MAD) from
        `image[good_pixels]`.
    agg : {'median', 'mean'}, optional (default 'median')
        How the per-pixel z-scores within a ring are aggregated into
        the single value compared against `k`. 'median' is robust to
        a ring being partly contaminated (e.g. by a background
        gradient or a neighboring source on one side).
    good_pixels : 2D bool array, optional, same shape as `image`
        True where a pixel is usable data (finite, not flagged bad in
        some separate bad-pixel/DQ sense). Excluded from every local
        fit's weight, and never grown into (a bad pixel is unusable,
        not a detection), but -- unlike `mask` -- not itself labeled
        as a growable region, so a bad pixel can't spuriously seed
        its own "source". Defaults to all-True (every pixel usable).

    Returns
    -------
    current_mask : 2D bool array, same shape as `mask`
        The grown source mask.
    current_labels : 2D int array, same shape as `mask`
        Per-pixel source label (0 = background), matching the labels
        `scipy.ndimage.label` assigned to the original seed regions
        -- i.e. label values are stable between the seed mask and the
        grown mask, so grown pixels can be traced back to the seed
        they grew from.
    """
    if label_structure is None:
        label_structure = ndi.generate_binary_structure(mask.ndim, 2)

    if good_pixels is None:
        good_pixels = np.ones(image.shape, dtype=bool)

    if sigma_pix is None:
        med = np.median(image[good_pixels])
        sigma_pix = 1.4826 * np.median(np.abs(image[good_pixels] - med))

    labels, nsrc = ndi.label(mask, structure=label_structure)
    if nsrc == 0:
        return mask.copy(), labels

    current_mask = mask.copy()
    current_labels = labels.copy()
    active = np.ones(nsrc + 1, dtype=bool)
    active[0] = False

    # Ring geometry is referenced to the ORIGINAL seed mask, computed
    # once, not recomputed against the current (already-grown) mask
    # each iteration. This matters: repeatedly thresholding distance
    # from an ever-changing current mask at a small fixed radial_step
    # just re-derives some fixed local neighborhood's own lattice
    # metric (diamond for step<sqrt(2), square for sqrt(2)<=step<2,
    # etc.) no matter how small the step is -- iterating a small local
    # rule never converges to a circle. Thresholding a single, fixed
    # distance map at a growing absolute radius does, because it's
    # just one correct Euclidean distance field read at increasing
    # cutoffs, with no per-step compounding of grid quantization.
    # (This also fixes the seed -> source-label assignment for the
    # whole run: each pixel's nearest *original* seed, a stable
    # Voronoi partition, rather than one that could drift as masks
    # grow unevenly.)
    dist0, (irow0, icol0) = ndi.distance_transform_edt(~mask, return_indices=True)
    nearest_label0 = labels[irow0, icol0]

    agg_func = np.median if agg == 'median' else np.mean

    for it in range(max_iter):
        if not active[1:].any():
            break

        z_map, valid, dist, irow, icol = gradient_significance_map(
            image, current_mask, half_size, sigma_pix, good_pixels=good_pixels
        )

        r = (it + 1) * radial_step
        shell = (~current_mask) & good_pixels & (dist0 <= r) & active[nearest_label0]
        if not shell.any():
            break

        shell_labels = np.where(shell, nearest_label0, 0)
        present = np.unique(shell_labels)
        present = present[present > 0]
        if present.size == 0:
            break

        ring_z = ndi.labeled_comprehension(
            z_map, shell_labels, present, agg_func, float, 0.0
        )

        keep = ring_z > k
        active[present[~keep]] = False

        grow_ids = present[keep]
        if grow_ids.size:
            add_px = np.isin(shell_labels, grow_ids)
            current_mask |= add_px
            current_labels[add_px] = shell_labels[add_px]

    return current_mask, current_labels


##################################################################################
# Testing
#
# ----------------------------------------------------------------------
# 1. Synthetic test image 
# ----------------------------------------------------------------------
def make_test_image(shape=(200, 200), sky_level=100.0, sky_sigma=5.0,
                     sources=None, seed=0):
    """
    Build a synthetic test image: a flat sky with Gaussian noise, plus
    zero or more 2D Gaussian sources.

    Parameters
    ----------
    shape : (int, int), optional (default (200, 200))
        Image shape (ny, nx).
    sky_level : float, optional (default 100.0)
        Constant sky background level.
    sky_sigma : float, optional (default 5.0)
        Standard deviation of the (Gaussian, uncorrelated) sky noise.
    sources : list of dict, or None, optional
        Each dict needs keys 'x0', 'y0' (center, column/row pixel
        coordinates), 'amp' (peak amplitude above sky), and 'sigma'
        (Gaussian width in pixels). If None (default), three example
        sources are used: bright/broad, moderate, and faint/diffuse.
    seed : int, optional (default 0)
        Seed for the noise random number generator (reproducible).

    Returns
    -------
    image : 2D array, shape `shape`
        The synthetic image (sky + sources + noise).
    sources : list of dict
        The source parameters actually used (the `sources` argument,
        or the default list if it was None).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    image = np.zeros(shape, dtype=float)
    if sources is None:
        sources = [
            dict(x0=50,  y0=50,  amp=80.0, sigma=6.0),   # bright, broad
            dict(x0=140, y0=60,  amp=40.0, sigma=4.0),   # moderate
            dict(x0=100, y0=140, amp=15.0, sigma=8.0),   # faint, very diffuse
        ]
    for s in sources:
        r2 = (xx - s['x0'])**2 + (yy - s['y0'])**2
        image += s['amp'] * np.exp(-0.5 * r2 / s['sigma']**2)
    image += sky_level
    image += rng.normal(0.0, sky_sigma, size=shape)
    return image, sources


def make_seed_mask(shape, sources, seed_radius=3):
    """
    Build a small circular seed mask (e.g. from a hypothetical
    detection step) around each source's center, for testing
    gradient_grow_mask.

    Parameters
    ----------
    shape : (int, int)
        Mask shape (ny, nx).
    sources : list of dict
        Source parameters as returned by make_test_image; only 'x0'
        and 'y0' are used.
    seed_radius : float, optional (default 3)
        Radius (pixels) of the circular seed placed at each source's
        center.

    Returns
    -------
    mask : 2D bool array, shape `shape`
        True within `seed_radius` pixels of any source center.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mask = np.zeros(shape, dtype=bool)
    for s in sources:
        r2 = (xx - s['x0'])**2 + (yy - s['y0'])**2
        mask |= r2 <= seed_radius**2
    return mask
# ----------------------------------------------------------------------
# 4. Run and compare against the old ring-mean method
# ----------------------------------------------------------------------
if __name__ == "__main__":
    shape = (200, 200)
    sky_level = 100.0
    sky_sigma = 5.0

    image, sources = make_test_image(shape=shape, sky_level=sky_level,
                                      sky_sigma=sky_sigma)
    seed_mask = make_seed_mask(shape, sources, seed_radius=3)

    grown_mask, labels = gradient_grow_mask(
        image, seed_mask, half_size=8, k=3.0, max_iter=40,
        sigma_pix=sky_sigma
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = sky_level - 3*sky_sigma, sky_level + 40

    axes[0].imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Synthetic image")

    axes[1].imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].contour(seed_mask, colors='cyan', linewidths=1)
    axes[1].contour(grown_mask, colors='red', linewidths=1)
    axes[1].set_title("Seed (cyan) vs gradient-grown (red)")

    axes[2].imshow(grown_mask, origin='lower', cmap='gray')
    axes[2].set_title("Final grown mask")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/gradient_grow_test.png", dpi=130)
    print("Saved plot to gradient_grow_test.png")

    for sid in range(1, len(sources) + 1):
        npix_seed = np.sum(ndi.label(seed_mask)[0] == sid)
        npix_grown = np.sum(labels == sid)
        print(f"Source {sid}: seed_npix={npix_seed}, grown_npix={npix_grown}, "
              f"amp={sources[sid-1]['amp']}, sigma={sources[sid-1]['sigma']}")
