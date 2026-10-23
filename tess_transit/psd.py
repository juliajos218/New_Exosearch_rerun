"""PSD estimation using Kay's smoothed periodogram.

Provides spectral estimation for TESS light curves with gaps, used by both
the FFT and gather detection pipelines.
"""


from typing import Tuple

import numpy as np

from tess_transit.data.lightcurve import LightCurve


def threshold_outliers(data: np.ndarray, level: float = 3.0) -> np.ndarray:
    """Two-pass sigma clip with Gaussian replacement.

    First pass: coarse threshold to remove extremal values.
    Second pass: recalculate std from clean data and apply final threshold.
    Outliers are replaced with Gaussian random samples at the clean std.

    Parameters
    ----------
    data : np.ndarray
        1-D array (modified copy is returned).
    level : float
        Threshold in units of standard deviation.

    Returns
    -------
    np.ndarray
        Cleaned copy of *data*.
    """
    data = data.copy()
    std_ = np.nanstd(data)
    diff = np.diff(data, prepend=data[0])
    thresh = level * std_

    # First pass
    mask = np.abs(data) > thresh
    mask |= np.abs(diff) > thresh

    std_clean = np.nanstd(data[~mask]) if np.sum(~mask) > 0 else std_
    thresh = level * std_clean

    # Second pass
    final_mask = np.abs(data) > thresh
    final_mask |= np.abs(diff) > thresh

    if final_mask.sum() > 0:
        data[final_mask] = np.random.normal(0, std_clean, size=final_mask.sum())
    return data


def smooth_periodogram(signal: np.ndarray, K: int = 3) -> np.ndarray:
    """Kay's smoothed periodogram: P(k) = |FFT|^2 / N, smoothed over 2K+1 bins.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal.
    K : int
        Smoothing half-width.

    Returns
    -------
    np.ndarray
        Smoothed power spectral density, same length as *signal*.
    """
    N = len(signal)
    p_raw = (1.0 / N) * np.abs(np.fft.fft(signal)) ** 2
    smoothed = np.zeros(N)
    for i in range(N):
        lo = max(0, i - K)
        hi = min(N, i + K + 1)
        smoothed[i] = np.sum(p_raw[lo:hi])
    smoothed /= 2 * K
    return smoothed


def kays_correction_factor(K: int) -> float:
    """Bias correction factor for Kay's smoothed periodogram.

    Returns beta = sqrt((2K - 1) / (2K)).
    """
    return np.sqrt((2 * K - 1) / (2 * K))


def detect_transit_candidates(
    flux: np.ndarray,
    mask: np.ndarray,
    duration: int,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Detect potential transit-like features for masking before PSD estimation.

    Uses a matched filter approach: convolve with a box of width `duration` to
    find regions with significant negative depth. These regions are marked for
    masking to prevent real transits from contaminating the noise estimate.

    Parameters
    ----------
    flux : np.ndarray
        Flux values on the full cadence grid (gaps filled with zeros or noise).
    mask : np.ndarray
        Boolean mask, True where data is observed.
    duration : int
        Transit duration in cadences.
    threshold_sigma : float
        Detection threshold in units of the box-filtered noise std.

    Returns
    -------
    np.ndarray
        Boolean mask, True where potential transits are detected.
        These regions should be excluded from PSD estimation.
    """
    N = len(flux)
    if duration >= N:
        return np.zeros(N, dtype=bool)

    # Box filter: running sum over duration-width window
    # Use cumsum for efficiency
    cumsum = np.cumsum(flux)
    cumsum = np.insert(cumsum, 0, 0)
    box_sum = cumsum[duration:] - cumsum[:-duration]  # length N - duration + 1

    # Normalize by number of observed points in each window
    mask_float = mask.astype(np.float64)
    mask_cumsum = np.cumsum(mask_float)
    mask_cumsum = np.insert(mask_cumsum, 0, 0)
    obs_count = mask_cumsum[duration:] - mask_cumsum[:-duration]

    # Avoid division by zero; require at least half the window observed
    min_obs = duration // 2
    valid_windows = obs_count >= min_obs
    box_mean = np.zeros(len(box_sum))
    box_mean[valid_windows] = box_sum[valid_windows] / obs_count[valid_windows]

    # Estimate noise level of the box-filtered signal
    # Use MAD (median absolute deviation) for robustness to outliers
    valid_vals = box_mean[valid_windows]
    if len(valid_vals) < 10:
        return np.zeros(N, dtype=bool)

    median_val = np.median(valid_vals)
    mad = np.median(np.abs(valid_vals - median_val))
    sigma = 1.4826 * mad  # MAD to sigma conversion for Gaussian

    if sigma < 1e-10:
        return np.zeros(N, dtype=bool)

    # Detect significant negative dips (transits are flux decreases)
    threshold = -threshold_sigma * sigma
    transit_windows = (box_mean < threshold) & valid_windows

    # Expand detected windows to mark all affected cadences
    # Each detected window at position i covers cadences [i, i+duration)
    transit_mask = np.zeros(N, dtype=bool)
    detected_indices = np.where(transit_windows)[0]

    for idx in detected_indices:
        start = max(0, idx)
        end = min(N, idx + duration)
        transit_mask[start:end] = True

    # Add a buffer of duration/2 on each side
    buffer = duration // 2
    if buffer > 0 and transit_mask.any():
        # Dilate the mask
        dilated = np.zeros(N, dtype=bool)
        for i in range(N):
            if transit_mask[i]:
                start = max(0, i - buffer)
                end = min(N, i + buffer + 1)
                dilated[start:end] = True
        transit_mask = dilated

    return transit_mask


def estimate_psd(lc: LightCurve, K: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate PSD from a gapped light curve.

    Steps:
        1. Threshold outliers in observed flux.
        2. Place cleaned flux on full cadence grid; fill gaps with white noise.
        3. Zero-pad to ``2*N - 1``.
        4. Compute smoothed periodogram.

    Parameters
    ----------
    lc : LightCurve
        Per-sector light curve.
    K : int
        Smoothing half-width for Kay's periodogram.

    Returns
    -------
    psd : np.ndarray
        Smoothed PSD of length ``2 * lc.n_full - 1``.
    cleaned_flux : np.ndarray
        Outlier-cleaned flux (same length as ``lc.flux``).
    """
    N = lc.n_full
    cadence_zero = lc.cadence_zero
    obs_mask = lc.observed_mask

    # Clean outliers
    cleaned_flux = threshold_outliers(lc.flux.copy(), level=3.0)

    # Place on full grid
    filled = np.zeros(N, dtype=np.float64)
    valid = (cadence_zero >= 0) & (cadence_zero < N)
    filled[cadence_zero[valid]] = cleaned_flux[valid]

    # Fill gaps with white noise at observed std
    std_ = np.std(cleaned_flux)
    gap_mask = ~obs_mask
    if gap_mask.sum() > 0:
        filled[gap_mask] = np.random.normal(0, std_, gap_mask.sum())

    # Zero-pad and compute PSD
    zp = np.zeros(2 * N - 1)
    zp[:N] = filled
    psd = smooth_periodogram(zp, K=K)

    return psd, cleaned_flux
