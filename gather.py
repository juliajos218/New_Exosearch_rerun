"""Gather-based transit detection (CPU + GPU reference/fallback).

This preserves the exact gather-based detection pipeline as a reference
implementation.  The FFT method (``fft_detect.py``) is preferred for
production use; gather is useful for validation and for cases where the
full covariance inverse is available.
"""


from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import convolve1d

from tess_transit.core.psd import threshold_outliers, smooth_periodogram
from tess_transit.data.lightcurve import LightCurve


# Re-use the same SectorResult dataclass as FFT for interoperability
from tess_transit.core.fft_detect import SectorResult


# =============================================================================
# CUDA kernel source (compiled on demand)
# =============================================================================

GATHER_KERNEL = r"""
__global__ void gather_transit_num(float *output, const float *input, int half_N)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int p = ty + 1;
    int no_transit = floorf(((2 * half_N) - tx - 1) / p) + 1;
    float value = 0.0f;

    if (tx < p && p <= half_N)
    {
        for (int i = 0; i < no_transit; ++i)
        {
            value += input[(i * p) + tx];
        }
        output[(half_N * ty) + tx] = value;
    }
}

__global__ void gather_transit_den(float *output, const float *input, int half_N)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int p = ty + 1;
    int t0 = tx;
    int no_transit = floorf(((2 * half_N) - t0 - 1) / p) + 1;
    float value = 0.0f;

    if (t0 < p && p <= half_N)
    {
        for (int i = 0; i < no_transit; ++i)
        {
            for (int j = 0; j < no_transit; ++j)
            {
                value += input[((i * p) + tx) * (2 * half_N) + (j * p) + tx];
            }
        }
        output[(half_N * (p - 1)) + t0] = value;
    }
}
"""


# =============================================================================
# CPU gather kernels
# =============================================================================

def gather_num_cpu(y_d: np.ndarray, half_N: int) -> np.ndarray:
    """CPU gather for numerator."""
    output = np.zeros((half_N, half_N), dtype=np.float64)
    for ty in range(half_N):
        p = ty + 1
        for t0 in range(p):
            output[ty, t0] = np.sum(y_d[t0::p])
    return output


def gather_den_cpu(K_d: np.ndarray, half_N: int) -> np.ndarray:
    """CPU gather for denominator."""
    output = np.zeros((half_N, half_N), dtype=np.float64)
    for ty in range(half_N):
        p = ty + 1
        for t0 in range(p):
            output[ty, t0] = np.sum(K_d[t0::p, t0::p])
    return output


# =============================================================================
# GPU gather kernels
# =============================================================================

def gather_gpu(y_d: np.ndarray, K_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run gather kernels on GPU via PyCUDA."""
    import math
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    mod = SourceModule(GATHER_KERNEL)
    gather_num_fn = mod.get_function("gather_transit_num")
    gather_den_fn = mod.get_function("gather_transit_den")

    half_N = len(y_d) // 2
    output_num = np.zeros((half_N, half_N), dtype=np.float32)
    output_den = np.zeros((half_N, half_N), dtype=np.float32)

    block_dim = (32, 8, 1)
    grid_dim = (
        int(math.ceil(half_N / block_dim[0])),
        int(math.ceil(half_N / block_dim[1])),
    )

    gather_num_fn(
        drv.Out(output_num),
        drv.In(y_d.astype(np.float32)),
        np.int32(half_N),
        block=block_dim,
        grid=grid_dim,
    )
    gather_den_fn(
        drv.Out(output_den),
        drv.In(K_d.astype(np.float32)),
        np.int32(half_N),
        block=block_dim,
        grid=grid_dim,
    )
    return output_num, output_den


# =============================================================================
# Full gather pipeline for a single sector
# =============================================================================

def gather_sector_search(
    lc: LightCurve,
    duration: int,
    delta: int = 5,
    K: int = 3,
    use_gpu: bool = False,
    regularization: float = 1e-6,
) -> SectorResult:
    """Run gather-based detection on a single sector.

    Pipeline:
        1. Estimate PSD -> autocorrelation -> Toeplitz covariance.
        2. Extract observed sub-matrix, invert (O(n^3)).
        3. Embed inverse covariance in full cadence grid.
        4. Convolve flux and covariance with transit template.
        5. Downsample to gather grid.
        6. Run gather kernels (CPU or GPU).

    Parameters
    ----------
    lc : LightCurve
        Light curve for the sector.
    duration : int
        Transit duration in cadences.
    delta : int
        Period / epoch step size in cadences.
    K : int
        PSD smoothing half-width.
    use_gpu : bool
        Use GPU gather kernels if available.
    regularization : float
        Tikhonov regularization for matrix inversion.

    Returns
    -------
    SectorResult
    """
    N_full = lc.n_full
    cadence_zero = lc.cadence_zero
    obs_mask = lc.observed_mask

    valid = (cadence_zero >= 0) & (cadence_zero < N_full)
    cadence_zero = cadence_zero[valid]
    flux_valid = lc.flux[valid]
    n_observed = len(cadence_zero)

    # ── Covariance estimation ───────────────────────────────────────────
    cleaned_flux = threshold_outliers(flux_valid.copy(), level=3)

    filled_flux = np.zeros(N_full, dtype=np.float64)
    filled_flux[cadence_zero] = cleaned_flux
    std_ = np.std(cleaned_flux)
    gap_mask = ~obs_mask
    if gap_mask.sum() > 0:
        filled_flux[gap_mask] = np.random.normal(0, std_, gap_mask.sum())

    # Zero-pad for spectral estimation
    zp_flux = np.zeros(2 * N_full - 1)
    zp_flux[:N_full] = filled_flux
    p_smooth = smooth_periodogram(zp_flux, K=K)
    ac = np.real(np.fft.ifft(p_smooth))[:N_full].astype(np.float64)

    # Build Toeplitz, extract observed portion, invert
    cov_full = toeplitz(ac)
    cov_obs = cov_full[np.ix_(cadence_zero, cadence_zero)]
    cov_obs_reg = cov_obs + np.eye(n_observed) * regularization
    try:
        cov_inv_obs = np.linalg.inv(cov_obs_reg)
    except np.linalg.LinAlgError:
        cov_inv_obs = np.linalg.pinv(cov_obs_reg)

    # Embed into full-size matrix
    cov_inv_full = np.zeros((N_full, N_full), dtype=np.float64)
    cov_inv_full[np.ix_(cadence_zero, cadence_zero)] = cov_inv_obs

    flux_full = np.zeros(N_full, dtype=np.float64)
    flux_full[cadence_zero] = cleaned_flux

    # ── Convolve with transit template ──────────────────────────────────
    transit_profile = -np.ones(duration)
    y_cov_inv = cov_inv_full.dot(flux_full)

    d = duration
    y_d = convolve1d(y_cov_inv, transit_profile, mode="constant")
    y_d = y_d[d // 2 : N_full - d // 2][::delta]

    K_conv = convolve1d(
        convolve1d(cov_inv_full, transit_profile, axis=0, mode="constant"),
        transit_profile,
        axis=1,
        mode="constant",
    )
    K_d = K_conv[d // 2 : N_full - d // 2, d // 2 : N_full - d // 2][
        ::delta, ::delta
    ]

    # Ensure even length
    if len(y_d) % 2 == 1:
        y_d = y_d[:-1]
        K_d = K_d[:-1, :-1]

    half_N = len(y_d) // 2

    # ── Gather ──────────────────────────────────────────────────────────
    if use_gpu:
        try:
            numerator, denominator = gather_gpu(y_d, K_d)
        except Exception:
            numerator = gather_num_cpu(y_d, half_N)
            denominator = gather_den_cpu(K_d, half_N)
    else:
        numerator = gather_num_cpu(y_d, half_N)
        denominator = gather_den_cpu(K_d, half_N)

    return SectorResult(
        sector=lc.sector,
        start_cadence=lc.start_cadence,
        end_cadence=lc.end_cadence,
        delta=delta,
        numerator=numerator,
        denominator=denominator,
    )
