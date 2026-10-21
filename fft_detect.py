"""FFT-based matched filter transit detection (DEFAULT method).

Implements the Fourier decomposition of periodic box transits with:
- PSD whitening (Kay's smoothed periodogram) for colored noise
- Mask-density denominator for correct gap handling
- Duration-adaptive transit masking for PSD estimation

The transit masking step identifies potential transit-like features at the
current duration and masks them before computing the PSD. This prevents real
transits from inflating the noise estimate, improving detection sensitivity.

Algorithm per sector, per duration D, step delta:

1. Estimate PSD S(k) from smoothed periodogram of gap-filled data.
2. Precompute:
   - Y_masked = FFT(flux_zerofilled)             data spectrum
   - A = Y_masked / S                             whitened data spectrum
   - F_box = FFT(box_of_width_D)                  single-transit template
   - base_corr = sqrt(N) * Re(IFFT(conj(F_box) * A))  single-transit matched filter
   - d0 = duty_cycle * sum_k |F_box(k)|^2 / S(k)  whitening weight (gap-corrected)
   - W_conv = IFFT(FFT(mask) * conj(FFT(ones_D))) observed count per window
   - F_Wconv = FFT(W_conv)

3. For each period P = (p_idx + 1) * delta:
   - M = (N - D) // P + 1                         transit count at epoch 0
   - G_M(k) = (1 - z^M) / (1 - z),  z = exp(-j*2*pi*k*P/N)

   Numerator (all epochs from one IFFT):
     num_all = sqrt(N) * Re(IFFT(conj(F_box * G_M) * A))

   Denominator (mask-density x whitening weight):
     m_fold_all = Re(IFFT(F_Wconv * conj(G_M)))
     den_all = (d0 / D) * m_fold_all

   Edge correction (epochs where M -> M-1 transits):
     num[edge] -= base_corr[wrapped_pos]
     den[edge] -= (d0 / D) * W_conv[wrapped_pos]

   SNR(t0) = num(t0) / sqrt(den(t0))

Complexity: O(half_N * N log N) -- one IFFT per trial period.

Denominator approximation: den(P, t0) = (d0/D) * m_fold(P, t0)
- d0 = duty_cycle * sum_k |F_box|^2 / S(k), gap-corrected whitened energy
- For white noise S=1: d0 = N_obs * D, matching gather's N_obs × N_obs covariance
- m_fold counts observed in-transit cadences across all transit repetitions
- SNR = sqrt(N) * corr / sqrt(N_obs * m_fold) matches time-domain matched filter
- Assumes cross-transit correlations in C^{-1} are negligible
- Exact for white noise; for colored noise accurate to ~1-5%

No NUFFT: TESS data sits on a regular 2-minute cadence grid with missing
points (gaps).  Standard FFT on the zero-padded array is correct.
"""

from dataclasses import dataclass

import numpy as np

from tess_transit.core.psd import (
    threshold_outliers,
    smooth_periodogram,
    detect_transit_candidates,
)
from tess_transit.data.lightcurve import LightCurve


@dataclass
class SectorResult:
    """Per-sector detection output from the FFT pipeline."""

    sector: int
    start_cadence: int
    end_cadence: int
    delta: int
    numerator: np.ndarray   # shape (half_N, half_N)
    denominator: np.ndarray  # shape (half_N, half_N)


def fft_sector_search(
    lc: LightCurve,
    duration: int,
    delta: int = 5,
    K: int = 3,
) -> SectorResult:
    """Run FFT matched filter search on a single sector.

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

    Returns
    -------
    SectorResult
        Numerator and denominator arrays on the (period, epoch) grid.
    """
    N_full = lc.n_full
    cadence_zero = lc.cadence_zero
    obs_mask = lc.observed_mask

    valid = (cadence_zero >= 0) & (cadence_zero < N_full)

    # ── Clean flux and build full-grid arrays ───────────────────────────
    cleaned_flux = lc.flux.copy()
    flux_full = np.zeros(N_full, dtype=np.float64)
    flux_full[cadence_zero[valid]] = cleaned_flux[valid]

    W = obs_mask.astype(np.float64)

    # ── PSD estimation at N_full frequency bins ─────────────────────────
    # Detect potential transit candidates at this duration and mask them
    # before PSD estimation.  This prevents real transits from contaminating
    # the noise spectrum, improving detection sensitivity.
    transit_mask = detect_transit_candidates(
        flux_full, obs_mask, duration, threshold_sigma=3.0
    )

    # Combined mask: gaps + transit candidates (both should be filled with noise)
    # for PSD estimation purposes
    fill_mask = ~obs_mask | transit_mask

    # Fill gaps and transit candidates with white noise for spectral estimation
    filled = flux_full.copy()
    std_ = np.std(cleaned_flux)
    if fill_mask.sum() > 0:
        filled[fill_mask] = np.random.normal(0, std_, fill_mask.sum())

    # Smoothed periodogram directly on N_full-length array (no zero-pad)
    # so PSD bins align with the FFT frequency grid of flux_full
    S = smooth_periodogram(filled, K=K)
    # Floor at a fraction of median PSD to prevent whitening explosion.
    # Using 1e-10 is far too small for real data and causes numerical
    # issues where 1/S amplifies noise by many orders of magnitude.
    psd_floor = np.median(S) * 1e-6
    S = np.maximum(S, psd_floor)

    # ── Precompute (once for all periods) ───────────────────────────────
    Y_masked = np.fft.fft(flux_full)            # data spectrum
    A = Y_masked / S                             # whitened data spectrum

    # Single box transit template (negative dip)
    box = np.zeros(N_full)
    box[:duration] = -1.0
    F_box = np.fft.fft(box)

    # Single-transit matched filter (for edge corrections)
    # Scale by sqrt(N) to get correct matched filter normalization
    # (IFFT includes 1/N factor; sqrt(N) gives SNR = corr / sqrt(m_fold))
    base_corr = np.real(np.fft.ifft(np.conj(F_box) * A)) * np.sqrt(N_full)

    # Whitening weight for one full transit: d0 = sum_k |F_box|^2 / S
    # By Parseval: sum_k |F_box|^2 = N_full * D, so d0 = N_full * D / S for white noise.
    # Scale by duty cycle to account for gaps: only N_obs cadences contribute signal,
    # but Parseval uses N_full. This matches the gather method which uses N_obs × N_obs
    # covariance matrix for the observed data only.
    duty_cycle = np.sum(obs_mask) / N_full
    F_box_sq_over_S = np.abs(F_box) ** 2 / S
    d0 = np.real(np.sum(F_box_sq_over_S)) * duty_cycle

    # Window correlation: W_conv[n] = number of observed cadences in
    # transit window [n, n+duration)
    box_ones = np.zeros(N_full)
    box_ones[:duration] = 1.0
    W_conv = np.real(
        np.fft.ifft(np.fft.fft(W) * np.conj(np.fft.fft(box_ones)))
    )
    F_Wconv = np.fft.fft(W_conv)

    # ── Downsample to gather grid ───────────────────────────────────────
    n_grid = N_full // delta
    if n_grid % 2 == 1:
        n_grid -= 1
    half_N = n_grid // 2

    num_out = np.zeros((half_N, half_N), dtype=np.float64)
    den_out = np.zeros((half_N, half_N), dtype=np.float64)

    freq_idx = np.arange(N_full)

    # ── Per-period computation ──────────────────────────────────────────
    for p_idx in range(half_N):
        period = (p_idx + 1) * delta
        if period < duration:
            continue

        # Number of transits for epoch=0
        M = (N_full - duration) // period + 1
        if M < 1:
            continue

        # Geometric sum: G_M[k] = sum_{m=0}^{M-1} exp(-j*2*pi*k*m*P/N)
        phase = 2 * np.pi * freq_idx * period / N_full
        z = np.exp(-1j * phase)
        one_minus_z = 1 - z
        singular = np.abs(one_minus_z) < 1e-10

        G_M = np.zeros(N_full, dtype=np.complex128)
        G_M[~singular] = (1 - z[~singular] ** M) / one_minus_z[~singular]
        G_M[singular] = M

        # ── Numerator: all epochs via IFFT ──────────────────────────────
        # Scale by sqrt(N) for correct matched filter normalization
        F_tmpl = F_box * G_M
        num_all = np.real(np.fft.ifft(np.conj(F_tmpl) * A)) * np.sqrt(N_full)

        # ── Denominator: mask-density x whitening weight ────────────────
        # m_fold(t0) = sum_m W_conv[(t0 + m*P) mod N]
        #            = IFFT(F_Wconv * conj(G_M))[t0]
        m_fold_all = np.real(np.fft.ifft(F_Wconv * np.conj(G_M)))
        # m_fold counts observed cadences — must be non-negative.
        # IFFT rounding can produce tiny negative values.
        m_fold_all = np.maximum(m_fold_all, 0.0)
        den_all = (d0 / duration) * m_fold_all

        # Edge threshold: for epoch > thresh, only M-1 transits fit
        epoch_thresh = (N_full - duration) - (M - 1) * period

        # Fill output at epoch grid points
        n_epochs = p_idx + 1
        epochs = np.arange(n_epochs) * delta

        num_out[p_idx, :n_epochs] = num_all[epochs % N_full]
        den_out[p_idx, :n_epochs] = den_all[epochs % N_full]

        # Edge corrections (M -> M-1 transits)
        edge_idx = np.where(epochs > epoch_thresh)[0]
        if len(edge_idx) > 0:
            corr_pos = (epochs[edge_idx] + (M - 1) * period) % N_full

            # Numerator: subtract wrapped transit contribution
            num_out[p_idx, edge_idx] -= base_corr[corr_pos]

            # Denominator: subtract window count at wrapped position
            den_out[p_idx, edge_idx] -= (d0 / duration) * W_conv[corr_pos]

        # After edge corrections, enforce a minimum denominator threshold.
        # The denominator represents noise variance of the matched filter.
        # Require at least duration/2 observed in-transit cadences total
        # (across all transit repetitions) for a reliable detection.
        # Below this, spectral leakage from whitening produces spurious SNR.
        # Threshold: den >= (d0/D) * (D/2) = d0/2, i.e. m_fold >= D/2.
        den_min = d0 * 0.5
        den_out[p_idx, :n_epochs] = np.where(
            den_out[p_idx, :n_epochs] >= den_min,
            den_out[p_idx, :n_epochs],
            0.0,
        )

    return SectorResult(
        sector=lc.sector,
        start_cadence=lc.start_cadence,
        end_cadence=lc.end_cadence,
        delta=delta,
        numerator=num_out,
        denominator=den_out,
    )
