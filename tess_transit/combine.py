"""Multi-sector phase-aligned combination of detection statistics.

Works identically for both FFT and gather outputs — they share the same
``SectorResult`` format and (period, epoch) grid indexing.
"""


from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from tess_transit.core.fft_detect import SectorResult


@dataclass
class CombinedResult:
    """Combined detection statistics across sectors."""

    numerator: np.ndarray    # shape (half_N, half_N)
    denominator: np.ndarray  # shape (half_N, half_N)
    detection: np.ndarray    # SNR = num / sqrt(den)
    delta: int


def compute_epoch_offset(
    period: int,
    delta: int,
    sector_start: int,
    ref_start: int,
) -> int:
    """Compute the global epoch offset for a sector relative to a reference.

    Parameters
    ----------
    period : int
        Period in gather grid units (p_idx + 1).
    delta : int
        Cadence step.
    sector_start : int
        Absolute cadence of this sector's start.
    ref_start : int
        Absolute cadence of the reference sector's start.

    Returns
    -------
    int
        Epoch offset in gather grid units.
    """
    sector_start_grid = sector_start // delta
    return sector_start_grid % period


def local_to_global_epoch(
    local_epoch: int,
    period: int,
    sector_start: int,
    delta: int,
) -> int:
    """Convert a local epoch index to a global (absolute cadence) epoch.

    Parameters
    ----------
    local_epoch : int
        Epoch index in sector's local grid (0 to period-1).
    period : int
        Period in gather grid units.
    sector_start : int
        Absolute cadence number of sector start.
    delta : int
        Cadence step.

    Returns
    -------
    int
        Epoch in global grid units, range [0, period-1].
    """
    absolute_cadence = sector_start + local_epoch * delta
    return (absolute_cadence // delta) % period


def count_transits_in_sector(
    period: int,
    delta: int,
    sector_start: int,
    sector_end: int,
) -> int:
    """Count how many complete transits fit within a sector's span."""
    span_cadences = sector_end - sector_start
    period_cadences = period * delta
    return span_cadences // period_cadences if period_cadences > 0 else 0


def combine_sectors(
    results_list: List[SectorResult],
    min_transits: int = 1,
) -> CombinedResult:
    """Combine per-sector statistics with proper phase alignment.

    For each (period, local_epoch) in each sector, maps to the corresponding
    global epoch and accumulates.  This ensures transits at the same absolute
    time contribute to the same detection cell.

    Parameters
    ----------
    results_list : list of SectorResult
        Per-sector detection statistics (from FFT or gather).
    min_transits : int
        Not used here (kept for API compatibility). The min_transits
        requirement is enforced in search.py based on total data span.

    Returns
    -------
    CombinedResult
    """
    if not results_list:
        raise ValueError("Empty results list")

    min_half_N = min(r.numerator.shape[0] for r in results_list)
    delta = results_list[0].delta

    num_global = np.zeros((min_half_N, min_half_N), dtype=np.float64)
    den_global = np.zeros((min_half_N, min_half_N), dtype=np.float64)

    for result in results_list:
        for p_idx in range(min_half_N):
            period = p_idx + 1  # period in gather grid units

            # Map each local epoch to global epoch
            for local_e in range(period):
                global_e = local_to_global_epoch(
                    local_e, period, result.start_cadence, delta
                )

                if global_e < min_half_N and local_e < result.numerator.shape[1]:
                    num_global[p_idx, global_e] += result.numerator[p_idx, local_e]
                    den_global[p_idx, global_e] += result.denominator[p_idx, local_e]

    # Compute detection statistic
    detection = np.divide(
        num_global,
        np.sqrt(den_global),
        out=np.zeros_like(num_global),
        where=den_global > 0,
    )

    return CombinedResult(
        numerator=num_global,
        denominator=den_global,
        detection=detection,
        delta=delta,
    )
