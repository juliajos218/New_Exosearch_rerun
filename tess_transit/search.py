"""Multi-sector search orchestration.

Coordinates loading, optional injection, per-sector detection, multi-sector
combination, and candidate extraction.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from tess_transit.data.loader import load_pickle, get_lightcurve, available_sectors
from tess_transit.data.lightcurve import LightCurve
from tess_transit.core.fft_detect import fft_sector_search
from tess_transit.core.gather import gather_sector_search
from tess_transit.core.combine import combine_sectors, CombinedResult
from tess_transit.inject.transit import inject_transit

# Import default durations from info.py (1-16 hours in cadences)
try:
    from info import durs as _default_durs
except ImportError:
    # Fallback if info.py not on path
    _default_durs = np.array([30, 60, 90, 120, 180, 240, 300, 360, 420, 480])


@dataclass
class SearchConfig:
    """Configuration for a transit search."""

    delta: int = 5
    durations: Optional[np.ndarray] = None
    K: int = 3
    method: str = "fft"          # 'fft' or 'gather'
    use_gpu: bool = False
    min_transits: int = 2        # require at least 2 transits with data
    min_period_days: float = 0.5  # default minimum 0.5 days
    max_period_days: Optional[float] = None  # None = data_length / 2

    def __post_init__(self):
        if self.durations is None:
            self.durations = _default_durs.copy()


def _search_sector(
    lc: LightCurve,
    duration: int,
    config: SearchConfig,
):
    """Dispatch to FFT or gather for a single sector."""
    if config.method == "fft":
        return fft_sector_search(lc, duration, config.delta, config.K)
    elif config.method == "gather":
        return gather_sector_search(
            lc, duration, config.delta, config.K, config.use_gpu
        )
    else:
        raise ValueError(f"Unknown method: {config.method!r} (use 'fft' or 'gather')")


def run_search(
    pickle_path: str,
    sectors: Optional[List[int]] = None,
    config: Optional[SearchConfig] = None,
    inject_params: Optional[Dict] = None,
) -> Dict:
    """Run a multi-sector transit search.

    Parameters
    ----------
    pickle_path : str
        Path to ``.p`` light curve file.
    sectors : list of int, optional
        Sectors to include.  ``None`` means all available.
    config : SearchConfig, optional
        Search configuration.  Defaults to FFT with standard durations.
    inject_params : dict, optional
        If provided, inject a synthetic transit before searching.
        Keys: ``period``, ``duration``, ``t0``, ``depth``.

    Returns
    -------
    dict
        ``results`` (per-duration), ``best_candidates`` (sorted by SNR),
        ``sectors``, ``config``, ``inject_params``.
    """
    if config is None:
        config = SearchConfig()

    print(f"Loading {pickle_path} ...")
    data = load_pickle(pickle_path)

    avail = available_sectors(data)
    print(f"Available sectors: {avail}")

    if sectors is None:
        sectors = avail
    sectors = [s for s in sectors if s in avail]
    print(f"Using sectors: {sectors}")

    # Build LightCurve objects
    lc_list: List[LightCurve] = [get_lightcurve(data, s) for s in sectors]

    # Compute data span (first to last cadence across all sectors)
    first_cadence = min(lc.start_cadence for lc in lc_list)
    last_cadence = max(lc.end_cadence for lc in lc_list)
    data_span_cadences = last_cadence - first_cadence
    data_length_days = data_span_cadences * 2 / (60 * 24)

    # Set max_period to data_length / 2 if not specified
    if config.max_period_days is None:
        config.max_period_days = data_length_days / 2
        print(f"Data span: {data_length_days:.1f} days, max period set to {config.max_period_days:.1f} days")
    print(f"Period range: {config.min_period_days:.2f} - {config.max_period_days:.1f} days")

    # Optional injection
    if inject_params:
        print(
            f"Injecting transit: period={inject_params['period']}, "
            f"duration={inject_params['duration']}, "
            f"t0={inject_params['t0']}, depth={inject_params['depth']}"
        )
        lc_list = [
            inject_transit(
                lc,
                inject_params["period"],
                inject_params["duration"],
                inject_params["t0"],
                inject_params["depth"],
            )
            for lc in lc_list
        ]

    # Process each duration
    all_results: Dict[int, Dict] = {}
    for duration in config.durations:
        dur_hrs = duration * 2 / 60
        print(f"\n--- Duration: {duration} cadences ({dur_hrs:.1f} hours) ---")

        sector_results = []
        for lc in lc_list:
            print(f"  Sector {lc.sector} ({config.method}) ...")
            result = _search_sector(lc, duration, config)
            sector_results.append(result)

        # Combine across sectors
        combined = combine_sectors(sector_results, min_transits=config.min_transits)

        all_results[duration] = {
            "combined": combined,
            "sector_results": sector_results,
        }

    # Find best candidates across all durations
    best_candidates = []

    # Convert period limits to cadences
    min_p_cad = int(config.min_period_days * 60 * 24 / 2)
    max_p_cad = int(config.max_period_days * 60 * 24 / 2)

    # Also require period short enough for min_transits to fit
    # max_period_for_transits = data_span / min_transits
    max_p_for_transits = int(data_span_cadences / config.min_transits)
    max_p_cad = min(max_p_cad, max_p_for_transits)

    print(f"Requiring >= {config.min_transits} transits (max period {max_p_cad * 2 / (60*24):.1f} days)")

    for duration, res in all_results.items():
        det = res["combined"].detection
        den = res["combined"].denominator

        # Build mask: valid denominator AND within period range
        valid_mask = den > 0
        period_cad_arr = (np.arange(det.shape[0]) + 1) * config.delta
        period_mask = (period_cad_arr >= min_p_cad) & (period_cad_arr <= max_p_cad)
        combined_mask = valid_mask & period_mask[:, np.newaxis]

        det_masked = np.where(combined_mask, det, -np.inf)
        if not np.any(np.isfinite(det_masked)):
            continue  # no valid candidates for this duration

        max_idx = np.unravel_index(np.argmax(det_masked), det.shape)
        max_snr = det[max_idx]
        period_cadences = (max_idx[0] + 1) * config.delta
        epoch_cadences = max_idx[1] * config.delta
        best_candidates.append(
            {
                "duration": duration,
                "period_cadences": period_cadences,
                "epoch_cadences": epoch_cadences,
                "snr": max_snr,
                "period_days": period_cadences * 2 / (60 * 24),
                "duration_hours": duration * 2 / 60,
            }
        )

    best_candidates.sort(key=lambda x: x["snr"], reverse=True)

    return {
        "results": all_results,
        "best_candidates": best_candidates,
        "sectors": sectors,
        "config": config,
        "inject_params": inject_params,
    }
