"""Load TESS light curve data from pickle files."""


import pickle
from typing import Dict, List

import numpy as np
import sys
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')
from info import cadence_bounds
from tess_transit.data.lightcurve import LightCurve


def load_pickle(path: str) -> Dict:
    """Unpack an 11-tuple pickle file produced by the TESS download pipeline.

    Parameters
    ----------
    path : str
        Path to the ``.p`` file.

    Returns
    -------
    dict
        Keys: ``lc_data``, ``processed_lc_data``, ``detrend_data``,
        ``norm_offset``, ``quality_data``, ``time_data``, ``cam_data``,
        ``ccd_data``, ``coeff_ls``, ``centroid_xy_data``, ``pos_xy_corr``.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    (
        lc_data,
        processed_lc_data,
        detrend_data,
        norm_offset,
        quality_data,
        time_data,
        cam_data,
        ccd_data,
        coeff_ls,
        centroid_xy_data,
        pos_xy_corr,
    ) = payload

    return {
        "lc_data": lc_data,
        "processed_lc_data": processed_lc_data,
        "detrend_data": detrend_data,
        "norm_offset": norm_offset,
        "quality_data": quality_data,
        "time_data": time_data,
        "cam_data": cam_data,
        "ccd_data": ccd_data,
        "coeff_ls": coeff_ls,
        "centroid_xy_data": centroid_xy_data,
        "pos_xy_corr": pos_xy_corr,
    }


def get_lightcurve(data: Dict, sector: int, use_processed: bool = False) -> LightCurve:
    """Extract a :class:`LightCurve` for one sector from loaded pickle data.

    Parameters
    ----------
    data : dict
        Output of :func:`load_pickle`.
    sector : int
        Sector number.
    use_processed : bool
        If True use ``processed_lc_data`` (systematics removed, stellar noise
        only).  If False (default) use ``detrend_data``.

    Returns
    -------
    LightCurve
    """
    flux_source = data["processed_lc_data"] if use_processed else data["detrend_data"]

    if sector not in flux_source:
        raise KeyError(f"Sector {sector} not found in data")
    if sector not in cadence_bounds:
        raise KeyError(f"Sector {sector} not in cadence_bounds (edit info.py)")

    start_cad, end_cad = cadence_bounds[sector]

    return LightCurve(
        sector=sector,
        flux=np.asarray(flux_source[sector], dtype=np.float64),
        cadence_numbers=np.asarray(data["time_data"][sector], dtype=np.int64),
        camera=data["cam_data"][sector],
        ccd=data["ccd_data"][sector],
        start_cadence=start_cad,
        end_cadence=end_cad,
    )


def available_sectors(data: Dict) -> List[int]:
    """Return sorted list of sectors present in loaded pickle data."""
    return sorted(data["detrend_data"].keys())
