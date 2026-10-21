"""Transit signal injection for testing and validation."""


import numpy as np

from tess_transit.data.lightcurve import LightCurve


def box_transit(
    times: np.ndarray,
    period: float,
    duration: float,
    t0: float,
    depth: float = 1.0,
) -> np.ndarray:
    """Generate a box-shaped transit signal evaluated at *times*.

    Parameters
    ----------
    times : np.ndarray
        Array of time points (cadence numbers or BJD).
    period : float
        Orbital period (same units as *times*).
    duration : float
        Transit duration.
    t0 : float
        Epoch of first transit centre.
    depth : float
        Transit depth (positive value; returned signal is negative during transit).

    Returns
    -------
    np.ndarray
        Transit time series: ``-depth`` during transit, ``0`` elsewhere.
    """
    phase = ((times - t0 + duration / 2) % period)
    return np.where(phase <= duration, -depth, 0.0)


def inject_transit(
    lc: LightCurve,
    period: float,
    duration: float,
    t0: float,
    depth: float,
) -> LightCurve:
    """Return a new :class:`LightCurve` with an injected box transit.

    Parameters
    ----------
    lc : LightCurve
        Original light curve.
    period, duration, t0 : float
        Transit parameters in cadence units.
    depth : float
        Transit depth.

    Returns
    -------
    LightCurve
        Copy with ``flux = lc.flux + transit_signal``.
    """
    signal = box_transit(
        lc.cadence_numbers.astype(float),
        float(period),
        float(duration),
        float(t0),
        depth,
    )
    return LightCurve(
        sector=lc.sector,
        flux=lc.flux + signal,
        cadence_numbers=lc.cadence_numbers,
        camera=lc.camera,
        ccd=lc.ccd,
        start_cadence=lc.start_cadence,
        end_cadence=lc.end_cadence,
    )
