"""LightCurve dataclass holding per-sector TESS data."""


from dataclasses import dataclass

import numpy as np


@dataclass
class LightCurve:
    """Container for a single sector's light curve data.

    Parameters
    ----------
    sector : int
        TESS sector number.
    flux : np.ndarray
        Detrended flux values, shape ``(n_obs,)``.
    cadence_numbers : np.ndarray
        Absolute cadence integers, shape ``(n_obs,)``.
    camera : int
        TESS camera (1-4).
    ccd : int
        TESS CCD (1-4).
    start_cadence : int
        First cadence in the sector (from ``cadence_bounds``).
    end_cadence : int
        Last cadence in the sector (from ``cadence_bounds``).
    """

    sector: int
    flux: np.ndarray
    cadence_numbers: np.ndarray
    camera: int
    ccd: int
    start_cadence: int
    end_cadence: int

    @property
    def n_full(self) -> int:
        """Number of cadences spanning the full sector range (including gaps)."""
        return self.end_cadence - self.start_cadence + 1

    @property
    def cadence_zero(self) -> np.ndarray:
        """Zero-based cadence indices relative to ``start_cadence``."""
        return (self.cadence_numbers - self.start_cadence).astype(int)

    @property
    def observed_mask(self) -> np.ndarray:
        """Boolean mask of shape ``(n_full,)`` — True at observed cadences."""
        mask = np.zeros(self.n_full, dtype=bool)
        idx = self.cadence_zero
        valid = (idx >= 0) & (idx < self.n_full)
        mask[idx[valid]] = True
        return mask

    @property
    def duty_cycle(self) -> float:
        """Fraction of cadences that are observed."""
        return len(self.cadence_numbers) / self.n_full
