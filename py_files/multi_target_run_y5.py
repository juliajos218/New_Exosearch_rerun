# multi_target_run_y5.py
# FFT Transit Search — Year 5 (Sectors 61-69)
# Adapted from multi_target_run.py by Julia Joseph, 2026
#
# This script runs the FFT-based matched filter transit search on all 1299
# preprocessed light curves from year 5 (sectors 61-69). For each target it:
#   1. Loads the preprocessed light curve pickle file
#   2. Runs the FFT search across all available sectors
#   3. Applies SNR >= 6 filter immediately before saving to keep file sizes small (~80MB vs 2.5GB)
#   4. Saves filtered results to output_y5/{target_id}_results.p
#   5. Skips targets that already have results (safe to restart if interrupted)
# Searching transit durations of 30, 60, 90, 120 cadences (1-4 hours)
# Searching periods from 0.5 days up to half the data span
# Requires at least 3 transits for a detection
# Estimated runtime: 10-15 hours for 1299 targets
# Run with: nohup python3 multi_target_run_y5.py > multi_target_run_y5.log 2>&1 &

import sys
sys.path.insert(0, '/home/juliajos')

import numpy as np
import glob
import os
import pickle
from tess_transit.search import run_search, SearchConfig
from tess_transit.data.loader import load_pickle, get_lightcurve, available_sectors

LC_DIR      = "/home/juliajos/TESS/data/lightcurves_y5/"
RESULTS_DIR = "/home/juliajos/tess_transit/data/output_y5"
SNR_MIN     = 6.0
os.makedirs(RESULTS_DIR, exist_ok=True)

lc_paths = sorted(glob.glob(os.path.join(LC_DIR, "*.p")))
print(f"Found {len(lc_paths)} light curves")

for path in lc_paths:
    target_id    = os.path.splitext(os.path.basename(path))[0]
    results_path = os.path.join(RESULTS_DIR, f"{target_id}_results.p")

    if os.path.exists(results_path):
        print(f"[{target_id}] Already processed, skipping.")
        continue

    try:
        data    = load_pickle(path)
        results = run_search(
            path,
            sectors=available_sectors(data),
            config=SearchConfig(
                durations=np.array([30, 60, 90, 120]),
                min_period_days=0.5,
                max_period_days=None,
                min_transits=3,
            )
        )

        # apply SNR filter immediately before saving to keep file sizes small
        # avoids accumulating 2.5GB files — filters to ~80MB per target
        for duration, res in results["results"].items():
            c    = res["combined"]
            keep = np.abs(c.detection) >= SNR_MIN
            c.detection   = np.where(keep, c.detection,   0).astype(np.float32)
            c.numerator   = np.where(keep, c.numerator,   0).astype(np.float32)
            c.denominator = np.where(keep, c.denominator, 0).astype(np.float32)
            res["sector_results"] = []

        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[{target_id}] Done.")
        del data, results

    except Exception as e:
        print(f"[{target_id}] Failed: {e}, skipping.")

print("All done!")
