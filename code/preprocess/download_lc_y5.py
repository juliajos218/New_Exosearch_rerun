# download_lc_y5.py
# Download and preprocess TESS light curves for year 5 (sectors 61-69)
# Based on original code by Dr. Jamila Taaki, adapted for year 5
# Modified by Julia Joseph, 2025-2026

import lightkurve as lk
import pickle
import numpy as np
import os
import sys
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')
from info_y5 import sectors, cadence_bounds

PRIORS_DIR = '/home/juliajos/TESS/data/priors/'
OUTPUT_DIR = os.path.expanduser('~/TESS/data/lightcurves_y5/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

tid_list = np.loadtxt('/home/juliajos/TESS/data/light_curves/info/persistant_tids_y5.txt', dtype=int)
print(f"Loaded {len(tid_list)} targets")

for tid in tid_list:
    tic_id = str(tid)
    output_file = os.path.join(OUTPUT_DIR, f'{tic_id}.p')

    if os.path.exists(output_file):
        print(f"Already exists: TIC {tic_id}, skipping")
        continue

    print(f"Processing TIC {tic_id}...")

    lc_data          = {}
    processed_lc_data = {}
    time_data        = {}
    cam_data         = {}
    ccd_data         = {}
    centroid_xy_data = {}
    coeff_ls         = {}
    detrend_data     = {}
    norm_offset      = {}
    quality_data     = {}
    pos_xy_corr      = {}

    for sector in range(sectors[0], sectors[1]+1):
        try:
            search_result = lk.search_lightcurve('TIC '+tic_id, mission='TESS', author='SPOC', cadence='short', sector=sector)
            lightcurve = search_result.download_all()[0]
        except:
            print(f"  Missing sector {sector} for TIC {tic_id}")
            continue

        lc_sap = lightcurve.sap_flux
        lc_pdc = lightcurve.pdcsap_flux
        quality = lightcurve.quality

        lc_data[sector]           = lc_sap.to_value()[~quality.astype(bool)]
        processed_lc_data[sector] = lc_pdc.to_value()[~quality.astype(bool)]
        time_data[sector]         = lightcurve.cadenceno[~quality.astype(bool)]
        cam_data[sector]          = lightcurve.camera
        ccd_data[sector]          = lightcurve.ccd

        qual = np.zeros_like(lightcurve.cadenceno.data)
        qual[quality.astype(bool)] = 1
        quality_data[sector] = qual

        centroid_xy_data[sector] = [
            lightcurve.mom_centr1.to_value()[~quality.astype(bool)],
            lightcurve.mom_centr2.to_value()[~quality.astype(bool)]
        ]
        pos_xy_corr[sector] = [
            lightcurve.pos_corr1.to_value()[~quality.astype(bool)],
            lightcurve.pos_corr2.to_value()[~quality.astype(bool)]
        ]

        prior_file = os.path.join(PRIORS_DIR, str(sector),
                                  f'evec_matrix_{sector}_{cam_data[sector]}_{ccd_data[sector]}.p')

        if not os.path.exists(prior_file):
            print(f"  Missing prior file: {prior_file}, skipping sector {sector}")
            continue

        evecs        = pickle.load(open(prior_file, 'rb'))
        cadence_data = time_data[sector] - cadence_bounds[sector][0] - 1

        # --- ADDED by Julia Joseph ---
        # Some cadences fall slightly outside cadence_bounds after quality flag filtering
        # causing IndexError. valid_mask filters these out before detrending.
        # This does not affect the original download_lc.py behavior for year 6
        # since year 6 data did not have out-of-bounds cadences.
        valid_mask              = (cadence_data >= 0) & (cadence_data < evecs.shape[1])
        cadence_data            = cadence_data[valid_mask]
        lc_data[sector]         = lc_data[sector][valid_mask]
        processed_lc_data[sector] = processed_lc_data[sector][valid_mask]
        time_data[sector]       = time_data[sector][valid_mask]
        quality_data[sector]    = quality_data[sector][valid_mask]
        centroid_xy_data[sector] = [
            centroid_xy_data[sector][0][valid_mask],
            centroid_xy_data[sector][1][valid_mask]
        ]
        pos_xy_corr[sector]     = [
            pos_xy_corr[sector][0][valid_mask],
            pos_xy_corr[sector][1][valid_mask]
        ]
        # --- END ADDED ---

        lc_sap_val   = lc_sap.to_value()[~quality.astype(bool)][valid_mask]
        lc_offset    = np.nanmedian(lc_sap_val)
        lc_sap_val  -= lc_offset
        lc_norm      = np.linalg.norm(lc_sap_val)
        lc_sap_val  /= lc_norm
        evecs_mask   = evecs[:, cadence_data]
        coeff        = np.dot(evecs_mask, lc_sap_val.T)

        norm_offset[sector]  = [lc_offset, lc_norm]
        coeff_ls[sector]     = coeff * lc_norm
        detrend_data[sector] = lc_norm * (lc_sap_val - np.dot(coeff, evecs_mask))

        print(f"  Sector {sector} done")

    if len(lc_data) > 0:
        pickle.dump(
            (lc_data, processed_lc_data, detrend_data, norm_offset, quality_data,
             time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr),
            open(output_file, 'wb')
        )
        print(f"  Saved TIC {tic_id} with {len(lc_data)} sectors")
    else:
        print(f"  No sectors processed for TIC {tic_id}")

print("All done!")
