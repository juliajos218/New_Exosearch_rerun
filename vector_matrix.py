# vector_matrix.py
# Format cotrending basis vectors into a full-size matrix for each sector/cam/ccd
# Based on original code by Dr. Jamila Taaki, adapted for year 5 (sectors 61-69)
# Modified by Julia Joseph, 2025-2026

from astropy.io import fits
import numpy as np
import glob
import pickle
import os
import sys
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')
from info_y5 import sectors, cadence_bounds

CBV_DIR    = '/home/juliajos/TESS/data/light_curves/info/cbvs/'
OUTPUT_DIR = '/home/juliajos/TESS/data/priors/'

for sector in range(sectors[0], sectors[1]+1):
    for cam in range(1, 5):
        for ccd in range(1, 5):
            N_vecs    = 30
            N_cadence = cadence_bounds[sector][1] - cadence_bounds[sector][0]
            evecs     = np.zeros((N_vecs, N_cadence), dtype='float32')
            print(f"Processing sector {sector}, cam {cam}, ccd {ccd}")

            pattern        = f'{CBV_DIR}*{sector:04d}*{cam}-{ccd}*.fits'
            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"  No CBV file found for sector {sector}, cam {cam}, ccd {ccd}")
                continue

            file_path = matching_files[0]
            print(f"  Found: {file_path}")

            try:
                with fits.open(file_path, memmap=True) as hdulist:
                    for j in range(30):
                        k = j + 1
                        try:
                            evec  = hdulist[1].data[f'VECTOR_{k}']
                            times = hdulist[1].data['CADENCENO']
                            if np.any(evec):
                                indices    = times - cadence_bounds[sector][0] - 1
                                valid_mask = (indices >= 0) & (indices < N_cadence)
                                evecs[j, indices[valid_mask]] = evec[valid_mask]
                        except:
                            continue

                output_path = os.path.join(OUTPUT_DIR, str(sector))
                os.makedirs(output_path, exist_ok=True)
                pickle.dump(evecs, open(f'{output_path}/evec_matrix_{sector}_{cam}_{ccd}.p', 'wb'))
                print(f"  Saved evec_matrix_{sector}_{cam}_{ccd}.p")

            except Exception as e:
                print(f"  Error: {e}")
                continue

print("Done!")
