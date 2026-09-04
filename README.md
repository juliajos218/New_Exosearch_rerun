# New_Exosearch_rerun

## GitHub Repo Structure
Scripts and notebooks only — no data files (data lives on Treefort)

```
tess_transit_rerun/ (GitHub)
├── README.md                         # This file
├── year5_pipeline.ipynb              # Main pipeline notebook
├── current_vetting_and_p...          # Vetting notebook
├── code/
│   └── preprocess/
│       ├── download_lc_y5.py         # downloads and preprocesses light curves (adapted from Dr. Taaki)
│       └── vector_matrix.py          # generates evec matrices from CBV files (Dr. Taaki)
├── info/
│   └── info_y5.py                    # year 5 cadence bounds and sector config
├── py_files/
│   └── multi_target_run_y5.py        # FFT search script for year 5 (FFR search method from Dr. Taaki)
└── sh_files/
    └── download_lists.sh             # downloads sector target lists from MIT
```

## TreeFort File Structure

```
tess_transit_rerun/
├── year5_pipeline.ipynb          # Main pipeline notebook
├── README.md                     # This file
└── NOTES.md                      # Pipeline decisions and known issues

TESS/ (on Treefort at ~/TESS/)
├── data/
│   ├── light_curves/
│   │   ├── info/
│   │   │   ├── all_targets_S061_v1.txt   # sector target lists (from MIT)
│   │   │   ├── ...
│   │   │   ├── all_targets_S069_v1.txt
│   │   │   ├── persistant_tids_y5.txt    # TICs persistent across all 9 sectors
│   │   │   ├── info_y5.py                # year 5 cadence bounds and sector config
│   │   │   └── cbvs/                     # CBV download scripts + FITS files
│   │   └── lightcurves_y5/
│   │       └── {TIC_ID}.p                # preprocessed light curve pickles
│   ├── priors/
│   │   └── {sector}/
│   │       └── evec_matrix_{sector}_{cam}_{ccd}.p
│   └── output_y5/
│       └── {TIC_ID}_results.p            # FFT search results (SNR filtered)
└── code/
    └── preprocess/
        ├── vector_matrix.py              # generates evec matrices from CBV files (Dr. Taaki)
        ├── download_lc_y5.py             # downloads and preprocesses light curves (Dr. Taaki with personal edits)
        └── systematics_cov.py            # computes systematics covariance (Dr. Taaki)

tess_transit/ (on Treefort at ~/tess_transit/)
├── multi_target_run_y5.py        # FFT search script for year 5 (FFT search script by Dr. Taaki)
├── data/
│   └── loader.py                 # updated to import from info_y5.py
└── search.py                     # FFT search pipeline (Dr. Taaki)

```

---

# TESS Exoplanet Transit Search Pipeline

## Setup & Data Download

### Step 1 — Download sector target lists
```bash
cd ~/TESS/data/light_curves/info/
bash download_lists.sh
```
**Output:** Downloads `all_targets_S061_v1.txt` through `all_targets_S069_v1.txt` from MIT

---

### Step 2 — Find persistent TIC IDs
Run `year5_pipeline.ipynb` Step 2 cell.

**Output:** 1299 persistent TICs saved to `persistant_tids_y5.txt`

---

### Step 3a — Download CBV shell scripts
```bash
cd ~/TESS/data/light_curves/info/cbvs/
for s in $(seq -f "%02g" 61 69); do
    wget "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_${s}_cbv.sh"
done
```

### Step 4 — Run CBV shell scripts to download FITS files
```bash
for s in $(seq -f "%02g" 61 69); do
    echo "Downloading CBVs for sector $s..."
    bash tesscurl_sector_${s}_cbv.sh
done
```
**Output:** 368 CBV FITS files downloaded

---

### Step 5a — Get cadence bounds from CBV FITS files
Run `year5_pipeline.ipynb` Step 5 cell.
**Output:** cadence bounds for sectors 61-69 saved to `info_y5.py`

---

### Step 5b — Create info file for your sectors
Create a new `info_y5.py` file in `~/TESS/data/light_curves/info/` with the cadence bounds from Step 5:

```python
year = 5
sectors = [61, 69]
cadence_bounds = {
    61: (1249444, 1267755),
    62: (1267906, 1286422),
    63: (1286574, 1305680),
    64: (1305831, 1325215),
    65: (1325721, 1345805),
    66: (1346554, 1367260),
    67: (1367411, 1387397),
    68: (1387549, 1407372),
    69: (1407524, 1426092)
}
```

Then update `~/tess_transit/data/loader.py` to import from your new info file:
```bash
nano ~/tess_transit/data/loader.py
```
Change:
```python
from info import cadence_bounds
```
To:
```python
import sys
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')
from info_y5 import cadence_bounds
```
**Note:** This step is required — if skipped every target will fail with `Sector X not in cadence_bounds`.

---
---

### Step 6 — Generate evec matrices from CBV files
```bash
python3 ~/TESS/code/preprocess/vector_matrix.py
```
**Output:** `evec_matrix_{sector}_{cam}_{ccd}.p` saved to `~/TESS/data/priors/{sector}/`

---

### Step 7 — Download and preprocess light curves
```bash
nohup python3 ~/TESS/code/preprocess/download_lc_y5.py > download_lc_y5.log 2>&1 &
tail -f download_lc_y5.log
```
**Estimated time:** ~10-11 hours for 1299 targets

---

### Step 8 — Run FFT search with inline SNR filter
```bash
nohup python3 ~/tess_transit/multi_target_run_y5.py > multi_target_run_y5.log 2>&1 &
tail -f multi_target_run_y5.log
```
**Output:** `{TIC_ID}_results.p` saved to `~/tess_transit/data/output_y5/`
**Note:** SNR ≥ 6 filter applied inline to keep file sizes ~80MB instead of ~2.5GB
**Estimated time:** 10-15 hours for 1299 targets
**Estimated storage:** ~104GB (vs ~3.25TB unfiltered)
---

### Step 9 — Apply SNR filter
```bash
python3 filter_snr.py
```

### Step 10 — Run vetting
Run vetting notebook with updated paths for year 5.

---

### Step 11 — Manual inspection
Inspect phase-folded lightcurves for planet candidates.

---

# FIRST RUN (Year 6) — Pipeline Notes & Decision Log

## Data
- Sectors 73-83 (TESS Year 6, Northern CVZ)
- 1011 persistent TIC IDs observed in all 11 sectors
- 2-minute cadence light curves downloaded from MAST via SPOC

## Key Parameter Choices

### SNR Filter
- Threshold: SNR ≥ 6
- Applied after FFT search to reduce storage from ~2.5GB to ~80MB per target
- 678/1011 targets passed (67%)

### Sigma Clipping
- 15σ threshold applied to flux before centroid motion test only
- NOT applied to even-odd test — clipping erases deep EB eclipse depths,
  causing the EOP test to incorrectly pass targets that should fail
- Originally 5σ, changed to 15σ after TIC 376976984 (deep EB) was incorrectly
  passing the EOP test due to eclipse depths being clipped

### Vetting Thresholds
- CM_FAIL_ALPHA = 0.01
- EOP_FAIL_ALPHA = 0.01
- SCORE_FAIL_ALPHA = 0.05
- SCORE_WARN_ALPHA = 0.02
- Weighted score = (3 × CM_p + EOP_p) / 4
- CM test weighted 3x more than EOP — directly measures physical centroid shift

### Depth Calculation
- Uses phase-folded binned flux (bin_size = 0.003 in phase units)
- Depth = minimum of binned flux within 2x half-duration window near phase 0
- Baseline = median of all out-of-transit bins
- norm_factor = median raw photon counts from lc_data across all sectors
- Converts MAD units to fractional depth: depth_fraction = (baseline - transit_min) / norm_factor
- Rp/Rs = sqrt(depth_fraction)
- Targets with Rp/Rs > 0.15 flagged as likely false positives

## Known Issues

### Sector 78 Outliers
- Many targets have a single outlying cadence in sector 78 that shifts the centroid
- Fixed by sector quality flag: excludes sectors where peak-to-peak centroid drift > 10x typical scatter

### Harmonic Aliasing
- FFT search sometimes detects integer multiples or fractions of the true period
- 6 confirmed cases identified by cross-matching with ExoFOP:
  - TIC 160583126: detected 2.35d, correct 6.998d (3x)
  - TIC 229750058: detected 5.09d, correct 10.18d (2x)
  - TIC 267542728: detected 7.96d, correct 39.74d (5x)
  - TIC 356822426: detected 7.78d, correct 1.56d (1/5x)
  - TIC 356978132: detected 8.60d, correct 60.18d (7x)
  - TIC 359629653: detected 8.35d, correct 1.67d (5x)
- Automated harmonic correction attempted but not reliably implemented
- Manual correction applied for known cases
- Future work: implement robust period validation

### Multi-Planet Systems
- Centroid motion test may be too sensitive for multi-planet systems
- Baseline window may contain transits from second planet, inflating centroid correlation
- TIC 287139872 (TOI-1752, confirmed 2-planet system) incorrectly failed vetting

### t0 Convention
- epoch_cadences stored as relative cadences from start of sector 73
- For phase folding: use epoch_to_btjd(epoch_cadences, duration) — no offset needed
- For BJD midpoint (ExoFOP submission): use t0_btjd + 2457000

## Results Summary
- 1011 targets searched
- 678 passed SNR ≥ 6 filter
- 160 passed automated vetting
- ~22 good candidates after manual inspection
- 7 strong detections with depth SNR > 3 and Rp/Rs < 15%
- 3 confirmed planets in sample (TOI-2071b, TOI-1291b, TOI-1752b/c)
- TOI-2071b and TOI-1291b correctly passed vetting
- TOI-1752b/c incorrectly failed (multi-planet system issue)

---

# SECOND RUN (Year 5) — Pipeline Notes & Decision Log

## Data
- Sectors 61-69 (TESS Year 5, Southern CVZ)
- 1299 persistent TIC IDs observed in all 9 sectors
- 2-minute cadence light curves downloaded from MAST via SPOC
- Cadence bounds derived from CBV FITS files

## Results Summary
- In progress

---

## Credits & Acknowledgments

This pipeline was developed as part of undergraduate research at the University of Michigan
Department of Astronomy, advised by Dr. Jamila Taaki.

The following scripts were written or provided by Dr. Taaki and adapted for this project:
- `vector_matrix.py` — formats CBV FITS files into evec matrices for systematics detrending
- `systematics_cov.py` — computes systematics covariance coefficients per sector/cam/ccd
- `download_lc_y5.py` — adapted from 'download_lc.py', downloads and preprocesses TESS light curves from MAST

The vetting tests (centroid motion test and even-odd phase test) were developed based on
ideas from the EXOMINER++ framework and adapted for this data format.
