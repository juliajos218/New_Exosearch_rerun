# New_Exosearch_rerun

# TESS Exoplanet Transit Search Pipeline

## Setup & Data Download

### 1. Download sector target lists
```bash
bash download_lists.sh
```

### 2. Download CBV files for each sector
```bash
cd ~/TESS/data/light_curves/info/cbvs/
for s in $(seq -f "%02g" 61 69); do
    wget "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_${s}_cbv.sh"
    bash tesscurl_sector_${s}_cbv.sh
done
```

### 3. Generate evec matrices from CBV files
```bash
python vector_matrix.py
```

### 4. Find persistent TIC IDs
Run `persistant_tic_list_and_download.ipynb`

### 5. Download and preprocess light curves
```bash
python download_lc.py
```

### 6. Run FFT search
...

### 7. Apply SNR filter
```bash
python filter_snr.py
```



# FIRST RUN (year 6) - Pipeline Notes & Decision Log

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

### 8. Run vetting
...
