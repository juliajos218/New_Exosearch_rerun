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

### 8. Run vetting
...
