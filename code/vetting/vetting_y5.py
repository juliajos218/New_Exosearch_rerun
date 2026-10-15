#!/usr/bin/env python3
"""
vetting_y5.py
-------------
Standalone vetting loop for TESS Year 5 (Sectors 61-69).
Run from the terminal:
    nohup python3 ~/tess_transit_rerun/vetting_y5.py > vetting_y5.log 2>&1 &
    tail -f ~/tess_transit_rerun/vetting_y5.log
"""

# ── imports ────────────────────────────────────────────────────────────────────
import gc
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/juliajos')
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')

from scipy.stats import pearsonr, binom, ttest_ind, norm

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = "/home/juliajos/tess_transit/data/output_y5"
LAB_DIR     = os.path.expanduser("~/TESS/data/lightcurves_y5")
VETTING_DIR = "/home/juliajos/tess_transit/data/vetting__results_y5"
IMAGES_DIR  = "/home/juliajos/tess_transit/data/vetting__images_y5"
TIDS_FILE   = "/home/juliajos/TESS/data/light_curves/info/persistant_tids_y5.txt"

os.makedirs(VETTING_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR,  exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
SECTORS            = [61, 62, 63, 64, 65, 66, 67, 68, 69]
CADENCE_DURATION_DAYS = 2 / (24 * 60)   # 2-minute cadence in days

CM_FAIL_ALPHA    = 0.05
EOP_FAIL_ALPHA   = 0.05
SCORE_FAIL_ALPHA = 0.05
SCORE_WARN_ALPHA = 0.02
SIGMA_CLIP       = 15.0

# ── utility functions ──────────────────────────────────────────────────────────
def cadence_to_days(n_cadences):
    return n_cadences * CADENCE_DURATION_DAYS

def cadences_to_btjd(cadence_numbers):
    return cadence_numbers * (120 / 86400)

def epoch_to_btjd(epoch_cadences, duration_cadences):
    return (epoch_cadences + duration_cadences / 2) * (120 / 86400)

# ── vetting functions ──────────────────────────────────────────────────────────
def stouffers_method(p_values):
    clean = [p for p in p_values if p is not None and not np.isnan(p)]
    if len(clean) == 0:
        return np.nan
    clipped = [max(min(p, 1 - 1e-10), 1e-10) for p in clean]
    z_scores = [norm.ppf(p) for p in clipped]
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))
    return float(norm.cdf(combined_z))

def combined_score_weighted(centroid_p, eop_p):
    return (3 * min(centroid_p, 1.0) + 1 * min(eop_p, 1.0)) / 4

def even_odd_phase_our_data(flux, time_btjd, period_days, t0_btjd, duration_days,
                             plot_size=0.15, same_axes=False, plot=False):
    flux      = np.array(flux,      dtype=float)
    time_btjd = np.array(time_btjd, dtype=float)

    phase_full   = ((time_btjd - t0_btjd) % period_days) / period_days
    phase_folded = phase_full - np.round(phase_full)
    transit_num  = np.floor((time_btjd - t0_btjd) / period_days).astype(int)

    half_window     = duration_days / (2 * period_days)
    in_transit      = np.abs(phase_folded) < half_window
    in_transit_odd  = in_transit & (transit_num % 2 == 1)
    in_transit_even = in_transit & (transit_num % 2 == 0)

    odd_depths  = 1 - flux[in_transit_odd]
    even_depths = 1 - flux[in_transit_even]
    odd_clean   = odd_depths[~np.isnan(odd_depths)]
    even_clean  = even_depths[~np.isnan(even_depths)]

    p_val = np.nan
    if len(odd_clean) > 1 and len(even_clean) > 1:
        try:
            _, p_val = ttest_ind(odd_clean, even_clean)
        except Exception:
            pass
    return p_val

def calculate_enhanced_centroid_motion_pvalue(
        lc_flux, time_btjd, mom1, mom2, pos1, pos2,
        period_days, t0_btjd, duration_days,
        baseline_factor=3.0, sector=None, plot=False):

    time_btjd = np.asarray(time_btjd, dtype=float)
    lc_flux   = np.asarray(lc_flux,   dtype=float)
    mom1      = np.asarray(mom1,      dtype=float)
    mom2      = np.asarray(mom2,      dtype=float)
    pos1      = np.asarray(pos1,      dtype=float)
    pos2      = np.asarray(pos2,      dtype=float)

    centroid_x = mom1 - pos1
    centroid_y = mom2 - pos2

    t_start, t_end = time_btjd[0], time_btjd[-1]
    n_min = int(np.floor((t_start - t0_btjd) / period_days))
    n_max = int(np.ceil( (t_end   - t0_btjd) / period_days))
    transit_times = [
        t0_btjd + n * period_days
        for n in range(n_min, n_max + 1)
        if t_start <= t0_btjd + n * period_days <= t_end
    ]

    if len(transit_times) == 0:
        return np.nan

    spike_dur       = min(duration_days, 0.5)
    baseline_window = duration_days * baseline_factor
    all_p_values    = []

    for tt in transit_times:
        transit_mask  = (time_btjd > tt - spike_dur / 2) & (time_btjd < tt + spike_dur / 2)
        baseline_mask = (
            (time_btjd > tt - baseline_window) &
            (time_btjd < tt + baseline_window) &
            ~transit_mask
        )

        if np.sum(transit_mask) < 2 or np.sum(baseline_mask) < 5:
            continue

        cx_tr    = centroid_x[transit_mask]
        cy_tr    = centroid_y[transit_mask]
        cx_bl    = centroid_x[baseline_mask]
        cy_bl    = centroid_y[baseline_mask]
        flux_tr  = lc_flux[transit_mask]
        flux_dev = (flux_tr / np.nanmedian(flux_tr)) - 1.0

        for cx in [cx_tr, cy_tr]:
            valid = ~np.isnan(cx) & ~np.isnan(flux_dev)
            if np.sum(valid) >= 3:
                _, p = pearsonr(flux_dev[valid], cx[valid])
                all_p_values.append(p)

        for c_tr, c_bl in [(cx_tr, cx_bl), (cy_tr, cy_bl)]:
            bl_clean = c_bl[~np.isnan(c_bl)]
            tr_clean = c_tr[~np.isnan(c_tr)]
            if len(bl_clean) > 5 and len(tr_clean) > 0:
                outliers = np.abs(tr_clean - np.median(bl_clean)) > 5 * np.std(bl_clean)
                if np.sum(outliers) > 0:
                    p_sp = 1 - binom.cdf(np.sum(outliers) - 1, len(tr_clean), 0.003)
                    all_p_values.append(p_sp)

    if len(all_p_values) == 0:
        return np.nan
    return stouffers_method(all_p_values)

# ── main vetting loop ──────────────────────────────────────────────────────────
def main():
    tids = np.loadtxt(TIDS_FILE, dtype=int)
    print(f"Total targets: {len(tids)}", flush=True)

    passed            = 0
    failed            = 0
    skipped_no_result = 0

    for TARGET_ID in tids:
        TARGET_ID = str(TARGET_ID)
        print(f"[{TARGET_ID}] starting...", flush=True)
        out_path = os.path.join(VETTING_DIR, f"{TARGET_ID}_vetting.p")

        if os.path.exists(out_path):
            continue

        results_path = os.path.join(RESULTS_DIR, f"{TARGET_ID}_results.p")
        if not os.path.exists(results_path):
            skipped_no_result += 1
            continue

        try:
            with open(results_path, "rb") as f:
                results_real = pickle.load(f)
        except Exception as e:
            print(f"[{TARGET_ID}] could not load results: {e}", flush=True)
            continue

        candidates = results_real.get('best_candidates', [])
        if len(candidates) == 0 or candidates[0].get('snr', 0) < 6:
            record = {'tid': TARGET_ID, 'status': 'no_candidate'}
            with open(out_path, "wb") as f:
                pickle.dump(record, f)
            failed += 1
            continue

        best_real         = candidates[0]
        det_period_days   = best_real['period_days']
        det_duration_days = cadence_to_days(best_real['duration'])

        lab_path = os.path.join(LAB_DIR, f"{TARGET_ID}.p")
        if not os.path.exists(lab_path):
            print(f"[{TARGET_ID}] lab file not found, skipping.", flush=True)
            continue

        try:
            with open(lab_path, "rb") as f:
                (lc_data, processed_lc_data, detrend_data, norm_offset, quality_data,
                 time_data, cam_data, ccd_data, coeff_ls,
                 centroid_xy_data, pos_xy_corr) = pickle.load(f)
        except Exception as e:
            print(f"[{TARGET_ID}] could not load LC: {e}", flush=True)
            continue

        sector61_start_cad = np.asarray(time_data[SECTORS[0]])[0]
        det_t0_btjd = epoch_to_btjd(best_real['epoch_cadences'], best_real['duration'])

        all_flux_raw = np.concatenate([np.asarray(lc_data[s]) for s in SECTORS if s in lc_data])
        norm_factor  = float(np.nanmedian(all_flux_raw))

        sector_cm_pvals  = []
        sector_eop_pvals = []

        for sector in SECTORS:
            try:
                lc_sector     = np.array(detrend_data[sector].unmasked)
                time_sector   = np.asarray(time_data[sector], dtype=float)
                time_btjd_s   = cadences_to_btjd(time_sector)
                pos_xy_sector = pos_xy_corr[sector]
                mom_xy        = centroid_xy_data[sector]
                pos_cor1      = np.array(pos_xy_sector[0].unmasked)
                pos_cor2      = np.array(pos_xy_sector[1].unmasked)
                mom1          = np.array(mom_xy[0].unmasked)
                mom2          = np.array(mom_xy[1].unmasked)
            except Exception:
                continue

            # CM test — apply 15-sigma clip to flux only for this test
            try:
                lc_cm = lc_sector.copy()
                med   = np.nanmedian(lc_cm)
                std   = np.nanstd(lc_cm)
                lc_cm[np.abs(lc_cm - med) > SIGMA_CLIP * std] = med

                cm_p = calculate_enhanced_centroid_motion_pvalue(
                    lc_cm, time_btjd=time_btjd_s,
                    mom1=mom1, mom2=mom2, pos1=pos_cor1, pos2=pos_cor2,
                    period_days=det_period_days, t0_btjd=det_t0_btjd,
                    duration_days=max(det_duration_days, cadence_to_days(6)),
                    baseline_factor=3.0, sector=sector, plot=False
                )
            except Exception:
                cm_p = np.nan
            sector_cm_pvals.append(cm_p)

            # EOP test — NO clipping
            try:
                bin_size    = 0.005
                bins        = np.arange(np.min(time_btjd_s), np.max(time_btjd_s) + bin_size, bin_size)
                bin_indices = np.digitize(time_btjd_s, bins)
                b_time, b_flux_eop = [], []
                for i in range(1, len(bins)):
                    mask = bin_indices == i
                    if np.any(mask):
                        b_time.append(np.mean(time_btjd_s[mask]))
                        b_flux_eop.append(np.mean(lc_sector[mask]))
                b_time     = np.array(b_time)
                b_flux_eop = np.array(b_flux_eop)

                eop = even_odd_phase_our_data(
                    b_flux_eop, time_btjd=b_time,
                    period_days=det_period_days, t0_btjd=det_t0_btjd,
                    duration_days=det_duration_days,
                    plot_size=0.15, same_axes=False, plot=False
                )
            except Exception:
                eop = np.nan
            sector_eop_pvals.append(eop)

            del lc_sector, time_sector, time_btjd_s
            del pos_cor1, pos_cor2, mom1, mom2, b_time, b_flux_eop
            gc.collect()

        combined_cm_p  = stouffers_method(sector_cm_pvals)
        combined_eop_p = stouffers_method(sector_eop_pvals)
        score = (
            combined_score_weighted(combined_cm_p, combined_eop_p)
            if not (np.isnan(combined_cm_p) or np.isnan(combined_eop_p))
            else np.nan
        )

        # depth / Rp/Rs
        try:
            bjd_all  = np.concatenate([cadences_to_btjd(np.asarray(time_data[s], dtype=float))
                                       for s in SECTORS if s in time_data])
            flux_all = np.concatenate([np.asarray(detrend_data[s].unmasked)
                                       for s in SECTORS if s in detrend_data])
            sort_idx = np.argsort(bjd_all)
            bjd_all  = bjd_all[sort_idx]
            flux_all = flux_all[sort_idx]

            phase_norm = (((bjd_all - det_t0_btjd + 0.5 * det_period_days)
                           % det_period_days) - 0.5 * det_period_days) / det_period_days
            half_dur   = (det_duration_days / det_period_days) / 2

            bin_size = 0.003
            bins = np.arange(-0.5, 0.5 + bin_size, bin_size)
            b_phase_d, b_flux_d = [], []
            for i in range(1, len(bins)):
                m = (phase_norm >= bins[i-1]) & (phase_norm < bins[i])
                if np.any(m):
                    b_phase_d.append(0.5 * (bins[i-1] + bins[i]))
                    b_flux_d.append(np.nanmedian(flux_all[m]))
            b_phase_d = np.array(b_phase_d)
            b_flux_d  = np.array(b_flux_d)

            near_transit   = np.abs(b_phase_d) < 1 * half_dur
            baseline       = np.nanmedian(b_flux_d[~near_transit]) if np.any(~near_transit) else np.nan
            transit_min    = np.nanmin(b_flux_d[near_transit])     if np.any(near_transit)  else np.nan
            depth_fraction = (baseline - transit_min) / norm_factor if norm_factor != 0 else np.nan
            rp_rs          = float(np.sqrt(np.abs(depth_fraction))) if not np.isnan(depth_fraction) else np.nan
        except Exception:
            depth_fraction = np.nan
            rp_rs          = np.nan

        # verdict
        if np.isnan(score):
            verdict = 'INSUFFICIENT_DATA'
        elif combined_cm_p < CM_FAIL_ALPHA:
            verdict = 'FAIL_CM'
        elif combined_eop_p < EOP_FAIL_ALPHA:
            verdict = 'FAIL_EOP'
        elif score < SCORE_WARN_ALPHA:
            verdict = 'FAIL_SCORE'
        elif score < SCORE_FAIL_ALPHA:
            verdict = 'WARN'
        else:
            verdict = 'PASS'

        record = {
            'tid':            TARGET_ID,
            'period_days':    det_period_days,
            'duration_days':  det_duration_days,
            't0_btjd':        det_t0_btjd,
            'snr':            best_real['snr'],
            'cm_p':           combined_cm_p,
            'eop_p':          combined_eop_p,
            'score':          score,
            'depth_fraction': depth_fraction,
            'rp_rs':          rp_rs,
            'verdict':        verdict,
        }
        with open(out_path, "wb") as f:
            pickle.dump(record, f)

        if verdict in ('PASS', 'WARN'):
            passed += 1
            print(
                f"[{TARGET_ID}] {verdict}  P={det_period_days:.3f}d  "
                f"SNR={best_real['snr']:.1f}  score={score:.3f}  Rp/Rs={rp_rs:.3f}",
                flush=True
            )
        else:
            failed += 1

        del lc_data, processed_lc_data, detrend_data, norm_offset, quality_data
        del time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr
        del results_real, all_flux_raw
        gc.collect()

    print(f"\nDone. Passed/Warn: {passed}  Failed: {failed}  No result file: {skipped_no_result}", flush=True)


if __name__ == "__main__":
    main()
