#!/usr/bin/env python3
"""
images_y5.py
------------
Generates 3-panel diagnostic figures for all PASS/WARN targets from
the Year 5 vetting results. Saves one PNG per target to IMAGES_DIR.

Run:
    nohup python3 ~/tess_transit_rerun/images_y5.py > ~/tess_transit_rerun/images_y5.log 2>&1 &
    echo $!
"""

import gc
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/juliajos')
sys.path.insert(0, '/home/juliajos/TESS/data/light_curves/info')

VETTING_DIR = "/home/juliajos/tess_transit/data/vetting__results_y5"
RESULTS_DIR = "/home/juliajos/tess_transit/data/output_y5"
LAB_DIR     = "/home/juliajos/TESS/data/lightcurves_y5"
IMAGES_DIR  = "/home/juliajos/tess_transit/data/vetting__images_y5"
SECTORS     = [61, 62, 63, 64, 65, 66, 67, 68, 69]

os.makedirs(IMAGES_DIR, exist_ok=True)

CADENCE_DURATION_DAYS = 2 / (24 * 60)

def cadence_to_days(n):
    return n * CADENCE_DURATION_DAYS

def cadences_to_btjd(cad):
    return np.asarray(cad, dtype=float) * (120 / 86400)

def epoch_to_btjd(epoch_cadences, duration_cadences):
    return (epoch_cadences + duration_cadences / 2) * (120 / 86400)


def make_figure(tid, rec, bjd, flux):
    """
    Build and save the 3-panel diagnostic figure for one target.
    rec  : vetting pickle dict
    bjd  : stitched BTJD array
    flux : stitched detrended flux array
    """
    P      = rec['period_days']
    t0     = rec['t0_btjd']
    dur    = rec['duration_days']
    snr    = rec['snr']
    score  = rec['score']
    cm_p   = rec.get('cm_p', float('nan'))
    eop_p  = rec.get('eop_p', float('nan'))
    rp_rs  = rec.get('rp_rs', float('nan'))
    verdict = rec.get('verdict', '?')

    # ── phase ──────────────────────────────────────────────────────────────────
    phase     = ((bjd - t0 + 0.5 * P) % P) - 0.5 * P
    phase_norm = phase / P          # fraction of period, centred on 0

    # ── binned phase fold (for panels 2 & 3) ──────────────────────────────────
    bin_size = 0.003
    xlim     = 0.10
    bins     = np.arange(-xlim, xlim + bin_size, bin_size)
    b_phase, b_flux = [], []
    for i in range(1, len(bins)):
        mask = (phase_norm >= bins[i-1]) & (phase_norm < bins[i])
        if np.any(mask):
            b_phase.append(0.5 * (bins[i-1] + bins[i]))
            b_flux.append(np.nanmedian(flux[mask]))
    b_phase = np.array(b_phase)
    b_flux  = np.array(b_flux)

    # ── depth estimate (panel 3) ───────────────────────────────────────────────
    half_dur_frac = (dur / P) / 2
    near_transit  = np.abs(b_phase) <= half_dur_frac * 2   # 2× window
    in_transit_b  = np.abs(b_phase) <= half_dur_frac
    baseline_mask = ~near_transit

    baseline_level = np.nanmedian(b_flux[baseline_mask]) if np.any(baseline_mask) else 0.0
    transit_min    = np.nanmin(b_flux[in_transit_b])     if np.any(in_transit_b)  else np.nan

    depth_ppm = (baseline_level - transit_min) * 1e6 if not np.isnan(transit_min) else float('nan')

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(f"TIC {tid}", fontsize=14, fontweight='bold', y=0.995)

    # ── Panel 1: stitched light curve + transit markers ───────────────────────
    ax = axes[0]
    ax.scatter(bjd, flux, s=0.5, alpha=0.4, color='steelblue', rasterized=True)

    n_min = int(np.floor((bjd.min() - t0) / P))
    n_max = int(np.ceil( (bjd.max() - t0) / P))
    for n in range(n_min, n_max + 1):
        tt = t0 + n * P
        if bjd.min() <= tt <= bjd.max():
            ax.axvline(tt, color='red', lw=0.6, linestyle='--', alpha=0.5)

    cm_str  = f"{cm_p:.2f}"  if not np.isnan(cm_p)  else "nan"
    eop_str = f"{eop_p:.3f}" if not np.isnan(eop_p) else "nan"
    ax.set_title(
        f"Stitched lightcurve — P={P:.4f} d  SNR={snr:.1f}  "
        f"Score={score:.3f}  CM p={cm_str}  EOP p={eop_str}",
        fontsize=9
    )
    ax.set_xlabel("BTJD [days]")
    ax.set_ylabel("Detrended Flux")

    # ── Panel 2: phase fold ───────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(phase_norm, flux, s=0.5, color='gray', alpha=0.2, rasterized=True)
    ax.plot(b_phase, b_flux, 'D-', ms=3, color='steelblue', label='binned median')
    ax.axvline(0, color='red', lw=0.8, linestyle='--', label='transit center')
    ax.axvline(-half_dur_frac, color='red', lw=0.5, linestyle=':')
    ax.axvline( half_dur_frac, color='red', lw=0.5, linestyle=':')
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("Phase (fraction of period)")
    ax.set_ylabel("Detrended Flux")
    rp_str = f"{rp_rs:.4f} ({rp_rs*100:.2f}%)" if not np.isnan(rp_rs) else "nan"
    depth_str = f"{depth_ppm:.0f}" if not np.isnan(depth_ppm) else "nan"
    ax.set_title(f"Phase fold — Depth={depth_str} ppm  Rp/Rs={rp_str}", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    # ── Panel 3: depth diagnostic (zoomed binned fold) ────────────────────────
    ax = axes[2]
    xlim3 = max(half_dur_frac * 6, 0.05)   # zoom to ~6× transit duration

    # colour points by location
    for i, (ph, fl) in enumerate(zip(b_phase, b_flux)):
        if abs(ph) > xlim3:
            continue
        color = 'red' if abs(ph) <= half_dur_frac * 2 else 'green'
        ax.scatter(ph, fl, color=color, s=20, zorder=3)

    ax.axhline(baseline_level, color='green',  lw=1.0, linestyle='--',
               label=f'baseline={baseline_level:.3f}')
    if not np.isnan(transit_min):
        ax.axhline(transit_min, color='orange', lw=1.0, linestyle='--',
                   label=f'transit min={transit_min:.3f}')
    ax.axvline( half_dur_frac, color='pink', lw=0.8, linestyle='-')
    ax.axvline(-half_dur_frac, color='pink', lw=0.8, linestyle='-')
    ax.axvline( half_dur_frac * 2, color='pink', lw=0.5, linestyle=':')
    ax.axvline(-half_dur_frac * 2, color='pink', lw=0.5, linestyle=':')
    ax.axvline(0, color='red', lw=0.8, linestyle='--')

    ax.set_xlim(-xlim3, xlim3)
    ax.set_xlabel("Phase (fraction of period)")
    ax.set_ylabel("Binned Flux")
    ax.set_title("Depth calculation diagnostic", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(IMAGES_DIR, f"{tid}_diagnostic.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ── main loop ──────────────────────────────────────────────────────────────────

import glob
vetting_files = sorted(glob.glob(os.path.join(VETTING_DIR, "*_vetting.p")))
pass_warn = []
for fpath in vetting_files:
    with open(fpath, "rb") as f:
        rec = pickle.load(f)
    if rec.get('verdict') in ('PASS', 'WARN'):
        pass_warn.append((fpath, rec))

print(f"Found {len(pass_warn)} PASS/WARN targets to plot")

done = 0
failed = 0

for fpath, rec in pass_warn:
    tid = str(rec['tid'])
    print(f"[{tid}] plotting...", flush=True)

    # load light curve
    lab_path = os.path.join(LAB_DIR, f"{tid}.p")
    try:
        with open(lab_path, "rb") as f:
            (lc_data, processed_lc_data, detrend_data, norm_offset, quality_data,
             time_data, cam_data, ccd_data, coeff_ls,
             centroid_xy_data, pos_xy_corr) = pickle.load(f)
    except Exception as e:
        print(f"  [{tid}] could not load LC: {e}")
        failed += 1
        continue

    # stitch sectors
    try:
        bjd_parts  = [cadences_to_btjd(np.asarray(time_data[s])) for s in SECTORS if s in time_data]
        flux_parts = [np.asarray(detrend_data[s].unmasked)        for s in SECTORS if s in detrend_data]
        bjd  = np.concatenate(bjd_parts)
        flux = np.concatenate(flux_parts)
        sort_idx = np.argsort(bjd)
        bjd  = bjd[sort_idx]
        flux = flux[sort_idx]
    except Exception as e:
        print(f"  [{tid}] could not stitch LC: {e}")
        failed += 1
        del lc_data, processed_lc_data, detrend_data, norm_offset, quality_data
        del time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr
        gc.collect()
        continue

    # make figure
    try:
        save_path = make_figure(tid, rec, bjd, flux)
        print(f"  [{tid}] saved → {save_path}")
        done += 1
    except Exception as e:
        print(f"  [{tid}] figure failed: {e}")
        failed += 1

    del lc_data, processed_lc_data, detrend_data, norm_offset, quality_data
    del time_data, cam_data, ccd_data, coeff_ls, centroid_xy_data, pos_xy_corr
    del bjd, flux
    gc.collect()

print(f"\nDone. Saved: {done}  Failed: {failed}")
print(f"Images in: {IMAGES_DIR}")
