#!/usr/bin/env bash
# download_all_targets.sh
# Fetch sector target-list text files S070–S083 (keeps existing files).

for s in $(seq -f "%03g" 61 69); do        # 061 … 069 change sector values here
    url="https://tess.mit.edu/public/target_lists/2m/all_targets_S${s}_v1.txt"
    wget -nc "$url"                        # -nc = “no clobber” (skip if already there)
done

